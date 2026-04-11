"""
Concept Swap + Classifier Degradation — Positive Control.

Demonstrates that C-MMD tracks downstream classifier degradation under
controlled semantic drift (label swap).

Setup:
  - Reference: 5000 IMDb reviews with clean labels
  - Stream: Exact clone of reference embeddings, but with progressively
    corrupted labels (0 → 50 % label swap)
  - A LogisticRegression sentiment classifier is trained on the clean
    reference and evaluated against the corrupted stream at each level.

Expected results:
  - Global MMD ≈ 0  (identical embeddings)
  - C-MMD rises with corruption
  - Classifier Macro-F1 falls with corruption
  - Strong negative Spearman ρ between C-MMD and F1
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from src.mmd import mmd_rbf_chunked as mmd_rbf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ------------------------------------------------------------------
# Helpers (reused from 03_concept_swap.py)
# ------------------------------------------------------------------

def split_by_label(embeddings, labels):
    return embeddings[labels == 1], embeddings[labels == 0]


def flip_labels(labels, flip_ratio, seed=42):
    torch.manual_seed(seed)
    corrupted = labels.clone()
    n_flip = int(len(labels) * flip_ratio)
    if n_flip > 0:
        idx = torch.randperm(len(labels))[:n_flip]
        corrupted[idx] = 1 - corrupted[idx]
    return corrupted


# ------------------------------------------------------------------
# Stronger baselines (same as experiment 12)
# ------------------------------------------------------------------

def compute_energy_distance(X_ref_np, X_win_np, max_samples=300):
    rng = np.random.RandomState(42)
    if len(X_ref_np) > max_samples:
        X_ref_np = X_ref_np[rng.choice(len(X_ref_np), max_samples, replace=False)]
    if len(X_win_np) > max_samples:
        X_win_np = X_win_np[rng.choice(len(X_win_np), max_samples, replace=False)]
    exy = np.mean(cdist(X_ref_np, X_win_np, 'euclidean'))
    exx = np.mean(cdist(X_ref_np, X_ref_np, 'euclidean'))
    eyy = np.mean(cdist(X_win_np, X_win_np, 'euclidean'))
    return float(2 * exy - exx - eyy)


def compute_c2st_nonlinear(X_ref_np, X_win_np, max_samples=300):
    rng = np.random.RandomState(42)
    if len(X_ref_np) > max_samples:
        X_ref_np = X_ref_np[rng.choice(len(X_ref_np), max_samples, replace=False)]
    if len(X_win_np) > max_samples:
        X_win_np = X_win_np[rng.choice(len(X_win_np), max_samples, replace=False)]
    X = np.vstack([X_ref_np, X_win_np])
    y = np.concatenate([np.zeros(len(X_ref_np)), np.ones(len(X_win_np))])
    mlp = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
    grid = GridSearchCV(
        mlp,
        {'hidden_layer_sizes': [(64,), (128,), (64, 32)],
         'alpha': [1e-4, 1e-3, 1e-2]},
        cv=3, scoring='accuracy', n_jobs=-1, refit=False,
    )
    grid.fit(X, y)
    return float(grid.best_score_)


# ------------------------------------------------------------------
# Dual-axis plot
# ------------------------------------------------------------------

def plot_dual_axis(df, rho_cmmd, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = df['Swap_Pct']

    # Left axis — MMD scores
    ax1.plot(x, df['Global_MMD'],      'b--s', linewidth=2.5, markersize=9,
             label='Global MMD')
    ax1.plot(x, df['Conditional_MMD'], 'r-o',  linewidth=2.5, markersize=9,
             label='C-MMD (Conditional)')
    ax1.set_xlabel('Label Swap Percentage (%)', fontsize=12)
    ax1.set_ylabel('MMD Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    # Right axis — Classifier F1
    ax2 = ax1.twinx()
    ax2.plot(x, df['Classifier_F1'], 'g-^', linewidth=2.5, markersize=9,
             alpha=0.85, label='Classifier F1')
    ax2.set_ylabel('Macro F1 Score', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center left', fontsize=11)

    ax1.set_title(
        f'Concept Swap: C-MMD vs Classifier Degradation '
        f'(Spearman \u03c1 = {rho_cmmd:+.3f})',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONCEPT SWAP + CLASSIFIER DEGRADATION  (Positive Control)")
    print("=" * 70)

    # Paths
    base_path   = project_root
    ref_path    = base_path / 'data' / 'processed' / 'ref_embeddings.pt'
    results_dir = base_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out  = results_dir / 'concept_swap_classifier.csv'
    plot_out = results_dir / 'concept_swap_classifier_plot.png'

    # Load reference embeddings (IMDb / DistilBERT)
    print("\nLoading reference embeddings…")
    ref_data       = torch.load(ref_path, map_location='cpu', weights_only=False)
    all_embeddings = ref_data['embeddings']
    all_labels     = ref_data['labels']
    print(f"Total samples: {all_embeddings.shape[0]}, dim={all_embeddings.shape[1]}")

    # Reference subset
    REF_SIZE = 5000
    torch.manual_seed(42)
    idx   = torch.randperm(all_embeddings.shape[0])[:REF_SIZE]
    X_ref = all_embeddings[idx]
    Y_ref = all_labels[idx]

    ref_pos, ref_neg = split_by_label(X_ref, Y_ref)
    print(f"Reference: {REF_SIZE} (pos={ref_pos.shape[0]}, neg={ref_neg.shape[0]})")

    # Stream = exact clone (marginal is identical)
    X_stream = X_ref.clone()

    # ------------------------------------------------------------------
    # Train sentiment classifier on CLEAN reference
    # ------------------------------------------------------------------
    print("\nTraining LogisticRegression on clean reference…")
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    clf.fit(X_ref.numpy(), Y_ref.numpy())
    train_pred = clf.predict(X_ref.numpy())
    print(f"  Train Macro-F1 : {f1_score(Y_ref.numpy(), train_pred, average='macro'):.4f}")
    print(f"  Train Accuracy : {accuracy_score(Y_ref.numpy(), train_pred):.4f}")

    # ------------------------------------------------------------------
    # Sweep over swap percentages
    # ------------------------------------------------------------------
    swap_pcts   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    SAMPLE_SIZE = 200
    results     = []

    # Pre-compute sub-sampled reference for MMD
    torch.manual_seed(42)
    sample_idx     = torch.randperm(REF_SIZE)[:SAMPLE_SIZE]
    ref_sample     = X_ref[sample_idx]
    ref_all_np     = ref_sample.numpy()

    ref_pos_sample = ref_pos[torch.randperm(ref_pos.shape[0])[:SAMPLE_SIZE]]
    ref_neg_sample = ref_neg[torch.randperm(ref_neg.shape[0])[:SAMPLE_SIZE]]

    print("\n" + "-" * 70)
    print(f"{'Swap%':>6}  {'Global MMD':>11}  {'C-MMD':>11}  "
          f"{'F1':>6}  {'Acc':>6}  {'E-Dist':>8}  {'C2ST':>6}")
    print("-" * 70)

    with torch.no_grad():
        for p in swap_pcts:
            Y_corrupt = flip_labels(Y_ref, p, seed=42)

            # ---- Global MMD (embeddings are identical → ≈ 0) ----
            stream_sample = X_stream[sample_idx]
            global_mmd = mmd_rbf(ref_sample, stream_sample).item()

            # ---- Conditional MMD ----
            s_pos, s_neg = split_by_label(X_stream, Y_corrupt)
            sp = s_pos[torch.randperm(s_pos.shape[0])[:SAMPLE_SIZE]] if s_pos.shape[0] > SAMPLE_SIZE else s_pos
            sn = s_neg[torch.randperm(s_neg.shape[0])[:SAMPLE_SIZE]] if s_neg.shape[0] > SAMPLE_SIZE else s_neg

            mmd_pos  = mmd_rbf(ref_pos_sample, sp).item()
            mmd_neg  = mmd_rbf(ref_neg_sample, sn).item()
            cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg

            # ---- Classifier evaluation (predict on stream, score vs corrupted labels) ----
            preds   = clf.predict(X_stream.numpy())
            win_f1  = f1_score(Y_corrupt.numpy(), preds, average='macro')
            win_acc = accuracy_score(Y_corrupt.numpy(), preds)

            # ---- Energy Distance (unsupervised baseline) ----
            e_dist = compute_energy_distance(ref_all_np, stream_sample.numpy())

            # ---- Non-linear C2ST ----
            c2st_acc = compute_c2st_nonlinear(ref_all_np, stream_sample.numpy())

            print(f"{p*100:>5.0f}%  {global_mmd:>11.6f}  {cond_mmd:>11.6f}  "
                  f"{win_f1:>6.4f}  {win_acc:>6.4f}  {e_dist:>8.5f}  {c2st_acc:>6.4f}")

            results.append({
                'Swap_Pct':          p * 100,
                'Global_MMD':        global_mmd,
                'Conditional_MMD':   cond_mmd,
                'Classifier_F1':     win_f1,
                'Classifier_Accuracy': win_acc,
                'Energy_Distance':   e_dist,
                'C2ST_CV_Accuracy':  c2st_acc,
            })

    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    print(f"\n✅ CSV saved to: {csv_out}")

    # ------------------------------------------------------------------
    # Spearman correlations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATIONS  (drift metric ↔ Classifier F1)")
    print("=" * 70)

    metrics = {
        'C-MMD':           'Conditional_MMD',
        'Global MMD':      'Global_MMD',
        'Energy Distance': 'Energy_Distance',
        'C2ST CV Acc':     'C2ST_CV_Accuracy',
    }
    for label, col in metrics.items():
        rho, p_val = spearmanr(df[col], df['Classifier_F1'])
        print(f"  {label:20s}  ρ = {rho:+.4f}   p = {p_val:.4e}")

    rho_cmmd, _ = spearmanr(df['Conditional_MMD'], df['Classifier_F1'])

    # ------------------------------------------------------------------
    # Dual-axis plot
    # ------------------------------------------------------------------
    plot_dual_axis(df, rho_cmmd, plot_out)

    print("\n" + "=" * 70)
    print("Concept Swap Classifier Experiment Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
