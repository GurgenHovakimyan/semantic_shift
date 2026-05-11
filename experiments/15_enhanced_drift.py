"""
Enhanced Drift Evaluation: C-MMD Validation & Stronger Baselines.

Reviewer-requested methodology upgrades:

TASK 1 — Link C-MMD to downstream model degradation:
  • Train LogisticRegression sentiment classifier on reference window
  • Predict on each stream window → Macro-F1 & Accuracy
  • Spearman ρ(C-MMD, F1) after the loop
  • Dual-axis plot: MMD scores (left) + Classifier F1 (right)

TASK 2 — Add stronger baselines:
  • Multivariate Energy Distance (unsupervised baseline)
  • Non-linear C2ST via MLP + GridSearchCV 3-fold CV (supervised baseline)

Dataset: Amazon Electronics (DistilBERT embeddings)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from src.mmd import mmd_rbf_chunked
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_embeddings(filepath: Path, device: torch.device) -> dict:
    """Load embeddings and move to device."""
    data = torch.load(filepath, map_location=device, weights_only=False)
    data['embeddings'] = data['embeddings'].to(device)
    data['labels'] = data['labels'].to(device)
    return data


def split_by_label(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split embeddings into positive (1) and negative (0) subsets."""
    return embeddings[labels == 1], embeddings[labels == 0]


# ---------------------------------------------------------------------------
# TASK 2a — Multivariate Energy Distance
# ---------------------------------------------------------------------------

def compute_energy_distance(
    X_ref_np: np.ndarray, X_win_np: np.ndarray, max_samples: int = 300
) -> float:
    """
    Multivariate energy distance:
        E(P, Q) = 2·E[||X−Y||] − E[||X−X'||] − E[||Y−Y'||]

    Sub-sampled for O(n²) tractability on each window.
    """
    rng = np.random.RandomState(42)

    if len(X_ref_np) > max_samples:
        X_ref_sub = X_ref_np[rng.choice(len(X_ref_np), max_samples, replace=False)]
    else:
        X_ref_sub = X_ref_np

    if len(X_win_np) > max_samples:
        X_win_sub = X_win_np[rng.choice(len(X_win_np), max_samples, replace=False)]
    else:
        X_win_sub = X_win_np

    exy = np.mean(cdist(X_ref_sub, X_win_sub, metric='euclidean'))
    exx = np.mean(cdist(X_ref_sub, X_ref_sub, metric='euclidean'))
    eyy = np.mean(cdist(X_win_sub, X_win_sub, metric='euclidean'))

    return float(2.0 * exy - exx - eyy)


# ---------------------------------------------------------------------------
# TASK 2b — Non-linear C2ST (MLP + GridSearchCV, 3-fold)
# ---------------------------------------------------------------------------

def compute_c2st_nonlinear(
    X_ref_np: np.ndarray, X_win_np: np.ndarray, max_samples: int = 300
) -> float:
    """
    Classifier Two-Sample Test with an MLP and 3-fold cross-validated
    grid search.  Returns the best CV accuracy (chance = 0.5).
    """
    rng = np.random.RandomState(42)

    if len(X_ref_np) > max_samples:
        X_ref_sub = X_ref_np[rng.choice(len(X_ref_np), max_samples, replace=False)]
    else:
        X_ref_sub = X_ref_np

    if len(X_win_np) > max_samples:
        X_win_sub = X_win_np[rng.choice(len(X_win_np), max_samples, replace=False)]
    else:
        X_win_sub = X_win_np

    X = np.vstack([X_ref_sub, X_win_sub])
    y = np.concatenate([np.zeros(len(X_ref_sub)), np.ones(len(X_win_sub))])

    mlp = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32)],
        'alpha': [1e-4, 1e-3, 1e-2],
    }
    grid = GridSearchCV(
        mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, refit=False
    )
    grid.fit(X, y)
    return float(grid.best_score_)


# ---------------------------------------------------------------------------
# Plotting — Dual-axis chart (TASK 1.4)
# ---------------------------------------------------------------------------

def plot_dual_axis(df: pd.DataFrame, rho_cmmd: float, output_path: Path) -> None:
    """Global MMD + C-MMD on left axis, Classifier F1 on right axis."""
    # Adaptive smoothing
    sw = min(5, max(1, len(df) // 3))
    if sw >= 3:
        g_smooth = df['Global_MMD'].rolling(window=sw, center=True).mean()
        c_smooth = df['Conditional_MMD'].rolling(window=sw, center=True).mean()
        f_smooth = df['Classifier_F1'].rolling(window=sw, center=True).mean()
    else:
        g_smooth = df['Global_MMD']
        c_smooth = df['Conditional_MMD']
        f_smooth = df['Classifier_F1']

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left y-axis: MMD scores
    ax1.plot(df['Date'], g_smooth, 'b--', linewidth=2, label='Global MMD')
    ax1.plot(df['Date'], c_smooth, 'r-',  linewidth=2, label='C-MMD (Conditional)')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('MMD Score', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Classifier F1
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], f_smooth, 'g-', linewidth=2, alpha=0.8,
             label='Classifier F1')
    ax2.set_ylabel('Macro F1 Score', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)

    ax1.set_title(
        f'C-MMD vs Classifier Degradation '
        f'(Spearman \u03c1 = {rho_cmmd:+.3f})',
        fontsize=14, fontweight='bold'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Configuration
    WINDOW_SIZE = 1000
    STEP_SIZE   = 500
    REF_SIZE    = 2000
    MAX_REF_SAMPLES = 500
    WINDOW_SAMPLE   = 500

    print("=" * 70)
    print("ENHANCED DRIFT EVALUATION: C-MMD + BASELINES")
    print("=" * 70)
    print(f"Window Size : {WINDOW_SIZE}")
    print(f"Step Size   : {STEP_SIZE}")
    print(f"Reference   : first {REF_SIZE} samples (oldest)")

    # Paths
    base_path       = project_root
    embeddings_path = base_path / 'data' / 'processed' / 'amazon_embeddings.pt'
    results_dir     = base_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path  = results_dir / 'enhanced_drift_results.csv'
    plot_output_path = results_dir / 'enhanced_drift_plot.png'

    device = get_device()

    # ------------------------------------------------------------------
    # Load & split
    # ------------------------------------------------------------------
    print("\nLoading Amazon embeddings...")
    data = load_embeddings(embeddings_path, device)
    all_embeddings = data['embeddings']
    all_labels     = data['labels']
    all_dates      = data['dates']
    print(f"Total samples: {all_embeddings.shape[0]}, dim={all_embeddings.shape[1]}")

    ref_embeddings  = all_embeddings[:REF_SIZE].cpu()
    ref_labels      = all_labels[:REF_SIZE].cpu()
    stream_embeddings = all_embeddings[REF_SIZE:].cpu()
    stream_labels     = all_labels[REF_SIZE:].cpu()
    stream_dates      = all_dates[REF_SIZE:]

    print(f"Reference : {ref_embeddings.shape[0]} samples")
    print(f"Stream    : {stream_embeddings.shape[0]} samples")

    # Reference class splits
    ref_pos, ref_neg = split_by_label(ref_embeddings, ref_labels)
    print(f"Ref Pos={ref_pos.shape[0]}, Ref Neg={ref_neg.shape[0]}")

    torch.manual_seed(42)

    # Sub-sample reference for MMD
    def _subsample(tensor, n):
        if tensor.shape[0] > n:
            return tensor[torch.randperm(tensor.shape[0])[:n]]
        return tensor

    ref_pos_sample = _subsample(ref_pos, MAX_REF_SAMPLES)
    ref_neg_sample = _subsample(ref_neg, MAX_REF_SAMPLES)
    ref_all_sample = _subsample(ref_embeddings, MAX_REF_SAMPLES)

    # Numpy copies for sklearn / scipy baselines
    ref_all_np = ref_all_sample.numpy()

    # ------------------------------------------------------------------
    # TASK 1 — Train downstream sentiment classifier on reference data
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TASK 1: Training LogisticRegression on reference embeddings…")
    print("-" * 70)

    ref_emb_np = ref_embeddings.numpy()
    ref_lab_np = ref_labels.numpy()

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    clf.fit(ref_emb_np, ref_lab_np)

    train_pred = clf.predict(ref_emb_np)
    train_f1  = f1_score(ref_lab_np, train_pred, average='macro')
    train_acc = accuracy_score(ref_lab_np, train_pred)
    print(f"  Reference Macro-F1 : {train_f1:.4f}")
    print(f"  Reference Accuracy : {train_acc:.4f}")

    # ------------------------------------------------------------------
    # Sliding window loop
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Processing stream windows…")
    print("-" * 70)

    results   = []
    n_samples = stream_embeddings.shape[0]
    n_windows = (n_samples - WINDOW_SIZE) // STEP_SIZE + 1

    with torch.no_grad():
        for i in tqdm(
            range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE),
            total=n_windows,
            desc="Windows",
        ):
            window_emb    = stream_embeddings[i : i + WINDOW_SIZE]
            window_labels = stream_labels[i : i + WINDOW_SIZE]
            window_dates  = stream_dates[i : i + WINDOW_SIZE]

            avg_date = pd.to_datetime(window_dates).mean()

            # Sub-sample window for MMD
            window_emb_sample = _subsample(window_emb, WINDOW_SAMPLE)

            # ---- Metric 1: Global MMD ----
            global_mmd = mmd_rbf_chunked(
                ref_all_sample, window_emb_sample, chunk_size=128
            ).item()

            # ---- Metric 2: Conditional MMD ----
            win_pos, win_neg = split_by_label(window_emb, window_labels)
            win_pos = _subsample(win_pos, WINDOW_SAMPLE)
            win_neg = _subsample(win_neg, WINDOW_SAMPLE)

            if win_pos.shape[0] > 0 and win_neg.shape[0] > 0:
                mmd_pos = mmd_rbf_chunked(
                    ref_pos_sample, win_pos, chunk_size=128
                ).item()
                mmd_neg = mmd_rbf_chunked(
                    ref_neg_sample, win_neg, chunk_size=128
                ).item()
                cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            elif win_pos.shape[0] > 0:
                cond_mmd = mmd_rbf_chunked(
                    ref_pos_sample, win_pos, chunk_size=128
                ).item()
            elif win_neg.shape[0] > 0:
                cond_mmd = mmd_rbf_chunked(
                    ref_neg_sample, win_neg, chunk_size=128
                ).item()
            else:
                cond_mmd = float('nan')

            # ---- TASK 1: Classifier prediction on window ----
            win_emb_np = window_emb.numpy()
            win_lab_np = window_labels.numpy()
            win_pred   = clf.predict(win_emb_np)
            win_f1     = f1_score(win_lab_np, win_pred, average='macro')
            win_acc    = accuracy_score(win_lab_np, win_pred)

            # ---- TASK 2a: Energy Distance ----
            e_dist = compute_energy_distance(ref_all_np, win_emb_np)

            # ---- TASK 2b: Non-linear C2ST (MLP + GridSearchCV) ----
            c2st_acc = compute_c2st_nonlinear(ref_all_np, win_emb_np)

            results.append({
                'Date': avg_date,
                'Global_MMD': global_mmd,
                'Conditional_MMD': cond_mmd,
                'Classifier_F1': win_f1,
                'Classifier_Accuracy': win_acc,
                'Energy_Distance': e_dist,
                'C2ST_CV_Accuracy': c2st_acc,
                'Window_Start': i,
                'Window_End': i + WINDOW_SIZE,
                'Pos_Count': win_pos.shape[0],
                'Neg_Count': win_neg.shape[0],
            })

    # ------------------------------------------------------------------
    # Post-loop analysis
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Date').reset_index(drop=True)
    df_results.to_csv(csv_output_path, index=False)
    print(f"\n✅ Results saved to: {csv_output_path}")

    # First 5 rows
    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)
    print(df_results.head().to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    cols = {
        'Global_MMD':        'Global MMD',
        'Conditional_MMD':   'C-MMD',
        'Classifier_F1':     'Classifier F1',
        'Energy_Distance':   'Energy Distance',
        'C2ST_CV_Accuracy':  'C2ST CV Acc',
    }
    for col, label in cols.items():
        s = df_results[col].dropna()
        print(f"  {label:20s}  Mean={s.mean():.6f}  Std={s.std():.6f}")

    # ------------------------------------------------------------------
    # TASK 1.3 — Spearman correlations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATIONS  (drift metric vs Classifier F1)")
    print("=" * 70)

    valid = df_results.dropna(subset=['Conditional_MMD', 'Classifier_F1'])
    metrics = {
        'C-MMD':           'Conditional_MMD',
        'Global MMD':      'Global_MMD',
        'Energy Distance': 'Energy_Distance',
        'C2ST CV Acc':     'C2ST_CV_Accuracy',
    }
    spearman_results = {}
    for label, col in metrics.items():
        rho, p = spearmanr(valid[col], valid['Classifier_F1'])
        spearman_results[col] = (rho, p)
        print(f"  {label:20s}  ρ = {rho:+.4f}   p = {p:.4e}")

    rho_cmmd = spearman_results['Conditional_MMD'][0]

    # ------------------------------------------------------------------
    # TASK 1.4 — Dual-axis plot
    # ------------------------------------------------------------------
    plot_dual_axis(df_results, rho_cmmd, plot_output_path)

    print("\n" + "=" * 70)
    print("Enhanced Drift Evaluation Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
