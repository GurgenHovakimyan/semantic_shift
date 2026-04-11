"""
Publication-Quality Visualizations for Enhanced Drift Evaluation.

Generates four figures:
  Fig 1 — Concept Swap: C-MMD vs Classifier Degradation (dual-axis)
  Fig 2 — Amazon Sliding Window: dual-axis (MMD + F1)
  Fig 3 — Amazon Sliding Window: All Baselines comparison
  Fig 4 — Combined Spearman Correlation Comparison (bar chart)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr

# ---------- style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
})

results_dir = project_root / 'results'

# ================================================================
# Load data
# ================================================================
df_amazon = pd.read_csv(results_dir / 'enhanced_drift_results.csv', parse_dates=['Date'])
df_swap   = pd.read_csv(results_dir / 'concept_swap_classifier.csv')


# ================================================================
# FIG 1 — Concept Swap: C-MMD vs Classifier Degradation
# ================================================================
def fig1_concept_swap():
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = df_swap['Swap_Pct']

    # Left axis — MMD
    l1, = ax1.plot(x, df_swap['Global_MMD'],      'b--s', lw=2.5, ms=9, label='Global MMD')
    l2, = ax1.plot(x, df_swap['Conditional_MMD'],  'r-o',  lw=2.5, ms=9, label='C-MMD')
    ax1.set_xlabel('Label Swap Percentage (%)')
    ax1.set_ylabel('MMD Score')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.25)

    # Shade region of "high drift"
    ax1.axvspan(30, 50, color='red', alpha=0.05)

    # Right axis — F1
    ax2 = ax1.twinx()
    l3, = ax2.plot(x, df_swap['Classifier_F1'], 'g-^', lw=2.5, ms=9,
                   alpha=0.85, label='Classifier Macro-F1')
    ax2.set_ylabel('Macro F1 Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0.40, 0.95)

    # Legend
    ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
               loc='center left', framealpha=0.9)

    rho, p = spearmanr(df_swap['Conditional_MMD'], df_swap['Classifier_F1'])
    ax1.set_title(f'Concept Swap — C-MMD Tracks Classifier Degradation\n'
                  f'Spearman \u03c1 = {rho:+.3f}  (p = {p:.4f})')

    plt.tight_layout()
    out = results_dir / 'fig1_concept_swap_dual_axis.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# FIG 2 — Amazon Sliding Window: dual-axis (MMD + F1)
# ================================================================
def fig2_amazon_dual_axis():
    df = df_amazon.copy()
    sw = min(5, max(1, len(df) // 3))
    for col in ['Global_MMD', 'Conditional_MMD', 'Classifier_F1']:
        df[col + '_s'] = df[col].rolling(window=sw, center=True).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    l1, = ax1.plot(df['Date'], df['Global_MMD_s'],      'b--', lw=2, label='Global MMD')
    l2, = ax1.plot(df['Date'], df['Conditional_MMD_s'],  'r-',  lw=2, label='C-MMD')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MMD Score')
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l3, = ax2.plot(df['Date'], df['Classifier_F1_s'], 'g-', lw=2, alpha=0.8,
                   label='Classifier Macro-F1')
    ax2.set_ylabel('Macro F1 Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
               loc='best', framealpha=0.9)

    rho, p = spearmanr(df_amazon['Conditional_MMD'], df_amazon['Classifier_F1'])
    ax1.set_title(f'Amazon Sliding Window — Mild Drift, Stable Classifier\n'
                  f'Spearman \u03c1 = {rho:+.3f}  (p = {p:.4f})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = results_dir / 'fig2_amazon_dual_axis.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# FIG 3 — Amazon Baselines Comparison (all metrics on one chart)
# ================================================================
def fig3_amazon_baselines():
    df = df_amazon.copy()
    sw = min(5, max(1, len(df) // 3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    # Smoothing helper
    def smooth(s):
        return s.rolling(window=sw, center=True).mean() if sw >= 3 else s

    # Panel A — Global MMD vs C-MMD
    ax = axes[0, 0]
    ax.plot(df['Date'], smooth(df['Global_MMD']),      'b--', lw=2, label='Global MMD')
    ax.plot(df['Date'], smooth(df['Conditional_MMD']), 'r-',  lw=2, label='C-MMD')
    ax.set_ylabel('MMD Score')
    ax.set_title('(a) Global MMD vs C-MMD')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel B — Classifier F1 & Accuracy
    ax = axes[0, 1]
    ax.plot(df['Date'], smooth(df['Classifier_F1']),       'g-',  lw=2, label='Macro F1')
    ax.plot(df['Date'], smooth(df['Classifier_Accuracy']), 'g--', lw=1.5, alpha=0.6, label='Accuracy')
    ax.set_ylabel('Score')
    ax.set_title('(b) Downstream Classifier Performance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel C — Energy Distance
    ax = axes[1, 0]
    ax.plot(df['Date'], smooth(df['Energy_Distance']), 'm-', lw=2, label='Energy Distance')
    ax.plot(df['Date'], smooth(df['Global_MMD']),      'b--', lw=1.5, alpha=0.5, label='Global MMD')
    ax.set_ylabel('Score')
    ax.set_xlabel('Date')
    ax.set_title('(c) Unsupervised Baseline: Energy Distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel D — C2ST CV Accuracy
    ax = axes[1, 1]
    ax.plot(df['Date'], smooth(df['C2ST_CV_Accuracy']), 'k-o', lw=2, ms=5, label='C2ST MLP (3-fold CV)')
    ax.axhline(y=0.5, color='gray', ls=':', lw=1, label='Chance level')
    ax.set_ylabel('CV Accuracy')
    ax.set_xlabel('Date')
    ax.set_title('(d) Supervised Baseline: Non-linear C2ST')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    for ax in axes[1, :]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.suptitle('Amazon Electronics — All Drift Metrics', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = results_dir / 'fig3_amazon_all_baselines.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# FIG 4 — Spearman Correlation Comparison (Amazon vs Concept Swap)
# ================================================================
def fig4_spearman_comparison():
    # Compute correlations for Amazon
    amazon_corrs = {}
    for col, label in [('Conditional_MMD', 'C-MMD'),
                       ('Global_MMD', 'Global MMD'),
                       ('Energy_Distance', 'Energy Dist.'),
                       ('C2ST_CV_Accuracy', 'C2ST MLP')]:
        rho, _ = spearmanr(df_amazon[col], df_amazon['Classifier_F1'])
        amazon_corrs[label] = rho

    # Compute correlations for Concept Swap
    swap_corrs = {}
    for col, label in [('Conditional_MMD', 'C-MMD'),
                       ('Global_MMD', 'Global MMD'),
                       ('Energy_Distance', 'Energy Dist.'),
                       ('C2ST_CV_Accuracy', 'C2ST MLP')]:
        rho, _ = spearmanr(df_swap[col], df_swap['Classifier_F1'])
        swap_corrs[label] = rho

    labels = list(amazon_corrs.keys())
    amazon_vals = [amazon_corrs[l] for l in labels]
    swap_vals   = [swap_corrs[l] for l in labels]

    # Replace NaN with 0 for display
    amazon_vals = [v if np.isfinite(v) else 0.0 for v in amazon_vals]
    swap_vals   = [v if np.isfinite(v) else 0.0 for v in swap_vals]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width/2, amazon_vals, width, label='Amazon (Mild Drift)',
                   color='#4C72B0', edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, swap_vals,   width, label='Concept Swap (Strong Drift)',
                   color='#DD8452', edgecolor='white', linewidth=0.8)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 * np.sign(h),
                f'{h:+.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 * np.sign(h),
                f'{h:+.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Spearman \u03c1 (Metric vs Classifier F1)')
    ax.set_title('Spearman Correlation: Drift Metric ↔ Downstream F1', fontweight='bold')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='gray', ls='-', lw=0.8)
    ax.set_ylim(-1.15, 0.45)
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    out = results_dir / 'fig4_spearman_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# FIG 5 — Concept Swap: Metric Sensitivity Panel
# ================================================================
def fig5_swap_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = df_swap['Swap_Pct']

    # Panel A — MMD scores
    ax = axes[0]
    ax.plot(x, df_swap['Global_MMD'],      'b--s', lw=2, ms=8, label='Global MMD')
    ax.plot(x, df_swap['Conditional_MMD'], 'r-o',  lw=2, ms=8, label='C-MMD')
    ax.set_xlabel('Label Swap (%)')
    ax.set_ylabel('MMD Score')
    ax.set_title('(a) MMD Scores')
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)

    # Panel B — Classifier metrics
    ax = axes[1]
    ax.plot(x, df_swap['Classifier_F1'],       'g-^', lw=2, ms=8, label='Macro F1')
    ax.plot(x, df_swap['Classifier_Accuracy'], 'g--D', lw=1.5, ms=6, alpha=0.6, label='Accuracy')
    ax.set_xlabel('Label Swap (%)')
    ax.set_ylabel('Score')
    ax.set_title('(b) Classifier Degradation')
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)

    # Panel C — Baselines (E-Dist & C2ST)
    ax = axes[2]
    ax.plot(x, df_swap['Energy_Distance'],  'm-v', lw=2, ms=8, label='Energy Dist.')
    ax.plot(x, df_swap['C2ST_CV_Accuracy'], 'k-P', lw=2, ms=8, label='C2ST MLP')
    ax.axhline(y=0.5, color='gray', ls=':', lw=1, label='Chance (C2ST)')
    ax.set_xlabel('Label Swap (%)')
    ax.set_ylabel('Score')
    ax.set_title('(c) Marginal Baselines (blind)')
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)

    fig.suptitle('Concept Swap — Only C-MMD Detects Semantic Drift',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = results_dir / 'fig5_swap_sensitivity_panel.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    fig1_concept_swap()
    fig2_amazon_dual_axis()
    fig3_amazon_baselines()
    fig4_spearman_comparison()
    fig5_swap_sensitivity()
    print("\nAll figures saved to:", results_dir)
