"""
Experiment 22 -- Publication figures for the revised manuscript.

Reads the CSV outputs of experiments 16-21 and writes PNG figures to
``results/figures``.  Figures are also copied to the manuscript folder by the
helper at the bottom when ``--copy <dir>`` is given.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.common import RESULTS_DIR
from src.statistics import PRETTY

FIG_DIR = RESULTS_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE = sns.color_palette('colorblind', 10)
COLORS = {
    'cmmd': '#B2182B', 'global_mmd': PALETTE[0], 'bbsd_ks': PALETTE[2], 'energy': PALETTE[9],
    'c2st_linear': PALETTE[7], 'c2st_mlp': PALETTE[4], 'cond_c2st': PALETTE[1], 'cond_energy': PALETTE[8],
    'centroid_shift': PALETTE[5], 'joint_mmd': PALETTE[6], 'error_rate': 'black', 'prior_shift': 'grey',
}
LW = {d: 1.4 for d in COLORS}
LW['cmmd'] = 2.4
KEY_TITLE = {'imdb_distilbert': 'IMDb / DistilBERT', 'imdb_roberta': 'IMDb / RoBERTa',
             'amazon_distilbert': 'Amazon / DistilBERT', 'amazon_roberta': 'Amazon / RoBERTa'}
STYLES = {
    'cmmd': '-', 'global_mmd': '--', 'bbsd_ks': '--', 'energy': '--', 'c2st_linear': '--', 'c2st_mlp': '--',
    'cond_c2st': '-', 'cond_energy': '-', 'centroid_shift': '-', 'joint_mmd': '-', 'error_rate': ':', 'prior_shift': ':',
}
MAIN_DETECTORS = ['cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'cond_energy',
                  'global_mmd', 'bbsd_ks', 'c2st_mlp', 'energy', 'error_rate']
SCENARIO_TITLE = {
    'natural': 'Natural temporal drift', 'attenuation': 'Polarity attenuation', 'boundary': 'Annotation-boundary drift',
    'asym_temporal': 'Asymmetric temporal drift', 'label_shift': 'Pure label (prior) shift', 'label_noise': 'Label noise (concept swap)',
}


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path.name}')


# ---------------------------------------------------------------------------
def fig_null_calibration():
    thr = pd.read_csv(RESULTS_DIR / 'calibration_thresholds.csv')
    fpr = pd.read_csv(RESULTS_DIR / 'calibration_fpr.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={'width_ratios': [1.2, 1.2, 1.6]})
    key = 'imdb_distilbert'
    for ax, det in zip(axes[:2], ['cmmd', 'global_mmd']):
        data, labels = [], []
        for n in [100, 200, 500, 1000]:
            p = RESULTS_DIR / 'null' / f'null_{key}_N{n}.csv'
            if p.exists():
                data.append(pd.read_csv(p)[det].values)
                labels.append(str(n))
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[det]); pc.set_alpha(0.5)
        for i, n in enumerate(labels):
            t = thr[(thr['key'] == key) & (thr['N'] == int(n)) & (thr['detector'] == det) & np.isclose(thr['alpha'], 0.05)]['tau'].iloc[0]
            ax.hlines(t, i + 0.6, i + 1.4, colors='k', linestyles='--', lw=1)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        ax.set_xlabel('window size $N$'); ax.set_title(f'{PRETTY[det]}: null distribution (IMDb)')
        ax.set_ylabel('statistic (unbiased MMD$^2$)')
    ax = axes[2]
    sub = fpr[(fpr['key'] == key) & (fpr['N'] == 200) & np.isclose(fpr['alpha'], 0.05)].set_index('detector')
    dets = [d for d in MAIN_DETECTORS if d in sub.index]
    ax.bar(range(len(dets)), sub.loc[dets, 'fpr_mean'], yerr=sub.loc[dets, 'fpr_std'], color=[COLORS[d] for d in dets], alpha=0.85, capsize=3)
    ax.axhline(0.05, color='k', ls='--', lw=1, label='nominal $\\alpha=0.05$')
    ax.set_xticks(range(len(dets))); ax.set_xticklabels([PRETTY[d] for d in dets], rotation=45, ha='right')
    ax.set_ylabel('false-positive rate'); ax.set_title('Fresh null windows, $N=200$ (10 runs)'); ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, 'fig_null_calibration.png')


def fig_power(key, n, name):
    s = pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv')
    s = s[(s['key'] == key) & (s['N'] == n)]
    scenarios = [sc for sc in ['attenuation', 'boundary', 'asym_temporal', 'label_noise', 'label_shift', 'natural'] if sc in s['scenario'].unique()]
    ncol = 3 if len(scenarios) > 4 else 2
    nrow = int(np.ceil(len(scenarios) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
    for ax, sc in zip(axes.flat, scenarios):
        g = s[s['scenario'] == sc]
        if sc == 'natural':
            dets = [d for d in MAIN_DETECTORS if d in g['detector'].values]
            vals = g.set_index('detector').loc[dets]
            ax.bar(range(len(dets)), vals['power_mean'], yerr=vals['power_std'], color=[COLORS[d] for d in dets], alpha=0.85, capsize=2)
            ax.set_xticks(range(len(dets))); ax.set_xticklabels([PRETTY[d] for d in dets], rotation=60, ha='right', fontsize=7)
            ax.set_ylabel('detection rate')
        else:
            for det in MAIN_DETECTORS:
                gg = g[g['detector'] == det].sort_values('lam')
                if gg.empty:
                    continue
                ax.errorbar(gg['lam'], gg['power_mean'], yerr=gg['power_std'], color=COLORS[det], ls=STYLES[det],
                            marker='o', ms=3, lw=LW[det], capsize=2, label=PRETTY[det])
            ax.set_xlabel('drift level $\\lambda$'); ax.set_ylabel('power at 5% FPR')
            ax.set_ylim(-0.03, 1.03); ax.axhline(0.05, color='grey', lw=0.7, ls=':')
        ax.set_title(SCENARIO_TITLE[sc])
    for ax in axes.flat[len(scenarios):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(f'{KEY_TITLE.get(key, key)}, $N={n}$', y=1.01)
    fig.tight_layout()
    savefig(fig, name)


def fig_delay():
    d = pd.read_csv(RESULTS_DIR / 'delay_summary.csv')
    d = d[(d['key'] == 'imdb_distilbert') & (d['N'] == 200) & (d['rule'] == 'k2')]
    drifts = [x for x in ['asym_temporal', 'boundary', 'attenuation', 'label_noise'] if x in d['drift'].unique()]
    dets = ['cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'global_mmd', 'bbsd_ks', 'c2st_mlp', 'error_rate']
    fig, axes = plt.subplots(2, len(drifts), figsize=(3.6 * len(drifts), 6.0), squeeze=False, sharey=True)
    horizon = 20
    for j, drift in enumerate(drifts):
        for i, onset in enumerate(['abrupt', 'gradual']):
            ax = axes[i, j]
            g = d[(d['drift'] == drift) & (d['onset_type'] == onset)]
            means, miss = [], []
            for det in dets:
                v = g[g['detector'] == det]['delay'].values
                v = np.where(np.isfinite(v), v, horizon)   # censor misses at the horizon
                means.append(v)
                miss.append(np.mean(~np.isfinite(g[g['detector'] == det]['delay'].values)))
            bp = ax.boxplot(means, widths=0.6, patch_artist=True, showfliers=False, medianprops={'color': 'k'})
            for patch, det in zip(bp['boxes'], dets):
                patch.set_facecolor(COLORS[det]); patch.set_alpha(0.7)
            for k, m in enumerate(miss):
                if m > 0:
                    ax.text(k + 1, horizon + 1.0, f'{int(round(100 * m))}%', ha='center', va='bottom', fontsize=6.5, color='firebrick')
            ax.axhline(horizon, color='grey', lw=0.6, ls=':')
            ax.set_xticks(range(1, len(dets) + 1)); ax.set_xticklabels([PRETTY[x] for x in dets], rotation=60, ha='right', fontsize=7)
            ax.set_ylim(-0.5, horizon + 4.5)
            ax.set_title(f'{SCENARIO_TITLE[drift]}\n({onset} onset)', fontsize=9.5)
            if j == 0:
                ax.set_ylabel('detection delay (windows)')
    fig.suptitle('IMDb / DistilBERT, $N=200$, sustained rule $k=2$. Misses are censored at the 20-window horizon (dotted); the red percentage is the miss rate.', y=1.0, fontsize=9.5)
    fig.tight_layout()
    savefig(fig, 'fig_detection_delay.png')


def fig_natural_streams():
    thr = pd.read_csv(RESULTS_DIR / 'calibration_thresholds.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2))
    for row, (key, n, title) in enumerate([('imdb_distilbert', 500, 'IMDb (2013-2017 stream vs 2005-2010 reference), $N=500$'),
                                            ('amazon_distilbert', 250, 'Amazon Electronics (2017-2023 stream vs 2001-2017 reference), $N=250$')]):
        p = RESULTS_DIR / 'stream' / f'stream_{key}.csv'
        if not p.exists():
            continue
        s = pd.read_csv(p, parse_dates=['date_mid'])
        ax = axes[row, 0]
        t = thr[(thr['key'] == key) & (thr['N'] == n) & np.isclose(thr['alpha'], 0.05)].set_index('detector')['tau']
        ax.plot(s['date_mid'], s['global_mmd'], color=COLORS['global_mmd'], lw=1.2, ls='--', label='Global MMD')
        ax.plot(s['date_mid'], s['cmmd'], color=COLORS['cmmd'], lw=1.4, label='C-MMD')
        ax.axhline(t['cmmd'], color=COLORS['cmmd'], lw=0.9, ls=':', label='$\\tau_{0.05}$ (C-MMD)')
        ax.axhline(t['global_mmd'], color=COLORS['global_mmd'], lw=0.9, ls=':', label='$\\tau_{0.05}$ (Global)')
        flagged = s[s['bh_cmmd']]
        ax.scatter(flagged['date_mid'], flagged['cmmd'], color=COLORS['cmmd'], s=10, zorder=5, label='BH-significant (C-MMD)')
        ax.set_ylabel('unbiased MMD$^2$'); ax.set_title(title, fontsize=9.5); ax.legend(ncol=3, loc='upper left', fontsize=7)
        ax2 = axes[row, 1]
        ax2.plot(s['date_mid'], s['f1'], color='black', lw=1.2, label='macro-F1 of reference classifier')
        ax2.plot(s['date_mid'], s['pos_frac'], color='grey', lw=1.0, ls='--', label='positive fraction in window')
        ax2.set_ylabel('F1 / class proportion'); ax2.set_title('Downstream performance and class prior', fontsize=9.5)
        ax2.legend(loc='lower left', fontsize=7)
        for a in (ax, ax2):
            a.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    savefig(fig, 'fig_natural_streams.png')


def fig_speed():
    s = pd.read_csv(RESULTS_DIR / 'speed_benchmark_v2.csv')
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for det in ['cmmd', 'global_mmd', 'cond_c2st', 'c2st_linear', 'c2st_mlp', 'bbsd_ks', 'energy', 'centroid_shift', 'joint_mmd']:
        g = s[(s['detector'] == det) & (s['device'] == 'cpu')].sort_values('N')
        if g.empty:
            continue
        ax.plot(g['N'], g['ms_median'], color=COLORS[det], ls=STYLES[det], marker='o', ms=3.5, lw=1.4, label=f'{PRETTY[det]} (CPU)')
    g = s[(s['detector'] == 'cmmd') & (s['device'] == 'cuda')].sort_values('N')
    if not g.empty:
        ax.plot(g['N'], g['ms_median'], color=COLORS['cmmd'], ls='-', marker='s', ms=3.5, lw=1.0, alpha=0.6, label='C-MMD (GPU)')
    g = s[(s['detector'] == 'cmmd_permutation200') & (s['device'] == 'cuda')].sort_values('N')
    if not g.empty:
        ax.plot(g['N'], g['ms_median'], color=COLORS['cmmd'], ls='-.', marker='^', ms=3.5, lw=1.0, alpha=0.6, label='C-MMD + 200 permutations (GPU)')
    g = s[(s['detector'] == 'distilbert_embedding') & (s['device'] == 'cuda')].sort_values('N')
    if not g.empty:
        ax.plot(g['N'], g['ms_median'], color='k', ls=':', marker='x', ms=5, lw=1.2, label='DistilBERT embedding of the window (GPU)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('window size $N$'); ax.set_ylabel('wall-clock time per window (ms)')
    ax.set_xticks([100, 200, 500, 1000, 2000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=7, ncol=2, loc='upper left'); ax.grid(True, which='both', alpha=0.25)
    ax.set_title('Cost per monitored window (reference sample $M=1000$, single CPU thread)')
    fig.tight_layout()
    savefig(fig, 'fig_speed.png')


def fig_concept_swap():
    thr = pd.read_csv(RESULTS_DIR / 'calibration_thresholds.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (key, title) in zip(axes, [('imdb_distilbert', 'IMDb / DistilBERT'), ('imdb_roberta', 'IMDb / RoBERTa')]):
        for n, marker in [(200, 'o'), (500, 's')]:
            p = RESULTS_DIR / 'benchmark' / f'bench_{key}_N{n}.csv'
            if not p.exists():
                continue
            b = pd.read_csv(p)
            null = pd.read_csv(RESULTS_DIR / 'null' / f'null_{key}_N{n}.csv')
            for det, lbl in [('cmmd', 'C-MMD'), ('global_mmd', 'Global MMD')]:
                g = b[b['scenario'] == 'label_noise'].groupby('lam')[det].agg(['mean', 'std']).reset_index()
                g = pd.concat([pd.DataFrame({'lam': [0.0], 'mean': [null[det].mean()], 'std': [null[det].std()]}), g])
                ax.errorbar(100 * g['lam'], g['mean'], yerr=g['std'], color=COLORS[det], marker=marker, ms=4, lw=1.3,
                            ls=STYLES[det], capsize=2, label=f'{lbl}, $N={n}$')
            t = thr[(thr['key'] == key) & (thr['N'] == n) & (thr['detector'] == 'cmmd') & np.isclose(thr['alpha'], 0.05)]['tau'].iloc[0]
            ax.axhline(t, color=COLORS['cmmd'], lw=0.8, ls=':', alpha=0.7)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_xlabel('fraction of flipped labels (%)'); ax.set_ylabel('unbiased MMD$^2$ (mean $\\pm$ s.d., 100 windows)')
        ax.set_title(title); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig(fig, 'fig_concept_swap.png')


def fig_downstream():
    corr = pd.read_csv(RESULTS_DIR / 'downstream_correlations.csv')
    key = 'imdb_distilbert'
    b = pd.read_csv(RESULTS_DIR / 'benchmark' / f'bench_{key}_N500.csv')
    null = pd.read_csv(RESULTS_DIR / 'null' / f'nulltest_{key}_N500.csv'); null['scenario'] = 'null'
    pooled = pd.concat([b, null], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    order = ['null', 'natural', 'attenuation', 'boundary', 'asym_temporal', 'label_shift', 'label_noise']
    pal = dict(zip(order, sns.color_palette('colorblind', len(order))))
    for sc in order:
        g = pooled[pooled['scenario'] == sc]
        ax.scatter(g['cmmd'], g['f1'], s=7, alpha=0.55, color=pal[sc], label=SCENARIO_TITLE.get(sc, 'Null (reference era)'))
    ax.set_xlabel('C-MMD (unbiased MMD$^2$)'); ax.set_ylabel('macro-F1 of reference classifier')
    ax.set_title('IMDb / DistilBERT, $N=500$: all benchmark windows'); ax.legend(fontsize=6.5, markerscale=2)
    ax = axes[1]
    dets = [d for d in MAIN_DETECTORS if d != 'error_rate']
    for i, (setting, lbl, off) in enumerate([('natural_stream', 'natural stream', -0.2), ('benchmark_pooled', 'pooled benchmark', 0.2)]):
        sub = corr[(corr['key'] == key) & (corr['setting'] == setting)].set_index('detector').reindex(dets)
        err = np.vstack([sub['rho'] - sub['ci_lo'], sub['ci_hi'] - sub['rho']])
        ax.bar(np.arange(len(dets)) + off, sub['rho'], width=0.38, yerr=err, capsize=2, alpha=0.9 if i else 0.45,
               color=[COLORS[d] for d in dets], edgecolor='k', linewidth=0.6, hatch='' if i else '///', label=lbl)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(range(len(dets))); ax.set_xticklabels([PRETTY[d] for d in dets], rotation=45, ha='right')
    ax.set_ylabel('Spearman $\\rho$ with macro-F1')
    ax.set_title('Rank correlation with F1 (95% bootstrap CI)\nhatched: natural stream ($n$=331); solid: pooled benchmark ($n$=2500)', fontsize=9)
    fig.tight_layout()
    savefig(fig, 'fig_downstream_link.png')


def fig_bandwidth():
    a = pd.read_csv(RESULTS_DIR / 'bandwidth_ablation.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, n in zip(axes, [200, 500]):
        piv = a[a['N'] == n].pivot(index='scenario', columns='variant', values='power')
        piv = piv.reindex(['label_noise', 'attenuation', 'asym_temporal', 'boundary', 'label_shift'])
        piv.plot.bar(ax=ax, width=0.8, color=sns.color_palette('colorblind', piv.shape[1]), alpha=0.9, legend=(n == 200))
        ax.set_xticklabels([SCENARIO_TITLE[s] for s in piv.index], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('power at 5% FPR'); ax.set_title(f'IMDb / DistilBERT, $N={n}$'); ax.set_xlabel('')
        ax.axhline(0.05, color='grey', lw=0.7, ls=':')
    axes[0].legend(fontsize=6.5, title='C-MMD variant', title_fontsize=7)
    fig.tight_layout()
    savefig(fig, 'fig_bandwidth_ablation.png')


def fig_power_vs_N():
    s = pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv')
    s = s[(s['key'] == 'imdb_distilbert') & (s['lam'].isin([1.0, 0.3]))]
    scenarios = ['attenuation', 'boundary', 'asym_temporal', 'label_noise']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
    for ax, sc in zip(axes, scenarios):
        lam = 0.3 if sc == 'label_noise' else 1.0
        g = s[(s['scenario'] == sc) & np.isclose(s['lam'], lam)]
        for det in MAIN_DETECTORS:
            gg = g[g['detector'] == det].sort_values('N')
            if gg.empty:
                continue
            ax.errorbar(gg['N'], gg['power_mean'], yerr=gg['power_std'], color=COLORS[det], ls=STYLES[det], marker='o', ms=3, lw=LW[det], capsize=2, label=PRETTY[det])
        ax.set_xscale('log'); ax.set_xticks([200, 500, 1000]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f'{SCENARIO_TITLE[sc]} ($\\lambda={lam}$)', fontsize=9.5); ax.set_xlabel('window size $N$')
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel('power at 5% FPR')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    savefig(fig, 'fig_power_vs_N.png')


def main():
    steps = [
        ('null calibration', fig_null_calibration),
        ('power IMDb N=500', lambda: fig_power('imdb_distilbert', 500, 'fig_power_imdb_N500.png')),
        ('power IMDb N=200', lambda: fig_power('imdb_distilbert', 200, 'fig_power_imdb_N200.png')),
        ('power IMDb RoBERTa N=200', lambda: fig_power('imdb_roberta', 200, 'fig_power_imdb_roberta_N200.png')),
        ('power Amazon N=200', lambda: fig_power('amazon_distilbert', 200, 'fig_power_amazon_N200.png')),
        ('power vs N', fig_power_vs_N),
        ('delay', fig_delay),
        ('natural streams', fig_natural_streams),
        ('speed', fig_speed),
        ('concept swap', fig_concept_swap),
        ('downstream', fig_downstream),
        ('bandwidth', fig_bandwidth),
    ]
    for label, fn in steps:
        try:
            fn()
        except Exception as exc:  # keep going so that partial results still produce figures
            print(f'  [{label}] skipped: {type(exc).__name__}: {exc}')
    if len(sys.argv) > 2 and sys.argv[1] == '--copy':
        dest = Path(sys.argv[2])
        for p in FIG_DIR.glob('fig_*.png'):
            shutil.copy(p, dest / p.name)
        print(f'  copied figures to {dest}')


if __name__ == '__main__':
    main()
