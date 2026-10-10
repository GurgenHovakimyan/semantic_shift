"""Print the numbers quoted in the manuscript text (helper for writing the Results section)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)


def section(title):
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)


def main():
    section('Reference-era (null) F1 and error per key / N')
    for p in sorted((RESULTS_DIR / 'null').glob('nulltest_*.csv')):
        d = pd.read_csv(p)
        print(f'{p.stem:40s} f1={d["f1"].mean():.3f}  error={d["error_rate"].mean():.3f}  pos_frac~ n={len(d)}')

    section('Permutation type-I')
    print(pd.read_csv(RESULTS_DIR / 'permutation_type1.csv').round(3).to_string(index=False))

    if (RESULTS_DIR / 'benchmark_summary.csv').exists():
        section('Power summary (cmmd, global_mmd, bbsd_ks, joint_mmd, centroid_shift, cond_c2st, error_rate)')
        s = pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv')
        piv = s.pivot_table(index=['key', 'N', 'scenario', 'lam'], columns='detector', values='power_mean')
        cols = [c for c in ['global_mmd', 'energy', 'c2st_mlp', 'bbsd_ks', 'cmmd', 'cond_energy', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'prior_shift', 'error_rate'] if c in piv.columns]
        print(piv[cols].round(2).to_string())
        section('AUROC summary (cmmd / global / joint / centroid)')
        piv = s.pivot_table(index=['key', 'N', 'scenario', 'lam'], columns='detector', values='auroc')
        print(piv[[c for c in ['global_mmd', 'bbsd_ks', 'cmmd', 'centroid_shift', 'joint_mmd', 'error_rate'] if c in piv.columns]].round(3).to_string())
        section('F1 per scenario/level (IMDb distilbert N=500)')
        print(s[(s['key'] == 'imdb_distilbert') & (s['N'] == 500) & (s['detector'] == 'cmmd')][['scenario', 'lam', 'f1_mean', 'f1_std', 'pos_frac_mean']].round(3).to_string(index=False))

    if (RESULTS_DIR / 'delay_summary.csv').exists():
        section('Delay summary (k2): median delay, detection rate within horizon, pre-onset false alarms')
        d = pd.read_csv(RESULTS_DIR / 'delay_summary.csv')
        for rule in ['k1', 'k2']:
            sub = d[d['rule'] == rule]
            g = sub.groupby(['key', 'N', 'drift', 'onset_type', 'detector']).agg(
                med_delay=('delay', lambda v: np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan),
                mean_delay=('delay', lambda v: np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan),
                det_rate=('delay', lambda v: np.mean(np.isfinite(v))),
                pre_fa=('pre_onset_alarms', 'mean'), f1_pre=('f1_pre', 'mean'), f1_post=('f1_post', 'mean')).reset_index()
            g = g[g['detector'].isin(['global_mmd', 'bbsd_ks', 'c2st_mlp', 'cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate'])]
            print(f'--- rule {rule}')
            print(g.round(2).to_string(index=False))

    if (RESULTS_DIR / 'stream_summary.csv').exists():
        section('Stream summary')
        print(pd.read_csv(RESULTS_DIR / 'stream_summary.csv').round(3).to_string(index=False))
        for p in sorted((RESULTS_DIR / 'stream').glob('stream_*.csv')):
            s = pd.read_csv(p, parse_dates=['date_mid'])
            print(f'{p.stem}: windows={len(s)} f1 mean={s["f1"].mean():.3f} min={s["f1"].min():.3f} max={s["f1"].max():.3f} '
                  f'pos_frac mean={s["pos_frac"].mean():.3f} [{s["pos_frac"].min():.2f},{s["pos_frac"].max():.2f}] '
                  f'cmmd mean={s["cmmd"].mean():.5f} global mean={s["global_mmd"].mean():.5f} '
                  f'class0 mean={s.get("cmmd_class0", pd.Series([np.nan])).mean():.5f} class1 mean={s.get("cmmd_class1", pd.Series([np.nan])).mean():.5f} '
                  f'p_cmmd<=.05={np.mean(s["p_cmmd"] <= 0.05):.2f} p_global<=.05={np.mean(s["p_global"] <= 0.05):.2f}')
            # by year
            s['year'] = s['date_mid'].dt.year
            print(s.groupby('year')[['cmmd', 'global_mmd', 'f1', 'pos_frac', 'exceed_cmmd', 'exceed_global_mmd', 'exceed_error_rate']].mean().round(4).to_string())

    if (RESULTS_DIR / 'downstream_correlations.csv').exists():
        section('Downstream correlations')
        c = pd.read_csv(RESULTS_DIR / 'downstream_correlations.csv')
        print(c[c['detector'].isin(['global_mmd', 'bbsd_ks', 'c2st_mlp', 'cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate'])].round(3).to_string(index=False))

    if (RESULTS_DIR / 'encoder_scale_check.csv').exists():
        section('Encoder scale check')
        print(pd.read_csv(RESULTS_DIR / 'encoder_scale_check.csv').round(5).to_string(index=False))

    if (RESULTS_DIR / 'bandwidth_ablation.csv').exists():
        section('Bandwidth ablation')
        a = pd.read_csv(RESULTS_DIR / 'bandwidth_ablation.csv')
        print(a.pivot_table(index=['N', 'variant'], columns='scenario', values=['power', 'auroc']).round(3).to_string())

    if (RESULTS_DIR / 'speed_benchmark_v2.csv').exists():
        section('Speed')
        sp = pd.read_csv(RESULTS_DIR / 'speed_benchmark_v2.csv')
        print(sp.pivot_table(index=['detector', 'device'], columns='N', values='ms_median').round(2).to_string())


if __name__ == '__main__':
    main()
