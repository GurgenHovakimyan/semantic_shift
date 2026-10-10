"""
Experiment 19 -- Natural deployment streams and the link to downstream
performance.

Part A (natural streams).  The chronologically sorted deployment stream of
each (dataset, encoder) pair is cut into non-overlapping windows (IMDb N=500,
Amazon N=250).  For every window we compute all detectors, the macro-F1 of
the reference-trained classifier, exact permutation p-values for Global MMD
and C-MMD, and the per-class C-MMD terms.  Alarms are declared with the
calibrated threshold tau_0.05 (single window and sustained k=2 rule) and,
offline, with Benjamini-Hochberg FDR control over the permutation p-values.

Part B (downstream link).  Spearman rank correlations (with bootstrap 95% CIs)
between each drift statistic and the window-level macro-F1 are reported
(i) on the natural streams and (ii) pooled over all benchmark windows of
experiment 17 (null + all scenarios), where F1 varies substantially.

Outputs
    results/stream/stream_{key}.csv       per-window statistics and alarms
    results/stream_summary.csv            alarm fractions per detector
    results/downstream_correlations.csv   Spearman rho with CIs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.common import CONFIGS, CONTEXT_SEED, DEVICE, RESULTS_DIR, Timer, key_of, load_thresholds, log
from src.datasets import load_dataset
from src.protocol import benjamini_hochberg, permutation_test_cmmd, permutation_test_mmd, sustained_alarms
from src.statistics import ALL_DETECTORS, build_context, cmmd_per_class, compute_all

STREAM_N = {'imdb': 500, 'amazon': 250}
BENCH_N_FOR_CORR = {'imdb': 500, 'amazon': 200}
N_PERM = 200
N_BOOT = 1000


def spearman_ci(x, y, rng, n_boot=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = spearmanr(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        boots.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return rho, p, lo, hi, len(x)


def merge_save(rows, path):
    new = pd.DataFrame(rows)
    if path.exists() and not new.empty:
        old = pd.read_csv(path)
        old = old[~old['key'].isin(new['key'].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main(selected=None):
    (RESULTS_DIR / 'stream').mkdir(exist_ok=True)
    summary_rows, corr_rows = [], []
    rng = np.random.RandomState(123)

    for name, model in CONFIGS:
        key = key_of(name, model)
        if selected and key not in selected:
            continue
        n = STREAM_N[name]
        ds = load_dataset(name, model, DEVICE)
        ctx = build_context(ds, seed=CONTEXT_SEED)
        taus = load_thresholds(key, n, alpha=0.05)
        rows = []
        n_windows = ds.stream_emb.shape[0] // n
        with Timer(f'{key} natural stream ({n_windows} windows of N={n})'):
            for w in range(n_windows):
                sl = slice(w * n, (w + 1) * n)
                e, l = ds.stream_emb[sl], ds.stream_lab[sl]
                r = compute_all(ctx, e, l)
                r['p_global'] = permutation_test_mmd(ctx.ref_global, e, ctx.beta, n_perm=N_PERM, seed=w)[1]
                r['p_cmmd'] = permutation_test_cmmd(ctx.ref_by_class, e, l, ctx.beta, n_perm=N_PERM, seed=w)[1]
                for c, v in cmmd_per_class(ctx, e, l).items():
                    r[f'cmmd_class{c}'] = v
                dates = pd.to_datetime(ds.stream_dates[sl])
                r.update({'window': w, 'date_start': dates.min(), 'date_mid': dates[len(dates) // 2],
                          'date_end': dates.max(), 'pos_frac': float(l.float().mean().item())})
                rows.append(r)
        df = pd.DataFrame(rows)
        for det in ALL_DETECTORS:
            df[f'exceed_{det}'] = df[det].values > taus[det]
            df[f'alarm_k2_{det}'] = sustained_alarms(df[f'exceed_{det}'].values, k=2)
        df['bh_global'] = benjamini_hochberg(df['p_global'].values, q=0.05)
        df['bh_cmmd'] = benjamini_hochberg(df['p_cmmd'].values, q=0.05)
        df.to_csv(RESULTS_DIR / 'stream' / f'stream_{key}.csv', index=False)

        for det in ALL_DETECTORS:
            row = {'key': key, 'dataset': name, 'model': model, 'N': n, 'detector': det, 'n_windows': len(df),
                   'frac_exceed': df[f'exceed_{det}'].mean(), 'frac_alarm_k2': df[f'alarm_k2_{det}'].mean(),
                   'stat_mean': df[det].mean(), 'tau': taus[det]}
            if det == 'cmmd':
                row['frac_bh'] = df['bh_cmmd'].mean()
                row['frac_p05'] = (df['p_cmmd'] <= 0.05).mean()
            if det == 'global_mmd':
                row['frac_bh'] = df['bh_global'].mean()
                row['frac_p05'] = (df['p_global'] <= 0.05).mean()
            summary_rows.append(row)
        log(f'  {key}: F1 ref-era={1 - ctx.ref_error:.3f} (acc) | stream F1 mean={df["f1"].mean():.3f} '
            f'[{df["f1"].min():.3f}, {df["f1"].max():.3f}] | flagged cmmd={df["exceed_cmmd"].mean():.2f} '
            f'(BH {df["bh_cmmd"].mean():.2f}) global={df["exceed_global_mmd"].mean():.2f} (BH {df["bh_global"].mean():.2f})')

        # ---- Part B: correlations with downstream F1 -------------------------------
        for det in ALL_DETECTORS:
            rho, p, lo, hi, k = spearman_ci(df[det], df['f1'], rng)
            corr_rows.append({'key': key, 'setting': 'natural_stream', 'N': n, 'detector': det, 'rho': rho, 'p': p,
                              'ci_lo': lo, 'ci_hi': hi, 'n': k, 'f1_min': df['f1'].min(), 'f1_max': df['f1'].max()})
        bench_path = RESULTS_DIR / 'benchmark' / f'bench_{key}_N{BENCH_N_FOR_CORR[name]}.csv'
        if bench_path.exists():
            b = pd.read_csv(bench_path)
            null = pd.read_csv(RESULTS_DIR / 'null' / f'nulltest_{key}_N{BENCH_N_FOR_CORR[name]}.csv')
            null['scenario'] = 'null'
            pooled = pd.concat([b, null], ignore_index=True)
            for det in ALL_DETECTORS:
                rho, p, lo, hi, k = spearman_ci(pooled[det], pooled['f1'], rng)
                corr_rows.append({'key': key, 'setting': 'benchmark_pooled', 'N': BENCH_N_FOR_CORR[name], 'detector': det,
                                  'rho': rho, 'p': p, 'ci_lo': lo, 'ci_hi': hi, 'n': k,
                                  'f1_min': pooled['f1'].min(), 'f1_max': pooled['f1'].max()})
                for sc, g in b.groupby('scenario'):
                    gg = pd.concat([g, null], ignore_index=True)
                    rho, p, lo, hi, k = spearman_ci(gg[det], gg['f1'], rng, n_boot=300)
                    corr_rows.append({'key': key, 'setting': f'benchmark_{sc}', 'N': BENCH_N_FOR_CORR[name],
                                      'detector': det, 'rho': rho, 'p': p, 'ci_lo': lo, 'ci_hi': hi, 'n': k,
                                      'f1_min': gg['f1'].min(), 'f1_max': gg['f1'].max()})
        merge_save(summary_rows, RESULTS_DIR / 'stream_summary.csv')
        merge_save(corr_rows, RESULTS_DIR / 'downstream_correlations.csv')
        del ds, ctx
        torch.cuda.empty_cache()
    log('\nExperiment 19 complete.')


if __name__ == '__main__':
    main(set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None)
