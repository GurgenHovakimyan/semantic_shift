"""
Experiment 16 -- Null calibration, alarm thresholds and false-positive control.

For every (dataset, encoder) pair and window size N:

1. Calibration: draw ``N_NULL_CAL`` drift-free windows from the held-out
   *calibration* pool of the reference period, evaluate every detector, and
   store the empirical null distribution.  The alarm threshold tau_alpha is
   the (1 - alpha) quantile of this distribution.
2. False-positive control: in ``N_RUNS`` independent runs draw ``N_NULL_TEST``
   fresh windows from the disjoint *test* pool and record how often each
   detector exceeds tau_alpha (per-window FPR) and how often the sustained
   k=2 rule fires.  Mean and standard deviation across runs are reported.
3. Permutation test type-I error: on the same fresh null windows compute exact
   permutation p-values for Global MMD and C-MMD and report the rejection
   rate at 5% and a KS test of p-value uniformity.

Outputs
    results/null/null_{key}_N{N}.csv        null statistics (one row per window)
    results/calibration_thresholds.csv      tau_alpha for every detector
    results/calibration_fpr.csv             FPR (mean, std over runs)
    results/permutation_type1.csv           permutation-test type-I error
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest

from experiments.common import (ALPHAS, CONFIGS, CONTEXT_SEED, DEVICE, N_GRID, N_NULL_CAL, N_NULL_TEST, N_RUNS,
                                RESULTS_DIR, Timer, key_of, log, null_path, thresholds_path)
from src.datasets import describe, load_dataset
from src.protocol import (draw_null_window, empirical_fpr, permutation_test_cmmd, permutation_test_mmd,
                          sustained_alarms, threshold_from_null)
from src.statistics import ALL_DETECTORS, build_context, compute_all

N_PERM = 200
N_PERM_WINDOWS = 100   # null windows used for the permutation type-I check


def evaluate_windows(ctx, pool_emb, pool_lab, n, n_windows, rng):
    rows = []
    for _ in range(n_windows):
        e, l = draw_null_window(pool_emb, pool_lab, n, rng)
        rows.append(compute_all(ctx, e, l))
    return pd.DataFrame(rows)


def merge_save(rows, path):
    """Replace rows of the processed keys in an existing CSV (so that configs can be re-run one at a time)."""
    new = pd.DataFrame(rows)
    if path.exists() and not new.empty:
        old = pd.read_csv(path)
        old = old[~old['key'].isin(new['key'].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main(selected=None, pool_modes=('default',)):
    (RESULTS_DIR / 'null').mkdir(exist_ok=True)
    thr_rows, fpr_rows, perm_rows = [], [], []

    for name, model in CONFIGS:
        base_key = key_of(name, model)
        if selected and base_key not in selected:
            continue
        for mode in pool_modes:
            # 'separate' forces disjoint calibration/test pools (reported with suffix _sep for Amazon)
            shared = None if mode == 'default' else False
            ds = load_dataset(name, model, DEVICE, shared_null_pool=shared)
            key = base_key if mode == 'default' or name == 'imdb' else base_key + '_sep'
            if mode != 'default' and name == 'imdb':
                continue
            log('\n' + describe(ds) + f'  [key={key}]')
            run_config(ds, key, name, model, thr_rows, fpr_rows, perm_rows)
            del ds
            torch.cuda.empty_cache()
            # save after every configuration so that downstream experiments can start early
            merge_save(thr_rows, thresholds_path())
            merge_save(fpr_rows, RESULTS_DIR / 'calibration_fpr.csv')
            merge_save(perm_rows, RESULTS_DIR / 'permutation_type1.csv')

    log('\nExperiment 16 complete.')


def run_config(ds, key, name, model, thr_rows, fpr_rows, perm_rows):
    ctx = build_context(ds, seed=CONTEXT_SEED)
    log(f'  bandwidth beta={ctx.beta:.3f}  reference error={ctx.ref_error:.3f}')
    cal_emb, cal_lab, _ = ds.pool('cal')
    test_emb, test_lab, _ = ds.pool('test')

    for n in N_GRID[name]:
        # ---- 1. calibration ------------------------------------------------
        with Timer(f'{key} N={n} calibration'):
            rng = np.random.RandomState(1000 + n)
            null_df = evaluate_windows(ctx, cal_emb, cal_lab, n, N_NULL_CAL, rng)
            null_df.to_csv(null_path(key, n), index=False)
        taus = {}
        for det in ALL_DETECTORS:
            for alpha in ALPHAS:
                tau = threshold_from_null(null_df[det].values, alpha)
                taus[(det, alpha)] = tau
                thr_rows.append({'key': key, 'dataset': name, 'model': model, 'N': n, 'detector': det,
                                 'alpha': alpha, 'tau': tau, 'null_mean': null_df[det].mean(),
                                 'null_std': null_df[det].std(), 'beta': ctx.beta})

        # ---- 2. false-positive control on fresh windows ---------------------
        with Timer(f'{key} N={n} FPR ({N_RUNS} runs x {N_NULL_TEST} windows)'):
            per_run = {det: {a: [] for a in ALPHAS} for det in ALL_DETECTORS}
            per_run_k2 = {det: [] for det in ALL_DETECTORS}
            all_test = []
            for run in range(N_RUNS):
                rng = np.random.RandomState(5000 + 100 * n + run)
                test_df = evaluate_windows(ctx, test_emb, test_lab, n, N_NULL_TEST, rng)
                test_df['run'] = run
                all_test.append(test_df)
                for det in ALL_DETECTORS:
                    for alpha in ALPHAS:
                        per_run[det][alpha].append(empirical_fpr(test_df[det].values, taus[(det, alpha)]))
                    exceed = test_df[det].values > taus[(det, 0.05)]
                    per_run_k2[det].append(float(sustained_alarms(exceed, k=2).mean()))
            pd.concat(all_test).to_csv(RESULTS_DIR / 'null' / f'nulltest_{key}_N{n}.csv', index=False)
            for det in ALL_DETECTORS:
                for alpha in ALPHAS:
                    v = np.array(per_run[det][alpha])
                    fpr_rows.append({'key': key, 'dataset': name, 'model': model, 'N': n, 'detector': det,
                                     'alpha': alpha, 'fpr_mean': v.mean(), 'fpr_std': v.std(ddof=1),
                                     'fpr_k2_mean': np.mean(per_run_k2[det]) if alpha == 0.05 else np.nan})

        # ---- 3. permutation-test type-I error -------------------------------
        with Timer(f'{key} N={n} permutation type-I ({N_PERM_WINDOWS} windows)'):
            rng = np.random.RandomState(9000 + n)
            p_g, p_c = [], []
            for _ in range(N_PERM_WINDOWS):
                e, l = draw_null_window(test_emb, test_lab, n, rng)
                p_g.append(permutation_test_mmd(ctx.ref_global, e, ctx.beta, n_perm=N_PERM, seed=int(rng.randint(1 << 30)))[1])
                p_c.append(permutation_test_cmmd(ctx.ref_by_class, e, l, ctx.beta, n_perm=N_PERM, seed=int(rng.randint(1 << 30)))[1])
            for det, pv in [('global_mmd', p_g), ('cmmd', p_c)]:
                pv = np.array(pv)
                perm_rows.append({'key': key, 'dataset': name, 'model': model, 'N': n, 'detector': det,
                                  'reject_05': float(np.mean(pv <= 0.05)), 'reject_01': float(np.mean(pv <= 0.01)),
                                  'ks_uniform_p': float(kstest(pv, 'uniform').pvalue), 'n_windows': len(pv)})
        log(f'  N={n}: tau05 cmmd={taus[("cmmd", 0.05)]:.5f} global={taus[("global_mmd", 0.05)]:.5f} | '
            f'FPR cmmd={np.mean(per_run["cmmd"][0.05]):.3f} global={np.mean(per_run["global_mmd"][0.05]):.3f} | '
            f'perm reject cmmd={perm_rows[-1]["reject_05"]:.3f} global={perm_rows[-2]["reject_05"]:.3f}')
    del ctx


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    modes = ('default', 'separate') if '--both-pools' in sys.argv else ('default',)
    main(set(args[0].split(',')) if args else None, modes)
