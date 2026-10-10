"""
Experiment 21 -- Kernel bandwidth ablation and encoder (DistilBERT vs RoBERTa)
scale check.

Part A.  Bandwidth ablation.  The code uses the *mean squared* pairwise
distance with a five-scale kernel mixture.  Here the C-MMD detector is re-run
under six variants (mean/median heuristic, single/multi scale, bandwidth
fixed on the reference vs. re-estimated on every pooled window, biased V- vs
unbiased U-statistic) and the power at a matched 5% FPR and the AUROC are
compared on IMDb/DistilBERT for several scenarios.

Part B.  Encoder scale check.  DistilBERT and RoBERTa are compared with a
consistent estimator: null level and spread of the statistics, the
concept-swap signal (label noise 50%) and the natural-stream level are
tabulated for both encoders together with embedding anisotropy (mean cosine
similarity of random pairs) and the bandwidth.

Outputs
    results/bandwidth_ablation.csv
    results/encoder_scale_check.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from experiments.common import CONTEXT_SEED, DEVICE, RESULTS_DIR, Timer, key_of, log
from src.datasets import load_dataset
from src.mmd import mean_sq_dist_bandwidth, median_sq_dist_bandwidth, mmd2_rbf
from src.protocol import auroc, draw_null_window, threshold_from_null
from src.scenarios import ScenarioSampler
from src.statistics import MIN_CLASS_COUNT, build_context

N_NULL, N_DRIFT = 200, 100
SCENARIOS = [('label_noise', 0.3), ('attenuation', 1.0), ('asym_temporal', 1.0), ('boundary', 1.0), ('label_shift', 1.0)]

VARIANTS = {
    'mean_multi_fixed_U (default)': dict(heuristic='mean', kernel_num=5, fixed=True, unbiased=True),
    'median_multi_fixed_U': dict(heuristic='median', kernel_num=5, fixed=True, unbiased=True),
    'mean_single_fixed_U': dict(heuristic='mean', kernel_num=1, fixed=True, unbiased=True),
    'median_single_fixed_U': dict(heuristic='median', kernel_num=1, fixed=True, unbiased=True),
    'mean_multi_adaptive_U': dict(heuristic='mean', kernel_num=5, fixed=False, unbiased=True),
    'mean_multi_fixed_V': dict(heuristic='mean', kernel_num=5, fixed=True, unbiased=False),
}


def cmmd_variant(ctx, e, l, beta_fixed, cfg):
    vals = []
    for c, R in ctx.ref_by_class.items():
        W = e[l == c]
        if W.shape[0] < MIN_CLASS_COUNT:
            continue
        if cfg['fixed']:
            beta = beta_fixed
        else:
            beta = (mean_sq_dist_bandwidth if cfg['heuristic'] == 'mean' else median_sq_dist_bandwidth)(R, W)
        vals.append(mmd2_rbf(R, W, beta=beta, unbiased=cfg['unbiased'], kernel_num=cfg['kernel_num']))
    return float(np.mean(vals))


def part_a():
    rows = []
    ds = load_dataset('imdb', 'distilbert', DEVICE)
    ctx = build_context(ds, seed=CONTEXT_SEED, train_classifier=False)
    betas = {'mean': mean_sq_dist_bandwidth(ctx.ref_global, None, max_samples=1000),
             'median': median_sq_dist_bandwidth(ctx.ref_global, None, max_samples=1000)}
    log(f'  bandwidths on reference: mean={betas["mean"]:.3f} median={betas["median"]:.3f}')
    cal_emb, cal_lab, _ = ds.pool('cal')
    for n in [200, 500]:
        sampler = ScenarioSampler(ds, n)
        with Timer(f'bandwidth ablation N={n}'):
            # draw the windows once so that every variant sees identical data
            rng = np.random.RandomState(777 + n)
            null_windows = [draw_null_window(cal_emb, cal_lab, n, rng) for _ in range(N_NULL)]
            drift_windows = {sc: [sampler.draw(sc, rng, lam) for _ in range(N_DRIFT)] for sc, lam in SCENARIOS}
            for vname, cfg in VARIANTS.items():
                beta_fixed = betas[cfg['heuristic']]
                null_vals = np.array([cmmd_variant(ctx, e, l, beta_fixed, cfg) for e, l in null_windows])
                tau = threshold_from_null(null_vals, 0.05)
                for (sc, lam) in SCENARIOS:
                    vals = np.array([cmmd_variant(ctx, e, l, beta_fixed, cfg) for e, l in drift_windows[sc]])
                    rows.append({'N': n, 'variant': vname, 'scenario': sc, 'lam': lam, 'power': float(np.mean(vals > tau)),
                                 'auroc': auroc(null_vals, vals), 'null_mean': null_vals.mean(), 'null_std': null_vals.std(),
                                 'drift_mean': vals.mean(), 'tau': tau, 'beta': beta_fixed if cfg['fixed'] else np.nan})
        df = pd.DataFrame(rows)
        piv = df[df['N'] == n].pivot(index='variant', columns='scenario', values='power')
        log(f'  N={n} power at 5% FPR:\n{piv.round(2).to_string()}')
    pd.DataFrame(rows).to_csv(RESULTS_DIR / 'bandwidth_ablation.csv', index=False)


def anisotropy(emb, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(emb.shape[0], generator=g)[:n].to(emb.device)
    x = torch.nn.functional.normalize(emb[idx], dim=1)
    cos = x @ x.t()
    n = x.shape[0]
    return float((cos.sum() - n).item() / (n * n - n))


def part_b():
    rows = []
    thr = pd.read_csv(RESULTS_DIR / 'calibration_thresholds.csv')
    for name in ['imdb', 'amazon']:
        for model in ['distilbert', 'roberta']:
            key = key_of(name, model)
            ds = load_dataset(name, model, DEVICE)
            ctx = build_context(ds, seed=CONTEXT_SEED, train_classifier=False)
            row = {'key': key, 'dataset': name, 'model': model, 'beta': ctx.beta,
                   'mean_pairwise_dist': float(np.sqrt(ctx.beta)), 'mean_norm': float(ds.ref_emb.norm(dim=1).mean().item()),
                   'anisotropy_cos': anisotropy(ds.ref_emb)}
            for n in [200]:
                for det in ['cmmd', 'global_mmd']:
                    t = thr[(thr['key'] == key) & (thr['N'] == n) & (thr['detector'] == det) & (np.isclose(thr['alpha'], 0.05))]
                    row[f'{det}_null_mean_N{n}'] = float(t['null_mean'].iloc[0])
                    row[f'{det}_null_std_N{n}'] = float(t['null_std'].iloc[0])
                    row[f'{det}_tau05_N{n}'] = float(t['tau'].iloc[0])
                bench = RESULTS_DIR / 'benchmark' / f'bench_{key}_N{n}.csv'
                if bench.exists():
                    b = pd.read_csv(bench)
                    swap = b[(b['scenario'] == 'label_noise') & (np.isclose(b['lam'], 0.5))]
                    row[f'cmmd_swap50_mean_N{n}'] = swap['cmmd'].mean()
                    row[f'cmmd_swap50_effect_N{n}'] = (swap['cmmd'].mean() - row[f'cmmd_null_mean_N{n}']) / row[f'cmmd_null_std_N{n}']
                    nat = b[b['scenario'] == 'natural']
                    row[f'cmmd_natural_mean_N{n}'] = nat['cmmd'].mean()
                    row[f'global_natural_mean_N{n}'] = nat['global_mmd'].mean()
            stream = RESULTS_DIR / 'stream' / f'stream_{key}.csv'
            if stream.exists():
                s = pd.read_csv(stream)
                row['stream_cmmd_mean'] = s['cmmd'].mean()
                row['stream_global_mean'] = s['global_mmd'].mean()
            rows.append(row)
            del ds, ctx
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    # ratios DistilBERT / RoBERTa
    ratio_rows = []
    for name in ['imdb', 'amazon']:
        d = df[(df['dataset'] == name) & (df['model'] == 'distilbert')].iloc[0]
        r = df[(df['dataset'] == name) & (df['model'] == 'roberta')].iloc[0]
        for col in df.columns:
            if col in ('key', 'dataset', 'model'):
                continue
            if pd.notna(d[col]) and pd.notna(r[col]) and r[col] != 0:
                ratio_rows.append({'dataset': name, 'quantity': col, 'distilbert': d[col], 'roberta': r[col], 'ratio_distil_over_roberta': d[col] / r[col]})
    out = pd.DataFrame(ratio_rows)
    out.to_csv(RESULTS_DIR / 'encoder_scale_check.csv', index=False)
    log(out.round(4).to_string())


if __name__ == '__main__':
    parts = sys.argv[1].upper() if len(sys.argv) > 1 else 'AB'
    if 'A' in parts:
        log('Part A: bandwidth ablation')
        part_a()
    if 'B' in parts:
        log('\nPart B: encoder scale check')
        part_b()
    log('\nExperiment 21 complete.')
