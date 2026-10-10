"""
Experiment 18 -- Detection delay under abrupt and gradual drift onset.

A monitored stream of T windows is simulated: the first ONSET windows are
drift-free (held-out reference), after which a drift scenario is switched on
either abruptly (lam = lam_max from the onset) or gradually (lam ramps
linearly to lam_max over RAMP windows).  Every detector is run with its
threshold tau_0.05 from experiment 16 and two alarm rules (k = 1 and k = 2
consecutive exceedances).  We record the delay (windows from onset to the
first alarm), whether an alarm occurred within the horizon, and the number of
false alarms before the onset.  Repeated over N_SEEDS independent streams.

Outputs
    results/delay/delay_{key}_N{N}.csv   per-window statistics for all streams
    results/delay_summary.csv            delay statistics per detector / rule
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from experiments.common import CONTEXT_SEED, DEVICE, RESULTS_DIR, Timer, key_of, load_thresholds, log
from src.datasets import load_dataset
from src.protocol import detection_delay, sustained_alarms
from src.scenarios import ScenarioSampler, ramp
from src.statistics import ALL_DETECTORS, build_context, compute_all

T, ONSET, RAMP = 30, 10, 10
N_SEEDS = 10
RULES = {'k1': 1, 'k2': 2}
PLAN = [
    ('imdb', 'distilbert', 200, [('asym_temporal', 1.0), ('boundary', 1.0), ('attenuation', 1.0), ('label_noise', 0.3)]),
    ('imdb', 'distilbert', 500, [('asym_temporal', 1.0), ('boundary', 1.0), ('attenuation', 1.0), ('label_noise', 0.3)]),
    ('imdb', 'roberta', 200, [('asym_temporal', 1.0), ('boundary', 1.0)]),
    ('amazon', 'distilbert', 200, [('asym_temporal', 1.0), ('label_noise', 0.3)]),
]


def main(selected=None):
    (RESULTS_DIR / 'delay').mkdir(exist_ok=True)
    cache = {}
    for name, model, n, drifts in PLAN:
        key = key_of(name, model)
        if selected and key not in selected:
            continue
        summary_rows = []
        if key not in cache:
            ds = load_dataset(name, model, DEVICE)
            cache[key] = (ds, build_context(ds, seed=CONTEXT_SEED))
        ds, ctx = cache[key]
        taus = load_thresholds(key, n, alpha=0.05)
        sampler = ScenarioSampler(ds, n)
        rows = []
        with Timer(f'{key} N={n} delay'):
            for drift, lam_max in drifts:
                for onset_type in ['abrupt', 'gradual']:
                    ramp_len = 1 if onset_type == 'abrupt' else RAMP
                    for seed in range(N_SEEDS):
                        rng = np.random.RandomState(30000 + 100 * seed + n)
                        stream = []
                        for t in range(T):
                            lam = ramp(t, ONSET, ramp_len, lam_max)
                            e, l = sampler.draw('null', rng, 0.0) if lam == 0.0 else sampler.draw(drift, rng, lam)
                            r = compute_all(ctx, e, l)
                            r.update({'drift': drift, 'onset_type': onset_type, 'seed': seed, 't': t, 'lam': lam})
                            stream.append(r)
                        sdf = pd.DataFrame(stream)
                        rows.append(sdf)
                        for det in ALL_DETECTORS:
                            exceed = sdf[det].values > taus[det]
                            for rule, k in RULES.items():
                                alarms = sustained_alarms(exceed, k=k)
                                summary_rows.append({
                                    'key': key, 'dataset': name, 'model': model, 'N': n, 'drift': drift,
                                    'lam_max': lam_max, 'onset_type': onset_type, 'seed': seed, 'detector': det,
                                    'rule': rule, 'delay': detection_delay(alarms, ONSET),
                                    'pre_onset_alarms': int(alarms[:ONSET].sum()),
                                    'f1_pre': sdf['f1'].values[:ONSET].mean(), 'f1_post': sdf['f1'].values[ONSET:].mean(),
                                })
        pd.concat(rows).to_csv(RESULTS_DIR / 'delay' / f'delay_{key}_N{n}.csv', index=False)
        sm = pd.DataFrame(summary_rows)
        sm.to_csv(RESULTS_DIR / 'delay' / f'summary_{key}_N{n}.csv', index=False)
        sub = sm[(sm['key'] == key) & (sm['N'] == n) & (sm['rule'] == 'k2')]
        for (drift, ot), g in sub.groupby(['drift', 'onset_type']):
            msg = []
            for det in ['global_mmd', 'bbsd_ks', 'cmmd', 'cond_c2st', 'error_rate']:
                d = g[g['detector'] == det]['delay'].values
                det_rate = np.mean(np.isfinite(d))
                med = np.median(d[np.isfinite(d)]) if det_rate > 0 else float('nan')
                msg.append(f'{det}: rate={det_rate:.2f} med={med:.1f}')
            log(f'  {key} N={n} {drift:14s} {ot:8s} | ' + ' | '.join(msg))
    parts = sorted((RESULTS_DIR / 'delay').glob('summary_*.csv'))
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(RESULTS_DIR / 'delay_summary.csv', index=False)
    log('\nExperiment 18 complete.')


if __name__ == '__main__':
    main(set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None)
