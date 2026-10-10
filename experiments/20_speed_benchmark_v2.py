"""
Experiment 20 -- Computational cost of every detector.

Wall-clock time per monitored window as a function of the window size N for
all detectors of ``src/statistics.py`` (reference sample fixed at M=1000
embeddings, 500 per class, as in the monitoring protocol).  Kernel and
energy statistics are timed on a single CPU thread *and* on the GPU; the
classifier-based statistics (C2ST, BBSD, error-rate) run in scikit-learn on
the CPU.  The exact permutation test (200 permutations) is timed separately.
For perspective, the time needed to *produce* the embeddings of one window
with DistilBERT is measured as well when the model is available locally.

Outputs
    results/speed_benchmark_v2.csv
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from experiments.common import CONTEXT_SEED, PROJECT_ROOT, RESULTS_DIR, log
from src.datasets import load_dataset
from src.protocol import permutation_test_cmmd
from src.statistics import ALL_DETECTORS, build_context

N_VALUES = [100, 200, 500, 1000, 2000]
N_REPEATS = 20
TORCH_BASED = {'global_mmd', 'energy', 'cmmd', 'cond_energy', 'centroid_shift', 'joint_mmd', 'prior_shift'}


def time_fn(fn, repeats=N_REPEATS, sync=False):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000 * np.median(times), 1000 * np.percentile(times, 25), 1000 * np.percentile(times, 75)


def embedding_cost(texts, device):
    try:
        from transformers import DistilBertModel, DistilBertTokenizer
        tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
        model = DistilBertModel.from_pretrained('distilbert-base-uncased', local_files_only=True).to(device).eval()
    except Exception as exc:  # model not cached locally
        log(f'  embedding cost skipped ({type(exc).__name__})')
        return None
    @torch.no_grad()
    def run():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            model(**enc).last_hidden_state[:, 0, :]
    return time_fn(run, repeats=3, sync=device == 'cuda')


def main():
    torch.set_num_threads(1)
    rows = []
    ds_cpu = load_dataset('imdb', 'distilbert', 'cpu')
    ctx_cpu = build_context(ds_cpu, seed=CONTEXT_SEED)
    test_emb, test_lab, _ = ds_cpu.pool('test')
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    ctx_gpu = None
    if 'cuda' in devices:
        ds_gpu = load_dataset('imdb', 'distilbert', 'cuda')
        ctx_gpu = build_context(ds_gpu, seed=CONTEXT_SEED, train_classifier=False)
        ctx_gpu.clf, ctx_gpu.ref_scores, ctx_gpu.ref_error = ctx_cpu.clf, ctx_cpu.ref_scores, ctx_cpu.ref_error

    for n in N_VALUES:
        rng = np.random.RandomState(n)
        idx = rng.choice(test_emb.shape[0], n, replace=False)
        e_cpu, l_cpu = test_emb[idx], test_lab[idx]
        for det, fn in ALL_DETECTORS.items():
            med, q1, q3 = time_fn(lambda: fn(ctx_cpu, e_cpu, l_cpu))
            rows.append({'detector': det, 'N': n, 'device': 'cpu', 'ms_median': med, 'ms_q1': q1, 'ms_q3': q3})
            if det in TORCH_BASED and ctx_gpu is not None:
                e_g, l_g = e_cpu.cuda(), l_cpu.cuda()
                med, q1, q3 = time_fn(lambda: fn(ctx_gpu, e_g, l_g), sync=True)
                rows.append({'detector': det, 'N': n, 'device': 'cuda', 'ms_median': med, 'ms_q1': q1, 'ms_q3': q3})
        # permutation test (C-MMD, 200 permutations)
        med, q1, q3 = time_fn(lambda: permutation_test_cmmd(ctx_cpu.ref_by_class, e_cpu, l_cpu, ctx_cpu.beta, n_perm=200), repeats=5)
        rows.append({'detector': 'cmmd_permutation200', 'N': n, 'device': 'cpu', 'ms_median': med, 'ms_q1': q1, 'ms_q3': q3})
        if ctx_gpu is not None:
            e_g, l_g = e_cpu.cuda(), l_cpu.cuda()
            med, q1, q3 = time_fn(lambda: permutation_test_cmmd(ctx_gpu.ref_by_class, e_g, l_g, ctx_gpu.beta, n_perm=200), repeats=5, sync=True)
            rows.append({'detector': 'cmmd_permutation200', 'N': n, 'device': 'cuda', 'ms_median': med, 'ms_q1': q1, 'ms_q3': q3})
        df = pd.DataFrame(rows)
        sub = df[(df['N'] == n) & (df['device'] == 'cpu')].set_index('detector')['ms_median']
        log(f'  N={n}: ' + ' '.join(f'{d}={sub[d]:.1f}ms' for d in ['cmmd', 'global_mmd', 'c2st_linear', 'c2st_mlp', 'cond_c2st', 'bbsd_ks']))

    # embedding extraction cost for one window of N texts
    try:
        texts = pd.read_pickle(PROJECT_ROOT / 'data' / 'processed' / 'reference_set.pkl')['full_text'].tolist()
        for n in [200, 500, 1000]:
            for device in devices:
                res = embedding_cost(texts[:n], device)
                if res is not None:
                    rows.append({'detector': 'distilbert_embedding', 'N': n, 'device': device, 'ms_median': res[0], 'ms_q1': res[1], 'ms_q3': res[2]})
                    log(f'  DistilBERT embedding of N={n} texts on {device}: {res[0]:.0f} ms')
    except Exception as exc:
        log(f'  embedding cost skipped ({exc})')

    pd.DataFrame(rows).to_csv(RESULTS_DIR / 'speed_benchmark_v2.csv', index=False)
    log('\nExperiment 20 complete.')


if __name__ == '__main__':
    main()
