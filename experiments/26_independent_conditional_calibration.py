"""Independent split-level check of class-count-conditional C-MMD calibration.

For IMDb/DistilBERT (N=500) and Amazon/DistilBERT (N=200), this experiment
repeats the fit/calibration/test partition, calibrates one natural-prior
threshold and thresholds conditional on the observed class counts, then
evaluates each on separate null and 95%-positive test windows. Every calibration
and test window is sampled without replacement, so all windows contain distinct
observations and the two thresholds are compared on the same footing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from experiments.common import DEVICE, RESULTS_DIR, log
from src.datasets import load_dataset
from src.protocol import (
    ClassCountConditionalCalibrator,
    draw_class_count_window,
    draw_null_window,
    empirical_fpr,
    threshold_from_null,
)
from src.statistics import build_context, cmmd

N_SPLITS = 5
N_CAL = 200
N_TEST = 200
ALPHA = 0.05
CONFIGS = [
    ('imdb', 'distilbert', 500),
    ('amazon', 'distilbert', 200),
]


@dataclass
class ReferenceSplit:
    ref_emb: torch.Tensor
    ref_lab: torch.Tensor
    fit_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def classes(self):
        return sorted(int(c) for c in torch.unique(self.ref_lab).tolist())

    @property
    def ref_prior(self):
        return float(self.ref_lab.float().mean().item())

    def pool(self, which):
        indices = {'fit': self.fit_idx, 'cal': self.cal_idx, 'test': self.test_idx}[which]
        indices = torch.as_tensor(indices, device=self.ref_emb.device)
        return self.ref_emb[indices], self.ref_lab[indices], None


def split_reference(emb, lab, seed):
    rng = np.random.RandomState(seed)
    fit_idx, cal_idx, test_idx = [], [], []
    for label in sorted(int(c) for c in torch.unique(lab).tolist()):
        indices = torch.where(lab == label)[0].cpu().numpy()
        rng.shuffle(indices)
        n_fit = int(0.4 * len(indices))
        n_cal = int(0.3 * len(indices))
        fit_idx.extend(indices[:n_fit])
        cal_idx.extend(indices[n_fit:n_fit + n_cal])
        test_idx.extend(indices[n_fit + n_cal:])
    return ReferenceSplit(
        emb,
        lab,
        np.asarray(fit_idx),
        np.asarray(cal_idx),
        np.asarray(test_idx),
    )


def counts_for_prior(classes, n, positive_prior):
    positive = int(round(n * positive_prior))
    counts = {c: 0 for c in classes}
    positive_class = max(classes)
    counts[positive_class] = positive
    counts[min(classes)] = n - positive
    return counts


def evaluate_counts(ctx, pool_emb, pool_lab, counts, n_windows, seed):
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_windows):
        emb, labels = draw_class_count_window(pool_emb, pool_lab, counts, rng)
        values.append(cmmd(ctx, emb, labels))
    return np.asarray(values)


def run_configuration(name, model, window_size):
    ds = load_dataset(name, model, DEVICE, shared_null_pool=False)
    rows = []
    for split_id in range(N_SPLITS):
        split = split_reference(ds.ref_emb, ds.ref_lab, seed=8100 + split_id)
        ctx = build_context(split, train_classifier=False, seed=9100 + split_id)
        cal_emb, cal_lab, _ = split.pool('cal')
        test_emb, test_lab, _ = split.pool('test')
        classes = split.classes
        natural_prior = split.ref_prior
        count_scenarios = {
            'natural_prior': counts_for_prior(classes, window_size, natural_prior),
            '95pct_positive': counts_for_prior(classes, window_size, 0.95),
        }

        cal_rng = np.random.RandomState(10100 + split_id)
        calibration_values = []
        for _ in range(N_CAL):
            emb, labels = draw_null_window(cal_emb, cal_lab, window_size, cal_rng)
            calibration_values.append(cmmd(ctx, emb, labels))
        fixed_threshold = threshold_from_null(np.asarray(calibration_values), ALPHA)

        conditional = ClassCountConditionalCalibrator(
            cmmd, ctx, cal_emb, cal_lab, classes,
            alpha=ALPHA, n_null=N_CAL, seed=11100 + split_id,
        )
        for scenario, counts in count_scenarios.items():
            test_values = evaluate_counts(
                ctx, test_emb, test_lab, counts, N_TEST,
                seed=12100 + 100 * split_id + sum(counts.values()),
            )
            conditional_threshold = conditional.threshold(counts)
            rows.append({
                'dataset': name,
                'model': model,
                'window_size': window_size,
                'split': split_id + 1,
                'scenario': scenario,
                'class_counts': str(tuple(counts[c] for c in classes)),
                'fixed_threshold': fixed_threshold,
                'conditional_threshold': conditional_threshold,
                'fpr_fixed_threshold': empirical_fpr(test_values, fixed_threshold),
                'fpr_conditional_threshold': empirical_fpr(test_values, conditional_threshold),
                'n_test_windows': len(test_values),
            })
        log(f'{name}/{model} split {split_id + 1}/{N_SPLITS} complete')
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for name, model, window_size in CONFIGS:
        rows.extend(run_configuration(name, model, window_size))
    result = pd.DataFrame(rows)
    result_path = RESULTS_DIR / 'independent_conditional_calibration.csv'
    result.to_csv(result_path, index=False)
    summary = (
        result.groupby(['dataset', 'model', 'window_size', 'scenario', 'class_counts'])
        .agg(
            fixed_threshold=('fixed_threshold', 'mean'),
            conditional_threshold=('conditional_threshold', 'mean'),
            fpr_fixed_mean=('fpr_fixed_threshold', 'mean'),
            fpr_fixed_sd=('fpr_fixed_threshold', 'std'),
            fpr_conditional_mean=('fpr_conditional_threshold', 'mean'),
            fpr_conditional_sd=('fpr_conditional_threshold', 'std'),
            n_splits=('split', 'count'),
            n_test_windows=('n_test_windows', 'sum'),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / 'independent_conditional_calibration_summary.csv'
    summary.to_csv(summary_path, index=False)
    log(f'Wrote {result_path}')
    log(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
