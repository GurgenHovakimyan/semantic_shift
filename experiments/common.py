"""Shared configuration and helpers for experiments 16-22."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# (dataset, encoder) pairs and the window sizes calibrated for each
CONFIGS = [('imdb', 'distilbert'), ('imdb', 'roberta'), ('amazon', 'distilbert'), ('amazon', 'roberta')]
N_GRID = {'imdb': [100, 200, 500, 1000], 'amazon': [100, 200, 250]}
ALPHAS = [0.05, 0.01]
N_NULL_CAL = 200      # null windows used to calibrate thresholds
N_RUNS = 10           # repeated runs for FPR / power estimates
N_NULL_TEST = 40      # fresh null windows per run
CONTEXT_SEED = 0


def key_of(name: str, model: str) -> str:
    return f'{name}_{model}'


def null_path(key: str, n: int) -> Path:
    return RESULTS_DIR / 'null' / f'null_{key}_N{n}.csv'


def thresholds_path() -> Path:
    return RESULTS_DIR / 'calibration_thresholds.csv'


def load_thresholds(key: str, n: int, alpha: float = 0.05) -> dict:
    """Return {detector: tau} calibrated by experiment 16."""
    df = pd.read_csv(thresholds_path())
    sub = df[(df['key'] == key) & (df['N'] == n) & (np.isclose(df['alpha'], alpha))]
    if sub.empty:
        raise FileNotFoundError(f'no thresholds for {key} N={n} alpha={alpha}; run experiment 16 first')
    return dict(zip(sub['detector'], sub['tau']))


def load_null(key: str, n: int) -> pd.DataFrame:
    return pd.read_csv(null_path(key, n))


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f'[{self.label}] {time.time() - self.t0:.1f}s', flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))
