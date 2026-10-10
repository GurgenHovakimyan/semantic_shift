"""
Unified access to the embedded datasets used in experiments 16-22.

Every dataset is returned as a :class:`DriftDataset` with

* a *reference* period (training-era data) that is randomly partitioned into
  three disjoint pools -- ``fit`` (fixed reference sample, bandwidth and
  downstream classifier), ``cal`` (null windows used to calibrate alarm
  thresholds) and ``test`` (held-out null windows used to measure the
  false-positive rate and to build the realistic drift scenarios);
* a *stream* period sorted chronologically (the original IMDb stream file is
  stored in corpus order, not in time order, so sorting is essential for any
  temporal analysis);
* per-sample star ratings for IMDb (needed by the rating-based scenarios).

Reference/stream definitions
----------------------------
IMDb   : reference 2005-2010 (50,000 reviews), stream 2013-01 .. 2018-01
         (165,790 reviews), neutral ratings (5-6) excluded.
Amazon : chronologically sorted balanced Electronics sample (10,000 reviews);
         reference = first 3,000 reviews (2001 .. late 2017), stream = the
         remaining 7,000 (late 2017 .. 2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

SPLIT_SEED = 0
AMAZON_REF_SIZE = 3000

_FILES = {
    ('imdb', 'distilbert'): ('ref_embeddings.pt', 'stream_embeddings.pt'),
    ('imdb', 'roberta'): ('ref_embeddings_roberta.pt', 'stream_embeddings_roberta.pt'),
    ('amazon', 'distilbert'): ('amazon_embeddings.pt', None),
    ('amazon', 'roberta'): ('amazon_roberta_embeddings.pt', None),
}


@dataclass
class DriftDataset:
    name: str
    model: str
    ref_emb: torch.Tensor
    ref_lab: torch.Tensor
    ref_dates: np.ndarray
    ref_rating: np.ndarray | None
    stream_emb: torch.Tensor
    stream_lab: torch.Tensor
    stream_dates: np.ndarray
    stream_rating: np.ndarray | None
    fit_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def key(self) -> str:
        return f'{self.name}_{self.model}'

    @property
    def classes(self) -> list[int]:
        return sorted(int(c) for c in torch.unique(self.ref_lab).tolist())

    @property
    def ref_prior(self) -> float:
        return float(self.ref_lab.float().mean().item())

    def pool(self, which: str):
        idx = {'fit': self.fit_idx, 'cal': self.cal_idx, 'test': self.test_idx}[which]
        idx_t = torch.as_tensor(idx, device=self.ref_emb.device)
        rating = None if self.ref_rating is None else self.ref_rating[idx]
        return self.ref_emb[idx_t], self.ref_lab[idx_t], rating

    def to(self, device) -> 'DriftDataset':
        self.ref_emb = self.ref_emb.to(device)
        self.ref_lab = self.ref_lab.to(device)
        self.stream_emb = self.stream_emb.to(device)
        self.stream_lab = self.stream_lab.to(device)
        return self


def _load_pt(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def _three_way_split(n: int, fracs=(0.4, 0.3, 0.3), seed: int = SPLIT_SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_fit = int(round(fracs[0] * n))
    n_cal = int(round(fracs[1] * n))
    return perm[:n_fit], perm[n_fit:n_fit + n_cal], perm[n_fit + n_cal:]


def load_dataset(name: str, model: str = 'distilbert', device: str | torch.device = 'cpu',
                 shared_null_pool: bool | None = None) -> DriftDataset:
    """Load one (dataset, encoder) pair with chronologically sorted stream.

    ``shared_null_pool`` merges the calibration and test pools.  This is the
    default for Amazon, whose reference period (3,000 reviews) is too small
    for two disjoint pools that are each large relative to the window size:
    with disjoint 750-review pools the between-pool sampling offset inflates
    the false-positive rate of every detector at N >= 200 (experiment 16
    reports both variants).  IMDb keeps disjoint pools of 15,000 reviews.
    """
    name, model = name.lower(), model.lower()
    ref_file, stream_file = _FILES[(name, model)]

    if name == 'imdb':
        ref = _load_pt(DATA_DIR / ref_file)
        stream = _load_pt(DATA_DIR / stream_file)
        ref_df = pd.read_pickle(DATA_DIR / 'reference_set.pkl')
        stream_df = pd.read_pickle(DATA_DIR / 'stream_set.pkl')
        assert len(ref_df) == ref['embeddings'].shape[0] and len(stream_df) == stream['embeddings'].shape[0]
        ref_emb, ref_lab = ref['embeddings'], ref['labels']
        ref_dates = pd.to_datetime(ref['dates']).values
        ref_rating = ref_df['rating'].to_numpy().astype(int)

        order = np.argsort(pd.to_datetime(stream['dates']).values, kind='stable')
        stream_emb = stream['embeddings'][torch.as_tensor(order)]
        stream_lab = stream['labels'][torch.as_tensor(order)]
        stream_dates = pd.to_datetime(stream['dates']).values[order]
        stream_rating = stream_df['rating'].to_numpy().astype(int)[order]
    elif name == 'amazon':
        data = _load_pt(DATA_DIR / ref_file)
        dates = pd.to_datetime(data['dates']).values
        order = np.argsort(dates, kind='stable')
        emb, lab, dates = data['embeddings'][torch.as_tensor(order)], data['labels'][torch.as_tensor(order)], dates[order]
        ref_emb, ref_lab, ref_dates = emb[:AMAZON_REF_SIZE], lab[:AMAZON_REF_SIZE], dates[:AMAZON_REF_SIZE]
        stream_emb, stream_lab, stream_dates = emb[AMAZON_REF_SIZE:], lab[AMAZON_REF_SIZE:], dates[AMAZON_REF_SIZE:]
        ref_rating = stream_rating = None
    else:
        raise ValueError(name)

    # Amazon's reference is small: give the fixed sample / classifier half and
    # the two null pools a quarter each.
    fracs = (0.4, 0.3, 0.3) if name == 'imdb' else (0.5, 0.25, 0.25)
    fit_idx, cal_idx, test_idx = _three_way_split(ref_emb.shape[0], fracs)
    if shared_null_pool is None:
        shared_null_pool = name == 'amazon'
    if shared_null_pool:
        cal_idx = test_idx = np.sort(np.concatenate([cal_idx, test_idx]))

    ds = DriftDataset(
        name=name, model=model,
        ref_emb=ref_emb.float(), ref_lab=ref_lab.long(), ref_dates=ref_dates, ref_rating=ref_rating,
        stream_emb=stream_emb.float(), stream_lab=stream_lab.long(), stream_dates=stream_dates,
        stream_rating=stream_rating, fit_idx=fit_idx, cal_idx=cal_idx, test_idx=test_idx,
    )
    return ds.to(device)


def describe(ds: DriftDataset) -> str:
    lines = [
        f'{ds.key}: reference {ds.ref_emb.shape[0]} ({pd.Timestamp(ds.ref_dates.min()).date()} .. '
        f'{pd.Timestamp(ds.ref_dates.max()).date()}, P(Y=1)={ds.ref_prior:.3f}); '
        f'stream {ds.stream_emb.shape[0]} ({pd.Timestamp(ds.stream_dates.min()).date()} .. '
        f'{pd.Timestamp(ds.stream_dates.max()).date()}, P(Y=1)={ds.stream_lab.float().mean().item():.3f})',
        f'  pools: fit={len(ds.fit_idx)}, cal={len(ds.cal_idx)}, test={len(ds.test_idx)}'
        + (' (cal and test pools shared)' if np.array_equal(ds.cal_idx, ds.test_idx) else ''),
    ]
    return '\n'.join(lines)
