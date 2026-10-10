"""
Drift scenarios built from *real* review embeddings.

All scenarios draw their samples from (i) the held-out reference ``test`` pool
(same distribution as the reference, never used for calibration) and (ii) the
chronologically sorted deployment stream.  No synthetic vectors are generated;
drift is created only by *which* real reviews (or which labels) enter a window.

Scenario            Type of shift                                          Datasets
------------------  -----------------------------------------------------  --------
null                none (held-out reference windows)                      both
natural             natural temporal drift: a contiguous stream window     both
attenuation         within-class covariate shift: a fraction lam of each   IMDb
                    class is replaced by *mild* reviews (7/10 for positive,
                    4/10 for negative) -- polarity attenuation; P(Y) fixed
boundary            annotation-policy (posterior) drift: borderline        IMDb
                    reviews (rating 4 / 7) receive the label of the
                    neighbouring class with probability lam
asym_temporal       asymmetric temporal drift: a fraction lam of the       both
                    negative class comes from the deployment era while the
                    positive class stays in the reference era
label_shift         pure prior shift: P(Y=1) moves from the reference      both
                    prior towards 0.95 (lam = 1), P(E|Y) unchanged
label_noise         label corruption ("concept swap"): a fraction lam of   both
                    labels is flipped on a fresh reference-distributed
                    window (diagnostic positive control)
"""

from __future__ import annotations

import numpy as np
import torch

from .datasets import DriftDataset

LEVELS = {
    'natural': [1.0],
    'attenuation': [0.25, 0.5, 0.75, 1.0],
    'boundary': [0.25, 0.5, 0.75, 1.0],
    'asym_temporal': [0.25, 0.5, 0.75, 1.0],
    'label_shift': [0.25, 0.5, 0.75, 1.0],
    'label_noise': [0.1, 0.2, 0.3, 0.5],
}

# Which scenarios genuinely change P(E|Y) (what C-MMD measures)?
CHANGES_COND = {'natural': True, 'attenuation': True, 'boundary': True, 'asym_temporal': True,
                'label_shift': False, 'label_noise': True}


def available_scenarios(ds: DriftDataset) -> list[str]:
    s = ['natural', 'asym_temporal', 'label_shift', 'label_noise']
    if ds.ref_rating is not None:
        s = ['natural', 'attenuation', 'boundary', 'asym_temporal', 'label_shift', 'label_noise']
    return s


def _choice(rng: np.random.RandomState, n_avail: int, n: int) -> np.ndarray:
    if n > n_avail:
        raise ValueError(f'requested {n} samples from a pool of {n_avail}')
    return rng.choice(n_avail, size=n, replace=False)


def _take(emb: torch.Tensor, lab: torch.Tensor, idx: np.ndarray):
    t = torch.as_tensor(idx, device=emb.device, dtype=torch.long)
    return emb[t], lab[t]


class ScenarioSampler:
    """Draw windows of size ``n`` for every scenario at drift level ``lam``."""

    def __init__(self, ds: DriftDataset, n: int):
        self.ds = ds
        self.n = n
        self.emb, self.lab, self.rating = ds.pool('test')
        self.lab_np = self.lab.cpu().numpy()
        self.pos = np.where(self.lab_np == 1)[0]
        self.neg = np.where(self.lab_np == 0)[0]
        if self.rating is not None:
            self.pos_mild = self.pos[self.rating[self.pos] == 7]
            self.neg_mild = self.neg[self.rating[self.neg] == 4]
        s_lab = ds.stream_lab.cpu().numpy()
        self.stream_neg = np.where(s_lab == 0)[0]
        self.stream_pos = np.where(s_lab == 1)[0]
        self.prior = ds.ref_prior

    # -- primitives --------------------------------------------------------
    def _class_counts(self, rng, prior=None):
        p = self.prior if prior is None else prior
        n_pos = int(rng.binomial(self.n, p))
        n_pos = min(max(n_pos, 1), self.n - 1)
        return n_pos, self.n - n_pos

    def _null_idx(self, rng, prior=None):
        n_pos, n_neg = self._class_counts(rng, prior)
        return np.r_[self.pos[_choice(rng, len(self.pos), n_pos)], self.neg[_choice(rng, len(self.neg), n_neg)]]

    # -- scenarios ---------------------------------------------------------
    def null(self, rng, lam=0.0):
        idx = self._null_idx(rng)
        return _take(self.emb, self.lab, idx)

    def natural(self, rng, lam=1.0, start=None):
        ds = self.ds
        if start is None:
            start = int(rng.randint(0, ds.stream_emb.shape[0] - self.n + 1))
        return ds.stream_emb[start:start + self.n], ds.stream_lab[start:start + self.n]

    def attenuation(self, rng, lam):
        n_pos, n_neg = self._class_counts(rng)
        k_pos, k_neg = int(round(lam * n_pos)), int(round(lam * n_neg))
        idx = np.r_[
            self.pos_mild[_choice(rng, len(self.pos_mild), k_pos)],
            self.pos[_choice(rng, len(self.pos), n_pos - k_pos)],
            self.neg_mild[_choice(rng, len(self.neg_mild), k_neg)],
            self.neg[_choice(rng, len(self.neg), n_neg - k_neg)],
        ]
        return _take(self.emb, self.lab, idx)

    def boundary(self, rng, lam):
        idx = self._null_idx(rng)
        emb, lab = _take(self.emb, self.lab, idx)
        r = self.rating[idx]
        flip = ((r == 4) | (r == 7)) & (rng.rand(len(idx)) < lam)
        lab = lab.clone()
        f = torch.as_tensor(flip, device=lab.device)
        lab[f] = 1 - lab[f]
        return emb, lab

    def asym_temporal(self, rng, lam):
        n_pos, n_neg = self._class_counts(rng)
        k = int(round(lam * n_neg))
        ref_idx = np.r_[self.pos[_choice(rng, len(self.pos), n_pos)], self.neg[_choice(rng, len(self.neg), n_neg - k)]]
        emb, lab = _take(self.emb, self.lab, ref_idx)
        if k > 0:
            s_idx = self.stream_neg[_choice(rng, len(self.stream_neg), k)]
            s_emb, s_lab = _take(self.ds.stream_emb, self.ds.stream_lab, s_idx)
            emb, lab = torch.cat([emb, s_emb]), torch.cat([lab, s_lab])
        return emb, lab

    def label_shift(self, rng, lam):
        prior = self.prior + lam * (0.95 - self.prior)
        idx = self._null_idx(rng, prior=prior)
        return _take(self.emb, self.lab, idx)

    def label_noise(self, rng, lam):
        idx = self._null_idx(rng)
        emb, lab = _take(self.emb, self.lab, idx)
        k = int(round(lam * self.n))
        lab = lab.clone()
        if k > 0:
            f = torch.as_tensor(rng.choice(self.n, size=k, replace=False), device=lab.device)
            lab[f] = 1 - lab[f]
        return emb, lab

    def draw(self, scenario: str, rng, lam: float):
        return getattr(self, scenario)(rng, lam)


def ramp(t: int, onset: int, ramp_len: int, lam_max: float) -> float:
    """Drift level at window t: 0 before onset, linear ramp over ramp_len windows, then lam_max."""
    if t < onset:
        return 0.0
    if ramp_len <= 1:
        return lam_max
    return lam_max * min(1.0, (t - onset + 1) / ramp_len)
