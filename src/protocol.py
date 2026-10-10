"""
Statistical monitoring protocol for class-conditional drift.

Components
----------
1. **Null calibration.**  A statistic ``s`` is evaluated on ``B`` windows of
   size ``N`` drawn from a held-out, drift-free reference pool.  The alarm
   threshold is the empirical ``(1 - alpha)`` quantile of these null values,
   ``tau_alpha``.  Calibration is repeated for every (dataset, encoder, N) --
   thresholds are *not* transferable across encoders or window sizes.
2. **Permutation test.**  For kernel statistics an exact per-window p-value is
   obtained by permuting reference/window membership (within each class for
   C-MMD) on the pooled kernel matrix.
3. **Alarm rule / multiplicity.**  Online: raise an alarm when ``s_t > tau_alpha``
   for ``k`` consecutive windows ("sustained alarm").  Offline: control the
   false-discovery rate over all windows with Benjamini-Hochberg.
4. **Evaluation metrics.**  False-positive rate on fresh null windows, power
   (detection rate) on drifted windows, AUROC between null and drifted
   statistic distributions, and detection delay (windows between drift onset
   and the first sustained alarm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .mmd import pooled_kernel, mmd2_from_kernel


# ---------------------------------------------------------------------------
# Null calibration and thresholds
# ---------------------------------------------------------------------------

def draw_null_window(pool_emb: torch.Tensor, pool_lab: torch.Tensor, n: int, rng: np.random.RandomState):
    idx = torch.as_tensor(rng.choice(pool_emb.shape[0], size=n, replace=False), device=pool_emb.device)
    return pool_emb[idx], pool_lab[idx]


def threshold_from_null(null_values: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical (1 - alpha) quantile with the conservative (higher) interpolation."""
    v = np.asarray(null_values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.quantile(v, 1.0 - alpha, method='higher'))


def empirical_fpr(values: np.ndarray, tau: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.mean(v > tau))


def power(values: np.ndarray, tau: float) -> float:
    return empirical_fpr(values, tau)


def auroc(null_values: np.ndarray, drift_values: np.ndarray) -> float:
    a = np.asarray(null_values, float)
    b = np.asarray(drift_values, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


# ---------------------------------------------------------------------------
# Permutation tests on the kernel matrix
# ---------------------------------------------------------------------------

def _batched_mmd2(K: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Unbiased MMD^2 for a batch of membership masks (B, n) over kernel K (n, n)."""
    m = masks.to(K.dtype)
    w = 1 - m
    n1 = m.sum(1)
    n2 = m.shape[1] - n1
    d = K.diagonal()
    KM = m @ K                                  # (B, n)
    KW = w @ K
    xx = ((KM * m).sum(1) - m @ d) / (n1 * (n1 - 1))
    yy = ((KW * w).sum(1) - w @ d) / (n2 * (n2 - 1))
    xy = (KM * w).sum(1) / (n1 * n2)
    return xx + yy - 2 * xy


def permutation_test_mmd(X: torch.Tensor, Y: torch.Tensor, beta: float, n_perm: int = 200,
                         seed: int = 0) -> tuple[float, float]:
    """Exact permutation p-value for the (unbiased) global MMD^2 between X and Y."""
    K = pooled_kernel(X, Y, beta)
    n1, n = X.shape[0], K.shape[0]
    obs = mmd2_from_kernel(K, n1, unbiased=True)
    g = torch.Generator(device='cpu').manual_seed(seed)
    masks = torch.zeros(n_perm, n, dtype=torch.bool)
    for b in range(n_perm):
        masks[b, torch.randperm(n, generator=g)[:n1]] = True
    perm_stats = _batched_mmd2(K, masks.to(K.device)).cpu().numpy()
    p = (1.0 + np.sum(perm_stats >= obs - 1e-12)) / (1.0 + n_perm)
    return obs, float(p)


def permutation_test_cmmd(ref_by_class: dict, win_emb: torch.Tensor, win_lab: torch.Tensor, beta: float,
                          n_perm: int = 200, seed: int = 0, min_count: int = 10) -> tuple[float, float]:
    """Permutation p-value for C-MMD: membership is permuted *within* each class,
    so the null hypothesis is P_ref(E|Y=c) = P_win(E|Y=c) for every class c."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    obs_terms, perm_terms = [], []
    for c, R in ref_by_class.items():
        W = win_emb[win_lab == c]
        if W.shape[0] < min_count:
            continue
        K = pooled_kernel(R, W, beta)
        n1, n = R.shape[0], K.shape[0]
        obs_terms.append(mmd2_from_kernel(K, n1, unbiased=True))
        masks = torch.zeros(n_perm, n, dtype=torch.bool)
        for b in range(n_perm):
            masks[b, torch.randperm(n, generator=g)[:n1]] = True
        perm_terms.append(_batched_mmd2(K, masks.to(K.device)).cpu().numpy())
    if not obs_terms:
        return float('nan'), float('nan')
    obs = float(np.mean(obs_terms))
    perm_stats = np.mean(np.stack(perm_terms, 0), 0)
    p = (1.0 + np.sum(perm_stats >= obs - 1e-12)) / (1.0 + n_perm)
    return obs, float(p)


# ---------------------------------------------------------------------------
# Multiplicity control and alarm rules
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    """Boolean rejection vector controlling FDR at level q."""
    p = np.asarray(pvals, float)
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranked = p[order]
    thresh = q * (np.arange(1, len(p) + 1) / max(m, 1))
    below = ranked <= thresh
    reject = np.zeros(len(p), dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        reject[order[:kmax + 1]] = True
    return reject


def sustained_alarms(exceed: np.ndarray, k: int = 2) -> np.ndarray:
    """Alarm at t if the statistic exceeded the threshold at t, t-1, ..., t-k+1."""
    e = np.asarray(exceed, bool)
    out = np.zeros_like(e)
    run = 0
    for t, flag in enumerate(e):
        run = run + 1 if flag else 0
        out[t] = run >= k
    return out


def detection_delay(alarms: np.ndarray, onset: int) -> float:
    """Windows from onset to first alarm at/after onset (inf if never)."""
    idx = np.where(np.asarray(alarms, bool)[onset:])[0]
    return float(idx[0]) if len(idx) else float('inf')


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

@dataclass
class MonitorState:
    tau: float
    k: int
    run: int = 0

    def update(self, value: float) -> bool:
        self.run = self.run + 1 if value > self.tau else 0
        return self.run >= self.k


class CMMDMonitor:
    """Deployable C-MMD monitor: calibrate once, then score windows online.

    Parameters
    ----------
    ctx : statistics.Context
        Fixed reference sample, bandwidth and classes.
    alpha : float
        Nominal per-window false-positive rate used for the threshold.
    k : int
        Number of consecutive exceedances required for an alarm.
    """

    def __init__(self, ctx, alpha: float = 0.05, k: int = 2):
        from .statistics import cmmd
        self.ctx = ctx
        self.alpha = alpha
        self.k = k
        self._stat = cmmd
        self.null_values: np.ndarray | None = None
        self.state: MonitorState | None = None

    def calibrate(self, pool_emb: torch.Tensor, pool_lab: torch.Tensor, window_size: int,
                  n_null: int = 200, seed: int = 0) -> float:
        rng = np.random.RandomState(seed)
        vals = []
        for _ in range(n_null):
            e, l = draw_null_window(pool_emb, pool_lab, window_size, rng)
            vals.append(self._stat(self.ctx, e, l))
        self.null_values = np.asarray(vals)
        tau = threshold_from_null(self.null_values, self.alpha)
        self.state = MonitorState(tau=tau, k=self.k)
        return tau

    def score(self, win_emb: torch.Tensor, win_lab: torch.Tensor, n_perm: int = 0) -> dict:
        value = self._stat(self.ctx, win_emb, win_lab)
        out = {'cmmd': value, 'tau': self.state.tau, 'exceed': value > self.state.tau,
               'alarm': self.state.update(value)}
        if n_perm:
            _, p = permutation_test_cmmd(self.ctx.ref_by_class, win_emb, win_lab, self.ctx.beta, n_perm=n_perm)
            out['p_value'] = p
        return out
