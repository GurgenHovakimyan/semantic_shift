"""
Statistical monitoring protocol for class-conditional drift.

Components
----------
1. **Null calibration.** The C-MMD threshold is conditional on the observed
   class counts in each window. Calibration windows are resampled separately
   within each class, preserving the requested counts. This accounts for the
   count-dependent variance of the unbiased per-class estimators.
2. **Permutation test.** For kernel statistics a Monte Carlo permutation
   p-value is obtained by permuting reference/window membership (within each
   class for C-MMD) on a pooled kernel matrix. The bandwidth is selected from
   the pooled observations, independently of their group assignment.
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


def draw_class_count_window(
    pool_emb: torch.Tensor,
    pool_lab: torch.Tensor,
    counts: dict[int, int],
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a null window with the specified count for every class."""
    embeddings, labels = [], []
    for label, count in counts.items():
        count = int(count)
        if count < 0:
            raise ValueError(f'negative count for class {label}: {count}')
        if count == 0:
            continue
        indices = torch.where(pool_lab == label)[0].cpu().numpy()
        if len(indices) == 0:
            raise ValueError(f'calibration pool has no observations for class {label}')
        selected = rng.choice(indices, size=count, replace=True)
        selected_t = torch.as_tensor(selected, device=pool_emb.device)
        embeddings.append(pool_emb[selected_t])
        labels.append(torch.full((count,), label, dtype=pool_lab.dtype, device=pool_lab.device))
    if not embeddings:
        raise ValueError('class counts must include at least one observation')
    return torch.cat(embeddings), torch.cat(labels)


def threshold_from_null(null_values: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical (1 - alpha) quantile with conservative higher interpolation."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    v = np.asarray(null_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError('null_values must contain at least one finite value')
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


def pooled_mean_sq_dist(Z: torch.Tensor) -> float:
    """Mean squared pairwise distance of all rows of ``Z`` (diagonal excluded), in O(n d).

    Uses sum_{i != j} ||z_i - z_j||^2 = 2 n sum_i ||z_i - mean(z)||^2, so it equals
    ``mmd.mean_sq_dist_bandwidth`` on the full sample without forming the distance matrix.
    """
    Zd = Z.double()
    Zc = Zd - Zd.mean(0, keepdim=True)
    return float(2.0 * (Zc * Zc).sum().item() / (Z.shape[0] - 1))


def permutation_test_mmd(X: torch.Tensor, Y: torch.Tensor, n_perm: int = 200,
                         seed: int = 0) -> tuple[float, float]:
    """Monte Carlo permutation test for global MMD with pooled, invariant bandwidth."""
    if n_perm < 1:
        raise ValueError('n_perm must be positive')
    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError('both samples must contain at least two observations')
    pooled = torch.cat([X, Y])
    beta = pooled_mean_sq_dist(pooled)
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


def permutation_test_cmmd(ref_by_class: dict, win_emb: torch.Tensor, win_lab: torch.Tensor,
                          n_perm: int = 200, seed: int = 0, min_count: int = 10) -> tuple[float, float]:
    """Monte Carlo C-MMD permutation test with a permutation-invariant bandwidth.

    The bandwidth is selected once from all pooled class observations before
    group memberships are permuted, so it cannot encode the observed split.
    Returns ``(nan, nan)`` when any class has fewer than ``min_count`` window
    observations, rather than testing only the remaining classes.
    """
    if n_perm < 1:
        raise ValueError('n_perm must be positive')
    if any(int((win_lab == c).sum().item()) < min_count for c in ref_by_class):
        return float('nan'), float('nan')
    pooled = torch.cat([torch.cat([R, win_emb[win_lab == c]]) for c, R in ref_by_class.items()])
    beta = pooled_mean_sq_dist(pooled)
    g = torch.Generator(device='cpu').manual_seed(seed)
    obs_terms, perm_terms = [], []
    for c, R in ref_by_class.items():
        W = win_emb[win_lab == c]
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
    """Return BH rejections; FDR control requires independence or PRDS dependence."""
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
    """Deployable C-MMD monitor with cached thresholds by observed class counts.

    Parameters
    ----------
    ctx : statistics.Context
        Fixed reference sample, bandwidth and classes.
    alpha : float
        Nominal per-window false-positive rate used for each conditional threshold.
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
        self._conditional_calibrator: ClassCountConditionalCalibrator | None = None

    def calibrate(self, pool_emb: torch.Tensor, pool_lab: torch.Tensor, window_size: int,
                  n_null: int = 200, seed: int = 0) -> float:
        classes = list(self.ctx.classes)
        totals = {c: int((pool_lab == c).sum().item()) for c in classes}
        total = sum(totals.values())
        if total == 0:
            raise ValueError('calibration pool must not be empty')
        counts = {c: int(round(window_size * totals[c] / total)) for c in classes}
        counts[classes[-1]] += window_size - sum(counts.values())
        self._conditional_calibrator = ClassCountConditionalCalibrator(
            self._stat, self.ctx, pool_emb, pool_lab, classes,
            alpha=self.alpha, n_null=n_null, seed=seed,
        )
        tau = self._conditional_calibrator.threshold(counts)
        self.state = MonitorState(tau=tau, k=self.k)
        return tau

    def score(self, win_emb: torch.Tensor, win_lab: torch.Tensor, n_perm: int = 0) -> dict:
        if self._conditional_calibrator is None or self.state is None:
            raise RuntimeError('calibrate the monitor before scoring windows')
        value = self._stat(self.ctx, win_emb, win_lab)
        counts = {c: int((win_lab == c).sum().item()) for c in self.ctx.classes}
        tau = self._conditional_calibrator.threshold(counts)
        self.state.tau = tau
        out = {'cmmd': value, 'tau': tau, 'exceed': value > tau,
               'alarm': self.state.update(value)}
        if n_perm:
            _, p = permutation_test_cmmd(self.ctx.ref_by_class, win_emb, win_lab, n_perm=n_perm)
            out['p_value'] = p
        return out


class ClassCountConditionalCalibrator:
    """Cache bootstrap C-MMD thresholds for observed class-count signatures.

    Calibration examples are sampled with replacement within each class. The
    resulting quantiles condition on the reference sample, class-specific
    calibration pools, and the exact class counts in the scored window.
    """

    def __init__(self, statistic, context, pool_emb, pool_lab, classes, alpha=0.05,
                 n_null=200, seed=0, min_count=10):
        if n_null < 1:
            raise ValueError('n_null must be positive')
        self.statistic = statistic
        self.context = context
        self.pool_emb = pool_emb
        self.pool_lab = pool_lab
        self.classes = tuple(classes)
        self.alpha = alpha
        self.n_null = n_null
        self.seed = seed
        self.min_count = min_count
        self._thresholds: dict[tuple[int, ...], float] = {}

    def threshold(self, counts: dict[int, int]) -> float:
        signature = tuple(int(counts.get(c, 0)) for c in self.classes)
        if any(n < self.min_count for n in signature):
            raise ValueError(
                f'C-MMD requires at least {self.min_count} samples per class; '
                f'observed counts are {signature}'
            )
        if signature not in self._thresholds:
            seed = int(
                np.random.SeedSequence([self.seed, *signature]).generate_state(1)[0]
            )
            rng = np.random.RandomState(seed)
            class_counts = dict(zip(self.classes, signature))
            values = [
                self.statistic(
                    self.context,
                    *draw_class_count_window(self.pool_emb, self.pool_lab, class_counts, rng),
                )
                for _ in range(self.n_null)
            ]
            self._thresholds[signature] = threshold_from_null(np.asarray(values), self.alpha)
        return self._thresholds[signature]
