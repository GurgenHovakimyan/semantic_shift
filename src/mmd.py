"""
Kernel two-sample statistics (MMD) used throughout the project.

Estimator specification (used consistently by every experiment):

* Kernel: mixture of Gaussian RBF kernels

      k(x, y) = (1/L) * sum_{l=0}^{L-1} exp(-||x - y||^2 / beta_l),
      beta_l  = beta * mul^(l - L//2),      L = 5, mul = 2

  i.e. bandwidths {beta/4, beta/2, beta, 2*beta, 4*beta}.
* Base bandwidth `beta`: MEAN squared pairwise Euclidean distance (diagonal
  excluded) of the pooled sample ("mean heuristic").  The median heuristic is
  available via ``median_sq_dist_bandwidth`` for the ablation in experiment 21.
* MMD^2 estimator: biased V-statistic (all pairs, including the diagonal),

      MMD_V^2 = mean(K_XX) + mean(K_YY) - 2 * mean(K_XY),

  which is non-negative and has a strictly positive finite-sample expectation
  under H0 (this is why 0% label swap yields ~0.005 rather than 0).
  The unbiased U-statistic is available through ``unbiased=True``.

Historical note: an earlier version of ``mmd_rbf`` returned the SUM over the
five kernels while ``mmd_rbf_chunked`` returned their MEAN, which inflated the
IMDb/DistilBERT stream values of experiment 02 by exactly 5x.  Both wrappers
now return the same quantity.
"""

from __future__ import annotations

import torch

KERNEL_MUL = 2.0
KERNEL_NUM = 5


# ---------------------------------------------------------------------------
# Distances and bandwidths
# ---------------------------------------------------------------------------

def sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix (N, M), clamped at zero."""
    x2 = (X * X).sum(dim=1, keepdim=True)
    y2 = (Y * Y).sum(dim=1, keepdim=True)
    D2 = x2 + y2.t() - 2.0 * (X @ Y.t())
    return D2.clamp_min_(0.0)


def _subsample(X: torch.Tensor, max_samples: int, generator: torch.Generator | None) -> torch.Tensor:
    if X.shape[0] <= max_samples:
        return X
    idx = torch.randperm(X.shape[0], generator=generator)[:max_samples].to(X.device)
    return X[idx]


def _pool(X, Y, max_samples, generator):
    if Y is None:
        return _subsample(X, 2 * max_samples, generator)
    return torch.cat([_subsample(X, max_samples, generator), _subsample(Y, max_samples, generator)], 0)


def mean_sq_dist_bandwidth(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
    max_samples: int = 1000,
    generator: torch.Generator | None = None,
) -> float:
    """Mean squared pairwise distance of the (pooled) sample, diagonal excluded."""
    Z = _pool(X, Y, max_samples, generator)
    n = Z.shape[0]
    D2 = sq_dists(Z, Z)
    return float(D2.sum().item() / (n * n - n))


def median_sq_dist_bandwidth(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
    max_samples: int = 1000,
    generator: torch.Generator | None = None,
) -> float:
    """Median squared pairwise distance (off-diagonal) of the pooled sample."""
    Z = _pool(X, Y, max_samples, generator)
    n = Z.shape[0]
    D2 = sq_dists(Z, Z)
    off = D2[~torch.eye(n, dtype=torch.bool, device=Z.device)]
    return float(off.median().item())


# ---------------------------------------------------------------------------
# Kernels and MMD
# ---------------------------------------------------------------------------

def bandwidth_list(beta: float, kernel_mul: float = KERNEL_MUL, kernel_num: int = KERNEL_NUM) -> list[float]:
    base = beta / (kernel_mul ** (kernel_num // 2))
    return [base * (kernel_mul ** i) for i in range(kernel_num)]


def rbf_mixture_kernel(D2: torch.Tensor, beta: float, kernel_mul: float = KERNEL_MUL, kernel_num: int = KERNEL_NUM) -> torch.Tensor:
    """Average of Gaussian kernels over the multi-scale bandwidth list."""
    K = torch.zeros_like(D2)
    for bw in bandwidth_list(beta, kernel_mul, kernel_num):
        K += torch.exp(-D2 / bw)
    return K / kernel_num


def mmd2_from_kernel(K: torch.Tensor, n1: int, unbiased: bool = False) -> float:
    """MMD^2 from a pooled kernel matrix whose first n1 rows/cols are sample X."""
    KXX = K[:n1, :n1]
    KYY = K[n1:, n1:]
    KXY = K[:n1, n1:]
    n2 = K.shape[0] - n1
    if unbiased:
        xx = (KXX.sum() - KXX.diagonal().sum()) / (n1 * (n1 - 1))
        yy = (KYY.sum() - KYY.diagonal().sum()) / (n2 * (n2 - 1))
    else:
        xx = KXX.mean()
        yy = KYY.mean()
    return float((xx + yy - 2.0 * KXY.mean()).item())


def pooled_kernel(X: torch.Tensor, Y: torch.Tensor, beta: float, kernel_mul: float = KERNEL_MUL, kernel_num: int = KERNEL_NUM) -> torch.Tensor:
    Z = torch.cat([X, Y], 0)
    return rbf_mixture_kernel(sq_dists(Z, Z), beta, kernel_mul, kernel_num)


def mmd2_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    beta: float | None = None,
    unbiased: bool = False,
    kernel_mul: float = KERNEL_MUL,
    kernel_num: int = KERNEL_NUM,
    generator: torch.Generator | None = None,
) -> float:
    """MMD^2 between samples X (N, D) and Y (M, D) with the project kernel.

    If ``beta`` is None the mean heuristic is computed on the pooled sample
    (window-adaptive bandwidth).  The monitoring protocol instead passes a
    bandwidth fixed on the reference set so that scores are comparable across
    windows and against a calibrated threshold.
    """
    if beta is None:
        beta = mean_sq_dist_bandwidth(X, Y, generator=generator)
    K = pooled_kernel(X, Y, beta, kernel_mul, kernel_num)
    return mmd2_from_kernel(K, X.shape[0], unbiased=unbiased)


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers used by experiments 01-15
# ---------------------------------------------------------------------------

def gaussian_kernel(source, target, kernel_mul=KERNEL_MUL, kernel_num=KERNEL_NUM, fix_sigma=None):
    """Pooled multi-scale kernel matrix (mean over kernels) for the old API."""
    Z = torch.cat([source, target], 0)
    D2 = sq_dists(Z, Z)
    n = Z.shape[0]
    beta = fix_sigma if fix_sigma else float(D2.sum().item() / (n * n - n))
    return rbf_mixture_kernel(D2, beta, kernel_mul, kernel_num)


def mmd_rbf(source, target, kernel_mul=KERNEL_MUL, kernel_num=KERNEL_NUM, fix_sigma=None):
    """Biased MMD^2 with window-adaptive mean-heuristic bandwidth (returns a tensor)."""
    K = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    return torch.tensor(mmd2_from_kernel(K, source.shape[0], unbiased=False))


def pairwise_distances_chunked(X, Y, chunk_size=256):
    """
    Compute pairwise squared L2 distances between X and Y in chunks.
    
    Args:
        X: Tensor of shape (N, D)
        Y: Tensor of shape (M, D)
        chunk_size: Number of rows to process at once
        
    Returns:
        Distance matrix of shape (N, M)
    """
    N = X.shape[0]
    M = Y.shape[0]
    
    # Pre-compute squared norms
    X_sqnorm = (X ** 2).sum(dim=1)  # (N,)
    Y_sqnorm = (Y ** 2).sum(dim=1)  # (M,)
    
    # Initialize result
    distances = torch.zeros(N, M, dtype=X.dtype, device=X.device)
    
    # Compute in chunks to save memory
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        X_chunk = X[i:i_end]  # (chunk, D)
        
        # Compute distances for this chunk
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        XY = torch.mm(X_chunk, Y.t())  # (chunk, M)
        distances[i:i_end] = X_sqnorm[i:i_end].unsqueeze(1) + Y_sqnorm.unsqueeze(0) - 2 * XY
    
    return distances


def compute_bandwidth(source, target, chunk_size=256):
    """Mean-heuristic bandwidth on (at most 1000 + 1000) pooled samples."""
    return mean_sq_dist_bandwidth(source, target, max_samples=1000)


def mmd_rbf_chunked(source, target, kernel_mul=KERNEL_MUL, kernel_num=KERNEL_NUM, fix_sigma=None, chunk_size=256):
    """Same estimator as ``mmd_rbf`` (mean over kernels, V-statistic); kept for
    backwards compatibility with experiments 03-15."""
    beta = fix_sigma if fix_sigma else compute_bandwidth(source, target, chunk_size)
    return torch.tensor(mmd2_rbf(source, target, beta=beta, kernel_mul=kernel_mul, kernel_num=kernel_num))