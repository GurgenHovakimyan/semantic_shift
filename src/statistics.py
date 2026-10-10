"""
Drift statistics (detectors) with a common interface.

Every detector is a function ``f(ctx, win_emb, win_lab) -> float`` where larger
values indicate *more* evidence of drift.  ``ctx`` is a :class:`Context` built
once per (dataset, encoder) from the reference ``fit`` pool and holds all
quantities that must stay fixed during monitoring: the fixed reference sample,
the kernel bandwidth, and the reference-trained downstream classifier.

Unsupervised (label-free) detectors
    global_mmd      MMD^2 between reference and window embeddings
    energy          multivariate energy distance (Szekely & Rizzo)
    c2st_linear     classifier two-sample test, logistic regression, hold-out accuracy
    c2st_mlp        classifier two-sample test, MLP, hold-out accuracy
    bbsd_ks         black-box shift detection: KS statistic on the deployed
                    classifier's predicted probabilities (Rabanser et al., 2019)

Label-aware (class-conditional) detectors
    cmmd            class-conditional MMD^2 averaged over classes (ours);
                    unbiased U-statistic so that the null expectation does not
                    depend on the per-class window counts
    cmmd_v          biased V-statistic variant (ablation only)
    cond_energy     class-conditional energy distance averaged over classes
    cond_c2st       class-conditional linear C2ST averaged over classes
    centroid_shift  class-conditional squared mean-embedding shift / bandwidth
                    (= linear-kernel C-MMD)
    joint_mmd       MMD^2 on the joint (E, Y) with the product kernel k(e,e')·1[y=y']
    prior_shift     |z| statistic of a binomial test on the window's class
                    proportion against the reference prior (label-shift monitor)
    error_rate      error rate of the reference-trained classifier on the
                    labelled window (supervised performance monitoring)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from .mmd import mean_sq_dist_bandwidth, mmd2_rbf, pooled_kernel, mmd2_from_kernel, sq_dists

MIN_CLASS_COUNT = 10
C2ST_CAP = 300


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class Context:
    beta: float
    ref_global: torch.Tensor
    ref_global_lab: torch.Tensor
    ref_by_class: dict
    classes: list
    prior: float = 0.5
    clf: Pipeline | None = None
    ref_scores: np.ndarray | None = None   # classifier scores on held-out reference (BBSD)
    ref_error: float = float('nan')
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def device(self):
        return self.ref_global.device


def build_context(ds, n_ref_global: int = 1000, n_ref_class: int = 500, train_classifier: bool = True,
                  n_clf_train: int = 20000, seed: int = 0) -> Context:
    """Fix reference sample, bandwidth and downstream classifier from the ``fit`` pool."""
    fit_emb, fit_lab, _ = ds.pool('fit')
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(fit_emb.shape[0], generator=g).to(fit_emb.device)

    ref_global = fit_emb[perm[:n_ref_global]]
    ref_global_lab = fit_lab[perm[:n_ref_global]]
    ref_by_class = {}
    for c in ds.classes:
        idx = perm[fit_lab[perm] == c][:n_ref_class]
        ref_by_class[c] = fit_emb[idx]

    # Bandwidth: mean squared pairwise distance of the fixed global reference sample.
    beta = mean_sq_dist_bandwidth(ref_global, None, max_samples=n_ref_global, generator=g)

    ctx = Context(beta=beta, ref_global=ref_global, ref_global_lab=ref_global_lab,
                  ref_by_class=ref_by_class, classes=list(ds.classes), prior=ds.ref_prior, seed=seed)

    if train_classifier:
        tr = perm[:min(n_clf_train, fit_emb.shape[0])]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(fit_emb[tr].cpu().numpy(), fit_lab[tr].cpu().numpy())
        ctx.clf = clf
        # Held-out reference scores/error for BBSD and performance monitoring
        cal_emb, cal_lab, _ = ds.pool('cal')
        sub = torch.randperm(cal_emb.shape[0], generator=g)[:2000].to(cal_emb.device)
        X = cal_emb[sub].cpu().numpy()
        ctx.ref_scores = clf.predict_proba(X)[:, 1]
        ctx.ref_error = float(np.mean(clf.predict(X) != cal_lab[sub].cpu().numpy()))
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _cap(X: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    if X.shape[0] <= n:
        return X
    g = torch.Generator().manual_seed(seed)
    return X[torch.randperm(X.shape[0], generator=g)[:n].to(X.device)]


def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    exy = sq_dists(X, Y).sqrt().mean()
    exx = sq_dists(X, X).sqrt().mean()
    eyy = sq_dists(Y, Y).sqrt().mean()
    return float((2 * exy - exx - eyy).item())


def _holdout_c2st(X: torch.Tensor, Y: torch.Tensor, model: str, seed: int) -> float:
    """Hold-out accuracy of a classifier separating X from Y (chance = 0.5)."""
    Xc, Yc = _cap(X, C2ST_CAP, seed), _cap(Y, C2ST_CAP, seed + 1)
    n = min(Xc.shape[0], Yc.shape[0])
    Xc, Yc = Xc[:n], Yc[:n]
    rng = np.random.RandomState(seed)
    half = n // 2
    px, py = rng.permutation(n), rng.permutation(n)
    Xtr = np.vstack([_np(Xc[px[:half]]), _np(Yc[py[:half]])])
    ytr = np.r_[np.zeros(half), np.ones(half)]
    Xte = np.vstack([_np(Xc[px[half:]]), _np(Yc[py[half:]])])
    yte = np.r_[np.zeros(n - half), np.ones(n - half)]
    if model == 'linear':
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    else:
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                                            early_stopping=True, random_state=seed))
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def _class_split(ctx: Context, win_emb: torch.Tensor, win_lab: torch.Tensor):
    for c in ctx.classes:
        Wc = win_emb[win_lab == c]
        if Wc.shape[0] >= MIN_CLASS_COUNT:
            yield c, Wc


# ---------------------------------------------------------------------------
# Unsupervised detectors
# ---------------------------------------------------------------------------

def global_mmd(ctx: Context, win_emb, win_lab=None) -> float:
    return mmd2_rbf(ctx.ref_global, win_emb, beta=ctx.beta, unbiased=True)


def energy(ctx: Context, win_emb, win_lab=None) -> float:
    return energy_distance(ctx.ref_global, win_emb)


def c2st_linear(ctx: Context, win_emb, win_lab=None) -> float:
    return _holdout_c2st(ctx.ref_global, win_emb, 'linear', ctx.seed)


def c2st_mlp(ctx: Context, win_emb, win_lab=None) -> float:
    return _holdout_c2st(ctx.ref_global, win_emb, 'mlp', ctx.seed)


def bbsd_ks(ctx: Context, win_emb, win_lab=None) -> float:
    scores = ctx.clf.predict_proba(_np(win_emb))[:, 1]
    return float(ks_2samp(ctx.ref_scores, scores).statistic)


# ---------------------------------------------------------------------------
# Label-aware detectors
# ---------------------------------------------------------------------------

def cmmd(ctx: Context, win_emb, win_lab) -> float:
    vals = [mmd2_rbf(ctx.ref_by_class[c], Wc, beta=ctx.beta, unbiased=True)
            for c, Wc in _class_split(ctx, win_emb, win_lab)]
    return float(np.mean(vals)) if vals else float('nan')


def cmmd_v(ctx: Context, win_emb, win_lab) -> float:
    vals = [mmd2_rbf(ctx.ref_by_class[c], Wc, beta=ctx.beta, unbiased=False)
            for c, Wc in _class_split(ctx, win_emb, win_lab)]
    return float(np.mean(vals)) if vals else float('nan')


def cmmd_per_class(ctx: Context, win_emb, win_lab) -> dict:
    return {c: mmd2_rbf(ctx.ref_by_class[c], Wc, beta=ctx.beta, unbiased=True)
            for c, Wc in _class_split(ctx, win_emb, win_lab)}


def prior_shift(ctx: Context, win_emb, win_lab) -> float:
    n = win_lab.shape[0]
    p_hat = float(win_lab.float().mean().item())
    se = np.sqrt(ctx.prior * (1 - ctx.prior) / n)
    return float(abs(p_hat - ctx.prior) / se)


def cond_energy(ctx: Context, win_emb, win_lab) -> float:
    vals = [energy_distance(ctx.ref_by_class[c], Wc) for c, Wc in _class_split(ctx, win_emb, win_lab)]
    return float(np.mean(vals)) if vals else float('nan')


def cond_c2st(ctx: Context, win_emb, win_lab) -> float:
    vals = [_holdout_c2st(ctx.ref_by_class[c], Wc, 'linear', ctx.seed) for c, Wc in _class_split(ctx, win_emb, win_lab)]
    return float(np.mean(vals)) if vals else float('nan')


def centroid_shift(ctx: Context, win_emb, win_lab) -> float:
    vals = [float(((ctx.ref_by_class[c].mean(0) - Wc.mean(0)) ** 2).sum().item()) / ctx.beta
            for c, Wc in _class_split(ctx, win_emb, win_lab)]
    return float(np.mean(vals)) if vals else float('nan')


def joint_mmd(ctx: Context, win_emb, win_lab) -> float:
    K = pooled_kernel(ctx.ref_global, win_emb, ctx.beta)
    y = torch.cat([ctx.ref_global_lab, win_lab]).view(-1, 1)
    K = K * (y == y.t()).to(K.dtype)
    return mmd2_from_kernel(K, ctx.ref_global.shape[0], unbiased=True)


def error_rate(ctx: Context, win_emb, win_lab) -> float:
    pred = ctx.clf.predict(_np(win_emb))
    return float(np.mean(pred != _np(win_lab)))


def downstream_f1(ctx: Context, win_emb, win_lab) -> float:
    pred = ctx.clf.predict(_np(win_emb))
    return float(f1_score(_np(win_lab), pred, average='macro'))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

UNSUPERVISED = {
    'global_mmd': global_mmd,
    'energy': energy,
    'c2st_linear': c2st_linear,
    'c2st_mlp': c2st_mlp,
    'bbsd_ks': bbsd_ks,
}

LABEL_AWARE = {
    'cmmd': cmmd,
    'cond_energy': cond_energy,
    'cond_c2st': cond_c2st,
    'centroid_shift': centroid_shift,
    'joint_mmd': joint_mmd,
    'prior_shift': prior_shift,
    'error_rate': error_rate,
}

ALL_DETECTORS = {**UNSUPERVISED, **LABEL_AWARE}

PRETTY = {
    'global_mmd': 'Global MMD', 'energy': 'Energy distance', 'c2st_linear': 'C2ST (linear)',
    'c2st_mlp': 'C2ST (MLP)', 'bbsd_ks': 'BBSD-KS', 'cmmd': 'C-MMD (ours)', 'cmmd_v': 'C-MMD (V-stat.)',
    'cond_energy': 'Cond. energy', 'cond_c2st': 'Cond. C2ST', 'centroid_shift': 'Centroid shift',
    'joint_mmd': 'Joint MMD (E,Y)', 'prior_shift': 'Prior-shift z', 'error_rate': 'Error-rate monitor',
}


def compute_all(ctx: Context, win_emb, win_lab, detectors=None) -> dict:
    detectors = detectors or ALL_DETECTORS
    out = {name: fn(ctx, win_emb, win_lab) for name, fn in detectors.items()}
    if ctx.clf is not None:
        out['f1'] = downstream_f1(ctx, win_emb, win_lab)
    return out
