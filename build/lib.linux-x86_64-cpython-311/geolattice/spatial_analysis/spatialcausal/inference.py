"""Inference under spatial dependence.

Every estimator in this package is asymptotically linear: ``theta_hat - theta
~ mean(psi_i)``, where ``psi_i`` is the unit-level influence (score)
contribution. Spatially robust standard errors therefore only need the
scores and the coordinates:

* ``"iid"`` — the usual ``sqrt(var(psi) / n)``.
* ``"conley"`` — Conley (1999) spatial HAC: covariance terms between pairs
  closer than ``cutoff`` are kept, down-weighted by a distance kernel.
* ``"block"`` — cluster-robust over spatial blocks (e.g. the cross-fitting
  blocks, or administrative units).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from geolattice.spatial_analysis.spatialcausal._utils import as_coords
from geolattice.spatial_analysis.spatialcausal.crossfit import SpatialBlockKFold


def _kernel(u: np.ndarray, kind: str) -> np.ndarray:
    if kind == "bartlett":
        return np.clip(1.0 - u, 0.0, None)
    if kind == "uniform":
        return (u <= 1.0).astype(float)
    if kind == "epanechnikov":
        return np.clip(1.0 - u ** 2, 0.0, None)
    raise ValueError("kernel must be 'bartlett', 'uniform' or 'epanechnikov'")


def conley_variance(psi: np.ndarray, coords, cutoff: float, kernel: str = "bartlett") -> float:
    """Spatial HAC variance of ``mean(psi)``.

    Parameters
    ----------
    psi : (n,) influence contributions (centred).
    coords : (n, 2) projected coordinates.
    cutoff : distance beyond which covariance is assumed zero.
    kernel : distance-decay kernel for pairs inside the cutoff.

    Notes
    -----
    Uses a sparse distance matrix (``cKDTree.sparse_distance_matrix``), so the
    cost scales with the number of pairs inside ``cutoff``, not with n².
    In two dimensions the Bartlett kernel is not guaranteed to give a
    positive estimate in small samples; the result is floored at zero and
    a non-positive value falls back to the i.i.d. variance.
    """
    psi = np.asarray(psi, float).ravel()
    coords = as_coords(coords)
    n = len(psi)
    tree = cKDTree(coords)
    D = tree.sparse_distance_matrix(tree, max_distance=cutoff, output_type="coo_matrix")
    off = D.row != D.col
    rows, cols, d = D.row[off], D.col[off], D.data[off]
    k = _kernel(d / cutoff, kernel)
    cross = float(np.sum(k * psi[rows] * psi[cols]))
    var = (np.sum(psi ** 2) + cross) / n ** 2
    if var <= 0:
        var = np.sum(psi ** 2) / n ** 2
    return float(var)


def block_variance(psi: np.ndarray, blocks: np.ndarray) -> float:
    """Cluster-robust variance of ``mean(psi)`` with spatial blocks as clusters."""
    psi = np.asarray(psi, float).ravel()
    n = len(psi)
    _, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=psi)
    g = len(sums)
    adj = g / (g - 1) if g > 1 else 1.0
    return float(adj * np.sum(sums ** 2) / n ** 2)


def standard_error(psi: np.ndarray, coords=None, method: str = "iid",
                   cutoff: Optional[float] = None, blocks: Optional[np.ndarray] = None,
                   kernel: str = "bartlett") -> float:
    """Standard error of ``mean(psi)`` under the chosen dependence model."""
    psi = np.asarray(psi, float).ravel()
    psi = psi - psi.mean()
    n = len(psi)
    if method == "iid":
        return float(np.sqrt(np.sum(psi ** 2) / n ** 2))
    if method == "conley":
        if coords is None or cutoff is None:
            raise ValueError("method='conley' needs coords and cutoff")
        return float(np.sqrt(conley_variance(psi, coords, cutoff, kernel)))
    if method == "block":
        if blocks is None:
            if coords is None:
                raise ValueError("method='block' needs blocks or coords")
            blocks = SpatialBlockKFold(n_splits=2, n_blocks=max(10, n // 50)).make_blocks(coords)
        return float(np.sqrt(block_variance(psi, blocks)))
    raise ValueError("method must be 'iid', 'conley' or 'block'")


def suggest_cutoff(residuals: np.ndarray, coords, w=None, quantile: float = 0.25) -> float:
    """Heuristic Conley cutoff: the distance at which the residual
    correlogram first drops below 0.1, or a quantile of pairwise distances.

    This is a starting point, not a substitute for a sensitivity check over
    several cutoffs.
    """
    coords = as_coords(coords)
    r = np.asarray(residuals, float)
    r = (r - r.mean()) / (r.std() + 1e-12)
    tree = cKDTree(coords)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=min(len(coords), 400), replace=False)
    sub = coords[idx]
    dd = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
    dmax = np.quantile(dd[dd > 0], 0.5)
    edges = np.linspace(0, dmax, 16)[1:]
    prev = 0.0
    for e in edges:
        D = tree.sparse_distance_matrix(tree, max_distance=e, output_type="coo_matrix")
        m = (D.row != D.col) & (D.data > prev)
        if m.sum() > 30:
            corr = np.mean(r[D.row[m]] * r[D.col[m]])
            if corr < 0.1:
                return float(e)
        prev = e
    return float(np.quantile(dd[dd > 0], quantile))


def spatial_block_bootstrap(statistic: Callable[[np.ndarray], float], coords, n_boot: int = 200,
                            n_blocks: Optional[int] = None, random_state: int = 0):
    """Block bootstrap: resample whole spatial blocks with replacement.

    ``statistic`` receives an index array (with repeats) and returns a scalar.
    Returns ``(estimates, se)``.
    """
    coords = as_coords(coords)
    n = len(coords)
    n_blocks = n_blocks or max(10, n // 40)
    blocks = SpatialBlockKFold(n_splits=2, n_blocks=n_blocks, random_state=random_state).make_blocks(coords)
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(random_state)
    est = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in pick])
        est[b] = statistic(idx)
    return est, float(np.std(est, ddof=1))


def block_permutation_test(statistic: Callable[[np.ndarray], float], treatment: np.ndarray, coords,
                           n_perm: int = 499, n_blocks: Optional[int] = None, random_state: int = 0):
    """Randomisation test that shuffles treatment *within* spatial blocks.

    Shuffling within blocks keeps the large-scale spatial pattern of
    treatment, so the null distribution is not inflated by spatial
    clustering of treatment. ``statistic(t)`` returns a scalar for a given
    treatment vector. Returns ``(observed, p_value, null_draws)``.
    """
    coords = as_coords(coords)
    t = np.asarray(treatment).copy()
    n_blocks = n_blocks or max(10, len(t) // 30)
    blocks = SpatialBlockKFold(n_splits=2, n_blocks=n_blocks, random_state=random_state).make_blocks(coords)
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(random_state)
    obs = statistic(t)
    null = np.empty(n_perm)
    for k in range(n_perm):
        tp = t.copy()
        for g in groups:
            tp[g] = rng.permutation(t[g])
        null[k] = statistic(tp)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return float(obs), float(p), null
