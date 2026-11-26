"""
Sequential Indicator Simulation with local prior means (SKlm).

At each node the local CDF is the secondary prior (for example from a
proportional-odds model), corrected by simple kriging of indicator residuals
at nearby data and previously simulated nodes:

    F*(u; z_k) = p(u; z_k) + sum_a lambda_a(u; z_k) [ i(u_a; z_k) - p(u_a; z_k) ]

Far from any neighbour the weights vanish and the node falls back on the
prior. Residual variograms are supplied per threshold as standardized models
(nugget + partial sill + range) together with residual variances that restore
indicator units.

This is the continuous-variable / local-prior-means counterpart of
`sisim` (global prior) and differs from `sicosim` (Markov–Bayes soft data).
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "order_relations",
    "residual_covariances",
    "sk_correction",
    "sklm_cdf",
    "sis_local_means",
]

ModelLike = Mapping[str, Union[str, float]]
ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Residual covariance helpers
# ---------------------------------------------------------------------------


def _spherical(h: np.ndarray, a: float) -> np.ndarray:
    """Standardized spherical structure (sill = 1), practical range ``a``."""
    hr = np.minimum(np.asarray(h, dtype=float) / float(a), 1.0)
    return 1.5 * hr - 0.5 * hr**3


def _exponential(h: np.ndarray, a: float) -> np.ndarray:
    """Standardized exponential structure (sill = 1), practical range ``a``."""
    return 1.0 - np.exp(-3.0 * np.asarray(h, dtype=float) / float(a))


_STRUCTURES: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "spherical": _spherical,
    "exponential": _exponential,
}


def _as_models(models: Sequence[ModelLike]) -> Tuple[Dict[str, float], ...]:
    out = []
    for i, m in enumerate(models):
        try:
            name = str(m["model"]).strip().lower()
            nugget = float(m["nugget"])
            psill = float(m["psill"])
            rng = float(m["range"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"models[{i}] must have keys model, nugget, psill, range; got {m!r}"
            ) from exc
        if name not in _STRUCTURES:
            known = ", ".join(sorted(_STRUCTURES))
            raise ValueError(f"Unknown residual structure {name!r}. Known: {known}")
        if rng <= 0:
            raise ValueError(f"models[{i}] range must be > 0")
        if nugget < 0 or psill < 0:
            raise ValueError(f"models[{i}] nugget and psill must be >= 0")
        out.append({"model": name, "nugget": nugget, "psill": psill, "range": rng})
    if not out:
        raise ValueError("models must contain at least one threshold model")
    return tuple(out)


def order_relations(F: np.ndarray) -> np.ndarray:
    """
    Correct a CDF (or a batch of CDFs) for order-relation violations.

    Clips to ``[0, 1]``, then averages an upward non-decreasing pass with a
    downward non-increasing pass (GSLIB ``ordrel`` continuous branch). Works
    on arrays whose last axis indexes ascending thresholds.
    """
    F = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)
    up = np.maximum.accumulate(F, axis=-1)
    down = np.minimum.accumulate(F[..., ::-1], axis=-1)[..., ::-1]
    return 0.5 * (up + down)


def residual_covariances(
    models: Sequence[ModelLike],
    variances: ArrayLike,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Build a residual-covariance evaluator for ``K`` thresholds.

    Parameters
    ----------
    models :
        Length-``K`` sequence of standardized residual models with keys
        ``model`` (``"spherical"`` / ``"exponential"``), ``nugget``, ``psill``,
        ``range``.
    variances :
        Residual variances (length ``K``) that rescale the standardized models
        back to indicator units.

    Returns
    -------
    cov : callable
        ``cov(h)`` returns shape ``(K, *h.shape)`` residual covariances for a
        distance array ``h``.
    sill0 : ndarray, shape (K,)
        Residual sill ``(nugget + psill) * variance`` per threshold.
    """
    models_t = _as_models(models)
    var = np.asarray(variances, dtype=float).ravel()
    if var.shape != (len(models_t),):
        raise ValueError(
            f"variances must have length {len(models_t)}, got shape {var.shape}"
        )
    if np.any(var < 0):
        raise ValueError("variances must be >= 0")
    sill0 = np.array(
        [(m["nugget"] + m["psill"]) * v for m, v in zip(models_t, var)],
        dtype=float,
    )

    def cov(h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return np.stack(
            [
                m["psill"] * v * (1.0 - _STRUCTURES[m["model"]](h, m["range"]))
                for m, v in zip(models_t, var)
            ]
        )

    return cov, sill0


def sk_correction(
    target: ArrayLike,
    nb_xy: np.ndarray,
    nb_resid: np.ndarray,
    cov: Callable[[np.ndarray], np.ndarray],
    sill0: np.ndarray,
) -> np.ndarray:
    """
    Simple-kriging estimate of the ``K`` residuals at ``target``.

    Parameters
    ----------
    target :
        Coordinates of the estimation point, shape ``(2,)``.
    nb_xy :
        Neighbour coordinates, shape ``(n, 2)``.
    nb_resid :
        Neighbour residuals, shape ``(n, K)``.
    cov, sill0 :
        From :func:`residual_covariances`.

    Returns
    -------
    ndarray, shape (K,)
        Kriged residual corrections.
    """
    target = np.asarray(target, dtype=float).ravel()
    nb_xy = np.asarray(nb_xy, dtype=float)
    nb_resid = np.asarray(nb_resid, dtype=float)
    if nb_xy.ndim != 2 or nb_xy.shape[1] != 2:
        raise ValueError("nb_xy must have shape (n, 2)")
    if nb_resid.ndim != 2 or nb_resid.shape[0] != nb_xy.shape[0]:
        raise ValueError("nb_resid must have shape (n, K) matching nb_xy")
    if target.shape != (2,):
        raise ValueError("target must have shape (2,)")

    d0 = np.sqrt(((nb_xy - target) ** 2).sum(axis=1))
    h = np.sqrt(((nb_xy[:, None, :] - nb_xy[None, :, :]) ** 2).sum(axis=-1))
    C = cov(h)
    idx = np.arange(len(nb_xy))
    C[:, idx, idx] = sill0[:, None] * (1.0 + 1e-8)
    lam = np.linalg.solve(C, cov(d0)[:, :, None])[:, :, 0]  # (K, n)
    return np.einsum("kn,nk->k", lam, nb_resid)


def sklm_cdf(
    points: np.ndarray,
    prior_points: np.ndarray,
    data_xy: np.ndarray,
    data_resid: np.ndarray,
    models: Sequence[ModelLike],
    variances: ArrayLike,
    *,
    k_nn: int = 16,
    radius: float = 6000.0,
) -> np.ndarray:
    """
    Local CDFs at ``points`` from priors plus data residuals (no sequential path).

    Useful for cross-validation of simple kriging with local means. Previously
    simulated nodes are not included; only ``data_xy`` / ``data_resid``.

    Parameters
    ----------
    points :
        Estimation locations, shape ``(m, 2)``.
    prior_points :
        Prior ``P(Z <= z_k)`` at those locations, shape ``(m, K)``.
    data_xy, data_resid :
        Hard-data coordinates ``(nd, 2)`` and residuals ``(nd, K)``.
    models, variances :
        Residual variogram models and residual variances (see
        :func:`residual_covariances`).
    k_nn, radius :
        Neighbour search limits.

    Returns
    -------
    ndarray, shape (m, K)
        Order-relation-corrected local CDFs.
    """
    points = np.asarray(points, dtype=float)
    prior_points = np.asarray(prior_points, dtype=float)
    data_xy = np.asarray(data_xy, dtype=float)
    data_resid = np.asarray(data_resid, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (m, 2)")
    if prior_points.shape != (points.shape[0], data_resid.shape[1]):
        raise ValueError("prior_points must have shape (m, K)")
    if data_xy.shape[0] != data_resid.shape[0] or data_xy.shape[1] != 2:
        raise ValueError("data_xy / data_resid shape mismatch")
    if k_nn < 1:
        raise ValueError("k_nn must be >= 1")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    cov, sill0 = residual_covariances(models, variances)
    if data_resid.shape[1] != len(sill0):
        raise ValueError("data_resid second axis must match len(models)")

    tree = cKDTree(data_xy)
    F = prior_points.copy()
    k_query = min(int(k_nn), len(data_xy))
    for i, p in enumerate(points):
        d, j = tree.query(p, k=k_query, distance_upper_bound=float(radius))
        j = np.atleast_1d(j)
        d = np.atleast_1d(d)
        j = j[np.isfinite(d)]
        if len(j):
            F[i] += sk_correction(p, data_xy[j], data_resid[j], cov, sill0)
    return order_relations(F)


def sis_local_means(
    nodes: np.ndarray,
    prior_nodes: np.ndarray,
    data_xy: np.ndarray,
    data_ind: np.ndarray,
    data_prior: np.ndarray,
    thresholds: ArrayLike,
    models: Sequence[ModelLike],
    variances: ArrayLike,
    *,
    zmin: float,
    zmax: float,
    p_low: float = 1.0,
    p_up: float = 1.0,
    k_nn: int = 16,
    radius: float = 6000.0,
    n_candidates: int = 160,
    nsim: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sequential indicator simulation of a continuous variable with local priors.

    Parameters
    ----------
    nodes :
        Simulation nodes, shape ``(n, 2)``.
    prior_nodes :
        Prior ``P(Z <= z_k)`` at the nodes, shape ``(n, K)``.
    data_xy :
        Hard-data coordinates, shape ``(nd, 2)``.
    data_ind :
        Hard indicators ``I(Z <= z_k)``, shape ``(nd, K)``.
    data_prior :
        Prior probabilities at the data locations, shape ``(nd, K)``.
    thresholds :
        Ascending continuous thresholds, length ``K``.
    models, variances :
        Residual variogram models (standardized) and residual variances.
    zmin, zmax :
        Lower and upper bounds used when drawing from the discrete CDF.
    p_low, p_up :
        Power exponents for lower- and upper-tail interpolation
        (``t ** p`` inside the first / last class).
    k_nn, radius, n_candidates :
        Neighbour search: keep up to ``k_nn`` informed candidates within
        ``radius``, chosen from the ``n_candidates`` nearest locations
        precomputed for each node.
    nsim, seed :
        Number of realizations and RNG seed.

    Returns
    -------
    ndarray, shape (nsim, n)
        Simulated continuous values.
    """
    nodes = np.asarray(nodes, dtype=float)
    prior_nodes = np.asarray(prior_nodes, dtype=float)
    data_xy = np.asarray(data_xy, dtype=float)
    data_ind = np.asarray(data_ind, dtype=float)
    data_prior = np.asarray(data_prior, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float).ravel()

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")
    if data_xy.ndim != 2 or data_xy.shape[1] != 2:
        raise ValueError("data_xy must have shape (nd, 2)")
    n, nd, nk = len(nodes), len(data_xy), len(thresholds)
    if nk < 1:
        raise ValueError("thresholds must be non-empty")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly ascending")
    if prior_nodes.shape != (n, nk):
        raise ValueError("prior_nodes must have shape (n, K)")
    if data_ind.shape != (nd, nk) or data_prior.shape != (nd, nk):
        raise ValueError("data_ind and data_prior must have shape (nd, K)")
    if zmax <= zmin:
        raise ValueError("zmax must be > zmin")
    if nsim < 1:
        raise ValueError("nsim must be >= 1")
    if k_nn < 1 or n_candidates < 1:
        raise ValueError("k_nn and n_candidates must be >= 1")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    cov, sill0 = residual_covariances(models, variances)
    if len(sill0) != nk:
        raise ValueError("len(models) must equal len(thresholds)")

    rng = np.random.default_rng(seed)
    all_xy = np.vstack([data_xy, nodes])
    k_cand = min(int(n_candidates), len(all_xy))
    cand_d, cand_i = cKDTree(all_xy).query(nodes, k=k_cand)
    cand_d = np.atleast_2d(cand_d)
    cand_i = np.atleast_2d(cand_i)
    knots = np.r_[float(zmin), thresholds, float(zmax)]
    out = np.empty((int(nsim), n), dtype=float)

    for s in range(int(nsim)):
        resid = np.zeros((nd + n, nk), dtype=float)
        resid[:nd] = data_ind - data_prior
        informed = np.zeros(nd + n, dtype=bool)
        informed[:nd] = True
        u_draw = rng.random(n)
        for node in rng.permutation(n):
            c = cand_i[node]
            nb = c[informed[c] & (cand_d[node] <= radius)][: int(k_nn)]
            F = prior_nodes[node].copy()
            if len(nb):
                F += sk_correction(nodes[node], all_xy[nb], resid[nb], cov, sill0)
            cdf = np.r_[0.0, order_relations(F), 1.0]
            u = float(u_draw[node])
            j = int(np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, nk))
            width = cdf[j + 1] - cdf[j]
            t = 0.5 if width < 1e-12 else (u - cdf[j]) / width
            if j == 0:
                t = t ** float(p_low)
            elif j == nk:
                t = t ** float(p_up)
            z = knots[j] + (knots[j + 1] - knots[j]) * t
            out[s, node] = z
            resid[nd + node] = (z <= thresholds).astype(float) - prior_nodes[node]
            informed[nd + node] = True
    return out
