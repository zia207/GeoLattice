"""Designing spatial experiments when treatment can spill over.

* :class:`ClusterRandomizedDesign` — randomise spatial clusters, with
  optional buffer zones: units within ``buffer`` of a differently assigned
  cluster are excluded from analysis, so (if spillovers do not reach
  further than ``buffer``) each analysed unit's exposure equals its own
  cluster's assignment (cf. graph cluster randomisation, Ugander et al., 2013).
* :class:`SaturationDesign` — two-stage randomised saturation (Baird et
  al., 2018): clusters are randomised to saturation levels, then units within
  each cluster are treated with that probability. Untreated units in
  partially treated clusters identify spillovers.
* :func:`balanced_allocation` — spatially balanced treated set with GRTS
  (``geolattice.spatial_statistics.sampling.grts``).
* :func:`rerandomize` — accept only assignments with good covariate balance
  and little spatial clustering of treatment (Morgan & Rubin, 2012).
* :func:`design_diagnostics` — balance, Moran's I of the assignment and
  exposure-condition counts.
* :func:`power_analysis` — simulation-based power for direct and spillover
  effects under a design, with spatially correlated outcomes.

Each design has ``sample(n, rng)``, so it plugs into
:func:`~geolattice.spatial_analysis.spatialcausal.exposure.exposure_probabilities` and the design-based
Hájek estimator in :class:`~geolattice.spatial_analysis.spatialcausal.interference.InterferenceEstimator`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from geolattice.spatial_analysis.spatialcausal._utils import as_2d, as_coords, as_sparse_weights, default_weights
from geolattice.spatial_analysis.spatialcausal.inference import block_variance


# ----------------------------------------------------------------------------
# Designs
# ----------------------------------------------------------------------------
class ClusterRandomizedDesign:
    """Randomise spatial clusters to treatment.

    Parameters
    ----------
    coords : (n, 2) unit locations.
    n_clusters : int
        Clusters formed by k-means on coordinates (or pass ``clusters``).
    p : float
        Probability a cluster is treated (``complete=True``: exactly
        ``round(p * n_clusters)`` clusters).
    buffer : float
        Exclusion distance for analysis (see :meth:`analysis_mask`).
    clusters : array, optional
        Pre-defined cluster labels (villages, districts …).
    """

    def __init__(self, coords, n_clusters: int = 20, p: float = 0.5, buffer: float = 0.0,
                 clusters=None, complete: bool = True, random_state: int = 0):
        self.coords = as_coords(coords)
        if clusters is None:
            clusters = KMeans(n_clusters=n_clusters, n_init=4, random_state=random_state).fit_predict(self.coords)
        _, self.clusters = np.unique(np.asarray(clusters), return_inverse=True)
        self.n_clusters = int(self.clusters.max() + 1)
        self.p, self.buffer, self.complete = float(p), float(buffer), complete

    def sample(self, n: Optional[int] = None, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        G = self.n_clusters
        if self.complete:
            on = np.zeros(G, bool)
            on[rng.choice(G, int(round(self.p * G)), replace=False)] = True
        else:
            on = rng.random(G) < self.p
        return on[self.clusters].astype(int)

    def analysis_mask(self, treatment) -> np.ndarray:
        """Units whose ``buffer`` neighbourhood contains only same-arm units."""
        t = np.asarray(treatment).astype(int)
        if self.buffer <= 0:
            return np.ones(len(t), bool)
        tree = cKDTree(self.coords)
        nb = tree.query_ball_point(self.coords, r=self.buffer)
        return np.array([np.all(t[idx] == t[i]) for i, idx in enumerate(nb)])

    def interior_share(self, n_draws: int = 20, rng=None) -> float:
        """Average share of units retained for analysis under the buffer."""
        rng = np.random.default_rng(rng)
        return float(np.mean([self.analysis_mask(self.sample(rng=rng)).mean() for _ in range(n_draws)]))


class SaturationDesign:
    """Two-stage randomised saturation design.

    Parameters
    ----------
    coords : (n, 2)
    n_clusters : int or cluster labels via ``clusters``
    saturations : sequence of floats in [0, 1]
        Within-cluster treatment probabilities; clusters are allocated to
        levels as evenly as possible (``weights`` to change that). Including
        ``0`` gives pure control clusters.
    """

    def __init__(self, coords, n_clusters: int = 30, saturations: Sequence[float] = (0.0, 0.33, 0.67),
                 weights: Optional[Sequence[float]] = None, clusters=None, random_state: int = 0):
        self.coords = as_coords(coords)
        if clusters is None:
            clusters = KMeans(n_clusters=n_clusters, n_init=4, random_state=random_state).fit_predict(self.coords)
        _, self.clusters = np.unique(np.asarray(clusters), return_inverse=True)
        self.n_clusters = int(self.clusters.max() + 1)
        self.saturations = np.asarray(saturations, float)
        w = np.ones(len(self.saturations)) if weights is None else np.asarray(weights, float)
        counts = np.floor(w / w.sum() * self.n_clusters).astype(int)
        counts[: self.n_clusters - counts.sum()] += 1
        self._levels = np.repeat(np.arange(len(self.saturations)), counts)
        self.last_saturation_ = None

    def sample(self, n: Optional[int] = None, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        lvl = rng.permutation(self._levels)
        sat = self.saturations[lvl][self.clusters]
        self.last_saturation_ = sat
        return (rng.random(len(sat)) < sat).astype(int)

    def unit_saturation(self, rng=None):
        """Draw an assignment and return ``(treatment, saturation of own cluster)``."""
        t = self.sample(rng=rng)
        return t, self.last_saturation_.copy()


# ----------------------------------------------------------------------------
# Allocation helpers
# ----------------------------------------------------------------------------
def balanced_allocation(coords, n_treated: int, size=None, seed: Optional[int] = None) -> np.ndarray:
    """Spatially balanced treatment allocation using GRTS (``geolattice.spatial_statistics.sampling.grts``).

    Every unit keeps its intended inclusion probability (``n_treated / n``,
    or ``∝ size``), so design-based inference stays valid, but the treated
    units are spread evenly over space, which lowers the variance of
    effect estimates when outcomes are spatially structured.
    """
    from geolattice.spatial_statistics.sampling import grts

    coords = as_coords(coords)
    s = grts(coords, n_treated, size=size, seed=seed)
    t = np.zeros(len(coords), int)
    t[np.asarray(s.indices)] = 1
    return t


def _imbalance(t, X):
    m1, m0 = X[t == 1].mean(0), X[t == 0].mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1
    return np.abs(m1 - m0) / sd


def rerandomize(coords, X=None, n_treated: Optional[int] = None, design=None, max_smd: float = 0.1,
                max_moran: Optional[float] = 0.05, w=None, n_draws: int = 1000, seed: int = 0):
    """Rerandomisation: draw assignments until balance criteria are met.

    Accepts an assignment when every covariate's standardised mean
    difference is below ``max_smd`` and Moran's I of the assignment is below
    ``max_moran`` (limits spatial clustering of treatment). Draws from
    ``design.sample`` if given, else complete randomisation of ``n_treated``.

    Returns ``(treatment, info)``. Analyse with randomisation inference that
    applies the same acceptance rule (the accepted assignments are the
    reference distribution); ``info['accepted']`` holds them.
    """
    coords = as_coords(coords)
    n = len(coords)
    rng = np.random.default_rng(seed)
    X = None if X is None else as_2d(X)
    Wm = None
    if max_moran is not None:
        Wm = as_sparse_weights(w if w is not None else default_weights(coords), n, row_standardize=True)
    accepted = []
    for _ in range(n_draws):
        if design is not None:
            t = design.sample(n, rng)
        else:
            t = np.zeros(n, int)
            t[rng.choice(n, n_treated, replace=False)] = 1
        smd = _imbalance(t, X).max() if X is not None else 0.0
        mi = 0.0
        if Wm is not None:
            z = t - t.mean()
            mi = float(z @ (Wm @ z) / (z @ z))
        if smd <= max_smd and (max_moran is None or mi <= max_moran):
            accepted.append(t)
    if not accepted:
        raise RuntimeError(f"no assignment met the criteria in {n_draws} draws; relax max_smd/max_moran")
    info = {"acceptance_rate": len(accepted) / n_draws, "accepted": np.array(accepted)}
    return accepted[0], info


def design_diagnostics(treatment, coords, X=None, w=None, exposure=None, names=None) -> dict:
    """Balance and spatial structure of an assignment.

    Returns a dict with ``balance`` (standardised mean differences),
    ``moran`` (Moran's I of treatment, from ``geolattice.spatial_statistics.esda.Moran``) and,
    if an :class:`ExposureMapping` is passed, ``exposure_counts``.
    """
    from geolattice.spatial_statistics.esda import Moran

    t = np.asarray(treatment).astype(int)
    coords = as_coords(coords)
    out = {"n": len(t), "treated_share": float(t.mean())}
    if X is not None:
        X = as_2d(X)
        names = names or [f"x{j}" for j in range(X.shape[1])]
        out["balance"] = pd.Series(_imbalance(t, X), index=names, name="smd")
    w = w if w is not None else default_weights(coords)
    m = Moran(t.astype(float), w, permutations=199, seed=0)
    out["moran"] = {"I": float(m.I), "p_value": float(m.p_value)}
    if exposure is not None:
        from geolattice.spatial_analysis.spatialcausal.exposure import CONDITIONS
        c = exposure.conditions(t)
        out["exposure_counts"] = {CONDITIONS[k]: int(np.sum(c == k)) for k in range(4)}
    return out


# ----------------------------------------------------------------------------
# Power analysis
# ----------------------------------------------------------------------------
def _smooth_noise(coords, length_scale, rng, n_bumps=80):
    centers = rng.uniform(coords.min(0), coords.max(0), size=(n_bumps, 2))
    d2 = ((coords[:, None] - centers[None]) ** 2).sum(-1)
    f = (np.exp(-0.5 * d2 / length_scale ** 2) * rng.normal(size=n_bumps)).sum(1)
    return (f - f.mean()) / (f.std() + 1e-12)


def power_analysis(design, coords, direct: float = 0.0, spillover: float = 0.0, exposure=None,
                   sigma: float = 1.0, spatial_sd: float = 0.5, length_scale: float = 0.15,
                   cluster_sd: float = 0.0, estimand: str = "direct", alpha: float = 0.05,
                   n_sims: int = 300, seed: int = 0) -> dict:
    """Simulation-based power for a spatial design.

    Outcomes are simulated as ``y = direct * T + spillover * H + spatial
    noise + cluster noise + iid noise``, where ``H`` is high neighbour
    exposure from ``exposure`` (an :class:`ExposureMapping`; required for
    spillovers). Each simulation draws a fresh assignment from ``design``.

    Estimands
    ---------
    * ``"direct"`` — treated vs control among low-exposure units (or all
      units if no exposure mapping), using the design's analysis mask.
    * ``"spillover"`` — high- vs low-exposure units among controls.
    * ``"total"`` — treated vs control in the analysis mask, ignoring
      exposure (what a cluster-randomised trial with buffers estimates:
      own plus neighbourhood effect).

    Inference uses cluster-robust SEs with the design's clusters (or the
    unit itself when the design has none) and a t(G - 1) critical value,
    since cluster-robust z-tests over-reject with few clusters. Returns power, the mean
    estimate, the mean SE and the rejection rate under the null
    (effect = 0) for a size check.
    """
    coords = as_coords(coords)
    n = len(coords)
    rng = np.random.default_rng(seed)
    clusters = getattr(design, "clusters", None)
    cl = np.arange(n) if clusters is None else clusters

    def one(effect_on: bool):
        t = design.sample(n, rng)
        H = exposure.high_exposure(t) if exposure is not None else np.zeros(n, int)
        noise = sigma * rng.standard_normal(n) + spatial_sd * _smooth_noise(coords, length_scale, rng)
        if cluster_sd > 0 and clusters is not None:
            noise = noise + cluster_sd * rng.standard_normal(cl.max() + 1)[cl]
        y = noise + (direct * t + spillover * H if effect_on else 0.0)
        mask = design.analysis_mask(t) if hasattr(design, "analysis_mask") else np.ones(n, bool)
        if estimand == "direct":
            a = mask & (H == 0) if exposure is not None else mask
            g1, g0 = a & (t == 1), a & (t == 0)
        elif estimand == "total":
            g1, g0 = mask & (t == 1), mask & (t == 0)
        elif estimand == "spillover":
            if exposure is None:
                raise ValueError("estimand='spillover' needs an exposure mapping")
            g1, g0 = mask & (t == 0) & (H == 1), mask & (t == 0) & (H == 0)
        else:
            raise ValueError("estimand must be 'direct', 'spillover' or 'total'")
        if g1.sum() < 2 or g0.sum() < 2:
            return np.nan, np.nan
        m1, m0 = y[g1].mean(), y[g0].mean()
        psi = np.zeros(n)
        psi[g1] = (y[g1] - m1) / g1.mean()
        psi[g0] = -(y[g0] - m0) / g0.mean()
        se = np.sqrt(block_variance(psi, cl))
        return m1 - m0, se

    G = int(cl.max() + 1)
    # small-number-of-clusters correction: t critical value with G - 1 df
    z = stats.t.ppf(1 - alpha / 2, G - 1) if clusters is not None else stats.norm.ppf(1 - alpha / 2)
    alt = np.array([one(True) for _ in range(n_sims)])
    null = np.array([one(False) for _ in range(min(n_sims, 200))])
    ok_a, ok_n = np.isfinite(alt).all(1), np.isfinite(null).all(1)
    rej = np.abs(alt[ok_a, 0] / alt[ok_a, 1]) > z
    rej0 = np.abs(null[ok_n, 0] / null[ok_n, 1]) > z
    size = float(rej0.mean())
    out = {"estimand": estimand, "power": float(rej.mean()), "size": size,
           "mean_estimate": float(alt[ok_a, 0].mean()), "mean_se": float(alt[ok_a, 1].mean()),
           "valid_sims": int(ok_a.sum())}
    # flag when the null rejection rate is clearly above alpha (binomial upper bound)
    n0 = int(ok_n.sum())
    out["size_warning"] = bool(size > alpha + 2.5 * np.sqrt(alpha * (1 - alpha) / max(n0, 1)))
    if out["size_warning"]:
        out["note"] = ("tests over-reject under the null: outcome correlation likely extends beyond the "
                       "randomisation clusters; use larger clusters or buffers, and treat the power as optimistic")
    return out


def power_curve(design_factory, coords, values: Sequence, vary: str = "n_clusters", **kwargs) -> pd.DataFrame:
    """Power over a design parameter, e.g. ``design_factory(n_clusters)``.

    ``design_factory(v)`` returns a design for each value ``v``; remaining
    keyword arguments go to :func:`power_analysis`.
    """
    rows = []
    for v in values:
        r = power_analysis(design_factory(v), coords, **kwargs)
        r[vary] = v
        rows.append(r)
    return pd.DataFrame(rows).set_index(vary)
