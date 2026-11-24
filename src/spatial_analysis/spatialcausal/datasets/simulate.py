"""Simulators with known causal effects, for teaching and benchmarking."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from geolattice.spatial_statistics.weights import KNN

from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData


def _smooth_field(coords: np.ndarray, length_scale: float, rng, n_bumps: int = 60) -> np.ndarray:
    """Cheap smooth random field: a random sum of Gaussian bumps, standardised."""
    centers = rng.uniform(coords.min(0), coords.max(0), size=(n_bumps, 2))
    amps = rng.normal(size=n_bumps)
    d2 = ((coords[:, None, :] - centers[None]) ** 2).sum(-1)
    f = (np.exp(-0.5 * d2 / length_scale ** 2) * amps).sum(1)
    return (f - f.mean()) / f.std()


def _coords(n, rng):
    return rng.uniform(0, 1, size=(n, 2))


def simulate_spatial_confounding(n: int = 800, ate: float = 1.0, confounder_range: float = 0.15,
                                 confounding_strength: float = 1.0, treatment: str = "continuous",
                                 noise_range: float = 0.0, random_state: int = 0) -> SpatialCausalData:
    """Unmeasured, spatially smooth confounder ``U`` affects both T and Y.

    Coordinates are in the unit square; ``confounder_range`` is the length
    scale of ``U``. ``noise_range > 0`` adds spatially correlated outcome
    noise (so naive SEs are too small). ``U`` is returned in
    ``data.truth['confounder']`` and is *not* part of ``X``.
    """
    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    U = _smooth_field(s, confounder_range, rng)
    X = rng.normal(size=(n, 3))
    X[:, 2] = 0.5 * X[:, 2] + 0.5 * _smooth_field(s, 0.3, rng)  # an observed spatial covariate
    lin_t = 0.6 * X[:, 0] - 0.4 * X[:, 2] + confounding_strength * U
    if treatment == "binary":
        T = rng.binomial(1, expit(lin_t)).astype(float)
    elif treatment == "continuous":
        T = lin_t + rng.normal(scale=0.7, size=n)
    else:
        raise ValueError("treatment must be 'continuous' or 'binary'")
    eps = rng.normal(scale=0.8, size=n)
    if noise_range > 0:
        eps = eps + 0.8 * _smooth_field(s, noise_range, rng)
    Y = ate * T + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.8 * X[:, 2] + 1.5 * confounding_strength * U + eps
    return SpatialCausalData(X, T, Y, s, w=KNN(s, k=8, transform="row"),
                             feature_names=["x0", "x1", "x2"],
                             truth={"ate": ate, "confounder": U})


def simulate_interference(n: int = 1000, direct: float = 1.0, spillover: float = 0.5,
                          spillover_treated: float = 0.25, k: int = 6, threshold: float = 0.5,
                          p_treat: float = 0.5, confounded: bool = False,
                          random_state: int = 0) -> SpatialCausalData:
    """Outcomes depend on own treatment and the share of treated neighbours.

    Neighbours are the ``k`` nearest units (PySpatialStats ``KNN``). A unit
    is "highly exposed" when at least ``threshold`` of its neighbours are
    treated. With ``confounded=True`` treatment probability depends on a
    spatially smooth covariate, so clustering of treatment and outcome
    must be adjusted for.

    Truth: ``direct``, ``spillover`` (controls), ``spillover_treated``,
    ``total = direct + spillover_treated``.
    """
    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    X = rng.normal(size=(n, 3))
    X[:, 0] = 0.5 * X[:, 0] + _smooth_field(s, 0.2, rng)
    if confounded:
        T = rng.binomial(1, expit(np.log(p_treat / (1 - p_treat)) + 0.8 * X[:, 0])).astype(int)
    else:
        T = rng.binomial(1, p_treat, size=n)
    w = KNN(s, k=k, transform="row")
    share = np.asarray(w.sparse @ T).ravel()
    H = (share >= threshold).astype(int)
    Y = (0.8 * X[:, 0] + np.cos(X[:, 1]) + 0.3 * X[:, 2]
         + direct * T + spillover * H * (1 - T) + spillover_treated * H * T
         + rng.normal(scale=0.7, size=n))
    return SpatialCausalData(X, T, Y, s, w=w, feature_names=["x0", "x1", "x2"],
                             truth={"direct": direct, "spillover": spillover,
                                    "spillover_treated": spillover_treated,
                                    "total": direct + spillover_treated,
                                    "k": k, "threshold": threshold, "p_treat": p_treat})


def simulate_spatial_heterogeneity(n: int = 1500, base_effect: float = 1.0, amplitude: float = 1.0,
                                   random_state: int = 0) -> SpatialCausalData:
    """Binary treatment whose effect varies smoothly over space.

    ``tau(s) = base_effect + amplitude * sin(2*pi*s_x) * cos(pi*s_y)``.
    ``data.truth['cate']`` holds the true unit-level effects.
    """
    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    X = rng.normal(size=(n, 3))
    tau = base_effect + amplitude * np.sin(2 * np.pi * s[:, 0]) * np.cos(np.pi * s[:, 1])
    e = expit(0.5 * X[:, 0] - 0.3 * X[:, 1])
    T = rng.binomial(1, e)
    Y = X[:, 0] + 0.5 * np.abs(X[:, 1]) + tau * T + rng.normal(scale=0.6, size=n)
    return SpatialCausalData(X, T, Y, s, w=KNN(s, k=8, transform="row"),
                             feature_names=["x0", "x1", "x2"],
                             truth={"ate": float(tau.mean()), "cate": tau})


# ----------------------------------------------------------------------------
# v0.2: geographic quasi-experiments and spatially varying effects
# ----------------------------------------------------------------------------
def simulate_geographic_rdd(n: int = 2000, effect: float = 1.0, heterogeneity: float = 0.0,
                            random_state: int = 0) -> SpatialCausalData:
    """Units north of a wavy border are treated.

    The outcome has a smooth spatial trend that is continuous across the
    border, so a naive treated-vs-control comparison is confounded but the
    discontinuity is not. With ``heterogeneity > 0`` the jump varies along
    the border: ``effect + heterogeneity * cos(2*pi*x)``.
    ``data.truth`` holds ``effect`` (mean jump along the border), the
    shapely ``boundary`` and ``effect_at(x)``.
    """
    from shapely.geometry import LineString

    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    xs = np.linspace(0, 1, 201)
    border_y = 0.5 + 0.08 * np.sin(2 * np.pi * xs)
    boundary = LineString(np.column_stack([xs, border_y]))
    T = (s[:, 1] > 0.5 + 0.08 * np.sin(2 * np.pi * s[:, 0])).astype(int)
    X = rng.normal(size=(n, 2))
    X[:, 0] += 1.5 * _smooth_field(s, 0.3, rng)
    trend = 3.0 * s[:, 1] + 1.0 * s[:, 0] ** 2 + 0.8 * _smooth_field(s, 0.25, rng)

    def effect_at(x):
        return effect + heterogeneity * np.cos(2 * np.pi * np.asarray(x))

    Y = trend + 0.5 * X[:, 0] + 0.3 * X[:, 1] + effect_at(s[:, 0]) * T + rng.normal(scale=0.5, size=n)
    return SpatialCausalData(X, T, Y, s, feature_names=["x0", "x1"],
                             truth={"effect": effect, "boundary": boundary, "effect_at": effect_at})


def simulate_spatial_did(n_units: int = 600, n_periods: int = 8, treat_time: int = 5,
                         att: float = 2.0, rings=(0.0, 0.05, 0.10, 0.15),
                         ring_spillovers=(1.0, 0.5, 0.2), n_clusters: int = 12,
                         treated_share: float = 0.08, random_state: int = 0):
    """Panel where treatment spills over to untreated neighbours.

    Treated units are clustered around ``n_clusters`` centres. Untreated
    units at distance ``d`` from the nearest treated unit receive the
    spillover of the ring ``d`` falls in (step function); beyond the last
    ring there is none. Parallel trends hold. Returns a :class:`SpatialPanel`.
    """
    from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(random_state)
    s = _coords(n_units, rng)
    centres = rng.uniform(0.1, 0.9, size=(n_clusters, 2))
    dc, _ = cKDTree(centres).query(s)
    n_tr = max(1, int(treated_share * n_units))
    treated = np.zeros(n_units, int)
    treated[np.argsort(dc)[:n_tr]] = 1
    d, _ = cKDTree(s[treated == 1]).query(s)
    spill = np.zeros(n_units)
    for r in range(len(rings) - 1):
        m = (treated == 0) & (d > rings[r]) & (d <= rings[r + 1])
        spill[m] = ring_spillovers[r]
    alpha = 2.0 * _smooth_field(s, 0.2, rng) + rng.normal(scale=0.5, size=n_units)
    lam = np.cumsum(rng.normal(0.3, 0.2, size=n_periods))
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_units)
    post = (time >= treat_time).astype(int)
    y = (alpha[unit] + lam[time - 1] + post * (att * treated[unit] + spill[unit])
         + rng.normal(scale=0.5, size=len(unit)))
    return SpatialPanel(unit, time, y, np.arange(n_units), s, treated=treated, treat_time=treat_time,
                        truth={"att": att, "rings": tuple(rings), "spillovers": tuple(ring_spillovers)})


def simulate_synthetic_control(n_units: int = 60, n_periods: int = 30, treat_time: int = 21,
                               effect: float = -4.0, spillover: float = -2.5, spill_radius: float = 0.25,
                               random_state: int = 0):
    """One treated unit at the centre; nearby units get a decaying spillover.

    Outcomes follow a factor model whose loadings vary smoothly in space,
    so nearby units are the most similar donors — exactly the ones a
    standard synthetic control would weight, and the ones contaminated by
    spillover (``spillover * (1 - d / spill_radius)`` within ``spill_radius``).
    """
    from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel

    rng = np.random.default_rng(random_state)
    s = rng.uniform(0, 1, size=(n_units, 2))
    s[0] = (0.5, 0.5)
    F = np.cumsum(rng.normal(size=(n_periods, 3)), axis=0) * 0.6
    L = np.column_stack([_smooth_field(s, 0.3, rng) for _ in range(3)])
    base = 10 + L @ F.T + rng.normal(scale=0.3, size=(n_units, n_periods))
    d = np.sqrt(((s - s[0]) ** 2).sum(1))
    post = np.arange(1, n_periods + 1) >= treat_time
    spill = np.where((d > 0) & (d < spill_radius), spillover * (1 - d / spill_radius), 0.0)
    eff = np.where(np.arange(n_units) == 0, effect, spill)
    Y = base + eff[:, None] * post[None, :]
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_units)
    treated = (np.arange(n_units) == 0).astype(int)
    return SpatialPanel(unit, time, Y.ravel(), np.arange(n_units), s, treated=treated, treat_time=treat_time,
                        truth={"att": effect, "spillover_by_unit": spill, "spill_radius": spill_radius})


# ----------------------------------------------------------------------------
# v0.3 simulators
# ----------------------------------------------------------------------------
def simulate_areal_causal(nrow: int = 15, ncol: int = 15, family: str = "gaussian", effect: float = 0.5,
                          spillover: float = 0.0, confounding: float = 0.0, field_sd: float = 0.6,
                          treat_share: float = 0.4, random_state: int = 0) -> SpatialCausalData:
    """Areal (lattice) data with an ICAR-like spatial effect and Queen neighbours.

    ``family="poisson"`` returns counts in ``y`` with expected counts in
    ``truth['expected']``; effects are then log rate ratios. ``confounding``
    makes treatment more likely where the (unmeasured) spatial field is high.
    The GeoDataFrame of cells is in ``data.gdf``.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from geolattice.spatial_statistics.weights import Queen

    rng = np.random.default_rng(random_state)
    cells = [box(i, j, i + 1, j + 1) for j in range(nrow) for i in range(ncol)]
    gdf = gpd.GeoDataFrame(geometry=cells)
    n = len(gdf)
    w = Queen(gdf)
    cent = np.column_stack([gdf.centroid.x, gdf.centroid.y]) / max(nrow, ncol)
    field = field_sd * _smooth_field(cent, 0.25, rng)
    X = rng.normal(size=(n, 2))
    lin = np.log(treat_share / (1 - treat_share)) + 0.5 * X[:, 0] + confounding * field / max(field_sd, 1e-9)
    T = rng.binomial(1, expit(lin)).astype(float)
    Wr = _as_sparse_row(w, n)
    share = np.asarray(Wr @ T).ravel()
    eta = 0.3 * X[:, 0] - 0.2 * X[:, 1] + effect * T + spillover * share + field
    truth = {"direct": effect, "spillover": spillover, "total": effect + spillover, "field": field}
    if family == "gaussian":
        y = 1.0 + eta + rng.normal(scale=0.4, size=n)
    elif family == "poisson":
        E = rng.uniform(20, 80, size=n)
        y = rng.poisson(E * np.exp(eta - 0.2)).astype(float)
        truth["expected"] = E
    else:
        raise ValueError("family must be 'gaussian' or 'poisson'")
    gdf["T"], gdf["y"] = T, y
    return SpatialCausalData(X, T, y, cent, w=w, feature_names=["x0", "x1"], gdf=gdf, truth=truth)


def _as_sparse_row(w, n):
    from geolattice.spatial_analysis.spatialcausal._utils import as_sparse_weights
    return as_sparse_weights(w, n, row_standardize=True)


def simulate_staggered_spillover(n_units: int = 800, n_periods: int = 10, cohorts=(4, 6, 8),
                                 cohort_share: float = 0.06, att: float = 1.0, growth: float = 0.3,
                                 spillover: float = 0.8, exposure_distance: float = 0.05,
                                 random_state: int = 0):
    """Staggered adoption with dynamic direct effects and spatial spillovers.

    Each cohort treats a spatial cluster of units at its adoption period.
    The direct effect ``k`` periods after adoption is ``att + growth * k``.
    Never-treated units within ``exposure_distance`` of any treated unit
    receive ``spillover`` from the period they are first exposed.
    Parallel trends hold. Returns a :class:`SpatialPanel` whose ``truth``
    has the effects and the true event-time function.
    """
    from scipy.spatial import cKDTree
    from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel

    rng = np.random.default_rng(random_state)
    s = _coords(n_units, rng)
    adopt = np.full(n_units, np.inf)
    free = np.ones(n_units, bool)
    for g in cohorts:
        k = int(cohort_share * n_units)
        for _ in range(3):  # three clusters per cohort
            c = rng.uniform(0.1, 0.9, size=2)
            d = np.sqrt(((s - c) ** 2).sum(1))
            d[~free] = np.inf
            pick = np.argsort(d)[: k // 3]
            adopt[pick] = g
            free[pick] = False
    times = np.arange(1, n_periods + 1)
    expo = np.full(n_units, np.inf)
    for t in times:
        src = np.flatnonzero(adopt <= t)
        if len(src):
            d, _ = cKDTree(s[src]).query(s, k=min(2, len(src)))
            d = d if d.ndim == 1 else np.where(d[:, 0] == 0, d[:, -1], d[:, 0])
            newly = np.isinf(expo) & (d <= exposure_distance)
            expo[newly] = t
    alpha = 1.5 * _smooth_field(s, 0.2, rng) + rng.normal(scale=0.5, size=n_units)
    lam = np.cumsum(rng.normal(0.2, 0.2, size=n_periods))
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(times, n_units)
    k_rel = time - adopt[unit]
    direct = np.where(k_rel >= 0, att + growth * k_rel, 0.0)
    spill = np.where(np.isinf(adopt[unit]) & (time >= expo[unit]), spillover, 0.0)
    y = alpha[unit] + lam[time - 1] + direct + spill + rng.normal(scale=0.5, size=len(unit))
    treated = np.isfinite(adopt).astype(int)
    post = (time >= adopt[unit]).astype(int)
    return SpatialPanel(unit, time, y, np.arange(n_units), s, treated=treated, post=post,
                        truth={"att": att, "growth": growth, "spillover": spillover,
                               "exposure_distance": exposure_distance,
                               "event_time": lambda k: att + growth * np.asarray(k)})


def simulate_spatial_granger(n_units: int = 300, n_periods: int = 30, local: float = 0.5,
                             spill: float = 0.4, rho: float = 0.4, k: int = 6, random_state: int = 0):
    """Panel where past ``x`` (own and neighbours') drives ``y``; ``y`` does not drive ``x``.

    ``y_it = rho y_i,t-1 + local x_i,t-1 + spill (Wx)_i,t-1 + a_i + e_it``.
    ``x`` is a spatially autocorrelated AR(1). Returns a :class:`SpatialPanel`
    with ``x`` in ``panel.X[:, 0]``.
    """
    from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel

    rng = np.random.default_rng(random_state)
    s = _coords(n_units, rng)
    W = KNN(s, k=k, transform="row").sparse
    T = n_periods + 20
    x = np.zeros((n_units, T))
    y = np.zeros((n_units, T))
    a = rng.normal(scale=0.5, size=n_units)
    for t in range(1, T):
        shock = rng.normal(size=n_units)
        x[:, t] = 0.5 * x[:, t - 1] + 0.5 * np.asarray(W @ shock) + 0.5 * shock
        y[:, t] = (rho * y[:, t - 1] + local * x[:, t - 1] + spill * np.asarray(W @ x[:, t - 1])
                   + a + rng.normal(scale=0.7, size=n_units))
    x, y = x[:, 20:], y[:, 20:]
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)
    return SpatialPanel(unit, time, y.ravel(), np.arange(n_units), s, X=x.ravel()[:, None],
                        truth={"local": local, "spill": spill})


def simulate_spatial_dag(n: int = 1500, k: int = 6, random_state: int = 0):
    """Cross-sectional data from a linear SEM with local and spillover edges.

    Local edges: x0 -> x1, x0 -> x2, x1 -> x3, x2 -> x3, x2 -> x4.
    Spillover edge: W.x1 -> x4 (neighbours' x1 affects a unit's x4).
    Returns ``(X, coords, w, truth)`` with ``truth['local']`` and
    ``truth['spillover']`` as sets of (source, target) names.
    """
    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    w = KNN(s, k=k, transform="row")
    W = w.sparse
    e = rng.normal(size=(n, 5))
    x0 = e[:, 0] * 1.0
    x1 = 1.2 * x0 + 0.6 * e[:, 1]
    x2 = -0.9 * x0 + 0.6 * e[:, 2]
    x3 = 0.8 * x1 + 0.7 * x2 + 0.5 * e[:, 3]
    x4 = 0.9 * x2 + 1.5 * np.asarray(W @ x1) + 0.5 * e[:, 4]
    X = np.column_stack([x0, x1, x2, x3, x4])
    truth = {"local": {("x0", "x1"), ("x0", "x2"), ("x1", "x3"), ("x2", "x3"), ("x2", "x4")},
             "spillover": {("W.x1", "x4")}}
    return X, s, w, truth


def simulate_gccm(n: int = 900, beta_yx: float = 0.3, beta_xy: float = 0.0, diffusion: float = 0.3,
                  n_steps: int = 300, k: int = 8, random_state: int = 0):
    """Coupled spatial logistic maps (the test system of Gao et al., 2023).

    ``x_{t+1} = r_x x (1 - x - beta_xy y)`` and
    ``y_{t+1} = r_y y (1 - y - beta_yx x)``, each followed by diffusion to
    the ``k`` nearest neighbours: ``v <- (1 - diffusion) v + diffusion W v``.
    With the defaults ``x`` drives ``y`` and ``y`` does not drive ``x``.
    The final cross-section is returned as ``(x, y, coords, w)``.
    """
    rng = np.random.default_rng(random_state)
    s = _coords(n, rng)
    w = KNN(s, k=k, transform="row")
    W = w.sparse
    x = rng.uniform(0.2, 0.4, size=n)
    y = rng.uniform(0.2, 0.4, size=n)
    rx, ry = 3.8, 3.5
    for _ in range(n_steps):
        xn = rx * x * (1 - x - beta_xy * y)
        yn = ry * y * (1 - y - beta_yx * x)
        x = np.clip((1 - diffusion) * xn + diffusion * np.asarray(W @ xn), 1e-6, 1 - 1e-6)
        y = np.clip((1 - diffusion) * yn + diffusion * np.asarray(W @ yn), 1e-6, 1 - 1e-6)
    return x, y, s, w
