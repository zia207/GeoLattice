"""Geographic quasi-experiments.

:class:`GeographicRDD`
    Boundary regression discontinuity: units on one side of a border are
    treated. The running variable is the signed distance to the border;
    optional boundary-segment fixed effects compare only units near the
    same stretch of border (Dell, 2010; Keele & Titiunik, 2015).

:class:`SpatialDiD`
    Difference-in-differences with spillover rings (Butts, 2023). Untreated
    units close to treated ones may themselves be affected, which biases a
    naive DiD that uses them as controls. Ring indicators absorb those
    spillovers, so the remaining (far) units form a clean control group and
    the ring coefficients estimate the spillover at each distance.

:class:`SpatialSyntheticControl`
    Synthetic control (Abadie et al., 2010) with a spatial buffer that
    removes potentially contaminated nearby donors, in-space placebo
    inference, estimates of spillovers onto the excluded neighbours, and a
    buffer-sensitivity analysis.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.spatial import cKDTree

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_2d, as_binary, as_coords, check_lengths
from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData
from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.crossfit import SpatialBlockKFold
from geolattice.spatial_analysis.spatialcausal.inference import standard_error


# ============================================================================
# Shared: weighted least squares with influence functions
# ============================================================================
def _wls(A: np.ndarray, y: np.ndarray, w: np.ndarray):
    """WLS coefficients and per-observation influence contributions (scaled by n)."""
    Aw = A * w[:, None]
    M = np.linalg.pinv(A.T @ Aw)
    beta = M @ (Aw.T @ y)
    e = y - A @ beta
    psi = len(y) * (Aw * e[:, None]) @ M.T  # (n, p); mean over rows ≈ beta - beta0
    return beta, psi, e


def _triangular(u):
    return np.clip(1.0 - np.abs(u), 0.0, None)


def _epanechnikov(u):
    return np.clip(0.75 * (1.0 - u ** 2), 0.0, None)


_RDD_KERNELS = {"triangular": _triangular, "epanechnikov": _epanechnikov,
                "uniform": lambda u: (np.abs(u) <= 1).astype(float)}
_IK_CONST = {"triangular": 3.4375, "epanechnikov": 3.1999, "uniform": 2.7016}


def ik_bandwidth(r: np.ndarray, y: np.ndarray, kernel: str = "triangular") -> float:
    """Imbens & Kalyanaraman (2012) MSE-optimal bandwidth for a sharp RDD at 0."""
    r, y = np.asarray(r, float), np.asarray(y, float)
    n = len(r)
    right = r >= 0
    sx = np.std(r)
    h1 = 1.84 * sx * n ** (-0.2)
    inl, inr = (~right) & (np.abs(r) <= h1), right & (np.abs(r) <= h1)
    if inl.sum() < 5 or inr.sum() < 5:
        h1 = np.quantile(np.abs(r), 0.5)
        inl, inr = (~right) & (np.abs(r) <= h1), right & (np.abs(r) <= h1)
    f0 = (inl.sum() + inr.sum()) / (2 * n * h1)
    s2l = np.var(y[inl], ddof=1) if inl.sum() > 2 else np.var(y)
    s2r = np.var(y[inr], ddof=1) if inr.sum() > 2 else np.var(y)
    lo, hi = np.median(r[~right]), np.median(r[right])
    mid = (r >= lo) & (r <= hi)
    if mid.sum() < 10:
        mid = np.ones(n, bool)
    Z = np.column_stack([np.ones(mid.sum()), right[mid], r[mid], r[mid] ** 2, r[mid] ** 3])
    m3 = 6 * np.linalg.lstsq(Z, y[mid], rcond=None)[0][4]
    m3 = m3 if abs(m3) > 1e-8 else 1e-8
    nl, nr = (~right).sum(), right.sum()
    h2l = 3.56 * (s2l / (f0 * m3 ** 2)) ** (1 / 7) * nl ** (-1 / 7)
    h2r = 3.56 * (s2r / (f0 * m3 ** 2)) ** (1 / 7) * nr ** (-1 / 7)

    def curv(side, h2):
        m = side & (np.abs(r) <= h2)
        if m.sum() < 5:
            return 0.0, m.sum()
        Zs = np.column_stack([np.ones(m.sum()), r[m], r[m] ** 2])
        return 2 * np.linalg.lstsq(Zs, y[m], rcond=None)[0][2], m.sum()

    m2l, n2l = curv(~right, h2l)
    m2r, n2r = curv(right, h2r)
    regl = 2160 * s2l / (max(n2l, 1) * h2l ** 4)
    regr = 2160 * s2r / (max(n2r, 1) * h2r ** 4)
    h = _IK_CONST[kernel] * ((s2l + s2r) / (f0 * ((m2r - m2l) ** 2 + regl + regr))) ** 0.2 * n ** (-0.2)
    return float(h)


# ============================================================================
# Geographic regression discontinuity
# ============================================================================
def _as_line(boundary):
    from shapely.geometry.base import BaseGeometry
    if not isinstance(boundary, BaseGeometry):
        from shapely.geometry import LineString
        boundary = LineString(np.asarray(boundary, float))
    if boundary.geom_type in ("Polygon", "MultiPolygon"):
        boundary = boundary.boundary
    if boundary.geom_type not in ("LineString", "MultiLineString", "LinearRing"):
        raise ValueError("boundary must be a LineString, MultiLineString, Polygon or (k, 2) array")
    return boundary


class GeographicRDD:
    """Sharp regression discontinuity at a geographic boundary.

    Parameters
    ----------
    boundary : shapely LineString / MultiLineString / Polygon, or (k, 2) array
        The border. For a polygon, its outline is used.
    bandwidth : "ik" or float
        Distance bandwidth; ``"ik"`` is the Imbens–Kalyanaraman MSE-optimal
        choice on the signed distance.
    kernel : {"triangular", "epanechnikov", "uniform"}
    polynomial : {1, 2}
        Local polynomial order in distance (separate on each side).
    n_segments : int, optional
        Split the border into this many equal-length segments and add
        segment fixed effects, so treated and control units are compared
        only along the same stretch of border.
    donut : float
        Drop units closer than this to the border (sorting / mismeasurement).
    use_covariates : bool
        Add covariates ``X`` for precision (they should be balanced).
    se : {"iid", "conley", "block"}
        ``"iid"`` is the heteroskedasticity-robust sandwich.
    cutoff : float, Conley cutoff.

    Notes
    -----
    Identification needs every other determinant of the outcome to change
    smoothly at the border. Check with :meth:`balance` (placebo RDD on
    covariates) and :meth:`bandwidth_sensitivity`. Compound borders (where
    several policies change at once) cannot be separated by any RDD.
    """

    def __init__(self, boundary, bandwidth="ik", kernel: str = "triangular", polynomial: int = 1,
                 n_segments: Optional[int] = None, donut: float = 0.0, use_covariates: bool = False,
                 se: str = "iid", cutoff: Optional[float] = None):
        if kernel not in _RDD_KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_RDD_KERNELS)}")
        if polynomial not in (1, 2):
            raise ValueError("polynomial must be 1 or 2")
        self.boundary = _as_line(boundary)
        self.bandwidth, self.kernel, self.polynomial = bandwidth, kernel, polynomial
        self.n_segments, self.donut, self.use_covariates = n_segments, float(donut), use_covariates
        self.se, self.cutoff = se, cutoff

    # ------------------------------------------------------------------
    def running_variable(self, coords, treatment) -> np.ndarray:
        """Signed distance to the boundary: positive on the treated side."""
        import shapely
        pts = shapely.points(as_coords(coords))
        d = shapely.distance(pts, self.boundary)
        t = as_binary(treatment)
        return np.where(t == 1, d, -d)

    def boundary_position(self, coords) -> np.ndarray:
        """Position of each unit's nearest boundary point, as a fraction of border length."""
        import shapely
        pts = shapely.points(as_coords(coords))
        return shapely.line_locate_point(self.boundary, pts, normalized=True) if \
            self.boundary.geom_type != "MultiLineString" else \
            np.asarray([self.boundary.project(p, normalized=True) for p in pts])

    def _design(self, r, t, X, seg, keep):
        cols = [np.ones(keep.sum()), t[keep], r[keep], r[keep] * t[keep]]
        if self.polynomial == 2:
            cols += [r[keep] ** 2, r[keep] ** 2 * t[keep]]
        A = np.column_stack(cols)
        if seg is not None:
            s = seg[keep]
            levels = np.unique(s)
            if len(levels) > 1:
                A = np.column_stack([A, (s[:, None] == levels[None, 1:]).astype(float)])
        if X is not None:
            A = np.column_stack([A, X[keep] - X[keep].mean(0)])
        return A

    def _estimate(self, r, t, y, coords, X, seg, h):
        u = r / h
        w = _RDD_KERNELS[self.kernel](u)
        keep = (w > 0) & (np.abs(r) >= self.donut)
        if (t[keep] == 1).sum() < 10 or (t[keep] == 0).sum() < 10:
            raise ValueError(f"fewer than 10 units on one side within bandwidth {h:.4g}; widen it")
        A = self._design(r, t, X, seg, keep)
        beta, psi, _ = _wls(A, y[keep], w[keep])
        se = standard_error(psi[:, 1], coords[keep], self.se, self.cutoff)
        return float(beta[1]), se, int(keep.sum())

    # ------------------------------------------------------------------
    def fit(self, data_or_X, treatment=None, y=None, coords=None):
        if isinstance(data_or_X, SpatialCausalData):
            d = data_or_X
            X, t, y, coords = d.X, d.treatment, d.y, d.coords
        else:
            X = None if data_or_X is None else as_2d(data_or_X)
            t, y, coords = as_binary(treatment), as_1d(y), as_coords(coords)
        t = as_binary(t)
        n = check_lengths(t=t, y=y, coords=coords)
        r = self.running_variable(coords, t)
        seg = None
        if self.n_segments:
            pos = self.boundary_position(coords)
            seg = np.minimum((pos * self.n_segments).astype(int), self.n_segments - 1)
        h = ik_bandwidth(r, y, self.kernel) if self.bandwidth == "ik" else float(self.bandwidth)
        Xuse = X if (self.use_covariates and X is not None) else None
        tau, se, n_eff = self._estimate(r, t, y, coords, Xuse, seg, h)
        self.running_ = r
        self.segments_ = seg
        self.bandwidth_ = h
        self._data = (t, y, coords, X, seg)
        self.result_ = SpatialCausalResult(
            "Geographic RDD", {"effect_at_boundary": EffectEstimate("effect_at_boundary", tau, se)},
            se_method=self.se, n=n, bandwidth=round(h, 5), n_in_bandwidth=n_eff,
            kernel=self.kernel, polynomial=self.polynomial,
            segment_fe=self.n_segments or 0, donut=self.donut)
        return self

    # ------------------------------------------------------------------
    def bandwidth_sensitivity(self, multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0)) -> pd.DataFrame:
        """Re-estimate at multiples of the chosen bandwidth."""
        t, y, coords, X, seg = self._data
        Xuse = X if (self.use_covariates and X is not None) else None
        rows = []
        for m in multipliers:
            h = m * self.bandwidth_
            try:
                tau, se, k = self._estimate(self.running_, t, y, coords, Xuse, seg, h)
            except ValueError:
                tau, se, k = np.nan, np.nan, 0
            rows.append({"multiplier": m, "bandwidth": h, "estimate": tau, "se": se,
                         "ci_lower": tau - 1.96 * se, "ci_upper": tau + 1.96 * se, "n_in_bandwidth": k})
        return pd.DataFrame(rows)

    def balance(self, covariates=None, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Placebo RDD on each covariate (no jump expected if the design is valid)."""
        t, _, coords, X, seg = self._data
        C = X if covariates is None else as_2d(covariates)
        if C is None:
            raise ValueError("no covariates available")
        names = names or [f"x{j}" for j in range(C.shape[1])]
        rows = []
        for j, nm in enumerate(names):
            tau, se, _ = self._estimate(self.running_, t, C[:, j], coords, None, seg, self.bandwidth_)
            rows.append({"covariate": nm, "jump": tau, "se": se,
                         "p_value": 2 * stats.norm.sf(abs(tau / se))})
        return pd.DataFrame(rows).set_index("covariate")

    def along_boundary(self, n_points: int = 10, bandwidth: Optional[float] = None) -> pd.DataFrame:
        """Effects at points along the border (Keele & Titiunik, 2015).

        At each boundary point ``b``, a local linear regression in the
        two coordinates (``s - b``) is fit separately on each side with
        kernel weights in Euclidean distance to ``b``; the effect is the
        difference of the two intercepts. Default bandwidth: 2 × the RDD
        bandwidth (2-D neighbourhoods hold fewer units).
        """
        t, y, coords, _, _ = self._data
        h = bandwidth or 2.0 * self.bandwidth_
        fracs = (np.arange(n_points) + 0.5) / n_points
        kern = _RDD_KERNELS[self.kernel]
        rows = []
        for f in fracs:
            p = self.boundary.interpolate(f, normalized=True)
            b = np.array([p.x, p.y])
            dvec = coords - b
            w = kern(np.sqrt((dvec ** 2).sum(1)) / h)
            est, var, ok = [], 0.0, True
            for side in (1, 0):
                m = (w > 0) & (t == side)
                if m.sum() < 10:
                    ok = False
                    break
                A = np.column_stack([np.ones(m.sum()), dvec[m]])
                beta, psi, _ = _wls(A, y[m], w[m])
                est.append(beta[0])
                var += np.sum(psi[:, 0] ** 2) / m.sum() ** 2
            if ok:
                tau, se = est[0] - est[1], np.sqrt(var)
            else:
                tau, se = np.nan, np.nan
            rows.append({"position": f, "x": b[0], "y": b[1], "estimate": tau, "se": se,
                         "ci_lower": tau - 1.96 * se, "ci_upper": tau + 1.96 * se})
        return pd.DataFrame(rows)

    def effect(self) -> float:
        return self.result_["effect_at_boundary"].estimate

    def summary(self) -> str:
        return self.result_.summary()


# ============================================================================
# Panel helpers
# ============================================================================
def _twfe_demean(Z: np.ndarray, uid: np.ndarray, tid: np.ndarray, tol: float = 1e-10, max_iter: int = 500):
    """Two-way within transformation by alternating projections (handles unbalanced panels)."""
    Z = np.array(Z, float, copy=True)
    if Z.ndim == 1:
        Z = Z[:, None]
    nu, nt = uid.max() + 1, tid.max() + 1
    cu = np.bincount(uid, minlength=nu).astype(float)
    ct = np.bincount(tid, minlength=nt).astype(float)
    for _ in range(max_iter):
        old = Z.copy()
        for j in range(Z.shape[1]):
            Z[:, j] -= (np.bincount(uid, Z[:, j], nu) / cu)[uid]
            Z[:, j] -= (np.bincount(tid, Z[:, j], nt) / ct)[tid]
        if np.max(np.abs(Z - old)) < tol:
            break
    return Z


def _cluster_vcov(psi: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Cluster-robust covariance of the mean of influence rows ``psi`` (n, p)."""
    n = psi.shape[0]
    _, inv = np.unique(clusters, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, psi.shape[1]))
    np.add.at(S, inv, psi)
    adj = G / (G - 1) if G > 1 else 1.0
    return adj * (S.T @ S) / n ** 2


def _ols_twfe(y, D, uid, tid):
    yt = _twfe_demean(y, uid, tid)[:, 0]
    Dt = _twfe_demean(D, uid, tid)
    M = np.linalg.pinv(Dt.T @ Dt)
    beta = M @ Dt.T @ yt
    e = yt - Dt @ beta
    psi = len(yt) * (Dt * e[:, None]) @ M.T
    return beta, psi


# ============================================================================
# Spatial difference-in-differences with spillover rings
# ============================================================================
class SpatialDiD:
    """Two-way fixed effects DiD with distance rings for spillovers.

    Model (static)::

        y_it = a_i + l_t + tau * D_it + sum_r delta_r * Ring_r(i) * Post_t + e_it

    where ``D_it = treated_i * post_it`` and ``Ring_r(i) = 1`` for untreated
    units whose distance to the nearest treated unit (or ``sources``) is in
    ``(rings[r], rings[r + 1]]``. Untreated units beyond ``rings[-1]`` are
    the control group.

    Parameters
    ----------
    rings : sequence of increasing distances starting at 0
        E.g. ``(0, 1, 2, 3)`` gives rings (0, 1], (1, 2], (2, 3]. Choose the
        outer edge beyond the plausible spillover range; too small a range
        leaves contaminated units in the control group.
    sources : (m, 2) array, optional
        Locations of the intervention (e.g. plants, clinics) used for
        distances instead of the treated units' own locations.
    cluster : {"unit", "spatial"}
        Cluster SEs by unit, or by spatial blocks of units (``n_clusters``),
        which also allows for correlation between nearby units.
    n_clusters : "auto" or int
        Number of spatial blocks for ``cluster="spatial"``; ``"auto"`` is
        ``max(50, n_units // 8)``. With few blocks, cluster-robust tests
        over-reject (about 20% at a nominal 5% with 40 blocks and 15
        pre-trend restrictions in our simulations).
    event_study : bool
        Estimate group × period effects (requires one common adoption date),
        with the last pre-period as reference.

    Notes
    -----
    With staggered adoption, static TWFE can be badly biased even without
    spillovers (Goodman-Bacon, 2021); the estimator warns in that case.
    """

    def __init__(self, rings: Sequence[float] = (0.0, 1.0, 2.0), sources=None, cluster: str = "unit",
                 n_clusters="auto", event_study: bool = False, random_state: int = 0):
        rings = np.asarray(rings, float)
        if rings[0] != 0 or np.any(np.diff(rings) <= 0) or len(rings) < 2:
            raise ValueError("rings must start at 0 and be strictly increasing (at least two values)")
        if cluster not in ("unit", "spatial"):
            raise ValueError("cluster must be 'unit' or 'spatial'")
        self.rings, self.sources, self.cluster = rings, sources, cluster
        self.n_clusters, self.event_study, self.random_state = n_clusters, event_study, random_state

    def _distance(self, panel: SpatialPanel) -> np.ndarray:
        if self.sources is not None:
            src = as_coords(self.sources)
        else:
            src = panel.unit_coords[panel.treated == 1]
        if len(src) == 0:
            raise ValueError("no treated units / sources")
        d, _ = cKDTree(src).query(panel.unit_coords)
        return d

    def unit_groups(self, panel: SpatialPanel) -> np.ndarray:
        """Group per unit: 0 = control, 1 = treated, 2.. = ring 1, 2, ..."""
        d = self._distance(panel)
        g = np.zeros(panel.n_units, int)
        g[panel.treated == 1] = 1
        for r in range(len(self.rings) - 1):
            m = (panel.treated == 0) & (d > self.rings[r]) & (d <= self.rings[r + 1])
            g[m] = 2 + r
        if self.sources is not None:  # untreated unit located exactly at a source
            g[(panel.treated == 0) & (d == 0)] = 2
        return g

    def _clusters(self, panel):
        if self.cluster == "unit":
            return panel.unit_index
        G = max(50, panel.n_units // 8) if self.n_clusters == "auto" else int(self.n_clusters)
        blocks = SpatialBlockKFold(n_splits=2, n_blocks=min(G, panel.n_units),
                                   random_state=self.random_state).make_blocks(panel.unit_coords)
        return blocks[panel.unit_index]

    # ------------------------------------------------------------------
    def fit(self, panel: SpatialPanel):
        if panel.post is None:
            raise ValueError("panel needs post or treat_time")
        uid, tid = panel.unit_index, panel.time_index
        g_unit = self.unit_groups(panel)
        g = g_unit[uid]
        n_rings = len(self.rings) - 1
        names = ["att"] + [f"spillover_ring_{r + 1}" for r in range(n_rings)]
        self.groups_ = pd.Series(g_unit, index=panel.units, name="group")
        counts = {"treated": int((g_unit == 1).sum()), "control": int((g_unit == 0).sum())}
        counts.update({f"ring_{r + 1}": int((g_unit == 2 + r).sum()) for r in range(n_rings)})
        self.group_counts_ = counts
        if counts["control"] < 10:
            raise ValueError("fewer than 10 control units beyond the outer ring; reduce rings")
        for r in range(n_rings):
            if counts[f"ring_{r + 1}"] == 0:
                warnings.warn(f"ring {r + 1} is empty", RuntimeWarning)

        adopt = panel.adoption_times()
        a = adopt[panel.treated == 1]
        a = a[np.isfinite(a)]
        self.staggered_ = len(np.unique(a)) > 1
        if self.staggered_:
            warnings.warn("treatment adoption is staggered; static TWFE may be biased "
                          "(Goodman-Bacon 2021). Interpret with care.", RuntimeWarning)
        post = panel.post
        cl = self._clusters(panel)

        # static ring model
        cols = [(g == 1) * post] + [(g == 2 + r) * post for r in range(n_rings)]
        keep_cols = [j for j, c in enumerate(cols) if c.any()]
        Dm = np.column_stack([cols[j] for j in keep_cols]).astype(float)
        beta, psi = _ols_twfe(panel.y, Dm, uid, tid)
        V = _cluster_vcov(psi, cl)
        effects = {names[j]: EffectEstimate(names[j], beta[i], np.sqrt(V[i, i]))
                   for i, j in enumerate(keep_cols)}

        # naive DiD: all untreated units are controls
        b0, p0 = _ols_twfe(panel.y, ((g == 1) * post).astype(float)[:, None], uid, tid)
        se0 = np.sqrt(_cluster_vcov(p0, cl)[0, 0])
        self.naive_ = EffectEstimate("att_naive", b0[0], se0)
        effects["att_naive"] = self.naive_
        self.vcov_ = pd.DataFrame(V, index=[names[j] for j in keep_cols], columns=[names[j] for j in keep_cols])
        self.result_ = SpatialCausalResult(
            "Spatial DiD with spillover rings", effects, se_method=f"cluster ({self.cluster})",
            n=panel.n_obs, rings=tuple(float(x) for x in np.round(self.rings, 6)), group_counts=counts,
            naive_bias=round(float(b0[0] - effects["att"].estimate), 4))
        if self.event_study:
            self.event_study_ = self._event_study(panel, g, cl, adopt)
        return self

    def _event_study(self, panel, g, cl, adopt):
        if self.staggered_:
            raise ValueError("event_study needs a single adoption date")
        T0 = np.nanmin(adopt)
        times = panel.times
        pre = times[times < T0]
        if len(pre) == 0:
            raise ValueError("no pre-treatment periods")
        ref = pre.max()
        groups = [1] + [2 + r for r in range(len(self.rings) - 1)]
        cols, meta = [], []
        for grp in groups:
            if not np.any(g == grp):
                continue
            for p in times:
                if p == ref:
                    continue
                cols.append(((g == grp) & (panel.time == p)).astype(float))
                meta.append((grp, p))
        D = np.column_stack(cols)
        beta, psi = _ols_twfe(panel.y, D, panel.unit_index, panel.time_index)
        V = _cluster_vcov(psi, cl)
        se = np.sqrt(np.diag(V))
        label = {1: "treated"} | {2 + r: f"ring_{r + 1}" for r in range(len(self.rings) - 1)}
        df = pd.DataFrame({"group": [label[m[0]] for m in meta], "time": [m[1] for m in meta],
                           "rel_time": [m[1] - T0 for m in meta], "estimate": beta, "se": se})
        df["ci_lower"], df["ci_upper"] = df.estimate - 1.96 * df.se, df.estimate + 1.96 * df.se
        # joint pre-trend test across all groups. A chi-square Wald test over-rejects
        # when restrictions are many relative to clusters, so use the Hotelling-type
        # F statistic F = W (G - q) / ((G - 1) q) ~ F(q, G - q).
        pre_idx = np.flatnonzero(df["time"].to_numpy() < T0)
        if len(pre_idx):
            b, Vp = beta[pre_idx], V[np.ix_(pre_idx, pre_idx)]
            W = float(b @ np.linalg.pinv(Vp) @ b)
            q, G = len(pre_idx), len(np.unique(cl))
            if G > q:
                F = W * (G - q) / ((G - 1) * q)
                p = float(stats.f.sf(F, q, G - q))
            else:
                F, p = np.nan, np.nan
                warnings.warn(f"pre-trend test needs more clusters ({G}) than restrictions ({q})", RuntimeWarning)
            self.pretrend_test_ = {"wald": W, "F": F, "df": (q, G - q), "n_clusters": G, "p_value": p}
        return df

    def effect(self, name: str = "att") -> float:
        return self.result_[name].estimate

    def summary(self) -> str:
        return self.result_.summary()


# ============================================================================
# Spatial synthetic control
# ============================================================================
def _sc_weights(y1: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """Simplex-constrained least squares: min ||y1 - Y0 w||, w >= 0, sum w = 1.

    Solved as non-negative least squares with the sum-to-one constraint added
    as a heavily weighted extra row (``M * 1' w = M``), which is exact up to
    ~1/M and about 1,000x faster than SLSQP for a few hundred donors. Falls
    back to SLSQP if NNLS does not converge.
    """
    J = Y0.shape[1]
    if J == 1:
        return np.ones(1)
    scale = np.sqrt(np.mean(Y0 ** 2)) + 1e-12
    M = 1e3 * scale * np.sqrt(len(y1))
    try:
        w, _ = optimize.nnls(np.vstack([Y0, M * np.ones((1, J))]), np.r_[y1, M], maxiter=50 * J)
        if w.sum() > 0:
            return w / w.sum()
    except RuntimeError:
        pass
    G, c = Y0.T @ Y0, Y0.T @ y1
    gs = max(np.abs(G).max(), 1e-12)
    res = optimize.minimize(lambda v: (v @ G @ v - 2 * c @ v) / gs, np.full(J, 1.0 / J),
                            jac=lambda v: (2 * G @ v - 2 * c) / gs, method="SLSQP",
                            bounds=[(0.0, 1.0)] * J,
                            constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1.0,
                                          "jac": lambda v: np.ones(J)}],
                            options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


class SpatialSyntheticControl:
    """Synthetic control with a spatial donor buffer.

    Parameters
    ----------
    buffer : float
        Donors within this distance of the treated unit are excluded, because
        spillovers would contaminate the counterfactual. ``0`` = standard SC.
    demean : bool
        Fit on pre-period-demeaned outcomes and add back the treated unit's
        pre-period mean (allows a level difference; Ferman & Pinto, 2021).
    placebo : bool
        Run in-space placebos (each donor in turn as the "treated" unit,
        with the same buffer rule) for a permutation p-value.
    pre_fit_filter : float or None
        Drop placebos whose pre-period RMSPE exceeds this multiple of the
        treated unit's (Abadie et al., 2010).
    """

    def __init__(self, buffer: float = 0.0, demean: bool = False, placebo: bool = True,
                 pre_fit_filter: Optional[float] = 5.0):
        self.buffer, self.demean, self.placebo, self.pre_fit_filter = float(buffer), demean, placebo, pre_fit_filter

    # ------------------------------------------------------------------
    def _fit_one(self, Y, pre, target, donors):
        y1, Y0 = Y[target], Y[donors].T
        if self.demean:
            m1, m0 = y1[pre].mean(), Y0[pre].mean(0)
            w = _sc_weights(y1[pre] - m1, Y0[pre] - m0)
            synth = (Y0 - m0) @ w + m1
        else:
            w = _sc_weights(y1[pre], Y0[pre])
            synth = Y0 @ w
        gap = y1 - synth
        pre_rmspe = np.sqrt(np.mean(gap[pre] ** 2))
        post_rmspe = np.sqrt(np.mean(gap[~pre] ** 2))
        return w, synth, gap, pre_rmspe, post_rmspe

    def _donor_pool(self, target, exclude_units, buffer):
        d = np.sqrt(((self._coords - self._coords[target]) ** 2).sum(1))
        ok = d > buffer
        ok[target] = False
        ok[list(exclude_units)] = False
        return np.flatnonzero(ok), d

    def fit(self, panel: SpatialPanel, treated_unit=None, treat_time=None):
        if not panel.is_balanced:
            raise ValueError("synthetic control needs a balanced panel")
        Y = panel.wide()
        self._panel, self._coords, self._Y = panel, panel.unit_coords, Y
        if treated_unit is None:
            tr = np.flatnonzero(panel.treated == 1)
            if len(tr) != 1:
                raise ValueError("pass treated_unit (the panel has %d treated units)" % len(tr))
            target = int(tr[0])
        else:
            target = int(np.flatnonzero(panel.units == treated_unit)[0])
        if treat_time is None:
            if panel.treat_time is not None:
                treat_time = panel.treat_time
            else:
                treat_time = np.nanmin(panel.adoption_times()[[target]])
        pre = panel.times < treat_time
        if pre.sum() < 3:
            raise ValueError("need at least 3 pre-treatment periods")
        self._pre, self._target, self.treat_time_ = pre, target, treat_time
        self._other_treated = set(np.flatnonzero(panel.treated == 1)) - {target}

        donors, dist = self._donor_pool(target, self._other_treated, self.buffer)
        if len(donors) < 2:
            raise ValueError("fewer than 2 donors outside the buffer")
        w, synth, gap, pre_r, post_r = self._fit_one(Y, pre, target, donors)
        self.donors_ = panel.units[donors]
        self.excluded_near_ = panel.units[(dist <= self.buffer) & (np.arange(panel.n_units) != target)]
        self.weights_ = pd.Series(w, index=panel.units[donors], name="weight").sort_values(ascending=False)
        self.synthetic_ = pd.Series(synth, index=panel.times)
        self.observed_ = pd.Series(Y[target], index=panel.times)
        self.gap_ = pd.Series(gap, index=panel.times)
        self.pre_rmspe_, self.post_rmspe_ = pre_r, post_r
        att = float(gap[~pre].mean())
        self.att_ = att

        p_value, placebo_se = np.nan, np.nan
        if self.placebo:
            self.placebos_ = self._placebos()
            ratios = self.placebos_["ratio"].to_numpy()
            mine = post_r / max(pre_r, 1e-12)
            keep = np.ones(len(ratios), bool)
            if self.pre_fit_filter is not None:
                keep = self.placebos_["pre_rmspe"].to_numpy() <= self.pre_fit_filter * pre_r
            p_value = (1 + np.sum(ratios[keep] >= mine)) / (1 + keep.sum())
            placebo_se = float(np.std(self.placebos_["att"].to_numpy()[keep], ddof=1)) if keep.sum() > 2 else np.nan
            self.p_value_ = float(p_value)
        eff = {"att": EffectEstimate("att", att, placebo_se if np.isfinite(placebo_se) else np.nan)}
        self.result_ = SpatialCausalResult(
            "Spatial synthetic control", eff, se_method="SD of in-space placebo ATTs", n=panel.n_units,
            treated_unit=panel.units[target], buffer=self.buffer, n_donors=len(donors),
            n_excluded_near=len(self.excluded_near_), pre_rmspe=round(pre_r, 4),
            post_pre_ratio=round(post_r / max(pre_r, 1e-12), 3),
            permutation_p_value=None if not self.placebo else round(float(p_value), 4),
            top_donors={k: round(float(v), 3) for k, v in self.weights_.head(5).items()})
        return self

    def _placebos(self) -> pd.DataFrame:
        rows = []
        Y, pre, target = self._Y, self._pre, self._target
        exclude = self._other_treated | {target}
        pool, _ = self._donor_pool(target, self._other_treated, self.buffer)
        for j in pool:
            donors, _ = self._donor_pool(j, exclude, self.buffer)
            if len(donors) < 2:
                continue
            _, _, gap, pr, po = self._fit_one(Y, pre, j, donors)
            rows.append({"unit": self._panel.units[j], "att": gap[~pre].mean(), "pre_rmspe": pr,
                         "post_rmspe": po, "ratio": po / max(pr, 1e-12)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def spillovers(self) -> pd.DataFrame:
        """Spillover onto each excluded nearby unit.

        Each unit inside the buffer gets its own synthetic control from the
        same far-away donor pool; its post-period gap estimates the
        spillover it received.
        """
        if not hasattr(self, "_target"):
            raise RuntimeError("fit first")
        Y, pre, target = self._Y, self._pre, self._target
        donors, dist = self._donor_pool(target, self._other_treated, self.buffer)
        rows = []
        for u in self.excluded_near_:
            j = int(np.flatnonzero(self._panel.units == u)[0])
            _, _, gap, pr, _ = self._fit_one(Y, pre, j, donors)
            rows.append({"unit": u, "distance": dist[j], "spillover": gap[~pre].mean(), "pre_rmspe": pr})
        return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)

    def buffer_sensitivity(self, buffers: Sequence[float]) -> pd.DataFrame:
        """ATT and pre-period fit for a range of buffer distances (no placebos)."""
        rows = []
        for b in buffers:
            donors, _ = self._donor_pool(self._target, self._other_treated, b)
            if len(donors) < 2:
                rows.append({"buffer": b, "n_donors": len(donors), "att": np.nan, "pre_rmspe": np.nan})
                continue
            _, _, gap, pr, _ = self._fit_one(self._Y, self._pre, self._target, donors)
            rows.append({"buffer": b, "n_donors": len(donors), "att": gap[~self._pre].mean(), "pre_rmspe": pr})
        return pd.DataFrame(rows)

    def effect(self) -> float:
        return self.att_

    def summary(self) -> str:
        return self.result_.summary()
