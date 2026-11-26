"""Spatially varying treatment effects with geographically weighted estimators.

Two estimators, both built on the kernels and bandwidth search of
``geolattice.spatial_statistics.gwmodel``:

:class:`GWDML`
    Geographically weighted double machine learning. Nuisance models are
    fitted once, globally, with spatial cross-fitting (as in
    :class:`~geolattice.spatial_analysis.spatialcausal.ml.SpatialDML`). The local effect is then a
    kernel-weighted residual-on-residual regression,

        theta(s_i) = sum_j K_ij (v_j - v_i*)(u_j - u_i*) / sum_j K_ij (v_j - v_i*)^2,

    i.e. a GWR of the outcome residual ``u`` on the treatment residual ``v``.
    Works for binary or continuous treatments and gives a local SE for every
    unit plus a Monte-Carlo test of spatial stationarity.

:class:`GWCausalForest`
    Geographically weighted causal forest (binary treatment). AIPW
    pseudo-outcomes are cross-fitted on spatial blocks, then regressed on
    covariates with *local* random forests whose training samples are
    kernel-weighted around calibration knots, blended with a global forest
    (the geographical random forest idea of Georganos et al., 2021). All
    reported CATEs are out-of-bag, so they are not in-sample fits.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from geolattice.spatial_statistics.gwmodel.core.bandwidth import BandwidthSelector
from geolattice.spatial_statistics.gwmodel.core.kernels import KERNEL_FUNCTIONS

from geolattice.spatial_analysis.spatialcausal._utils import as_2d, as_coords, check_lengths
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.crossfit import make_folds
from geolattice.spatial_analysis.spatialcausal.inference import standard_error
from geolattice.spatial_analysis.spatialcausal.ml import (
    FeatureSpec,
    _is_binary,
    _unpack,
    aipw_pseudo_outcomes,
    dml_residuals,
    spatial_features,
)

_CHUNK = 1024


# ----------------------------------------------------------------------------
# Kernel weights (vectorised version of geolattice.spatial_statistics SpatialKernel semantics)
# ----------------------------------------------------------------------------
def _kernel_rows(D: np.ndarray, kernel: str, bandwidth: float, fixed: bool) -> np.ndarray:
    """Kernel weights for a block of rows of a distance matrix.

    Matches ``geolattice.spatial_statistics.gwmodel.core.kernels.SpatialKernel``: a fixed
    bandwidth is a distance; an adaptive bandwidth ``k`` uses, for each row,
    the distance to the ``k``-th nearest point (index 0 being the point
    itself) as that row's bandwidth.
    """
    fn = KERNEL_FUNCTIONS[kernel]
    if fixed:
        return fn(D, float(bandwidth))
    k = int(min(max(int(round(bandwidth)), 1), D.shape[1] - 1))
    h = np.partition(D, k, axis=1)[:, k]
    if np.any(h == 0):
        h2 = np.partition(D, min(k + 1, D.shape[1] - 1), axis=1)[:, min(k + 1, D.shape[1] - 1)]
        h = np.where(h == 0, h2, h)
        h = np.where(h == 0, 1.0, h)
    return fn(D, h[:, None])


class _KernelEngine:
    """Computes ``K @ V`` and row-wise kernel quantities without holding K in memory
    for large n (rows are processed in chunks)."""

    def __init__(self, coords: np.ndarray, kernel: str, fixed: bool, targets: Optional[np.ndarray] = None):
        if kernel not in KERNEL_FUNCTIONS:
            raise ValueError(f"kernel must be one of {sorted(KERNEL_FUNCTIONS)}")
        self.coords = coords
        self.targets = coords if targets is None else targets
        self.kernel, self.fixed = kernel, fixed
        self._D = cdist(self.targets, coords) if len(self.targets) * len(coords) <= 9_000_000 else None

    def _dist(self, rows: slice) -> np.ndarray:
        if self._D is not None:
            return self._D[rows]
        return cdist(self.targets[rows], self.coords)

    def chunks(self, bandwidth: float, exclude_self: bool = False):
        m = len(self.targets)
        for start in range(0, m, _CHUNK):
            rows = slice(start, min(start + _CHUNK, m))
            K = _kernel_rows(self._dist(rows), self.kernel, bandwidth, self.fixed)
            if exclude_self:
                idx = np.arange(rows.start, rows.stop)
                K[idx - rows.start, idx] = 0.0
            yield rows, K

    def matmul(self, V: np.ndarray, bandwidth: float, exclude_self: bool = False) -> np.ndarray:
        out = np.empty((len(self.targets), V.shape[1]))
        for rows, K in self.chunks(bandwidth, exclude_self):
            out[rows] = K @ V
        return out

    def default_bounds(self, n_min: int = 30) -> Tuple[float, float]:
        n = len(self.coords)
        if not self.fixed:
            return float(min(n_min, n - 2)), float(n - 1)
        sub = self.coords[np.random.default_rng(0).choice(n, size=min(n, 500), replace=False)]
        d = cdist(sub, sub)
        nn = np.sort(d, axis=1)[:, min(n_min, len(sub) - 1)]
        return float(np.median(nn)), float(d.max())


def _select_bandwidth(score_fn, engine: _KernelEngine, bounds, n_min: int):
    lo, hi = bounds if bounds is not None else engine.default_bounds(n_min)
    tol = 1.0 if not engine.fixed else (hi - lo) / 200.0
    sel = BandwidthSelector(score_fn, fixed=engine.fixed, bw_min=lo, bw_max=hi, criterion="CV", tol=tol)
    bw, score = sel.golden_section()
    if not engine.fixed:
        bw = float(int(round(bw)))
    return bw, score, sel


# ----------------------------------------------------------------------------
# GW-DML
# ----------------------------------------------------------------------------
class GWDML:
    """Geographically weighted double machine learning.

    Parameters
    ----------
    model_y, model_t : str or sklearn estimator
        Global nuisance learners (as in :class:`SpatialDML`).
    spatial_features : spatial features for the nuisance models.
    bandwidth : "auto", or a number
        ``"auto"`` minimises the leave-one-out CV error of the local
        residual-on-residual fit with ``geolattice.spatial_statistics`` golden-section search.
        Adaptive bandwidths (``fixed=False``) are numbers of neighbours.
    kernel : one of the ``geolattice.spatial_statistics.gwmodel`` kernels
        ``"bisquare"`` (default), ``"gaussian"``, ``"tricube"``,
        ``"exponential"``, ``"boxcar"``, ``"triangular"``.
    fixed : bool
        Fixed-distance (True) or adaptive k-NN (False) bandwidth.
    bw_bounds : (low, high), optional
        Search bounds for the bandwidth.
    min_neighbors : int
        Lower bound for an adaptive bandwidth search (default 30).
    cv, n_splits, buffer : cross-fitting of the nuisance models.
    se, cutoff : SE type for the *global* effect.

    Attributes
    ----------
    local_effects_ : DataFrame
        One row per unit: ``theta``, ``se``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``eff_n`` (kernel effective sample size).
    bandwidth_ : float
    result_ : SpatialCausalResult with the global effect.

    Notes
    -----
    Local SEs are kernel-weighted sandwich SEs that treat the cross-fitted
    residuals as independent. Neighbouring local estimates share data, so
    pointwise intervals must not be read as a joint band; use
    :meth:`stationarity_test` to test whether the effect varies at all.
    """

    def __init__(self, model_y="rf", model_t="rf", spatial_features: FeatureSpec = ("coords", "lag"),
                 bandwidth="auto", kernel: str = "bisquare", fixed: bool = False,
                 bw_bounds: Optional[Tuple[float, float]] = None, min_neighbors: int = 30,
                 cv="spatial_block", n_splits: int = 5, buffer: float = 0.0, n_estimators: int = 200,
                 se: str = "iid", cutoff: Optional[float] = None, trim: float = 0.01, random_state: int = 0):
        self.model_y, self.model_t = model_y, model_t
        self.spatial_features = spatial_features
        self.bandwidth, self.kernel, self.fixed = bandwidth, kernel, fixed
        self.bw_bounds, self.min_neighbors = bw_bounds, min_neighbors
        self.cv, self.n_splits, self.buffer = cv, n_splits, buffer
        self.n_estimators = n_estimators
        self.se, self.cutoff, self.trim = se, cutoff, trim
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def _local_slope(M: np.ndarray):
        """Local intercept/slope from kernel moments [S0, Su, Sv, Suv, Svv]."""
        S0 = np.where(M[:, 0] > 0, M[:, 0], np.nan)
        ub, vb = M[:, 1] / S0, M[:, 2] / S0
        cov = M[:, 3] / S0 - ub * vb
        var = M[:, 4] / S0 - vb ** 2
        theta = cov / np.where(var > 1e-12, var, np.nan)
        return ub, vb, theta

    def _cv_score(self, bw: float) -> float:
        u, v = self.residuals_["y"], self.residuals_["t"]
        V = np.column_stack([np.ones_like(u), u, v, u * v, v * v])
        ub, vb, th = self._local_slope(self._engine.matmul(V, bw, exclude_self=True))
        pred = ub + th * (v - vb)
        err = (u - pred) ** 2
        if not np.all(np.isfinite(err)):
            return 1e300
        return float(np.mean(err))

    def _local_fit(self, u, v, bw):
        n = len(u)
        theta = np.empty(n)
        se = np.empty(n)
        neff = np.empty(n)
        for rows, K in self._engine.chunks(bw):
            S0 = K.sum(1)
            ub = K @ u / S0
            vb = K @ v / S0
            dv = v[None, :] - vb[:, None]
            du = u[None, :] - ub[:, None]
            sxx = np.sum(K * dv * dv, 1)
            th = np.sum(K * dv * du, 1) / sxx
            e = du - th[:, None] * dv
            theta[rows] = th
            se[rows] = np.sqrt(np.sum(K ** 2 * dv ** 2 * e ** 2, 1)) / sxx
            neff[rows] = S0 ** 2 / np.sum(K ** 2, 1)
        return theta, se, neff

    # ------------------------------------------------------------------
    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        X, t, y, coords, w = _unpack(data_or_X, treatment, y, coords, w)
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        F = spatial_features(X, coords, w, self.spatial_features, random_state=self.random_state)
        kw = {"buffer": self.buffer} if self.cv == "spatial_block" else {}
        folds = make_folds(coords, self.cv, self.n_splits, self.random_state, **kw)
        u, v, _, _ = dml_residuals(F, t, y, folds, self.model_y, self.model_t,
                                   self.n_estimators, self.trim, self.random_state)
        self.residuals_ = {"y": u, "t": v}
        self.coords_ = coords
        self._engine = _KernelEngine(coords, self.kernel, self.fixed)

        if self.bandwidth == "auto":
            bw, score, sel = _select_bandwidth(self._cv_score, self._engine, self.bw_bounds, self.min_neighbors)
            self.bandwidth_history_ = pd.DataFrame({"bandwidth": sel.bw_history_, "cv": sel.score_history_})
        else:
            bw = float(self.bandwidth)
            score = self._cv_score(bw)
        self.bandwidth_, self.cv_score_ = bw, score

        theta, se, neff = self._local_fit(u, v, bw)
        z = stats.norm.ppf(0.975)
        self.local_effects_ = pd.DataFrame({
            "theta": theta, "se": se, "ci_lower": theta - z * se, "ci_upper": theta + z * se,
            "p_value": 2 * stats.norm.sf(np.abs(theta / se)), "eff_n": neff})

        th_g = float(np.sum(v * u) / np.sum(v * v))
        psi = v * (u - th_g * v) / np.mean(v * v)
        se_g = standard_error(psi, coords, self.se, self.cutoff)
        self.result_ = SpatialCausalResult(
            "GW-DML", {"ate": EffectEstimate("ate", th_g, se_g)}, se_method=self.se, n=n,
            kernel=f"{self.kernel} ({'fixed' if self.fixed else 'adaptive'})",
            bandwidth=round(bw, 4), local_theta_range=(round(float(np.nanmin(theta)), 3),
                                                       round(float(np.nanmax(theta)), 3)),
            share_local_significant=round(float(np.mean(self.local_effects_["p_value"] < 0.05)), 3))
        return self

    # ------------------------------------------------------------------
    def stationarity_test(self, n_perm: int = 99, random_state: int = 0):
        """Monte-Carlo test of H0: the effect does not vary over space.

        Statistic: SD of the local estimates. Under H0, randomly reassigning
        the residual pairs ``(u_j, v_j)`` to locations leaves the
        distribution of the statistic unchanged (cf. the GWR Monte-Carlo
        test of Brunsdon, Fotheringham & Charlton, 1998).

        Returns ``(statistic, p_value, null_draws)``.
        """
        u, v = self.residuals_["y"], self.residuals_["t"]
        V = lambda uu, vv: np.column_stack([np.ones_like(uu), uu, vv, uu * vv, vv * vv])  # noqa: E731
        obs = float(np.nanstd(self._local_slope(self._engine.matmul(V(u, v), self.bandwidth_))[2]))
        rng = np.random.default_rng(random_state)
        null = np.empty(n_perm)
        for b in range(n_perm):
            p = rng.permutation(len(u))
            null[b] = np.nanstd(self._local_slope(self._engine.matmul(V(u[p], v[p]), self.bandwidth_))[2])
        pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
        return obs, float(pval), null

    def predict(self, coords) -> np.ndarray:
        """Local effect at new locations, using the training residuals."""
        tgt = as_coords(coords)
        eng = _KernelEngine(self.coords_, self.kernel, self.fixed, targets=tgt)
        u, v = self.residuals_["y"], self.residuals_["t"]
        V = np.column_stack([np.ones_like(u), u, v, u * v, v * v])
        return self._local_slope(eng.matmul(V, self.bandwidth_))[2]

    def effect(self) -> float:
        return self.result_["ate"].estimate

    def summary(self) -> str:
        return self.result_.summary()


# ----------------------------------------------------------------------------
# GW causal forest
# ----------------------------------------------------------------------------
class GWCausalForest:
    """Geographically weighted causal forest for a binary treatment.

    Parameters
    ----------
    model_y, model_t : nuisance learners for the AIPW pseudo-outcome.
    spatial_features : spatial features for the nuisance models.
    cate_features : {"X", "X+coords"}
        Covariates for the forests. Location already enters through the
        geographic weighting, so ``"X"`` is the default.
    bandwidth : "auto" or a number
        ``"auto"`` minimises the leave-one-out error of a kernel-weighted
        mean of the pseudo-outcome (the spatial scale of effect
        heterogeneity), via ``geolattice.spatial_statistics`` golden-section search.
    kernel, fixed : ``geolattice.spatial_statistics.gwmodel`` kernel and bandwidth type.
    n_knots : int
        Calibration points (k-means centres) at which local forests are fit.
        Each unit uses the forest of its nearest knot.
    local_weight : "auto" or float in [0, 1]
        Weight of the local forest against the global forest. ``"auto"``
        minimises the out-of-bag error.
    engine : {"knots", "spatialstats"}
        ``"knots"`` (default) is the fast knot-based forest above.
        ``"spatialstats"`` (alias ``"spatial_statistics"``) runs ``geolattice.spatial_statistics.gwmodel.ensemble.GWRandomForest``
        on the pseudo-outcomes, cross-fitted, with one local forest per unit
        (much slower; for n ≲ 1000).
    n_estimators_local : trees per local forest.
    min_samples_leaf : "auto" or int
        ``"auto"`` chooses among three sizes by out-of-bag error.

    Attributes
    ----------
    cate_ : out-of-bag (or out-of-fold) CATE for each unit.
    local_ate_ : DataFrame — kernel-weighted AIPW mean at each unit with SE.
    knots_, local_importance_ : knot coordinates and local feature importances.
    bandwidth_, local_weight_ : selected tuning values.
    """

    def __init__(self, model_y="gbr", model_t="gbr", spatial_features: FeatureSpec = ("coords", "lag"),
                 cate_features: str = "X", bandwidth="auto", kernel: str = "bisquare", fixed: bool = False,
                 bw_bounds: Optional[Tuple[float, float]] = None, min_neighbors: int = 50,
                 n_knots: int = 50, local_weight="auto", engine: str = "knots",
                 n_estimators_local: int = 150, min_samples_leaf="auto",
                 cv="spatial_block", n_splits: int = 5, buffer: float = 0.0, n_estimators: int = 200,
                 se: str = "iid", cutoff: Optional[float] = None, trim: float = 0.01, random_state: int = 0):
        if cate_features not in ("X", "X+coords"):
            raise ValueError("cate_features must be 'X' or 'X+coords'")
        if engine == "spatial_statistics":
            engine = "spatialstats"
        if engine not in ("knots", "spatialstats"):
            raise ValueError("engine must be 'knots' or 'spatialstats'")
        self.model_y, self.model_t = model_y, model_t
        self.spatial_features = spatial_features
        self.cate_features = cate_features
        self.bandwidth, self.kernel, self.fixed = bandwidth, kernel, fixed
        self.bw_bounds, self.min_neighbors = bw_bounds, min_neighbors
        self.n_knots, self.local_weight, self.engine = n_knots, local_weight, engine
        self.n_estimators_local, self.min_samples_leaf = n_estimators_local, min_samples_leaf
        self.cv, self.n_splits, self.buffer = cv, n_splits, buffer
        self.n_estimators = n_estimators
        self.se, self.cutoff, self.trim = se, cutoff, trim
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _features(self, X, coords):
        X = as_2d(X)
        if self.cate_features == "X":
            return X
        return np.hstack([X, (as_coords(coords) - self._cmean) / self._cstd])

    def _forest(self, m: int, oob: bool = True, leaf: Optional[int] = None, n_trees: Optional[int] = None):
        if leaf is None:
            leaf = self.leaf_ if hasattr(self, "leaf_") else max(10, m // 15)
        return RandomForestRegressor(n_estimators=n_trees or self.n_estimators_local, min_samples_leaf=int(leaf),
                                     max_features=1.0 if self._p <= 3 else 0.6, oob_score=oob,
                                     random_state=self.random_state, n_jobs=-1)

    def _smooth_cv(self, bw: float) -> float:
        g = self.pseudo_outcome_
        M = self._engine.matmul(np.column_stack([np.ones_like(g), g]), bw, exclude_self=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = M[:, 1] / M[:, 0]
        err = (g - pred) ** 2
        return float(np.mean(err)) if np.all(np.isfinite(err)) else 1e300

    def _local_ate(self, bw):
        g = self.pseudo_outcome_
        est, se, neff = np.empty(len(g)), np.empty(len(g)), np.empty(len(g))
        for rows, K in self._engine.chunks(bw):
            S0 = K.sum(1)
            m = K @ g / S0
            est[rows] = m
            se[rows] = np.sqrt(np.sum(K ** 2 * (g[None] - m[:, None]) ** 2, 1)) / S0
            neff[rows] = S0 ** 2 / np.sum(K ** 2, 1)
        z = stats.norm.ppf(0.975)
        return pd.DataFrame({"ate": est, "se": se, "ci_lower": est - z * se, "ci_upper": est + z * se,
                             "p_value": 2 * stats.norm.sf(np.abs(est / se)), "eff_n": neff})

    # ------------------------------------------------------------------
    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        X, t, y, coords, w = _unpack(data_or_X, treatment, y, coords, w)
        if not _is_binary(t):
            raise ValueError("GWCausalForest needs a binary treatment; use GWDML for continuous")
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        F = spatial_features(X, coords, w, self.spatial_features, random_state=self.random_state)
        kw = {"buffer": self.buffer} if self.cv == "spatial_block" else {}
        folds = make_folds(coords, self.cv, self.n_splits, self.random_state, **kw)
        gamma, e, _, _ = aipw_pseudo_outcomes(F, t, y, folds, self.model_y, self.model_t,
                                              self.n_estimators, self.trim, self.random_state)
        self.pseudo_outcome_, self.propensity_ = gamma, e
        self.coords_ = coords
        self._cmean, self._cstd = coords.mean(0), coords.std(0)
        Fc = self._features(X, coords)
        self._p = Fc.shape[1]
        self._engine = _KernelEngine(coords, self.kernel, self.fixed)

        if self.bandwidth == "auto":
            bw, _, _ = _select_bandwidth(self._smooth_cv, self._engine, self.bw_bounds, self.min_neighbors)
        else:
            bw = float(self.bandwidth)
        self.bandwidth_ = bw
        self.local_ate_ = self._local_ate(bw)

        if self.engine == "knots":
            self._fit_knots(Fc, gamma, coords)
        else:
            self._fit_spatialstats(Fc, gamma, coords)

        ate = float(gamma.mean())
        se = standard_error(gamma, coords, self.se, self.cutoff)
        self.result_ = SpatialCausalResult(
            "GW causal forest", {"ate": EffectEstimate("ate", ate, se)}, se_method=self.se, n=n,
            engine=self.engine, kernel=f"{self.kernel} ({'fixed' if self.fixed else 'adaptive'})",
            bandwidth=round(bw, 4), local_weight=round(float(self.local_weight_), 2),
            min_samples_leaf=getattr(self, "leaf_", None),
            propensity_range=(round(float(e.min()), 3), round(float(e.max()), 3)))
        return self

    @staticmethod
    def _knot_samples(wts: np.ndarray, n: int):
        mask = wts > 0.01 * wts.max()
        if mask.sum() < 30:
            mask = np.isin(np.arange(n), np.argsort(-wts)[:30])
        return np.flatnonzero(mask)

    def _fit_local(self, Fc, gamma, idx, wts, leaf=None, n_trees=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._forest(len(idx), leaf=leaf, n_trees=n_trees).fit(
                Fc[idx], gamma[idx], sample_weight=wts[idx] / wts.max())

    def _choose_leaf(self, Fc, gamma, Kc, nearest, rng):
        """Pick min_samples_leaf by out-of-bag error on a subset of knots.

        AIPW pseudo-outcomes are very noisy, so the leaf size that works
        depends on the noise level; candidates scale with the median local
        sample size ``m``: ``m/15``, ``m/6``, ``m/3``. Evaluated on every
        unit with cheap 50-tree forests, because the error differences are
        small relative to the pseudo-outcome noise.
        """
        n = len(gamma)
        sizes = [len(self._knot_samples(w, n)) for w in Kc]
        m = int(np.median(sizes))
        cands = sorted({max(10, m // 15), max(10, m // 6), max(10, m // 3)})
        sub = np.arange(len(Kc))
        best, best_err = cands[0], np.inf
        for leaf in cands:
            err, cnt = 0.0, 0
            for r in sub:
                idx = self._knot_samples(Kc[r], n)
                rf = self._fit_local(Fc, gamma, idx, Kc[r], leaf=leaf, n_trees=50)
                own = np.searchsorted(idx, idx[nearest[idx] == r])
                pred = rf.oob_prediction_[own]
                ok = np.isfinite(pred)
                err += np.sum((gamma[idx[own]][ok] - pred[ok]) ** 2)
                cnt += ok.sum()
            if cnt and err / cnt < best_err:
                best, best_err = leaf, err / cnt
        return best

    def _fit_knots(self, Fc, gamma, coords):
        n = len(gamma)
        k = min(self.n_knots, n)
        km = KMeans(n_clusters=k, n_init=4, random_state=self.random_state).fit(coords)
        self.knots_ = km.cluster_centers_
        nearest = km.predict(coords)
        eng = _KernelEngine(coords, self.kernel, self.fixed, targets=self.knots_)
        Kc = np.vstack([K for _, K in eng.chunks(self.bandwidth_)])
        if self.min_samples_leaf == "auto":
            self.leaf_ = self._choose_leaf(Fc, gamma, Kc, nearest, np.random.default_rng(self.random_state))
        else:
            self.leaf_ = int(self.min_samples_leaf)
        self.global_forest_ = self._forest(n).fit(Fc, gamma)
        glob_oob = self.global_forest_.oob_prediction_
        loc_oob = np.full(n, np.nan)
        self.local_forests_, self.local_importance_ = [], np.zeros((k, Fc.shape[1]))
        for r, wts in enumerate(Kc):
            idx = self._knot_samples(wts, n)
            rf = self._fit_local(Fc, gamma, idx, wts)
            self.local_forests_.append(rf)
            self.local_importance_[r] = rf.feature_importances_
            own = idx[nearest[idx] == r]
            loc_oob[own] = rf.oob_prediction_[np.searchsorted(idx, own)]
        bad = ~np.isfinite(loc_oob)
        loc_oob[bad] = glob_oob[bad]
        self._nearest_knot = nearest
        self.local_weight_ = (self._choose_weight(gamma, loc_oob, glob_oob) if self.local_weight == "auto"
                              else float(self.local_weight))
        lw = self.local_weight_
        self.cate_ = lw * loc_oob + (1 - lw) * glob_oob

    def _fit_spatialstats(self, Fc, gamma, coords):
        from geolattice.spatial_statistics.gwmodel.ensemble import GWRandomForest

        if len(gamma) > 1500:
            warnings.warn("engine='spatialstats' fits one forest per unit; this will be slow", RuntimeWarning)
        lw = 1.0 if self.local_weight == "auto" else float(self.local_weight)

        def make():
            return GWRandomForest(n_estimators=min(self.n_estimators_local, 100), bandwidth=self.bandwidth_,
                                  fixed=self.fixed, kernel=self.kernel, local_weight=lw,
                                  random_state=self.random_state)

        oof = np.full(len(gamma), np.nan)
        for train, test in make_folds(coords, "random", self.n_splits, self.random_state):
            oof[test] = make().fit(Fc[train], gamma[train], coords[train]).predict(Fc[test], coords[test])
        self.gwrf_ = make().fit(Fc, gamma, coords)
        self.knots_, self.local_importance_ = coords, self.gwrf_.feature_importances_local_
        self.local_weight_ = lw
        self.cate_ = oof

    @staticmethod
    def _choose_weight(gamma, loc, glob):
        grid = np.linspace(0, 1, 21)
        mse = [np.mean((gamma - (a * loc + (1 - a) * glob)) ** 2) for a in grid]
        return float(grid[int(np.argmin(mse))])

    def _resolved_local_weight(self):
        return self.local_weight_ if self.local_weight == "auto" else float(self.local_weight)

    # ------------------------------------------------------------------
    def predict(self, X, coords) -> np.ndarray:
        """CATE for new units (nearest-knot local forest blended with the global forest)."""
        coords = as_coords(coords)
        Fc = self._features(X, coords)
        if self.engine == "spatialstats":
            return self.gwrf_.predict(Fc, coords)
        knot = np.argmin(cdist(coords, self.knots_), axis=1)
        loc = np.empty(len(Fc))
        for k in np.unique(knot):
            m = knot == k
            loc[m] = self.local_forests_[k].predict(Fc[m])
        lw = self._resolved_local_weight()
        return lw * loc + (1 - lw) * self.global_forest_.predict(Fc)

    def effect(self) -> float:
        return self.result_["ate"].estimate

    def summary(self) -> str:
        return self.result_.summary()
