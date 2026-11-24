"""Spatial causal discovery.

Discovery methods suggest causal structure from observational data under
strong assumptions; treat their output as hypotheses to test with a design
(the rest of this package), not as established effects.

:class:`SpatialGranger`
    Panel Granger causality with spatial lags. Tests whether past values of
    ``x`` at a unit (*local* channel) and at its neighbours (*spillover*
    channel) predict ``y`` beyond ``y``'s own past, its neighbours' past
    and unit/time fixed effects.

:class:`GCCM`
    Geographical convergent cross mapping (Gao et al., 2023) for
    cross-sectional spatial data: state-space reconstruction with spatial
    lags of increasing order replaces time-delay embedding. If ``x`` causes
    ``y``, the manifold of ``y`` carries information about ``x``, and the
    skill of cross-mapping ``x`` from ``y`` rises and converges as the
    library grows.

:class:`SpatialNOTEARS`
    Linear NOTEARS (Zheng et al., 2018) — continuous-optimisation structure
    learning with a smooth acyclicity constraint — implemented in NumPy /
    SciPy (no deep-learning dependencies), with spatial-lag nodes ``W x_j``
    added as exogenous candidate causes, so the learned graph separates
    local edges (``x_j -> x_k``) from spillover edges (``W x_j -> x_k``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
from scipy.linalg import expm

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_2d, as_coords, as_sparse_weights, default_weights
from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel


# ============================================================================
# Spatial Granger causality (panel)
# ============================================================================
class SpatialGranger:
    """Panel Granger causality test with spatial lags.

    Model::

        y_it = a_i + l_t + sum_l [ r_l y_i,t-l + g_l (Wy)_i,t-l
                                   + b_l x_i,t-l + h_l (Wx)_i,t-l ] + e_it

    Tests: ``local`` (all ``b_l = 0``), ``spillover`` (all ``h_l = 0``) and
    ``joint``. Wald statistics use cluster-robust covariances and a
    Hotelling-type F reference distribution.

    Parameters
    ----------
    lags : int
    w : spatial weights over units (default 8-NN of the unit coordinates).
    spatial : bool
        Include the ``Wy`` and ``Wx`` terms.
    cluster : {"spatial", "unit", "time"}
        ``"spatial"`` (default) clusters on spatial blocks of units over all
        periods, which allows shocks to be correlated between neighbours as
        well as over time. Unit clustering ignores cross-sectional
        correlation and over-rejects when shocks are spatially correlated
        (a false "y causes x" at p < 0.01 in our simulation).
    n_clusters : "auto" or int
        Spatial blocks; ``"auto"`` = ``max(30, n_units // 10)``.

    bias_correction : {"jackknife", None}
        With unit fixed effects and lagged regressors, OLS has Nickell bias
        of order 1/T. In large panels that bias is "significant" and creates
        spurious Granger causality (in our simulation the false direction was
        rejected 25–57% of the time without correction). ``"jackknife"``
        applies the half-panel jackknife of Dhaene & Jochmans (2015),
        ``2 b - (b_first_half + b_second_half) / 2``; SEs are from the full
        panel.

    Notes
    -----
    Granger causality is predictive: an unobserved common driver of both
    series can produce it.
    """

    def __init__(self, lags: int = 1, w=None, spatial: bool = True, cluster: str = "spatial",
                 n_clusters="auto", bias_correction: Optional[str] = "jackknife", random_state: int = 0):
        if lags < 1:
            raise ValueError("lags must be >= 1")
        if cluster not in ("unit", "spatial", "time"):
            raise ValueError("cluster must be 'unit', 'spatial' or 'time'")
        self.lags, self.w, self.spatial = int(lags), w, spatial
        self.cluster, self.n_clusters, self.random_state = cluster, n_clusters, random_state
        if bias_correction not in (None, "jackknife"):
            raise ValueError("bias_correction must be None or 'jackknife'")
        self.bias_correction = bias_correction

    @staticmethod
    def _series(panel: SpatialPanel, v) -> np.ndarray:
        if v is None:
            return panel.wide()
        if isinstance(v, (int, np.integer)):
            if panel.X is None:
                raise ValueError("panel has no covariates X")
            arr = panel.X[:, int(v)]
        else:
            arr = as_1d(v, "series")
        M = np.full((panel.n_units, len(panel.times)), np.nan)
        M[panel.unit_index, panel.time_index] = arr
        return M

    def fit(self, panel: SpatialPanel, cause, effect=None):
        """``cause``: column index of ``panel.X`` or an (n_obs,) array. ``effect``: same, default ``panel.y``."""
        if not panel.is_balanced:
            raise ValueError("SpatialGranger needs a balanced panel")
        Y, Xc = self._series(panel, effect), self._series(panel, cause)
        N, T = Y.shape
        L = self.lags
        if T - L < 3:
            raise ValueError("too few periods for the number of lags")
        Wm = as_sparse_weights(self.w if self.w is not None else default_weights(panel.unit_coords),
                               N, row_standardize=True)
        WY, WX = np.asarray(Wm @ Y), np.asarray(Wm @ Xc)
        blocks = [("y", Y), ("Wy", WY), ("x", Xc), ("Wx", WX)] if self.spatial else [("y", Y), ("x", Xc)]
        from geolattice.spatial_analysis.spatialcausal.quasi import _cluster_vcov, _ols_twfe

        def design(t0, t1):
            """Regression rows for outcome periods t0..t1-1 (needs t0 >= L)."""
            cols, nms = [], []
            for nm, M in blocks:
                for l in range(1, L + 1):
                    cols.append(M[:, t0 - l:t1 - l].ravel())
                    nms.append(f"{nm}_lag{l}")
            yv_ = Y[:, t0:t1].ravel()
            u_ = np.repeat(np.arange(N), t1 - t0)
            t_ = np.tile(np.arange(t1 - t0), N)
            return np.column_stack(cols), yv_, u_, t_, nms

        D, yv, uid, tid, names = design(L, T)
        beta, psi = _ols_twfe(yv, D, uid, tid)
        if self.bias_correction == "jackknife":
            # half-panel jackknife (Dhaene & Jochmans 2015): removes the O(1/T) Nickell bias
            mid = L + (T - L) // 2
            if mid - L < 3 or T - mid < 3:
                raise ValueError("too few periods for the half-panel jackknife; use bias_correction=None")
            b1 = _ols_twfe(*[design(L, mid)[i] for i in (1, 0, 2, 3)])[0]
            b2 = _ols_twfe(*[design(mid, T)[i] for i in (1, 0, 2, 3)])[0]
            self.uncorrected_coef_ = pd.Series(beta, index=names)
            beta = 2 * beta - 0.5 * (b1 + b2)
        if self.cluster == "unit":
            cl = uid
        elif self.cluster == "time":
            cl = tid
        else:
            from geolattice.spatial_analysis.spatialcausal.crossfit import SpatialBlockKFold
            G0 = max(30, N // 10) if self.n_clusters == "auto" else int(self.n_clusters)
            cl = SpatialBlockKFold(n_splits=2, n_blocks=min(G0, N), random_state=self.random_state).make_blocks(
                panel.unit_coords)[uid]
        V = _cluster_vcov(psi, cl)
        G_cl = len(np.unique(cl))
        se = np.sqrt(np.diag(V))
        self.coef_ = pd.DataFrame({"coef": beta, "se": se, "t": beta / se,
                                   "p_value": 2 * stats.norm.sf(np.abs(beta / se))}, index=names)

        def wald(prefix):
            idx = [i for i, nm in enumerate(names) if nm.split("_lag")[0] in prefix]
            b, Vs = beta[idx], V[np.ix_(idx, idx)]
            W = float(b @ np.linalg.pinv(Vs) @ b)
            q, G = len(idx), G_cl
            F = W * (G - q) / ((G - 1) * q)
            return {"wald": W, "F": F, "df": (q, G - q), "p_value": float(stats.f.sf(F, q, G - q))}

        self.tests_ = {"local": wald(("x",))}
        if self.spatial:
            self.tests_["spillover"] = wald(("Wx",))
            self.tests_["joint"] = wald(("x", "Wx"))
        self._panel, self._cause, self._effect = panel, cause, effect
        return self

    def summary(self) -> str:
        lines = [f"Spatial Granger causality (lags={self.lags})", "=" * 40, self.coef_.round(4).to_string(), ""]
        for k, v in self.tests_.items():
            lines.append(f"  H0 no {k:9s} Granger causality: F={v['F']:.3f}, df={v['df']}, p={v['p_value']:.4g}")
        return "\n".join(lines)


# ============================================================================
# Geographical convergent cross mapping
# ============================================================================
def spatial_lag_rings(w, n: int, order: int):
    """Row-normalised operators averaging over the k-th order neighbour ring, k = 1..order.

    Ring k = units reachable in exactly k steps on the (symmetrised, binary)
    neighbour graph.
    """
    S = as_sparse_weights(w, n)
    A = ((S + S.T) > 0).astype(float)
    Ai = (A + sparse.identity(n, format="csr")).tocsr()
    reach_prev = sparse.identity(n, format="csr")
    reach = sparse.identity(n, format="csr")
    rings = []
    for _ in range(order):
        reach = ((reach @ Ai) > 0).astype(float)
        ring = (reach - reach_prev).tocsr()
        ring.data = (ring.data > 0).astype(float)
        ring.eliminate_zeros()
        rs = np.asarray(ring.sum(1)).ravel()
        rs[rs == 0] = 1.0
        rings.append(sparse.diags(1.0 / rs) @ ring)
        reach_prev = reach
    return rings


class GCCM:
    """Geographical convergent cross mapping.

    Parameters
    ----------
    E : int
        Embedding dimension: the value at a unit plus the means of its first
        ``E - 1`` neighbour rings.
    lib_sizes : sequence of int, optional
        Library sizes (default 10 values from ``E + 2`` to ``n - 1``).
    n_rep : int
        Random libraries per size.
    w : spatial weights (default 8-NN of ``coords``).
    exclusion : "auto" or int
        Exclude a unit's neighbours up to this ring order from its candidate
        nearest neighbours in state space — a spatial Theiler window. The
        default ``2 (E - 1)`` covers every unit whose neighbour rings overlap
        the target's. Without it, two independent but spatially smooth
        fields showed cross-map skill of ~0.5 in both directions in our
        simulations, purely from spatial autocorrelation.

    Attributes
    ----------
    skill_ : DataFrame indexed by library size with columns
        ``"x_causes_y"`` (skill of recovering ``x`` from ``y``'s manifold)
        and ``"y_causes_x"``, averaged over replications.
    tests_ : dict per direction — final skill and its Fisher-z p-value,
        and the convergence gain (skill at the largest minus smallest
        library) with a p-value from the replication spread.

    Notes
    -----
    CCM is designed for deterministic, nonlinear coupled systems. Read the
    *convergence* of each direction, not only its skill: with x driving y,
    both directions can show skill, but only ``x_causes_y`` rises with the
    library size. Strong forcing can make both converge (generalised
    synchrony). A common driver — including two processes evolving on the
    same neighbour graph — can also produce converging skill, which CCM
    cannot tell apart from causation.
    """

    def __init__(self, E: int = 3, lib_sizes: Optional[Sequence[int]] = None, n_rep: int = 20,
                 w=None, exclusion="auto", random_state: int = 0):
        if E < 2:
            raise ValueError("E must be >= 2")
        self.E, self.lib_sizes, self.n_rep, self.w = int(E), lib_sizes, int(n_rep), w
        self.exclusion = exclusion
        self.random_state = random_state

    def embed(self, v: np.ndarray, rings) -> np.ndarray:
        cols = [v] + [np.asarray(R @ v).ravel() for R in rings[: self.E - 1]]
        M = np.column_stack(cols)
        return (M - M.mean(0)) / (M.std(0) + 1e-12)

    def _cross_map(self, M_src, target, lib, rng):
        """Predict ``target`` at every unit from neighbours in ``M_src`` restricted to ``lib``.

        Candidate neighbours exclude the unit itself and every unit inside its
        exclusion zone (``self._forbid``), the spatial analogue of a Theiler
        window: units that share neighbour rings have similar embeddings by
        construction, and using them would let spatial autocorrelation
        masquerade as cross-map skill.
        """
        k = self.E + 1
        n = len(M_src)
        out = np.empty(n)
        Ml = M_src[lib]
        for start in range(0, n, 1024):
            rows = slice(start, min(start + 1024, n))
            D = ((M_src[rows, None, :] - Ml[None, :, :]) ** 2).sum(-1)
            D[self._forbid[rows][:, lib]] = np.inf
            kk = min(k, max(1, int(np.isfinite(D).sum(1).min())))
            idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
            d = np.sqrt(np.take_along_axis(D, idx, 1))
            order = np.argsort(d, axis=1)
            d, idx = np.take_along_axis(d, order, 1), np.take_along_axis(idx, order, 1)
            d1 = np.maximum(d[:, :1], 1e-12)
            wts = np.where(np.isfinite(d), np.exp(-d / d1), 0.0)
            wts /= np.maximum(wts.sum(1, keepdims=True), 1e-300)
            out[rows] = (wts * target[lib][idx]).sum(1)
        return out

    def fit(self, x, y, w=None, coords=None):
        x, y = as_1d(x, "x"), as_1d(y, "y")
        n = len(x)
        if len(y) != n:
            raise ValueError("x and y must have the same length")
        w = w if w is not None else self.w
        if w is None:
            if coords is None:
                raise ValueError("pass w or coords")
            w = default_weights(as_coords(coords))
        rings = spatial_lag_rings(w, n, self.E - 1)
        Mx, My = self.embed(x, rings), self.embed(y, rings)
        excl = 2 * (self.E - 1) if self.exclusion == "auto" else int(self.exclusion)
        if n > 6000:
            raise ValueError("GCCM is implemented with dense distance blocks; n > 6000 is not supported")
        forbid = np.eye(n, dtype=bool)
        if excl > 0:
            for R in spatial_lag_rings(w, n, excl):
                forbid |= (R.toarray() > 0)
        self._forbid = forbid
        libs = self.lib_sizes or np.unique(np.linspace(max(self.E + 2, n // 20), n - 1, 10).astype(int))
        rng = np.random.default_rng(self.random_state)
        res = {"x_causes_y": [], "y_causes_x": []}
        for L in libs:
            rx, ry = [], []
            for _ in range(self.n_rep):
                lib = np.sort(rng.choice(n, size=int(L), replace=False))
                rx.append(np.corrcoef(self._cross_map(My, x, lib, rng), x)[0, 1])
                ry.append(np.corrcoef(self._cross_map(Mx, y, lib, rng), y)[0, 1])
            res["x_causes_y"].append(rx)
            res["y_causes_x"].append(ry)
        self.replicates_ = {k: np.array(v) for k, v in res.items()}
        self.skill_ = pd.DataFrame({k: v.mean(1) for k, v in self.replicates_.items()},
                                   index=pd.Index(libs, name="lib_size"))
        self.tests_ = {}
        for k, R in self.replicates_.items():
            rho = float(R[-1].mean())
            zf = np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt(n - 3)
            gain = R[-1] - R[0]
            self.tests_[k] = {"skill": rho, "p_skill": float(stats.norm.sf(zf)),
                              "convergence_gain": float(gain.mean()),
                              "p_convergence": float(stats.ttest_1samp(gain, 0.0, alternative="greater").pvalue)}
        return self

    def summary(self) -> str:
        lines = [f"Geographical CCM (E={self.E})", "=" * 30, self.skill_.round(3).to_string(), ""]
        for k, v in self.tests_.items():
            lines.append(f"  {k}: skill={v['skill']:.3f} (p={v['p_skill']:.3g}), "
                         f"convergence gain={v['convergence_gain']:.3f} (p={v['p_convergence']:.3g})")
        return "\n".join(lines)


# ============================================================================
# NOTEARS (linear) with spatial-lag nodes
# ============================================================================
def notears_linear(X: np.ndarray, lambda1: float = 0.05, max_iter: int = 100, h_tol: float = 1e-8,
                   rho_max: float = 1e16, w_threshold: float = 0.3, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Linear NOTEARS with least-squares loss and L1 penalty (Zheng et al., 2018).

    ``mask[i, j] = False`` forbids the edge ``i -> j``. Returns the weighted
    adjacency matrix ``W`` (``W[i, j]`` is the effect of ``i`` on ``j``).
    """
    X = as_2d(X)
    X = X - X.mean(0)
    n, d = X.shape
    allowed = np.ones((d, d), bool) if mask is None else np.asarray(mask, bool).copy()
    np.fill_diagonal(allowed, False)

    def adj(wv):
        return (wv[: d * d] - wv[d * d:]).reshape(d, d)

    def h_fn(W):
        E = expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def func(wv):
        W = adj(wv)
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        h, G_h = h_fn(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * wv.sum()
        G = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate([G + lambda1, -G + lambda1], axis=None)

    bounds = [(0, 0) if not allowed[i, j] else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    wv, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            sol = optimize.minimize(func, wv, method="L-BFGS-B", jac=True, bounds=bounds)
            w_new = sol.x
            h_new, _ = h_fn(adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        wv, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = adj(wv)
    W[np.abs(W) < w_threshold] = 0.0
    return W


class SpatialNOTEARS:
    """Structure learning with local and spillover edges.

    The node set is the ``d`` variables plus their spatial lags ``W x_j``.
    Lag nodes are exogenous: edges *into* them are forbidden, so every
    learned edge is either local (``x_j -> x_k``) or a spillover
    (``W x_j -> x_k``, the neighbours' ``x_j`` affecting a unit's ``x_k``).

    Parameters
    ----------
    lambda1 : L1 penalty.
    w_threshold : edges with ``|weight| < w_threshold`` are dropped.
    standardize : bool
        Scale variables to unit variance first. Off by default: NOTEARS
        partly exploits differences in marginal variance ("varsortability",
        Reisach et al., 2021), so results on standardised data differ and
        should be compared, not assumed equal.
    spatial : bool
        Add spatial-lag nodes (``False`` gives plain NOTEARS).

    Notes
    -----
    Treating ``W x`` as exogenous is a working approximation: with true
    spatial feedback (``y_i`` depending on ``y_j`` and vice versa) the
    unit-level model is not a DAG. Linear-Gaussian structure is only
    partially identifiable; use the result to propose hypotheses.
    """

    def __init__(self, lambda1: float = 0.05, w_threshold: float = 0.3, standardize: bool = False,
                 spatial: bool = True, max_iter: int = 100):
        self.lambda1, self.w_threshold, self.standardize = lambda1, w_threshold, standardize
        self.spatial, self.max_iter = spatial, max_iter

    def fit(self, X, w=None, coords=None, names: Optional[Sequence[str]] = None):
        if isinstance(X, pd.DataFrame):
            names = names or [str(c) for c in X.columns]
            X = X.to_numpy(float)
        X = as_2d(X)
        n, d = X.shape
        names = list(names) if names is not None else [f"x{j}" for j in range(d)]
        Z = X - X.mean(0)
        if self.standardize:
            Z = Z / Z.std(0)
        if self.spatial:
            if w is None:
                if coords is None:
                    raise ValueError("pass w or coords (or spatial=False)")
                w = default_weights(as_coords(coords))
            Wm = as_sparse_weights(w, n, row_standardize=True)
            Lg = np.asarray(Wm @ Z)
            data = np.hstack([Z, Lg])
            mask = np.ones((2 * d, 2 * d), bool)
            mask[:, d:] = False           # nothing causes a lag node
        else:
            data, mask = Z, None
        Wfull = notears_linear(data, self.lambda1, self.max_iter, w_threshold=self.w_threshold, mask=mask)
        self.local_ = pd.DataFrame(Wfull[:d, :d], index=names, columns=names)
        if self.spatial:
            self.spillover_ = pd.DataFrame(Wfull[d:, :d], index=[f"W.{nm}" for nm in names], columns=names)
        else:
            self.spillover_ = None
        rows = []
        for i in range(d):
            for j in range(d):
                if Wfull[i, j] != 0:
                    rows.append({"source": names[i], "target": names[j], "weight": Wfull[i, j], "type": "local"})
                if self.spatial and Wfull[d + i, j] != 0:
                    rows.append({"source": f"W.{names[i]}", "target": names[j], "weight": Wfull[d + i, j],
                                 "type": "spillover"})
        self.edges_ = pd.DataFrame(rows, columns=["source", "target", "weight", "type"])
        return self

    def summary(self) -> str:
        return "Spatial NOTEARS edges\n=====================\n" + (
            self.edges_.round(3).to_string(index=False) if len(self.edges_) else "(no edges)")
