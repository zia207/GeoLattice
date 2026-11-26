"""Spatial confounding: diagnostics, spatial bases, and spatial+.

Unmeasured confounders that vary smoothly in space bias effect estimates.
Adding a spatial random effect or smooth term to the outcome model does
not reliably remove that bias: the smooth term and the treatment compete
for the same large-scale spatial variation (Hodges & Reich, 2010; Paciorek,
2010). *spatial+* (Dupont, Wood & Augustin, 2022) first removes the spatially
smooth part of the treatment, then estimates the effect from the remaining,
spatially rough treatment variation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV

from geolattice.spatial_statistics.esda import Moran

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_2d, as_coords, as_sparse_weights, check_lengths, default_weights
from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.inference import standard_error


# ----------------------------------------------------------------------------
# Spatial bases
# ----------------------------------------------------------------------------
def moran_eigenvectors(w, k: Optional[int] = None, n: Optional[int] = None,
                       positive_only: bool = True) -> np.ndarray:
    """Moran eigenvector maps (Griffith; Dray et al.).

    Eigenvectors of ``M S M`` with ``S = (W + W') / 2`` and ``M`` the
    centring matrix, ordered from strongest positive autocorrelation down.
    Dense for n ≤ 3000, Lanczos (``eigsh``) above that.
    """
    S = as_sparse_weights(w, n)
    S = (S + S.T) * 0.5
    n = S.shape[0]
    k = k or max(1, min(50, n // 10))
    if n <= 3000:
        Sd = S.toarray()
        Sd = Sd - Sd.mean(0, keepdims=True)
        Sd = Sd - Sd.mean(1, keepdims=True)
        vals, vecs = np.linalg.eigh(Sd)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
    else:
        ones = np.ones(n) / np.sqrt(n)

        def mv(x):
            x = x - ones * (ones @ x)
            y = S @ x
            return y - ones * (ones @ y)

        op = sparse.linalg.LinearOperator((n, n), matvec=mv, dtype=float)
        vals, vecs = eigsh(op, k=min(k + 5, n - 2), which="LA")
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
    if positive_only:
        keep = vals > 1e-10
        vals, vecs = vals[keep], vecs[:, keep]
    return vecs[:, :k]


def rbf_basis(coords, k: int = 30, bandwidth: Optional[float] = None, random_state: int = 0) -> np.ndarray:
    """Gaussian radial basis functions centred on k-means knots."""
    coords = as_coords(coords)
    k = min(k, len(coords))
    knots = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(coords).cluster_centers_
    if bandwidth is None:
        d, _ = cKDTree(knots).query(knots, k=2)
        bandwidth = 1.5 * float(np.median(d[:, 1]))
    D2 = ((coords[:, None, :] - knots[None]) ** 2).sum(-1)
    B = np.exp(-0.5 * D2 / bandwidth ** 2)
    return B - B.mean(0)


class SpatialBasis:
    """A spatial smooth expressed as basis columns.

    Parameters
    ----------
    kind : {"eigen", "rbf", "coords"}
        ``eigen`` uses Moran eigenvectors of ``w``; ``rbf`` uses radial basis
        functions on ``coords``; ``coords`` uses a quadratic trend surface.
    k : int
        Number of basis columns.
    """

    def __init__(self, kind: str = "eigen", k: int = 20, random_state: int = 0):
        if kind not in ("eigen", "rbf", "coords"):
            raise ValueError("kind must be 'eigen', 'rbf' or 'coords'")
        self.kind, self.k, self.random_state = kind, int(k), random_state

    def build(self, coords, w=None) -> np.ndarray:
        coords = as_coords(coords)
        if self.kind == "eigen":
            w = w if w is not None else default_weights(coords)
            return moran_eigenvectors(w, self.k, len(coords))
        if self.kind == "rbf":
            return rbf_basis(coords, self.k, random_state=self.random_state)
        z = (coords - coords.mean(0)) / coords.std(0)
        x, y = z[:, 0], z[:, 1]
        return np.column_stack([x, y, x * y, x ** 2, y ** 2])


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def _resid(target, X):
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, target)
    return target - m.predict(X)


def diagnose_spatial_confounding(data: SpatialCausalData, w=None, permutations: int = 199,
                                 seed: int = 0) -> pd.DataFrame:
    """Moran's I (via PySpatialStats) on treatment, outcome and their residuals.

    Read the table this way: if the *treatment residual* (what is left
    of treatment after the measured covariates) is strongly spatially
    autocorrelated, any unmeasured, spatially smooth confounder has room
    to bias the estimate. If the *outcome residual* is too, that risk is
    real and standard errors must also account for dependence.
    """
    w = w if w is not None else data.w
    t_res = _resid(data.treatment, data.X)
    y_res = _resid(data.y, data.X)
    rows = []
    for name, v in [("treatment", data.treatment), ("outcome", data.y),
                    ("treatment | X residual", t_res), ("outcome | X residual", y_res)]:
        r = Moran(v, w, permutations=permutations, seed=seed)
        rows.append({"variable": name, "moran_I": r.I, "z": r.z_score, "p_value": r.p_value})
    df = pd.DataFrame(rows).set_index("variable")
    flag_t = df.loc["treatment | X residual", "p_value"] < 0.05
    flag_y = df.loc["outcome | X residual", "p_value"] < 0.05
    df.attrs["interpretation"] = (
        "High risk: residual treatment and outcome both spatially structured; use spatial+ or spatial DML."
        if flag_t and flag_y else
        "Moderate risk: residual treatment is spatially structured." if flag_t else
        "Low risk from spatially smooth confounding (residual treatment not autocorrelated)."
    )
    return df


# ----------------------------------------------------------------------------
# OLS helper with influence functions
# ----------------------------------------------------------------------------
def _ols_effect(y, Z, j: int = 0):
    """OLS coefficient j and its influence contributions."""
    n = len(y)
    Z1 = np.column_stack([np.ones(n), Z])
    j += 1
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    e = y - Z1 @ beta
    Qinv = np.linalg.pinv(Z1.T @ Z1 / n)
    psi = (Z1 * e[:, None]) @ Qinv[j]
    return float(beta[j]), psi


# ----------------------------------------------------------------------------
# Penalised least squares (unpenalised Z, ridge-penalised spatial basis S)
# ----------------------------------------------------------------------------
def _penalized_effect(y, Z, S, lambdas=None, j: int = 0):
    """Fit ``y ~ 1 + Z + S`` with a ridge penalty on S only; lambda by GCV.

    Returns (coefficient j of Z, influence contributions, chosen lambda).
    """
    n = len(y)
    A = np.column_stack([np.ones(n), Z, S])
    p_unpen = 1 + Z.shape[1]
    pen = np.r_[np.zeros(p_unpen), np.ones(S.shape[1])]
    AtA, Aty = A.T @ A, A.T @ y
    lambdas = np.logspace(-3, 4, 30) * n if lambdas is None else np.asarray(lambdas)
    best = None
    for lam in lambdas:
        M = np.linalg.pinv(AtA + lam * np.diag(pen))
        beta = M @ Aty
        e = y - A @ beta
        edf = np.trace(A @ M @ A.T) if n <= 3000 else np.sum(np.einsum("ij,jk,ik->i", A, M, A))
        gcv = n * np.sum(e ** 2) / max(n - edf, 1.0) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lam, beta, e, M)
    _, lam, beta, e, M = best
    psi = n * (A * e[:, None]) @ M[1 + j]
    return float(beta[1 + j]), psi, float(lam)


# ----------------------------------------------------------------------------
# spatial+
# ----------------------------------------------------------------------------
class SpatialPlus:
    """spatial+ (Dupont, Wood & Augustin, 2022).

    1. Regress treatment on a rich spatial basis ``S_t`` (``k_treatment``
       columns, unpenalised); keep the residual ``t~``, which is
       orthogonal to every large- and mid-scale spatial pattern in ``S_t``.
    2. Regress outcome on ``t~`` and ``X`` (unpenalised) plus a spatial
       smooth ``S_y`` whose coefficients are ridge-penalised, with the
       penalty chosen by GCV. The coefficient on ``t~`` is the effect.

    Why the two steps differ: with the *same unpenalised* basis in both
    steps the estimator collapses (Frisch–Waugh–Lovell) to ordinary
    "OLS + spatial smooth". The gain comes from (a) a treatment basis rich
    enough to strip the confounded spatial component from ``t`` and
    (b) a penalised outcome smooth that no longer competes with ``t~``.

    Parameters
    ----------
    basis : SpatialBasis
        Basis family. ``basis.k`` columns are used in the outcome smooth.
    k_treatment : int, optional
        Columns for step 1 (default ``max(3 * basis.k, n // 10)``, capped at
        ``n // 3``). Larger values remove finer-scale confounding at the
        cost of discarding more treatment variation (wider CIs).
    se : {"iid", "conley", "block"}
    cutoff : float, Conley cutoff.
    """

    def __init__(self, basis: Optional[SpatialBasis] = None, k_treatment: Optional[int] = None,
                 se: str = "iid", cutoff: Optional[float] = None):
        self.basis = basis or SpatialBasis("eigen", k=30)
        self.k_treatment = k_treatment
        self.se = se
        self.cutoff = cutoff

    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        if isinstance(data_or_X, SpatialCausalData):
            d = data_or_X
            X, t, y, coords, w = d.X, d.treatment, d.y, d.coords, (w if w is not None else d.w)
        else:
            X, t, y, coords = as_2d(data_or_X), as_1d(treatment), as_1d(y), as_coords(coords)
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        kt = self.k_treatment or min(max(3 * self.basis.k, n // 10), n // 3)
        big = SpatialBasis(self.basis.kind, k=max(kt, self.basis.k), random_state=self.basis.random_state)
        S_all = big.build(coords, w)
        kt = min(kt, S_all.shape[1])
        St, Sy = S_all[:, :kt], S_all[:, : min(self.basis.k, S_all.shape[1])]
        if self.basis.kind == "rbf":  # rbf columns are not nested; build the outcome basis separately
            Sy = self.basis.build(coords, w)
        t_fit = np.column_stack([np.ones(n), St])
        gam, *_ = np.linalg.lstsq(t_fit, t, rcond=None)
        t_tilde = t - t_fit @ gam
        kept = float(np.var(t_tilde) / np.var(t))
        if kept < 0.05:
            import warnings
            warnings.warn(f"only {kept:.1%} of treatment variance survives step 1; "
                          "reduce k_treatment", RuntimeWarning)
        theta, psi, lam = _penalized_effect(y, np.column_stack([t_tilde, X]), Sy)
        se = standard_error(psi, coords, self.se, self.cutoff)
        self.treatment_residual_ = t_tilde
        self.result_ = SpatialCausalResult(
            "spatial+", {"ate": EffectEstimate("ate", theta, se)}, se_method=self.se, n=n,
            basis=f"{self.basis.kind}(k_outcome={Sy.shape[1]}, k_treatment={kt})",
            treatment_variance_kept=round(kept, 3), ridge_lambda=round(lam, 3))
        return self

    def effect(self) -> float:
        return self.result_["ate"].estimate

    def summary(self) -> str:
        return self.result_.summary()


def compare_adjustments(data: SpatialCausalData, basis: Optional[SpatialBasis] = None,
                        se: str = "iid", cutoff: Optional[float] = None,
                        include_dml: bool = True, random_state: int = 0) -> pd.DataFrame:
    """Side-by-side estimates: naive OLS, OLS + spatial smooth, spatial+, spatial DML.

    With simulated data (``data.truth['ate']``) the table also shows bias.
    """
    basis = basis or SpatialBasis("eigen", k=30)
    X, t, y, coords = data.X, data.treatment, data.y, data.coords
    S = basis.build(coords, data.w)
    out = {}
    th, psi = _ols_effect(y, np.column_stack([t, X]))
    out["OLS (X only)"] = (th, standard_error(psi, coords, se, cutoff))
    th, psi = _ols_effect(y, np.column_stack([t, X, S]))
    out["OLS + spatial smooth"] = (th, standard_error(psi, coords, se, cutoff))
    sp = SpatialPlus(basis, se=se, cutoff=cutoff).fit(data)
    out["spatial+"] = (sp.effect(), sp.result_["ate"].se)
    if include_dml:
        from geolattice.spatial_analysis.spatialcausal.ml import SpatialDML
        dml = SpatialDML(model_y="rf", model_t="rf", spatial_features=("coords", "eigen"),
                         se=se, cutoff=cutoff, random_state=random_state).fit(data)
        out["spatial DML"] = (dml.effect(), dml.result_["ate"].se)
    df = pd.DataFrame(out, index=["estimate", "se"]).T
    if "ate" in data.truth:
        df["bias"] = df["estimate"] - data.truth["ate"]
    return df
