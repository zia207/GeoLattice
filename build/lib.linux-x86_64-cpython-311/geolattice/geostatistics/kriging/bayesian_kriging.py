"""
Bayesian kriging.

Classical kriging fits a variogram and then predicts as if its parameters
were known. Bayesian kriging treats the trend coefficients ``beta``, the
variance ``sigma^2`` and the correlation parameters (range ``phi``, relative
nugget ``tau = nugget / sigma^2``, and optionally a Box-Cox ``lambda``) as
unknown, and predicts with the posterior predictive distribution

    p(z0 | z) = integral p(z0 | z, beta, sigma^2, phi, tau) p(beta, sigma^2, phi, tau | z),

so that parameter uncertainty widens the prediction intervals.

Model
-----
``Z(x) = f(x)' beta + S(x) + e(x)`` with ``Cov(Z) = sigma^2 V``,
``V = R(phi) + tau I``: ``R`` is the correlation matrix of the chosen model
(``rho(h) = 1 - gamma(h)`` with unit sill, from
`geolattice.geostatistics.core.variogram_models`), ``tau`` the nugget relative to
the partial sill, and ``f(x)`` a polynomial trend in the coordinates plus
optional covariates.

Classes
-------
`BayesianKriging`
    Correlation parameters on a discrete grid with discrete priors
    (Kitanidis, 1986; Handcock & Stein, 1993; Diggle & Ribeiro, 2007, ch. 7;
    geoR ``krige.bayes``). For every grid point the normal-inverse-gamma
    conjugate prior on ``(beta, sigma^2)`` gives the marginal likelihood and
    a Student-t predictive in closed form; the grid posterior weights mix
    them. The predictive is a finite mixture of Student-t distributions:
    moments, quantiles and exceedance probabilities are exact, and joint
    predictive samples are available.
`ConjugateBayesianKriging`
    The same with fixed range and nugget ratio (a one-point grid):
    uncertainty in ``beta`` and ``sigma^2`` only.
`BayesianTransformedKriging`
    Adds a Box-Cox parameter ``lambda`` to the grid (De Oliveira, Kedem &
    Short, 1997), using the geometric-mean normalised transform so marginal
    likelihoods are comparable across ``lambda``; predictions are
    back-transformed by Monte Carlo.
`BayesianGLMKriging`
    Spatial generalized linear mixed model for counts (Poisson, log link,
    optional exposure) or binomial data (logit link) (Diggle, Tawn & Moyeed,
    1998), sampled by MCMC: a whitened latent Gaussian field updated by MALA
    (Christensen, 2004), random-walk Metropolis for ``beta`` and
    ``log sigma^2``, and a discrete grid for the range.

Priors
------
``beta_prior``: ``"flat"`` (default) or ``("normal", mean, cov)`` (``cov``
scaled by ``sigma^2``). ``sigma2_prior``: ``"reciprocal"`` (``1/sigma^2``,
default) or ``("scaled_inv_chi2", df, scale2)``. Discrete priors for the
range / nugget ratio / lambda: ``"uniform"``, ``"reciprocal"``,
``"squared_reciprocal"``, or an array of weights.

Coordinates are used as given (isotropic); transform them first for
geometric anisotropy (`geolattice.geostatistics.transform_aniso`).
"""

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import gammaln

from ..core.variogram_models import MODEL_FUNCS

__all__ = [
    "BayesianKriging",
    "ConjugateBayesianKriging",
    "BayesianTransformedKriging",
    "BayesianGLMKriging",
    "bayesian_kriging",
    "correlation_function",
]

_JITTER = 1e-10


# --------------------------------------------------------------- helpers


def correlation_function(h, model: str = "exponential", range_: float = 1.0, nu: float = 1.5,
                         alpha: float = 1.0) -> np.ndarray:
    """
    Correlation ``rho(h) = 1 - gamma(h)`` of a unit-sill, zero-nugget model
    from `MODEL_FUNCS` (``rho(0) = 1``).

    Parameters
    ----------
    h : array-like
        Distances.
    model : str
        'spherical', 'exponential', 'gaussian', 'matern', 'cubic' or 'stable'.
    range_ : float
        Range parameter, in the `Variogram` convention of each model.
    nu : float, default 1.5
        Matern smoothness (``model='matern'`` only).
    alpha : float, default 1.0
        Stable exponent in (0, 2] (``model='stable'`` only): 1 is the
        exponential, 2 the Gaussian model.
    """
    h = np.asarray(h, dtype=float)
    if model not in MODEL_FUNCS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODEL_FUNCS)}")
    if model == "matern":
        g = MODEL_FUNCS[model](h, 0.0, 1.0, range_, nu)
    elif model == "stable":
        if not 0.0 < alpha <= 2.0:
            raise ValueError("stable exponent alpha must be in (0, 2]")
        g = MODEL_FUNCS[model](h, 0.0, 1.0, range_, alpha)
    else:
        g = MODEL_FUNCS[model](h, 0.0, 1.0, range_)
    rho = 1.0 - np.asarray(g, dtype=float)
    return np.where(h <= 0.0, 1.0, rho)


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.clip(d2, 0.0, None))


def _discrete_prior(grid: np.ndarray, prior) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if isinstance(prior, str):
        if prior == "uniform":
            w = np.ones_like(grid)
        elif prior in ("reciprocal", "squared_reciprocal"):
            if np.any(grid <= 0):
                raise ValueError(f"a {prior} prior needs a strictly positive grid")
            w = 1.0 / grid if prior == "reciprocal" else 1.0 / grid ** 2
        else:
            raise ValueError("prior must be 'uniform', 'reciprocal', 'squared_reciprocal' or an array")
    else:
        w = np.asarray(prior, dtype=float)
        if w.shape != grid.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("prior weights must be non-negative, one per grid value, not all zero")
    return w / w.sum()


def _parse_beta_prior(beta_prior, p: int):
    if isinstance(beta_prior, str):
        if beta_prior != "flat":
            raise ValueError("beta_prior must be 'flat' or ('normal', mean, cov)")
        return None
    kind, mean, cov = beta_prior
    if kind != "normal":
        raise ValueError("beta_prior must be 'flat' or ('normal', mean, cov)")
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if mean.shape != (p,) or cov.shape != (p, p):
        raise ValueError(f"normal beta prior needs mean ({p},) and cov ({p}, {p})")
    return mean, cov


def _parse_sigma2_prior(sigma2_prior):
    if isinstance(sigma2_prior, str):
        if sigma2_prior != "reciprocal":
            raise ValueError("sigma2_prior must be 'reciprocal' or ('scaled_inv_chi2', df, scale2)")
        return 0.0, 0.0
    kind, df, scale2 = sigma2_prior
    if kind != "scaled_inv_chi2" or df <= 0 or scale2 <= 0:
        raise ValueError("sigma2_prior must be 'reciprocal' or ('scaled_inv_chi2', df > 0, scale2 > 0)")
    return float(df), float(scale2)


class _Trend:
    """Polynomial trend in (centred, scaled) coordinates plus optional covariates."""

    def __init__(self, degree: int):
        if degree not in (0, 1, 2):
            raise ValueError("trend degree must be 0 (constant), 1 (linear) or 2 (quadratic)")
        self.degree = degree

    def fit(self, xy: np.ndarray, covariates: Optional[np.ndarray]) -> np.ndarray:
        self.centre = xy.mean(0)
        self.scale = xy.std(0)
        self.scale[self.scale == 0] = 1.0
        self.n_cov = 0 if covariates is None else np.atleast_2d(np.asarray(covariates)).reshape(len(xy), -1).shape[1]
        return self.transform(xy, covariates)

    def transform(self, xy: np.ndarray, covariates: Optional[np.ndarray]) -> np.ndarray:
        u = (xy - self.centre) / self.scale
        cols = [np.ones(len(xy))]
        if self.degree >= 1:
            cols += [u[:, 0], u[:, 1]]
        if self.degree == 2:
            cols += [u[:, 0] ** 2, u[:, 0] * u[:, 1], u[:, 1] ** 2]
        F = np.column_stack(cols)
        if self.n_cov:
            if covariates is None:
                raise ValueError("the model was fitted with covariates; pass them for prediction")
            c = np.asarray(covariates, dtype=float).reshape(len(xy), -1)
            if c.shape[1] != self.n_cov:
                raise ValueError(f"expected {self.n_cov} covariate column(s)")
            F = np.column_stack([F, c])
        elif covariates is not None:
            raise ValueError("the model was fitted without covariates")
        return F

    def names(self):
        base = ["intercept", "x", "y", "x2", "xy", "y2"][: {0: 1, 1: 3, 2: 6}[self.degree]]
        return base + [f"cov{i + 1}" for i in range(self.n_cov)]


# --------------------------------------------------------- BayesianKriging


class BayesianKriging:
    """
    Bayesian kriging with discrete priors on the range and the relative
    nugget and a conjugate normal-inverse-gamma prior on the trend and
    variance.

    Parameters
    ----------
    model : str, default 'exponential'
        Correlation model: 'exponential', 'spherical', 'gaussian', 'matern',
        'cubic' or 'stable'. The range follows the `Variogram` convention of
        the same model (e.g. ``exp(-h / range)`` for 'exponential').
    nu : float, default 1.5
        Shape parameter: Matern smoothness (``model='matern'``) or the
        stable exponent alpha in (0, 2] (``model='stable'``).
    trend : int, default 0
        Polynomial trend degree in the coordinates: 0 constant (ordinary
        kriging-like), 1 linear, 2 quadratic (universal kriging-like).
    range_grid : array-like, optional
        Candidate ranges. Default: 30 values log-spaced from 2% to 100% of
        the largest inter-point distance.
    nugget_grid : array-like, default (0, 0.1, 0.25, 0.5, 1.0)
        Candidate relative nuggets ``tau = nugget / partial sill``.
    range_prior, nugget_prior : str or array, default 'uniform'
        Discrete priors over the grids.
    beta_prior : 'flat' or ('normal', mean, cov), default 'flat'
    sigma2_prior : 'reciprocal' or ('scaled_inv_chi2', df, scale2), default 'reciprocal'

    Attributes
    ----------
    posterior_ : pandas.DataFrame
        One row per grid point: range, nugget ratio, prior, log marginal
        likelihood, posterior weight, posterior mean of sigma^2, degrees of
        freedom, posterior mean of beta.
    beta_names_ : list of str

    Examples
    --------
    >>> bk = BayesianKriging("exponential", trend=0).fit(xy, z)
    >>> mean, sd = bk.predict(xy_new, return_std=True)
    >>> lo, hi = bk.predict_quantiles(xy_new, [0.05, 0.95]).T
    """

    def __init__(self, model: str = "exponential", nu: float = 1.5, trend: int = 0,
                 range_grid: Optional[Sequence[float]] = None,
                 nugget_grid: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
                 range_prior="uniform", nugget_prior="uniform",
                 beta_prior="flat", sigma2_prior="reciprocal"):
        if model not in MODEL_FUNCS:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODEL_FUNCS)}")
        self.model = model
        self.nu = float(nu)
        self.trend = int(trend)
        self.range_grid = None if range_grid is None else np.asarray(range_grid, dtype=float)
        self.nugget_grid = np.asarray(nugget_grid, dtype=float)
        if self.range_grid is not None and np.any(self.range_grid <= 0):
            raise ValueError("range_grid values must be positive")
        if np.any(self.nugget_grid < 0):
            raise ValueError("nugget_grid values must be >= 0")
        self.range_prior = range_prior
        self.nugget_prior = nugget_prior
        self.beta_prior = beta_prior
        self.sigma2_prior = sigma2_prior

    # ----------------------------------------------------------------- fit

    @classmethod
    def from_variogram(cls, variogram, trend: int = 0, **kwargs) -> "ConjugateBayesianKriging":
        """
        Conjugate Bayesian kriging with the range and nugget ratio of a
        fitted `Variogram` (its ``fitted_params`` are ``[nugget, sill, range(, nu)]``).
        """
        params = getattr(variogram, "fitted_params", None)
        if params is None:
            raise RuntimeError("the variogram is not fitted")
        nugget, sill, rng = float(params[0]), float(params[1]), float(params[2])
        nu = float(params[3]) if variogram.model in ("matern", "stable") and len(params) > 3 else 1.5
        return ConjugateBayesianKriging(variogram.model, range_=rng, nugget_ratio=nugget / max(sill, 1e-300),
                                        nu=nu, trend=trend, **kwargs)

    def _check_xy(self, coords):
        xy = np.asarray(coords, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        return xy

    def _components(self, z: np.ndarray, dist: np.ndarray, F: np.ndarray, beta_spec, s2_spec):
        """Closed-form posterior for every (range, nugget ratio) grid point."""
        n, p = F.shape
        n0, s0 = s2_spec
        if beta_spec is None and n + n0 <= p:
            raise ValueError("not enough data for the trend: need n > number of trend terms")
        comps = []
        if beta_spec is not None:
            m, Vb = beta_spec
            cVb = cho_factor(Vb + _JITTER * np.eye(p), lower=True)
            Vb_inv = cho_solve(cVb, np.eye(p))
        for rng in self._range_values:
            R = correlation_function(dist, self.model, rng, self.nu, self.nu)
            for tau in self.nugget_grid:
                V = R + (tau + _JITTER) * np.eye(n)
                try:
                    cV = cho_factor(V, lower=True)
                except np.linalg.LinAlgError:
                    continue
                logdetV = 2.0 * np.log(np.diag(cV[0])).sum()
                Vi_z = cho_solve(cV, z)
                Vi_F = cho_solve(cV, F)
                A = F.T @ Vi_F
                if beta_spec is None:
                    cA = cho_factor(A + _JITTER * np.eye(p), lower=True)
                    beta = cho_solve(cA, F.T @ Vi_z)
                    resid = z - F @ beta
                    Q = float(resid @ cho_solve(cV, resid))
                    dof = n0 + n - p
                    ss = n0 * s0 + Q
                    logdetA = 2.0 * np.log(np.diag(cA[0])).sum()
                    logml = -0.5 * logdetV - 0.5 * logdetA - 0.5 * dof * np.log(ss)
                    Vbeta = cho_solve(cA, np.eye(p))
                else:
                    P = Vb_inv + A
                    cP = cho_factor(P, lower=True)
                    Vbeta = cho_solve(cP, np.eye(p))
                    beta = Vbeta @ (Vb_inv @ m + F.T @ Vi_z)
                    r0 = z - F @ m
                    # (z - Fm)' W^-1 (z - Fm), W = V + F Vb F' (Woodbury)
                    t = F.T @ cho_solve(cV, r0)
                    Q = float(r0 @ cho_solve(cV, r0) - t @ cho_solve(cP, t))
                    dof = n0 + n
                    ss = n0 * s0 + Q
                    logdetP = 2.0 * np.log(np.diag(cP[0])).sum()
                    logml = -0.5 * (logdetV + logdetP) - 0.5 * dof * np.log(ss)
                comps.append({"range": float(rng), "nugget_ratio": float(tau), "cV": cV, "Vi_F": Vi_F,
                              "beta": beta, "Vbeta": Vbeta, "dof": float(dof), "s2": ss / dof,
                              "logml": float(logml)})
        if not comps:
            raise np.linalg.LinAlgError("no grid point gave a positive-definite covariance matrix")
        return comps

    def fit(self, coords, values, covariates=None) -> "BayesianKriging":
        """Compute the grid posterior for data `values` at `coords` (n, 2)."""
        xy = self._check_xy(coords)
        z = np.asarray(values, dtype=float).ravel()
        if len(z) != len(xy):
            raise ValueError("coords and values must have the same length")
        if not np.all(np.isfinite(z)):
            raise ValueError("values must be finite")
        self._trend = _Trend(self.trend)
        F = self._trend.fit(xy, covariates)
        dist = _pairwise(xy, xy)
        dmax = dist.max() if len(xy) > 1 else 1.0
        self._range_values = (np.geomspace(0.02 * dmax, dmax, 30) if self.range_grid is None
                              else self.range_grid)
        beta_spec = _parse_beta_prior(self.beta_prior, F.shape[1])
        s2_spec = _parse_sigma2_prior(self.sigma2_prior)
        comps = self._components(z, dist, F, beta_spec, s2_spec)
        self._finalize(comps, xy, z, F)
        return self

    def _finalize(self, comps, xy, z, F, extra_logprior=None):
        pr = _discrete_prior(self._range_values, self.range_prior)
        pn = _discrete_prior(self.nugget_grid, self.nugget_prior)
        r_index = {float(r): i for i, r in enumerate(self._range_values)}
        n_index = {float(t): i for i, t in enumerate(self.nugget_grid)}
        logprior = np.array([np.log(max(pr[r_index[c["range"]]] * pn[n_index[c["nugget_ratio"]]], 1e-300))
                             for c in comps])
        if extra_logprior is not None:
            logprior = logprior + extra_logprior
        lp = np.array([c["logml"] for c in comps]) + logprior
        w = np.exp(lp - lp.max())
        w /= w.sum()
        self._comps, self._weights = comps, w
        self._xy, self._z, self._F = xy, z, F
        self.beta_names_ = self._trend.names()
        rows = []
        for c, wk, lpk in zip(comps, w, logprior):
            row = {"range": c["range"], "nugget_ratio": c["nugget_ratio"], "log_prior": lpk,
                   "log_marginal": c["logml"], "weight": wk, "dof": c["dof"],
                   "sigma2_mean": c["s2"] * c["dof"] / (c["dof"] - 2) if c["dof"] > 2 else np.inf}
            for name, b in zip(self.beta_names_, c["beta"]):
                row[f"beta_{name}"] = b
            if "lam" in c:
                row["lambda"] = c["lam"]
            rows.append(row)
        self.posterior_ = pd.DataFrame(rows)
        return self

    # ------------------------------------------------------------ predict

    def _check_fitted(self):
        if not hasattr(self, "_comps"):
            raise RuntimeError("call fit() first")

    def _component_predictions(self, coords, covariates=None, signal: bool = True):
        """Per-component Student-t location, scale and dof at the prediction points."""
        self._check_fitted()
        xp = self._check_xy(coords)
        f0 = self._trend.transform(xp, covariates)
        d0 = _pairwise(xp, self._xy)
        locs, scales, dofs = [], [], []
        corr_cache = {}
        for c in self._comps:
            key = c["range"]
            if key not in corr_cache:
                corr_cache[key] = correlation_function(d0, self.model, key, self.nu, self.nu)
            r = corr_cache[key]
            resid = self._z - self._F @ c["beta"]
            mean = f0 @ c["beta"] + r @ cho_solve(c["cV"], resid)
            Vi_rT = cho_solve(c["cV"], r.T)
            g = f0 - r @ c["Vi_F"]
            v00 = 1.0 if signal else 1.0 + c["nugget_ratio"]
            var_unit = v00 - np.einsum("ij,ji->i", r, Vi_rT) + np.einsum("ij,jk,ik->i", g, c["Vbeta"], g)
            locs.append(mean)
            scales.append(np.sqrt(np.clip(c["s2"] * var_unit, 0.0, None)))
            dofs.append(c["dof"])
        return np.array(locs), np.array(scales), np.array(dofs)

    def predict(self, coords, covariates=None, return_std: bool = False, signal: bool = True):
        """
        Posterior predictive mean (and standard deviation) at `coords`.

        `signal=True` predicts the noise-free process ``f'beta + S(x)``;
        ``False`` a new observation including the nugget.
        """
        loc, scale, dof = self._component_predictions(coords, covariates, signal)
        w = self._weights[:, None]
        mean = (w * loc).sum(0)
        if not return_std:
            return mean
        if np.any(dof <= 2):
            warnings.warn("posterior predictive variance is infinite for dof <= 2", stacklevel=2)
        tvar = np.where(dof[:, None] > 2, scale ** 2 * dof[:, None] / np.maximum(dof[:, None] - 2, 1e-12), np.inf)
        var = (w * (tvar + loc ** 2)).sum(0) - mean ** 2
        return mean, np.sqrt(np.clip(var, 0.0, None))

    def _mixture_cdf(self, x, loc, scale, dof):
        z = (x[None, :] - loc) / np.maximum(scale, 1e-300)
        cdf = stats.t.cdf(z, df=dof[:, None])
        cdf = np.where(scale > 0, cdf, (x[None, :] >= loc).astype(float))
        return (self._weights[:, None] * cdf).sum(0)

    def predict_quantiles(self, coords, quantiles: Sequence[float] = (0.05, 0.5, 0.95), covariates=None,
                          signal: bool = True) -> np.ndarray:
        """Quantiles of the mixture predictive, shape ``(n_points, n_quantiles)`` (bisection)."""
        loc, scale, dof = self._component_predictions(coords, covariates, signal)
        qs = np.atleast_1d(np.asarray(quantiles, dtype=float))
        if np.any((qs <= 0) | (qs >= 1)):
            raise ValueError("quantiles must be in (0, 1)")
        out = np.empty((loc.shape[1], len(qs)))
        for j, q in enumerate(qs):
            tq = stats.t.ppf(q, df=dof)[:, None]
            cand = loc + tq * scale
            lo, hi = cand.min(0) - 1e-9, cand.max(0) + 1e-9
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                below = self._mixture_cdf(mid, loc, scale, dof) < q
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            out[:, j] = 0.5 * (lo + hi)
        return out

    def predict_proba(self, coords, threshold: float, covariates=None, greater: bool = True,
                      signal: bool = True) -> np.ndarray:
        """``P(Z(x) > threshold | data)`` (or ``<=`` with ``greater=False``)."""
        loc, scale, dof = self._component_predictions(coords, covariates, signal)
        cdf = self._mixture_cdf(np.full(loc.shape[1], float(threshold)), loc, scale, dof)
        return 1.0 - cdf if greater else cdf

    def sample_parameters(self, n: int = 1000, seed=None) -> pd.DataFrame:
        """Posterior draws of (range, nugget_ratio, sigma2, beta...)."""
        self._check_fitted()
        rng = np.random.default_rng(seed)
        k = rng.choice(len(self._comps), size=n, p=self._weights)
        rows = []
        for i in k:
            c = self._comps[i]
            s2 = c["s2"] * c["dof"] / rng.chisquare(c["dof"])
            beta = rng.multivariate_normal(c["beta"], s2 * c["Vbeta"])
            row = {"range": c["range"], "nugget_ratio": c["nugget_ratio"], "sigma2": s2,
                   "nugget": c["nugget_ratio"] * s2}
            row.update({f"beta_{nm}": b for nm, b in zip(self.beta_names_, beta)})
            if "lam" in c:
                row["lambda"] = c["lam"]
            rows.append(row)
        return pd.DataFrame(rows)

    def predictive_samples(self, coords, n_samples: int = 1000, covariates=None, joint: bool = False,
                           signal: bool = True, seed=None) -> np.ndarray:
        """
        Draws from the posterior predictive, shape ``(n_samples, n_points)``.

        Each draw takes a grid point by its posterior weight, then ``sigma^2``,
        then ``beta``, then the conditional Gaussian field. `joint=True` draws
        the points jointly (uses the ``n_points x n_points`` covariance; keep
        ``n_points`` moderate), otherwise each point marginally.
        """
        self._check_fitted()
        xp = self._check_xy(coords)
        f0 = self._trend.transform(xp, covariates)
        d0 = _pairwise(xp, self._xy)
        d00 = _pairwise(xp, xp) if joint else None
        rng = np.random.default_rng(seed)
        k_all = rng.choice(len(self._comps), size=n_samples, p=self._weights)
        out = np.empty((n_samples, len(xp)))
        for k in np.unique(k_all):
            idx = np.flatnonzero(k_all == k)
            c = self._comps[k]
            r = correlation_function(d0, self.model, c["range"], self.nu, self.nu)
            Vi_rT = cho_solve(c["cV"], r.T)
            if joint:
                C = correlation_function(d00, self.model, c["range"], self.nu, self.nu) - r @ Vi_rT
                if not signal:
                    C = C + c["nugget_ratio"] * np.eye(len(xp))
                Lc = np.linalg.cholesky(C + 1e-9 * np.eye(len(xp)))
            else:
                v = (1.0 if signal else 1.0 + c["nugget_ratio"]) - np.einsum("ij,ji->i", r, Vi_rT)
                sd_unit = np.sqrt(np.clip(v, 0.0, None))
            for i in idx:
                s2 = c["s2"] * c["dof"] / rng.chisquare(c["dof"])
                beta = rng.multivariate_normal(c["beta"], s2 * c["Vbeta"])
                mean = f0 @ beta + r @ cho_solve(c["cV"], self._z - self._F @ beta)
                eps = rng.standard_normal(len(xp))
                out[i] = mean + np.sqrt(s2) * (Lc @ eps if joint else sd_unit * eps)
        return out


class ConjugateBayesianKriging(BayesianKriging):
    """
    Bayesian kriging with known correlation (fixed range and nugget ratio):
    conjugate normal-inverse-gamma posterior for the trend and variance, and a
    Student-t predictive (Kitanidis, 1986). See `BayesianKriging` for the
    other parameters; ``BayesianKriging.from_variogram`` builds one from a
    fitted `Variogram`.
    """

    def __init__(self, model: str = "exponential", range_: float = 1.0, nugget_ratio: float = 0.0,
                 nu: float = 1.5, trend: int = 0, beta_prior="flat", sigma2_prior="reciprocal"):
        super().__init__(model=model, nu=nu, trend=trend, range_grid=[range_], nugget_grid=[nugget_ratio],
                         beta_prior=beta_prior, sigma2_prior=sigma2_prior)


# ------------------------------------------------- transformed Gaussian


def _boxcox(z, lam):
    return np.log(z) if lam == 0 else (z ** lam - 1.0) / lam


def _boxcox_inv(t, lam):
    if lam == 0:
        return np.exp(t)
    base = lam * t + 1.0
    return np.where(base > 0, np.abs(base) ** (1.0 / lam), 0.0 if lam > 0 else np.inf)


class BayesianTransformedKriging(BayesianKriging):
    """
    Bayesian transformed Gaussian kriging (De Oliveira, Kedem & Short, 1997)
    for positive, skewed data: the Box-Cox parameter ``lambda`` joins the
    range and nugget-ratio grid.

    The data are transformed with the geometric-mean normalised Box-Cox
    transform ``w = t_lambda(z) / g^(lambda - 1)`` (``g`` the geometric mean),
    whose Jacobian is 1, so marginal likelihoods compare directly across
    ``lambda``. Predictions are posterior predictive draws back-transformed
    to the data scale; draws whose transform is outside the Box-Cox domain
    map to 0 (``lambda > 0``) or to +inf (``lambda < 0``, with a warning:
    a transformed Gaussian with ``lambda < 0`` has no finite mean, so use
    medians and quantiles).

    Parameters
    ----------
    lambda_grid : array-like, default (-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1)
    lambda_prior : str or array, default 'uniform'
    Other parameters as in `BayesianKriging` (``signal`` predictions are not
    defined on the back-transformed scale; observations are predicted).
    """

    def __init__(self, model: str = "exponential", nu: float = 1.5, trend: int = 0,
                 lambda_grid: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
                 lambda_prior="uniform", **kwargs):
        super().__init__(model=model, nu=nu, trend=trend, **kwargs)
        self.lambda_grid = np.asarray(lambda_grid, dtype=float)
        self.lambda_prior = lambda_prior

    def fit(self, coords, values, covariates=None) -> "BayesianTransformedKriging":
        xy = self._check_xy(coords)
        z = np.asarray(values, dtype=float).ravel()
        if np.any(z <= 0):
            raise ValueError("BayesianTransformedKriging needs strictly positive values")
        self._trend = _Trend(self.trend)
        F = self._trend.fit(xy, covariates)
        dist = _pairwise(xy, xy)
        dmax = dist.max()
        self._range_values = (np.geomspace(0.02 * dmax, dmax, 30) if self.range_grid is None
                              else self.range_grid)
        beta_spec = _parse_beta_prior(self.beta_prior, F.shape[1])
        s2_spec = _parse_sigma2_prior(self.sigma2_prior)
        self._gmean = float(np.exp(np.log(z).mean()))
        pl = _discrete_prior(self.lambda_grid, self.lambda_prior)
        comps, extra = [], []
        self._w_by_lambda = {}
        for lam, plam in zip(self.lambda_grid, pl):
            w = _boxcox(z, lam) / self._gmean ** (lam - 1.0)
            cl = self._components(w, dist, F, beta_spec, s2_spec)
            for c in cl:
                c["lam"] = float(lam)
                c["zt"] = w
            comps += cl
            extra += [np.log(max(plam, 1e-300))] * len(cl)
            self._w_by_lambda[float(lam)] = w
        self._finalize(comps, xy, z, F, extra_logprior=np.array(extra))
        lam_post = self.posterior_.groupby("lambda")["weight"].sum()
        self.lambda_posterior_ = lam_post
        return self

    def predictive_samples(self, coords, n_samples: int = 2000, covariates=None, joint: bool = False,
                           signal: bool = False, seed=None) -> np.ndarray:
        """Back-transformed posterior predictive draws, shape ``(n_samples, n_points)``."""
        self._check_fitted()
        xp = self._check_xy(coords)
        f0 = self._trend.transform(xp, covariates)
        d0 = _pairwise(xp, self._xy)
        d00 = _pairwise(xp, xp) if joint else None
        rng = np.random.default_rng(seed)
        k_all = rng.choice(len(self._comps), size=n_samples, p=self._weights)
        out = np.empty((n_samples, len(xp)))
        for k in np.unique(k_all):
            idx = np.flatnonzero(k_all == k)
            c = self._comps[k]
            r = correlation_function(d0, self.model, c["range"], self.nu, self.nu)
            Vi_rT = cho_solve(c["cV"], r.T)
            if joint:
                C = correlation_function(d00, self.model, c["range"], self.nu, self.nu) - r @ Vi_rT
                C = C + c["nugget_ratio"] * np.eye(len(xp))
                Lc = np.linalg.cholesky(C + 1e-9 * np.eye(len(xp)))
            else:
                v = 1.0 + c["nugget_ratio"] - np.einsum("ij,ji->i", r, Vi_rT)
                sd_unit = np.sqrt(np.clip(v, 0.0, None))
            for i in idx:
                s2 = c["s2"] * c["dof"] / rng.chisquare(c["dof"])
                beta = rng.multivariate_normal(c["beta"], s2 * c["Vbeta"])
                mean = f0 @ beta + r @ cho_solve(c["cV"], c["zt"] - self._F @ beta)
                eps = rng.standard_normal(len(xp))
                wdraw = mean + np.sqrt(s2) * (Lc @ eps if joint else sd_unit * eps)
                out[i] = _boxcox_inv(wdraw * self._gmean ** (c["lam"] - 1.0), c["lam"])
        bad = ~np.isfinite(out)
        if bad.any():
            warnings.warn(f"{bad.mean():.3%} of the predictive draws fall outside the Box-Cox domain for "
                          "lambda < 0 and are infinite (the predictive mean does not exist); "
                          "summarize with medians or quantiles", stacklevel=2)
        return out

    def predict(self, coords, covariates=None, n_samples: int = 2000, seed=None) -> pd.DataFrame:
        """Predictive mean, median, sd and 5% / 95% quantiles on the data scale (Monte Carlo)."""
        s = self.predictive_samples(coords, n_samples, covariates, seed=seed)
        return pd.DataFrame({"mean": s.mean(0), "median": np.median(s, 0), "sd": s.std(0),
                             "q05": np.quantile(s, 0.05, 0), "q95": np.quantile(s, 0.95, 0)})

    def predict_quantiles(self, coords, quantiles=(0.05, 0.5, 0.95), covariates=None, n_samples: int = 2000,
                          seed=None) -> np.ndarray:
        s = self.predictive_samples(coords, n_samples, covariates, seed=seed)
        return np.quantile(s, np.atleast_1d(quantiles), axis=0).T

    def predict_proba(self, coords, threshold: float, covariates=None, greater: bool = True,
                      n_samples: int = 2000, seed=None) -> np.ndarray:
        s = self.predictive_samples(coords, n_samples, covariates, seed=seed)
        return (s > threshold).mean(0) if greater else (s <= threshold).mean(0)


# ------------------------------------------------------ GLM (MCMC)


class BayesianGLMKriging:
    """
    Bayesian generalized linear kriging (spatial GLMM) by MCMC.

    ``Y_i | S ~ Poisson(E_i exp(eta_i))`` or ``Binomial(n_i, logistic(eta_i))``
    with ``eta = F beta + S``, ``S ~ N(0, sigma^2 (R(phi) + tau I))``.

    Sampler (Christensen, 2004): the latent field is whitened,
    ``S = sigma L(phi) Gamma`` with ``Gamma ~ N(0, I)``, and ``Gamma`` is
    updated by MALA (adaptive step size, target acceptance 0.574);
    ``beta`` by random-walk Metropolis with a ``(F'F)^-1``-shaped proposal;
    ``log sigma^2`` by random-walk Metropolis; the range ``phi`` on its
    discrete grid by a move to a neighbouring grid value. Step sizes adapt
    during burn-in only. Priors: ``beta ~ N(0, beta_prior_sd^2 I)``,
    ``sigma^2 ~ scaled-inv-chi2(df, scale2)``, discrete prior on ``phi``.

    With ``interweave=True`` (default) every iteration also updates the
    parameters in the centred parametrization with the linear predictor
    ``eta`` held fixed (ancillarity-sufficiency interweaving, Yu & Meng,
    2011): ``beta | eta`` is Gaussian, ``sigma^2 | S`` scaled-inv-chi2, and
    ``phi | S, sigma^2`` is drawn exactly from its discrete posterior every
    `range_gibbs_every` iterations. These moves leave the likelihood
    unchanged and remove the slow mixing of the intercept, ``sigma^2`` and
    the range that the whitened parametrization alone suffers from.

    Parameters
    ----------
    family : {'poisson', 'binomial'}
    model, nu, trend : as in `BayesianKriging`
    range_grid : array-like, optional
        Default 20 values log-spaced from 2% to 100% of the largest distance.
    range_prior : str or array, default 'uniform'
    nugget_ratio : float, default 0.0
        Fixed relative nugget of the latent field (extra-Poisson /
        extra-binomial unstructured variation).
    sigma2_prior : (df, scale2), default (1.0, 0.5)
    beta_prior_sd : float, default 10.0
    interweave : bool, default True
        Add the centred Gibbs updates described above.
    range_gibbs_every : int, default 5
        Iterations between exact discrete updates of the range (each costs
        one triangular solve per grid value).

    Attributes
    ----------
    trace_ : pandas.DataFrame
        Kept samples of beta, sigma2, range and log-likelihood.
    acceptance_ : dict
        Acceptance rates of the four updates after burn-in.
    """

    def __init__(self, family: str = "poisson", model: str = "exponential", nu: float = 1.5, trend: int = 0,
                 range_grid: Optional[Sequence[float]] = None, range_prior="uniform",
                 nugget_ratio: float = 0.0, sigma2_prior=(1.0, 0.5), beta_prior_sd: float = 10.0,
                 interweave: bool = True, range_gibbs_every: int = 5):
        if family not in ("poisson", "binomial"):
            raise ValueError("family must be 'poisson' or 'binomial'")
        if model not in MODEL_FUNCS:
            raise ValueError(f"unknown model {model!r}")
        self.family = family
        self.model = model
        self.nu = float(nu)
        self.trend = int(trend)
        self.range_grid = None if range_grid is None else np.asarray(range_grid, dtype=float)
        self.range_prior = range_prior
        self.nugget_ratio = float(nugget_ratio)
        self.sigma2_prior = tuple(float(v) for v in sigma2_prior)
        self.beta_prior_sd = float(beta_prior_sd)
        if self.sigma2_prior[0] <= 0 or self.sigma2_prior[1] <= 0:
            raise ValueError("sigma2_prior must be (df > 0, scale2 > 0)")
        self.interweave = bool(interweave)
        if range_gibbs_every < 1:
            raise ValueError("range_gibbs_every must be >= 1")
        self.range_gibbs_every = int(range_gibbs_every)

    def _loglik(self, eta):
        y, a = self._y, self._aux
        if self.family == "poisson":
            return float(np.sum(y * eta - a * np.exp(eta)))
        return float(np.sum(y * eta - a * np.logaddexp(0.0, eta)))

    def _grad_eta(self, eta):
        if self.family == "poisson":
            return self._y - self._aux * np.exp(eta)
        return self._y - self._aux / (1.0 + np.exp(-eta))

    def fit(self, coords, values, trials_or_exposure=None, covariates=None, n_iter: int = 6000,
            burn: int = 2000, thin: int = 4, seed=None, verbose: bool = False) -> "BayesianGLMKriging":
        """
        Run the MCMC. `trials_or_exposure` is the number of trials (binomial,
        default 1) or the exposure / expected count ``E_i`` (Poisson, default 1).
        """
        xy = np.asarray(coords, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        y = np.asarray(values, dtype=float).ravel()
        n = len(y)
        if len(xy) != n:
            raise ValueError("coords and values must have the same length")
        aux = np.ones(n) if trials_or_exposure is None else np.asarray(trials_or_exposure, dtype=float).ravel()
        if np.any(y < 0) or np.any(aux <= 0):
            raise ValueError("counts must be >= 0 and trials / exposure > 0")
        if self.family == "binomial" and np.any(y > aux):
            raise ValueError("binomial successes cannot exceed the number of trials")
        if burn >= n_iter:
            raise ValueError("burn must be smaller than n_iter")
        self._y, self._aux, self._xy = y, aux, xy
        self._trend = _Trend(self.trend)
        F = self._trend.fit(xy, covariates)
        p = F.shape[1]
        dist = _pairwise(xy, xy)
        dmax = dist.max()
        self._range_values = (np.geomspace(0.02 * dmax, dmax, 20) if self.range_grid is None
                              else self.range_grid)
        log_pr = np.log(np.maximum(_discrete_prior(self._range_values, self.range_prior), 1e-300))
        self._L = [np.linalg.cholesky(correlation_function(dist, self.model, r, self.nu, self.nu)
                                      + (self.nugget_ratio + 1e-8) * np.eye(n)) for r in self._range_values]
        df0, sc0 = self.sigma2_prior
        J = len(self._range_values)
        logdetL = np.array([np.log(np.diag(L)).sum() for L in self._L])
        A_all = [solve_triangular(L, F, lower=True) for L in self._L]       # L^-1 F per range
        rng = np.random.default_rng(seed)

        # initial values: GLM-ish start
        if self.family == "poisson":
            emp = np.log((y + 0.5) / aux)
        else:
            emp = np.log((y + 0.5) / (aux - y + 0.5))
        beta = np.linalg.lstsq(F, emp, rcond=None)[0]
        s2 = max(float(np.var(emp - F @ beta)), 0.1)
        j = len(self._range_values) // 2
        gam = np.zeros(n)
        FtF_inv = np.linalg.inv(F.T @ F + 1e-9 * np.eye(p))
        Lb = np.linalg.cholesky(FtF_inv)

        def eta_of(b, g, s2_, jj):
            return F @ b + np.sqrt(s2_) * (self._L[jj] @ g)

        def log_post_gamma(g, eta):
            return self._loglik(eta) - 0.5 * g @ g

        def log_prior_s2(s2_):
            return -(df0 / 2 + 1) * np.log(s2_) - df0 * sc0 / (2 * s2_)

        h, sb, sl = 0.05, 0.5, 0.3
        acc = {"gamma": 0, "beta": 0, "sigma2": 0, "range": 0}
        tries = {k: 0 for k in acc}
        keep = []
        eta = eta_of(beta, gam, s2, j)
        ll = self._loglik(eta)
        win = {"gamma": 0, "beta": 0, "sigma2": 0}
        for it in range(n_iter):
            # --- MALA on Gamma
            sig = np.sqrt(s2)
            grad = sig * (self._L[j].T @ self._grad_eta(eta)) - gam
            mean_f = gam + 0.5 * h * grad
            g_new = mean_f + np.sqrt(h) * rng.standard_normal(n)
            eta_new = eta_of(beta, g_new, s2, j)
            grad_new = sig * (self._L[j].T @ self._grad_eta(eta_new)) - g_new
            mean_b = g_new + 0.5 * h * grad_new
            ll_new = self._loglik(eta_new)
            log_a = (ll_new - 0.5 * g_new @ g_new) - (ll - 0.5 * gam @ gam) \
                - ((gam - mean_b) @ (gam - mean_b) - (g_new - mean_f) @ (g_new - mean_f)) / (2 * h)
            ok = np.log(rng.uniform()) < log_a
            if ok:
                gam, eta, ll = g_new, eta_new, ll_new
            win["gamma"] += ok
            # --- beta (RW)
            b_new = beta + sb * (Lb @ rng.standard_normal(p))
            eta_new = eta + F @ (b_new - beta)
            ll_new = self._loglik(eta_new)
            log_a = ll_new - ll - 0.5 * (b_new @ b_new - beta @ beta) / self.beta_prior_sd ** 2
            ok = np.log(rng.uniform()) < log_a
            if ok:
                beta, eta, ll = b_new, eta_new, ll_new
            win["beta"] += ok
            # --- log sigma^2 (RW, with Jacobian)
            s2_new = s2 * np.exp(sl * rng.standard_normal())
            eta_new = eta_of(beta, gam, s2_new, j)
            ll_new = self._loglik(eta_new)
            log_a = ll_new - ll + log_prior_s2(s2_new) - log_prior_s2(s2) + np.log(s2_new / s2)
            ok = np.log(rng.uniform()) < log_a
            if ok:
                s2, eta, ll = s2_new, eta_new, ll_new
            win["sigma2"] += ok
            # --- range: neighbouring grid value
            j_new = j + (1 if rng.uniform() < 0.5 else -1)
            if 0 <= j_new < len(self._range_values):
                eta_new = eta_of(beta, gam, s2, j_new)
                ll_new = self._loglik(eta_new)
                if np.log(rng.uniform()) < ll_new - ll + log_pr[j_new] - log_pr[j]:
                    j, eta, ll = j_new, eta_new, ll_new
                    if it >= burn:
                        acc["range"] += 1
            if it >= burn:
                tries["range"] += 1
            # --- interweaving: centred Gibbs updates with eta fixed (likelihood unchanged)
            if self.interweave:
                sig = np.sqrt(s2)
                r = solve_triangular(self._L[j], eta, lower=True) / sig
                Aj = A_all[j] / sig
                cP = cho_factor(Aj.T @ Aj + np.eye(p) / self.beta_prior_sd ** 2, lower=True)
                beta = cho_solve(cP, Aj.T @ r) + solve_triangular(cP[0].T, rng.standard_normal(p), lower=False)
                S_lat = eta - F @ beta
                if it % self.range_gibbs_every == 0 and J > 1:
                    # range | S with sigma^2 integrated out (collapsed: moves along the
                    # sigma^2-range ridge), then sigma^2 | S, range
                    us = [solve_triangular(L, S_lat, lower=True) for L in self._L]
                    q = np.array([v @ v for v in us])
                    lp = log_pr - logdetL - 0.5 * (df0 + n) * np.log(df0 * sc0 + q)
                    pr_j = np.exp(lp - lp.max())
                    j = int(rng.choice(J, p=pr_j / pr_j.sum()))
                    u = us[j]
                else:
                    u = solve_triangular(self._L[j], S_lat, lower=True)
                s2 = (df0 * sc0 + u @ u) / rng.chisquare(df0 + n)
                gam = u / np.sqrt(s2)
            # --- adaptation (burn-in only), acceptance bookkeeping
            if it < burn and (it + 1) % 50 == 0:
                h *= np.exp((win["gamma"] / 50 - 0.574) * 2)
                sb *= np.exp((win["beta"] / 50 - 0.30) * 2)
                sl *= np.exp((win["sigma2"] / 50 - 0.30) * 2)
                win = {k: 0 for k in win}
            elif it >= burn:
                for k in ("gamma", "beta", "sigma2"):
                    acc[k] += win[k]
                    tries[k] += 1
                win = {k: 0 for k in win}
            else:
                pass
            if it >= burn and (it - burn) % thin == 0:
                keep.append((beta.copy(), s2, j, gam.copy(), ll))
            if verbose and (it + 1) % 1000 == 0:
                print(f"iteration {it + 1}/{n_iter}: loglik {ll:.1f}, sigma2 {s2:.3f}, "
                      f"range {self._range_values[j]:.3g}")

        self._samples = keep
        self._F = F
        self.beta_names_ = self._trend.names()
        self.trace_ = pd.DataFrame(
            [{**{f"beta_{nm}": b for nm, b in zip(self.beta_names_, bb)}, "sigma2": ss,
              "range": self._range_values[jj], "loglik": lk} for bb, ss, jj, _, lk in keep])
        self.acceptance_ = {k: acc[k] / max(tries[k], 1) for k in acc}
        return self

    def latent_samples(self) -> np.ndarray:
        """Posterior draws of the linear predictor ``eta`` at the data, shape (n_kept, n)."""
        if not hasattr(self, "_samples"):
            raise RuntimeError("call fit() first")
        return np.stack([self._F @ b + np.sqrt(s2) * (self._L[j] @ g) for b, s2, j, g, _ in self._samples])

    def predictive_samples(self, coords, covariates=None, scale: str = "response", seed=None) -> np.ndarray:
        """
        One posterior draw per kept MCMC sample at `coords`, shape (n_kept, n_points):
        the linear predictor (``scale='link'``) or the rate / probability
        (``'response'``: ``exp(eta)`` or ``logistic(eta)``).
        """
        if not hasattr(self, "_samples"):
            raise RuntimeError("call fit() first")
        if scale not in ("response", "link"):
            raise ValueError("scale must be 'response' or 'link'")
        xp = np.asarray(coords, dtype=float)
        f0 = self._trend.transform(xp, covariates)
        d0 = _pairwise(xp, self._xy)
        rng = np.random.default_rng(seed)
        out = np.empty((len(self._samples), len(xp)))
        js = np.array([s[2] for s in self._samples])
        for j in np.unique(js):
            r = correlation_function(d0, self.model, self._range_values[j], self.nu, self.nu)
            L = self._L[j]
            # S0 | S ~ N(r V^-1 S, sigma^2 (1 + tau - r V^-1 r')) with V = L L' (S0 has its own nugget)
            A = np.linalg.solve(L, r.T)                         # L^-1 r'
            var_unit = np.clip(1.0 + self.nugget_ratio - (A ** 2).sum(0), 0.0, None)
            for i in np.flatnonzero(js == j):
                b, s2, _, g, _ = self._samples[i]
                mean = f0 @ b + np.sqrt(s2) * (A.T @ g)            # r V^-1 (sigma L g) = sigma r L^-T g
                eta0 = mean + np.sqrt(s2 * var_unit) * rng.standard_normal(len(xp))
                out[i] = eta0
        if scale == "link":
            return out
        return np.exp(out) if self.family == "poisson" else 1.0 / (1.0 + np.exp(-out))

    def predict(self, coords, covariates=None, scale: str = "response", seed=None) -> pd.DataFrame:
        """Posterior mean, sd, median and 5% / 95% quantiles at `coords`."""
        s = self.predictive_samples(coords, covariates, scale, seed)
        return pd.DataFrame({"mean": s.mean(0), "sd": s.std(0), "median": np.median(s, 0),
                             "q05": np.quantile(s, 0.05, 0), "q95": np.quantile(s, 0.95, 0)})

    def predict_proba(self, coords, threshold: float, covariates=None, greater: bool = True,
                      scale: str = "response", seed=None) -> np.ndarray:
        """Posterior probability that the rate / probability exceeds `threshold`."""
        s = self.predictive_samples(coords, covariates, scale, seed)
        return (s > threshold).mean(0) if greater else (s <= threshold).mean(0)


# ------------------------------------------------------------ one call


def bayesian_kriging(coords, values, coords_pred, model: str = "exponential", trend: int = 0,
                     quantiles: Sequence[float] = (0.05, 0.5, 0.95), covariates=None,
                     covariates_pred=None, signal: bool = True, return_model: bool = False, **kwargs):
    """
    Fit `BayesianKriging` and return a table of predictive mean, sd and
    quantiles at `coords_pred`.

    Parameters
    ----------
    coords, values : training locations (n, 2) and values (n,).
    coords_pred : prediction locations (m, 2).
    model, trend : correlation model and polynomial trend degree.
    quantiles : predictive quantiles to report (columns ``qNN``).
    covariates, covariates_pred : optional covariates at the data and at the
        prediction locations.
    signal : bool, default True
        Predict the noise-free process (True) or a new observation
        including the nugget (False; use this to compare with held-out data).
    return_model : bool, default False
        Also return the fitted `BayesianKriging` (posterior, draws, ...).
    **kwargs
        Other `BayesianKriging` arguments (``range_grid``, ``nugget_grid``,
        ``range_prior``, ``nugget_prior``, ``beta_prior``, ``sigma2_prior``,
        ``nu``).

    Returns
    -------
    pandas.DataFrame with columns ``mean``, ``sd`` and one per quantile, or
    ``(table, model)`` with `return_model`.
    """
    bk = BayesianKriging(model=model, trend=trend, **kwargs).fit(coords, values, covariates)
    mean, sd = bk.predict(coords_pred, covariates_pred, return_std=True, signal=signal)
    q = bk.predict_quantiles(coords_pred, quantiles, covariates_pred, signal=signal)
    out = pd.DataFrame({"mean": mean, "sd": sd})
    for j, qq in enumerate(quantiles):
        out[f"q{int(round(qq * 100)):02d}"] = q[:, j]
    return (out, bk) if return_model else out
