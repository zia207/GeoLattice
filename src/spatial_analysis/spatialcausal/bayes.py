"""Bayesian spatial causal models with ICAR / BYM random effects.

:class:`BayesianSpatialCausal` fits

    Gaussian:  y_i = a + tau * T_i + delta * (W T)_i + x_i' b + u_i + e_i
    Poisson :  O_i ~ Poisson(E_i theta_i),  log theta_i = a + tau * T_i + delta * (W T)_i + x_i' b + u_i + v_i

with ``u`` an intrinsic CAR (ICAR) field on the neighbour graph and, for
counts, ``v`` an unstructured effect (the BYM model). ``tau`` is the direct
effect, ``delta`` the spillover from the share of treated neighbours.

* **Gaussian** outcomes use an exact, partially collapsed Gibbs sampler
  written for this package. The ICAR precision ``Q = D - A`` is
  diagonalised once (``Q = V diag(lambda) V'``); in that basis the field's
  full conditional is diagonal and the sum-to-zero constraint simply drops
  the null-space directions. Each iteration (1) updates the noise and ICAR
  precisions jointly by Metropolis on their marginal likelihood with the
  field integrated out (O(n) per evaluation), (2) draws the coefficients
  with the field integrated out, and (3) draws the field. Mixing therefore
  does not degrade when treatment and the field are correlated, or when the
  two variance components trade off.
* **Count** outcomes call :class:`geolattice.spatial_statistics.bayes.BYM` with treatment
  (and its spatial lag) as covariates; effects are reported as log rate
  ratios with ``exp`` summaries.

Spatial confounding: adding ``u`` does *not* by itself remove bias from a
spatially smooth unmeasured confounder (Hodges & Reich, 2010). With
``adjust="spatial+"`` the treatment is first residualised on the leading
Moran eigenvectors (Dupont et al., 2022), the Bayesian analogue of
:class:`~geolattice.spatial_analysis.spatialcausal.confounding.SpatialPlus`.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from geolattice.spatial_statistics.bayes.icar import ess, icar_structure, rhat

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_2d, as_sparse_weights, check_lengths
from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult

_MAX_N_GAUSSIAN = 5000


def _icar_eigen(w, n: int):
    """Eigen-decomposition of the ICAR precision ``Q = D - A`` (binary, symmetrised).

    Returns ``(V_plus, lam_plus, n_null)``: eigenvectors with positive
    eigenvalue (the constrained field lives in their span) and the number of
    null directions (one per connected component, including islands).
    """
    from geolattice.spatial_statistics.weights import W

    if isinstance(w, W):
        struct = icar_structure(w)
        A = struct["A"].toarray()
    else:
        S = as_sparse_weights(w, n)
        A = ((S + S.T) > 0).astype(float).toarray()
    Q = np.diag(A.sum(1)) - A
    lam, V = np.linalg.eigh(Q)
    tol = 1e-8 * max(1.0, lam.max())
    keep = lam > tol
    return V[:, keep], lam[keep], int((~keep).sum())


def _moran_residualize(t: np.ndarray, w, n: int, k: Optional[int]) -> np.ndarray:
    from geolattice.spatial_analysis.spatialcausal.confounding import moran_eigenvectors

    k = k or max(5, n // 10)
    E = moran_eigenvectors(w, k, n)
    Z = np.column_stack([np.ones(n), E])
    return t - Z @ np.linalg.lstsq(Z, t, rcond=None)[0]


def _summ(draws: np.ndarray, level: float = 0.95):
    a = (1 - level) / 2
    flat = draws.reshape(-1)
    return {"mean": float(flat.mean()), "sd": float(flat.std()),
            "q_lo": float(np.quantile(flat, a)), "q_hi": float(np.quantile(flat, 1 - a)),
            "p_gt_0": float(np.mean(flat > 0)),
            "rhat": float(rhat(draws)) if draws.ndim == 2 and draws.shape[0] > 1 else np.nan,
            "ess": float(ess(draws)) if draws.ndim == 2 else np.nan}


class BayesianSpatialCausal:
    """Bayesian treatment and spillover effects with an ICAR / BYM spatial field.

    Parameters
    ----------
    family : {"gaussian", "poisson"}
        For ``"poisson"``, ``y`` holds counts and ``expected`` must be given
        to :meth:`fit` (see ``geolattice.spatial_statistics.bayes.expected_counts``).
    spillover : bool
        Include ``delta * (W T)``: the share of treated neighbours
        (row-standardised weights).
    spatial : {"icar", "none"}
        Spatial random effect (Gaussian family). Counts always use BYM.
    adjust : {"none", "spatial+"}
        Residualise treatment (and its lag) on ``k_adjust`` Moran
        eigenvectors before fitting.
    draws, tune, chains : MCMC settings (per chain; ``tune`` discarded).
    thin : int
        Keep every ``thin``-th draw (Poisson only; passed to BYM, whose
        single-site sampler mixes slowly: check ``max_rhat`` and ``min_ess``).
    n_mh : int
        Metropolis steps per iteration for the variance components (Gaussian).
    backend : {"gibbs", "pymc", "auto"}
        Sampler for the Poisson/BYM model (``geolattice.spatial_statistics.bayes.BYM``).
    prior_coef_sd : float
        Normal prior SD for coefficients, on the scale of the standardised
        outcome (Gaussian).
    priors : (shape, rate)
        Gamma prior for the ICAR precision and for the noise precision
        (Gaussian; on the standardised outcome scale).
    seed : int

    Attributes
    ----------
    posterior_ : dict of arrays (chains, draws)
        ``"tau"``, ``"delta"`` (if spillover), ``"sigma"``, ``"tau_u"``,
        ``"beta"`` (chains, draws, p).
    field_ : posterior mean of the spatial effect ``u`` (outcome scale).
    diagnostics_ : DataFrame with posterior summaries, R-hat and ESS.
    result_ : SpatialCausalResult (estimate = posterior mean, se = posterior SD).
    """

    def __init__(self, family: str = "gaussian", spillover: bool = False, spatial: str = "icar",
                 adjust: str = "none", k_adjust: Optional[int] = None, draws: int = 1000,
                 tune: int = 500, chains: int = 2, thin: int = 10, prior_coef_sd: float = 10.0,
                 priors=(1.0, 0.01), n_mh: int = 10, backend: str = "gibbs", seed: int = 0):
        if family not in ("gaussian", "poisson"):
            raise ValueError("family must be 'gaussian' or 'poisson'")
        if spatial not in ("icar", "none"):
            raise ValueError("spatial must be 'icar' or 'none'")
        if adjust not in ("none", "spatial+"):
            raise ValueError("adjust must be 'none' or 'spatial+'")
        if chains < 2:
            raise ValueError("use at least 2 chains (needed for R-hat)")
        self.family, self.spillover, self.spatial, self.adjust = family, spillover, spatial, adjust
        self.k_adjust = k_adjust
        self.draws, self.tune, self.chains, self.thin = draws, tune, chains, thin
        self.prior_coef_sd, self.priors, self.seed = prior_coef_sd, priors, seed
        self.n_mh, self.backend = n_mh, backend

    # ------------------------------------------------------------------
    def _design(self, X, t, w, n):
        Ws = as_sparse_weights(w, n, row_standardize=True)
        cols, names = [t], ["tau"]
        if self.spillover:
            cols.append(np.asarray(Ws @ t).ravel())
            names.append("delta")
        if self.adjust == "spatial+":
            cols = [_moran_residualize(c, w, n, self.k_adjust) for c in cols]
        if X is not None:
            cols += [X[:, j] for j in range(X.shape[1])]
            names += [f"x{j}" for j in range(X.shape[1])]
        return np.column_stack(cols), names

    def fit(self, data_or_X, treatment=None, y=None, w=None, expected=None, coords=None):
        if isinstance(data_or_X, SpatialCausalData):
            d = data_or_X
            X, t, y, w = d.X, d.treatment, d.y, (w if w is not None else d.w)
        else:
            if w is None:
                raise ValueError("pass spatial weights w (or a SpatialCausalData)")
            X = None if data_or_X is None else as_2d(data_or_X)
            t, y = as_1d(treatment), as_1d(y)
        n = check_lengths(t=t, y=y, X=X)
        A, names = self._design(X, t, w, n)
        self.coef_names_ = names
        if self.family == "gaussian":
            post = self._gaussian(A, y, w, n)
        else:
            if expected is None:
                raise ValueError("family='poisson' needs expected counts")
            post = self._poisson(A, y, as_1d(expected, "expected"), w, names)
        self.posterior_ = post
        self._finish(post, n)
        return self

    # ------------------------------------------------------------------
    def _gaussian(self, A, y, w, n):
        if n > _MAX_N_GAUSSIAN and self.spatial == "icar":
            raise ValueError(f"the Gaussian ICAR sampler is dense; n={n} > {_MAX_N_GAUSSIAN}")
        my, sy = y.mean(), y.std()
        z = (y - my) / sy
        A1 = np.column_stack([np.ones(n), A])
        p = A1.shape[1]
        a0, b0 = self.priors
        prior_prec = np.full(p, 1.0 / self.prior_coef_sd ** 2)
        prior_prec[0] = 1e-6  # flat intercept
        if self.spatial == "icar":
            V, lam, _ = _icar_eigen(w, n)
            VA, Vz = V.T @ A1, V.T @ z
        else:
            V = lam = None
        AtA, Atz = A1.T @ A1, A1.T @ z
        master = np.random.SeedSequence(self.seed)
        out = {k: [] for k in ("beta", "sigma", "tau_u", "u")}
        accept = []

        def log_marg(ls2, ltu, rv, rest):
            """log p(r | sigma^2, tau_u) with the field integrated out, + log priors + Jacobian.

            ``rv = V'r`` and ``rest`` = squared norm of r in the null space of Q.
            """
            s2_, tu_ = np.exp(ls2), np.exp(ltu)
            var_k = s2_ + 1.0 / (tu_ * lam)
            ll = -0.5 * (np.sum(np.log(var_k)) + np.sum(rv ** 2 / var_k)
                         + (n - len(lam)) * ls2 + rest / s2_)
            # Gamma(a0, b0) priors on 1/sigma^2 and tau_u, with log-scale Jacobians
            lp = a0 * (-ls2) - b0 / s2_ + a0 * ltu - b0 * tu_
            return ll + lp

        for ch, ss in enumerate(master.spawn(self.chains)):
            rng = np.random.default_rng(ss)
            s2 = 0.5 * (ch + 1)
            tu = 1.0 * (ch + 1)
            step, acc_n = 0.3, 0
            beta = np.linalg.lstsq(A1, z, rcond=None)[0]
            keep_b, keep_s, keep_t, u_sum = [], [], [], np.zeros(n)
            for it in range(self.tune + self.draws):
                # (1) variance components | beta, field integrated out (random-walk MH)
                if V is not None:
                    r0 = z - A1 @ beta
                    rv = V.T @ r0
                    rest = r0 @ r0 - rv @ rv
                    cur = np.array([np.log(s2), np.log(tu)])
                    lcur = log_marg(cur[0], cur[1], rv, rest)
                    for _ in range(self.n_mh):              # cheap: O(n) per step
                        prop = cur + step * rng.standard_normal(2)
                        lprop = log_marg(prop[0], prop[1], rv, rest)
                        if np.log(rng.random()) < lprop - lcur:
                            cur, lcur = prop, lprop
                            acc_n += 1
                    s2, tu = np.exp(cur[0]), np.exp(cur[1])
                    if it < self.tune and (it + 1) % 50 == 0:   # adapt toward ~35% acceptance
                        step *= np.exp(acc_n / (50 * self.n_mh) - 0.35)
                        acc_n = 0
                # (2) beta | variances, field integrated out
                if V is not None:
                    D = tu * lam + 1.0 / s2           # field conditional precision (eigenbasis)
                    G = VA / (D[:, None])
                    P = AtA / s2 - (VA.T @ G) / s2 ** 2 + np.diag(prior_prec)
                    rhs = Atz / s2 - (VA.T @ (Vz / D)) / s2 ** 2
                else:
                    P = AtA / s2 + np.diag(prior_prec)
                    rhs = Atz / s2
                L = np.linalg.cholesky(P)
                mean = np.linalg.solve(P, rhs)
                beta = mean + np.linalg.solve(L.T, rng.standard_normal(p))
                r = z - A1 @ beta
                # (3) field | beta, variances
                if V is not None:
                    c = (V.T @ r) / s2 / D + rng.standard_normal(len(D)) / np.sqrt(D)
                    u = V @ c
                else:
                    u = np.zeros(n)
                    res = r
                    s2 = 1.0 / rng.gamma(a0 + n / 2, 1.0 / (b0 + 0.5 * res @ res))
                if it >= self.tune:
                    if V is not None and it == self.tune:
                        acc_n = 0
                    keep_b.append(beta.copy())
                    keep_s.append(np.sqrt(s2))
                    keep_t.append(tu)
                    u_sum += u
            if V is not None:
                accept.append(acc_n / (self.draws * self.n_mh))
            out["beta"].append(np.array(keep_b))
            out["sigma"].append(np.array(keep_s))
            out["tau_u"].append(np.array(keep_t))
            out["u"].append(u_sum / self.draws)
        self.acceptance_ = float(np.mean(accept)) if accept else None
        beta = np.stack(out["beta"]) * sy                      # back to outcome scale
        beta[..., 0] = beta[..., 0] + my
        post = {"intercept": beta[..., 0], "beta": beta[..., 1:], "sigma": np.stack(out["sigma"]) * sy,
                "tau_u": np.stack(out["tau_u"]) / sy ** 2 if V is not None else None}
        self.field_ = np.mean(out["u"], axis=0) * sy
        return post

    def _poisson(self, A, O, E, w, names):
        from geolattice.spatial_statistics.bayes import BYM
        from geolattice.spatial_statistics.weights import W

        if not isinstance(w, W):
            w = W.from_sparse(as_sparse_weights(w, len(O)))
        model = BYM(O, E, w, X=A, names=names, model="bym", backend=self.backend, draws=self.draws,
                    tune=self.tune, chains=self.chains, thin=self.thin, seed=self.seed)
        xsd = A.std(0)
        beta = model._raw["beta"] / xsd                         # original scale (chains, draws, p)
        self.bym_ = model
        self.field_ = None
        return {"beta": beta, "intercept": None, "sigma": None, "tau_u": model._raw["tau_u"]}

    # ------------------------------------------------------------------
    def _finish(self, post, n):
        beta = post["beta"]
        rows = {nm: _summ(beta[..., j]) for j, nm in enumerate(self.coef_names_)}
        if post["sigma"] is not None:
            rows["sigma"] = _summ(post["sigma"])
        if post["tau_u"] is not None and not np.all(np.isnan(post["tau_u"])):
            rows["tau_u"] = _summ(post["tau_u"])
        self.diagnostics_ = pd.DataFrame(rows).T
        self.posterior_["tau"] = beta[..., 0]
        draws = {"direct": beta[..., 0]}
        if self.spillover:
            self.posterior_["delta"] = beta[..., 1]
            draws["spillover"] = beta[..., 1]
            draws["total"] = beta[..., 0] + beta[..., 1]
        effects = {k: EffectEstimate(k, float(v.mean()), float(v.std())) for k, v in draws.items()}
        meta = {"credible_95": {k: (round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4))
                                for k, v in draws.items()},
                "p_gt_0": {k: round(float(np.mean(v > 0)), 4) for k, v in draws.items()},
                "max_rhat": round(float(np.nanmax(self.diagnostics_["rhat"])), 3),
                "min_ess": round(float(np.nanmin(self.diagnostics_["ess"])), 1),
                "family": self.family, "adjust": self.adjust}
        if self.family == "poisson":
            meta["rate_ratio"] = {k: round(float(np.exp(v).mean()), 4) for k, v in draws.items()}
            meta["effect_scale"] = "log rate ratio"
        self._draws = draws
        if meta["max_rhat"] > 1.05:
            warnings.warn(f"max R-hat {meta['max_rhat']} > 1.05; run longer chains", RuntimeWarning)
        self.result_ = SpatialCausalResult(f"Bayesian spatial causal model ({self.family})", effects,
                                           se_method="posterior SD", n=n, **meta)

    # ------------------------------------------------------------------
    def effect_draws(self, name: str = "direct") -> np.ndarray:
        """Flattened posterior draws of ``"direct"``, ``"spillover"`` or ``"total"``."""
        return self._draws[name].reshape(-1)

    def probability(self, name: str = "direct", threshold: float = 0.0) -> float:
        """Posterior probability that the effect exceeds ``threshold``."""
        return float(np.mean(self.effect_draws(name) > threshold))

    def effect(self, name: str = "direct") -> float:
        return self.result_[name].estimate

    def summary(self) -> str:
        return self.result_.summary()
