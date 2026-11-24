"""
multi_poisson_kriging.py
========================

Multivariate Poisson (co)kriging for areal disease-count data.

Python conversion of the two R scripts
``Multivariate-Poisson-cokriging.R`` / ``-utils.R`` implementing the method of

    Payares-Garcia D., Osei F., Mateu J., Stein A. (2024).
    "Multivariate Poisson cokriging: a geostatistical model for health count
    data." Statistical Methods in Medical Research. doi:10.1177/09622802241268488

The model predicts/smooths a target disease *risk* from noisy areal counts by
borrowing strength across several spatially co-regionalised diseases. Each
count is treated as Poisson, ``Y_l(v_i) ~ Poisson(n_i R_l(v_i))``; the latent
risks ``R_l`` are jointly modelled with a Linear Model of Coregionalization
(LMC). This module converts, and generalises to an arbitrary number ``P`` of
variables, the R functions:

    pck.variogram          -> pck_variogram         (direct Poisson semivariogram)
    pck.crossvariogram     -> pck_crossvariogram    (cross Poisson semivariogram)
    rlk                    -> shared_risk           (Poisson shared/joint risk)
    eb.risk                -> eb_risk               (empirical-Bayes rates)
    Cexp/Csph              -> c_exp / c_sph          (covariance models)
    lmc.poisson.cokrige.*  -> fit_lmc               (LMC coregionalization fit)
    poisson.cokrige.{bi,three,four} + *.pred.*
                           -> poisson_cokrige       (N-variate co-kriging)
    poisson.krige.one + .pred.one
                           -> poisson_krige         (univariate Poisson kriging)

Both prediction (new locations) and leave-one-out smoothing (prediction
locations == data locations) are supported, matching the R behaviour.

Conventions (as in the R code)
------------------------------
* ``cov`` functions return the covariance C(h) directly (sill at h=0), NOT the
  semivariance. The LMC covariance for a pair (l,k) is ``B[l,k] * rho(h)``.
* A rate is ``pop_rate * cases / population`` (e.g. ``pop_rate = 100000``).
* Poisson reliability terms are added to the (co)kriging matrix diagonal:
  direct block ``pop_rate**2 * m*_l / n_i`` with ``m*_l = sum(y_l)/sum(n_l)``
  (the variance of a Poisson rate expressed per ``pop_rate``); cross block at
  co-located points ``pop_rate * mean(R_lk) / n_i`` with ``R_lk`` the Poisson
  shared risk on the ``pop_rate`` scale. This is the multivariate extension of
  the univariate Poisson-kriging error variance. Results at ``pop_rate = c``
  equal ``c`` times the results at ``pop_rate = 1`` (variances ``c**2``).

  Versions before 2026-11 used ``m*_l * pop_rate`` (one factor of
  ``pop_rate`` short) in the direct bias and reliability terms, which was
  correct only for ``pop_rate = 1``.
* A dataset may carry its own measurement-error variance ``noise_var`` (on the
  ``pop_rate**2`` scale); it then replaces the Poisson direct reliability term
  (heteroscedastic kriging with known error variances, e.g. survey SEs).

GPU (CuPy) support
------------------
:func:`poisson_cokrige` in **k-nearest-neighbor** mode
(``number_of_neighbors=k``) batches every local system into stacked
``xp.linalg.solve`` calls (chunks of ``chunk_size`` targets) — NumPy on CPU,
CuPy on GPU. Pass ``use_gpu='auto'|True|False``. The k-NN path handles
variables observed at different locations (each variable takes its own ``k``
nearest data) and leave-one-out smoothing (``smooth=True``: the target's own
datum is dropped, co-located auxiliary data are kept). The default **global**
path (every known area in the system) keeps the original per-target CPU loop.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

from ..utils.backend import (
    as_gpu_array,
    get_array_module,
    resolve_cupy_use_gpu,
    to_numpy,
)

__all__ = [
    "c_exp", "c_sph", "covariance_model",
    "shared_risk", "eb_risk",
    "pck_variogram", "pck_crossvariogram",
    "LMC", "fit_lmc",
    "poisson_cokrige", "poisson_krige",
]

# ---------------------------------------------------------------------------
# Covariance models  (R: Cexp, Csph, covariance.model)
# ---------------------------------------------------------------------------
def c_exp(h, rng, sill):
    """Exponential covariance C(h) = sill * exp(-h / range)."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    return sill * xp.exp(-h / rng)

def c_sph(h, rng, sill):
    """Spherical covariance C(h) = sill*(1 - 1.5 h/a + 0.5 (h/a)^3), 0 beyond a."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    hr = xp.clip(h / rng, 0, 1)
    return xp.where(h <= rng, sill * (1 - 1.5 * hr + 0.5 * hr ** 3), 0.0)

def _unit_correlation(h, rng, model, xp):
    """Unit-sill spatial correlation ρ(h) on ``xp`` (NumPy or CuPy)."""
    if model == "Exp":
        return xp.exp(-h / rng)
    hr = xp.clip(h / rng, 0, 1)
    return xp.where(h <= rng, 1.0 - 1.5 * hr + 0.5 * hr ** 3, 0.0)

def covariance_model(name):
    """Map 'Exp'/'Sph' to the covariance function (R: covariance.model)."""
    return {"Exp": c_exp, "Sph": c_sph}[name]

# ---------------------------------------------------------------------------
# Shared (joint) risk and empirical-Bayes rates
# ---------------------------------------------------------------------------
def shared_risk(Rl, Rk, rho, log=False):
    """
    Poisson shared risk between two diseases (R: rlk).

    ``rho * sqrt(Rl * Rk)`` (or the log-scale variant). ``Rl``/``Rk`` are the
    (per-``pop_rate``) rates of the two diseases and ``rho`` their Poisson
    correlation. Used to build the cross-semivariogram bias term and the
    co-located cross reliability term.
    """
    Rl, Rk = np.asarray(Rl, float), np.asarray(Rk, float)
    if log:
        return np.log(rho * np.sqrt(np.exp(Rl) * np.exp(Rk)))
    return rho * np.sqrt(Rl * Rk)

def eb_risk(cases, pop, pop_rate=100000.0):
    """Empirical-Bayes standardized rate (R: eb.risk)."""
    r = np.asarray(cases, float)
    n = np.asarray(pop, float)
    x = r / n
    N = len(r)
    m = np.nansum(r) / np.nansum(n)
    s2 = np.nansum(n * (x - m) ** 2) / np.nansum(n)
    a = s2 - (m / (np.nansum(n) / N))
    if a < (m / (np.nansum(n) / N)):
        a = 0.0
    v = a + (m / n)
    return (x - m) / np.sqrt(v)

# ---------------------------------------------------------------------------
# Empirical population-weighted Poisson (cross-)semivariograms
# ---------------------------------------------------------------------------
def _bin_semivariogram(d, wdiff_minus_bias, w, maxd, nbins):
    """Shared binning: gamma(bin) = sum(w*diff - bias) / (2 sum(w))."""
    if maxd is None:
        maxd = d.max() / 2.0
    edges = np.linspace(0, maxd, nbins + 1)
    idx = np.digitize(d, edges) - 1
    lags, gam, npair = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 1:
            continue
        sw = w[m].sum()
        if sw <= 0:
            continue
        lags.append(d[m].mean())
        gam.append(wdiff_minus_bias[m].sum() / (2.0 * sw))
        npair.append(int(m.sum()))
    return (np.array(lags), np.array(gam), np.array(npair, dtype=float))

def pck_variogram(coords, cases, pop, pop_rate=100000.0, maxd=None, nbins=15):
    """
    Population-weighted **direct** Poisson semivariogram (R: pck.variogram).

    For unique location pairs (i<j)::

        diff = (pop_rate*(y_i/n_i - y_j/n_j))^2
        w    = n_i n_j / (n_i + n_j)
        bias = pop_rate**2 * sum(y)/sum(n)         # noise variance of w*diff
        gamma(h) = [ sum_pairs(w*diff) - n_pairs*bias ] / [ 2 sum_pairs(w) ]

    Returns ``(lags, gamma, npairs)``; negative gammas are clipped to 0.
    """
    coords = np.asarray(coords, float)
    y = np.asarray(cases, float)
    n = np.asarray(pop, float)
    iu = np.triu_indices(len(coords), 1)
    d = squareform(pdist(coords))[iu]
    diff = (pop_rate * (y[iu[0]] / n[iu[0]] - y[iu[1]] / n[iu[1]])) ** 2
    w = (n[iu[0]] * n[iu[1]]) / (n[iu[0]] + n[iu[1]])
    # Poisson noise of a rate scaled by pop_rate has variance pop_rate**2 * m / n,
    # so E[w * diff] carries the bias pop_rate**2 * m  (m = sum(y) / sum(n)).
    bias = pop_rate ** 2 * y.sum() / n.sum()
    lags, gam, npair = _bin_semivariogram(d, w * diff - bias, w, maxd, nbins)
    return lags, np.clip(gam, 0, None), npair

def pck_crossvariogram(coords, cases_l, cases_k, pop, r_lk,
                       pop_rate=100000.0, maxd=None, nbins=15):
    """
    Population-weighted **cross** Poisson semivariogram (R: pck.crossvariogram).

    ``r_lk`` is the Poisson shared-risk vector (see :func:`shared_risk`) on
    the same ``pop_rate`` scale as the rates, so ``bias = pop_rate * mean(r_lk)``
    equals ``pop_rate**2`` times the shared risk as a proportion. Cross gammas may be negative; returned as
    given (NaNs dropped).
    """
    coords = np.asarray(coords, float)
    yl = np.asarray(cases_l, float)
    yk = np.asarray(cases_k, float)
    n = np.asarray(pop, float)
    iu = np.triu_indices(len(coords), 1)
    d = squareform(pdist(coords))[iu]
    dl = pop_rate * (yl[iu[0]] / n[iu[0]] - yl[iu[1]] / n[iu[1]])
    dk = pop_rate * (yk[iu[0]] / n[iu[0]] - yk[iu[1]] / n[iu[1]])
    diff = dl * dk
    w = (n[iu[0]] * n[iu[1]]) / (n[iu[0]] + n[iu[1]])
    bias = pop_rate * np.nanmean(np.asarray(r_lk, float))
    lags, gam, npair = _bin_semivariogram(d, w * diff - bias, w, maxd, nbins)
    return lags, gam, npair

# ---------------------------------------------------------------------------
# Linear Model of Coregionalization
# ---------------------------------------------------------------------------
def _nearest_psd(B):
    """Project a symmetric matrix onto the nearest positive-semidefinite one."""
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)
    w = np.clip(w, 0, None)
    return (V * w) @ V.T

class LMC:
    """
    Fitted Linear Model of Coregionalization: one spatial structure with a
    common ``range`` and a P x P positive-semidefinite sill matrix ``B``, plus
    an optional P x P nugget matrix ``B0``. The covariance between variables
    l and k is ``cov(l, k, h) = B0[l, k] * (h == 0) + B[l, k] * rho(h)``.
    """
    def __init__(self, B, rng, model="Exp", nugget=None):
        self.B = np.asarray(B, float)
        self.P = self.B.shape[0]
        self.range = float(rng)
        self.model = model
        self._cov = covariance_model(model)
        self.B0 = (np.zeros((self.P, self.P)) if nugget is None
                   else np.asarray(nugget, float).reshape(self.P, self.P))

    def cov(self, l, k, h):
        h = np.asarray(h, float)
        c = self.B[l, k] * self._cov(h, self.range, 1.0)
        if self.B0[l, k] != 0.0:
            c = c + self.B0[l, k] * np.isclose(h, 0.0)
        return c

    def sill(self, l):
        """Total sill (nugget + structured) of variable ``l``."""
        return self.B[l, l] + self.B0[l, l]

    def correlation(self):
        """Structural correlation matrix ``B[l,k] / sqrt(B[l,l] B[k,k])``."""
        d = np.sqrt(np.clip(np.diag(self.B), 1e-300, None))
        return self.B / np.outer(d, d)

    def gamma(self, l, k, h):
        """Model (cross-)semivariogram ``C(0) - C(h)`` for h > 0."""
        h = np.asarray(h, float)
        return self.B0[l, k] + self.B[l, k] * (1.0 - self._cov(h, self.range, 1.0))

    def __repr__(self):
        nug = ", nugget" if np.any(self.B0) else ""
        return f"LMC(model={self.model}, range={self.range:.3g}, P={self.P}{nug})"

def fit_lmc(direct_vars, cross_vars, cross_pairs, P, rng, model="Exp",
            fit_range=False, nugget=False, range_grid=None):
    """
    Fit an LMC to empirical direct and cross semivariograms (R: fit.lmc via
    lmc.poisson.cokrige.*).

    Parameters
    ----------
    direct_vars : list of (lags, gamma, npairs), length P
        Direct semivariograms, variable ``l`` at index ``l``.
    cross_vars : list of (lags, gamma, npairs)
        Cross semivariograms, aligned with ``cross_pairs``.
    cross_pairs : list of (l, k)
        Variable index pairs (l < k) for each entry of ``cross_vars``.
    P : int
        Number of variables.
    rng : float
        Common range parameter (fixed unless ``fit_range=True``, matching the
        R default ``fit.ranges = F``). For ``Exp`` it is the scale ``a`` in
        ``exp(-h/a)``; the practical range is ``3a``.
    model : {"Exp", "Sph"}
    fit_range : bool
        If True, also choose the common range by a 1-D grid search that
        minimises the pair-weighted squared error over all semivariograms.
    nugget : bool
        If True, also fit a nugget matrix ``B0`` (each entry by weighted least
        squares together with the sill, then both matrices projected to PSD).
    range_grid : array, optional
        Candidate ranges for ``fit_range`` (default 40 values in
        ``[0.25 rng, 4 rng]``).

    Returns
    -------
    LMC
        With PSD-projected coregionalization matrices ``B`` (and ``B0``).
    """
    covfun = covariance_model(model)

    def unit_gamma(lags, a):
        return covfun(0.0, a, 1.0) - covfun(np.asarray(lags, float), a, 1.0)

    def fit_pair(lags, gam, npv, a):
        ok = np.isfinite(gam)
        if ok.sum() == 0:
            return 0.0, 0.0
        g, gm, w = unit_gamma(lags[ok], a), gam[ok], npv[ok]
        if not nugget:
            den = np.sum(w * g * g)
            return 0.0, (np.sum(w * g * gm) / den if den > 0 else 0.0)
        X = np.column_stack([np.ones_like(g), g]) * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X, gm * np.sqrt(w), rcond=None)
        return float(coef[0]), float(coef[1])

    def solve(a):
        B = np.zeros((P, P)); B0 = np.zeros((P, P))
        for l in range(P):
            lags, gam, npv = direct_vars[l]
            c0, c1 = fit_pair(lags, gam, npv, a)
            if nugget and c0 < 0:                       # refit without nugget
                den = np.sum(npv * unit_gamma(lags, a) ** 2)
                c0, c1 = 0.0, np.sum(npv * unit_gamma(lags, a) * gam) / den
            B0[l, l], B[l, l] = max(c0, 0.0), max(c1, 0.0)
        for (l, k), (lags, gam, npv) in zip(cross_pairs, cross_vars):
            c0, c1 = fit_pair(lags, gam, npv, a)
            B0[l, k] = B0[k, l] = c0
            B[l, k] = B[k, l] = c1
        return _nearest_psd(B), (_nearest_psd(B0) if nugget else None)

    def sse(a):
        B, B0 = solve(a)
        m = LMC(B, a, model, B0)
        tot = 0.0
        for l in range(P):
            lags, gam, npv = direct_vars[l]
            tot += np.sum(npv * (m.gamma(l, l, lags) - gam) ** 2)
        for (l, k), (lags, gam, npv) in zip(cross_pairs, cross_vars):
            ok = np.isfinite(gam)
            tot += np.sum(npv[ok] * (m.gamma(l, k, lags[ok]) - gam[ok]) ** 2)
        return tot

    if fit_range:
        grid = (np.linspace(0.25 * rng, 4 * rng, 40) if range_grid is None
                else np.asarray(range_grid, float))
        rng = float(grid[int(np.argmin([sse(a) for a in grid]))])
    B, B0 = solve(rng)
    return LMC(B, rng, model, B0)

# ---------------------------------------------------------------------------
# N-variate Poisson cokriging
# (R: poisson.cokrige.{bi,three,four} + poisson.cokrige.pred.*)
# ---------------------------------------------------------------------------
def _shared(shared_mean, l, k):
    return shared_mean.get((l, k), shared_mean.get((k, l), 0.0))

def _direct_noise(d, pop_rate, mstar):
    """Direct reliability term for dataset ``d`` (Poisson or user-supplied)."""
    if d.get("noise_var") is not None:
        return np.asarray(d["noise_var"], float)
    return pop_rate ** 2 * mstar / np.asarray(d["pop"], float)

def _cokrige_solve(datasets, target, lmc, pop_rate, shared_mean, coords_pred):
    """
    Global co-kriging system for ``datasets`` (list of dicts with keys x, y,
    cases, pop and optional noise_var); predicts the target risk at
    ``coords_pred``. Returns an array with columns [x, y, pred, var].
    """
    P = len(datasets)
    coords = [np.column_stack([d["x"], d["y"]]).astype(float) for d in datasets]
    y = [np.asarray(d["cases"], float) for d in datasets]
    n = [np.asarray(d["pop"], float) for d in datasets]
    sizes = [len(c) for c in coords]
    off = np.concatenate([[0], np.cumsum(sizes)])
    N = off[-1]

    C = np.zeros((N, N))
    for l in range(P):
        for k in range(l, P):
            D = cdist(coords[l], coords[k])
            block = lmc.cov(l, k, D)
            if l == k:
                mstar = y[l].sum() / n[l].sum()
                block = block + np.diag(_direct_noise(datasets[l], pop_rate, mstar))
            else:
                rlk = _shared(shared_mean, l, k)
                co = np.isclose(D, 0.0)
                if co.any() and rlk != 0.0:
                    ii, jj = np.where(co)
                    block = block.copy()
                    block[ii, jj] = block[ii, jj] + pop_rate * rlk / n[l][ii]
            C[off[l]:off[l+1], off[k]:off[k+1]] = block
            if l != k:
                C[off[k]:off[k+1], off[l]:off[l+1]] = block.T

    Cbias = np.zeros((N, P))
    for l in range(P):
        Cbias[off[l]:off[l+1], l] = 1.0
    K = np.zeros((N + P, N + P))
    K[:N, :N] = C
    K[:N, N:] = Cbias
    K[N:, :N] = Cbias.T
    Kinv = np.linalg.inv(K)

    rates_all = np.concatenate([y[l] / n[l] * pop_rate for l in range(P)])
    out = np.empty((len(np.atleast_2d(coords_pred)), 4))
    for r, x0 in enumerate(np.atleast_2d(coords_pred)):
        rhs = np.empty(N + P)
        for k in range(P):
            dk = np.sqrt(((coords[k] - x0) ** 2).sum(1))
            rhs[off[k]:off[k+1]] = lmc.cov(target, k, dk)
        rhs[N:] = 0.0
        rhs[N + target] = 1.0
        lam = Kinv @ rhs
        pred = max(float((lam[:N] * rates_all).sum()), 0.0)
        var = lmc.sill(target) - float((lam * rhs).sum())
        out[r] = [x0[0], x0[1], pred, var]
    return out


def _knn_indices(tree, pts, k, exclude=None):
    """k nearest data indices for ``pts``; ``exclude[i]`` (data index) dropped."""
    kk = k + (1 if exclude is not None else 0)
    _, idx = tree.query(pts, k=kk)
    idx = np.asarray(idx, int).reshape(len(pts), kk)
    if exclude is None:
        return idx
    hit = idx == np.asarray(exclude)[:, None]
    # drop the excluded index where present, else the farthest neighbour
    drop = np.where(hit.any(1), hit.argmax(1), kk - 1)
    keep = np.ones_like(idx, bool)
    keep[np.arange(len(pts)), drop] = False
    return idx[keep].reshape(len(pts), k)


def _cokrige_knn(datasets, target, lmc, pop_rate, shared_mean, coords_pred,
                 number_of_neighbors, use_gpu=False, exclude_target=None,
                 mean_rate="local", chunk_size=1024):
    """
    Local k-NN Poisson cokriging, batched with ``xp.linalg.solve``.

    Each variable ``l`` contributes its own ``k_l`` nearest data to every
    target (variables may sit at different locations). ``exclude_target[i]``
    removes one target-variable datum from target ``i`` (leave-one-out).
    ``mean_rate`` sets ``m*_l`` of the Poisson term: ``"local"`` (mean over
    the neighbours) or ``"global"`` (all data of variable ``l``).
    """
    P = len(datasets)
    coords = [np.column_stack([d["x"], d["y"]]).astype(float) for d in datasets]
    rates = [np.asarray(d["cases"], float) / np.asarray(d["pop"], float) * pop_rate
             for d in datasets]
    pops = [np.asarray(d["pop"], float) for d in datasets]
    ks = ([int(number_of_neighbors)] * P if np.isscalar(number_of_neighbors)
          else [int(v) for v in number_of_neighbors])
    for l in range(P):
        if ks[l] < 1:
            raise ValueError("number_of_neighbors must be >= 1")
        need = ks[l] + (1 if (exclude_target is not None and l == target) else 0)
        if need > len(coords[l]):
            raise ValueError(f"variable {l}: only {len(coords[l])} data for "
                             f"number_of_neighbors={ks[l]}")
    noise = []
    for l, d in enumerate(datasets):
        if d.get("noise_var") is not None:
            noise.append(("given", np.asarray(d["noise_var"], float)))
        else:
            noise.append(("poisson", None))
    gstar = [np.asarray(d["cases"], float).sum() / pops[l].sum()
             for l, d in enumerate(datasets)]
    trees = [cKDTree(c) for c in coords]
    coords_pred = np.atleast_2d(np.asarray(coords_pred, float))
    n_pred = len(coords_pred)
    offs = np.concatenate([[0], np.cumsum(ks)])
    m = int(offs[-1]); dim = m + P
    out = np.empty((n_pred, 4))
    out[:, :2] = coords_pred
    xp = np
    if use_gpu:
        import cupy as xp  # noqa: F811

    # keep each (batch, dim, dim) array near 2e7 elements (~160 MB in float64)
    step = int(max(1, min(chunk_size, 2.0e7 // (dim * dim))))
    for c0 in range(0, n_pred, step):
        sl_pred = slice(c0, min(c0 + step, n_pred))
        tg = coords_pred[sl_pred]; nb = len(tg)
        ex = (None if exclude_target is None
              else np.asarray(exclude_target)[sl_pred])
        idx = [_knn_indices(trees[l], tg, ks[l], ex if l == target else None)
               for l in range(P)]
        nc = np.concatenate([coords[l][idx[l]] for l in range(P)], axis=1)  # (nb, m, 2)
        var_of = np.repeat(np.arange(P), ks)                                  # (m,)
        rate_n = np.concatenate([rates[l][idx[l]] for l in range(P)], axis=1)
        pop_n = np.concatenate([pops[l][idx[l]] for l in range(P)], axis=1)
        noise_n = np.empty((nb, m))
        for l in range(P):
            s = slice(offs[l], offs[l + 1])
            if noise[l][0] == "given":
                noise_n[:, s] = noise[l][1][idx[l]]
            else:
                if mean_rate == "global":
                    ms = np.full((nb, 1), gstar[l])
                else:
                    yv = np.asarray(datasets[l]["cases"], float)[idx[l]]
                    ms = (yv.sum(1) / pops[l][idx[l]].sum(1))[:, None]
                noise_n[:, s] = pop_rate ** 2 * ms / pop_n[:, s]

        ncx = xp.asarray(nc); tgx = xp.asarray(tg)
        dmat = xp.sqrt((ncx[:, :, None, 0] - ncx[:, None, :, 0]) ** 2
                       + (ncx[:, :, None, 1] - ncx[:, None, :, 1]) ** 2)
        d0 = xp.sqrt(xp.sum((ncx - tgx[:, None, :]) ** 2, -1))
        rho_nn = _unit_correlation(dmat, lmc.range, lmc.model, xp)
        rho_0 = _unit_correlation(d0, lmc.range, lmc.model, xp)
        Bv = xp.asarray(lmc.B[np.ix_(var_of, var_of)])                       # (m, m)
        B0v = xp.asarray(lmc.B0[np.ix_(var_of, var_of)])
        zero_nn = dmat < 1e-9
        C = Bv[None] * rho_nn + B0v[None] * zero_nn
        # Poisson cross reliability at co-located data of different variables
        R = np.zeros((P, P))
        for l in range(P):
            for p in range(P):
                if l != p:
                    R[l, p] = _shared(shared_mean, l, p)
        if np.any(R):
            Rv = xp.asarray(R[np.ix_(var_of, var_of)])
            C = C + zero_nn * (pop_rate * Rv[None] / xp.asarray(pop_n)[:, :, None])
        ii = xp.arange(m)
        C[:, ii, ii] = C[:, ii, ii] + xp.asarray(noise_n)

        K = xp.zeros((nb, dim, dim))
        K[:, :m, :m] = C
        for l in range(P):
            K[:, offs[l]:offs[l + 1], m + l] = 1.0
            K[:, m + l, offs[l]:offs[l + 1]] = 1.0
        rhs = xp.zeros((nb, dim))
        bt = xp.asarray(lmc.B[target, var_of]); b0t = xp.asarray(lmc.B0[target, var_of])
        rhs[:, :m] = bt[None] * rho_0 + b0t[None] * (d0 < 1e-9)
        rhs[:, m + target] = 1.0
        try:
            w = xp.linalg.solve(K, rhs[:, :, None])[:, :, 0]
        except Exception:                                                     # singular block
            w = xp.stack([xp.linalg.lstsq(K[i], rhs[i], rcond=None)[0] for i in range(nb)])
        pred = xp.maximum(xp.sum(w[:, :m] * xp.asarray(rate_n), 1), 0.0)
        var = lmc.sill(target) - xp.sum(w * rhs, 1)
        out[sl_pred, 2] = to_numpy(pred)
        out[sl_pred, 3] = to_numpy(var)
    return out


def _cokrige_solve_knn_batch(datasets, target, lmc, pop_rate, shared_mean,
                             coords_pred, number_of_neighbors, use_gpu):
    """Backward-compatible wrapper (co-located k-NN prediction)."""
    return _cokrige_knn(datasets, target, lmc, pop_rate, shared_mean,
                        coords_pred, number_of_neighbors, use_gpu=use_gpu)


def poisson_cokrige(datasets, target, lmc, pop_rate=100000.0,
                    shared_mean=None, coords_pred=None, smooth=None,
                    number_of_neighbors=None, use_gpu="auto",
                    mean_rate="local", chunk_size=1024):
    """
    N-variate Poisson cokriging (generalises poisson.cokrige.bi/three/four).

    Parameters
    ----------
    datasets : list of dict
        One entry per variable, each with arrays ``x, y, cases, pop`` and an
        optional ``noise_var`` (measurement-error variance on the
        ``pop_rate**2`` scale, replacing the Poisson term). Element ``target``
        is the variable being predicted; the rest are auxiliary. The global
        path and the empirical cross-semivariogram need co-located variables;
        the k-NN path accepts variables observed at different locations.
    target : int
        Index of the target variable in ``datasets``.
    lmc : LMC
        Fitted coregionalization model.
    pop_rate : float
        Rate base (1 for proportions, 100 for percent, 100000 per 100k).
    shared_mean : dict
        ``{(l, k): mean shared risk R_lk}`` on the ``pop_rate`` scale, for the
        co-located cross reliability terms. Missing pairs default to 0.
    coords_pred : array (m, 2), optional
        Prediction locations. If omitted (or equal to the target locations),
        leave-one-out **smoothing** is performed instead.
    smooth : bool, optional
        Force smoothing (True) or prediction (False). Auto-detected when None.
    number_of_neighbors : int or list of int, optional
        Local k-NN cokriging: each variable contributes its ``k`` nearest data
        (a list gives one ``k`` per variable). Works for prediction and for
        leave-one-out smoothing, where the target datum at the smoothed area
        is dropped and co-located auxiliary data are kept. If omitted, the
        original **global** system (every known area) is used.
    use_gpu : bool or {'auto'}, default='auto'
        Solve the batched k-NN systems on the GPU via CuPy when usable.
    mean_rate : {"local", "global"}
        k-NN path only: ``m*_l`` in the Poisson term is the rate of the
        neighbours ("local", the earlier behaviour) or of all data ("global",
        as in the global path and the paper).
    chunk_size : int
        Maximum targets per batched solve on the k-NN path (reduced
        automatically so each stacked system stays near 2e7 elements).

    Returns
    -------
    dict of numpy arrays
        Prediction mode: ``x, y, pred, var``.
        Smoothing mode: ``rate_pred, rate_var, observed, residual, zscore``.
    """
    shared_mean = shared_mean or {}
    tgt = datasets[target]
    tcoords = np.column_stack([tgt["x"], tgt["y"]]).astype(float)

    if smooth is None:
        smooth = coords_pred is None or (
            np.shape(coords_pred) == tcoords.shape and np.allclose(coords_pred, tcoords))

    use_gpu_flag = (resolve_cupy_use_gpu(use_gpu)
                    if number_of_neighbors is not None else False)

    if not smooth:
        if number_of_neighbors is not None:
            res = _cokrige_knn(datasets, target, lmc, pop_rate, shared_mean,
                               coords_pred, number_of_neighbors, use_gpu_flag,
                               mean_rate=mean_rate, chunk_size=chunk_size)
        else:
            res = _cokrige_solve(datasets, target, lmc, pop_rate, shared_mean, coords_pred)
        return {"x": res[:, 0], "y": res[:, 1], "pred": res[:, 2], "var": res[:, 3]}

    m = len(tcoords)
    observed = np.asarray(tgt["cases"], float) / np.asarray(tgt["pop"], float) * pop_rate
    if number_of_neighbors is not None:
        res = _cokrige_knn(datasets, target, lmc, pop_rate, shared_mean,
                           tcoords, number_of_neighbors, use_gpu_flag,
                           exclude_target=np.arange(m), mean_rate=mean_rate,
                           chunk_size=chunk_size)
        rate_pred, rate_var = res[:, 2], res[:, 3]
    else:
        rate_pred = np.full(m, np.nan); rate_var = np.full(m, np.nan)
        for i in range(m):
            keep = np.ones(m, bool); keep[i] = False
            ds = []
            for l, d in enumerate(datasets):
                dd = dict(x=d["x"], y=d["y"], cases=d["cases"], pop=d["pop"],
                          noise_var=d.get("noise_var"))
                if l == target:
                    dd = {k: (None if v is None else np.asarray(v)[keep])
                          for k, v in dd.items()}
                ds.append(dd)
            r = _cokrige_solve(ds, target, lmc, pop_rate, shared_mean,
                               tcoords[i:i+1])
            rate_pred[i], rate_var[i] = r[0, 2], r[0, 3]
    residual = rate_pred - observed
    zscore = residual / np.sqrt(np.clip(rate_var, 1e-300, None))
    return {"rate_pred": rate_pred, "rate_var": rate_var, "observed": observed,
            "residual": residual, "zscore": zscore}

# ---------------------------------------------------------------------------
# Univariate Poisson kriging  (R: poisson.krige.one + .pred.one)
# ---------------------------------------------------------------------------
def poisson_krige(data, sill, rng, nugget=0.0, model="Exp",
                  pop_rate=100000.0, coords_pred=None, smooth=None,
                  number_of_neighbors=None, mean_rate="local", use_gpu="auto",
                  chunk_size=1024):
    """
    Ordinary univariate Poisson kriging (R: poisson.krige.one).

    ``data`` is a dict with ``x, y, cases, pop`` (optional ``noise_var``). The
    kriging matrix adds the Poisson error variance ``pop_rate**2 * m* / n_i``
    and the nugget to the diagonal. Prediction vs leave-one-out smoothing is
    auto-detected from ``coords_pred`` (see :func:`poisson_cokrige`). With
    ``number_of_neighbors`` it runs the batched local k-NN solver.
    """
    if number_of_neighbors is not None:
        lmc = LMC([[sill]], rng, model, [[nugget]])
        return poisson_cokrige([data], 0, lmc, pop_rate=pop_rate,
                               coords_pred=coords_pred, smooth=smooth,
                               number_of_neighbors=number_of_neighbors,
                               use_gpu=use_gpu, mean_rate=mean_rate,
                               chunk_size=chunk_size)
    covfun = covariance_model(model)
    x = np.asarray(data["x"], float); yc = np.asarray(data["y"], float)
    y = np.asarray(data["cases"], float); n = np.asarray(data["pop"], float)
    nv = data.get("noise_var")
    coords = np.column_stack([x, yc])

    if smooth is None:
        smooth = coords_pred is None or (
            np.shape(coords_pred) == coords.shape and np.allclose(coords_pred, coords))

    def solve(coords_tr, y_tr, n_tr, nv_tr, targets):
        size = len(coords_tr)
        D = squareform(pdist(coords_tr)) if size > 1 else np.zeros((1, 1))
        mstar = y_tr.sum() / n_tr.sum()
        noise = (pop_rate ** 2 * mstar / n_tr) if nv_tr is None else nv_tr
        C = covfun(D, rng, sill) + np.diag(noise) + np.diag(np.full(size, nugget))
        A = np.zeros((size + 1, size + 1))
        A[:size, :size] = C; A[:size, size] = 1; A[size, :size] = 1
        Ainv = np.linalg.inv(A)
        rates = y_tr / n_tr * pop_rate
        out = np.empty((len(np.atleast_2d(targets)), 4))
        for r, x0 in enumerate(np.atleast_2d(targets)):
            d0 = np.sqrt(((coords_tr - x0) ** 2).sum(1))
            c0 = covfun(d0, rng, sill)
            pos = np.isclose(d0, 0.0)
            if pos.any():
                c0 = c0.copy(); c0[pos] = c0[pos] + nugget
            rhs = np.append(c0, 1.0)
            lam = Ainv @ rhs
            pred = max(float((lam[:size] * rates).sum()), 0.0)
            var = (covfun(0.0, rng, sill) + nugget) - float((lam * rhs).sum())
            out[r] = [x0[0], x0[1], pred, var]
        return out

    if not smooth:
        res = solve(coords, y, n, None if nv is None else np.asarray(nv, float), coords_pred)
        return {"x": res[:, 0], "y": res[:, 1], "pred": res[:, 2], "var": res[:, 3]}

    m = len(coords)
    rate_pred = np.full(m, np.nan); rate_var = np.full(m, np.nan)
    observed = y / n * pop_rate
    for i in range(m):
        keep = np.ones(m, bool); keep[i] = False
        r = solve(coords[keep], y[keep], n[keep],
                  None if nv is None else np.asarray(nv, float)[keep], coords[i:i+1])
        rate_pred[i], rate_var[i] = r[0, 2], r[0, 3]
    residual = rate_pred - observed
    zscore = residual / np.sqrt(np.clip(rate_var, 1e-300, None))
    return {"rate_pred": rate_pred, "rate_var": rate_var, "observed": observed,
            "residual": residual, "zscore": zscore}
