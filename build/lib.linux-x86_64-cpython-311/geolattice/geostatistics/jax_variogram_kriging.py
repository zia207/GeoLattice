"""
JAX variograms and kriging for CPU and GPU.

Same functions as ``geolattice.geostatistics.core``: theoretical models, empirical
variograms (Matheron, Cressie–Hawkins, Dowd), variogram clouds, indicator
variograms, ordinary kriging, simple kriging, and block kriging. Pairwise
distances, binning, and the kriging solve run as JAX array operations on the
selected device.

``use_gpu`` follows the core classes:

* ``"auto"`` — GPU when JAX has one, otherwise CPU
* ``True`` — GPU, with a warning and a CPU fallback when no GPU is available
* ``False`` — CPU

Results are NumPy arrays. A JAX array passed to a model function stays on
its device.

>>> from geolattice.geostatistics.jax_variogram_kriging import Variogram, OrdinaryKriging
"""

from __future__ import annotations

import os

# Take only the memory a call needs. The default JAX pool reserves most of
# the GPU and fails when another process already holds it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma as gamma_fn
from scipy.optimize import minimize

from .core.variogram_cloud import VariogramCloud as _NumpyVariogramCloud
from .core.variogram_cloud import plot_variogram_cloud
from .utils.anisotropy import transform_aniso as _transform_aniso_host

jax.config.update("jax_enable_x64", True)

_GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def resolve_device(use_gpu="auto", device=None):
    """
    Pick a JAX device.

    Parameters
    ----------
    use_gpu : bool or ``"auto"``
        Ignored when ``device`` is given.
    device : jax.Device, optional
        Explicit device. Overrides ``use_gpu``.
    """
    if device is not None:
        return device
    gpus = [d for d in jax.devices() if d.platform in _GPU_PLATFORMS]
    cpus = jax.devices("cpu")
    if use_gpu == "auto":
        return gpus[0] if gpus else cpus[0]
    if bool(use_gpu):
        if not gpus:
            warnings.warn(
                "GPU requested (use_gpu=True) but JAX has no GPU device; using CPU.",
                stacklevel=3,
            )
            return cpus[0]
        return gpus[0]
    return cpus[0]


def to_numpy(arr):
    """Copy a JAX or NumPy array onto the host as a NumPy array."""
    return np.asarray(arr)


def _put(arr, device, dtype=jnp.float64):
    with jax.default_device(device):
        return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


def _is_gpu(device):
    return device.platform in _GPU_PLATFORMS


def _oom(exc):
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in message or "resource_exhausted" in message


def _float(x):
    """``x`` as a JAX float array. float32 input stays float32; anything else becomes float64."""
    x = jnp.asarray(x)
    return x if x.dtype in (jnp.float32, jnp.float64) else x.astype(jnp.float64)


# ---------------------------------------------------------------------------
# Theoretical models
# ---------------------------------------------------------------------------

def spherical(h, nugget, sill, range_):
    """Spherical semivariogram γ(h)."""
    h = _float(h)
    a = jnp.clip(h / range_, 0.0, 1.0)
    return nugget + sill * jnp.where(h < range_, 1.5 * a - 0.5 * a ** 3, 1.0)


def exponential(h, nugget, sill, range_):
    """Exponential semivariogram γ(h)."""
    h = _float(h)
    return nugget + sill * (1.0 - jnp.exp(-h / range_))


def gaussian(h, nugget, sill, range_):
    """Gaussian semivariogram γ(h)."""
    h = _float(h)
    return nugget + sill * (1.0 - jnp.exp(-(h / range_) ** 2))


def _matern_correlation(x, nu, n_terms=80):
    """
    Matérn correlation ``2**(1-ν) / Γ(ν) * x**ν * K_ν(x)``.

    ``x**ν K_ν(x)`` is computed from the power series of the modified Bessel
    function so the product stays finite at the origin and on the GPU.
    Smoothness within 1e-3 of an integer is shifted by that amount; Γ(1-ν)
    has a pole there and the series quotient is undefined.
    """
    x = _float(x)
    nu = jnp.asarray(nu, dtype=x.dtype)
    nu = jnp.where(jnp.abs(nu - jnp.round(nu)) < 1e-3, nu + 1e-3, nu)
    half2 = (x / 2.0) ** 2

    def series(a0):
        def body(k, carry):
            term, total = carry
            term = term * half2 / ((k + 1.0) * (a0 + k))
            return term, total + term

        ones = jnp.ones_like(x)
        _, total = jax.lax.fori_loop(0, n_terms, body, (ones, ones))
        return total

    left = (2.0 ** nu) / gamma_fn(1.0 - nu) * series(1.0 - nu)
    right = jnp.exp(2.0 * nu * jnp.log(jnp.maximum(x, 1e-30)) - nu * jnp.log(2.0))
    right = right / gamma_fn(1.0 + nu) * series(1.0 + nu)
    series_val = (jnp.pi / 2.0) / jnp.sin(nu * jnp.pi) * (left - right)

    mu = 4.0 * nu * nu
    inv = 1.0 / jnp.maximum(x, 1e-30)
    asym_poly = (
        1.0
        + (mu - 1.0) * inv / 8.0
        + (mu - 1.0) * (mu - 9.0) * inv ** 2 / 128.0
        + (mu - 1.0) * (mu - 9.0) * (mu - 25.0) * inv ** 3 / 3072.0
    )
    asym = jnp.exp(nu * jnp.log(jnp.maximum(x, 1e-30)) - x) * jnp.sqrt(jnp.pi / (2.0 * jnp.maximum(x, 1e-30))) * asym_poly
    xnu_k = jnp.where(x < 12.0, series_val, asym)
    return (2.0 ** (1.0 - nu) / gamma_fn(nu)) * xnu_k


def matern(h, nugget, sill, range_, nu=1.5):
    """Matérn semivariogram γ(h). ``γ(0) = nugget``."""
    h = _float(h)
    x = jnp.maximum(h / range_, 1e-8)
    corr = _matern_correlation(x, nu)
    gamma = nugget + sill * (1.0 - corr)
    return jnp.where(h < 1e-8, nugget, gamma)


def stable(h, nugget, sill, range_, alpha=1.0):
    """Stable (powered exponential) semivariogram γ(h)."""
    h = _float(h)
    alpha = jnp.clip(jnp.asarray(alpha, dtype=h.dtype), 0.01, 2.0)
    return nugget + sill * (1.0 - jnp.exp(-(h / range_) ** alpha))


def cubic(h, nugget, sill, range_):
    """
    Cubic semivariogram (Chilès & Delfiner).

    γ(h) = nugget + sill * (7u² − 8.75u³ + 3.5u⁵ − 0.75u⁷) for u = h/a < 1.
    """
    h = _float(h)
    u = h / range_
    gamma_u = 7.0 * u ** 2 - 8.75 * u ** 3 + 3.5 * u ** 5 - 0.75 * u ** 7
    return nugget + sill * jnp.where(h < range_, gamma_u, 1.0)


MODEL_FUNCS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "matern": matern,
    "stable": stable,
    "cubic": cubic,
}

# Every model here runs on the device selected by ``use_gpu``.
CPU_ONLY_MODELS = frozenset()


def covariance_from_variogram(h, nugget, sill, range_, model="spherical", **kwargs):
    """
    Covariance C(h) = C(0) − γ(h), with C(0) = nugget + sill restored at h = 0.
    """
    if model not in MODEL_FUNCS:
        raise ValueError(f"model must be one of {list(MODEL_FUNCS)}")
    h = _float(h)
    gamma_h = MODEL_FUNCS[model](h, nugget, sill, range_, **kwargs)
    c0 = nugget + sill
    return jnp.where(h == 0, c0, c0 - gamma_h)


def transform_aniso(coords, angle=0.0, ratio=1.0, device=None):
    """
    Rotate ``coords`` by ``angle`` degrees and scale the second axis by ``ratio``.

    Matches ``geolattice.geostatistics.utils.anisotropy.transform_aniso``. The result
    is a JAX array on ``device`` (the default device when ``device`` is None).
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    if device is not None:
        coords = jax.device_put(coords, device)
    theta = jnp.deg2rad(jnp.asarray(angle, dtype=jnp.float64))
    rotation = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]],
        dtype=jnp.float64,
    )
    scale = jnp.diag(jnp.array([1.0, ratio], dtype=jnp.float64))
    return coords @ rotation @ scale


def _pairwise_upper(coords, values):
    """Upper-triangle distances and value differences. Diagonal excluded."""
    n = coords.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    diff = coords[iu] - coords[ju]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    dval = values[iu] - values[ju]
    return dist, dval, iu, ju


def _pairwise(a, b, same):
    """Euclidean distances between rows, via the Gram matrix.

    float32 inputs are measured in float64 and the result cast back, so the
    test for coincident points (distance 0 gets C(0), nugget included) gives
    the same answer in both precisions. The product has only two coordinates
    per point, so float64 costs little even on a GPU.
    """
    out_dtype = jnp.result_type(a, b)
    if out_dtype == jnp.float32:
        return _pairwise(a.astype(jnp.float64), b.astype(jnp.float64), same).astype(jnp.float32)
    a2 = jnp.sum(a * a, axis=1)
    b2 = jnp.sum(b * b, axis=1)
    # HIGHEST keeps GPUs from running a float32 product in TF32, which would
    # lose the small differences between large squared norms.
    ab = jnp.matmul(a, b.T, precision=jax.lax.Precision.HIGHEST)
    d2 = jnp.maximum(a2[:, None] + b2[None, :] - 2.0 * ab, 0.0)
    if same:
        d2 = d2.at[jnp.diag_indices(a.shape[0])].set(0.0)
    # ||a||^2 + ||a||^2 - 2 a·a is not always a hard zero. Isotopic cokriging
    # needs those pairs to keep C(0), the same way cdist does.
    scale = jnp.maximum(jnp.maximum(jnp.max(a2), jnp.max(b2)), 1.0)
    d2 = jnp.where(d2 <= 1e-10 * scale, 0.0, d2)
    return jnp.sqrt(d2)


def _apply_model(model, h, params):
    params = tuple(float(p) for p in params)
    if model == "matern":
        return matern(h, *params[:3], nu=params[3] if len(params) > 3 else 1.5)
    if model == "stable":
        return stable(h, *params[:3], alpha=params[3] if len(params) > 3 else 1.0)
    return MODEL_FUNCS[model](h, *params[:3])


def _cov_from_gamma(gamma, c0, dist):
    return jnp.where(dist == 0, c0, c0 - gamma)


# jax.numpy.linalg has no LinAlgError in current JAX; a singular solve shows up as non-finite weights.
_SOLVE_ERRORS = tuple({FloatingPointError, np.linalg.LinAlgError, getattr(jnp.linalg, "LinAlgError", FloatingPointError)})


def _solve(matrix, rhs):
    """Solve ``matrix @ x = rhs``. Fall back to least squares when singular."""
    try:
        solution = jnp.linalg.solve(matrix, rhs)
        solution.block_until_ready()
        if not bool(jnp.all(jnp.isfinite(solution))):
            raise FloatingPointError("kriging matrix solve produced non-finite weights")
        return solution
    except _SOLVE_ERRORS:
        return jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]


# ---------------------------------------------------------------------------
# Factored kriging
#
# Simple, ordinary, universal and cokriging share one generalized least
# squares form: a covariance matrix C over the data and, except for simple
# kriging, a drift matrix F whose columns are the unbiasedness constraints
# (a column of ones for ordinary kriging, polynomial terms for universal
# kriging, one indicator per variable for cokriging). ``fit`` factors
# C = L Lᵀ (Cholesky) once and keeps on the device
#
#   W = L⁻¹F,   Q = Wᵀ W = FᵀC⁻¹F,   b = Q⁻¹ Wᵀ L⁻¹ y,   a = C⁻¹(y − F b)
#
# so that, for a prediction point with covariances c to the data and drift f,
#
#   prediction = cᵀa + fᵀb   (or vᵀ L⁻¹(y − F b) + fᵀb, the same number)
#   variance   = C(0) − vᵀv + uᵀQ⁻¹u,   v = L⁻¹c,   u = f − Wᵀv.
#
# These are exactly the predictions and variances of the bordered kriging
# system the older code solved for every call, but a prediction batch now
# costs one matrix product and one triangular solve instead of an LU
# factorization of the whole system. The kernels are compiled with
# ``jax.jit``; variogram parameters are traced arguments and prediction
# batches have a fixed size, so new parameters or a new number of points do
# not trigger a recompile. When the Cholesky factor is not finite (for
# example duplicate locations with no nugget) the classes fall back to the
# bordered solve with a least-squares fallback.
# ---------------------------------------------------------------------------

_DTYPES = {None: jnp.float64, "float64": jnp.float64, "float32": jnp.float32,
           np.float64: jnp.float64, np.float32: jnp.float32}


def _resolve_dtype(dtype):
    try:
        return _DTYPES[dtype]
    except (KeyError, TypeError):
        raise ValueError("dtype must be 'float64' or 'float32'") from None


def _params_array(params, dtype):
    return jnp.asarray([float(p) for p in params], dtype=dtype)


def _gamma_traced(model, h, params):
    """γ(h) with the variogram parameters as a traced array (no host round trip)."""
    if model == "matern":
        nu = params[3] if params.shape[0] > 3 else 1.5
        return matern(h, params[0], params[1], params[2], nu=nu)
    if model == "stable":
        alpha = params[3] if params.shape[0] > 3 else 1.0
        return stable(h, params[0], params[1], params[2], alpha=alpha)
    return MODEL_FUNCS[model](h, params[0], params[1], params[2])


def _cov_traced(model, a, b, params, same):
    dist = _pairwise(a, b, same)
    c0 = params[0] + params[1]
    return jnp.where(dist == 0, c0, c0 - _gamma_traced(model, dist, params))


def _gls_factor(cov, y, drift):
    """Cholesky factor and dual weights. ``drift`` is None for simple kriging."""
    chol = jax.scipy.linalg.cholesky(cov, lower=True)
    ly = jax.scipy.linalg.solve_triangular(chol, y, lower=True)
    if drift is None:
        a = jax.scipy.linalg.solve_triangular(chol.T, ly, lower=False)
        return chol, a, None, None, None, ly
    w = jax.scipy.linalg.solve_triangular(chol, drift, lower=True)
    q_chol = jax.scipy.linalg.cholesky(w.T @ w, lower=True)
    b = jax.scipy.linalg.cho_solve((q_chol, True), w.T @ ly)
    r = ly - w @ b  # whitened residual L⁻¹(y − F b)
    a = jax.scipy.linalg.solve_triangular(chol.T, r, lower=False)
    return chol, a, b, w, q_chol, r


def _gls_predict(c, f, c0, chol, a, b, w, q_chol, r, with_var):
    """Predictions (and variances) for a batch with covariances ``c`` (m × n).

    Without variances in float64 the prediction is the dual form cᵀa (one
    matrix-vector product). When v = L⁻¹c is computed anyway (variances) or
    the arithmetic is float32, it is the whitened form vᵀr, which avoids the
    cancellation in cᵀa when the dual weights a are large and alternate in sign.
    """
    whiten = with_var or c.dtype == jnp.float32
    if whiten:
        v = jax.scipy.linalg.solve_triangular(chol, c.T, lower=True)
        pred = v.T @ r
    else:
        pred = c @ a
    if b is not None:
        pred = pred + f @ b
    if not with_var:
        return pred, jnp.zeros_like(pred)
    var = c0 - jnp.sum(v * v, axis=0)
    if b is not None:
        u = f.T - w.T @ v
        var = var + jnp.sum(u * jax.scipy.linalg.cho_solve((q_chol, True), u), axis=0)
    return pred, jnp.sqrt(jnp.clip(var, 0.0, None))


from functools import partial  # noqa: E402


@partial(jax.jit, static_argnames=("model",))
def _factor_single(x, y, drift, params, model):
    return _gls_factor(_cov_traced(model, x, x, params, True), y, drift)


@partial(jax.jit, static_argnames=("model",))
def _factor_single_sk(x, y, params, model):
    return _gls_factor(_cov_traced(model, x, x, params, True), y, None)


@partial(jax.jit, static_argnames=("model", "with_var"))
def _predict_single(x_pred, f_pred, x, chol, a, b, w, q_chol, r, params, model, with_var):
    c = _cov_traced(model, x_pred, x, params, False)
    return _gls_predict(c, f_pred, params[0] + params[1], chol, a, b, w, q_chol, r, with_var)


@partial(jax.jit, static_argnames=("model", "with_var"))
def _predict_single_sk(x_pred, x, chol, a, r, params, model, with_var):
    c = _cov_traced(model, x_pred, x, params, False)
    return _gls_predict(c, None, params[0] + params[1], chol, a, None, None, None, r, with_var)


def _co_cov_train(xp, xs, pp, ps, pc, mp, ms, mc):
    c_pp = _cov_traced(mp, xp, xp, pp, True)
    c_ss = _cov_traced(ms, xs, xs, ps, True)
    c_ps = _cov_traced(mc, xp, xs, pc, False)
    return _jitter_diag(jnp.block([[c_pp, c_ps], [c_ps.T, c_ss]]))


@partial(jax.jit, static_argnames=("mp", "ms", "mc"))
def _factor_co(xp, xs, values, drift, pp, ps, pc, mp, ms, mc):
    return _gls_factor(_co_cov_train(xp, xs, pp, ps, pc, mp, ms, mc), values, drift)


@partial(jax.jit, static_argnames=("mp", "mc", "with_var"))
def _predict_co(x_pred, f_pred, xp, xs, chol, a, b, w, q_chol, r, pp, pc, mp, mc, with_var):
    c = jnp.concatenate([_cov_traced(mp, x_pred, xp, pp, False), _cov_traced(mc, x_pred, xs, pc, False)], axis=1)
    return _gls_predict(c, f_pred, pp[0] + pp[1], chol, a, b, w, q_chol, r, with_var)


def _all_finite(*arrays):
    return all(bool(jnp.all(jnp.isfinite(arr))) for arr in arrays if arr is not None)


def _pred_batch(n_train, n_pred, dtype):
    """Fixed batch size: a power of two that keeps one m × n block near 64 MiB."""
    itemsize = jnp.dtype(dtype).itemsize
    cap = max(256, (64 * 1024 * 1024) // max(n_train * itemsize, 1))
    batch = 256
    while batch * 2 <= cap and batch < n_pred:
        batch *= 2
    return batch


def _run_batched(fn, x_pred, f_pred, batch, device, dtype):
    """Call ``fn(x_batch, f_batch)`` over fixed-size, padded batches; return NumPy."""
    m = len(x_pred)
    preds, ses = [], []
    for start in range(0, m, batch):
        xb = x_pred[start:start + batch]
        fb = None if f_pred is None else f_pred[start:start + batch]
        pad = batch - len(xb)
        if pad:  # repeat the last point so every call has the compiled shape
            xb = np.concatenate([xb, np.repeat(xb[-1:], pad, axis=0)])
            if fb is not None:
                fb = np.concatenate([fb, np.repeat(fb[-1:], pad, axis=0)])
        xb = _put(xb, device, dtype)
        fb = None if fb is None else _put(fb, device, dtype)
        p, s = fn(xb, fb)
        preds.append(p[:batch - pad] if pad else p)
        ses.append(s[:batch - pad] if pad else s)
    pred = np.asarray(jnp.concatenate(preds)) if preds else np.empty(0)
    se = np.asarray(jnp.concatenate(ses)) if ses else np.empty(0)
    return pred, se


# ---------------------------------------------------------------------------
# Empirical variogram
# ---------------------------------------------------------------------------

def _empirical_gamma(dist, dval, maxlag, n_lags, estimator):
    """
    Bin semivariances. Bins are (left, right], matching ``Variogram``.
    """
    edges = jnp.linspace(0.0, maxlag, n_lags + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = jnp.searchsorted(edges, dist, side="left") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_lags) & jnp.isfinite(dval)
    safe = jnp.clip(bin_idx, 0, n_lags - 1)
    counts = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, 1.0, 0.0))

    if estimator == "matheron":
        numer = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, dval ** 2, 0.0))
        gamma = jnp.where(counts > 0, 0.5 * numer / jnp.maximum(counts, 1.0), jnp.nan)
    elif estimator == "cressie":
        numer = jnp.zeros(n_lags).at[safe].add(
            jnp.where(valid, jnp.sqrt(jnp.abs(dval)), 0.0)
        )
        # The mean is unchanged if both triangles are kept, but Cressie's
        # finite-sample term depends on the pair count. The core estimator
        # counts (i, j) and (j, i), so N is twice the upper triangle.
        term = numer / jnp.maximum(counts, 1.0)
        n_pairs = jnp.maximum(counts * 2.0, 1.0)
        gamma = jnp.where(
            counts > 0,
            0.5 * term ** 4 / (0.457 + 0.494 / n_pairs),
            jnp.nan,
        )
    elif estimator == "dowd":
        def one_bin(i):
            mask = valid & (bin_idx == i)
            d2 = jnp.where(mask, dval ** 2, jnp.nan)
            return 0.5 * 2.198 * jnp.nanmedian(d2)

        gamma = jax.vmap(one_bin)(jnp.arange(n_lags))
    else:
        raise ValueError(f"estimator must be one of {list(ESTIMATORS)}")
    return centers, gamma


ESTIMATORS = ("matheron", "cressie", "dowd")


class Variogram:
    """
    Empirical variogram and a fitted theoretical model, evaluated with JAX.

    Parameters match ``geolattice.geostatistics.core.variogram.Variogram``.
    ``use_gpu`` selects the JAX device used to form pairs and bins.
    """

    VALID_MODELS = ["spherical", "exponential", "gaussian", "matern", "stable", "cubic"]

    def __init__(
        self,
        coords,
        values,
        model="spherical",
        estimator="matheron",
        maxlag=None,
        n_lags=10,
        anisotropy=None,
        use_gpu="auto",
        device=None,
    ):
        if model not in self.VALID_MODELS:
            raise ValueError(f"model must be one of {self.VALID_MODELS}")
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {list(ESTIMATORS)}")

        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float).reshape(-1)
        if self.coords.ndim != 2:
            raise ValueError("coords must be a 2D array of shape (n, dim)")
        if len(self.values) != len(self.coords):
            raise ValueError("coords and values must have the same length")
        if len(self.values) < 2:
            raise ValueError("at least 2 observations are required to compute a variogram")

        self.model = model
        self.estimator = estimator
        self.maxlag = maxlag
        self.n_lags = int(n_lags)
        self.anisotropy = anisotropy
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)

        if self.anisotropy:
            self.coords = np.asarray(
                _transform_aniso_host(self.coords, **self.anisotropy), dtype=float
            )

        self.lags = None
        self.experimental = None
        self.fitted_params = None

    def _pairs(self):
        with jax.default_device(self.device):
            coords = _put(self.coords, self.device)
            values = _put(self.values, self.device)
            return _pairwise_upper(coords, values)

    def _compute_empirical(self):
        dist, dval, _, _ = self._pairs()
        dist_host = np.asarray(dist)
        positive = dist_host[dist_host > 0]
        if positive.size == 0:
            raise ValueError("All locations are coincident; cannot estimate a variogram")
        if self.maxlag is None:
            # Both triangles, matching the NumPy squareform percentile.
            both = np.concatenate([positive, positive])
            self.maxlag = float(np.percentile(both, 50))

        with jax.default_device(self.device):
            centers, gamma = _empirical_gamma(
                dist, dval, float(self.maxlag), self.n_lags, self.estimator
            )
        self.lags = np.asarray(centers)
        self.experimental = np.asarray(gamma)
        return self

    def fit(self, method="auto"):
        """Estimate the empirical variogram and, unless ``method='manual'``, fit the model."""
        try:
            self._compute_empirical()
        except Exception as exc:
            if self.use_gpu and _oom(exc):
                warnings.warn("Variogram ran out of GPU memory; recomputing on CPU.", stacklevel=2)
                self.device = resolve_device(False)
                self.use_gpu = False
                self._compute_empirical()
            else:
                raise

        if method == "auto":
            self._fit_model()
        elif method == "manual":
            self.fitted_params = None
        else:
            raise ValueError("method must be 'auto' or 'manual'")
        return self

    def set_params(self, nugget, sill, range_, nu=None, alpha=None):
        """
        Set theoretical model parameters.

        Order is ``nugget, sill, range_``, plus ``nu`` (Matérn) or ``alpha`` (stable).
        """
        params = [float(nugget), float(sill), float(range_)]
        if self.model == "matern":
            if nu is None:
                raise ValueError("matern requires nu")
            params.append(float(nu))
        elif self.model == "stable":
            if alpha is None:
                raise ValueError("stable requires alpha")
            params.append(float(alpha))
        elif nu is not None or alpha is not None:
            raise ValueError(f"{self.model} does not take nu or alpha")
        self.fitted_params = params
        return self

    def _fit_model(self):
        """Least-squares fit of the theoretical model. The optimizer runs on the host."""
        sill0 = float(np.nanvar(self.values))
        range0 = self.maxlag / 2.0
        nugget0 = 0.0

        if self.model == "matern":
            x0 = [nugget0, sill0, range0, 1.5]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3), (0.1, 10)]
        elif self.model == "stable":
            x0 = [nugget0, sill0, range0, 1.0]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3), (0.01, 2.0)]
        else:
            x0 = [nugget0, sill0, range0]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3)]

        lags = self.lags
        obs = self.experimental
        model = self.model
        device = self.device

        def obj(params):
            with jax.default_device(device):
                pred = np.asarray(_apply_model(model, lags, params))
            mask = ~np.isnan(obs)
            if not np.any(mask):
                return np.inf
            return float(np.sum((pred[mask] - obs[mask]) ** 2))

        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        self.fitted_params = [float(v) for v in (res.x if res.success else x0)]
        self.fit_success_ = bool(res.success)
        return self

    def __call__(self, h):
        if self.fitted_params is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        on_device = isinstance(h, jax.Array)
        device = h.devices().pop() if on_device and h.devices() else self.device
        with jax.default_device(device):
            out = _apply_model(self.model, h, self.fitted_params)
        if on_device:
            return out
        return np.asarray(out)


# ---------------------------------------------------------------------------
# Variogram cloud and indicator variogram
# ---------------------------------------------------------------------------

def compute_variogram_cloud(coords, values, maxlag=None, use_gpu="auto", device=None):
    """
    Pairwise distances and semivariances on the selected JAX device.

    Returns NumPy arrays ``(dist, gamma, left, right)``, upper triangle only.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(coords):
        raise ValueError("coords and values must have same length")
    device = resolve_device(use_gpu, device)
    with jax.default_device(device):
        dist, dval, left, right = _pairwise_upper(_put(coords, device), _put(values, device))
        gamma = 0.5 * dval ** 2
        if maxlag is not None:
            mask = dist <= maxlag
            dist, gamma, left, right = dist[mask], gamma[mask], left[mask], right[mask]
    return np.asarray(dist), np.asarray(gamma), np.asarray(left), np.asarray(right)


class VariogramCloud(_NumpyVariogramCloud):
    """Variogram cloud. ``from_coords`` builds the pairs with JAX."""

    @classmethod
    def from_coords(cls, coords, values, maxlag=None, use_gpu="auto", device=None):
        dist, gamma, left, right = compute_variogram_cloud(
            coords, values, maxlag=maxlag, use_gpu=use_gpu, device=device
        )
        return cls(dist=dist, gamma=gamma, left=left, right=right)


def fit_indicator_variogram(
    coords, values, threshold, maxlag=None, n_lags=10, use_gpu="auto", device=None
):
    """
    Empirical indicator variogram of ``I(Z <= threshold)``.

    Bins are ``[lower, upper)``, matching ``core.indicator_variogram``.
    Returns NumPy ``(lags, experimental_gamma)``.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array with shape (n_points, dim)")
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values must be a 1D array with the same length as coords")

    device = resolve_device(use_gpu, device)
    n_lags = int(n_lags)
    with jax.default_device(device):
        indicators = (_put(values, device) <= threshold).astype(jnp.float64)
        dist, dval, _, _ = _pairwise_upper(_put(coords, device), indicators)
        if dist.shape[0] == 0:
            return np.array([]), np.array([])
        if maxlag is None:
            maxlag = float(np.asarray(jnp.max(dist)) / 2.0)
        edges = jnp.linspace(0.0, float(maxlag), n_lags + 1)
        lags = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = jnp.searchsorted(edges, dist, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_lags)
        safe = jnp.clip(bin_idx, 0, max(n_lags - 1, 0))
        numer = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, dval ** 2, 0.0))
        counts = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, 1.0, 0.0))
        gamma = jnp.where(counts > 0, 0.5 * numer / jnp.maximum(counts, 1.0), jnp.nan)
    return np.asarray(lags), np.asarray(gamma)


# ---------------------------------------------------------------------------
# Kriging
# ---------------------------------------------------------------------------

def _gpu_pred_batch(n_train, n_pred):
    """Prediction batch that keeps one cross-covariance under 16 MiB."""
    if n_train <= 0 or n_pred <= 0:
        return max(n_pred, 1)
    batch = (16 * 1024 * 1024) // (n_train * 8)
    return max(1, min(n_pred, batch))


def _apply_variogram_anisotropy(variogram, coords):
    coords = np.asarray(coords, dtype=float)
    aniso = getattr(variogram, "anisotropy", None)
    if aniso:
        return np.asarray(_transform_aniso_host(coords, **aniso), dtype=float)
    return coords


class _FactoredKriging:
    """Factor once in ``fit``, predict with compiled kernels (see "Factored kriging")."""

    def _factor_with_fallback(self, build):
        """Run ``build()`` (which returns the factor tuple) on the device.

        A GPU out-of-memory error moves the model to the CPU. A factor that is
        not finite (C not positive definite) disables the fast path, and
        ``predict`` then solves the bordered system as before.
        """
        self._factor = None
        try:
            factor = build()
        except Exception as exc:
            if not (self.use_gpu and _oom(exc)):
                raise
            warnings.warn("Kriging ran out of GPU memory; using CPU.", stacklevel=3)
            self.device, self.use_gpu = resolve_device(False), False
            factor = build()
        if _all_finite(*factor):
            self._factor = factor

    def _with_cpu_retry(self, fn):
        try:
            return fn()
        except Exception as exc:
            if not (self.use_gpu and _oom(exc)):
                raise
            warnings.warn("Kriging ran out of GPU memory; predicting on CPU.", stacklevel=3)
            saved_device, saved_gpu = self.device, self.use_gpu
            self.device, self.use_gpu = resolve_device(False), False
            try:
                if getattr(self, "_factor", None) is not None:
                    self._factor = tuple(None if t is None else jax.device_put(t, self.device) for t in self._factor)
                    self._refresh_device_arrays()
                return fn()
            finally:
                self.device, self.use_gpu = saved_device, saved_gpu

    def _refresh_device_arrays(self):
        """Move cached training arrays to ``self.device`` (after a CPU fallback)."""


class OrdinaryKriging(_FactoredKriging):
    """
    Ordinary kriging on a JAX device (CPU or GPU).

    ``fit`` builds the covariance matrix of the data and factors it once
    (Cholesky); ``predict`` then runs compiled kernels on fixed-size batches,
    so repeated predictions only pay for the cross-covariances. ``variogram``
    is a fitted :class:`Variogram` (this module or ``core.variogram``).
    Predictions are NumPy arrays. A GPU allocation failure retries on CPU.

    ``dtype="float32"`` runs the factorization and the predictions in single
    precision (distances are still measured in float64). Most consumer and
    laptop GPUs run float64 at a small fraction of their float32 speed, so
    with ``return_variance=True`` this is often an order of magnitude faster
    on such GPUs; for predictions alone float64 is already one matrix-vector
    product per batch and float32 gains little. On 5,000 soil
    organic carbon samples (values 0 to 24) float32 predictions were within
    0.006 of float64 and standard errors within 4e-5; check the difference on
    your own data before relying on it.
    """

    def __init__(self, variogram, use_gpu="auto", device=None, dtype=None):
        self.variogram = variogram
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)
        self.dtype = _resolve_dtype(dtype)

    def fit(self, X, y):
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        self.X_train = _apply_variogram_anisotropy(self.variogram, X)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")
        self._factor_with_fallback(self._build_factor)
        return self

    def _build_factor(self):
        self._refresh_device_arrays()
        self._params = _put(np.asarray(self.variogram.fitted_params, dtype=float), self.device, self.dtype)
        ones = _put(np.ones((len(self.y_train), 1)), self.device, self.dtype)
        with jax.default_device(self.device):
            return _factor_single(self._x_dev, self._y_dev, ones, self._params, self.variogram.model)

    def _refresh_device_arrays(self):
        # distances do not change under a shift; centring keeps them accurate in float32
        self._center = self.X_train.mean(axis=0)
        self._x_dev = _put(self.X_train - self._center, self.device, self.dtype)
        self._y_dev = _put(self.y_train, self.device, self.dtype)
        if hasattr(self, "_params"):
            self._params = jax.device_put(self._params, self.device)

    def predict(self, X_pred, return_variance=False, block_size=None):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if block_size is not None:
            return self._predict_block(X_pred, return_variance, block_size)
        if getattr(self, "_factor", None) is not None:
            return self._predict_fast(np.asarray(X_pred, dtype=float), return_variance)
        return self._predict_point(X_pred, return_variance)

    def _predict_fast(self, X_pred, return_variance):
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred) - self._center
        ones = np.ones((len(X_pred), 1))

        def run():
            chol, a, b, w, q_chol, r = self._factor
            model = self.variogram.model

            def fn(xb, fb):
                return _predict_single(xb, fb, self._x_dev, chol, a, b, w, q_chol, r,
                                       self._params, model, bool(return_variance))

            batch = _pred_batch(len(self.X_train), len(X_pred), self.dtype)
            with jax.default_device(self.device):
                return _run_batched(fn, X_pred, ones, batch, self.device, self.dtype)

        pred, se = self._with_cpu_retry(run)
        return (pred, se) if return_variance else pred

    def _predict_point(self, X_pred, return_variance):
        X_pred = np.asarray(X_pred, dtype=float)
        batch = _gpu_pred_batch(len(self.X_train), len(X_pred)) if self.use_gpu else len(X_pred)
        try:
            if batch >= len(X_pred):
                return self._predict_point_arrays(X_pred, return_variance)
            parts = [
                self._predict_point_arrays(X_pred[start:start + batch], return_variance)
                for start in range(0, len(X_pred), batch)
            ]
        except Exception as exc:
            if not (self.use_gpu and _oom(exc)):
                raise
            warnings.warn("Kriging ran out of GPU memory; predicting on CPU.", stacklevel=2)
            saved_device, saved_gpu = self.device, self.use_gpu
            self.device, self.use_gpu = resolve_device(False), False
            try:
                return self._predict_point_arrays(X_pred, return_variance)
            finally:
                self.device, self.use_gpu = saved_device, saved_gpu
        if not return_variance:
            return np.concatenate(parts)
        pred = np.concatenate([part[0] for part in parts])
        se = np.concatenate([part[1] for part in parts])
        return pred, se

    def _predict_point_arrays(self, X_pred, return_variance):
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred)
        with jax.default_device(self.device):
            pred, se = _krige_ordinary(
                _put(self.X_train, self.device),
                _put(self.y_train, self.device),
                _put(X_pred, self.device),
                self.variogram.model,
                self.variogram.fitted_params,
            )
        pred = np.asarray(pred)
        if not return_variance:
            return pred
        return pred, np.asarray(se)

    def _predict_block(self, X_centers, return_variance, block_size):
        """
        Block kriging by Monte Carlo average of point predictions inside each block.

        The reported block variance is the mean of the point kriging variances.
        """
        X_centers = np.asarray(X_centers, dtype=float)
        n_blocks = len(X_centers)
        n_samples = 50
        block_preds = np.empty(n_blocks)
        block_vars = np.empty(n_blocks) if return_variance else None

        for i, center in enumerate(X_centers):
            offsets = np.random.uniform(-block_size / 2, block_size / 2, size=(n_samples, 2))
            block_points = np.asarray(center, dtype=float) + offsets
            if return_variance:
                preds, stds = self.predict(block_points, return_variance=True)
                block_preds[i] = np.mean(preds)
                block_vars[i] = np.mean(stds ** 2)
            else:
                preds = self.predict(block_points, return_variance=False)
                block_preds[i] = np.mean(preds)

        if return_variance:
            return block_preds, np.sqrt(block_vars)
        return block_preds


class SimpleKriging(_FactoredKriging):
    """
    Simple kriging with a known mean. See :class:`OrdinaryKriging` for
    ``use_gpu``, ``dtype`` and how ``fit`` and ``predict`` divide the work.
    """

    def __init__(self, variogram, mean=None, use_gpu="auto", device=None, dtype=None):
        self.variogram = variogram
        self.mean = mean
        self._mean_provided = mean is not None
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)
        self.dtype = _resolve_dtype(dtype)

    def fit(self, X, y):
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        self.X_train = _apply_variogram_anisotropy(self.variogram, X)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")
        if not self._mean_provided:
            self.mean = float(np.mean(self.y_train))
        self._factor_with_fallback(self._build_factor)
        return self

    def _build_factor(self):
        self._refresh_device_arrays()
        self._params = _put(np.asarray(self.variogram.fitted_params, dtype=float), self.device, self.dtype)
        resid = _put(self.y_train - float(self.mean), self.device, self.dtype)
        with jax.default_device(self.device):
            return _factor_single_sk(self._x_dev, resid, self._params, self.variogram.model)

    def _refresh_device_arrays(self):
        self._center = self.X_train.mean(axis=0)
        self._x_dev = _put(self.X_train - self._center, self.device, self.dtype)
        if hasattr(self, "_params"):
            self._params = jax.device_put(self._params, self.device)

    def predict(self, X_pred, return_variance=False):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred)
        if getattr(self, "_factor", None) is not None:
            def run():
                chol, a, _, _, _, r = self._factor
                model = self.variogram.model

                def fn(xb, _fb):
                    return _predict_single_sk(xb, self._x_dev, chol, a, r, self._params, model, bool(return_variance))

                batch = _pred_batch(len(self.X_train), len(X_pred), self.dtype)
                with jax.default_device(self.device):
                    return _run_batched(fn, X_pred - self._center, None, batch, self.device, self.dtype)

            pred, se = self._with_cpu_retry(run)
            pred = pred + float(self.mean)
            return (pred, se) if return_variance else pred
        try:
            return self._predict_arrays(X_pred, return_variance)
        except Exception as exc:
            if not (self.use_gpu and _oom(exc)):
                raise
            warnings.warn("Kriging ran out of GPU memory; predicting on CPU.", stacklevel=2)
            saved_device, saved_gpu = self.device, self.use_gpu
            self.device, self.use_gpu = resolve_device(False), False
            try:
                return self._predict_arrays(X_pred, return_variance)
            finally:
                self.device, self.use_gpu = saved_device, saved_gpu

    def _predict_arrays(self, X_pred, return_variance):
        with jax.default_device(self.device):
            pred, se = _krige_simple(
                _put(self.X_train, self.device),
                _put(self.y_train, self.device),
                _put(X_pred, self.device),
                float(self.mean),
                self.variogram.model,
                self.variogram.fitted_params,
            )
        pred = np.asarray(pred)
        if not return_variance:
            return pred
        return pred, np.asarray(se)


def _krige_ordinary(x_train, y_train, x_pred, model, params):
    n, m = x_train.shape[0], x_pred.shape[0]
    dist_train = _pairwise(x_train, x_train, same=True)
    dist_pred = _pairwise(x_pred, x_train, same=False)
    c0 = float(params[0]) + float(params[1])
    c_train = _cov_from_gamma(_apply_model(model, dist_train, params), c0, dist_train)
    c_pred = _cov_from_gamma(_apply_model(model, dist_pred, params), c0, dist_pred)

    ones_n = jnp.ones((n, 1))
    matrix = jnp.block([
        [c_train, ones_n],
        [ones_n.T, jnp.zeros((1, 1))],
    ])
    rhs = jnp.concatenate([c_pred, jnp.ones((m, 1))], axis=1)
    weights = _solve(matrix, rhs.T).T
    pred = weights[:, :-1] @ y_train
    var = c0 - jnp.sum(weights[:, :-1] * c_pred, axis=1) - weights[:, -1]
    se = jnp.sqrt(jnp.clip(var, 0.0, None))
    return pred, se


def _krige_simple(x_train, y_train, x_pred, mean, model, params):
    dist_train = _pairwise(x_train, x_train, same=True)
    dist_pred = _pairwise(x_pred, x_train, same=False)
    c0 = float(params[0]) + float(params[1])
    c_train = _cov_from_gamma(_apply_model(model, dist_train, params), c0, dist_train)
    c_pred = _cov_from_gamma(_apply_model(model, dist_pred, params), c0, dist_pred)
    weights = _solve(c_train, c_pred.T).T
    pred = mean + weights @ (y_train - mean)
    var = c0 - jnp.sum(weights * c_pred, axis=1)
    se = jnp.sqrt(jnp.clip(var, 0.0, None))
    return pred, se


class UniversalKriging(_FactoredKriging):
    """
    Universal kriging with a polynomial drift.

    The drift matrix is built with scikit-learn ``PolynomialFeatures`` on the
    original coordinates, matching ``kriging.universal_kriging.UniversalKriging``.
    Distances and the solve run on the selected JAX device.
    """

    def __init__(self, variogram, degree=1, use_gpu="auto", device=None, dtype=None):
        self.variogram = variogram
        self.degree = int(degree)
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)
        self.dtype = _resolve_dtype(dtype)

    def fit(self, X, y):
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        from sklearn.preprocessing import PolynomialFeatures

        self.X_train_orig = np.asarray(X, dtype=float)
        self.X_train = _apply_variogram_anisotropy(self.variogram, self.X_train_orig)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.F = self._poly.fit_transform(self.X_train_orig)
        # The fast path builds the drift from centred, scaled coordinates. A
        # polynomial in affinely transformed coordinates spans the same space,
        # so predictions are unchanged, and F^T C^-1 F stays well scaled (which
        # float32 needs when coordinates are large, such as metres).
        self._drift_shift = self.X_train_orig.mean(axis=0)
        spread = self.X_train_orig.std(axis=0)
        self._drift_scale = np.where(spread > 0, spread, 1.0)
        self._poly_std = PolynomialFeatures(degree=self.degree, include_bias=True)
        self._F_std = self._poly_std.fit_transform((self.X_train_orig - self._drift_shift) / self._drift_scale)
        self._factor_with_fallback(self._build_factor)
        return self

    def _build_factor(self):
        self._refresh_device_arrays()
        self._params = _put(np.asarray(self.variogram.fitted_params, dtype=float), self.device, self.dtype)
        with jax.default_device(self.device):
            return _factor_single(self._x_dev, self._y_dev, self._f_dev, self._params, self.variogram.model)

    def _refresh_device_arrays(self):
        self._center = self.X_train.mean(axis=0)
        self._x_dev = _put(self.X_train - self._center, self.device, self.dtype)
        self._y_dev = _put(self.y_train, self.device, self.dtype)
        self._f_dev = _put(self._F_std, self.device, self.dtype)
        if hasattr(self, "_params"):
            self._params = jax.device_put(self._params, self.device)

    def predict(self, X_pred, return_variance=False, block_size=None):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if block_size is not None:
            raise NotImplementedError(
                "block_size is not implemented for UniversalKriging; "
                "use OrdinaryKriging for block kriging, or predict at point support."
            )
        X_pred_orig = np.asarray(X_pred, dtype=float)
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred_orig)
        if getattr(self, "_factor", None) is not None:
            F_pred = self._poly_std.transform((X_pred_orig - self._drift_shift) / self._drift_scale)
            def run():
                chol, a, b, w, q_chol, r = self._factor
                model = self.variogram.model

                def fn(xb, fb):
                    return _predict_single(xb, fb, self._x_dev, chol, a, b, w, q_chol, r,
                                           self._params, model, bool(return_variance))

                batch = _pred_batch(len(self.X_train), len(X_pred), self.dtype)
                with jax.default_device(self.device):
                    return _run_batched(fn, X_pred - self._center, F_pred, batch, self.device, self.dtype)

            pred, se = self._with_cpu_retry(run)
            return (pred, se) if return_variance else pred
        F_pred = self._poly.transform(X_pred_orig)
        with jax.default_device(self.device):
            pred, se = _krige_universal(
                _put(self.X_train, self.device),
                _put(self.y_train, self.device),
                _put(X_pred, self.device),
                _put(self.F, self.device),
                _put(F_pred, self.device),
                self.variogram.model,
                self.variogram.fitted_params,
            )
        pred = np.asarray(pred)
        if not return_variance:
            return pred
        return pred, np.asarray(se)


def _krige_universal(x_train, y_train, x_pred, drift, drift_pred, model, params):
    n = x_train.shape[0]
    p = drift.shape[1]
    dist_train = _pairwise(x_train, x_train, same=True)
    dist_pred = _pairwise(x_pred, x_train, same=False)
    c0 = float(params[0]) + float(params[1])
    c_train = _cov_from_gamma(_apply_model(model, dist_train, params), c0, dist_train)
    c_pred = _cov_from_gamma(_apply_model(model, dist_pred, params), c0, dist_pred)
    top = jnp.concatenate([c_train, drift], axis=1)
    bottom = jnp.concatenate([drift.T, jnp.zeros((p, p))], axis=1)
    matrix = jnp.concatenate([top, bottom], axis=0)
    rhs = jnp.concatenate([c_pred, drift_pred], axis=1)
    weights = _solve(matrix, rhs.T).T
    pred = weights[:, :n] @ y_train
    var = c0 - jnp.sum(weights[:, :n] * c_pred, axis=1) - jnp.sum(weights[:, n:] * drift_pred, axis=1)
    se = jnp.sqrt(jnp.clip(var, 0.0, None))
    return pred, se


class CrossVariogram:
    """
    Empirical and theoretical cross-variogram on a JAX device.

    ``fitted_params`` is ``[nugget, c12, range, ...]``. ``c12`` may be negative.
    """

    def __init__(
        self,
        coords,
        primary,
        secondary,
        maxlag=None,
        n_lags=10,
        model="exponential",
        use_gpu="auto",
        device=None,
    ):
        if model not in MODEL_FUNCS:
            raise ValueError(f"model must be one of {list(MODEL_FUNCS)}")
        self.coords = np.asarray(coords, dtype=float)
        self.primary = np.asarray(primary, dtype=float).reshape(-1)
        self.secondary = np.asarray(secondary, dtype=float).reshape(-1)
        self.maxlag = maxlag
        self.n_lags = int(n_lags)
        self.model = model
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)
        self.lags = None
        self.experimental = None
        self.fitted_params = None
        self.fit_success_ = None

    def fit(self, method="empirical"):
        if len(self.primary) != len(self.secondary):
            raise ValueError("Primary and secondary must be co-located (same length).")
        if len(self.coords) != len(self.primary):
            raise ValueError("coords must have the same length as primary/secondary.")
        with jax.default_device(self.device):
            coords = _put(self.coords, self.device)
            primary = _put(self.primary, self.device)
            secondary = _put(self.secondary, self.device)
            dist, _, iu, ju = _pairwise_upper(coords, primary)
            pair = 0.5 * (primary[iu] - primary[ju]) * (secondary[iu] - secondary[ju])
        dist_host = np.asarray(dist)
        positive = dist_host[dist_host > 0]
        if positive.size == 0:
            raise ValueError("All locations are coincident; cannot estimate a cross-variogram")
        if self.maxlag is None:
            both = np.concatenate([positive, positive])
            self.maxlag = float(np.percentile(both, 50))
        with jax.default_device(self.device):
            centers, gamma = _binned_mean(dist, pair, float(self.maxlag), self.n_lags)
        self.lags = np.asarray(centers)
        self.experimental = np.asarray(gamma)
        if method == "auto":
            self.fit_model()
        elif method != "empirical":
            raise ValueError("method must be 'empirical' or 'auto'")
        return self

    def set_params(self, nugget, c12, range_, nu=None, alpha=None):
        params = [float(nugget), float(c12), float(range_)]
        if self.model == "matern":
            if nu is None:
                raise ValueError("matern requires nu")
            params.append(float(nu))
        elif self.model == "stable":
            if alpha is None:
                raise ValueError("stable requires alpha")
            params.append(float(alpha))
        elif nu is not None or alpha is not None:
            raise ValueError(f"{self.model} does not take nu or alpha")
        self.fitted_params = params
        return self

    def fit_model(self, range_=None, nu=None, alpha=None):
        if self.experimental is None:
            raise RuntimeError("Call fit() before fit_model().")
        obs = self.experimental
        mask = ~np.isnan(obs)
        if not np.any(mask):
            raise RuntimeError("Empirical cross-variogram has no finite lags.")
        cov0 = float(np.cov(self.primary, self.secondary)[0, 1])
        range0 = float(range_) if range_ is not None else self.maxlag / 2.0
        if self.model == "matern":
            x0 = [0.0, cov0, range0, 1.5 if nu is None else float(nu)]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3), (0.1, 10)]
        elif self.model == "stable":
            x0 = [0.0, cov0, range0, 1.0 if alpha is None else float(alpha)]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3), (0.01, 2.0)]
        else:
            x0 = [0.0, cov0, range0]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3)]
        lock_range = range_ is not None
        lock_extra = (self.model == "matern" and nu is not None) or (
            self.model == "stable" and alpha is not None
        )
        model = self.model
        lags = self.lags
        device = self.device

        def unpack(params):
            p = list(params)
            if lock_range:
                p[2] = range0
            if lock_extra and len(p) > 3:
                p[3] = x0[3]
            return p

        def obj(params):
            p = unpack(params)
            with jax.default_device(device):
                pred = np.asarray(_apply_model(model, lags, p))
            return float(np.sum((pred[mask] - obs[mask]) ** 2))

        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        self.fitted_params = [float(v) for v in unpack(res.x if res.success else x0)]
        self.fit_success_ = bool(res.success)
        return self

    def __call__(self, h):
        if self.fitted_params is None:
            raise RuntimeError("Call .fit() and set .fitted_params first.")
        on_device = isinstance(h, jax.Array)
        device = next(iter(h.devices())) if on_device and h.devices() else self.device
        with jax.default_device(device):
            out = _apply_model(self.model, h, self.fitted_params)
        if on_device:
            return out
        return np.asarray(out)


def _binned_mean(dist, pair_value, maxlag, n_lags):
    """Mean of ``pair_value`` on bins (left, right]."""
    edges = jnp.linspace(0.0, maxlag, n_lags + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = jnp.searchsorted(edges, dist, side="left") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_lags) & jnp.isfinite(pair_value)
    safe = jnp.clip(bin_idx, 0, n_lags - 1)
    numer = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, pair_value, 0.0))
    counts = jnp.zeros(n_lags).at[safe].add(jnp.where(valid, 1.0, 0.0))
    gamma = jnp.where(counts > 0, numer / jnp.maximum(counts, 1.0), jnp.nan)
    return centers, gamma


def _jitter_diag(matrix, rel=1e-8):
    scale = jnp.nanmax(jnp.abs(jnp.diag(matrix)))
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.0)
    idx = jnp.arange(matrix.shape[0])
    return matrix.at[idx, idx].add(rel * scale)


class Cokriging(_FactoredKriging):
    """
    Ordinary cokriging of one primary and one secondary variable.

    Two unbiasedness constraints: primary weights sum to 1, secondary weights
    sum to 0. Matches ``kriging.cokriging.Cokriging``.
    """

    def __init__(self, primary_var, secondary_var, cross_var, use_gpu="auto", device=None, dtype=None):
        self.primary_var = primary_var
        self.secondary_var = secondary_var
        self.cross_var = cross_var
        self.device = resolve_device(use_gpu, device)
        self.use_gpu = _is_gpu(self.device)
        self.dtype = _resolve_dtype(dtype)

    def fit(self, coords_primary, primary, coords_secondary, secondary):
        self.coords_primary = np.asarray(coords_primary, dtype=float)
        self.primary = np.asarray(primary, dtype=float).reshape(-1)
        self.coords_secondary = np.asarray(coords_secondary, dtype=float)
        self.secondary = np.asarray(secondary, dtype=float).reshape(-1)
        if len(self.coords_primary) != len(self.primary):
            raise ValueError("coords_primary and primary must have same length.")
        if len(self.coords_secondary) != len(self.secondary):
            raise ValueError("coords_secondary and secondary must have same length.")
        self.all_values = np.concatenate([self.primary, self.secondary])
        self.n_primary = len(self.primary)
        self.n_secondary = len(self.secondary)
        self._factor_with_fallback(self._build_factor)
        return self

    def _build_factor(self):
        self._refresh_device_arrays()
        drift = np.zeros((self.n_primary + self.n_secondary, 2))
        drift[:self.n_primary, 0] = 1.0
        drift[self.n_primary:, 1] = 1.0
        put = lambda v: _put(np.asarray(v, dtype=float), self.device, self.dtype)  # noqa: E731
        self._pp = put(self.primary_var.fitted_params)
        self._ps = put(self.secondary_var.fitted_params)
        self._pc = put(self.cross_var.fitted_params)
        with jax.default_device(self.device):
            return _factor_co(self._xp_dev, self._xs_dev, put(self.all_values), put(drift),
                              self._pp, self._ps, self._pc,
                              self.primary_var.model, self.secondary_var.model, self.cross_var.model)

    def _refresh_device_arrays(self):
        self._center = self.coords_primary.mean(axis=0)
        self._xp_dev = _put(self.coords_primary - self._center, self.device, self.dtype)
        self._xs_dev = _put(self.coords_secondary - self._center, self.device, self.dtype)
        for name in ("_pp", "_ps", "_pc"):
            if hasattr(self, name):
                setattr(self, name, jax.device_put(getattr(self, name), self.device))

    def predict(self, coords_pred, return_variance=False):
        if not hasattr(self, "coords_primary"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        if getattr(self, "_factor", None) is not None:
            f_pred = np.tile(np.array([[1.0, 0.0]]), (len(coords_pred), 1))

            def run():
                chol, a, b, w, q_chol, r = self._factor

                def fn(xb, fb):
                    return _predict_co(xb, fb, self._xp_dev, self._xs_dev, chol, a, b, w, q_chol, r,
                                       self._pp, self._pc, self.primary_var.model, self.cross_var.model,
                                       bool(return_variance))

                n_all = self.n_primary + self.n_secondary
                batch = _pred_batch(n_all, len(coords_pred), self.dtype)
                with jax.default_device(self.device):
                    return _run_batched(fn, coords_pred - self._center, f_pred, batch, self.device, self.dtype)

            pred, se = self._with_cpu_retry(run)
            return (pred, se) if return_variance else pred
        with jax.default_device(self.device):
            pred, se = _krige_co(
                _put(self.coords_primary, self.device),
                _put(self.coords_secondary, self.device),
                _put(self.all_values, self.device),
                _put(coords_pred, self.device),
                self.primary_var.model,
                self.primary_var.fitted_params,
                self.secondary_var.model,
                self.secondary_var.fitted_params,
                self.cross_var.model,
                self.cross_var.fitted_params,
            )
        pred = np.asarray(pred)
        if not return_variance:
            return pred
        return pred, np.asarray(se)


def _krige_co(
    coords_p, coords_s, values, coords_pred,
    model_p, params_p, model_s, params_s, model_c, params_c,
):
    n_p = coords_p.shape[0]
    n_s = coords_s.shape[0]
    n = n_p + n_s
    m = coords_pred.shape[0]
    c0_p = float(params_p[0]) + float(params_p[1])
    c0_s = float(params_s[0]) + float(params_s[1])
    c0_c = float(params_c[0]) + float(params_c[1])

    dist_pp = _pairwise(coords_p, coords_p, same=True)
    dist_ss = _pairwise(coords_s, coords_s, same=True)
    dist_ps = _pairwise(coords_p, coords_s, same=False)
    c_pp = _cov_from_gamma(_apply_model(model_p, dist_pp, params_p), c0_p, dist_pp)
    c_ss = _cov_from_gamma(_apply_model(model_s, dist_ss, params_s), c0_s, dist_ss)
    c_ps = _cov_from_gamma(_apply_model(model_c, dist_ps, params_c), c0_c, dist_ps)
    covariance = jnp.block([[c_pp, c_ps], [c_ps.T, c_ss]])
    covariance = _jitter_diag(covariance)

    dist_pred_p = _pairwise(coords_pred, coords_p, same=False)
    dist_pred_s = _pairwise(coords_pred, coords_s, same=False)
    rhs_p = _cov_from_gamma(_apply_model(model_p, dist_pred_p, params_p), c0_p, dist_pred_p)
    rhs_s = _cov_from_gamma(_apply_model(model_c, dist_pred_s, params_c), c0_c, dist_pred_s)
    rhs = jnp.concatenate([rhs_p, rhs_s], axis=1)

    drift = jnp.zeros((n, 2)).at[:n_p, 0].set(1.0).at[n_p:, 1].set(1.0)
    matrix = jnp.block([
        [covariance, drift],
        [drift.T, jnp.zeros((2, 2))],
    ])
    rhs_aug = jnp.concatenate([rhs, jnp.ones((m, 1)), jnp.zeros((m, 1))], axis=1)
    weights = _solve(matrix, rhs_aug.T).T
    pred = weights[:, :n] @ values
    var = c0_p - jnp.sum(weights[:, :n] * rhs, axis=1) - weights[:, n]
    se = jnp.sqrt(jnp.clip(var, 0.0, None))
    return pred, se


JaxVariogram = Variogram
JaxOrdinaryKriging = OrdinaryKriging
JaxSimpleKriging = SimpleKriging
JaxUniversalKriging = UniversalKriging
JaxCokriging = Cokriging

__all__ = [
    "resolve_device",
    "to_numpy",
    "spherical",
    "exponential",
    "gaussian",
    "matern",
    "stable",
    "cubic",
    "MODEL_FUNCS",
    "CPU_ONLY_MODELS",
    "covariance_from_variogram",
    "transform_aniso",
    "ESTIMATORS",
    "Variogram",
    "JaxVariogram",
    "compute_variogram_cloud",
    "VariogramCloud",
    "plot_variogram_cloud",
    "fit_indicator_variogram",
    "OrdinaryKriging",
    "SimpleKriging",
    "UniversalKriging",
    "CrossVariogram",
    "Cokriging",
    "JaxOrdinaryKriging",
    "JaxSimpleKriging",
    "JaxUniversalKriging",
    "JaxCokriging",
]
