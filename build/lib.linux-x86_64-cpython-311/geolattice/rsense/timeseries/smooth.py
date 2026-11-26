"""Smoothing and gap-filling of image time series (pixel-wise, vectorized)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..backend import get_xp, to_cpu
from ._common import decimal_days, from_matrix, interp_gaps, to_matrix


def interpolate(da: xr.DataArray, *, dim: str = "time") -> xr.DataArray:
    """Fill NaNs by linear interpolation in time (respecting actual acquisition dates)."""
    M, shape, tpl = to_matrix(da, dim)
    return from_matrix(interp_gaps(M, decimal_days(tpl[dim].values)), shape, tpl, da.name)


def savgol(da: xr.DataArray, *, window: int = 7, polyorder: int = 2, dim: str = "time",
           fill: bool = True, upper_envelope: int = 0) -> xr.DataArray:
    """Savitzky–Golay filter along time.

    ``upper_envelope`` > 0 runs the Chen et al. (2004) iterative upper-envelope fit
    (values below the smooth curve, typically residual cloud, are replaced and refit).
    """
    from scipy.signal import savgol_filter

    M, shape, tpl = to_matrix(da, dim)
    if fill:
        M = interp_gaps(M, decimal_days(tpl[dim].values))
    ok = np.all(np.isfinite(M), axis=0)
    S = np.full_like(M, np.nan)
    Y = M[:, ok]
    sm = savgol_filter(Y, window, polyorder, axis=0, mode="interp")
    for _ in range(upper_envelope):
        Y = np.maximum(Y, sm)
        sm = savgol_filter(Y, window, polyorder, axis=0, mode="interp")
    S[:, ok] = sm
    return from_matrix(S, shape, tpl, da.name)


def whittaker(da: xr.DataArray, *, lmbda: float = 10.0, dim: str = "time", order: int = 2,
              fill: bool = True) -> xr.DataArray:
    """Whittaker smoother (Eilers 2003): minimizes ``|y - z|^2 + lambda |D^d z|^2``.

    One (T×T) system is solved for all pixels at once on the active backend, so this is
    fast even for large cubes. NaNs are linearly interpolated first (``fill=True``).
    """
    M, shape, tpl = to_matrix(da, dim)
    if fill:
        M = interp_gaps(M, decimal_days(tpl[dim].values))
    T = M.shape[0]
    D = np.diff(np.eye(T), n=order, axis=0)
    A = np.eye(T) + lmbda * D.T @ D
    ok = np.all(np.isfinite(M), axis=0)
    xp = get_xp()
    Z = np.full_like(M, np.nan)
    Z[:, ok] = to_cpu(xp.linalg.solve(xp.asarray(A), xp.asarray(M[:, ok])))
    return from_matrix(Z, shape, tpl, da.name)


def harmonic_fit(da: xr.DataArray, *, n_harmonics: int = 2, dim: str = "time",
                 trend: bool = True) -> xr.Dataset:
    """Per-pixel harmonic regression (Fourier series + optional linear trend).

    Returns the fitted series plus amplitude and phase (days) of the first harmonic
    and the trend slope (per year). NaNs are ignored per pixel.
    """
    from ._common import harmonic_design, to_spatial

    M, shape, tpl = to_matrix(da, dim)
    days = decimal_days(tpl[dim].values)
    X = harmonic_design(days, n_harmonics, trend)
    coef = np.full((X.shape[1], M.shape[1]), np.nan)
    full = np.all(np.isfinite(M), axis=0)
    xp = get_xp()
    if full.any():
        coef[:, full] = to_cpu(xp.linalg.pinv(xp.asarray(X)) @ xp.asarray(M[:, full]))
    for j in np.flatnonzero(~full & (np.isfinite(M).sum(0) > X.shape[1])):
        ok = np.isfinite(M[:, j])
        coef[:, j] = np.linalg.lstsq(X[ok], M[ok, j], rcond=None)[0]
    fitted = X @ coef
    i0 = 2 if trend else 1
    amp = np.hypot(coef[i0], coef[i0 + 1])
    phase = (np.arctan2(coef[i0 + 1], coef[i0]) % (2 * np.pi)) / (2 * np.pi) * 365.25
    out = xr.Dataset({"fitted": from_matrix(fitted, shape, tpl),
                      "amplitude": to_spatial(amp, shape, tpl, dim),
                      "phase_days": to_spatial(phase, shape, tpl, dim)})
    if trend:
        out["trend_per_year"] = to_spatial(coef[1], shape, tpl, dim)
    return out
