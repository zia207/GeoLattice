"""Shared helpers: move time to axis 0, flatten pixels, gap-fill, rebuild DataArrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu


def to_matrix(da: xr.DataArray, dim: str = "time") -> tuple[np.ndarray, tuple, xr.DataArray]:
    """(T, N) float64 matrix, spatial shape, and a template for rebuilding."""
    da = da.transpose(dim, ...)
    a = np.asarray(to_cpu(da).values, dtype="float64")
    return a.reshape(a.shape[0], -1), a.shape[1:], da


def from_matrix(M: np.ndarray, shape: tuple, template: xr.DataArray, name: str | None = None
                ) -> xr.DataArray:
    return template.copy(data=M.reshape((M.shape[0],) + shape).astype("float32")).rename(name)


def to_spatial(v: np.ndarray, shape: tuple, template: xr.DataArray, dim: str = "time",
               name: str | None = None) -> xr.DataArray:
    t = template.isel({dim: 0}, drop=True)
    return t.copy(data=v.reshape(shape)).rename(name)


def interp_gaps(M: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Linear interpolation of NaNs along axis 0 per column (edges: nearest valid)."""
    T, N = M.shape
    t = np.arange(T, dtype=float) if t is None else np.asarray(t, float)
    out = M.copy()
    nan = np.isnan(M)
    cols = np.flatnonzero(nan.any(0) & (~nan).any(0))
    for j in cols:
        ok = ~nan[:, j]
        out[nan[:, j], j] = np.interp(t[nan[:, j]], t[ok], M[ok, j])
    return out


def decimal_days(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return ((t - t[0]).total_seconds() / 86400.0).to_numpy()


def harmonic_design(days: np.ndarray, n_harmonics: int = 2, trend: bool = True,
                    period: float = 365.25) -> np.ndarray:
    cols = [np.ones_like(days)]
    if trend:
        cols.append(days / period)
    for k in range(1, n_harmonics + 1):
        w = 2 * np.pi * k * days / period
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, 1)
