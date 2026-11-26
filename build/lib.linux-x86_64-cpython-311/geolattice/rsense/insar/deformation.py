"""Displacement from unwrapped phase: LOS conversion, decomposition, SBAS time series, velocity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import get_xp, to_cpu
from .interferogram import SENTINEL1_WAVELENGTH


def phase_to_los(phase: Any, wavelength: float = SENTINEL1_WAVELENGTH, *,
                 toward_positive: bool = True) -> Any:
    """Unwrapped phase (rad) → line-of-sight displacement (m).

    ``d = -lambda / (4 pi) * phi``. With ``toward_positive=True`` (MintPy/ISCE convention)
    positive values mean motion toward the satellite (uplift or eastward on ascending).
    Conventions differ between processors: check the sign against a known reference.
    """
    d = -wavelength / (4 * np.pi) * phase
    d = d if toward_positive else -d
    if isinstance(d, xr.DataArray):
        d = d.rename("los_displacement").assign_attrs(units="m")
    return d


def los_to_vertical(los: Any, incidence_deg: Any) -> Any:
    """Project LOS displacement to vertical assuming purely vertical motion: ``d_v = d_los / cos(theta)``."""
    v = los / np.cos(np.radians(incidence_deg))
    return v.rename("vertical_displacement") if isinstance(v, xr.DataArray) else v


def decompose_asc_desc(los_asc: Any, los_desc: Any, inc_asc: Any, inc_desc: Any,
                       heading_asc: float = -10.0, heading_desc: float = -170.0) -> dict:
    """Combine ascending and descending LOS into vertical and east-west components.

    North-south motion is neglected (near-polar orbits are insensitive to it). LOS unit
    vectors use incidence angle and satellite heading (degrees from north, clockwise);
    the radar looks right. Positive LOS = toward satellite.
    """
    def coeffs(inc, head):
        inc_r, h = np.radians(inc), np.radians(head)
        look_az = h + np.pi / 2           # right-looking
        e = -np.sin(inc_r) * np.sin(look_az)
        u = np.cos(inc_r)
        return e, u

    ea, ua = coeffs(inc_asc, heading_asc)
    ed, ud = coeffs(inc_desc, heading_desc)
    det = ea * ud - ed * ua
    east = (los_asc * ud - los_desc * ua) / det
    up = (ea * los_desc - ed * los_asc) / det
    return {"east": east, "up": up}


def sbas_inversion(ifgs: Any, pairs: Sequence[tuple[Any, Any]], *, dim: str = "pair",
                   reference_date: Any = None, weights: Any = None) -> xr.DataArray | np.ndarray:
    """Small-baseline network inversion (Berardino et al. 2002) to a displacement time series.

    Parameters
    ----------
    ifgs : unwrapped interferograms (displacement or phase) stacked on ``dim``
        (DataArray (pair, y, x) or array (n_pairs, ...)).
    pairs : list of (reference_date, secondary_date) for each interferogram.
    reference_date : date fixed at zero (default: earliest).
    weights : optional per-pixel weights (e.g. coherence) with the same shape as ``ifgs``.

    Solves ``A d = ifg`` per pixel by (weighted) least squares, vectorized across pixels on
    the active backend. Returns cumulative displacement per date (same units as input).
    """
    dates = sorted({pd.Timestamp(d) for p in pairs for d in p})
    ref = pd.Timestamp(reference_date) if reference_date is not None else dates[0]
    unknown = [d for d in dates if d != ref]
    col = {d: i for i, d in enumerate(unknown)}
    A = np.zeros((len(pairs), len(unknown)))
    for i, (d1, d2) in enumerate(pairs):
        d1, d2 = pd.Timestamp(d1), pd.Timestamp(d2)
        if d2 in col:
            A[i, col[d2]] += 1
        if d1 in col:
            A[i, col[d1]] -= 1
    if np.linalg.matrix_rank(A) < len(unknown):
        raise ValueError("Interferogram network is disconnected; add pairs linking all subsets")
    xp = get_xp()
    data = np.asarray(to_cpu(getattr(ifgs, "values", ifgs)), dtype="float64")
    spatial = data.shape[1:]
    Y = data.reshape(len(pairs), -1)
    ok = np.all(np.isfinite(Y), axis=0)
    D = np.full((len(unknown), Y.shape[1]), np.nan)
    if weights is None:
        sol = xp.linalg.pinv(xp.asarray(A)) @ xp.asarray(Y[:, ok])
        D[:, ok] = to_cpu(sol)
    else:
        Wt = np.asarray(to_cpu(getattr(weights, "values", weights)), dtype="float64")
        Wt = Wt.reshape(len(pairs), -1)
        for j in np.flatnonzero(ok):
            w = np.sqrt(np.clip(Wt[:, j], 1e-6, None))
            D[:, j] = np.linalg.lstsq(A * w[:, None], Y[:, j] * w, rcond=None)[0]
    full = np.zeros((len(dates), Y.shape[1]))
    full[[dates.index(d) for d in unknown]] = D
    full[:, ~ok] = np.nan
    full = full.reshape((len(dates),) + spatial)
    if isinstance(ifgs, xr.DataArray):
        template = ifgs.isel({dim: 0}, drop=True)
        da = xr.concat([template.copy(data=f) for f in full],
                       dim=pd.Index(dates, name="time"))
        return da.rename("displacement")
    return full


def velocity(series: xr.DataArray, *, dim: str = "time", seasonal: bool = False,
             per: str = "year") -> xr.Dataset:
    """Per-pixel linear velocity (units per year) from a displacement time series by least
    squares, optionally with annual sine/cosine terms. Also returns the residual RMSE."""
    t = (pd.DatetimeIndex(series[dim].values) - pd.Timestamp(series[dim].values[0])).days.values
    ty = t / 365.25
    cols = [np.ones_like(ty), ty]
    if seasonal:
        cols += [np.sin(2 * np.pi * ty), np.cos(2 * np.pi * ty)]
    A = np.stack(cols, 1)
    xp = get_xp()
    s = series.transpose(dim, ...)
    Y = np.asarray(to_cpu(s).values, dtype="float64").reshape(len(ty), -1)
    ok = np.all(np.isfinite(Y), axis=0)
    coef = np.full((A.shape[1], Y.shape[1]), np.nan)
    coef[:, ok] = to_cpu(xp.linalg.pinv(xp.asarray(A)) @ xp.asarray(Y[:, ok]))
    resid = Y - A @ np.nan_to_num(coef)
    rmse = np.sqrt(np.nanmean(resid**2, axis=0))
    rmse[~ok] = np.nan
    shape = s.shape[1:]
    template = s.isel({dim: 0}, drop=True)
    out = xr.Dataset({"velocity": template.copy(data=coef[1].reshape(shape)),
                      "rmse": template.copy(data=rmse.reshape(shape))})
    if seasonal:
        out["seasonal_amplitude"] = template.copy(
            data=np.hypot(coef[2], coef[3]).reshape(shape))
    out["velocity"].attrs["units"] = f"{series.attrs.get('units', 'units')}/{per}"
    return out
