"""Land surface phenology metrics from smoothed vegetation-index time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ._common import to_matrix, to_spatial


def phenology(vi: xr.DataArray, *, dim: str = "time", threshold: float = 0.2,
              eos_threshold: float | None = None, min_amplitude: float = 0.1) -> xr.Dataset:
    """Seasonal metrics by the amplitude-threshold method (Jönsson & Eklundh 2004, TIMESAT).

    Expects one growing season of a smoothed index (see :mod:`geolattice.rsense.timeseries.smooth`).
    SOS/EOS are where the curve crosses ``base + threshold * amplitude`` before/after
    the peak (bases are left/right minima). Linear interpolation between dates gives
    day-of-year precision. Returns ``sos``, ``pos``, ``eos`` (day of year), ``los`` (days),
    ``peak``, ``base``, ``amplitude``, ``integral`` (season sum of VI above base × days),
    ``greenup_rate``, ``senescence_rate`` (per day).
    """
    eos_threshold = threshold if eos_threshold is None else eos_threshold
    M, shape, tpl = to_matrix(vi, dim)
    t = pd.DatetimeIndex(tpl[dim].values)
    doy0 = t[0].dayofyear
    days = ((t - t[0]).total_seconds() / 86400.0).to_numpy() + doy0
    T, N = M.shape
    keys = ("sos", "pos", "eos", "los", "peak", "base", "amplitude", "integral",
            "greenup_rate", "senescence_rate")
    out = {k: np.full(N, np.nan) for k in keys}
    filled = np.where(np.isfinite(M), M, -np.inf)
    ip = np.argmax(filled, axis=0)
    for j in np.flatnonzero(np.isfinite(M).sum(0) >= 4):
        y = M[:, j]
        p = ip[j]
        left, right = y[:p + 1], y[p:]
        if not (np.isfinite(left).any() and np.isfinite(right).any()):
            continue
        bl, br = np.nanmin(left), np.nanmin(right)
        peak = y[p]
        amp = peak - (bl + br) / 2
        if not np.isfinite(amp) or amp < min_amplitude:
            continue
        lvl_s = bl + threshold * (peak - bl)
        lvl_e = br + eos_threshold * (peak - br)
        sos = _crossing(days[:p + 1], left, lvl_s, rising=True)
        eos = _crossing(days[p:], right, lvl_e, rising=False)
        out["pos"][j], out["peak"][j] = days[p], peak
        out["base"][j], out["amplitude"][j] = (bl + br) / 2, amp
        out["sos"][j], out["eos"][j] = sos, eos
        if np.isfinite(sos) and np.isfinite(eos):
            out["los"][j] = eos - sos
            sel = (days >= sos) & (days <= eos) & np.isfinite(y)
            if sel.sum() >= 2:
                out["integral"][j] = np.trapezoid(y[sel] - out["base"][j], days[sel]) \
                    if hasattr(np, "trapezoid") else np.trapz(y[sel] - out["base"][j], days[sel])
            out["greenup_rate"][j] = (peak - lvl_s) / max(days[p] - sos, 1e-9)
            out["senescence_rate"][j] = (peak - lvl_e) / max(eos - days[p], 1e-9)
    return xr.Dataset({k: to_spatial(v.astype("float32"), shape, tpl, dim) for k, v in out.items()},
                      attrs={"method": "amplitude threshold", "threshold": threshold})


def _crossing(d: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float:
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    if rising:
        idx = np.flatnonzero((y[:-1] < level) & (y[1:] >= level))
        if not idx.size:
            return np.nan
        i = idx[-1]
    else:
        idx = np.flatnonzero((y[:-1] >= level) & (y[1:] < level))
        if not idx.size:
            return np.nan
        i = idx[0]
    return float(d[i] + (level - y[i]) * (d[i + 1] - d[i]) / (y[i + 1] - y[i]))
