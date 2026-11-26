"""Breakpoint detection: BFAST-Monitor-like MOSUM monitoring of a harmonic model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import get_xp, to_cpu
from ._common import decimal_days, harmonic_design, interp_gaps, to_matrix, to_spatial


def bfast_monitor(da: xr.DataArray, monitor_start: Any, *, dim: str = "time",
                  n_harmonics: int = 1, trend: bool = True, h: float = 0.25,
                  lam: float | None = None, history_start: Any = None) -> xr.Dataset:
    """Detect the first disturbance after ``monitor_start`` (Verbesselt et al. 2012, simplified).

    A season-trend model (``n_harmonics`` + optional linear trend) is fit by OLS on the
    stable history period. In the monitoring period the moving sum (window ``h`` × history
    length) of standardized residuals is compared to the boundary
    ``lam * sqrt(2 * log+(t / n))``. The default ``lam`` approximates the 5% critical value
    for ``h = 0.25``; treat it as a tunable sensitivity parameter rather than an exact test.

    Returns ``breakpoint`` (date of first boundary crossing), ``magnitude`` (median
    monitoring residual; negative = loss for vegetation indices) and ``detected``.
    Gaps are linearly interpolated before fitting.
    """
    M, shape, tpl = to_matrix(da, dim)
    times = pd.DatetimeIndex(tpl[dim].values)
    days = decimal_days(times)
    M = interp_gaps(M, days)
    hist = times < pd.Timestamp(monitor_start)
    if history_start is not None:
        hist &= times >= pd.Timestamp(history_start)
    mon = times >= pd.Timestamp(monitor_start)
    n = int(hist.sum())
    X = harmonic_design(days, n_harmonics, trend)
    if n < X.shape[1] + 2 or not mon.any():
        raise ValueError("History too short for the model, or no monitoring observations")
    ok = np.all(np.isfinite(M), axis=0)
    xp = get_xp()
    Xh = xp.asarray(X[hist])
    Y = xp.asarray(M[:, ok])
    coef = xp.linalg.pinv(Xh) @ Y[xp.asarray(hist)]
    resid = Y - xp.asarray(X) @ coef
    rh = resid[xp.asarray(hist)]
    sigma = xp.sqrt((rh**2).sum(0) / (n - X.shape[1]))
    rm = resid[xp.asarray(mon)]
    win = max(1, int(np.floor(h * n)))
    # moving sum over the history tail + monitoring residuals
    tail = xp.concatenate([rh[-(win - 1):] if win > 1 else rh[:0], rm], 0)
    cs = xp.cumsum(tail, 0)
    cs = xp.concatenate([xp.zeros((1, cs.shape[1])), cs], 0)
    mosum = (cs[win:] - cs[:-win]) / (sigma * np.sqrt(n))
    t_idx = np.arange(n + 1, n + 1 + int(mon.sum()), dtype=float)
    lam = lam if lam is not None else _lambda_default(h)
    bound = lam * np.sqrt(2 * np.maximum(np.log(t_idx / n), 1.0))
    cross = to_cpu(xp.abs(mosum)) > bound[:, None]
    detected = cross.any(0)
    first = np.argmax(cross, 0)
    mon_times = times[mon].values
    bp = np.full(M.shape[1], np.datetime64("NaT"), dtype="datetime64[ns]")
    mag = np.full(M.shape[1], np.nan)
    det = np.zeros(M.shape[1], bool)
    okidx = np.flatnonzero(ok)
    bp[okidx[detected]] = mon_times[first[detected]]
    mag[okidx] = np.median(to_cpu(rm), 0)
    det[okidx] = detected
    return xr.Dataset({"breakpoint": to_spatial(bp, shape, tpl, dim),
                       "magnitude": to_spatial(mag.astype("float32"), shape, tpl, dim),
                       "detected": to_spatial(det, shape, tpl, dim)},
                      attrs={"monitor_start": str(pd.Timestamp(monitor_start).date()), "h": h,
                             "lambda": lam})


def _lambda_default(h: float) -> float:
    # rough critical values (alpha = 0.05) for the MOSUM log boundary, by window size
    table = {0.25: 2.0, 0.5: 1.8, 1.0: 1.6}
    k = min(table, key=lambda x: abs(x - h))
    return table[k]
