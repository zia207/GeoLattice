"""CCDC-lite: Continuous Change Detection and Classification (Zhu & Woodcock 2014), simplified."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2

from ..backend import to_cpu
from ._common import decimal_days, harmonic_design


def _fit(X, Y):
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ coef
    dof = max(X.shape[0] - X.shape[1], 1)
    rmse = np.sqrt((r**2).sum(0) / dof)
    return coef, np.maximum(rmse, 1e-6)


def _ccdc_pixel(days, Y, init_obs, consecutive, crit, n_harm, refit_every, init_days,
                min_rmse):
    """Y: (T, B) valid observations only. Returns list of (break_index, magnitude)."""
    breaks = []
    T = len(days)
    start = 0
    while start + init_obs + consecutive <= T:
        # initial window: at least init_obs observations spanning at least init_days
        end = start + init_obs
        while end < T and days[end - 1] - days[start] < init_days:
            end += 1
        if end + consecutive > T:
            break
        X = harmonic_design(days[start:end] - days[start], n_harm, True)
        coef, rmse = _fit(X, Y[start:end])
        rmse = np.maximum(rmse, min_rmse)
        i = end
        since_fit = 0
        found = False
        while i + consecutive <= T:
            Xn = harmonic_design(days[i:i + consecutive] - days[start], n_harm, True)
            z = (((Y[i:i + consecutive] - Xn @ coef) / rmse) ** 2).sum(1)
            if np.all(z > crit):
                mag = (Y[i:i + consecutive] - Xn @ coef).mean(0)
                breaks.append((i, mag))
                start = i
                found = True
                break
            i += 1
            since_fit += 1
            if since_fit >= refit_every:
                X = harmonic_design(days[start:i] - days[start], n_harm, True)
                coef, rmse = _fit(X, Y[start:i])
                rmse = np.maximum(rmse, min_rmse)
                since_fit = 0
        if not found:
            break
    return breaks


def ccdc(obj: xr.Dataset | xr.DataArray, *, bands: Sequence[str] | None = None, dim: str = "time",
         init_obs: int = 12, consecutive: int = 6, probability: float = 0.99,
         n_harmonics: int = 1, refit_every: int = 6, init_days: float = 365.0,
         min_rmse: float | Sequence[float] = 0.0) -> xr.Dataset:
    """Detect multiple land-cover breaks per pixel from a multi-band reflectance series.

    For each pixel a harmonic + trend model is fit to the first ``init_obs`` clear
    observations spanning at least ``init_days`` (a full year, so the seasonal terms are
    identifiable); ``min_rmse`` (per band) floors the residual scale; a break is declared
    when ``consecutive`` observations in a row exceed
    the chi-square (``probability``, dof = number of bands) of RMSE-normalized residuals;
    a new model starts after each break. Models are periodically refit (``refit_every``).

    Simplified relative to the full CCDC (no Tmask cloud screening, fixed model order).
    Pixel loop on CPU, so use for areas up to ~10^5 pixels or tile with Dask.

    Returns ``n_breaks``, ``first_break``, ``last_break`` and ``first_magnitude``
    (mean residual of the first band at the first break).
    """
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset(name=obj.name or "value")
    bands = list(bands or obj.data_vars)
    stack = xr.concat([obj[b] for b in bands], "band").transpose(dim, "band", ...)
    A = np.asarray(to_cpu(stack).values, dtype="float64")
    T, B = A.shape[:2]
    spatial = A.shape[2:]
    A = A.reshape(T, B, -1)
    times = pd.DatetimeIndex(obj[dim].values)
    days = decimal_days(times)
    crit = chi2.ppf(probability, B)
    N = A.shape[2]
    nb = np.zeros(N, np.int16)
    fb = np.full(N, np.datetime64("NaT"), "datetime64[ns]")
    lb = fb.copy()
    fm = np.full(N, np.nan, np.float32)
    for j in range(N):
        Y = A[:, :, j]
        ok = np.all(np.isfinite(Y), axis=1)
        if ok.sum() < init_obs + consecutive:
            continue
        brk = _ccdc_pixel(days[ok], Y[ok], init_obs, consecutive, crit, n_harmonics, refit_every,
                          init_days, np.broadcast_to(np.asarray(min_rmse, float), (B,)))
        if brk:
            tv = times[ok]
            nb[j] = len(brk)
            fb[j] = tv[brk[0][0]].to_datetime64()
            lb[j] = tv[brk[-1][0]].to_datetime64()
            fm[j] = brk[0][1][0]
    tpl = obj[bands[0]].isel({dim: 0}, drop=True)

    def wrap(v):
        return tpl.copy(data=v.reshape(spatial))

    return xr.Dataset({"n_breaks": wrap(nb), "first_break": wrap(fb), "last_break": wrap(lb),
                       "first_magnitude": wrap(fm)},
                      attrs={"method": "CCDC-lite", "probability": probability,
                             "consecutive": consecutive})
