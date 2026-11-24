"""Climatologies, anomalies and drought condition indices."""

from __future__ import annotations

from typing import Any, Literal

import xarray as xr

Freq = Literal["month", "dayofyear", "season"]


def climatology(da: xr.DataArray, *, freq: Freq = "month", period: tuple[Any, Any] | None = None,
                dim: str = "time") -> xr.Dataset:
    """Mean, std, min and max by calendar month / day-of-year / season over a base ``period``."""
    base = da.sel({dim: slice(*period)}) if period else da
    g = base.groupby(f"{dim}.{freq}")
    return xr.Dataset({"mean": g.mean(dim), "std": g.std(dim), "min": g.min(dim),
                       "max": g.max(dim)})


def _expand(clim: xr.DataArray, da: xr.DataArray, freq: str, dim: str) -> xr.DataArray:
    """Broadcast a climatology statistic back onto every time step of ``da``."""
    key = getattr(da[dim].dt, freq)
    return clim.sel({freq: key}).drop_vars(freq, errors="ignore")


def anomaly(da: xr.DataArray, *, kind: Literal["absolute", "standardized", "percent"] = "absolute",
            freq: Freq = "month", period: tuple[Any, Any] | None = None,
            dim: str = "time") -> xr.DataArray:
    """Departure from climatology: ``x - mean`` (absolute), ``(x - mean)/std`` (z-score) or
    ``100 (x - mean)/mean`` (percent)."""
    c = climatology(da, freq=freq, period=period, dim=dim)
    m = _expand(c["mean"], da, freq, dim)
    if kind == "absolute":
        out = da - m
    elif kind == "standardized":
        out = (da - m) / _expand(c["std"], da, freq, dim)
    elif kind == "percent":
        out = 100 * (da - m) / m
    else:
        raise ValueError(f"Unknown kind {kind!r}")
    return out.rename(f"{da.name or 'value'}_anomaly")


def vci(ndvi: xr.DataArray, *, freq: Freq = "month", period: tuple[Any, Any] | None = None,
        dim: str = "time") -> xr.DataArray:
    """Vegetation Condition Index (Kogan 1995): ``100 (NDVI - min) / (max - min)`` per calendar
    period. 0 = worst observed condition, 100 = best."""
    c = climatology(ndvi, freq=freq, period=period, dim=dim)
    lo, hi = _expand(c["min"], ndvi, freq, dim), _expand(c["max"], ndvi, freq, dim)
    return (100 * (ndvi - lo) / (hi - lo)).clip(0, 100).rename("vci")


def tci(lst: xr.DataArray, *, freq: Freq = "month", period: tuple[Any, Any] | None = None,
        dim: str = "time") -> xr.DataArray:
    """Temperature Condition Index: ``100 (max - LST) / (max - min)`` (hot = stressed = low)."""
    c = climatology(lst, freq=freq, period=period, dim=dim)
    lo, hi = _expand(c["min"], lst, freq, dim), _expand(c["max"], lst, freq, dim)
    return (100 * (hi - lst) / (hi - lo)).clip(0, 100).rename("tci")


def vhi(vci_da: xr.DataArray, tci_da: xr.DataArray, alpha: float = 0.5) -> xr.DataArray:
    """Vegetation Health Index ``alpha * VCI + (1 - alpha) * TCI``."""
    return (alpha * vci_da + (1 - alpha) * tci_da).rename("vhi")
