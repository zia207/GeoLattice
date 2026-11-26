"""Zonal statistics: summarize raster values inside vector polygons."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr

_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean, "median": np.median, "min": np.min, "max": np.max,
    "std": np.std, "sum": np.sum, "count": lambda v: float(v.size),
}


def zonal_stats(raster: xr.DataArray, zones, stats: Sequence[str] = ("mean", "min", "max", "count"),
                *, all_touched: bool = False, prefix: str | None = None):
    """Compute statistics of a single-band raster within each polygon.

    Parameters
    ----------
    raster : 2-D DataArray (``y``, ``x``) with a CRS, or 3-D with a size-1 ``band`` dim.
    zones : GeoDataFrame of polygons; reprojected to the raster CRS if needed.
    stats : any of ``mean, median, min, max, std, sum, count`` or ``pNN`` percentiles (``p90``).
    prefix : column prefix; defaults to the raster name or ``"value"``.

    Returns a copy of ``zones`` with one column per statistic. NaN pixels are ignored.
    """
    from rasterio.features import geometry_mask

    from ..backend import to_cpu

    raster = to_cpu(raster)
    if "band" in raster.dims:
        if raster.sizes["band"] != 1:
            raise ValueError("zonal_stats expects a single band; select one with .sel(band=...)")
        raster = raster.squeeze("band", drop=True)
    if raster.rio.crs is None:
        raise ValueError("raster has no CRS")
    zones = zones.to_crs(raster.rio.crs) if zones.crs and zones.crs != raster.rio.crs else zones
    values = np.asarray(raster.values, dtype="float64")
    transform = raster.rio.transform()
    shape = values.shape
    prefix = prefix or (raster.name if isinstance(raster.name, str) else "value")

    funcs = {}
    for s in stats:
        if s in _STATS:
            funcs[s] = _STATS[s]
        elif s.startswith("p") and s[1:].replace(".", "", 1).isdigit():
            q = float(s[1:])
            funcs[s] = lambda v, q=q: np.percentile(v, q)
        else:
            raise ValueError(f"Unknown statistic {s!r}")

    rows = []
    for geom in zones.geometry:
        if geom is None or geom.is_empty:
            rows.append({k: np.nan for k in funcs})
            continue
        inside = ~geometry_mask([geom], out_shape=shape, transform=transform,
                                all_touched=all_touched)
        v = values[inside]
        v = v[~np.isnan(v)]
        rows.append({k: (f(v) if v.size else (0.0 if k == "count" else np.nan))
                     for k, f in funcs.items()})

    out = zones.copy()
    for k in funcs:
        out[f"{prefix}_{k}"] = [r[k] for r in rows]
    return out
