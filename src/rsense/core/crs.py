"""Coordinate reference system helpers."""

from __future__ import annotations

from typing import Any

import xarray as xr


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing a lon/lat point (Norway/Svalbard exceptions)."""
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError("lon/lat outside UTM domain")
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    elif 72 <= lat < 84 and lon >= 0:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37 if lon < 42 else zone
    zone = min(zone, 60)
    return (32600 if lat >= 0 else 32700) + zone


def estimate_utm_crs(obj: xr.DataArray | xr.Dataset | Any) -> int:
    """UTM EPSG for the centre of a raster or GeoDataFrame."""
    from pyproj import Transformer

    if hasattr(obj, "rio"):
        crs = obj.rio.crs
        b = obj.rio.bounds()
    else:
        crs = obj.crs
        b = obj.total_bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(cx, cy)
    return utm_epsg(lon, lat)


def transform_bounds(bounds: tuple[float, float, float, float], src: Any, dst: Any,
                     densify: int = 21) -> tuple[float, float, float, float]:
    """Reproject a bounding box (minx, miny, maxx, maxy), densifying edges."""
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True).transform_bounds(*bounds,
                                                                           densify_pts=densify)


def ensure_crs(obj: xr.DataArray | xr.Dataset, crs: Any = None):
    """Return ``obj`` with a CRS, writing ``crs`` if missing; raise if neither is available."""
    if obj.rio.crs is not None:
        return obj
    if crs is None:
        raise ValueError("Object has no CRS; pass crs=...")
    return obj.rio.write_crs(crs)
