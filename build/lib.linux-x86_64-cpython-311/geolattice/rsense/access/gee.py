"""Google Earth Engine bridge: run server-side, pull results into xarray.

Requires ``earthengine-api`` plus ``xee`` (``pip install geolattice[rsense-gee]``).
"""

from __future__ import annotations

from typing import Any

from .stac import _require, aoi_to_geojson


def gee_init(project: str | None = None, *, authenticate: bool = False) -> None:
    """Initialize Earth Engine (optionally running the browser authentication flow)."""
    ee = _require("ee", "gee")
    if authenticate:
        ee.Authenticate()
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")


def gee_geometry(aoi: Any):
    ee = _require("ee", "gee")
    return ee.Geometry(aoi_to_geojson(aoi))


def gee_to_xarray(collection: Any, *, aoi: Any, scale: float, crs: str = "EPSG:4326",
                  bands: list[str] | None = None, dates: tuple[str, str] | None = None):
    """Open an ``ee.ImageCollection`` (object or asset id) lazily as an xarray Dataset via xee."""
    ee = _require("ee", "gee")
    _require("xee", "gee")
    import xarray as xr

    col = ee.ImageCollection(collection) if isinstance(collection, str) else collection
    geom = gee_geometry(aoi)
    col = col.filterBounds(geom)
    if dates:
        col = col.filterDate(*dates)
    if bands:
        col = col.select(bands)
    ds = xr.open_dataset(col, engine="ee", crs=crs, scale=scale, geometry=geom)
    ds = ds.rename({"lon": "x", "lat": "y"}) if "lon" in ds.dims else ds
    ds = ds.transpose("time", "y", "x", missing_dims="ignore")
    return ds.rio.write_crs(crs)


def gee_image_to_xarray(image: Any, *, aoi: Any, scale: float, crs: str = "EPSG:4326"):
    """Single ``ee.Image`` → xarray Dataset (wrapped as a one-image collection)."""
    ee = _require("ee", "gee")
    ds = gee_to_xarray(ee.ImageCollection([image]), aoi=aoi, scale=scale, crs=crs)
    return ds.isel(time=0, drop=True) if "time" in ds.dims else ds
