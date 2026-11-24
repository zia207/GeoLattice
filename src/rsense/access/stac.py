"""Search and load imagery from STAC catalogs as lazy xarray data cubes.

Requires the ``access`` extra: ``pip install geolattice[rsense-access]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

CATALOGS = {
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "usgs-landsat": "https://landsatlook.usgs.gov/stac-server",
    "cdse": "https://stac.dataspace.copernicus.eu/v1",
    "nasa-lpcloud": "https://cmr.earthdata.nasa.gov/stac/LPCLOUD",
}

_SENSOR_FROM_COLLECTION = (("sentinel-2", "sentinel2"), ("landsat", "landsat"),
                           ("modis-09", "modis"), ("sentinel-1", "sentinel1"))


def _require(module: str, extra: str = "access"):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        from geolattice.install_options import install_hint

        raise ImportError(f"{module} is required; install with `{install_hint('rsense-' + extra)}`") from e


def aoi_to_geojson(aoi: Any) -> dict:
    """Normalize an AOI to a GeoJSON geometry in EPSG:4326.

    Accepts a vector file path, GeoDataFrame/GeoSeries, shapely geometry,
    GeoJSON dict (geometry, Feature or FeatureCollection) or a bbox tuple.
    """
    import geopandas as gpd
    from shapely.geometry import box, mapping, shape

    if isinstance(aoi, (str, PathLike)):
        aoi = gpd.read_file(aoi)
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4 and all(isinstance(v, (int, float)) for v in aoi):
        return mapping(box(*aoi))
    if hasattr(aoi, "to_crs") and hasattr(aoi, "geometry"):  # GeoDataFrame / GeoSeries
        g = aoi.to_crs(4326) if aoi.crs else aoi
        return mapping(g.geometry.union_all() if hasattr(g.geometry, "union_all")
                       else g.geometry.unary_union)
    if isinstance(aoi, dict):
        if aoi.get("type") == "FeatureCollection":
            from shapely.ops import unary_union

            return mapping(unary_union([shape(f["geometry"]) for f in aoi["features"]]))
        if aoi.get("type") == "Feature":
            return aoi["geometry"]
        return aoi
    if hasattr(aoi, "__geo_interface__"):
        return json.loads(json.dumps(aoi.__geo_interface__))
    raise TypeError(f"Unsupported AOI type: {type(aoi).__name__}")


def _date_range(dates: Any) -> str | None:
    if dates is None or isinstance(dates, str):
        return dates
    start, end = dates
    return f"{start}/{end}"


def stac_search(collection: str | Sequence[str], *, aoi: Any = None,
                dates: str | tuple[str, str] | None = None, max_cloud: float | None = None,
                catalog: str = "planetary-computer", query: Mapping | None = None,
                limit: int | None = None, **kwargs: Any):
    """Search a STAC catalog and return a ``pystac.ItemCollection``.

    Example::

        items = rs.access.stac_search("sentinel-2-l2a", aoi="field.geojson",
                                      dates=("2026-05-01", "2026-09-30"), max_cloud=20)
    """
    pystac_client = _require("pystac_client")
    url = CATALOGS.get(catalog, catalog)
    modifier = None
    if "planetarycomputer" in url:
        pc = _require("planetary_computer")
        modifier = pc.sign_inplace
    client = pystac_client.Client.open(url, modifier=modifier)
    q = dict(query or {})
    if max_cloud is not None:
        q["eo:cloud_cover"] = {"lt": max_cloud}
    search = client.search(
        collections=[collection] if isinstance(collection, str) else list(collection),
        intersects=aoi_to_geojson(aoi) if aoi is not None else None,
        datetime=_date_range(dates), query=q or None, max_items=limit, **kwargs,
    )
    items = search.item_collection()
    if len(items) == 0:
        import warnings

        warnings.warn("STAC search returned no items; check AOI, dates and cloud threshold.",
                      UserWarning, stacklevel=2)
    return items


def stac_load(items, *, bands: Sequence[str] | None = None, aoi: Any = None,
              resolution: float | None = None, crs: Any = None,
              chunks: Mapping[str, int] | None = None, groupby: str | None = "solar_day",
              sensor: str | None = None, **kwargs: Any):
    """Load STAC items into a lazy, Dask-backed ``xarray.Dataset`` (one variable per band).

    ``groupby="solar_day"`` merges tiles from the same overpass. The detected sensor is
    stored in ``attrs["pyrsense:sensor"]`` so downstream functions can find QA bands.
    """
    odc_stac = _require("odc.stac")
    items = list(items)
    if not items:
        raise ValueError("No items to load")
    geom = None
    if aoi is not None:
        from odc.geo.geom import Geometry

        geom = Geometry(aoi_to_geojson(aoi), crs="EPSG:4326")
    ds = odc_stac.load(
        items, bands=list(bands) if bands else None, geopolygon=geom, resolution=resolution,
        crs=crs, chunks=dict(chunks or {"x": 2048, "y": 2048}), groupby=groupby, **kwargs,
    )
    coll = (items[0].collection_id or "").lower()
    detected = sensor or next((s for key, s in _SENSOR_FROM_COLLECTION if key in coll), None)
    if detected:
        ds.attrs["pyrsense:sensor"] = detected
    ds.attrs["pyrsense:collection"] = coll
    return ds
