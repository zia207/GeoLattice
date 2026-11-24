"""Search STAC catalogues and keep the least cloudy scene.

``search_stac`` is the same search as ``geolattice.rsense.access.stac_search``.
Install the clients with ``pip install geolattice[rsense]`` (and
``geolattice[rsense-access]`` for Planetary Computer signing).
"""

from __future__ import annotations

from typing import Any


def search_stac(
    collection,
    aoi=None,
    dates=None,
    *,
    max_cloud=None,
    catalog: str = "planetary-computer",
    **kwargs,
):
    """Search ``collection`` over ``aoi`` and ``dates``.

    ``aoi`` may be a :class:`geolattice.vector.VectorBBox`, a ``(minx, miny, maxx, maxy)``
    tuple in longitude/latitude, a GeoDataFrame, or a GeoJSON geometry.
    ``dates`` is ``(start, end)``. ``max_cloud`` is the maximum scene cloud cover.
    """
    from geolattice.rsense.access.stac import stac_search

    return stac_search(
        collection,
        aoi=_aoi_arg(aoi),
        dates=dates,
        max_cloud=max_cloud,
        catalog=catalog,
        **kwargs,
    )


def get_clearest_item(collection, aoi=None, dates=None, *, items=None, **kwargs):
    """Return the scene with the lowest ``eo:cloud_cover`` that covers ``aoi``.

    Pass a collection name to search first, or pass ``items`` from
    :func:`search_stac`. Scenes whose footprint contains the area are preferred.
    When none contain it, the least cloudy intersecting scene is returned.
    An empty search returns ``None``.
    """
    from shapely.geometry import box, shape

    found = list(items if items is not None else search_stac(collection, aoi, dates, **kwargs))
    if not found:
        return None
    area = _lonlat_box(aoi)
    pool = found
    if area is not None:
        geom = box(*area)
        covering = [item for item in found if _contains(item, geom, shape)]
        pool = covering or found
    return min(pool, key=_cloud)


def _aoi_arg(aoi):
    if aoi is None:
        return None
    box = _lonlat_box(aoi)
    return box if box is not None else aoi


def _lonlat_box(aoi):
    from geolattice.vector.bbox import VectorBBox, to_crs

    if isinstance(aoi, VectorBBox):
        return to_crs(aoi, 4326).bounds
    if hasattr(aoi, "total_bounds") and hasattr(aoi, "crs"):
        return tuple(to_crs(aoi, 4326).total_bounds)
    if (
        isinstance(aoi, (tuple, list))
        and len(aoi) == 4
        and all(isinstance(v, (int, float)) for v in aoi)
    ):
        return tuple(float(v) for v in aoi)
    return None


def _contains(item, geom, shape) -> bool:
    geometry = getattr(item, "geometry", None)
    if not geometry and isinstance(item, dict):
        geometry = item.get("geometry")
    if not geometry:
        return False
    try:
        return shape(geometry).contains(geom)
    except Exception:
        return False


def _cloud(item) -> float:
    props = getattr(item, "properties", None)
    if props is None and isinstance(item, dict):
        props = item.get("properties")
    value = (props or {}).get("eo:cloud_cover", 100)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 100.0


__all__ = ["get_clearest_item", "search_stac"]
