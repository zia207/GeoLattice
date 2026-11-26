"""Normalize tables into GeoDataFrames for plotting."""

from __future__ import annotations

from typing import Any

from geolattice.viz._imports import VisualizationError, require

_POLYGON = {"Polygon", "MultiPolygon"}
_LINE = {"LineString", "LinearRing", "MultiLineString"}
_POINT = {"Point", "MultiPoint"}


def as_geodataframe(data: Any) -> Any:
    """Return a GeoDataFrame without copying more than GeoPandas already does."""
    gpd = require("geopandas")
    pd = require("pandas")
    if isinstance(data, gpd.GeoDataFrame):
        return data
    if isinstance(data, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=data, crs=getattr(data, "crs", None))
    if isinstance(data, pd.DataFrame):
        if "geometry" not in getattr(data, "columns", ()):
            raise VisualizationError("The table has no geometry column.")
        return gpd.GeoDataFrame(data, geometry="geometry", crs=getattr(data, "crs", None))
    to_dataframe = getattr(data, "to_dataframe", None)
    if callable(to_dataframe):
        return as_geodataframe(to_dataframe())
    raise VisualizationError(
        "Expected a GeoDataFrame, a GeoSeries, or a table with a geometry column."
    )


def require_features(frame: Any) -> Any:
    """Reject an empty layer before handing it to Matplotlib."""
    if len(frame) == 0 or frame.geometry.isna().all():
        raise VisualizationError("The layer has no geometries to draw.")
    return frame


def geometry_kind(frame: Any) -> str:
    """Return ``polygon``, ``line``, ``point``, or ``mixed``."""
    kinds = set(frame.geom_type.dropna().unique())
    if not kinds or kinds <= _POLYGON:
        return "polygon"
    if kinds <= _LINE:
        return "line"
    if kinds <= _POINT:
        return "point"
    return "mixed"
