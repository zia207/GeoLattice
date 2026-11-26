"""
Geoprocessing for vector layers.

These tools follow the desktop GIS operations and the geometry rules used by
`sf <https://github.com/r-spatial/sf>`_,
`terra <https://github.com/rspatial/terra>`_,
`GeoPandas <https://github.com/geopandas/geopandas>`_, and
`Shapely <https://github.com/shapely/shapely>`_:

- ``clip`` extracts the part of each input feature that falls inside a mask.
  Attributes come from the input. This is ``geopandas.clip`` and ``terra::crop``.
- ``union`` with one layer merges every geometry (``sf::st_union``,
  ``shapely.union_all``). With two layers it builds an overlay union that keeps
  the pieces covered by either layer (``GeoDataFrame.overlay(how="union")``).
- ``dissolve`` groups features by attributes and unions each group
  (``GeoDataFrame.dissolve``, ``terra::aggregate``).
- ``intersect`` keeps the pairwise overlap and the attributes of both layers
  (``sf::st_intersection``, ``overlay(how="intersection")``).
- ``erase`` removes one layer from another (``sf::st_difference``,
  ``terra::erase``, ``overlay(how="difference")``).
- ``convex_hull`` builds the convex hull of each feature, or of the whole layer
  (``sf::st_convex_hull``, ``terra::convHull``).
- ``buffer`` offsets each feature by a distance (``sf::st_buffer``,
  ``GeoSeries.buffer``).

Calculations are planar and run in the coordinate units of the input layer.
Project a geographic layer before buffering or overlaying when the distance
should be metres or feet. When two layers both have a CRS and the definitions
differ, the second layer is reprojected to the input CRS.

Inputs can be a GeoDataFrame, a GeoSeries, a pandas DataFrame with a geometry
column, a Shapely geometry, WKT, a GeoJSON geometry mapping, a sequence of
those, a sequence of ``VectorFeature``, or an open vector layer from
``open_vectors_data`` or ``open_cloud_native_vector_data``. The result is a
GeoDataFrame when GeoPandas is installed, otherwise a DataFrame whose
``geometry`` column holds Shapely geometries and whose ``attrs["crs"]`` stores
the CRS.

Example:
    from geolattice.vector import buffer, clip, dissolve, open_vectors_data

    parcels = open_vectors_data("parcels.gpkg")
    zoning = open_vectors_data("zoning.shp")
    inside = clip(parcels, zoning)
    dissolved = dissolve(inside, by="zone")
    buffered = buffer(dissolved, 30)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from shapely.geometry import GeometryCollection, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkt import loads as _load_wkt

from geolattice.core.bounds import Bounds
from geolattice.vector.vector_CRS import (
    VectorCRSError,
    _align_crs as _align_crs_impl,
    _crs_equal,
    _crs_for_output,
    _is_geographic,
    _reproject as _reproject_impl,
    _to_pyproj,
)

try:
    from shapely import union_all as _shapely_union_all
except ImportError:  # shapely 1.8
    _shapely_union_all = None

_READER_TYPES = {
    "VectorDataset",
    "VectorLayer",
    "CloudNativeDataset",
    "CloudNativeLayer",
}
_POLYGON = {"Polygon", "MultiPolygon"}
_LINE = {"LineString", "LinearRing", "MultiLineString"}
_POINT = {"Point", "MultiPoint"}
_CAP_STYLES = {
    "round": "round",
    "flat": "flat",
    "square": "square",
}
_JOIN_STYLES = {
    "round": "round",
    "mitre": "mitre",
    "miter": "mitre",
    "bevel": "bevel",
}
_AGGREGATIONS = ("first", "last", "sum", "mean", "min", "max", "median", "count", "nunique")


class GeoprocessError(ValueError):
    """Raised when a geoprocessing input cannot be used."""


class _Layer:
    """Geometries, attributes, and CRS normalized for one geoprocessing input."""

    def __init__(
        self,
        geometries: List[Optional[BaseGeometry]],
        attributes: pd.DataFrame,
        crs: Any,
    ) -> None:
        if len(attributes) != len(geometries):
            raise GeoprocessError(
                f"Attribute table has {len(attributes)} rows and "
                f"there are {len(geometries)} geometries."
            )
        self.geometries = geometries
        self.attributes = attributes.reset_index(drop=True)
        self.crs = crs


def clip(
    data: Any,
    mask: Any,
    *,
    keep_geom_type: bool = False,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Clip features to a mask.

    Each input feature is cut to the combined outline of the mask. Features
    that miss the mask are dropped. Attributes are copied from the input.
    A mask may be another layer, one geometry, or ``(minx, miny, maxx, maxy)``.

    Args:
        data: Features to clip
        mask: Polygon layer, geometry, or bounding box that defines the cookie cutter
        keep_geom_type: When True, drop result parts that are not the input geometry type
        make_valid: Repair invalid polygonal geometries before clipping
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Clipped features in the input CRS

    Example:
        clipped = clip(parcels, city_boundary)
        clipped = clip(parcels, (500000, 4100000, 501000, 4101000))
    """
    left, right = _load_pair(data, mask, geometry_column, "mask", make_valid, keep_geom_type)
    family = _result_family(left.geometries, keep_geom_type)
    cutter = _union_all(right.geometries)
    rows: List[Dict[str, Any]] = []
    dropped = 0
    if cutter is not None:
        for index, geom in enumerate(left.geometries):
            if geom is None or geom.is_empty:
                continue
            piece, rejected = _emit(geom.intersection(cutter), family)
            if rejected:
                dropped += 1
            if piece is None:
                continue
            rows.append(_attribute_row(left, index, piece))
    _warn_dropped(dropped, family)
    return _frame_from_rows(rows, list(left.attributes.columns), left.crs)


def union(
    data: Any,
    other: Any = None,
    *,
    keep_geom_type: bool = True,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Union geometries.

    With one layer, every geometry is merged into a single geometry and the
    result has an ``n_features`` count. Attribute aggregation of one layer is
    ``dissolve``.

    With two layers, overlapping areas are split. Each output feature is covered
    by the first layer, the second layer, or both, and carries the attributes
    of the layers that cover it. Shared attribute names get ``_1`` and ``_2``
    suffixes.

    Args:
        data: First layer, or the only layer for a unary union
        other: Second layer. Omit it to union every geometry in ``data``.
        keep_geom_type: When True, two-layer results keep the geometry type of ``data``
        make_valid: Repair invalid polygonal geometries first
        geometry_column: Geometry column name when an input is a DataFrame

    Returns:
        One merged feature, or the overlay pieces, in the CRS of ``data``
    """
    if other is None:
        layer = _prepare_one(data, geometry_column, make_valid, keep_geom_type=False)
        merged = _union_all(layer.geometries)
        row = {
            "n_features": _nonempty_count(layer.geometries),
            "geometry": merged if merged is not None else GeometryCollection(),
        }
        return _frame_from_rows([row], ["n_features"], layer.crs)

    left, right = _load_pair(data, other, geometry_column, "union", make_valid, keep_geom_type)
    family = _result_family(left.geometries, keep_geom_type)
    left_names = list(left.attributes.columns)
    right_names = list(right.attributes.columns)
    left_out, right_out = _suffix_names(left_names, right_names)
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for index, other_index, crossed in _crossings(left.geometries, right.geometries):
        piece, rejected = _emit(crossed, family)
        if rejected:
            dropped += 1
        if piece is None:
            continue
        rows.append(_overlay_row(left, right, left_out, right_out, index, other_index, piece))

    right_union = _union_all(right.geometries)
    left_union = _union_all(left.geometries)
    for index, geom in enumerate(left.geometries):
        piece, rejected = _difference_piece(geom, right_union, family)
        if rejected:
            dropped += 1
        if piece is None:
            continue
        rows.append(_overlay_row(left, right, left_out, right_out, index, None, piece))
    for other_index, geom in enumerate(right.geometries):
        piece, rejected = _difference_piece(geom, left_union, family)
        if rejected:
            dropped += 1
        if piece is None:
            continue
        rows.append(_overlay_row(left, right, left_out, right_out, None, other_index, piece))
    _warn_dropped(dropped, family)
    return _frame_from_rows(rows, [*left_out, *right_out], left.crs)


def dissolve(
    data: Any,
    by: Any = None,
    aggfunc: Any = "first",
    *,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Dissolve features by attribute.

    Geometries that share the same ``by`` values are unioned. Other attributes
    are aggregated. ``by=None`` dissolves the whole layer into one feature.

    Args:
        data: Features to dissolve
        by: Attribute name, or a sequence of names. None dissolves every feature.
        aggfunc: ``first``, ``last``, ``sum``, ``mean``, ``min``, ``max``,
            ``median``, ``count``, ``nunique``, a callable, or a dict of
            column names to one of those
        make_valid: Repair invalid polygonal geometries before the union
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        One feature per group, in the input CRS

    Example:
        regions = dissolve(counties, by="state", aggfunc={"population": "sum"})
    """
    layer = _prepare_one(data, geometry_column, make_valid, keep_geom_type=False)
    key_names = _group_keys(by, layer.attributes)
    value_columns = _value_columns(layer.attributes, key_names, aggfunc)
    columns = [*key_names, *value_columns]
    rows: List[Dict[str, Any]] = []
    if not key_names:
        groups: Iterable[Tuple[Tuple[Any, ...], pd.DataFrame]] = [((), layer.attributes)]
    elif len(layer.attributes) == 0:
        groups = []
    else:
        groups = _attribute_groups(layer.attributes, key_names)
    for key_vals, frame in groups:
        positions = [int(label) for label in frame.index]
        merged = _union_all(layer.geometries[position] for position in positions)
        row = {name: value for name, value in zip(key_names, key_vals)}
        row.update(_aggregate_frame(frame.loc[:, value_columns], aggfunc))
        row["geometry"] = merged if merged is not None else GeometryCollection()
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def intersect(
    data: Any,
    other: Any,
    *,
    keep_geom_type: bool = True,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Intersect two layers.

    Each overlapping pair becomes one output feature. The geometry is the
    shared area, line, or point, and the attributes come from both features.
    Shared attribute names get ``_1`` and ``_2`` suffixes. Pairs that only
    touch at a boundary are omitted when ``keep_geom_type`` is True and that
    boundary is a lower-dimension geometry.

    Args:
        data: First layer
        other: Second layer, geometry, or bounding box
        keep_geom_type: When True, keep only results of the same type as ``data``
        make_valid: Repair invalid polygonal geometries first
        geometry_column: Geometry column name when an input is a DataFrame

    Returns:
        Pairwise intersection features in the CRS of ``data``
    """
    left, right = _load_pair(data, other, geometry_column, "intersect", make_valid, keep_geom_type)
    family = _result_family(left.geometries, keep_geom_type)
    left_names = list(left.attributes.columns)
    right_names = list(right.attributes.columns)
    left_out, right_out = _suffix_names(left_names, right_names)
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for index, other_index, crossed in _crossings(left.geometries, right.geometries):
        piece, rejected = _emit(crossed, family)
        if rejected:
            dropped += 1
        if piece is None:
            continue
        rows.append(_overlay_row(left, right, left_out, right_out, index, other_index, piece))
    _warn_dropped(dropped, family)
    return _frame_from_rows(rows, [*left_out, *right_out], left.crs)


def erase(
    data: Any,
    other: Any,
    *,
    keep_geom_type: bool = True,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Erase one layer from another.

    The erase layer is combined into one geometry and subtracted from each
    input feature. Attributes come from the input. A feature that is completely
    covered by the erase layer is dropped.

    Args:
        data: Features to keep
        other: Features, geometry, or bounding box to remove
        keep_geom_type: When True, drop result parts that are not the input geometry type
        make_valid: Repair invalid polygonal geometries first
        geometry_column: Geometry column name when an input is a DataFrame

    Returns:
        Input features with the erase layer removed, in the input CRS
    """
    left, right = _load_pair(data, other, geometry_column, "erase", make_valid, keep_geom_type)
    family = _result_family(left.geometries, keep_geom_type)
    cutter = _union_all(right.geometries)
    rows: List[Dict[str, Any]] = []
    dropped = 0
    for index, geom in enumerate(left.geometries):
        piece, rejected = _difference_piece(geom, cutter, family)
        if rejected:
            dropped += 1
        if piece is None:
            continue
        rows.append(_attribute_row(left, index, piece))
    _warn_dropped(dropped, family)
    return _frame_from_rows(rows, list(left.attributes.columns), left.crs)


def convex_hull(
    data: Any,
    *,
    as_one: bool = False,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Build convex hulls.

    By default each feature is replaced by its own convex hull and attributes
    are kept. ``as_one=True`` returns one hull around every feature and an
    ``n_features`` count.

    Args:
        data: Features
        as_one: When True, return one hull for the whole layer
        make_valid: Repair invalid polygonal geometries first
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Hull features in the input CRS
    """
    layer = _prepare_one(data, geometry_column, make_valid, keep_geom_type=False)
    if as_one:
        merged = _union_all(layer.geometries)
        hull = GeometryCollection() if merged is None else merged.convex_hull
        row = {"n_features": _nonempty_count(layer.geometries), "geometry": hull}
        return _frame_from_rows([row], ["n_features"], layer.crs)
    rows: List[Dict[str, Any]] = []
    for index, geom in enumerate(layer.geometries):
        hull = None if geom is None else geom.convex_hull
        rows.append(_attribute_row(layer, index, hull))
    return _frame_from_rows(rows, list(layer.attributes.columns), layer.crs)


def buffer(
    data: Any,
    distance: Any,
    *,
    quad_segs: int = 16,
    resolution: Optional[int] = None,
    cap_style: Any = "round",
    join_style: Any = "round",
    mitre_limit: float = 5.0,
    single_sided: bool = False,
    dissolve: bool = False,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Buffer features by a distance.

    ``distance`` is a number in the units of the layer CRS, a sequence with
    one distance per feature, or the name of a numeric attribute. ``resolution``
    is an alias of ``quad_segs``, matching GeoPandas. Cap styles are ``round``,
    ``flat``, and ``square``. Join styles are ``round``, ``mitre``, and ``bevel``.
    ``dissolve=True`` unions the buffers into one geometry.

    A geographic CRS is buffered in degrees. Project the layer when the
    distance is a ground distance.

    Args:
        data: Features to buffer
        distance: Buffer width, a width per feature, or an attribute name
        quad_segs: Segments used to approximate a quarter circle
        resolution: Alias of ``quad_segs``
        cap_style: ``round``, ``flat``, or ``square``
        join_style: ``round``, ``mitre`` (or ``miter``), or ``bevel``
        mitre_limit: Limit for mitre joins
        single_sided: Buffer only the left side of a line
        dissolve: Union every buffer into one geometry
        make_valid: Repair invalid polygonal geometries first
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Buffered features in the input CRS

    Example:
        buffered = buffer(roads, 15, cap_style="flat", dissolve=True)
        buffered = buffer(sites, "radius_m")
    """
    layer = _prepare_one(data, geometry_column, make_valid, keep_geom_type=False)
    segments = _quad_segs(quad_segs if resolution is None else resolution)
    cap = _style_name(cap_style, _CAP_STYLES, "cap_style")
    join = _style_name(join_style, _JOIN_STYLES, "join_style")
    limit = float(mitre_limit)
    if not math.isfinite(limit) or limit <= 0:
        raise GeoprocessError("mitre_limit must be a finite number greater than zero")
    distances = _distances(distance, layer)
    _warn_geographic_buffer(layer.crs)
    rows: List[Dict[str, Any]] = []
    for index, (geom, width) in enumerate(zip(layer.geometries, distances)):
        if geom is None or geom.is_empty:
            buffered: Optional[BaseGeometry] = geom
        else:
            buffered = geom.buffer(
                width,
                quad_segs=segments,
                cap_style=cap,
                join_style=join,
                mitre_limit=limit,
                single_sided=bool(single_sided),
            )
        rows.append(_attribute_row(layer, index, buffered))
    if dissolve:
        merged = _union_all(row["geometry"] for row in rows)
        dissolved = {
            "n_features": _nonempty_count(layer.geometries),
            "geometry": merged if merged is not None else GeometryCollection(),
        }
        return _frame_from_rows([dissolved], ["n_features"], layer.crs)
    return _frame_from_rows(rows, list(layer.attributes.columns), layer.crs)


clipping = clip


def _load_pair(
    data: Any,
    other: Any,
    geometry_column: Optional[str],
    other_name: str,
    make_valid: bool,
    keep_geom_type: bool,
) -> Tuple[_Layer, _Layer]:
    left = _prepare_one(data, geometry_column, make_valid, keep_geom_type)
    right = _prepare_one(other, geometry_column, make_valid, keep_geom_type, allow_bbox=True)
    right = _align_crs(left, right, other_name)
    return left, right


def _prepare_one(
    data: Any,
    geometry_column: Optional[str],
    make_valid: bool,
    keep_geom_type: bool,
    allow_bbox: bool = False,
) -> _Layer:
    layer = _parse(data, geometry_column, allow_bbox=allow_bbox)
    repaired: List[Optional[BaseGeometry]] = []
    for geom in layer.geometries:
        family = None
        if keep_geom_type and geom is not None and not geom.is_empty:
            family = _family(geom.geom_type)
        geom = _repair(geom, make_valid)
        if family is not None:
            geom = _coerce_output(geom, family)
        repaired.append(geom)
    layer.geometries = repaired
    return layer


def _parse(data: Any, geometry_column: Optional[str], allow_bbox: bool) -> _Layer:
    if _is_bbox(data):
        if not allow_bbox:
            raise TypeError(
                "A bounding box can be a mask or overlay layer, not the input layer."
            )
        return _layer_from_bbox(data)
    if isinstance(data, Bounds):
        if not allow_bbox:
            raise TypeError("Bounds can be a mask or overlay layer, not the input layer.")
        return _layer_from_bbox((data.minx, data.miny, data.maxx, data.maxy))
    if isinstance(data, BaseGeometry):
        return _Layer([data], pd.DataFrame(index=range(1)), None)
    if isinstance(data, pd.DataFrame):
        return _layer_from_frame(data, geometry_column)
    if isinstance(data, pd.Series):
        return _layer_from_series(data)
    if _is_feature(data):
        return _layer_from_features([data])
    if _is_reader(data):
        return _layer_from_reader(data, geometry_column)
    if _is_geometry_record(data):
        return _Layer([_as_geometry(data)], pd.DataFrame(index=range(1)), None)
    if _is_sequence(data):
        return _layer_from_sequence(list(data))
    raise TypeError(
        "Expected a vector layer, GeoDataFrame, DataFrame, GeoSeries, "
        f"Shapely geometry, or sequence of features, got {type(data).__name__}"
    )


def _layer_from_bbox(bbox: Sequence[float]) -> _Layer:
    min_x, min_y, max_x, max_y = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
        raise GeoprocessError("Bounding box values must be finite")
    if min_x > max_x or min_y > max_y:
        raise GeoprocessError(
            f"Invalid bounding box: minx ({min_x}) > maxx ({max_x}) "
            f"or miny ({min_y}) > maxy ({max_y})"
        )
    return _Layer([box(min_x, min_y, max_x, max_y)], pd.DataFrame(index=range(1)), None)


def _layer_from_frame(frame: pd.DataFrame, geometry_column: Optional[str]) -> _Layer:
    geom_name = geometry_column
    if geom_name is None:
        active = getattr(frame, "_geometry_column_name", None)
        if isinstance(active, str) and active in frame.columns:
            geom_name = active
        elif "geometry" in frame.columns:
            geom_name = "geometry"
    if geom_name is None or geom_name not in frame.columns:
        raise GeoprocessError(
            "No geometry column was found. Pass geometry_column, or use a column named geometry."
        )
    geometries = [_as_geometry(value) for value in frame[geom_name].tolist()]
    attributes = frame.drop(columns=[geom_name]).copy()
    crs = getattr(frame, "crs", None)
    if crs is None:
        crs = frame.attrs.get("crs")
    return _Layer(geometries, attributes, crs)


def _layer_from_series(series: pd.Series) -> _Layer:
    geometries = [_as_geometry(value) for value in series.tolist()]
    return _Layer(
        geometries,
        pd.DataFrame(index=range(len(geometries))),
        getattr(series, "crs", None),
    )


def _layer_from_features(features: Sequence[Any]) -> _Layer:
    rows: List[Dict[str, Any]] = []
    geometries: List[Optional[BaseGeometry]] = []
    for feature in features:
        geometries.append(_as_geometry(feature.geometry))
        row = {"fid": feature.fid}
        properties = feature.properties
        if not isinstance(properties, dict):
            raise TypeError("VectorFeature.properties must be a dictionary")
        row.update(properties)
        rows.append(row)
    return _Layer(geometries, pd.DataFrame(rows), None)


def _layer_from_reader(reader: Any, geometry_column: Optional[str]) -> _Layer:
    frame = reader.to_dataframe()
    if not isinstance(frame, pd.DataFrame):
        raise GeoprocessError(
            f"{type(reader).__name__}.to_dataframe() returned "
            f"{type(frame).__name__}, not a DataFrame"
        )
    layer = _layer_from_frame(frame, geometry_column)
    if layer.crs is None:
        layer.crs = getattr(reader, "crs", None)
    return layer


def _layer_from_sequence(items: List[Any]) -> _Layer:
    if not items:
        raise GeoprocessError("Cannot geoprocess an empty sequence of features")
    first = items[0]
    if isinstance(first, BaseGeometry) or _is_missing(first) or _is_geometry_record(first):
        geometries = [_as_geometry(item) for item in items]
        return _Layer(geometries, pd.DataFrame(index=range(len(geometries))), None)
    if _is_feature(first):
        if not all(_is_feature(item) for item in items):
            raise TypeError("A feature sequence cannot mix VectorFeature objects with other values")
        return _layer_from_features(items)
    raise TypeError(
        "A sequence must contain Shapely geometries or VectorFeature objects, "
        f"not {type(first).__name__}"
    )


def _as_geometry(value: Any) -> Optional[BaseGeometry]:
    if isinstance(value, BaseGeometry):
        return value
    if _is_missing(value):
        return None
    if isinstance(value, dict) and "type" in value:
        from shapely.geometry import shape

        try:
            geometry = shape(value)
        except Exception as exc:
            raise TypeError(f"Could not read a GeoJSON geometry from {value!r}") from exc
        if not isinstance(geometry, BaseGeometry):
            raise TypeError(f"Could not read a GeoJSON geometry from {value!r}")
        return geometry
    if isinstance(value, str):
        text = value.strip()
        if text.upper().startswith(
            ("POINT", "LINESTRING", "POLYGON", "MULTI", "GEOMETRYCOLLECTION", "LINEARRING")
        ):
            try:
                return _load_wkt(text)
            except Exception as exc:
                raise TypeError(f"Could not read WKT geometry {value!r}") from exc
    raise TypeError(
        f"Expected a Shapely geometry, GeoJSON mapping, or WKT, got {type(value).__name__}"
    )


def _is_missing(value: Any) -> bool:
    if isinstance(value, BaseGeometry):
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return isinstance(missing, bool) and bool(missing)


def _is_bbox(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or isinstance(value, pd.DataFrame):
        return False
    if not isinstance(value, Sequence) or len(value) != 4:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _is_feature(value: Any) -> bool:
    return (
        not isinstance(value, (pd.DataFrame, pd.Series, BaseGeometry))
        and hasattr(value, "geometry")
        and hasattr(value, "properties")
        and hasattr(value, "fid")
    )


def _is_reader(value: Any) -> bool:
    return type(value).__name__ in _READER_TYPES and hasattr(value, "to_dataframe") and hasattr(
        value, "features"
    )


def _is_geometry_record(value: Any) -> bool:
    return isinstance(value, (dict, str)) and not isinstance(value, pd.Series)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def _repair(geom: Optional[BaseGeometry], make_valid: bool) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    try:
        valid = bool(geom.is_valid)
    except Exception:
        valid = True
    if valid:
        return geom
    if not make_valid:
        raise GeoprocessError(
            "Input contains an invalid geometry. Set make_valid=True, "
            "or repair the geometry before geoprocessing."
        )
    try:
        from shapely import make_valid as _make_valid
    except ImportError:
        return geom.buffer(0)
    return _make_valid(geom)


def _family(geom_type: str) -> Optional[str]:
    if geom_type in _POLYGON:
        return "polygon"
    if geom_type in _LINE:
        return "line"
    if geom_type in _POINT:
        return "point"
    return None


def _result_family(
    geometries: Sequence[Optional[BaseGeometry]],
    keep_geom_type: bool,
) -> Optional[str]:
    if not keep_geom_type:
        return None
    found = {
        family
        for geom in geometries
        if geom is not None and not geom.is_empty
        for family in [_family(geom.geom_type)]
        if family is not None
    }
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        warnings.warn(
            "Input has mixed geometry types, so keep_geom_type was not applied.",
            UserWarning,
            stacklevel=3,
        )
    return None


def _flatten(geom: Optional[BaseGeometry]) -> List[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "GeometryCollection":
        parts: List[BaseGeometry] = []
        for part in geom.geoms:
            parts.extend(_flatten(part))
        return parts
    return [geom]


def _coerce_output(geom: Optional[BaseGeometry], family: Optional[str]) -> Optional[BaseGeometry]:
    parts = _flatten(geom)
    if family is not None:
        parts = [part for part in parts if _family(part.geom_type) == family]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return _union_all(parts)


def _emit(
    geom: Optional[BaseGeometry],
    family: Optional[str],
) -> Tuple[Optional[BaseGeometry], bool]:
    if geom is None or geom.is_empty:
        return None, False
    result = _coerce_output(geom, family)
    return result, result is None


def _difference_piece(
    geom: Optional[BaseGeometry],
    cutter: Optional[BaseGeometry],
    family: Optional[str],
) -> Tuple[Optional[BaseGeometry], bool]:
    if geom is None or geom.is_empty:
        return None, False
    if cutter is None or cutter.is_empty:
        return _emit(geom, family)
    return _emit(geom.difference(cutter), family)


def _union_all(geometries: Iterable[Optional[BaseGeometry]]) -> Optional[BaseGeometry]:
    parts = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if _shapely_union_all is not None:
        return _shapely_union_all(parts)
    return unary_union(parts)


def _nonempty_count(geometries: Sequence[Optional[BaseGeometry]]) -> int:
    return sum(1 for geom in geometries if geom is not None and not geom.is_empty)


def _filled(geometries: Sequence[Optional[BaseGeometry]]) -> List[BaseGeometry]:
    empty = GeometryCollection()
    return [geom if geom is not None and not geom.is_empty else empty for geom in geometries]


def _crossings(
    left: Sequence[Optional[BaseGeometry]],
    right: Sequence[Optional[BaseGeometry]],
) -> List[Tuple[int, int, BaseGeometry]]:
    """Return the pairwise intersection of features whose bounding boxes overlap."""
    crossings: List[Tuple[int, int, BaseGeometry]] = []
    for index, other_index in _intersecting_index_pairs(left, right):
        geom = left[index]
        other = right[other_index]
        if geom is None or other is None or geom.is_empty or other.is_empty:
            continue
        crossings.append((index, other_index, geom.intersection(other)))
    return crossings


def _intersecting_index_pairs(
    left: Sequence[Optional[BaseGeometry]],
    right: Sequence[Optional[BaseGeometry]],
) -> List[Tuple[int, int]]:
    if not left or not right:
        return []
    left_geoms = _filled(left)
    right_geoms = _filled(right)
    try:
        from shapely.strtree import STRtree
    except ImportError:
        return _brute_pairs(left_geoms, right_geoms)
    tree = STRtree(right_geoms)
    try:
        hits = tree.query(left_geoms, predicate="intersects")
    except TypeError:
        return _brute_pairs(left_geoms, right_geoms)
    return [
        (int(left_index), int(right_index))
        for left_index, right_index in zip(hits[0], hits[1])
    ]


def _brute_pairs(
    left: Sequence[BaseGeometry],
    right: Sequence[BaseGeometry],
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for left_index, geom in enumerate(left):
        if geom.is_empty:
            continue
        for right_index, other in enumerate(right):
            if other.is_empty:
                continue
            if geom.intersects(other):
                pairs.append((left_index, right_index))
    return pairs


def _attribute_row(layer: _Layer, index: int, geometry: Optional[BaseGeometry]) -> Dict[str, Any]:
    values = layer.attributes.iloc[index].to_dict()
    row = {str(key): value for key, value in values.items()}
    row["geometry"] = geometry
    return row


def _overlay_row(
    left: _Layer,
    right: _Layer,
    left_names: Sequence[str],
    right_names: Sequence[str],
    left_index: Optional[int],
    right_index: Optional[int],
    geometry: BaseGeometry,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    left_values = None if left_index is None else left.attributes.iloc[left_index]
    right_values = None if right_index is None else right.attributes.iloc[right_index]
    for source, name in zip(left.attributes.columns, left_names):
        row[name] = None if left_values is None else left_values[source]
    for source, name in zip(right.attributes.columns, right_names):
        row[name] = None if right_values is None else right_values[source]
    row["geometry"] = geometry
    return row


def _suffix_names(left: Sequence[str], right: Sequence[str]) -> Tuple[List[str], List[str]]:
    overlap = set(left) & set(right)
    return (
        [f"{name}_1" if name in overlap else name for name in left],
        [f"{name}_2" if name in overlap else name for name in right],
    )


def _warn_dropped(count: int, family: Optional[str]) -> None:
    if count and family is not None:
        warnings.warn(
            f"Dropped {count} result geometry(ies) that were not {family} features "
            "because keep_geom_type is True.",
            UserWarning,
            stacklevel=3,
        )


def _group_keys(by: Any, attributes: pd.DataFrame) -> List[str]:
    if by is None:
        return []
    if isinstance(by, str):
        keys = [by]
    elif isinstance(by, Sequence) and not isinstance(by, (bytes, bytearray)):
        keys = [str(name) for name in by]
    else:
        raise TypeError("by must be an attribute name or a sequence of attribute names")
    if not keys:
        raise GeoprocessError("by must name at least one attribute")
    missing = [name for name in keys if name not in attributes.columns]
    if missing:
        raise GeoprocessError(
            f"Dissolve field(s) {missing} are not in the attribute table. "
            f"Available fields: {list(attributes.columns)}"
        )
    return keys


def _value_columns(attributes: pd.DataFrame, keys: Sequence[str], aggfunc: Any) -> List[str]:
    if isinstance(aggfunc, dict):
        unknown = [name for name in aggfunc if name not in attributes.columns]
        grouped = [name for name in aggfunc if name in keys]
        if unknown:
            raise GeoprocessError(
                f"aggfunc column(s) {unknown} are not in the attribute table. "
                f"Available fields: {list(attributes.columns)}"
            )
        if grouped:
            raise GeoprocessError(
                f"aggfunc cannot aggregate the dissolve field(s) {grouped}. "
                "Those fields define the groups."
            )
        return [str(name) for name in aggfunc]
    return [name for name in attributes.columns if name not in keys]


def _attribute_groups(
    attributes: pd.DataFrame,
    keys: Sequence[str],
) -> List[Tuple[Tuple[Any, ...], pd.DataFrame]]:
    groups: List[Tuple[Tuple[Any, ...], pd.DataFrame]] = []
    grouped = attributes.groupby(list(keys), dropna=False, sort=False, observed=False)
    for key, frame in grouped:
        # groupby returns a tuple whenever the key list has one or more names.
        key_vals = tuple(key) if isinstance(key, tuple) else (key,)
        groups.append((key_vals, frame))
    return groups


def _aggregate_frame(frame: pd.DataFrame, aggfunc: Any) -> Dict[str, Any]:
    if isinstance(aggfunc, dict):
        return {str(name): _apply_agg(frame[name], func) for name, func in aggfunc.items()}
    return {name: _apply_agg(frame[name], aggfunc) for name in frame.columns}


def _apply_agg(series: pd.Series, aggfunc: Any) -> Any:
    if isinstance(aggfunc, str):
        name = aggfunc.strip().lower()
        if name not in _AGGREGATIONS:
            raise GeoprocessError(
                f"aggfunc must be one of {list(_AGGREGATIONS)}, a callable, "
                f"or a dict, got {aggfunc!r}"
            )
        present = series.dropna()
        if name == "first":
            return pd.NA if present.empty else present.iloc[0]
        if name == "last":
            return pd.NA if present.empty else present.iloc[-1]
        if name == "count":
            return int(present.shape[0])
        if name == "nunique":
            return int(present.nunique())
        method = getattr(series, name)
        return method()
    if callable(aggfunc):
        return aggfunc(series)
    raise TypeError("aggfunc must be an aggregation name, a callable, or a dict of those")


def _distances(distance: Any, layer: _Layer) -> List[float]:
    count = len(layer.geometries)
    if isinstance(distance, str):
        if distance not in layer.attributes.columns:
            raise GeoprocessError(
                f"Buffer distance field {distance!r} is not in the attribute table. "
                f"Available fields: {list(layer.attributes.columns)}"
            )
        values = layer.attributes[distance].tolist()
    elif isinstance(distance, (int, float)) and not isinstance(distance, bool):
        values = [distance] * count
    elif isinstance(distance, Sequence) and not isinstance(distance, (str, bytes)):
        values = list(distance)
    else:
        raise TypeError(
            "distance must be a number, a sequence of numbers, or the name of a numeric attribute"
        )
    if len(values) != count:
        raise GeoprocessError(
            f"distance has {len(values)} value(s) for {count} feature(s)"
        )
    numbers: List[float] = []
    for value in values:
        if _is_missing(value):
            raise GeoprocessError("Buffer distance is missing for a feature")
        try:
            width = float(value)
        except (TypeError, ValueError) as exc:
            raise GeoprocessError(f"Buffer distance {value!r} is not a number") from exc
        if not math.isfinite(width):
            raise GeoprocessError(f"Buffer distance {value!r} is not finite")
        numbers.append(width)
    return numbers


def _quad_segs(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("quad_segs must be an integer")
    if value < 1:
        raise GeoprocessError("quad_segs must be 1 or greater")
    return value


def _style_name(value: Any, mapping: Dict[str, str], label: str) -> str:
    raw = value.name if hasattr(value, "name") else value
    key = str(raw).strip().lower()
    if key not in mapping:
        choices = sorted(set(mapping))
        raise GeoprocessError(f"{label} must be one of {choices}, got {value!r}")
    return mapping[key]


def _warn_geographic_buffer(crs: Any) -> None:
    if not _is_geographic(crs):
        return
    warnings.warn(
        "The layer CRS is geographic, so the buffer distance is in degrees. "
        "Project the layer to a metre or foot CRS when the distance is a ground distance.",
        UserWarning,
        stacklevel=3,
    )


# CRS helpers are defined in vector_CRS and wrapped here so CRS failures stay
# GeoprocessError for existing callers.


def _align_crs(data: _Layer, other: _Layer, other_name: str) -> _Layer:
    try:
        return _align_crs_impl(data, other, other_name)
    except VectorCRSError as exc:
        raise GeoprocessError(str(exc)) from exc


def _reproject(layer: _Layer, target_crs: Any) -> _Layer:
    try:
        return _reproject_impl(layer, target_crs)
    except VectorCRSError as exc:
        raise GeoprocessError(str(exc)) from exc


def _frame_from_rows(rows: Sequence[Dict[str, Any]], columns: Sequence[str], crs: Any) -> Any:
    ordered = [name for name in columns if name != "geometry"]
    if rows:
        records = [{name: row.get(name) for name in ordered} for row in rows]
        frame = pd.DataFrame(records, columns=ordered)
        frame["geometry"] = [row.get("geometry") for row in rows]
    else:
        frame = pd.DataFrame({name: pd.Series(dtype="object") for name in ordered})
        frame["geometry"] = pd.Series(dtype="object")
    return _as_geodataframe(frame, crs)


def _as_geodataframe(frame: pd.DataFrame, crs: Any) -> Any:
    output_crs = _crs_for_output(crs)
    try:
        import geopandas as gpd
    except ImportError:
        frame.attrs["crs"] = output_crs
        return frame
    try:
        return gpd.GeoDataFrame(frame, geometry="geometry", crs=output_crs)
    except Exception:
        frame.attrs["crs"] = output_crs
        return frame


__all__ = [
    "GeoprocessError",
    "buffer",
    "clip",
    "clipping",
    "convex_hull",
    "dissolve",
    "erase",
    "intersect",
    "union",
]
