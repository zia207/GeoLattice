"""
Geometry construction and cleanup for vector layers.

These tools follow the desktop GIS geometry operations used to lay out plots,
rows, and sections, then clean digitized corners before area is calculated:

- ``rectangles``, ``ovals``, and ``diamonds`` turn a point into a plot polygon
  of a fixed size. A rotation field, or an angle from
  ``oriented_minimum_bounding_box``, turns the polygon with the row.
- ``rotate`` and ``affine_transform`` move a generated grid onto the real row
  direction. Affine operations are applied in this order: scale, rotation,
  translation.
- ``translate`` shifts a whole section by an easting and northing.
- ``offset_lines`` copies one surveyed row into parallel rows.
- ``points_along_geometry`` places positions along a row at a fixed spacing.
- ``transect`` draws perpendicular lines that divide a path into plot widths.
- ``orthogonalize`` squares corners that were digitized slightly off 90 degrees.
- ``snap_points_to_grid`` rounds coordinates onto a regular grid.
- ``snap_geometries_to_layer`` pulls vertices onto a reference layer so edges
  that should meet actually meet.
- ``build_sections`` creates a labeled section grid, or dissolves plots that
  share a section attribute into one section polygon.
- ``add_geometry_attributes`` writes polygon area and perimeter, line length,
  and point X/Y coordinates.
- ``collect_geometries`` groups features into one multipart geometry without
  dissolving overlaps.
- ``centroids`` and ``point_on_surface`` create one marker point per feature.
  Point on surface is always inside a polygon.
- ``points_displacement`` spreads records that share one plot coordinate.
- ``check_validity`` and ``fix_geometries`` report and repair invalid geometry.
- ``remove_duplicate_vertices`` and ``delete_holes`` drop repeated vertices and
  interior rings.

Angles are degrees counterclockwise from east, which is the positive X axis.
The angle returned by ``oriented_minimum_bounding_box`` can be passed straight
to ``rectangles``, ``ovals``, ``diamonds``, ``rotate``, and ``affine_transform``.
Distances are in the coordinate units of the layer. Project a geographic layer
before using a ground distance in metres or feet.

Inputs can be a GeoDataFrame, a GeoSeries, a pandas DataFrame with a geometry
column, a Shapely geometry, WKT, a GeoJSON geometry mapping, a sequence of
those, a sequence of ``VectorFeature``, or an open vector layer. The result is
a GeoDataFrame when GeoPandas is installed.

Example:
    from geolattice.vector import (
        offset_lines,
        oriented_minimum_bounding_box,
        points_along_geometry,
        rectangles,
    )

    angle = oriented_minimum_bounding_box(cemetery)["angle"].iloc[0]
    rows = offset_lines(surveyed_row, 2.5, count=12, side="left", keep_original=True)
    graves = points_along_geometry(rows, 1.2)
    plots = rectangles(graves, 1.2, 2.5, angle=angle)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from shapely.affinity import rotate as _rotate_geom
from shapely.affinity import scale as _scale_geom
from shapely.affinity import translate as _translate_geom
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as _transform_coords
from shapely.validation import explain_validity

from geolattice.vector.geoprocessing_vector_data import (
    GeoprocessError,
    _aggregate_frame,
    _frame_from_rows,
    _group_keys,
    _is_geographic,
    _parse,
    _union_all,
    _value_columns,
)

try:
    from shapely import make_valid as _make_valid
except ImportError:  # shapely 1.8
    try:
        from shapely.validation import make_valid as _make_valid
    except ImportError:
        _make_valid = None

try:
    from shapely.strtree import STRtree
except ImportError:  # pragma: no cover - shapely has had STRtree since 1.6
    STRtree = None  # type: ignore[misc, assignment]

_POLYGON = {"Polygon", "MultiPolygon"}
_LINE = {"LineString", "LinearRing", "MultiLineString"}
_POINT = {"Point", "MultiPoint"}
_MULTI = {"MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"}
_SIDES = {"left", "right", "both"}
_SNAP_MODES = {
    "vertices": "vertices",
    "vertex": "vertices",
    "segments": "segments",
    "segment": "segments",
    "vertices_and_segments": "vertices_and_segments",
    "both": "vertices_and_segments",
}
_FIX_METHODS = {"make_valid", "buffer"}
_SHAPE_KINDS = ("rectangle", "oval", "diamond")


def rectangles(
    data: Any,
    width: Any,
    height: Any,
    angle: Any = 0,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create a rectangle centered on each point.

    ``width`` runs east-west and ``height`` runs north-south before ``angle``
    is applied. Both sizes, and the angle, can be a number, a sequence with
    one value per feature, or the name of a numeric attribute. ``angle`` is
    degrees counterclockwise from east.

    A multipoint becomes one rectangle per part. Attributes are copied onto
    each rectangle.

    Args:
        data: Point features, one per grave or plot corner anchor
        width: Rectangle width in layer units, or an attribute name
        height: Rectangle height in layer units, or an attribute name
        angle: Rotation in degrees, or an attribute name such as ``"rotation"``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Rectangle polygons in the input CRS

    Example:
        plots = rectangles(graves, 1.2, 2.5, angle="rotation")
    """
    return _marker_shapes(data, width, height, angle, "rectangle", 16, geometry_column)


def ovals(
    data: Any,
    width: Any,
    height: Any,
    angle: Any = 0,
    *,
    quad_segs: int = 16,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create an oval (ellipse) centered on each point.

    ``width`` and ``height`` are the full axis lengths before rotation.
    ``quad_segs`` is the number of segments used to approximate a quarter
    of the ellipse.

    Args:
        data: Point features
        width: Full width of the oval, or an attribute name
        height: Full height of the oval, or an attribute name
        angle: Rotation in degrees counterclockwise from east, or an attribute name
        quad_segs: Segments used to approximate a quarter ellipse
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Oval polygons in the input CRS

    Example:
        markers = ovals(graves, 0.6, 1.0, angle=row_angle)
    """
    return _marker_shapes(data, width, height, angle, "oval", quad_segs, geometry_column)


def diamonds(
    data: Any,
    width: Any,
    height: Any,
    angle: Any = 0,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create a diamond centered on each point.

    ``width`` is the horizontal diagonal and ``height`` is the vertical
    diagonal before rotation.

    Args:
        data: Point features
        width: Horizontal diagonal, or an attribute name
        height: Vertical diagonal, or an attribute name
        angle: Rotation in degrees counterclockwise from east, or an attribute name
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Diamond polygons in the input CRS

    Example:
        markers = diamonds(graves, 0.8, 1.4, angle=row_angle)
    """
    return _marker_shapes(data, width, height, angle, "diamond", 16, geometry_column)


def oriented_minimum_bounding_box(
    data: Any,
    *,
    per_feature: bool = False,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Build the rotated rectangle that encloses the features.

    The long-side ``angle`` is degrees counterclockwise from east, in the
    range ``(-90, 90]``. Pass that angle to ``rectangles`` or ``rotate`` to
    line plots and grids up with the dominant direction of the layer.
    ``width`` is the longer side and ``height`` is the shorter side.

    With ``per_feature=False`` the box encloses the whole layer and the result
    has one row. With ``per_feature=True`` each feature keeps its attributes
    and gets its own box.

    Args:
        data: Features to enclose
        per_feature: When True, bound each feature instead of the whole layer
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Rotated rectangle polygons with ``angle``, ``width``, and ``height``

    Example:
        frame = oriented_minimum_bounding_box(cemetery)
        plots = rectangles(graves, 1.2, 2.5, angle=frame["angle"].iloc[0])
    """
    layer = _load(data, geometry_column)
    if per_feature:
        geometries: List[Optional[BaseGeometry]] = []
        extras: List[Dict[str, Any]] = []
        for geom in layer.geometries:
            box_geom, measures = _oriented_box(geom)
            geometries.append(box_geom)
            extras.append(measures)
        return _output(layer, geometries, extras)

    merged = _union_all(layer.geometries)
    box_geom, measures = _oriented_box(merged)
    measures["n_features"] = _nonempty_count(layer.geometries)
    row = dict(measures)
    row["geometry"] = box_geom
    columns = ["angle", "width", "height", "n_features"]
    return _frame_from_rows([row], columns, layer.crs)


def affine_transform(
    data: Any,
    scale: Any = 1,
    angle: float = 0,
    translation: Any = (0.0, 0.0),
    *,
    origin: Any = "center",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Scale, rotate, and then translate the features.

    The operations are always applied in that order around ``origin``.
    ``scale`` is one number for both axes or an ``(x_scale, y_scale)`` pair.
    ``angle`` is degrees counterclockwise from east. ``translation`` is an
    ``(x, y)`` shift applied after rotation. ``origin`` is ``"center"`` (the
    middle of the layer bounds), ``"centroid"``, ``"feature"`` (each feature
    about its own centroid), or an ``(x, y)`` point. Use one layer origin when
    a generated grid should turn onto a surveyed row.

    Args:
        data: Features to transform
        scale: Uniform scale, or ``(scale_x, scale_y)``
        angle: Rotation in degrees counterclockwise, applied after scaling
        translation: ``(x, y)`` shift applied after rotation
        origin: Shared origin, or ``"feature"`` to use each centroid
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Transformed features in the input CRS

    Example:
        aligned = affine_transform(grid, scale=1, angle=row_angle, translation=(2, -1))
    """
    layer = _load(data, geometry_column)
    scale_x, scale_y = _scale_pair(scale)
    degrees = _finite(angle, "angle")
    xoff, yoff = _xy_pair(translation, "translation")
    shared = _shared_origin(layer, origin)
    geometries: List[Optional[BaseGeometry]] = []
    for geom in layer.geometries:
        if geom is None or geom.is_empty:
            geometries.append(geom)
            continue
        anchor = _feature_origin(geom) if shared is None else shared
        geometries.append(_apply_affine(geom, scale_x, scale_y, degrees, xoff, yoff, anchor))
    return _output(layer, geometries)


def rotate(
    data: Any,
    angle: float,
    *,
    origin: Any = "center",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Rotate features counterclockwise about a shared origin.

    ``angle`` is degrees from east. The default origin is the center of the
    layer bounds, so a grid rotates as one piece. ``origin="feature"`` rotates
    each feature about its own centroid. ``origin=(x, y)`` uses that point.

    Args:
        data: Features to rotate
        angle: Rotation in degrees counterclockwise
        origin: ``"center"``, ``"centroid"``, ``"feature"``, or an ``(x, y)`` point
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Rotated features in the input CRS

    Example:
        aligned = rotate(grid, row_angle, origin=(500000, 4100000))
    """
    return affine_transform(
        data,
        scale=1,
        angle=angle,
        translation=(0.0, 0.0),
        origin=origin,
        geometry_column=geometry_column,
    )


def translate(
    data: Any,
    xoff: float = 0,
    yoff: float = 0,
    zoff: float = 0,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Shift every feature by the same easting, northing, and height.

    Use this to nudge a section until it sits on the imagery. Offsets are in
    layer units.

    Args:
        data: Features to shift
        xoff: Shift in the X direction
        yoff: Shift in the Y direction
        zoff: Shift in the Z direction when the geometry has Z
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Shifted features in the input CRS

    Example:
        placed = translate(section, xoff=1.5, yoff=-0.4)
    """
    layer = _load(data, geometry_column)
    dx = _finite(xoff, "xoff")
    dy = _finite(yoff, "yoff")
    dz = _finite(zoff, "zoff")
    geometries: List[Optional[BaseGeometry]] = []
    for geom in layer.geometries:
        if geom is None or geom.is_empty or (dx == 0 and dy == 0 and dz == 0):
            geometries.append(geom)
        else:
            geometries.append(_translate_geom(geom, xoff=dx, yoff=dy, zoff=dz))
    return _output(layer, geometries)


def offset_lines(
    data: Any,
    distance: Any,
    count: int = 1,
    side: str = "left",
    *,
    keep_original: bool = False,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create parallel lines from each input line.

    Positive ``distance`` is the spacing between copies. ``side="left"`` is the
    left of the line direction, ``"right"`` is the opposite side, and
    ``"both"`` copies each side. ``count`` is the number of copies on each
    requested side. The result includes ``offset_index`` (negative on the
    right, positive on the left, zero for the source) and ``offset_distance``.

    A polygon is offset along its exterior ring.

    Args:
        data: Surveyed row lines
        distance: Spacing between parallel lines, or an attribute name
        count: Number of parallel copies on each requested side
        side: ``"left"``, ``"right"``, or ``"both"``
        keep_original: Include the surveyed line as ``offset_index`` 0
        quad_segs: Segments used to approximate a quarter circle at bends
        join_style: ``"round"``, ``"mitre"``, or ``"bevel"``
        mitre_limit: Limit for mitre joins
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Parallel line features in the input CRS

    Example:
        rows = offset_lines(surveyed, 2.5, count=12, side="left", keep_original=True)
    """
    layer = _load(data, geometry_column)
    distances = _measures(distance, layer, "distance")
    copies = _positive_int(count, "count", allow_zero=True)
    which = _choice(side, _SIDES, "side")
    if copies == 0 and not keep_original:
        raise GeoprocessError("count is 0 and keep_original is False, so no lines would be created")
    segments = _positive_int(quad_segs, "quad_segs")
    join = _choice(join_style, {"round", "mitre", "miter", "bevel"}, "join_style")
    if join == "miter":
        join = "mitre"
    limit = _finite(mitre_limit, "mitre_limit")
    if limit <= 0:
        raise GeoprocessError("mitre_limit must be greater than zero")
    _warn_geographic(layer.crs, "the offset distance")
    rows: List[Dict[str, Any]] = []
    columns = [*_column_names(layer), "offset_index", "offset_distance"]
    for index, (geom, spacing) in enumerate(zip(layer.geometries, distances)):
        if spacing == 0:
            raise GeoprocessError("Offset distance must not be zero")
        if spacing < 0:
            raise GeoprocessError("Offset distance must be greater than zero")
        lines = _linear_parts(geom)
        if geom is not None and not geom.is_empty and not lines:
            raise GeoprocessError(
                "offset_lines expects lines or polygons, "
                f"got {geom.geom_type}"
            )
        indexes = _offset_indexes(copies, which, keep_original)
        if not lines:
            if keep_original:
                rows.append(_row_from(layer, index, geom, offset_index=0, offset_distance=0.0))
            continue
        for offset_index in indexes:
            signed = float(offset_index) * spacing
            if offset_index == 0:
                rows.append(
                    _row_from(layer, index, geom, offset_index=0, offset_distance=0.0)
                )
                continue
            shifted = _offset_parts(lines, signed, segments, join, limit)
            if shifted is None:
                continue
            rows.append(
                _row_from(
                    layer,
                    index,
                    shifted,
                    offset_index=offset_index,
                    offset_distance=signed,
                )
            )
    return _frame_from_rows(rows, columns, layer.crs)


def points_along_geometry(
    data: Any,
    distance: Any,
    *,
    start_offset: Any = 0,
    end_offset: Any = 0,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Place points along each line at a fixed spacing.

    Points start at ``start_offset`` and repeat every ``distance`` until
    ``end_offset`` before the end of the line. Both offsets and the distance
    can be numbers or attribute names. Each point keeps the source attributes
    and records ``distance_along``. A closed ring does not repeat the start
    point at the end.

    Args:
        data: Row lines, or polygons whose exterior rings are walked
        distance: Spacing between points, or an attribute name
        start_offset: Distance to skip at the start of each line
        end_offset: Distance to leave unused at the end of each line
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Point features in the input CRS

    Example:
        graves = points_along_geometry(rows, 1.2, start_offset=0.6)
    """
    layer = _load(data, geometry_column)
    distances = _measures(distance, layer, "distance")
    starts = _measures(start_offset, layer, "start_offset")
    ends = _measures(end_offset, layer, "end_offset")
    _warn_geographic(layer.crs, "the point spacing")
    rows: List[Dict[str, Any]] = []
    columns = [*_column_names(layer), "distance_along"]
    for index, geom in enumerate(layer.geometries):
        spacing = distances[index]
        if spacing <= 0:
            raise GeoprocessError("Point spacing must be greater than zero")
        start = starts[index]
        end = ends[index]
        if start < 0 or end < 0:
            raise GeoprocessError("start_offset and end_offset must be zero or greater")
        for line in _linear_parts(geom):
            rows.extend(_points_on_line(layer, index, line, spacing, start, end))
    return _frame_from_rows(rows, columns, layer.crs)


def transect(
    data: Any,
    length: Any,
    interval: Any = None,
    *,
    angle: float = 90,
    side: str = "both",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Draw lines perpendicular to a path.

    ``interval`` is the plot width: a transect is drawn at the start, at every
    multiple of that width, and at the end. When ``interval`` is omitted, a
    transect is drawn at each vertex. ``angle`` is measured counterclockwise
    from the path direction, so 90 is perpendicular and points to the left of
    the path. ``length`` is the full transect length when ``side`` is
    ``"both"`` (the path sits in the middle). For ``"left"`` or ``"right"``
    the transect starts on the path and extends ``length`` to that side.

    Args:
        data: Path lines, or polygons whose exterior rings are split
        length: Transect length in layer units, or an attribute name
        interval: Plot width along the path, or an attribute name. Vertices are
            used when this is omitted.
        angle: Degrees counterclockwise from the path direction
        side: ``"both"``, ``"left"``, or ``"right"``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Transect lines with ``distance_along`` in the input CRS

    Example:
        dividers = transect(path, length=2.5, interval=1.2)
    """
    layer = _load(data, geometry_column)
    lengths = _measures(length, layer, "length")
    if interval is None:
        intervals: Optional[List[float]] = None
    else:
        intervals = _measures(interval, layer, "interval")
    degrees = _finite(angle, "angle")
    which = _choice(side, _SIDES, "side")
    _warn_geographic(layer.crs, "the transect length")
    rows: List[Dict[str, Any]] = []
    columns = [*_column_names(layer), "distance_along"]
    for index, geom in enumerate(layer.geometries):
        reach = lengths[index]
        if reach <= 0:
            raise GeoprocessError("Transect length must be greater than zero")
        step = None if intervals is None else intervals[index]
        if step is not None and step <= 0:
            raise GeoprocessError("Transect interval must be greater than zero")
        for line in _linear_parts(geom):
            for position in _transect_positions(line, step):
                cut = _transect_line(line, position, reach, degrees, which)
                if cut is None:
                    continue
                rows.append(_row_from(layer, index, cut, distance_along=position))
    return _frame_from_rows(rows, columns, layer.crs)


def orthogonalize(
    data: Any,
    *,
    angle_tolerance: float = 15,
    max_iterations: int = 25,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Square corners that are already close to a right angle.

    Each corner whose angle is within ``angle_tolerance`` degrees of 90 is
    moved onto the right angle that keeps the two neighboring vertices fixed
    and moves this vertex the shortest distance. The pass repeats until
    corners stop moving or ``max_iterations`` is reached. Line endpoints stay
    where they are. Rings and holes are each squared on their own.

    Args:
        data: Line or polygon features with hand-digitized corners
        angle_tolerance: How far from 90 degrees a corner may be and still be squared
        max_iterations: Maximum passes over the vertices
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with squared corners, in the input CRS

    Example:
        squared = orthogonalize(plots, angle_tolerance=15)
    """
    tolerance = _finite(angle_tolerance, "angle_tolerance")
    if tolerance < 0 or tolerance >= 90:
        raise GeoprocessError("angle_tolerance must be from 0 up to, but not including, 90")
    iterations = _positive_int(max_iterations, "max_iterations")
    layer = _load(data, geometry_column)
    geometries = [
        _map_lines_and_polygons(geom, lambda coords, closed: _orthogonal_coords(
            coords, closed, tolerance, iterations
        ))
        for geom in layer.geometries
    ]
    return _output(layer, geometries)


def snap_points_to_grid(
    data: Any,
    spacing_x: float,
    spacing_y: Optional[float] = None,
    *,
    origin_x: float = 0,
    origin_y: float = 0,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Round every vertex onto a regular grid.

    ``spacing_x`` and ``spacing_y`` are the grid cell sizes. When ``spacing_y``
    is omitted the cells are square. ``origin_x`` and ``origin_y`` are the
    coordinates of one grid intersection. GPS points and plot corners both
    move to the nearest intersection. Z values are left unchanged.

    Args:
        data: Features whose vertices should be rounded
        spacing_x: Grid size in the X direction
        spacing_y: Grid size in the Y direction. Defaults to ``spacing_x``.
        origin_x: X coordinate of a grid intersection
        origin_y: Y coordinate of a grid intersection
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features snapped to the grid, in the input CRS

    Example:
        cleaned = snap_points_to_grid(headstones, 0.1)
    """
    layer = _load(data, geometry_column)
    step_x = _finite(spacing_x, "spacing_x")
    step_y = step_x if spacing_y is None else _finite(spacing_y, "spacing_y")
    if step_x <= 0 or step_y <= 0:
        raise GeoprocessError("Grid spacing must be greater than zero")
    start_x = _finite(origin_x, "origin_x")
    start_y = _finite(origin_y, "origin_y")
    _warn_geographic(layer.crs, "the grid spacing")

    def _snap(x: float, y: float, z: Optional[float] = None) -> Tuple[float, ...]:
        snapped_x = start_x + _round_half_away((x - start_x) / step_x) * step_x
        snapped_y = start_y + _round_half_away((y - start_y) / step_y) * step_y
        if z is None:
            return (snapped_x, snapped_y)
        return (snapped_x, snapped_y, z)

    geometries = [_snap_geometry(geom, _snap) for geom in layer.geometries]
    return _output(layer, geometries)


def snap_geometries_to_layer(
    data: Any,
    reference: Any,
    tolerance: float,
    *,
    mode: str = "vertices_and_segments",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Move vertices onto a reference layer.

    ``mode="vertices"`` snaps to reference vertices. ``mode="segments"`` snaps
    to the nearest location on a reference edge. ``mode="vertices_and_segments"``
    uses a vertex when one is in range, and otherwise uses an edge. Nothing
    farther than ``tolerance`` moves.

    Pass the same layer as ``data`` and ``reference`` to close gaps between
    features. Vertices within ``tolerance`` of each other then collapse to one
    anchor, the vertex closest to the group centroid, so shared plot edges meet.
    A vertex that still does not meet a neighbor can snap to another feature's
    edge when the mode includes segments.

    Args:
        data: Features to adjust
        reference: Features that supply the snap targets
        tolerance: Maximum snap distance in layer units
        mode: ``"vertices"``, ``"segments"``, or ``"vertices_and_segments"``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Snapped features in the input CRS

    Example:
        closed = snap_geometries_to_layer(plots, plots, 0.15)
        heads = snap_geometries_to_layer(gps_points, plot_corners, 0.4)
    """
    layer = _load(data, geometry_column)
    reach = _finite(tolerance, "tolerance")
    if reach < 0:
        raise GeoprocessError("tolerance must be zero or greater")
    which = _SNAP_MODES.get(str(mode).strip().lower())
    if which is None:
        choices = ("vertices", "segments", "vertices_and_segments")
        raise GeoprocessError(f"mode must be one of {choices}, got {mode!r}")
    _warn_geographic(layer.crs, "the snap tolerance")
    if data is reference:
        snapped = _snap_layer_to_itself(layer.geometries, reach, which)
    else:
        other = _load(reference, geometry_column)
        other = _reproject_reference(layer, other)
        snapped = _snap_layer_to_reference(layer.geometries, other.geometries, reach, which)
    return _output(layer, snapped)


def build_sections(
    data: Any = None,
    *,
    by: Any = None,
    aggfunc: Any = "first",
    origin: Any = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    rows: Optional[int] = None,
    columns: Optional[int] = None,
    angle: float = 0,
    bounds: Any = None,
    crs: Any = None,
    section_field: str = "section",
    row_field: str = "row",
    column_field: str = "column",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Build section polygons and their attributes.

    Two inputs select the mode.

    Grid mode lays out a rows-by-columns grid of rectangles. ``width`` and
    ``height`` are the section size. ``origin`` is the southwest corner before
    rotation. When origin is omitted it is the southwest corner of ``bounds``
    or of ``data``. ``rows`` and ``columns`` are counted from that extent when
    they are omitted. ``angle`` rotates the whole grid counterclockwise about
    the origin. Each cell is named ``R{row}C{column}`` and stores 1-based
    ``row`` and ``column`` values. Row 1 is the southern row before rotation.

    Attribute mode unions the features that share a ``by`` value into one
    section polygon and aggregates the other attributes. ``n_features`` counts
    the plots in the section.

    Args:
        data: Layer used for the extent, the CRS, or the plots to group
        by: Attribute used to group existing plots into sections
        aggfunc: Aggregation used in attribute mode. A name, a callable, or a
            dict of those, matching ``dissolve``.
        origin: Southwest corner ``(x, y)`` of the grid before rotation
        width: Section width in layer units
        height: Section height in layer units
        rows: Number of section rows
        columns: Number of section columns
        angle: Grid rotation in degrees counterclockwise about the origin
        bounds: ``(minx, miny, maxx, maxy)`` used when the grid extent is not
            taken from ``data``
        crs: CRS assigned to a grid that was not taken from a layer
        section_field: Name of the section id attribute
        row_field: Name of the row number attribute
        column_field: Name of the column number attribute
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Section polygons in the input CRS

    Example:
        sections = build_sections(origin=(0, 0), width=20, height=30, rows=4, columns=6)
        grouped = build_sections(plots, by="section")
    """
    if by is not None and any(
        value is not None for value in (width, height, rows, columns, origin, bounds)
    ):
        raise GeoprocessError(
            "Pass either by= to group existing plots, or width and height to build a grid."
        )
    if by is not None:
        if data is None:
            raise GeoprocessError("Attribute mode requires a layer in data")
        return _sections_from_attribute(
            data, by, aggfunc, geometry_column
        )
    return _sections_from_grid(
        data,
        origin=origin,
        width=width,
        height=height,
        rows=rows,
        columns=columns,
        angle=angle,
        bounds=bounds,
        crs=crs,
        section_field=section_field,
        row_field=row_field,
        column_field=column_field,
        geometry_column=geometry_column,
    )


def add_geometry_attributes(
    data: Any,
    *,
    area: bool = True,
    perimeter: bool = True,
    length: bool = True,
    coordinates: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Add measurements taken from each geometry.

    Polygons can receive ``area`` and ``perimeter``. Lines can receive
    ``length``. Points can receive ``x`` and ``y``, and ``z`` when a point has
    a Z coordinate. Perimeter includes holes. Area excludes holes. Measurements
    are in the units of the layer CRS. A column is added only when the layer
    contains that geometry family and the matching flag is True.

    Args:
        data: Features to measure
        area: Add polygon area
        perimeter: Add polygon perimeter
        length: Add line length
        coordinates: Add point X, Y, and Z
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        The input features with measurement columns added

    Example:
        measured = add_geometry_attributes(plots)
    """
    layer = _load(data, geometry_column)
    families = _families(layer.geometries)
    want_area = bool(area) and "polygon" in families
    want_perimeter = bool(perimeter) and "polygon" in families
    want_length = bool(length) and "line" in families
    want_xy = bool(coordinates) and "point" in families
    want_z = want_xy and any(_has_z(geom) for geom in layer.geometries)
    if want_area or want_perimeter:
        _warn_geographic(layer.crs, "area and perimeter")
    extras: List[Dict[str, Any]] = []
    for geom in layer.geometries:
        extras.append(
            _measurement(
                geom,
                want_area,
                want_perimeter,
                want_length,
                want_xy,
                want_z,
            )
        )
    return _output(layer, list(layer.geometries), extras)


def collect_geometries(
    data: Any,
    by: Any = None,
    aggfunc: Any = "first",
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Group features into multipart geometries.

    Unlike ``dissolve``, overlapping parts are not unioned. Two family plots
    become one ``MultiPolygon``. Mixed geometry types become a
    ``GeometryCollection``. Attributes are aggregated with ``aggfunc``, which
    accepts the same names and dicts as ``dissolve``. ``n_features`` records
    how many input features went into the part.

    Args:
        data: Features to group
        by: Attribute name, or names, that define each group. Omit it to collect
            the whole layer.
        aggfunc: Aggregation for the other attributes
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        One multipart feature per group, in the input CRS

    Example:
        families = collect_geometries(plots, by="family_id")
    """
    layer = _load(data, geometry_column)
    keys = _group_keys(by, layer.attributes)
    columns, rows = _group_rows(layer, keys, aggfunc, _collect)
    return _frame_from_rows(rows, columns, layer.crs)


def centroids(
    data: Any,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create the centroid of each feature.

    The centroid of a polygon can fall outside a concave plot. Use
    ``point_on_surface`` when the marker must lie inside the polygon.

    Args:
        data: Features to label
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Point features in the input CRS

    Example:
        labels = centroids(plots)
    """
    layer = _load(data, geometry_column)
    geometries = [_centroid_of(geom) for geom in layer.geometries]
    return _output(layer, geometries)


def point_on_surface(
    data: Any,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create a point guaranteed to lie inside each feature.

    For a polygon this is a point on the surface, not the centroid. For a
    line it lies on the line. For a point it is the point itself.

    Args:
        data: Features to label
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Point features in the input CRS

    Example:
        markers = point_on_surface(plots)
    """
    layer = _load(data, geometry_column)
    geometries: List[Optional[BaseGeometry]] = []
    for geom in layer.geometries:
        if geom is None or geom.is_empty:
            geometries.append(geom)
        else:
            geometries.append(geom.representative_point())
    return _output(layer, geometries)


def points_displacement(
    data: Any,
    distance: float,
    *,
    proximity: float = 0,
    start_angle: float = 0,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Spread point features that were recorded at the same location.

    Points within ``proximity`` of each other are one stack of burials.
    ``proximity=0`` treats coordinates that match to 8 decimal places as the
    same plot. A stack is moved onto a circle of radius ``distance`` around
    the stack centroid. The first feature in the stack is placed at
    ``start_angle`` degrees counterclockwise from east, and the rest follow
    counterclockwise. Features that are alone stay where they are.
    ``displacement_group`` identifies the stack.

    Args:
        data: Point features, one per burial
        distance: Radius of the circle the stack is spread onto
        proximity: Distance within which points belong to the same plot
        start_angle: Direction of the first burial, in degrees from east
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Point features in the input CRS

    Example:
        spread = points_displacement(burials, distance=0.3, proximity=0.05)
    """
    layer = _load(data, geometry_column)
    radius = _finite(distance, "distance")
    if radius < 0:
        raise GeoprocessError("distance must be zero or greater")
    near = _finite(proximity, "proximity")
    if near < 0:
        raise GeoprocessError("proximity must be zero or greater")
    degrees = _finite(start_angle, "start_angle")
    _warn_geographic(layer.crs, "the displacement distance")
    anchors: List[Tuple[int, float, float]] = []
    for index, geom in enumerate(layer.geometries):
        if geom is None or geom.is_empty:
            continue
        if _family(geom.geom_type) != "point":
            raise GeoprocessError(
                "points_displacement expects points, "
                f"got {geom.geom_type}"
            )
        center = geom.centroid
        anchors.append((index, float(center.x), float(center.y)))
    groups = _cluster_pairs([(x, y) for _, x, y in anchors], near)
    geometries: List[Optional[BaseGeometry]] = list(layer.geometries)
    extras: List[Dict[str, Any]] = [
        {"displacement_group": pd.NA} for _ in layer.geometries
    ]
    for group_id, members in enumerate(groups, start=1):
        if len(members) < 2:
            continue
        feature_indexes = [anchors[member][0] for member in members]
        center_x = sum(anchors[member][1] for member in members) / len(members)
        center_y = sum(anchors[member][2] for member in members) / len(members)
        for slot, feature_index in enumerate(feature_indexes):
            theta = math.radians(degrees + slot * (360.0 / len(members)))
            target_x = center_x + radius * math.cos(theta)
            target_y = center_y + radius * math.sin(theta)
            geometries[feature_index] = _move_point_feature(
                layer.geometries[feature_index], target_x, target_y
            )
            extras[feature_index] = {"displacement_group": group_id}
    return _output(layer, geometries, extras)


def check_validity(
    data: Any,
    *,
    invalid_only: bool = False,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Report whether each geometry is valid.

    The result adds ``is_valid`` and ``validity_reason``. Run this before
    calculating area or publishing a section. ``invalid_only=True`` drops the
    valid rows.

    Args:
        data: Features to inspect
        invalid_only: Return only the invalid rows
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        The input features with validity columns

    Example:
        problems = check_validity(plots, invalid_only=True)
    """
    layer = _load(data, geometry_column)
    rows: List[Dict[str, Any]] = []
    for index, geom in enumerate(layer.geometries):
        valid, reason = _validity(geom)
        if invalid_only and valid:
            continue
        rows.append(_row_from(layer, index, geom, is_valid=valid, validity_reason=reason))
    columns = [
        name
        for name in _column_names(layer)
        if name not in ("is_valid", "validity_reason")
    ]
    columns.extend(["is_valid", "validity_reason"])
    return _frame_from_rows(rows, columns, layer.crs)


def fix_geometries(
    data: Any,
    *,
    method: str = "make_valid",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Repair invalid geometries.

    ``method="make_valid"`` uses the GEOS make-valid operation and keeps the
    location of the original vertices. ``method="buffer"`` uses a zero-width
    buffer, which repairs most invalid polygons. Already valid geometries are
    copied through unchanged. Repair a layer before calculating plot area.

    Args:
        data: Features to repair
        method: ``"make_valid"`` or ``"buffer"``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Repaired features in the input CRS

    Example:
        repaired = fix_geometries(plots)
    """
    chosen = _choice(method, _FIX_METHODS, "method")
    layer = _load(data, geometry_column)
    geometries = [_fix_one(geom, chosen) for geom in layer.geometries]
    return _output(layer, geometries)


def remove_duplicate_vertices(
    data: Any,
    tolerance: float = 0,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Drop repeated or nearly repeated vertices.

    Consecutive vertices closer than ``tolerance`` collapse to the first of
    the pair. ``tolerance=0`` removes exact duplicates only. A ring that no
    longer has three distinct corners becomes empty.

    Args:
        data: Features to clean
        tolerance: Maximum distance between vertices that are treated as duplicates
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features without duplicate vertices, in the input CRS

    Example:
        cleaned = remove_duplicate_vertices(plots, tolerance=0.01)
    """
    gap = _finite(tolerance, "tolerance")
    if gap < 0:
        raise GeoprocessError("tolerance must be zero or greater")
    layer = _load(data, geometry_column)
    if gap:
        _warn_geographic(layer.crs, "the vertex tolerance")
    geometries = [
        _map_lines_and_polygons(
            geom,
            lambda coords, closed: _dedupe_coords(coords, closed, gap),
        )
        for geom in layer.geometries
    ]
    return _output(layer, geometries)


def delete_holes(
    data: Any,
    minimum_area: float = 0,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Remove interior rings from polygons.

    ``minimum_area=0`` deletes every hole. A positive value deletes only holes
    whose area is smaller than that value. Lines and points pass through
    unchanged.

    Args:
        data: Polygon features to clean
        minimum_area: Holes smaller than this area are deleted. Zero deletes all holes.
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Polygons without the deleted holes, in the input CRS

    Example:
        solid = delete_holes(plots)
        kept = delete_holes(plots, minimum_area=1)
    """
    limit = _finite(minimum_area, "minimum_area")
    if limit < 0:
        raise GeoprocessError("minimum_area must be zero or greater")
    layer = _load(data, geometry_column)
    if limit:
        _warn_geographic(layer.crs, "the hole area")
    geometries = [_delete_holes_geom(geom, limit) for geom in layer.geometries]
    return _output(layer, geometries)


def _marker_shapes(
    data: Any,
    width: Any,
    height: Any,
    angle: Any,
    kind: str,
    quad_segs: int,
    geometry_column: Optional[str],
) -> Any:
    if kind not in _SHAPE_KINDS:
        raise GeoprocessError(f"Unknown shape {kind!r}")
    segments = _positive_int(quad_segs, "quad_segs")
    layer = _load(data, geometry_column)
    widths = _measures(width, layer, "width")
    heights = _measures(height, layer, "height")
    angles = _measures(angle, layer, "angle")
    _warn_geographic(layer.crs, "the plot width and height")
    rows: List[Dict[str, Any]] = []
    for index, geom in enumerate(layer.geometries):
        if geom is None or geom.is_empty:
            rows.append(_row_from(layer, index, Polygon()))
            continue
        if _family(geom.geom_type) != "point":
            raise GeoprocessError(
                f"{kind}s expects points, got {geom.geom_type}. "
                "Use centroids or point_on_surface to turn plots into anchor points."
            )
        for part in _point_parts(geom):
            shape = _shape_at(
                float(part.x),
                float(part.y),
                widths[index],
                heights[index],
                angles[index],
                kind,
                segments,
            )
            rows.append(_row_from(layer, index, shape))
    return _frame_from_rows(rows, _column_names(layer), layer.crs)


def _shape_at(
    x: float,
    y: float,
    width: float,
    height: float,
    angle: float,
    kind: str,
    quad_segs: int,
) -> BaseGeometry:
    if width <= 0 or height <= 0:
        raise GeoprocessError("width and height must be greater than zero")
    if kind == "rectangle":
        geom: BaseGeometry = box(-width / 2.0, -height / 2.0, width / 2.0, height / 2.0)
    elif kind == "diamond":
        geom = Polygon(
            [
                (0.0, height / 2.0),
                (width / 2.0, 0.0),
                (0.0, -height / 2.0),
                (-width / 2.0, 0.0),
            ]
        )
    else:
        geom = _scale_geom(
            Point(0.0, 0.0).buffer(1.0, quad_segs=quad_segs),
            xfact=width / 2.0,
            yfact=height / 2.0,
            origin=(0.0, 0.0),
        )
    if angle:
        geom = _rotate_geom(geom, angle, origin=(0.0, 0.0))
    return _translate_geom(geom, xoff=x, yoff=y)


def _oriented_box(geom: Optional[BaseGeometry]) -> Tuple[Optional[BaseGeometry], Dict[str, Any]]:
    empty = {"angle": math.nan, "width": math.nan, "height": math.nan}
    if geom is None or geom.is_empty:
        return geom, empty
    try:
        rectangle = geom.minimum_rotated_rectangle
    except Exception as exc:
        raise GeoprocessError(f"Could not build an oriented bounding box: {exc}") from exc
    if rectangle is None or rectangle.is_empty or rectangle.geom_type != "Polygon":
        return rectangle, empty
    coords = list(rectangle.exterior.coords)
    edges = []
    for start, end in zip(coords, coords[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))))
    if not edges:
        return rectangle, empty
    long_edge = max(edges, key=lambda item: item[0])
    short_edge = min(edges, key=lambda item: item[0])
    return rectangle, {
        "angle": _normalize_axis_angle(long_edge[1]),
        "width": long_edge[0],
        "height": short_edge[0],
    }


def _normalize_axis_angle(degrees: float) -> float:
    """Fold a line direction into (-90, 90] so either way along the line matches."""
    angle = degrees
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _apply_affine(
    geom: BaseGeometry,
    scale_x: float,
    scale_y: float,
    angle: float,
    xoff: float,
    yoff: float,
    origin: Point,
) -> BaseGeometry:
    scaled = geom
    if scale_x != 1 or scale_y != 1:
        scaled = _scale_geom(geom, xfact=scale_x, yfact=scale_y, origin=origin)
    rotated = scaled
    if angle:
        rotated = _rotate_geom(scaled, angle, origin=origin)
    if xoff or yoff:
        return _translate_geom(rotated, xoff=xoff, yoff=yoff)
    return rotated


def _shared_origin(layer: Any, origin: Any) -> Optional[Point]:
    if isinstance(origin, str):
        key = origin.strip().lower()
        if key == "feature":
            return None
        if key == "center":
            minx, miny, maxx, maxy = _layer_bounds(layer)
            return Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        if key == "centroid":
            merged = _union_all(layer.geometries)
            if merged is None or merged.is_empty:
                raise GeoprocessError("The layer has no geometries to use as an origin")
            center = merged.centroid
            return Point(float(center.x), float(center.y))
        raise GeoprocessError(
            'origin must be "center", "centroid", "feature", or an (x, y) point, '
            f"got {origin!r}"
        )
    x_coord, y_coord = _xy_pair(origin, "origin")
    return Point(x_coord, y_coord)


def _feature_origin(geom: BaseGeometry) -> Point:
    center = geom.centroid
    return Point(float(center.x), float(center.y))


def _offset_indexes(count: int, side: str, keep_original: bool) -> List[int]:
    indexes: List[int] = []
    if side in ("right", "both"):
        indexes.extend(range(-count, 0))
    if keep_original:
        indexes.append(0)
    if side in ("left", "both"):
        indexes.extend(range(1, count + 1))
    return indexes


def _offset_parts(
    lines: Sequence[LineString],
    distance: float,
    quad_segs: int,
    join_style: str,
    mitre_limit: float,
) -> Optional[BaseGeometry]:
    parts: List[BaseGeometry] = []
    for line in lines:
        if line.is_empty or line.length == 0:
            continue
        offset = _offset_curve(line, distance, quad_segs, join_style, mitre_limit)
        if offset is None or offset.is_empty:
            continue
        parts.extend(_linear_parts(offset))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiLineString(parts)


def _offset_curve(
    line: LineString,
    distance: float,
    quad_segs: int,
    join_style: str,
    mitre_limit: float,
) -> Optional[BaseGeometry]:
    if hasattr(line, "offset_curve"):
        try:
            return line.offset_curve(
                distance,
                quad_segs=quad_segs,
                join_style=join_style,
                mitre_limit=mitre_limit,
            )
        except TypeError:
            return line.offset_curve(distance, quad_segs=quad_segs)
    from shapely.geometry import JOIN_STYLE

    styles = {
        "round": JOIN_STYLE.round,
        "mitre": JOIN_STYLE.mitre,
        "bevel": JOIN_STYLE.bevel,
    }
    return line.parallel_offset(
        distance,
        resolution=quad_segs,
        join_style=styles[join_style],
        mitre_limit=mitre_limit,
    )


def _points_on_line(
    layer: Any,
    index: int,
    line: LineString,
    distance: float,
    start_offset: float,
    end_offset: float,
) -> List[Dict[str, Any]]:
    length = float(line.length)
    limit = length - end_offset
    if length == 0 or start_offset > limit + 1e-9:
        return []
    closed = bool(getattr(line, "is_ring", False))
    rows: List[Dict[str, Any]] = []
    position = start_offset
    while position <= limit + 1e-9:
        if closed and rows and abs(position - length) <= 1e-8:
            break
        point = line.interpolate(min(position, length))
        rows.append(_row_from(layer, index, point, distance_along=min(position, length)))
        position += distance
    return rows


def _transect_positions(line: LineString, interval: Optional[float]) -> List[float]:
    length = float(line.length)
    if length == 0:
        return []
    if interval is None:
        positions: List[float] = []
        walked = 0.0
        coords = list(line.coords)
        for previous, current in zip(coords, coords[1:]):
            positions.append(walked)
            walked += math.hypot(current[0] - previous[0], current[1] - previous[1])
        if not positions or abs(positions[-1] - length) > 1e-8:
            positions.append(length)
        if len(positions) >= 2 and abs(positions[-1] - positions[0] - length) <= 1e-6:
            if _coord_distance(coords[0], coords[-1]) <= 1e-8:
                positions.pop()
        return positions
    positions = [0.0]
    cursor = interval
    while cursor < length - 1e-8:
        positions.append(cursor)
        cursor += interval
    if abs(positions[-1] - length) > 1e-8:
        positions.append(length)
    return positions


def _transect_line(
    line: LineString,
    position: float,
    length: float,
    angle: float,
    side: str,
) -> Optional[LineString]:
    if line.length == 0:
        return None
    point = line.interpolate(position)
    tangent = _tangent_angle(line, position)
    heading = tangent + math.radians(angle)
    unit_x = math.cos(heading)
    unit_y = math.sin(heading)
    if side == "both":
        start = (point.x - unit_x * length / 2.0, point.y - unit_y * length / 2.0)
        end = (point.x + unit_x * length / 2.0, point.y + unit_y * length / 2.0)
    elif side == "left":
        start = (point.x, point.y)
        end = (point.x + unit_x * length, point.y + unit_y * length)
    else:
        start = (point.x, point.y)
        end = (point.x - unit_x * length, point.y - unit_y * length)
    if _coord_distance(start, end) == 0:
        return None
    return LineString([start, end])


def _tangent_angle(line: LineString, position: float) -> float:
    length = float(line.length)
    step = max(length * 1e-6, 1e-8)
    start = max(0.0, position - step)
    end = min(length, position + step)
    if abs(end - start) < 1e-12:
        return 0.0
    first = line.interpolate(start)
    second = line.interpolate(end)
    return math.atan2(second.y - first.y, second.x - first.x)


def _orthogonal_coords(
    coords: Sequence[Sequence[float]],
    closed: bool,
    tolerance: float,
    max_iterations: int,
) -> Optional[List[Tuple[float, ...]]]:
    points = [tuple(float(value) for value in coord) for coord in coords]
    if closed and len(points) >= 2 and _coord_distance(points[0], points[-1]) <= 1e-12:
        points = points[:-1]
    count = len(points)
    minimum = 3 if closed else 2
    if count < minimum:
        return None
    for _ in range(max_iterations):
        moved = False
        for index in range(count):
            if not closed and (index == 0 or index == count - 1):
                continue
            previous = points[(index - 1) % count]
            current = points[index]
            nxt = points[(index + 1) % count]
            updated = _square_corner(previous, current, nxt, tolerance)
            if updated is None:
                continue
            if _coord_distance(updated, current) <= 1e-8:
                continue
            points[index] = updated
            moved = True
        if not moved:
            break
    if closed:
        if len(points) < 3:
            return None
        return points + [points[0]]
    return points


def _square_corner(
    previous: Sequence[float],
    current: Sequence[float],
    nxt: Sequence[float],
    tolerance: float,
) -> Optional[Tuple[float, ...]]:
    """Move a corner onto the nearest location that makes a right angle.

    The locus of right angles with the two neighbors fixed is the circle that
    has those neighbors as its diameter. Projecting the corner onto that circle
    changes the digitized vertex as little as possible.
    """
    to_previous = (previous[0] - current[0], previous[1] - current[1])
    to_next = (nxt[0] - current[0], nxt[1] - current[1])
    length_previous = math.hypot(*to_previous)
    length_next = math.hypot(*to_next)
    if length_previous <= 1e-12 or length_next <= 1e-12:
        return None
    corner = _angle_between(to_previous, to_next)
    if abs(corner - 90.0) > tolerance or abs(corner - 90.0) <= 1e-6:
        return None
    center_x = (previous[0] + nxt[0]) / 2.0
    center_y = (previous[1] + nxt[1]) / 2.0
    radius = math.hypot(nxt[0] - previous[0], nxt[1] - previous[1]) / 2.0
    offset_x = current[0] - center_x
    offset_y = current[1] - center_y
    offset = math.hypot(offset_x, offset_y)
    if radius <= 1e-12 or offset <= 1e-12:
        return None
    scale = radius / offset
    updated_xy = (center_x + offset_x * scale, center_y + offset_y * scale)
    move = _coord_distance(updated_xy, current)
    if move > length_previous + length_next:
        return None
    return _keep_measures(current, updated_xy)


def _angle_between(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    start = math.atan2(first[1], first[0])
    end = math.atan2(second[1], second[0])
    difference = abs(math.degrees(end - start)) % 360.0
    if difference > 180.0:
        difference = 360.0 - difference
    return difference


def _snap_geometry(
    geom: Optional[BaseGeometry],
    snap: Callable[..., Tuple[float, ...]],
) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    return _transform_coords(snap, geom)


def _snap_layer_to_itself(
    geometries: Sequence[Optional[BaseGeometry]],
    tolerance: float,
    mode: str,
) -> List[Optional[BaseGeometry]]:
    flat: List[Tuple[float, ...]] = []
    owners: List[int] = []
    segments: List[Tuple[int, LineString]] = []
    for feature_index, geom in enumerate(geometries):
        coords = _collect_coords(geom)
        flat.extend(coords)
        owners.extend([feature_index] * len(coords))
        segments.extend(_segments_from_feature(geom, feature_index))
    if mode != "segments":
        groups = _cluster_pairs([(coord[0], coord[1]) for coord in flat], tolerance)
        for group in groups:
            if len(group) < 2:
                continue
            anchor = _cluster_anchor(flat, group)
            for member in group:
                flat[member] = _keep_measures(flat[member], anchor)
    if mode != "vertices":
        _snap_loose_vertices_to_segments(flat, owners, segments, tolerance, skip_same_feature=True)
    return _rebuild_all(geometries, flat)


def _snap_layer_to_reference(
    geometries: Sequence[Optional[BaseGeometry]],
    reference: Sequence[Optional[BaseGeometry]],
    tolerance: float,
    mode: str,
) -> List[Optional[BaseGeometry]]:
    ref_coords: List[Tuple[float, ...]] = []
    ref_points: List[Point] = []
    ref_segments: List[LineString] = []
    for geom in reference:
        for coord in _collect_coords(geom):
            ref_coords.append(coord)
            ref_points.append(Point(coord[0], coord[1]))
        ref_segments.extend(segment for _, segment in _segments_from_feature(geom, 0))
    point_tree = _strtree(ref_points) if mode != "segments" else None
    segment_tree = _strtree(ref_segments) if mode != "vertices" else None
    flat: List[Tuple[float, ...]] = []
    for geom in geometries:
        for coord in _collect_coords(geom):
            flat.append(
                _snap_one_coordinate(
                    coord,
                    tolerance,
                    mode,
                    ref_coords,
                    ref_points,
                    point_tree,
                    ref_segments,
                    segment_tree,
                )
            )
    return _rebuild_all(geometries, flat)


def _snap_one_coordinate(
    coord: Tuple[float, ...],
    tolerance: float,
    mode: str,
    ref_coords: Sequence[Tuple[float, ...]],
    ref_points: Sequence[Point],
    point_tree: Any,
    ref_segments: Sequence[LineString],
    segment_tree: Any,
) -> Tuple[float, ...]:
    vertex_hit: Optional[Tuple[float, float]] = None
    if mode != "segments" and ref_points:
        found = _nearest_vertex(coord[0], coord[1], tolerance, ref_coords, ref_points, point_tree)
        if found is not None:
            vertex_hit = found[0]
    segment_hit: Optional[Tuple[float, float]] = None
    if mode != "vertices" and ref_segments:
        found_segment = _nearest_segment(
            coord[0], coord[1], tolerance, ref_segments, segment_tree, None
        )
        if found_segment is not None:
            segment_hit = found_segment[0]
    # A reference vertex inside the tolerance wins over a closer edge, so GPS
    # points land on plot corners and digitized nodes line up with each other.
    if vertex_hit is not None:
        return _keep_measures(coord, vertex_hit)
    if segment_hit is not None:
        return _keep_measures(coord, segment_hit)
    return coord


def _snap_loose_vertices_to_segments(
    coords: List[Tuple[float, ...]],
    owners: Sequence[int],
    segments: Sequence[Tuple[int, LineString]],
    tolerance: float,
    skip_same_feature: bool,
) -> None:
    if not segments:
        return
    lines = [segment for _, segment in segments]
    tree = _strtree(lines)
    # A vertex that already shares its location with another vertex was anchored.
    location_counts: Dict[Tuple[float, float], int] = {}
    for coord in coords:
        key = (round(coord[0], 8), round(coord[1], 8))
        location_counts[key] = location_counts.get(key, 0) + 1
    for index, coord in enumerate(coords):
        key = (round(coord[0], 8), round(coord[1], 8))
        if location_counts.get(key, 0) > 1:
            continue
        owner = owners[index] if skip_same_feature else None
        found = _nearest_segment(coord[0], coord[1], tolerance, lines, tree, owner, segments)
        if found is None:
            continue
        point, _distance = found
        coords[index] = _keep_measures(coord, point)


def _nearest_vertex(
    x_coord: float,
    y_coord: float,
    tolerance: float,
    coords: Sequence[Tuple[float, ...]],
    points: Sequence[Point],
    tree: Any,
) -> Optional[Tuple[Tuple[float, float], float]]:
    indexes = _nearby_indexes(tree, points, x_coord, y_coord, tolerance)
    best: Optional[Tuple[Tuple[float, float], float, int]] = None
    for index in indexes:
        distance = math.hypot(coords[index][0] - x_coord, coords[index][1] - y_coord)
        if distance > tolerance + 1e-9:
            continue
        if best is None or distance < best[1] or (distance == best[1] and index < best[2]):
            best = ((coords[index][0], coords[index][1]), distance, index)
    if best is None:
        return None
    return best[0], best[1]


def _nearest_segment(
    x_coord: float,
    y_coord: float,
    tolerance: float,
    segments: Sequence[LineString],
    tree: Any,
    owner: Optional[int],
    described: Optional[Sequence[Tuple[int, LineString]]] = None,
) -> Optional[Tuple[Tuple[float, float], float]]:
    indexes = _nearby_indexes(tree, segments, x_coord, y_coord, tolerance)
    probe = Point(x_coord, y_coord)
    best: Optional[Tuple[Tuple[float, float], float, int]] = None
    for index in indexes:
        if owner is not None and described is not None and described[index][0] == owner:
            continue
        segment = segments[index]
        distance = float(segment.distance(probe))
        if distance > tolerance + 1e-9:
            continue
        nearest = segment.interpolate(segment.project(probe))
        point = (float(nearest.x), float(nearest.y))
        if best is None or distance < best[1] or (distance == best[1] and index < best[2]):
            best = (point, distance, index)
    if best is None:
        return None
    return best[0], best[1]


def _nearby_indexes(
    tree: Any,
    pool: Sequence[BaseGeometry],
    x_coord: float,
    y_coord: float,
    tolerance: float,
) -> List[int]:
    if not pool:
        return []
    if tolerance == 0:
        probe: BaseGeometry = Point(x_coord, y_coord)
    else:
        probe = box(
            x_coord - tolerance,
            y_coord - tolerance,
            x_coord + tolerance,
            y_coord + tolerance,
        )
    if tree is None:
        return [index for index, geom in enumerate(pool) if geom.intersects(probe)]
    found = tree.query(probe)
    indexes: List[int] = []
    for item in list(found):
        try:
            indexes.append(int(item))
        except (TypeError, ValueError):
            for index, geom in enumerate(pool):
                if geom.equals(item):
                    indexes.append(index)
                    break
    return indexes


def _strtree(geometries: Sequence[BaseGeometry]) -> Any:
    if not geometries or STRtree is None:
        return None
    return STRtree(list(geometries))


def _segments_from_feature(
    geom: Optional[BaseGeometry],
    feature_index: int,
) -> List[Tuple[int, LineString]]:
    segments: List[Tuple[int, LineString]] = []

    def _add(coords: Sequence[Sequence[float]]) -> None:
        for start, end in zip(coords, coords[1:]):
            if _coord_distance(start, end) <= 1e-15:
                continue
            segments.append((feature_index, LineString([start, end])))

    def _walk(part: BaseGeometry) -> None:
        kind = part.geom_type
        if kind in ("LineString", "LinearRing"):
            _add(list(part.coords))
        elif kind == "Polygon":
            _add(list(part.exterior.coords))
            for ring in part.interiors:
                _add(list(ring.coords))
        elif kind in _MULTI:
            for child in part.geoms:
                _walk(child)

    if geom is not None and not geom.is_empty:
        _walk(geom)
    return segments


def _cluster_anchor(
    coords: Sequence[Tuple[float, ...]],
    members: Sequence[int],
) -> Tuple[float, float]:
    """Pick the vertex closest to the centroid of the distinct positions.

    A closed ring repeats its first vertex, so that coordinate is counted once.
    Equal distances keep the earliest vertex.
    """
    unique: List[int] = []
    seen = set()
    for index in members:
        key = (round(coords[index][0], 8), round(coords[index][1], 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append(index)
    center_x = sum(coords[index][0] for index in unique) / len(unique)
    center_y = sum(coords[index][1] for index in unique) / len(unique)
    chosen = min(
        unique,
        key=lambda index: (
            (coords[index][0] - center_x) ** 2 + (coords[index][1] - center_y) ** 2,
            index,
        ),
    )
    return (coords[chosen][0], coords[chosen][1])


def _collect_coords(geom: Optional[BaseGeometry]) -> List[Tuple[float, ...]]:
    coords: List[Tuple[float, ...]] = []

    def _walk(part: BaseGeometry) -> None:
        kind = part.geom_type
        if kind == "Point":
            coords.append(tuple(part.coords[0]))
        elif kind in ("LineString", "LinearRing"):
            coords.extend(tuple(coord) for coord in part.coords)
        elif kind == "Polygon":
            coords.extend(tuple(coord) for coord in part.exterior.coords)
            for ring in part.interiors:
                coords.extend(tuple(coord) for coord in ring.coords)
        elif kind in _MULTI:
            for child in part.geoms:
                _walk(child)

    if geom is not None and not geom.is_empty:
        _walk(geom)
    return coords


def _rebuild_all(
    geometries: Sequence[Optional[BaseGeometry]],
    coords: Sequence[Tuple[float, ...]],
) -> List[Optional[BaseGeometry]]:
    cursor = [0]
    rebuilt: List[Optional[BaseGeometry]] = []
    for geom in geometries:
        rebuilt.append(_rebuild_geom(geom, coords, cursor))
    if cursor[0] != len(coords):
        raise GeoprocessError("Internal error while rebuilding snapped geometries")
    return rebuilt


def _rebuild_geom(
    geom: Optional[BaseGeometry],
    coords: Sequence[Tuple[float, ...]],
    cursor: List[int],
) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom

    def _take(count: int) -> List[Tuple[float, ...]]:
        start = cursor[0]
        chunk = list(coords[start : start + count])
        cursor[0] += count
        if len(chunk) != count:
            raise GeoprocessError("Internal error while reading snapped coordinates")
        return chunk

    def _walk(part: BaseGeometry) -> BaseGeometry:
        kind = part.geom_type
        if kind == "Point":
            return Point(_take(1)[0])
        if kind == "LineString":
            return LineString(_take(len(list(part.coords))))
        if kind == "LinearRing":
            return LineString(_take(len(list(part.coords))))
        if kind == "Polygon":
            exterior = _take(len(list(part.exterior.coords)))
            holes = [_take(len(list(ring.coords))) for ring in part.interiors]
            return Polygon(exterior, holes)
        if kind == "MultiPoint":
            return MultiPoint([_walk(child) for child in part.geoms])
        if kind == "MultiLineString":
            return MultiLineString([_walk(child) for child in part.geoms])
        if kind == "MultiPolygon":
            return MultiPolygon([_walk(child) for child in part.geoms])
        if kind == "GeometryCollection":
            return GeometryCollection([_walk(child) for child in part.geoms])
        raise GeoprocessError(f"Cannot snap {kind} geometries")

    return _walk(geom)


def _sections_from_attribute(
    data: Any,
    by: Any,
    aggfunc: Any,
    geometry_column: Optional[str],
) -> Any:
    layer = _load(data, geometry_column)
    keys = _group_keys(by, layer.attributes)
    columns, rows = _group_rows(layer, keys, aggfunc, _union_or_empty)
    return _frame_from_rows(rows, columns, layer.crs)


def _sections_from_grid(
    data: Any,
    *,
    origin: Any,
    width: Optional[float],
    height: Optional[float],
    rows: Optional[int],
    columns: Optional[int],
    angle: float,
    bounds: Any,
    crs: Any,
    section_field: str,
    row_field: str,
    column_field: str,
    geometry_column: Optional[str],
) -> Any:
    if width is None or height is None:
        raise GeoprocessError("Grid mode requires width and height")
    cell_width = _finite(width, "width")
    cell_height = _finite(height, "height")
    if cell_width <= 0 or cell_height <= 0:
        raise GeoprocessError("width and height must be greater than zero")
    degrees = _finite(angle, "angle")
    names = (section_field, row_field, column_field)
    if any(not isinstance(name, str) or not name for name in names):
        raise GeoprocessError("Section field names must be non-empty strings")
    if len(set(names)) < 3:
        raise GeoprocessError("section, row, and column field names must be different")
    layer = None if data is None else _load(data, geometry_column)
    output_crs = layer.crs if layer is not None and layer.crs is not None else crs
    extent = _section_extent(layer, bounds)
    if origin is None:
        if extent is None:
            raise GeoprocessError("Grid mode requires origin, bounds, or a layer to measure")
        origin_x, origin_y = extent[0], extent[1]
    else:
        origin_x, origin_y = _xy_pair(origin, "origin")
    row_count = rows
    column_count = columns
    if row_count is None or column_count is None:
        if extent is None:
            raise GeoprocessError("rows and columns are required when no extent is available")
        span_x = extent[2] - extent[0]
        span_y = extent[3] - extent[1]
        if column_count is None:
            if span_x <= 0:
                raise GeoprocessError("columns is required when the extent has no width")
            column_count = max(1, math.ceil(span_x / cell_width - 1e-9))
        if row_count is None:
            if span_y <= 0:
                raise GeoprocessError("rows is required when the extent has no height")
            row_count = max(1, math.ceil(span_y / cell_height - 1e-9))
    row_count = _positive_int(row_count, "rows")
    column_count = _positive_int(column_count, "columns")
    if layer is not None:
        _warn_geographic(layer.crs, "the section width and height")
    records: List[Dict[str, Any]] = []
    anchor = Point(origin_x, origin_y)
    for row_index in range(row_count):
        for column_index in range(column_count):
            minx = origin_x + column_index * cell_width
            miny = origin_y + row_index * cell_height
            cell = box(minx, miny, minx + cell_width, miny + cell_height)
            if degrees:
                cell = _rotate_geom(cell, degrees, origin=anchor)
            row_number = row_index + 1
            column_number = column_index + 1
            records.append(
                {
                    section_field: f"R{row_number}C{column_number}",
                    row_field: row_number,
                    column_field: column_number,
                    "geometry": cell,
                }
            )
    return _frame_from_rows(
        records,
        [section_field, row_field, column_field],
        output_crs,
    )


def _section_extent(layer: Any, bounds: Any) -> Optional[Tuple[float, float, float, float]]:
    if bounds is not None:
        return _bounds_tuple(bounds)
    if layer is None:
        return None
    try:
        return _layer_bounds(layer)
    except GeoprocessError:
        return None


def _group_rows(
    layer: Any,
    keys: Sequence[str],
    aggfunc: Any,
    combine: Callable[[Sequence[Optional[BaseGeometry]]], BaseGeometry],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    value_cols = [
        name
        for name in _value_columns(layer.attributes, keys, aggfunc)
        if name != "n_features"
    ]
    columns = [*keys, *value_cols, "n_features"]
    if not keys:
        merged = combine(layer.geometries)
        aggregated = (
            _aggregate_frame(layer.attributes[value_cols], aggfunc) if value_cols else {}
        )
        row = dict(aggregated)
        row["n_features"] = _nonempty_count(layer.geometries)
        row["geometry"] = merged
        return columns, [row]
    grouped = layer.attributes.groupby(list(keys), dropna=False, sort=False, observed=False)
    rows: List[Dict[str, Any]] = []
    for key, frame in grouped:
        key_values = tuple(key) if isinstance(key, tuple) else (key,)
        indexes = [int(value) for value in frame.index]
        aggregated = _aggregate_frame(frame[value_cols], aggfunc) if value_cols else {}
        row = {name: value for name, value in zip(keys, key_values)}
        row.update(aggregated)
        row["n_features"] = len(indexes)
        row["geometry"] = combine([layer.geometries[index] for index in indexes])
        rows.append(row)
    return columns, rows


def _union_or_empty(geometries: Sequence[Optional[BaseGeometry]]) -> BaseGeometry:
    merged = _union_all(_repair_quiet(geom) for geom in geometries)
    if merged is None:
        return GeometryCollection()
    return merged


def _measurement(
    geom: Optional[BaseGeometry],
    want_area: bool,
    want_perimeter: bool,
    want_length: bool,
    want_xy: bool,
    want_z: bool,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if geom is None or geom.is_empty:
        if want_area:
            values["area"] = math.nan
        if want_perimeter:
            values["perimeter"] = math.nan
        if want_length:
            values["length"] = math.nan
        if want_xy:
            values["x"] = math.nan
            values["y"] = math.nan
            if want_z:
                values["z"] = math.nan
        return values
    family = _family(geom.geom_type)
    if want_area:
        values["area"] = float(geom.area) if family == "polygon" else math.nan
    if want_perimeter:
        values["perimeter"] = float(geom.length) if family == "polygon" else math.nan
    if want_length:
        values["length"] = float(geom.length) if family == "line" else math.nan
    if want_xy:
        if family == "point":
            center = geom.centroid
            values["x"] = float(center.x)
            values["y"] = float(center.y)
            if want_z:
                values["z"] = _z_value(geom)
        else:
            values["x"] = math.nan
            values["y"] = math.nan
            if want_z:
                values["z"] = math.nan
    return values


def _z_value(geom: BaseGeometry) -> float:
    if geom.geom_type == "Point" and geom.has_z:
        return float(geom.coords[0][2])
    return math.nan


def _has_z(geom: Optional[BaseGeometry]) -> bool:
    return bool(geom is not None and not geom.is_empty and geom.has_z)


def _collect(geometries: Sequence[Optional[BaseGeometry]]) -> BaseGeometry:
    parts: List[BaseGeometry] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in _MULTI:
            parts.extend(list(geom.geoms))
        else:
            parts.append(geom)
    if not parts:
        return GeometryCollection()
    kinds = {part.geom_type for part in parts}
    if len(parts) == 1:
        return parts[0]
    if kinds <= {"Point"}:
        return MultiPoint(parts)
    if kinds <= {"LineString", "LinearRing"}:
        return MultiLineString([LineString(part.coords) for part in parts])
    if kinds <= {"Polygon"}:
        return MultiPolygon(parts)
    return GeometryCollection(parts)


def _centroid_of(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    return geom.centroid


def _move_point_feature(
    geom: Optional[BaseGeometry],
    x_coord: float,
    y_coord: float,
) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    center = geom.centroid
    return _translate_geom(geom, xoff=x_coord - float(center.x), yoff=y_coord - float(center.y))


def _cluster_pairs(
    coords: Sequence[Tuple[float, float]],
    proximity: float,
) -> List[List[int]]:
    count = len(coords)
    parent = list(range(count))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    if proximity <= 0:
        buckets: Dict[Tuple[float, float], List[int]] = {}
        for index, (x_coord, y_coord) in enumerate(coords):
            key = (round(x_coord, 8), round(y_coord, 8))
            buckets.setdefault(key, []).append(index)
        for members in buckets.values():
            for other in members[1:]:
                _union(members[0], other)
    else:
        cell = proximity
        grid: Dict[Tuple[int, int], List[int]] = {}
        placed: List[Tuple[int, int]] = []
        for index, (x_coord, y_coord) in enumerate(coords):
            cell_x = math.floor(x_coord / cell)
            cell_y = math.floor(y_coord / cell)
            placed.append((cell_x, cell_y))
            grid.setdefault((cell_x, cell_y), []).append(index)
        for index, (x_coord, y_coord) in enumerate(coords):
            cell_x, cell_y = placed[index]
            for step_x in (-1, 0, 1):
                for step_y in (-1, 0, 1):
                    for other in grid.get((cell_x + step_x, cell_y + step_y), []):
                        if other <= index:
                            continue
                        other_x, other_y = coords[other]
                        if math.hypot(other_x - x_coord, other_y - y_coord) <= proximity:
                            _union(index, other)
    groups: Dict[int, List[int]] = {}
    for index in range(count):
        groups.setdefault(_find(index), []).append(index)
    return list(groups.values())


def _validity(geom: Optional[BaseGeometry]) -> Tuple[bool, str]:
    if geom is None:
        return False, "Missing geometry"
    try:
        valid = bool(geom.is_valid)
    except Exception as exc:
        return False, str(exc)
    if valid:
        return True, "Valid Geometry"
    try:
        return False, str(explain_validity(geom))
    except Exception as exc:
        return False, str(exc)


def _fix_one(geom: Optional[BaseGeometry], method: str) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    try:
        valid = bool(geom.is_valid)
    except Exception:
        valid = False
    if valid:
        return geom
    if method == "buffer":
        return geom.buffer(0)
    if _make_valid is not None:
        return _make_valid(geom)
    return geom.buffer(0)


def _repair_quiet(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    try:
        valid = bool(geom.is_valid)
    except Exception:
        valid = True
    if valid:
        return geom
    return _fix_one(geom, "make_valid")


def _dedupe_coords(
    coords: Sequence[Sequence[float]],
    closed: bool,
    tolerance: float,
) -> Optional[List[Tuple[float, ...]]]:
    raw = [tuple(float(value) for value in coord) for coord in coords]
    if closed and len(raw) >= 2 and _coord_distance(raw[0], raw[-1]) <= max(tolerance, 0.0):
        raw = raw[:-1]
    if not raw:
        return None
    kept: List[Tuple[float, ...]] = [raw[0]]
    for coord in raw[1:]:
        if _coord_distance(kept[-1], coord) <= tolerance:
            continue
        kept.append(coord)
    if closed and len(kept) >= 2 and _coord_distance(kept[0], kept[-1]) <= tolerance:
        kept.pop()
    if closed:
        if len(kept) < 3:
            return None
        return kept + [kept[0]]
    if len(kept) < 2:
        return None
    return kept


def _delete_holes_geom(geom: Optional[BaseGeometry], minimum_area: float) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    kind = geom.geom_type
    if kind == "Polygon":
        return _delete_polygon_holes(geom, minimum_area)
    if kind == "MultiPolygon":
        parts = [_delete_polygon_holes(part, minimum_area) for part in geom.geoms]
        return MultiPolygon(parts)
    if kind == "GeometryCollection":
        return GeometryCollection([_delete_holes_geom(part, minimum_area) for part in geom.geoms])
    return geom


def _delete_polygon_holes(geom: Polygon, minimum_area: float) -> Polygon:
    if not list(geom.interiors):
        return geom
    kept = []
    for ring in geom.interiors:
        if minimum_area <= 0 or abs(Polygon(ring).area) < minimum_area:
            continue
        kept.append(ring)
    return Polygon(geom.exterior, kept)


def _map_lines_and_polygons(
    geom: Optional[BaseGeometry],
    ring_fn: Callable[[Sequence[Sequence[float]], bool], Optional[Sequence[Sequence[float]]]],
) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom
    kind = geom.geom_type
    if kind == "LineString":
        return _line_from_coords(ring_fn(list(geom.coords), False), LineString)
    if kind == "LinearRing":
        return _line_from_coords(ring_fn(list(geom.coords), True), LineString)
    if kind == "Polygon":
        return _polygon_from_rings(geom, ring_fn)
    if kind == "MultiLineString":
        parts = [
            _line_from_coords(ring_fn(list(part.coords), False), LineString) for part in geom.geoms
        ]
        parts = [part for part in parts if part is not None and not part.is_empty]
        if not parts:
            return MultiLineString()
        return MultiLineString(parts)
    if kind == "MultiPolygon":
        parts = [_polygon_from_rings(part, ring_fn) for part in geom.geoms]
        parts = [part for part in parts if part is not None and not part.is_empty]
        if not parts:
            return MultiPolygon()
        return MultiPolygon(parts)
    if kind == "GeometryCollection":
        parts = [_map_lines_and_polygons(part, ring_fn) for part in geom.geoms]
        parts = [part for part in parts if part is not None]
        return GeometryCollection(parts)
    return geom


def _line_from_coords(
    coords: Optional[Sequence[Sequence[float]]],
    constructor: Callable[[Sequence[Sequence[float]]], BaseGeometry],
) -> BaseGeometry:
    if not coords or len(list(coords)) < 2:
        return LineString()
    return constructor(list(coords))


def _polygon_from_rings(
    geom: Polygon,
    ring_fn: Callable[[Sequence[Sequence[float]], bool], Optional[Sequence[Sequence[float]]]],
) -> Polygon:
    exterior = ring_fn(list(geom.exterior.coords), True)
    if not exterior or len(list(exterior)) < 4:
        return Polygon()
    holes = []
    for ring in geom.interiors:
        hole = ring_fn(list(ring.coords), True)
        if hole and len(list(hole)) >= 4:
            holes.append(list(hole))
    return Polygon(list(exterior), holes)


def _linear_parts(geom: Optional[BaseGeometry]) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    kind = geom.geom_type
    if kind == "LineString":
        return [geom]
    if kind == "LinearRing":
        return [LineString(geom.coords)]
    if kind == "MultiLineString":
        return [part for part in geom.geoms if not part.is_empty]
    if kind == "Polygon":
        return [LineString(geom.exterior.coords)]
    if kind == "MultiPolygon":
        return [LineString(part.exterior.coords) for part in geom.geoms if not part.is_empty]
    if kind == "GeometryCollection":
        parts: List[LineString] = []
        for child in geom.geoms:
            parts.extend(_linear_parts(child))
        return parts
    return []


def _point_parts(geom: BaseGeometry) -> List[Point]:
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return [part for part in geom.geoms if not part.is_empty]
    if geom.geom_type == "GeometryCollection":
        parts: List[Point] = []
        for child in geom.geoms:
            parts.extend(_point_parts(child))
        return parts
    return []


def _load(data: Any, geometry_column: Optional[str]) -> Any:
    return _parse(data, geometry_column, allow_bbox=False)


def _output(
    layer: Any,
    geometries: Sequence[Optional[BaseGeometry]],
    extras: Optional[Sequence[Dict[str, Any]]] = None,
) -> Any:
    base = _column_names(layer)
    extra_names: List[str] = []
    if extras:
        for item in extras:
            for name in item:
                if name not in base and name not in extra_names:
                    extra_names.append(str(name))
    if extras is not None and len(extras) != len(geometries):
        raise GeoprocessError("Internal error while attaching geometry attributes")
    rows: List[Dict[str, Any]] = []
    for index, geom in enumerate(geometries):
        row = _row_from(layer, index, geom)
        if extras:
            row.update(extras[index])
        rows.append(row)
    return _frame_from_rows(rows, [*base, *extra_names], layer.crs)


def _row_from(
    layer: Any,
    index: int,
    geometry: Optional[BaseGeometry],
    **extra: Any,
) -> Dict[str, Any]:
    values = layer.attributes.iloc[index].to_dict()
    row = {str(key): value for key, value in values.items()}
    row.update(extra)
    row["geometry"] = geometry
    return row


def _column_names(layer: Any) -> List[str]:
    return [str(name) for name in layer.attributes.columns]


def _measures(value: Any, layer: Any, label: str) -> List[float]:
    count = len(layer.geometries)
    if isinstance(value, str):
        if value not in layer.attributes.columns:
            raise GeoprocessError(
                f"{label} field {value!r} is not in the attribute table. "
                f"Available fields: {list(layer.attributes.columns)}"
            )
        values = layer.attributes[value].tolist()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [value] * count
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(
            f"{label} must be a number, a sequence of numbers, or the name of a numeric attribute"
        )
    if len(values) != count:
        raise GeoprocessError(f"{label} has {len(values)} value(s) for {count} feature(s)")
    numbers: List[float] = []
    for item in values:
        numbers.append(_finite(item, label))
    return numbers


def _scale_pair(scale: Any) -> Tuple[float, float]:
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        value = _finite(scale, "scale")
        return value, value
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)) and len(scale) == 2:
        return _finite(scale[0], "scale_x"), _finite(scale[1], "scale_y")
    raise TypeError("scale must be a number or an (x, y) pair")


def _xy_pair(value: Any, label: str) -> Tuple[float, float]:
    if isinstance(value, BaseGeometry):
        if value.is_empty:
            raise GeoprocessError(f"{label} geometry is empty")
        point = value if value.geom_type == "Point" else value.centroid
        return float(point.x), float(point.y)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return _finite(value[0], label), _finite(value[1], label)
    raise TypeError(f"{label} must be an (x, y) pair or a point geometry")


def _bounds_tuple(value: Any) -> Tuple[float, float, float, float]:
    if isinstance(value, BaseGeometry):
        if value.is_empty:
            raise GeoprocessError("bounds geometry is empty")
        minx, miny, maxx, maxy = value.bounds
        return (float(minx), float(miny), float(maxx), float(maxy))
    if hasattr(value, "minx") and hasattr(value, "maxy"):
        return (float(value.minx), float(value.miny), float(value.maxx), float(value.maxy))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        minx, miny, maxx, maxy = (_finite(item, "bounds") for item in value)
        if minx > maxx or miny > maxy:
            raise GeoprocessError("bounds min values must not exceed max values")
        return (minx, miny, maxx, maxy)
    raise TypeError("bounds must be (minx, miny, maxx, maxy) or a geometry")


def _layer_bounds(layer: Any) -> Tuple[float, float, float, float]:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    found = False
    for geom in layer.geometries:
        if geom is None or geom.is_empty:
            continue
        left, bottom, right, top = geom.bounds
        minx = min(minx, left)
        miny = min(miny, bottom)
        maxx = max(maxx, right)
        maxy = max(maxy, top)
        found = True
    if not found:
        raise GeoprocessError("The layer has no geometries to measure")
    return (float(minx), float(miny), float(maxx), float(maxy))


def _round_half_away(value: float) -> float:
    """Round half away from zero so a coordinate on a cell boundary has one home."""
    if value >= 0:
        return math.floor(value + 0.5 + 1e-10)
    return math.ceil(value - 0.5 - 1e-10)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise GeoprocessError(f"{label} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GeoprocessError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise GeoprocessError(f"{label} must be a finite number, got {value!r}")
    return number


def _positive_int(value: Any, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{label} must be an integer")
    number = int(value)
    minimum = 0 if allow_zero else 1
    if number < minimum:
        raise GeoprocessError(f"{label} must be {minimum} or greater")
    return number


def _choice(value: Any, options: Iterable[str], label: str) -> str:
    key = str(value).strip().lower()
    allowed = {item.lower(): item.lower() for item in options}
    if key not in allowed:
        raise GeoprocessError(f"{label} must be one of {sorted(set(allowed))}, got {value!r}")
    return key


def _families(geometries: Sequence[Optional[BaseGeometry]]) -> set:
    found = set()
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        family = _family(geom.geom_type)
        if family is not None:
            found.add(family)
    return found


def _family(geom_type: str) -> Optional[str]:
    if geom_type in _POLYGON:
        return "polygon"
    if geom_type in _LINE:
        return "line"
    if geom_type in _POINT:
        return "point"
    return None


def _nonempty_count(geometries: Sequence[Optional[BaseGeometry]]) -> int:
    return sum(1 for geom in geometries if geom is not None and not geom.is_empty)


def _coord_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _keep_measures(template: Sequence[float], xy: Tuple[float, float]) -> Tuple[float, ...]:
    if len(template) > 2:
        return (xy[0], xy[1], *template[2:])
    return (xy[0], xy[1])


def _reproject_reference(data: Any, reference: Any) -> Any:
    from geolattice.vector.vector_CRS import align_vector_crs

    return align_vector_crs(data, reference, "reference")


def _warn_geographic(crs: Any, what: str) -> None:
    if not _is_geographic(crs):
        return
    warnings.warn(
        f"The layer CRS is geographic, so {what} is in degrees. "
        "Project the layer to a metre or foot CRS when the measurement is a ground distance.",
        UserWarning,
        stacklevel=3,
    )


__all__ = [
    "add_geometry_attributes",
    "affine_transform",
    "build_sections",
    "centroids",
    "check_validity",
    "collect_geometries",
    "delete_holes",
    "diamonds",
    "fix_geometries",
    "offset_lines",
    "oriented_minimum_bounding_box",
    "orthogonalize",
    "ovals",
    "point_on_surface",
    "points_along_geometry",
    "points_displacement",
    "rectangles",
    "remove_duplicate_vertices",
    "rotate",
    "snap_geometries_to_layer",
    "snap_points_to_grid",
    "transect",
    "translate",
]
