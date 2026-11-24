"""
Vector analysis tools for attribute and spatial statistics.

These tools follow the desktop GIS vector analysis operations used by
`QGIS <https://docs.qgis.org/latest/en/docs/user_manual/processing_algs/qgis/vectoranalysis.html>`_
and common attribute workflows from ``sf`` / ``dplyr``:

Spatial / statistical analysis:

- ``basic_statistics_for_fields`` summarizes one attribute field.
- ``climb_along_line`` measures climb and descent on 3D line vertices.
- ``count_points_in_polygon`` counts points inside each polygon.
- ``dbscan_clustering`` and ``stdbscan_clustering`` densify clusters of points.
- ``distance_matrix`` reports distances from input points to target points.
- ``distance_to_nearest_hub_lines`` and ``distance_to_nearest_hub_points``
  attach each feature to its nearest hub.
- ``join_by_lines`` draws hub-and-spoke lines that match on an ID field.
- ``kmeans_clustering`` partitions features into *k* clusters.
- ``list_unique_values`` lists distinct attribute values and their counts.
- ``mean_coordinates`` computes the (optionally weighted) center of mass.
- ``nearest_neighbour_analysis`` reports Clark–Evans nearest-neighbour statistics.
- ``overlap_analysis`` measures how much each input polygon is covered by overlays.
- ``shortest_line_between_features`` draws the shortest link to neighbors.
- ``statistics_by_categories`` summarizes a field within category groups.
- ``sum_line_lengths`` totals line length and count inside each polygon.

Attribute manipulation:

- ``filter_features`` selects features that match attribute criteria.
- ``select_fields`` keeps a subset of attribute columns (geometry is retained).
- ``rename_fields`` renames attribute columns.
- ``calculate_fields`` creates or overwrites attributes from expressions.
- ``aggregate_attributes`` groups features and aggregates attributes
  (optionally dissolving geometries).
- ``sort_features`` orders features by attribute values.
- ``join_attributes`` joins a table or layer onto features by a common key.
- ``transform_field`` applies a function to one attribute column.
- ``string_field`` applies text operations to string attributes.
- ``update_where`` updates attributes where a condition is true.
- ``reshape_attributes`` melts (wide→long) or pivots (long→wide) attributes.
- ``drop_geometry`` returns the attribute table without geometries.

Distances are planar in the coordinate units of the layer unless a ground
unit is requested for hub distances. Project a geographic layer before using
metre or foot distances for clustering and neighbour tools.

Inputs can be a GeoDataFrame, a GeoSeries, a pandas DataFrame with a geometry
column, a Shapely geometry, WKT, a GeoJSON geometry mapping, a sequence of
those, a sequence of ``VectorFeature``, or an open vector layer. Spatial
results are a GeoDataFrame when GeoPandas is installed. Table-only tools
return a pandas DataFrame.

Example:
    from geolattice.vector import (
        calculate_fields,
        count_points_in_polygon,
        filter_features,
        join_attributes,
        kmeans_clustering,
        sum_line_lengths,
    )

    counted = count_points_in_polygon(parcels, trees)
    subset = filter_features(districts, "DIV_Name in ('Rangpur', 'Rajshahi')")
    totals = calculate_fields(rice, {"Total": "Aus + Aman + Boro"})
    joined = join_attributes(upazilas, poverty, on="ADM3_PCODE")
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from geolattice.vector.geoprocessing_vector_data import (
    GeoprocessError,
    _align_crs,
    _attribute_row,
    _frame_from_rows,
    _group_keys,
    _intersecting_index_pairs,
    _is_geographic,
    _is_missing,
    _parse,
    _suffix_names,
    _to_pyproj,
    dissolve,
)

try:
    from scipy.cluster.vq import kmeans2
except ImportError:  # pragma: no cover
    kmeans2 = None  # type: ignore[misc, assignment]

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None  # type: ignore[misc, assignment]

_LINE = {"LineString", "LinearRing", "MultiLineString"}
_MATRIX_TYPES = {
    "linear": "linear",
    "0": "linear",
    0: "linear",
    "standard": "standard",
    "1": "standard",
    1: "standard",
    "summary": "summary",
    "2": "summary",
    2: "summary",
}
_HUB_UNITS = {
    "meters": "meters",
    "metres": "meters",
    "m": "meters",
    0: "meters",
    "0": "meters",
    "feet": "feet",
    "ft": "feet",
    1: "feet",
    "1": "feet",
    "miles": "miles",
    "mi": "miles",
    2: "miles",
    "2": "miles",
    "kilometers": "kilometers",
    "kilometres": "kilometers",
    "km": "kilometers",
    3: "kilometers",
    "3": "kilometers",
    "layer": "layer",
    "layer_units": "layer",
    "layer units": "layer",
    4: "layer",
    "4": "layer",
}
_KMEANS_METHODS = {
    "farthest": "farthest",
    "farthest_points": "farthest",
    "farthest points": "farthest",
    0: "farthest",
    "0": "farthest",
    "kmeans++": "kmeans++",
    "k-means++": "kmeans++",
    "plusplus": "kmeans++",
    1: "kmeans++",
    "1": "kmeans++",
}
_SHORTEST_METHODS = {
    "nearest_point": "nearest_point",
    "nearest point": "nearest_point",
    "distance to nearest point on feature": "nearest_point",
    0: "nearest_point",
    "0": "nearest_point",
    "centroid": "centroid",
    "distance to feature centroid": "centroid",
    1: "centroid",
    "1": "centroid",
}
_TIME_UNITS = {
    "milliseconds": 1.0 / 86_400_000.0,
    "ms": 1.0 / 86_400_000.0,
    "seconds": 1.0 / 86_400.0,
    "s": 1.0 / 86_400.0,
    "minutes": 1.0 / 1_440.0,
    "min": 1.0 / 1_440.0,
    "hours": 1.0 / 24.0,
    "h": 1.0 / 24.0,
    "days": 1.0,
    "d": 1.0,
    "weeks": 7.0,
    "w": 7.0,
}
_METERS_PER = {
    "meters": 1.0,
    "feet": 0.3048,
    "miles": 1609.344,
    "kilometers": 1000.0,
}
_JOIN_HOW = {
    "left": "left",
    "inner": "inner",
    "right": "right",
    "outer": "outer",
    "full": "outer",
}
_STRING_OPS = {
    "lower": "lower",
    "upper": "upper",
    "title": "title",
    "capitalize": "capitalize",
    "strip": "strip",
    "lstrip": "lstrip",
    "rstrip": "rstrip",
    "replace": "replace",
    "contains": "contains",
    "startswith": "startswith",
    "endswith": "endswith",
    "length": "length",
    "len": "length",
    "slice": "slice",
    "substring": "slice",
    "cat": "cat",
    "concat": "cat",
    "pad": "pad",
    "zfill": "zfill",
}
_RESHAPE_HOW = {
    "long": "long",
    "melt": "long",
    "wide": "wide",
    "pivot": "wide",
}


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def basic_statistics_for_fields(
    data: Any,
    field: str,
    *,
    geometry_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute basic statistics for one attribute field.

    Numeric fields receive count, unique, empty, filled, min, max, range, sum,
    mean, median, std, CV, minority, majority, and quartiles. String-like fields
    receive count, unique, empty, filled, min, max, and string-length statistics.

    Args:
        data: Features whose attribute table is summarized
        field: Attribute name to summarize
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        One-row table of statistics

    Example:
        stats = basic_statistics_for_fields(parcels, "area_ha")
    """
    layer = _load(data, geometry_column)
    series = _field_series(layer, field)
    stats = _field_statistics(series)
    frame = pd.DataFrame([stats])
    frame.attrs["field"] = field
    return frame


def climb_along_line(
    data: Any,
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Calculate climb and descent along 3D line geometries.

    Each line must carry Z values on its vertices. The result keeps the input
    attributes and adds ``climb``, ``descent``, ``minelev``, and ``maxelev``.
    Layer totals are stored on ``result.attrs``.

    Args:
        data: Line features with Z coordinates
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Line features with climb statistics

    Example:
        climbed = climb_along_line(trails_3d)
    """
    layer = _load(data, geometry_column)
    rows: List[Dict[str, Any]] = []
    columns = [
        *_column_names(layer),
        "climb",
        "descent",
        "minelev",
        "maxelev",
    ]
    total_climb = 0.0
    total_descent = 0.0
    layer_min: Optional[float] = None
    layer_max: Optional[float] = None
    for index, geom in enumerate(layer.geometries):
        climb, descent, minelev, maxelev = _climb_stats(geom)
        row = _attribute_row(layer, index, geom)
        row["climb"] = climb
        row["descent"] = descent
        row["minelev"] = minelev
        row["maxelev"] = maxelev
        rows.append(row)
        if climb is not None:
            total_climb += float(climb)
            total_descent += float(descent)
            layer_min = float(minelev) if layer_min is None else min(layer_min, float(minelev))
            layer_max = float(maxelev) if layer_max is None else max(layer_max, float(maxelev))
    result = _frame_from_rows(rows, columns, layer.crs)
    result.attrs["total_climb"] = total_climb
    result.attrs["total_descent"] = total_descent
    result.attrs["min_elevation"] = layer_min
    result.attrs["max_elevation"] = layer_max
    return result


def count_points_in_polygon(
    polygons: Any,
    points: Any,
    *,
    weight_field: Optional[str] = None,
    class_field: Optional[str] = None,
    count_field: str = "NUMPOINTS",
    geometry_column: Optional[str] = None,
    points_geometry_column: Optional[str] = None,
) -> Any:
    """
    Count points that fall inside each polygon.

    When ``weight_field`` is set, the count is the sum of that numeric field.
    When ``class_field`` is set and no weight is given, each distinct class
    inside a polygon is counted once.

    Args:
        polygons: Polygon features that receive the count
        points: Point features to count
        weight_field: Optional numeric weight on the point layer
        class_field: Optional class field for unique-class counts
        count_field: Name of the output count column
        geometry_column: Geometry column for ``polygons``
        points_geometry_column: Geometry column for ``points``

    Returns:
        Polygon features with a count column

    Example:
        counted = count_points_in_polygon(parcels, trees, weight_field="biomass")
    """
    poly_layer = _load(polygons, geometry_column)
    point_layer = _align_crs(
        poly_layer,
        _load(points, points_geometry_column),
        "points",
    )
    if weight_field is not None:
        _require_field(point_layer, weight_field, "weight_field")
    if class_field is not None:
        _require_field(point_layer, class_field, "class_field")
    count_name = _unique_column_name(poly_layer, count_field)
    counts = [0.0 for _ in poly_layer.geometries]
    pairs = _intersecting_index_pairs(poly_layer.geometries, point_layer.geometries)
    class_sets: List[set] = [set() for _ in poly_layer.geometries]
    for poly_index, point_index in pairs:
        poly = poly_layer.geometries[poly_index]
        point = point_layer.geometries[point_index]
        if poly is None or point is None or poly.is_empty or point.is_empty:
            continue
        if not poly.intersects(point) and not poly.contains(point):
            # intersects covers boundary; still guard multiparts that only bbox-hit
            if not any(
                poly.contains(part) or poly.touches(part) or poly.covers(part)
                for part in _point_parts(point)
            ):
                continue
        if weight_field is not None:
            value = point_layer.attributes.iloc[point_index][weight_field]
            try:
                weight = float(value)
            except (TypeError, ValueError):
                weight = 0.0
            if not math.isfinite(weight):
                weight = 0.0
            counts[poly_index] += weight
        elif class_field is not None:
            class_sets[poly_index].add(point_layer.attributes.iloc[point_index][class_field])
        else:
            counts[poly_index] += float(len(_point_parts(point)))
    if weight_field is None and class_field is not None:
        counts = [float(len(values)) for values in class_sets]
    if weight_field is None:
        extras = [{count_name: int(count)} for count in counts]
    else:
        extras = [{count_name: float(count)} for count in counts]
    return _output_with_extras(poly_layer, extras)


def dbscan_clustering(
    data: Any,
    *,
    min_size: int = 5,
    eps: float = 1.0,
    border_as_noise: bool = False,
    cluster_field: str = "CLUSTER_ID",
    size_field: str = "CLUSTER_SIZE",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Cluster point features with DBSCAN.

    ``min_size`` is the minimum number of points in a cluster. ``eps`` is the
    neighbourhood radius in layer units. Noise points receive a null cluster
    id and a cluster size of 0.

    Args:
        data: Point features to cluster
        min_size: Minimum cluster size
        eps: Maximum neighbour distance
        border_as_noise: When True, treat border points as noise (DBSCAN*)
        cluster_field: Output cluster id column
        size_field: Output cluster size column
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Input features with cluster id and size columns

    Example:
        clusters = dbscan_clustering(samples, min_size=5, eps=50)
    """
    return _cluster_points(
        data,
        min_size=min_size,
        eps=eps,
        datetime_field=None,
        eps2=None,
        border_as_noise=border_as_noise,
        cluster_field=cluster_field,
        size_field=size_field,
        geometry_column=geometry_column,
    )


def distance_matrix(
    data: Any,
    target: Any,
    *,
    input_field: Optional[str] = None,
    target_field: Optional[str] = None,
    matrix_type: Union[str, int] = "linear",
    nearest_points: int = 0,
    geometry_column: Optional[str] = None,
    target_geometry_column: Optional[str] = None,
) -> Any:
    """
    Build a point-to-point distance matrix.

    Matrix types:

    - ``linear``: up to *k* rows per input point (InputID, TargetID, Distance)
    - ``standard``: one row per input point and one column per target
    - ``summary``: mean, std, min, and max distance per input point

    Args:
        data: Origin point features
        target: Destination point features
        input_field: Unique id field on the input layer
        target_field: Unique id field on the target layer
        matrix_type: ``linear``, ``standard``, or ``summary``
        nearest_points: Limit to the *k* nearest targets; ``0`` means all
        geometry_column: Geometry column for ``data``
        target_geometry_column: Geometry column for ``target``

    Returns:
        Point features carrying the distance matrix attributes

    Example:
        matrix = distance_matrix(stores, customers, matrix_type="summary", nearest_points=5)
    """
    kind = _choice_map(matrix_type, _MATRIX_TYPES, "matrix_type")
    left = _load(data, geometry_column)
    right = _align_crs(left, _load(target, target_geometry_column), "target")
    left_points = _feature_centers(left.geometries)
    right_points = _feature_centers(right.geometries)
    if not left_points or not right_points:
        return _frame_from_rows([], ["geometry"], left.crs)
    left_ids = _id_values(left, input_field, "InputID")
    right_ids = _id_values(right, target_field, "TargetID")
    k = len(right_points) if nearest_points in (None, 0) else int(nearest_points)
    if k < 1:
        raise GeoprocessError("nearest_points must be 0 (all) or a positive integer")
    k = min(k, len(right_points))
    _warn_geographic(left.crs, "distance matrix distances")
    distances = _pairwise_distances(left_points, right_points)
    rows: List[Dict[str, Any]] = []
    if kind == "linear":
        columns = ["InputID", "TargetID", "Distance", "geometry"]
        for i, origin in enumerate(left_points):
            order = np.argsort(distances[i])[:k]
            for j in order:
                rows.append(
                    {
                        "InputID": left_ids[i],
                        "TargetID": right_ids[int(j)],
                        "Distance": float(distances[i, j]),
                        "geometry": Point(origin),
                    }
                )
        return _frame_from_rows(rows, columns, left.crs)
    if kind == "standard":
        if k == len(right_points):
            columns = ["InputID", *[f"Target_{rid}" for rid in right_ids], "geometry"]
            for i, origin in enumerate(left_points):
                row = {"InputID": left_ids[i], "geometry": Point(origin)}
                for j, rid in enumerate(right_ids):
                    row[f"Target_{rid}"] = float(distances[i, j])
                rows.append(row)
        else:
            columns = ["InputID", *[f"NEAREST_{rank + 1}" for rank in range(k)], "geometry"]
            for i, origin in enumerate(left_points):
                order = list(np.argsort(distances[i])[:k])
                row = {"InputID": left_ids[i], "geometry": Point(origin)}
                for rank, j in enumerate(order):
                    row[f"NEAREST_{rank + 1}"] = float(distances[i, j])
                rows.append(row)
        return _frame_from_rows(rows, columns, left.crs)
    columns = ["InputID", "MEAN", "STDDEV", "MIN", "MAX", "geometry"]
    for i, origin in enumerate(left_points):
        values = distances[i, np.argsort(distances[i])[:k]]
        rows.append(
            {
                "InputID": left_ids[i],
                "MEAN": float(np.mean(values)),
                "STDDEV": float(np.std(values, ddof=0)),
                "MIN": float(np.min(values)),
                "MAX": float(np.max(values)),
                "geometry": Point(origin),
            }
        )
    return _frame_from_rows(rows, columns, left.crs)


def distance_to_nearest_hub_lines(
    data: Any,
    hubs: Any,
    *,
    hub_field: str,
    unit: Union[str, int] = "meters",
    geometry_column: Optional[str] = None,
    hubs_geometry_column: Optional[str] = None,
) -> Any:
    """
    Create lines from each feature center to its nearest hub center.

    Args:
        data: Source features
        hubs: Destination hub features
        hub_field: Hub attribute used as the hub name / id
        unit: ``meters``, ``feet``, ``miles``, ``kilometers``, or ``layer``
        geometry_column: Geometry column for ``data``
        hubs_geometry_column: Geometry column for ``hubs``

    Returns:
        Line features with hub name and HubDist

    Example:
        links = distance_to_nearest_hub_lines(houses, schools, hub_field="name")
    """
    return _distance_to_nearest_hub(
        data,
        hubs,
        hub_field=hub_field,
        unit=unit,
        as_lines=True,
        geometry_column=geometry_column,
        hubs_geometry_column=hubs_geometry_column,
    )


def distance_to_nearest_hub_points(
    data: Any,
    hubs: Any,
    *,
    hub_field: str,
    unit: Union[str, int] = "meters",
    geometry_column: Optional[str] = None,
    hubs_geometry_column: Optional[str] = None,
) -> Any:
    """
    Create points at each feature center with nearest-hub name and distance.

    Args:
        data: Source features
        hubs: Destination hub features
        hub_field: Hub attribute used as the hub name / id
        unit: ``meters``, ``feet``, ``miles``, ``kilometers``, or ``layer``
        geometry_column: Geometry column for ``data``
        hubs_geometry_column: Geometry column for ``hubs``

    Returns:
        Point features with HubName and HubDist

    Example:
        nearest = distance_to_nearest_hub_points(houses, schools, hub_field="name")
    """
    return _distance_to_nearest_hub(
        data,
        hubs,
        hub_field=hub_field,
        unit=unit,
        as_lines=False,
        geometry_column=geometry_column,
        hubs_geometry_column=hubs_geometry_column,
    )


def join_by_lines(
    hubs: Any,
    spokes: Any,
    *,
    hub_field: str,
    spoke_field: str,
    hub_fields: Optional[Sequence[str]] = None,
    spoke_fields: Optional[Sequence[str]] = None,
    geodesic: bool = False,
    geodesic_distance_km: float = 1000.0,
    antimeridian_split: bool = False,
    geometry_column: Optional[str] = None,
    spokes_geometry_column: Optional[str] = None,
) -> Any:
    """
    Join hubs to spokes with matching ID fields by drawing connecting lines.

    Matching uses equality between ``hub_field`` and ``spoke_field``. Non-point
    geometries connect at a point on their surface. Optional geodesic lines
    follow the ellipsoid and may be densified or split at the antimeridian.

    Args:
        hubs: Hub features
        spokes: Spoke features
        hub_field: Hub ID field
        spoke_field: Spoke ID field that matches the hub ID
        hub_fields: Hub attributes to copy; default is all
        spoke_fields: Spoke attributes to copy; default is all
        geodesic: Draw geodesic lines on the ellipsoid
        geodesic_distance_km: Vertex spacing for geodesic densification
        antimeridian_split: Split geodesic lines at ±180° longitude
        geometry_column: Geometry column for ``hubs``
        spokes_geometry_column: Geometry column for ``spokes``

    Returns:
        Line features connecting matching hubs and spokes

    Example:
        lines = join_by_lines(depots, deliveries, hub_field="depot_id", spoke_field="depot_id")
    """
    hub_layer = _load(hubs, geometry_column)
    spoke_layer = _align_crs(
        hub_layer,
        _load(spokes, spokes_geometry_column),
        "spokes",
    )
    _require_field(hub_layer, hub_field, "hub_field")
    _require_field(spoke_layer, spoke_field, "spoke_field")
    hub_cols = _selected_fields(hub_layer, hub_fields)
    spoke_cols = _selected_fields(spoke_layer, spoke_fields)
    hub_names, spoke_names = _suffix_names(hub_cols, spoke_cols)
    hub_lookup: Dict[Any, List[int]] = defaultdict(list)
    for index, value in enumerate(hub_layer.attributes[hub_field].tolist()):
        hub_lookup[_normalize_key(value)].append(index)
    hub_points = [_surface_point(geom) for geom in hub_layer.geometries]
    spoke_points = [_surface_point(geom) for geom in spoke_layer.geometries]
    rows: List[Dict[str, Any]] = []
    columns = [*hub_names, *spoke_names]
    for spoke_index, spoke_point in enumerate(spoke_points):
        if spoke_point is None:
            continue
        key = _normalize_key(spoke_layer.attributes.iloc[spoke_index][spoke_field])
        for hub_index in hub_lookup.get(key, []):
            hub_point = hub_points[hub_index]
            if hub_point is None:
                continue
            line = _connect_points(
                hub_point,
                spoke_point,
                hub_layer.crs,
                geodesic=geodesic,
                geodesic_distance_km=geodesic_distance_km,
                antimeridian_split=antimeridian_split,
            )
            if line is None:
                continue
            row: Dict[str, Any] = {"geometry": line}
            hub_values = hub_layer.attributes.iloc[hub_index]
            spoke_values = spoke_layer.attributes.iloc[spoke_index]
            for source, name in zip(hub_cols, hub_names):
                row[name] = hub_values[source]
            for source, name in zip(spoke_cols, spoke_names):
                row[name] = spoke_values[source]
            rows.append(row)
    return _frame_from_rows(rows, columns, hub_layer.crs)


hub_lines = join_by_lines


def kmeans_clustering(
    data: Any,
    n_clusters: int = 5,
    *,
    method: Union[str, int] = "farthest",
    cluster_field: str = "CLUSTER_ID",
    size_field: str = "CLUSTER_SIZE",
    random_state: Optional[int] = None,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Partition features into *k* clusters with 2D k-means.

    Clustering uses feature centroids. ``method`` is ``farthest`` (farthest
    points seeding) or ``kmeans++``.

    Args:
        data: Features to cluster
        n_clusters: Number of clusters
        method: Initial center selection method
        cluster_field: Output cluster id column
        size_field: Output cluster size column
        random_state: Seed for ``kmeans++``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Input features with cluster id and size columns

    Example:
        clusters = kmeans_clustering(samples, n_clusters=5, method="kmeans++")
    """
    if kmeans2 is None:
        raise ImportError("kmeans_clustering requires scipy")
    layer = _load(data, geometry_column)
    centers = _feature_centers(layer.geometries)
    if not centers:
        return _output_with_extras(layer, [])
    k = _positive_int(n_clusters, "n_clusters")
    if k > len(centers):
        raise GeoprocessError(
            f"n_clusters ({k}) cannot exceed the number of features ({len(centers)})"
        )
    _warn_geographic(layer.crs, "k-means distances")
    coords = np.asarray(centers, dtype=float)
    seed_method = _choice_map(method, _KMEANS_METHODS, "method")
    if seed_method == "farthest":
        labels = _kmeans_farthest(coords, k)
    else:
        rng = np.random.default_rng(random_state)
        # scipy uses global random state for minit='++'; seed numpy for stability
        if random_state is not None:
            np.random.seed(int(random_state) % (2**32 - 1))
        _, labels = kmeans2(coords, k, minit="++", missing="warn")
        labels = np.asarray(labels, dtype=int)
        _ = rng  # keep parameter meaningful for API users
    labels = np.asarray(labels, dtype=int)
    sizes = Counter(int(label) for label in labels)
    cluster_name = _unique_column_name(layer, cluster_field)
    size_name = _unique_column_name(layer, size_field, reserved={cluster_name})
    extras = [
        {
            cluster_name: int(label) + 1,
            size_name: int(sizes[int(label)]),
        }
        for label in labels
    ]
    # Map only features that contributed a center; empty geometries get nulls
    full_extras: List[Dict[str, Any]] = []
    center_index = 0
    for geom in layer.geometries:
        if geom is None or geom.is_empty:
            full_extras.append({cluster_name: pd.NA, size_name: 0})
        else:
            full_extras.append(extras[center_index])
            center_index += 1
    result = _output_with_extras(layer, full_extras)
    result.attrs["num_clusters"] = int(len(set(int(label) for label in labels)))
    return result


def list_unique_values(
    data: Any,
    fields: Union[str, Sequence[str]],
    *,
    geometry_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    List unique values of one or more attribute fields and count occurrences.

    Args:
        data: Features to summarize
        fields: Field name or sequence of field names
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Table of unique value combinations with a ``COUNT`` column. Totals are
        stored on ``result.attrs``.

    Example:
        unique = list_unique_values(parcels, ["zone", "landuse"])
    """
    layer = _load(data, geometry_column)
    keys = _group_keys(fields, layer.attributes)
    frame = layer.attributes.loc[:, keys].copy()
    grouped = (
        frame.groupby(keys, dropna=False, sort=False, observed=False)
        .size()
        .reset_index(name="COUNT")
    )
    values = [
        ", ".join("" if _is_missing(v) else str(v) for v in row)
        for row in grouped[keys].itertuples(index=False, name=None)
    ]
    grouped.attrs["total_values"] = int(len(grouped))
    grouped.attrs["unique_values"] = ", ".join(values)
    return grouped


def mean_coordinates(
    data: Any,
    *,
    weight_field: Optional[str] = None,
    uid_field: Optional[str] = None,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Compute the center of mass of geometries.

    When ``uid_field`` is set, one mean point is produced per unique value.
    When ``weight_field`` is set, feature centroids are weighted by that field.

    Args:
        data: Features to summarize
        weight_field: Optional numeric weight field
        uid_field: Optional grouping field
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Point features at the mean coordinate(s)

    Example:
        center = mean_coordinates(trees, weight_field="biomass", uid_field="stand_id")
    """
    layer = _load(data, geometry_column)
    if weight_field is not None:
        _require_field(layer, weight_field, "weight_field")
    if uid_field is not None:
        _require_field(layer, uid_field, "uid_field")
    groups: Dict[Any, List[int]] = defaultdict(list)
    for index in range(len(layer.geometries)):
        key = None if uid_field is None else layer.attributes.iloc[index][uid_field]
        groups[key].append(index)
    rows: List[Dict[str, Any]] = []
    columns = ([] if uid_field is None else [uid_field])
    for key, indices in groups.items():
        xs: List[float] = []
        ys: List[float] = []
        weights: List[float] = []
        for index in indices:
            geom = layer.geometries[index]
            if geom is None or geom.is_empty:
                continue
            point = geom.centroid
            weight = 1.0
            if weight_field is not None:
                value = layer.attributes.iloc[index][weight_field]
                try:
                    weight = float(value)
                except (TypeError, ValueError) as exc:
                    raise GeoprocessError(
                        f"weight_field value {value!r} is not numeric"
                    ) from exc
                if not math.isfinite(weight):
                    raise GeoprocessError(f"weight_field value {value!r} is not finite")
            xs.append(float(point.x))
            ys.append(float(point.y))
            weights.append(weight)
        if not xs:
            continue
        total = sum(weights)
        if total == 0:
            raise GeoprocessError("Sum of weights is zero for a mean-coordinate group")
        x = sum(x * w for x, w in zip(xs, weights)) / total
        y = sum(y * w for y, w in zip(ys, weights)) / total
        row: Dict[str, Any] = {"geometry": Point(x, y)}
        if uid_field is not None:
            row[uid_field] = key
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def nearest_neighbour_analysis(
    data: Any,
    *,
    geometry_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform Clark–Evans nearest-neighbour analysis on a point layer.

    Returns observed mean distance, expected mean distance under CSR, the
    nearest-neighbour index, point count, and Z-score.

    Args:
        data: Point features to analyze
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        One-row statistics table

    Example:
        nna = nearest_neighbour_analysis(samples)
    """
    layer = _load(data, geometry_column)
    points = _feature_centers(layer.geometries)
    n = len(points)
    if n < 2:
        raise GeoprocessError("Nearest neighbour analysis needs at least two points")
    _warn_geographic(layer.crs, "nearest neighbour distances")
    coords = np.asarray(points, dtype=float)
    if cKDTree is None:
        raise ImportError("nearest_neighbour_analysis requires scipy")
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    observed = float(np.mean(distances[:, 1]))
    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)
    area = max((maxx - minx) * (maxy - miny), 0.0)
    if area <= 0:
        raise GeoprocessError("Points are collinear or coincident; study area has no area")
    density = n / area
    expected = 0.5 / math.sqrt(density)
    nn_index = observed / expected if expected else float("nan")
    se = 0.26136 / math.sqrt(n * density)
    z_score = (observed - expected) / se if se else float("nan")
    frame = pd.DataFrame(
        [
            {
                "OBSERVED_MD": observed,
                "EXPECTED_MD": expected,
                "NN_INDEX": nn_index,
                "POINT_COUNT": n,
                "Z_SCORE": z_score,
            }
        ]
    )
    return frame


nearest_neighbor_analysis = nearest_neighbour_analysis


def overlap_analysis(
    data: Any,
    overlays: Union[Any, Sequence[Any]],
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Measure how much each input polygon is overlapped by overlay layers.

    For each overlay layer, two columns are added: absolute overlap area and
    percentage of the input feature area. Overlay layers may be passed as one
    layer or a sequence of layers.

    Args:
        data: Input polygon features
        overlays: One overlay layer or a sequence of overlay layers
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Input polygons with overlap area and percentage columns

    Example:
        overlaps = overlap_analysis(parcels, [flood, wetlands])
    """
    layer = _load(data, geometry_column)
    overlay_layers = _as_layer_list(overlays, geometry_column)
    _warn_geographic(layer.crs, "overlap areas")
    extras: List[Dict[str, Any]] = [{} for _ in layer.geometries]
    used_names: set = set(_column_names(layer))
    for overlay_index, overlay in enumerate(overlay_layers):
        aligned = _align_crs(layer, overlay, f"overlay[{overlay_index}]")
        label = _overlay_label(overlay, overlay_index)
        area_name = _unique_name(f"{label}_area", used_names)
        pct_name = _unique_name(f"{label}_pc", used_names)
        used_names.update({area_name, pct_name})
        for index, geom in enumerate(layer.geometries):
            if geom is None or geom.is_empty:
                extras[index][area_name] = 0.0
                extras[index][pct_name] = 0.0
                continue
            base_area = float(geom.area)
            hit_indexes = [
                other_index
                for _, other_index in _intersecting_index_pairs([geom], aligned.geometries)
            ]
            overlap_area = 0.0
            if base_area > 0 and hit_indexes:
                pieces = [
                    aligned.geometries[other_index]
                    for other_index in hit_indexes
                    if aligned.geometries[other_index] is not None
                    and not aligned.geometries[other_index].is_empty
                ]
                if pieces:
                    merged = unary_union(pieces)
                    if merged is not None and not merged.is_empty:
                        overlap_area = float(geom.intersection(merged).area)
            extras[index][area_name] = overlap_area
            extras[index][pct_name] = (
                0.0 if base_area <= 0 else 100.0 * overlap_area / base_area
            )
    return _output_with_extras(layer, extras)


def shortest_line_between_features(
    source: Any,
    destination: Any,
    *,
    method: Union[str, int] = "nearest_point",
    neighbors: int = 1,
    max_distance: Optional[float] = None,
    geometry_column: Optional[str] = None,
    destination_geometry_column: Optional[str] = None,
) -> Any:
    """
    Create the shortest line from each source feature to nearest destinations.

    ``method`` is ``nearest_point`` (closest points on the geometries) or
    ``centroid`` (distance between centroids).

    Args:
        source: Origin features
        destination: Destination features
        method: Distance method
        neighbors: Maximum number of nearest neighbours to keep
        max_distance: Optional search radius in layer units
        geometry_column: Geometry column for ``source``
        destination_geometry_column: Geometry column for ``destination``

    Returns:
        Line features with source attributes, destination attributes, and distance

    Example:
        links = shortest_line_between_features(points, roads, neighbors=2)
    """
    left = _load(source, geometry_column)
    right = _align_crs(
        left,
        _load(destination, destination_geometry_column),
        "destination",
    )
    kind = _choice_map(method, _SHORTEST_METHODS, "method")
    n_neighbors = _positive_int(neighbors, "neighbors")
    _warn_geographic(left.crs, "shortest-line distances")
    left_names, right_names = _suffix_names(
        _column_names(left),
        _column_names(right),
    )
    dest_centers = [
        None if geom is None or geom.is_empty else geom.centroid
        for geom in right.geometries
    ]
    rows: List[Dict[str, Any]] = []
    columns = [*left_names, *right_names, "distance"]
    for source_index, geom in enumerate(left.geometries):
        if geom is None or geom.is_empty:
            continue
        scored: List[Tuple[float, int, BaseGeometry]] = []
        origin_ref = geom.centroid if kind == "centroid" else geom
        for dest_index, other in enumerate(right.geometries):
            if other is None or other.is_empty:
                continue
            if kind == "centroid":
                dest_ref = dest_centers[dest_index]
                assert dest_ref is not None
                distance = float(origin_ref.distance(dest_ref))
                line = LineString(
                    [(origin_ref.x, origin_ref.y), (dest_ref.x, dest_ref.y)]
                )
            else:
                distance = float(geom.distance(other))
                start, end = nearest_points(geom, other)
                if start.equals(end):
                    line = LineString([(start.x, start.y), (end.x + 0.0, end.y + 0.0)])
                else:
                    line = LineString([start, end])
            if max_distance is not None and distance > float(max_distance):
                continue
            scored.append((distance, dest_index, line))
        scored.sort(key=lambda item: item[0])
        for distance, dest_index, line in scored[:n_neighbors]:
            row: Dict[str, Any] = {"geometry": line, "distance": distance}
            left_values = left.attributes.iloc[source_index]
            right_values = right.attributes.iloc[dest_index]
            for source_name, name in zip(left.attributes.columns, left_names):
                row[name] = left_values[source_name]
            for source_name, name in zip(right.attributes.columns, right_names):
                row[name] = right_values[source_name]
            rows.append(row)
    return _frame_from_rows(rows, columns, left.crs)


def stdbscan_clustering(
    data: Any,
    datetime_field: str,
    *,
    min_size: int = 5,
    eps: float = 1.0,
    eps2: float = 0.0,
    time_unit: str = "days",
    border_as_noise: bool = False,
    cluster_field: str = "CLUSTER_ID",
    size_field: str = "CLUSTER_SIZE",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Cluster point features with spatiotemporal DBSCAN (ST-DBSCAN).

    Points must be within ``eps`` spatially and within ``eps2`` temporally
    (converted to days using ``time_unit``) to share a neighbourhood.

    Args:
        data: Point features to cluster
        datetime_field: Date/time attribute
        min_size: Minimum cluster size
        eps: Maximum neighbour distance in layer units
        eps2: Maximum time separation
        time_unit: Unit for ``eps2`` (milliseconds…weeks)
        border_as_noise: When True, treat border points as noise
        cluster_field: Output cluster id column
        size_field: Output cluster size column
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Input features with cluster id and size columns

    Example:
        clusters = stdbscan_clustering(events, "timestamp", eps=100, eps2=1, time_unit="days")
    """
    unit_key = str(time_unit).strip().lower()
    if unit_key not in _TIME_UNITS:
        raise GeoprocessError(
            f"time_unit must be one of {sorted(_TIME_UNITS)}, got {time_unit!r}"
        )
    eps2_days = float(eps2) * _TIME_UNITS[unit_key]
    return _cluster_points(
        data,
        min_size=min_size,
        eps=eps,
        datetime_field=datetime_field,
        eps2=eps2_days,
        border_as_noise=border_as_noise,
        cluster_field=cluster_field,
        size_field=size_field,
        geometry_column=geometry_column,
    )


def statistics_by_categories(
    data: Any,
    categories: Union[str, Sequence[str]],
    *,
    values_field: Optional[str] = None,
    geometry_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate field statistics within category groups.

    When ``values_field`` is omitted, only the row count per category is
    returned.

    Args:
        data: Features to summarize
        categories: Category field name or sequence of names
        values_field: Optional field to summarize within each category
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Table of per-category statistics

    Example:
        by_zone = statistics_by_categories(parcels, "zone", values_field="value")
    """
    layer = _load(data, geometry_column)
    keys = _group_keys(categories, layer.attributes)
    if values_field is None:
        return (
            layer.attributes.groupby(keys, dropna=False, sort=False, observed=False)
            .size()
            .reset_index(name="COUNT")
        )
    _require_field(layer, values_field, "values_field")
    rows: List[Dict[str, Any]] = []
    grouped = layer.attributes.groupby(keys, dropna=False, sort=False, observed=False)
    for key, frame in grouped:
        key_vals = key if isinstance(key, tuple) else (key,)
        stats = _field_statistics(frame[values_field])
        row = {name: value for name, value in zip(keys, key_vals)}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def sum_line_lengths(
    polygons: Any,
    lines: Any,
    *,
    length_field: str = "LENGTH",
    count_field: str = "COUNT",
    geometry_column: Optional[str] = None,
    lines_geometry_column: Optional[str] = None,
) -> Any:
    """
    Sum the length and count of lines that cross each polygon.

    Args:
        polygons: Polygon features that receive the totals
        lines: Line features to measure
        length_field: Output length column name
        count_field: Output count column name
        geometry_column: Geometry column for ``polygons``
        lines_geometry_column: Geometry column for ``lines``

    Returns:
        Polygon features with length and count columns

    Example:
        totals = sum_line_lengths(watersheds, streams)
    """
    poly_layer = _load(polygons, geometry_column)
    line_layer = _align_crs(
        poly_layer,
        _load(lines, lines_geometry_column),
        "lines",
    )
    _warn_geographic(poly_layer.crs, "summed line lengths")
    length_name = _unique_column_name(poly_layer, length_field)
    count_name = _unique_column_name(poly_layer, count_field, reserved={length_name})
    lengths = [0.0 for _ in poly_layer.geometries]
    counts = [0 for _ in poly_layer.geometries]
    for poly_index, line_index in _intersecting_index_pairs(
        poly_layer.geometries,
        line_layer.geometries,
    ):
        poly = poly_layer.geometries[poly_index]
        line = line_layer.geometries[line_index]
        if poly is None or line is None or poly.is_empty or line.is_empty:
            continue
        piece = poly.intersection(line)
        length = _line_length(piece)
        if length <= 0 and not poly.intersects(line):
            continue
        if length > 0 or poly.intersects(line):
            lengths[poly_index] += length
            counts[poly_index] += 1
    extras = [
        {length_name: length, count_name: count}
        for length, count in zip(lengths, counts)
    ]
    return _output_with_extras(poly_layer, extras)


# ---------------------------------------------------------------------------
# Attribute manipulation
# ---------------------------------------------------------------------------


def filter_features(
    data: Any,
    query: Optional[str] = None,
    *,
    mask: Optional[Any] = None,
    columns: Optional[Sequence[str]] = None,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Select features that match attribute criteria.

    Pass a pandas ``query`` string, a boolean ``mask`` (sequence or callable
    that receives the attribute table), or both (combined with AND).

    Args:
        data: Features to filter
        query: Attribute expression evaluated with ``DataFrame.query``
        mask: Boolean sequence, or ``callable(attributes) -> boolean Series``
        columns: Optional attribute columns to keep (geometry is always kept)
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Matching features in the input CRS

    Example:
        north = filter_features(districts, "DIV_Name in ('Rangpur', 'Rajshahi')")
        low = filter_features(rice, "Boro < @mean_boro", mask=None)
    """
    layer = _load(data, geometry_column)
    selected = _boolean_mask(layer, query=query, mask=mask)
    return _subset_layer(layer, selected, columns=columns)


def select_fields(
    data: Any,
    fields: Union[str, Sequence[str]],
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Keep a subset of attribute columns. Geometry is always retained.

    Args:
        data: Features
        fields: Field name or sequence of names to keep
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with the selected attributes

    Example:
        slim = select_fields(districts, ["DIST_Name", "DIV_Name", "Aman"])
    """
    layer = _load(data, geometry_column)
    return _subset_layer(layer, np.ones(len(layer.geometries), dtype=bool), columns=fields)


def rename_fields(
    data: Any,
    mapping: Dict[str, str],
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Rename attribute columns.

    Args:
        data: Features
        mapping: ``{old_name: new_name, ...}``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with renamed attributes

    Example:
        renamed = rename_fields(districts, {"ADM1_EN": "DIV_Name", "ADM2_EN": "DIST_Name"})
    """
    layer = _load(data, geometry_column)
    if not mapping:
        return _subset_layer(layer, np.ones(len(layer.geometries), dtype=bool))
    missing = [name for name in mapping if name not in layer.attributes.columns]
    if missing:
        raise GeoprocessError(
            f"Cannot rename missing field(s) {missing}. "
            f"Available fields: {list(layer.attributes.columns)}"
        )
    renamed = layer.attributes.rename(columns=dict(mapping))
    rows: List[Dict[str, Any]] = []
    columns = [str(name) for name in renamed.columns]
    for index, geom in enumerate(layer.geometries):
        row = {name: renamed.iloc[index][name] for name in columns}
        row["geometry"] = geom
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def calculate_fields(
    data: Any,
    assignments: Dict[str, Any],
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Create or overwrite attributes from expressions or callables.

    Each value in ``assignments`` may be:

    - a pandas ``eval`` expression string that refers to attribute columns
    - a callable that receives the attribute ``DataFrame`` and returns a Series
    - a scalar copied to every feature

    Args:
        data: Features
        assignments: ``{field_name: expression_or_callable_or_scalar, ...}``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with the calculated attributes

    Example:
        totals = calculate_fields(rice, {"Total": "Aus + Aman + Boro"})
        rated = calculate_fields(rice, {"high": lambda df: df["Boro"] > df["Boro"].mean()})
    """
    layer = _load(data, geometry_column)
    if not assignments:
        raise GeoprocessError("assignments must contain at least one field")
    frame = layer.attributes.copy()
    for name, value in assignments.items():
        frame[str(name)] = _resolve_assignment(frame, value, str(name))
    rows: List[Dict[str, Any]] = []
    columns = [str(name) for name in frame.columns]
    for index, geom in enumerate(layer.geometries):
        row = {name: frame.iloc[index][name] for name in columns}
        row["geometry"] = geom
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def aggregate_attributes(
    data: Any,
    by: Any,
    aggregations: Any = "first",
    *,
    keep_geometry: bool = True,
    make_valid: bool = True,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Group features by attributes and aggregate values.

    When ``keep_geometry`` is True (default), geometries that share the group
    keys are dissolved (same behaviour as ``dissolve``). When False, a table
    of aggregated attributes is returned without geometries.

    Args:
        data: Features to summarize
        by: Grouping field name or sequence of names
        aggregations: Aggregation for remaining fields: ``first``, ``last``,
            ``sum``, ``mean``, ``min``, ``max``, ``median``, ``count``,
            ``nunique``, a callable, or a dict of field → aggregation
        keep_geometry: Dissolve geometries when True
        make_valid: Repair invalid polygons before dissolving
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        One feature (or row) per group

    Example:
        by_div = aggregate_attributes(rice, "DIV_Name", {"Boro": "sum", "Aman": "sum"})
        table = aggregate_attributes(rice, "DIV_Name", {"Boro": "sum"}, keep_geometry=False)
    """
    if keep_geometry:
        return dissolve(
            data,
            by=by,
            aggfunc=aggregations,
            make_valid=make_valid,
            geometry_column=geometry_column,
        )
    layer = _load(data, geometry_column)
    keys = _group_keys(by, layer.attributes)
    if not keys:
        raise GeoprocessError("by must name at least one grouping field when keep_geometry is False")
    value_columns = [
        name
        for name in layer.attributes.columns
        if name not in keys
    ]
    if isinstance(aggregations, dict):
        value_columns = [name for name in aggregations if name not in keys]
        missing = [name for name in value_columns if name not in layer.attributes.columns]
        if missing:
            raise GeoprocessError(
                f"Aggregation field(s) {missing} are not in the attribute table. "
                f"Available fields: {list(layer.attributes.columns)}"
            )
    if not value_columns and not isinstance(aggregations, dict):
        grouped = (
            layer.attributes.groupby(keys, dropna=False, sort=False, observed=False)
            .size()
            .reset_index(name="COUNT")
        )
        return grouped
    from geolattice.vector.geoprocessing_vector_data import _aggregate_frame

    rows: List[Dict[str, Any]] = []
    for key, frame in layer.attributes.groupby(keys, dropna=False, sort=False, observed=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(keys, key_vals)}
        subset = frame.loc[:, value_columns] if value_columns else frame.iloc[:, 0:0]
        row.update(_aggregate_frame(subset, aggregations))
        rows.append(row)
    return pd.DataFrame(rows)


def sort_features(
    data: Any,
    by: Union[str, Sequence[str]],
    *,
    ascending: Union[bool, Sequence[bool]] = True,
    na_position: str = "last",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Order features by one or more attribute fields.

    Args:
        data: Features to sort
        by: Field name or sequence of field names
        ascending: Sort direction, or one direction per field
        na_position: ``first`` or ``last`` for missing values
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features in sorted order

    Example:
        ordered = sort_features(rice, ["DIV_Name", "Boro"], ascending=[True, False])
    """
    layer = _load(data, geometry_column)
    keys = _group_keys(by, layer.attributes)
    frame = layer.attributes.copy()
    frame["_row_"] = np.arange(len(frame))
    sorted_frame = frame.sort_values(
        by=keys,
        ascending=ascending,
        na_position=na_position,
        kind="mergesort",
    )
    order = sorted_frame["_row_"].to_numpy(dtype=int)
    rows: List[Dict[str, Any]] = []
    columns = _column_names(layer)
    for index in order:
        rows.append(_attribute_row(layer, int(index), layer.geometries[int(index)]))
    return _frame_from_rows(rows, columns, layer.crs)


def join_attributes(
    data: Any,
    other: Any,
    *,
    on: Optional[Union[str, Sequence[str]]] = None,
    left_on: Optional[Union[str, Sequence[str]]] = None,
    right_on: Optional[Union[str, Sequence[str]]] = None,
    how: str = "left",
    suffixes: Tuple[str, str] = ("_1", "_2"),
    geometry_column: Optional[str] = None,
    other_geometry_column: Optional[str] = None,
) -> Any:
    """
    Join attributes from a table or layer onto features by a common key.

    Geometries come from ``data``. Non-spatial tables (DataFrame / CSV-like
    objects without a geometry column) are accepted as ``other``.

    Args:
        data: Left (spatial) features
        other: Right table or layer
        on: Shared key field name(s)
        left_on: Key field(s) on ``data`` when names differ
        right_on: Key field(s) on ``other`` when names differ
        how: ``left``, ``inner``, ``right``, or ``outer``
        suffixes: Suffixes for overlapping non-key column names
        geometry_column: Geometry column for ``data``
        other_geometry_column: Geometry column for ``other`` when it is spatial

    Returns:
        Features with joined attributes in the CRS of ``data``

    Example:
        joined = join_attributes(upazilas, poverty, on="ADM3_PCODE")
    """
    kind = _choice_map(how, _JOIN_HOW, "how")
    left = _load(data, geometry_column)
    right_frame, right_geoms = _attributes_for_join(other, other_geometry_column)
    left_keys, right_keys = _join_keys(on, left_on, right_on, left.attributes, right_frame)
    left_attrs = left.attributes.copy()
    left_attrs["__left_ix__"] = np.arange(len(left_attrs))
    right_attrs = right_frame.copy()
    right_attrs["__right_ix__"] = np.arange(len(right_attrs))
    merged = left_attrs.merge(
        right_attrs,
        left_on=left_keys,
        right_on=right_keys,
        how=kind,
        suffixes=suffixes,
    )
    rows: List[Dict[str, Any]] = []
    skip = {"__left_ix__", "__right_ix__"}
    columns = [str(name) for name in merged.columns if name not in skip]
    for _, record in merged.iterrows():
        left_ix = record["__left_ix__"]
        if pd.isna(left_ix):
            geom = None
        else:
            geom = left.geometries[int(left_ix)]
        row = {name: record[name] for name in columns}
        row["geometry"] = geom
        rows.append(row)
    return _frame_from_rows(rows, columns, left.crs)


def transform_field(
    data: Any,
    field: str,
    func: Callable[[Any], Any],
    *,
    output_field: Optional[str] = None,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Apply a function to one attribute column.

    Args:
        data: Features
        field: Source attribute name
        func: Callable applied element-wise (receives each cell value)
        output_field: Name of the result column; defaults to overwriting ``field``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with the transformed attribute

    Example:
        logs = transform_field(samples, "SOC", np.log1p, output_field="log_SOC")
    """
    layer = _load(data, geometry_column)
    series = _field_series(layer, field)
    target = output_field or field
    transformed = series.map(func)
    extras = [{str(target): value} for value in transformed.tolist()]
    if target in layer.attributes.columns and target == field:
        # overwrite: rebuild without duplicating the column via extras path
        frame = layer.attributes.copy()
        frame[field] = transformed.to_numpy()
        rows: List[Dict[str, Any]] = []
        columns = _column_names(layer)
        for index, geom in enumerate(layer.geometries):
            row = {name: frame.iloc[index][name] for name in columns}
            row["geometry"] = geom
            rows.append(row)
        return _frame_from_rows(rows, columns, layer.crs)
    if target in layer.attributes.columns:
        frame = layer.attributes.copy()
        frame[target] = transformed.to_numpy()
        rows = []
        columns = _column_names(layer)
        for index, geom in enumerate(layer.geometries):
            row = {name: frame.iloc[index][name] for name in columns}
            row["geometry"] = geom
            rows.append(row)
        return _frame_from_rows(rows, columns, layer.crs)
    return _output_with_extras(layer, extras)


def string_field(
    data: Any,
    field: str,
    operation: str,
    *,
    output_field: Optional[str] = None,
    other: Optional[Any] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    width: Optional[int] = None,
    fillchar: str = " ",
    case: bool = False,
    regex: bool = False,
    n: int = -1,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Apply a string operation to a text attribute.

    Supported ``operation`` values: ``lower``, ``upper``, ``title``,
    ``capitalize``, ``strip``, ``lstrip``, ``rstrip``, ``replace``,
    ``contains``, ``startswith``, ``endswith``, ``length``, ``slice``,
    ``cat`` / ``concat``, ``pad``, ``zfill``.

    Args:
        data: Features
        field: Source text field
        operation: String operation name
        output_field: Result column name; defaults to overwriting ``field``
            for mutating ops, or ``{field}_{operation}`` for predicates / length
        other: Replacement text, pattern, concat partner, or pad side
        start: Slice start (for ``slice``) or replace count via ``n``
        stop: Slice stop
        width: Width for ``pad`` / ``zfill``
        fillchar: Fill character for ``pad``
        case: Case-sensitive match for contains / startswith / endswith
        regex: Treat ``other`` as a regular expression for ``replace`` / ``contains``
        n: Maximum replacements for ``replace`` (``-1`` means all)
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with the string result column

    Example:
        titled = string_field(districts, "DIST_Name", "title")
        codes = string_field(districts, "DIST_Name", "slice", start=0, stop=3, output_field="code")
    """
    layer = _load(data, geometry_column)
    op = _choice_map(operation, _STRING_OPS, "operation")
    series = _field_series(layer, field).astype("string")
    text = series.fillna("")
    accessor = text.str
    if op == "lower":
        result = accessor.lower()
    elif op == "upper":
        result = accessor.upper()
    elif op == "title":
        result = accessor.title()
    elif op == "capitalize":
        result = accessor.capitalize()
    elif op == "strip":
        result = accessor.strip(other) if other is not None else accessor.strip()
    elif op == "lstrip":
        result = accessor.lstrip(other) if other is not None else accessor.lstrip()
    elif op == "rstrip":
        result = accessor.rstrip(other) if other is not None else accessor.rstrip()
    elif op == "replace":
        if other is None:
            raise GeoprocessError("string_field(..., operation='replace') requires other=(pat, repl) or pat")
        if isinstance(other, (tuple, list)) and len(other) == 2:
            pat, repl = other
        else:
            raise GeoprocessError(
                "For replace, pass other=(pattern, replacement)"
            )
        result = accessor.replace(pat, repl, regex=regex, n=n)
    elif op == "contains":
        if other is None:
            raise GeoprocessError("contains requires other=substring")
        result = accessor.contains(str(other), case=case, regex=regex)
    elif op == "startswith":
        if other is None:
            raise GeoprocessError("startswith requires other=prefix")
        result = accessor.startswith(str(other))
    elif op == "endswith":
        if other is None:
            raise GeoprocessError("endswith requires other=suffix")
        result = accessor.endswith(str(other))
    elif op == "length":
        result = accessor.len()
    elif op == "slice":
        result = accessor.slice(start=start, stop=stop)
    elif op == "cat":
        if other is None:
            raise GeoprocessError("cat/concat requires other=field_name or literal")
        left_text = text.astype(str)
        if isinstance(other, str) and other in layer.attributes.columns:
            partner = layer.attributes[other].astype("string").fillna("").astype(str)
            result = left_text + partner
        else:
            result = left_text + str(other)
    elif op == "pad":
        if width is None:
            raise GeoprocessError("pad requires width=")
        side = "left" if other in (None, "left", "l") else str(other)
        result = accessor.pad(width=int(width), side=side, fillchar=fillchar)
    elif op == "zfill":
        if width is None:
            raise GeoprocessError("zfill requires width=")
        result = accessor.zfill(int(width))
    else:  # pragma: no cover
        raise GeoprocessError(f"Unsupported string operation {operation!r}")

    default_name = (
        field
        if op not in {"contains", "startswith", "endswith", "length"}
        else f"{field}_{op}"
    )
    target = output_field or default_name
    return _assign_series(layer, target, result)


def update_where(
    data: Any,
    condition: Any,
    updates: Dict[str, Any],
    *,
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Update attributes where a condition is true.

    ``condition`` may be a pandas query string, a boolean sequence, or a
    callable that receives the attribute table. Each value in ``updates`` may
    be a scalar, an ``eval`` expression string, or a callable of the attribute
    table (evaluated for the whole table; only matching rows are written).

    Args:
        data: Features
        condition: Rows to update
        updates: ``{field: value_or_expression, ...}``
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Features with conditional updates applied

    Example:
        flagged = update_where(rice, "Boro < 1000", {"status": "low"})
        scaled = update_where(rice, "DIV_Name == 'Rangpur'", {"Boro": "Boro * 1.1"})
    """
    layer = _load(data, geometry_column)
    if not updates:
        raise GeoprocessError("updates must contain at least one field")
    selected = _boolean_mask(layer, query=condition if isinstance(condition, str) else None, mask=None if isinstance(condition, str) else condition)
    frame = layer.attributes.copy()
    for name, value in updates.items():
        resolved = _resolve_assignment(frame, value, str(name))
        if not isinstance(resolved, pd.Series):
            resolved = pd.Series([resolved] * len(frame), index=frame.index)
        frame.loc[selected, str(name)] = resolved.loc[selected].to_numpy()
    rows: List[Dict[str, Any]] = []
    columns = [str(name) for name in frame.columns]
    for index, geom in enumerate(layer.geometries):
        row = {name: frame.iloc[index][name] for name in columns}
        row["geometry"] = geom
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def reshape_attributes(
    data: Any,
    *,
    to: str = "long",
    id_vars: Optional[Union[str, Sequence[str]]] = None,
    value_vars: Optional[Union[str, Sequence[str]]] = None,
    var_name: str = "variable",
    value_name: str = "value",
    columns: Optional[str] = None,
    values: Optional[str] = None,
    index: Optional[Union[str, Sequence[str]]] = None,
    aggfunc: Any = "first",
    geometry_column: Optional[str] = None,
) -> Any:
    """
    Reshape attribute tables between wide and long forms.

    ``to="long"`` (alias ``melt``) stacks ``value_vars`` into ``var_name`` /
    ``value_name`` columns and duplicates geometries. ``to="wide"`` (alias
    ``pivot``) spreads a long table using ``index``, ``columns``, and ``values``.

    Args:
        data: Features
        to: ``long`` / ``melt`` or ``wide`` / ``pivot``
        id_vars: Identifier columns kept in long form
        value_vars: Measure columns stacked in long form
        var_name: Name of the melted variable column
        value_name: Name of the melted value column
        columns: Column that becomes new field names in wide form
        values: Column that fills wide-form cells
        index: Row identifier(s) for wide form
        aggfunc: Aggregation when wide-form cells collide
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Reshaped features (long form keeps geometries; wide form dissolves
        duplicate geometries per index key when present)

    Example:
        long = reshape_attributes(rice, to="long", id_vars=["DIST_Name"], value_vars=["Aus", "Aman", "Boro"])
        wide = reshape_attributes(long, to="wide", index="DIST_Name", columns="variable", values="value")
    """
    kind = _choice_map(to, _RESHAPE_HOW, "to")
    layer = _load(data, geometry_column)
    frame = layer.attributes.copy()
    frame["__geom_ix__"] = np.arange(len(frame))
    if kind == "long":
        id_names = _optional_fields(frame, id_vars, allow_empty=True)
        value_names = _optional_fields(frame, value_vars, allow_empty=False)
        if not value_names:
            value_names = [name for name in frame.columns if name not in {*id_names, "__geom_ix__"}]
        melted = frame.melt(
            id_vars=[*id_names, "__geom_ix__"],
            value_vars=value_names,
            var_name=var_name,
            value_name=value_name,
        )
        rows: List[Dict[str, Any]] = []
        columns_out = [name for name in melted.columns if name != "__geom_ix__"]
        for _, record in melted.iterrows():
            geom = layer.geometries[int(record["__geom_ix__"])]
            row = {name: record[name] for name in columns_out}
            row["geometry"] = geom
            rows.append(row)
        return _frame_from_rows(rows, columns_out, layer.crs)

    if columns is None or values is None:
        raise GeoprocessError("reshape_attributes(..., to='wide') requires columns= and values=")
    index_names = _optional_fields(frame, index, allow_empty=True)
    if not index_names:
        raise GeoprocessError("reshape_attributes(..., to='wide') requires index=")
    for name in [*index_names, columns, values]:
        if name not in frame.columns:
            raise GeoprocessError(
                f"Field {name!r} is not in the attribute table. "
                f"Available fields: {list(frame.columns)}"
            )
    geom_lookup: Dict[Tuple[Any, ...], Optional[BaseGeometry]] = {}
    for position, record in frame.iterrows():
        key = tuple(record[name] for name in index_names)
        if key not in geom_lookup:
            geom_lookup[key] = layer.geometries[int(record["__geom_ix__"])]
    pivoted = frame.drop(columns="__geom_ix__").pivot_table(
        index=index_names,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
    )
    if isinstance(pivoted.columns, pd.MultiIndex):
        pivoted.columns = [str(col) for col in pivoted.columns.to_flat_index()]
    else:
        pivoted.columns = [str(col) for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    rows = []
    columns_out = [str(name) for name in pivoted.columns]
    for _, record in pivoted.iterrows():
        key = tuple(record[name] for name in index_names)
        row = {name: record[name] for name in columns_out}
        row["geometry"] = geom_lookup.get(key)
        rows.append(row)
    return _frame_from_rows(rows, columns_out, layer.crs)


def drop_geometry(
    data: Any,
    *,
    geometry_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the attribute table without geometries.

    Args:
        data: Features
        geometry_column: Geometry column name when ``data`` is a DataFrame

    Returns:
        Copy of the attribute ``DataFrame``

    Example:
        table = drop_geometry(districts)
    """
    layer = _load(data, geometry_column)
    return layer.attributes.copy()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load(data: Any, geometry_column: Optional[str]):
    return _parse(data, geometry_column, allow_bbox=False)


def _boolean_mask(
    layer: Any,
    *,
    query: Optional[str],
    mask: Optional[Any],
) -> np.ndarray:
    n = len(layer.geometries)
    selected = np.ones(n, dtype=bool)
    if query is not None:
        try:
            evaluated = layer.attributes.eval(query)
        except Exception:
            try:
                matched = layer.attributes.query(query)
            except Exception as exc:
                raise GeoprocessError(f"Invalid filter query {query!r}: {exc}") from exc
            evaluated = layer.attributes.index.isin(matched.index)
        if isinstance(evaluated, pd.Series):
            values = evaluated.reindex(layer.attributes.index).fillna(False)
            selected &= np.asarray(values, dtype=bool)
        else:
            selected &= np.asarray(evaluated, dtype=bool)
        if selected.shape[0] != n:
            raise GeoprocessError(
                f"Filter query produced {selected.shape[0]} values for {n} features"
            )
    if mask is None:
        return selected
    if callable(mask):
        values = mask(layer.attributes)
        if isinstance(values, pd.Series):
            values = values.reindex(layer.attributes.index).fillna(False).to_numpy(dtype=bool)
        else:
            values = np.asarray(values, dtype=bool)
    else:
        values = np.asarray(mask, dtype=bool)
    if values.shape[0] != n:
        raise GeoprocessError(
            f"mask length ({values.shape[0]}) must match the number of features ({n})"
        )
    return selected & values


def _subset_layer(
    layer: Any,
    selected: np.ndarray,
    *,
    columns: Optional[Union[str, Sequence[str]]] = None,
) -> Any:
    if columns is None:
        keep = _column_names(layer)
    else:
        keep = _selected_fields(layer, [columns] if isinstance(columns, str) else columns)
    rows: List[Dict[str, Any]] = []
    for index, keep_row in enumerate(selected):
        if not keep_row:
            continue
        row = {name: layer.attributes.iloc[index][name] for name in keep}
        row["geometry"] = layer.geometries[index]
        rows.append(row)
    return _frame_from_rows(rows, keep, layer.crs)


def _resolve_assignment(frame: pd.DataFrame, value: Any, name: str) -> Any:
    if callable(value):
        result = value(frame)
        if isinstance(result, pd.Series):
            return result.reindex(frame.index)
        return result
    if isinstance(value, str):
        if _looks_like_expression(value, frame.columns):
            try:
                return frame.eval(value)
            except Exception as exc:
                raise GeoprocessError(
                    f"Could not evaluate expression {value!r} for field {name!r}: {exc}"
                ) from exc
        return value
    return value


def _looks_like_expression(value: str, columns: Sequence[str]) -> bool:
    text = value.strip()
    if not text:
        return False
    if text in {str(name) for name in columns}:
        return True
    markers = set("+-*/%<>=()!&|~")
    if any(char in markers for char in text):
        return True
    tokens = {token for token in text.replace(",", " ").split() if token}
    return any(token in columns for token in tokens)


def _attributes_for_join(
    other: Any,
    geometry_column: Optional[str],
) -> Tuple[pd.DataFrame, Optional[List[Optional[BaseGeometry]]]]:
    if isinstance(other, pd.DataFrame) and not hasattr(other, "geometry"):
        return other.reset_index(drop=True).copy(), None
    # GeoDataFrame / spatial inputs
    try:
        layer = _load(other, geometry_column)
    except (TypeError, GeoprocessError, ValueError):
        if isinstance(other, pd.DataFrame):
            cols = [name for name in other.columns if name != (geometry_column or "geometry")]
            return other.loc[:, cols].reset_index(drop=True).copy(), None
        raise
    return layer.attributes.copy(), layer.geometries


def _join_keys(
    on: Optional[Union[str, Sequence[str]]],
    left_on: Optional[Union[str, Sequence[str]]],
    right_on: Optional[Union[str, Sequence[str]]],
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    if on is not None:
        if left_on is not None or right_on is not None:
            raise GeoprocessError("Pass either on= or left_on=/right_on=, not both")
        keys = [on] if isinstance(on, str) else list(on)
        for name in keys:
            if name not in left.columns or name not in right.columns:
                raise GeoprocessError(
                    f"Join key {name!r} must exist on both sides. "
                    f"Left fields: {list(left.columns)}; right fields: {list(right.columns)}"
                )
        return keys, keys
    if left_on is None or right_on is None:
        raise GeoprocessError("join_attributes requires on= or both left_on= and right_on=")
    left_keys = [left_on] if isinstance(left_on, str) else list(left_on)
    right_keys = [right_on] if isinstance(right_on, str) else list(right_on)
    if len(left_keys) != len(right_keys):
        raise GeoprocessError("left_on and right_on must have the same number of fields")
    for name in left_keys:
        if name not in left.columns:
            raise GeoprocessError(f"left_on field {name!r} is not in the left attribute table")
    for name in right_keys:
        if name not in right.columns:
            raise GeoprocessError(f"right_on field {name!r} is not in the right attribute table")
    return left_keys, right_keys


def _assign_series(layer: Any, name: str, values: pd.Series) -> Any:
    frame = layer.attributes.copy()
    frame[str(name)] = values.to_numpy()
    rows: List[Dict[str, Any]] = []
    columns = [str(col) for col in frame.columns]
    for index, geom in enumerate(layer.geometries):
        row = {col: frame.iloc[index][col] for col in columns}
        row["geometry"] = geom
        rows.append(row)
    return _frame_from_rows(rows, columns, layer.crs)


def _optional_fields(
    frame: pd.DataFrame,
    fields: Optional[Union[str, Sequence[str]]],
    *,
    allow_empty: bool,
) -> List[str]:
    if fields is None:
        if allow_empty:
            return []
        return []
    names = [fields] if isinstance(fields, str) else [str(name) for name in fields]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise GeoprocessError(
            f"Field(s) {missing} are not in the attribute table. "
            f"Available fields: {list(frame.columns)}"
        )
    return names


def _column_names(layer: Any) -> List[str]:
    return [str(name) for name in layer.attributes.columns]


def _require_field(layer: Any, field: str, label: str) -> None:
    if field not in layer.attributes.columns:
        raise GeoprocessError(
            f"{label} {field!r} is not in the attribute table. "
            f"Available fields: {list(layer.attributes.columns)}"
        )


def _field_series(layer: Any, field: str) -> pd.Series:
    _require_field(layer, field, "field")
    return layer.attributes[field]


def _unique_column_name(
    layer: Any,
    name: str,
    reserved: Optional[set] = None,
) -> str:
    used = set(_column_names(layer))
    if reserved:
        used.update(reserved)
    return _unique_name(name, used)


def _unique_name(name: str, used: set) -> str:
    candidate = str(name)
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    result = f"{candidate}_{index}"
    used.add(result)
    return result


def _output_with_extras(layer: Any, extras: Sequence[Dict[str, Any]]) -> Any:
    if extras and len(extras) != len(layer.geometries):
        raise GeoprocessError("Internal error while attaching analysis attributes")
    extra_names: List[str] = []
    for item in extras:
        for name in item:
            if name not in extra_names:
                extra_names.append(str(name))
    rows: List[Dict[str, Any]] = []
    for index, geom in enumerate(layer.geometries):
        row = _attribute_row(layer, index, geom)
        if extras:
            row.update(extras[index])
        rows.append(row)
    return _frame_from_rows(rows, [*_column_names(layer), *extra_names], layer.crs)


def _choice_map(value: Any, mapping: Dict[Any, str], label: str) -> str:
    if isinstance(value, str):
        key: Any = value.strip().lower()
    else:
        key = value
    if key not in mapping:
        choices = sorted({str(item) for item in mapping.values()})
        raise GeoprocessError(f"{label} must be one of {choices}, got {value!r}")
    return mapping[key]


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{label} must be an integer")
    number = int(value)
    if number < 1:
        raise GeoprocessError(f"{label} must be 1 or greater")
    return number


def _warn_geographic(crs: Any, what: str) -> None:
    if not _is_geographic(crs):
        return
    warnings.warn(
        f"The layer CRS is geographic, so {what} are in degrees. "
        "Project the layer to a metre or foot CRS when the measurement is a ground distance.",
        UserWarning,
        stacklevel=3,
    )


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


def _feature_centers(
    geometries: Sequence[Optional[BaseGeometry]],
) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        point = geom.centroid
        centers.append((float(point.x), float(point.y)))
    return centers


def _surface_point(geom: Optional[BaseGeometry]) -> Optional[Point]:
    if geom is None or geom.is_empty:
        return None
    try:
        point = geom.representative_point()
    except Exception:
        point = geom.centroid
    return Point(float(point.x), float(point.y))


def _id_values(layer: Any, field: Optional[str], default_prefix: str) -> List[Any]:
    if field is None:
        return [f"{default_prefix}_{index}" for index in range(len(layer.geometries))]
    _require_field(layer, field, "id field")
    return layer.attributes[field].tolist()


def _pairwise_distances(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
) -> np.ndarray:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    delta = left_arr[:, None, :] - right_arr[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _normalize_key(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _selected_fields(layer: Any, fields: Optional[Sequence[str]]) -> List[str]:
    if fields is None:
        return _column_names(layer)
    names = [str(name) for name in fields]
    missing = [name for name in names if name not in layer.attributes.columns]
    if missing:
        raise GeoprocessError(
            f"Field(s) {missing} are not in the attribute table. "
            f"Available fields: {list(layer.attributes.columns)}"
        )
    return names


def _as_layer_list(overlays: Union[Any, Sequence[Any]], geometry_column: Optional[str]):
    if isinstance(overlays, (str, bytes, dict)) or not isinstance(overlays, Sequence):
        return [_load(overlays, geometry_column)]
    # A GeoDataFrame/Series is a Sequence; treat objects with geometry as one layer.
    if hasattr(overlays, "geometry") or hasattr(overlays, "geom_type"):
        return [_load(overlays, geometry_column)]
    if not overlays:
        raise GeoprocessError("overlays must contain at least one layer")
    # Sequence of geometries vs sequence of layers: if first item looks like a layer, load each
    first = overlays[0]
    if hasattr(first, "geometry") or hasattr(first, "to_dataframe") or hasattr(first, "features"):
        return [_load(item, geometry_column) for item in overlays]
    return [_load(overlays, geometry_column)]


def _overlay_label(layer: Any, index: int) -> str:
    name = layer.attributes.attrs.get("layer_name") if hasattr(layer.attributes, "attrs") else None
    if name:
        return str(name)
    return f"layer_{index + 1}"


def _line_length(geom: Optional[BaseGeometry]) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type in _LINE:
        return float(geom.length)
    if geom.geom_type == "GeometryCollection":
        return sum(_line_length(part) for part in geom.geoms)
    return 0.0


def _climb_stats(
    geom: Optional[BaseGeometry],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if geom is None or geom.is_empty:
        return None, None, None, None
    zs = _line_z_values(geom)
    if not zs:
        raise GeoprocessError(
            "climb_along_line requires line geometries with Z values on vertices"
        )
    climb = 0.0
    descent = 0.0
    for first, second in zip(zs, zs[1:]):
        delta = second - first
        if delta > 0:
            climb += delta
        elif delta < 0:
            descent += -delta
    return climb, descent, min(zs), max(zs)


def _line_z_values(geom: BaseGeometry) -> List[float]:
    values: List[float] = []
    if geom.geom_type in {"LineString", "LinearRing"}:
        for coord in geom.coords:
            if len(coord) < 3 or coord[2] is None:
                return []
            values.append(float(coord[2]))
        return values
    if geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            part_values = _line_z_values(part)
            if not part_values:
                return []
            values.extend(part_values)
        return values
    if geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            if part.geom_type in _LINE:
                part_values = _line_z_values(part)
                if not part_values:
                    return []
                values.extend(part_values)
        return values
    return []


def _field_statistics(series: pd.Series) -> Dict[str, Any]:
    count = int(series.shape[0])
    null_mask = series.isna()
    empty = int(null_mask.sum())
    filled_series = series.loc[~null_mask]
    filled = int(filled_series.shape[0])
    unique = int(filled_series.nunique(dropna=True))
    stats: Dict[str, Any] = {
        "COUNT": count,
        "UNIQUE": unique,
        "EMPTY": empty,
        "FILLED": filled,
    }
    if filled == 0:
        return stats
    numeric = pd.to_numeric(filled_series, errors="coerce")
    is_numeric = bool(numeric.notna().all()) and not (
        pd.api.types.is_bool_dtype(filled_series)
        or (
            filled_series.dtype == object
            and filled_series.map(lambda value: isinstance(value, bool)).any()
        )
    )
    # Prefer string stats when original dtype is clearly string-like
    if pd.api.types.is_string_dtype(filled_series) or (
        filled_series.dtype == object
        and filled_series.map(lambda value: isinstance(value, str)).all()
    ):
        is_numeric = False
    if is_numeric:
        values = numeric.astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        counts = values.value_counts(dropna=True)
        stats.update(
            {
                "MIN": float(values.min()),
                "MAX": float(values.max()),
                "RANGE": float(values.max() - values.min()),
                "SUM": float(values.sum()),
                "MEAN": mean,
                "MEDIAN": float(values.median()),
                "STD_DEV": std,
                "CV": float("nan") if mean == 0 else std / abs(mean),
                "MINORITY": _mode_value(counts, rarest=True),
                "MAJORITY": _mode_value(counts, rarest=False),
                "FIRSTQUARTILE": q1,
                "THIRDQUARTILE": q3,
                "IQR": q3 - q1,
            }
        )
        return stats
    as_text = filled_series.map(lambda value: "" if _is_missing(value) else str(value))
    lengths = as_text.map(len)
    counts = as_text.value_counts(dropna=True)
    stats.update(
        {
            "MIN": as_text.min(),
            "MAX": as_text.max(),
            "MIN_LENGTH": int(lengths.min()),
            "MAX_LENGTH": int(lengths.max()),
            "MEAN_LENGTH": float(lengths.mean()),
            "MINORITY": _mode_value(counts, rarest=True),
            "MAJORITY": _mode_value(counts, rarest=False),
        }
    )
    return stats


def _mode_value(counts: pd.Series, *, rarest: bool) -> Any:
    if counts.empty:
        return pd.NA
    target = counts.min() if rarest else counts.max()
    candidates = counts[counts == target]
    return candidates.index[0]


def _distance_to_nearest_hub(
    data: Any,
    hubs: Any,
    *,
    hub_field: str,
    unit: Union[str, int],
    as_lines: bool,
    geometry_column: Optional[str],
    hubs_geometry_column: Optional[str],
) -> Any:
    source = _load(data, geometry_column)
    hub_layer = _align_crs(
        source,
        _load(hubs, hubs_geometry_column),
        "hubs",
    )
    _require_field(hub_layer, hub_field, "hub_field")
    unit_name = _choice_map(unit, _HUB_UNITS, "unit")
    source_points = [_surface_point(geom) for geom in source.geometries]
    hub_points = [_surface_point(geom) for geom in hub_layer.geometries]
    valid_hubs = [(index, point) for index, point in enumerate(hub_points) if point is not None]
    if not valid_hubs:
        raise GeoprocessError("The hubs layer has no usable geometries")
    hub_coords = np.asarray([(point.x, point.y) for _, point in valid_hubs], dtype=float)
    if cKDTree is None:
        raise ImportError("distance to nearest hub requires scipy")
    tree = cKDTree(hub_coords)
    rows: List[Dict[str, Any]] = []
    columns = [*_column_names(source), "HubName", "HubDist"]
    for index, origin in enumerate(source_points):
        if origin is None:
            continue
        distance_layer, hub_pos = tree.query((origin.x, origin.y), k=1)
        hub_index = valid_hubs[int(hub_pos)][0]
        hub_point = hub_points[hub_index]
        assert hub_point is not None
        distance = _convert_distance(
            float(distance_layer),
            origin,
            hub_point,
            source.crs,
            unit_name,
        )
        geometry: BaseGeometry
        if as_lines:
            geometry = LineString(
                [(origin.x, origin.y), (hub_point.x, hub_point.y)]
            )
        else:
            geometry = Point(origin.x, origin.y)
        row = _attribute_row(source, index, geometry)
        row["HubName"] = hub_layer.attributes.iloc[hub_index][hub_field]
        row["HubDist"] = distance
        rows.append(row)
    return _frame_from_rows(rows, columns, source.crs)


def _convert_distance(
    layer_distance: float,
    start: Point,
    end: Point,
    crs: Any,
    unit: str,
) -> float:
    if unit == "layer":
        return layer_distance
    meters: float
    if _is_geographic(crs):
        geod = pyproj.Geod(ellps="WGS84")
        proj = _to_pyproj(crs)
        if proj is not None and not proj.equals(pyproj.CRS.from_epsg(4326)):
            transformer = pyproj.Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
            lon1, lat1 = transformer.transform(start.x, start.y)
            lon2, lat2 = transformer.transform(end.x, end.y)
        else:
            lon1, lat1, lon2, lat2 = start.x, start.y, end.x, end.y
        meters = float(geod.inv(lon1, lat1, lon2, lat2)[2])
    else:
        # Assume projected CRS stores metres unless unit name suggests otherwise.
        meters = layer_distance
        proj = _to_pyproj(crs)
        if proj is not None:
            try:
                factor = proj.axis_info[0].unit_conversion_factor
                if factor:
                    meters = layer_distance * float(factor)
            except Exception:
                meters = layer_distance
    return meters / _METERS_PER[unit]


def _connect_points(
    start: Point,
    end: Point,
    crs: Any,
    *,
    geodesic: bool,
    geodesic_distance_km: float,
    antimeridian_split: bool,
) -> Optional[BaseGeometry]:
    if start.equals(end):
        return None
    if not geodesic:
        return LineString([(start.x, start.y), (end.x, end.y)])
    geod = pyproj.Geod(ellps="WGS84")
    proj = _to_pyproj(crs)
    to_wgs = None
    from_wgs = None
    if proj is not None and not proj.is_geographic:
        to_wgs = pyproj.Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        from_wgs = pyproj.Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        lon1, lat1 = to_wgs.transform(start.x, start.y)
        lon2, lat2 = to_wgs.transform(end.x, end.y)
    else:
        lon1, lat1, lon2, lat2 = start.x, start.y, end.x, end.y
    spacing_m = max(float(geodesic_distance_km) * 1000.0, 1.0)
    points = geod.npts(lon1, lat1, lon2, lat2, max(1, int(geod.inv(lon1, lat1, lon2, lat2)[2] // spacing_m)))
    coords = [(lon1, lat1), *points, (lon2, lat2)]
    if from_wgs is not None:
        coords = [from_wgs.transform(x, y) for x, y in coords]
    line: BaseGeometry = LineString(coords)
    if antimeridian_split and (proj is None or proj.is_geographic):
        line = _split_antimeridian(LineString([(lon1, lat1), *points, (lon2, lat2)]))
        if from_wgs is not None and isinstance(line, (LineString, MultiLineString)):
            if line.geom_type == "LineString":
                line = LineString([from_wgs.transform(x, y) for x, y in line.coords])
            else:
                line = MultiLineString(
                    [
                        LineString([from_wgs.transform(x, y) for x, y in part.coords])
                        for part in line.geoms
                    ]
                )
    return line


def _split_antimeridian(line: LineString) -> BaseGeometry:
    coords = list(line.coords)
    parts: List[List[Tuple[float, float]]] = []
    current = [coords[0]]
    for previous, current_coord in zip(coords, coords[1:]):
        lon1, lat1 = previous[0], previous[1]
        lon2, lat2 = current_coord[0], current_coord[1]
        if abs(lon2 - lon1) > 180:
            # split at antimeridian
            parts.append(current)
            current = [(lon2, lat2)]
        else:
            current.append((lon2, lat2))
    parts.append(current)
    lines = [LineString(part) for part in parts if len(part) >= 2]
    if not lines:
        return line
    if len(lines) == 1:
        return lines[0]
    return MultiLineString(lines)


def _cluster_points(
    data: Any,
    *,
    min_size: int,
    eps: float,
    datetime_field: Optional[str],
    eps2: Optional[float],
    border_as_noise: bool,
    cluster_field: str,
    size_field: str,
    geometry_column: Optional[str],
) -> Any:
    if cKDTree is None:
        raise ImportError("DBSCAN clustering requires scipy")
    layer = _load(data, geometry_column)
    min_samples = _positive_int(min_size, "min_size")
    if float(eps) <= 0:
        raise GeoprocessError("eps must be greater than zero")
    _warn_geographic(layer.crs, "clustering distances")
    points: List[Tuple[float, float]] = []
    keep_indices: List[int] = []
    times: List[Optional[float]] = []
    if datetime_field is not None:
        _require_field(layer, datetime_field, "datetime_field")
    for index, geom in enumerate(layer.geometries):
        if geom is None or geom.is_empty:
            continue
        center = geom.centroid
        points.append((float(center.x), float(center.y)))
        keep_indices.append(index)
        if datetime_field is None:
            times.append(None)
        else:
            times.append(_as_day_number(layer.attributes.iloc[index][datetime_field]))
    labels = np.full(len(layer.geometries), -1, dtype=int)
    if points:
        coords = np.asarray(points, dtype=float)
        tree = cKDTree(coords)
        neighbors = tree.query_ball_tree(tree, r=float(eps))
        if datetime_field is not None:
            assert eps2 is not None
            filtered: List[List[int]] = []
            for i, idxs in enumerate(neighbors):
                t0 = times[i]
                kept = []
                for j in idxs:
                    t1 = times[j]
                    if t0 is None or t1 is None or abs(t0 - t1) <= float(eps2):
                        kept.append(j)
                filtered.append(kept)
            neighbors = filtered
        clustered = _dbscan_labels(neighbors, min_samples, border_as_noise)
        for local_index, label in enumerate(clustered):
            labels[keep_indices[local_index]] = int(label)
    sizes = Counter(int(label) for label in labels if int(label) >= 0)
    cluster_name = _unique_column_name(layer, cluster_field)
    size_name = _unique_column_name(layer, size_field, reserved={cluster_name})
    extras = []
    for label in labels:
        if int(label) < 0:
            extras.append({cluster_name: pd.NA, size_name: 0})
        else:
            extras.append(
                {
                    cluster_name: int(label) + 1,
                    size_name: int(sizes[int(label)]),
                }
            )
    result = _output_with_extras(layer, extras)
    result.attrs["num_clusters"] = int(len(sizes))
    return result


def _as_day_number(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise GeoprocessError(f"datetime value {value!r} could not be parsed")
    # Unix days
    return float(stamp.value) / 86_400_000_000_000.0


def _dbscan_labels(
    neighbors: Sequence[Sequence[int]],
    min_samples: int,
    border_as_noise: bool,
) -> List[int]:
    n = len(neighbors)
    labels = [-1] * n
    cluster_id = 0
    for index in range(n):
        if labels[index] != -1:
            continue
        seeds = list(neighbors[index])
        if len(seeds) < min_samples:
            continue
        labels[index] = cluster_id
        queue = deque(seeds)
        seen = set(seeds)
        while queue:
            point = queue.popleft()
            point_neighbors = neighbors[point]
            is_core = len(point_neighbors) >= min_samples
            if labels[point] == -1:
                if border_as_noise and not is_core:
                    continue
                labels[point] = cluster_id
            if not is_core:
                continue
            for other in point_neighbors:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
                if labels[other] == -1 and not (border_as_noise and len(neighbors[other]) < min_samples):
                    labels[other] = cluster_id
        cluster_id += 1
    return labels


def _kmeans_farthest(coords: np.ndarray, k: int) -> np.ndarray:
    """Seed with farthest points, then assign / update means until stable."""
    n = coords.shape[0]
    centers_idx = [0]
    # Start from the point farthest from the mean for a better first seed
    mean = coords.mean(axis=0)
    centers_idx = [int(np.argmax(np.sum((coords - mean) ** 2, axis=1)))]
    while len(centers_idx) < k:
        dist = np.min(
            np.sqrt(((coords[:, None, :] - coords[centers_idx][None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        centers_idx.append(int(np.argmax(dist)))
    centers = coords[centers_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        delta = coords[:, None, :] - centers[None, :, :]
        distances = np.sum(delta * delta, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = coords[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


__all__ = [
    "basic_statistics_for_fields",
    "climb_along_line",
    "count_points_in_polygon",
    "dbscan_clustering",
    "distance_matrix",
    "distance_to_nearest_hub_lines",
    "distance_to_nearest_hub_points",
    "hub_lines",
    "join_by_lines",
    "kmeans_clustering",
    "list_unique_values",
    "mean_coordinates",
    "nearest_neighbor_analysis",
    "nearest_neighbour_analysis",
    "overlap_analysis",
    "shortest_line_between_features",
    "statistics_by_categories",
    "stdbscan_clustering",
    "sum_line_lengths",
]
