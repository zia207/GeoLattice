"""
Vector CRS helpers: define, validate, and reproject geometries.

Shared CRS definitions live in :mod:`geolattice.crs`. This module is the
vector-specific side: parse a CRS for a layer, check equivalence, choose a
datum transformation, and reproject Shapely geometries (or a whole layer /
GeoDataFrame).

Geoprocessing tools call these helpers when two layers have different CRS
definitions. Raster warping lives in :mod:`geolattice.raster.raster_CRS`.

Example:
    from geolattice.vector.vector_CRS import reproject_vector, define_crs

    projected = reproject_vector(parcels, 32618)
    crs = define_crs(4326)
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as _transform_geom

from geolattice.crs import (
    CRS,
    CRSSettings,
    DatumTransformation,
    DatumTransformationDialog,
    DatumTransformationRequired,
    crs_settings,
)

CRSLike = Union[CRS, str, int, pyproj.CRS, None]


class VectorCRSError(ValueError):
    """Raised when a vector CRS cannot be defined, validated, or applied."""


def define_crs(value: CRSLike) -> CRS:
    """
    Build a :class:`~geolattice.crs.CRS` from an EPSG code, WKT, PROJ string,
    authority string, pyproj CRS, or existing :class:`CRS`.
    """
    if value is None:
        raise VectorCRSError("A CRS definition is required.")
    if isinstance(value, CRS):
        return value
    if isinstance(value, pyproj.CRS):
        try:
            auth = value.to_authority()
        except Exception:
            auth = None
        if auth:
            return CRS.from_authority(str(auth[0]), auth[1])
        wkt = value.to_wkt()
        if wkt:
            return CRS.from_wkt(wkt)
        return CRS(str(value))
    try:
        return CRS(value) if isinstance(value, (str, int)) else CRS(str(value))
    except Exception as exc:
        raise VectorCRSError(f"Could not define CRS from {value!r}: {exc}") from exc


def validate_crs(value: CRSLike) -> CRS:
    """Return a CRS after checking that pyproj can bind the definition."""
    crs = define_crs(value)
    if to_pyproj(crs) is None:
        raise VectorCRSError(
            f"CRS {crs!r} could not be loaded with pyproj. "
            "Check the EPSG code, WKT, or PROJ string."
        )
    return crs


def to_pyproj(crs: CRSLike) -> Optional[pyproj.CRS]:
    """Convert a CRS-like value to a :class:`pyproj.CRS`, or ``None``."""
    if crs is None:
        return None
    bound = getattr(crs, "_proj", None)
    if bound is not None:
        return bound
    if isinstance(crs, pyproj.CRS):
        return crs
    try:
        if isinstance(crs, CRS):
            return pyproj.CRS.from_user_input(crs.wkt or crs.identifier or str(crs))
        return pyproj.CRS.from_user_input(crs)
    except Exception:
        return None


def crs_equal(left: CRSLike, right: CRSLike) -> bool:
    """True when two CRS definitions describe the same coordinate system."""
    left_proj = to_pyproj(left)
    right_proj = to_pyproj(right)
    if left_proj is not None and right_proj is not None:
        try:
            return bool(left_proj.equals(right_proj))
        except Exception:
            return False
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, CRS) and isinstance(right, CRS):
        return left.equivalent(right)
    return str(left) == str(right)


def is_geographic(crs: CRSLike) -> bool:
    """True when the CRS is geographic (longitude / latitude)."""
    proj = to_pyproj(crs)
    if proj is None:
        return False
    try:
        return bool(proj.is_geographic)
    except Exception:
        return False


def crs_for_output(crs: CRSLike) -> Any:
    """CRS object suitable for attaching to a GeoDataFrame / DataFrame attrs."""
    proj = to_pyproj(crs)
    if proj is not None:
        return proj
    return None if crs is None else str(crs)


def choose_datum_transformation(
    source: CRSLike,
    target: CRSLike,
    settings: Optional[CRSSettings] = None,
    *,
    operation: Optional[str] = None,
) -> Optional[DatumTransformation]:
    """
    Resolve the datum transformation between two CRS definitions.

    When several paths exist and settings say to ask, raises
    :class:`~geolattice.crs.DatumTransformationRequired`. Pass ``operation``
    to pick a path by name without raising.
    """
    source_crs = validate_crs(source)
    target_crs = validate_crs(target)
    if source_crs.equivalent(target_crs):
        return None
    dialog = DatumTransformationDialog(
        source_crs,
        target_crs,
        settings if settings is not None else crs_settings,
    )
    if operation is not None:
        return dialog.select(operation)
    return dialog.resolve()


def reproject_geometry(
    geometry: Optional[BaseGeometry],
    source_crs: CRSLike,
    target_crs: CRSLike,
    *,
    always_xy: bool = True,
    coordinate_operation: Optional[str] = None,
) -> Optional[BaseGeometry]:
    """
    Reproject one Shapely geometry from ``source_crs`` to ``target_crs``.

    ``coordinate_operation`` is an optional PROJ pipeline / WKT operation
    string (the same idea as GDAL's ``-ct`` / ``coordinateOperation``).
    """
    if geometry is None or geometry.is_empty:
        return geometry
    source = to_pyproj(source_crs)
    target = to_pyproj(target_crs)
    if source is None or target is None:
        raise VectorCRSError("Cannot reproject because a CRS definition was not recognized.")
    if source.equals(target) and not coordinate_operation:
        return geometry
    if coordinate_operation:
        transformer = pyproj.Transformer.from_pipeline(coordinate_operation)
    else:
        transformer = pyproj.Transformer.from_crs(source, target, always_xy=always_xy)
    return _transform_geom(transformer.transform, geometry)


def reproject_geometries(
    geometries: Sequence[Optional[BaseGeometry]],
    source_crs: CRSLike,
    target_crs: CRSLike,
    *,
    always_xy: bool = True,
    coordinate_operation: Optional[str] = None,
) -> List[Optional[BaseGeometry]]:
    """Reproject a sequence of Shapely geometries."""
    source = to_pyproj(source_crs)
    target = to_pyproj(target_crs)
    if source is None or target is None:
        raise VectorCRSError("Cannot reproject because a CRS definition was not recognized.")
    if source.equals(target) and not coordinate_operation:
        return list(geometries)
    if coordinate_operation:
        transformer = pyproj.Transformer.from_pipeline(coordinate_operation)
    else:
        transformer = pyproj.Transformer.from_crs(source, target, always_xy=always_xy)
    out: List[Optional[BaseGeometry]] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            out.append(geom)
        else:
            out.append(_transform_geom(transformer.transform, geom))
    return out


def reproject_vector(
    data: Any,
    target_crs: CRSLike,
    *,
    source_crs: CRSLike = None,
    always_xy: bool = True,
    coordinate_operation: Optional[str] = None,
    datum_operation: Optional[str] = None,
) -> Any:
    """
    Reproject vector features to ``target_crs``.

    ``data`` may be a GeoDataFrame / GeoSeries, a DataFrame with a geometry
    column, a Shapely geometry, a sequence of geometries, a mapping of
    ``{"geometry": ..., ...}`` features, or an open ``VectorDataset`` /
    ``VectorLayer`` / cloud-native layer (converted through geoprocessing
    parsers when needed).
    """
    target = validate_crs(target_crs)
    if datum_operation and coordinate_operation is None:
        source_for_dialog = source_crs
        if source_for_dialog is None and hasattr(data, "crs"):
            source_for_dialog = data.crs
        if source_for_dialog is not None:
            chosen = choose_datum_transformation(
                source_for_dialog, target, operation=datum_operation
            )
            if chosen is not None:
                coordinate_operation = chosen.definition or None

    # GeoPandas fast path
    if hasattr(data, "to_crs") and hasattr(data, "geometry"):
        src = source_crs if source_crs is not None else getattr(data, "crs", None)
        if src is None:
            raise VectorCRSError(
                "GeoDataFrame has no CRS. Pass source_crs= or set data.crs first."
            )
        if coordinate_operation:
            # GeoPandas to_crs does not take an arbitrary pipeline; fall through.
            pass
        else:
            try:
                return data.to_crs(to_pyproj(target) or target.identifier)
            except Exception:
                pass

    # Shapely geometry
    if isinstance(data, BaseGeometry):
        if source_crs is None:
            raise VectorCRSError("source_crs is required when reprojecting a geometry.")
        return reproject_geometry(
            data,
            source_crs,
            target,
            always_xy=always_xy,
            coordinate_operation=coordinate_operation,
        )

    # Sequence of geometries
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], BaseGeometry):
        if source_crs is None:
            raise VectorCRSError("source_crs is required when reprojecting geometries.")
        return reproject_geometries(
            data,
            source_crs,
            target,
            always_xy=always_xy,
            coordinate_operation=coordinate_operation,
        )

    # Geoprocessing _Layer or reader types via local parser
    layer = _as_layer(data, source_crs)
    return _reproject_layer(
        layer,
        target,
        always_xy=always_xy,
        coordinate_operation=coordinate_operation,
    )


def align_vector_crs(
    data: Any,
    other: Any,
    other_name: str = "other",
) -> Any:
    """
    Reproject ``other`` to the CRS of ``data`` when both have CRS and they differ.

    Used by geoprocessing and vector analysis. Returns ``other`` unchanged when
    either CRS is missing or the definitions already match.
    """
    data_layer = _as_layer(data, None)
    other_layer = _as_layer(other, None)
    if data_layer.crs is None and other_layer.crs is None:
        return other_layer if _is_internal_layer(other) else other
    if data_layer.crs is not None and other_layer.crs is None:
        return other_layer if _is_internal_layer(other) else other
    if data_layer.crs is None and other_layer.crs is not None:
        warnings.warn(
            f"The input layer has no CRS, so the {other_name} CRS was not applied. "
            "Coordinates are compared as stored.",
            UserWarning,
            stacklevel=3,
        )
        return other_layer if _is_internal_layer(other) else other
    if crs_equal(data_layer.crs, other_layer.crs):
        return other_layer if _is_internal_layer(other) else other
    warnings.warn(
        f"Reprojecting the {other_name} layer to the input CRS.",
        UserWarning,
        stacklevel=3,
    )
    reprojected = _reproject_layer(other_layer, data_layer.crs)
    return reprojected


# ---------------------------------------------------------------------------
# Layer adapter (avoids circular imports with geoprocessing)
# ---------------------------------------------------------------------------


class _CRSLayer:
    """Minimal geometry + attributes + CRS container for CRS operations."""

    def __init__(
        self,
        geometries: Sequence[Optional[BaseGeometry]],
        attributes: Any,
        crs: CRSLike,
    ) -> None:
        self.geometries = list(geometries)
        self.attributes = attributes
        self.crs = crs


def _is_internal_layer(value: Any) -> bool:
    return type(value).__name__ == "_Layer" or isinstance(value, _CRSLayer)


def _as_layer(data: Any, source_crs: CRSLike) -> Any:
    if _is_internal_layer(data):
        if source_crs is not None and getattr(data, "crs", None) is None:
            data.crs = source_crs
        return data
    # Lazy import keeps geoprocessing optional for pure geometry reprojection.
    from geolattice.vector.geoprocessing_vector_data import _parse

    layer = _parse(data, None, False)
    if source_crs is not None and layer.crs is None:
        layer.crs = source_crs
    return layer


def _reproject_layer(
    layer: Any,
    target_crs: CRSLike,
    *,
    always_xy: bool = True,
    coordinate_operation: Optional[str] = None,
) -> Any:
    target = validate_crs(target_crs)
    source = to_pyproj(layer.crs)
    target_proj = to_pyproj(target)
    if source is None or target_proj is None:
        raise VectorCRSError("Cannot reproject because a CRS definition was not recognized.")
    geometries = reproject_geometries(
        layer.geometries,
        layer.crs,
        target,
        always_xy=always_xy,
        coordinate_operation=coordinate_operation,
    )
    attributes = layer.attributes.copy() if hasattr(layer.attributes, "copy") else layer.attributes
    # Prefer reconstructing the original _Layer type when available.
    layer_type = type(layer)
    if layer_type.__name__ == "_Layer":
        return layer_type(geometries, attributes, target)
    return _CRSLayer(geometries, attributes, target)


# Compatibility aliases used by geoprocessing / analysis historically.
_to_pyproj = to_pyproj
_crs_equal = crs_equal
_is_geographic = is_geographic
_crs_for_output = crs_for_output


def _align_crs(data: Any, other: Any, other_name: str) -> Any:
    """Alias of :func:`align_vector_crs` that always returns a layer object."""
    return align_vector_crs(data, other, other_name)


def _reproject(layer: Any, target_crs: Any) -> Any:
    """Alias of layer reprojection for geoprocessing."""
    return _reproject_layer(layer, target_crs)
