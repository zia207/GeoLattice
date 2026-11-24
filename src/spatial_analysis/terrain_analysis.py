"""
Terrain analysis and geomorphometry from digital elevation models (DEMs).

File-based helpers align with ``gdaldem``
(https://gdal.org/en/stable/programs/gdaldem.html) for first-order products
and roughness indices, and with Whitebox terrain / geomorphometry concepts
(https://www.whiteboxgeo.com/manuals/api/python/terrain-analysis.html)
for curvature, wetness, landform classification, multiscale metrics, and
viewsheds.

GDAL ``DEMProcessing`` is used when the product maps to a ``gdaldem`` mode
(slope, aspect, hillshade, TRI, TPI, roughness). Curvature, TWI, landform
classification, multiscale analysis, and viewshed are computed with NumPy
(Horn / Zevenbergen–Thorne style finite differences). Contours reuse the
existing ``contour`` helper (``gdal.ContourGenerateEx``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.raster.basic_raster_operation import (
    RasterOperationError,
    _creation_options,
    _format_from_path,
    _gdal,
    _prepare_destination,
    contour,
)
from geolattice.raster.raster_read_write import open_raster_data, write_raster_data

PathLike = Union[str, Path]

# Output nodata used by gdaldem float products
_GDALDEM_NODATA = -9999.0

_CURVATURE_TYPES = {
    "mean",
    "tangential",
    "profile",
    "plan",
    "gaussian",
}

_LANDFORM_CLASSES = {
    1: "valley",
    2: "lower_slope",
    3: "gentle_slope",
    4: "steep_slope",
    5: "upper_slope",
    6: "ridge",
    7: "flat",
}


class TerrainAnalysisError(RasterOperationError):
    """Raised when a terrain-analysis operation cannot be completed."""


# ---------------------------------------------------------------------------
# Public gdaldem-backed derivatives
# ---------------------------------------------------------------------------


def slope(
    source: PathLike,
    destination: PathLike,
    *,
    units: str = "degrees",
    band: int = 1,
    algorithm: str = "Horn",
    scale: Optional[float] = None,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Slope from a DEM (``gdaldem slope``).

    ``units`` is ``"degrees"`` (default) or ``"percent"`` (percent rise).
    ``algorithm`` is ``"Horn"`` or ``"ZevenbergenThorne"``.
    """
    units_key = str(units).strip().lower()
    if units_key in {"degree", "degrees", "deg"}:
        slope_format = "degree"
    elif units_key in {"percent", "percentage", "pct", "p"}:
        slope_format = "percent"
    else:
        raise TerrainAnalysisError(
            f"units must be 'degrees' or 'percent'; got {units!r}."
        )
    return _dem_processing(
        source,
        destination,
        mode="slope",
        band=band,
        algorithm=algorithm,
        scale=scale,
        compute_edges=compute_edges,
        creation_options=creation_options,
        overwrite=overwrite,
        slopeFormat=slope_format,
    )


def aspect(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    algorithm: str = "Horn",
    trigonometric: bool = False,
    zero_for_flat: bool = False,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Aspect from a DEM (``gdaldem aspect``).

    Values are degrees clockwise from north (0–360) unless ``trigonometric``
    is True. Flat cells are nodata (-9999) unless ``zero_for_flat`` is True.
    """
    return _dem_processing(
        source,
        destination,
        mode="aspect",
        band=band,
        algorithm=algorithm,
        compute_edges=compute_edges,
        creation_options=creation_options,
        overwrite=overwrite,
        trigonometric=trigonometric,
        zeroForFlat=zero_for_flat,
    )


def hillshade(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    algorithm: str = "Horn",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    scale: Optional[float] = None,
    combined: bool = False,
    multidirectional: bool = False,
    igor: bool = False,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Hillshade / shaded relief from a DEM (``gdaldem hillshade``).

    Set ``multidirectional=True`` for even illumination from several azimuths,
    ``combined=True`` for slope+oblique shading, or ``igor=True`` for Igor's
    shading (mutually exclusive with ``altitude`` semantics in gdaldem).
    """
    flags = sum(bool(v) for v in (combined, multidirectional, igor))
    if flags > 1:
        raise TerrainAnalysisError(
            "Specify at most one of combined, multidirectional, or igor."
        )
    return _dem_processing(
        source,
        destination,
        mode="hillshade",
        band=band,
        algorithm=algorithm,
        scale=scale,
        compute_edges=compute_edges,
        creation_options=creation_options,
        overwrite=overwrite,
        zFactor=float(z_factor),
        azimuth=None if multidirectional or igor else float(azimuth),
        altitude=None if multidirectional or igor else float(altitude),
        combined=combined,
        multiDirectional=multidirectional,
        igor=igor,
    )


def tri(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    algorithm: str = "Riley",
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Terrain Ruggedness Index (``gdaldem TRI``).

    ``algorithm`` is ``"Riley"`` (default, terrestrial) or ``"Wilson"``
    (bathymetric / mean absolute difference).
    """
    alg = str(algorithm).strip()
    if alg.lower() not in {"riley", "wilson"}:
        raise TerrainAnalysisError(
            f"TRI algorithm must be 'Riley' or 'Wilson'; got {algorithm!r}."
        )
    return _dem_processing(
        source,
        destination,
        mode="TRI",
        band=band,
        algorithm=alg.capitalize() if alg.lower() == "riley" else "Wilson",
        compute_edges=compute_edges,
        creation_options=creation_options,
        overwrite=overwrite,
    )


def tpi(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    window_size: Optional[int] = None,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Topographic Position Index.

    With the default 3×3 neighbourhood this uses ``gdaldem TPI``. When
    ``window_size`` is an odd integer >= 3, TPI is computed as elevation minus
    the mean of the surrounding window (centre cell excluded), matching the
    Whitebox relative topographic position concept at a chosen scale.
    """
    if window_size is None or int(window_size) == 3:
        return _dem_processing(
            source,
            destination,
            mode="TPI",
            band=band,
            compute_edges=compute_edges,
            creation_options=creation_options,
            overwrite=overwrite,
        )
    size = int(window_size)
    if size < 3 or size % 2 == 0:
        raise TerrainAnalysisError("window_size must be an odd integer >= 3.")
    dem, meta = _read_dem_band(source, band=band)
    result = _tpi_array(dem, size=size)
    return _write_float_raster(
        destination,
        result,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def roughness(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """Terrain roughness (``gdaldem roughness``): max − min in the 3×3 window."""
    return _dem_processing(
        source,
        destination,
        mode="roughness",
        band=band,
        compute_edges=compute_edges,
        creation_options=creation_options,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Curvature suite
# ---------------------------------------------------------------------------


def curvature(
    source: PathLike,
    destination: PathLike,
    *,
    curvature_type: str = "mean",
    band: int = 1,
    z_factor: float = 1.0,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Surface curvature from a DEM.

    ``curvature_type`` is one of:

    - ``"mean"`` — mean curvature
    - ``"tangential"`` — tangential (normal contour) curvature
    - ``"profile"`` — profile (normal slope-line) curvature
    - ``"plan"`` — plan (projected contour) curvature
    - ``"gaussian"`` — Gaussian curvature

    Derivatives follow Zevenbergen & Thorne (1987) finite differences on a
    3×3 window (Whitebox terrain-derivative family).
    """
    kind = str(curvature_type).strip().lower()
    aliases = {
        "mean_curvature": "mean",
        "tangential_curvature": "tangential",
        "normal_contour": "tangential",
        "profile_curvature": "profile",
        "normal_slope": "profile",
        "plan_curvature": "plan",
        "planform": "plan",
        "gaussian_curvature": "gaussian",
        "gauss": "gaussian",
    }
    kind = aliases.get(kind, kind)
    if kind not in _CURVATURE_TYPES:
        known = ", ".join(sorted(_CURVATURE_TYPES))
        raise TerrainAnalysisError(
            f"Unknown curvature_type {curvature_type!r}. Known: {known}."
        )
    dem, meta = _read_dem_band(source, band=band)
    result = _curvature_array(dem, meta["cellsize"], kind, z_factor=float(z_factor))
    return _write_float_raster(
        destination,
        result,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def mean_curvature(
    source: PathLike,
    destination: PathLike,
    **kwargs: Any,
) -> str:
    """Mean curvature (see :func:`curvature`)."""
    return curvature(source, destination, curvature_type="mean", **kwargs)


def tangential_curvature(
    source: PathLike,
    destination: PathLike,
    **kwargs: Any,
) -> str:
    """Tangential (normal contour) curvature (see :func:`curvature`)."""
    return curvature(source, destination, curvature_type="tangential", **kwargs)


def profile_curvature(
    source: PathLike,
    destination: PathLike,
    **kwargs: Any,
) -> str:
    """Profile (normal slope-line) curvature (see :func:`curvature`)."""
    return curvature(source, destination, curvature_type="profile", **kwargs)


def plan_curvature(
    source: PathLike,
    destination: PathLike,
    **kwargs: Any,
) -> str:
    """Plan (projected contour) curvature (see :func:`curvature`)."""
    return curvature(source, destination, curvature_type="plan", **kwargs)


def gaussian_curvature(
    source: PathLike,
    destination: PathLike,
    **kwargs: Any,
) -> str:
    """Gaussian curvature (see :func:`curvature`)."""
    return curvature(source, destination, curvature_type="gaussian", **kwargs)


# ---------------------------------------------------------------------------
# Wetness, landform, multiscale, viewshed, contours
# ---------------------------------------------------------------------------


def twi(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    z_factor: float = 1.0,
    min_slope_degrees: float = 0.001,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Topographic Wetness Index: ``ln(a / tan(β))``.

    Specific catchment area ``a`` is estimated from D8 flow accumulation and
    cell size; ``β`` is local slope in radians. Flat cells use
    ``min_slope_degrees`` to avoid division by zero.
    """
    dem, meta = _read_dem_band(source, band=band)
    cell = meta["cellsize"]
    slope_rad = _slope_radians(dem, cell, z_factor=float(z_factor))
    flow_acc = _d8_flow_accumulation(dem)
    # specific catchment area (m) ≈ (n_cells * cell^2) / cell = n_cells * cell
    sca = np.maximum(flow_acc.astype(np.float64) * cell, cell)
    min_slope = math.radians(max(float(min_slope_degrees), 1e-12))
    slope_safe = np.maximum(slope_rad, min_slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(sca / np.tan(slope_safe))
    result = np.where(np.isfinite(dem) & np.isfinite(result), result, np.nan)
    return _write_float_raster(
        destination,
        result,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def landform_classification(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    small_window: int = 3,
    large_window: int = 45,
    slope_threshold: float = 5.0,
    tpi_threshold: float = 1.0,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Landform classification from nested TPI scales (Weiss-style).

    Combines a small-neighbourhood TPI, a large-neighbourhood TPI, and slope
    (degrees) into integer landform classes:

    1 valley, 2 lower slope, 3 gentle slope, 4 steep slope,
    5 upper slope, 6 ridge, 7 flat.

    Returns a dict with ``path`` and ``classes`` (code → name).
    """
    small = int(small_window)
    large = int(large_window)
    if small < 3 or small % 2 == 0 or large < 3 or large % 2 == 0:
        raise TerrainAnalysisError("Window sizes must be odd integers >= 3.")
    if large <= small:
        raise TerrainAnalysisError("large_window must be greater than small_window.")

    dem, meta = _read_dem_band(source, band=band)
    cell = meta["cellsize"]
    tpi_small = _tpi_array(dem, size=small)
    tpi_large = _tpi_array(dem, size=large)
    slope_deg = np.degrees(_slope_radians(dem, cell))

    thr = float(tpi_threshold)
    # Standardize roughly by local SD of small TPI where available
    finite = np.isfinite(tpi_small)
    sd = float(np.nanstd(tpi_small)) if np.any(finite) else 1.0
    if sd < 1e-9:
        sd = 1.0
    zs = tpi_small / sd
    zl = tpi_large / sd

    classes = np.full(dem.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(dem) & np.isfinite(zs) & np.isfinite(zl) & np.isfinite(slope_deg)

    # Ridge / upper / valley / lower from nested TPI signs
    classes[valid & (zs > thr) & (zl > thr)] = 6  # ridge
    classes[valid & (zs > thr) & (np.abs(zl) <= thr)] = 5  # upper slope
    classes[valid & (zs < -thr) & (zl < -thr)] = 1  # valley
    classes[valid & (zs < -thr) & (np.abs(zl) <= thr)] = 2  # lower slope

    mid = valid & (np.abs(zs) <= thr)
    classes[mid & (slope_deg < float(slope_threshold))] = 7  # flat
    classes[
        mid
        & (slope_deg >= float(slope_threshold))
        & (slope_deg < float(slope_threshold) * 3.0)
    ] = 3  # gentle slope
    classes[mid & (slope_deg >= float(slope_threshold) * 3.0)] = 4  # steep slope

    # Remaining mid-slope with opposing nested TPI
    classes[valid & np.isnan(classes) & (zs > thr) & (zl < -thr)] = 5
    classes[valid & np.isnan(classes) & (zs < -thr) & (zl > thr)] = 2
    classes[valid & np.isnan(classes)] = 3

    path = _write_float_raster(
        destination,
        classes,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=-9999.0,
    )
    return {"path": path, "classes": dict(_LANDFORM_CLASSES)}


def multiscale_analysis(
    source: PathLike,
    destination: PathLike,
    *,
    metric: str = "tpi",
    band: int = 1,
    min_scale: int = 3,
    max_scale: int = 21,
    step: int = 2,
    z_factor: float = 1.0,
    output: str = "max",
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Multiscale terrain metric (Whitebox multiscale-signatures style).

    ``metric`` is one of ``"tpi"``, ``"roughness"``, ``"mean_curvature"``,
    ``"tri"``. Scales are odd window sizes from ``min_scale`` to ``max_scale``
    inclusive.

    ``output``:

    - ``"max"`` — magnitude of the metric at the scale of maximum |value|
      (writes ``destination`` and a sibling ``*_scale`` raster)
    - ``"stack"`` — multi-band stack of every scale (single destination)

    Returns paths in a dict (``value`` / ``scale`` or ``stack``).
    """
    metric_key = str(metric).strip().lower()
    aliases = {
        "relative_topographic_position": "tpi",
        "topographic_position": "tpi",
        "mean_curv": "mean_curvature",
        "curvature": "mean_curvature",
        "ruggedness": "tri",
    }
    metric_key = aliases.get(metric_key, metric_key)
    if metric_key not in {"tpi", "roughness", "mean_curvature", "tri"}:
        raise TerrainAnalysisError(
            "metric must be one of: tpi, roughness, mean_curvature, tri."
        )

    scales = _odd_scales(min_scale, max_scale, step)
    dem, meta = _read_dem_band(source, band=band)
    cell = meta["cellsize"]
    layers: List[np.ndarray] = []
    for size in scales:
        if metric_key == "tpi":
            layers.append(_tpi_array(dem, size=size))
        elif metric_key == "roughness":
            layers.append(_window_roughness(dem, size=size))
        elif metric_key == "tri":
            layers.append(_window_tri(dem, size=size))
        else:
            # Approximate multiscale mean curvature via DEM smoothed then 3×3
            smooth = _focal_mean(dem, size=size)
            layers.append(
                _curvature_array(smooth, cell, "mean", z_factor=float(z_factor))
            )

    stack = np.stack(layers, axis=0).astype(np.float32)
    out_mode = str(output).strip().lower()
    if out_mode == "stack":
        path = write_raster_data(
            destination,
            stack,
            crs=meta["crs"],
            geotransform=meta["geotransform"],
            nodata=_GDALDEM_NODATA,
            overwrite=overwrite,
            creation_options=creation_options,
        )
        return {"stack": path}

    if out_mode != "max":
        raise TerrainAnalysisError("output must be 'max' or 'stack'.")

    abs_stack = np.abs(stack)
    abs_stack = np.where(np.isfinite(abs_stack), abs_stack, -np.inf)
    idx = np.argmax(abs_stack, axis=0)
    rows, cols = np.indices(dem.shape)
    value = stack[idx, rows, cols]
    scale_raster = np.array(scales, dtype=np.float32)[idx]
    value = np.where(np.isfinite(dem), value, np.nan)
    scale_raster = np.where(np.isfinite(dem), scale_raster, np.nan)

    dest = Path(destination)
    scale_dest = dest.with_name(f"{dest.stem}_scale{dest.suffix or '.tif'}")
    value_path = _write_float_raster(
        dest,
        value,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )
    scale_path = _write_float_raster(
        scale_dest,
        scale_raster,
        meta,
        overwrite=True,
        creation_options=creation_options,
    )
    return {"value": value_path, "scale": scale_path}


def viewshed(
    source: PathLike,
    destination: PathLike,
    *,
    observer: Union[Sequence[float], Sequence[Sequence[float]]],
    band: int = 1,
    observer_height: float = 1.8,
    target_height: float = 0.0,
    max_distance: Optional[float] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Binary viewshed from one or more observer locations.

    ``observer`` is ``(x, y)`` in the DEM CRS, or a sequence of such points.
    Visible cells are 1, hidden cells are 0. Uses line-of-sight sampling along
    rays from each observer (Whitebox viewshed concept).
    """
    dem, meta = _read_dem_band(source, band=band)
    gt = meta["geotransform"]
    cell = meta["cellsize"]
    points = _normalize_observers(observer)
    visible = np.zeros(dem.shape, dtype=np.uint8)
    max_dist_cells = None
    if max_distance is not None:
        max_dist_cells = max(1, int(math.ceil(float(max_distance) / cell)))

    for x, y in points:
        col, row = _map_to_pixel(x, y, gt)
        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            raise TerrainAnalysisError(
                f"Observer ({x}, {y}) falls outside the DEM extent."
            )
        if not np.isfinite(dem[row, col]):
            raise TerrainAnalysisError(
                f"Observer ({x}, {y}) lands on a nodata DEM cell."
            )
        visible |= _viewshed_from_pixel(
            dem,
            row,
            col,
            observer_height=float(observer_height),
            target_height=float(target_height),
            max_distance_cells=max_dist_cells,
        )

    return write_raster_data(
        destination,
        visible,
        crs=meta["crs"],
        geotransform=gt,
        nodata=None,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def contours(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    interval: Optional[float] = None,
    fixed_levels: Optional[Sequence[float]] = None,
    base: float = 0.0,
    elev_field: str = "elev",
    id_field: str = "id",
    overwrite: bool = False,
) -> str:
    """
    Vector elevation contours from a DEM (``gdal raster contour``).

    Alias of :func:`geolattice.raster.basic_raster_operation.contour`.
    Provide either ``interval`` or ``fixed_levels``.
    """
    return contour(
        source,
        destination,
        band=band,
        interval=interval,
        fixed_levels=fixed_levels,
        base=base,
        elev_field=elev_field,
        id_field=id_field,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Internals — GDAL
# ---------------------------------------------------------------------------


def _dem_processing(
    source: PathLike,
    destination: PathLike,
    *,
    mode: str,
    band: int = 1,
    algorithm: Optional[str] = None,
    scale: Optional[float] = None,
    compute_edges: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
    **options: Any,
) -> str:
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    kwargs: Dict[str, Any] = {
        "band": int(band),
        "computeEdges": bool(compute_edges),
        "format": _format_from_path(dest),
    }
    co = _creation_options(creation_options)
    if co:
        kwargs["creationOptions"] = co
    if algorithm is not None:
        kwargs["alg"] = algorithm
    if scale is not None:
        kwargs["scale"] = float(scale)
    for key, value in options.items():
        if value is not None:
            kwargs[key] = value

    dem_options = gdal.DEMProcessingOptions(**kwargs)
    result = gdal.DEMProcessing(
        os.fspath(dest),
        os.fspath(source),
        mode,
        options=dem_options,
    )
    if result is None:
        raise TerrainAnalysisError(f"gdaldem {mode} failed for {dest}")
    result.FlushCache()
    result = None
    return str(dest)


# ---------------------------------------------------------------------------
# Internals — DEM I/O helpers
# ---------------------------------------------------------------------------


def _read_dem_band(source: PathLike, *, band: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        nodata = dataset.get_band(band).nodata
        gt = dataset.geotransform
        crs = dataset.crs
    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)
    cell_x = abs(float(gt[1]))
    cell_y = abs(float(gt[5]))
    if cell_x <= 0 or cell_y <= 0:
        raise TerrainAnalysisError("DEM geotransform has non-positive cell size.")
    cell = 0.5 * (cell_x + cell_y)
    meta = {
        "crs": crs,
        "geotransform": gt,
        "cellsize": cell,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "nodata": nodata,
    }
    return array, meta


def _write_float_raster(
    destination: PathLike,
    array: np.ndarray,
    meta: Mapping[str, Any],
    *,
    overwrite: bool,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    nodata: float = _GDALDEM_NODATA,
) -> str:
    out = np.asarray(array, dtype=np.float32)
    if nodata is not None:
        out = np.where(np.isfinite(out), out, nodata)
    return write_raster_data(
        destination,
        out,
        crs=meta["crs"],
        geotransform=meta["geotransform"],
        nodata=nodata,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# Internals — derivatives / windows
# ---------------------------------------------------------------------------


def _pad(dem: np.ndarray) -> np.ndarray:
    return np.pad(dem, 1, mode="edge")


def _horn_gradients(
    dem: np.ndarray,
    cell_x: float,
    cell_y: float,
    *,
    z_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dz/dx (p) and dz/dy (q) using Horn's 3×3 weighted differences."""
    z = _pad(dem * float(z_factor))
    # rows increase southward for north-up rasters (negative gt[5])
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z6 = z[1:-1, :-2], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    p = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * cell_x)
    q = ((z1 + 2.0 * z2 + z3) - (z7 + 2.0 * z8 + z9)) / (8.0 * cell_y)
    return p, q


def _zevenbergen_partials(
    dem: np.ndarray,
    cell: float,
    *,
    z_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return p, q, r, s, t for Zevenbergen & Thorne curvature formulas."""
    z = _pad(dem * float(z_factor))
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z5, z6 = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    l = float(cell)
    p = (z6 - z4) / (2.0 * l)
    q = (z2 - z8) / (2.0 * l)
    r = (z4 + z6 - 2.0 * z5) / (l * l)
    t = (z2 + z8 - 2.0 * z5) / (l * l)
    s = (z3 + z7 - z1 - z9) / (4.0 * l * l)
    return p, q, r, s, t


def _slope_radians(
    dem: np.ndarray,
    cell: float,
    *,
    z_factor: float = 1.0,
) -> np.ndarray:
    p, q = _horn_gradients(dem, cell, cell, z_factor=z_factor)
    return np.arctan(np.hypot(p, q))


def _curvature_array(
    dem: np.ndarray,
    cell: float,
    kind: str,
    *,
    z_factor: float = 1.0,
) -> np.ndarray:
    p, q, r, s, t = _zevenbergen_partials(dem, cell, z_factor=z_factor)
    p2 = p * p
    q2 = q * q
    pq = p2 + q2
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "gaussian":
            denom = (1.0 + pq) ** 2
            out = (r * t - s * s) / denom
        elif kind == "mean":
            denom = 2.0 * (1.0 + pq) ** 1.5
            out = -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t) / denom
        elif kind == "profile":
            denom = pq * (1.0 + pq) ** 1.5
            out = -(p2 * r + 2.0 * p * q * s + q2 * t) / denom
        elif kind == "plan":
            denom = pq ** 1.5
            out = -(q2 * r - 2.0 * p * q * s + p2 * t) / denom
        elif kind == "tangential":
            denom = pq * np.sqrt(1.0 + pq)
            out = -(q2 * r - 2.0 * p * q * s + p2 * t) / denom
        else:
            raise TerrainAnalysisError(f"Unsupported curvature kind {kind!r}.")
    # Undefined where surface is locally flat for directional curvatures
    if kind in {"profile", "plan", "tangential"}:
        out = np.where(pq > 1e-12, out, 0.0)
    out = np.where(np.isfinite(dem), out, np.nan)
    return out.astype(np.float64)


def _tpi_array(dem: np.ndarray, *, size: int) -> np.ndarray:
    mean = _focal_mean(dem, size=size)
    # Exclude centre from neighbourhood mean: (sum - z) / (n - 1)
    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy) <= (radius * radius + 1e-9)
    n = int(kernel.sum())
    if n <= 1:
        raise TerrainAnalysisError("TPI window must contain more than one cell.")
    neighbourhood_mean = (mean * n - dem) / (n - 1)
    return np.where(np.isfinite(dem), dem - neighbourhood_mean, np.nan)


def _focal_mean(dem: np.ndarray, *, size: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy) <= (radius * radius + 1e-9)
    padded = np.pad(dem, radius, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (size, size))
    # windows shape: (rows, cols, size, size)
    masked = np.where(kernel, windows, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(masked, axis=(2, 3))


def _window_roughness(dem: np.ndarray, *, size: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    radius = size // 2
    padded = np.pad(dem, radius, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (size, size))
    with np.errstate(all="ignore"):
        out = np.nanmax(windows, axis=(2, 3)) - np.nanmin(windows, axis=(2, 3))
    return np.where(np.isfinite(dem), out, np.nan)


def _window_tri(dem: np.ndarray, *, size: int) -> np.ndarray:
    """Riley-style TRI generalized to a circular window: sqrt(sum(diff^2))."""
    from numpy.lib.stride_tricks import sliding_window_view

    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy) <= (radius * radius + 1e-9)
    kernel[radius, radius] = False
    padded = np.pad(dem, radius, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (size, size))
    centre = dem[:, :, np.newaxis, np.newaxis]
    diff2 = (windows - centre) ** 2
    diff2 = np.where(kernel, diff2, np.nan)
    with np.errstate(all="ignore"):
        out = np.sqrt(np.nansum(diff2, axis=(2, 3)))
    return np.where(np.isfinite(dem), out, np.nan)


def _odd_scales(min_scale: int, max_scale: int, step: int) -> List[int]:
    lo = int(min_scale)
    hi = int(max_scale)
    st = int(step)
    if lo < 3 or lo % 2 == 0:
        raise TerrainAnalysisError("min_scale must be an odd integer >= 3.")
    if hi < lo or hi % 2 == 0:
        raise TerrainAnalysisError("max_scale must be an odd integer >= min_scale.")
    if st < 1:
        raise TerrainAnalysisError("step must be a positive integer.")
    scales = list(range(lo, hi + 1, st))
    scales = [s if s % 2 == 1 else s + 1 for s in scales]
    # de-duplicate while preserving order
    seen = set()
    out: List[int] = []
    for s in scales:
        if s <= hi and s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        raise TerrainAnalysisError("No valid odd scales in the requested range.")
    return out


# ---------------------------------------------------------------------------
# Internals — hydrology helpers for TWI
# ---------------------------------------------------------------------------


def _d8_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """D8 flow accumulation (cell counts, including self)."""
    rows, cols = dem.shape
    # D8 offsets indexed 0..7: E, SE, S, SW, W, NW, N, NE
    offsets = np.array(
        [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ],
        dtype=np.int32,
    )
    dist = np.array(
        [1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0)],
        dtype=np.float64,
    )

    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    best_drop = np.zeros(dem.shape, dtype=np.float64)
    flow_dir = np.full(dem.shape, -1, dtype=np.int8)

    for i, (dr, dc) in enumerate(offsets):
        neigh = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            drop = (centre - neigh) / dist[i]
        better = np.isfinite(drop) & (drop > best_drop)
        best_drop = np.where(better, drop, best_drop)
        flow_dir = np.where(better, i, flow_dir)
    flow_dir = np.where(np.isfinite(dem), flow_dir, -1)

    # Receiver index for each cell (-1 = sink / nodata / no downslope neighbour)
    flat = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    receiver = np.full(rows * cols, -1, dtype=np.int32)
    rows_g, cols_g = np.indices((rows, cols))
    for i, (dr, dc) in enumerate(offsets):
        mask = flow_dir == i
        rr = rows_g + int(dr)
        cc = cols_g + int(dc)
        in_bounds = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        use = mask & in_bounds
        receiver[flat[use]] = flat[rr[use], cc[use]]

    # Process cells in descending elevation order (donors before receivers)
    order = np.argsort(-np.nan_to_num(dem, nan=-np.inf), axis=None)
    acc = np.ones(rows * cols, dtype=np.float64)
    acc[~np.isfinite(dem.ravel())] = 0.0
    for idx in order:
        recv = receiver[idx]
        if recv >= 0:
            acc[recv] += acc[idx]

    out = acc.reshape(rows, cols)
    out[~np.isfinite(dem)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Internals — viewshed
# ---------------------------------------------------------------------------


def _normalize_observers(
    observer: Union[Sequence[float], Sequence[Sequence[float]]],
) -> List[Tuple[float, float]]:
    if not observer:
        raise TerrainAnalysisError("Provide at least one observer location.")
    # Single point: (x, y)
    if len(observer) == 2 and not isinstance(observer[0], (list, tuple, np.ndarray)):
        return [(float(observer[0]), float(observer[1]))]
    points: List[Tuple[float, float]] = []
    for item in observer:
        if len(item) < 2:
            raise TerrainAnalysisError("Each observer must be an (x, y) pair.")
        points.append((float(item[0]), float(item[1])))
    return points


def _map_to_pixel(
    x: float,
    y: float,
    gt: Sequence[float],
) -> Tuple[int, int]:
    # Inverse affine for north-up grids (gt[2]=gt[4]=0)
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])
    return col, row


def _viewshed_from_pixel(
    dem: np.ndarray,
    row0: int,
    col0: int,
    *,
    observer_height: float,
    target_height: float,
    max_distance_cells: Optional[int],
) -> np.ndarray:
    rows, cols = dem.shape
    visible = np.zeros(dem.shape, dtype=np.uint8)
    visible[row0, col0] = 1
    z0 = float(dem[row0, col0]) + float(observer_height)

    # Cast rays to every border cell
    border: List[Tuple[int, int]] = []
    for c in range(cols):
        border.append((0, c))
        border.append((rows - 1, c))
    for r in range(1, rows - 1):
        border.append((r, 0))
        border.append((r, cols - 1))

    for r1, c1 in border:
        _trace_ray(
            dem,
            visible,
            row0,
            col0,
            r1,
            c1,
            z0,
            target_height=target_height,
            max_distance_cells=max_distance_cells,
        )
    return visible


def _trace_ray(
    dem: np.ndarray,
    visible: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    z0: float,
    *,
    target_height: float,
    max_distance_cells: Optional[int],
) -> None:
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return
    max_angle = -math.inf
    for i in range(1, steps + 1):
        t = i / steps
        r = int(round(r0 + dr * t))
        c = int(round(c0 + dc * t))
        if r < 0 or r >= dem.shape[0] or c < 0 or c >= dem.shape[1]:
            break
        dist = math.hypot(r - r0, c - c0)
        if dist <= 0:
            continue
        if max_distance_cells is not None and dist > max_distance_cells:
            break
        z = dem[r, c]
        if not np.isfinite(z):
            continue
        angle = ((float(z) + float(target_height)) - z0) / dist
        if angle >= max_angle:
            visible[r, c] = 1
            max_angle = angle


__all__ = [
    "TerrainAnalysisError",
    "slope",
    "aspect",
    "hillshade",
    "tri",
    "tpi",
    "twi",
    "roughness",
    "curvature",
    "mean_curvature",
    "tangential_curvature",
    "profile_curvature",
    "plan_curvature",
    "gaussian_curvature",
    "landform_classification",
    "multiscale_analysis",
    "viewshed",
    "contours",
    "contour",
]
