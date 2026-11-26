"""
Surface hydrological analysis from digital elevation models (DEMs).

Implements WhiteboxTools-style hydrological workflows with NumPy / GDAL I/O:

1. DEM preprocessing — breach / fill depressions
2. Flow direction — D8, Rho8, D-infinity, FD8, MD-infinity, Quinn, minimal
   dispersion
3. Flow accumulation — D8 cell counts and D-infinity specific contributing area
4. Stream definition — threshold extraction
5. Stream segmentation and catchment delineation
6. Watershed delineation — single / multiple outlets, nested / subbasins
7. Topographic Wetness Index (TWI)
8. Terrain hydrologic indices — elevation above stream, downslope distance
9. Advanced — stochastic depression analysis, hydrologic connectivity,
   impoundment size index

References:
- O'Callaghan & Mark (1984) D8
- Fairfield & Leymarie (1991) Rho8
- Tarboton (1997) D-infinity
- Freeman (1991) FD8
- Seibert & McGlynn (2007) MD-infinity
- Quinn et al. (1991, 1995) multiple flow direction / wetness
- Lindsay (2016) Whitebox hydrological tools
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.raster.basic_raster_operation import (
    RasterOperationError,
    _prepare_destination,
)
from geolattice.raster.raster_read_write import open_raster_data, write_raster_data

PathLike = Union[str, Path]

_NODATA = -9999.0

# D8 neighbour order and Whitebox pointer codes (powers of 2)
# E, SE, S, SW, W, NW, N, NE
_D8_OFFSETS = np.array(
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
_D8_DIST = np.array(
    [1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0)],
    dtype=np.float64,
)
_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
_CODE_TO_DIR = {int(c): i for i, c in enumerate(_D8_CODES)}

# Facet pairs for D-infinity (Tarboton): consecutive neighbour pairs around cell
_DINF_FACETS = [
    (0, 1),  # E–SE
    (1, 2),  # SE–S
    (2, 3),  # S–SW
    (3, 4),  # SW–W
    (4, 5),  # W–NW
    (5, 6),  # NW–N
    (6, 7),  # N–NE
    (7, 0),  # NE–E
]
# Angle (radians from east, CCW) of the cardinal edge of each facet
_DINF_FACET_ANGLE = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64
) * math.pi


class SurfaceHydrologyError(RasterOperationError):
    """Raised when a surface-hydrology operation cannot be completed."""


# ---------------------------------------------------------------------------
# 1. DEM preprocessing
# ---------------------------------------------------------------------------


def breach_depressions(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    max_depth: Optional[float] = None,
    max_length: Optional[float] = None,
    fill_pits: bool = True,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Breach depressions in a DEM (Whitebox ``BreachDepressions`` style).

    Single-cell pits are opened to the lowest neighbour. Deeper / larger
    sinks are breached along a least-cost path toward the lowest pour point
    within ``max_length`` cells (when set). Remaining closed sinks can be
    left as-is or filled when ``fill_pits`` is True.
    """
    dem, meta = _read_dem_band(source, band=band)
    breached = dem.copy()
    breached = _breach_single_cell_pits(breached)

    max_len_cells = None
    if max_length is not None:
        max_len_cells = max(1, int(math.ceil(float(max_length) / meta["cellsize"])))

    breached = _breach_depressions_priority(
        breached,
        max_depth=max_depth,
        max_length_cells=max_len_cells,
    )

    if fill_pits:
        breached = _fill_depressions_priority_flood(breached)

    return _write_float_raster(
        destination,
        breached,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def fill_depressions(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    fix_flats: bool = True,
    flat_increment: Optional[float] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Fill depressions in a DEM (Wang & Liu / priority-flood style).

    All interior sinks are raised to their pour-point elevation. The flood
    starts from every valid cell on the data boundary, that is, on the raster
    edge or next to a nodata cell, so DEMs clipped to a polygon drain through
    their outline.

    When ``fix_flats`` is True the fill uses Priority-Flood + epsilon
    (Barnes, Lehman and Mulla, 2014): each flooded cell is raised slightly
    above the cell it was reached from, so filled depressions and existing
    flats keep an unambiguous flow direction. With ``flat_increment=None``
    the step is the smallest float32 increment at that elevation, which
    survives the float32 output; a positive ``flat_increment`` (map units)
    is used as the step when it is larger than that minimum.
    """
    dem, meta = _read_dem_band(source, band=band)
    if flat_increment is not None and float(flat_increment) <= 0:
        raise SurfaceHydrologyError("flat_increment must be positive.")
    filled = _fill_depressions_priority_flood(
        dem,
        epsilon=bool(fix_flats),
        increment=None if flat_increment is None else float(flat_increment),
    )
    return _write_float_raster(
        destination,
        filled,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# 2. Flow direction / pointers
# ---------------------------------------------------------------------------


def d8_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    esri_style: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    D8 flow pointer (O'Callaghan & Mark, 1984).

    Whitebox codes: 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE.
    Set ``esri_style=True`` for ESRI D8 codes (1 E … 128 NE clockwise from E).
    """
    dem, meta = _read_dem_band(source, band=band)
    direction = _d8_direction_index(dem)
    codes = _ESRI_D8_CODES if esri_style else _D8_CODES
    pointer = np.full(dem.shape, 0, dtype=np.float32)
    valid = direction >= 0
    pointer[valid] = codes[direction[valid]].astype(np.float32)
    pointer[~np.isfinite(dem)] = np.nan
    return _write_float_raster(
        destination,
        pointer,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=0.0,
    )


def rho8_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    seed: Optional[int] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Rho8 stochastic single-flow pointer (Fairfield & Leymarie, 1991).

    Among downslope neighbours, one receiver is drawn with probability
    proportional to drop / distance, reducing D8 cardinal bias.
    """
    dem, meta = _read_dem_band(source, band=band)
    rng = np.random.default_rng(seed)
    direction = _rho8_direction_index(dem, rng)
    pointer = np.full(dem.shape, 0, dtype=np.float32)
    valid = direction >= 0
    pointer[valid] = _D8_CODES[direction[valid]].astype(np.float32)
    pointer[~np.isfinite(dem)] = np.nan
    return _write_float_raster(
        destination,
        pointer,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=0.0,
    )


def dinf_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    D-infinity flow pointer (Tarboton, 1997).

    Output is flow direction in degrees clockwise from east (0–360), matching
    Whitebox ``DInfPointer``. Nodata / flat sinks are 0.
    """
    dem, meta = _read_dem_band(source, band=band)
    angle, _ = _dinf_flow(dem, meta["cell_x"], meta["cell_y"])
    # Convert CCW-from-east radians → CW-from-east degrees (Whitebox)
    deg = np.degrees(angle)
    cw = np.mod(360.0 - deg, 360.0)
    cw = np.where(np.isfinite(angle), cw, 0.0)
    cw = np.where(np.isfinite(dem), cw, np.nan)
    return _write_float_raster(
        destination,
        cw,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=-1.0,
    )


def fd8_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    exponent: float = 1.1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    FD8 (Freeman) multiple-flow-direction weights.

    Writes an 8-band raster of fractional outflow to E, SE, S, SW, W, NW, N,
    NE (band order). Weights are proportional to ``(drop/distance)**exponent``.
    """
    dem, meta = _read_dem_band(source, band=band)
    weights = _mfd_weights(dem, exponent=float(exponent), quinn=False)
    return _write_multiband_raster(
        destination,
        weights,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=_NODATA,
    )


def mdinf_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    MD-infinity flow apportionment (Seibert & McGlynn, 2007).

    Writes an 8-band fractional outflow raster (same band order as FD8).
    Flow is split among triangular facets using infinite-direction slopes.
    """
    dem, meta = _read_dem_band(source, band=band)
    weights = _mdinf_weights(dem, meta["cell_x"], meta["cell_y"])
    return _write_multiband_raster(
        destination,
        weights,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=_NODATA,
    )


def quinn_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    exponent: float = 1.0,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Quinn et al. FD8 variant (often used for wetness / slope-area workflows).

    Same multi-band weight layout as ``fd8_pointer``, default exponent 1.0.
    """
    dem, meta = _read_dem_band(source, band=band)
    weights = _mfd_weights(dem, exponent=float(exponent), quinn=True)
    return _write_multiband_raster(
        destination,
        weights,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=_NODATA,
    )


def minimal_dispersion_pointer(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Adaptive flow routing between D8 and D-infinity (minimal dispersion).

    On divergent / planar hillslopes, D-infinity proportions are used; in
    strongly convergent cells, flow collapses toward a single D8 receiver.
    Writes an 8-band fractional outflow raster.
    """
    dem, meta = _read_dem_band(source, band=band)
    weights = _minimal_dispersion_weights(dem, meta["cell_x"], meta["cell_y"])
    return _write_multiband_raster(
        destination,
        weights,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=_NODATA,
    )


# Aliases matching common Whitebox naming
d8_flow_direction = d8_pointer
dinf_flow_direction = dinf_pointer
d_infinity_pointer = dinf_pointer


# ---------------------------------------------------------------------------
# 3. Flow accumulation
# ---------------------------------------------------------------------------


def d8_flow_accumulation(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    out_type: str = "cells",
    log_transform: bool = False,
    pointer: Optional[PathLike] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Basic D8 flow accumulation.

    ``out_type``:

    - ``"cells"`` — upslope cell count including self (default)
    - ``"catchment_area"`` / ``"sca"`` — specific catchment area (m² / m ≈ m)
    - ``"area"`` — upslope area (m²)
    """
    dem, meta = _read_dem_band(source, band=band)
    if pointer is not None:
        pntr, _ = _read_dem_band(pointer, band=1)
        direction = _pointer_to_direction(pntr)
    else:
        direction = _d8_direction_index(dem)
    acc_cells = _accumulate_d8(dem, direction)

    key = str(out_type).strip().lower()
    cell = meta["cellsize"]
    if key in {"cells", "cell", "ncells"}:
        out = acc_cells
    elif key in {"area", "ca", "catchment area"}:
        out = acc_cells * (cell * cell)
    elif key in {"sca", "specific contributing area", "catchment_area", "specific_catchment_area"}:
        out = acc_cells * cell
    else:
        raise SurfaceHydrologyError(
            "out_type must be 'cells', 'area', or 'sca' / 'catchment_area'."
        )
    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(np.maximum(out, 1e-12))
    out = np.where(np.isfinite(dem), out, np.nan)
    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def dinf_flow_accumulation(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    out_type: str = "sca",
    log_transform: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    D-infinity flow accumulation / specific contributing area (Tarboton, 1997).

    Default ``out_type="sca"`` returns specific contributing area (m).
    Use ``"cells"`` or ``"area"`` for cell counts or upslope area (m²).
    """
    dem, meta = _read_dem_band(source, band=band)
    angle, proportions = _dinf_flow(dem, meta["cell_x"], meta["cell_y"])
    acc = _accumulate_mfd(dem, proportions)

    key = str(out_type).strip().lower()
    cell = meta["cellsize"]
    if key in {"cells", "cell", "ncells"}:
        out = acc
    elif key in {"area", "ca"}:
        out = acc * (cell * cell)
    elif key in {
        "sca",
        "specific contributing area",
        "specific_catchment_area",
        "catchment_area",
    }:
        out = acc * cell
    else:
        raise SurfaceHydrologyError(
            "out_type must be 'cells', 'area', or 'sca'."
        )
    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(np.maximum(out, 1e-12))
    out = np.where(np.isfinite(dem) & np.isfinite(angle), out, np.nan)
    out = np.where(np.isfinite(dem), out, np.nan)
    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# 4. Stream definition
# ---------------------------------------------------------------------------


def extract_streams(
    flow_accumulation: PathLike,
    destination: PathLike,
    *,
    threshold: float,
    band: int = 1,
    zero_background: bool = True,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Simple threshold stream extraction (Whitebox ``ExtractStreams``).

    Cells with flow accumulation ≥ ``threshold`` are marked as stream (1);
    background is 0 when ``zero_background`` else nodata.
    """
    if float(threshold) <= 0:
        raise SurfaceHydrologyError("threshold must be positive.")
    fa, meta = _read_dem_band(flow_accumulation, band=band)
    streams = np.where(np.isfinite(fa) & (fa >= float(threshold)), 1.0, 0.0)
    if not zero_background:
        streams = np.where(streams > 0, 1.0, np.nan)
    streams = np.where(np.isfinite(fa), streams, np.nan)
    return _write_float_raster(
        destination,
        streams,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=0.0 if zero_background else _NODATA,
    )


# ---------------------------------------------------------------------------
# 5. Stream segmentation & catchment delineation
# ---------------------------------------------------------------------------


def stream_segmentation(
    streams: PathLike,
    d8_pointer: PathLike,
    destination: PathLike,
    *,
    streams_band: int = 1,
    pointer_band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Assign unique IDs to stream links between junctions / heads / outlets.

    Input ``streams`` is a binary stream raster; ``d8_pointer`` is a D8 pointer
    grid (Whitebox codes).
    """
    stream, meta = _read_dem_band(streams, band=streams_band)
    pntr, _ = _read_dem_band(d8_pointer, band=pointer_band)
    binary = np.isfinite(stream) & (stream > 0)
    links = _segment_streams(binary, pntr)
    links = np.where(binary, links.astype(np.float64), np.nan)
    return _write_float_raster(
        destination,
        links,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def catchment_delineation(
    streams: PathLike,
    d8_pointer: PathLike,
    destination: PathLike,
    *,
    streams_band: int = 1,
    pointer_band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Delineate catchments draining to each stream link ID.

    Requires a stream-link raster (from ``stream_segmentation``) or a binary
    stream grid (then links are computed first). Non-stream upslope cells
    inherit the ID of the first stream cell they drain into.
    """
    stream, meta = _read_dem_band(streams, band=streams_band)
    pntr, _ = _read_dem_band(d8_pointer, band=pointer_band)
    if np.nanmax(stream) <= 1.0:
        binary = np.isfinite(stream) & (stream > 0)
        links = _segment_streams(binary, pntr)
    else:
        links = np.where(np.isfinite(stream) & (stream > 0), stream, 0.0)

    catchments = _label_upslope_from_seeds(links.astype(np.float64), pntr)
    catchments = np.where(catchments > 0, catchments, np.nan)
    return _write_float_raster(
        destination,
        catchments,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# 6. Watershed delineation
# ---------------------------------------------------------------------------


def snap_pour_points(
    pour_points: Union[PathLike, Sequence[Sequence[float]]],
    streams: PathLike,
    destination: Optional[PathLike] = None,
    *,
    snap_distance: float,
    streams_band: int = 1,
    prefer_flow_accumulation: Optional[PathLike] = None,
) -> Union[str, List[Tuple[float, float]]]:
    """
    Snap outlet points to the nearest stream cell (Jenson-style).

    ``pour_points`` may be a vector file path or a sequence of ``(x, y)``
    coordinates in the raster CRS. When ``destination`` is set, snapped points
    are written as a simple GeoJSON / shapefile-compatible point layer via
    OGR; otherwise a list of ``(x, y)`` tuples is returned.
    """
    stream, meta = _read_dem_band(streams, band=streams_band)
    binary = np.isfinite(stream) & (stream > 0)
    if not np.any(binary):
        raise SurfaceHydrologyError("Stream raster contains no stream cells.")

    weight = None
    if prefer_flow_accumulation is not None:
        fa, _ = _read_dem_band(prefer_flow_accumulation, band=1)
        weight = fa

    points = _load_points(pour_points)
    snapped: List[Tuple[float, float]] = []
    cell = meta["cellsize"]
    max_cells = max(1, int(math.ceil(float(snap_distance) / cell)))
    gt = meta["geotransform"]

    for x, y in points:
        col, row = _map_to_pixel(x, y, gt)
        sr, sc = _snap_to_stream(binary, row, col, max_cells, weight=weight)
        if sr is None:
            snapped.append((x, y))
        else:
            sx, sy = _pixel_to_map(sc, sr, gt)
            snapped.append((sx, sy))

    if destination is None:
        return snapped
    return _write_points(destination, snapped, meta["crs"])


def watershed(
    d8_pointer: PathLike,
    pour_points: Union[PathLike, Sequence[Sequence[float]]],
    destination: PathLike,
    *,
    pointer_band: int = 1,
    esri_style: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Delineate watersheds for one or more outlet points.

    Each outlet receives a unique integer ID. Works for a single outlet or
    multiple outlets / nested basins (upstream basins keep distinct IDs).
    """
    pntr, meta = _read_dem_band(d8_pointer, band=pointer_band)
    points = _load_points(pour_points)
    gt = meta["geotransform"]
    seeds = np.zeros(pntr.shape, dtype=np.float64)
    for i, (x, y) in enumerate(points, start=1):
        col, row = _map_to_pixel(x, y, gt)
        if 0 <= row < pntr.shape[0] and 0 <= col < pntr.shape[1]:
            seeds[row, col] = float(i)
    if not np.any(seeds > 0):
        raise SurfaceHydrologyError("No pour points fall inside the pointer raster.")

    basins = _label_upslope_from_seeds(seeds, pntr, esri_style=esri_style)
    basins = np.where(basins > 0, basins, np.nan)
    return _write_float_raster(
        destination,
        basins,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def subbasins(
    d8_pointer: PathLike,
    streams: PathLike,
    destination: PathLike,
    *,
    pointer_band: int = 1,
    streams_band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Subbasin delineation from stream links (Whitebox ``Subbasins`` style).

    Each stream segment outlet seeds an upslope subbasin.
    """
    return catchment_delineation(
        streams,
        d8_pointer,
        destination,
        streams_band=streams_band,
        pointer_band=pointer_band,
        creation_options=creation_options,
        overwrite=overwrite,
    )


def basins(
    d8_pointer: PathLike,
    destination: PathLike,
    *,
    pointer_band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Delineate whole drainage basins draining to DEM edges / sinks.

    Edge cells and internal no-flow sinks act as outlets.
    """
    pntr, meta = _read_dem_band(d8_pointer, band=pointer_band)
    direction = _pointer_to_direction(pntr)
    rows, cols = pntr.shape
    seeds = np.zeros((rows, cols), dtype=np.float64)
    outlet_id = 1
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(pntr[r, c]) or pntr[r, c] <= 0:
                continue
            d = direction[r, c]
            if d < 0:
                seeds[r, c] = outlet_id
                outlet_id += 1
                continue
            nr = r + int(_D8_OFFSETS[d, 0])
            nc = c + int(_D8_OFFSETS[d, 1])
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                seeds[r, c] = outlet_id
                outlet_id += 1
            elif not np.isfinite(pntr[nr, nc]):
                seeds[r, c] = outlet_id
                outlet_id += 1
    labeled = _label_upslope_from_seeds(seeds, pntr)
    labeled = np.where(labeled > 0, labeled, np.nan)
    return _write_float_raster(
        destination,
        labeled,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# 7. Topographic Wetness Index
# ---------------------------------------------------------------------------


def twi(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    method: str = "dinf",
    z_factor: float = 1.0,
    min_slope_degrees: float = 0.001,
    sca: Optional[PathLike] = None,
    slope_raster: Optional[PathLike] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Topographic Wetness Index: ``ln(a / tan(β))``.

    By default specific catchment area ``a`` is estimated with D-infinity
    (``method="dinf"``); use ``method="d8"`` for D8 SCA. Provide ``sca`` and
    ``slope_raster`` to skip internal derivation (Whitebox ``WetnessIndex``).
    """
    if sca is not None and slope_raster is not None:
        sca_arr, meta = _read_dem_band(sca, band=1)
        slope_deg, _ = _read_dem_band(slope_raster, band=1)
        slope_rad = np.radians(slope_deg)
        dem_mask = np.isfinite(sca_arr) & np.isfinite(slope_rad)
    else:
        dem, meta = _read_dem_band(source, band=band)
        cell = meta["cellsize"]
        slope_rad = _slope_radians(dem, cell, z_factor=float(z_factor))
        key = str(method).strip().lower()
        if key in {"dinf", "d-infinity", "d_infinity"}:
            _, proportions = _dinf_flow(dem, meta["cell_x"], meta["cell_y"])
            acc = _accumulate_mfd(dem, proportions)
            sca_arr = np.maximum(acc * cell, cell)
        elif key in {"d8"}:
            direction = _d8_direction_index(dem)
            acc = _accumulate_d8(dem, direction)
            sca_arr = np.maximum(acc * cell, cell)
        else:
            raise SurfaceHydrologyError("method must be 'dinf' or 'd8'.")
        dem_mask = np.isfinite(dem)

    min_slope = math.radians(max(float(min_slope_degrees), 1e-12))
    slope_safe = np.maximum(slope_rad, min_slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.maximum(sca_arr, 1e-12) / np.tan(slope_safe))
    result = np.where(dem_mask & np.isfinite(result), result, np.nan)
    return _write_float_raster(
        destination,
        result,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


wetness_index = twi


# ---------------------------------------------------------------------------
# 8. Terrain hydrologic indices
# ---------------------------------------------------------------------------


def elevation_above_stream(
    dem: PathLike,
    streams: PathLike,
    d8_pointer: PathLike,
    destination: PathLike,
    *,
    dem_band: int = 1,
    streams_band: int = 1,
    pointer_band: int = 1,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Elevation difference between each cell and the stream cell it drains to.

    Whitebox ``ElevationAboveStream`` analogue.
    """
    elev, meta = _read_dem_band(dem, band=dem_band)
    stream, _ = _read_dem_band(streams, band=streams_band)
    pntr, _ = _read_dem_band(d8_pointer, band=pointer_band)
    binary = np.isfinite(stream) & (stream > 0)
    stream_z = _downstream_stream_value(elev, binary, pntr, value=elev)
    out = elev - stream_z
    out = np.where(np.isfinite(elev) & np.isfinite(stream_z), out, np.nan)
    out = np.where(binary, 0.0, out)
    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def downslope_distance_to_stream(
    streams: PathLike,
    d8_pointer: PathLike,
    destination: PathLike,
    *,
    streams_band: int = 1,
    pointer_band: int = 1,
    dem: Optional[PathLike] = None,
    vertical: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Downslope flow-path distance from each cell to the nearest stream.

    Distances follow the D8 pointer. When ``vertical=True`` and ``dem`` is
    provided, the vertical drop to the stream is returned instead.
    """
    stream, meta = _read_dem_band(streams, band=streams_band)
    pntr, _ = _read_dem_band(d8_pointer, band=pointer_band)
    binary = np.isfinite(stream) & (stream > 0)
    cell = meta["cellsize"]

    if vertical:
        if dem is None:
            raise SurfaceHydrologyError("dem is required when vertical=True.")
        elev, _ = _read_dem_band(dem, band=1)
        stream_z = _downstream_stream_value(elev, binary, pntr, value=elev)
        out = elev - stream_z
        out = np.where(binary, 0.0, out)
    else:
        out = _downslope_distance(binary, pntr, cell)
    out = np.where(np.isfinite(pntr) | binary, out, np.nan)
    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


# ---------------------------------------------------------------------------
# 9 / 10. Advanced analyses
# ---------------------------------------------------------------------------


def stochastic_depression_analysis(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    n_iter: int = 100,
    error_magnitude: float = 0.1,
    seed: Optional[int] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Stochastic depression probability (Whitebox ``StochasticDepressionAnalysis``).

    Adds Gaussian elevation error of magnitude ``error_magnitude``, fills
    depressions, and records the fraction of iterations in which each cell was
    raised (i.e. inside a depression).
    """
    if int(n_iter) < 1:
        raise SurfaceHydrologyError("n_iter must be >= 1.")
    dem, meta = _read_dem_band(source, band=band)
    rng = np.random.default_rng(seed)
    count = np.zeros(dem.shape, dtype=np.float64)
    sigma = float(error_magnitude)
    for _ in range(int(n_iter)):
        noise = rng.normal(0.0, sigma, size=dem.shape)
        noisy = dem + noise
        noisy = np.where(np.isfinite(dem), noisy, np.nan)
        filled = _fill_depressions_priority_flood(noisy)
        dep = (filled - noisy) > 1e-9
        count += dep.astype(np.float64)
    prob = count / float(n_iter)
    prob = np.where(np.isfinite(dem), prob, np.nan)
    return _write_float_raster(
        destination,
        prob,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def hydrologic_connectivity(
    dem: PathLike,
    streams: PathLike,
    destination: PathLike,
    *,
    dem_band: int = 1,
    streams_band: int = 1,
    max_distance: Optional[float] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Binary hydrologic connectivity to the stream network along D8 flow paths.

    Cells that drain to a stream within ``max_distance`` (map units; optional)
    are 1; disconnected / beyond-threshold cells are 0.
    """
    elev, meta = _read_dem_band(dem, band=dem_band)
    stream, _ = _read_dem_band(streams, band=streams_band)
    binary = np.isfinite(stream) & (stream > 0)
    direction = _d8_direction_index(elev)
    # Build temporary pointer codes for distance walk
    pntr = np.zeros(elev.shape, dtype=np.float64)
    valid = direction >= 0
    pntr[valid] = _D8_CODES[direction[valid]]
    dist = _downslope_distance(binary, pntr, meta["cellsize"])
    connected = np.isfinite(dist)
    if max_distance is not None:
        connected &= dist <= float(max_distance)
    out = connected.astype(np.float64)
    out = np.where(np.isfinite(elev), out, np.nan)
    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
        nodata=0.0,
    )


def impoundment_analysis(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    dam_length: float = 11.0,
    dam_height: Optional[float] = None,
    out_type: str = "volume",
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Impoundment size index (simplified Whitebox ``ImpoundmentSizeIndex``).

    For each potential dam cell, estimates the upslope depression volume (m³)
    or area (m²) created by a dam of width ``dam_length`` (map units) and
    optional ``dam_height``. ``out_type`` is ``"volume"`` or ``"area"``.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    dem, meta = _read_dem_band(source, band=band)
    cell = meta["cellsize"]
    half = max(1, int(math.floor(float(dam_length) / (2.0 * cell))))
    win = max(half * 3, 5)
    area_cell = cell * cell
    rows, cols = dem.shape

    # Crest along an E–W dam axis (max in a 1 × (2*half+1) window)
    dam_w = 2 * half + 1
    pad_c = np.pad(dem, ((0, 0), (half, half)), mode="constant", constant_values=np.nan)
    crest = np.nanmax(sliding_window_view(pad_c, dam_w, axis=1), axis=-1)
    if dam_height is not None:
        crest = np.fmin(crest, dem + float(dam_height))

    # Local flood window: depth = max(crest - z, 0) within (2*win+1)^2 neighbourhood
    size = 2 * win + 1
    pad = np.pad(dem, win, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(pad, (size, size))  # (rows, cols, size, size)
    # Broadcast per-cell crest into the window stack
    crest_b = crest[:, :, None, None]
    depth = np.maximum(crest_b - windows, 0.0)
    valid = np.isfinite(windows) & (windows < crest_b)
    depth = np.where(valid, depth, 0.0)

    want_area = str(out_type).strip().lower() in {"area", "a"}
    if want_area:
        out = np.count_nonzero(depth > 0, axis=(2, 3)).astype(np.float64) * area_cell
    else:
        out = np.nansum(depth, axis=(2, 3)) * area_cell

    # No impoundment where crest does not exceed the cell, or DEM is nodata
    out = np.where(np.isfinite(dem) & np.isfinite(crest) & (crest > dem), out, 0.0)
    out = np.where(np.isfinite(dem), out, np.nan)

    return _write_float_raster(
        destination,
        out,
        meta,
        overwrite=overwrite,
        creation_options=creation_options,
    )


impoundment_size_index = impoundment_analysis


# ---------------------------------------------------------------------------
# Internals — I/O
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
        raise SurfaceHydrologyError("DEM geotransform has non-positive cell size.")
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
    nodata: float = _NODATA,
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


def _write_multiband_raster(
    destination: PathLike,
    bands: np.ndarray,
    meta: Mapping[str, Any],
    *,
    overwrite: bool,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    nodata: float = _NODATA,
) -> str:
    """``bands`` shaped (8, rows, cols)."""
    out = np.asarray(bands, dtype=np.float32)
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


def _write_points(
    destination: PathLike,
    points: Sequence[Tuple[float, float]],
    crs: Any,
) -> str:
    from osgeo import ogr, osr

    dest = Path(destination)
    _prepare_destination(dest, overwrite=True)
    suffix = dest.suffix.lower()
    if suffix in {".shp"}:
        driver_name = "ESRI Shapefile"
    elif suffix in {".geojson", ".json"}:
        driver_name = "GeoJSON"
    elif suffix in {".gpkg"}:
        driver_name = "GPKG"
    else:
        driver_name = "GeoJSON"
        if suffix == "":
            dest = dest.with_suffix(".geojson")

    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise SurfaceHydrologyError(f"OGR driver {driver_name!r} is unavailable.")
    if dest.exists():
        driver.DeleteDataSource(os.fspath(dest))
    ds = driver.CreateDataSource(os.fspath(dest))
    if ds is None:
        raise SurfaceHydrologyError(f"Could not create {dest}")

    srs = osr.SpatialReference()
    if crs is not None:
        wkt = getattr(crs, "to_wkt", None)
        if callable(wkt):
            srs.ImportFromWkt(crs.to_wkt())
        elif isinstance(crs, str):
            if crs.upper().startswith("EPSG:"):
                srs.ImportFromEPSG(int(crs.split(":")[1]))
            else:
                srs.ImportFromWkt(crs)

    layer = ds.CreateLayer("pour_points", srs, ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    for i, (x, y) in enumerate(points, start=1):
        feat = ogr.Feature(defn)
        feat.SetField("id", i)
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(float(x), float(y))
        feat.SetGeometry(pt)
        layer.CreateFeature(feat)
        feat = None
    ds = None
    return str(dest)


# ---------------------------------------------------------------------------
# Internals — depression handling
# ---------------------------------------------------------------------------


def _breach_single_cell_pits(dem: np.ndarray) -> np.ndarray:
    out = dem.copy()
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    lower = np.full(dem.shape, False)
    min_neigh = np.full(dem.shape, np.inf, dtype=np.float64)
    for dr, dc in _D8_OFFSETS:
        neigh = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            min_neigh = np.fmin(min_neigh, neigh)
            lower |= np.isfinite(neigh) & (neigh < centre)
    pits = np.isfinite(dem) & ~lower & np.isfinite(min_neigh)
    # Breach: lower pit to just below lowest neighbour
    out[pits] = min_neigh[pits] - 1e-4
    return out


def _breach_depressions_priority(
    dem: np.ndarray,
    *,
    max_depth: Optional[float],
    max_length_cells: Optional[int],
) -> np.ndarray:
    """
    Approximate least-cost breaching via priority flood with optional limits.

    Cells inside depressions deeper / farther than the limits are left for a
    subsequent fill pass when the caller sets ``fill_pits=True``.
    """
    filled = _fill_depressions_priority_flood(dem)
    depth = filled - dem
    depth = np.where(np.isfinite(depth), depth, 0.0)
    if max_depth is not None:
        # Only breach shallow depressions; deep ones remain for filling
        breach_mask = depth <= float(max_depth)
    else:
        breach_mask = depth > 0

    out = dem.copy()
    # Carve a path: set breached cells to filled elevation along shallow sinks
    # (equivalent to filling shallow ones — true channel cutting is heavier;
    # combine with single-cell breach above for common DEM prep).
    if max_length_cells is None:
        out = np.where(breach_mask & (depth > 0), filled, out)
        return out

    # Restrict breach to compact depressions using a cheap distance proxy
    from numpy.lib.stride_tricks import sliding_window_view

    size = 2 * int(max_length_cells) + 1
    radius = size // 2
    padded = np.pad((depth > 0).astype(np.float64), radius, mode="constant")
    windows = sliding_window_view(padded, (size, size))
    local = np.nansum(windows, axis=(2, 3))
    compact = local <= (math.pi * (max_length_cells**2))
    out = np.where(breach_mask & compact & (depth > 0), filled, out)
    return out


def _data_boundary(valid: np.ndarray) -> np.ndarray:
    """Valid cells on the raster edge or with at least one invalid D8 neighbour."""
    rows, cols = valid.shape
    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    interior = np.ones(valid.shape, dtype=bool)
    for dr, dc in _D8_OFFSETS:
        interior &= padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    return valid & ~interior


def _fill_depressions_priority_flood(
    dem: np.ndarray,
    *,
    epsilon: bool = False,
    increment: Optional[float] = None,
) -> np.ndarray:
    """
    Priority-flood depression fill (Barnes / Wang–Liu family).

    The flood is seeded from every valid cell on the data boundary (raster
    edge or adjacent to nodata), so nodata-bounded DEMs drain outward.

    With ``epsilon=True`` this is Priority-Flood + epsilon: a cell that is not
    higher than the cell it is reached from is raised just above it, which
    fills depressions and drains flats in one pass. The arithmetic is done in
    float32 so the gradient is preserved when the result is written as a
    float32 raster. ``increment`` optionally sets a larger step in map units.
    """
    import heapq

    rows, cols = dem.shape
    valid = np.isfinite(dem)
    out = dem.astype(np.float32) if epsilon else dem.copy()
    visited = ~valid
    heap: List[Tuple[float, int, int]] = []

    for r, c in zip(*np.where(_data_boundary(valid))):
        heap.append((float(out[r, c]), int(r), int(c)))
        visited[r, c] = True
    heapq.heapify(heap)

    inf32 = np.float32(np.inf)
    step = None if increment is None else np.float32(increment)
    offsets = [(int(dr), int(dc)) for dr, dc in _D8_OFFSETS]

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True
            nz = out[nr, nc]
            if epsilon:
                if nz <= z:
                    z32 = np.float32(z)
                    raised = np.nextafter(z32, inf32)
                    if step is not None:
                        raised = max(raised, np.float32(z32 + step))
                    out[nr, nc] = raised
                heapq.heappush(heap, (float(out[nr, nc]), nr, nc))
            elif nz < z:
                out[nr, nc] = z
                heapq.heappush(heap, (z, nr, nc))
            else:
                heapq.heappush(heap, (float(nz), nr, nc))

    if epsilon:
        return np.where(valid, out.astype(np.float64), np.nan)
    return out


# ---------------------------------------------------------------------------
# Internals — flow direction
# ---------------------------------------------------------------------------

_ESRI_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)


def _d8_direction_index(dem: np.ndarray) -> np.ndarray:
    """Return per-cell D8 neighbour index 0..7, or -1 if none."""
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    best_drop = np.zeros(dem.shape, dtype=np.float64)
    flow_dir = np.full(dem.shape, -1, dtype=np.int8)
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        neigh = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            drop = (centre - neigh) / _D8_DIST[i]
        better = np.isfinite(drop) & (drop > best_drop)
        # Tie-break: prefer first (deterministic) — already handled by >
        best_drop = np.where(better, drop, best_drop)
        flow_dir = np.where(better, i, flow_dir)
    flow_dir = np.where(np.isfinite(dem), flow_dir, -1)
    return flow_dir


def _rho8_direction_index(dem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    slopes = np.zeros((8, rows, cols), dtype=np.float64)
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        neigh = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            drop = (centre - neigh) / _D8_DIST[i]
        slopes[i] = np.where(np.isfinite(drop) & (drop > 0), drop, 0.0)

    total = slopes.sum(axis=0)
    flow_dir = np.full(dem.shape, -1, dtype=np.int8)
    flat = total <= 0
    # Vectorized roulette: cumulative / random
    cdf = np.cumsum(slopes, axis=0)
    u = rng.random(dem.shape) * np.maximum(total, 1e-30)
    chosen = np.full(dem.shape, -1, dtype=np.int8)
    for i in range(8):
        chosen = np.where((chosen < 0) & (u <= cdf[i]) & ~flat, i, chosen)
    flow_dir = chosen
    flow_dir = np.where(np.isfinite(dem), flow_dir, -1)
    return flow_dir


def _mfd_weights(
    dem: np.ndarray,
    *,
    exponent: float,
    quinn: bool,
) -> np.ndarray:
    """Return (8, rows, cols) fractional outflow weights."""
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    centre = padded[1:-1, 1:-1]
    weights = np.zeros((8, rows, cols), dtype=np.float64)
    exp = float(exponent)
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        neigh = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        with np.errstate(invalid="ignore"):
            drop = (centre - neigh) / _D8_DIST[i]
        positive = np.isfinite(drop) & (drop > 0)
        # Quinn originally uses contour length factor; approximate with distance
        factor = (1.0 / _D8_DIST[i]) if quinn else 1.0
        weights[i] = np.where(positive, (drop ** exp) * factor, 0.0)
    total = weights.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(total > 0, weights / total, 0.0)
    weights[:, ~np.isfinite(dem)] = np.nan
    return weights


def _dinf_flow(
    dem: np.ndarray,
    cell_x: float,
    cell_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (angle_ccw_from_east radians or nan, proportions[8,r,c]).
    """
    rows, cols = dem.shape
    e0 = dem
    proportions = np.zeros((8, rows, cols), dtype=np.float64)
    angle = np.full(dem.shape, np.nan, dtype=np.float64)

    # Neighbour elevations
    neigh = []
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    for dr, dc in _D8_OFFSETS:
        neigh.append(padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols])

    best_s = np.full(dem.shape, -np.inf, dtype=np.float64)
    best_r = np.full(dem.shape, -1, dtype=np.int8)
    best_s1 = np.zeros(dem.shape, dtype=np.float64)
    best_s2 = np.zeros(dem.shape, dtype=np.float64)

    ac = float(cell_x)
    af = float(cell_y)
    for facet, (i1, i2) in enumerate(_DINF_FACETS):
        e1 = neigh[i1]
        e2 = neigh[i2]
        # Map facet to (cardinal, diagonal) with correct dx/dy
        # Use Tarboton geometric form on unit cell
        with np.errstate(invalid="ignore"):
            s1 = (e0 - e1) / ac
            s2 = (e1 - e2) / af
        # Re-orient: alternate facets swap roles of card/diag distances
        if facet % 2 == 1:
            with np.errstate(invalid="ignore"):
                s1 = (e0 - e1) / af
                s2 = (e1 - e2) / ac
        r = np.arctan2(s2, s1)  # angle within facet
        s = np.hypot(s1, s2)
        # Constrain r to facet (0 .. atan(af/ac) approx pi/4)
        rmax = math.atan(af / ac) if facet % 2 == 0 else math.atan(ac / af)
        r = np.clip(r, 0.0, rmax)
        # Recompute slope when clipped
        use1 = np.isfinite(s1) & (s1 > 0) & np.isfinite(e1)
        use2 = use1 & np.isfinite(e2) & (s2 > 0)
        # If only cardinal downslope
        s_card = np.where(use1, s1, -np.inf)
        s_use = np.where(use2, s, s_card)
        r_use = np.where(use2, r, 0.0)
        better = s_use > best_s
        best_s = np.where(better, s_use, best_s)
        best_r = np.where(better, facet, best_r)
        best_s1 = np.where(better, s1, best_s1)
        best_s2 = np.where(better, np.where(use2, s2, 0.0), best_s2)

        # provisional store angle for this facet when selected later
        _ = r_use  # silence lint; angle assigned below from best facet

    # Assign proportions from winning facet
    for facet, (i1, i2) in enumerate(_DINF_FACETS):
        mask = best_r == facet
        if not np.any(mask):
            continue
        s1 = best_s1
        s2 = best_s2
        with np.errstate(invalid="ignore"):
            r = np.arctan2(s2, s1)
        rmax = math.atan(af / ac) if facet % 2 == 0 else math.atan(ac / af)
        r = np.clip(r, 0.0, rmax)
        # Proportion along diagonal vs cardinal within facet
        p_diag = np.where(rmax > 0, r / rmax, 0.0)
        p_card = 1.0 - p_diag
        proportions[i1] = np.where(mask, p_card, proportions[i1])
        proportions[i2] = np.where(mask, p_diag, proportions[i2])
        ang = _DINF_FACET_ANGLE[facet] + r
        angle = np.where(mask, ang, angle)

    # Fallback steepest D8 where DInf failed
    d8 = _d8_direction_index(dem)
    fallback = np.isnan(angle) & (d8 >= 0)
    for i in range(8):
        m = fallback & (d8 == i)
        proportions[i] = np.where(m, 1.0, proportions[i])
        angle = np.where(m, _DINF_FACET_ANGLE[i], angle)

    total = proportions.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        proportions = np.where(total > 0, proportions / total, 0.0)
    proportions[:, ~np.isfinite(dem)] = np.nan
    angle = np.where(np.isfinite(dem), angle, np.nan)
    return angle, proportions


def _mdinf_weights(dem: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """MD-infinity: apportion to all downslope triangular facets."""
    rows, cols = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    e0 = padded[1:-1, 1:-1]
    neigh = []
    for dr, dc in _D8_OFFSETS:
        neigh.append(padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols])

    weights = np.zeros((8, rows, cols), dtype=np.float64)
    ac, af = float(cell_x), float(cell_y)
    for facet, (i1, i2) in enumerate(_DINF_FACETS):
        e1, e2 = neigh[i1], neigh[i2]
        with np.errstate(invalid="ignore"):
            if facet % 2 == 0:
                s1 = (e0 - e1) / ac
                s2 = (e1 - e2) / af
            else:
                s1 = (e0 - e1) / af
                s2 = (e1 - e2) / ac
            s = np.hypot(s1, s2)
            valid = np.isfinite(s) & (s > 0) & (s1 > 0)
            rmax = math.atan(af / ac) if facet % 2 == 0 else math.atan(ac / af)
            r = np.clip(np.arctan2(s2, s1), 0.0, rmax)
            p_diag = np.where(rmax > 0, r / rmax, 0.0)
            p_card = 1.0 - p_diag
            w = np.where(valid, s, 0.0)
            weights[i1] += np.where(valid, w * p_card, 0.0)
            weights[i2] += np.where(valid, w * p_diag, 0.0)

    total = weights.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(total > 0, weights / total, 0.0)
    # Fallback D8
    d8 = _d8_direction_index(dem)
    none = (weights.sum(axis=0) <= 0) & (d8 >= 0)
    for i in range(8):
        weights[i] = np.where(none & (d8 == i), 1.0, weights[i])
    weights[:, ~np.isfinite(dem)] = np.nan
    return weights


def _minimal_dispersion_weights(
    dem: np.ndarray,
    cell_x: float,
    cell_y: float,
) -> np.ndarray:
    """Blend D8 and DInf based on local convergence."""
    _, dinf_w = _dinf_flow(dem, cell_x, cell_y)
    d8 = _d8_direction_index(dem)
    d8_w = np.zeros_like(dinf_w)
    for i in range(8):
        d8_w[i] = (d8 == i).astype(np.float64)

    rows, cols = dem.shape
    donors = np.zeros(dem.shape, dtype=np.float64)
    rows_g, cols_g = np.indices((rows, cols))
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        mask = d8 == i
        rr = rows_g + int(dr)
        cc = cols_g + int(dc)
        in_bounds = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        use = mask & in_bounds
        # Vectorized donor increment via flat indices
        flat_recv = rr[use] * cols + cc[use]
        np.add.at(donors.ravel(), flat_recv, 1.0)

    alpha = np.clip(donors / 3.0, 0.0, 1.0)
    weights = (1.0 - alpha) * dinf_w + alpha * d8_w
    total = weights.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(total > 0, weights / total, 0.0)
    weights[:, ~np.isfinite(dem)] = np.nan
    return weights


def _pointer_to_direction(pntr: np.ndarray, *, esri_style: bool = False) -> np.ndarray:
    codes = _ESRI_D8_CODES if esri_style else _D8_CODES
    mapping = {int(c): i for i, c in enumerate(codes)}
    direction = np.full(pntr.shape, -1, dtype=np.int8)
    flat_p = pntr.ravel()
    flat_d = direction.ravel()
    for idx, val in enumerate(flat_p):
        if not np.isfinite(val) or val <= 0:
            continue
        code = int(val)
        if code in mapping:
            flat_d[idx] = mapping[code]
    return direction


# ---------------------------------------------------------------------------
# Internals — accumulation
# ---------------------------------------------------------------------------


def _accumulate_d8(dem: np.ndarray, direction: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    flat = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    receiver = np.full(rows * cols, -1, dtype=np.int32)
    rows_g, cols_g = np.indices((rows, cols))
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        mask = direction == i
        rr = rows_g + int(dr)
        cc = cols_g + int(dc)
        in_bounds = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        use = mask & in_bounds
        receiver[flat[use]] = flat[rr[use], cc[use]]

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


def _accumulate_mfd(dem: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    order = np.argsort(-np.nan_to_num(dem, nan=-np.inf), axis=None)
    acc = np.ones(rows * cols, dtype=np.float64)
    acc[~np.isfinite(dem.ravel())] = 0.0
    flat_w = weights.reshape(8, -1)

    for idx in order:
        r, c = divmod(int(idx), cols)
        a = acc[idx]
        if a <= 0:
            continue
        for i, (dr, dc) in enumerate(_D8_OFFSETS):
            w = flat_w[i, idx]
            if not np.isfinite(w) or w <= 0:
                continue
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < rows and 0 <= nc < cols:
                acc[nr * cols + nc] += a * w
    out = acc.reshape(rows, cols)
    out[~np.isfinite(dem)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Internals — streams / watersheds
# ---------------------------------------------------------------------------


def _segment_streams(binary: np.ndarray, pntr: np.ndarray) -> np.ndarray:
    """Label stream links between junctions."""
    rows, cols = binary.shape
    direction = _pointer_to_direction(pntr)
    n_down = np.zeros(binary.shape, dtype=np.int8)
    n_up = np.zeros(binary.shape, dtype=np.int8)
    rows_g, cols_g = np.indices((rows, cols))

    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        flows = (direction == i) & binary
        rr = rows_g + int(dr)
        cc = cols_g + int(dc)
        inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        use = flows & inside
        valid_dest = np.zeros(binary.shape, dtype=bool)
        valid_dest[use] = binary[rr[use], cc[use]]
        n_down[valid_dest] = 1
        recv_r = rr[valid_dest]
        recv_c = cc[valid_dest]
        np.add.at(n_up, (recv_r, recv_c), 1)

    junction = binary & ((n_up >= 2) | (n_down == 0))
    head = binary & (n_up == 0)

    links = np.zeros(binary.shape, dtype=np.int32)
    link_id = 0
    starts = list(zip(*np.where(head | junction)))
    for sr, sc in starts:
        if not binary[sr, sc]:
            continue
        r, c = sr, sc
        if links[r, c] == 0 and head[r, c]:
            link_id += 1
            while True:
                if not binary[r, c]:
                    break
                links[r, c] = link_id
                if (junction[r, c] and not (r == sr and c == sc)) or direction[r, c] < 0:
                    break
                d = direction[r, c]
                nr, nc = r + int(_D8_OFFSETS[d, 0]), c + int(_D8_OFFSETS[d, 1])
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    break
                if junction[nr, nc]:
                    links[nr, nc] = link_id
                    break
                if links[nr, nc] > 0:
                    break
                r, c = nr, nc

    for r, c in zip(*np.where(binary & (links == 0))):
        link_id += 1
        cr, cc_ = r, c
        while binary[cr, cc_] and links[cr, cc_] == 0:
            links[cr, cc_] = link_id
            d = direction[cr, cc_]
            if d < 0:
                break
            nr, nc = cr + int(_D8_OFFSETS[d, 0]), cc_ + int(_D8_OFFSETS[d, 1])
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                break
            if links[nr, nc] > 0 or junction[nr, nc]:
                break
            cr, cc_ = nr, nc

    return links


def _label_upslope_from_seeds(
    seeds: np.ndarray,
    pntr: np.ndarray,
    *,
    esri_style: bool = False,
) -> np.ndarray:
    """Assign each cell the ID of the seed it drains to (upstream flood)."""
    rows, cols = seeds.shape
    direction = _pointer_to_direction(pntr, esri_style=esri_style)
    labels = seeds.copy().astype(np.float64)

    # Build donor lists: for each cell, donors that flow into it
    donors: List[List[int]] = [[] for _ in range(rows * cols)]
    for r in range(rows):
        for c in range(cols):
            d = direction[r, c]
            if d < 0:
                continue
            nr = r + int(_D8_OFFSETS[d, 0])
            nc = c + int(_D8_OFFSETS[d, 1])
            if 0 <= nr < rows and 0 <= nc < cols:
                donors[nr * cols + nc].append(r * cols + c)

    # BFS upstream from seeds
    from collections import deque

    q: deque = deque()
    for r, c in zip(*np.where(seeds > 0)):
        q.append((r, c, float(seeds[r, c])))

    while q:
        r, c, lab = q.popleft()
        for src in donors[r * cols + c]:
            sr, sc = divmod(src, cols)
            if labels[sr, sc] > 0:
                continue
            labels[sr, sc] = lab
            q.append((sr, sc, lab))
    return labels


def _downstream_stream_value(
    dem: np.ndarray,
    binary: np.ndarray,
    pntr: np.ndarray,
    *,
    value: np.ndarray,
) -> np.ndarray:
    """For each cell, value at the first stream cell along the D8 path."""
    rows, cols = dem.shape
    direction = _pointer_to_direction(pntr)
    out = np.full(dem.shape, np.nan, dtype=np.float64)
    out[binary] = value[binary]

    order = np.argsort(np.nan_to_num(dem, nan=np.inf), axis=None)  # low to high? need downslope first
    # Process from streams upward: visit in reverse flow order (low elev first if filled)
    # Better: memoized walk
    memo = out.copy()

    def walk(r: int, c: int) -> float:
        if np.isfinite(memo[r, c]):
            return float(memo[r, c])
        if binary[r, c]:
            memo[r, c] = float(value[r, c])
            return float(memo[r, c])
        d = direction[r, c]
        if d < 0:
            return np.nan
        nr, nc = r + int(_D8_OFFSETS[d, 0]), c + int(_D8_OFFSETS[d, 1])
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            return np.nan
        # cycle guard
        memo[r, c] = np.inf
        result = walk(nr, nc)
        memo[r, c] = result
        return result

    for r in range(rows):
        for c in range(cols):
            if np.isfinite(dem[r, c]):
                walk(r, c)
    memo[memo == np.inf] = np.nan
    return memo


def _downslope_distance(
    binary: np.ndarray,
    pntr: np.ndarray,
    cell: float,
) -> np.ndarray:
    rows, cols = binary.shape
    direction = _pointer_to_direction(pntr)
    out = np.full(binary.shape, np.nan, dtype=np.float64)
    out[binary] = 0.0
    memo = out.copy()

    def walk(r: int, c: int) -> float:
        if np.isfinite(memo[r, c]) and memo[r, c] != np.inf:
            return float(memo[r, c])
        if binary[r, c]:
            memo[r, c] = 0.0
            return 0.0
        d = direction[r, c]
        if d < 0:
            return np.nan
        nr, nc = r + int(_D8_OFFSETS[d, 0]), c + int(_D8_OFFSETS[d, 1])
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            return np.nan
        memo[r, c] = np.inf
        rest = walk(nr, nc)
        if not np.isfinite(rest):
            memo[r, c] = np.nan
            return np.nan
        step = float(cell) * _D8_DIST[d]
        memo[r, c] = rest + step
        return float(memo[r, c])

    for r in range(rows):
        for c in range(cols):
            if np.isfinite(pntr[r, c]) or binary[r, c]:
                walk(r, c)
    memo[memo == np.inf] = np.nan
    return memo


def _snap_to_stream(
    binary: np.ndarray,
    row: int,
    col: int,
    max_cells: int,
    *,
    weight: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], Optional[int]]:
    rows, cols = binary.shape
    if 0 <= row < rows and 0 <= col < cols and binary[row, col]:
        return row, col
    best = None
    best_score = -np.inf
    for dr in range(-max_cells, max_cells + 1):
        for dc in range(-max_cells, max_cells + 1):
            if dr * dr + dc * dc > max_cells * max_cells:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and binary[r, c]:
                if weight is not None and np.isfinite(weight[r, c]):
                    score = float(weight[r, c])
                else:
                    score = -math.hypot(dr, dc)
                if score > best_score:
                    best_score = score
                    best = (r, c)
    if best is None:
        return None, None
    return best


def _load_points(
    pour_points: Union[PathLike, Sequence[Sequence[float]]],
) -> List[Tuple[float, float]]:
    if isinstance(pour_points, (str, Path)):
        from geolattice.vector.read_vector_data import open_vectors_data

        points: List[Tuple[float, float]] = []
        with open_vectors_data(pour_points) as ds:
            for feat in ds.layer.features():
                g = feat.geometry
                if g is None:
                    continue
                if hasattr(g, "geom_type") and g.geom_type != "Point":
                    c = g.centroid
                    points.append((float(c.x), float(c.y)))
                else:
                    points.append((float(g.x), float(g.y)))
        if not points:
            raise SurfaceHydrologyError(f"No points found in {pour_points}")
        return points

    pts: List[Tuple[float, float]] = []
    for item in pour_points:
        if len(item) < 2:
            raise SurfaceHydrologyError("Each pour point must be an (x, y) pair.")
        pts.append((float(item[0]), float(item[1])))
    if not pts:
        raise SurfaceHydrologyError("Provide at least one pour point.")
    return pts


def _map_to_pixel(x: float, y: float, gt: Sequence[float]) -> Tuple[int, int]:
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])
    return col, row


def _pixel_to_map(col: int, row: int, gt: Sequence[float]) -> Tuple[float, float]:
    x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
    y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return float(x), float(y)


def _slope_radians(dem: np.ndarray, cell: float, *, z_factor: float = 1.0) -> np.ndarray:
    z = np.pad(dem * float(z_factor), 1, mode="edge")
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z6 = z[1:-1, :-2], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    p = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * cell)
    q = ((z1 + 2.0 * z2 + z3) - (z7 + 2.0 * z8 + z9)) / (8.0 * cell)
    return np.arctan(np.hypot(p, q))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SurfaceHydrologyError",
    # DEM prep
    "breach_depressions",
    "fill_depressions",
    # Flow direction
    "d8_pointer",
    "d8_flow_direction",
    "rho8_pointer",
    "dinf_pointer",
    "dinf_flow_direction",
    "d_infinity_pointer",
    "fd8_pointer",
    "mdinf_pointer",
    "quinn_pointer",
    "minimal_dispersion_pointer",
    # Flow accumulation
    "d8_flow_accumulation",
    "dinf_flow_accumulation",
    # Streams
    "extract_streams",
    "stream_segmentation",
    "catchment_delineation",
    # Watersheds
    "snap_pour_points",
    "watershed",
    "subbasins",
    "basins",
    # Indices
    "twi",
    "wetness_index",
    "elevation_above_stream",
    "downslope_distance_to_stream",
    # Advanced
    "stochastic_depression_analysis",
    "hydrologic_connectivity",
    "impoundment_analysis",
    "impoundment_size_index",
]
