"""
Basic raster operations built on GDAL (and NumPy for algebra / focal stats).

These helpers mirror the high-level ``gdal raster`` sub-commands documented at
https://gdal.org/en/stable/programs/gdal_raster.html while remaining compatible
with classic GDAL Python bindings (``gdal.Translate``, ``gdal.Warp``,
``gdal.FillNodata``, ``gdal.Polygonize``, ``gdal.ComputeProximity``,
``gdal.ContourGenerate``, ``gdal.BuildVRT``, ``gdal.DEMProcessing``).


Functions:
- create: Create a new raster dataset
- edit: Edit raster metadata without changing pixel dimensions
- convert: Convert a raster dataset
- resize: Resize a raster without changing georeferenced extents
- unscale: Convert scaled values into unscaled values using band scale/offset
- fill_nodata: Fill nodata regions by interpolation from edges
- neighbors: Compute each pixel from its neighbors / focal statistics
- focal_statistics: Moving-window focal statistics over a single raster band
- calc: Perform raster algebra
- raster_algebra: Evaluate a NumPy-backed raster algebra expression and write the result
- blend: Blend two rasters
- compare: Compare two rasters
- clip: Clip a raster to a bounding box
- reclassify: Reclassify a raster with explicit value / range rules
- raster_histogram: Histogram breaks and counts for a raster band
- class_intervals: Classification break points (classInt-compatible styles)
- reclassify_by: Histogram / quantile / cluster-based reclassification
- resample: Resample a raster to a new resolution
- aggregate: Aggregate a raster by a factor
- disaggregate: Disaggregate a raster by a factor
- mosaic: Mosaic multiple rasters
- stack: Stack multiple rasters
- summarize_raster: Summarize a raster
- zonal_stats: Calculate zonal statistics
- rasterize: Burn vector features into a raster grid
- polygonize: Polygonize a raster
- proximity: Calculate proximity
- color_map: Create a color map
- contour: Create a contour
- _gdal: Get the GDAL library
- _gdal_ogr: Get the GDAL OGR library
- _band_nodata: Get the nodata value for a band
- _normalize_dtype: Normalize a dtype
- _gdal_dtype: Normalize a GDAL dtype
- _normalize_resample: Normalize a resample method
- _creation_options: Normalize creation options
- _format_from_path: Get the format from a path
- _vector_driver_from_path: Get the vector driver from a path
- _prepare_destination: Prepare a destination
- _broadcast_band_values: Broadcast band values
- _bounds_to_proj_win: Convert bounds to projection window
- _translate: Translate a raster
- _focal_kernel: Get the focal kernel
- _apply_focal: Apply a focal statistic
- _eval_expression: Evaluate an expression
- _eval_node: Evaluate a node
- _zone_summary: Calculate zone summary
- _read_band_array: Read a band array
- _polygonize: Polygonize a raster
- _proximity: Calculate proximity



"""

from __future__ import annotations

import ast
import math
import operator
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.raster.raster_read_write import (
    RasterDataset,
    open_raster_data,
    write_raster_data,
)

PathLike = Union[str, Path]

_DTYPE_ALIASES: Dict[str, str] = {
    "byte": "uint8",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "float32": "float32",
    "float64": "float64",
    "float": "float32",
    "double": "float64",
}

_NUMPY_TO_GDAL_NAME: Dict[str, str] = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}

_RESAMPLE_ALIASES = {
    "near": "near",
    "nearest": "near",
    "bilinear": "bilinear",
    "cubic": "cubic",
    "cubicspline": "cubicspline",
    "lanczos": "lanczos",
    "average": "average",
    "mode": "mode",
    "max": "max",
    "min": "min",
    "med": "med",
    "q1": "q1",
    "q3": "q3",
    "sum": "sum",
    "rms": "rms",
}

_FOCAL_STATS = {
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "std",
    "var",
    "range",
    "majority",
    "count",
}

# classInt-compatible classification styles (R classIntervals)
_CLASSIFY_STYLES = {
    "fixed",
    "sd",
    "equal",
    "pretty",
    "quantile",
    "kmeans",
    "hclust",
    "bclust",
    "fisher",
    "jenks",
}

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_SAFE_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}
_SAFE_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "clip": np.clip,
    "isnan": np.isnan,
    "isfinite": np.isfinite,
    "nan_to_num": np.nan_to_num,
}


class RasterOperationError(ValueError):
    """Raised when a basic raster operation cannot be completed."""


# ---------------------------------------------------------------------------
# create / edit / convert / resize / unscale
# ---------------------------------------------------------------------------


def create(
    path: PathLike,
    *,
    width: int,
    height: int,
    bands: int = 1,
    dtype: Union[str, np.dtype] = "float32",
    crs: Any = None,
    geotransform: Optional[Sequence[float]] = None,
    nodata: Optional[float] = None,
    fill: Optional[float] = None,
    driver: Optional[str] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Create a new raster dataset (``gdal raster create``).

    When ``fill`` is omitted the array is zeros (or ``nodata`` when set).
    """
    if width <= 0 or height <= 0 or bands <= 0:
        raise RasterOperationError("width, height, and bands must be positive.")
    np_dtype = np.dtype(_normalize_dtype(dtype))
    value = 0 if fill is None else fill
    if fill is None and nodata is not None:
        value = nodata
    array = np.full((bands, height, width), value, dtype=np_dtype)
    if bands == 1:
        array = array[0]
    return write_raster_data(
        path,
        array,
        driver=driver,
        crs=crs,
        geotransform=geotransform,
        nodata=nodata,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def edit(
    source: PathLike,
    destination: Optional[PathLike] = None,
    *,
    crs: Any = None,
    geotransform: Optional[Sequence[float]] = None,
    nodata: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    band_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    scale: Optional[Union[float, Sequence[float]]] = None,
    offset: Optional[Union[float, Sequence[float]]] = None,
    description: Optional[Union[str, Sequence[str]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Edit raster metadata without changing pixel dimensions (``gdal raster edit``).

    Writes a CreateCopy of ``source`` and applies CRS, geotransform, nodata,
    scale/offset, descriptions, and metadata. When ``destination`` is omitted
    the source file is updated in place (via a temporary sibling file).
    """
    gdal = _gdal()
    source_path = Path(source)
    in_place = destination is None
    if in_place:
        destination = source_path.with_name(
            f".{source_path.stem}_edit_tmp{source_path.suffix or '.tif'}"
        )
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite or in_place)

    src_ds = gdal.Open(os.fspath(source_path), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RasterOperationError(f"Could not open raster: {source_path}")
    try:
        driver = src_ds.GetDriver()
        if driver is None:
            driver = gdal.GetDriverByName("GTiff")
        copy = driver.CreateCopy(os.fspath(dest), src_ds, strict=0)
        if copy is None:
            raise RasterOperationError(f"Could not create editable copy at {dest}")
        try:
            if crs is not None:
                from geolattice.raster.raster_CRS import crs_to_gdal

                copy.SetProjection(crs_to_gdal(crs))
            if geotransform is not None:
                if len(geotransform) != 6:
                    raise RasterOperationError("geotransform must contain six values.")
                copy.SetGeoTransform(tuple(float(v) for v in geotransform))
            if metadata is not None:
                copy.SetMetadata({str(k): str(v) for k, v in metadata.items()})

            band_count = copy.RasterCount
            scales = _broadcast_band_values(scale, band_count, "scale")
            offsets = _broadcast_band_values(offset, band_count, "offset")
            descriptions = _broadcast_band_values(description, band_count, "description")
            band_meta = list(band_metadata or [])

            for index in range(1, band_count + 1):
                band = copy.GetRasterBand(index)
                if nodata is not None:
                    band.SetNoDataValue(float(nodata))
                if scales[index - 1] is not None:
                    band.SetScale(float(scales[index - 1]))
                if offsets[index - 1] is not None:
                    band.SetOffset(float(offsets[index - 1]))
                if descriptions[index - 1] is not None:
                    band.SetDescription(str(descriptions[index - 1]))
                if index - 1 < len(band_meta) and band_meta[index - 1]:
                    band.SetMetadata(
                        {str(k): str(v) for k, v in band_meta[index - 1].items()}
                    )
                band = None
            copy.FlushCache()
        finally:
            copy = None
    finally:
        src_ds = None

    if in_place:
        os.replace(os.fspath(dest), os.fspath(source_path))
        return str(source_path)
    return str(dest)


def convert(
    source: PathLike,
    destination: PathLike,
    *,
    format: Optional[str] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    bands: Optional[Sequence[int]] = None,
    nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
    **translate_kwds: Any,
) -> str:
    """Convert a raster dataset (``gdal raster convert`` / ``gdal_translate``)."""
    return _translate(
        source,
        destination,
        format=format,
        outputType=_gdal_dtype(dtype) if dtype is not None else None,
        bandList=list(bands) if bands is not None else None,
        noData=nodata,
        creationOptions=_creation_options(creation_options) if creation_options else None,
        overwrite=overwrite,
        **translate_kwds,
    )


def resize(
    source: PathLike,
    destination: PathLike,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    width_pct: Optional[float] = None,
    height_pct: Optional[float] = None,
    resample: str = "near",
    format: Optional[str] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Resize a raster without changing georeferenced extents
    (``gdal raster resize``).

    Provide either absolute ``width`` / ``height`` or percentage
    ``width_pct`` / ``height_pct`` (100 = original size).
    """
    if not any(v is not None for v in (width, height, width_pct, height_pct)):
        raise RasterOperationError(
            "Provide width/height or width_pct/height_pct for resize."
        )
    return _translate(
        source,
        destination,
        format=format,
        width=int(width or 0),
        height=int(height or 0),
        widthPct=float(width_pct or 0.0),
        heightPct=float(height_pct or 0.0),
        resampleAlg=_normalize_resample(resample),
        creationOptions=_creation_options(creation_options) if creation_options else None,
        overwrite=overwrite,
    )


def unscale(
    source: PathLike,
    destination: PathLike,
    *,
    format: Optional[str] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Convert scaled values into unscaled values using band scale/offset
    (``gdal raster unscale``).
    """
    return _translate(
        source,
        destination,
        format=format,
        unscale=True,
        outputType=_gdal_dtype(dtype) if dtype is not None else None,
        creationOptions=_creation_options(creation_options) if creation_options else None,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# fill-nodata / neighbors / focal / calc / algebra
# ---------------------------------------------------------------------------


def fill_nodata(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    max_distance: float = 100.0,
    smoothing_iterations: int = 0,
    mask_band: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """
    Fill nodata regions by interpolation from edges (``gdal raster fill-nodata``).
    """
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    src_ds = gdal.Open(os.fspath(source), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RasterOperationError(f"Could not open raster: {source}")
    try:
        driver = gdal.GetDriverByName(_format_from_path(dest))
        copy = driver.CreateCopy(os.fspath(dest), src_ds, strict=0)
        if copy is None:
            raise RasterOperationError(f"Could not create output at {dest}")
        try:
            target = copy.GetRasterBand(band)
            mask = None
            if mask_band is not None:
                mask = copy.GetRasterBand(mask_band)
            status = gdal.FillNodata(
                targetBand=target,
                maskBand=mask,
                maxSearchDist=float(max_distance),
                smoothingIterations=int(smoothing_iterations),
            )
            if status != 0:
                raise RasterOperationError(f"FillNodata failed with status {status}")
            target.FlushCache()
            copy.FlushCache()
        finally:
            copy = None
    finally:
        src_ds = None
    return str(dest)


def neighbors(
    source: PathLike,
    destination: PathLike,
    *,
    statistic: str = "mean",
    size: int = 3,
    band: int = 1,
    circular: bool = False,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Compute each pixel from its neighbors / focal statistics
    (``gdal raster neighbors``, alias ``neighbours``).
    """
    return focal_statistics(
        source,
        destination,
        statistic=statistic,
        size=size,
        band=band,
        circular=circular,
        nodata=nodata,
        overwrite=overwrite,
    )


neighbours = neighbors


def focal_statistics(
    source: PathLike,
    destination: PathLike,
    *,
    statistic: str = "mean",
    size: int = 3,
    band: int = 1,
    circular: bool = False,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """Moving-window focal statistics over a single raster band."""
    stat = str(statistic).strip().lower()
    if stat not in _FOCAL_STATS:
        known = ", ".join(sorted(_FOCAL_STATS))
        raise RasterOperationError(f"Unknown focal statistic {statistic!r}. Known: {known}")
    if size < 1 or size % 2 == 0:
        raise RasterOperationError("Focal window size must be an odd integer >= 1.")

    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        src_nodata = _band_nodata(dataset, band) if nodata is None else nodata
        crs = dataset.crs
        gt = dataset.geotransform
        out_nodata = src_nodata if src_nodata is not None else np.nan

    if src_nodata is not None:
        valid = array != src_nodata
        work = np.where(valid, array, np.nan)
    else:
        work = array.astype(np.float64, copy=True)

    kernel = _focal_kernel(size, circular=circular)
    result = _apply_focal(work, kernel, stat)
    if out_nodata is not None and not (isinstance(out_nodata, float) and math.isnan(out_nodata)):
        result = np.where(np.isnan(result), out_nodata, result)

    return write_raster_data(
        destination,
        result.astype(np.float32),
        crs=crs,
        geotransform=gt,
        nodata=None if (isinstance(out_nodata, float) and math.isnan(out_nodata)) else out_nodata,
        overwrite=overwrite,
    )


def calc(
    expression: str,
    destination: PathLike,
    *,
    inputs: Optional[Mapping[str, PathLike]] = None,
    source: Optional[PathLike] = None,
    dtype: Union[str, np.dtype] = "float32",
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Perform raster algebra (``gdal raster calc``).

    ``expression`` is a Python expression over band / input names. For a single
    ``source``, bands are available as ``A`` / ``B`` / … or ``band1`` / ``band2``.
    With ``inputs={"DEM": "dem.tif", "MASK": "mask.tif"}``, use those names.
    """
    return raster_algebra(
        expression,
        destination,
        inputs=inputs,
        source=source,
        dtype=dtype,
        nodata=nodata,
        overwrite=overwrite,
    )


def raster_algebra(
    expression: str,
    destination: PathLike,
    *,
    inputs: Optional[Mapping[str, PathLike]] = None,
    source: Optional[PathLike] = None,
    dtype: Union[str, np.dtype] = "float32",
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """Evaluate a NumPy-backed raster algebra expression and write the result."""
    namespace: Dict[str, Any] = {}
    crs = None
    gt = None
    ref_nodata = nodata

    if inputs:
        for name, path in inputs.items():
            with open_raster_data(path) as dataset:
                data = np.asarray(dataset.read())
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                namespace[str(name)] = data.astype(np.float64, copy=False)
                if crs is None:
                    crs = dataset.crs
                    gt = dataset.geotransform
                    if ref_nodata is None:
                        ref_nodata = _band_nodata(dataset)
    elif source is not None:
        with open_raster_data(source) as dataset:
            data = np.asarray(dataset.read(), dtype=np.float64)
            crs = dataset.crs
            gt = dataset.geotransform
            if ref_nodata is None:
                ref_nodata = _band_nodata(dataset)
            if data.ndim == 2:
                namespace["A"] = data
                namespace["band1"] = data
            else:
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for index, plane in enumerate(data):
                    namespace[f"band{index + 1}"] = plane
                    if index < len(letters):
                        namespace[letters[index]] = plane
    else:
        raise RasterOperationError("Provide source= or inputs= for raster algebra.")

    result = _eval_expression(expression, namespace)
    result = np.asarray(result)
    if nodata is not None:
        result = np.where(np.isnan(result), nodata, result)
    return write_raster_data(
        destination,
        result.astype(_normalize_dtype(dtype)),
        crs=crs,
        geotransform=gt,
        nodata=nodata if nodata is not None else ref_nodata,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# blend / compare / clip / reclassify
# ---------------------------------------------------------------------------


def blend(
    top: PathLike,
    bottom: PathLike,
    destination: PathLike,
    *,
    opacity: float = 0.5,
    band: int = 1,
    overwrite: bool = False,
) -> str:
    """
    Blend / compose two aligned rasters (``gdal raster blend``).

    ``result = opacity * top + (1 - opacity) * bottom``. Inputs must share
    shape; the bottom raster supplies CRS / geotransform.
    """
    if not 0.0 <= float(opacity) <= 1.0:
        raise RasterOperationError("opacity must be between 0 and 1.")
    with open_raster_data(top) as top_ds, open_raster_data(bottom) as bottom_ds:
        a = np.asarray(top_ds.read(bands=band), dtype=np.float64)
        b = np.asarray(bottom_ds.read(bands=band), dtype=np.float64)
        if a.shape != b.shape:
            raise RasterOperationError(
                f"blend inputs must share shape; got {a.shape} and {b.shape}."
            )
        result = float(opacity) * a + (1.0 - float(opacity)) * b
        return write_raster_data(
            destination,
            result.astype(np.float32),
            crs=bottom_ds.crs,
            geotransform=bottom_ds.geotransform,
            nodata=_band_nodata(bottom_ds, band),
            overwrite=overwrite,
        )


def compare(
    left: PathLike,
    right: PathLike,
    *,
    band: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare two rasters (``gdal raster compare``).

    Returns a summary dict with shape/CRS checks, difference statistics, and
    whether values are approximately equal.
    """
    with open_raster_data(left) as a_ds, open_raster_data(right) as b_ds:
        a = np.asarray(a_ds.read(bands=band), dtype=np.float64)
        b = np.asarray(b_ds.read(bands=band), dtype=np.float64)
        shape_match = a.shape == b.shape
        crs_match = str(a_ds.crs) == str(b_ds.crs)
        gt_match = tuple(a_ds.geotransform) == tuple(b_ds.geotransform)
        summary: Dict[str, Any] = {
            "shape_match": shape_match,
            "crs_match": crs_match,
            "geotransform_match": gt_match,
            "left_shape": a.shape,
            "right_shape": b.shape,
            "equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rmse": None,
            "n_diff": None,
        }
        if not shape_match:
            return summary
        left_nodata = _band_nodata(a_ds, band)
        right_nodata = _band_nodata(b_ds, band)
        mask = np.ones(a.shape, dtype=bool)
        if left_nodata is not None:
            mask &= a != left_nodata
        if right_nodata is not None:
            mask &= b != right_nodata
        if not np.any(mask):
            summary["equal"] = True
            summary["max_abs_diff"] = 0.0
            summary["mean_abs_diff"] = 0.0
            summary["rmse"] = 0.0
            summary["n_diff"] = 0
            return summary
        diff = a[mask] - b[mask]
        abs_diff = np.abs(diff)
        summary["max_abs_diff"] = float(np.max(abs_diff))
        summary["mean_abs_diff"] = float(np.mean(abs_diff))
        summary["rmse"] = float(np.sqrt(np.mean(diff * diff)))
        close = np.isclose(a[mask], b[mask], rtol=rtol, atol=atol)
        summary["n_diff"] = int(np.size(close) - int(np.count_nonzero(close)))
        summary["equal"] = bool(np.all(close))
        return summary


def clip(
    source: PathLike,
    destination: PathLike,
    *,
    bounds: Optional[Sequence[float]] = None,
    cutline: Optional[PathLike] = None,
    crop_to_cutline: bool = True,
    src_win: Optional[Sequence[float]] = None,
    resample: str = "near",
    nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Clip a raster by projected bounds, pixel window, or vector cutline
    (``gdal raster clip``).

    ``bounds`` is ``(minx, miny, maxx, maxy)``. ``src_win`` is
    ``[left_x, top_y, width, height]`` in pixels.
    """
    if bounds is None and cutline is None and src_win is None:
        raise RasterOperationError("Provide bounds=, cutline=, or src_win= for clip.")

    if cutline is not None or (bounds is not None and src_win is None):
        from geolattice.raster.raster_CRS import warp

        warp_kwds: Dict[str, Any] = {
            "resample": resample,
            "creation_options": creation_options,
            "overwrite": overwrite,
        }
        if nodata is not None:
            warp_kwds["dstNodata"] = nodata
            warp_kwds["srcNodata"] = nodata
        if cutline is not None:
            warp_kwds["cutlineDSName"] = os.fspath(cutline)
            warp_kwds["cropToCutline"] = crop_to_cutline
        if bounds is not None:
            minx, miny, maxx, maxy = [float(v) for v in bounds]
            warp_kwds["outputBounds"] = [minx, miny, maxx, maxy]
        return str(warp(source, destination, **warp_kwds))

    return _translate(
        source,
        destination,
        srcWin=list(src_win) if src_win is not None else None,
        projWin=_bounds_to_proj_win(bounds) if bounds is not None else None,
        noData=nodata,
        creationOptions=_creation_options(creation_options) if creation_options else None,
        overwrite=overwrite,
    )


def reclassify(
    source: PathLike,
    destination: PathLike,
    mapping: Sequence[Tuple[Any, ...]],
    *,
    band: int = 1,
    default: Optional[float] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Reclassify raster values (``gdal raster reclassify``).

    ``mapping`` entries are either ``(value, new_value)`` for exact matches or
    ``(min, max, new_value)`` for half-open ranges ``[min, max)``.

    For histogram / quantile / cluster schemes matching R ``classIntervals``,
    use :func:`class_intervals` and :func:`reclassify_by`.
    """
    if not mapping:
        raise RasterOperationError("mapping must contain at least one rule.")
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        crs = dataset.crs
        gt = dataset.geotransform
        src_nodata = _band_nodata(dataset, band) if nodata is None else nodata

    out = np.full(array.shape, np.nan if default is None else float(default), dtype=np.float64)
    for rule in mapping:
        if len(rule) == 2:
            old, new = rule
            mask = array == old
            out[mask] = float(new)
        elif len(rule) == 3:
            lo, hi, new = rule
            mask = (array >= float(lo)) & (array < float(hi))
            out[mask] = float(new)
        else:
            raise RasterOperationError(
                f"Invalid reclassify rule {rule!r}; use (value, new) or (min, max, new)."
            )
    if src_nodata is not None:
        out[array == src_nodata] = src_nodata if nodata is None else nodata
    out_nodata = nodata if nodata is not None else src_nodata
    if out_nodata is not None:
        out = np.where(np.isnan(out), out_nodata, out)
    return write_raster_data(
        destination,
        out.astype(np.float32),
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def raster_histogram(
    source: PathLike,
    *,
    bins: Union[int, Sequence[float]] = 10,
    band: int = 1,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Histogram of raster values (R ``hist()``-style breaks and counts).

    Returns a dict with ``breaks``, ``counts``, ``mids``, ``n`` (valid pixels),
    and ``density`` (counts / n).
    """
    values = _valid_band_values(source, band=band)
    if values.size == 0:
        return {
            "breaks": [],
            "counts": [],
            "mids": [],
            "n": 0,
            "density": [],
        }
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    breaks = [float(v) for v in edges]
    count_list = [int(v) for v in counts]
    mids = [0.5 * (breaks[i] + breaks[i + 1]) for i in range(len(breaks) - 1)]
    n = int(values.size)
    density = [c / n if n else 0.0 for c in count_list]
    return {
        "breaks": breaks,
        "counts": count_list,
        "mids": mids,
        "n": n,
        "density": density,
    }


def class_intervals(
    source: PathLike,
    *,
    n: int = 5,
    style: str = "quantile",
    band: int = 1,
    fixed: Optional[Sequence[float]] = None,
    sample: Optional[int] = 10_000,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Classification break points for a raster band (R ``classIntervals``).

    Supported ``style`` values (classInt-compatible)::

        fixed, sd, equal, pretty, quantile, kmeans, hclust, bclust, fisher, jenks

    ``fixed`` requires ``fixed=`` break points (length >= 2). Returns ``n + 1``
    breaks including the data minimum and maximum (except for ``fixed`` /
    ``pretty``, which may differ).
    """
    style_key = str(style).strip().lower()
    if style_key not in _CLASSIFY_STYLES:
        known = ", ".join(sorted(_CLASSIFY_STYLES))
        raise RasterOperationError(f"Unknown classification style {style!r}. Known: {known}")
    if style_key != "fixed" and n < 2:
        raise RasterOperationError("n must be an integer >= 2.")
    values = _valid_band_values(source, band=band)
    if values.size == 0:
        raise RasterOperationError("No valid pixels available for classification.")
    return _classification_breaks(
        values,
        n=int(n),
        style=style_key,
        fixed=fixed,
        sample=sample,
        seed=seed,
    )


def reclassify_by(
    source: PathLike,
    destination: PathLike,
    *,
    n: int = 5,
    style: str = "quantile",
    band: int = 1,
    fixed: Optional[Sequence[float]] = None,
    start_value: int = 1,
    default: Optional[float] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
    sample: Optional[int] = 10_000,
    seed: Optional[int] = None,
) -> str:
    """
    Reclassify a raster into ``n`` classes using a histogram / quantile /
    cluster scheme (R ``classIntervals`` + ``classify``).

    Class labels are consecutive integers starting at ``start_value`` (default
    1). Ranges are half-open ``[lo, hi)`` except the last class, which is
    closed ``[lo, hi]``.
    """
    breaks = class_intervals(
        source,
        n=n,
        style=style,
        band=band,
        fixed=fixed,
        sample=sample,
        seed=seed,
    )
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        crs = dataset.crs
        gt = dataset.geotransform
        src_nodata = _band_nodata(dataset, band) if nodata is None else nodata

    out = _apply_class_breaks(
        array,
        breaks,
        start_value=int(start_value),
        default=default,
        src_nodata=src_nodata,
        out_nodata=nodata if nodata is not None else src_nodata,
    )
    out_nodata = nodata if nodata is not None else src_nodata
    return write_raster_data(
        destination,
        out.astype(np.float32),
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# resample / aggregate / disaggregate / mosaic / stack
# ---------------------------------------------------------------------------


def resample(
    source: PathLike,
    destination: PathLike,
    *,
    resolution: Optional[Union[float, Tuple[float, float]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    method: str = "bilinear",
    nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """Resample a raster to a new resolution or dimensions via GDAL Warp."""
    from geolattice.raster.raster_CRS import warp

    x_res = y_res = None
    if resolution is not None:
        if isinstance(resolution, (int, float)):
            x_res = y_res = float(resolution)
        else:
            x_res, y_res = float(resolution[0]), float(resolution[1])
    warp_kwds: Dict[str, Any] = {
        "resample": method,
        "xRes": x_res,
        "yRes": y_res,
        "creation_options": creation_options,
        "overwrite": overwrite,
    }
    if width is not None:
        warp_kwds["width"] = int(width)
    if height is not None:
        warp_kwds["height"] = int(height)
    if nodata is not None:
        warp_kwds["dstNodata"] = nodata
        warp_kwds["srcNodata"] = nodata
    return str(warp(source, destination, **warp_kwds))


def aggregate(
    source: PathLike,
    destination: PathLike,
    *,
    factor: int = 2,
    method: str = "average",
    nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Aggregate (coarsen) a raster by an integer factor.

    Cell size becomes ``factor`` times larger; ``method`` is a GDAL resampling
    algorithm such as ``average``, ``mode``, ``sum``, ``min``, ``max``.
    """
    if factor < 2:
        raise RasterOperationError("aggregate factor must be an integer >= 2.")
    with open_raster_data(source) as dataset:
        gt = dataset.geotransform
        x_res = abs(float(gt[1])) * int(factor)
        y_res = abs(float(gt[5])) * int(factor)
    return resample(
        source,
        destination,
        resolution=(x_res, y_res),
        method=method,
        nodata=nodata,
        creation_options=creation_options,
        overwrite=overwrite,
    )


def disaggregate(
    source: PathLike,
    destination: PathLike,
    *,
    factor: int = 2,
    method: str = "near",
    nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Disaggregate (refine) a raster by an integer factor.

    Cell size becomes ``1/factor`` of the original. Nearest-neighbour is the
    usual method for categorical data.
    """
    if factor < 2:
        raise RasterOperationError("disaggregate factor must be an integer >= 2.")
    with open_raster_data(source) as dataset:
        gt = dataset.geotransform
        x_res = abs(float(gt[1])) / int(factor)
        y_res = abs(float(gt[5])) / int(factor)
    return resample(
        source,
        destination,
        resolution=(x_res, y_res),
        method=method,
        nodata=nodata,
        creation_options=creation_options,
        overwrite=overwrite,
    )


def mosaic(
    sources: Sequence[PathLike],
    destination: PathLike,
    *,
    virtual: bool = False,
    resample: str = "near",
    src_nodata: Any = None,
    dst_nodata: Any = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Build a mosaic from multiple rasters (``gdal raster mosaic``).

    When ``virtual`` is true, writes a VRT. Otherwise materializes with
    ``gdal.Warp`` / Translate from a temporary VRT.
    """
    if not sources:
        raise RasterOperationError("mosaic requires at least one source raster.")
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)
    inputs = [os.fspath(path) for path in sources]

    if virtual or dest.suffix.lower() == ".vrt":
        options = gdal.BuildVRTOptions(
            resampleAlg=_normalize_resample(resample),
            srcNodata=src_nodata,
            VRTNodata=dst_nodata,
        )
        vrt = gdal.BuildVRT(os.fspath(dest), inputs, options=options)
        if vrt is None:
            raise RasterOperationError(f"BuildVRT failed for {dest}")
        vrt.FlushCache()
        vrt = None
        return str(dest)

    with tempfile.TemporaryDirectory(prefix="geolattice-mosaic-") as folder:
        vrt_path = Path(folder) / "mosaic.vrt"
        options = gdal.BuildVRTOptions(
            resampleAlg=_normalize_resample(resample),
            srcNodata=src_nodata,
            VRTNodata=dst_nodata,
        )
        vrt = gdal.BuildVRT(os.fspath(vrt_path), inputs, options=options)
        if vrt is None:
            raise RasterOperationError("BuildVRT failed while building mosaic.")
        vrt.FlushCache()
        vrt = None
        return _translate(
            vrt_path,
            dest,
            creationOptions=_creation_options(creation_options) if creation_options else None,
            noData=dst_nodata,
            overwrite=True,
        )


def stack(
    sources: Sequence[PathLike],
    destination: PathLike,
    *,
    separate: bool = True,
    virtual: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Combine input bands into a multi-band output (``gdal raster stack``).

    When ``separate`` is true (default), each input becomes one band. When
    ``virtual`` is true, writes a VRT.
    """
    if not sources:
        raise RasterOperationError("stack requires at least one source raster.")
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)
    inputs = [os.fspath(path) for path in sources]

    if virtual or dest.suffix.lower() == ".vrt":
        options = gdal.BuildVRTOptions(separate=separate)
        vrt = gdal.BuildVRT(os.fspath(dest), inputs, options=options)
        if vrt is None:
            raise RasterOperationError(f"BuildVRT failed for stack at {dest}")
        vrt.FlushCache()
        vrt = None
        return str(dest)

    planes: List[np.ndarray] = []
    crs = None
    gt = None
    nodata = None
    for path in sources:
        with open_raster_data(path) as dataset:
            data = np.asarray(dataset.read())
            if crs is None:
                crs = dataset.crs
                gt = dataset.geotransform
                nodata = _band_nodata(dataset)
            if data.ndim == 2:
                planes.append(data)
            else:
                for plane in data:
                    planes.append(plane)
    stacked = np.stack(planes, axis=0)
    return write_raster_data(
        dest,
        stacked,
        crs=crs,
        geotransform=gt,
        nodata=nodata,
        creation_options=creation_options,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# summarize / zonal-stats / polygonize / proximity / color-map / contour
# ---------------------------------------------------------------------------


def summarize_raster(
    source: PathLike,
    *,
    band: int = 1,
    percentiles: Sequence[float] = (2, 98),
) -> Dict[str, Any]:
    """Summarize raster values (min, max, mean, std, count, percentiles)."""
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        nodata = _band_nodata(dataset, band)
    mask = np.ones(array.shape, dtype=bool)
    if nodata is not None:
        mask &= array != nodata
    mask &= np.isfinite(array)
    values = array[mask]
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "sum": None,
            "percentiles": {},
        }
    pct = {
        f"p{int(p) if float(p).is_integer() else p}": float(np.percentile(values, p))
        for p in percentiles
    }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "sum": float(np.sum(values)),
        "percentiles": pct,
    }


def zonal_stats(
    source: PathLike,
    zones: PathLike,
    *,
    band: int = 1,
    zone_field: Optional[str] = None,
    stats: Sequence[str] = ("count", "min", "max", "mean", "sum"),
    all_touched: bool = False,
) -> List[Dict[str, Any]]:
    """
    Calculate raster zonal statistics (``gdal raster zonal-stats``).

    ``zones`` is a vector dataset. When ``zone_field`` is set, features are
    grouped by that attribute; otherwise each feature is its own zone
    (``zone_id`` = feature FID).
    """
    gdal, ogr = _gdal_ogr()
    with open_raster_data(source) as raster_ds:
        raster = np.asarray(raster_ds.read(bands=band), dtype=np.float64)
        nodata = _band_nodata(raster_ds, band)
        gt = raster_ds.geotransform
        width = raster_ds.width
        height = raster_ds.height
        projection = raster_ds.projection_wkt or ""

    zone_ds = ogr.Open(os.fspath(zones))
    if zone_ds is None:
        raise RasterOperationError(f"Could not open zones vector: {zones}")
    try:
        layer = zone_ds.GetLayer(0)
        if layer is None:
            raise RasterOperationError(f"No layer found in {zones}")

        mem_driver = gdal.GetDriverByName("MEM")
        zone_raster_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Int32)
        zone_raster_ds.SetGeoTransform(gt)
        if projection:
            zone_raster_ds.SetProjection(projection)
        zone_band = zone_raster_ds.GetRasterBand(1)
        zone_band.Fill(0)
        zone_band.SetNoDataValue(0)

        burn_options = ["ATTRIBUTE=" + zone_field] if zone_field else []
        if all_touched:
            burn_options.append("ALL_TOUCHED=TRUE")

        if zone_field:
            err = gdal.RasterizeLayer(
                zone_raster_ds,
                [1],
                layer,
                options=burn_options,
            )
            if err != 0:
                raise RasterOperationError(f"RasterizeLayer failed with status {err}")
            zone_ids = _read_band_array(zone_band, width, height)
            labels = sorted({int(v) for v in np.unique(zone_ids) if int(v) != 0})
            results: List[Dict[str, Any]] = []
            for label in labels:
                results.append(
                    _zone_summary(
                        raster,
                        zone_ids == label,
                        nodata,
                        stats,
                        zone_id=label,
                        zone_field=zone_field,
                    )
                )
            return results

        # Per-feature FID burn via temporary one-feature memory layer.
        results = []
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature is not None:
            fid = int(feature.GetFID())
            zone_band.Fill(0)
            mem_vec = ogr.GetDriverByName("Memory").CreateDataSource("zone")
            mem_layer = mem_vec.CreateLayer("zone", layer.GetSpatialRef(), layer.GetGeomType())
            defn = mem_layer.GetLayerDefn()
            out_feat = ogr.Feature(defn)
            out_feat.SetGeometry(feature.GetGeometryRef().Clone())
            mem_layer.CreateFeature(out_feat)
            options = ["ALL_TOUCHED=TRUE"] if all_touched else []
            err = gdal.RasterizeLayer(
                zone_raster_ds, [1], mem_layer, burn_values=[1], options=options
            )
            if err != 0:
                raise RasterOperationError(f"RasterizeLayer failed for FID {fid}")
            mask = _read_band_array(zone_band, width, height) == 1
            row = _zone_summary(raster, mask, nodata, stats, zone_id=fid, zone_field="fid")
            results.append(row)
            feature = layer.GetNextFeature()
            mem_vec = None
        return results
    finally:
        zone_ds = None


def rasterize(
    vectors: PathLike,
    destination: PathLike,
    *,
    attribute: Optional[str] = None,
    burn_value: float = 1.0,
    template: Optional[PathLike] = None,
    resolution: Optional[Union[float, Sequence[float]]] = None,
    bounds: Optional[Sequence[float]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    crs: Any = None,
    dtype: Union[str, np.dtype] = "float32",
    nodata: Optional[float] = None,
    all_touched: bool = False,
    where: Optional[str] = None,
    overwrite: bool = False,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> str:
    """
    Burn vector features into a raster grid (R ``raster::rasterize`` / GDAL Rasterize).

    Supply either:

    * ``template`` — copy resolution / CRS from an existing raster (extent defaults
      to the vector layer, or ``bounds`` when given); or
    * ``resolution`` with optional ``bounds``; or
    * ``width`` and ``height`` with ``bounds``.

    When ``attribute`` is set, feature field values are burned; otherwise
    ``burn_value`` is used for all geometries.
    """
    gdal, ogr = _gdal_ogr()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    vec_ds = ogr.Open(os.fspath(vectors))
    if vec_ds is None:
        raise RasterOperationError(f"Could not open vector dataset: {vectors}")
    try:
        layer = vec_ds.GetLayer(0)
        if layer is None:
            raise RasterOperationError(f"No layer found in {vectors}")
        if where:
            layer.SetAttributeFilter(str(where))

        if attribute:
            defn = layer.GetLayerDefn()
            field_names = {
                defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())
            }
            if attribute not in field_names:
                raise RasterOperationError(
                    f"Attribute {attribute!r} not found in {vectors}. "
                    f"Available: {sorted(field_names)}"
                )

        minx, maxx, miny, maxy = layer.GetExtent(True)
        if bounds is not None:
            if len(bounds) != 4:
                raise RasterOperationError("bounds must be (minx, miny, maxx, maxy).")
            minx, miny, maxx, maxy = (float(v) for v in bounds)
        if not (maxx > minx and maxy > miny):
            raise RasterOperationError("Rasterize extent is empty or invalid.")

        projection = ""
        xres: Optional[float] = None
        yres: Optional[float] = None
        out_width = width
        out_height = height

        if template is not None:
            with open_raster_data(template) as tmpl:
                gt = tmpl.geotransform
                xres = abs(float(gt[1]))
                yres = abs(float(gt[5]))
                projection = tmpl.projection_wkt or ""
                if crs is None:
                    crs = tmpl.crs
        elif resolution is not None:
            if isinstance(resolution, (int, float)):
                xres = yres = abs(float(resolution))
            else:
                if len(resolution) != 2:
                    raise RasterOperationError("resolution must be a number or (xres, yres).")
                xres = abs(float(resolution[0]))
                yres = abs(float(resolution[1]))
        elif width is None or height is None:
            raise RasterOperationError(
                "Provide template=, resolution=, or both width= and height= (with bounds)."
            )

        if xres is not None and yres is not None:
            if out_width is None:
                out_width = max(1, int(math.ceil((maxx - minx) / xres)))
            if out_height is None:
                out_height = max(1, int(math.ceil((maxy - miny) / yres)))
            # Snap max extent to pixel grid so geotransform matches size.
            maxx = minx + out_width * xres
            maxy = miny + out_height * yres
        else:
            if out_width is None or out_height is None:
                raise RasterOperationError("width and height are required without resolution.")
            xres = (maxx - minx) / float(out_width)
            yres = (maxy - miny) / float(out_height)

        if not projection:
            srs = layer.GetSpatialRef()
            if srs is not None:
                projection = srs.ExportToWkt()
        if crs is not None:
            from geolattice.raster.raster_CRS import crs_to_gdal

            projection = crs_to_gdal(crs)
        crs_for_write: Any = crs if crs is not None else (projection or None)

        fill = float(nodata) if nodata is not None else 0.0
        geotransform = (minx, xres, 0.0, maxy, 0.0, -yres)
        create(
            dest,
            width=int(out_width),
            height=int(out_height),
            bands=1,
            dtype=dtype,
            crs=crs_for_write,
            geotransform=geotransform,
            nodata=nodata,
            fill=fill,
            creation_options=creation_options,
            overwrite=True,
        )

        out_ds = gdal.Open(os.fspath(dest), gdal.GA_Update)
        if out_ds is None:
            raise RasterOperationError(f"Could not open raster for update: {dest}")
        try:
            options = []
            if attribute:
                options.append("ATTRIBUTE=" + attribute)
            if all_touched:
                options.append("ALL_TOUCHED=TRUE")
            kwargs: Dict[str, Any] = {"options": options}
            if not attribute:
                kwargs["burn_values"] = [float(burn_value)]
            err = gdal.RasterizeLayer(out_ds, [1], layer, **kwargs)
            if err != 0:
                raise RasterOperationError(f"RasterizeLayer failed with status {err}")
        finally:
            out_ds = None
    finally:
        vec_ds = None
    return str(dest)


def _read_band_array(band: Any, width: int, height: int) -> np.ndarray:
    """Read a GDAL band into NumPy, with ReadRaster fallback when gdal_array is broken."""
    try:
        array = band.ReadAsArray()
        if array is not None:
            return np.asarray(array)
    except Exception:
        pass
    gdal = _gdal()
    dtype = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }.get(band.DataType, np.float64)
    raw = band.ReadRaster(0, 0, width, height)
    return np.frombuffer(raw, dtype=dtype).reshape(height, width)


def polygonize(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    field_name: str = "value",
    connectedness: int = 4,
    mask_band: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """
    Create polygons from a raster band (``gdal raster polygonize``).
    """
    gdal, ogr = _gdal_ogr()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    src_ds = gdal.Open(os.fspath(source), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RasterOperationError(f"Could not open raster: {source}")
    try:
        src_band = src_ds.GetRasterBand(band)
        mask = src_ds.GetRasterBand(mask_band) if mask_band is not None else None
        driver_name = _vector_driver_from_path(dest)
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise RasterOperationError(f"Vector driver unavailable: {driver_name}")
        if dest.exists():
            driver.DeleteDataSource(os.fspath(dest))
        out_ds = driver.CreateDataSource(os.fspath(dest))
        if out_ds is None:
            raise RasterOperationError(f"Could not create vector dataset: {dest}")
        srs = None
        wkt = src_ds.GetProjection()
        if wkt:
            from osgeo import osr

            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
        layer = out_ds.CreateLayer(dest.stem, srs=srs, geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger64))
        options = []
        if int(connectedness) == 8:
            options.append("8CONNECTED=8")
        status = gdal.Polygonize(src_band, mask, layer, 0, options)
        if status != 0:
            raise RasterOperationError(f"Polygonize failed with status {status}")
        out_ds = None
    finally:
        src_ds = None
    return str(dest)


def proximity(
    source: PathLike,
    destination: PathLike,
    *,
    band: int = 1,
    target_values: Optional[Sequence[float]] = None,
    max_distance: Optional[float] = None,
    dist_units: str = "GEO",
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Produce a raster proximity map (``gdal raster proximity``).
    """
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    src_ds = gdal.Open(os.fspath(source), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RasterOperationError(f"Could not open raster: {source}")
    try:
        driver = gdal.GetDriverByName(_format_from_path(dest))
        out_ds = driver.Create(
            os.fspath(dest),
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,
            gdal.GDT_Float32,
        )
        if out_ds is None:
            raise RasterOperationError(f"Could not create proximity raster: {dest}")
        out_ds.SetGeoTransform(src_ds.GetGeoTransform())
        out_ds.SetProjection(src_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(float(nodata))
        options = [f"DISTUNITS={dist_units}"]
        if target_values is not None:
            options.append("VALUES=" + ",".join(str(v) for v in target_values))
        if max_distance is not None:
            options.append(f"MAXDIST={float(max_distance)}")
        if nodata is not None:
            options.append(f"NODATA={float(nodata)}")
        status = gdal.ComputeProximity(src_ds.GetRasterBand(band), out_band, options=options)
        if status != 0:
            raise RasterOperationError(f"ComputeProximity failed with status {status}")
        out_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None
    finally:
        src_ds = None
    return str(dest)


def color_map(
    source: PathLike,
    destination: PathLike,
    color_filename: PathLike,
    *,
    band: int = 1,
    add_alpha: bool = False,
    color_selection: str = "linear_interpolation",
    overwrite: bool = False,
) -> str:
    """
    Generate an RGB/RGBA dataset from a single band using a color map file
    (``gdal raster color-map`` / ``gdaldem color-relief``).

    ``color_filename`` is a gdaldem color-relief text file.
    ``color_selection`` is one of ``linear_interpolation``,
    ``nearest_color_entry``, or ``exact_color_entry`` (``interpolate`` is
    accepted as an alias of ``linear_interpolation``).
    """
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)
    selection = str(color_selection).strip().lower()
    if selection in {"interpolate", "linear", "linear_interpolation"}:
        selection = "linear_interpolation"
    elif selection in {"nearest", "nearest_color_entry"}:
        selection = "nearest_color_entry"
    elif selection in {"exact", "exact_color_entry"}:
        selection = "exact_color_entry"
    else:
        raise RasterOperationError(
            "color_selection must be linear_interpolation, nearest_color_entry, "
            f"or exact_color_entry; got {color_selection!r}."
        )
    options = gdal.DEMProcessingOptions(
        colorFilename=os.fspath(color_filename),
        band=band,
        addAlpha=add_alpha,
        colorSelection=selection,
    )
    result = gdal.DEMProcessing(
        os.fspath(dest),
        os.fspath(source),
        "color-relief",
        options=options,
    )
    if result is None:
        raise RasterOperationError(f"color-map / DEMProcessing failed for {dest}")
    result.FlushCache()
    result = None
    return str(dest)


def contour(
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
    Create vector contours from a DEM (``gdal raster contour``).

    Provide either ``interval`` or ``fixed_levels``.
    """
    if interval is None and not fixed_levels:
        raise RasterOperationError("Provide interval= or fixed_levels= for contour.")
    gdal, ogr = _gdal_ogr()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)

    src_ds = gdal.Open(os.fspath(source), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RasterOperationError(f"Could not open raster: {source}")
    try:
        src_band = src_ds.GetRasterBand(band)
        driver_name = _vector_driver_from_path(dest)
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise RasterOperationError(f"Vector driver unavailable: {driver_name}")
        if dest.exists():
            driver.DeleteDataSource(os.fspath(dest))
        out_ds = driver.CreateDataSource(os.fspath(dest))
        srs = None
        wkt = src_ds.GetProjection()
        if wkt:
            from osgeo import osr

            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
        layer = out_ds.CreateLayer(dest.stem, srs=srs, geom_type=ogr.wkbLineString)
        layer.CreateField(ogr.FieldDefn(id_field, ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn(elev_field, ogr.OFTReal))

        options = [f"ID_FIELD=0", f"ELEV_FIELD=1"]
        if interval is not None:
            options.append(f"LEVEL_INTERVAL={float(interval)}")
            options.append(f"LEVEL_BASE={float(base)}")
        if fixed_levels:
            options.append(
                "FIXED_LEVELS=" + ",".join(str(float(level)) for level in fixed_levels)
            )
        nodata = src_band.GetNoDataValue()
        if nodata is not None:
            options.append(f"NODATA={float(nodata)}")
        status = gdal.ContourGenerateEx(src_band, layer, options=options)
        if status != 0:
            raise RasterOperationError(f"ContourGenerateEx failed with status {status}")
        out_ds = None
    finally:
        src_ds = None
    return str(dest)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gdal() -> Any:
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RasterOperationError(
            "GDAL Python bindings are required for basic raster operations. "
            "Install the gdal package that matches the GDAL library on this machine."
        ) from exc
    gdal._UseExceptions()
    return gdal


def _gdal_ogr() -> Tuple[Any, Any]:
    gdal = _gdal()
    try:
        from osgeo import ogr
    except ImportError as exc:
        raise RasterOperationError("GDAL OGR bindings are required for this operation.") from exc
    try:
        ogr._UseExceptions()
    except AttributeError:
        ogr.UseExceptions()
    return gdal, ogr


def _band_nodata(dataset: RasterDataset, band: int = 1) -> Optional[float]:
    return dataset.get_band(band).nodata


def _normalize_dtype(dtype: Union[str, np.dtype]) -> str:
    key = str(np.dtype(dtype).name if not isinstance(dtype, str) else dtype).strip().lower()
    if key not in _DTYPE_ALIASES:
        raise RasterOperationError(f"Unsupported dtype {dtype!r}.")
    return _DTYPE_ALIASES[key]


def _gdal_dtype(dtype: Optional[Union[str, np.dtype]]) -> Optional[int]:
    if dtype is None:
        return None
    gdal = _gdal()
    name = _NUMPY_TO_GDAL_NAME[_normalize_dtype(dtype)]
    value = gdal.GetDataTypeByName(name)
    if value is None or value == 0:
        # Int8 may be missing on older GDAL; fall back.
        if name == "Int8":
            return gdal.GDT_Byte
        raise RasterOperationError(f"GDAL does not support dtype {dtype!r}.")
    return value


def _normalize_resample(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _RESAMPLE_ALIASES:
        known = ", ".join(sorted(_RESAMPLE_ALIASES))
        raise RasterOperationError(f"Unknown resample algorithm {value!r}. Known: {known}")
    return _RESAMPLE_ALIASES[key]


def _creation_options(
    options: Optional[Union[Sequence[str], Mapping[str, Any]]],
) -> Optional[List[str]]:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return [f"{key}={value}" for key, value in options.items()]
    return [str(item) for item in options]


def _format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".tif": "GTiff",
        ".tiff": "GTiff",
        ".img": "HFA",
        ".vrt": "VRT",
        ".asc": "AAIGrid",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
    }
    return mapping.get(suffix, "GTiff")


def _vector_driver_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".gpkg": "GPKG",
        ".shp": "ESRI Shapefile",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".gml": "GML",
    }
    return mapping.get(suffix, "GPKG")


def _prepare_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {path}. Pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        try:
            gdal = _gdal()
            gdal.Unlink(os.fspath(path))
        except Exception:
            path.unlink(missing_ok=True)


def _broadcast_band_values(
    value: Optional[Union[Any, Sequence[Any]]],
    band_count: int,
    name: str,
) -> List[Any]:
    if value is None:
        return [None] * band_count
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value] * band_count
    values = list(value)
    if len(values) == 1:
        return values * band_count
    if len(values) != band_count:
        raise RasterOperationError(
            f"{name} must have length 1 or {band_count}, got {len(values)}."
        )
    return values


def _bounds_to_proj_win(bounds: Sequence[float]) -> List[float]:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    # Translate projWin is ulx, uly, lrx, lry
    return [minx, maxy, maxx, miny]


def _translate(
    source: PathLike,
    destination: PathLike,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    gdal = _gdal()
    dest = Path(destination)
    _prepare_destination(dest, overwrite=overwrite)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if "format" not in kwargs:
        kwargs["format"] = _format_from_path(dest)
    result = gdal.Translate(os.fspath(dest), os.fspath(source), **kwargs)
    if result is None:
        raise RasterOperationError(f"gdal.Translate failed for {dest}")
    result.FlushCache()
    result = None
    return str(dest)


def _focal_kernel(size: int, *, circular: bool) -> np.ndarray:
    kernel = np.ones((size, size), dtype=bool)
    if circular:
        radius = size // 2
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        kernel = (xx * xx + yy * yy) <= (radius * radius + 1e-9)
    return kernel


def _apply_focal(array: np.ndarray, kernel: np.ndarray, statistic: str) -> np.ndarray:
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError as exc:  # pragma: no cover
        raise RasterOperationError("numpy>=1.20 is required for focal statistics.") from exc

    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, kernel.shape)
    # windows shape: (H, W, ky, kx)
    masked = np.where(kernel, windows, np.nan)
    flat = masked.reshape(masked.shape[0], masked.shape[1], -1)

    if statistic == "mean":
        return np.nanmean(flat, axis=2)
    if statistic == "sum":
        return np.nansum(flat, axis=2)
    if statistic == "min":
        return np.nanmin(flat, axis=2)
    if statistic == "max":
        return np.nanmax(flat, axis=2)
    if statistic == "median":
        return np.nanmedian(flat, axis=2)
    if statistic == "std":
        return np.nanstd(flat, axis=2)
    if statistic == "var":
        return np.nanvar(flat, axis=2)
    if statistic == "range":
        return np.nanmax(flat, axis=2) - np.nanmin(flat, axis=2)
    if statistic == "count":
        return np.sum(np.isfinite(flat), axis=2).astype(np.float64)
    if statistic == "majority":
        # Mode over finite values; ties pick the smallest value.
        out = np.full(array.shape, np.nan, dtype=np.float64)
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                vals = flat[row, col]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                values, counts = np.unique(vals, return_counts=True)
                out[row, col] = float(values[np.argmax(counts)])
        return out
    raise RasterOperationError(f"Unhandled focal statistic {statistic!r}.")


def _eval_expression(expression: str, namespace: Mapping[str, Any]) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise RasterOperationError(f"Invalid raster expression: {exc}") from exc
    return _eval_node(tree.body, namespace)


def _eval_node(node: ast.AST, namespace: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise RasterOperationError(f"Unknown name in expression: {node.id}")
        return namespace[node.id]
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_BINOPS:
            raise RasterOperationError(f"Operator not allowed: {op_type.__name__}")
        return _SAFE_BINOPS[op_type](_eval_node(node.left, namespace), _eval_node(node.right, namespace))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_UNARY:
            raise RasterOperationError(f"Unary operator not allowed: {op_type.__name__}")
        return _SAFE_UNARY[op_type](_eval_node(node.operand, namespace))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, namespace)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_CMP:
                raise RasterOperationError(f"Comparison not allowed: {op_type.__name__}")
            right = _eval_node(comparator, namespace)
            result = np.logical_and(result, _SAFE_CMP[op_type](left, right))
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(value, namespace) for value in node.values]
        if isinstance(node.op, ast.And):
            out = values[0]
            for value in values[1:]:
                out = np.logical_and(out, value)
            return out
        if isinstance(node.op, ast.Or):
            out = values[0]
            for value in values[1:]:
                out = np.logical_or(out, value)
            return out
    if isinstance(node, ast.IfExp):
        return np.where(
            _eval_node(node.test, namespace),
            _eval_node(node.body, namespace),
            _eval_node(node.orelse, namespace),
        )
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise RasterOperationError("Only approved NumPy helpers are allowed in expressions.")
        args = [_eval_node(arg, namespace) for arg in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, namespace) for kw in node.keywords if kw.arg}
        return _SAFE_FUNCS[node.func.id](*args, **kwargs)
    raise RasterOperationError(f"Unsupported expression node: {type(node).__name__}")


def _zone_summary(
    raster: np.ndarray,
    mask: np.ndarray,
    nodata: Optional[float],
    stats: Sequence[str],
    *,
    zone_id: Any,
    zone_field: str,
) -> Dict[str, Any]:
    valid = mask.copy()
    if nodata is not None:
        valid &= raster != nodata
    valid &= np.isfinite(raster)
    values = raster[valid]
    row: Dict[str, Any] = {zone_field: zone_id}
    for name in stats:
        key = str(name).strip().lower()
        if key == "count":
            row["count"] = int(values.size)
        elif key == "min":
            row["min"] = float(np.min(values)) if values.size else None
        elif key == "max":
            row["max"] = float(np.max(values)) if values.size else None
        elif key == "mean":
            row["mean"] = float(np.mean(values)) if values.size else None
        elif key == "sum":
            row["sum"] = float(np.sum(values)) if values.size else None
        elif key == "std":
            row["std"] = float(np.std(values)) if values.size else None
        elif key == "median":
            row["median"] = float(np.median(values)) if values.size else None
        else:
            raise RasterOperationError(f"Unknown zonal statistic {name!r}.")
    return row


def _valid_band_values(source: PathLike, *, band: int = 1) -> np.ndarray:
    """Return finite, non-nodata values from one raster band as a 1-D array."""
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        nodata = _band_nodata(dataset, band)
    mask = np.isfinite(array)
    if nodata is not None:
        mask &= array != nodata
    return array[mask]


def _sample_values(
    values: np.ndarray,
    sample: Optional[int],
    seed: Optional[int],
) -> np.ndarray:
    """Optionally subsample values for expensive classifiers."""
    if sample is None or sample <= 0 or values.size <= sample:
        return values
    rng = np.random.default_rng(seed)
    index = rng.choice(values.size, size=int(sample), replace=False)
    return values[index]


def _unique_sorted_breaks(breaks: Sequence[float]) -> List[float]:
    cleaned = sorted({float(v) for v in breaks if np.isfinite(v)})
    if len(cleaned) < 2:
        raise RasterOperationError("Classification requires at least two finite break points.")
    return cleaned


def _pretty_breaks(vmin: float, vmax: float, n: int) -> List[float]:
    """Approximate R ``pretty()`` / mapclassify PrettyBreaks."""
    try:
        from mapclassify import PrettyBreaks

        dummy = np.linspace(vmin, vmax, max(n * 20, 50))
        classifier = PrettyBreaks(dummy, k=n)
        bins = [float(vmin)] + [float(v) for v in classifier.bins]
        return _unique_sorted_breaks(bins)
    except Exception:
        pass
    if vmax <= vmin:
        return [float(vmin), float(vmax if vmax > vmin else vmin + 1.0)]
    raw = np.linspace(vmin, vmax, n + 1)
    span = vmax - vmin
    step = span / n
    magnitude = 10 ** math.floor(math.log10(step)) if step > 0 else 1.0
    nice_steps = np.array([1.0, 2.0, 2.5, 5.0, 10.0]) * magnitude
    nice = float(nice_steps[np.argmin(np.abs(nice_steps - step))])
    start = math.floor(vmin / nice) * nice
    stop = math.ceil(vmax / nice) * nice
    breaks = list(np.arange(start, stop + nice * 0.5, nice))
    if breaks[0] > vmin:
        breaks.insert(0, float(vmin))
    if breaks[-1] < vmax:
        breaks.append(float(vmax))
    return _unique_sorted_breaks(breaks)


def _sd_breaks(values: np.ndarray, n: int) -> List[float]:
    """Standard-deviation classes around the mean (classInt ``sd``)."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if std == 0.0 or not np.isfinite(std):
        return [vmin, vmax]
    # Build symmetric sd cuts, then trim / pad to n classes.
    half = max(n // 2, 1)
    cuts = [mean + k * std for k in range(-half, half + 1)]
    breaks = _unique_sorted_breaks([vmin, vmax, *cuts])
    if len(breaks) - 1 == n:
        return breaks
    # Fall back to equal width over the sd-expanded range when cut count differs.
    lo = min(vmin, mean - half * std)
    hi = max(vmax, mean + half * std)
    return _unique_sorted_breaks(np.linspace(lo, hi, n + 1))


def _kmeans_breaks(values: np.ndarray, n: int, seed: Optional[int]) -> List[float]:
    from scipy.cluster.vq import kmeans2, whiten

    sample = values.astype(np.float64, copy=False)
    if sample.size < n:
        return _unique_sorted_breaks([float(np.min(values)), float(np.max(values))])
    data = whiten(sample.reshape(-1, 1))
    rng = int(seed) if seed is not None else 0
    try:
        centroids, labels = kmeans2(data, n, minit="points", seed=rng)
    except TypeError:
        # Older SciPy without seed=
        np.random.seed(rng)
        centroids, labels = kmeans2(data, n, minit="points")
    # Map whitened centroids back via class means in original units.
    centers = []
    for index in range(n):
        members = sample[labels == index]
        if members.size:
            centers.append(float(np.mean(members)))
    centers = sorted(centers)
    if len(centers) < 2:
        return _unique_sorted_breaks([float(np.min(values)), float(np.max(values))])
    edges = [float(np.min(values))]
    for left, right in zip(centers[:-1], centers[1:]):
        edges.append(0.5 * (left + right))
    edges.append(float(np.max(values)))
    return _unique_sorted_breaks(edges)


def _hclust_breaks(values: np.ndarray, n: int) -> List[float]:
    from scipy.cluster.hierarchy import fcluster, linkage

    sample = np.sort(values.astype(np.float64, copy=False))
    if sample.size < n:
        return _unique_sorted_breaks([float(sample[0]), float(sample[-1])])
    # Cap hierarchical clustering cost.
    if sample.size > 2_000:
        idx = np.linspace(0, sample.size - 1, 2_000).astype(int)
        sample = sample[idx]
    z = linkage(sample.reshape(-1, 1), method="ward")
    labels = fcluster(z, t=n, criterion="maxclust")
    centers = []
    for index in range(1, n + 1):
        members = sample[labels == index]
        if members.size:
            centers.append(float(np.mean(members)))
    centers = sorted(centers)
    edges = [float(np.min(values))]
    for left, right in zip(centers[:-1], centers[1:]):
        edges.append(0.5 * (left + right))
    edges.append(float(np.max(values)))
    return _unique_sorted_breaks(edges)


def _bclust_breaks(values: np.ndarray, n: int, seed: Optional[int]) -> List[float]:
    """Bagged k-means consensus breaks (simplified classInt ``bclust``)."""
    rng = np.random.default_rng(seed)
    sample = values.astype(np.float64, copy=False)
    if sample.size < n:
        return _unique_sorted_breaks([float(np.min(values)), float(np.max(values))])
    bag = min(20, max(5, sample.size // 50))
    collected: List[float] = []
    for bag_index in range(bag):
        draw = rng.choice(sample, size=min(sample.size, 2_000), replace=True)
        breaks = _kmeans_breaks(draw, n, seed=None if seed is None else int(seed) + bag_index)
        collected.extend(breaks[1:-1])
    if not collected:
        return _unique_sorted_breaks([float(np.min(values)), float(np.max(values))])
    inner = np.quantile(collected, np.linspace(0, 1, max(n - 1, 1)))
    return _unique_sorted_breaks(
        [float(np.min(values)), *[float(v) for v in np.atleast_1d(inner)], float(np.max(values))]
    )


def _mapclassify_breaks(values: np.ndarray, n: int, style: str) -> List[float]:
    try:
        import mapclassify
    except ImportError as exc:
        raise RasterOperationError(
            f"style={style!r} requires mapclassify. Install with: pip install mapclassify"
        ) from exc
    classifiers = {
        "fisher": getattr(mapclassify, "FisherJenks", None),
        "jenks": getattr(mapclassify, "JenksCaspall", None)
        or getattr(mapclassify, "NaturalBreaks", None),
        "pretty": getattr(mapclassify, "PrettyBreaks", None),
    }
    cls = classifiers.get(style)
    if cls is None:
        raise RasterOperationError(f"mapclassify does not provide a classifier for style={style!r}.")
    classifier = cls(values, k=n)
    bins = [float(np.min(values))] + [float(v) for v in classifier.bins]
    return _unique_sorted_breaks(bins)


def _classification_breaks(
    values: np.ndarray,
    *,
    n: int,
    style: str,
    fixed: Optional[Sequence[float]] = None,
    sample: Optional[int] = 10_000,
    seed: Optional[int] = None,
) -> List[float]:
    """Compute class breaks for 1-D values using a classInt-compatible style."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RasterOperationError("No finite values for classification.")
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if style == "fixed":
        if fixed is None or len(fixed) < 2:
            raise RasterOperationError("style='fixed' requires fixed= with at least two breaks.")
        return _unique_sorted_breaks(fixed)

    sampled = _sample_values(values, sample, seed)

    if style == "equal":
        return _unique_sorted_breaks(np.linspace(vmin, vmax, n + 1))
    if style == "quantile":
        probs = np.linspace(0, 1, n + 1)
        return _unique_sorted_breaks(np.quantile(sampled, probs))
    if style == "pretty":
        return _pretty_breaks(vmin, vmax, n)
    if style == "sd":
        return _sd_breaks(sampled, n)
    if style == "kmeans":
        return _kmeans_breaks(sampled, n, seed)
    if style == "hclust":
        return _hclust_breaks(sampled, n)
    if style == "bclust":
        return _bclust_breaks(sampled, n, seed)
    if style in {"fisher", "jenks"}:
        try:
            return _mapclassify_breaks(sampled, n, style)
        except RasterOperationError:
            # NumPy fallback: equal-interval when mapclassify is unavailable.
            if style == "fisher":
                # Lightweight Fisher-Jenks via dynamic programming for modest samples.
                return _fisher_jenks_breaks(sampled, n)
            return _unique_sorted_breaks(np.linspace(vmin, vmax, n + 1))
    raise RasterOperationError(f"Unhandled classification style {style!r}.")


def _fisher_jenks_breaks(values: np.ndarray, n: int) -> List[float]:
    """Small-sample Fisher-Jenks natural breaks (dynamic programming)."""
    data = np.sort(np.unique(values.astype(np.float64, copy=False)))
    m = data.size
    if m <= n:
        return _unique_sorted_breaks([float(data[0]), float(data[-1])])
    # Cap DP cost.
    if m > 3_000:
        idx = np.linspace(0, m - 1, 3_000).astype(int)
        data = data[idx]
        m = data.size

    lower = np.zeros((m + 1, n + 1), dtype=np.float64)
    variance = np.zeros((m + 1, n + 1), dtype=np.float64)
    for i in range(1, m + 1):
        lower[i, 1] = 1
        variance[i, 1] = np.var(data[:i]) * i if i > 1 else 0.0

    for k in range(2, n + 1):
        for i in range(k, m + 1):
            best = np.inf
            best_j = k
            for j in range(k, i + 1):
                # variance of data[j-1:i]
                segment = data[j - 1 : i]
                seg_var = float(np.var(segment) * segment.size) if segment.size > 1 else 0.0
                candidate = variance[j - 1, k - 1] + seg_var
                if candidate < best:
                    best = candidate
                    best_j = j
            variance[i, k] = best
            lower[i, k] = best_j

    breaks = [float(data[-1])]
    k = n
    i = m
    while k > 1:
        j = int(lower[i, k])
        breaks.append(float(data[j - 1]))
        i = j - 1
        k -= 1
    breaks.append(float(data[0]))
    return _unique_sorted_breaks(list(reversed(breaks)))


def _apply_class_breaks(
    array: np.ndarray,
    breaks: Sequence[float],
    *,
    start_value: int = 1,
    default: Optional[float] = None,
    src_nodata: Optional[float] = None,
    out_nodata: Optional[float] = None,
) -> np.ndarray:
    """Map continuous values to integer classes using ordered break points."""
    edges = _unique_sorted_breaks(breaks)
    out = np.full(array.shape, np.nan if default is None else float(default), dtype=np.float64)
    for index in range(len(edges) - 1):
        lo = edges[index]
        hi = edges[index + 1]
        label = float(start_value + index)
        if index == len(edges) - 2:
            mask = (array >= lo) & (array <= hi)
        else:
            mask = (array >= lo) & (array < hi)
        out[mask] = label
    if src_nodata is not None:
        out[array == src_nodata] = out_nodata if out_nodata is not None else src_nodata
    if out_nodata is not None:
        out = np.where(np.isnan(out), out_nodata, out)
    return out


__all__ = [
    "RasterOperationError",
    "create",
    "edit",
    "fill_nodata",
    "unscale",
    "neighbors",
    "neighbours",
    "focal_statistics",
    "calc",
    "raster_algebra",
    "blend",
    "compare",
    "clip",
    "reclassify",
    "raster_histogram",
    "class_intervals",
    "reclassify_by",
    "aggregate",
    "disaggregate",
    "resample",
    "mosaic",
    "stack",
    "summarize_raster",
    "zonal_stats",
    "polygonize",
    "proximity",
    "resize",
    "convert",
    "color_map",
    "contour",
]
