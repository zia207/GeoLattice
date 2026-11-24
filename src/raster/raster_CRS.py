"""
Raster CRS helpers: define, validate, datum choice, and GDAL warp / reproject.

Shared CRS definitions live in :mod:`geolattice.crs`. This module is the
raster-specific side:

* read and validate the CRS of an open dataset or file
* choose a datum transformation when several paths exist
* reproject / warp rasters with :func:`gdal.Warp` and :func:`gdal.WarpOptions`

Vector geometry reprojection lives in :mod:`geolattice.vector.vector_CRS`.

Example:
    from geolattice.raster.raster_CRS import reproject_raster, warp

    reproject_raster("dem.tif", "dem_utm.tif", dst_crs=32618, resample="bilinear")
    warp("src.tif", "out.tif", dstSRS="EPSG:3857", xRes=30, yRes=30)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from geolattice.crs import (
    CRS,
    CRSSettings,
    DatumTransformation,
    DatumTransformationDialog,
    DatumTransformationRequired,
    crs_settings,
)
from geolattice.raster.raster_read_write import (
    RasterDataset,
    RasterOpenError,
    RasterWriteError,
    open_raster_data,
    write_raster_data,
)

CRSLike = Union[CRS, str, int, None]
PathLike = Union[str, Path]
ResampleAlg = Union[str, int]

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


class RasterCRSError(ValueError):
    """Raised when a raster CRS cannot be defined, validated, or warped."""


def _gdal() -> Any:
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RasterCRSError(
            "GDAL Python bindings are required for raster CRS / warp. "
            "Install the gdal package that matches the GDAL library on this machine."
        ) from exc
    gdal._UseExceptions()
    return gdal


def define_crs(value: CRSLike) -> CRS:
    """Build a :class:`~geolattice.crs.CRS` from an EPSG code, WKT, or PROJ string."""
    if value is None:
        raise RasterCRSError("A CRS definition is required.")
    if isinstance(value, CRS):
        return value
    try:
        return CRS(value) if isinstance(value, (str, int)) else CRS(str(value))
    except Exception as exc:
        raise RasterCRSError(f"Could not define CRS from {value!r}: {exc}") from exc


def validate_crs(value: CRSLike) -> CRS:
    """Return a CRS after checking that a WKT or authority string is available."""
    crs = define_crs(value)
    if not (crs.wkt or crs.identifier or crs.proj):
        raise RasterCRSError(f"CRS {crs!r} has no WKT, authority, or PROJ definition.")
    return crs


def crs_to_gdal(value: CRSLike) -> str:
    """Export a CRS as a string GDAL accepts for ``srcSRS`` / ``dstSRS``."""
    crs = validate_crs(value)
    if crs.authority and crs.code:
        return f"{crs.authority}:{crs.code}"
    if crs.wkt:
        return crs.wkt
    if crs.proj:
        return crs.proj
    return str(crs)


def get_raster_crs(source: Union[PathLike, RasterDataset, Any]) -> Optional[CRS]:
    """
    Return the CRS of a raster file or open :class:`RasterDataset`.

    Raises:
        RasterOpenError: If a path cannot be opened.
    """
    if isinstance(source, RasterDataset):
        return source.crs
    if hasattr(source, "crs") and type(source).__name__ == "RasterDataset":
        return source.crs
    with open_raster_data(source) as dataset:
        return dataset.crs


def validate_raster_crs(
    source: Union[PathLike, RasterDataset, Any],
    *,
    require: bool = True,
) -> Optional[CRS]:
    """
    Read and validate the CRS of a raster dataset.

    Args:
        source: Path or open raster.
        require: When true, raise if the dataset has no CRS.
    """
    crs = get_raster_crs(source)
    if crs is None:
        if require:
            path = getattr(source, "path", source)
            raise RasterCRSError(f"Raster has no CRS definition: {path}")
        return None
    return validate_crs(crs)


def assign_raster_crs(
    source: PathLike,
    crs: CRSLike,
    destination: Optional[PathLike] = None,
    *,
    overwrite: bool = False,
) -> str:
    """
    Write a CRS onto a copy of the raster without warping pixel values.

    GDAL ``SetProjection`` is applied to a CreateCopy of ``source``. Pass
    ``destination`` or the result is written beside the source as
    ``*_crs.tif``.
    """
    gdal = _gdal()
    source_path = Path(source)
    if destination is None:
        destination = source_path.with_name(f"{source_path.stem}_crs{source_path.suffix or '.tif'}")
    dest = Path(destination)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {dest}. Pass overwrite=True to replace it.")

    src_ds = gdal.Open(os.fspath(source_path))
    if src_ds is None:
        raise RasterCRSError(f"Could not open raster: {source_path}")
    driver_name = src_ds.GetDriver().ShortName
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        driver = gdal.GetDriverByName("GTiff")
    if dest.exists():
        try:
            driver.Delete(os.fspath(dest))
        except Exception:
            dest.unlink(missing_ok=True)  # type: ignore[arg-type]
    copy = driver.CreateCopy(os.fspath(dest), src_ds)
    src_ds = None
    if copy is None:
        raise RasterCRSError(f"Could not create copy at {dest}")
    try:
        copy.SetProjection(crs_to_gdal(crs))
        copy.FlushCache()
    finally:
        copy = None
    return str(dest)


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
    to select a path by name.
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


def list_datum_transformations(
    source: CRSLike,
    target: CRSLike,
    settings: Optional[CRSSettings] = None,
) -> List[DatumTransformation]:
    """List every datum transformation path between ``source`` and ``target``."""
    dialog = DatumTransformationDialog(
        validate_crs(source),
        validate_crs(target),
        settings if settings is not None else crs_settings,
    )
    return list(dialog.paths)


def warp_options(
    options: Optional[Union[Sequence[str], str]] = None,
    *,
    format: Optional[str] = None,
    outputBounds: Optional[Sequence[float]] = None,
    outputBoundsSRS: Optional[str] = None,
    xRes: Optional[float] = None,
    yRes: Optional[float] = None,
    targetAlignedPixels: bool = False,
    width: int = 0,
    height: int = 0,
    srcSRS: Optional[CRSLike] = None,
    dstSRS: Optional[CRSLike] = None,
    coordinateOperation: Optional[str] = None,
    resampleAlg: Optional[ResampleAlg] = None,
    srcNodata: Any = None,
    dstNodata: Any = None,
    creationOptions: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    multithread: bool = False,
    cutlineDSName: Optional[str] = None,
    cropToCutline: bool = False,
    warpMemoryLimit: Optional[float] = None,
    errorThreshold: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """
    Build a :func:`osgeo.gdal.WarpOptions` object.

    CRS arguments accept EPSG codes, WKT, PROJ strings, or
    :class:`~geolattice.crs.CRS`. ``resampleAlg`` accepts GDAL names such as
    ``"bilinear"`` or aliases such as ``"nearest"``.
    """
    gdal = _gdal()
    kwargs_out: Dict[str, Any] = dict(kwargs)
    if options is not None:
        kwargs_out["options"] = options
    if format is not None:
        kwargs_out["format"] = format
    if outputBounds is not None:
        kwargs_out["outputBounds"] = list(outputBounds)
    if outputBoundsSRS is not None:
        kwargs_out["outputBoundsSRS"] = outputBoundsSRS
    if xRes is not None:
        kwargs_out["xRes"] = xRes
    if yRes is not None:
        kwargs_out["yRes"] = yRes
    if targetAlignedPixels:
        kwargs_out["targetAlignedPixels"] = True
    if width:
        kwargs_out["width"] = width
    if height:
        kwargs_out["height"] = height
    if srcSRS is not None:
        kwargs_out["srcSRS"] = crs_to_gdal(srcSRS)
    if dstSRS is not None:
        kwargs_out["dstSRS"] = crs_to_gdal(dstSRS)
    if coordinateOperation is not None:
        kwargs_out["coordinateOperation"] = coordinateOperation
    if resampleAlg is not None:
        kwargs_out["resampleAlg"] = _normalize_resample(resampleAlg)
    if srcNodata is not None:
        kwargs_out["srcNodata"] = srcNodata
    if dstNodata is not None:
        kwargs_out["dstNodata"] = dstNodata
    if creationOptions is not None:
        kwargs_out["creationOptions"] = _creation_options(creationOptions)
    if multithread:
        kwargs_out["multithread"] = True
    if cutlineDSName is not None:
        kwargs_out["cutlineDSName"] = cutlineDSName
    if cropToCutline:
        kwargs_out["cropToCutline"] = True
    if warpMemoryLimit is not None:
        kwargs_out["warpMemoryLimit"] = warpMemoryLimit
    if errorThreshold is not None:
        kwargs_out["errorThreshold"] = errorThreshold
    return gdal.WarpOptions(**kwargs_out)


def warp(
    src: Union[PathLike, RasterDataset, Any],
    dest: Optional[PathLike] = None,
    *,
    options: Any = None,
    format: Optional[str] = None,
    dst_crs: Optional[CRSLike] = None,
    src_crs: Optional[CRSLike] = None,
    resample: Optional[ResampleAlg] = None,
    xRes: Optional[float] = None,
    yRes: Optional[float] = None,
    outputBounds: Optional[Sequence[float]] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    coordinate_operation: Optional[str] = None,
    datum_operation: Optional[str] = None,
    multithread: bool = False,
    overwrite: bool = False,
    return_dataset: bool = False,
    **warp_kwds: Any,
) -> Union[str, RasterDataset]:
    """
    Warp a raster with :func:`osgeo.gdal.Warp`.

    Args:
        src: Input path, open :class:`RasterDataset`, or GDAL dataset.
        dest: Output path. Required unless ``return_dataset`` writes to a temp file.
        options: Pre-built :func:`warp_options` / ``gdal.WarpOptions`` object.
        dst_crs / src_crs: Destination and optional source CRS.
        resample: Resampling algorithm (defaults to bilinear when building options).
        coordinate_operation: PROJ pipeline / WKT coordinate operation (``-ct``).
        datum_operation: Name of a path from :func:`choose_datum_transformation`.
        return_dataset: Open the result with :func:`open_raster_data`.

    Returns:
        Output path, or an open :class:`RasterDataset` when ``return_dataset``.
    """
    gdal = _gdal()
    src_ds, owns_src = _open_gdal_source(src)
    destination = Path(dest) if dest is not None else None
    tempdir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if destination is None:
            tempdir = tempfile.TemporaryDirectory(prefix="geolattice-warp-")
            destination = Path(tempdir.name) / "warped.tif"
            format = format or "GTiff"
        elif destination.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {destination}. Pass overwrite=True to replace it."
            )

        op = coordinate_operation
        if op is None and datum_operation is not None:
            source_crs = src_crs or _crs_from_gdal(src_ds)
            if source_crs is None or dst_crs is None:
                raise RasterCRSError(
                    "datum_operation requires both a source CRS and dst_crs."
                )
            chosen = choose_datum_transformation(
                source_crs, dst_crs, operation=datum_operation
            )
            if chosen is not None and chosen.definition:
                op = chosen.definition

        if options is None:
            options = warp_options(
                format=format or _format_from_path(destination),
                srcSRS=src_crs,
                dstSRS=dst_crs,
                resampleAlg=resample if resample is not None else "bilinear",
                xRes=xRes,
                yRes=yRes,
                outputBounds=outputBounds,
                creationOptions=creation_options,
                coordinateOperation=op,
                multithread=multithread,
                **warp_kwds,
            )
        elif any(
            value is not None
            for value in (dst_crs, src_crs, resample, xRes, yRes, outputBounds, op)
        ):
            raise RasterCRSError(
                "Pass either a pre-built options= object or keyword arguments, not both."
            )

        if destination.exists() and overwrite:
            try:
                gdal.Unlink(os.fspath(destination))
            except Exception:
                destination.unlink(missing_ok=True)  # type: ignore[arg-type]

        result = gdal.Warp(os.fspath(destination), src_ds, options=options)
        if result is None:
            raise RasterCRSError(f"gdal.Warp failed for {destination}")
        result.FlushCache()
        result = None
        path = str(destination)
        if return_dataset:
            return open_raster_data(path)
        return path
    finally:
        if owns_src:
            src_ds = None
        if tempdir is not None and not return_dataset:
            tempdir.cleanup()


def reproject_raster(
    src: Union[PathLike, RasterDataset, Any],
    dest: Optional[PathLike] = None,
    dst_crs: CRSLike = None,
    *,
    src_crs: Optional[CRSLike] = None,
    resample: ResampleAlg = "bilinear",
    resolution: Optional[Union[float, Tuple[float, float]]] = None,
    bounds: Optional[Sequence[float]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    creation_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    format: Optional[str] = None,
    coordinate_operation: Optional[str] = None,
    datum_operation: Optional[str] = None,
    multithread: bool = False,
    overwrite: bool = False,
    return_dataset: bool = False,
    **warp_kwds: Any,
) -> Union[str, RasterDataset]:
    """
    Reproject a raster to ``dst_crs`` (thin wrapper around :func:`warp`).

    ``resolution`` may be a single cell size or ``(xRes, yRes)``.
    """
    if dst_crs is None:
        raise RasterCRSError("dst_crs is required for reproject_raster.")
    x_res = y_res = None
    if resolution is not None:
        if isinstance(resolution, (int, float)):
            x_res = y_res = float(resolution)
        else:
            x_res, y_res = float(resolution[0]), float(resolution[1])
    return warp(
        src,
        dest,
        format=format,
        dst_crs=dst_crs,
        src_crs=src_crs,
        resample=resample,
        xRes=x_res,
        yRes=y_res,
        outputBounds=bounds,
        width=width or 0,
        height=height or 0,
        creation_options=creation_options,
        coordinate_operation=coordinate_operation,
        datum_operation=datum_operation,
        multithread=multithread,
        overwrite=overwrite,
        return_dataset=return_dataset,
        **warp_kwds,
    )


def reproject_raster_array(
    data: Any,
    src_crs: CRSLike,
    dst_crs: CRSLike,
    *,
    geotransform: Optional[Sequence[float]] = None,
    resample: ResampleAlg = "bilinear",
    nodata: Optional[float] = None,
) -> Tuple[Any, Tuple[float, float, float, float, float, float], CRS]:
    """
    Reproject an in-memory NumPy array (or open dataset) and return
    ``(array, geotransform, crs)``.

    Writes a temporary GeoTIFF, warps it, and reads the result back.
    """
    import numpy as np

    src_crs_obj = validate_crs(src_crs)
    dst_crs_obj = validate_crs(dst_crs)
    with tempfile.TemporaryDirectory(prefix="geolattice-reproject-array-") as folder:
        source_path = Path(folder) / "source.tif"
        warped_path = Path(folder) / "warped.tif"
        write_raster_data(
            source_path,
            data,
            crs=src_crs_obj,
            geotransform=geotransform,
            nodata=nodata,
            overwrite=True,
        )
        warp(
            source_path,
            warped_path,
            dst_crs=dst_crs_obj,
            src_crs=src_crs_obj,
            resample=resample,
            overwrite=True,
        )
        with open_raster_data(warped_path) as dataset:
            array = np.asarray(dataset.read())
            gt = dataset.geotransform
            crs = dataset.crs or dst_crs_obj
            return array, gt, crs


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _normalize_resample(value: ResampleAlg) -> Union[str, int]:
    if isinstance(value, int):
        return value
    key = str(value).strip().lower()
    if key not in _RESAMPLE_ALIASES:
        known = ", ".join(sorted(_RESAMPLE_ALIASES))
        raise RasterCRSError(f"Unknown resample algorithm {value!r}. Known: {known}")
    return _RESAMPLE_ALIASES[key]


def _creation_options(
    options: Union[Sequence[str], Mapping[str, Any]],
) -> List[str]:
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


def _open_gdal_source(source: Any) -> Tuple[Any, bool]:
    gdal = _gdal()
    if isinstance(source, RasterDataset):
        if source._dataset is None:
            raise RasterCRSError(f"Dataset is closed: {source.path}")
        return source._dataset, False
    if hasattr(source, "GetRasterBand"):
        return source, False
    path = os.fspath(source)
    dataset = gdal.Open(path)
    if dataset is None:
        raise RasterCRSError(f"Could not open raster: {path}")
    return dataset, True


def _crs_from_gdal(dataset: Any) -> Optional[CRS]:
    wkt = dataset.GetProjection() or None
    if not wkt:
        return None
    try:
        return CRS.from_wkt(wkt)
    except Exception:
        try:
            return CRS(wkt)
        except Exception:
            return None


# Re-export for convenience / discovery.
__all__ = [
    "RasterCRSError",
    "DatumTransformation",
    "DatumTransformationRequired",
    "define_crs",
    "validate_crs",
    "crs_to_gdal",
    "get_raster_crs",
    "validate_raster_crs",
    "assign_raster_crs",
    "choose_datum_transformation",
    "list_datum_transformations",
    "warp_options",
    "warp",
    "reproject_raster",
    "reproject_raster_array",
]
