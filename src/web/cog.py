"""
Create and validate Cloud-Optimized GeoTIFFs (COGs).

A COG is an ordinary GeoTIFF arranged so a client can read any part of it
with a few HTTP range requests:

1. the image is **tiled** (typically 256 or 512 pixel blocks), not striped;
2. it carries **internal overviews** (reduced-resolution copies) for zoomed-out views;
3. all image file directories (IFDs, the TIFF headers) come **first**, in
   order full resolution, overview 1, overview 2, ...; and
4. the pixel data follow, **smallest overview first** and full resolution last.

:func:`validate_cog` checks these rules with GDAL; :func:`to_cog` converts any
GDAL-readable raster with GDAL's COG driver.

Example:
    from geolattice.web import to_cog, validate_cog

    to_cog("ndvi.tif", "ndvi_cog.tif", overview_resampling="average")
    report = validate_cog("ndvi_cog.tif")
    print(report.valid, report.errors, report.warnings)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from geolattice.web.cloud import to_vsi_path

PathLike = Union[str, Path]


@dataclass
class COGReport:
    """Result of :func:`validate_cog`."""

    path: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        head = f"{self.path}: {'valid' if self.valid else 'NOT a valid'} Cloud-Optimized GeoTIFF"
        lines = [head]
        lines += [f"  error: {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_cog(path: PathLike, *, strict: bool = False, min_size_for_overviews: int = 512) -> COGReport:
    """
    Check whether a GeoTIFF is cloud-optimized.

    Args:
        path: Local path, ``/vsi...`` path or URL (``s3://``, ``https://``, ...).
        strict: Treat warnings (missing overviews on a large image, missing
            ``LAYOUT=COG`` marker) as errors.
        min_size_for_overviews: Images larger than this in width or height
            should have internal overviews.

    Returns:
        A :class:`COGReport`; ``report.valid`` is true when there are no errors.
    """
    gdal = _gdal()
    location = to_vsi_path(os.fspath(path))
    errors: List[str] = []
    warnings: List[str] = []
    try:
        ds = gdal.Open(location)
    except Exception as exc:
        return COGReport(str(path), False, [f"GDAL cannot open the file: {exc}"])
    if ds is None:
        return COGReport(str(path), False, ["GDAL cannot open the file."])
    driver = ds.GetDriver().ShortName
    if driver != "GTiff":
        return COGReport(str(path), False, [f"Not a GeoTIFF (GDAL driver {driver})."])

    band = ds.GetRasterBand(1)
    width, height = ds.RasterXSize, ds.RasterYSize
    block_x, block_y = band.GetBlockSize()
    structure = ds.GetMetadata("IMAGE_STRUCTURE") or {}
    files = ds.GetFileList() or []
    overviews = [band.GetOverview(i) for i in range(band.GetOverviewCount())]
    details: Dict[str, Any] = {
        "size": (width, height),
        "bands": ds.RasterCount,
        "dtype": gdal.GetDataTypeName(band.DataType),
        "block_size": (block_x, block_y),
        "overview_count": len(overviews),
        "overview_sizes": [(o.XSize, o.YSize) for o in overviews],
        "compression": structure.get("COMPRESSION"),
        "interleave": structure.get("INTERLEAVE"),
        "layout": structure.get("LAYOUT"),
    }

    # 1. tiling
    tiled = not (block_x == width and block_y < height) and not (block_y == 1 and height > 1)
    if (width > block_x or height > block_y) and not tiled:
        errors.append(f"The image is striped (block {block_x}×{block_y}), not tiled.")
    elif block_x != block_y and tiled:
        warnings.append(f"Blocks are not square ({block_x}×{block_y}).")
    details["tiled"] = tiled

    # 2. overviews: present and internal
    if any(str(f).lower().endswith(".ovr") for f in files):
        errors.append("Overviews are stored in an external .ovr file; a COG needs internal overviews.")
    if not overviews and max(width, height) > min_size_for_overviews:
        warnings.append(f"The image is larger than {min_size_for_overviews} px but has no internal overviews.")
    for i, ovr in enumerate(overviews):
        ox, oy = ovr.GetBlockSize()
        if ox == ovr.XSize and oy < ovr.YSize and ovr.YSize > 1:
            errors.append(f"Overview {i + 1} is striped, not tiled.")

    # 3. IFD order: full resolution first, then overviews in decreasing size
    ifd_main = _int(band.GetMetadataItem("IFD_OFFSET", "TIFF"))
    ifd_ovr = [_int(o.GetMetadataItem("IFD_OFFSET", "TIFF")) for o in overviews]
    details["ifd_offsets"] = [ifd_main, *ifd_ovr]
    if ifd_main is not None and ifd_ovr and all(v is not None for v in ifd_ovr):
        sequence = [ifd_main, *ifd_ovr]
        if sequence != sorted(sequence):
            errors.append("Image file directories are not in order (full resolution, then overviews).")
        if ifd_main > 1024 * 1024:
            warnings.append(f"The main IFD starts at byte {ifd_main:,}, far from the file start.")

    # 4. data order: headers before data, smallest overview's data first
    data_main = _first_block_offset(band)
    data_ovr = [_first_block_offset(o) for o in overviews]
    details["data_offsets"] = [data_main, *data_ovr]
    last_ifd = max(v for v in [ifd_main, *ifd_ovr] if v is not None) if ifd_main is not None else None
    first_data = min(v for v in [data_main, *data_ovr] if v is not None) if data_main is not None else None
    if last_ifd is not None and first_data is not None and first_data < last_ifd:
        errors.append("Some pixel data come before the image file directories.")
    if data_main is not None and data_ovr and all(v is not None for v in data_ovr):
        if [*data_ovr[::-1], data_main] != sorted([*data_ovr[::-1], data_main]):
            errors.append("Pixel data are not ordered from the smallest overview to full resolution.")

    if details["layout"] != "COG":
        warnings.append("No LAYOUT=COG marker: not written by GDAL's COG driver (may still be valid).")
    if details["compression"] is None:
        warnings.append("The image is uncompressed.")
    ds = None
    if strict:
        errors, warnings = errors + warnings, []
    return COGReport(str(path), not errors, errors, warnings, details)


def to_cog(
    src: PathLike,
    dst: PathLike,
    *,
    compress: str = "DEFLATE",
    blocksize: int = 512,
    overview_resampling: str = "AVERAGE",
    resampling: str = "BILINEAR",
    predictor: Optional[Union[int, str]] = None,
    level: Optional[int] = None,
    quality: Optional[int] = None,
    nodata: Optional[float] = None,
    web_optimized: bool = False,
    bigtiff: str = "IF_SAFER",
    num_threads: Union[int, str] = "ALL_CPUS",
    overwrite: bool = True,
) -> str:
    """
    Convert any GDAL-readable raster to a Cloud-Optimized GeoTIFF.

    Args:
        src: Input raster (path, ``/vsi`` path or URL).
        dst: Output ``.tif`` path.
        compress: ``DEFLATE``, ``ZSTD``, ``LZW``, ``WEBP``, ``JPEG``, ``LERC`` or ``NONE``.
        blocksize: Tile size in pixels (256 or 512 are typical).
        overview_resampling: ``AVERAGE`` for continuous data, ``NEAREST`` or
            ``MODE`` for classes and masks.
        resampling: Resampling for reprojection when ``web_optimized``.
        predictor: TIFF predictor (``2`` for integers, ``3`` for floats, or ``"YES"``).
        level: Compression level for DEFLATE/ZSTD.
        quality: Quality for JPEG/WEBP (lossy).
        nodata: Set this no-data value on the output.
        web_optimized: Reproject to Web Mercator aligned to the XYZ tile grid,
            so tiles can be served without resampling.
        bigtiff: ``YES``, ``NO``, ``IF_NEEDED`` or ``IF_SAFER``.
        num_threads: Threads for compression.
        overwrite: Replace an existing file.

    Returns:
        The output path.
    """
    gdal = _gdal()
    if gdal.GetDriverByName("COG") is None:
        raise RuntimeError("GDAL has no COG driver; GDAL 3.1 or later is required.")
    destination = Path(dst)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; pass overwrite=True.")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = [f"COMPRESS={compress.upper()}", f"BLOCKSIZE={int(blocksize)}",
               f"OVERVIEW_RESAMPLING={overview_resampling.upper()}", f"BIGTIFF={bigtiff}",
               f"NUM_THREADS={num_threads}"]
    if predictor is not None:
        options.append(f"PREDICTOR={predictor}")
    if level is not None:
        options.append(f"LEVEL={int(level)}")
    if quality is not None:
        options.append(f"QUALITY={int(quality)}")
    if web_optimized:
        options += ["TILING_SCHEME=GoogleMapsCompatible", f"WARP_RESAMPLING={resampling.upper()}"]
    kwargs: Dict[str, Any] = {"format": "COG", "creationOptions": options}
    if nodata is not None:
        kwargs["noData"] = nodata
    result = gdal.Translate(str(destination), to_vsi_path(os.fspath(src)), **kwargs)
    if result is None:
        raise RuntimeError(f"GDAL could not write {destination}.")
    result = None
    return str(destination)


def _first_block_offset(band: Any) -> Optional[int]:
    return _int(band.GetMetadataItem("BLOCK_OFFSET_0_0", "TIFF"))


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _gdal() -> Any:
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


__all__ = ["COGReport", "validate_cog", "to_cog"]
