"""
Open and create file-based raster datasets with GDAL raster drivers.

The reader and writer target the formats documented in the GDAL raster driver
reference: https://gdal.org/en/stable/drivers/raster/index.html

Example:
    from geolattice.raster import open_raster_data, write_raster_data
    import numpy as np

    array = np.arange(20, dtype=np.float32).reshape(4, 5)
    write_raster_data(
        "grid.tif",
        array,
        crs=4326,
        geotransform=(0, 1, 0, 4, 0, -1),
        nodata=-9999,
    )
    with open_raster_data("grid.tif") as dataset:
        print(dataset.driver, dataset.shape, dataset.crs)
        grid = dataset.read()
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.core.bounds import Bounds
from geolattice.crs import CRS

_GDAL: Optional[Tuple[Any, Any]] = None
_DRIVER_AVAILABLE: Dict[str, bool] = {}

_GDAL_TO_NUMPY: Dict[int, Any] = {}
_NUMPY_TO_GDAL: Dict[Any, int] = {}


class RasterOpenError(Exception):
    """Raised when a raster dataset cannot be opened or read."""


class DriverUnavailableError(RasterOpenError):
    """Raised when this GDAL build does not include the driver for a format."""


class RasterWriteError(RasterOpenError):
    """Raised when raster pixels cannot be written."""


class RasterFormat:
    """
    One file-based raster format and the GDAL drivers that read or write it.

    Drivers are listed in the order they are tried when opening.
    """

    def __init__(
        self,
        name: str,
        extensions: Sequence[str],
        drivers: Sequence[str],
        description: str,
        notes: str = "",
    ) -> None:
        self.name = name
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.drivers = tuple(drivers)
        self.description = description
        self.notes = notes

    def __repr__(self) -> str:
        return f"RasterFormat({self.name!r}, extensions={self.extensions!r})"


RASTER_FORMATS: Tuple[RasterFormat, ...] = (
    RasterFormat(
        name="GeoTIFF / COG",
        extensions=(".tif", ".tiff"),
        drivers=("GTiff",),
        description="GeoTIFF and Cloud Optimized GeoTIFF (COG).",
        notes="COG is written with GTiff creation options such as COMPRESS and TILED.",
    ),
    RasterFormat(
        name="JPEG",
        extensions=(".jpg", ".jpeg"),
        drivers=("JPEG",),
        description="JPEG JFIF image.",
    ),
    RasterFormat(
        name="PNG",
        extensions=(".png",),
        drivers=("PNG",),
        description="Portable Network Graphics.",
    ),
    RasterFormat(
        name="GIF",
        extensions=(".gif",),
        drivers=("GIF",),
        description="Graphics Interchange Format.",
    ),
    RasterFormat(
        name="BMP",
        extensions=(".bmp",),
        drivers=("BMP",),
        description="Windows Bitmap.",
    ),
    RasterFormat(
        name="Erdas Imagine",
        extensions=(".img",),
        drivers=("HFA", "ENVI"),
        description="Erdas Imagine (.img).",
        notes="ENVI binary files also use .img; HFA is tried first, then ENVI.",
    ),
    RasterFormat(
        name="NetCDF",
        extensions=(".nc",),
        drivers=("netCDF",),
        description="Network Common Data Form.",
        notes="Multi-dimensional files expose GDAL subdatasets; pass subdataset= to open a grid.",
    ),
    RasterFormat(
        name="HDF4",
        extensions=(".hdf",),
        drivers=("HDF4",),
        description="Hierarchical Data Format 4.",
        notes="Subdatasets are listed on the root dataset.",
    ),
    RasterFormat(
        name="HDF5",
        extensions=(".h5", ".hdf5"),
        drivers=("HDF5",),
        description="Hierarchical Data Format 5.",
        notes="Subdatasets are listed on the root dataset.",
    ),
    RasterFormat(
        name="ENVI",
        extensions=(".hdr", ".envi", ".img"),
        drivers=("ENVI",),
        description="ENVI header / binary raster.",
        notes=(
            "Open the binary data file (.img / .dat), not the .hdr. "
            "A .img extension can also be Erdas Imagine (HFA); both drivers are tried."
        ),
    ),
    RasterFormat(
        name="ESRI ASCII Grid",
        extensions=(".asc", ".grd"),
        drivers=("AAIGrid",),
        description="Arc/Info ASCII Grid.",
    ),
    RasterFormat(
        name="VRT",
        extensions=(".vrt",),
        drivers=("VRT",),
        description="GDAL Virtual Raster.",
    ),
    RasterFormat(
        name="JPEG2000",
        extensions=(".jp2", ".j2k", ".j2c"),
        drivers=("JP2OpenJPEG", "JP2KAK", "JP2ECW"),
        description="JPEG 2000.",
        notes="JP2OpenJPEG is tried first when it is installed.",
    ),
    RasterFormat(
        name="GeoPackage raster",
        extensions=(".gpkg",),
        drivers=("GPKG",),
        description="GeoPackage raster tiles.",
        notes="The same extension as vector GeoPackage; GDAL selects raster or vector on open.",
    ),
    RasterFormat(
        name="Zarr",
        extensions=(".zarr", ".zar"),
        drivers=("Zarr",),
        description="Zarr arrays.",
        notes="Requires GDAL 3.4 or newer with the Zarr driver.",
    ),
    RasterFormat(
        name="MBTiles",
        extensions=(".mbtiles",),
        drivers=("MBTiles",),
        description="Mapbox MBTiles tile cache.",
    ),
    RasterFormat(
        name="SAGA GIS",
        extensions=(".sdat", ".sgrd"),
        drivers=("SAGA",),
        description="SAGA GIS grid.",
    ),
    RasterFormat(
        name="USGS DEM",
        extensions=(".dem",),
        drivers=("USGSDEM", "GTOPO30"),
        description="USGS ASCII DEM and related formats.",
    ),
)


class RasterBand:
    """Metadata and read helpers for one raster band."""

    def __init__(self, owner: "RasterDataset", index: int, band: Any) -> None:
        self._owner = owner
        self.index = index
        self._band = band

    @property
    def nodata(self) -> Optional[float]:
        """NoData value for the band, when the driver defines one."""
        try:
            value = self._band.GetNoDataValue()
        except RuntimeError:
            return None
        if value is None:
            return None
        if math.isnan(value):
            return value
        return float(value)

    @property
    def dtype(self) -> str:
        """NumPy dtype name for the band."""
        gdal, _ = _gdal_modules()
        code = self._band.DataType
        dtype = _GDAL_TO_NUMPY.get(code)
        return np.dtype(dtype).name if dtype is not None else gdal.GetDataTypeName(code)

    @property
    def color_interpretation(self) -> str:
        gdal, _ = _gdal_modules()
        return gdal.GetColorInterpretationName(self._band.GetColorInterpretation())

    @property
    def metadata(self) -> Dict[str, str]:
        return dict(self._band.GetMetadata() or {})

    @property
    def statistics(self) -> Optional[Tuple[float, float, float, float]]:
        """Minimum, maximum, mean, and standard deviation when cached."""
        try:
            stats = self._band.GetStatistics(False, True)
        except RuntimeError:
            return None
        if stats is None or len(stats) < 4:
            return None
        return tuple(float(item) for item in stats[:4])

    def read(
        self,
        window: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Read band pixels into a 2-D NumPy array.

        Args:
            window: Optional ``(xoff, yoff, xsize, ysize)`` pixel window.
        """
        if self._owner._dataset is None:
            raise RasterOpenError(f"Dataset is closed: {self._owner.path}")
        xoff, yoff, xsize, ysize = _window_or_full(window, self._owner.width, self._owner.height)
        return _read_array(self._band, xoff, yoff, xsize, ysize)


class RasterDataset:
    """
    An open GDAL raster dataset.

    Use it as a context manager so file handles are released::

        with open_raster_data("dem.tif") as dataset:
            elevation = dataset.read()
    """

    def __init__(
        self,
        dataset: Any,
        path: str,
        driver: str,
        format_name: str,
        subdataset: Optional[str] = None,
    ) -> None:
        self.path = path
        self.driver = driver
        self.format_name = format_name
        self.subdataset = subdataset
        self._dataset = dataset
        self._bands: Tuple[RasterBand, ...] = tuple(
            RasterBand(self, index, dataset.GetRasterBand(index))
            for index in range(1, dataset.RasterCount + 1)
        )

    @property
    def width(self) -> int:
        return int(self._dataset.RasterXSize)

    @property
    def height(self) -> int:
        return int(self._dataset.RasterYSize)

    @property
    def band_count(self) -> int:
        return len(self._bands)

    @property
    def shape(self) -> Tuple[int, int]:
        """Raster size as ``(rows, columns)``."""
        return self.height, self.width

    @property
    def bands(self) -> Tuple[RasterBand, ...]:
        return self._bands

    @property
    def geotransform(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(float(value) for value in self._dataset.GetGeoTransform())

    @property
    def projection_wkt(self) -> Optional[str]:
        wkt = self._dataset.GetProjection()
        return wkt or None

    @property
    def crs(self) -> Optional[CRS]:
        return _crs_from_dataset(self._dataset)

    @property
    def bounds(self) -> Optional[Bounds]:
        gt = self.geotransform
        if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) and self._dataset.GetProjection() in ("", None):
            return None
        return _bounds_from_geotransform(gt, self.width, self.height)

    @property
    def metadata(self) -> Dict[str, str]:
        return dict(self._dataset.GetMetadata() or {})

    @property
    def subdatasets(self) -> Tuple[Tuple[str, str], ...]:
        """``(name, description)`` pairs when the format exposes subdatasets."""
        pairs = self._dataset.GetSubDatasets() or ()
        return tuple((str(name), str(description)) for name, description in pairs)

    @property
    def has_subdatasets(self) -> bool:
        return bool(self.subdatasets)

    def get_band(self, index: int) -> RasterBand:
        if index < 1 or index > len(self._bands):
            raise RasterOpenError(
                f"Band index {index} is outside 1..{len(self._bands)} for {self.path}"
            )
        return self._bands[index - 1]

    def read(
        self,
        bands: Union[int, Sequence[int], None] = None,
        window: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Read one or more bands into a NumPy array.

        A single band returns shape ``(rows, columns)``. Multiple bands return
        shape ``(bands, rows, columns)``.
        """
        if self._dataset is None:
            raise RasterOpenError(f"Dataset is closed: {self.path}")
        if bands is None:
            if self.band_count == 1:
                return self.get_band(1).read(window=window)
            bands = tuple(range(1, self.band_count + 1))
        elif isinstance(bands, int):
            return self.get_band(bands).read(window=window)
        arrays = [self.get_band(index).read(window=window) for index in bands]
        return np.stack(arrays, axis=0)

    def copy_metadata_from(self, other: "RasterDataset") -> None:
        """Copy dataset and band metadata from another open dataset."""
        if self._dataset is None:
            raise RasterOpenError(f"Dataset is closed: {self.path}")
        for key, value in other.metadata.items():
            self._dataset.SetMetadataItem(str(key), str(value))
        for left, right in zip(self._bands, other._bands):
            for key, value in right.metadata.items():
                left._band.SetMetadataItem(str(key), str(value))

    def close(self) -> None:
        """Release the GDAL dataset handle."""
        self._dataset = None

    def __enter__(self) -> "RasterDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[RasterBand]:
        return iter(self._bands)


def open_raster_data(
    path: Union[str, Path],
    driver: Optional[str] = None,
    subdataset: Union[int, str, None] = None,
    open_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
) -> RasterDataset:
    """
    Open a raster dataset for reading.

    Args:
        path: Dataset path or GDAL virtual path (``/vsi*``, ``s3://``, …).
        driver: GDAL raster driver short name, such as ``"GTiff"``.
        subdataset: Subdataset index or GDAL subdataset name for multi-grid
            formats such as NetCDF and HDF.
        open_options: GDAL open options as ``KEY=VALUE`` strings or a mapping.

    Returns:
        Open :class:`RasterDataset`.
    """
    requested = os.fspath(path)
    location, local = _resolve_input(requested)
    formats = _candidate_formats(local, location)
    drivers = _driver_list(formats, driver)
    user_options = _normalize_open_options(open_options)

    root, driver_name = _open_with_drivers(location, drivers, formats, user_options)
    target: Any = root
    try:
        target, sub_name = _select_subdataset(root, subdataset)
        format_name = _format_name(driver_name, formats)
        return RasterDataset(
            dataset=target,
            path=location if sub_name is None else sub_name,
            driver=driver_name,
            format_name=format_name,
            subdataset=sub_name,
        )
    finally:
        if target is not root:
            root = None


def read_raster_data(
    path: Union[str, Path],
    driver: Optional[str] = None,
    subdataset: Union[int, str, None] = None,
    **kwargs: Any,
) -> RasterDataset:
    """Read a raster dataset. Alias of :func:`open_raster_data`."""
    return open_raster_data(path, driver=driver, subdataset=subdataset, **kwargs)


def open_raster(path: Union[str, Path], **kwargs: Any) -> RasterDataset:
    """Open a raster dataset. Alias of :func:`open_raster_data`."""
    return open_raster_data(path, **kwargs)


def write_raster_data(
    path: Union[str, Path],
    data: Any,
    driver: Optional[str] = None,
    crs: Any = None,
    geotransform: Optional[Sequence[float]] = None,
    nodata: Optional[Union[float, Sequence[float]]] = None,
    overwrite: bool = False,
    creation_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    band_metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    Write raster pixels with a GDAL raster driver.

    ``data`` may be a 2-D or 3-D NumPy array (bands first when 3-D), an
    :class:`RasterDataset`, or an xarray ``DataArray`` / ``Dataset`` when
    xarray is installed.

    Args:
        path: Output path. The extension selects the driver when ``driver`` is omitted.
        data: Pixel values to write.
        driver: GDAL driver short name.
        crs: CRS stored with the dataset.
        geotransform: GDAL six-parameter geotransform. When omitted, a north-up
            unit grid anchored at the origin is used.
        nodata: NoData value for every band, or one value per band.
        overwrite: Replace an existing dataset at ``path``.
        creation_options: Driver-specific creation options.
        metadata: Dataset metadata keys and values.
        band_metadata: Per-band metadata mappings.

    Returns:
        Path of the written dataset.
    """
    destination = Path(path)
    driver_name = _choose_driver(destination, driver)
    _prepare_destination(destination, driver_name, overwrite)

    if isinstance(data, RasterDataset):
        return _write_copy(
            destination,
            data,
            driver_name,
            creation_options,
            overwrite=True,
        )

    array, band_count = _array_from_data(data)
    gt = _geotransform_or_default(geotransform, array.shape[-2], array.shape[-1])
    crs_value = crs
    if crs_value is None and hasattr(data, "rio"):
        try:
            crs_value = data.rio.crs
        except Exception:
            crs_value = None
    if crs_value is None and hasattr(data, "attrs"):
        crs_value = data.attrs.get("crs") or data.attrs.get("spatial_ref")

    _write_array(
        destination,
        array,
        driver_name,
        crs=crs_value,
        geotransform=gt,
        nodata=nodata,
        creation_options=_option_map(creation_options),
        metadata=metadata or {},
        band_metadata=band_metadata or [],
    )
    return str(destination)


def write_geotiff(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write a GeoTIFF that :func:`read_raster_data` can open."""
    kwargs["driver"] = "GTiff"
    return write_raster_data(path, data, **kwargs)


def write_png(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write a PNG image that :func:`read_raster_data` can open."""
    kwargs["driver"] = "PNG"
    return write_raster_data(path, data, **kwargs)


def write_jpeg(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write a JPEG image that :func:`read_raster_data` can open."""
    kwargs["driver"] = "JPEG"
    return write_raster_data(path, data, **kwargs)


def supported_formats() -> List[Dict[str, Any]]:
    """
    List file-based raster formats this module knows about.

    Returns:
        One dictionary per format with extensions, drivers, availability,
        description, and notes.
    """
    records: List[Dict[str, Any]] = []
    for fmt in RASTER_FORMATS:
        available = [name for name in fmt.drivers if driver_available(name)]
        records.append(
            {
                "name": fmt.name,
                "extensions": list(fmt.extensions),
                "drivers": list(fmt.drivers),
                "available_drivers": available,
                "available": bool(available),
                "description": fmt.description,
                "notes": fmt.notes,
            }
        )
    return records


def supported_write_formats() -> List[Dict[str, Any]]:
    """
    List raster formats and whether this GDAL build can create them.

    Returns:
        One dictionary per known format with creatable drivers.
    """
    gdal, _ = _gdal_modules()
    records: List[Dict[str, Any]] = []
    for fmt in RASTER_FORMATS:
        creatable = [
            name
            for name in fmt.drivers
            if _driver_can_write(gdal.GetDriverByName(name) if driver_available(name) else None)
        ]
        records.append(
            {
                "name": fmt.name,
                "extensions": list(fmt.extensions),
                "drivers": list(fmt.drivers),
                "creatable_drivers": creatable,
                "writable": bool(creatable),
                "description": fmt.description,
            }
        )
    return records


def driver_available(name: str) -> bool:
    """
    Return whether this GDAL build includes raster driver ``name``.

    Args:
        name: GDAL driver short name, such as ``"GTiff"`` or ``"PNG"``.
    """
    cached = _DRIVER_AVAILABLE.get(name)
    if cached is not None:
        return cached
    gdal, _ = _gdal_modules()
    driver = gdal.GetDriverByName(name)
    found = driver is not None and driver.GetMetadataItem("DCAP_RASTER") == "YES"
    _DRIVER_AVAILABLE[name] = found
    return found


def list_gdal_raster_drivers() -> List[Dict[str, str]]:
    """
    List every raster driver registered in this GDAL build.

    Returns:
        Driver short name, long name, and extensions when GDAL publishes them.
    """
    gdal, _ = _gdal_modules()
    records: List[Dict[str, str]] = []
    for index in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(index)
        if driver.GetMetadataItem("DCAP_RASTER") != "YES":
            continue
        records.append(
            {
                "short_name": driver.ShortName,
                "long_name": driver.LongName,
                "extensions": driver.GetMetadataItem("DMD_EXTENSIONS") or "",
            }
        )
    records.sort(key=lambda item: item["short_name"].lower())
    return records


def _gdal_modules() -> Tuple[Any, Any]:
    """Import GDAL/OSR and enable exceptions without loading the NumPy array bridge."""
    global _GDAL, _GDAL_TO_NUMPY, _NUMPY_TO_GDAL
    if _GDAL is not None:
        return _GDAL
    try:
        from osgeo import gdal, osr
    except ImportError as exc:
        raise RasterOpenError(
            "GDAL Python bindings are required for raster I/O. "
            "Install the gdal package that matches the GDAL library on this machine."
        ) from exc
    gdal._UseExceptions()
    osr._UseExceptions()
    _GDAL = (gdal, osr)
    _GDAL_TO_NUMPY = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_Int8: np.int8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    complex16 = getattr(gdal, "GDT_CInt16", None)
    complex32 = getattr(gdal, "GDT_CInt32", None)
    cfloat32 = getattr(gdal, "GDT_CFloat32", None)
    cfloat64 = getattr(gdal, "GDT_CFloat64", None)
    if complex16 is not None:
        _GDAL_TO_NUMPY[complex16] = np.complex64
    if complex32 is not None:
        _GDAL_TO_NUMPY[complex32] = np.complex128
    if cfloat32 is not None:
        _GDAL_TO_NUMPY[cfloat32] = np.complex64
    if cfloat64 is not None:
        _GDAL_TO_NUMPY[cfloat64] = np.complex128
    _NUMPY_TO_GDAL = {
        np.dtype(code): gdal_type for gdal_type, code in _GDAL_TO_NUMPY.items()
    }
    return _GDAL


def _resolve_input(text: str) -> Tuple[str, Optional[Path]]:
    if _is_virtual_path(text):
        return text, None
    local = Path(text).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Raster dataset does not exist: {local}")
    return str(local), local


def _is_virtual_path(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("/vsi") or "://" in text


def _candidate_formats(local: Optional[Path], location: str) -> List[RasterFormat]:
    suffix = Path(location).suffix.lower()
    formats = [fmt for fmt in RASTER_FORMATS if suffix in fmt.extensions]
    return formats


def _driver_list(formats: Sequence[RasterFormat], explicit: Optional[str]) -> List[str]:
    if explicit:
        return [explicit]
    ordered: List[str] = []
    for fmt in formats:
        for name in fmt.drivers:
            if name not in ordered:
                ordered.append(name)
    return ordered


def _format_name(driver_name: str, formats: Sequence[RasterFormat]) -> str:
    for fmt in formats:
        if driver_name in fmt.drivers:
            return fmt.name
    return driver_name


def _normalize_open_options(
    options: Optional[Union[Sequence[str], Dict[str, Any]]],
) -> Dict[str, str]:
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(key): str(value) for key, value in options.items()}
    normalized: Dict[str, str] = {}
    for item in options:
        text = str(item)
        if "=" not in text:
            raise ValueError(f"open option must be KEY=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        normalized[key] = value
    return normalized


def _open_with_drivers(
    location: str,
    drivers: Sequence[str],
    formats: Sequence[RasterFormat],
    user_options: Dict[str, str],
) -> Tuple[Any, str]:
    if drivers:
        available = [name for name in drivers if driver_available(name)]
        if not available:
            raise DriverUnavailableError(_unavailable_message(location, drivers, formats))
    else:
        available = []

    errors: List[str] = []
    attempts = available or [""]
    for name in attempts:
        option_list = [f"{key}={value}" for key, value in user_options.items()]
        try:
            dataset = _open_ex(location, name or None, option_list)
        except RuntimeError as exc:
            message = str(exc).strip()
            if "No such file or directory" in message:
                raise FileNotFoundError(message) from exc
            errors.append(f"{name or 'GDAL'}: {message}")
            continue
        if dataset is None:
            errors.append(f"{name or 'GDAL'}: open returned None")
            continue
        actual = dataset.GetDriver().ShortName
        return dataset, actual

    raise RasterOpenError(_open_failure_message(location, available, formats, errors))


def _open_ex(location: str, driver_name: Optional[str], option_list: Sequence[str]) -> Any:
    gdal, _ = _gdal_modules()
    flags = gdal.OF_RASTER | gdal.OF_READONLY | getattr(gdal, "OF_VERBOSE_ERROR", 0)
    kwargs: Dict[str, Any] = {"nOpenFlags": flags}
    if driver_name:
        kwargs["allowed_drivers"] = [driver_name]
    if option_list:
        kwargs["open_options"] = [str(item) for item in option_list]
    return gdal.OpenEx(location, **kwargs)


def _unavailable_message(
    location: str,
    drivers: Sequence[str],
    formats: Sequence[RasterFormat],
) -> str:
    hints = []
    for name in drivers:
        for fmt in formats:
            if name in fmt.drivers and fmt.notes:
                hints.append(fmt.notes)
                break
    detail = " ".join(dict.fromkeys(hints))
    return (
        f"None of the requested GDAL raster drivers are available for {location!r}: "
        f"{', '.join(drivers)}. {detail}".strip()
    )


def _open_failure_message(
    location: str,
    drivers: Sequence[str],
    formats: Sequence[RasterFormat],
    errors: Sequence[str],
) -> str:
    if not formats:
        known = ", ".join(sorted({ext for fmt in RASTER_FORMATS for ext in fmt.extensions}))
        return (
            f"Could not open {location!r} as a raster dataset. "
            f"No registered extension matched. Known extensions include: {known}. "
            f"Errors: {'; '.join(errors) or 'none'}"
        )
    tried = ", ".join(drivers) if drivers else "automatic detection"
    return (
        f"Could not open {location!r} as a raster dataset with driver(s) {tried}. "
        f"Errors: {'; '.join(errors)}"
    )


def _select_subdataset(
    dataset: Any,
    subdataset: Union[int, str, None],
) -> Tuple[Any, Optional[str]]:
    pairs = dataset.GetSubDatasets() or []
    if subdataset is None:
        if dataset.RasterCount == 0 and pairs:
            names = [f"{index}: {desc}" for index, (_, desc) in enumerate(pairs)]
            raise RasterOpenError(
                "Dataset exposes GDAL subdatasets but no grid is opened at the root. "
                f"Pass subdataset= with an index or name. Available: {'; '.join(names)}"
            )
        return dataset, None

    if not pairs:
        raise RasterOpenError("Dataset has no subdatasets to select.")

    if isinstance(subdataset, bool) or not isinstance(subdataset, (int, str)):
        raise TypeError("subdataset must be an integer index or subdataset name string")

    if isinstance(subdataset, int):
        if subdataset < 0 or subdataset >= len(pairs):
            raise RasterOpenError(
                f"Subdataset index {subdataset} is outside 0..{len(pairs) - 1}."
            )
        name, _ = pairs[subdataset]
    else:
        exact = [item for item in pairs if item[0] == subdataset]
        if not exact:
            folded = [item for item in pairs if subdataset.lower() in item[0].lower()]
            if len(folded) == 1:
                exact = folded
        if len(exact) != 1:
            raise RasterOpenError(
                f"Subdataset {subdataset!r} was not found. "
                f"Available names: {[item[0] for item in pairs]}"
            )
        name, _ = exact[0]

    gdal, _ = _gdal_modules()
    opened = gdal.Open(str(name))
    if opened is None:
        raise RasterOpenError(f"Could not open subdataset {name!r}.")
    return opened, str(name)


def _bounds_from_geotransform(
    geotransform: Sequence[float],
    width: int,
    height: int,
) -> Bounds:
    corners = (
        _pixel_to_world(geotransform, 0, 0),
        _pixel_to_world(geotransform, width, 0),
        _pixel_to_world(geotransform, 0, height),
        _pixel_to_world(geotransform, width, height),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def _pixel_to_world(
    geotransform: Sequence[float],
    pixel: float,
    line: float,
) -> Tuple[float, float]:
    x = geotransform[0] + pixel * geotransform[1] + line * geotransform[2]
    y = geotransform[3] + pixel * geotransform[4] + line * geotransform[5]
    return x, y


def _crs_from_dataset(dataset: Any) -> Optional[CRS]:
    _, osr = _gdal_modules()
    spatial_ref = None
    if hasattr(dataset, "GetSpatialRef"):
        try:
            spatial_ref = dataset.GetSpatialRef()
        except RuntimeError:
            spatial_ref = None
    wkt = None
    if spatial_ref is not None:
        try:
            cloned = spatial_ref.Clone()
        except RuntimeError:
            cloned = None
        if cloned is not None:
            traditional = getattr(osr, "OAMS_TRADITIONAL_GIS_ORDER", None)
            if traditional is not None and hasattr(cloned, "SetAxisMappingStrategy"):
                try:
                    cloned.SetAxisMappingStrategy(traditional)
                except RuntimeError:
                    pass
            try:
                wkt = cloned.ExportToWkt() or None
            except RuntimeError:
                wkt = None
            authority = code = None
            try:
                authority = cloned.GetAuthorityName(None)
                code = cloned.GetAuthorityCode(None)
            except RuntimeError:
                authority = code = None
            if authority and code and str(authority).upper() == "EPSG":
                try:
                    return CRS(int(code))
                except (TypeError, ValueError):
                    pass
            if wkt:
                try:
                    return CRS.from_wkt(wkt)
                except (RuntimeError, ValueError):
                    return None
    projection = dataset.GetProjection()
    if projection:
        try:
            return CRS.from_wkt(projection)
        except (RuntimeError, ValueError):
            return None
    return None


def _window_or_full(
    window: Optional[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    if window is None:
        return 0, 0, width, height
    if len(window) != 4:
        raise ValueError("window must be (xoff, yoff, xsize, ysize)")
    xoff, yoff, xsize, ysize = (int(value) for value in window)
    if xsize <= 0 or ysize <= 0:
        raise ValueError("window width and height must be positive")
    if xoff < 0 or yoff < 0 or xoff + xsize > width or yoff + ysize > height:
        raise ValueError("window falls outside the raster extent")
    return xoff, yoff, xsize, ysize


def _read_array(
    band: Any,
    xoff: int,
    yoff: int,
    xsize: int,
    ysize: int,
) -> np.ndarray:
    try:
        array = band.ReadAsArray(xoff, yoff, xsize, ysize)
        if array is not None:
            return array
    except Exception:
        pass
    gdal, _ = _gdal_modules()
    dtype = _GDAL_TO_NUMPY.get(band.DataType)
    if dtype is None:
        raise RasterOpenError(
            f"Unsupported GDAL data type {gdal.GetDataTypeName(band.DataType)} for NumPy I/O."
        )
    raw = band.ReadRaster(xoff, yoff, xsize, ysize, buf_xsize=xsize, buf_ysize=ysize)
    return np.frombuffer(raw, dtype=dtype).reshape(ysize, xsize)


def _write_array(
    destination: Path,
    array: np.ndarray,
    driver_name: str,
    crs: Any,
    geotransform: Sequence[float],
    nodata: Optional[Union[float, Sequence[float]]],
    creation_options: Dict[str, str],
    metadata: Dict[str, Any],
    band_metadata: Sequence[Dict[str, Any]],
) -> None:
    gdal, osr = _gdal_modules()
    driver = gdal.GetDriverByName(driver_name)
    if not _driver_can_write(driver):
        raise DriverUnavailableError(
            f"GDAL driver {driver_name!r} cannot create raster datasets in this build."
        )

    if array.ndim == 2:
        bands = 1
        planes = [array]
    elif array.ndim == 3:
        bands = int(array.shape[0])
        planes = [array[index] for index in range(bands)]
    else:
        raise RasterWriteError(f"Expected a 2-D or 3-D array, got shape {array.shape!r}.")

    rows, cols = planes[0].shape
    gdal_type = _numpy_to_gdal(planes[0].dtype)
    options = [f"{key}={value}" for key, value in creation_options.items()]
    if _driver_can_create(driver):
        try:
            dataset = driver.Create(str(destination), cols, rows, bands, gdal_type, options=options)
        except Exception as exc:
            raise RasterWriteError(
                f"Could not create {driver_name} dataset {destination}: {exc}"
            ) from exc
        if dataset is None:
            raise RasterWriteError(f"Could not create {driver_name} dataset {destination}")
        try:
            _populate_dataset(
                dataset,
                planes,
                gdal_type,
                geotransform,
                crs,
                osr,
                metadata,
                nodata,
                band_metadata,
            )
        finally:
            dataset = None
        return

    if not _driver_can_createcopy(driver):
        raise DriverUnavailableError(
            f"GDAL driver {driver_name!r} cannot create raster datasets in this build."
        )
    mem_driver = gdal.GetDriverByName("MEM")
    if mem_driver is None:
        raise DriverUnavailableError(
            f"GDAL driver {driver_name!r} requires CreateCopy but the MEM driver is missing."
        )
    mem_dataset = mem_driver.Create("", cols, rows, bands, gdal_type)
    if mem_dataset is None:
        raise RasterWriteError("Could not create in-memory raster for CreateCopy output.")
    try:
        _populate_dataset(
            mem_dataset,
            planes,
            gdal_type,
            geotransform,
            crs,
            osr,
            metadata,
            nodata,
            band_metadata,
        )
        copy = driver.CreateCopy(str(destination), mem_dataset, options=options)
    except Exception as exc:
        raise RasterWriteError(
            f"Could not copy raster to {driver_name} dataset {destination}: {exc}"
        ) from exc
    finally:
        mem_dataset = None
    if copy is None:
        raise RasterWriteError(f"Could not copy raster to {driver_name} dataset {destination}")
    copy.FlushCache()
    copy = None


def _write_copy(
    destination: Path,
    source: RasterDataset,
    driver_name: str,
    creation_options: Optional[Union[Sequence[str], Dict[str, Any]]],
    overwrite: bool,
) -> str:
    if source._dataset is None:
        raise RasterWriteError("Cannot copy a closed RasterDataset.")
    gdal, _ = _gdal_modules()
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise DriverUnavailableError(f"GDAL driver {driver_name!r} is not available.")
    if not _driver_can_write(driver):
        raise DriverUnavailableError(
            f"GDAL driver {driver_name!r} cannot create or copy datasets in this build."
        )
    options = [f"{key}={value}" for key, value in _option_map(creation_options).items()]
    if overwrite and destination.exists():
        driver.Delete(str(destination))
    try:
        copy = driver.CreateCopy(str(destination), source._dataset, options=options)
    except Exception as exc:
        raise RasterWriteError(f"Could not copy raster to {destination}: {exc}") from exc
    if copy is None:
        raise RasterWriteError(f"Could not copy raster to {destination}")
    copy.FlushCache()
    copy = None
    return str(destination)


def _populate_dataset(
    dataset: Any,
    planes: Sequence[np.ndarray],
    gdal_type: int,
    geotransform: Sequence[float],
    crs: Any,
    osr: Any,
    metadata: Dict[str, Any],
    nodata: Optional[Union[float, Sequence[float]]],
    band_metadata: Sequence[Dict[str, Any]],
) -> None:
    dataset.SetGeoTransform(tuple(float(value) for value in geotransform))
    if crs is not None:
        dataset.SetProjection(_spatial_reference_from_crs(crs, osr).ExportToWkt())
    for key, value in metadata.items():
        dataset.SetMetadataItem(str(key), str(value))
    nodata_values = _nodata_list(nodata, len(planes))
    for index, plane in enumerate(planes, start=1):
        band = dataset.GetRasterBand(index)
        _write_band_array(band, plane, gdal_type)
        if nodata_values[index - 1] is not None:
            band.SetNoDataValue(float(nodata_values[index - 1]))
        if index - 1 < len(band_metadata):
            for key, value in band_metadata[index - 1].items():
                band.SetMetadataItem(str(key), str(value))
    dataset.FlushCache()


def _write_band_array(band: Any, array: np.ndarray, gdal_type: int) -> None:
    if array.shape != (band.YSize, band.XSize):
        raise RasterWriteError(
            f"Array shape {array.shape!r} does not match band size {(band.YSize, band.XSize)!r}."
        )
    contiguous = np.ascontiguousarray(array)
    try:
        band.WriteArray(contiguous)
        return
    except Exception:
        pass
    band.WriteRaster(
        0,
        0,
        band.XSize,
        band.YSize,
        contiguous.tobytes(),
        buf_xsize=band.XSize,
        buf_ysize=band.YSize,
        buf_type=gdal_type,
    )


def _array_from_data(data: Any) -> Tuple[np.ndarray, int]:
    if isinstance(data, np.ndarray):
        array = data
    elif hasattr(data, "values") and hasattr(data, "dims"):
        array = _array_from_xarray(data)
    else:
        raise RasterWriteError(
            "data must be a NumPy array, xarray DataArray/Dataset, or RasterDataset."
        )
    if array.ndim == 2:
        return array, 1
    if array.ndim == 3:
        return array, int(array.shape[0])
    raise RasterWriteError(f"Expected a 2-D or 3-D array, got shape {array.shape!r}.")


def _array_from_xarray(data: Any) -> np.ndarray:
    name = type(data).__name__
    if name == "Dataset":
        variables = list(getattr(data, "data_vars", {}))
        if not variables:
            raise RasterWriteError("xarray Dataset has no data variables.")
        if len(variables) != 1:
            raise RasterWriteError(
                f"xarray Dataset has {len(variables)} variables; pass one DataArray instead."
            )
        data = data[variables[0]]
    dims = tuple(getattr(data, "dims", ()))
    values = np.asarray(data.values)
    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise RasterWriteError(f"xarray object must be 2-D or 3-D, got shape {values.shape!r}.")
    band_dim = None
    for candidate in ("band", "bands"):
        if candidate in dims:
            band_dim = candidate
            break
    if band_dim is None:
        for candidate in dims:
            if candidate not in {"x", "y", "lat", "lon", "latitude", "longitude", "time", "t"}:
                band_dim = candidate
                break
    if band_dim is None:
        raise RasterWriteError(
            "Could not determine the band dimension on the xarray object."
        )
    axis = dims.index(band_dim)
    return np.moveaxis(values, axis, 0)


def _geotransform_or_default(
    geotransform: Optional[Sequence[float]],
    rows: int,
    cols: int,
) -> Tuple[float, float, float, float, float, float]:
    if geotransform is not None:
        if len(geotransform) != 6:
            raise ValueError("geotransform must contain six values.")
        return tuple(float(value) for value in geotransform)
    return (0.0, 1.0, 0.0, float(rows), 0.0, -1.0)


def _numpy_to_gdal(dtype: np.dtype) -> int:
    _gdal_modules()
    gdal_type = _NUMPY_TO_GDAL.get(np.dtype(dtype))
    if gdal_type is None:
        raise RasterWriteError(f"NumPy dtype {dtype!r} is not supported for GDAL raster output.")
    return gdal_type


def _nodata_list(
    nodata: Optional[Union[float, Sequence[float]]],
    bands: int,
) -> List[Optional[float]]:
    if nodata is None:
        return [None] * bands
    if isinstance(nodata, (int, float)) or (
        isinstance(nodata, float) and math.isnan(nodata)
    ):
        return [float(nodata)] * bands
    values = [float(value) for value in nodata]
    if len(values) != bands:
        raise ValueError(f"Expected {bands} nodata values, got {len(values)}.")
    return values


def _spatial_reference_from_crs(crs_value: Any, osr: Any) -> Any:
    if isinstance(crs_value, CRS):
        wkt = crs_value.wkt
    elif hasattr(crs_value, "to_wkt"):
        wkt = crs_value.to_wkt()
    else:
        wkt = CRS(crs_value).wkt
    if not wkt:
        raise RasterWriteError(f"CRS {crs_value!r} has no WKT definition to write.")
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(wkt)
    if hasattr(spatial_ref, "SetAxisMappingStrategy"):
        spatial_ref.SetAxisMappingStrategy(getattr(osr, "OAMS_TRADITIONAL_GIS_ORDER", 0))
    return spatial_ref


def _choose_driver(path: Path, explicit: Optional[str]) -> str:
    gdal, _ = _gdal_modules()
    if explicit:
        driver = gdal.GetDriverByName(explicit)
        if not _driver_can_write(driver):
            raise DriverUnavailableError(
                f"GDAL driver {explicit!r} is not available or cannot create rasters."
            )
        return explicit
    suffix = path.suffix.lower()
    candidates: List[str] = []
    for fmt in RASTER_FORMATS:
        if suffix in fmt.extensions:
            candidates.extend(fmt.drivers)
    if not candidates:
        known = ", ".join(sorted({ext for fmt in RASTER_FORMATS for ext in fmt.extensions}))
        raise RasterWriteError(
            f"No raster writer is registered for {path.name!r}. Known extensions include: {known}"
        )
    for name in candidates:
        if _driver_can_write(gdal.GetDriverByName(name)):
            return name
    raise DriverUnavailableError(
        f"None of the GDAL drivers for {path.name!r} can create a dataset: {', '.join(candidates)}"
    )


def _driver_can_create(driver: Any) -> bool:
    return driver is not None and driver.GetMetadataItem("DCAP_CREATE") == "YES"


def _driver_can_createcopy(driver: Any) -> bool:
    return driver is not None and driver.GetMetadataItem("DCAP_CREATECOPY") == "YES"


def _driver_can_write(driver: Any) -> bool:
    return _driver_can_create(driver) or _driver_can_createcopy(driver)


def _prepare_destination(path: Path, driver_name: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}. Pass overwrite=True to replace it.")
    gdal, _ = _gdal_modules()
    driver = gdal.GetDriverByName(driver_name)
    try:
        if driver is not None:
            driver.Delete(str(path))
    except Exception:
        pass
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _option_map(options: Optional[Union[Sequence[str], Dict[str, Any]]]) -> Dict[str, str]:
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(key): str(value) for key, value in options.items()}
    mapped: Dict[str, str] = {}
    for item in options:
        text = str(item)
        if "=" not in text:
            raise ValueError(f"creation option must be KEY=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        mapped[key] = value
    return mapped
