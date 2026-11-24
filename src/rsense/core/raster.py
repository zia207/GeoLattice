"""Raster I/O built on rioxarray, returning georeferenced xarray objects."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any

import xarray as xr


def read(
    path: str | PathLike,
    *,
    chunks: Any = None,
    band_names: Sequence[str] | None = None,
    masked: bool = True,
    as_dataset: bool = False,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Read a raster (GeoTIFF/COG/JP2/HDF subdataset/NetCDF) as a georeferenced DataArray.

    Parameters
    ----------
    path : file path or URL (``/vsicurl/`` and ``s3://`` paths are supported by GDAL).
    chunks : Dask chunking, e.g. ``"auto"`` or ``{"x": 1024, "y": 1024}``. ``None`` loads eagerly.
    band_names : names for the ``band`` coordinate, e.g. ``["blue", "green", "red", "nir"]``.
        If omitted, GDAL band descriptions are used when present.
    masked : convert nodata to NaN.
    as_dataset : return a Dataset with one variable per band (handy for index formulas).
    """
    import rioxarray

    da = rioxarray.open_rasterio(path, chunks=chunks, masked=masked, **kwargs)
    if isinstance(da, list):
        raise ValueError("File holds several subdatasets; pass one explicitly via its GDAL path.")  # noqa: TRY004
    if isinstance(da, xr.Dataset):
        return da
    names = list(band_names) if band_names else _descriptions(da)
    if names:
        if len(names) != da.sizes.get("band", 1):
            raise ValueError(f"{len(names)} band names given for {da.sizes.get('band', 1)} bands")
        da = da.assign_coords(band=names)
    if as_dataset:
        return da.to_dataset(dim="band")
    return da


def _descriptions(da: xr.DataArray) -> list[str] | None:
    desc = da.attrs.get("long_name")
    if isinstance(desc, (list, tuple)) and all(desc) and len(set(desc)) == len(desc):
        return [str(d) for d in desc]
    return None


def write(
    obj: xr.DataArray | xr.Dataset,
    path: str | PathLike,
    *,
    cog: bool = True,
    compress: str = "DEFLATE",
    **kwargs: Any,
) -> None:
    """Write a DataArray/Dataset to GeoTIFF, as a Cloud-Optimized GeoTIFF by default.

    GPU-resident data is moved to host memory automatically.
    """
    from ..backend import to_cpu

    obj = to_cpu(obj)
    if isinstance(obj, xr.Dataset):
        obj = obj.to_array(dim="band")
    if obj.rio.crs is None:
        raise ValueError("Object has no CRS; set one with `.rio.write_crs(...)` before writing.")
    opts: dict[str, Any] = {"compress": compress, **kwargs}
    if cog:
        opts.setdefault("driver", "COG")
    else:
        opts.setdefault("tiled", True)
    obj.rio.to_raster(path, **opts)


def reproject(obj: xr.DataArray | xr.Dataset, dst_crs: Any, *, resolution: float | None = None,
              resampling: str = "bilinear", **kwargs: Any):
    """Reproject to ``dst_crs`` (EPSG code, WKT or pyproj CRS)."""
    from rasterio.enums import Resampling

    from ..backend import to_cpu

    return to_cpu(obj).rio.reproject(dst_crs, resolution=resolution,
                                     resampling=Resampling[resampling], **kwargs)


def clip(obj: xr.DataArray | xr.Dataset, geometries: Any, *, crs: Any = None,
         all_touched: bool = False, drop: bool = True):
    """Clip to geometries (a GeoDataFrame, GeoSeries or list of shapely/GeoJSON geometries)."""
    from ..backend import to_cpu

    obj = to_cpu(obj)
    if hasattr(geometries, "geometry"):  # GeoDataFrame / GeoSeries
        crs = crs or geometries.crs
        geometries = list(geometries.geometry)
    return obj.rio.clip(geometries, crs=crs or obj.rio.crs, all_touched=all_touched, drop=drop)
