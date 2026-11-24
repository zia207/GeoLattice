"""Readers/writers for HDF4/HDF5, NetCDF, Zarr and Sentinel SAFE products."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import xarray as xr


def list_subdatasets(path: str | PathLike) -> list[str]:
    """GDAL subdataset paths of an HDF/NetCDF file (e.g. MODIS HDF-EOS layers)."""
    import rasterio

    with rasterio.open(path) as src:
        return list(src.subdatasets)


def read_hdf(path: str | PathLike, layer: str | None = None, *, chunks: Any = None,
             masked: bool = True) -> xr.DataArray | dict[str, str]:
    """Read one layer of an HDF4/HDF5 file through GDAL.

    With ``layer=None`` returns ``{short_name: gdal_path}`` for all layers so you can pick one.
    ``layer`` may be the short name (``"sur_refl_b01"``) or a full GDAL subdataset path.
    MODIS sinusoidal CRS and transforms are preserved.
    """
    import rioxarray

    subs = list_subdatasets(path)
    names = {s.split(":")[-1]: s for s in subs}
    if layer is None:
        return names
    gdal_path = names.get(layer, layer)
    da = rioxarray.open_rasterio(gdal_path, chunks=chunks, masked=masked)
    return da.squeeze("band", drop=True) if "band" in da.dims and da.sizes["band"] == 1 else da


def read_netcdf(path: str | PathLike, variable: str | None = None, *, crs: Any = None,
                x_dim: str | None = None, y_dim: str | None = None, chunks: Any = None):
    """Open NetCDF with spatial dims and CRS set for rioxarray (CF ``grid_mapping`` honored)."""
    ds = xr.open_dataset(path, chunks=chunks, decode_coords="all")
    obj = ds[variable] if variable else ds
    xd = x_dim or next((d for d in ("x", "lon", "longitude") if d in obj.dims), None)
    yd = y_dim or next((d for d in ("y", "lat", "latitude") if d in obj.dims), None)
    if xd and yd:
        obj = obj.rio.set_spatial_dims(x_dim=xd, y_dim=yd)
    if obj.rio.crs is None:
        if crs is not None:
            obj = obj.rio.write_crs(crs)
        elif xd in ("lon", "longitude"):
            obj = obj.rio.write_crs(4326)
    return obj


def write_zarr(obj: xr.Dataset | xr.DataArray, store: str | PathLike, *, mode: str = "w",
               consolidated: bool = True, **kwargs: Any) -> None:
    """Write to Zarr, keeping CRS (``spatial_ref``) so it round-trips with :func:`read_zarr`."""
    from ..backend import to_cpu

    obj = to_cpu(obj)
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset(name=obj.name or "data")
    obj.to_zarr(store, mode=mode, consolidated=consolidated, **kwargs)


def read_zarr(store: str | PathLike, *, chunks: Any = "auto", consolidated: bool | None = None):
    ds = xr.open_zarr(store, chunks=chunks, consolidated=consolidated, decode_coords="all")
    return ds


def read_safe(safe_dir: str | PathLike) -> dict[str, Any]:
    """Index a Sentinel-1/2 ``.SAFE`` directory.

    Returns ``{"mission", "measurement": {pol_or_band: path}, "annotation": {...},
    "calibration": {...}, "noise": {...}, "manifest": path}``. For Sentinel-1 the keys
    are polarizations (``vv``, ``vh``); for Sentinel-2 they are band names (``B04``...).
    """
    root = Path(safe_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    out: dict[str, Any] = {"mission": root.name[:3], "manifest": root / "manifest.safe",
                           "measurement": {}, "annotation": {}, "calibration": {}, "noise": {}}
    if root.name.startswith("S1"):
        for tif in sorted((root / "measurement").glob("*.tif*")):
            pol = tif.stem.split("-")[3]
            out["measurement"][pol] = tif
        for xml in sorted((root / "annotation").glob("*.xml")):
            out["annotation"][xml.stem.split("-")[3]] = xml
        cal = root / "annotation" / "calibration"
        for xml in sorted(cal.glob("calibration-*.xml")):
            out["calibration"][xml.stem.split("-")[4]] = xml
        for xml in sorted(cal.glob("noise-*.xml")):
            out["noise"][xml.stem.split("-")[4]] = xml
    else:  # Sentinel-2
        for jp2 in sorted(root.glob("GRANULE/*/IMG_DATA/**/*.jp2")):
            parts = jp2.stem.split("_")
            band = parts[-2] if parts[-1].endswith("m") else parts[-1]
            out["measurement"].setdefault(band, jp2)
    return out
