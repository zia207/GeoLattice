"""Parallel and chunked readers and writers for vector and raster data.

Readers default to ``compute="cpu"`` (data lands in host memory); pass ``compute="gpu"``
to get CuPy-backed raster chunks straight away.
"""

from __future__ import annotations

import glob
import os
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._backend import Compute, get_defaults, has, resolve_compute, to_cpu, to_gpu
from ._partition import default_npartitions, plan_chunks, spatial_partition

_PARQUET = (".parquet", ".geoparquet", ".pq")


def _expand(path: str | PathLike | list) -> list[str]:
    if isinstance(path, (list, tuple)):
        return [str(p) for p in path]
    p = str(path)
    if any(c in p for c in "*?["):
        files = sorted(glob.glob(p))
        if not files:
            raise FileNotFoundError(f"No files match {p!r}")
        return files
    if os.path.isdir(p):
        files = sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                       if f.lower().endswith(_PARQUET))
        if files:
            return files
    return [p]


# --------------------------------------------------------------------------- vector
def read_vector(path: str | PathLike | list, *, npartitions: int | None = None,
                lazy: bool = False, columns: list[str] | None = None,
                bbox: tuple[float, float, float, float] | None = None,
                layer: str | int | None = None, compute: Compute = "cpu", **kwargs: Any) -> Any:
    """Read vector data in parallel.

    ``path`` may be a file, a glob (``"tiles/*.parquet"``), a directory of GeoParquet
    files or a list of paths.

    * GeoParquet files are read one task per file.
    * A single OGR file (GeoPackage, FlatGeobuf, Shapefile ...) is read in
      ``npartitions`` row ranges in parallel with ``pyogrio``.
    * ``lazy=True`` returns a ``dask_geopandas`` frame (needs dask-geopandas).

    Without ``lazy`` the result is one GeoDataFrame.
    """
    import dask
    import geopandas as gpd

    resolve_compute(compute, op="io.read_vector", gpu_available=False)
    files = _expand(path)
    is_parquet = all(f.lower().endswith(_PARQUET) for f in files)

    if lazy:
        if not has("dask_geopandas"):
            raise ImportError("lazy=True needs dask-geopandas: pip install geolattice[distributed]")
        import dask_geopandas

        if is_parquet:
            return dask_geopandas.read_parquet(files if len(files) > 1 else files[0],
                                               columns=columns, **kwargs)
        if len(files) > 1:
            raise ValueError("lazy reading of several non-Parquet files is not supported")
        return dask_geopandas.read_file(files[0], npartitions=npartitions or os.cpu_count() or 1,
                                        layer=layer, **kwargs)

    if is_parquet:
        tasks = [dask.delayed(gpd.read_parquet)(f, columns=columns, bbox=bbox, **kwargs)
                 for f in files]
    elif len(files) == 1 and bbox is None:
        import pyogrio

        n_rows = int(pyogrio.read_info(files[0], layer=layer)["features"])
        n = default_npartitions(n_rows, npartitions or get_defaults().npartitions)
        edges = np.linspace(0, n_rows, n + 1).astype(int)
        tasks = [dask.delayed(gpd.read_file)(files[0], layer=layer, columns=columns,
                                             skip_features=int(a), max_features=int(b - a),
                                             engine="pyogrio", **kwargs)
                 for a, b in zip(edges[:-1], edges[1:]) if b > a]
    else:
        tasks = [dask.delayed(gpd.read_file)(f, layer=layer, columns=columns, bbox=bbox,
                                             engine="pyogrio", **kwargs) for f in files]

    import pandas as pd

    parts = list(dask.compute(*tasks))

    out = pd.concat(parts, ignore_index=not is_parquet or len(parts) > 1)
    if out.crs is None and parts and parts[0].crs is not None:
        out = out.set_crs(parts[0].crs)
    return out


def write_vector(gdf: Any, path: str | PathLike, *, npartitions: int | None = None,
                 partition_on: str | None = None, spatial: bool = True, **kwargs: Any) -> str:
    """Write a partitioned GeoParquet dataset (one file per partition, written in parallel).

    ``path`` ending in ``.parquet`` with ``npartitions`` of 1 writes a single file.
    ``partition_on`` writes Hive-style subdirectories (``col=value/part.0.parquet``).
    ``spatial`` orders rows along a Hilbert curve so each file covers a compact area, and a
    GeoParquet 1.1 bbox covering column is written (``write_covering_bbox=True``), so
    readers can skip files and row groups by bounding box.
    """
    import dask

    # a bbox covering column lets readers (including read_vector(bbox=...)) skip row groups
    kwargs.setdefault("write_covering_bbox", True)
    if type(gdf).__module__.startswith("dask_geopandas"):
        gdf.to_parquet(str(path), partition_on=partition_on, **kwargs)
        return str(path)

    path = Path(path)
    if path.suffix.lower() in _PARQUET and (npartitions or 1) == 1 and partition_on is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(path, **kwargs)
        return str(path)

    path.mkdir(parents=True, exist_ok=True)
    groups = [(None, gdf)] if partition_on is None else list(gdf.groupby(partition_on, sort=True))
    tasks = []
    for key, sub in groups:
        out_dir = path if key is None else path / f"{partition_on}={key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        data = sub if key is None else sub.drop(columns=partition_on)
        parts = spatial_partition(data, npartitions, method="hilbert" if spatial else "none")
        for i, part in enumerate(parts):
            tasks.append(dask.delayed(part.to_parquet)(out_dir / f"part.{i}.parquet", **kwargs))
    dask.compute(*tasks)
    return str(path)


# --------------------------------------------------------------------------- raster
def _place(obj: Any, compute: Compute, op: str) -> Any:
    used = resolve_compute(compute, op=op)
    return to_gpu(obj) if used == "gpu" else obj


def read_raster(path: str | PathLike, *, chunks: Any = "auto", compute: Compute = "cpu",
                masked: bool = True, squeeze: bool = True, **kwargs: Any) -> Any:
    """Open a raster lazily as a Dask-backed DataArray (``rioxarray.open_rasterio``).

    ``chunks="auto"`` picks spatial chunks with
    :func:`~geolattice.distributed._partition.plan_chunks` (bands kept whole).
    ``squeeze`` drops a single-band ``band`` dimension. ``compute="gpu"`` makes each chunk
    a CuPy array when computed.
    """
    import rioxarray

    da = rioxarray.open_rasterio(path, chunks=True, masked=masked, lock=False, **kwargs)
    # rioxarray records an identity scale and offset as attributes; written to Zarr or NetCDF
    # they make xarray decode the data as float64 on reading, so drop them when they do nothing
    if da.attrs.get("scale_factor", 1.0) == 1.0 and da.attrs.get("add_offset", 0.0) == 0.0:
        da.attrs.pop("scale_factor", None)
        da.attrs.pop("add_offset", None)
    if squeeze and "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    if chunks == "auto":
        used = resolve_compute(compute, op="io.read_raster")
        da = da.chunk(dict(zip(da.dims, plan_chunks(da.shape, da.dtype, compute=used))))
    elif chunks is not None:
        da = da.chunk(chunks)
    return _place(da, compute, "io.read_raster")


def read_zarr(store: str | PathLike, *, chunks: Any = None, compute: Compute = "cpu",
              **kwargs: Any) -> Any:
    """Open a Zarr store lazily as an ``xarray.Dataset`` (chunks default to the store's)."""
    import xarray as xr

    ds = xr.open_zarr(store, chunks={} if chunks is None else chunks, **kwargs)
    return _place(ds, compute, "io.read_zarr")


def read_stac(items: Any, bands: str | list[str] | None = None, *, chunks: dict | None = None,
              resolution: float | None = None, crs: Any = None,
              bbox: tuple[float, float, float, float] | None = None, compute: Compute = "cpu",
              **kwargs: Any) -> Any:
    """Load STAC items lazily into an ``xarray.Dataset`` with ``odc.stac.load``.

    ``chunks`` defaults to 2048 x 2048 spatial chunks. Extra ``kwargs`` go to
    ``odc.stac.load`` (``groupby``, ``resampling``, ``dtype`` ...).
    """
    from odc.stac import load

    opts: dict[str, Any] = dict(bands=bands, chunks=chunks or {"x": 2048, "y": 2048}, **kwargs)
    if resolution is not None:
        opts["resolution"] = resolution
    if crs is not None:
        opts["crs"] = crs
    if bbox is not None:
        opts["bbox"] = bbox
    return _place(load(items, **opts), compute, "io.read_stac")


def write_raster(obj: Any, path: str | PathLike, *,
                 format: Literal["zarr", "gtiff", "cog"] = "zarr", name: str | None = None,
                 compress: str = "DEFLATE", **kwargs: Any) -> str:
    """Write a DataArray/Dataset, computing chunks in parallel.

    * ``zarr``: chunked store, written chunk by chunk (best for large or cluster jobs).
    * ``gtiff``: tiled GeoTIFF written chunk by chunk under a lock.
    * ``cog``: tiled GeoTIFF as above, then converted with :func:`geolattice.web.cog.to_cog`.

    GPU data is moved to host memory chunk by chunk first.
    """
    import xarray as xr

    obj = to_cpu(obj)
    path = Path(path)
    if format == "zarr":
        ds = obj.to_dataset(name=name or obj.name or "data") if isinstance(obj, xr.DataArray) else obj
        ds.to_zarr(path, mode="w", **kwargs)
        return str(path)
    if format not in ("gtiff", "cog"):
        raise ValueError("format must be 'zarr', 'gtiff' or 'cog'")

    import threading

    import rioxarray  # noqa: F401

    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if format == "gtiff" else path.with_suffix(".tmp.tif")
    lock: Any = threading.Lock()
    try:
        from distributed import Lock, default_client

        default_client()
        lock = Lock("geolattice-write-raster")
    except Exception:
        pass
    obj.rio.to_raster(target, tiled=True, compress=compress, lock=lock, **kwargs)
    if format == "cog":
        from ..web.cog import to_cog

        to_cog(target, path, compress=compress)
        target.unlink(missing_ok=True)
    return str(path)
