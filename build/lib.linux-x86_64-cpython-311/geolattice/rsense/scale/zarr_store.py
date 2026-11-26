"""Zarr data cubes: write once, read lazily anywhere (local disk or cloud object storage)."""

from __future__ import annotations

from os import PathLike
from typing import Any

import xarray as xr


def to_zarr_cube(obj: xr.Dataset | xr.DataArray, store: str | PathLike, *,
                 chunks: dict[str, int] | None = None, append_dim: str | None = None,
                 compressor: str = "zstd", level: int = 3, **kwargs: Any) -> None:
    """Write (or append along ``append_dim``) a cube to Zarr with sensible chunks/compression.

    Default chunks: 512×512 spatially, full time — efficient for both maps and time series.
    """
    from ..backend import to_cpu

    obj = to_cpu(obj)
    ds = obj.to_dataset(name=obj.name or "data") if isinstance(obj, xr.DataArray) else obj
    if chunks is None:
        chunks = {d: (512 if d in ("x", "y") else -1) for d in ds.dims}
    ds = ds.chunk({d: c for d, c in chunks.items() if d in ds.dims})
    for v in ds.variables:
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding.pop("preferred_chunks", None)
    if append_dim:
        ds.to_zarr(store, append_dim=append_dim, **kwargs)
    else:
        ds.to_zarr(store, mode="w", **kwargs)


def open_cube(store: str | PathLike, *, chunks: Any = {}, **kwargs: Any) -> xr.Dataset:  # noqa: B006
    """Open a Zarr cube lazily with its native chunks and CRS."""
    return xr.open_zarr(store, chunks=chunks, decode_coords="all", **kwargs)
