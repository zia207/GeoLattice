"""Tiled processing of large rasters with overlap (halo) for moving-window operations.

:func:`apply_tiled` runs a function block by block on the active backend. Each
block is padded by ``overlap`` pixels so filters near tile edges see their full
neighbourhood, then cropped back. With ``overlap`` at least the filter radius,
the result matches running the function on the whole image at once, while GPU
memory only ever holds one tile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from ..backend import get_backend, get_xp, to_cpu


@dataclass(frozen=True)
class Window:
    """A tile: the ``core`` region it owns and the ``padded`` region it reads."""

    row: slice
    col: slice
    pad_row: slice
    pad_col: slice

    @property
    def inner(self) -> tuple[slice, slice]:
        """Core region expressed relative to the padded block."""
        r0 = self.row.start - self.pad_row.start
        c0 = self.col.start - self.pad_col.start
        return (slice(r0, r0 + (self.row.stop - self.row.start)),
                slice(c0, c0 + (self.col.stop - self.col.start)))


def iter_windows(height: int, width: int, tile: int | tuple[int, int] = 512,
                 overlap: int = 0) -> Iterator[Window]:
    """Yield non-overlapping core windows, each with a padded read window."""
    th, tw = (tile, tile) if isinstance(tile, int) else tile
    if th <= 0 or tw <= 0 or overlap < 0:
        raise ValueError("tile must be positive and overlap non-negative")
    for r0 in range(0, height, th):
        r1 = min(r0 + th, height)
        for c0 in range(0, width, tw):
            c1 = min(c0 + tw, width)
            yield Window(
                slice(r0, r1), slice(c0, c1),
                slice(max(r0 - overlap, 0), min(r1 + overlap, height)),
                slice(max(c0 - overlap, 0), min(c1 + overlap, width)),
            )


def auto_tile_size(dtype_bytes: int = 4, n_layers: int = 1, safety: float = 0.25,
                   default: int = 1024, max_tile: int = 8192) -> int:
    """Pick a square tile edge that fits in a fraction of free GPU memory (``default`` on CPU)."""
    from ..backend import free_gpu_memory_bytes

    free = free_gpu_memory_bytes()
    if free is None:
        return default
    edge = int(np.sqrt(free * safety / (dtype_bytes * max(n_layers, 1) * 4)))  # 4x for temporaries
    return int(np.clip(edge // 256 * 256, 256, max_tile))


def apply_tiled(
    data: Any,
    func: Callable[[Any], Any],
    *,
    tile: int | tuple[int, int] | None = 1024,
    overlap: int = 0,
    dtype: Any = None,
) -> Any:
    """Apply ``func`` to the last two axes of ``data`` tile by tile on the active backend.

    ``func`` receives a NumPy or CuPy block shaped ``(..., rows, cols)`` and must
    return a block with the same trailing spatial shape. Leading dimensions
    (bands, time) are passed through whole.

    Accepts NumPy arrays, DataArrays (coords are preserved) and Dask-backed
    DataArrays (dispatched to ``dask.array.map_overlap`` and kept lazy).
    The result is returned in host memory.
    """
    if isinstance(data, xr.DataArray):
        out = apply_tiled(data.data, func, tile=tile, overlap=overlap, dtype=dtype)
        return data.copy(data=out)

    if hasattr(data, "map_overlap") and hasattr(data, "chunks"):  # dask array
        return _apply_dask(data, func, overlap=overlap, dtype=dtype)

    arr = to_cpu(data)
    if arr.ndim < 2:
        raise ValueError("data must have at least two (spatial) dimensions")
    xp = get_xp()
    tile = tile or auto_tile_size(arr.dtype.itemsize, int(np.prod(arr.shape[:-2])) or 1)
    out = None
    for w in iter_windows(arr.shape[-2], arr.shape[-1], tile, overlap):
        block = xp.asarray(arr[..., w.pad_row, w.pad_col])
        res = func(block)
        if res.shape[-2:] != block.shape[-2:]:
            raise ValueError(
                f"func changed spatial shape {block.shape[-2:]} -> {res.shape[-2:]}; "
                "apply_tiled requires shape-preserving functions"
            )
        ir, ic = w.inner
        res = to_cpu(res[..., ir, ic])
        if out is None:
            out = np.empty(res.shape[:-2] + arr.shape[-2:], dtype=dtype or res.dtype)
        out[..., w.row, w.col] = res
    return out


def _apply_dask(arr: Any, func: Callable, *, overlap: int, dtype: Any):
    backend = get_backend()

    def _block(b):
        from ..backend import device

        with device(str(backend)):
            xp = get_xp()
            return to_cpu(func(xp.asarray(b)))

    depth = {i: 0 for i in range(arr.ndim)}
    depth[arr.ndim - 2] = depth[arr.ndim - 1] = overlap
    return arr.map_overlap(_block, depth=depth, boundary="none", trim=True,
                           dtype=dtype or arr.dtype)
