"""Internal helpers shared by native algorithms.

Convention: native functions compute on the *active* backend and return results
in the same memory space as their input (NumPy in -> NumPy out, CuPy in -> CuPy
out). So CPU data still benefits from a GPU backend without code changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from ..backend import array_module, get_xp, is_gpu_array, ndimage, to_cpu


def where(cond: Any, a: Any, b: Any) -> Any:
    """``xr.where``/``np.where``, picked so a DataArray input stays a DataArray.

    Plain ``np.where`` on a DataArray silently returns a bare ndarray (no NEP-18 dispatch),
    which then makes ``xr.Dataset({...})`` promote it to its own anonymous dimension
    coordinate instead of aligning it with the other variables.
    """
    if isinstance(cond, xr.DataArray) or isinstance(a, xr.DataArray) or isinstance(b, xr.DataArray):
        return xr.where(cond, a, b)
    return np.where(cond, a, b)


def materialize(data: Any) -> Any:
    """Compute Dask arrays (native filters need whole arrays; use apply_tiled for huge data)."""
    if hasattr(data, "compute") and hasattr(data, "dask"):
        return data.compute()
    return data


def run_array(data: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn(xp_array, xp, *args, **kwargs)`` on the active backend.

    Accepts NumPy/CuPy/Dask arrays or DataArrays; returns the same container type,
    in the input's memory space. Output with the input's shape keeps its coords.
    """
    if isinstance(data, xr.DataArray):
        res = run_array(data.data, fn, *args, **kwargs)
        if getattr(res, "shape", None) == data.shape:
            return data.copy(data=res)
        return res
    data = materialize(data)
    was_gpu = is_gpu_array(data)
    xp = get_xp()
    arr = xp.asarray(data) if xp is not np else np.asarray(to_cpu(data))
    res = fn(arr, xp, *args, **kwargs)
    return res if was_gpu else to_cpu(res)


def as_float(arr: Any, xp: Any) -> Any:
    return arr if arr.dtype in (xp.float32, xp.float64) else arr.astype(xp.float32)


def nan_box_mean(arr: Any, size: int | tuple[int, int], xp: Any) -> Any:
    """Moving-window mean ignoring NaN, on the last two axes."""
    nd = ndimage(arr)
    sz = (size, size) if isinstance(size, int) else size
    sz = (1,) * (arr.ndim - 2) + tuple(sz)
    valid = ~xp.isnan(arr)
    filled = xp.where(valid, arr, 0)
    num = nd.uniform_filter(filled, sz, mode="reflect")
    den = nd.uniform_filter(valid.astype(arr.dtype), sz, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return xp.where(den > 0, out, xp.nan)


def local_mean_var(arr: Any, size: int, xp: Any) -> tuple[Any, Any]:
    """NaN-aware local mean and variance."""
    m = nan_box_mean(arr, size, xp)
    m2 = nan_box_mean(arr * arr, size, xp)
    return m, xp.maximum(m2 - m * m, 0)


def like2d(template: xr.DataArray | None, data: Any, name: str | None = None,
           attrs: dict | None = None) -> Any:
    """Wrap a 2-D (or (..., y, x)) result with the spatial coords/CRS of ``template``."""
    if template is None or not isinstance(template, xr.DataArray):
        return data
    sdims = template.dims[-2:]
    if data.ndim == 2:
        da = xr.DataArray(data, coords={d: template[d] for d in sdims if d in template.coords},
                          dims=sdims, name=name, attrs=attrs or {})
    else:
        da = template.copy(data=data).rename(name) if name else template.copy(data=data)
        if attrs:
            da.attrs = attrs
    if template.rio.crs is not None:
        da = da.rio.write_crs(template.rio.crs)
    return da


def xp_of(arr: Any):
    return array_module(arr)


def to_xp(arr: Any, xp: Any) -> Any:
    """Convert an array to module ``xp`` (NumPy or CuPy), unwrapping DataArrays."""
    arr = materialize(getattr(arr, "data", arr))
    return np.asarray(to_cpu(arr)) if xp is np else xp.asarray(arr)


def pix_to_map(transform: Any, col: Any, row: Any) -> tuple[Any, Any]:
    """Affine pixel (col, row) → map (x, y), independent of the ``affine`` package version."""
    t = transform
    col, row = np.asarray(col, float), np.asarray(row, float)
    return t.c + t.a * col + t.b * row, t.f + t.d * col + t.e * row


def map_to_pix(transform: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """Map (x, y) → fractional pixel (col, row)."""
    return pix_to_map(~transform, x, y)
