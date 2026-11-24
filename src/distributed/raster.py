"""Chunked raster processing on CPU (NumPy) or GPU (CuPy), locally or on a Dask cluster.

Every function accepts a NumPy/CuPy/Dask array or an ``xarray.DataArray`` whose last
two dimensions are ``(y, x)``; leading dimensions (bands, time) are carried through.

* In-memory input is chunked with :func:`~geolattice.distributed._partition.plan_chunks`,
  processed in parallel and returned in memory.
* Dask-backed input stays lazy; call ``.compute()`` (or :func:`geolattice.distributed.compute`).
* ``compute="gpu"`` moves chunks to CuPy for the computation. The result is returned in
  the same memory space as the input (host input gives host output).

Moving-window operations read a halo of neighbouring pixels from adjacent chunks
(``dask.array.map_overlap`` with ``boundary="nearest"``), so results match running on
the whole array at once, edge pixels included.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from ._backend import Compute, get_defaults, is_gpu, is_lazy, resolve_compute, to_cpu, to_gpu
from ._partition import plan_chunks
from ._registry import register

FocalStat = Literal["mean", "sum", "min", "max", "median", "std"]


# --------------------------------------------------------------------------- plumbing
@dataclass(frozen=True)
class _Ctx:
    was_da: bool
    lazy: bool
    was_gpu: bool
    used: str


def _xp(block: Any):
    if type(block).__module__.startswith("cupy"):
        import cupy as cp

        return cp
    return np


def _ndimage(block: Any):
    if type(block).__module__.startswith("cupy"):
        import cupyx.scipy.ndimage as ndi

        return ndi
    import scipy.ndimage as ndi

    return ndi


def _prepare(obj: Any, *, compute: Compute | None, chunks: Any, op: str, halo: int = 0
             ) -> tuple[xr.DataArray, _Ctx]:
    """Wrap ``obj`` as a Dask-backed DataArray on the resolved device."""
    was_da = isinstance(obj, xr.DataArray)
    if not was_da:
        arr = obj
        if getattr(arr, "ndim", 0) < 2:
            raise ValueError("raster input needs at least two (y, x) dimensions")
        dims = [f"dim_{i}" for i in range(arr.ndim - 2)] + ["y", "x"]
        obj = xr.DataArray(arr, dims=dims)
    ctx = _Ctx(was_da, is_lazy(obj), is_gpu(obj), resolve_compute(compute, op=op))
    da = obj
    if isinstance(chunks, int):
        chunks = (chunks, chunks)
    if isinstance(chunks, tuple):  # (rows, cols) for the spatial dims
        chunks = {**{d: -1 for d in da.dims[:-2]}, **dict(zip(da.dims[-2:], chunks[-2:]))}
    if not ctx.lazy:
        chunks = get_defaults().chunks if chunks is None else chunks
        if chunks == "auto":
            plan = plan_chunks(da.shape, da.dtype, halo=halo, compute=ctx.used)
            chunks = dict(zip(da.dims, plan))
        # a fresh token skips Dask's content hash, which pickles CuPy data through the host
        da = da.chunk(chunks, token=uuid.uuid4().hex)
    elif chunks not in (None, "auto"):
        da = da.chunk(chunks)
    if ctx.used == "gpu" and not ctx.was_gpu:
        da = to_gpu(da)
    elif ctx.used == "cpu" and ctx.was_gpu:
        da = to_cpu(da)
    return da, ctx


def _finish(da: xr.DataArray, ctx: _Ctx) -> Any:
    """Return ``da`` in the input's container, laziness and memory space."""
    if ctx.used == "gpu" and not ctx.was_gpu:
        da = to_cpu(da)
    elif ctx.used == "cpu" and ctx.was_gpu:
        da = to_gpu(da)
    if not ctx.lazy:
        da = da.compute()
    return da if ctx.was_da else da.data


def _map_overlap(da: xr.DataArray, fn: Callable[[Any, Any], Any], *, halo: int,
                 dtype: Any = None, boundary: Any = "nearest") -> xr.DataArray:
    """Run ``fn(block, xp)`` on each chunk with ``halo`` pixels of overlap in y and x.
    ``boundary`` fills the halo outside the array (``"nearest"`` or a constant)."""
    data = da.data
    dtype = np.dtype(dtype or data.dtype)
    meta = data._meta.astype(dtype)
    if halo == 0:
        out = data.map_blocks(lambda b: fn(b, _xp(b)), dtype=dtype, meta=meta)
    else:
        depth = {i: 0 for i in range(data.ndim)}
        depth[data.ndim - 2] = depth[data.ndim - 1] = halo
        out = data.map_overlap(lambda b: fn(b, _xp(b)), depth=depth, boundary=boundary,
                               trim=True, dtype=dtype, meta=meta)
    return da.copy(data=out)


def _named(da: Any, name: str, units: str | None = None) -> Any:
    if isinstance(da, xr.DataArray):
        da = da.rename(name)
        if units is not None:
            da.attrs["units"] = units
    return da


def _float(b: Any, xp: Any) -> Any:
    return b if b.dtype in (xp.float32, xp.float64) else b.astype(xp.float32)


def map_tiles(obj: Any, func: Callable[[Any, Any], Any], *, overlap: int = 0, dtype: Any = None,
              compute: Compute | None = None, chunks: Any = None) -> Any:
    """Apply ``func(block, xp)`` chunk by chunk, with ``overlap`` halo pixels.

    ``func`` receives a NumPy or CuPy block shaped ``(..., rows, cols)`` plus the matching
    array module, and must return a block of the same shape. Write it with ``xp``
    instead of ``np`` and it runs unchanged on CPU and GPU.
    """
    da, ctx = _prepare(obj, compute=compute, chunks=chunks, op="raster.map_tiles", halo=overlap)
    return _finish(_map_overlap(da, func, halo=overlap, dtype=dtype), ctx)


# --------------------------------------------------------------------------- local algebra
_EXPR_FUNCS = {f: getattr(np, f) for f in (
    "sqrt", "exp", "log", "log10", "log1p", "abs", "sin", "cos", "tan", "arctan", "arctan2",
    "minimum", "maximum", "floor", "ceil", "isnan", "isfinite", "clip",
)}


def _where(cond: Any, a: Any, b: Any) -> Any:
    """``where`` that keeps Python scalars as scalars, so CuPy chunks accept them."""
    import dask.array as dsa

    args = [cond, a, b]
    arrays = [i for i, v in enumerate(args) if isinstance(v, xr.DataArray)]
    if not arrays:
        return dsa.where(cond, a, b)
    aligned = xr.broadcast(*[args[i] for i in arrays])
    for i, v in zip(arrays, aligned):
        args[i] = v
    like = args[arrays[0]]
    out = dsa.where(*[v.data if isinstance(v, xr.DataArray) else v for v in args])
    return like.copy(data=out)


def calc(expr: str | Callable[..., Any], *, compute: Compute | None = None, chunks: Any = None,
         **layers: Any) -> Any:
    """Raster calculator over named layers.

    ``expr`` is a callable taking the layers as keyword arguments, or an arithmetic string
    such as ``"(nir - red) / (nir + red)"``. Strings may use the layer names, ``where``,
    and NumPy ufuncs (``sqrt``, ``log``, ``minimum``, ...). Layers are aligned by xarray.

    >>> ndvi = calc("(nir - red) / (nir + red)", nir=b8, red=b4)
    """
    if not layers:
        raise ValueError("calc needs at least one layer, e.g. calc('a * 2', a=dem)")
    prepared, ctxs = {}, []
    for name, lyr in layers.items():
        prepared[name], ctx = _prepare(lyr, compute=compute, chunks=chunks, op="raster.calc")
        ctxs.append(ctx)
    if callable(expr):
        out = expr(**prepared)
    else:
        ns = dict(_EXPR_FUNCS, where=_where, nan=np.nan, pi=np.pi, **prepared)
        out = eval(compile(expr, "<calc>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
    return _finish(out, ctxs[0])


def where(cond: Any, a: Any, b: Any, *, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Element-wise ``a`` where ``cond`` is true, else ``b`` (``a``/``b`` may be scalars)."""
    layers = {k: v for k, v in (("a", a), ("b", b)) if not np.isscalar(v)}
    return calc(lambda cond, **ab: _where(cond, ab.get("a", a), ab.get("b", b)),
                compute=compute, chunks=chunks, cond=cond, **layers)


def reclassify(obj: Any, *, bins: Sequence[float] | None = None,
               values: Sequence[float] | None = None, mapping: Mapping[float, float] | None = None,
               nodata: float = np.nan, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Reclassify cell values.

    ``bins`` (increasing edges) assigns class ``i`` to ``bins[i-1] <= v < bins[i]`` as in
    ``numpy.digitize``; pass ``values`` (length ``len(bins) + 1``) to output those values
    instead of class numbers. ``mapping`` replaces exact values (``{1: 10, 2: 20}``);
    unmapped cells get ``nodata``. NaN input stays ``nodata``.
    """
    if (bins is None) == (mapping is None):
        raise ValueError("pass exactly one of bins= or mapping=")
    if values is not None and (bins is None or len(values) != len(bins) + 1):
        raise ValueError("values needs len(bins) + 1 entries")

    def f(b, xp):
        bf = _float(b, xp)
        if bins is not None:
            idx = xp.digitize(bf, xp.asarray(bins, dtype=bf.dtype))
            out = idx.astype(xp.float32) if values is None else xp.asarray(values, xp.float32)[idx]
        else:
            out = xp.full(bf.shape, nodata, dtype=xp.float32)
            for k, v in mapping.items():
                out = xp.where(bf == k, xp.float32(v), out)
        return xp.where(xp.isnan(bf), xp.float32(nodata), out)

    return map_tiles(obj, f, dtype="float32", compute=compute, chunks=chunks)


def normalize(obj: Any, *, method: Literal["minmax", "zscore"] = "minmax",
              compute: Compute | None = None, chunks: Any = None) -> Any:
    """Rescale to 0-1 (``minmax``) or to zero mean and unit variance (``zscore``), ignoring NaN.
    Global statistics are reduced across all chunks."""
    da, ctx = _prepare(obj, compute=compute, chunks=chunks, op="raster.normalize")
    if method == "minmax":
        lo, hi = da.min(skipna=True), da.max(skipna=True)
        out = (da - lo) / (hi - lo)
    elif method == "zscore":
        out = (da - da.mean(skipna=True)) / da.std(skipna=True)
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")
    return _finish(out, ctx)


def normalized_difference(a: Any, b: Any, *, compute: Compute | None = None,
                          chunks: Any = None) -> Any:
    """``(a - b) / (a + b)``, e.g. NDVI with ``a=nir, b=red``."""
    return calc("(a - b) / (a + b)", a=a, b=b, compute=compute, chunks=chunks)


# --------------------------------------------------------------------------- focal
def _footprint(ndim: int, size: int) -> tuple[int, ...]:
    return (1,) * (ndim - 2) + (size, size)


def focal(obj: Any, stat: FocalStat = "mean", *, size: int = 3, compute: Compute | None = None,
          chunks: Any = None) -> Any:
    """Moving-window statistic over a ``size`` x ``size`` window (edges use nearest values).

    NaN propagates into every window that contains it; fill gaps first if needed.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    def f(b, xp):
        ndi = _ndimage(b)
        bf = _float(b, xp)
        fp = _footprint(b.ndim, size)
        if stat == "mean":
            return ndi.uniform_filter(bf, fp, mode="nearest")
        if stat == "sum":
            return ndi.uniform_filter(bf, fp, mode="nearest") * (size * size)
        if stat == "min":
            return ndi.minimum_filter(bf, fp, mode="nearest")
        if stat == "max":
            return ndi.maximum_filter(bf, fp, mode="nearest")
        if stat == "median":
            return ndi.median_filter(bf, fp, mode="nearest")
        if stat == "std":
            # E[x^2] - E[x]^2 loses most float32 digits when the values are large compared with
            # their spread (elevations of 3000 m varying by 5 m): shift by the block mean and
            # work in float64, then return the input's precision
            d = bf.astype(xp.float64)
            finite = d[xp.isfinite(d)]
            d = d - (finite.mean() if finite.size else 0.0)
            m = ndi.uniform_filter(d, fp, mode="nearest")
            m2 = ndi.uniform_filter(d * d, fp, mode="nearest")
            return xp.sqrt(xp.maximum(m2 - m * m, 0)).astype(bf.dtype)
        raise ValueError(f"Unknown stat {stat!r}")

    da, ctx = _prepare(obj, compute=compute, chunks=chunks, op="raster.focal", halo=size // 2)
    dtype = np.result_type(da.dtype, np.float32)
    return _named(_finish(_map_overlap(da, f, halo=size // 2, dtype=dtype), ctx), f"focal_{stat}")


def convolve(obj: Any, kernel: Any, *, normalize_kernel: bool = False,
             compute: Compute | None = None, chunks: Any = None) -> Any:
    """Convolve with a 2-D ``kernel`` (odd shape; edges use nearest values)."""
    k = np.asarray(kernel, dtype="float64")
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd side lengths")
    if normalize_kernel:
        k = k / k.sum()
    halo = max(k.shape) // 2

    def f(b, xp):
        bf = _float(b, xp)
        kk = xp.asarray(k, dtype=bf.dtype).reshape((1,) * (b.ndim - 2) + k.shape)
        return _ndimage(b).convolve(bf, kk, mode="nearest")

    da, ctx = _prepare(obj, compute=compute, chunks=chunks, op="raster.convolve", halo=halo)
    dtype = np.result_type(da.dtype, np.float32)
    return _finish(_map_overlap(da, f, halo=halo, dtype=dtype), ctx)


# --------------------------------------------------------------------------- terrain
def _res(dem: Any, res: float | tuple[float, float] | None) -> tuple[float, float, bool]:
    """``(dx, dy, north_up)`` in map units, from ``res`` or the DataArray coordinates."""
    if res is not None:
        dx, dy = (res, res) if np.isscalar(res) else res
        return float(dx), float(dy), True
    if isinstance(dem, xr.DataArray):
        ydim, xdim = dem.dims[-2:]
        x, y = np.asarray(dem[xdim].values), np.asarray(dem[ydim].values)
        if x.size < 2 or y.size < 2:
            raise ValueError("DEM coordinates too short to infer resolution; pass res=")
        dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
        crs = None
        try:
            import rioxarray  # noqa: F401

            crs = dem.rio.crs
        except Exception:
            pass
        if crs is not None and crs.is_geographic:
            lat = float(np.mean(y))
            warnings.warn("DEM is in geographic coordinates; converting degrees to metres at the "
                          "scene centre. Reproject to a projected CRS for accurate results.",
                          UserWarning, stacklevel=3)
            dx, dy = dx * 111_320 * np.cos(np.radians(lat)), dy * 110_540
        return dx, dy, bool(y[1] < y[0])
    raise ValueError("Pass res= for plain arrays")


def _neighbors(z: Any, xp: Any) -> dict[str, Any]:
    """3x3 neighbourhood (edge-padded) as shifted views a..i, row-major from top-left."""
    p = xp.pad(z, [(0, 0)] * (z.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    r, c = z.shape[-2:]
    return {k: p[..., dr:dr + r, dc:dc + c]
            for k, (dr, dc) in zip("abcdefghi", [(i, j) for i in range(3) for j in range(3)])}


def _gradients(z: Any, xp: Any, dx: float, dy: float, north_up: bool, z_factor: float):
    n = _neighbors(z * z_factor, xp)
    gx = ((n["c"] + 2 * n["f"] + n["i"]) - (n["a"] + 2 * n["d"] + n["g"])) / (8 * dx)
    gn = ((n["a"] + 2 * n["b"] + n["c"]) - (n["g"] + 2 * n["h"] + n["i"])) / (8 * dy)
    return gx, (gn if north_up else -gn)


def _terrain(dem: Any, f: Callable, *, halo: int, op: str, compute: Compute | None,
             chunks: Any, name: str, units: str) -> Any:
    da, ctx = _prepare(dem, compute=compute, chunks=chunks, op=op, halo=halo)
    dtype = np.result_type(da.dtype, np.float32)
    return _named(_finish(_map_overlap(da, f, halo=halo, dtype=dtype), ctx), name, units)


def slope(dem: Any, *, units: Literal["degrees", "radians", "percent"] = "degrees",
          res: Any = None, z_factor: float = 1.0, compute: Compute | None = None,
          chunks: Any = None) -> Any:
    """Slope (Horn 1981)."""
    dx, dy, nu = _res(dem, res)

    def f(z, xp):
        gx, gn = _gradients(_float(z, xp), xp, dx, dy, nu, z_factor)
        g = xp.sqrt(gx**2 + gn**2)
        if units == "percent":
            return g * 100
        s = xp.arctan(g)
        return xp.degrees(s) if units == "degrees" else s

    return _terrain(dem, f, halo=1, op="raster.slope", compute=compute, chunks=chunks,
                    name="slope", units=units)


def aspect(dem: Any, *, res: Any = None, flat: float = np.nan, compute: Compute | None = None,
           chunks: Any = None) -> Any:
    """Aspect: direction of steepest descent, degrees clockwise from north (``flat`` where level)."""
    dx, dy, nu = _res(dem, res)

    def f(z, xp):
        gx, gn = _gradients(_float(z, xp), xp, dx, dy, nu, 1.0)
        a = xp.degrees(xp.arctan2(-gx, -gn)) % 360
        return xp.where((gx == 0) & (gn == 0), flat, a)

    return _terrain(dem, f, halo=1, op="raster.aspect", compute=compute, chunks=chunks,
                    name="aspect", units="degrees")


def hillshade(dem: Any, *, azimuth: float = 315, altitude: float = 45, res: Any = None,
              z_factor: float = 1.0, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Analytical hillshade scaled 0-255."""
    dx, dy, nu = _res(dem, res)
    zen, az = np.radians(90 - altitude), np.radians(azimuth)

    def f(z, xp):
        gx, gn = _gradients(_float(z, xp), xp, dx, dy, nu, z_factor)
        s = xp.arctan(xp.sqrt(gx**2 + gn**2))
        a = xp.arctan2(-gx, -gn)
        cos_i = np.cos(zen) * xp.cos(s) + np.sin(zen) * xp.sin(s) * xp.cos(az - a)
        return xp.clip(cos_i, 0, 1) * 255

    return _terrain(dem, f, halo=1, op="raster.hillshade", compute=compute, chunks=chunks,
                    name="hillshade", units="")


def curvature(dem: Any, *, res: Any = None, compute: Compute | None = None,
              chunks: Any = None) -> Any:
    """Total (Laplacian) curvature, 1/m: negative on ridges, positive in hollows."""
    dx, dy, _ = _res(dem, res)

    def f(z, xp):
        n = _neighbors(_float(z, xp), xp)
        return (n["d"] - 2 * n["e"] + n["f"]) / dx**2 + (n["b"] - 2 * n["e"] + n["h"]) / dy**2

    return _terrain(dem, f, halo=1, op="raster.curvature", compute=compute, chunks=chunks,
                    name="curvature", units="1/m")


def tpi(dem: Any, *, window: int = 9, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Topographic Position Index: elevation minus the ``window`` x ``window`` mean."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    def f(z, xp):
        zf = _float(z, xp)
        return zf - _ndimage(z).uniform_filter(zf, _footprint(z.ndim, window), mode="nearest")

    return _terrain(dem, f, halo=window // 2, op="raster.tpi", compute=compute, chunks=chunks,
                    name="tpi", units="m")


def tri(dem: Any, *, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Terrain Ruggedness Index (Riley et al. 1999)."""
    def f(z, xp):
        zf = _float(z, xp)
        n = _neighbors(zf, xp)
        return xp.sqrt(sum((n[k] - zf) ** 2 for k in "abcdfghi"))

    return _terrain(dem, f, halo=1, op="raster.tri", compute=compute, chunks=chunks,
                    name="tri", units="m")


def roughness(dem: Any, *, compute: Compute | None = None, chunks: Any = None) -> Any:
    """Maximum minus minimum elevation in the 3x3 neighbourhood."""
    def f(z, xp):
        n = _neighbors(_float(z, xp), xp)
        stack = xp.stack([n[k] for k in "abcdefghi"])
        return stack.max(0) - stack.min(0)

    return _terrain(dem, f, halo=1, op="raster.roughness", compute=compute, chunks=chunks,
                    name="roughness", units="m")


# Every raster op runs the same xp-generic kernel on NumPy or CuPy chunks.
for _op in ("map_tiles", "calc", "where", "reclassify", "normalize", "normalized_difference",
            "focal", "convolve", "slope", "aspect", "hillshade", "curvature", "tpi", "tri",
            "roughness"):
    _fn = globals()[_op]
    register(f"raster.{_op}", "cpu")(_fn)
    register(f"raster.{_op}", "gpu")(_fn)
del _op, _fn
