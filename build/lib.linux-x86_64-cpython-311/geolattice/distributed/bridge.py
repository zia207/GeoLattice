"""Raster and vector together: rasterize, zonal statistics, point extraction, polygonize.

All four work chunk by chunk on Dask-backed rasters, so they scale to rasters larger
than memory and run on a cluster:

* ``rasterize`` sends each raster chunk only the features whose bounding boxes touch it.
  Polygons burn on the GPU (pixel-centre rule); points, lines and ``all_touched`` use
  ``rasterio`` on CPU.
* ``zonal_stats`` rasterizes zones onto the raster's chunks and reduces partial
  statistics per chunk (NumPy or CuPy ``bincount`` / scatter), then combines them.
* ``extract_values`` gathers pixels for points from just the chunks that hold them.
* ``polygonize`` traces each chunk with ``rasterio.features.shapes`` (CPU) and merges
  same-valued polygons that meet across chunk seams.

Where zones overlap, each pixel belongs to the last zone in row order (as in
``rasterio.features.rasterize``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ._backend import Compute, resolve_compute
from ._registry import dispatch, register
from .raster import _prepare

Stat = Literal["count", "sum", "mean", "min", "max", "std"]
_ALL_STATS: tuple[Stat, ...] = ("count", "sum", "mean", "min", "max", "std")


# --------------------------------------------------------------------------- grid helpers
def _grid(da: xr.DataArray) -> tuple[float, float, float, float]:
    """``(x0, dx, y0, dy)``: outer corner of pixel (0, 0) and signed pixel sizes."""
    ydim, xdim = da.dims[-2:]
    x, y = np.asarray(da[xdim].values, dtype=float), np.asarray(da[ydim].values, dtype=float)
    if x.size >= 2 and y.size >= 2:
        dx, dy = x[1] - x[0], y[1] - y[0]
        return x[0] - dx / 2, dx, y[0] - dy / 2, dy
    t = da.rio.transform()
    return t.c, t.a, t.f, t.e


def _edges(chunks: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(chunks)]).astype(int)


def _crs(da: xr.DataArray) -> Any:
    try:
        import rioxarray  # noqa: F401

        return da.rio.crs
    except Exception:
        return None


def _check_crs(gdf: Any, da: xr.DataArray) -> None:
    rc = _crs(da)
    if rc is not None and gdf.crs is not None and gdf.crs != rc:
        raise ValueError(f"CRS mismatch: vector {gdf.crs} vs raster {rc}; reproject first")


def _template(like: Any, compute: Compute | None, chunks: Any, op: str):
    """Spatial (y, x) template DataArray (Dask-backed) plus the resolved context."""
    da, ctx = _prepare(like, compute=compute, chunks=chunks, op=op)
    lead = {d: 0 for d in da.dims[:-2]}
    return (da.isel(lead, drop=True) if lead else da), da, ctx


# --------------------------------------------------------------------------- rasterize
def _burn_cpu(geoms: np.ndarray, vals: np.ndarray, shape: tuple[int, int], x0: float, dx: float,
              y0: float, dy: float, fill: float, all_touched: bool, dtype: Any) -> np.ndarray:
    from affine import Affine
    from rasterio.features import rasterize as rio_rasterize

    if len(geoms) == 0:
        return np.full(shape, fill, dtype=dtype)
    return rio_rasterize(zip(geoms, vals), out_shape=shape, transform=Affine(dx, 0, x0, 0, dy, y0),
                         fill=fill, all_touched=all_touched, dtype=dtype)


def _burn_gpu(geoms: np.ndarray, vals: np.ndarray, shape: tuple[int, int], x0: float, dx: float,
              y0: float, dy: float, fill: float, all_touched: bool, dtype: Any) -> Any:
    from . import _gpu_geom as G

    if len(geoms) == 0 or all_touched or G.family(geoms) != "polygon":
        import cupy as cp

        return cp.asarray(_burn_cpu(geoms, vals, shape, x0, dx, y0, dy, fill, all_touched, dtype))
    import shapely

    out = G.rasterize_polygons(G.to_ragged(geoms), np.asarray(vals, dtype=float),
                               shapely.bounds(geoms), shape, x0, dx, y0, dy, fill)
    return out.astype(dtype)


register("bridge.rasterize", "cpu")(_burn_cpu)
# GDAL's scanline fill beats a per-pixel GPU test for typical polygon counts (measured:
# 3000 polygons on 8000 x 8000, 0.56 s CPU vs 1.14 s GPU), so GPU is opt-in.
register("bridge.rasterize", "gpu", auto=False)(_burn_gpu)


def rasterize(gdf: Any, like: Any, *, column: str | None = None, values: Any = None,
              fill: float = 0, all_touched: bool = False, dtype: Any = "float32",
              compute: Compute | None = None, chunks: Any = None) -> Any:
    """Burn features onto the grid of ``like`` (a DataArray; its last two dims are y, x).

    Burn values come from ``column``, an array ``values``, or 1. Later rows overwrite
    earlier ones. The result is a ``(y, x)`` DataArray with ``like``'s coordinates and
    chunks; it is lazy if ``like`` is lazy, otherwise computed.
    """
    import dask
    import dask.array as dsa
    from shapely import box

    if not isinstance(like, xr.DataArray):
        raise TypeError("like must be an xarray.DataArray with y/x coordinates")
    _check_crs(gdf, like)
    tmpl, _, ctx = _template(like, compute, chunks, "bridge.rasterize")
    impl, used = dispatch("bridge.rasterize", ctx.used)
    vals = (np.asarray(gdf[column]) if column is not None else
            np.ones(len(gdf)) if values is None else np.asarray(values))
    if len(vals) != len(gdf):
        raise ValueError("values must have one entry per feature")
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    tree = gdf.sindex
    x0, dx, y0, dy = _grid(tmpl)
    re, ce = _edges(tmpl.data.chunks[0]), _edges(tmpl.data.chunks[1])
    dtype = np.dtype(dtype)
    meta = tmpl.data._meta.astype(dtype)
    rows = []
    for i in range(len(re) - 1):
        row = []
        for j in range(len(ce) - 1):
            bx0, by0 = x0 + ce[j] * dx, y0 + re[i] * dy
            bx1, by1 = x0 + ce[j + 1] * dx, y0 + re[i + 1] * dy
            idx = np.sort(tree.query(box(min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))))
            shape = (int(re[i + 1] - re[i]), int(ce[j + 1] - ce[j]))
            blk = dask.delayed(impl)(geoms[idx], vals[idx], shape, bx0, dx, by0, dy, fill,
                                     all_touched, dtype)
            row.append(dsa.from_delayed(blk, shape=shape, dtype=dtype, meta=meta))
        rows.append(row)
    out = tmpl.copy(data=dsa.block(rows)).rename(column or "rasterized")
    out.attrs = {}
    if not ctx.was_gpu and used == "gpu":
        from ._backend import to_cpu

        out = to_cpu(out)
    if not ctx.lazy:
        out = out.compute()
    return out if ctx.was_da else out.data


# --------------------------------------------------------------------------- zonal statistics
def _zonal_partial(labels: Any, values: Any, n: int) -> dict[str, np.ndarray]:
    """Per-zone count, sum, sum of squares, min and max for one chunk (NumPy or CuPy)."""
    from ..rsense.backend.dispatch import scatter_reduce

    from .raster import _xp

    xp = _xp(values)
    lab = xp.asarray(labels).ravel()
    inzone = lab > 0
    v = values.reshape(-1, lab.size)  # a view; bands are handled one at a time
    out = {k: np.zeros((v.shape[0], n + 1)) for k in ("count", "sum", "sumsq")}
    out["min"] = np.full((v.shape[0], n + 1), np.inf)
    out["max"] = np.full((v.shape[0], n + 1), -np.inf)
    for b in range(v.shape[0]):
        # only the valid pixels inside a zone are converted to float64, so a large block
        # never exists as a whole float64 copy
        ok = inzone & xp.isfinite(v[b])
        idx, val = lab[ok].astype(xp.int64), v[b][ok].astype(xp.float64)
        if idx.size == 0:
            continue
        cnt = xp.bincount(idx, minlength=n + 1)
        sm = xp.bincount(idx, weights=val, minlength=n + 1)
        sq = xp.bincount(idx, weights=val * val, minlength=n + 1)
        mn = scatter_reduce(xp.full(n + 1, xp.inf), idx, val, "min")
        mx = scatter_reduce(xp.full(n + 1, -xp.inf), idx, val, "max")
        for key, arr in (("count", cnt), ("sum", sm), ("sumsq", sq), ("min", mn), ("max", mx)):
            out[key][b] = arr.get() if hasattr(arr, "get") else arr
    return out


def _zonal_combine(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: parts[0][k].copy() for k in parts[0]}
    for p in parts[1:]:
        for k in ("count", "sum", "sumsq"):
            out[k] += p[k]
        out["min"] = np.minimum(out["min"], p["min"])
        out["max"] = np.maximum(out["max"], p["max"])
    return out


register("bridge.zonal_stats", "cpu")(_zonal_partial)
register("bridge.zonal_stats", "gpu")(_zonal_partial)


def zonal_stats(zones: Any, raster: Any, *, stats: Sequence[Stat] = _ALL_STATS,
                all_touched: bool = False, prefix: str = "", compute: Compute | None = None,
                chunks: Any = None) -> Any:
    """Statistics of ``raster`` cells inside each zone polygon.

    ``raster`` is a ``(y, x)`` or ``(band, y, x)`` DataArray; NaN cells are ignored.
    Returns ``zones`` with one column per statistic (``{prefix}{stat}``, or
    ``{prefix}{stat}_{band}`` for several bands). ``std`` is the population standard
    deviation. Zones with no cells get count 0 and NaN for the rest.
    """
    import dask

    bad = set(stats) - set(_ALL_STATS)
    if bad:
        raise ValueError(f"Unknown stats {sorted(bad)}; choose from {_ALL_STATS}")
    if not isinstance(raster, xr.DataArray):
        raise TypeError("raster must be an xarray.DataArray with y/x coordinates")
    if raster.ndim not in (2, 3):
        raise ValueError("raster must be (y, x) or (band, y, x)")
    _check_crs(zones, raster)
    da, ctx = _prepare(raster, compute=compute, chunks=chunks, op="bridge.zonal_stats")
    if da.ndim == 3:
        da = da.chunk({da.dims[0]: -1})
    impl, used = dispatch("bridge.zonal_stats", ctx.used)
    n = len(zones)
    labels = rasterize(zones, da.isel({da.dims[0]: 0}) if da.ndim == 3 else da,
                       values=np.arange(1, n + 1), fill=0, all_touched=all_touched,
                       dtype="int32", compute=compute)
    lab_blocks = labels.data.to_delayed()
    val_blocks = da.data.to_delayed()
    if da.ndim == 3:
        val_blocks = val_blocks[0]
    parts = [dask.delayed(impl)(lab_blocks[i, j], val_blocks[i, j], n)
             for i in range(lab_blocks.shape[0]) for j in range(lab_blocks.shape[1])]
    # Reduce in batches: with one task combining every partial at once, Dask may load far
    # more chunks than it can finish (a 21-billion-pixel, 3-band raster exhausted 120 GB).
    # A batch of a few chunks per thread keeps memory bounded by the batch size.
    from ._partition import default_npartitions

    batch = max(8, 2 * default_npartitions(10**9))
    tot = None
    for k in range(0, len(parts), batch):
        done = dask.compute(*parts[k:k + batch])
        tot = _zonal_combine(*done) if tot is None else _zonal_combine(tot, *done)

    cnt = tot["count"][:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot["sum"][:, 1:] / cnt
        table = {
            "count": cnt,
            "sum": np.where(cnt > 0, tot["sum"][:, 1:], np.nan),
            "mean": mean,
            "min": np.where(cnt > 0, tot["min"][:, 1:], np.nan),
            "max": np.where(cnt > 0, tot["max"][:, 1:], np.nan),
            "std": np.sqrt(np.maximum(tot["sumsq"][:, 1:] / cnt - mean**2, 0)),
        }
    out = zones.copy()
    bands = list(da[da.dims[0]].values) if da.ndim == 3 else [None]
    for s in stats:
        for b, band in enumerate(bands):
            col = f"{prefix}{s}" if band is None else f"{prefix}{s}_{band}"
            out[col] = table[s][b].astype("int64") if s == "count" else table[s][b]
    return out


# --------------------------------------------------------------------------- point extraction
def _take(block: Any, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    from .raster import _xp

    xp = _xp(block)
    vals = block[..., xp.asarray(r), xp.asarray(c)]
    return vals.get() if hasattr(vals, "get") else np.asarray(vals)


register("bridge.extract_values", "cpu")(_take)
register("bridge.extract_values", "gpu")(_take)


def _gather(data: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``data[..., rows[k], cols[k]]`` for in-bounds integer positions, touching only the
    chunks that hold them. Returns ``(lead, n)`` floats."""
    import dask

    re, ce = _edges(data.chunks[-2]), _edges(data.chunks[-1])
    bi = np.searchsorted(re, rows, side="right") - 1
    bj = np.searchsorted(ce, cols, side="right") - 1
    if data.ndim > 2 and any(len(c) > 1 for c in data.chunks[:-2]):
        data = data.rechunk({i: -1 for i in range(data.ndim - 2)})
    blocks = data.to_delayed().reshape(len(re) - 1, len(ce) - 1)
    lead = int(np.prod(data.shape[:-2])) if data.ndim > 2 else 1
    out = np.full((lead, len(rows)), np.nan)
    keys = bi * (len(ce) - 1) + bj
    jobs, where = [], []
    for key in np.unique(keys):
        sel = np.nonzero(keys == key)[0]
        i, j = divmod(int(key), len(ce) - 1)
        jobs.append(dask.delayed(_take)(blocks[i, j], rows[sel] - re[i], cols[sel] - ce[j]))
        where.append(sel)
    for sel, vals in zip(where, dask.compute(*jobs)):
        out[:, sel] = np.asarray(vals, dtype=float).reshape(lead, -1)
    return out


def extract_values(points: Any, raster: Any, *, method: Literal["nearest", "linear"] = "nearest",
                   name: str | None = None, compute: Compute | None = None,
                   chunks: Any = None) -> Any:
    """Sample ``raster`` at point locations.

    Returns ``points`` with a column ``name`` (default: the raster's name or ``"value"``),
    or one column per band (``{name}_{band}``) for a ``(band, y, x)`` raster. Points
    outside the raster get NaN. ``linear`` is bilinear between pixel centres (edge pixels
    are clamped); any NaN neighbour gives NaN.
    """
    import shapely

    if not isinstance(raster, xr.DataArray):
        raise TypeError("raster must be an xarray.DataArray with y/x coordinates")
    if raster.ndim not in (2, 3):
        raise ValueError("raster must be (y, x) or (band, y, x)")
    _check_crs(points, raster)
    t = shapely.get_type_id(np.asarray(points.geometry.values, dtype=object))
    if not np.all(t == 0):
        raise TypeError("extract_values needs Point geometries")
    da, ctx = _prepare(raster, compute=compute, chunks=chunks, op="bridge.extract_values")
    if da.ndim == 3:
        da = da.chunk({da.dims[0]: -1})
    dispatch("bridge.extract_values", ctx.used)
    x0, dx, y0, dy = _grid(da)
    h, w = da.shape[-2:]
    px = shapely.get_x(points.geometry.values)
    py = shapely.get_y(points.geometry.values)
    fc, fr = (px - x0) / dx, (py - y0) / dy
    inside = (fc >= 0) & (fc < w) & (fr >= 0) & (fr < h)
    lead = da.shape[0] if da.ndim == 3 else 1
    vals = np.full((lead, len(px)), np.nan)
    ii = np.nonzero(inside)[0]
    if ii.size:
        if method == "nearest":
            vals[:, ii] = _gather(da.data, np.floor(fr[ii]).astype(int), np.floor(fc[ii]).astype(int))
        elif method == "linear":
            c, r = fc[ii] - 0.5, fr[ii] - 0.5
            c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
            wc, wr = c - c0, r - r0
            acc = np.zeros((lead, ii.size))
            for dr, dc, wt in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc),
                               (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
                rr, cc = np.clip(r0 + dr, 0, h - 1), np.clip(c0 + dc, 0, w - 1)
                acc += _gather(da.data, rr, cc) * wt
            vals[:, ii] = acc
        else:
            raise ValueError("method must be 'nearest' or 'linear'")
    out = points.copy()
    base = name or (raster.name if isinstance(raster.name, str) else None) or "value"
    if da.ndim == 3:
        for b, band in enumerate(da[da.dims[0]].values):
            out[f"{base}_{band}"] = vals[b]
    else:
        out[base] = vals[0]
    return out


# --------------------------------------------------------------------------- polygonize
_SHAPES_DTYPES = {np.dtype(t) for t in ("int16", "int32", "uint8", "uint16", "float32")}


def _shapes_dtype(dtype: np.dtype) -> np.dtype:
    if dtype in _SHAPES_DTYPES:
        return dtype
    if dtype == np.bool_:
        return np.dtype("uint8")
    if np.issubdtype(dtype, np.integer):
        return np.dtype("int32")
    return np.dtype("float32")


@register("bridge.polygonize", "cpu")
def _trace(block: np.ndarray, x0: float, dx: float, y0: float, dy: float, connectivity: int,
           nodata: float | None) -> tuple[list, list]:
    import shapely.geometry
    from affine import Affine
    from rasterio.features import shapes

    block = np.asarray(block.get() if hasattr(block, "get") else block)
    valid = np.isfinite(block) if np.issubdtype(block.dtype, np.floating) else \
        np.ones(block.shape, dtype=bool)
    if nodata is not None:
        valid &= block != nodata
    arr = block.astype(_shapes_dtype(block.dtype), copy=False)
    geoms, vals = [], []
    for g, v in shapes(arr, mask=valid, connectivity=connectivity,
                       transform=Affine(dx, 0, x0, 0, dy, y0)):
        geoms.append(shapely.geometry.shape(g))
        vals.append(v)
    return geoms, vals


def _merge_seams(gdf: Any, xs: np.ndarray, ys: np.ndarray, tol: float, connectivity: int,
                 name: str) -> Any:
    """Union same-valued polygons that meet across chunk seams."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(gdf) == 0 or (xs.size == 0 and ys.size == 0):
        return gdf
    b = gdf.bounds.to_numpy()

    def near(col: np.ndarray, seams: np.ndarray) -> np.ndarray:
        if seams.size == 0:
            return np.zeros(len(col), dtype=bool)
        return np.min(np.abs(col[:, None] - seams[None, :]), axis=1) < tol

    cand = np.nonzero(near(b[:, 0], xs) | near(b[:, 2], xs) | near(b[:, 1], ys) | near(b[:, 3], ys))[0]
    if cand.size < 2:
        return gdf
    sub = gdf.iloc[cand]
    geoms = sub.geometry.values
    li, ri = sub.sindex.query(geoms, predicate="intersects")
    vals = sub[name].to_numpy()
    keep = (li < ri) & (vals[li] == vals[ri])
    li, ri = li[keep], ri[keep]
    if connectivity == 4 and li.size:
        shared = shapely.length(shapely.intersection(geoms[li], geoms[ri])) > tol
        li, ri = li[shared], ri[shared]
    if li.size == 0:
        return gdf
    m = len(cand)
    _, comp = connected_components(coo_matrix((np.ones(li.size), (li, ri)), shape=(m, m)),
                                   directed=False)
    merged = sub.assign(_comp=comp).dissolve(by="_comp", aggfunc="first", as_index=False)
    merged = merged.drop(columns="_comp")
    rest = gdf.drop(index=gdf.index[cand])
    import geopandas as gpd

    return gpd.GeoDataFrame(pd.concat([rest, merged], ignore_index=True), crs=gdf.crs)


def polygonize(raster: Any, *, connectivity: Literal[4, 8] = 4, nodata: float | None = None,
               name: str = "value", compute: Compute | None = None, chunks: Any = None) -> Any:
    """Vectorize a ``(y, x)`` raster into polygons of equal value.

    NaN cells and ``nodata`` are skipped. Runs on CPU (``rasterio.features.shapes``) one
    chunk per task; polygons cut by chunk seams are merged back, so the result matches
    polygonizing the whole raster. Returns a GeoDataFrame with ``name`` and geometry.
    """
    import dask
    import geopandas as gpd

    if not isinstance(raster, xr.DataArray) or raster.ndim != 2:
        raise TypeError("raster must be a 2-D xarray.DataArray with y/x coordinates")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    resolve_compute(compute, op="bridge.polygonize", gpu_available=False)
    da, _ = _prepare(raster, compute="cpu", chunks=chunks, op="bridge.polygonize")
    impl, _ = dispatch("bridge.polygonize", "cpu")
    x0, dx, y0, dy = _grid(da)
    re, ce = _edges(da.data.chunks[0]), _edges(da.data.chunks[1])
    blocks = da.data.to_delayed()
    jobs = [dask.delayed(impl)(blocks[i, j], x0 + ce[j] * dx, dx, y0 + re[i] * dy, dy,
                               connectivity, nodata)
            for i in range(len(re) - 1) for j in range(len(ce) - 1)]
    geoms, vals = [], []
    for g, v in dask.compute(*jobs):
        geoms.extend(g)
        vals.extend(v)
    gdf = gpd.GeoDataFrame({name: vals}, geometry=geoms, crs=_crs(raster))
    xs, ys = x0 + ce[1:-1] * dx, y0 + re[1:-1] * dy
    tol = 1e-6 * min(abs(dx), abs(dy))
    return _merge_seams(gdf, np.asarray(xs, float), np.asarray(ys, float), tol, connectivity, name)
