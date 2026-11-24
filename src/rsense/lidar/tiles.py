"""Tiling large point clouds with buffers, parallel tile processing and merging results."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal

import numpy as np

from ..core.pointcloud import PointCloud


@dataclass(frozen=True)
class Tile:
    """A tile: ``core`` bounds (no overlap between tiles) and ``buffered`` bounds."""

    id: str
    row: int
    col: int
    core: tuple[float, float, float, float]
    buffered: tuple[float, float, float, float]


def tile_grid(bounds: tuple[float, float, float, float], size: float = 500.0, *,
              buffer: float = 20.0, align: bool = True) -> list[Tile]:
    """Regular tiles of ``size`` m covering ``bounds`` (minx, miny, maxx, maxy).

    The ``buffer`` (m) is added on every side so neighbourhood operations (ground filters,
    tree detection, metrics) do not suffer edge effects; trim results back to ``core``.
    """
    minx, miny, maxx, maxy = bounds
    if align:
        minx, miny = np.floor(minx / size) * size, np.floor(miny / size) * size
    nx = max(1, int(np.ceil((maxx - minx) / size)))
    ny = max(1, int(np.ceil((maxy - miny) / size)))
    tiles = []
    for r in range(ny):
        for c in range(nx):
            x0, y0 = minx + c * size, miny + r * size
            core = (x0, y0, x0 + size, y0 + size)
            buf = (x0 - buffer, y0 - buffer, x0 + size + buffer, y0 + size + buffer)
            tiles.append(Tile(f"{int(x0)}_{int(y0)}", r, c, core, buf))
    return tiles


def iter_tiles(pc: PointCloud, size: float = 500.0, *, buffer: float = 20.0,
               min_points: int = 1) -> Iterator[tuple[Tile, PointCloud]]:
    """Yield ``(tile, buffered sub-cloud)`` for every tile holding ``min_points`` points."""
    for t in tile_grid(pc.bbox, size, buffer=buffer):
        sub = pc.crop(*t.buffered)
        if len(sub) >= min_points:
            yield t, sub


def trim(obj, tile: Tile):
    """Clip a PointCloud or raster (DataArray/Dataset) back to the tile's core bounds."""
    minx, miny, maxx, maxy = tile.core
    if isinstance(obj, PointCloud):
        m = (obj.x >= minx) & (obj.x < maxx) & (obj.y >= miny) & (obj.y < maxy)
        return obj.filter(m)
    return obj.sel(x=slice(minx, maxx), y=slice(maxy, miny))


def merge_pointclouds(clouds: list[PointCloud]) -> PointCloud:
    """Concatenate point clouds (fields present in all of them are kept)."""
    clouds = [c.to_cpu() for c in clouds if c is not None and len(c)]
    if not clouds:
        raise ValueError("Nothing to merge")
    common = set(clouds[0].attrs).intersection(*[c.attrs for c in clouds[1:]])
    cat = np.concatenate
    return PointCloud(cat([c.x for c in clouds]), cat([c.y for c in clouds]),
                      cat([c.z for c in clouds]),
                      {k: cat([c.attrs[k] for c in clouds]) for k in sorted(common)},
                      clouds[0].crs)


def merge_rasters(rasters: list):
    """Mosaic tile rasters (first valid value wins) into one DataArray/Dataset."""
    import xarray as xr

    rasters = [r for r in rasters if r is not None]
    if not rasters:
        raise ValueError("Nothing to merge")
    if isinstance(rasters[0], xr.Dataset):
        return xr.Dataset({v: merge_rasters([r[v] for r in rasters]) for v in rasters[0].data_vars})
    from rioxarray.merge import merge_arrays

    crs = rasters[0].rio.crs
    return merge_arrays([r.rio.write_nodata(np.nan, encoded=False) if crs is not None else r
                         for r in rasters]).rename(rasters[0].name)


def process_tiles(pc: PointCloud, func: Callable[[PointCloud], Any], *, size: float = 500.0,
                  buffer: float = 20.0, n_jobs: int = 1,
                  executor: Literal["thread", "process"] = "thread", min_points: int = 10,
                  merge: bool = True, progress: bool = False):
    """Run ``func(sub_cloud)`` on every buffered tile and merge the trimmed results.

    ``func`` may return a PointCloud (trimmed to the tile core, then concatenated), an
    ``xarray`` raster (trimmed, then mosaicked), a GeoDataFrame (features whose centroid falls
    in the core are kept, then concatenated) or anything else (returned as a list).
    ``n_jobs > 1`` runs tiles in parallel threads (NumPy/SciPy release the GIL) or processes
    (``func`` must then be picklable, i.e. defined at module level).
    """
    work = list(iter_tiles(pc, size, buffer=buffer, min_points=min_points))

    def run(item):
        t, sub = item
        return t, func(sub)

    if n_jobs == 1:
        it = map(run, work)
    else:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        Ex = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
        ex = Ex(max_workers=n_jobs if n_jobs > 0 else os.cpu_count())
        it = ex.map(run, work) if executor == "thread" else \
            ex.map(_run_process, [(func, t, sub) for t, sub in work])
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, total=len(work))
        except ImportError:
            pass
    results = list(it)
    if n_jobs != 1:
        ex.shutdown()
    if not merge:
        return results
    outs = [_trim_any(r, t) for t, r in results]
    first = next((o for o in outs if o is not None), None)
    if isinstance(first, PointCloud):
        return merge_pointclouds(outs)
    if hasattr(first, "rio"):
        return merge_rasters(outs)
    if hasattr(first, "geometry"):
        import pandas as pd

        return pd.concat([o for o in outs if o is not None], ignore_index=True)
    return outs


def _run_process(args):
    func, t, sub = args
    return t, func(sub)


def _trim_any(obj, t: Tile):
    if obj is None:
        return None
    if isinstance(obj, PointCloud) or hasattr(obj, "rio"):
        return trim(obj, t)
    if hasattr(obj, "geometry"):
        if len(obj) == 0:
            return obj
        c = obj.geometry.centroid
        minx, miny, maxx, maxy = t.core
        return obj[(c.x >= minx) & (c.x < maxx) & (c.y >= miny) & (c.y < maxy)]
    return obj


def write_tiles(pc: PointCloud, out_dir: str | PathLike, size: float = 500.0, *,
                buffer: float = 0.0, prefix: str = "tile", ext: str = ".laz") -> list[str]:
    """Split a point cloud into LAS/LAZ tiles ``{prefix}_{x0}_{y0}{ext}`` and return the paths."""
    from .io import write_las

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for t, sub in iter_tiles(pc, size, buffer=buffer):
        p = os.path.join(out_dir, f"{prefix}_{t.id}{ext}")
        write_las(sub, p)
        paths.append(p)
    return paths


def process_files(paths: list[str | PathLike], func: Callable[[PointCloud], Any], *,
                  n_jobs: int = 1, read_kwargs: dict | None = None) -> list:
    """Apply ``func`` to each LAS/LAZ file (read with :func:`read_las`) in parallel threads."""
    from .io import read_las

    def run(p):
        return func(read_las(p, **(read_kwargs or {})))

    if n_jobs == 1:
        return [run(p) for p in paths]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(run, paths))
