"""Hydrology on DEMs.

D8 flow direction is local (each cell looks at its 8 neighbours), so :func:`flow_direction`
runs chunk by chunk with a 1-pixel halo on CPU or GPU and matches the whole-array result.

Depression filling and flow accumulation are global: a cell's answer can depend on cells
anywhere upstream or across the DEM. :func:`fill_depressions` and :func:`flow_accumulation`
therefore compute in memory with the existing :mod:`geolattice.rsense.lidar.terrain` algorithms
(priority-flood; one sparse linear solve). They accept lazy input but load it whole. For
DEMs that do not fit in memory, use the WhiteboxTools wrappers in :mod:`geolattice.whitebox`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ._backend import Compute, resolve_compute, to_cpu
from ._registry import register
from .raster import _finish, _map_overlap, _prepare, _res

# N, NE, E, SE, S, SW, W, NW, the same codes as geolattice.rsense.lidar.terrain.flow_direction
D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def flow_direction(dem: Any, *, res: Any = None, compute: Compute | None = None,
                   chunks: Any = None) -> Any:
    """D8 flow direction 0-7 (N, NE, E, SE, S, SW, W, NW), -1 for pits, flats draining
    nowhere lower and cells whose steepest descent leaves the DEM. Fill depressions first
    for a fully connected drainage network."""
    dx, dy, _ = _res(dem, res)
    dists = [float(np.hypot(dr * dy, dc * dx)) for dr, dc in D8]

    def f(z, xp):
        z = z.astype(xp.float64)
        r, c = z.shape[-2:]
        p = xp.pad(z, [(0, 0)] * (z.ndim - 2) + [(1, 1), (1, 1)], constant_values=np.nan)
        best = xp.zeros(z.shape)
        out = xp.full(z.shape, -1, dtype=xp.int8)
        for k, (dr, dc) in enumerate(D8):
            nb = p[..., 1 + dr:1 + dr + r, 1 + dc:1 + dc + c]
            drop = xp.nan_to_num((z - nb) / dists[k], nan=-np.inf)
            better = drop > best
            best = xp.where(better, drop, best)
            out = xp.where(better, xp.int8(k), out)
        return out

    da, ctx = _prepare(dem, compute=compute, chunks=chunks, op="hydrology.flow_direction", halo=1)
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype("float64")
    out = _finish(_map_overlap(da, f, halo=1, dtype="int8", boundary=np.nan), ctx)
    return out.rename("flow_direction") if isinstance(out, xr.DataArray) else out


register("hydrology.flow_direction", "cpu")(flow_direction)
register("hydrology.flow_direction", "gpu")(flow_direction)


def _in_memory(dem: Any) -> Any:
    if isinstance(dem, xr.DataArray):
        return to_cpu(dem).compute() if hasattr(dem.data, "dask") else to_cpu(dem)
    return np.asarray(to_cpu(dem.compute() if hasattr(dem, "dask") else dem))


@register("hydrology.fill_depressions", "cpu")
def fill_depressions(dem: Any, *, epsilon: float = 1e-4, compute: Compute | None = None) -> Any:
    """Priority-flood depression filling (Barnes et al. 2014), in memory on CPU."""
    from ..rsense.lidar.terrain import fill_depressions as _fill

    resolve_compute(compute, op="hydrology.fill_depressions", gpu_available=False)
    return _fill(_in_memory(dem), epsilon=epsilon)


@register("hydrology.flow_accumulation", "cpu")
def flow_accumulation(dem: Any, *, res: Any = None, filled: bool = False, weights: Any = None,
                      compute: Compute | None = None) -> Any:
    """D8 contributing cells (or summed ``weights``), solved as one sparse linear system
    in memory on CPU. Fills depressions first unless ``filled=True``."""
    from ..rsense.lidar.terrain import flow_accumulation as _acc

    resolve_compute(compute, op="hydrology.flow_accumulation", gpu_available=False)
    w = None if weights is None else _in_memory(weights)
    return _acc(_in_memory(dem), res=res, filled=filled, weights=w)
