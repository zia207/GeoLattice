"""Mosaicking overlapping rasters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu


def mosaic(rasters: Sequence[xr.DataArray], method: Literal["first", "last", "min", "max", "mean"]
           = "first", *, crs=None, resolution: float | None = None,
           nodata: float = np.nan) -> xr.DataArray:
    """Merge rasters onto one grid (reprojecting to the first raster's CRS unless ``crs``).

    ``first``/``last`` give priority by list order; ``min``/``max``/``mean`` combine overlaps.
    """
    from rioxarray.merge import merge_arrays

    if not rasters:
        raise ValueError("No rasters to mosaic")
    rasters = [to_cpu(r) for r in rasters]
    crs = crs or rasters[0].rio.crs
    rasters = [r if r.rio.crs == crs else r.rio.reproject(crs, resolution=resolution)
               for r in rasters]
    rasters = [r.astype("float32") if not np.issubdtype(r.dtype, np.floating) else r for r in rasters]
    if method in ("first", "last", "min", "max"):
        return merge_arrays(list(rasters), method=method, nodata=nodata, res=resolution)
    if method == "mean":
        grid = merge_arrays(list(rasters), method="first", nodata=nodata, res=resolution)
        stack = [r.rio.reproject_match(grid, nodata=np.nan) for r in rasters]
        out = xr.concat(stack, "_m").mean("_m", skipna=True)
        return out.rio.write_crs(crs)
    raise ValueError(f"Unknown mosaic method {method!r}")
