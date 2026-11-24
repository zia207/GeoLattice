"""Spatial partitioning for vector data and chunk planning for rasters.

Spatial joins and overlays scale only if each partition covers a compact area, so a
partition's bounding box can rule out most of the other table. :func:`spatial_partition`
orders features along a Hilbert curve and cuts the order into equal-sized partitions.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

Method = Literal["hilbert", "none"]


def default_npartitions(n_rows: int, npartitions: int | None = None) -> int:
    """Partitions to use: the explicit value, else the active session's thread count
    (or ``os.cpu_count()``), capped so each partition holds at least ~1000 rows."""
    if npartitions:
        return max(1, min(int(npartitions), max(n_rows, 1)))
    from ._session import status

    threads = status().get("threads") or os.cpu_count() or 1
    return max(1, min(int(threads), max(n_rows // 1000, 1)))


def hilbert_order(gdf: Any, level: int = 16) -> np.ndarray:
    """Row order of ``gdf`` along a Hilbert curve through its feature bounding-box centres."""
    if len(gdf) == 0:
        return np.arange(0)
    dist = np.asarray(gdf.geometry.hilbert_distance(total_bounds=gdf.total_bounds, level=level))
    return np.argsort(dist, kind="stable")


def spatial_partition(gdf: Any, npartitions: int | None = None, *,
                      method: Method = "hilbert") -> list[Any]:
    """Split a GeoDataFrame into ``npartitions`` spatially compact GeoDataFrames.

    ``method="hilbert"`` sorts along a Hilbert curve first; ``"none"`` keeps row order
    (use for per-feature operations where locality does not matter). The original index
    is kept, so ``pd.concat(parts).loc[gdf.index]`` restores the input order.
    """
    n = default_npartitions(len(gdf), npartitions)
    if method == "hilbert":
        gdf = gdf.iloc[hilbert_order(gdf)]
    elif method != "none":
        raise ValueError("method must be 'hilbert' or 'none'")
    bounds = np.linspace(0, len(gdf), n + 1).astype(int)
    return [gdf.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def partition_bounds(parts: list[Any]) -> np.ndarray:
    """``(n, 4)`` array of ``minx, miny, maxx, maxy`` for each partition."""
    return np.array([p.total_bounds for p in parts], dtype=float).reshape(-1, 4)


def plan_chunks(shape: tuple[int, ...], dtype: Any = "float32", *, n_layers: int = 1,
                halo: int = 0, compute: str = "cpu", target_mb: float | None = None
                ) -> tuple[int, ...]:
    """Dask chunks for a raster: full leading dims, square spatial chunks.

    CPU targets ~128 MB chunks; GPU targets ~5% of free device memory (from
    :func:`geolattice.rsense.backend.recommend_chunks`). Chunks are kept at least
    ``8 * halo`` pixels wide so the overlap stays a small fraction of each block.
    """
    from ..rsense.backend import recommend_chunks

    if target_mb is None and compute == "cpu":
        target_mb = 128
    chunks = list(recommend_chunks(tuple(shape), dtype, n_layers=n_layers, target_mb=target_mb))
    for i in (-2, -1):
        chunks[i] = min(shape[i], max(chunks[i], 8 * halo))
    return tuple(chunks)
