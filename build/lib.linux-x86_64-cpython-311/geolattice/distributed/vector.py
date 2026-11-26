"""Partition-parallel vector processing.

Inputs can be a ``geopandas`` GeoDataFrame/GeoSeries or, if ``dask-geopandas`` is installed,
a ``dask_geopandas`` frame.

* GeoPandas input is split into partitions, each partition runs as a Dask task (on the
  active :func:`~geolattice.distributed.start` session, or on local threads, where Shapely 2
  releases the GIL), and the result comes back as GeoPandas in the input's row order.
* dask-geopandas input stays lazy (``map_partitions``).

Geometry operations return the input type with the geometry replaced, so attribute
columns survive a pipeline. Measures (``area``, ``length``, ``bounds``) return pandas
objects.

GPU implementations (CuPy and cuDF kernels in :mod:`._gpu_geom`) exist for ``area``,
``length``, ``bounds``, point-in-polygon ``sjoin`` and point ``knn``. The measures and ``knn``
are opt-in (``compute="gpu"``): on typical inputs the CPU is faster, so ``"auto"`` keeps it. Other
operations, or inputs a GPU kernel does not handle (mixed geometry families), run on
CPU; an explicit ``compute="gpu"`` request then warns once.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from ._backend import Compute, get_defaults
from ._partition import spatial_partition
from ._registry import dispatch, register

_POS = "__geolattice_pos__"


# --------------------------------------------------------------------------- plumbing
def _is_dask_gdf(obj: Any) -> bool:
    return type(obj).__module__.startswith("dask_geopandas")


def _concat(parts: list[Any], like: Any) -> Any:
    if not parts:
        return like.iloc[0:0]
    out = pd.concat(parts)
    if hasattr(like, "crs") and hasattr(out, "set_crs") and out.crs is None and like.crs is not None:
        out = out.set_crs(like.crs)
    return out


def _run_partitions(func: Callable[[Any], Any], parts: list[Any]) -> list[Any]:
    import dask

    if len(parts) == 1:
        return [func(parts[0])]
    return list(dask.compute(*[dask.delayed(func)(p) for p in parts]))


def _npartitions(npartitions: int | None, used: str) -> int | None:
    """Explicit value, else the default; GPU work uses one partition per GPU worker."""
    npartitions = npartitions or get_defaults().npartitions
    if npartitions is None and used == "gpu":
        from ._backend import gpu_count
        from ._session import get_session

        s = get_session()
        npartitions = s.status()["workers"] if s is not None and s.is_gpu else max(gpu_count(), 1)
    return npartitions


def _apply(obj: Any, op: str, func: Callable[[Any], Any] | None = None, *,
           npartitions: int | None, compute: Compute | None) -> Any:
    """Run a per-feature function partition by partition, keeping row order.

    ``func`` defaults to the registered implementation for ``op`` (CPU or GPU).
    """
    impl, used = dispatch(op, compute)
    func = func or impl
    if _is_dask_gdf(obj):
        return obj.map_partitions(func)
    parts = spatial_partition(obj, _npartitions(npartitions, used), method="none")
    return _concat(_run_partitions(func, parts), obj)


def _replace_geometry(part: Any, geom: Any) -> Any:
    if hasattr(part, "geometry") and hasattr(part, "columns"):  # GeoDataFrame
        out = part.copy()
        out[part.geometry.name] = geom
        return out
    return geom


# --------------------------------------------------------------------------- per-feature ops
def _geom_op(name: str, fn: Callable[..., Any]):
    op = f"vector.{name}"

    def cpu(part: Any, **kw: Any) -> Any:
        return _replace_geometry(part, fn(part.geometry if hasattr(part, "columns") else part, **kw))

    register(op, "cpu")(cpu)
    return op, cpu


_BUFFER = _geom_op("buffer", lambda g, distance, **kw: g.buffer(distance, **kw))
_SIMPLIFY = _geom_op("simplify", lambda g, tolerance, preserve_topology=True:
                     g.simplify(tolerance, preserve_topology=preserve_topology))
_CENTROID = _geom_op("centroid", lambda g: g.centroid)
_MAKE_VALID = _geom_op("make_valid", lambda g: g.make_valid())


def buffer(gdf: Any, distance: float | Any, *, npartitions: int | None = None,
           compute: Compute | None = None, **kwargs: Any) -> Any:
    """Buffer each geometry by ``distance`` (CRS units). ``kwargs`` go to ``GeoSeries.buffer``."""
    if not np.isscalar(distance):
        raise TypeError("distance must be a scalar; per-row distances are not supported yet")
    op, cpu = _BUFFER
    return _apply(gdf, op, partial(cpu, distance=distance, **kwargs),
                  npartitions=npartitions, compute=compute)


def simplify(gdf: Any, tolerance: float, *, preserve_topology: bool = True,
             npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """Douglas-Peucker simplification of each geometry."""
    op, cpu = _SIMPLIFY
    return _apply(gdf, op, partial(cpu, tolerance=tolerance, preserve_topology=preserve_topology),
                  npartitions=npartitions, compute=compute)


def centroid(gdf: Any, *, npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """Replace each geometry by its centroid."""
    op, cpu = _CENTROID
    return _apply(gdf, op, cpu, npartitions=npartitions, compute=compute)


def make_valid(gdf: Any, *, npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """Repair invalid geometries (``shapely.make_valid``)."""
    op, cpu = _MAKE_VALID
    return _apply(gdf, op, cpu, npartitions=npartitions, compute=compute)


@register("vector.to_crs", "cpu")
def _to_crs_cpu(part: Any, crs: Any) -> Any:
    return part.to_crs(crs)


def to_crs(gdf: Any, crs: Any, *, npartitions: int | None = None,
           compute: Compute | None = None) -> Any:
    """Reproject to ``crs``."""
    if gdf.crs is None:
        raise ValueError("Input has no CRS; set one with .set_crs() first")
    return _apply(gdf, "vector.to_crs", partial(_to_crs_cpu, crs=crs),
                  npartitions=npartitions, compute=compute)


@register("vector.area", "cpu")
def _area_cpu(part: Any) -> Any:
    return part.area


@register("vector.length", "cpu")
def _length_cpu(part: Any) -> Any:
    return part.length


@register("vector.bounds", "cpu")
def _bounds_cpu(part: Any) -> Any:
    return part.bounds


def _gpu_measure(part: Any, kernel: str, cpu: Callable[[Any], Any]) -> Any:
    from . import _gpu_geom as G

    geoms = part.geometry if hasattr(part, "columns") else part
    if len(geoms) == 0 or G.family(geoms) is None:
        return cpu(part)
    r = G.to_ragged(geoms)
    if kernel == "bounds":
        return pd.DataFrame(G.bounds(r), index=part.index, columns=["minx", "miny", "maxx", "maxy"])
    return pd.Series(getattr(G, kernel)(r), index=part.index)


# Packing geometries into flat arrays costs far more than the arithmetic of a simple measure
# (measured: 1.6 million building polygons, area 0.03 s on 24 CPU threads vs 1.2 s on the GPU),
# so the GPU measures and the GPU knn below are opt-in: compute="auto" keeps the CPU.
@register("vector.area", "gpu", auto=False)
def _area_gpu(part: Any) -> Any:
    return _gpu_measure(part, "area", _area_cpu)


@register("vector.length", "gpu", auto=False)
def _length_gpu(part: Any) -> Any:
    return _gpu_measure(part, "length", _length_cpu)


@register("vector.bounds", "gpu", auto=False)
def _bounds_gpu(part: Any) -> Any:
    return _gpu_measure(part, "bounds", _bounds_cpu)


def area(gdf: Any, *, npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """Area of each geometry (CRS units squared), as a Series aligned to the input index."""
    return _apply(gdf, "vector.area", npartitions=npartitions, compute=compute)


def length(gdf: Any, *, npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """Length or perimeter of each geometry, as a Series aligned to the input index."""
    return _apply(gdf, "vector.length", npartitions=npartitions, compute=compute)


def bounds(gdf: Any, *, npartitions: int | None = None, compute: Compute | None = None) -> Any:
    """``minx, miny, maxx, maxy`` of each geometry, as a DataFrame."""
    return _apply(gdf, "vector.bounds", npartitions=npartitions, compute=compute)


# --------------------------------------------------------------------------- selection
@register("vector.clip", "cpu")
def _clip_cpu(part: Any, mask: Any, keep_geom_type: bool = False) -> Any:
    import geopandas as gpd

    return gpd.clip(part, mask, keep_geom_type=keep_geom_type)


def clip(gdf: Any, mask: Any, *, keep_geom_type: bool = False, npartitions: int | None = None,
         compute: Compute | None = None) -> Any:
    """Clip features to ``mask`` (a geometry, GeoSeries/GeoDataFrame or ``(minx, miny, maxx, maxy)``).

    Features are partitioned along a Hilbert curve and partitions whose bounding box
    misses the mask are dropped before any clipping work is scheduled.
    """
    dispatch("vector.clip", compute)
    func = partial(_clip_cpu, mask=mask, keep_geom_type=keep_geom_type)
    if _is_dask_gdf(gdf):
        return gdf.map_partitions(func)
    if not hasattr(gdf, "columns"):  # GeoSeries
        return clip(gdf.to_frame(), mask, keep_geom_type=keep_geom_type,
                    npartitions=npartitions, compute=compute).geometry
    mbounds = _mask_bounds(mask)
    tagged = gdf.assign(**{_POS: np.arange(len(gdf))})
    parts = spatial_partition(tagged, npartitions or get_defaults().npartitions, method="hilbert")
    parts = [p for p in parts if _bbox_hit(p.total_bounds, mbounds)]
    out = _concat(_run_partitions(func, parts), tagged)
    return out.sort_values(_POS, kind="stable").drop(columns=_POS)


def _mask_bounds(mask: Any) -> np.ndarray:
    if isinstance(mask, (tuple, list, np.ndarray)) and len(mask) == 4:
        return np.asarray(mask, dtype=float)
    if hasattr(mask, "total_bounds"):
        return np.asarray(mask.total_bounds, dtype=float)
    return np.asarray(mask.bounds, dtype=float)


def _bbox_hit(a: np.ndarray, b: np.ndarray) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# --------------------------------------------------------------------------- joins
_BBOX_PREDICATES = {"intersects", "within", "contains", "contains_properly", "covers",
                    "covered_by", "touches", "crosses", "overlaps", "dwithin"}


@register("vector.sjoin", "cpu")
def _sjoin_cpu(part: Any, right: Any, **kw: Any) -> Any:
    import geopandas as gpd

    return gpd.sjoin(part, right, **kw)


def _only_points(geoms: Any) -> bool:
    import shapely

    t = shapely.get_type_id(np.asarray(geoms.values, dtype=object))
    return bool(len(t)) and bool(np.all((t == 0) | (t == -1)))


@register("vector.sjoin", "gpu")
def _sjoin_gpu(part: Any, right: Any, **kw: Any) -> Any:
    """Point-in-polygon join on the GPU; other geometry/predicate combinations use CPU.

    Candidate pairs come from the polygon STRtree's bounding-box query (fast, vectorised C);
    the exact crossing-number test for every candidate runs as one CUDA kernel.
    """
    from . import _gpu_geom as G

    pred = kw.get("predicate", "intersects")
    if len(part) == 0 or len(right) == 0:
        return _sjoin_cpu(part, right, **kw)
    if pred in ("intersects", "within") and _only_points(part.geometry) \
            and G.family(right.geometry) == "polygon":
        pts, polys, swap = part, right, False
    elif pred in ("intersects", "contains") and _only_points(right.geometry) \
            and G.family(part.geometry) == "polygon":
        pts, polys, swap = right, part, True
    else:
        return _sjoin_cpu(part, right, **kw)
    try:
        from geopandas.tools.sjoin import _frame_join
    except ImportError:  # pragma: no cover - private geopandas API moved
        return _sjoin_cpu(part, right, **kw)

    import shapely

    pg = pts.geometry.values
    pi, gi = polys.sindex.query(pg, predicate=None)
    keep = G.points_in_polygons(shapely.get_x(pg), shapely.get_y(pg), G.to_ragged(polys.geometry),
                                pi, gi)
    li, ri = (gi[keep], pi[keep]) if swap else (pi[keep], gi[keep])
    order = np.lexsort((ri, li))
    joined, _ = _frame_join(part, right, (li[order], ri[order]), None, kw.get("how", "inner"),
                            kw.get("lsuffix", "left"), kw.get("rsuffix", "right"), pred)
    return joined



def sjoin(left: Any, right: Any, *, how: str = "inner", predicate: str = "intersects",
          distance: float | None = None, lsuffix: str = "left", rsuffix: str = "right",
          npartitions: int | None = None, compute: Compute | None = None,
          engine: str = "auto") -> Any:
    """Spatial join, parallel over spatially partitioned ``left``.

    ``left`` is cut into Hilbert-ordered partitions. For each partition only the rows of
    ``right`` whose bounding boxes touch that partition's bounding box (grown by
    ``distance`` for ``predicate="dwithin"``) are sent to its task, so work and data
    transfer shrink with the number of partitions. Results match ``geopandas.sjoin``
    (same columns, ``index_right``), in ``left`` row order.

    ``how`` is ``"inner"`` or ``"left"``. ``engine="sedona"`` runs the join as Spark SQL
    with Apache Sedona (see :mod:`geolattice.distributed.sedona`); ``"auto"`` picks Sedona when
    either input is a Spark DataFrame, otherwise Dask.
    """
    from . import sedona as _sedona

    if engine not in ("auto", "dask", "sedona"):
        raise ValueError("engine must be 'auto', 'dask' or 'sedona'")
    if engine == "sedona" or (engine == "auto" and (_sedona.is_spark(left) or _sedona.is_spark(right))):
        return _sedona.sjoin(left, right, how=how, predicate=predicate, distance=distance,
                             lsuffix=lsuffix, rsuffix=rsuffix)
    if how not in ("inner", "left"):
        raise ValueError("how must be 'inner' or 'left' for a partitioned join")
    if predicate not in _BBOX_PREDICATES:
        raise ValueError(f"Unsupported predicate {predicate!r}")
    if predicate == "dwithin" and distance is None:
        raise ValueError("predicate='dwithin' needs distance=")
    impl, used = dispatch("vector.sjoin", compute)
    kw: dict[str, Any] = dict(how=how, predicate=predicate, lsuffix=lsuffix, rsuffix=rsuffix)
    if distance is not None:
        kw["distance"] = distance

    if _is_dask_gdf(left) or _is_dask_gdf(right):
        import dask_geopandas

        if used == "gpu" and _is_dask_gdf(left) and not _is_dask_gdf(right):
            return left.map_partitions(impl, right, **kw)
        if not _is_dask_gdf(left):
            left = dask_geopandas.from_geopandas(left, npartitions=npartitions or 1)
        if not _is_dask_gdf(right):
            right = dask_geopandas.from_geopandas(right, npartitions=1)
        extra = {"distance": distance} if distance is not None else {}
        return dask_geopandas.sjoin(left, right, how=how, predicate=predicate, **extra)

    if left.crs is not None and right.crs is not None and left.crs != right.crs:
        raise ValueError(f"CRS mismatch: left {left.crs} vs right {right.crs}; reproject first")

    from shapely import box

    pad = float(distance or 0.0)
    tagged = left.assign(**{_POS: np.arange(len(left))})
    # CPU threads still drive the bbox query and result assembly, so a GPU join keeps the
    # CPU partition count; each partition's exact test is one kernel launch.
    parts = spatial_partition(tagged, npartitions or get_defaults().npartitions, method="hilbert")
    tree = right.sindex
    jobs = []
    for p in parts:
        x0, y0, x1, y1 = p.total_bounds
        idx = tree.query(box(x0 - pad, y0 - pad, x1 + pad, y1 + pad), predicate="intersects")
        if len(idx) or how == "left":
            jobs.append((p, right.iloc[np.sort(idx)]))

    import dask

    if len(jobs) == 1:
        results = [impl(jobs[0][0], jobs[0][1], **kw)]
    else:
        results = list(dask.compute(*[dask.delayed(impl)(p, r, **kw) for p, r in jobs]))
    if not results:
        out = impl(tagged.iloc[0:0], right.iloc[0:0], **kw)
    else:
        out = _concat(results, left)
    out = out.sort_values(_POS, kind="stable")
    return out.drop(columns=_POS)


# --------------------------------------------------------------------------- nearest neighbours
def _xy(geoms: Any) -> np.ndarray:
    import shapely

    return np.column_stack([shapely.get_x(geoms.values), shapely.get_y(geoms.values)])


def _knn_result(left_index: Any, right_index: Any, dist: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    n, k = idx.shape
    return pd.DataFrame({
        "index_right": np.asarray(right_index)[idx.ravel()],
        "distance": dist.ravel(),
        "k": np.tile(np.arange(1, k + 1), n),
    }, index=np.repeat(np.asarray(left_index), k))


@register("vector.knn", "cpu")
def _knn_cpu(left_xy: np.ndarray, right_xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(right_xy).query(left_xy, k=k, workers=-1)
    return dist.reshape(len(left_xy), k), idx.reshape(len(left_xy), k)


# brute force: 184,000 points against themselves took 21 s on the GPU vs 0.1 s with the KD-tree
@register("vector.knn", "gpu", auto=False)
def _knn_gpu(left_xy: np.ndarray, right_xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact brute-force k-NN on the GPU in float64, tiled over ``left`` to bound memory.

    Distances use coordinate differences (not the expanded dot-product form), so they
    stay exact at large projected coordinates.
    """
    import cupy as cp

    rx = cp.asarray(right_xy[:, 0], dtype=cp.float64)
    ry = cp.asarray(right_xy[:, 1], dtype=cp.float64)
    m = len(right_xy)
    tile = max(1, int(2e8 // (8 * max(m, 1))))
    dist = np.empty((len(left_xy), k))
    idx = np.empty((len(left_xy), k), dtype=np.int64)
    for s in range(0, len(left_xy), tile):
        lx = cp.asarray(left_xy[s:s + tile, 0], dtype=cp.float64)[:, None]
        ly = cp.asarray(left_xy[s:s + tile, 1], dtype=cp.float64)[:, None]
        d2 = (lx - rx[None, :]) ** 2 + (ly - ry[None, :]) ** 2
        part = cp.argpartition(d2, k - 1, axis=1)[:, :k] if k < m else \
            cp.broadcast_to(cp.arange(m), (d2.shape[0], m))
        pd2 = cp.take_along_axis(d2, part, 1)
        order = cp.argsort(pd2, axis=1)
        idx[s:s + tile] = cp.asnumpy(cp.take_along_axis(part, order, 1))
        dist[s:s + tile] = cp.asnumpy(cp.sqrt(cp.take_along_axis(pd2, order, 1)))
    return dist, idx


def knn(left: Any, right: Any | None = None, *, k: int = 1, max_distance: float | None = None,
        compute: Compute | None = None) -> pd.DataFrame:
    """``k`` nearest ``right`` points for every ``left`` point (Euclidean, CRS units).

    Returns a long DataFrame indexed by the ``left`` index with ``index_right``,
    ``distance`` and ``k`` (rank, 1 = nearest). With ``right=None`` the neighbours come
    from ``left`` itself, excluding each point. ``max_distance`` drops farther pairs.
    CPU uses a multi-threaded KD-tree (SciPy); GPU an exact float64 brute-force search
    (CuPy), which pays off when ``right`` is small to moderate (up to ~10^5 points).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if _is_dask_gdf(left):
        return left.map_partitions(lambda p: knn(p, right, k=k, max_distance=max_distance,
                                                 compute=compute))
    self_join = right is None
    right = left if self_join else right
    if _is_dask_gdf(right):
        right = right.compute()
    for name, g in (("left", left), ("right", right)):
        if not _only_points(g.geometry):
            raise TypeError(f"knn needs Point geometries in {name}; use geopandas.sjoin_nearest")
    if not self_join and left.crs is not None and right.crs is not None and left.crs != right.crs:
        raise ValueError(f"CRS mismatch: left {left.crs} vs right {right.crs}; reproject first")
    impl, _ = dispatch("vector.knn", compute)
    kk = min(k + int(self_join), len(right))
    dist, idx = impl(_xy(left.geometry), _xy(right.geometry), kk)
    if self_join:
        rows = np.arange(len(left))[:, None]
        not_self = idx != rows
        # drop the self match from each row (it may not be first when points coincide)
        order = np.argsort(~not_self, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, 1)[:, :kk - 1]
        dist = np.take_along_axis(dist, order, 1)[:, :kk - 1]
    out = _knn_result(left.index, right.index, dist, idx)
    if max_distance is not None:
        out = out[out["distance"] <= max_distance]
    return out
