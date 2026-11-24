"""GPU geometry kernels on CuPy and cuDF (no cuSpatial needed).

Geometries are converted once with ``shapely.to_ragged_array`` into flat coordinate
arrays plus offsets (the GeoArrow layout), copied to the GPU, and processed by
vectorised CuPy code or small CUDA kernels:

* ``area``, ``length``: shoelace / segment sums with cumulative-sum segment reductions.
* ``bounds``: cuDF group-by min/max over vertices.
* ``points_in_polygons``: even-odd crossing test for candidate (point, polygon) pairs.
* ``rasterize_polygons``: pixel-centre burn of polygons into a grid ("last wins").

Points lying exactly on a polygon boundary may be classed either way (a property of
the crossing-number test); interior and exterior points match Shapely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_FAMILY = {0: "point", 1: "line", 2: "line", 3: "polygon", 4: "point", 5: "line", 6: "polygon"}
# shapely.GeometryType: 0 Point, 1 LineString, 2 LinearRing, 3 Polygon, 4 MultiPoint,
# 5 MultiLineString, 6 MultiPolygon


@dataclass
class Ragged:
    """Flat GPU arrays for one homogeneous geometry family.

    ``levels`` are offset arrays from outermost (geometry) to innermost (coordinates):
    polygons ``[geom->polygon, polygon->ring, ring->coord]``; lines
    ``[geom->line, line->coord]``; points ``[geom->coord]``.
    """

    family: str
    x: Any
    y: Any
    levels: list[Any]
    n: int


def to_ragged(geoms: Any) -> Ragged:
    """Convert a GeoSeries/array of one geometry family to :class:`Ragged` on the GPU."""
    import cupy as cp
    import shapely

    arr = np.asarray(getattr(geoms, "values", geoms), dtype=object)
    n = len(arr)
    if n == 0:
        z = cp.zeros(0)
        return Ragged("point", z, z, [cp.zeros(1, dtype=cp.int64)], 0)
    gtype, coords, offsets = shapely.to_ragged_array(arr)
    gtype = int(gtype)
    fam = {0: "point", 1: "line", 3: "polygon", 4: "point", 5: "line", 6: "polygon"}[gtype]
    offs = [np.asarray(o, dtype=np.int64) for o in offsets]
    ident = np.arange(n + 1, dtype=np.int64)
    if gtype == 0:      # Point: one coordinate per geometry (empties have NaN coords)
        levels = [ident]
    elif gtype == 4:    # MultiPoint
        levels = [offs[0]]
    elif gtype == 1:    # LineString
        levels = [ident, offs[0]]
    elif gtype == 5:    # MultiLineString
        levels = [offs[1], offs[0]]
    elif gtype == 3:    # Polygon
        levels = [ident, offs[1], offs[0]]
    else:               # MultiPolygon
        levels = [offs[2], offs[1], offs[0]]
    return Ragged(fam, cp.asarray(coords[:, 0], dtype=cp.float64),
                  cp.asarray(coords[:, 1], dtype=cp.float64),
                  [cp.asarray(o) for o in levels], n)


def family(geoms: Any) -> str | None:
    """``"point"``, ``"line"`` or ``"polygon"`` if all non-empty geometries share a family."""
    import shapely

    t = shapely.get_type_id(np.asarray(getattr(geoms, "values", geoms), dtype=object))
    fams = {_FAMILY.get(int(v)) for v in np.unique(t[t >= 0])}
    return fams.pop() if len(fams) == 1 else None


def _segment_sum(values: Any, offsets: Any) -> Any:
    """Sum ``values[offsets[i]:offsets[i+1]]`` for every segment i (empty segments give 0)."""
    import cupy as cp

    cs = cp.concatenate([cp.zeros(1, dtype=values.dtype), cp.cumsum(values)])
    return cs[offsets[1:]] - cs[offsets[:-1]]


def _edge_terms(r: Ragged, coord_off: Any) -> tuple[Any, Any, Any]:
    """Per-vertex (dx, dy, cross) to the next vertex, zeroed on each part's last vertex."""
    import cupy as cp

    x, y = r.x, r.y
    nx, ny = cp.roll(x, -1), cp.roll(y, -1)
    last = cp.zeros(len(x), dtype=bool)
    ends = coord_off[1:] - 1
    last[ends[ends >= coord_off[:-1]]] = True
    dx, dy = cp.where(last, 0, nx - x), cp.where(last, 0, ny - y)
    cross = cp.where(last, 0, x * ny - nx * y)
    return dx, dy, cross


def area(r: Ragged) -> np.ndarray:
    """Planar area of each geometry (exterior rings minus holes)."""
    import cupy as cp

    if r.family != "polygon":
        return np.zeros(r.n)
    geom_poly, poly_ring, ring_coord = r.levels
    _, _, cross = _edge_terms(r, ring_coord)
    ring_area = cp.abs(_segment_sum(cross, ring_coord)) / 2
    is_exterior = cp.zeros(len(ring_area), dtype=bool)
    starts = poly_ring[:-1][poly_ring[1:] > poly_ring[:-1]]
    is_exterior[starts] = True
    signed = cp.where(is_exterior, ring_area, -ring_area)
    return cp.asnumpy(_segment_sum(_segment_sum(signed, poly_ring), geom_poly))


def length(r: Ragged) -> np.ndarray:
    """Length of lines, perimeter of polygons (all rings), 0 for points."""
    import cupy as cp

    if r.family == "point":
        return np.zeros(r.n)
    dx, dy, _ = _edge_terms(r, r.levels[-1])
    seg = cp.sqrt(dx * dx + dy * dy)
    out = _segment_sum(seg, r.levels[-1])
    for off in reversed(r.levels[:-1]):
        out = _segment_sum(out, off)
    return cp.asnumpy(out)


def bounds(r: Ragged) -> np.ndarray:
    """``(n, 4)`` minx, miny, maxx, maxy per geometry with cuDF group-by (NaN for empties)."""
    import cupy as cp

    from ._backend import rapids

    cudf = rapids("cudf")

    coord_off = r.levels[-1]
    for off in reversed(r.levels[:-1]):
        coord_off = coord_off[off]
    counts = cp.diff(coord_off)
    gid = cp.repeat(cp.arange(r.n), counts.astype(cp.int64))
    df = cudf.DataFrame({"g": gid, "x": r.x[:len(gid)], "y": r.y[:len(gid)]})
    df = df[df["x"].notna()]
    agg = df.groupby("g").agg({"x": ["min", "max"], "y": ["min", "max"]})
    out = np.full((r.n, 4), np.nan)
    idx = agg.index.to_numpy()
    out[idx, 0] = agg[("x", "min")].to_numpy()
    out[idx, 1] = agg[("y", "min")].to_numpy()
    out[idx, 2] = agg[("x", "max")].to_numpy()
    out[idx, 3] = agg[("y", "max")].to_numpy()
    return out


# --------------------------------------------------------------------------- CUDA kernels
_PIP_SRC = r"""
extern "C" __device__ bool pip(double x, double y, long long g,
        const long long* geom_poly, const long long* poly_ring, const long long* ring_coord,
        const double* cx, const double* cy) {
    bool inside = false;
    for (long long p = geom_poly[g]; p < geom_poly[g + 1]; ++p)
        for (long long r = poly_ring[p]; r < poly_ring[p + 1]; ++r) {
            long long s = ring_coord[r], e = ring_coord[r + 1];
            for (long long v = s; v < e - 1; ++v) {
                double xi = cx[v], yi = cy[v], xj = cx[v + 1], yj = cy[v + 1];
                if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
                    inside = !inside;
            }
        }
    return inside;
}

extern "C" __global__ void pip_pairs(long long n, const long long* pt, const long long* pg,
        const double* px, const double* py,
        const long long* geom_poly, const long long* poly_ring, const long long* ring_coord,
        const double* cx, const double* cy, bool* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pip(px[pt[i]], py[pt[i]], pg[i], geom_poly, poly_ring, ring_coord, cx, cy);
}

extern "C" __global__ void burn(int H, int W, double x0, double dx, double y0, double dy,
        long long n, const double* bb, const double* val,
        const long long* geom_poly, const long long* poly_ring, const long long* ring_coord,
        const double* cx, const double* cy, double* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)H * W) return;
    int r = i / W, c = i % W;
    double x = x0 + (c + 0.5) * dx, y = y0 + (r + 0.5) * dy;
    for (long long k = 0; k < n; ++k) {
        if (x < bb[4 * k] || x > bb[4 * k + 2] || y < bb[4 * k + 1] || y > bb[4 * k + 3]) continue;
        if (pip(x, y, k, geom_poly, poly_ring, ring_coord, cx, cy)) out[i] = val[k];
    }
}
"""
_KERNELS: dict[str, Any] = {}


def _kernel(name: str):
    if name not in _KERNELS:
        import cupy as cp

        mod = cp.RawModule(code=_PIP_SRC)
        _KERNELS["pip_pairs"] = mod.get_function("pip_pairs")
        _KERNELS["burn"] = mod.get_function("burn")
    return _KERNELS[name]


def points_in_polygons(px: np.ndarray, py: np.ndarray, polys: Ragged, pt_idx: np.ndarray,
                       poly_idx: np.ndarray) -> np.ndarray:
    """For candidate pairs ``(pt_idx[i], poly_idx[i])``, True where the point is inside."""
    import cupy as cp

    if polys.family != "polygon":
        raise ValueError("points_in_polygons needs polygon geometries")
    n = len(pt_idx)
    if n == 0:
        return np.zeros(0, dtype=bool)
    out = cp.zeros(n, dtype=cp.bool_)
    gp, pr, rc = polys.levels
    threads = 256
    _kernel("pip_pairs")(((n + threads - 1) // threads,), (threads,), (
        np.int64(n), cp.asarray(pt_idx, dtype=cp.int64), cp.asarray(poly_idx, dtype=cp.int64),
        cp.asarray(px, dtype=cp.float64), cp.asarray(py, dtype=cp.float64),
        gp, pr, rc, polys.x, polys.y, out))
    return cp.asnumpy(out)


def rasterize_polygons(polys: Ragged, values: np.ndarray, bboxes: np.ndarray,
                       shape: tuple[int, int], x0: float, dx: float, y0: float, dy: float,
                       fill: float) -> Any:
    """Burn polygons into a ``shape`` grid (CuPy float64) by pixel centre; later polygons win.

    ``x0, y0`` is the outer corner of pixel (0, 0); ``dx, dy`` are signed pixel sizes.
    """
    import cupy as cp

    h, w = shape
    out = cp.full(h * w, fill, dtype=cp.float64)
    if polys.n:
        gp, pr, rc = polys.levels
        threads = 256
        _kernel("burn")((((h * w) + threads - 1) // threads,), (threads,), (
            np.int32(h), np.int32(w), np.float64(x0), np.float64(dx), np.float64(y0),
            np.float64(dy), np.int64(polys.n), cp.asarray(bboxes, dtype=cp.float64).ravel(),
            cp.asarray(values, dtype=cp.float64), gp, pr, rc, polys.x, polys.y, out))
    return out.reshape(h, w)
