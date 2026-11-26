"""Terrain derivatives from DEMs.

Local-window derivatives (slope, aspect, hillshade, curvature, TPI, TRI,
illumination, geomorphons) run on the active backend. Hydrological tools
(depression filling, D8 flow, accumulation, TWI) are sequential algorithms and
run on CPU; for very large DEMs use the optional WhiteboxTools wrappers.
"""

from __future__ import annotations

import heapq
import warnings
from typing import Any

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..core._utils import as_float, like2d, nan_box_mean, run_array


# ----------------------------------------------------------------------------- helpers
def _res(dem: Any, res: float | tuple[float, float] | None) -> tuple[float, float, bool]:
    """Return (dx, dy, north_up) in map units."""
    if res is not None:
        dx, dy = (res, res) if np.isscalar(res) else res
        return float(dx), float(dy), True
    if isinstance(dem, xr.DataArray):
        ydim, xdim = dem.dims[-2:]
        x, y = dem[xdim].values, dem[ydim].values
        dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
        crs = dem.rio.crs
        if crs is not None and crs.is_geographic:
            lat = float(np.mean(y))
            warnings.warn("DEM is in geographic coordinates; converting degrees to metres at the "
                          "scene centre. Reproject to a projected CRS for accurate results.",
                          UserWarning, stacklevel=3)
            dx, dy = dx * 111_320 * np.cos(np.radians(lat)), dy * 110_540
        return dx, dy, bool(y[1] < y[0])
    raise ValueError("Pass res= for plain arrays")


def _neighbors(z: Any, xp: Any) -> dict[str, Any]:
    """3x3 neighbourhood as shifted views (edge-padded): keys a..i, row-major from top-left."""
    p = xp.pad(z, 1, mode="edge")
    keys = "abcdefghi"
    out = {}
    for k, (dr, dc) in zip(keys, [(r, c) for r in range(3) for c in range(3)]):
        out[k] = p[dr:dr + z.shape[0], dc:dc + z.shape[1]]
    return out


def _gradients(z: Any, xp: Any, dx: float, dy: float, north_up: bool, z_factor: float):
    n = _neighbors(z * z_factor, xp)
    gx = ((n["c"] + 2 * n["f"] + n["i"]) - (n["a"] + 2 * n["d"] + n["g"])) / (8 * dx)
    gn = ((n["a"] + 2 * n["b"] + n["c"]) - (n["g"] + 2 * n["h"] + n["i"])) / (8 * dy)
    if not north_up:
        gn = -gn
    return gx, gn  # east and north components of the gradient


# ----------------------------------------------------------------------------- derivatives
def slope(dem: Any, *, units: str = "degrees", res: Any = None, z_factor: float = 1.0):
    """Slope (Horn 1981) in ``degrees``, ``radians`` or ``percent``."""
    dx, dy, nu = _res(dem, res)

    def f(z, xp):
        gx, gn = _gradients(as_float(z, xp), xp, dx, dy, nu, z_factor)
        g = xp.sqrt(gx**2 + gn**2)
        if units == "percent":
            return g * 100
        s = xp.arctan(g)
        return xp.degrees(s) if units == "degrees" else s

    return _name(run_array(dem, f), "slope", units)


def aspect(dem: Any, *, res: Any = None, flat: float = np.nan):
    """Aspect: compass direction of steepest descent, degrees clockwise from north."""
    dx, dy, nu = _res(dem, res)

    def f(z, xp):
        gx, gn = _gradients(as_float(z, xp), xp, dx, dy, nu, 1.0)
        a = xp.degrees(xp.arctan2(-gx, -gn)) % 360
        return xp.where((gx == 0) & (gn == 0), flat, a)

    return _name(run_array(dem, f), "aspect", "degrees")


def illumination(dem: Any, sun_azimuth: float, sun_elevation: float, *, res: Any = None,
                 z_factor: float = 1.0):
    """Cosine of the local solar incidence angle, cos(i), used for topographic correction."""
    dx, dy, nu = _res(dem, res)
    zen, az = np.radians(90 - sun_elevation), np.radians(sun_azimuth)

    def f(z, xp):
        gx, gn = _gradients(as_float(z, xp), xp, dx, dy, nu, z_factor)
        s = xp.arctan(xp.sqrt(gx**2 + gn**2))
        a = xp.arctan2(-gx, -gn)
        return np.cos(zen) * xp.cos(s) + np.sin(zen) * xp.sin(s) * xp.cos(az - a)

    return _name(run_array(dem, f), "cos_i", "")


def hillshade(dem: Any, *, azimuth: float = 315, altitude: float = 45, res: Any = None,
              z_factor: float = 1.0):
    """Analytical hillshade scaled 0-255."""
    cos_i = illumination(dem, azimuth, altitude, res=res, z_factor=z_factor)
    return _name(run_array(cos_i, lambda c, xp: xp.clip(c, 0, 1) * 255), "hillshade", "")


def curvature(dem: Any, *, res: Any = None):
    """Total (Laplacian) curvature, 1/m: negative = convex (ridges), positive = concave."""
    dx, dy, _ = _res(dem, res)

    def f(z, xp):
        n = _neighbors(as_float(z, xp), xp)
        d2x = (n["d"] - 2 * n["e"] + n["f"]) / dx**2
        d2y = (n["b"] - 2 * n["e"] + n["h"]) / dy**2
        return d2x + d2y

    return _name(run_array(dem, f), "curvature", "1/m")


def tpi(dem: Any, *, window: int = 9):
    """Topographic Position Index: elevation minus the mean of a ``window``×``window`` neighbourhood."""
    return _name(run_array(dem, lambda z, xp: as_float(z, xp) - nan_box_mean(as_float(z, xp),
                                                                              window, xp)), "tpi", "m")


def tri(dem: Any):
    """Terrain Ruggedness Index (Riley et al. 1999): sqrt of summed squared differences to 8 neighbours."""
    def f(z, xp):
        z = as_float(z, xp)
        n = _neighbors(z, xp)
        return xp.sqrt(sum((n[k] - z) ** 2 for k in "abcdfghi"))

    return _name(run_array(dem, f), "tri", "m")


def roughness(dem: Any):
    """Max minus min elevation in the 3×3 neighbourhood."""
    def f(z, xp):
        n = _neighbors(as_float(z, xp), xp)
        stack = xp.stack([n[k] for k in "abcdefghi"])
        return stack.max(0) - stack.min(0)

    return _name(run_array(dem, f), "roughness", "m")


# ----------------------------------------------------------------------------- hydrology (CPU)
_D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # N, NE, E, ...


def fill_depressions(dem: Any, *, epsilon: float = 1e-4):
    """Priority-flood depression filling (Barnes et al. 2014), with a small ``epsilon`` gradient
    so filled flats still drain. Runs on CPU (sequential algorithm)."""
    z = np.array(to_cpu(dem.data if isinstance(dem, xr.DataArray) else dem), dtype="float64")
    rows, cols = z.shape
    nodata = np.isnan(z)
    done = nodata.copy()
    heap: list[tuple[float, int, int]] = []
    edge = np.zeros_like(done)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    # cells next to nodata also act as outlets
    pad = np.pad(nodata, 1, constant_values=False)
    near_nan = np.zeros_like(nodata)
    for dr, dc in _D8:
        near_nan |= pad[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    seeds = (edge | near_nan) & ~nodata
    for r, c in zip(*np.nonzero(seeds)):
        heapq.heappush(heap, (z[r, c], r, c))
        done[r, c] = True
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and not done[rr, cc]:
                done[rr, cc] = True
                if z[rr, cc] <= h:
                    z[rr, cc] = h + epsilon
                heapq.heappush(heap, (z[rr, cc], rr, cc))
    out = z.astype("float32")
    return like2d(dem, out, "dem_filled") if isinstance(dem, xr.DataArray) else out


def flow_direction(dem: Any, *, res: Any = None):
    """D8 flow direction as an index 0-7 (N, NE, E, SE, S, SW, W, NW); -1 for sinks/edges."""
    dx, dy, _ = _res(dem, res)
    z = np.asarray(to_cpu(dem.data if isinstance(dem, xr.DataArray) else dem), dtype="float64")
    p = np.pad(z, 1, constant_values=np.nan)
    rows, cols = z.shape
    best = np.full(z.shape, 0.0)
    out = np.full(z.shape, -1, dtype="int8")
    for k, (dr, dc) in enumerate(_D8):
        dist = np.hypot(dr * dy, dc * dx)
        nb = p[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
        drop = (z - nb) / dist
        better = np.nan_to_num(drop, nan=-np.inf) > best
        best = np.where(better, drop, best)
        out[better] = k
    return like2d(dem, out, "flow_direction") if isinstance(dem, xr.DataArray) else out


def _receivers(fd: np.ndarray) -> np.ndarray:
    """Flat index of each cell's D8 receiver (-1 for sinks, edges and nodata)."""
    rows, cols = fd.shape
    dr = np.array([d[0] for d in _D8])
    dc = np.array([d[1] for d in _D8])
    r, c = np.indices(fd.shape)
    k = np.where(fd >= 0, fd, 0)
    rr, cc = r + dr[k], c + dc[k]
    ok = (fd >= 0) & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    return np.where(ok, rr * cols + cc, -1).ravel()


def _accumulate(rec: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Sum ``w`` down a D8 forest (``rec`` = receiver index or -1) in topological order.

    Kahn's algorithm, one vectorised step per level: cells with no remaining donors pass their
    total to their receiver together. Linear in memory and time; the number of steps is the
    longest flow path in cells.
    """
    acc = w.astype("float64").copy()
    has = rec >= 0
    indeg = np.bincount(rec[has], minlength=rec.size)
    frontier = np.flatnonzero(indeg == 0)
    while frontier.size:
        frontier = frontier[has[frontier]]  # outlets pass nothing on
        if not frontier.size:
            break
        r = rec[frontier]
        np.add.at(acc, r, acc[frontier])
        np.subtract.at(indeg, r, 1)
        r = np.unique(r)
        frontier = r[indeg[r] == 0]
    return acc


def flow_accumulation(dem: Any, *, res: Any = None, filled: bool = False, weights: Any = None):
    """D8 upslope contributing cells (including the cell itself). Fills depressions first
    unless ``filled=True``. ``weights`` (same shape) accumulates e.g. rainfall or runoff
    instead of cell counts. Computed by a level-by-level topological sweep down the flow
    paths, linear in time and memory, so it scales to large DEMs. (An earlier version solved
    ``(I - A) acc = w`` with a general sparse LU, which ran out of memory at 17 million
    cells.)"""
    zf = dem if filled else fill_depressions(dem)
    fd = np.asarray(to_cpu(flow_direction(zf, res=res).data if isinstance(zf, xr.DataArray)
                           else flow_direction(zf, res=res)))
    z = np.asarray(to_cpu(zf.data if isinstance(zf, xr.DataArray) else zf), dtype="float64")
    rows, cols = z.shape
    w = np.where(np.isnan(z), 0.0, 1.0).ravel() if weights is None else \
        np.nan_to_num(np.asarray(getattr(weights, "values", weights), dtype="float64")).ravel()
    acc = _accumulate(_receivers(fd), w).reshape(rows, cols).astype("float32")
    acc[np.isnan(z)] = 0
    return like2d(dem, acc, "flow_accumulation") if isinstance(dem, xr.DataArray) else acc


def hand(dem: Any, *, threshold: float = 1000, res: Any = None, filled: bool = False):
    """Height Above Nearest Drainage (Rennó et al. 2008): elevation above the stream cell each
    cell drains to along D8 flow paths. Streams are cells with ``threshold`` or more upslope
    cells. Low HAND = flood-prone; see :func:`inundation`."""
    zf = dem if filled else fill_depressions(dem)
    fd = np.asarray(to_cpu(flow_direction(zf, res=res).data if isinstance(zf, xr.DataArray)
                           else flow_direction(zf, res=res)))
    acc = np.asarray(getattr(flow_accumulation(zf, res=res, filled=True), "values",
                             flow_accumulation(zf, res=res, filled=True)))
    z = np.asarray(to_cpu(zf.data if isinstance(zf, xr.DataArray) else zf), dtype="float64").ravel()
    stream = (acc.ravel() >= threshold)
    rec = _receivers(fd)
    idx = np.arange(z.size)
    tgt = np.where(stream | (rec < 0), idx, rec)
    for _ in range(64):                       # pointer jumping: ⌈log2(path length)⌉ rounds
        nxt = tgt[tgt]
        if np.array_equal(nxt, tgt):
            break
        tgt = nxt
    out = np.where(stream[tgt], z - z[tgt], np.nan).reshape(acc.shape).astype("float32")
    out[np.isnan(z.reshape(acc.shape))] = np.nan
    return like2d(dem, out, "hand") if isinstance(dem, xr.DataArray) else out


def inundation(surface: Any, level: float, *, seeds: Any = None, connected: bool = True):
    """Flood extent and depth for a water ``level``.

    With a HAND raster ``level`` is the stage above the channel (m); with a DEM it is an
    absolute water elevation (bathtub). ``connected`` keeps only flooded cells 8-connected
    to ``seeds`` (list of (x, y) map coordinates; default: the lowest cell). Returns an
    ``xr.Dataset`` with ``depth`` (m, NaN where dry) and ``extent`` (0/1), or a tuple of
    arrays for NumPy input.
    """
    from scipy import ndimage as snd

    a = np.asarray(to_cpu(getattr(surface, "values", surface)), dtype="float64")
    wet = np.isfinite(a) & (a < level)
    if connected and wet.any():
        lab, _ = snd.label(wet, structure=np.ones((3, 3)))
        if seeds is None:
            r, c = np.unravel_index(np.nanargmin(np.where(wet, a, np.nan)), a.shape)
            keep = {lab[r, c]}
        else:
            from ..core._utils import map_to_pix

            xs, ys = np.asarray(seeds, float).T
            cc, rr = map_to_pix(surface.rio.transform(), xs, ys)
            keep = {lab[int(r), int(c)] for r, c in zip(rr, cc)
                    if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]} - {0}
        wet &= np.isin(lab, list(keep))
    depth = np.where(wet, level - a, np.nan).astype("float32")
    if isinstance(surface, xr.DataArray):
        return xr.Dataset({"depth": like2d(surface, depth, "depth"),
                           "extent": like2d(surface, wet.astype("uint8"), "extent")})
    return depth, wet


def flatten_water(dem: xr.DataArray, water, *, percentile: float = 5.0,
                  ring_cells: int = 2) -> xr.DataArray:
    """Hydro-flatten water bodies: set each polygon (GeoDataFrame or shapely geometries) to
    a single elevation, the ``percentile`` of the DEM on a ``ring_cells``-wide shoreline
    ring. Removes the noisy water returns that break hydrology on lidar DEMs."""
    from rasterio.features import geometry_mask
    from scipy import ndimage as snd

    geoms = list(water.to_crs(dem.rio.crs).geometry) if hasattr(water, "geometry") else \
        list(water if isinstance(water, (list, tuple)) else [water])
    a = np.array(np.asarray(dem.values), dtype="float64")
    T = dem.rio.transform()
    for g in geoms:
        inside = ~geometry_mask([g], out_shape=a.shape, transform=T)
        if not inside.any():
            continue
        ring = snd.binary_dilation(inside, iterations=ring_cells) & ~inside & np.isfinite(a)
        if ring.any():
            a[inside] = np.percentile(a[ring], percentile)
    return dem.copy(data=a.astype(dem.dtype if dem.dtype.kind == "f" else "float32"))


def twi(dem: Any, *, res: Any = None, min_slope_deg: float = 0.1):
    """Topographic Wetness Index ln(a / tan β), with a = specific catchment area (m)."""
    dx, dy, _ = _res(dem, res)
    acc = flow_accumulation(dem, res=res)
    s = slope(dem, units="radians", res=res)
    a = np.asarray(to_cpu(getattr(acc, "data", acc))) * dx * dy / ((dx + dy) / 2)
    b = np.maximum(np.asarray(to_cpu(getattr(s, "data", s))), np.radians(min_slope_deg))
    out = np.log(a / np.tan(b)).astype("float32")
    return like2d(dem, out, "twi") if isinstance(dem, xr.DataArray) else out


# ----------------------------------------------------------------------------- geomorphons
GEOMORPHON_CLASSES = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur", 6: "slope",
                      7: "hollow", 8: "footslope", 9: "valley", 10: "pit"}
_FL, _PK, _RI, _SH, _SP, _SL, _HL, _FS, _VL, _PT = range(1, 11)
_LOOKUP = np.zeros((9, 9), dtype="int8")
_TABLE = [  # rows: number of lower directions (-), cols: number of higher directions (+)
    [_FL, _FL, _FL, _FS, _FS, _VL, _VL, _VL, _PT],
    [_FL, _FL, _FS, _FS, _FS, _VL, _VL, _VL, 0],
    [_FL, _SH, _SL, _SL, _HL, _HL, _VL, 0, 0],
    [_SH, _SH, _SL, _SL, _SL, _HL, 0, 0, 0],
    [_SH, _SH, _SP, _SL, _SL, 0, 0, 0, 0],
    [_RI, _RI, _SP, _SP, 0, 0, 0, 0, 0],
    [_RI, _RI, _RI, 0, 0, 0, 0, 0, 0],
    [_RI, _RI, 0, 0, 0, 0, 0, 0, 0],
    [_PK, 0, 0, 0, 0, 0, 0, 0, 0],
]
_LOOKUP[:] = np.array(_TABLE, dtype="int8")


def geomorphons(dem: Any, *, search: int = 10, flat_deg: float = 1.0, res: Any = None):
    """Geomorphon landforms (Jasiewicz & Stepinski 2013), classes 1-10 (see GEOMORPHON_CLASSES).

    ``search`` is the line-of-sight radius in cells; ``flat_deg`` the flatness threshold.
    """
    dx, dy, _ = _res(dem, res)
    t = np.radians(flat_deg)

    def f(z, xp):
        z = as_float(z, xp)
        rows, cols = z.shape
        p = xp.pad(z, search, constant_values=xp.nan)
        plus = xp.zeros(z.shape, dtype=xp.int8)
        minus = xp.zeros(z.shape, dtype=xp.int8)
        for dr, dc in _D8:
            step = float(np.hypot(dr * dy, dc * dx))
            hi = xp.full(z.shape, -np.inf, dtype=z.dtype)
            lo = xp.full(z.shape, np.inf, dtype=z.dtype)
            for k in range(1, search + 1):
                r0, c0 = search + dr * k, search + dc * k
                nb = p[r0:r0 + rows, c0:c0 + cols]
                ang = xp.arctan((nb - z) / (k * step))
                hi = xp.fmax(hi, ang)
                lo = xp.fmin(lo, ang)
            up = (hi > t) & (hi > -lo)
            down = (lo < -t) & (-lo > hi)
            plus += up.astype(xp.int8)
            minus += down.astype(xp.int8)
        lut = xp.asarray(_LOOKUP)
        out = lut[minus.astype(xp.int64), plus.astype(xp.int64)]
        return xp.where(xp.isnan(z), 0, out).astype(xp.int8)

    return _name(run_array(dem, f), "geomorphon", "class")


def _name(obj: Any, name: str, units: str):
    if isinstance(obj, xr.DataArray):
        obj = obj.rename(name)
        obj.attrs = {"units": units} if units else {}
    return obj
