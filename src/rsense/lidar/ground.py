"""Ground classification: SMRF (Pingel et al. 2013), CSF (Zhang et al. 2016) and PMF
(Zhang et al. 2003, in :mod:`.filters`).

Both are native NumPy implementations of the published algorithms (simplified where
noted); results are written to the ASPRS ``classification`` field (2 = ground).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as snd

from ..backend import to_cpu
from ..core.pointcloud import PointCloud


def _grid_min(x, y, z, cell, origin=None):
    x0, y0 = origin if origin is not None else (x.min(), y.max())
    col = np.floor((x - x0) / cell).astype(np.int64)
    row = np.floor((y0 - y) / cell).astype(np.int64)
    H, W = row.max() + 1, col.max() + 1
    g = np.full(H * W, np.inf)
    np.minimum.at(g, row * W + col, z)
    g = g.reshape(H, W)
    g[np.isinf(g)] = np.nan
    return g, row, col, (x0, y0)


def _fill_nearest(g: np.ndarray) -> np.ndarray:
    mask = np.isnan(g)
    if not mask.any():
        return g
    idx = snd.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return g[tuple(idx)]


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx**2 + yy**2) <= r * r


def _bilinear(g: np.ndarray, fr: np.ndarray, fc: np.ndarray) -> np.ndarray:
    return snd.map_coordinates(g, [fr, fc], order=1, mode="nearest")


def smrf(pc: PointCloud, *, cell: float = 1.0, slope: float = 0.15, window: float = 18.0,
         elevation_threshold: float = 0.5, elevation_scalar: float = 1.25,
         inplace: bool = False) -> PointCloud:
    """Simple Morphological Filter (Pingel et al. 2013).

    1. Minimum-elevation surface on a ``cell`` grid (gaps filled).
    2. Progressive morphological openings with disk radii up to ``window`` (m); cells
       whose height above the opening exceeds ``slope * radius * cell`` are non-ground.
    3. Provisional DTM interpolated from ground cells; points within
       ``elevation_threshold + elevation_scalar * local_slope * cell`` of it are ground.
    """
    p = pc.to_cpu()
    x, y, z = p.x, p.y, p.z
    zmin, row, col, origin = _grid_min(x, y, z, cell)
    surf = _fill_nearest(zmin)
    nonground = np.zeros(surf.shape, bool)
    last = surf.copy()
    for r in range(1, int(np.ceil(window / cell)) + 1):
        opened = snd.grey_opening(last, footprint=_disk(r))
        nonground |= (last - opened) > slope * r * cell
        last = opened
    dtm = np.where(nonground | np.isnan(zmin), np.nan, zmin)
    dtm = _fill_nearest(dtm)
    gy, gx = np.gradient(dtm, cell)
    sl = np.hypot(gx, gy)
    fr = (origin[1] - y) / cell - 0.5
    fc = (x - origin[0]) / cell - 0.5
    zdtm = _bilinear(dtm, fr, fc)
    thr = elevation_threshold + elevation_scalar * _bilinear(sl, fr, fc) * cell
    ground = np.abs(z - zdtm) <= thr
    return _set_class(pc, ground, inplace)


def csf(pc: PointCloud, *, resolution: float = 1.0, rigidness: int = 2,
        class_threshold: float = 0.5, iterations: int = 500, time_step: float = 0.65,
        slope_smooth: bool = True, inplace: bool = False) -> PointCloud:
    """Cloth Simulation Filter (Zhang et al. 2016), simplified.

    The cloud is inverted and a cloth of particles (spacing ``resolution``) is dropped
    onto it under gravity. Each particle stops at the highest inverted point below it;
    internal springs (``rigidness`` 1 = steep terrain, 3 = flat) pull particles toward
    their neighbours. Points within ``class_threshold`` of the settled cloth are ground.
    Uses the official ``CSF`` package instead if installed (``pip install cloth-simulation-filter``).
    """
    p = pc.to_cpu()
    try:
        import CSF as _csf  # official implementation, faster and exact

        c = _csf.CSF()
        c.params.bSloopSmooth = slope_smooth
        c.params.cloth_resolution = resolution
        c.params.rigidness = rigidness
        c.params.class_threshold = class_threshold
        c.params.interations = iterations
        c.params.time_step = time_step
        c.setPointCloud(np.column_stack([p.x, p.y, p.z]))
        g, ng = _csf.VecInt(), _csf.VecInt()
        c.do_filtering(g, ng, exportCloth=False)
        ground = np.zeros(len(p), bool)
        ground[np.asarray(g, dtype=int)] = True
        return _set_class(pc, ground, inplace)
    except ImportError:
        pass
    x, y, zi = p.x, p.y, -p.z
    # collision surface: highest inverted point per cell (= lowest original)
    zmin, row, col, origin = _grid_min(x, y, -zi, resolution)
    collide = -_fill_nearest(zmin)
    H, W = collide.shape
    cloth = np.full((H, W), collide.max() + 1.0)
    movable = np.ones((H, W), bool)
    disp = 0.2 * time_step**2 * resolution  # displacement per step under (scaled) gravity
    for _ in range(iterations):
        prev = cloth.copy()
        cloth = np.where(movable, cloth - disp, cloth)
        hit = cloth <= collide
        cloth = np.where(hit, collide, cloth)
        movable &= ~hit
        # spring constraints (rigidness repetitions of neighbour averaging on movable particles)
        for _ in range(rigidness):
            pad = np.pad(cloth, 1, mode="edge")
            nb = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
            cloth = np.where(movable, 0.5 * (cloth + nb), cloth)
        if np.max(np.abs(cloth - prev)) < 1e-3:
            break
    if slope_smooth:  # pull movable particles adjacent to fixed ones toward the collision surface
        cloth = np.where(movable & (cloth - collide < 0.3), collide, cloth)
    fr = (origin[1] - y) / resolution - 0.5
    fc = (x - origin[0]) / resolution - 0.5
    ground = np.abs(zi - _bilinear(cloth, fr, fc)) <= class_threshold
    return _set_class(pc, ground, inplace)


def _set_class(pc: PointCloud, ground: np.ndarray, inplace: bool) -> PointCloud:
    out = pc if inplace else pc.copy()
    xp = out.xp
    cls = out.attrs.get("classification")
    if cls is None:
        cls = xp.ones(len(out), dtype=xp.uint8)
    else:
        cls = cls.copy()
        cls = xp.where(cls == 2, 1, cls)  # reset previous ground
    g = xp.asarray(ground) if xp is not np else np.asarray(to_cpu(ground))
    out.attrs["classification"] = xp.where(g, 2, cls).astype(xp.uint8)
    return out


def classify_ground(pc: PointCloud, method: str = "smrf", **kwargs) -> PointCloud:
    """Dispatch to :func:`smrf`, :func:`csf` or :func:`~geolattice.rsense.lidar.filters.pmf`."""
    from .filters import pmf

    return {"smrf": smrf, "csf": csf, "pmf": pmf}[method](pc, **kwargs)
