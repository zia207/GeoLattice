"""Point cloud cleaning and preparation: outlier removal, PMF ground filter, thinning,
reprojection, polygon clipping and summaries."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.pointcloud import ASPRS_CLASSES, PointCloud
from .features import _tree, _xyz


def _apply(pc: PointCloud, bad: np.ndarray, action: str, cls: int) -> PointCloud:
    if action == "drop":
        return pc.filter(pc.xp.asarray(~bad) if pc.xp is not np else ~bad)
    if action == "classify":
        c = np.asarray(pc.to_cpu()["classification"]).copy() if "classification" in pc \
            else np.ones(len(pc), np.uint8)
        c[bad] = cls
        return pc.with_field("classification", c.astype(np.uint8) if pc.xp is np
                             else pc.xp.asarray(c.astype(np.uint8)))
    if action == "mask":
        return pc.with_field("outlier", bad)
    raise ValueError("action must be 'drop', 'classify' or 'mask'")


def statistical_outlier_removal(pc: PointCloud, k: int = 8, std_ratio: float = 2.0, *,
                                action: Literal["drop", "classify", "mask"] = "drop",
                                ) -> PointCloud:
    """Statistical outlier removal (Rusu et al. 2008), as in PCL / PDAL ``filters.outlier``.

    A point is an outlier when its mean distance to its ``k`` neighbours exceeds the global
    mean by ``std_ratio`` standard deviations. ``action``: ``drop`` the points, ``classify``
    them as noise (class 7 below, 18 above the local median) or add a boolean ``outlier``
    field (``mask``).
    """
    xyz = _xyz(pc)
    d, _ = _tree(xyz).query(xyz, k=min(k + 1, len(xyz)))
    md = d[:, 1:].mean(1)
    bad = md > md.mean() + std_ratio * md.std()
    if action == "classify":
        return _classify_noise(pc, bad, xyz)
    return _apply(pc, bad, action, 7)


def radius_outlier_removal(pc: PointCloud, radius: float = 1.0, min_neighbors: int = 2, *,
                           action: Literal["drop", "classify", "mask"] = "drop") -> PointCloud:
    """Remove points with fewer than ``min_neighbors`` other points within ``radius`` (m)."""
    xyz = _xyz(pc)
    tree = _tree(xyz)
    cnt = np.asarray(tree.query_ball_point(xyz, radius, return_length=True)) - 1
    bad = cnt < min_neighbors
    if action == "classify":
        return _classify_noise(pc, bad, xyz)
    return _apply(pc, bad, action, 7)


def _classify_noise(pc, bad, xyz, idx=None) -> PointCloud:
    """Noise above the surrounding clean surface → 18 (high noise), otherwise 7 (low noise).
    The reference is the median z of the 8 nearest non-outlier points in plan view."""
    c = np.asarray(pc.to_cpu()["classification"]).copy() if "classification" in pc \
        else np.ones(len(pc), np.uint8)
    good = np.flatnonzero(~bad)
    if bad.any() and good.size:
        _, j = _tree(xyz[good, :2]).query(xyz[bad, :2], k=min(8, good.size))
        med = np.median(xyz[good[np.atleast_2d(j).reshape(bad.sum(), -1)], 2], axis=1)
        hi = xyz[bad, 2] > med
        bi = np.flatnonzero(bad)
        c[bi[hi]] = 18
        c[bi[~hi]] = 7
    return pc.with_field("classification", c.astype(np.uint8))


def remove_outliers(pc: PointCloud, method: Literal["statistical", "radius"] = "statistical",
                    **kwargs: Any) -> PointCloud:
    """Dispatch to :func:`statistical_outlier_removal` or :func:`radius_outlier_removal`."""
    fn = {"statistical": statistical_outlier_removal, "radius": radius_outlier_removal}[method]
    return fn(pc, **kwargs)


def pmf(pc: PointCloud, *, cell: float = 1.0, max_window: float = 33.0, slope: float = 0.15,
        initial_distance: float = 0.15, max_distance: float = 2.5, exponential: bool = True,
        inplace: bool = False) -> PointCloud:
    """Progressive Morphological Filter (Zhang et al. 2003).

    Openings of a minimum-elevation grid with window sizes growing linearly or
    exponentially (``2·2^k + 1`` cells) up to ``max_window`` (m). A cell is non-ground when
    it rises more than ``dh_k = min(slope·(w_k − w_{k−1})·cell + initial_distance,
    max_distance)`` above the opened surface. Ground points are written as class 2.
    """
    from scipy import ndimage as snd

    from .ground import _bilinear, _fill_nearest, _grid_min, _set_class

    p = pc.to_cpu()
    x, y, z = np.asarray(p.x, float), np.asarray(p.y, float), np.asarray(p.z, float)
    zmin, row, col, origin = _grid_min(x, y, z, cell)
    surf = _fill_nearest(zmin)
    max_cells = max(3, int(max_window / cell))
    windows, k = [], 0
    while True:
        w = (2 * (2 ** k) + 1) if exponential else (2 * k + 3)
        if w > max_cells:
            break
        windows.append(w)
        k += 1
    nonground = np.zeros(surf.shape, bool)
    last, prev_w = surf, 1
    for w in windows:
        opened = snd.grey_dilation(snd.grey_erosion(last, size=(w, w)), size=(w, w))
        dh = min(slope * (w - prev_w) * cell + initial_distance, max_distance)
        nonground |= (last - opened) > dh
        last, prev_w = opened, w
    dtm = _fill_nearest(np.where(nonground | np.isnan(zmin), np.nan, zmin))
    fr = (origin[1] - y) / cell - 0.5
    fc = (x - origin[0]) / cell - 0.5
    ground = np.abs(z - _bilinear(dtm, fr, fc)) <= initial_distance + 0.5 * cell * slope + 0.15
    return _set_class(pc, ground, inplace)


def thin(pc: PointCloud, size: float = 1.0, *,
         method: Literal["voxel", "grid_min", "grid_max", "random"] = "voxel",
         fraction: float | None = None, seed: int = 0) -> PointCloud:
    """Reduce point density: one point per ``size`` voxel, the lowest (``grid_min``) or highest
    (``grid_max``) point per ``size`` grid cell, or a ``random`` ``fraction`` of points."""
    if method == "voxel":
        return pc.voxel_downsample(size)
    if method == "random":
        if fraction is None:
            raise ValueError("random thinning needs fraction=")
        rng = np.random.default_rng(seed)
        return pc.filter(np.sort(rng.choice(len(pc), int(len(pc) * fraction), replace=False)))
    p = pc.to_cpu()
    cx = np.floor((np.asarray(p.x) - np.min(p.x)) / size).astype(np.int64)
    cy = np.floor((np.asarray(p.y) - np.min(p.y)) / size).astype(np.int64)
    key = cx * (cy.max() + 1) + cy
    z = np.asarray(p.z)
    order = np.lexsort((z if method == "grid_min" else -z, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    return pc.filter(np.sort(order[first]))


def reproject(pc: PointCloud, dst_crs: Any, *, src_crs: Any = None) -> PointCloud:
    """Transform x/y (and z for compound CRSs with a vertical datum, e.g. ``"EPSG:6339+5703"``)
    to ``dst_crs`` with pyproj."""
    from pyproj import CRS, Transformer

    src = src_crs if src_crs is not None else pc.crs
    if src is None:
        raise ValueError("PointCloud has no CRS; pass src_crs=")
    dst = CRS.from_user_input(dst_crs)
    t = Transformer.from_crs(CRS.from_user_input(src), dst, always_xy=True)
    p = pc.to_cpu()
    x, y, z = t.transform(np.asarray(p.x, float), np.asarray(p.y, float), np.asarray(p.z, float))
    out = PointCloud(np.asarray(x), np.asarray(y), np.asarray(z), dict(p.attrs),
                     dst.to_epsg() or dst.to_wkt())
    return out.to_device("cuda") if pc.xp is not np else out


def clip_polygon(pc: PointCloud, geometry, *, invert: bool = False) -> PointCloud:
    """Keep points inside (or outside, ``invert=True``) a shapely geometry or GeoDataFrame
    (dissolved, reprojected to the cloud CRS)."""
    import shapely

    if hasattr(geometry, "geometry"):
        g = geometry.to_crs(pc.crs) if (pc.crs is not None and geometry.crs is not None) \
            else geometry
        geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.unary_union
    else:
        geom = geometry
    p = pc.to_cpu()
    inside = shapely.contains_xy(geom, np.asarray(p.x, float), np.asarray(p.y, float))
    return pc.filter(~inside if invert else inside)


def summary(pc: PointCloud) -> dict[str, Any]:
    """Quick inspection: point count, bounds, area, density, spacing, and tables of
    classification and return-number counts."""
    p = pc.to_cpu()
    minx, miny, maxx, maxy = p.bbox if len(p) else (np.nan,) * 4
    area = (maxx - minx) * (maxy - miny)
    out: dict[str, Any] = {
        "points": len(p), "crs": p.crs, "fields": p.fields,
        "bounds": {"minx": minx, "miny": miny, "minz": float(np.min(p.z)) if len(p) else np.nan,
                   "maxx": maxx, "maxy": maxy, "maxz": float(np.max(p.z)) if len(p) else np.nan},
        "area_m2": float(area),
        "density_pts_m2": float(len(p) / area) if area > 0 else np.nan,
        "spacing_m": float(np.sqrt(area / len(p))) if len(p) else np.nan,
    }
    if "classification" in p:
        v, c = np.unique(np.asarray(p["classification"]), return_counts=True)
        out["classes"] = pd.DataFrame({"class": v, "name": [ASPRS_CLASSES.get(int(i), "other")
                                                            for i in v],
                                       "count": c, "percent": (100 * c / c.sum()).round(2)})
    if "return_number" in p:
        v, c = np.unique(np.asarray(p["return_number"]), return_counts=True)
        out["returns"] = pd.DataFrame({"return_number": v, "count": c,
                                       "percent": (100 * c / c.sum()).round(2)})
    return out


def density_map(pc: PointCloud, res: float = 5.0, *, first_returns: bool = False):
    """Points per m² on a ``res`` grid (a QC layer for coverage gaps)."""
    from .surfaces import rasterize

    p = pc.first_returns() if first_returns else pc
    g = rasterize(p, res, value="z", stat="count", bounds=pc.bbox, name="density")
    return (g / res**2).rename("density").assign_attrs(units="pts/m2")


def vertical_accuracy(dtm, checkpoints, *, z_col: str = "z") -> dict[str, Any]:
    """Compare a DTM with surveyed checkpoints (GeoDataFrame with ``z_col``): RMSEz, mean
    error, NVA at 95% (1.96·RMSEz, ASPRS 2014) and the per-point residuals."""
    from ..core._utils import map_to_pix

    g = checkpoints.to_crs(dtm.rio.crs) if checkpoints.crs is not None else checkpoints
    cols, rows = map_to_pix(dtm.rio.transform(), g.geometry.x.values, g.geometry.y.values)
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (r >= 0) & (r < dtm.shape[-2]) & (c >= 0) & (c < dtm.shape[-1])
    est = np.full(len(g), np.nan)
    est[ok] = np.asarray(dtm.values)[r[ok], c[ok]]
    resid = est - g[z_col].to_numpy(float)
    v = resid[np.isfinite(resid)]
    rmse = float(np.sqrt(np.mean(v**2))) if v.size else np.nan
    out = g.copy()
    out["dtm_z"] = est
    out["residual"] = resid
    return {"rmse_z": rmse, "mean_error": float(v.mean()) if v.size else np.nan,
            "std": float(v.std()) if v.size else np.nan, "nva_95": 1.96 * rmse, "n": int(v.size),
            "points": out}


def swath_overlap(pc: PointCloud, res: float = 2.0, *, field: str = "point_source_id",
                  min_points: int = 2):
    """Inter-swath consistency: per cell, the elevation difference between flight lines
    (``point_source_id``) using the lowest point of each line. Large values reveal strip
    misalignment. Returns an ``xr.Dataset`` with ``dz`` (max − min across lines) and
    ``n_lines``."""
    import xarray as xr

    from .surfaces import _grid_spec, _to_dataarray

    if field not in pc:
        raise KeyError(f"PointCloud has no {field!r} field")
    p = pc.drop_noise().to_cpu()
    x0, y1, H, W = _grid_spec(p, res)
    col = np.clip(np.floor((np.asarray(p.x) - x0) / res).astype(np.int64), 0, W - 1)
    row = np.clip(np.floor((y1 - np.asarray(p.y)) / res).astype(np.int64), 0, H - 1)
    df = pd.DataFrame({"cell": row * W + col, "line": np.asarray(p[field]), "z": np.asarray(p.z)})
    per = df.groupby(["cell", "line"])["z"].agg(["min", "size"])
    per = per[per["size"] >= min_points]["min"].groupby(level=0)
    dz = (per.max() - per.min())
    nl = per.size()
    dz = dz[nl >= 2]
    a = np.full(H * W, np.nan, "float32")
    a[dz.index.values] = dz.values
    b = np.zeros(H * W, "float32")
    b[nl.index.values] = nl.values
    return xr.Dataset({"dz": _to_dataarray(a.reshape(H, W), x0, y1, res, pc.crs, "dz"),
                       "n_lines": _to_dataarray(b.reshape(H, W), x0, y1, res, pc.crs, "n_lines")})
