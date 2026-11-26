"""Registration and terrestrial/UAV lidar tools: ICP alignment, rigid transforms, stem
detection and DBH by circle fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.pointcloud import PointCloud
from .features import _tree, _xyz, euclidean_clusters


def rigid_transform(rotation_deg: tuple[float, float, float] = (0, 0, 0),
                    translation: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """4×4 matrix from rotations about x, y, z (degrees, applied x→y→z) and a translation."""
    rx, ry, rz = np.radians(rotation_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = translation
    return T


def apply_transform(pc: PointCloud, T: np.ndarray, *, origin: Any = None) -> PointCloud:
    """Apply a 4×4 rigid transform. ``origin`` (x, y, z) is subtracted before and added after,
    which keeps precision with large projected coordinates."""
    xyz = _xyz(pc)
    o = np.zeros(3) if origin is None else np.asarray(origin, float)
    q = (xyz - o) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3] + o
    p = pc.to_cpu()
    return PointCloud(q[:, 0], q[:, 1], q[:, 2], dict(p.attrs), p.crs)


def _best_fit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform mapping A onto B (Arun et al. 1987 / Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def icp(source: PointCloud, target: PointCloud, *, init: np.ndarray | None = None,
        max_iterations: int = 50, tolerance: float = 1e-6, max_distance: float | None = None,
        sample: int | None = 50_000, trim: float = 0.0, seed: int = 0) -> dict[str, Any]:
    """Point-to-point Iterative Closest Point (Besl & McKay 1992).

    Aligns ``source`` to ``target``. ``max_distance`` rejects far correspondences, ``trim``
    drops that fraction of the worst pairs each iteration (trimmed ICP, robust to partial
    overlap). Returns ``{"transform" (4×4, in world coordinates), "aligned" (PointCloud),
    "rmse", "iterations", "fitness" (share of matched points), "history"}``.
    """
    rng = np.random.default_rng(seed)
    S_all, Tt = _xyz(source), _xyz(target)
    o = Tt.mean(0)
    S_all, Tt = S_all - o, Tt - o
    S = S_all[rng.choice(len(S_all), sample, replace=False)] if sample and len(S_all) > sample \
        else S_all
    tree = _tree(Tt)
    T = np.eye(4)
    if init is not None:   # init is in world coordinates: shift it to the local origin
        T = _to_local(np.asarray(init, float), o)
    cur = S @ T[:3, :3].T + T[:3, 3]
    prev, hist, it, rmse, fit = np.inf, [], 0, np.nan, 0.0
    for it in range(1, max_iterations + 1):
        d, i = tree.query(cur, k=1)
        m = np.isfinite(d)
        if max_distance is not None:
            m &= d <= max_distance
        if trim > 0 and m.sum() > 10:
            m &= d <= np.quantile(d[m], 1 - trim)
        if m.sum() < 3:
            break
        step = _best_fit(cur[m], Tt[i[m]])
        cur = cur @ step[:3, :3].T + step[:3, 3]
        T = step @ T
        rmse = float(np.sqrt(np.mean(d[m] ** 2)))
        fit = float(m.mean())
        hist.append(rmse)
        if abs(prev - rmse) < tolerance:
            break
        prev = rmse
    d, _ = tree.query(cur, k=1)
    rmse = float(np.sqrt(np.mean(np.minimum(d, max_distance or np.inf) ** 2)))
    Tw = _to_world(T, o)
    return {"transform": Tw, "aligned": apply_transform(source, Tw), "rmse": rmse,
            "iterations": it, "fitness": fit, "history": hist}


def _to_local(T: np.ndarray, o: np.ndarray) -> np.ndarray:
    L = T.copy()
    L[:3, 3] = T[:3, :3] @ o + T[:3, 3] - o
    return L


def _to_world(T: np.ndarray, o: np.ndarray) -> np.ndarray:
    W = T.copy()
    W[:3, 3] = T[:3, 3] + o - T[:3, :3] @ o
    return W


def fit_circle(xy: np.ndarray, *, iterations: int = 0) -> tuple[float, float, float, float]:
    """Algebraic (Kåsa) least-squares circle fit, optionally refined by Gauss–Newton
    ``iterations``. Returns ``(cx, cy, radius, rmse)``."""
    xy = np.asarray(xy, float)
    o = xy.mean(0)
    x, y = (xy - o).T
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a1 / 2, a2 / 2
    r = np.sqrt(max(a3 + cx**2 + cy**2, 0))
    for _ in range(iterations):
        dx, dy = x - cx, y - cy
        di = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.column_stack([-dx / di, -dy / di, -np.ones_like(di)])
        res = di - r
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
    rmse = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx + o[0]), float(cy + o[1]), float(abs(r)), rmse


def detect_stems(pc: PointCloud, *, slice_height: tuple[float, float] = (1.0, 1.6),
                 cluster_radius: float = 0.15, min_points: int = 20,
                 dbh_range: tuple[float, float] = (0.05, 1.5), max_rmse: float = 0.05,
                 min_arc: float = 0.35):
    """Stem positions and diameter at breast height from TLS/MLS/dense UAV lidar.

    Points in the ``slice_height`` band above ground (needs a ``height`` field) are
    clustered in x/y; a circle is fitted to each cluster. Fits with a diameter outside
    ``dbh_range`` (m), an RMSE above ``max_rmse`` (m) or angular coverage below ``min_arc``
    (fraction of the full circle) are rejected. Returns a GeoDataFrame with ``dbh_cm``,
    ``radius_m``, ``rmse_m``, ``arc_coverage``, ``n_points``.
    """
    import geopandas as gpd

    if "height" not in pc:
        raise KeyError("PointCloud needs a 'height' field; call normalize_heights(pc, dtm)")
    h = np.asarray(pc.to_cpu()["height"], float)
    band = pc.filter(np.flatnonzero((h >= slice_height[0]) & (h <= slice_height[1])))
    xyz = _xyz(band)
    rows: list[dict] = []
    if len(xyz):
        lab = euclidean_clusters(xyz, cluster_radius, min_points=min_points, dims=2)
        for k in np.unique(lab[lab >= 0]):
            P = xyz[lab == k, :2]
            cx, cy, r, rmse = fit_circle(P, iterations=5)
            ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
            arc = len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 36))) / 36
            d = 2 * r
            if not (dbh_range[0] <= d <= dbh_range[1]) or rmse > max_rmse or arc < min_arc:
                continue
            rows.append({"x": cx, "y": cy, "dbh_cm": 100 * d, "radius_m": r, "rmse_m": rmse,
                         "arc_coverage": arc, "n_points": len(P)})
    import pandas as pd

    df = pd.DataFrame(rows, columns=["x", "y", "dbh_cm", "radius_m", "rmse_m", "arc_coverage",
                                     "n_points"])
    return gpd.GeoDataFrame(df.drop(columns=["x", "y"]),
                            geometry=gpd.points_from_xy(df.x, df.y), crs=pc.crs)
