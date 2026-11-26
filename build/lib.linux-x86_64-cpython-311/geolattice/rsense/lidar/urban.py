"""Urban and infrastructure lidar: vegetation/building classification, footprints, roof planes,
power lines and rooftop solar potential."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.pointcloud import PointCloud
from .features import _xyz, eigen_features, euclidean_clusters

VEG_CLASSES = {3: (0.5, 2.0), 4: (2.0, 5.0), 5: (5.0, np.inf)}


def _height(pc: PointCloud, dtm=None) -> np.ndarray:
    if "height" in pc:
        return np.asarray(pc.to_cpu()["height"], float)
    if dtm is None:
        raise KeyError("Needs a 'height' field or dtm=; see normalize_heights")
    from .surfaces import normalize_heights

    return np.asarray(normalize_heights(pc, dtm)["height"], float)


def _classes(pc: PointCloud) -> np.ndarray:
    return np.asarray(pc.to_cpu()["classification"]).copy() if "classification" in pc \
        else np.ones(len(pc), np.uint8)


def classify_vegetation(pc: PointCloud, *, dtm=None, breaks: dict[int, tuple[float, float]]
                        | None = None, only_unclassified: bool = True) -> PointCloud:
    """ASPRS low (3), medium (4) and high (5) vegetation by height above ground.

    Defaults: 0.5–2 m, 2–5 m, > 5 m. Ground (2), buildings (6), noise and water stay as they
    are; with ``only_unclassified`` only classes 0/1 (and 3–5) are relabelled.
    """
    h = _height(pc, dtm)
    c = _classes(pc)
    free = np.isin(c, [0, 1, 3, 4, 5]) if only_unclassified else ~np.isin(c, [2, 7, 9, 18])
    for cls, (lo, hi) in (breaks or VEG_CLASSES).items():
        c[free & (h >= lo) & (h < hi)] = cls
    return pc.with_field("classification", c.astype(np.uint8))


def classify_buildings(pc: PointCloud, *, dtm=None, min_height: float = 2.5, k: int = 20,
                       max_scattering: float = 0.05, max_linearity: float = 0.8,
                       cluster_radius: float = 1.5, min_points: int = 50,
                       plane_distance: float = 0.25, plane_fraction: float = 0.8,
                       max_planes: int = 6, max_roof_slope: float = 60.0,
                       grow_distance: float = 1.0,
                       classify_veg: bool = True, seed: int = 0) -> PointCloud:
    """Separate buildings (6) from vegetation using local and object-level geometry.

    1. Points ``min_height`` above ground on locally flat surfaces (eigen ``scattering``
       ≤ ``max_scattering``, ``linearity`` ≤ ``max_linearity``) are roof candidates.
    2. Candidates are clustered; a cluster is a building when up to ``max_planes`` RANSAC
       planes no steeper than ``max_roof_slope`` explain at least ``plane_fraction`` of its
       points (tree crowns do not).
    3. Accepted roofs grow to above-ground points within ``grow_distance`` (roof edges,
       walls). Remaining above-ground points become vegetation (3/4/5) if ``classify_veg``.

    Ground must already be classified (class 2).
    """
    from scipy.spatial import cKDTree

    h = _height(pc, dtm)
    c = _classes(pc)
    xyz_all = _xyz(pc)
    above = (h >= min_height) & ~np.isin(c, [2, 7, 9, 18])
    idx = np.flatnonzero(above)
    roof = np.zeros(len(c), bool)
    if idx.size >= k:
        f = eigen_features(pc.filter(idx), k, features=("scattering", "linearity"))
        cand = idx[(np.asarray(f["scattering"]) <= max_scattering)
                   & (np.asarray(f["linearity"]) <= max_linearity)]
        if cand.size:
            lab = euclidean_clusters(xyz_all[cand], cluster_radius, min_points=min_points)
            for kk in np.unique(lab[lab >= 0]):
                members = cand[lab == kk]
                P = xyz_all[members]
                left = np.ones(len(P), bool)
                for j in range(max_planes):
                    if left.sum() < 3:
                        break
                    plane, inl = ransac_plane(P[left], distance=plane_distance, iterations=200,
                                              seed=seed + j)
                    if inl.sum() < max(3, 0.05 * len(P)):
                        break
                    if np.degrees(np.arccos(min(abs(plane[2]), 1.0))) > max_roof_slope:
                        break
                    left[np.flatnonzero(left)[inl]] = False
                if 1 - left.mean() >= plane_fraction:
                    roof[members] = True
    if roof.any() and grow_distance > 0:
        rest = np.flatnonzero(above & ~roof)
        if rest.size:
            d, _ = cKDTree(xyz_all[roof]).query(xyz_all[rest], k=1,
                                                distance_upper_bound=grow_distance)
            roof[rest[np.isfinite(d)]] = True
    c[roof] = 6
    out = pc.with_field("classification", c.astype(np.uint8))
    if classify_veg:
        out = classify_vegetation(out.with_field("height", h.astype("float32")))
    return out


def building_footprints(pc: PointCloud, res: float = 1.0, *, building_class: int = 6,
                        min_area: float = 20.0, simplify: float | None = 0.5, dtm=None,
                        close_iter: int = 1):
    """Building outlines from class-6 points → GeoDataFrame with ``area_m2``, ``height_m``
    (95th percentile of roof heights, when heights are available), ``z_max`` and
    ``n_points``."""
    import geopandas as gpd
    from rasterio.features import shapes
    from scipy import ndimage as snd
    from shapely.geometry import shape

    from .surfaces import rasterize

    b = pc.by_class(building_class)
    if len(b) == 0:
        return gpd.GeoDataFrame({"area_m2": []}, geometry=[], crs=pc.crs)
    cnt = rasterize(b, res, value="z", stat="count", bounds=pc.bbox)
    mask = np.asarray(cnt.values) > 0
    if close_iter:
        mask = snd.binary_closing(mask, iterations=close_iter)
        mask = snd.binary_fill_holes(mask)
    lab, n = snd.label(mask)
    T = cnt.rio.transform()
    geoms, ids = [], []
    for geom, val in shapes(lab.astype("int32"), mask=lab > 0, transform=T):
        geoms.append(shape(geom))
        ids.append(int(val))
    g = gpd.GeoDataFrame({"building_id": ids}, geometry=geoms, crs=pc.crs)
    g = g.dissolve("building_id", as_index=False)
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify)
    g["area_m2"] = g.geometry.area
    g = g[g.area_m2 >= min_area].reset_index(drop=True)
    p = b.to_cpu()
    from ..core._utils import map_to_pix

    cols, rows = map_to_pix(T, np.asarray(p.x), np.asarray(p.y))
    r = np.clip(rows.astype(int), 0, lab.shape[0] - 1)
    cc = np.clip(cols.astype(int), 0, lab.shape[1] - 1)
    pid = lab[r, cc]
    hh = np.asarray(p["height"], float) if "height" in p else \
        (_height(b, dtm) if dtm is not None else None)
    df = pd.DataFrame({"bid": pid, "z": np.asarray(p.z)})
    if hh is not None:
        df["h"] = hh
    grp = df[df.bid > 0].groupby("bid")
    stats = pd.DataFrame({"z_max": grp["z"].max(), "n_points": grp.size()})
    if hh is not None:
        stats["height_m"] = grp["h"].quantile(0.95)
    g = g.join(stats, on="building_id")
    return g


def ransac_plane(xyz: np.ndarray, *, distance: float = 0.15, iterations: int = 500,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit one plane by RANSAC (refined by least squares on the inliers).

    Returns ``(plane, inliers)`` with ``plane = (a, b, c, d)``, ``ax + by + cz + d = 0``,
    unit normal pointing up.
    """
    rng = np.random.default_rng(seed)
    n = len(xyz)
    if n < 3:
        return np.array([0, 0, 1, 0.0]), np.zeros(n, bool)
    o = xyz.mean(0)
    P = xyz - o
    best, best_cnt = None, -1
    for _ in range(iterations):
        s = P[rng.choice(n, 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm /= L
        d = -nrm @ s[0]
        inl = np.abs(P @ nrm + d) <= distance
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best, best_cnt = inl, cnt
    if best is None:
        return np.array([0, 0, 1, 0.0]), np.zeros(n, bool)
    Q = P[best]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    nrm = vt[-1] * (1 if vt[-1][2] >= 0 else -1)
    d = -nrm @ c
    inl = np.abs(P @ nrm + d) <= distance
    return np.r_[nrm, d - nrm @ o], inl


def segment_roof_planes(pc: PointCloud, *, distance: float = 0.15, min_points: int = 30,
                        max_planes: int = 20, iterations: int = 500, cluster_radius: float = 1.0,
                        seed: int = 0) -> tuple[PointCloud, pd.DataFrame]:
    """Sequential RANSAC roof-plane segmentation of (building) points.

    Planes are extracted one by one; each plane's inliers are split into spatially
    connected patches. Returns the cloud with a ``plane_id`` field (-1 = none) and a table of
    planes with ``slope_deg``, ``aspect_deg``, ``n_points`` and the plane ``normal``.
    """
    xyz = _xyz(pc)
    remaining = np.arange(len(xyz))
    pid = np.full(len(xyz), -1, np.int32)
    rows, next_id = [], 0
    for _ in range(max_planes):
        if remaining.size < min_points:
            break
        plane, inl = ransac_plane(xyz[remaining], distance=distance, iterations=iterations,
                                  seed=seed + next_id)
        if inl.sum() < min_points:
            break
        members = remaining[inl]
        lab = euclidean_clusters(xyz[members], cluster_radius, min_points=min_points)
        if (lab >= 0).sum() == 0:
            remaining = remaining[~inl]
            continue
        for k in np.unique(lab[lab >= 0]):
            m = members[lab == k]
            pid[m] = next_id
            a, b, c = plane[:3]
            rows.append({"plane_id": next_id, "n_points": len(m),
                         "slope_deg": float(np.degrees(np.arccos(np.clip(abs(c), 0, 1)))),
                         "aspect_deg": float(np.degrees(np.arctan2(a, b)) % 360),
                         "normal": (float(a), float(b), float(c)),
                         "z_mean": float(xyz[m, 2].mean())})
            next_id += 1
        remaining = remaining[~inl]
    return pc.with_field("plane_id", pid), pd.DataFrame(rows)


def detect_power_lines(pc: PointCloud, *, dtm=None, min_height: float = 5.0, k: int = 15,
                       linearity: float = 0.85, cluster_radius: float = 2.0,
                       min_points: int = 15, min_length: float = 15.0,
                       wire_class: int = 14) -> tuple[PointCloud, Any]:
    """Wire conductors: elevated, strongly linear, near-horizontal points.

    Candidates are points ``min_height`` above ground with eigen ``linearity`` above the
    threshold; they are clustered and each cluster longer than ``min_length`` (m) is kept,
    classified as ASPRS 14 and returned as a line (GeoDataFrame with ``length_m``,
    ``height_m``, ``n_points``).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    h = _height(pc, dtm)
    c = _classes(pc)
    idx = np.flatnonzero((h >= min_height) & ~np.isin(c, [2, 6, 7, 18]))
    lines, heights, counts, lengths = [], [], [], []
    if idx.size >= k:
        sub = pc.filter(idx)
        f = eigen_features(sub, k, features=("linearity", "verticality"))
        lin = np.asarray(f["linearity"]) >= linearity
        cand = idx[lin]
        if cand.size >= min_points:
            xyz = _xyz(pc.filter(cand))
            lab = euclidean_clusters(xyz, cluster_radius, min_points=min_points)
            for kk in np.unique(lab[lab >= 0]):
                m = lab == kk
                P = xyz[m]
                cen = P.mean(0)
                _, _, vt = np.linalg.svd(P[:, :2] - cen[:2], full_matrices=False)
                t = (P[:, :2] - cen[:2]) @ vt[0]
                L = float(t.max() - t.min())
                if L < min_length:
                    continue
                a = cen[:2] + t.min() * vt[0]
                b = cen[:2] + t.max() * vt[0]
                lines.append(LineString([a, b]))
                lengths.append(L)
                heights.append(float(np.median(h[cand[m]])))
                counts.append(int(m.sum()))
                c[cand[m]] = wire_class
    g = gpd.GeoDataFrame({"length_m": lengths, "height_m": heights, "n_points": counts},
                         geometry=lines, crs=pc.crs)
    return pc.with_field("classification", c.astype(np.uint8)), g


def sun_positions(latitude: float, *, days: Any = None, hours: Any = None):
    """Solar azimuth/elevation (degrees) for day-of-year × local solar hour combinations
    (Spencer 1971 declination). Defaults: the 15th of each month, 06–18 h hourly."""
    days = np.asarray(days if days is not None else [15 + 30.4 * m for m in range(12)], float)
    hours = np.asarray(hours if hours is not None else np.arange(6, 19), float)
    D, Hh = np.meshgrid(days, hours, indexing="ij")
    g = 2 * np.pi * (D - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians(15 * (Hh - 12))
    lat = np.radians(latitude)
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(dec) * np.cos(lat)))
    az = (az + 180) % 360
    m = el > 0
    return pd.DataFrame({"day": D[m], "hour": Hh[m], "azimuth": az[m], "elevation": el[m]})


def solar_potential(dsm, *, latitude: float | None = None, footprints=None, days: Any = None,
                    hours: Any = None, max_slope: float = 60.0, shading: bool = True,
                    shade_distance: float = 100.0) -> Any:
    """Relative clear-sky insolation on a DSM: mean of cos(incidence) over sun positions,
    optionally with cast shadows (line-of-sight along each sun azimuth).

    Values are 0–1 (1 = surface facing the sun all the time). With ``footprints``
    (building GeoDataFrame) returns the footprints with ``solar_mean`` and
    ``suitable_area_m2`` (roof cells > 0.6 of the maximum and slope < ``max_slope``).
    """
    from .terrain import illumination, slope

    if latitude is None:
        from pyproj import Transformer

        cx, cy = float(dsm.x.mean()), float(dsm.y.mean())
        latitude = Transformer.from_crs(dsm.rio.crs, 4326, always_xy=True).transform(cx, cy)[1]
    sp = sun_positions(latitude, days=days, hours=hours)
    a = np.asarray(dsm.values, float)
    res = abs(float(dsm.x[1] - dsm.x[0]))
    acc = np.zeros_like(a)
    for az, el in zip(sp.azimuth, sp.elevation):
        cos_i = np.clip(np.asarray(illumination(dsm, az, el).values, float), 0, None)
        if shading:
            cos_i = cos_i * ~_shadow(a, az, el, res, shade_distance)
        acc += cos_i
    sol = dsm.copy(data=(acc / max(len(sp), 1)).astype("float32")).rename("solar")
    if footprints is None:
        return sol
    import shapely

    sl = np.asarray(slope(dsm).values, float)
    X, Y = np.meshgrid(dsm.x.values, dsm.y.values)
    out = footprints.to_crs(dsm.rio.crs).copy()
    means, areas = [], []
    for geom in out.geometry:
        inside = shapely.contains_xy(geom, X, Y)
        v = sol.values[inside]
        means.append(float(np.nanmean(v)) if v.size else np.nan)
        good = inside & (sol.values > 0.6 * np.nanmax(sol.values)) & (sl < max_slope)
        areas.append(float(good.sum() * res * res))
    out["solar_mean"] = means
    out["suitable_area_m2"] = areas
    return out


def _shadow(z: np.ndarray, az: float, el: float, res: float, max_dist: float) -> np.ndarray:
    """Cells shaded by terrain/objects for a sun at ``az``/``el`` (horizon ray-march)."""
    dx, dy = np.sin(np.radians(az)), np.cos(np.radians(az))   # toward the sun (east, north)
    tan_el = np.tan(np.radians(el))
    zz = np.nan_to_num(z, nan=np.nanmin(z) if np.isfinite(z).any() else 0)
    shade = np.zeros(z.shape, bool)
    H, W = z.shape
    rows, cols = np.mgrid[0:H, 0:W]
    for step in np.arange(res, max_dist + res, res):
        rr = np.round(rows - dy * step / res).astype(int)
        cc = np.round(cols + dx * step / res).astype(int)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        h = np.full(z.shape, -np.inf)
        h[ok] = zz[rr[ok], cc[ok]]
        shade |= h > zz + step * tan_el
    return shade
