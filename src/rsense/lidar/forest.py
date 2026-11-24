"""Forest structure from lidar: grid metrics, individual tree detection and crown segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..core._utils import map_to_pix, pix_to_map
from ..core.pointcloud import PointCloud
from .surfaces import _grid_spec, _to_dataarray

DEFAULT_METRICS = ("hmax", "hmean", "hstd", "p25", "p50", "p75", "p95", "cover", "density")


def grid_metrics(pc: PointCloud, res: float = 20.0, *, metrics: Sequence[str] = DEFAULT_METRICS,
                 cover_height: float = 2.0, min_height: float = 0.0) -> xr.Dataset:
    """Area-based structure metrics per ``res`` cell from height-normalized points.

    Requires a ``height`` field (see :func:`normalize_heights`). Metrics: ``hmax``, ``hmean``,
    ``hstd``, ``pNN`` percentiles, ``cover`` (fraction of first returns above
    ``cover_height``), ``density`` (points/m²), ``cv`` (coefficient of variation).
    """
    if "height" not in pc:
        raise KeyError("PointCloud needs a 'height' field; call normalize_heights(pc, dtm)")
    import pandas as pd

    p = pc.to_cpu()
    x0, y1, H, W = _grid_spec(p, res)
    col = np.clip(np.floor((p.x - x0) / res).astype(np.int64), 0, W - 1)
    row = np.clip(np.floor((y1 - p.y) / res).astype(np.int64), 0, H - 1)
    cell = row * W + col
    h = np.asarray(p["height"], dtype="float64")
    first = np.asarray(p["return_number"] == 1) if "return_number" in p else np.ones(len(p), bool)
    df = pd.DataFrame({"cell": cell, "h": h, "first": first})
    veg = df[df.h > min_height].groupby("cell")["h"]
    out: dict[str, np.ndarray] = {}

    def put(series):
        a = np.full(H * W, np.nan, dtype="float32")
        a[series.index.values] = series.values
        return a

    for m in metrics:
        if m == "density":
            out[m] = put(df.groupby("cell").size() / res**2)
        elif m == "cover":
            fr = df[df["first"]]
            out[m] = put(fr.assign(c=fr.h > cover_height).groupby("cell")["c"].mean())
        elif m == "hmax":
            out[m] = put(veg.max())
        elif m == "hmean":
            out[m] = put(veg.mean())
        elif m == "hstd":
            out[m] = put(veg.std(ddof=0))
        elif m == "cv":
            mu = veg.mean()
            out[m] = put((veg.std(ddof=0) / mu).where(mu > 0))
        elif m.startswith("p") and m[1:].isdigit():
            out[m] = put(veg.quantile(int(m[1:]) / 100))
        else:
            raise ValueError(f"Unknown metric {m!r}")
    return xr.Dataset({m: _to_dataarray(v.reshape(H, W), x0, y1, res, pc.crs, m)
                       for m, v in out.items()})


def detect_trees(chm: xr.DataArray, *, min_height: float = 2.0, window: float | None = None,
                 min_distance_px: int | None = None, smooth_sigma: float = 0.5):
    """Treetops as local maxima of a CHM → GeoDataFrame (x, y, height).

    With ``window=None`` a variable window is used: the minimum separation grows with
    height (``3 + 0.08 * h`` m, a common conifer allometry); otherwise a fixed ``window`` (m).
    """
    import geopandas as gpd
    from scipy import ndimage as snd

    a = np.nan_to_num(np.asarray(to_cpu(chm).values, dtype="float64"))
    if smooth_sigma:
        a = snd.gaussian_filter(a, smooth_sigma)
    res = abs(float(chm.x[1] - chm.x[0]))
    if min_distance_px is not None:
        sizes = [2 * min_distance_px + 1]
    elif window is not None:
        sizes = [max(3, int(round(window / res)) | 1)]
    else:
        hmax = float(np.nanmax(a)) if a.size else 0
        sizes = sorted({max(3, int(round((3 + 0.08 * h) / res)) | 1)
                        for h in np.linspace(min_height, max(hmax, min_height), 6)})
    peaks = np.zeros(a.shape, bool)
    if len(sizes) == 1:
        peaks = (a == snd.maximum_filter(a, size=sizes[0])) & (a >= min_height)
    else:
        bins = np.linspace(min_height, max(float(a.max()), min_height) + 1e-6, len(sizes) + 1)
        for k, s in enumerate(sizes):
            in_bin = (a >= bins[k]) & (a < bins[k + 1])
            peaks |= in_bin & (a == snd.maximum_filter(a, size=s))
    # merge plateau duplicates: keep one per connected plateau
    lab, n = snd.label(peaks)
    if n == 0:
        return gpd.GeoDataFrame({"height": []}, geometry=[], crs=chm.rio.crs)
    cy, cx = np.array(snd.center_of_mass(peaks, lab, np.arange(1, n + 1))).T
    h = np.asarray(to_cpu(chm).values)[np.round(cy).astype(int), np.round(cx).astype(int)]
    T = chm.rio.transform()
    xs, ys = pix_to_map(T, cx + 0.5, cy + 0.5)
    return gpd.GeoDataFrame({"height": h.astype("float32")}, geometry=gpd.points_from_xy(xs, ys),
                            crs=chm.rio.crs)


def segment_crowns(chm: xr.DataArray, treetops=None, *, min_height: float = 2.0,
                   compactness: float = 0.0, as_polygons: bool = True, **detect_kwargs):
    """Marker-controlled watershed crown segmentation of a CHM.

    Returns ``(labels DataArray, crowns GeoDataFrame)`` with per-crown height, area and
    an equivalent crown diameter. ``treetops`` default to :func:`detect_trees`.
    """
    import geopandas as gpd
    from skimage.segmentation import watershed

    a = np.nan_to_num(np.asarray(to_cpu(chm).values, dtype="float64"))
    if treetops is None:
        treetops = detect_trees(chm, min_height=min_height, **detect_kwargs)
    markers = np.zeros(a.shape, np.int32)
    for i, (x, y) in enumerate(zip(treetops.geometry.x, treetops.geometry.y), start=1):
        c, r = map_to_pix(chm.rio.transform(), x, y)
        r, c = int(r), int(c)
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            markers[r, c] = i
    labels = watershed(-a, markers, mask=a >= min_height, compactness=compactness)
    lab_da = chm.copy(data=labels.astype("int32")).rename("crown_id")
    if not as_polygons:
        return lab_da, None
    from rasterio.features import shapes
    from shapely.geometry import shape

    res = abs(float(chm.x[1] - chm.x[0]))
    geoms, ids = [], []
    for geom, val in shapes(labels.astype("int32"), mask=labels > 0, transform=chm.rio.transform()):
        geoms.append(shape(geom))
        ids.append(int(val))
    crowns = gpd.GeoDataFrame({"crown_id": ids}, geometry=geoms, crs=chm.rio.crs)
    crowns = crowns.dissolve("crown_id", as_index=False)
    from scipy import ndimage as snd

    idx = crowns["crown_id"].to_numpy()
    crowns["height"] = snd.maximum(a, labels, idx)
    crowns["area_m2"] = snd.sum(np.ones_like(a), labels, idx) * res * res
    crowns["diameter_m"] = 2 * np.sqrt(crowns["area_m2"] / np.pi)
    return lab_da, crowns


def assign_points_to_crowns(pc: PointCloud, crown_labels: xr.DataArray) -> PointCloud:
    """Add ``tree_id`` to points by looking up the crown raster (0 = none)."""
    p = pc.to_cpu()
    cols, rows = map_to_pix(crown_labels.rio.transform(), p.x, p.y)
    r = np.clip(rows.astype(int), 0, crown_labels.shape[0] - 1)
    c = np.clip(cols.astype(int), 0, crown_labels.shape[1] - 1)
    return pc.with_field("tree_id", np.asarray(crown_labels.values)[r, c].astype("int32"))


def segment_trees_points(pc: PointCloud, treetops, *, min_height: float = 2.0,
                         crown_ratio: float = 0.3, min_radius: float = 1.0,
                         max_radius: float = 12.0) -> PointCloud:
    """Point-based individual tree segmentation from treetops (no CHM rasterisation).

    Each vegetation point above ``min_height`` (needs ``height``) joins the treetop that is
    closest relative to that tree's expected crown radius (``crown_ratio × tree height``,
    clipped to ``min_radius``–``max_radius``) and not lower than the point; points outside
    every crown get ``tree_id`` 0. ``tree_id`` matches the row order of ``treetops`` + 1.
    """
    from scipy.spatial import cKDTree

    if "height" not in pc:
        raise KeyError("PointCloud needs a 'height' field; call normalize_heights(pc, dtm)")
    p = pc.to_cpu()
    h = np.asarray(p["height"], float)
    tid = np.zeros(len(p), np.int32)
    if len(treetops) == 0:
        return pc.with_field("tree_id", tid)
    tops = treetops.to_crs(pc.crs) if (pc.crs is not None and treetops.crs is not None) \
        else treetops
    tx, ty = tops.geometry.x.to_numpy(), tops.geometry.y.to_numpy()
    th = tops["height"].to_numpy(float)
    rad = np.clip(crown_ratio * th, min_radius, max_radius)
    sel = np.flatnonzero(h > min_height)
    xy = np.column_stack([np.asarray(p.x)[sel], np.asarray(p.y)[sel]])
    k = min(8, len(tx))
    d, i = cKDTree(np.column_stack([tx, ty])).query(xy, k=k)
    d, i = d.reshape(len(sel), k), i.reshape(len(sel), k)
    score = d / rad[i]
    score[(score > 1) | (h[sel][:, None] > th[i] + 1.0)] = np.inf
    best = np.argmin(score, axis=1)
    ok = np.isfinite(score[np.arange(len(sel)), best])
    tid[sel[ok]] = i[np.arange(len(sel)), best][ok] + 1
    return pc.with_field("tree_id", tid)
