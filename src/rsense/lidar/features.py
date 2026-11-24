"""Point-level features: eigenvalue geometry, normals, clustering, vertical structure and
plot metrics.

Everything here runs on NumPy/SciPy (``cKDTree``). Point clouds on the GPU are moved to the
host first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..core.pointcloud import PointCloud

EIGEN_FEATURES = ("linearity", "planarity", "scattering", "omnivariance", "anisotropy",
                  "eigenentropy", "curvature", "verticality", "nx", "ny", "nz")


def _xyz(pc: PointCloud) -> np.ndarray:
    p = pc.to_cpu()
    return np.column_stack([np.asarray(p.x, float), np.asarray(p.y, float),
                            np.asarray(p.z, float)])


def _tree(xyz: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(xyz)


def _eigen_block(xyz: np.ndarray, idx: np.ndarray):
    """Sorted eigenvalues (descending) and eigenvectors of each neighbourhood covariance."""
    nb = xyz[idx]                                   # (n, k, 3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / idx.shape[1]
    w, v = np.linalg.eigh(cov)                      # ascending
    return np.clip(w[:, ::-1], 0, None), v


def eigen_features(pc: PointCloud, k: int = 16, *, features: Sequence[str] = EIGEN_FEATURES,
                   chunk: int = 200_000, prefix: str = "") -> PointCloud:
    """Covariance (eigenvalue) features of each point's ``k`` nearest neighbours.

    With eigenvalues λ1 ≥ λ2 ≥ λ3: ``linearity`` (λ1−λ2)/λ1, ``planarity`` (λ2−λ3)/λ1,
    ``scattering`` λ3/λ1, ``omnivariance``, ``anisotropy``, ``eigenentropy``,
    ``curvature`` (change of curvature λ3/Σλ), ``verticality`` 1−|nz| and the upward unit
    normal ``nx, ny, nz``. Used for point classification, building and wire detection.
    """
    xyz = _xyz(pc)
    n = len(xyz)
    k = int(min(k, n))
    tree = _tree(xyz)
    out = {f: np.empty(n, "float32") for f in features}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, idx = tree.query(xyz[s:e], k=k)
        idx = np.atleast_2d(idx)
        lam, vec = _eigen_block(xyz, idx)
        l1 = np.maximum(lam[:, 0], 1e-12)
        tot = np.maximum(lam.sum(1), 1e-12)
        ev = lam / tot[:, None]
        normal = vec[:, :, 0]
        normal = normal * np.where(normal[:, 2:3] < 0, -1, 1)
        calc = {
            "linearity": (lam[:, 0] - lam[:, 1]) / l1,
            "planarity": (lam[:, 1] - lam[:, 2]) / l1,
            "scattering": lam[:, 2] / l1,
            "omnivariance": np.cbrt(np.prod(ev, axis=1)),
            "anisotropy": (lam[:, 0] - lam[:, 2]) / l1,
            "eigenentropy": -np.sum(ev * np.log(np.maximum(ev, 1e-12)), axis=1),
            "curvature": lam[:, 2] / tot,
            "verticality": 1 - np.abs(normal[:, 2]),
            "nx": normal[:, 0], "ny": normal[:, 1], "nz": normal[:, 2],
        }
        for f in features:
            out[f][s:e] = calc[f]
    res = pc
    for f in features:
        res = res.with_field(prefix + f, out[f] if pc.xp is np else pc.xp.asarray(out[f]))
    return res


def normals(pc: PointCloud, k: int = 16) -> PointCloud:
    """Upward-oriented unit surface normals (``nx, ny, nz``) from local PCA."""
    return eigen_features(pc, k, features=("nx", "ny", "nz"))


def euclidean_clusters(pc_or_xyz: PointCloud | np.ndarray, radius: float = 1.0, *,
                       min_points: int = 10, dims: int = 3) -> np.ndarray:
    """Connected-component clustering: points closer than ``radius`` share a cluster.

    ``dims=2`` clusters on x/y only. Returns integer labels (0, 1, ... by decreasing size);
    clusters smaller than ``min_points`` get ``-1``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    xyz = _xyz(pc_or_xyz) if isinstance(pc_or_xyz, PointCloud) else np.asarray(pc_or_xyz, float)
    xyz = xyz[:, :dims]
    n = len(xyz)
    if n == 0:
        return np.zeros(0, int)
    pairs = _tree(xyz).query_pairs(radius, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    out = rank[lab]
    out[sizes[lab] < min_points] = -1
    keep = np.unique(out[out >= 0])
    remap = np.full(out.max() + 2 if len(out) else 1, -1)
    remap[keep] = np.arange(len(keep))
    return np.where(out >= 0, remap[np.maximum(out, 0)], -1)


# ----------------------------------------------------------------------------- vertical structure
def _heights(pc: PointCloud) -> np.ndarray:
    if "height" not in pc:
        raise KeyError("PointCloud needs a 'height' field; call normalize_heights(pc, dtm)")
    return np.asarray(pc.to_cpu()["height"], float)


def height_profile(pc: PointCloud, bin_size: float = 1.0, *, max_height: float | None = None,
                   k: float = 0.5, ground_height: float = 0.5) -> pd.DataFrame:
    """Vertical profile of returns with gap fraction and leaf area density (MacArthur–Horn).

    Columns: ``z_bottom, z_top, count, fraction, gap_fraction`` (share of returns below the
    layer top) and ``lad`` (m²/m³, extinction coefficient ``k``). Returns below
    ``ground_height`` count as ground (they passed through every layer).
    """
    h = _heights(pc)
    top = max_height or max(float(np.nanmax(h)), bin_size)
    edges = np.arange(0, top + bin_size, bin_size)
    counts, _ = np.histogram(np.clip(h, 0, None), bins=edges)
    n = max(len(h), 1)
    below = np.array([(h < e).sum() for e in edges]) / n          # gap fraction at each edge
    below[0] = (h < ground_height).sum() / n
    with np.errstate(divide="ignore", invalid="ignore"):
        lad = np.log(np.maximum(below[1:], 1e-9) / np.maximum(below[:-1], 1e-9)) / (k * bin_size)
    lad = np.where(np.isfinite(lad), np.clip(lad, 0, None), 0.0)
    return pd.DataFrame({"z_bottom": edges[:-1], "z_top": edges[1:], "count": counts,
                         "fraction": counts / n, "gap_fraction": below[1:], "lad": lad})


def _fhd_counts(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, counts / np.maximum(tot, 1), 0)
        return -np.sum(np.where(p > 0, p * np.log(p), 0), axis=-1)


def foliage_height_diversity(pc: PointCloud, bin_size: float = 1.0, *,
                             min_height: float = 0.5) -> float:
    """Shannon diversity of returns across height layers above ``min_height`` (MacArthur 1961)."""
    h = _heights(pc)
    h = h[h >= min_height]
    if h.size == 0:
        return 0.0
    edges = np.arange(min_height, h.max() + bin_size, bin_size)
    counts, _ = np.histogram(h, bins=edges if len(edges) > 1 else 1)
    return float(_fhd_counts(counts.astype(float)))


def _stats(h: np.ndarray, first: np.ndarray, metrics: Sequence[str], *, area: float | None,
           cover_height: float, min_height: float, bin_size: float, k: float) -> dict[str, float]:
    out: dict[str, float] = {}
    veg = h[h > min_height]
    for m in metrics:
        if m == "density":
            out[m] = len(h) / area if area else np.nan
        elif m == "cover":
            out[m] = float(np.mean(h[first] > cover_height)) if first.any() else np.nan
        elif m == "lai":
            gap = np.mean(h <= cover_height) if len(h) else np.nan
            out[m] = float(-np.log(max(gap, 1e-3)) / k) if len(h) else np.nan
        elif m == "fhd":
            if veg.size:
                edges = np.arange(min_height, veg.max() + bin_size, bin_size)
                c, _ = np.histogram(veg, bins=edges if len(edges) > 1 else 1)
                out[m] = float(_fhd_counts(c.astype(float)))
            else:
                out[m] = np.nan
        elif veg.size == 0:
            out[m] = np.nan
        elif m == "hmax":
            out[m] = float(veg.max())
        elif m == "hmean":
            out[m] = float(veg.mean())
        elif m == "hstd":
            out[m] = float(veg.std())
        elif m == "cv":
            out[m] = float(veg.std() / veg.mean()) if veg.mean() > 0 else np.nan
        elif m == "crr":
            rng = veg.max() - veg.min()
            out[m] = float((veg.mean() - veg.min()) / rng) if rng > 0 else np.nan
        elif m.startswith("p") and m[1:].isdigit():
            out[m] = float(np.percentile(veg, int(m[1:])))
        else:
            raise ValueError(f"Unknown metric {m!r}")
    return out


PLOT_METRICS = ("hmax", "hmean", "hstd", "p25", "p50", "p75", "p95", "cover", "density",
                "fhd", "lai", "crr")


def plot_metrics(pc: PointCloud, plots, *, radius: float | None = None,
                 metrics: Sequence[str] = PLOT_METRICS, cover_height: float = 2.0,
                 min_height: float = 0.0, bin_size: float = 1.0, k: float = 0.5):
    """Structure metrics for each field plot (the area-based approach for biomass models).

    ``plots`` is a GeoDataFrame of polygons, or of points with ``radius`` (m) for circular
    plots. Needs a ``height`` field. Returns the plots with one column per metric and
    ``n_points``; feed it to a regression model with the plot biomass as the target.
    """
    import shapely

    h_all = _heights(pc)
    p = pc.to_cpu()
    x, y = np.asarray(p.x, float), np.asarray(p.y, float)
    first_all = (np.asarray(p["return_number"]) == 1) if "return_number" in p \
        else np.ones(len(p), bool)
    g = plots.to_crs(pc.crs) if (pc.crs is not None and plots.crs is not None) else plots
    geoms = g.geometry
    if radius is not None:
        geoms = geoms.buffer(radius)
    rows = []
    for geom in geoms:
        minx, miny, maxx, maxy = geom.bounds
        sel = np.flatnonzero((x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy))
        inside = sel[shapely.contains_xy(geom, x[sel], y[sel])]
        r = _stats(h_all[inside], first_all[inside], metrics, area=geom.area,
                   cover_height=cover_height, min_height=min_height, bin_size=bin_size, k=k)
        r["n_points"] = len(inside)
        rows.append(r)
    out = plots.copy()
    for col, vals in pd.DataFrame(rows, index=plots.index).items():
        out[col] = vals
    return out


def structure_metrics(pc: PointCloud, res: float = 20.0, *, bin_size: float = 1.0,
                      min_height: float = 0.5, cover_height: float = 2.0,
                      k: float = 0.5) -> xr.Dataset:
    """Gridded vertical-structure metrics: foliage height diversity (``fhd``), effective
    ``lai`` (Beer–Lambert from the gap fraction below ``cover_height``) and canopy relief
    ratio (``crr``). Complements :func:`~geolattice.rsense.lidar.grid_metrics`."""
    from .surfaces import _grid_spec, _to_dataarray

    h = _heights(pc)
    p = pc.to_cpu()
    x0, y1, H, W = _grid_spec(p, res)
    col = np.clip(np.floor((np.asarray(p.x) - x0) / res).astype(np.int64), 0, W - 1)
    row = np.clip(np.floor((y1 - np.asarray(p.y)) / res).astype(np.int64), 0, H - 1)
    cell = row * W + col
    df = pd.DataFrame({"cell": cell, "h": h})
    n = df.groupby("cell").size()
    gap = df.assign(b=df.h <= cover_height).groupby("cell")["b"].mean()
    veg = df[df.h > min_height]
    layer = np.floor((veg.h - min_height) / bin_size).astype(int)
    tab = pd.crosstab(veg.cell, layer)
    fhd = pd.Series(_fhd_counts(tab.to_numpy(float)), index=tab.index)
    agg = veg.groupby("cell")["h"].agg(["mean", "min", "max"])
    rng = (agg["max"] - agg["min"]).replace(0, np.nan)
    crr = (agg["mean"] - agg["min"]) / rng

    def grid(s: pd.Series, fill=np.nan):
        a = np.full(H * W, np.nan, "float32")
        a[n.index.values] = fill
        a[s.index.values] = s.values
        return a.reshape(H, W)

    lai = -np.log(gap.clip(lower=1e-3)) / k
    return xr.Dataset({"fhd": _to_dataarray(grid(fhd, 0.0), x0, y1, res, pc.crs, "fhd"),
                       "lai": _to_dataarray(grid(lai), x0, y1, res, pc.crs, "lai"),
                       "crr": _to_dataarray(grid(crr), x0, y1, res, pc.crs, "crr")})


# ----------------------------------------------------------------------------- ML on points
POINT_ATTRS = ("z", "height", "intensity", "return_number", "number_of_returns")


def point_feature_table(pc: PointCloud, *, k: int = 16, scales: Sequence[int] | None = None,
                        attrs: Sequence[str] = POINT_ATTRS,
                        label: str | None = None) -> pd.DataFrame:
    """Per-point feature table for supervised classification.

    Eigen features at one or several neighbourhood sizes (``scales``, e.g. ``(10, 30)``,
    prefixed ``k10_``…) plus the point attributes in ``attrs`` that exist. ``label`` adds
    that field (e.g. ``"classification"``) as the last column.
    """
    scales = list(scales) if scales else [k]
    cols: dict[str, Any] = {}
    p = pc.to_cpu()
    for a in attrs:
        if a in p:
            cols[a] = np.asarray(p[a], float)
    for s in scales:
        pre = f"k{s}_" if len(scales) > 1 else ""
        f = eigen_features(p, s, prefix=pre, features=EIGEN_FEATURES[:8] + ("nz",))
        for name in (*EIGEN_FEATURES[:8], "nz"):
            cols[pre + name] = np.asarray(f[pre + name], float)
    df = pd.DataFrame(cols)
    if label is not None:
        df[label] = np.asarray(p[label])
    return df


def train_point_classifier(pc: PointCloud, label: str = "classification", *, model: str = "rf",
                           k: int = 16, scales: Sequence[int] | None = None,
                           sample: int | None = 200_000, test_size: float = 0.25, seed: int = 0,
                           **model_kw: Any) -> dict:
    """Fit a point classifier (``rf``, ``xgb``, ``gbm``...) on eigen + attribute features.

    Returns ``{"model", "features", "report", "confusion"}``; the report holds overall
    accuracy, kappa and per-class F1 on a random holdout of ``test_size``.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    from ..classify import Classifier

    df = point_feature_table(pc, k=k, scales=scales, label=label)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    feats = [c for c in df.columns if c != label]
    X, y = df[feats], df[label].to_numpy()
    strat = y if np.min(np.unique(y, return_counts=True)[1]) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed,
                                          stratify=strat)
    if model == "rf":
        model_kw.setdefault("random_state", seed)
    clf = Classifier(model, **model_kw).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels = np.unique(np.concatenate([yte, pred]))
    report = {"accuracy": float(accuracy_score(yte, pred)),
              "kappa": float(cohen_kappa_score(yte, pred)),
              "f1": dict(zip(labels.tolist(), f1_score(yte, pred, labels=labels,
                                                       average=None).round(4).tolist()))}
    conf = pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels)
    return {"model": clf, "features": feats, "k": k, "scales": scales, "report": report,
            "confusion": conf}


def predict_point_classes(pc: PointCloud, fitted: dict, *, field: str = "classification",
                          chunk: int = 500_000) -> PointCloud:
    """Apply a model from :func:`train_point_classifier` and write ``field``."""
    df = point_feature_table(pc, k=fitted["k"], scales=fitted["scales"])
    pred = np.empty(len(df), dtype=np.asarray(fitted["model"].classes_).dtype
                    if fitted["model"].classes_ is not None else "int64")
    for s in range(0, len(df), chunk):
        pred[s:s + chunk] = fitted["model"].predict(df[fitted["features"]].iloc[s:s + chunk])
    return pc.with_field(field, pred.astype("uint8") if np.issubdtype(pred.dtype, np.integer)
                         else pred)
