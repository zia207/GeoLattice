"""Spatial statistics and clustering at scale, on CPU (NumPy/SciPy) or GPU (CuPy/cuML).

* :func:`weights` builds a sparse spatial weights matrix (k-nearest neighbours, distance
  band, queen or rook contiguity) using the partition-parallel joins in :mod:`.vector`.
* Global :func:`morans_i` and :func:`gearys_c`, local :func:`local_morans` (LISA) and
  :func:`getis_ord_local` (G*). Permutation inference is batched: every permutation is a
  column of one matrix, so 999 permutations are a few sparse matrix products. Results
  match ``esda`` (statistics exactly, pseudo p-values in distribution).
* :func:`dbscan`, :func:`hdbscan`, :func:`kmeans` dispatch to cuML on GPU, scikit-learn on CPU.
* :func:`kde` computes a kernel density surface chunk by chunk on a raster grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse, stats

from ._backend import Compute, resolve_compute
from ._registry import dispatch, register


# --------------------------------------------------------------------------- weights
@dataclass
class Weights:
    """Sparse spatial weights: ``sparse`` is an ``(n, n)`` SciPy CSR matrix (row i lists
    the neighbours of observation i), ``ids`` the matching index labels."""

    sparse: sparse.csr_matrix
    ids: pd.Index
    kind: str
    transform: Literal["b", "r"] = "b"

    @property
    def n(self) -> int:
        return self.sparse.shape[0]

    @property
    def cardinalities(self) -> np.ndarray:
        return np.diff(self.sparse.indptr)

    @property
    def islands(self) -> np.ndarray:
        """Positions of observations without neighbours."""
        return np.flatnonzero(self.cardinalities == 0)

    def row_standardize(self) -> Weights:
        rs = np.asarray(self.sparse.sum(axis=1)).ravel()
        inv = np.divide(1.0, rs, out=np.zeros_like(rs, dtype=float), where=rs != 0)
        return Weights(sparse.diags(inv) @ self.sparse, self.ids, self.kind, "r")

    def binary(self) -> Weights:
        b = self.sparse.copy()
        b.data[:] = 1.0
        return Weights(b, self.ids, self.kind, "b")

    def to_gpu(self) -> Any:
        import cupyx.scipy.sparse as cps

        return cps.csr_matrix(self.sparse.astype(np.float64))

    def to_libpysal(self) -> Any:
        from libpysal.weights import WSP

        return WSP(self.sparse, id_order=list(self.ids)).to_W()


def _coords(gdf: Any) -> np.ndarray:
    import shapely

    g = gdf.geometry.values
    t = shapely.get_type_id(np.asarray(g, dtype=object))
    if not np.all(t == 0):
        g = gdf.geometry.centroid.values
    return np.column_stack([shapely.get_x(g), shapely.get_y(g)])


def weights(gdf: Any, kind: Literal["knn", "distance", "queen", "rook"] = "knn", *, k: int = 8,
            threshold: float | None = None, binary: bool = True, alpha: float = -1.0,
            row_standardize: bool = True, npartitions: int | None = None,
            compute: Compute | None = None) -> Weights:
    """Spatial weights for the rows of ``gdf``.

    ``knn``: the ``k`` nearest neighbours (points, or polygon centroids); GPU-capable.
    ``distance``: all pairs within ``threshold``; binary or inverse distance ``d**alpha``.
    ``queen`` / ``rook``: polygons sharing a boundary point / a boundary segment, found
    with the partition-parallel :func:`~geolattice.distributed.vector.sjoin` (``touches``).
    """
    from . import vector

    n = len(gdf)
    if kind == "knn":
        nb = vector.knn(gdf.assign(geometry=gdf.geometry.centroid) if _needs_centroid(gdf) else gdf,
                        k=k, compute=compute)
        rows = gdf.index.get_indexer(nb.index)
        cols = gdf.index.get_indexer(nb["index_right"])
        vals = np.ones(len(rows))
    elif kind == "distance":
        if threshold is None:
            raise ValueError("kind='distance' needs threshold=")
        resolve_compute(compute, op="analysis.weights.distance", gpu_available=False)
        from scipy.spatial import cKDTree

        tree = cKDTree(_coords(gdf))
        d = tree.sparse_distance_matrix(tree, threshold, output_type="coo_matrix")
        keep = d.row != d.col
        rows, cols, dist = d.row[keep], d.col[keep], d.data[keep]
        vals = np.ones(len(rows)) if binary else np.power(np.maximum(dist, 1e-12), alpha)
    elif kind in ("queen", "rook"):
        import shapely

        left = gdf[[gdf.geometry.name]].reset_index(drop=True)
        pairs = vector.sjoin(left, left, predicate="touches", npartitions=npartitions,
                             compute="cpu")
        rows = pairs.index.to_numpy()
        cols = pairs["index_right"].to_numpy()
        if kind == "rook":
            geoms = left.geometry.values
            shared = shapely.length(shapely.intersection(geoms[rows], geoms[cols])) > 0
            rows, cols = rows[shared], cols[shared]
        vals = np.ones(len(rows))
    else:
        raise ValueError("kind must be 'knn', 'distance', 'queen' or 'rook'")
    w = Weights(sparse.csr_matrix((vals, (rows, cols)), shape=(n, n)), gdf.index, kind, "b")
    w.sparse.sum_duplicates()
    return w.row_standardize() if row_standardize else w


def _needs_centroid(gdf: Any) -> bool:
    import shapely

    t = shapely.get_type_id(np.asarray(gdf.geometry.values, dtype=object))
    return not np.all(t == 0)


# --------------------------------------------------------------------------- shared helpers
def _values(y: Any, w: Weights) -> np.ndarray:
    v = np.asarray(y, dtype=np.float64).ravel()
    if len(v) != w.n:
        raise ValueError(f"y has {len(v)} values but weights have {w.n} observations")
    if not np.all(np.isfinite(v)):
        raise ValueError("y contains NaN or infinite values")
    return v


def _xp_and_matrix(w: Weights, used: str):
    if used == "gpu":
        import cupy as cp

        return cp, w.to_gpu()
    return np, w.sparse


def _fold_p(sims: Any, obs: Any, permutations: int, xp: Any) -> np.ndarray:
    """esda's folded pseudo p-value: (min(#>=obs, #<obs) + 1) / (permutations + 1)."""
    larger = (sims >= obs).sum(axis=0)
    larger = xp.where(permutations - larger < larger, permutations - larger, larger)
    out = (larger + 1.0) / (permutations + 1.0)
    return out.get() if hasattr(out, "get") else np.asarray(out)


def _budget(xp: Any) -> float:
    """Elements per permutation batch: smaller on the GPU, whose memory is smaller and whose
    sort temporaries are large (n = 10^6 with 2e8 elements ran out of 16 GB of device memory)."""
    return 2e8 if xp is np else 2.5e7


def _batches(n: int, permutations: int, budget: float = 2e8) -> int:
    return max(1, min(permutations, int(budget // max(n, 1))))


def _perm_columns(z: Any, size: int, rng: Any, xp: Any) -> Any:
    """``(n, size)`` matrix whose columns are random permutations of ``z``."""
    keys = rng.random((size, len(z)))
    return z[xp.argsort(keys, axis=1)].T


# --------------------------------------------------------------------------- global statistics
@dataclass
class GlobalResult:
    """A global autocorrelation statistic with normal-approximation and permutation inference."""

    statistic: float
    expected: float
    variance_norm: float
    z_norm: float
    p_norm: float
    p_sim: float | None
    z_sim: float | None
    permutations: int
    sims: np.ndarray | None = field(default=None, repr=False)


def _morans_core(y: Any, w: Weights, permutations: int, seed: int | None, compute: Compute | None,
                 two_tailed: bool, keep_simulations: bool) -> GlobalResult:
    v = _values(y, w)
    n = w.n
    impl, used = dispatch("analysis.morans_i", compute)
    xp, W = _xp_and_matrix(w, used)
    z = xp.asarray(v - v.mean())
    zz = float((z * z).sum())
    s0 = float(w.sparse.sum())
    stat = n / s0 * float(z @ (W @ z)) / zz
    ei = -1.0 / (n - 1)
    ws = w.sparse
    s1 = float((ws + ws.T).power(2).sum()) / 2
    s2 = float(np.square(np.asarray(ws.sum(axis=1)).ravel() + np.asarray(ws.sum(axis=0)).ravel()).sum())
    vi = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n * n - 1) * s0 * s0) - ei * ei
    zn = (stat - ei) / np.sqrt(vi)
    pn = stats.norm.sf(abs(zn)) * (2 if two_tailed else 1)
    sims = impl(z, W, n, s0, zz, permutations, seed, xp) if permutations else None
    p_sim = z_sim = None
    if sims is not None:
        p_sim = float(_fold_p(sims[:, None], stat, permutations, np)[0])
        z_sim = float((stat - sims.mean()) / sims.std())
    return GlobalResult(stat, ei, vi, float(zn), float(pn), p_sim, z_sim, permutations,
                        sims if keep_simulations else None)


def _morans_perm(z: Any, W: Any, n: int, s0: float, zz: float, permutations: int,
                 seed: int | None, xp: Any) -> np.ndarray:
    rng = xp.random.default_rng(seed)
    out = []
    step = _batches(n, permutations, _budget(xp))
    for s in range(0, permutations, step):
        zp = _perm_columns(z, min(step, permutations - s), rng, xp)
        out.append((n / s0) * (zp * (W @ zp)).sum(axis=0) / zz)
    sims = xp.concatenate(out)
    return sims.get() if hasattr(sims, "get") else np.asarray(sims)


register("analysis.morans_i", "cpu")(_morans_perm)
register("analysis.morans_i", "gpu")(_morans_perm)


def morans_i(y: Any, w: Weights, *, permutations: int = 999, seed: int | None = None,
             two_tailed: bool = True, keep_simulations: bool = False,
             compute: Compute | None = None) -> GlobalResult:
    """Global Moran's I with normal-approximation and permutation (``p_sim``) inference.

    Matches ``esda.Moran`` for the same weights (``I``, ``EI``, ``VI_norm``, ``z_norm``,
    ``p_norm``); ``p_sim`` uses esda's folded definition.
    """
    return _morans_core(y, w, permutations, seed, compute, two_tailed, keep_simulations)


def _geary_perm(x: Any, W: Any, rc: Any, n: int, s0: float, zz: float, permutations: int,
                seed: int | None, xp: Any) -> np.ndarray:
    rng = xp.random.default_rng(seed)
    out = []
    step = _batches(n, permutations, _budget(xp))
    for s in range(0, permutations, step):
        xpm = _perm_columns(x, min(step, permutations - s), rng, xp)
        q = ((xpm * xpm) * rc[:, None]).sum(axis=0) - 2 * (xpm * (W @ xpm)).sum(axis=0)
        out.append((n - 1) * q / (2 * s0 * zz))
    sims = xp.concatenate(out)
    return sims.get() if hasattr(sims, "get") else np.asarray(sims)


register("analysis.gearys_c", "cpu")(_geary_perm)
register("analysis.gearys_c", "gpu")(_geary_perm)


def gearys_c(y: Any, w: Weights, *, permutations: int = 999, seed: int | None = None,
             compute: Compute | None = None) -> GlobalResult:
    """Geary's C (expected value 1; below 1 means positive autocorrelation).

    Uses ``sum_ij w_ij (x_i - x_j)^2 = sum_i x_i^2 (r_i + c_i) - 2 x'Wx`` (row and column
    sums), so permutations are batched matrix products. Normal inference uses the
    randomisation variance, as ``esda.Geary``.
    """
    v = _values(y, w)
    n = w.n
    impl, used = dispatch("analysis.gearys_c", compute)
    xp, W = _xp_and_matrix(w, used)
    ws = w.sparse
    rc_np = np.asarray(ws.sum(axis=1)).ravel() + np.asarray(ws.sum(axis=0)).ravel()
    x = xp.asarray(v)
    rc = xp.asarray(rc_np)
    zz = float(((v - v.mean()) ** 2).sum())
    s0 = float(ws.sum())
    q = float(((x * x) * rc).sum() - 2 * (x @ (W @ x)))
    c = (n - 1) * q / (2 * s0 * zz)
    s1 = float((ws + ws.T).power(2).sum()) / 2
    s2 = float(np.square(rc_np).sum())
    z = v - v.mean()
    k = (n * (z**4).sum()) / ((z * z).sum() ** 2)
    a = (n - 1) * s1 * (n * n - 3 * n + 3 - (n - 1) * k)
    b = 0.25 * (n - 1) * s2 * (n * n + 3 * n - 6 - (n * n - n + 2) * k)
    cc = s0 * s0 * (n * n - 3 - (n - 1) ** 2 * k)
    vc = (a - b + cc) / (n * (n - 2) * (n - 3) * s0 * s0)
    zn = (c - 1.0) / np.sqrt(vc)
    pn = stats.norm.sf(abs(zn))
    sims = impl(x, W, rc, n, s0, zz, permutations, seed, xp) if permutations else None
    p_sim = z_sim = None
    if sims is not None:
        p_sim = float(_fold_p(sims[:, None], c, permutations, np)[0])
        z_sim = float((c - sims.mean()) / sims.std())
    return GlobalResult(c, 1.0, vc, float(zn), float(pn), p_sim, z_sim, permutations)


# --------------------------------------------------------------------------- local statistics
def _padded_neighbors(w: Weights, rows: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """``(len(rows), kmax)`` neighbour weights (zero-padded) for the given rows."""
    ws = w.sparse
    wt = np.zeros((len(rows), kmax))
    for j, i in enumerate(rows):
        a, b = ws.indptr[i], ws.indptr[i + 1]
        wt[j, :b - a] = ws.data[a:b]
    return wt


def _local_perm(stat_obs: np.ndarray, zvals: np.ndarray, w: Weights, permutations: int,
                seed: int | None, xp: Any, combine) -> tuple[np.ndarray, np.ndarray]:
    """Conditional permutation shared across observations, as ``esda``'s ``crand``:
    each permutation draws one random set of ``kmax`` other observations; observation i
    uses its first ``k_i`` (indices at or above i shifted by one to skip i itself).

    ``combine(z_i, lag_perm)`` returns the permuted statistic. Returns ``(p_sim, sim_mean,
    sim_std)``, one value per observation each.
    """
    n = w.n
    card = w.cardinalities
    kmax = int(card.max()) if n else 0
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.permutation(n - 1)[:kmax] for _ in range(permutations)])  # (P, kmax)
    draws_x = xp.asarray(draws)
    z = xp.asarray(zvals)
    obs = xp.asarray(stat_obs)
    p_sim = np.empty(n)
    mean = np.empty(n)
    std = np.empty(n)
    block = max(1, int(_budget(xp) // max(permutations * max(kmax, 1), 1)))
    for s in range(0, n, block):
        rows = np.arange(s, min(n, s + block))
        wt = xp.asarray(_padded_neighbors(w, rows, kmax))                  # (B, kmax)
        ii = xp.asarray(rows)[:, None, None]
        idx = draws_x[None, :, :] + (draws_x[None, :, :] >= ii)           # (B, P, kmax)
        lag = (z[idx] * wt[:, None, :]).sum(axis=2)                       # (B, P)
        sims = combine(z[xp.asarray(rows)][:, None], lag)                 # (B, P)
        p_sim[rows] = _fold_p(sims.T, obs[xp.asarray(rows)][None, :], permutations, xp)
        m, sd = sims.mean(axis=1), sims.std(axis=1)
        mean[rows] = m.get() if hasattr(m, "get") else m
        std[rows] = sd.get() if hasattr(sd, "get") else sd
    return p_sim, mean, std


register("analysis.local", "cpu")(_local_perm)
register("analysis.local", "gpu")(_local_perm)


def local_morans(y: Any, w: Weights, *, permutations: int = 999, seed: int | None = None,
                 compute: Compute | None = None) -> pd.DataFrame:
    """Local Moran's I (LISA).

    Returns a DataFrame indexed like ``w.ids`` with ``Is``, quadrant ``q`` (1 HH, 2 LH,
    3 LL, 4 HL, as ``esda``), ``p_sim``, ``z_sim`` and ``EI_sim``. ``Is`` matches
    ``esda.Moran_Local``; use row-standardised weights for the usual interpretation.
    """
    v = _values(y, w)
    impl, used = dispatch("analysis.local", compute)
    xp, W = _xp_and_matrix(w, used)
    z = v - v.mean()
    m2 = (z * z).sum() / (w.n - 1)  # esda's scaling: Is = (n - 1) z_i lag_i / sum(z^2)
    lag = w.sparse @ z
    Is = z * lag / m2
    q = np.select([(z > 0) & (lag > 0), (z < 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0)],
                  [1, 2, 3, 4], 0)
    out = pd.DataFrame({"Is": Is, "q": q}, index=w.ids)
    if permutations:
        p, mean, sd = impl(Is, z, w, permutations, seed, xp, lambda zi, lg: zi * lg / m2)
        out["p_sim"] = p
        out["EI_sim"] = mean
        out["z_sim"] = (Is - mean) / sd
    return out


def getis_ord_local(y: Any, w: Weights, *, star: bool = True, permutations: int = 0,
                    seed: int | None = None, compute: Compute | None = None) -> pd.DataFrame:
    """Local Getis-Ord G (``star=False``) or G* (``star=True``) hot-spot statistic.

    Returns ``G`` and the analytic z-score ``Zs`` (positive = hot spot), plus ``p_norm``
    (one-sided, as ``esda.G_Local``) and, with ``permutations``, ``p_sim``. Uses binary
    weights; G* adds each observation to its own neighbourhood.
    """
    v = _values(y, w)
    n = w.n
    wb = w.binary().sparse
    if star:
        wb = (wb + sparse.identity(n, format="csr")).tocsr()
        wb.data[:] = 1.0
    wi = np.asarray(wb.sum(axis=1)).ravel()
    lag = wb @ v
    if star:
        g = lag / v.sum()
        xbar = v.mean()
        s = np.sqrt((v * v).mean() - xbar * xbar)
        zs = (lag - xbar * wi) / (s * np.sqrt((n * wi - wi * wi) / (n - 1)))
    else:
        others = v.sum() - v
        g = lag / others
        xbar = others / (n - 1)
        s = np.sqrt(((v * v).sum() - v * v) / (n - 1) - xbar * xbar)
        zs = (lag - xbar * wi) / (s * np.sqrt(((n - 1) * wi - wi * wi) / (n - 2)))
    out = pd.DataFrame({"G": g, "Zs": zs, "p_norm": stats.norm.sf(np.abs(zs))}, index=w.ids)
    if permutations:
        impl, used = dispatch("analysis.local", compute)
        xp, _ = _xp_and_matrix(w, used)
        total = v.sum()

        def combine(vi, lg):  # the focal value stays put; its neighbours are permuted
            return (vi + lg) / total if star else lg / (total - vi)

        p, _, _ = impl(g, v, w.binary(), permutations, seed, xp, combine)
        out["p_sim"] = p
    return out


# --------------------------------------------------------------------------- clustering
def _feature_matrix(data: Any) -> tuple[np.ndarray, Any]:
    if hasattr(data, "geometry"):
        return _coords(data), data.index
    arr = np.asarray(data, dtype=np.float64)
    return (arr[:, None] if arr.ndim == 1 else arr), None


def _centre(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the column means. Euclidean clustering does not change under a shift, but
    projected coordinates of millions of metres lose precision on the GPU: cuML's HDBSCAN on
    raw EPSG:5070 coordinates of 185,000 buildings agreed with scikit-learn at an adjusted
    Rand index of 0.0, and at 0.6 after centring."""
    m = x.mean(axis=0) if len(x) else np.zeros(x.shape[1])
    return x - m, m


def _labels(labels: Any, index: Any) -> Any:
    lab = labels.get() if hasattr(labels, "get") else np.asarray(labels)
    lab = np.asarray(lab).astype(np.int64)
    return pd.Series(lab, index=index, name="cluster") if index is not None else lab


@register("analysis.dbscan", "cpu")
def _dbscan_cpu(x: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x)


# cuML's DBSCAN searches neighbours by brute force; scikit-learn's KD-tree is far faster on
# 2-D points (measured: 200k points, 0.43 s CPU vs 13.3 s GPU), so GPU is opt-in.
@register("analysis.dbscan", "gpu", auto=False)
def _dbscan_gpu(x: np.ndarray, eps: float, min_samples: int) -> Any:
    import cupy as cp
    from ._backend import rapids

    DBSCAN = rapids("cuml.cluster").DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, output_type="cupy").fit_predict(
        cp.asarray(x, dtype=cp.float64))


def dbscan(data: Any, eps: float, *, min_samples: int = 5, compute: Compute | None = None) -> Any:
    """DBSCAN clusters (-1 = noise) for point geometries or an ``(n, d)`` array."""
    x, index = _feature_matrix(data)
    impl, _ = dispatch("analysis.dbscan", compute)
    return _labels(impl(_centre(x)[0], eps, min_samples), index)


@register("analysis.hdbscan", "cpu")
def _hdbscan_cpu(x: np.ndarray, min_cluster_size: int, min_samples: int | None) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(x)


# cuML's HDBSCAN builds its tree differently and, even on centred coordinates, gave a partition
# with 130 clusters where scikit-learn found 171 (adjusted Rand index 0.6, 185,000 buildings);
# it is much faster but not a drop-in replacement, so the GPU version is opt-in.
@register("analysis.hdbscan", "gpu", auto=False)
def _hdbscan_gpu(x: np.ndarray, min_cluster_size: int, min_samples: int | None) -> Any:
    import cupy as cp
    from ._backend import rapids

    HDBSCAN = rapids("cuml.cluster").HDBSCAN

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   output_type="cupy").fit_predict(cp.asarray(x, dtype=cp.float64))


def hdbscan(data: Any, *, min_cluster_size: int = 5, min_samples: int | None = None,
            compute: Compute | None = None) -> Any:
    """HDBSCAN clusters (-1 = noise) for point geometries or an ``(n, d)`` array.

    ``compute="auto"`` uses scikit-learn. ``compute="gpu"`` uses cuML, which is much faster
    but builds a different hierarchy, so its clusters can differ substantially.
    """
    x, index = _feature_matrix(data)
    impl, _ = dispatch("analysis.hdbscan", compute)
    return _labels(impl(_centre(x)[0], min_cluster_size, min_samples), index)


@register("analysis.kmeans", "cpu")
def _kmeans_cpu(x: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(x)
    return km.labels_, km.cluster_centers_


@register("analysis.kmeans", "gpu")
def _kmeans_gpu(x: np.ndarray, n_clusters: int, seed: int) -> tuple[Any, Any]:
    import cupy as cp
    from ._backend import rapids

    KMeans = rapids("cuml.cluster").KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, output_type="cupy")
    km.fit(cp.asarray(x, dtype=cp.float64))
    return km.labels_, km.cluster_centers_


def kmeans(data: Any, n_clusters: int, *, seed: int = 0,
           compute: Compute | None = None) -> tuple[Any, np.ndarray]:
    """K-means; returns ``(labels, centers)``."""
    x, index = _feature_matrix(data)
    impl, _ = dispatch("analysis.kmeans", compute)
    xc, shift = _centre(x)
    labels, centers = impl(xc, n_clusters, seed)
    centers = centers.get() if hasattr(centers, "get") else np.asarray(centers)
    return _labels(labels, index), centers + shift


# --------------------------------------------------------------------------- kernel density
_KDE_CUTOFF = {"gaussian": 5.0, "quartic": 1.0}


def _kde_block(px: np.ndarray, py: np.ndarray, pw: np.ndarray, shape: tuple[int, int],
               x0: float, dx: float, y0: float, dy: float, h: float, kernel: str,
               gpu: bool) -> Any:
    if gpu:
        import cupy as xp
    else:
        xp = np
    rows, cols = shape
    gx = xp.asarray(x0 + (np.arange(cols) + 0.5) * dx)
    gy = xp.asarray(y0 + (np.arange(rows) + 0.5) * dy)
    out = xp.zeros((rows, cols))
    if len(px) == 0:
        return out
    cx, cy, cw = xp.asarray(px), xp.asarray(py), xp.asarray(pw)
    step = max(1, int(5e7 // max(rows * cols, 1)))
    for s in range(0, len(px), step):
        ux = (gx[None, None, :] - cx[s:s + step, None, None]) / h
        uy = (gy[None, :, None] - cy[s:s + step, None, None]) / h
        u2 = ux * ux + uy * uy
        if kernel == "gaussian":
            k = xp.exp(-0.5 * u2) / (2 * np.pi)
            k = xp.where(u2 <= _KDE_CUTOFF["gaussian"] ** 2, k, 0.0)
        else:
            k = xp.where(u2 < 1, 3 / np.pi * (1 - u2) ** 2, 0.0)
        out += (k * cw[s:s + step, None, None]).sum(axis=0)
    return out / (h * h)


register("analysis.kde", "cpu")(_kde_block)
register("analysis.kde", "gpu")(_kde_block)


def kde(points: Any, like: xr.DataArray, *, bandwidth: float,
        kernel: Literal["gaussian", "quartic"] = "gaussian", weights: Sequence[float] | str | None = None,
        compute: Compute | None = None, chunks: Any = None) -> Any:
    """Kernel density (points per unit area, optionally weighted) on the grid of ``like``.

    Each raster chunk only receives points within the kernel's reach (5 bandwidths for
    the Gaussian, 1 for the quartic), so it scales to large grids and point sets.
    """
    import dask
    import dask.array as dsa
    import shapely

    from .bridge import _edges, _grid, _template

    if kernel not in _KDE_CUTOFF:
        raise ValueError("kernel must be 'gaussian' or 'quartic'")
    tmpl, _, ctx = _template(like, compute, chunks, "analysis.kde")
    impl, used = dispatch("analysis.kde", ctx.used)
    xy = _coords(points)
    w = (np.asarray(points[weights], dtype=float) if isinstance(weights, str) else
         np.ones(len(xy)) if weights is None else np.asarray(weights, dtype=float))
    tree = shapely.STRtree(shapely.points(xy))
    reach = _KDE_CUTOFF[kernel] * bandwidth
    x0, dx, y0, dy = _grid(tmpl)
    re, ce = _edges(tmpl.data.chunks[0]), _edges(tmpl.data.chunks[1])
    meta = tmpl.data._meta.astype(np.float64)
    rows = []
    for i in range(len(re) - 1):
        row = []
        for j in range(len(ce) - 1):
            bx0, by0 = x0 + ce[j] * dx, y0 + re[i] * dy
            bx1, by1 = x0 + ce[j + 1] * dx, y0 + re[i + 1] * dy
            idx = tree.query(shapely.box(min(bx0, bx1) - reach, min(by0, by1) - reach,
                                         max(bx0, bx1) + reach, max(by0, by1) + reach))
            shape = (int(re[i + 1] - re[i]), int(ce[j + 1] - ce[j]))
            blk = dask.delayed(impl)(xy[idx, 0], xy[idx, 1], w[idx], shape, bx0, dx, by0, dy,
                                     bandwidth, kernel, used == "gpu")
            row.append(dsa.from_delayed(blk, shape=shape, dtype=np.float64, meta=meta))
        rows.append(row)
    out = tmpl.copy(data=dsa.block(rows)).rename("density")
    out.attrs = {"kernel": kernel, "bandwidth": bandwidth}
    if not ctx.was_gpu and used == "gpu":
        from ._backend import to_cpu

        out = to_cpu(out)
    if not ctx.lazy:
        out = out.compute()
    return out if ctx.was_da else out.data
