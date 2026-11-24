"""Exposure mappings for spatial interference.

Under interference a unit's outcome depends on its own treatment and on the
treatments of others. An *exposure mapping* (Aronow & Samii, 2017) reduces
the full treatment vector to a small number of conditions per unit, e.g.

    (own treated?, enough neighbours treated?)  ->  c00, c01, c10, c11

Neighbourhoods come from any spatial weights object (PySpatialStats ``W``
from ``Queen``, ``KNN``, ``DistanceBand``, ``Kernel``…), from a distance
kernel, or from a bipartite source → receptor kernel.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_coords, as_sparse_weights

CONDITIONS = ("c00", "c01", "c10", "c11")
CONDITION_LABELS = {
    "c00": "control, low exposure",
    "c01": "control, high exposure",
    "c10": "treated, low exposure",
    "c11": "treated, high exposure",
}


# ----------------------------------------------------------------------------
# Kernel weights
# ----------------------------------------------------------------------------
def _decay(u: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-0.5 * u ** 2)
    if kernel == "exponential":
        return np.exp(-u)
    if kernel == "bisquare":
        return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    if kernel == "uniform":
        return (u <= 1).astype(float)
    raise ValueError("kernel must be 'gaussian', 'exponential', 'bisquare' or 'uniform'")


def _truncation(kernel: str) -> float:
    return {"gaussian": 3.0, "exponential": 5.0, "bisquare": 1.0, "uniform": 1.0}[kernel]


def kernel_weights(coords, bandwidth: float, kernel: str = "gaussian",
                   target_coords=None) -> sparse.csr_matrix:
    """Sparse distance-decay weights ``K[i, j] = k(d_ij / bandwidth)``.

    With ``target_coords`` the matrix is (n_target × n_source), for
    bipartite designs; otherwise it is square with a zero diagonal.
    """
    src = as_coords(coords)
    tgt = src if target_coords is None else as_coords(target_coords)
    r = _truncation(kernel) * bandwidth
    D = cKDTree(tgt).sparse_distance_matrix(cKDTree(src), max_distance=r, output_type="coo_matrix")
    vals = _decay(D.data / bandwidth, kernel)
    K = sparse.csr_matrix((vals, (D.row, D.col)), shape=(len(tgt), len(src)))
    if target_coords is None:
        K = sparse.lil_matrix(K)
        K.setdiag(0.0)
        K = sparse.csr_matrix(K)
    K.eliminate_zeros()
    return K


# ----------------------------------------------------------------------------
# Exposure mapping
# ----------------------------------------------------------------------------
class ExposureMapping:
    """Map a treatment vector to neighbourhood exposure and exposure conditions.

    Parameters
    ----------
    w : spatial weights
        PySpatialStats ``W``, libpysal W, scipy sparse, or ndarray.
    kind : {"share", "count", "any", "weighted"}
        * ``share``    — fraction of neighbours treated (row-standardised).
        * ``count``    — number of treated neighbours (binary weights).
        * ``any``      — 1 if at least one neighbour is treated.
        * ``weighted`` — ``W @ t`` with the weights as given (e.g. a kernel).
    threshold : float
        Neighbour exposure ``>= threshold`` counts as "high" when forming
        the four conditions. For ``any`` it is ignored.
    """

    def __init__(self, w, kind: str = "share", threshold: float = 0.5):
        if kind not in ("share", "count", "any", "weighted"):
            raise ValueError("kind must be 'share', 'count', 'any' or 'weighted'")
        self.w = w
        self.kind = kind
        self.threshold = float(threshold)

    def _matrix(self, n: int) -> sparse.csr_matrix:
        if self.kind == "share":
            return as_sparse_weights(self.w, n, row_standardize=True)
        if self.kind in ("count", "any"):
            m = as_sparse_weights(self.w, n)
            m.data = np.ones_like(m.data)
            return m
        return as_sparse_weights(self.w, n)

    def neighbor_exposure(self, treatment) -> np.ndarray:
        """Continuous neighbourhood exposure ``g_i``."""
        t = as_1d(treatment, "treatment")
        g = np.asarray(self._matrix(len(t)) @ t).ravel()
        if self.kind == "any":
            g = (g > 0).astype(float)
        return g

    def high_exposure(self, treatment) -> np.ndarray:
        g = self.neighbor_exposure(treatment)
        if self.kind == "any":
            return g.astype(int)
        return (g >= self.threshold).astype(int)

    def conditions(self, treatment) -> np.ndarray:
        """Four-level exposure condition as integer codes 0..3 = c00, c01, c10, c11."""
        t = as_1d(treatment).astype(int)
        return 2 * t + self.high_exposure(t)

    def condition_names(self, treatment) -> np.ndarray:
        return np.asarray(CONDITIONS)[self.conditions(treatment)]

    def __repr__(self) -> str:
        return f"ExposureMapping(kind='{self.kind}', threshold={self.threshold})"


class DistanceDecayExposure(ExposureMapping):
    """Exposure as a kernel-weighted sum of treated units within range.

    ``g_i = sum_j k(d_ij / bandwidth) t_j``. Set ``normalize=True`` to
    divide by ``sum_j k(d_ij / bandwidth)``, giving a kernel-weighted share.
    """

    def __init__(self, coords, bandwidth: float, kernel: str = "gaussian",
                 normalize: bool = True, threshold: float = 0.5):
        K = kernel_weights(coords, bandwidth, kernel)
        super().__init__(K, kind="share" if normalize else "weighted", threshold=threshold)
        self.bandwidth = float(bandwidth)
        self.kernel = kernel


class BipartiteExposure:
    """Exposure of receptor units to treatments applied at *other* locations.

    Typical case: interventions at power plants (sources), outcomes at
    residential zip codes (receptors) — Zigler & Papadogeorgou (2021).

    Parameters
    ----------
    source_coords, receptor_coords : (m, 2), (n, 2)
    bandwidth : float
    kernel : distance-decay kernel
    normalize : divide by total source influence on each receptor
    """

    def __init__(self, source_coords, receptor_coords, bandwidth: float,
                 kernel: str = "gaussian", normalize: bool = False, threshold: float = 0.5):
        self.K = kernel_weights(source_coords, bandwidth, kernel, target_coords=receptor_coords)
        self.normalize = normalize
        self.threshold = float(threshold)

    def neighbor_exposure(self, source_treatment) -> np.ndarray:
        t = as_1d(source_treatment, "source_treatment")
        g = np.asarray(self.K @ t).ravel()
        if self.normalize:
            tot = np.asarray(self.K.sum(axis=1)).ravel()
            g = np.divide(g, tot, out=np.zeros_like(g), where=tot > 0)
        return g

    def high_exposure(self, source_treatment) -> np.ndarray:
        return (self.neighbor_exposure(source_treatment) >= self.threshold).astype(int)

    def key_source(self) -> np.ndarray:
        """Index of the most influential source for each receptor."""
        return np.asarray(self.K.argmax(axis=1)).ravel()


# ----------------------------------------------------------------------------
# Randomisation designs (for design-based exposure probabilities)
# ----------------------------------------------------------------------------
class BernoulliDesign:
    """Each unit treated independently with probability ``p`` (scalar or vector)."""

    def __init__(self, p):
        self.p = p

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(n) < np.broadcast_to(self.p, (n,))).astype(int)


class CompleteDesign:
    """Exactly ``n_treated`` units treated, chosen uniformly at random."""

    def __init__(self, n_treated: int):
        self.n_treated = int(n_treated)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        t = np.zeros(n, dtype=int)
        t[rng.choice(n, self.n_treated, replace=False)] = 1
        return t


class ClusterDesign:
    """Whole clusters (e.g. villages, blocks) treated with probability ``p``."""

    def __init__(self, clusters, p: float = 0.5):
        self.clusters = np.asarray(clusters)
        self.p = float(p)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        uniq, inv = np.unique(self.clusters, return_inverse=True)
        on = rng.random(len(uniq)) < self.p
        return on[inv].astype(int)


def exposure_probabilities(mapping: ExposureMapping, design, n: int, n_sims: int = 2000,
                           random_state: int = 0) -> np.ndarray:
    """Monte-Carlo probability that each unit falls in each exposure condition.

    Returns an (n, 4) array ``pi[i, c]`` for c in ``CONDITIONS``. These are
    the inverse-probability weights for Horvitz–Thompson / Hájek estimators
    when treatment was randomised by a known ``design``.
    """
    rng = np.random.default_rng(random_state)
    counts = np.zeros((n, 4))
    rows = np.arange(n)
    for _ in range(n_sims):
        c = mapping.conditions(design.sample(n, rng))
        counts[rows, c] += 1
    return counts / n_sims
