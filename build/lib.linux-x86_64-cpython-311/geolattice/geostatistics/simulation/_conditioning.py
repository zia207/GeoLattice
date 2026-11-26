"""
Global dual kriging used to condition unconditional simulations.

Conditioning by kriging (Journel & Huijbregts, 1978) turns any
unconditional Gaussian simulation ``y`` into a conditional one::

    z_cs(x) = z*_K(x) + [y(x) - y*_K(x)]

Both kriged terms use the same weights, so the kriging system is factorized
once and applied to the data and to every realization's values at the data
locations (dual form), with the grid processed in chunks. Shared by
`tbsim` and `fftsim`.

Also holds the nearest-informed-neighbour search shared by the sequential
simulators (`sicosim`, `sgcosim`, `dssim`).
"""

from typing import Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import cKDTree

from .lusim import _cov_block

_TARGET_CHUNK = 20000


class DualKriging:
    """
    Simple or ordinary kriging from all of `data_xy`, factorized once.

    `predict` krigs columns of values at the data locations onto targets;
    for simple kriging the values must already have the mean removed.

    With ``exact_at_data=True`` a target that coincides with a datum gets
    the full sill as cross-covariance (the datum *is* that node's value,
    nugget included), so kriging reproduces it exactly. Use this when the
    unconditional field shares one value per location (lattice methods);
    leave it False when the nugget is simulated independently per point.
    """

    def __init__(self, data_xy: np.ndarray, vario: Sequence, rotation: np.ndarray,
                 kriging_type: str = "simple", exact_at_data: bool = False):
        self.data_xy = np.asarray(data_xy, dtype=float)
        self.exact_at_data = exact_at_data
        self.vario = vario
        self.rotation = rotation
        self.kriging_type = kriging_type
        self.sill = float(vario[4])
        n1 = len(self.data_xy)
        C11 = _cov_block(self.data_xy, self.data_xy, vario, rotation, same=True)
        C11[np.diag_indices(n1)] += 1e-10 * self.sill
        if kriging_type == "simple":
            K = C11
        else:
            K = np.zeros((n1 + 1, n1 + 1))
            K[:n1, :n1] = C11
            K[:n1, n1] = K[n1, :n1] = 1.0
        self._K = lu_factor(K, check_finite=False)

    def _cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        C = _cov_block(a, b, self.vario, self.rotation)
        if self.exact_at_data and float(self.vario[1]) > 0:
            same = (np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2) < 1e-9)
            C[same] = self.sill
        return C

    def predict(self, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Krige `values` (n_data, k) onto `targets` (m, 2); returns (m, k)."""
        n1 = len(self.data_xy)
        rhs = values if self.kriging_type == "simple" else np.vstack(
            [values, np.zeros((1, values.shape[1]))])
        coef = lu_solve(self._K, rhs, check_finite=False)
        out = np.empty((len(targets), values.shape[1]))
        for p0 in range(0, len(targets), _TARGET_CHUNK):
            t = targets[p0:p0 + _TARGET_CHUNK]
            out[p0:p0 + len(t)] = self._cross(t, self.data_xy) @ coef[:n1]
            if self.kriging_type == "ordinary":
                out[p0:p0 + len(t)] += coef[n1]
        return out

    def std(self, targets: np.ndarray, chunk: int = 2000) -> np.ndarray:
        """Kriging standard deviation at `targets` (m, 2)."""
        n1 = len(self.data_xy)
        var = np.empty(len(targets))
        for p0 in range(0, len(targets), chunk):
            t = targets[p0:p0 + chunk]
            C12 = self._cross(self.data_xy, t)
            rhs = C12 if self.kriging_type == "simple" else np.vstack(
                [C12, np.ones((1, len(t)))])
            sol = lu_solve(self._K, rhs, check_finite=False)
            v = self.sill - np.einsum("ij,ij->j", sol[:n1], C12)
            if self.kriging_type == "ordinary":
                v -= sol[n1]
            var[p0:p0 + len(t)] = v
        return np.sqrt(np.clip(var, 0.0, None))


def condition_on_values(dk: "DualKriging", mean: float, Yg: np.ndarray, Yd: np.ndarray,
                        data_values, targets: np.ndarray) -> np.ndarray:
    """
    Condition unconditional values `Yg` (targets, nsim) / `Yd` (data, nsim)
    on new `data_values` at the same data locations: one vector for all
    realizations, or one column per realization.
    """
    dv = np.asarray(data_values, dtype=float)
    if dv.ndim == 1 and dv.shape[0] == Yd.shape[0]:
        dv = np.broadcast_to(dv[:, None], Yd.shape)
    if dv.shape != Yd.shape:
        raise ValueError(f"data_values must have shape ({Yd.shape[0]},) or {Yd.shape}")
    base = mean if dk.kriging_type == "simple" else 0.0
    return base + Yg + dk.predict(dv - base - Yd, targets)


def sequential_neighbors(data_xy: np.ndarray, grid: np.ndarray, num_points: int,
                         radius: float = np.inf):
    """
    Candidate neighbour lists for sequential simulation.

    Locations are the data followed by the grid nodes (``loc = [data; grid]``).
    For every node, returns the indices into ``loc`` of its nearest
    ``max(8 * num_points, 64)`` locations within `radius` (itself excluded,
    -1 padding), from which a simulator picks the first `num_points` that
    are informed; and, per node, the index of a datum sitting exactly on it
    (-1 if none).

    Returns
    -------
    loc : (n_data + n_nodes, 2), nbrs : (n_nodes, m) int64, coincident : (n_nodes,) int64
    """
    data_xy = np.asarray(data_xy, dtype=float).reshape(-1, 2)
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    loc = np.vstack([data_xy, grid])
    n1 = len(data_xy)
    m = int(min(len(loc), max(8 * num_points, 64)))
    d, idx = cKDTree(loc).query(grid, k=m, distance_upper_bound=radius)
    idx = np.asarray(idx).reshape(len(grid), -1)
    d = np.asarray(d).reshape(len(grid), -1)
    self_idx = n1 + np.arange(len(grid))
    nbrs = np.where((idx == self_idx[:, None]) | ~np.isfinite(d), -1, idx).astype(np.int64)
    if n1:
        dd, di = cKDTree(data_xy).query(grid)
        coincident = np.where(dd <= 1e-9, di, -1).astype(np.int64)
    else:
        coincident = np.full(len(grid), -1, dtype=np.int64)
    return loc, nbrs, coincident
