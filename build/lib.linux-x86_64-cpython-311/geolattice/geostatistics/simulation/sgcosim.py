"""
Sequential Gaussian co-simulation with a secondary variable.

Sequential Gaussian simulation in which every node is drawn from its
*collocated simple cokriging* distribution: the primary normal scores of
the nearest informed locations (hard data and previously simulated nodes)
plus the secondary normal score at the node itself, under Markov Model 1
(Xu, Tran, Srivastava & Journel, 1992; GSLIB ``sgsim`` collocated
co-simulation). With primary covariance ``C_Z`` and correlation ``rho``
between primary and secondary normal scores,

    Cov(Z(u_a), Y(u)) = rho * C_Z(u_a - u) / sqrt(C_Z(0)),     Var(Y) = 1,

so only the primary variogram and ``rho`` are needed. The node value is
drawn from ``N(estimate, cokriging variance)`` and then joins the
conditioning data. ``rho = 0`` gives plain SGS; ``rho -> 1`` makes the
primary a rescaled copy of the secondary.

Workflow
--------
* The primary is normal-score transformed from the hard data (`itrans`,
  default True) and realizations are back-transformed (``zmin`` / ``zmax``
  clip the linear tails).
* The secondary must be known at every node (seismic, a DEM, a satellite
  index, ...); it is normal-score transformed over the grid.
* ``rho`` is estimated from the collocated normal scores at the data (the
  secondary value there is `secondary_col`, or the nearest node's) unless
  given.
* If the secondary is itself only known at the data, use
  `sgcosim_hierarchical`: simulate the secondary by SGS first, then the
  primary conditioned on each secondary realization; the pair reproduces
  their correlation.

Collocated vs multi-collocated
------------------------------
With ``secondary_neighbors=1`` (default, GSLIB) only the collocated
secondary is used. That is robust when the secondary is at least as
continuous as the primary (seismic, DEMs), but under-reproduces ``rho`` when
the secondary is *noisy*, because the secondary at not-yet-simulated nodes
is ignored. ``secondary_neighbors=m > 1`` uses the secondary at the ``m``
nearest nodes with the full MM1 covariance
``Cov(Y_a, Y_b) = rho^2 C(h) / C(0) + (1 - rho^2) [a == b]``, which restores
``rho`` when that model holds (secondary with a relative nugget near
``1 - rho^2``), but over-fits badly when the secondary is smooth. Check
`mm1_nugget_ratio_`: the secondary's relative nugget (lag-1 cell semivariogram
of its normal scores, rescaled) divided by the MM1 value ``1 - rho^2``. Near
or above 1 favours ``m > 1``; well below 1 (smooth secondary), keep ``m = 1``.

Neighbours are the `num_points` nearest informed locations within `radius`
(no octant constraint).
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._conditioning import sequential_neighbors
from .lusim import _cov_block
from ..gslib.nscore import nscore_back, nscore_forward
from .sgsim import _covar, _get_random_generator, _make_rotation_matrix, _validate_vario

__all__ = ["GaussianCoSimulator", "sgcosim", "sgcosim_hierarchical", "sgsim_gpu"]


def _collocated_cokrige(c_zz: np.ndarray, c_z0: np.ndarray, z: np.ndarray, c00: float,
                        y0: Optional[float], rho: float):
    """
    Collocated simple cokriging (MM1) of a zero-mean primary. Returns the
    estimate and the cokriging variance.
    """
    n = len(z)
    s = np.sqrt(c00)
    use_y = y0 is not None and rho != 0.0
    if use_y:
        K = np.empty((n + 1, n + 1))
        K[:n, :n] = c_zz
        K[:n, n] = K[n, :n] = rho * c_z0 / s
        K[n, n] = 1.0
        rhs = np.concatenate([c_z0, [rho * s]])
        vals = np.concatenate([z, [y0]])
    else:
        K, rhs, vals = c_zz, c_z0, z
    if len(rhs) == 0:
        return 0.0, c00
    try:
        w = np.linalg.solve(K + 1e-10 * c00 * np.eye(len(K)), rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return float(w @ vals), float(max(c00 - w @ rhs, 0.0))


def _mm1_cokrige(xy_z: np.ndarray, z: np.ndarray, xy_y: np.ndarray, y: np.ndarray,
                 target: np.ndarray, vario, rotation, rho: float):
    """
    Simple cokriging under Markov Model 1 with primary data at `xy_z` and
    secondary data at `xy_y` (the collocated node included). MM1 implies

        Cov(Z_a, Z_b) = C(h),   Cov(Z_a, Y_b) = rho C(h) / s,
        Cov(Y_a, Y_b) = rho^2 C(h) / C(0) + (1 - rho^2) [a == b],

    with ``s = sqrt(C(0))`` and ``Var(Y) = 1``. Returns estimate and variance.
    """
    c00 = float(vario[4])
    s = np.sqrt(c00)
    n, m = len(z), len(y)
    pts = np.vstack([xy_z, xy_y])
    C = _cov_block(pts, pts, vario, rotation, same=True)
    c0 = _cov_block(pts, target[None, :], vario, rotation)[:, 0]
    c0[n:][np.abs(xy_y - target).sum(axis=1) < 1e-9] = c00     # collocated: C(0)
    K = C.copy()
    K[:n, n:] *= rho / s
    K[n:, :n] *= rho / s
    K[n:, n:] = rho ** 2 * C[n:, n:] / c00 + (1.0 - rho ** 2) * np.eye(m)
    rhs = c0.copy()
    rhs[n:] *= rho / s
    vals = np.concatenate([z, y])
    try:
        w = np.linalg.solve(K + 1e-10 * c00 * np.eye(n + m), rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return float(w @ vals), float(max(c00 - w @ rhs, 0.0))


def _ordinary_krige(xy: np.ndarray, z: np.ndarray, target: np.ndarray, vario, rotation):
    """Ordinary kriging of residuals; falls back to the mean with no neighbours."""
    c00 = float(vario[4])
    n = len(z)
    if n == 0:
        return 0.0, c00
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = _cov_block(xy, xy, vario, rotation, same=True) + 1e-10 * c00 * np.eye(n)
    K[:n, n] = K[n, :n] = 1.0
    rhs = np.concatenate([_cov_block(xy, target[None, :], vario, rotation)[:, 0], [1.0]])
    try:
        w = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return float(w[:n] @ z), float(max(c00 - w @ rhs, 0.0))


def _nscore_with_table(values: np.ndarray, v_sorted: np.ndarray, ns_sorted: np.ndarray) -> np.ndarray:
    """Normal scores of `values` through an existing transform table (linear tails)."""
    out = np.interp(values, v_sorted, ns_sorted)
    if len(v_sorted) > 1:
        lo = (ns_sorted[1] - ns_sorted[0]) / max(v_sorted[1] - v_sorted[0], 1e-12)
        hi = (ns_sorted[-1] - ns_sorted[-2]) / max(v_sorted[-1] - v_sorted[-2], 1e-12)
        out = np.where(values < v_sorted[0], ns_sorted[0] + lo * (values - v_sorted[0]), out)
        out = np.where(values > v_sorted[-1], ns_sorted[-1] + hi * (values - v_sorted[-1]), out)
    return out


class GaussianCoSimulator:
    """
    Sequential Gaussian co-simulator (collocated cokriging, Markov Model 1).

    Parameters
    ----------
    vario : sequence
        GSLIB-style variogram of the primary (normal scores when `itrans`):
        ``[azimuth, nugget, major, minor, sill, vtype(, smoothness)]``.
    rho : float, optional
        Correlation between primary and secondary normal scores; estimated
        from the data when None.
    num_points : int, default 16
        Maximum informed neighbours per node.
    radius : float, optional
        Search radius (default: unlimited).
    itrans : bool, default True
        Normal-score transform the primary data and back-transform results.
    mean : float, optional
        Simple-kriging mean of the (transformed) primary; default 0 with
        `itrans`, else the data mean.
    zmin, zmax : float, optional
        Clip back-transformed realizations (the transform's linear tails are
        unbounded).
    secondary_neighbors : int, default 1
        Secondary values used per node: 1 = collocated only (GSLIB); more =
        multi-collocated MM1 cokriging with the nearest nodes (see module
        docstring and `mm1_nugget_ratio_`).
    kriging_type : {'simple', 'ordinary'}, default 'simple'
        Kriging at each node. Ordinary kriging re-estimates the local mean
        and is only available without a secondary variable (MM1 collocated
        cokriging is a simple-kriging system).
    use_gpu : bool or 'auto', default False
        Simulate all realizations in lockstep on a CUDA GPU, one thread per
        realization and one fused kernel per step (`geolattice.geostatistics.sgs_gpu`).
        Gives the same realizations as the CPU engine up to rounding.
        Supports simple / ordinary kriging and the collocated secondary
        (``secondary_neighbors=1``) for spherical, exponential and Gaussian
        variograms, with up to 48 neighbours.

    Attributes
    ----------
    rho_ : float
        Correlation used.
    mm1_nugget_ratio_ : float or None
        Secondary relative nugget over the MM1-implied ``1 - rho^2``.
    secondary_scores_ : ndarray (n_nodes,) or None
        Secondary normal scores at the nodes.

    Examples
    --------
    >>> cosim = GaussianCoSimulator([0, 0, 40, 20, 1.0, "spherical"])
    >>> cosim.fit(grid, wells, "X", "Y", "porosity", secondary=impedance, secondary_col="ai")
    >>> reals = cosim.simulate(nsim=50, seed=1)
    """

    def __init__(self, vario: Sequence, rho: Optional[float] = None, num_points: int = 16,
                 radius: Optional[float] = None, itrans: bool = True,
                 mean: Optional[float] = None, zmin: Optional[float] = None,
                 zmax: Optional[float] = None, secondary_neighbors: int = 1,
                 kriging_type: str = "simple", use_gpu: Union[bool, str] = False):
        _validate_vario(vario)
        if rho is not None and not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        if num_points < 1:
            raise ValueError("num_points must be >= 1")
        self.vario = list(vario)
        self.rho = rho
        self.num_points = int(num_points)
        self.radius = np.inf if radius is None else float(radius)
        self.itrans = itrans
        self.mean = mean
        self.zmin, self.zmax = zmin, zmax
        if secondary_neighbors < 1:
            raise ValueError("secondary_neighbors must be >= 1")
        self.secondary_neighbors = int(secondary_neighbors)
        if kriging_type not in ("simple", "ordinary"):
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")
        self.kriging_type = kriging_type
        self.use_gpu = use_gpu
        self._R = _make_rotation_matrix(vario[0], vario[2], vario[3])

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z",
            secondary: Optional[np.ndarray] = None,
            secondary_col: Optional[str] = None) -> "GaussianCoSimulator":
        """
        Prepare the simulation.

        `secondary` (n_nodes,) is the exhaustive secondary at the grid nodes
        (None = plain SGS). Its value at the data is `df[secondary_col]` or
        the nearest node's; it is only used to estimate ``rho``.
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        has_data = df is not None and len(df) > 0
        if has_data:
            xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            if self.itrans:
                z, self._v_sorted, self._ns_sorted = nscore_forward(z, tails="linear")
        else:
            if self.itrans:
                raise ValueError("itrans needs hard data to build the transform")
            xy, z = np.empty((0, 2)), np.empty(0)
        self._has_data = has_data
        if self.mean is not None:
            self.mean_ = float(self.mean)
        elif self.itrans or not has_data:
            self.mean_ = 0.0
        else:
            self.mean_ = float(z.mean())

        self.secondary_scores_ = None
        self.rho_ = 0.0
        self.mm1_nugget_ratio_ = None
        if secondary is not None:
            sec = np.asarray(secondary, dtype=float).ravel()
            if len(sec) != len(grid):
                raise ValueError("secondary must have one value per grid node")
            y, sv, sn = nscore_forward(sec, tails="linear")
            self.secondary_scores_ = y
            if self.rho is not None:
                self.rho_ = float(self.rho)
            else:
                if not has_data or len(z) < 3:
                    raise ValueError("estimating rho needs at least 3 hard data; pass rho")
                s_data = (df[secondary_col].to_numpy(dtype=float) if secondary_col is not None
                          else sec[cKDTree(grid).query(xy)[1]])
                y_data = _nscore_with_table(s_data, sv, sn)
                self.rho_ = float(np.corrcoef(z, y_data)[0, 1])
        elif self.rho not in (None, 0.0):
            raise ValueError("rho was given but no secondary variable")
        if self.kriging_type == "ordinary" and self.secondary_scores_ is not None:
            raise ValueError("kriging_type='ordinary' is not available with a secondary variable")

        if self.secondary_scores_ is not None:
            # MM1 diagnostic: secondary relative nugget vs the implied 1 - rho^2
            gtree = cKDTree(grid)
            if len(grid) > 1 and abs(self.rho_) < 1.0:
                dist, nn = gtree.query(grid, k=2)
                h_vec = (grid[nn[:, 1]] - grid) @ self._R
                c00 = float(self.vario[4])
                gz = c00 - _covar(np.sqrt((h_vec ** 2).sum(axis=1)), c00, float(self.vario[1]),
                                  self.vario[5], self.vario[6] if str(self.vario[5]).lower() == "matern" else None)
                ys = self.secondary_scores_
                gy = 0.5 * np.mean((ys - ys[nn[:, 1]]) ** 2)
                self.mm1_nugget_ratio_ = float((gy - self.rho_ ** 2 * gz.mean() / c00) / (1.0 - self.rho_ ** 2))
            if self.secondary_neighbors > 1:
                k = min(self.secondary_neighbors, len(grid))
                self._sec_nbrs = np.asarray(gtree.query(grid, k=k)[1]).reshape(len(grid), -1)

        self._grid = grid
        self._data_z = z
        self._loc, self._nbrs, self._coincident = sequential_neighbors(
            xy, grid, self.num_points, self.radius)
        self.n_nodes_ = len(grid)
        return self

    # ------------------------------------------------------------- simulate

    def _simulate_one(self, rng: np.random.Generator, y: Optional[np.ndarray]) -> np.ndarray:
        n1 = len(self._data_z)
        c00 = float(self.vario[4])
        m = self.mean_
        vals = np.full(len(self._loc), np.nan)
        vals[:n1] = self._data_z - m
        out = np.empty(self.n_nodes_)
        for node in rng.permutation(self.n_nodes_):
            if self._coincident[node] >= 0:
                vals[n1 + node] = out[node] = self._data_z[self._coincident[node]] - m
                continue
            cand = self._nbrs[node]
            cand = cand[cand >= 0]
            informed = cand[~np.isnan(vals[cand])][: self.num_points]
            if self.kriging_type == "ordinary" and (y is None or self.rho_ == 0.0):
                est, var = _ordinary_krige(self._loc[informed], vals[informed], self._grid[node],
                                           self.vario, self._R)
            elif y is not None and self.secondary_neighbors > 1 and self.rho_ != 0.0:
                sn = self._sec_nbrs[node]
                est, var = _mm1_cokrige(self._loc[informed], vals[informed], self._grid[sn], y[sn],
                                        self._grid[node], self.vario, self._R, self.rho_)
            else:
                if len(informed):
                    xy_n = self._loc[informed]
                    c_zz = _cov_block(xy_n, xy_n, self.vario, self._R, same=True)
                    c_z0 = _cov_block(xy_n, self._grid[node][None, :], self.vario, self._R)[:, 0]
                else:
                    c_zz, c_z0 = np.empty((0, 0)), np.empty(0)
                est, var = _collocated_cokrige(c_zz, c_z0, vals[informed], c00,
                                               None if y is None else y[node], self.rho_)
            vals[n1 + node] = out[node] = est + np.sqrt(var) * rng.standard_normal()
        return out + m

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None,
                 secondary_scores: Optional[np.ndarray] = None,
                 back_transform: bool = True) -> np.ndarray:
        """
        Realizations at the grid nodes, shape ``(nsim, n_nodes)``.

        `secondary_scores` overrides the fitted secondary normal scores, as
        ``(n_nodes,)`` or one row per realization ``(nsim, n_nodes)`` (used by
        `sgcosim_hierarchical`). `back_transform=False` returns normal scores.
        """
        if not hasattr(self, "_loc"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        y = self.secondary_scores_ if secondary_scores is None else np.asarray(secondary_scores, float)
        if y is not None:
            if y.shape == (self.n_nodes_,):
                y = np.broadcast_to(y, (nsim, self.n_nodes_))
            elif y.shape != (nsim, self.n_nodes_):
                raise ValueError(f"secondary_scores must be ({self.n_nodes_},) or ({nsim}, {self.n_nodes_})")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        from ..utils.backend import resolve_cupy_use_gpu

        sims = None
        if resolve_cupy_use_gpu(self.use_gpu):
            from ..gpu.sgs_gpu import sgs_simulate_gpu

            try:
                sims = sgs_simulate_gpu(self, seeds, y)
            except Exception as exc:          # GPU busy / out of memory: same result on CPU
                if "memory" not in str(exc).lower() and "OutOfMemory" not in type(exc).__name__:
                    raise
                import warnings

                warnings.warn(f"GPU simulation failed ({type(exc).__name__}: {exc}); "
                              "running on the CPU, which gives the same realizations", stacklevel=2)
        if sims is None:
            sims = np.stack([self._simulate_one(_get_random_generator(int(s)),
                                                None if y is None else y[r])
                             for r, s in enumerate(seeds)])
        if back_transform and self.itrans:
            sims = nscore_back(sims.ravel(), self._v_sorted, self._ns_sorted,
                               tails="linear").reshape(sims.shape)
        if back_transform and (self.zmin is not None or self.zmax is not None):
            sims = np.clip(sims, self.zmin, self.zmax)
        return sims


def sgcosim(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    vario: Sequence,
    secondary: np.ndarray,
    secondary_col: Optional[str] = None,
    rho: Optional[float] = None,
    num_points: int = 16,
    radius: Optional[float] = None,
    nsim: int = 1,
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Sequential Gaussian co-simulation with an exhaustive secondary, in one call.

    Parameters
    ----------
    prediction_grid : (n, 2) node coordinates.
    df : hard data with columns `xx`, `yy`, `zz` (and optionally `secondary_col`).
    vario : primary variogram (normal scores by default).
    secondary : (n,) secondary variable at every node.
    secondary_col : column of `df` holding the secondary at the data.
    rho : correlation of normal scores; estimated when None.
    num_points, radius, nsim, seed : see `GaussianCoSimulator`.
    **kwargs : ``itrans``, ``mean``, ``zmin``, ``zmax``, ``secondary_neighbors``.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = GaussianCoSimulator(vario, rho=rho, num_points=num_points, radius=radius, **kwargs)
    sim.fit(prediction_grid, df, xx, yy, zz, secondary=secondary, secondary_col=secondary_col)
    return sim.simulate(nsim, seed)


def sgcosim_hierarchical(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    primary: str,
    secondary: str,
    vario_primary: Sequence,
    vario_secondary: Sequence,
    rho: Optional[float] = None,
    num_points: int = 16,
    radius: Optional[float] = None,
    nsim: int = 1,
    seed: Optional[Union[int, np.random.Generator]] = None,
    zmin=None,
    zmax=None,
):
    """
    Joint co-simulation of two variables known only at the data.

    The secondary is simulated first by SGS (`vario_secondary`), then the
    primary by collocated co-simulation on each secondary realization, with
    ``rho`` the correlation of their normal scores at the data (estimated
    when None). Both variables are normal-score transformed from the data.

    Returns
    -------
    (primary_sims, secondary_sims) : two arrays of shape (nsim, n_nodes)
    """
    rng = _get_random_generator(seed)
    s1, s2 = (int(v) for v in rng.integers(0, 2**31 - 1, size=2))
    lo = (None, None) if zmin is None else (zmin if np.ndim(zmin) else (zmin, zmin))
    hi = (None, None) if zmax is None else (zmax if np.ndim(zmax) else (zmax, zmax))

    sec_sim = GaussianCoSimulator(vario_secondary, num_points=num_points, radius=radius,
                                  zmin=lo[1], zmax=hi[1])
    sec_sim.fit(prediction_grid, df, xx, yy, secondary)
    sec_scores = sec_sim.simulate(nsim, s1, back_transform=False)
    sec_values = nscore_back(sec_scores.ravel(), sec_sim._v_sorted, sec_sim._ns_sorted,
                             tails="linear").reshape(sec_scores.shape)
    if lo[1] is not None or hi[1] is not None:
        sec_values = np.clip(sec_values, lo[1], hi[1])

    if rho is None:
        zp, _, _ = nscore_forward(df[primary].to_numpy(dtype=float), tails="linear")
        zs, _, _ = nscore_forward(df[secondary].to_numpy(dtype=float), tails="linear")
        rho = float(np.corrcoef(zp, zs)[0, 1])
    prim = GaussianCoSimulator(vario_primary, rho=rho, num_points=num_points, radius=radius,
                               zmin=lo[0], zmax=hi[0])
    # the secondary's own scores are its normal scores already (same transform as at the data)
    prim.fit(prediction_grid, df, xx, yy, primary, secondary=sec_scores[0])
    prim_values = prim.simulate(nsim, s2, secondary_scores=sec_scores)
    return prim_values, sec_values


def sgsim_gpu(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    vario: Sequence,
    num_points: int = 16,
    radius: Optional[float] = None,
    kriging_type: str = "simple",
    nsim: int = 1,
    itrans: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
    use_gpu: Union[bool, str] = True,
    **kwargs,
) -> np.ndarray:
    """
    GPU-accelerated sequential Gaussian simulation.

    Plain SGS (no secondary) on the `GaussianCoSimulator` engine with
    ``use_gpu=True``: every realization is one CUDA thread walking its own
    random path, so many realizations run in parallel. Results equal the CPU
    engine for the same seed; without a usable GPU (or if it runs out of
    memory) it warns and runs on the CPU.

    Differences from `sgsim`: neighbours are the `num_points` nearest
    informed locations within `radius` (no octant search), and the default
    is simple kriging of normal scores (``kriging_type='ordinary'`` is
    available).

    Parameters
    ----------
    prediction_grid : (n, 2) node coordinates.
    df : hard data with columns `xx`, `yy`, `zz`.
    vario : GSLIB-style variogram ``[azimuth, nugget, major, minor, sill, vtype]``
        (spherical, exponential or Gaussian on the GPU).
    num_points : int, default 16 (at most 48 on the GPU).
    radius, kriging_type, nsim, itrans, seed : see `GaussianCoSimulator`.
    use_gpu : bool or 'auto', default True
    **kwargs : ``mean``, ``zmin``, ``zmax``.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = GaussianCoSimulator(vario, num_points=num_points, radius=radius, itrans=itrans,
                              kriging_type=kriging_type, use_gpu=use_gpu, **kwargs)
    return sim.fit(prediction_grid, df, xx, yy, zz).simulate(nsim, seed)
