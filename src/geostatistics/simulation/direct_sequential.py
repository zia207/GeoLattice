"""
Direct sequential simulation (DSS) in original units.

Sequential Gaussian simulation reproduces the variogram of the *normal
scores*; after back-transformation the original-unit variogram generally
differs from the one that was modelled. Direct sequential simulation
(Journel, 1994; Soares, 2001) instead krigs the variable itself: each node's
simple kriging estimate ``z*`` and variance ``sigma2_sk`` come from the
original-unit variogram, and a value is drawn from a local distribution
built from the global histogram so that the histogram is reproduced too.

Two published ways to build that local distribution:

* ``method='lookup'`` (Oz, Deutsch, Tran & Xie, 2003), default. Tabulate, by
  Gauss–Hermite quadrature, the original-unit mean and variance of every
  back-transformed Gaussian window ``F^-1(G(N(y_m, y_s^2)))``; at each node
  pick the window whose mean and variance best match ``(z*, sigma2_sk)``
  and draw from it. Local means and variances, and so the original-unit
  variogram, are reproduced; the histogram approximately.
* ``method='soares'`` (Soares, 2001). Map the estimate into
  Gaussian space through the global distribution,
  ``y* = G^-1(F(z*))``, draw ``y ~ N(y*, sigma2_sk / sigma2_Z)`` and return
  ``F^-1(G(y))``. Every value is a quantile of the global distribution, so
  values come from the global distribution, but the back-transformed
  window's mean exceeds ``z*`` whenever the distribution is skewed, and the
  sequential feedback compounds it: the simulated mean and variance drift
  upward (about +10 % on the mean at a coefficient of variation of 0.4,
  far more for strongly skewed data). Use it only for near-symmetric
  variables; a warning is issued otherwise.

With an exhaustive secondary variable the estimate is collocated simple
cokriging under Markov Model 1 on the standardized secondary (direct
sequential co-simulation, Soares, 2001), as in `sgcosim`.

The global distribution is the (optionally declustering-weighted) data
histogram, interpolated linearly between data values and to `zmin` /
`zmax` in the tails. The variogram is in original units: its sill is the
variance of the variable. Neighbours are the `num_points` nearest informed
locations within `radius`.
"""

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri

from ._conditioning import sequential_neighbors
from .lusim import _cov_block
from .sgcosim import _collocated_cokrige
from .sgsim import _get_random_generator, _make_rotation_matrix, _validate_vario

__all__ = ["GlobalDistribution", "DirectSequentialSimulator", "dssim_direct"]


# --------------------------------------------------------- global distribution


class GlobalDistribution:
    """
    Global cdf ``F`` of a variable with the transforms ``z -> y = G^-1(F(z))``
    and back, built from (optionally weighted) data.

    Cumulative probabilities use the midpoint rule (``F`` at the i-th sorted
    value is the cumulative weight up to it minus half its own weight), and
    are extended linearly to `zmin` / `zmax` at probability 0 / 1.
    """

    def __init__(self, values, weights=None, zmin: Optional[float] = None,
                 zmax: Optional[float] = None):
        z = np.asarray(values, dtype=float).ravel()
        w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float).ravel()
        if len(z) < 2:
            raise ValueError("need at least two values")
        if len(w) != len(z) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per value, not all zero")
        order = np.argsort(z)
        z, w = z[order], w[order] / w.sum()
        p = np.cumsum(w) - 0.5 * w
        zmin = z[0] if zmin is None else float(zmin)
        zmax = z[-1] if zmax is None else float(zmax)
        if zmin > z[0] or zmax < z[-1]:
            raise ValueError("zmin / zmax must bracket the data")
        self.z_ = np.concatenate([[zmin], z, [zmax]])
        self.p_ = np.concatenate([[0.0], p, [1.0]])
        self.mean_ = float((w * z).sum())
        self.var_ = float((w * (z - self.mean_) ** 2).sum())

    def cdf(self, z):
        return np.interp(z, self.z_, self.p_)

    def quantile(self, p):
        return np.interp(p, self.p_, self.z_)

    def to_gaussian(self, z):
        return ndtri(np.clip(self.cdf(z), 1e-12, 1 - 1e-12))

    def from_gaussian(self, y):
        return self.quantile(ndtr(y))


# ------------------------------------------------------------- lookup table


class _WindowTable:
    """Original-unit mean / sd of back-transformed Gaussian windows N(y_m, y_s^2)."""

    def __init__(self, dist: GlobalDistribution, n_mean: int = 161, n_sd: int = 80,
                 n_quad: int = 40):
        x, wq = np.polynomial.hermite_e.hermegauss(n_quad)
        wq = wq / wq.sum()
        ym = np.linspace(-4.0, 4.0, n_mean)
        ys = np.linspace(0.0, 2.0, n_sd + 1)[1:]
        M, S = np.meshgrid(ym, ys, indexing="ij")
        vals = dist.from_gaussian(M[..., None] + S[..., None] * x)
        mu = (vals * wq).sum(axis=-1)
        sd = np.sqrt(np.clip((vals ** 2 * wq).sum(axis=-1) - mu ** 2, 0.0, None))
        self.ym, self.ys = M.ravel(), S.ravel()
        scale = np.sqrt(dist.var_) or 1.0
        self._scale = scale
        self._tree = cKDTree(np.column_stack([mu.ravel(), sd.ravel()]) / scale)
        self.mu, self.sd = mu.ravel(), sd.ravel()

    def window(self, mean: float, sd: float):
        _, i = self._tree.query([mean / self._scale, sd / self._scale])
        return self.ym[i], self.ys[i]


# --------------------------------------------------------------- simulator


class DirectSequentialSimulator:
    """
    Direct sequential (co-)simulator in original units.

    Parameters
    ----------
    vario : sequence
        GSLIB-style variogram of the variable in original units
        ``[azimuth, nugget, major, minor, sill, vtype(, smoothness)]``; the
        sill is its variance.
    method : {'lookup', 'soares'}, default 'lookup'
        Local distribution, see the module docstring.
    num_points : int, default 16
        Maximum informed neighbours per node.
    radius : float, optional
        Search radius (default: unlimited).
    mean : float, optional
        Simple-kriging mean; default the (weighted) data mean.
    weights : array-like, optional
        Declustering weights for the global histogram and mean.
    zmin, zmax : float, optional
        Tail limits of the global distribution (default data min / max).
    rho : float, optional
        Correlation with the standardized secondary for co-simulation;
        estimated from the data when None.

    Attributes
    ----------
    distribution_ : GlobalDistribution
    rho_ : float
        Correlation used (0 without secondary).

    Notes
    -----
    ``use_gpu`` (bool or 'auto', default False) simulates all realizations
    in lockstep on a CUDA GPU, one thread per realization
    (`geolattice.geostatistics.dss_gpu`), with the same realizations as the CPU
    engine up to rounding; it pays off with many realizations. Supports up
    to 48 neighbours and spherical, exponential and Gaussian variograms.

    Examples
    --------
    >>> dss = DirectSequentialSimulator([0, 0, 40, 20, 2.5, "spherical"])
    >>> reals = dss.fit(grid, wells, "X", "Y", "grade").simulate(nsim=50, seed=1)
    """

    def __init__(self, vario: Sequence, method: str = "lookup", num_points: int = 16,
                 radius: Optional[float] = None, mean: Optional[float] = None,
                 weights=None, zmin: Optional[float] = None, zmax: Optional[float] = None,
                 rho: Optional[float] = None, use_gpu: Union[bool, str] = False):
        _validate_vario(vario)
        if method not in ("lookup", "soares"):
            raise ValueError("method must be 'lookup' or 'soares'")
        if num_points < 1:
            raise ValueError("num_points must be >= 1")
        if rho is not None and not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        self.vario = list(vario)
        self.method = method
        self.num_points = int(num_points)
        self.radius = np.inf if radius is None else float(radius)
        self.mean = mean
        self.weights = weights
        self.zmin, self.zmax = zmin, zmax
        self.rho = rho
        self._R = _make_rotation_matrix(vario[0], vario[2], vario[3])
        self.use_gpu = use_gpu

    def fit(self, prediction_grid: np.ndarray, df: pd.DataFrame, xx: str, yy: str, zz: str,
            secondary: Optional[np.ndarray] = None,
            secondary_col: Optional[str] = None) -> "DirectSequentialSimulator":
        """
        Prepare the simulation. `secondary` (n_nodes,) enables co-simulation;
        its value at the data is `df[secondary_col]` or the nearest node's.
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        xy = df[[xx, yy]].to_numpy(dtype=float)
        z = df[zz].to_numpy(dtype=float)
        self.distribution_ = GlobalDistribution(z, self.weights, self.zmin, self.zmax)
        self.mean_ = self.distribution_.mean_ if self.mean is None else float(self.mean)
        self._table = _WindowTable(self.distribution_) if self.method == "lookup" else None
        if self.method == "soares":
            skew = float(np.mean((z - z.mean()) ** 3) / max(z.std(), 1e-300) ** 3)
            if abs(skew) > 1.0:          # sample skewness is noisy for small correlated samples
                warnings.warn(
                    f"method='soares' is biased for skewed data (skewness {skew:.2f}): "
                    "simulated mean and variance drift away from the data; "
                    "method='lookup' reproduces them", stacklevel=2)

        self.rho_ = 0.0
        self._y = None
        if secondary is not None:
            sec = np.asarray(secondary, dtype=float).ravel()
            if len(sec) != len(grid):
                raise ValueError("secondary must have one value per grid node")
            mu, sd = sec.mean(), sec.std()
            if sd == 0:
                raise ValueError("secondary is constant")
            self._y = (sec - mu) / sd
            if self.rho is not None:
                self.rho_ = float(self.rho)
            else:
                s_data = (df[secondary_col].to_numpy(dtype=float) if secondary_col is not None
                          else sec[cKDTree(grid).query(xy)[1]])
                self.rho_ = float(np.corrcoef(z, s_data)[0, 1])
        elif self.rho not in (None, 0.0):
            raise ValueError("rho was given but no secondary variable")

        self._grid = grid
        self._data_z = z
        self._loc, self._nbrs, self._coincident = sequential_neighbors(
            xy, grid, self.num_points, self.radius)
        self.n_nodes_ = len(grid)
        return self

    def _draw(self, est: float, var: float, rng: np.random.Generator) -> float:
        dist = self.distribution_
        if self.method == "soares":
            y0 = float(dist.to_gaussian(est))
            sd = np.sqrt(var / dist.var_) if dist.var_ > 0 else 0.0
            return float(dist.from_gaussian(y0 + sd * rng.standard_normal()))
        noise = rng.standard_normal()       # always one draw per node (GPU engine relies on it)
        if var <= 0:
            return float(np.clip(est, dist.z_[0], dist.z_[-1]))
        ym, ys = self._table.window(est, np.sqrt(var))
        return float(dist.from_gaussian(ym + ys * noise))

    def _simulate_one(self, rng: np.random.Generator) -> np.ndarray:
        n1 = len(self._data_z)
        c00 = float(self.vario[4])
        m = self.mean_
        vals = np.full(len(self._loc), np.nan)
        vals[:n1] = self._data_z
        out = np.empty(self.n_nodes_)
        for node in rng.permutation(self.n_nodes_):
            if self._coincident[node] >= 0:
                vals[n1 + node] = out[node] = self._data_z[self._coincident[node]]
                continue
            cand = self._nbrs[node]
            cand = cand[cand >= 0]
            informed = cand[~np.isnan(vals[cand])][: self.num_points]
            if len(informed):
                xy_n = self._loc[informed]
                c_zz = _cov_block(xy_n, xy_n, self.vario, self._R, same=True)
                c_z0 = _cov_block(xy_n, self._grid[node][None, :], self.vario, self._R)[:, 0]
            else:
                c_zz, c_z0 = np.empty((0, 0)), np.empty(0)
            est, var = _collocated_cokrige(c_zz, c_z0, vals[informed] - m, c00,
                                           None if self._y is None else self._y[node], self.rho_)
            vals[n1 + node] = out[node] = self._draw(m + est, var, rng)
        return out

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations at the grid nodes, shape ``(nsim, n_nodes)``, original units."""
        if not hasattr(self, "_loc"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        from ..utils.backend import resolve_cupy_use_gpu

        if resolve_cupy_use_gpu(self.use_gpu):
            from ..gpu.dss_gpu import dss_simulate_gpu

            try:
                return dss_simulate_gpu(self, seeds)
            except Exception as exc:          # GPU busy / out of memory: same result on CPU
                if "memory" not in str(exc).lower() and "OutOfMemory" not in type(exc).__name__:
                    raise
                warnings.warn(f"GPU simulation failed ({type(exc).__name__}: {exc}); "
                              "running on the CPU, which gives the same realizations", stacklevel=2)
        return np.stack([self._simulate_one(_get_random_generator(int(s))) for s in seeds])


def dssim_direct(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    vario: Sequence,
    method: str = "lookup",
    num_points: int = 16,
    radius: Optional[float] = None,
    nsim: int = 1,
    seed: Optional[Union[int, np.random.Generator]] = None,
    secondary: Optional[np.ndarray] = None,
    secondary_col: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Direct sequential simulation (or co-simulation with `secondary`) in one call.

    Named ``dssim_direct`` because ``dssim`` is the Direct Sampling MPS
    simulator.

    Parameters
    ----------
    prediction_grid : (n, 2) node coordinates.
    df : hard data with columns `xx`, `yy`, `zz`.
    vario : variogram in original units.
    method : {'lookup', 'soares'}
    num_points, radius, nsim, seed : see `DirectSequentialSimulator`.
    secondary, secondary_col : optional exhaustive secondary for co-simulation.
    **kwargs : ``mean``, ``weights``, ``zmin``, ``zmax``, ``rho``.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = DirectSequentialSimulator(vario, method=method, num_points=num_points,
                                    radius=radius, **kwargs)
    sim.fit(prediction_grid, df, xx, yy, zz, secondary=secondary, secondary_col=secondary_col)
    return sim.simulate(nsim, seed)
