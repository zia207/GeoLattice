"""
Sequential indicator co-simulation (SIS with secondary data).

Sequential indicator simulation in which every node's indicator
probabilities come from *collocated indicator cokriging* with a secondary
(soft) variable under the Markov–Bayes model (Zhu & Journel, 1993; the
GSLIB ``sisim`` soft-data option). Works for categorical variables
(classes) and continuous ones (thresholds), and reduces to plain SIS when no
secondary information is given.

Soft indicators
---------------
An exhaustive secondary variable ``s(u)`` (seismic impedance, a
remote-sensing index, ...) is turned into soft indicators
``y_k(u) = P(I_k(u) = 1 | s(u))`` by a calibration table built from the
collocated hard / secondary pairs (`SoftCalibration`): the secondary is
binned by quantiles, and each bin holds the class frequencies of the hard
data in it, shrunk toward the global proportions. Soft probabilities from
elsewhere can be passed directly instead.

Markov–Bayes covariances
------------------------
For class ``k`` with indicator covariance ``C_I`` and calibration parameter

    B_k = E[y_k | I_k = 1] - E[y_k | I_k = 0]      (0 = useless, 1 = perfect)

the soft indicator has cross-covariance ``C_IY(h) = B_k C_I(h)`` and
variance ``C_Y(0) = |B_k| C_I(0)``. At each node the estimate is

    p_k = m_k + sum_a lambda_a (i_k(u_a) - m_k) + mu (y_k(u) - m_k)

from the simple cokriging system over the nearest informed primary
indicators (hard data and previously simulated nodes) plus the collocated
soft indicator. ``m_k`` are the global proportions (or ``P(Z <= t_k)``).

Probabilities are corrected for order relations (categorical: clipped and
renormalised; continuous: GSLIB ``ordrel``) and a value is drawn; for
continuous variables the value between thresholds uses `sisim`'s
``beyond`` interpolation and tails.

Neighbours are the `num_points` nearest informed locations within `radius`
(no octant constraint), searched in a precomputed nearest-candidate list.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._conditioning import sequential_neighbors
from .lusim import _cov_block
from .sgsim import _get_random_generator, _make_rotation_matrix, _validate_vario
from .sisim import _beyond_zval, _ordrel

__all__ = ["SoftCalibration", "IndicatorCoSimulator", "sicosim"]


# ------------------------------------------------------------ calibration


class SoftCalibration:
    """
    Calibration of a continuous secondary variable into soft indicators.

    Parameters
    ----------
    n_bins : int, default 10
        Number of quantile bins of the secondary variable.
    smoothing : float, default 1.0
        Pseudo-count pulling each bin's frequencies toward the global ones
        (stabilises bins with few hard data).

    Bin edges are secondary quantiles, chosen without regard to class
    boundaries, so a bin straddling a class boundary mixes classes and
    lowers ``B`` even for a perfectly separating secondary (by roughly
    ``1 / n_bins``). More bins reduce that, at the cost of fewer hard data
    per bin.

    Attributes
    ----------
    edges_ : ndarray (n_bins + 1,)
        Secondary bin edges (outer edges are +-inf).
    table_ : ndarray (n_bins, n_classes)
        Soft indicator value per bin and class.
    B_ : ndarray (n_classes,)
        Markov–Bayes calibration parameter per class, clipped to [0, 1].
    """

    def __init__(self, n_bins: int = 10, smoothing: float = 1.0):
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)

    def fit(self, indicators: np.ndarray, secondary: np.ndarray,
            prior: Optional[np.ndarray] = None) -> "SoftCalibration":
        """`indicators` (n, K) of 0/1 hard indicators and collocated `secondary` (n,)."""
        ind = np.asarray(indicators, dtype=float)
        s = np.asarray(secondary, dtype=float).ravel()
        if ind.ndim != 2 or len(ind) != len(s):
            raise ValueError("indicators must be (n, K) with one row per secondary value")
        ok = ~np.isnan(s)
        ind, s = ind[ok], s[ok]
        if len(s) < 2:
            raise ValueError("need at least two hard data with a secondary value")
        prior = ind.mean(axis=0) if prior is None else np.asarray(prior, dtype=float)
        q = np.unique(np.quantile(s, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        self.edges_ = np.concatenate([[-np.inf], q, [np.inf]])
        b = self._bin(s)
        nb = len(self.edges_) - 1
        counts = np.zeros((nb, ind.shape[1]))
        np.add.at(counts, b, ind)
        n = np.bincount(b, minlength=nb)[:, None]
        self.table_ = (counts + self.smoothing * prior) / (n + self.smoothing)
        y = self.table_[b]
        B = np.zeros(ind.shape[1])
        for k in range(ind.shape[1]):
            one, zero = ind[:, k] == 1, ind[:, k] == 0
            if one.any() and zero.any():
                B[k] = y[one, k].mean() - y[zero, k].mean()
        self.B_ = np.clip(B, 0.0, 1.0)
        return self

    def _bin(self, s: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges_, s, side="right") - 1, 0, len(self.edges_) - 2)

    def transform(self, secondary: np.ndarray) -> np.ndarray:
        """Soft indicators (m, K) for secondary values (m,); NaN gives NaN rows."""
        s = np.asarray(secondary, dtype=float).ravel()
        out = self.table_[self._bin(np.nan_to_num(s))]
        out[np.isnan(s)] = np.nan
        return out


# -------------------------------------------------------------- cokriging


def _cokrige(c_ii: np.ndarray, c_i0: np.ndarray, ind: np.ndarray, sill: float, mean: float,
             soft: Optional[float], B: float) -> float:
    """Simple (co)kriging estimate of one indicator; soft term per Markov–Bayes."""
    n = len(ind)
    use_soft = soft is not None and not np.isnan(soft) and B > 1e-6
    if n == 0:
        # only the collocated soft datum: its weight is B C(0) / (|B| C(0)) = 1
        return float(soft) if use_soft else mean
    if use_soft:
        K = np.empty((n + 1, n + 1))
        K[:n, :n] = c_ii
        K[:n, n] = K[n, :n] = B * c_i0
        K[n, n] = B * sill
        rhs = np.concatenate([c_i0, [B * sill]])
        resid = np.concatenate([ind - mean, [soft - mean]])
    else:
        K, rhs, resid = c_ii, c_i0, ind - mean
    try:
        w = np.linalg.solve(K + 1e-10 * sill * np.eye(len(K)), rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return float(mean + w @ resid)


# -------------------------------------------------------------- simulator


class IndicatorCoSimulator:
    """
    Sequential indicator (co-)simulator with Markov–Bayes soft data.

    Parameters
    ----------
    classes : sequence
        Category values (``variable='categorical'``) or ascending thresholds
        (``variable='continuous'``; indicators are ``I(Z <= t)``).
    vario : sequence or dict
        GSLIB-style indicator variogram ``[azimuth, nugget, major, minor,
        sill, vtype(, smoothness)]``, one for all classes or a dict
        ``{class: vario}``.
    variable : {'categorical', 'continuous'}, default 'categorical'
    num_points : int, default 16
        Maximum informed neighbours per node.
    radius : float, optional
        Search radius (default: unlimited).
    proportions : sequence, optional
        Global means ``m_k`` (class proportions, or ``P(Z <= t_k)``);
        default from the hard data (pass declustered values if sampling is
        preferential).
    n_bins, smoothing : calibration settings, see `SoftCalibration`.
    zmin, zmax, ltail, ltpar, middle, mpar, utail, utpar
        Continuous variables only: value limits and tail / middle models, as
        in `sisim`.
    use_gpu : bool or 'auto', default False
        Simulate all realizations in lockstep on a CUDA GPU, one thread per
        realization (`geolattice.geostatistics.sis_gpu`). Gives the same
        realizations as the CPU engine up to rounding; pays off with many
        realizations. Supports up to 48 neighbours and 16 classes, with
        spherical, exponential and Gaussian variograms.

    Attributes
    ----------
    calibration_ : SoftCalibration or None
    B_ : ndarray (n_classes,)
        Markov–Bayes parameter per class (zeros without secondary data).
    soft_ : ndarray (n_nodes, n_classes) or None
        Soft indicators at the grid nodes.
    proportions_ : ndarray (n_classes,)

    Examples
    --------
    >>> cosim = IndicatorCoSimulator([0, 1, 2], [0, 0, 40, 20, 0.25, "spherical"])
    >>> cosim.fit(grid, wells, "X", "Y", "facies", secondary=seismic, secondary_col="seis")
    >>> reals = cosim.simulate(nsim=20, seed=1)              # (20, len(grid))
    """

    def __init__(self, classes: Sequence, vario: Union[Sequence, Dict], variable: str = "categorical",
                 num_points: int = 16, radius: Optional[float] = None,
                 proportions: Optional[Sequence[float]] = None, n_bins: int = 10,
                 smoothing: float = 1.0, zmin: Optional[float] = None, zmax: Optional[float] = None,
                 ltail: str = "linear", ltpar: float = 1.0, middle: str = "linear",
                 mpar: float = 1.0, utail: str = "linear", utpar: float = 1.0,
                 use_gpu: Union[bool, str] = False):
        if variable not in ("categorical", "continuous"):
            raise ValueError("variable must be 'categorical' or 'continuous'")
        classes = np.asarray(classes, dtype=float)
        if len(classes) < 2:
            raise ValueError("need at least two classes / thresholds")
        if variable == "continuous" and not np.all(np.diff(classes) > 0):
            raise ValueError("thresholds must be strictly ascending")
        if variable == "categorical" and len(np.unique(classes)) != len(classes):
            raise ValueError("classes must be distinct")
        if isinstance(vario, dict):
            varios = [list(vario[c]) if c in vario else list(vario[float(c)]) for c in classes]
        else:
            varios = [list(vario)] * len(classes)
        for v in varios:
            _validate_vario(v)
        if num_points < 1:
            raise ValueError("num_points must be >= 1")
        self.classes = classes
        self.varios = varios
        self.variable = variable
        self.num_points = int(num_points)
        self.radius = np.inf if radius is None else float(radius)
        self.proportions = proportions
        self.n_bins, self.smoothing = n_bins, smoothing
        self.zmin, self.zmax = zmin, zmax
        self.tails = dict(ltail=ltail, ltpar=ltpar, middle=middle, mpar=mpar,
                          utail=utail, utpar=utpar)
        self._rot = [_make_rotation_matrix(v[0], v[2], v[3]) for v in varios]
        self.use_gpu = use_gpu

    def _indicators(self, values: np.ndarray) -> np.ndarray:
        if self.variable == "categorical":
            return (values[:, None] == self.classes[None, :]).astype(float)
        return (values[:, None] <= self.classes[None, :]).astype(float)

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: pd.DataFrame, xx: str, yy: str, zz: str,
            secondary: Optional[np.ndarray] = None, secondary_col: Optional[str] = None,
            soft: Optional[np.ndarray] = None, soft_data: Optional[np.ndarray] = None,
            B: Optional[Sequence[float]] = None) -> "IndicatorCoSimulator":
        """
        Prepare the simulation.

        Parameters
        ----------
        prediction_grid : (n, 2) node coordinates.
        df : hard data with columns `xx`, `yy`, `zz`.
        secondary : (n,) secondary variable at the grid nodes, calibrated
            against the hard data into soft indicators. Its value at the data
            is `df[secondary_col]`, or the nearest node's value if omitted.
        soft : (n, K) soft indicators at the nodes, used directly instead of
            `secondary`. `B` is then taken from `B`, or estimated from
            `soft_data` (soft indicators at the data, (n_data, K)).
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        xy = df[[xx, yy]].to_numpy(dtype=float)
        z = df[zz].to_numpy(dtype=float)
        if self.variable == "categorical":
            unknown = set(np.unique(z)) - set(self.classes)
            if unknown:
                raise ValueError(f"data values {sorted(unknown)} are not in classes")
        ind = self._indicators(z)
        K = len(self.classes)
        self.proportions_ = (ind.mean(axis=0) if self.proportions is None
                             else np.asarray(self.proportions, dtype=float))
        if len(self.proportions_) != K:
            raise ValueError("proportions must have one value per class")
        if self.variable == "continuous":
            self._zmin = float(z.min()) if self.zmin is None else float(self.zmin)
            self._zmax = float(z.max()) if self.zmax is None else float(self.zmax)

        self.calibration_ = None
        self.soft_ = None
        self.B_ = np.zeros(K)
        if secondary is not None and soft is not None:
            raise ValueError("pass either secondary or soft, not both")
        if secondary is not None:
            secondary = np.asarray(secondary, dtype=float).ravel()
            if len(secondary) != len(grid):
                raise ValueError("secondary must have one value per grid node")
            if secondary_col is not None:
                s_data = df[secondary_col].to_numpy(dtype=float)
            else:
                s_data = secondary[cKDTree(grid).query(xy)[1]]
            self.calibration_ = SoftCalibration(self.n_bins, self.smoothing).fit(
                ind, s_data, prior=self.proportions_)
            self.soft_ = self.calibration_.transform(secondary)
            self.B_ = self.calibration_.B_.copy()
        elif soft is not None:
            soft = np.asarray(soft, dtype=float)
            if soft.shape != (len(grid), K):
                raise ValueError(f"soft must have shape ({len(grid)}, {K})")
            self.soft_ = soft
            if B is not None:
                self.B_ = np.clip(np.asarray(B, dtype=float), 0.0, 1.0)
            elif soft_data is not None:
                sd = np.asarray(soft_data, dtype=float)
                self.B_ = np.array([
                    (sd[ind[:, k] == 1, k].mean() - sd[ind[:, k] == 0, k].mean())
                    if (ind[:, k] == 1).any() and (ind[:, k] == 0).any() else 0.0
                    for k in range(K)])
                self.B_ = np.clip(self.B_, 0.0, 1.0)
            else:
                raise ValueError("with soft probabilities pass B or soft_data to calibrate B")

        # locations: data first, then grid nodes; candidate neighbours per node
        self._grid = grid
        self._data_xy, self._data_z = xy, z
        self._loc, self._nbrs, self._coincident = sequential_neighbors(
            xy, grid, self.num_points, self.radius)
        self.n_nodes_ = len(grid)
        return self

    # ------------------------------------------------------------- simulate

    def _probabilities(self, node: int, informed: np.ndarray, values: np.ndarray) -> np.ndarray:
        target = self._grid[node][None, :]
        xy_n = self._loc[informed]
        ind = self._indicators(values)
        p = np.empty(len(self.classes))
        soft = self.soft_[node] if self.soft_ is not None else None
        for k, (v, R) in enumerate(zip(self.varios, self._rot)):
            sill = float(v[4])
            if len(informed):
                c_ii = _cov_block(xy_n, xy_n, v, R, same=True)
                c_i0 = _cov_block(xy_n, target, v, R)[:, 0]
            else:
                c_ii, c_i0 = np.empty((0, 0)), np.empty(0)
            p[k] = _cokrige(c_ii, c_i0, ind[:, k], sill, self.proportions_[k],
                            None if soft is None else soft[k], self.B_[k])
        return p

    def _simulate_one(self, rng: np.random.Generator) -> np.ndarray:
        n1 = len(self._data_z)
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
            p = self._probabilities(node, informed, vals[informed])
            if self.variable == "categorical":
                p = np.clip(p, 0.0, 1.0)
                s = p.sum()
                if s <= 0:
                    p = (self.soft_[node] if self.soft_ is not None else self.proportions_).copy()
                    s = p.sum()
                c = min(int(np.searchsorted(np.cumsum(p), rng.uniform() * s)), len(p) - 1)
                value = self.classes[c]
            else:
                ccdf = _ordrel(p)
                value = _beyond_zval(rng.uniform(), self.classes, ccdf, self._zmin, self._zmax,
                                     **self.tails)
            vals[n1 + node] = out[node] = value
        return out

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations at the grid nodes, shape ``(nsim, n_nodes)``."""
        if not hasattr(self, "_loc"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        from ..utils.backend import resolve_cupy_use_gpu

        if resolve_cupy_use_gpu(self.use_gpu):
            from ..gpu.sis_gpu import sis_simulate_gpu

            try:
                return sis_simulate_gpu(self, seeds)
            except Exception as exc:          # GPU busy / out of memory: same result on CPU
                if "memory" not in str(exc).lower() and "OutOfMemory" not in type(exc).__name__:
                    raise
                import warnings

                warnings.warn(f"GPU simulation failed ({type(exc).__name__}: {exc}); "
                              "running on the CPU, which gives the same realizations", stacklevel=2)
        return np.stack([self._simulate_one(_get_random_generator(int(s))) for s in seeds])


def sicosim(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    classes: Sequence,
    vario: Union[Sequence, Dict],
    secondary: Optional[np.ndarray] = None,
    secondary_col: Optional[str] = None,
    variable: str = "categorical",
    num_points: int = 16,
    radius: Optional[float] = None,
    nsim: int = 1,
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Sequential indicator co-simulation in one call.

    Parameters
    ----------
    prediction_grid : (n, 2) node coordinates.
    df : hard data.
    xx, yy, zz : column names for x, y and the variable.
    classes : categories, or ascending thresholds for a continuous variable.
    vario : indicator variogram(s), see `IndicatorCoSimulator`.
    secondary : (n,) secondary variable at the nodes, or None for plain SIS.
    secondary_col : column of `df` with the secondary value at the data.
    variable : {'categorical', 'continuous'}
    num_points, radius, nsim, seed : see `IndicatorCoSimulator`.
    **kwargs
        Other `IndicatorCoSimulator` options (``proportions``, ``n_bins``,
        ``smoothing``, ``zmin``, ``zmax``, tail models).

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = IndicatorCoSimulator(classes, vario, variable=variable, num_points=num_points,
                               radius=radius, **kwargs)
    sim.fit(prediction_grid, df, xx, yy, zz, secondary=secondary, secondary_col=secondary_col)
    return sim.simulate(nsim, seed)
