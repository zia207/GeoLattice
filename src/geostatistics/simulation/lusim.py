"""
LU (Cholesky) decomposition simulation -- Python port of GSLIB's lusim.

Gaussian simulation by factorizing the full covariance matrix (Davis, 1987;
Deutsch & Journel, *GSLIB*, §V.2.4). Unlike sequential simulation
(`sgsim`), nothing is approximated: there is no search neighbourhood and no
random path, so every realization has *exactly* the target covariance and,
when conditioned, the exact simple / ordinary kriging conditional
distribution. The price is building and factorizing an ``N x N`` matrix,
``O(N^3)`` time and ``O(N^2)`` memory for ``N = n_data + n_nodes``, which
limits it to a few thousand locations. Once the factor is computed, each
extra realization only costs a matrix-vector product, so `LUSimulator`
keeps the factor around and `simulate` can be called repeatedly.

Conditioning
------------
Split locations into data (1) and grid nodes (2)::

    C = | C11  C12 |        L = | L11   0  |       C = L L^T
        | C21  C22 |            | L21  L22 |

* ``kriging_type='simple'`` (the GSLIB `lusim` algorithm): with known mean
  ``m`` and data residuals ``r = z1 - m``,

      y2 = m + L21 L11^-1 r + L22 w,        w ~ N(0, I)

  where ``L11 = chol(C11)``, ``L21 = C21 L11^-T`` and
  ``L22 = chol(C22 - L21 L21^T)``. ``m + L21 L11^-1 r`` is the simple
  kriging estimate and ``L22 L22^T`` the simple kriging covariance.
* ``kriging_type='ordinary'``: the mean is unknown, so condition by kriging
  (Journel & Huijbregts, 1978): simulate a zero-mean unconditional field
  ``u`` jointly at data and nodes from ``chol(C)``, then

      y2 = z2*_OK + (u2 - u2*_OK)

  where ``*_OK`` is ordinary kriging from the data locations with the same
  weights ``Lambda`` for ``z`` and ``u``.

Variogram convention: `vario` is the same GSLIB-style list as `sgsim`
(``[azimuth, nugget, major_range, minor_range, sill, vtype(, smoothness)]``,
practical-range covariance, ``sill`` = total C(0)), and covariances come from
`sgsim`'s own functions so the two simulators agree. A nugget acts on the
diagonal only, so with ``nugget > 0`` a node that coincides with a datum is
not forced to that datum's value (GSLIB behaviour).

Near-singular covariance matrices (Gaussian models, duplicate or coincident
locations, nodes on top of data with zero nugget) are factorized after
adding the smallest diagonal jitter that succeeds; the value used is stored
as `LUSimulator.jitter_`.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from ..gslib.nscore import nscore_back, nscore_forward
from .sgsim import (
    _covar,
    _get_random_generator,
    _make_rotation_matrix,
    _pairwise_effective_lag,
    _validate_vario,
)

__all__ = ["LUSimulator", "lusim", "lusim_unconditional"]

_JITTERS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)


def _cov_block(A: np.ndarray, B: np.ndarray, vario: Sequence, rotation: np.ndarray,
               same: bool = False) -> np.ndarray:
    """Covariance between every row of `A` and every row of `B`."""
    nugget, sill, vtype = vario[1], vario[4], vario[5]
    smoothness = vario[6] if vtype.lower() == "matern" else None
    lag = _pairwise_effective_lag(A @ rotation, B @ rotation, np)
    cov = _covar(lag, sill, nugget, vtype, smoothness)
    if same:
        np.fill_diagonal(cov, sill)
    return cov


def _robust_cholesky(C: np.ndarray, scale: float):
    """Lower Cholesky factor of `C`, adding diagonal jitter only if needed."""
    for jitter in _JITTERS:
        try:
            M = C if jitter == 0.0 else C + (jitter * scale) * np.eye(len(C))
            return cholesky(M, lower=True, check_finite=False), jitter * scale
        except LinAlgError:
            continue
    raise LinAlgError(
        "covariance matrix is not positive definite even with jitter "
        f"{_JITTERS[-1]:g} x sill; check the variogram and for duplicate locations"
    )


class LUSimulator:
    """
    LU (Cholesky) decomposition Gaussian simulator.

    Factorizes the covariance once in `fit`, after which `simulate` draws any
    number of realizations cheaply.

    Parameters
    ----------
    vario : sequence
        ``[azimuth, nugget, major_range, minor_range, sill, vtype]`` (plus
        smoothness for ``'matern'``), same convention as `sgsim`.
    kriging_type : {'simple', 'ordinary'}, default 'simple'
        How realizations are conditioned to data. Ignored without data.
    mean : float, optional
        Mean for simple kriging and unconditional simulation. Defaults to 0
        when `itrans` is True or there is no data, otherwise the data mean
        (pass a declustered mean if sampling is preferential).
    itrans : bool, default False
        Normal-score transform the data before simulating and back-transform
        realizations (as `sgsim`'s `itrans`).
    max_nodes : int, default 10000
        Refuse problems with more than this many data + grid locations; the
        dense covariance needs ``8 * N^2`` bytes (800 MB at 10 000).
    zmin, zmax : float, optional
        Clip realizations to these bounds (GSLIB tail limits). Useful with
        `itrans`, whose linear tail extrapolation is otherwise unbounded,
        e.g. ``zmin=0`` for concentrations.

    Attributes
    ----------
    conditional_mean_ : ndarray (n_nodes,)
        Kriging estimate at the nodes (the mean field, in Gaussian units when
        `itrans` is True). Equals `mean` everywhere for unconditional runs.
    conditional_std_ : ndarray (n_nodes,)
        Kriging standard deviation at the nodes (sill**0.5 when unconditional).
    jitter_ : float
        Diagonal jitter that was needed to factorize (0.0 if none).

    Examples
    --------
    >>> sim = LUSimulator([0, 0, 40, 40, 1.0, "spherical"]).fit(grid, df, "X", "Y", "Z")
    >>> reals = sim.simulate(nsim=100, seed=7)       # (100, len(grid))
    """

    def __init__(self, vario: Sequence, kriging_type: str = "simple",
                 mean: Optional[float] = None, itrans: bool = False,
                 max_nodes: int = 10000, zmin: Optional[float] = None,
                 zmax: Optional[float] = None):
        _validate_vario(vario)
        if kriging_type not in ("simple", "ordinary"):
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")
        self.vario = list(vario)
        self.kriging_type = kriging_type
        self.mean = mean
        self.itrans = itrans
        self.max_nodes = max_nodes
        self.zmin = zmin
        self.zmax = zmax

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z") -> "LUSimulator":
        """
        Build and factorize the covariance for `prediction_grid` (n, 2),
        conditioned on `df[[xx, yy, zz]]` when `df` is given.
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        vario = self.vario
        sill = float(vario[4])
        R = _make_rotation_matrix(vario[0], vario[2], vario[3])

        has_data = df is not None and len(df) > 0
        if has_data:
            data_xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            if self.itrans:
                z, self._v_sorted, self._ns_sorted = nscore_forward(z, tails="linear")
        else:
            data_xy = np.empty((0, 2))
            z = np.empty(0)
        n1, n2 = len(z), len(grid)
        if n1 + n2 > self.max_nodes:
            raise ValueError(
                f"{n1 + n2} data + grid locations exceeds max_nodes={self.max_nodes} "
                f"(dense covariance would need {8 * (n1 + n2) ** 2 / 1e9:.1f} GB); "
                "use sgsim for large grids or raise max_nodes"
            )

        if self.mean is not None:
            m = float(self.mean)
        elif self.itrans or not has_data:
            m = 0.0
        else:
            m = float(z.mean())
        self.mean_ = m
        self.n_data_, self.n_nodes_ = n1, n2
        self._has_data = has_data
        self.jitter_ = 0.0

        C22 = _cov_block(grid, grid, vario, R, same=True)

        if not has_data:
            self._mode = "unconditional"
            self._L, self.jitter_ = _robust_cholesky(C22, sill)
            self.conditional_mean_ = np.full(n2, m)
            self.conditional_std_ = np.full(n2, np.sqrt(sill))
            return self

        C11 = _cov_block(data_xy, data_xy, vario, R, same=True)
        C12 = _cov_block(data_xy, grid, vario, R)

        if self.kriging_type == "simple":
            self._mode = "simple"
            L11, j1 = _robust_cholesky(C11, sill)
            A = solve_triangular(L11, C12, lower=True, check_finite=False)   # L21^T
            S = C22 - A.T @ A                                                # SK covariance
            S = 0.5 * (S + S.T)
            L22, j2 = _robust_cholesky(S, sill)
            self._L = L22
            self.jitter_ = max(j1, j2)
            self.conditional_mean_ = m + A.T @ solve_triangular(
                L11, z - m, lower=True, check_finite=False
            )
            self.conditional_std_ = np.sqrt(np.clip(np.diag(S), 0.0, None))
        else:
            self._mode = "ordinary"
            K = np.zeros((n1 + 1, n1 + 1))
            K[:n1, :n1] = C11
            K[:n1, n1] = K[n1, :n1] = 1.0
            rhs = np.vstack([C12, np.ones((1, n2))])
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            self._weights = sol[:n1]                                         # Lambda (n1, n2)
            mu = sol[n1]
            self.conditional_mean_ = self._weights.T @ z
            var = sill - np.einsum("ij,ij->j", self._weights, C12) - mu
            self.conditional_std_ = np.sqrt(np.clip(var, 0.0, None))
            C = np.block([[C11, C12], [C12.T, C22]])
            self._L, self.jitter_ = _robust_cholesky(C, sill)
        return self

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """
        Draw `nsim` realizations, shape ``(nsim, n_nodes)``, in grid order.
        Each realization gets its own sub-generator from `seed`, as in `sgsim`.
        """
        if not hasattr(self, "_L"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        child_seeds = rng.integers(0, 2**31 - 1, size=nsim)
        dim = self._L.shape[0]
        W = np.column_stack([
            _get_random_generator(int(s)).standard_normal(dim) for s in child_seeds
        ])
        U = self._L @ W                                                      # (dim, nsim)

        if self._mode == "ordinary":
            n1 = self.n_data_
            resid = U[n1:] - self._weights.T @ U[:n1]
            sims = (self.conditional_mean_[:, None] + resid).T
        else:
            sims = (self.conditional_mean_[:, None] + U).T

        if self.itrans and self._has_data:
            sims = nscore_back(sims.ravel(), self._v_sorted, self._ns_sorted,
                               tails="linear").reshape(sims.shape)
        if self.zmin is not None or self.zmax is not None:
            sims = np.clip(sims, self.zmin, self.zmax)
        return sims


def lusim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    zz: str,
    vario: Sequence,
    kriging_type: str = "simple",
    mean: Optional[float] = None,
    nsim: int = 1,
    itrans: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
    max_nodes: int = 10000,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> np.ndarray:
    """
    LU (Cholesky) decomposition simulation.

    Exact Gaussian simulation for small problems; same calling convention as
    `sgsim` minus the neighbourhood arguments (every datum and node is used).

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        x, y coordinates of the nodes to simulate.
    df : pandas.DataFrame or None
        Conditioning data; None gives unconditional realizations.
    xx, yy, zz : str
        Column names in `df` for x, y and the variable.
    vario : sequence
        GSLIB-style variogram, see `sgsim`.
    kriging_type : {'simple', 'ordinary'}, default 'simple'
        Conditioning by simple kriging (GSLIB `lusim`) or ordinary kriging.
    mean : float, optional
        Simple-kriging / unconditional mean; see `LUSimulator`.
    nsim : int, default 1
        Number of realizations.
    itrans : bool, default False
        Normal-score transform in, back-transform out.
    seed : int, numpy.random.Generator, or None
        Reproducibility seed.
    max_nodes : int, default 10000
        Upper limit on data + nodes (dense ``N x N`` covariance).
    zmin, zmax : float, optional
        Clip realizations to these bounds.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = LUSimulator(vario, kriging_type=kriging_type, mean=mean,
                      itrans=itrans, max_nodes=max_nodes, zmin=zmin, zmax=zmax)
    return sim.fit(prediction_grid, df, xx, yy, zz).simulate(nsim, seed)


def lusim_unconditional(
    prediction_grid: np.ndarray,
    vario: Sequence,
    nsim: int = 1,
    mean: float = 0.0,
    seed: Optional[Union[int, np.random.Generator]] = None,
    max_nodes: int = 10000,
) -> np.ndarray:
    """
    Unconditional Gaussian realizations with mean `mean` and covariance
    `vario` at `prediction_grid`; shape ``(nsim, len(prediction_grid))``.
    """
    sim = LUSimulator(vario, mean=mean, max_nodes=max_nodes)
    return sim.fit(prediction_grid).simulate(nsim, seed)
