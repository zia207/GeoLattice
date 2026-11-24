"""
FFT spectral simulation by circulant embedding.

Exact Gaussian simulation on a regular lattice (Dietrich & Newsam, 1993;
Wood & Chan, 1994). The lattice covariance matrix is block-Toeplitz; it is
embedded in a block-circulant matrix on a periodic torus of size
``Mx x My`` (padded beyond the lattice and the correlation range), whose
eigenvalues are the 2-D FFT of its first row::

    lambda = FFT2(c),   c[i, j] = C(signed lag (i dx, j dy) on the torus)

If every eigenvalue is non-negative the embedding is a valid covariance, and

    Y = FFT2( sqrt(lambda / (Mx My)) * (e1 + i e2) ),   e1, e2 ~ N(0, I)

gives **two** independent exact realizations, ``Re(Y)`` and ``Im(Y)``,
restricted to the lattice. Cost is ``O(N log N)`` for ``N = Mx My``, so
grids of millions of nodes take seconds. This is the same spectral
factorization used by FFT moving-average simulation (FFT-MA, Le Ravalec
et al., 2000).

Padding starts at twice the lattice size and grows (x1.5 per step, up to
`max_cells`) until the embedding is non-negative definite; models whose
correlation reaches far beyond the grid (large-range Gaussian / Matérn) may
need a lot. If the limit is reached, negative eigenvalues are clipped to
zero and a warning is issued; the clipped share of the total spectrum is
reported as `FFTSimulator.negative_eigen_fraction_` (0.0 = exact).

Conditioning
------------
FFT needs a lattice, so conditioning data are moved to their nearest lattice
node (several data on one node are averaged), the lattice is extended to
cover all data, and realizations are conditioned by kriging exactly as in
`tbsim`::

    z_cs(x) = z*_K(x) + [y(x) - y*_K(x)]

A relocated datum is that node's value, nugget included (as GSLIB `sgsim`
does when assigning data to nodes), so realizations reproduce it exactly at
its node and `conditional_std_` is 0 there. The largest relocation distance
is `FFTSimulator.max_data_shift_`.

Variogram convention: `vario` is the GSLIB-style list used by `sgsim`,
`lusim` and `tbsim` (``[azimuth, nugget, major_range, minor_range, sill,
vtype(, smoothness)]``, practical-range covariance, ``sill`` = total C(0)),
with any of `sgsim`'s model types.
"""

import math
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import fft as sfft

from ._conditioning import DualKriging, condition_on_values
from ..gslib.nscore import nscore_back, nscore_forward
from .sgsim import _covar, _get_random_generator, _make_rotation_matrix, _validate_vario

__all__ = ["FFTSimulator", "fftsim", "fftsim_unconditional"]

_LATTICE_TOL = 1e-6


def _axis_spacing(values: np.ndarray) -> float:
    u = np.unique(values)
    if len(u) < 2:
        return 0.0
    return float(np.min(np.diff(u)))


def _infer_lattice(points: np.ndarray, spacing=None):
    """Origin and spacing of the regular lattice that `points` (n, 2) lie on."""
    if spacing is not None:
        dx, dy = (float(spacing), float(spacing)) if np.isscalar(spacing) else map(float, spacing)
    else:
        dx, dy = _axis_spacing(points[:, 0]), _axis_spacing(points[:, 1])
        if dx == 0.0 and dy == 0.0:
            dx = dy = 1.0
        dx = dx or dy
        dy = dy or dx
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    fx, fy = (points[:, 0] - x0) / dx, (points[:, 1] - y0) / dy
    if (np.abs(fx - np.round(fx)).max() > _LATTICE_TOL
            or np.abs(fy - np.round(fy)).max() > _LATTICE_TOL):
        raise ValueError(
            "prediction_grid must lie on a regular lattice (constant x and y spacing); "
            "pass spacing=... if the spacing cannot be inferred"
        )
    return x0, y0, dx, dy


def _torus_covariance(Mx: int, My: int, dx: float, dy: float, vario: Sequence,
                      rotation: np.ndarray) -> np.ndarray:
    """First row of the block-circulant embedding: C at signed torus lags."""
    sx = np.arange(Mx)
    sx = np.where(sx <= Mx // 2, sx, sx - Mx) * dx
    sy = np.arange(My)
    sy = np.where(sy <= My // 2, sy, sy - My) * dy
    hx, hy = np.meshgrid(sx, sy, indexing="ij")
    lag = np.column_stack([hx.ravel(), hy.ravel()]) @ rotation
    nugget, sill, vtype = vario[1], vario[4], vario[5]
    smoothness = vario[6] if str(vtype).lower() == "matern" else None
    c = _covar(np.sqrt((lag ** 2).sum(axis=1)), sill, nugget, vtype, smoothness)
    c = c.reshape(Mx, My)
    c[0, 0] = sill
    return c


class FFTSimulator:
    """
    FFT (circulant embedding) Gaussian simulator on a regular lattice.

    Parameters
    ----------
    vario : sequence
        ``[azimuth, nugget, major_range, minor_range, sill, vtype]`` (plus
        smoothness for ``'matern'``), same convention as `sgsim`.
    kriging_type : {'simple', 'ordinary'}, default 'simple'
        Kriging used to condition realizations. Ignored without data.
    mean : float, optional
        Simple-kriging / unconditional mean. Defaults to 0 when `itrans` is
        True or there is no data, otherwise the data mean.
    itrans : bool, default False
        Normal-score transform the data in, back-transform realizations out.
    zmin, zmax : float, optional
        Clip realizations to these bounds (GSLIB tail limits).
    max_cells : int, default 2**24
        Largest allowed embedding torus ``Mx * My`` (~16.8 M cells, ~270 MB
        of complex workspace).
    max_data : int, default 5000
        Upper limit on conditioning data (dense kriging system).
    spacing : float or (dx, dy), optional
        Lattice spacing, if it cannot be inferred from `prediction_grid`
        (e.g. a single row or column of nodes).

    Attributes
    ----------
    conditional_mean_ : ndarray (n_nodes,)
        Kriging estimate at the nodes (``mean`` everywhere if unconditional).
    conditional_std_ : ndarray (n_nodes,)
        Kriging standard deviation at the nodes, computed on first access.
    lattice_ : dict
        ``x0, y0, dx, dy, nx, ny`` of the simulated lattice.
    embedding_shape_ : tuple
        ``(Mx, My)`` torus used for the FFT.
    negative_eigen_fraction_ : float
        Share of the spectrum clipped to make the embedding valid (0.0 = exact).
    max_data_shift_ : float
        Largest distance a datum was moved to reach its lattice node.

    Examples
    --------
    >>> sim = FFTSimulator([30, 0, 80, 40, 1.0, "spherical"]).fit(grid, df, "X", "Y", "Z")
    >>> reals = sim.simulate(nsim=100, seed=7)       # (100, len(grid))
    """

    def __init__(self, vario: Sequence, kriging_type: str = "simple",
                 mean: Optional[float] = None, itrans: bool = False,
                 zmin: Optional[float] = None, zmax: Optional[float] = None,
                 max_cells: int = 2 ** 24, max_data: int = 5000, spacing=None):
        _validate_vario(vario)
        vtype = str(vario[5]).lower()
        if vtype == "matern" and (len(vario) < 7 or vario[6] is None or vario[6] <= 0):
            raise ValueError("matern needs a positive smoothness as vario[6]")
        if kriging_type not in ("simple", "ordinary"):
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")
        self.vario = list(vario)
        self.kriging_type = kriging_type
        self.mean = mean
        self.itrans = itrans
        self.zmin = zmin
        self.zmax = zmax
        self.max_cells = int(max_cells)
        self.max_data = max_data
        self.spacing = spacing

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z") -> "FFTSimulator":
        """Set up the lattice and circulant embedding, and krige the data if given."""
        vario = self.vario
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        self._grid = grid
        self._R = _make_rotation_matrix(vario[0], vario[2], vario[3])
        x0, y0, dx, dy = _infer_lattice(grid, self.spacing)

        has_data = df is not None and len(df) > 0
        self._has_data = has_data
        gi = np.rint((grid - [x0, y0]) / [dx, dy]).astype(np.int64)
        if has_data:
            data_xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            if self.itrans:
                z, self._v_sorted, self._ns_sorted = nscore_forward(z, tails="linear")
            di = np.rint((data_xy - [x0, y0]) / [dx, dy]).astype(np.int64)
            lo = np.minimum(gi.min(axis=0), di.min(axis=0))
            hi = np.maximum(gi.max(axis=0), di.max(axis=0))
        else:
            z = np.empty(0)
            lo, hi = gi.min(axis=0), gi.max(axis=0)
        nx, ny = (hi - lo + 1).tolist()
        self.lattice_ = {"x0": x0 + lo[0] * dx, "y0": y0 + lo[1] * dy,
                         "dx": dx, "dy": dy, "nx": nx, "ny": ny}
        self._grid_idx = gi - lo

        # data -> nearest lattice node, duplicates averaged
        if has_data:
            node_idx = di - lo
            keys, inv = np.unique(node_idx, axis=0, return_inverse=True)
            inv = inv.ravel()
            z_node = np.bincount(inv, weights=z) / np.bincount(inv)
            if len(keys) > self.max_data:
                raise ValueError(
                    f"{len(keys)} conditioning nodes exceeds max_data={self.max_data} "
                    "(dense kriging system); thin the data or raise max_data"
                )
            self._data_idx = keys
            self._data_node_xy = self._node_xy(keys)
            self.max_data_shift_ = float(np.sqrt(
                ((data_xy - self._data_node_xy[inv]) ** 2).sum(axis=1)).max())
            z = z_node
        else:
            self.max_data_shift_ = 0.0

        if self.mean is not None:
            m = float(self.mean)
        elif self.itrans or not has_data:
            m = 0.0
        else:
            m = float(z.mean())
        self.mean_ = m
        self.n_data_, self.n_nodes_ = len(z), len(grid)
        self._std = None

        self._embed(nx, ny, dx, dy)

        if has_data:
            self._dk = DualKriging(self._data_node_xy, vario, self._R, self.kriging_type,
                                   exact_at_data=True)
            rhs = z - m if self.kriging_type == "simple" else z
            self.conditional_mean_ = self._dk.predict(rhs[:, None], grid)[:, 0]
            if self.kriging_type == "simple":
                self.conditional_mean_ += m
        else:
            self.conditional_mean_ = np.full(self.n_nodes_, m)
        return self

    def _node_xy(self, idx: np.ndarray) -> np.ndarray:
        L = self.lattice_
        return np.column_stack([L["x0"] + idx[:, 0] * L["dx"], L["y0"] + idx[:, 1] * L["dy"]])

    def _embed(self, nx: int, ny: int, dx: float, dy: float) -> None:
        """Choose the torus size and store sqrt of the circulant eigenvalues."""
        def size(n):
            return 1 if n == 1 else sfft.next_fast_len(2 * n)

        Mx, My = size(nx), size(ny)
        while True:
            if Mx * My > self.max_cells:
                raise ValueError(
                    f"embedding {Mx} x {My} exceeds max_cells={self.max_cells}; "
                    "coarsen the grid or raise max_cells"
                )
            lam = sfft.fft2(_torus_covariance(Mx, My, dx, dy, self.vario, self._R),
                            workers=-1).real
            negative = lam < -1e-10 * lam.max()
            if not negative.any():
                break
            nMx = 1 if nx == 1 else sfft.next_fast_len(math.ceil(Mx * 1.5))
            nMy = 1 if ny == 1 else sfft.next_fast_len(math.ceil(My * 1.5))
            if nMx * nMy > self.max_cells:
                break
            Mx, My = nMx, nMy
        clipped = np.abs(lam[lam < 0]).sum()
        self.negative_eigen_fraction_ = float(clipped / np.abs(lam).sum())
        if negative.any():
            warnings.warn(
                f"circulant embedding is not non-negative definite at {Mx} x {My} "
                f"(max_cells={self.max_cells}); clipped {self.negative_eigen_fraction_:.2%} "
                "of the spectrum, so realizations are approximate. The correlation range "
                "is likely large relative to the grid; raise max_cells to pad further.",
                stacklevel=3,
            )
        self.embedding_shape_ = (Mx, My)
        self._sqrt_lam = np.sqrt(np.clip(lam, 0.0, None) / (Mx * My))

    @property
    def conditional_std_(self) -> np.ndarray:
        if not hasattr(self, "_grid"):
            raise RuntimeError("call fit() first")
        if self._std is None:
            if self._has_data:
                self._std = self._dk.std(self._grid)
            else:
                self._std = np.full(self.n_nodes_, np.sqrt(float(self.vario[4])))
        return self._std

    # ------------------------------------------------------------- simulate

    def simulate_lattice(self, nsim: int = 1,
                         seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """
        Unconditional zero-mean realizations on the full lattice, shape
        ``(nsim, nx, ny)`` (axis 1 = x, axis 2 = y). Each FFT yields two
        independent realizations; each pair gets its own sub-generator.
        """
        if not hasattr(self, "_sqrt_lam"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        nx, ny = self.lattice_["nx"], self.lattice_["ny"]
        rng = _get_random_generator(seed)
        child_seeds = rng.integers(0, 2**31 - 1, size=(nsim + 1) // 2)
        out = np.empty((nsim, nx, ny))
        shape = self._sqrt_lam.shape
        for k, s in enumerate(child_seeds):
            r = _get_random_generator(int(s))
            noise = r.standard_normal(shape) + 1j * r.standard_normal(shape)
            field = sfft.fft2(self._sqrt_lam * noise, workers=-1)[:nx, :ny]
            out[2 * k] = field.real
            if 2 * k + 1 < nsim:
                out[2 * k + 1] = field.imag
        return out

    @property
    def conditioning_xy_(self) -> np.ndarray:
        """Coordinates of the conditioning data, in kriging order."""
        if not getattr(self, "_has_data", False):
            return np.empty((0, 2))
        return self._dk.data_xy

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None,
                 data_values=None) -> np.ndarray:
        """
        Draw `nsim` realizations at the grid nodes, shape ``(nsim, n_nodes)``.

        `data_values` re-conditions on new values at the fitted data
        locations (`conditioning_xy_` order, in simulation units, i.e. normal
        scores when `itrans` is True) without refitting: shape ``(n_data,)``
        for all realizations or ``(n_data, nsim)`` for one column each.
        """
        fields = self.simulate_lattice(nsim, seed)
        gx, gy = self._grid_idx[:, 0], self._grid_idx[:, 1]
        Yg = fields[:, gx, gy].T                                            # (n_nodes, nsim)
        if self._has_data:
            Yd = fields[:, self._data_idx[:, 0], self._data_idx[:, 1]].T    # (n_data, nsim)
            if data_values is not None:
                sims = condition_on_values(self._dk, self.mean_, Yg, Yd, data_values, self._grid)
            else:
                sims = self.conditional_mean_[:, None] + Yg - self._dk.predict(Yd, self._grid)
        else:
            sims = self.mean_ + Yg
        sims = sims.T

        if self.itrans and self._has_data:
            sims = nscore_back(sims.ravel(), self._v_sorted, self._ns_sorted,
                               tails="linear").reshape(sims.shape)
        if self.zmin is not None or self.zmax is not None:
            sims = np.clip(sims, self.zmin, self.zmax)
        return sims


def fftsim(
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
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    max_cells: int = 2 ** 24,
    max_data: int = 5000,
    spacing=None,
) -> np.ndarray:
    """
    FFT spectral (circulant embedding) simulation on a regular lattice,
    conditioned by simple or ordinary kriging.

    Same calling convention as `sgsim` / `lusim` / `tbsim`.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        x, y coordinates of the nodes; they must lie on a regular lattice
        (any order, holes allowed).
    df : pandas.DataFrame or None
        Conditioning data, moved to their nearest lattice node; None gives
        unconditional realizations.
    xx, yy, zz : str
        Column names in `df` for x, y and the variable.
    vario : sequence
        GSLIB-style variogram, see `sgsim`.
    kriging_type : {'simple', 'ordinary'}, default 'simple'
    mean : float, optional
        Simple-kriging / unconditional mean; see `FFTSimulator`.
    nsim : int, default 1
    itrans : bool, default False
        Normal-score transform in, back-transform out.
    seed : int, numpy.random.Generator, or None
    zmin, zmax : float, optional
        Clip realizations to these bounds.
    max_cells : int, default 2**24
        Largest embedding torus.
    max_data : int, default 5000
    spacing : float or (dx, dy), optional
        Lattice spacing if it cannot be inferred.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = FFTSimulator(vario, kriging_type=kriging_type, mean=mean, itrans=itrans,
                       zmin=zmin, zmax=zmax, max_cells=max_cells, max_data=max_data,
                       spacing=spacing)
    return sim.fit(prediction_grid, df, xx, yy, zz).simulate(nsim, seed)


def fftsim_unconditional(
    prediction_grid: np.ndarray,
    vario: Sequence,
    nsim: int = 1,
    mean: float = 0.0,
    seed: Optional[Union[int, np.random.Generator]] = None,
    max_cells: int = 2 ** 24,
    spacing=None,
) -> np.ndarray:
    """
    Unconditional FFT realizations with mean `mean` and covariance `vario`
    at the lattice `prediction_grid`; shape ``(nsim, len(prediction_grid))``.
    """
    sim = FFTSimulator(vario, mean=mean, max_cells=max_cells, spacing=spacing)
    return sim.fit(prediction_grid).simulate(nsim, seed)
