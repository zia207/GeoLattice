"""
Turning bands simulation (TBSIM) -- Gaussian random fields from 1-D line processes.

Matheron's turning bands method (Matheron, 1973; Journel, 1974) builds a
multidimensional Gaussian field as a sum of independent 1-D processes, each
running along a line with direction ``u_i``:

    Y(x) = (1 / sqrt(L)) * sum_i X_i(<x, u_i>)

In 3-D an isotropic covariance ``C3`` is reproduced exactly when each line
process has covariance ``C1(h) = d/dh [h C3(h)]``. A 2-D field is simulated
as a plane section of a 3-D field (Emery & Lantuéjoul, 2006, TBSIM), which
avoids the awkward 2-D turning bands integral transform. Line directions are
a quasi-uniform (Fibonacci) set on the sphere, given a fresh random rotation
per realization, so banding artefacts average out and the ensemble
covariance is exact for any number of lines.

Line processes
--------------
* ``'spherical'``: dilution of a random-sign sawtooth. Each line is cut into
  cells of length one practical range with a uniform random offset, and
  ``X(t) = sqrt(12) * eps_j * (t - centre_j)`` in cell ``j`` (``eps_j = ±1``).
  Its covariance is exactly ``1 - 3h + 2h^3``, the 1-D counterpart of the 3-D
  spherical model (Lantuéjoul, 2002).
* ``'exponential'``, ``'gaussian'``, ``'matern'``: spectral lines
  ``X(t) = sqrt(2) cos(w t + phi)`` with ``w`` drawn from the model's radial
  3-D spectral density (Matérn / exponential: ``k = g / (a sqrt(W))``,
  ``g ~ N(0, I3)``, ``W ~ chi2(2 nu)``; Gaussian: ``k ~ N(0, 2/a^2 I3)``).

The nugget is added as independent white noise at every location.

Conditioning
------------
Realizations are conditioned by kriging (Journel & Huijbregts, 1978): the
unconditional field ``y`` is simulated jointly at the data and the grid, and

    z_cs(x) = z*_K(x) + [y(x) - y*_K(x)]

where ``*_K`` is simple or ordinary kriging from all data (global dual
kriging, evaluated over the grid in chunks). Unlike `lusim`, the grid can be
very large: cost is ``O(n_nodes * n_lines)`` per realization plus
``O(n_nodes * n_data)`` for conditioning. The data count is limited by the
dense ``n_data x n_data`` kriging system (`max_data`).

Variogram convention: `vario` is the GSLIB-style list used by `sgsim` and
`lusim` (``[azimuth, nugget, major_range, minor_range, sill, vtype(,
smoothness)]``, practical-range covariance, ``sill`` = total C(0)).
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ._conditioning import DualKriging, condition_on_values
from ..gslib.nscore import nscore_back, nscore_forward
from .sgsim import _get_random_generator, _make_rotation_matrix, _validate_vario

__all__ = ["TBSimulator", "tbsim", "tbsim_unconditional"]

_POINT_CHUNK = 20000
_LINE_CHUNK = 64


def _fibonacci_directions(n: int) -> np.ndarray:
    """`n` quasi-uniform unit vectors on the upper hemisphere (lines are unsigned)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n                                   # cos(polar) uniform in (0, 1)
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i            # golden angle
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _matern_scale(smoothness: float) -> float:
    """Matérn ``a`` such that `sgsim._covar`'s practical-range Matérn is
    ``2^(1-nu)/Gamma(nu) (h/a)^nu K_nu(h/a)``."""
    s = 0.45246434 * np.exp(-0.70449189 * smoothness) + 1.7863836
    return 1.0 / (2.0 * s * np.sqrt(smoothness))


def _spectral_frequencies(vtype: str, smoothness: Optional[float], n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Radial frequencies |k| drawn from the 3-D spectral density, in effective-lag units."""
    g = np.linalg.norm(rng.standard_normal((n, 3)), axis=1)      # chi(3)
    if vtype == "gaussian":
        # exp(-3 h^2) = exp(-h^2 / a^2), a = 1/sqrt(3); spectrum N(0, 2/a^2)
        return np.sqrt(6.0) * g
    nu, a = (0.5, 1.0 / 3.0) if vtype == "exponential" else (smoothness, _matern_scale(smoothness))
    W = rng.chisquare(2.0 * nu, size=n)
    return g / (a * np.sqrt(W))


def _unconditional_standard(points: np.ndarray, vtype: str, smoothness: Optional[float],
                            nlines: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance turning bands field at `points` (m, 3) in effective-lag space."""
    dirs = _fibonacci_directions(nlines) @ Rotation.random(random_state=rng).as_matrix().T
    out = np.zeros(len(points))
    if vtype == "spherical":
        offsets = rng.uniform(0.0, 1.0, nlines)
        # Cells j = floor(t - offset) of unit length on each line; first find
        # the cell range each line spans, then draw one sign per (line, cell).
        lo = np.full(nlines, np.inf)
        hi = np.full(nlines, -np.inf)
        for p0 in range(0, len(points), _POINT_CHUNK):
            j = np.floor(points[p0:p0 + _POINT_CHUNK] @ dirs.T - offsets)
            lo = np.minimum(lo, j.min(axis=0))
            hi = np.maximum(hi, j.max(axis=0))
        signs = rng.choice((-1.0, 1.0), size=(nlines, int((hi - lo).max()) + 1))
        for p0 in range(0, len(points), _POINT_CHUNK):
            pts = points[p0:p0 + _POINT_CHUNK]
            for l0 in range(0, nlines, _LINE_CHUNK):
                sl = slice(l0, l0 + _LINE_CHUNK)
                t = pts @ dirs[sl].T
                j = np.floor(t - offsets[sl])
                eps = np.take_along_axis(signs[sl].T, (j - lo[sl]).astype(int), axis=0)
                out[p0:p0 + len(pts)] += np.sqrt(12.0) * (
                    eps * (t - (j + offsets[sl] + 0.5))).sum(axis=1)
    else:
        freqs = _spectral_frequencies(vtype, smoothness, nlines, rng)
        phases = rng.uniform(0.0, 2.0 * np.pi, nlines)
        k = dirs * freqs[:, None]                                  # (L, 3) wave vectors
        for p0 in range(0, len(points), _POINT_CHUNK):
            pts = points[p0:p0 + _POINT_CHUNK]
            for l0 in range(0, nlines, _LINE_CHUNK):
                arg = pts @ k[l0:l0 + _LINE_CHUNK].T + phases[l0:l0 + _LINE_CHUNK]
                out[p0:p0 + len(pts)] += np.sqrt(2.0) * np.cos(arg).sum(axis=1)
    return out / np.sqrt(nlines)


class TBSimulator:
    """
    Turning bands Gaussian simulator, unconditional or conditioned by kriging.

    Parameters
    ----------
    vario : sequence
        ``[azimuth, nugget, major_range, minor_range, sill, vtype]`` (plus
        smoothness for ``'matern'``), same convention as `sgsim` / `lusim`.
        ``vtype`` is ``'spherical'``, ``'exponential'``, ``'gaussian'`` or
        ``'matern'``.
    kriging_type : {'simple', 'ordinary'}, default 'simple'
        Kriging used to condition realizations. Ignored without data.
    mean : float, optional
        Simple-kriging / unconditional mean. Defaults to 0 when `itrans` is
        True or there is no data, otherwise the data mean.
    nlines : int, default 1000
        Number of turning bands lines per realization. Fewer lines are
        faster but each realization is less Gaussian and shows more banding;
        a few hundred is a practical minimum.
    itrans : bool, default False
        Normal-score transform the data in, back-transform realizations out.
    zmin, zmax : float, optional
        Clip realizations to these bounds (GSLIB tail limits).
    max_data : int, default 5000
        Upper limit on conditioning data (dense kriging system).
    use_gpu : bool or 'auto', default False
        Evaluate the line processes and the kriging conditioning on a CUDA
        GPU (`geolattice.geostatistics.tbsim_gpu`); random numbers stay on the
        host, so realizations equal the CPU engine's up to rounding. Turning
        bands is data-parallel, so this helps even for one realization.

    Attributes
    ----------
    conditional_mean_ : ndarray (n_nodes,)
        Kriging estimate at the nodes (``mean`` everywhere if unconditional).
    conditional_std_ : ndarray (n_nodes,)
        Kriging standard deviation at the nodes, computed on first access.

    Examples
    --------
    >>> sim = TBSimulator([0, 0, 40, 20, 1.0, "exponential"]).fit(grid, df, "X", "Y", "Z")
    >>> reals = sim.simulate(nsim=50, seed=7)        # (50, len(grid))
    """

    def __init__(self, vario: Sequence, kriging_type: str = "simple",
                 mean: Optional[float] = None, nlines: int = 1000, itrans: bool = False,
                 zmin: Optional[float] = None, zmax: Optional[float] = None,
                 max_data: int = 5000, use_gpu: Union[bool, str] = False):
        _validate_vario(vario)
        vtype = str(vario[5]).lower()
        if vtype not in ("spherical", "exponential", "gaussian", "matern"):
            raise ValueError("vtype must be 'spherical', 'exponential', 'gaussian', or 'matern'")
        if vtype == "matern" and (len(vario) < 7 or vario[6] is None or vario[6] <= 0):
            raise ValueError("matern needs a positive smoothness as vario[6]")
        if kriging_type not in ("simple", "ordinary"):
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")
        if nlines < 1:
            raise ValueError("nlines must be >= 1")
        self.vario = list(vario)
        self.kriging_type = kriging_type
        self.mean = mean
        self.nlines = int(nlines)
        self.itrans = itrans
        self.zmin = zmin
        self.zmax = zmax
        self.max_data = max_data
        self.use_gpu = use_gpu

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z") -> "TBSimulator":
        """Store the grid and, when `df` is given, factorize the kriging system."""
        vario = self.vario
        self._grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        self._R = _make_rotation_matrix(vario[0], vario[2], vario[3])
        self._vtype = str(vario[5]).lower()
        self._smooth = float(vario[6]) if self._vtype == "matern" else None

        has_data = df is not None and len(df) > 0
        if has_data:
            self._data_xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            if self.itrans:
                z, self._v_sorted, self._ns_sorted = nscore_forward(z, tails="linear")
            if len(z) > self.max_data:
                raise ValueError(
                    f"{len(z)} conditioning data exceeds max_data={self.max_data} "
                    "(dense kriging system); thin the data or raise max_data"
                )
        else:
            self._data_xy = np.empty((0, 2))
            z = np.empty(0)
        self._has_data = has_data

        if self.mean is not None:
            m = float(self.mean)
        elif self.itrans or not has_data:
            m = 0.0
        else:
            m = float(z.mean())
        self.mean_ = m
        self.n_data_, self.n_nodes_ = len(z), len(self._grid)
        self._std = None

        if has_data:
            self._dk = DualKriging(self._data_xy, vario, self._R, self.kriging_type)
            rhs = z - m if self.kriging_type == "simple" else z
            self.conditional_mean_ = self._dk.predict(rhs[:, None], self._grid)[:, 0]
            if self.kriging_type == "simple":
                self.conditional_mean_ += m
        else:
            self.conditional_mean_ = np.full(self.n_nodes_, m)
        return self

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
        Draw `nsim` realizations, shape ``(nsim, n_nodes)``, in grid order.
        Each realization gets its own sub-generator from `seed`.

        `data_values` re-conditions on new values at the fitted data
        locations (`conditioning_xy_` order, in simulation units, i.e. normal
        scores when `itrans` is True) without refitting: shape ``(n_data,)``
        for all realizations or ``(n_data, nsim)`` for one column each.
        """
        if not hasattr(self, "_grid"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        child_seeds = rng.integers(0, 2**31 - 1, size=nsim)

        from ..utils.backend import resolve_cupy_use_gpu

        if resolve_cupy_use_gpu(self.use_gpu):
            from ..gpu.tbsim_gpu import tb_simulate_gpu

            try:
                return self._postprocess(tb_simulate_gpu(self, child_seeds, data_values))
            except Exception as exc:          # GPU busy / out of memory: same result on CPU
                if "memory" not in str(exc).lower() and "OutOfMemory" not in type(exc).__name__:
                    raise
                import warnings

                warnings.warn(f"GPU simulation failed ({type(exc).__name__}: {exc}); "
                              "running on the CPU, which gives the same realizations", stacklevel=2)

        nugget, sill = float(self.vario[1]), float(self.vario[4])
        pts2 = np.vstack([self._data_xy, self._grid]) @ self._R
        pts = np.column_stack([pts2, np.zeros(len(pts2))])
        n1 = self.n_data_

        Y = np.empty((len(pts), nsim))
        for r, s in enumerate(child_seeds):
            r_rng = _get_random_generator(int(s))
            y = np.sqrt(sill - nugget) * _unconditional_standard(
                pts, self._vtype, self._smooth, self.nlines, r_rng)
            if nugget > 0:
                y += np.sqrt(nugget) * r_rng.standard_normal(len(pts))
            Y[:, r] = y

        if self._has_data and data_values is not None:
            sims = condition_on_values(self._dk, self.mean_, Y[n1:], Y[:n1], data_values, self._grid)
        elif self._has_data:
            sims = self.conditional_mean_[:, None] + Y[n1:] - self._dk.predict(Y[:n1], self._grid)
        else:
            sims = self.mean_ + Y[n1:]
        return self._postprocess(sims.T)

    def _postprocess(self, sims: np.ndarray) -> np.ndarray:
        if self.itrans and self._has_data:
            sims = nscore_back(sims.ravel(), self._v_sorted, self._ns_sorted,
                               tails="linear").reshape(sims.shape)
        if self.zmin is not None or self.zmax is not None:
            sims = np.clip(sims, self.zmin, self.zmax)
        return sims


def tbsim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    zz: str,
    vario: Sequence,
    kriging_type: str = "simple",
    mean: Optional[float] = None,
    nsim: int = 1,
    nlines: int = 1000,
    itrans: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    max_data: int = 5000,
) -> np.ndarray:
    """
    Turning bands simulation, conditioned by simple or ordinary kriging.

    Same calling convention as `sgsim` / `lusim`; scales to large grids.

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
    mean : float, optional
        Simple-kriging / unconditional mean; see `TBSimulator`.
    nsim : int, default 1
    nlines : int, default 1000
        Turning bands lines per realization.
    itrans : bool, default False
        Normal-score transform in, back-transform out.
    seed : int, numpy.random.Generator, or None
    zmin, zmax : float, optional
        Clip realizations to these bounds.
    max_data : int, default 5000

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sim = TBSimulator(vario, kriging_type=kriging_type, mean=mean, nlines=nlines,
                      itrans=itrans, zmin=zmin, zmax=zmax, max_data=max_data)
    return sim.fit(prediction_grid, df, xx, yy, zz).simulate(nsim, seed)


def tbsim_unconditional(
    prediction_grid: np.ndarray,
    vario: Sequence,
    nsim: int = 1,
    mean: float = 0.0,
    nlines: int = 1000,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Unconditional turning bands realizations with mean `mean` and covariance
    `vario` at `prediction_grid`; shape ``(nsim, len(prediction_grid))``.
    """
    sim = TBSimulator(vario, mean=mean, nlines=nlines)
    return sim.fit(prediction_grid).simulate(nsim, seed)
