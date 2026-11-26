"""
GPU engine for turning bands simulation.

Turning bands has no sequential dependency: a realization is the sum, over
``nlines`` lines, of 1-D processes evaluated at every location, so it is
data-parallel even for a single realization. This engine keeps the random
part on the host -- each realization's line directions (random rotation of
the Fibonacci set), spectral frequencies and phases, or sawtooth offsets and
signs, and nugget noise are drawn with exactly the calls `TBSimulator`'s CPU
engine makes -- and evaluates the field on the GPU with CuPy in chunks of
points x lines:

* spectral lines: ``sqrt(2) * sum cos(p . k_l + phi_l)``;
* spherical (sawtooth dilution): the per-line cell range is reduced on the
  GPU first (the host needs it to draw the signs, as on the CPU), then
  ``sqrt(12) * sum eps * (t - centre)``.

Conditioning by kriging also runs on the GPU: the dual kriging coefficients
come from the CPU engine's factorised system, and the expensive part -- the
target-to-data covariance times the coefficients, for every node and
realization -- is computed in chunks on the GPU.

Results equal the CPU engine up to floating-point rounding. (For the
spherical model a projection that lands within rounding of a cell boundary
can fall in the neighbouring cell on one device and not the other; this
affects a vanishing fraction of point-line pairs.)
"""

import numpy as np
from scipy.linalg import lu_solve
from scipy.spatial.transform import Rotation

from ..simulation.sgsim import _covar, _get_random_generator

__all__ = ["tb_simulate_gpu"]

_POINT_CHUNK = 20000
_LINE_CHUNK = 256
_TARGET_CHUNK = 20000


def _field_gpu(pts, vtype, smoothness, nlines, rng):
    """Unit-variance turning bands field at device points `pts` (m, 3)."""
    import cupy as cp

    from ..simulation.tbsim import _fibonacci_directions, _spectral_frequencies

    dirs_h = _fibonacci_directions(nlines) @ Rotation.random(random_state=rng).as_matrix().T
    dirs = cp.asarray(dirs_h)
    m = pts.shape[0]
    out = cp.zeros(m)
    if vtype == "spherical":
        off_h = rng.uniform(0.0, 1.0, nlines)
        off = cp.asarray(off_h)
        lo = cp.full(nlines, cp.inf)
        hi = cp.full(nlines, -cp.inf)
        for p0 in range(0, m, _POINT_CHUNK):
            j = cp.floor(pts[p0:p0 + _POINT_CHUNK] @ dirs.T - off)
            lo = cp.minimum(lo, j.min(axis=0))
            hi = cp.maximum(hi, j.max(axis=0))
        lo_h, hi_h = cp.asnumpy(lo), cp.asnumpy(hi)
        signs = cp.asarray(rng.choice((-1.0, 1.0), size=(nlines, int((hi_h - lo_h).max()) + 1)))
        for p0 in range(0, m, _POINT_CHUNK):
            p = pts[p0:p0 + _POINT_CHUNK]
            for l0 in range(0, nlines, _LINE_CHUNK):
                sl = slice(l0, l0 + _LINE_CHUNK)
                t = p @ dirs[sl].T
                j = cp.floor(t - off[sl])
                cols = cp.arange(sl.start, min(sl.stop, nlines))
                eps = signs[cols[None, :], (j - lo[sl]).astype(cp.int64)]
                out[p0:p0 + len(p)] += np.sqrt(12.0) * (eps * (t - (j + off[sl] + 0.5))).sum(axis=1)
    else:
        freqs = _spectral_frequencies(vtype, smoothness, nlines, rng)
        phases = cp.asarray(rng.uniform(0.0, 2.0 * np.pi, nlines))
        k = dirs * cp.asarray(freqs)[:, None]
        for p0 in range(0, m, _POINT_CHUNK):
            p = pts[p0:p0 + _POINT_CHUNK]
            for l0 in range(0, nlines, _LINE_CHUNK):
                arg = p @ k[l0:l0 + _LINE_CHUNK].T + phases[l0:l0 + _LINE_CHUNK]
                out[p0:p0 + len(p)] += np.sqrt(2.0) * cp.cos(arg).sum(axis=1)
    return out / np.sqrt(nlines)


def _krige_gpu(dk, values, targets_d, data_rot_d, vario, rot_d):
    """GPU version of `DualKriging.predict`: krige `values` (n1, k) onto targets."""
    import cupy as cp

    n1 = len(dk.data_xy)
    rhs = values if dk.kriging_type == "simple" else np.vstack([values, np.zeros((1, values.shape[1]))])
    coef = cp.asarray(lu_solve(dk._K, rhs, check_finite=False))
    nugget, sill, vtype = float(vario[1]), float(vario[4]), vario[5]
    smoothness = vario[6] if str(vtype).lower() == "matern" else None
    out = cp.empty((targets_d.shape[0], values.shape[1]))
    for p0 in range(0, targets_d.shape[0], _TARGET_CHUNK):
        t = targets_d[p0:p0 + _TARGET_CHUNK] @ rot_d
        lag = cp.sqrt(((t[:, None, :] - data_rot_d[None, :, :]) ** 2).sum(axis=2))
        C = _covar(lag, sill, nugget, vtype, smoothness)
        out[p0:p0 + len(t)] = C @ coef[:n1]
        if dk.kriging_type == "ordinary":
            out[p0:p0 + len(t)] += coef[n1]
    return out


def tb_simulate_gpu(sim, seeds, data_values=None) -> np.ndarray:
    """
    Turning bands realizations of a fitted `TBSimulator` on the GPU,
    ``(nsim, n_nodes)`` in simulation units (before back-transform / clipping).
    """
    import cupy as cp

    nsim = len(seeds)
    nugget, sill = float(sim.vario[1]), float(sim.vario[4])
    n1 = sim.n_data_
    pts2 = np.vstack([sim._data_xy, sim._grid]) @ sim._R
    pts = cp.asarray(np.column_stack([pts2, np.zeros(len(pts2))]))
    m = pts.shape[0]

    Y = cp.empty((m, nsim))
    for r, s in enumerate(seeds):
        g = _get_random_generator(int(s))
        y = np.sqrt(sill - nugget) * _field_gpu(pts, sim._vtype, sim._smooth, sim.nlines, g)
        if nugget > 0:
            y += np.sqrt(nugget) * cp.asarray(g.standard_normal(m))
        Y[:, r] = y

    if not sim._has_data:
        return cp.asnumpy(sim.mean_ + Y[n1:]).T

    dk = sim._dk
    grid_d = cp.asarray(sim._grid)
    rot_d = cp.asarray(sim._R)
    data_rot_d = cp.asarray(dk.data_xy @ sim._R)
    Yd = cp.asnumpy(Y[:n1])
    if data_values is not None:
        dv = np.asarray(data_values, dtype=float)
        if dv.ndim == 1 and dv.shape[0] == Yd.shape[0]:
            dv = np.broadcast_to(dv[:, None], Yd.shape)
        if dv.shape != Yd.shape:
            raise ValueError(f"data_values must have shape ({Yd.shape[0]},) or {Yd.shape}")
        base = sim.mean_ if dk.kriging_type == "simple" else 0.0
        sims = base + Y[n1:] + _krige_gpu(dk, dv - base - Yd, grid_d, data_rot_d, sim.vario, rot_d)
    else:
        sims = cp.asarray(sim.conditional_mean_)[:, None] + Y[n1:] - _krige_gpu(
            dk, Yd, grid_d, data_rot_d, sim.vario, rot_d)
    return cp.asnumpy(sims).T
