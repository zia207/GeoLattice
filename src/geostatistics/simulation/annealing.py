"""
Simulated annealing simulation on a regular lattice.

Optimization-based simulation (Deutsch & Journel, GSLIB ``sasim`` /
``anneal``; Deutsch, 1992): start from a random image with the target
histogram, then repeatedly swap the values of two non-data nodes. Swaps keep
the histogram exactly; each swap changes an objective function ``O`` that
measures how far the image is from the target statistics, and is accepted
with the Metropolis rule

    P(accept) = 1                      if O_new <= O_old
                exp((O_old - O_new)/t) otherwise,

while the temperature ``t`` is lowered geometrically (``t *= reduction``
after `k_accept` accepted or `k_max` attempted swaps). Annealing stops when
``O`` falls below ``tol`` times its initial value, after `max_swaps`, or
when `n_stop` successive temperatures accept nothing.

Objective components
--------------------
* ``'variogram'`` (continuous): ``sum_h w_h (gamma_img(h) - gamma(h))^2 /
  gamma(h)^2`` over lag vectors ``k * d`` (``k = 1..n_lags``) along each
  direction ``d`` (in cells), with ``gamma`` from `vario` and ``w_h ∝ 1/k``.
* ``'indicator'`` (categorical): the same for each class's indicator
  variogram, from `indicator_varios` ``{class: vario}``.
* ``'correlation'``: ``(rho_img - rho)^2`` with an exhaustive secondary
  variable and target correlation `target_correlation` (default: the
  correlation of the data with the secondary).

As in ``sasim`` each component is divided by its mean absolute change over
random swaps, so the user `component_weights` are relative importances.
Every swap updates the objective incrementally from the lag pairs that
involve the two swapped nodes only (NaN-padded flat lattice), and the
correlation in O(1).

Hard data are moved to their nearest node and never swapped. The target
histogram is `target_values` (e.g. declustered data or a reference
distribution) or the data; the free nodes start from its stratified
quantiles in random order, so the image reproduces it exactly.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .fftsim import _infer_lattice
from .lusim import _cov_block
from .sgsim import _get_random_generator, _make_rotation_matrix, _validate_vario

__all__ = ["SimulatedAnnealing", "sasim"]

_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class SimulatedAnnealing:
    """
    Simulated annealing simulator.

    Parameters
    ----------
    vario : sequence, optional
        Target variogram (original units) for a continuous variable.
    indicator_varios : dict, optional
        ``{class: vario}`` target indicator variograms for a categorical
        variable (give this instead of `vario`).
    n_lags : int, default 10
        Lags per direction.
    directions : sequence of (dx, dy), default x, y and both diagonals
        Lag directions in cells.
    target_values : array-like, optional
        Values defining the target histogram (default: the hard data).
    target_correlation : float, optional
        Target correlation with the secondary variable.
    component_weights : dict, optional
        Relative importance, e.g. ``{"variogram": 1, "correlation": 2}``.
    t0 : float, default 1.0
        Initial temperature, in units of the mean objective increase of a
        random swap (1 accepts a typical bad swap with probability ~0.37).
    reduction : float, default 0.1
        Temperature reduction factor.
    k_accept, k_max : int, optional
        Accepted / attempted swaps per temperature (default 5 and 50 times the
        number of free nodes).
    max_swaps : int, optional
        Swap budget (default 500 times the number of free nodes).
    tol : float, default 1e-3
        Stop when the objective falls below ``tol`` times its initial value.
    n_stop : int, default 3
        Stop after this many temperatures without an accepted swap.
    spacing : float or (dx, dy), optional
        Lattice spacing when it cannot be inferred from the grid.

    Attributes
    ----------
    history_ : list of ndarray
        Objective after each temperature, per realization.
    objective_ : ndarray (nsim,)
        Final objective relative to the initial one.
    n_swaps_ : ndarray (nsim,)
        Swaps attempted per realization.

    Examples
    --------
    >>> sa = SimulatedAnnealing(vario=[0, 0, 30, 15, 1.0, "spherical"], n_lags=12)
    >>> reals = sa.fit(grid, wells, "X", "Y", "Z").simulate(nsim=5, seed=1)
    """

    def __init__(self, vario: Optional[Sequence] = None,
                 indicator_varios: Optional[Dict] = None, n_lags: int = 10,
                 directions: Sequence[Tuple[int, int]] = _DIRECTIONS,
                 target_values=None, target_correlation: Optional[float] = None,
                 component_weights: Optional[Dict[str, float]] = None,
                 t0: float = 1.0, reduction: float = 0.1, k_accept: Optional[int] = None,
                 k_max: Optional[int] = None, max_swaps: Optional[int] = None,
                 tol: float = 1e-3, n_stop: int = 3, spacing=None):
        if (vario is None) == (indicator_varios is None):
            raise ValueError("give exactly one of vario (continuous) or indicator_varios (categorical)")
        if vario is not None:
            _validate_vario(vario)
        else:
            for v in indicator_varios.values():
                _validate_vario(v)
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if not 0.0 < reduction < 1.0:
            raise ValueError("reduction must be in (0, 1)")
        if t0 <= 0 or tol < 0:
            raise ValueError("t0 must be > 0 and tol >= 0")
        if target_correlation is not None and not -1.0 <= target_correlation <= 1.0:
            raise ValueError("target_correlation must be in [-1, 1]")
        self.vario = None if vario is None else list(vario)
        self.indicator_varios = None if indicator_varios is None else {
            float(k): list(v) for k, v in indicator_varios.items()}
        self.kind = "continuous" if vario is not None else "categorical"
        self.n_lags = int(n_lags)
        self.directions = [tuple(int(c) for c in d) for d in directions]
        self.target_values = target_values
        self.target_correlation = target_correlation
        self.component_weights = dict(component_weights or {})
        self.t0, self.reduction = float(t0), float(reduction)
        self.k_accept, self.k_max, self.max_swaps = k_accept, k_max, max_swaps
        self.tol, self.n_stop = float(tol), int(n_stop)
        self.spacing = spacing

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z",
            secondary: Optional[np.ndarray] = None,
            secondary_col: Optional[str] = None) -> "SimulatedAnnealing":
        """Set up the lattice, hard data, target statistics and secondary."""
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        x0, y0, dx, dy = _infer_lattice(grid, self.spacing)
        gi = np.rint((grid - [x0, y0]) / [dx, dy]).astype(np.int64)
        has_data = df is not None and len(df) > 0
        if has_data:
            xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            di = np.rint((xy - [x0, y0]) / [dx, dy]).astype(np.int64)
            lo = np.minimum(gi.min(axis=0), di.min(axis=0))
            hi = np.maximum(gi.max(axis=0), di.max(axis=0))
        else:
            lo, hi = gi.min(axis=0), gi.max(axis=0)
        nx, ny = (hi - lo + 1).tolist()
        self.lattice_ = {"x0": x0 + lo[0] * dx, "y0": y0 + lo[1] * dy,
                         "dx": dx, "dy": dy, "nx": nx, "ny": ny}
        self._grid_idx = gi - lo
        self.n_nodes_ = len(grid)

        hard = np.full((nx, ny), np.nan)
        if has_data:
            idx = di - lo
            keys, inv = np.unique(idx, axis=0, return_inverse=True)
            inv = inv.ravel()
            for k, (i, j) in enumerate(keys):
                v = z[inv == k]
                if self.kind == "categorical":
                    u, c = np.unique(v, return_counts=True)
                    hard[i, j] = u[c.argmax()]
                else:
                    hard[i, j] = v.mean()
        self._hard = hard

        target = (np.asarray(self.target_values, dtype=float).ravel()
                  if self.target_values is not None else (z if has_data else None))
        if target is None or len(target) == 0:
            raise ValueError("need hard data or target_values for the target histogram")
        self._target_values = target
        if self.kind == "categorical":
            self.classes_ = np.array(sorted(self.indicator_varios))
            bad = set(np.unique(target)) - set(self.classes_)
            if bad or (has_data and set(np.unique(z)) - set(self.classes_)):
                raise ValueError("data / target values must all be classes of indicator_varios")

        # lag vectors and target variograms
        lags = [(k * a, k * b) for (a, b) in self.directions for k in range(1, self.n_lags + 1)]
        self._lags = np.array(lags, dtype=np.int64)
        kk = np.array([k for _ in self.directions for k in range(1, self.n_lags + 1)], dtype=float)
        h_xy = self._lags * np.array([dx, dy])

        def model_gamma(v):
            R = _make_rotation_matrix(v[0], v[2], v[3])
            c = _cov_block(np.zeros((1, 2)), h_xy, v, R)[0]
            return float(v[4]) - c

        if self.kind == "continuous":
            self._targets = [model_gamma(self.vario)[None, :]]                 # (1, L)
        else:
            self._targets = [np.stack([model_gamma(self.indicator_varios[c]) for c in self.classes_])]
        self._lag_w = (1.0 / kk) / (1.0 / kk).sum()

        # secondary
        self._y = None
        if secondary is not None:
            sec = np.asarray(secondary, dtype=float).ravel()
            if len(sec) != len(grid):
                raise ValueError("secondary must have one value per grid node")
            ylat = np.full((nx, ny), np.nan)
            ylat[self._grid_idx[:, 0], self._grid_idx[:, 1]] = sec
            if np.isnan(ylat).any():
                # lattice nodes outside the requested grid take the nearest node's value
                miss = np.argwhere(np.isnan(ylat))
                ylat[miss[:, 0], miss[:, 1]] = sec[cKDTree(self._grid_idx).query(miss)[1]]
            self._y = ylat
            if self.target_correlation is not None:
                self.rho_target_ = float(self.target_correlation)
            elif has_data and len(z) >= 3:
                s_data = (df[secondary_col].to_numpy(dtype=float) if secondary_col is not None
                          else sec[cKDTree(grid).query(xy)[1]])
                self.rho_target_ = float(np.corrcoef(z, s_data)[0, 1])
            else:
                raise ValueError("give target_correlation (or >= 3 hard data) with a secondary")
        elif self.target_correlation is not None:
            raise ValueError("target_correlation needs a secondary variable")
        return self

    # ------------------------------------------------------- objective state

    def _setup_state(self, img: np.ndarray):
        nx, ny = img.shape
        r = int(np.abs(self._lags).max())
        self._pad = r
        P = np.full((nx + 2 * r, ny + 2 * r), np.nan)
        P[r:r + nx, r:r + ny] = img
        self._ncol = P.shape[1]
        self._lag_flat = self._lags[:, 0] * self._ncol + self._lags[:, 1]
        flat = P.ravel()
        # pair counts per lag (fixed: padding never changes)
        inside = np.zeros_like(P, dtype=bool)
        inside[r:r + nx, r:r + ny] = True
        f_in = inside.ravel()
        idx = np.flatnonzero(f_in)
        self._n_pairs = np.array([f_in[idx + L].sum() for L in self._lag_flat], dtype=float)
        if (self._n_pairs == 0).any():
            raise ValueError("some lags are longer than the grid; reduce n_lags")
        return flat, idx

    def _indicators(self, values):
        """(n_comp, ...) arrays whose squared differences give the variogram terms."""
        if self.kind == "continuous":
            return values[None, ...]
        return (values[None, ...] == self.classes_.reshape((-1,) + (1,) * np.ndim(values))).astype(float)

    def _gamma_sums(self, flat, idx):
        """Sum over pairs of squared differences per component and lag, (n_comp, L)."""
        base = self._indicators(flat[idx])
        out = np.empty((base.shape[0], len(self._lag_flat)))
        for l, L in enumerate(self._lag_flat):
            nb = flat[idx + L]
            ok = ~np.isnan(nb)
            out[:, l] = ((base[:, ok] - self._indicators(nb[ok])) ** 2).sum(axis=1)
        return out

    def _local_sums(self, flat, nodes):
        """Squared-difference sums of the pairs touching `nodes`, per component and lag."""
        L = self._lag_flat
        nbr = np.concatenate([nodes[:, None] + L[None, :], nodes[:, None] - L[None, :]], axis=0)
        centre = np.concatenate([nodes, nodes])[:, None]
        cv, nv = flat[np.broadcast_to(centre, nbr.shape)], flat[nbr]
        ok = ~np.isnan(nv)
        d = (self._indicators(cv) - self._indicators(np.where(ok, nv, 0.0))) ** 2
        d = np.where(ok[None, ...], d, 0.0)
        return d.sum(axis=1)                                            # (n_comp, L)

    def _variogram_objective(self, sums):
        g = sums / (2.0 * self._n_pairs)
        tgt = self._targets[0]
        return float((self._lag_w * (g - tgt) ** 2 / np.maximum(tgt, 1e-12) ** 2).sum())

    # ------------------------------------------------------------- simulate

    def _simulate_one(self, rng: np.random.Generator):
        nx, ny = self.lattice_["nx"], self.lattice_["ny"]
        img = self._hard.copy()
        free2d = np.argwhere(np.isnan(img))
        # stratified quantiles of the target histogram in random positions: the
        # image reproduces the target histogram exactly, and swaps keep it
        tv = np.sort(self._target_values)
        q = ((np.arange(len(free2d)) + 0.5) / max(len(free2d), 1) * len(tv)).astype(np.int64)
        img[free2d[:, 0], free2d[:, 1]] = rng.permutation(tv[np.minimum(q, len(tv) - 1)])
        flat, idx = self._setup_state(img)
        r = self._pad
        free = (free2d[:, 0] + r) * self._ncol + (free2d[:, 1] + r)
        nfree = len(free)
        if nfree < 2:
            return img, np.array([0.0]), 0, 0.0

        sums = self._gamma_sums(flat, idx)
        use_corr = self._y is not None
        if use_corr:
            ypad = np.full_like(flat, np.nan)
            Yp = ypad.reshape(-1, self._ncol)
            Yp[r:r + nx, r:r + ny] = self._y
            yflat = ypad
            zc = flat[idx]
            yc = yflat[idx]
            n_all = len(idx)
            mz, sz = zc.mean(), zc.std()
            my, sy = yc.mean(), yc.std()
            szy = float((zc * yc).sum())

            def rho_of(s):
                return (s / n_all - mz * my) / max(sz * sy, 1e-300)

        def objective(sums_, szy_):
            o_v = self._variogram_objective(sums_)
            o_c = (rho_of(szy_) - self.rho_target_) ** 2 if use_corr else 0.0
            return o_v, o_c

        def delta(a, b):
            nodes = np.array([a, b])
            old = self._local_sums(flat, nodes)
            flat[a], flat[b] = flat[b], flat[a]
            new = self._local_sums(flat, nodes)
            flat[a], flat[b] = flat[b], flat[a]
            d_szy = (flat[b] - flat[a]) * (yflat[a] - yflat[b]) if use_corr else 0.0
            return new - old, d_szy

        # component scaling: mean |dO| over random swaps (sasim)
        o_v0, o_c0 = objective(sums, szy if use_corr else 0.0)
        dv, dc = [], []
        for _ in range(min(200, 10 * nfree)):
            a, b = free[rng.choice(nfree, 2, replace=False)]
            ds, dz = delta(a, b)
            ov, oc = objective(sums + ds, (szy + dz) if use_corr else 0.0)
            dv.append(abs(ov - o_v0))
            dc.append(abs(oc - o_c0))
        w_v = self.component_weights.get("variogram", self.component_weights.get("indicator", 1.0))
        w_c = self.component_weights.get("correlation", 1.0)
        s_v = w_v / max(np.mean(dv), 1e-300)
        s_c = (w_c / max(np.mean(dc), 1e-300)) if use_corr else 0.0

        def total(ov, oc):
            return s_v * ov + s_c * oc

        O = total(o_v0, o_c0)
        O_init = O if O > 0 else 1.0
        t = self.t0 * float(np.mean([s_v * x for x in dv]) + np.mean([s_c * x for x in dc]))
        k_accept = self.k_accept or 5 * nfree
        k_max = self.k_max or 50 * nfree
        max_swaps = self.max_swaps or 500 * nfree
        history, swaps, idle = [O / O_init], 0, 0
        while swaps < max_swaps and O > self.tol * O_init and idle < self.n_stop:
            accepted = attempts = 0
            while attempts < k_max and accepted < k_accept and swaps < max_swaps:
                a, b = free[rng.integers(nfree)], free[rng.integers(nfree)]
                attempts += 1
                swaps += 1
                if a == b or flat[a] == flat[b]:
                    continue
                ds, dz = delta(a, b)
                new_sums = sums + ds
                new_szy = (szy + dz) if use_corr else 0.0
                O_new = total(*objective(new_sums, new_szy))
                if O_new <= O or rng.uniform() < np.exp((O - O_new) / t):
                    flat[a], flat[b] = flat[b], flat[a]
                    sums, O = new_sums, O_new
                    if use_corr:
                        szy = new_szy
                    accepted += 1
                    if O <= self.tol * O_init:
                        break
            idle = idle + 1 if accepted == 0 else 0
            history.append(O / O_init)
            t *= self.reduction
        out = flat.reshape(-1, self._ncol)[r:r + nx, r:r + ny].copy()
        self._tracked = (sums.copy(), szy if use_corr else None)   # for consistency checks
        return out, np.array(history), swaps, O / O_init

    def simulate_lattice(self, nsim: int = 1,
                         seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations on the full lattice, shape ``(nsim, nx, ny)``."""
        if not hasattr(self, "_hard"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        res = [self._simulate_one(_get_random_generator(int(s))) for s in seeds]
        self.history_ = [h for _, h, _, _ in res]
        self.n_swaps_ = np.array([n for _, _, n, _ in res])
        self.objective_ = np.array([o for _, _, _, o in res])
        return np.stack([img for img, _, _, _ in res])

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations at the grid nodes, shape ``(nsim, n_nodes)``, in grid order."""
        lat = self.simulate_lattice(nsim, seed)
        return lat[:, self._grid_idx[:, 0], self._grid_idx[:, 1]]

    def experimental_variogram(self, image: np.ndarray) -> np.ndarray:
        """Image (indicator) variograms at the fitted lags, shape (n_components, n_lags_total)."""
        flat, idx = self._setup_state(np.asarray(image, dtype=float))
        return self._gamma_sums(flat, idx) / (2.0 * self._n_pairs)

    def target_variogram(self) -> np.ndarray:
        """Target (indicator) variograms at the fitted lags, shape (n_components, n_lags_total)."""
        return self._targets[0].copy()


def sasim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    zz: str,
    vario: Optional[Sequence] = None,
    indicator_varios: Optional[Dict] = None,
    nsim: int = 1,
    seed: Optional[Union[int, np.random.Generator]] = None,
    secondary: Optional[np.ndarray] = None,
    secondary_col: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Simulated annealing simulation in one call; see `SimulatedAnnealing`.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    sa = SimulatedAnnealing(vario=vario, indicator_varios=indicator_varios, **kwargs)
    sa.fit(prediction_grid, df, xx, yy, zz, secondary=secondary, secondary_col=secondary_col)
    return sa.simulate(nsim, seed)
