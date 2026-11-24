"""
SNESIM: single normal equation simulation of categorical variables.

Multiple-point simulation from a categorical training image (TI)
(Strebelle, 2002). Instead of a variogram, the conditional probability of
each category at a node is read from the TI as the frequency of that
category at the centre of every TI pattern that matches the node's *data
event* (the informed nodes of a fixed template around it):

    P(c | data event) = #(TI replicates with centre c) / #(TI replicates)

Algorithm
---------
* **Template**: the `n_template` nodes closest to the centre (within
  `search_radius` cells), ordered by distance.
* **Multigrid**: simulation proceeds from the coarsest grid (spacing
  ``2**(n_multigrid - 1)`` cells) to the finest, with the template expanded
  by the grid spacing, so large structures are fixed first by a small
  template.
* **Pattern statistics**: for every level the TI is scanned once and its
  unique template patterns are stored with per-category counts of their
  centre value (the list-based equivalent of Strebelle's search tree, as in
  IMPALA, Straubhaar et al., 2011). TI cells whose template nodes fall
  outside the TI never match those nodes.
* **Data event reduction**: if fewer than `min_replicates` TI patterns
  match, the farthest informed nodes are dropped until enough do (all
  prefixes are counted in one filtering pass).
* **Servosystem**: probabilities are pulled toward the target proportions,
  ``p_c += servo / (1 - servo) * (target_c - current_c)``, to stop the
  simulated proportions from drifting. The correction is applied only to
  categories with ``p_c > 0``: a category the TI never shows for the data
  event stays impossible, so the servosystem cannot create patterns (e.g.
  contacts between classes) that do not occur in the TI. The result is
  clipped at 0 and renormalised; if nothing is left, the uncorrected
  probabilities are used.
* **Hard data** sit on their nearest nodes and, on coarse grids, are also
  copied temporarily to the nearest coarse node so the coarse structure
  honours them.

The TI is indexed ``ti[ix, iy]`` with the same cell size as the simulation
lattice; NaN cells are ignored. Categories are the TI's distinct values.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ..simulation.fftsim import _infer_lattice
from ..simulation.sgsim import _get_random_generator

__all__ = ["SNESIM", "snesim", "tau_model"]


def tau_model(p_prior: np.ndarray, p_ti: np.ndarray, p_soft: np.ndarray,
              tau=(1.0, 1.0)) -> np.ndarray:
    """
    Combine training-image and soft probabilities with the tau model
    (Journel, 2002): with distances ``x = (1 - P) / P``,

        x / x0 = (x_ti / x0) ** tau1 * (x_soft / x0) ** tau2,

    ``x0`` from the prior (target proportions), then ``P = 1 / (1 + x)`` per
    category, renormalised to sum to 1. ``tau = (1, 1)`` is permanence of
    ratios; ``tau2 = 0`` ignores the soft data, ``tau1 = 0`` the TI.
    """
    eps = 1e-9
    p0 = np.clip(np.asarray(p_prior, float), eps, 1 - eps)
    p1 = np.clip(np.asarray(p_ti, float), eps, 1 - eps)
    p2 = np.clip(np.asarray(p_soft, float), eps, 1 - eps)
    x0, x1, x2 = (1 - p0) / p0, (1 - p1) / p1, (1 - p2) / p2
    x = x0 * (x1 / x0) ** tau[0] * (x2 / x0) ** tau[1]
    p = 1.0 / (1.0 + x)
    return p / p.sum(axis=-1, keepdims=True)

_MISSING = 255


def _template(n_template: int, radius: float) -> np.ndarray:
    """The `n_template` offsets nearest the centre within `radius` (deterministic order)."""
    r = int(np.ceil(radius))
    ox, oy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    off = np.column_stack([ox.ravel(), oy.ravel()])
    d = np.hypot(off[:, 0], off[:, 1])
    keep = (d > 0) & (d <= radius)
    off, d = off[keep], d[keep]
    angle = np.arctan2(off[:, 1], off[:, 0])
    order = np.lexsort((angle, d))
    return off[order][:n_template]


class SNESIM:
    """
    SNESIM multiple-point simulator for categorical variables.

    Parameters
    ----------
    training_image : 2-D array
        Categorical TI indexed ``[ix, iy]`` (at most 254 categories); NaN
        cells are ignored.
    n_template : int, default 40
        Number of template nodes.
    search_radius : float, default 8.0
        Template radius in cells of the current grid.
    n_multigrid : int, default 4
        Number of multiple grids (1 = no multigrid).
    min_replicates : int, default 3
        Minimum matching TI patterns before the farthest data-event nodes are
        dropped. Small values keep more conditioning nodes and so better
        continuity; GSLIB-style values (~10) suit large TIs.
    servo : float, default 0.5
        Servosystem strength in ``[0, 1)``; 0 switches it off. Only
        categories with non-zero TI conditional probability are corrected,
        so categories the TI rules out for a data event are never revived.
    target_proportions : dict, optional
        ``{category: proportion}`` for the servosystem; default the TI
        proportions.
    spacing : float or (dx, dy), optional
        Lattice spacing when it cannot be inferred from the grid.
    use_gpu : bool or 'auto', default False
        Run all realizations in lockstep with batched array operations on the
        GPU (CuPy); 'auto' uses the GPU when one is usable. Statistically
        equivalent to the sequential engine, not bit-identical; pays off with
        many realizations (see `geolattice.geostatistics.mps_gpu`).
    tau : (float, float), default (1.0, 1.0)
        Tau-model exponents for TI and soft probabilities when `soft` is
        passed to `fit` (multiple-point co-simulation); see `tau_model`.

    Attributes
    ----------
    categories_ : ndarray
        Category values (realizations contain these values).
    target_proportions_ : ndarray
        Target proportion per category.
    n_patterns_ : list of int
        Unique TI patterns stored per grid level (finest first).
    lattice_ : dict
        ``x0, y0, dx, dy, nx, ny`` of the simulated lattice.

    Examples
    --------
    >>> ti = channel_training_image(150, 150, seed=0)
    >>> sim = SNESIM(ti, n_template=40, n_multigrid=4).fit(grid, wells, "X", "Y", "facies")
    >>> reals = sim.simulate(nsim=10, seed=1)            # (10, len(grid))
    """

    def __init__(self, training_image, n_template: int = 40, search_radius: float = 8.0,
                 n_multigrid: int = 4, min_replicates: int = 3, servo: float = 0.5,
                 target_proportions: Optional[Dict] = None, spacing=None,
                 tau=(1.0, 1.0), use_gpu: Union[bool, str] = False):
        ti = np.asarray(training_image, dtype=float)
        if ti.ndim != 2:
            raise ValueError("training_image must be a 2-D array")
        valid = ~np.isnan(ti)
        if not valid.any():
            raise ValueError("training_image has no valid cells")
        cats = np.unique(ti[valid])
        if len(cats) > 254:
            raise ValueError("SNESIM is for categorical training images (at most 254 categories)")
        if n_template < 1 or n_multigrid < 1 or min_replicates < 1:
            raise ValueError("n_template, n_multigrid and min_replicates must be >= 1")
        if not 0.0 <= servo < 1.0:
            raise ValueError("servo must be in [0, 1)")
        self.ti = ti
        self.categories_ = cats
        self.n_template = int(n_template)
        self.search_radius = float(search_radius)
        self.n_multigrid = int(n_multigrid)
        self.min_replicates = int(min_replicates)
        self.servo = float(servo)
        self.spacing = spacing
        self.use_gpu = use_gpu
        if len(tau) != 2 or min(tau) < 0:
            raise ValueError("tau must be two non-negative exponents")
        self.tau = (float(tau[0]), float(tau[1]))

        codes = np.full(ti.shape, _MISSING, dtype=np.uint8)
        codes[valid] = np.searchsorted(cats, ti[valid]).astype(np.uint8)
        self._ti_codes = codes
        k = len(cats)
        ti_prop = np.bincount(codes[valid], minlength=k) / valid.sum()
        self._user_target = target_proportions is not None
        if target_proportions is None:
            self.target_proportions_ = ti_prop
        else:
            tp = np.array([float(target_proportions.get(c, 0.0)) for c in cats])
            if (tp < 0).any() or tp.sum() <= 0:
                raise ValueError("target_proportions must be non-negative and not all zero")
            self.target_proportions_ = tp / tp.sum()

        self._template = _template(self.n_template, self.search_radius)
        self._levels = [self._pattern_table(2 ** g) for g in range(self.n_multigrid)]
        self.n_patterns_ = [len(lv[0]) for lv in self._levels]

    def _pattern_table(self, step: int):
        """Unique TI patterns for the template scaled by `step`, with centre counts."""
        codes = self._ti_codes
        nx, ny = codes.shape
        off = self._template * step
        r = int(np.abs(off).max())
        pad = np.full((nx + 2 * r, ny + 2 * r), _MISSING, dtype=np.uint8)
        pad[r:r + nx, r:r + ny] = codes
        centres = np.argwhere(codes != _MISSING)
        pats = pad[centres[:, 0, None] + r + off[None, :, 0], centres[:, 1, None] + r + off[None, :, 1]]
        centre_codes = codes[centres[:, 0], centres[:, 1]]
        uniq, inv = np.unique(pats, axis=0, return_inverse=True)
        inv = inv.ravel()
        k = len(self.categories_)
        counts = np.zeros((len(uniq), k))
        np.add.at(counts, (inv, centre_codes), 1.0)
        return uniq, counts

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", cat: str = "facies",
            soft: Optional[np.ndarray] = None) -> "SNESIM":
        """
        Set up the simulation lattice and place hard data `df[cat]` on nodes.
        `soft` (n_nodes, n_categories), columns in `categories_` order, are
        soft probabilities (e.g. calibrated seismic) merged with the TI
        probabilities by the tau model. Unless `target_proportions` was
        given, their mean becomes the target proportions (servosystem and
        tau prior), since soft data estimate the area's proportions better
        than the TI.
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        x0, y0, dx, dy = _infer_lattice(grid, self.spacing)
        gi = np.rint((grid - [x0, y0]) / [dx, dy]).astype(np.int64)
        has_data = df is not None and len(df) > 0
        if has_data:
            xy = df[[xx, yy]].to_numpy(dtype=float)
            vals = df[cat].to_numpy(dtype=float)
            unknown = set(np.unique(vals)) - set(self.categories_)
            if unknown:
                raise ValueError(f"data values {sorted(unknown)} are not categories of the TI")
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

        self._hard = np.full((nx, ny), _MISSING, dtype=np.uint8)
        if has_data:
            idx = di - lo
            codes = np.searchsorted(self.categories_, vals)
            keys, inv = np.unique(idx, axis=0, return_inverse=True)
            inv = inv.ravel()
            for k, (i, j) in enumerate(keys):
                self._hard[i, j] = np.bincount(codes[inv == k]).argmax()

        self._soft = None
        if soft is not None:
            soft = np.asarray(soft, dtype=float)
            K = len(self.categories_)
            if soft.shape != (len(grid), K):
                raise ValueError(f"soft must have shape ({len(grid)}, {K})")
            if (soft < 0).any() or not np.allclose(soft.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("soft rows must be probabilities summing to 1")
            if not self._user_target:
                # the soft data estimate this area's proportions better than the TI
                self.target_proportions_ = soft.mean(axis=0)
            lat = np.tile(self.target_proportions_, (nx, ny, 1))
            lat[self._grid_idx[:, 0], self._grid_idx[:, 1]] = soft
            self._soft = lat
        elif not self._user_target:
            valid = self._ti_codes != _MISSING
            self.target_proportions_ = (np.bincount(self._ti_codes[valid], minlength=len(self.categories_))
                                        / valid.sum())
        return self

    # ------------------------------------------------------------- simulate

    def _probabilities(self, level: int, values: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Category probabilities for a data event (template columns `cols`, nearest first)."""
        pats, counts = self._levels[level]
        rows = np.arange(len(pats))
        best = None
        for col, v in zip(cols, values):
            rows = rows[pats[rows, col] == v]
            c = counts[rows].sum(axis=0)
            if c.sum() < self.min_replicates:
                break
            best = c
        if best is None:
            return self.target_proportions_.copy()
        return best / best.sum()

    def _simulate_one(self, rng: np.random.Generator) -> np.ndarray:
        nx, ny = self.lattice_["nx"], self.lattice_["ny"]
        k = len(self.categories_)
        sim = self._hard.copy()
        hard_mask = sim != _MISSING
        n_sim = np.bincount(sim[hard_mask], minlength=k).astype(float)
        target = self.target_proportions_
        gain = self.servo / (1.0 - self.servo) if self.servo else 0.0

        for g in reversed(range(self.n_multigrid)):
            step = 2 ** g
            off = self._template * step
            # temporarily copy hard data onto the nearest node of this grid
            relocated = []
            if g > 0 and hard_mask.any():
                for i, j in np.argwhere(hard_mask):
                    ci = min(int(round(i / step)) * step, (nx - 1) // step * step)
                    cj = min(int(round(j / step)) * step, (ny - 1) // step * step)
                    if sim[ci, cj] == _MISSING:
                        sim[ci, cj] = sim[i, j]
                        relocated.append((ci, cj))

            ii, jj = np.meshgrid(np.arange(0, nx, step), np.arange(0, ny, step), indexing="ij")
            nodes = np.column_stack([ii.ravel(), jj.ravel()])
            nodes = nodes[sim[nodes[:, 0], nodes[:, 1]] == _MISSING]
            nodes = nodes[rng.permutation(len(nodes))]
            for i, j in nodes:
                pi, pj = i + off[:, 0], j + off[:, 1]
                inside = (pi >= 0) & (pi < nx) & (pj >= 0) & (pj < ny)
                vals = np.full(len(off), _MISSING, dtype=np.uint8)
                vals[inside] = sim[pi[inside], pj[inside]]
                cols = np.flatnonzero(vals != _MISSING)
                p = self._probabilities(g, vals[cols], cols)
                if self._soft is not None:
                    p = tau_model(target, p, self._soft[i, j], self.tau)
                if gain:
                    total = n_sim.sum()
                    if total > 0:
                        adj = np.where(p > 0, np.clip(p + gain * (target - n_sim / total), 0.0, None), 0.0)
                        s = adj.sum()
                        if s > 0:
                            p = adj / s
                c = int(np.searchsorted(np.cumsum(p), rng.uniform() * p.sum()))
                c = min(c, k - 1)
                sim[i, j] = c
                n_sim[c] += 1

            for ci, cj in relocated:
                sim[ci, cj] = _MISSING
        return self.categories_[sim]

    def simulate_lattice(self, nsim: int = 1,
                         seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations on the full lattice, shape ``(nsim, nx, ny)``, as category values."""
        if not hasattr(self, "_hard"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        from ..utils.backend import resolve_cupy_use_gpu

        if resolve_cupy_use_gpu(self.use_gpu):
            import cupy

            from ..gpu.mps_gpu import snesim_simulate_batched

            return snesim_simulate_batched(self, seeds, cupy)
        return np.stack([self._simulate_one(_get_random_generator(int(s))) for s in seeds])

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations at the grid nodes, shape ``(nsim, n_nodes)``, in grid order."""
        lat = self.simulate_lattice(nsim, seed)
        return lat[:, self._grid_idx[:, 0], self._grid_idx[:, 1]]


def snesim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    cat: str,
    training_image,
    nsim: int = 1,
    n_template: int = 40,
    n_multigrid: int = 4,
    min_replicates: int = 3,
    servo: float = 0.5,
    seed: Optional[Union[int, np.random.Generator]] = None,
    soft: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    SNESIM simulation in one call.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        Nodes on a regular lattice with the TI's cell size.
    df : pandas.DataFrame or None
        Categorical hard data (moved to their nearest node); None for
        unconditional.
    xx, yy, cat : str
        Column names for x, y and the category.
    training_image : 2-D array
        Categorical TI indexed ``[ix, iy]``.
    nsim, n_template, n_multigrid, min_replicates, servo, seed
        See `SNESIM`.
    soft : (n, n_categories) array, optional
        Soft probabilities for tau-model co-simulation.
    **kwargs
        ``search_radius``, ``target_proportions``, ``spacing``, ``tau``.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
        Category values.
    """
    sim = SNESIM(training_image, n_template=n_template, n_multigrid=n_multigrid,
                 min_replicates=min_replicates, servo=servo, **kwargs)
    return sim.fit(prediction_grid, df, xx, yy, cat, soft=soft).simulate(nsim, seed)
