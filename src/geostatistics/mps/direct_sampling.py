"""
Multiple-point statistics (MPS) simulation by Direct Sampling.

Two-point statistics (variograms) cannot describe curvilinear, connected
structures such as channels: very different images share the same
variogram. MPS borrows those patterns from a *training image* (TI), a
conceptual picture of the expected geology, instead of a variogram.

This module implements Direct Sampling (Mariethoz, Renard & Straubhaar,
2010), which works for categorical and continuous variables without
building a pattern database:

1. Visit the simulation nodes along a random path (data are pasted first).
2. At each node take the *data event*: the ``n_neighbors`` closest nodes
   that already have a value (within ``search_radius`` cells), with their
   lags and values.
3. Scan the TI from a random position. For each TI location compute the
   distance between its pattern and the data event (fraction of mismatches
   for categories, mean absolute difference / TI range for continuous
   values, optionally weighted by lag ``** -distance_power``). Accept the
   first location with distance ``<= threshold`` and paste its central
   value; if ``max_scan_fraction`` of the TI is scanned without success,
   paste the best match seen.
4. Optionally re-simulate every non-data node in ``n_postprocess`` further
   passes, now with a fully informed neighbourhood (Mariethoz et al., 2010,
   §2.5), which removes noise from nodes pasted early on sparse data events.

``threshold`` and ``n_neighbors`` trade pattern fidelity against speed and
variability: ``threshold = 0`` with many neighbours copies the TI almost
verbatim; larger thresholds give more noise. For a continuous TI, expect
somewhat more roughness than the TI at the finest lag (about 1.5x the
lag-1 variogram with the defaults); lower `threshold`, more neighbours, or
more `n_postprocess` passes reduce it. Direct Sampling does not enforce
global proportions or histograms; check them with `pattern_histogram` /
`connectivity` and the realizations' statistics. The TI must use the same cell
size as the simulation lattice, and is indexed ``ti[ix, iy]`` (axis 0 = x,
axis 1 = y), like `FFTSimulator.simulate_lattice`. NaN cells in the TI are
never used.

Also provided: `pattern_histogram` and `connectivity` to compare
realizations with the TI, and `channel_training_image` to make a synthetic
sinuous-channel TI.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..simulation.fftsim import _infer_lattice
from ..simulation.sgsim import _get_random_generator

__all__ = [
    "DirectSampling",
    "dssim",
    "pattern_histogram",
    "pattern_histogram_distance",
    "connectivity",
    "channel_training_image",
]

_BATCH = 128


def _template(radius: float, rng: np.random.Generator) -> np.ndarray:
    """Offsets within `radius` cells, sorted by distance (ties in random order)."""
    r = int(np.ceil(radius))
    ox, oy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    off = np.column_stack([ox.ravel(), oy.ravel()])
    d = np.hypot(off[:, 0], off[:, 1])
    keep = (d > 0) & (d <= radius)
    off, d = off[keep], d[keep]
    order = np.lexsort((rng.random(len(d)), d))
    return off[order]


def _event_distance(ti_flat: np.ndarray, centres: np.ndarray, lag_flat: np.ndarray,
                    values: np.ndarray, weights: np.ndarray, categorical: bool,
                    value_range: float) -> np.ndarray:
    """Data-event distance at flat padded-TI `centres` (B,); NaN = mismatch."""
    tv = ti_flat[centres[:, None] + lag_flat[None, :]]                 # (B, k)
    if categorical:
        mis = (tv != values).astype(float)                             # NaN != v -> 1
    else:
        mis = np.abs(tv - values) / value_range
        mis = np.where(np.isnan(mis), 1.0, np.minimum(mis, 1.0))
    return mis @ weights


class DirectSampling:
    """
    Direct Sampling multiple-point simulator.

    Parameters
    ----------
    training_image : 2-D array
        TI indexed ``[ix, iy]`` with the simulation lattice's cell size. NaN
        cells are ignored.
    kind : {'auto', 'categorical', 'continuous'}, default 'auto'
        'auto' treats integer-valued TIs with at most 20 distinct values as
        categorical.
    n_neighbors : int, default 30
        Maximum number of informed nodes in a data event.
    threshold : float, default 0.05
        Acceptance distance in ``[0, 1]``.
    max_scan_fraction : float, default 0.3
        Largest share of the TI scanned per node.
    search_radius : float, optional
        Data-event search radius in cells; default a quarter of the smaller
        TI dimension (at least 3).
    distance_power : float, default 0.0
        Weight data-event nodes by ``lag ** -distance_power`` (0 = equal).
    secondary_ti : 2-D array, optional
        Secondary training image paired cell by cell with `training_image`
        (e.g. seismic forward-modelled on the TI). Enables co-simulation
        with the exhaustive secondary passed to `fit`.
    secondary_weight : float, default 0.5
        Weight ``w`` of the secondary in the distance
        ``(1 - w) d_primary + w d_secondary``.
    n_secondary : int, default 5
        Secondary data-event size: the collocated value plus the nearest
        ``n_secondary - 1`` template offsets.
    use_gpu : bool or 'auto', default False
        Run all realizations in lockstep with batched array operations on the
        GPU (CuPy); 'auto' uses the GPU when one is usable. Statistically
        equivalent to the sequential engine, not bit-identical; pays off with
        many realizations (see `geolattice.geostatistics.mps_gpu`).
    secondary_kind : {'auto', 'categorical', 'continuous'}, default 'auto'
    n_postprocess : int, default 1
        Extra passes re-simulating every non-data node once its whole
        neighbourhood is informed (Mariethoz et al., 2010, §2.5). Removes the
        fine-scale noise left by nodes simulated early from sparse, distant
        neighbours; each pass costs about as much as the main pass.
    spacing : float or (dx, dy), optional
        Lattice spacing when it cannot be inferred from the grid.

    Attributes
    ----------
    categories_ : ndarray or None
        Category values for a categorical TI.
    lattice_ : dict
        ``x0, y0, dx, dy, nx, ny`` of the simulated lattice.
    max_data_shift_ : float
        Largest distance a datum was moved to reach its node.

    Examples
    --------
    >>> ti = channel_training_image(150, 150, seed=0)
    >>> ds = DirectSampling(ti, n_neighbors=30, threshold=0.05).fit(grid, wells, "X", "Y", "facies")
    >>> reals = ds.simulate(nsim=10, seed=1)          # (10, len(grid))
    """

    def __init__(self, training_image, kind: str = "auto", n_neighbors: int = 30,
                 threshold: float = 0.05, max_scan_fraction: float = 0.3,
                 search_radius: Optional[float] = None, distance_power: float = 0.0,
                 n_postprocess: int = 1, spacing=None, secondary_ti=None,
                 secondary_weight: float = 0.5, n_secondary: int = 5,
                 secondary_kind: str = "auto", use_gpu: Union[bool, str] = False):
        ti = np.asarray(training_image, dtype=float)
        if ti.ndim != 2:
            raise ValueError("training_image must be a 2-D array")
        if np.isnan(ti).all():
            raise ValueError("training_image has no valid cells")
        if kind not in ("auto", "categorical", "continuous"):
            raise ValueError("kind must be 'auto', 'categorical', or 'continuous'")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 < max_scan_fraction <= 1.0:
            raise ValueError("max_scan_fraction must be in (0, 1]")
        vals = ti[~np.isnan(ti)]
        if kind == "auto":
            kind = ("categorical" if np.all(vals == np.round(vals)) and len(np.unique(vals)) <= 20
                    else "continuous")
        self.ti = ti
        self.kind = kind
        self.n_neighbors = int(n_neighbors)
        self.threshold = float(threshold)
        self.max_scan_fraction = float(max_scan_fraction)
        self.search_radius = (max(3.0, min(ti.shape) / 4.0) if search_radius is None
                              else float(search_radius))
        self.distance_power = float(distance_power)
        if n_postprocess < 0:
            raise ValueError("n_postprocess must be >= 0")
        self.n_postprocess = int(n_postprocess)
        self.spacing = spacing
        self.use_gpu = use_gpu
        self.categories_ = np.unique(vals) if kind == "categorical" else None
        self._range = float(vals.max() - vals.min()) or 1.0
        valid = np.argwhere(~np.isnan(ti))
        self._ti_valid = valid                                    # candidate centres

        self.secondary_ti = None
        if secondary_ti is not None:
            sti = np.asarray(secondary_ti, dtype=float)
            if sti.shape != ti.shape:
                raise ValueError("secondary_ti must have the same shape as training_image")
            if not 0.0 <= secondary_weight <= 1.0:
                raise ValueError("secondary_weight must be in [0, 1]")
            if n_secondary < 1:
                raise ValueError("n_secondary must be >= 1")
            svals = sti[~np.isnan(sti)]
            if secondary_kind == "auto":
                secondary_kind = ("categorical" if np.all(svals == np.round(svals))
                                  and len(np.unique(svals)) <= 20 else "continuous")
            if secondary_kind not in ("categorical", "continuous"):
                raise ValueError("secondary_kind must be 'auto', 'categorical', or 'continuous'")
            self.secondary_ti = sti
            self.secondary_kind = secondary_kind
            self._sec_range = float(svals.max() - svals.min()) or 1.0
        self.secondary_weight = float(secondary_weight)
        self.n_secondary = int(n_secondary)

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", zz: str = "Z",
            secondary: Optional[np.ndarray] = None) -> "DirectSampling":
        """
        Set up the simulation lattice and paste hard data `df[zz]` onto nodes.
        With a `secondary_ti`, `secondary` (n_nodes,) is the exhaustive
        secondary at the grid nodes.
        """
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        x0, y0, dx, dy = _infer_lattice(grid, self.spacing)
        gi = np.rint((grid - [x0, y0]) / [dx, dy]).astype(np.int64)
        has_data = df is not None and len(df) > 0
        if has_data:
            xy = df[[xx, yy]].to_numpy(dtype=float)
            z = df[zz].to_numpy(dtype=float)
            if self.kind == "categorical":
                unknown = set(np.unique(z)) - set(self.categories_)
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

        self._hard = np.full((nx, ny), np.nan)
        self.max_data_shift_ = 0.0
        if has_data:
            idx = di - lo
            keys, inv = np.unique(idx, axis=0, return_inverse=True)
            inv = inv.ravel()
            for k, (i, j) in enumerate(keys):
                v = z[inv == k]
                if self.kind == "categorical":
                    u, c = np.unique(v, return_counts=True)
                    self._hard[i, j] = u[c.argmax()]
                else:
                    self._hard[i, j] = v.mean()
            node_xy = np.column_stack([self.lattice_["x0"] + idx[:, 0] * dx,
                                       self.lattice_["y0"] + idx[:, 1] * dy])
            self.max_data_shift_ = float(np.sqrt(((xy - node_xy) ** 2).sum(axis=1)).max())

        self._sec_lattice = None
        if self.secondary_ti is not None:
            if secondary is None:
                raise ValueError("secondary_ti was given: pass the exhaustive secondary to fit")
            sec = np.asarray(secondary, dtype=float).ravel()
            if len(sec) != len(grid):
                raise ValueError("secondary must have one value per grid node")
            lat = np.full((nx, ny), np.nan)
            lat[self._grid_idx[:, 0], self._grid_idx[:, 1]] = sec
            miss = np.argwhere(np.isnan(lat))
            if len(miss):                      # lattice nodes outside the grid: nearest node
                from scipy.spatial import cKDTree
                lat[miss[:, 0], miss[:, 1]] = sec[cKDTree(self._grid_idx).query(miss)[1]]
            self._sec_lattice = lat
        elif secondary is not None:
            raise ValueError("secondary needs a secondary_ti")
        return self

    # ------------------------------------------------------------- simulate

    def _distance(self, centres: np.ndarray, lag_flat: np.ndarray, values: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
        """Primary data-event distance at flat padded-TI `centres` (B,)."""
        return _event_distance(self._ti_pad, centres, lag_flat, values, weights,
                               self.kind == "categorical", self._range)

    def _simulate_one(self, rng: np.random.Generator) -> np.ndarray:
        nx, ny = self.lattice_["nx"], self.lattice_["ny"]
        r = int(np.ceil(self.search_radius))
        template = _template(self.search_radius, rng)
        tdist = np.hypot(template[:, 0], template[:, 1])

        # NaN-padded simulation grid and TI, addressed by flat indices so no
        # bounds checks are needed: padding is simply "uninformed" / mismatch.
        sim = np.full((nx + 2 * r, ny + 2 * r), np.nan)
        sim[r:r + nx, r:r + ny] = self._hard
        sim_flat = sim.ravel()
        sim_lag = template[:, 0] * sim.shape[1] + template[:, 1]
        tpad = np.full((self.ti.shape[0] + 2 * r, self.ti.shape[1] + 2 * r), np.nan)
        tpad[r:-r, r:-r] = self.ti
        self._ti_pad = tpad.ravel()
        ti_lag = template[:, 0] * tpad.shape[1] + template[:, 1]
        valid = (self._ti_valid[:, 0] + r) * tpad.shape[1] + (self._ti_valid[:, 1] + r)
        n_valid = len(valid)
        max_scan = max(1, int(np.ceil(self.max_scan_fraction * n_valid)))
        scan_order = valid[rng.permutation(n_valid)]
        ti_flat = self._ti_pad

        cosim = self._sec_lattice is not None
        if cosim:
            spad = np.full(sim.shape, np.nan)
            spad[r:r + nx, r:r + ny] = self._sec_lattice
            s_flat = spad.ravel()
            stpad = np.full(tpad.shape, np.nan)
            stpad[r:-r, r:-r] = self.secondary_ti
            st_flat = stpad.ravel()
            soff = np.vstack([[0, 0], template[: self.n_secondary - 1]])
            s_sim_lag = soff[:, 0] * sim.shape[1] + soff[:, 1]
            s_ti_lag = soff[:, 0] * tpad.shape[1] + soff[:, 1]
            w_sec = self.secondary_weight
            sec_cat = self.secondary_kind == "categorical"

        free = np.flatnonzero(np.isnan(self._hard).ravel())
        free = (free // ny + r) * sim.shape[1] + (free % ny + r)       # flat padded nodes
        # main pass along a random path, then post-processing passes that
        # re-simulate every non-data node with a fully informed neighbourhood
        path = np.concatenate([free[rng.permutation(len(free))]
                               for _ in range(1 + self.n_postprocess)])
        for node in path:
            vals = sim_flat[node + sim_lag]                            # centre is not in the template
            informed = np.flatnonzero(~np.isnan(vals))[: self.n_neighbors]
            if cosim:
                sv = s_flat[node + s_sim_lag]
                sok = ~np.isnan(sv)
                s_vals, s_lag = sv[sok], s_ti_lag[sok]
                s_w = np.full(len(s_vals), 1.0 / max(len(s_vals), 1))
            if len(informed) == 0 and not cosim:
                sim_flat[node] = ti_flat[scan_order[rng.integers(n_valid)]]
                continue
            values = vals[informed]
            lag_flat = ti_lag[informed]
            if len(informed) == 0:
                w = np.empty(0)
            elif self.distance_power:
                w = tdist[informed] ** -self.distance_power
                w = w / w.sum()
            else:
                w = np.full(len(informed), 1.0 / len(informed))

            if cosim:
                def dist(c):
                    d2 = _event_distance(st_flat, c, s_lag, s_vals, s_w, sec_cat, self._sec_range)
                    if len(informed) == 0:
                        return d2
                    return (1.0 - w_sec) * self._distance(c, lag_flat, values, w) + w_sec * d2
            else:
                def dist(c):
                    return self._distance(c, lag_flat, values, w)

            start = rng.integers(n_valid)
            best_d, best_v = np.inf, np.nan
            scanned = 0
            batch = _BATCH
            while scanned < max_scan:
                b = min(batch, max_scan - scanned)
                pos = start + scanned
                if pos + b <= n_valid:
                    centres = scan_order[pos:pos + b]
                else:
                    centres = scan_order[(pos + np.arange(b)) % n_valid]
                d = dist(centres)
                hit = np.flatnonzero(d <= self.threshold)
                if len(hit):
                    best_v = ti_flat[centres[hit[0]]]
                    break
                k = int(d.argmin())
                if d[k] < best_d:
                    best_d, best_v = d[k], ti_flat[centres[k]]
                scanned += b
                batch = min(batch * 2, 4096)                           # hard nodes: bigger batches
            sim_flat[node] = best_v
        return sim[r:r + nx, r:r + ny].copy()

    def simulate_lattice(self, nsim: int = 1,
                         seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations on the full lattice, shape ``(nsim, nx, ny)``."""
        if not hasattr(self, "_hard"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=nsim)
        from ..utils.backend import resolve_cupy_use_gpu

        if resolve_cupy_use_gpu(self.use_gpu):
            import cupy

            from ..gpu.mps_gpu import ds_simulate_batched

            return ds_simulate_batched(self, seeds, cupy)
        return np.stack([self._simulate_one(_get_random_generator(int(s))) for s in seeds])

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Realizations at the grid nodes, shape ``(nsim, n_nodes)``, in grid order."""
        lat = self.simulate_lattice(nsim, seed)
        return lat[:, self._grid_idx[:, 0], self._grid_idx[:, 1]]


def dssim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    zz: str,
    training_image,
    nsim: int = 1,
    n_neighbors: int = 30,
    threshold: float = 0.05,
    max_scan_fraction: float = 0.3,
    kind: str = "auto",
    seed: Optional[Union[int, np.random.Generator]] = None,
    secondary: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Direct Sampling MPS simulation in one call.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        Nodes on a regular lattice with the TI's cell size.
    df : pandas.DataFrame or None
        Hard data (moved to their nearest node); None for unconditional.
    xx, yy, zz : str
        Column names for x, y and the variable.
    training_image : 2-D array
        TI indexed ``[ix, iy]``.
    nsim, n_neighbors, threshold, max_scan_fraction, kind, seed
        See `DirectSampling`.
    secondary : (n,) array, optional
        Exhaustive secondary at the nodes (needs ``secondary_ti`` in kwargs).
    **kwargs
        ``search_radius``, ``distance_power``, ``n_postprocess``, ``spacing``,
        ``secondary_ti``, ``secondary_weight``, ``n_secondary``, ``secondary_kind``.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
    """
    ds = DirectSampling(training_image, kind=kind, n_neighbors=n_neighbors,
                        threshold=threshold, max_scan_fraction=max_scan_fraction, **kwargs)
    return ds.fit(prediction_grid, df, xx, yy, zz, secondary=secondary).simulate(nsim, seed)


# --------------------------------------------------------------- diagnostics


def pattern_histogram(image, size: Union[int, Sequence[int]] = 2) -> Dict[tuple, float]:
    """
    Multiple-point histogram: frequency of every ``size x size`` (or
    ``(sx, sy)``) pattern of a categorical image, as ``{pattern: frequency}``.
    Patterns containing NaN are skipped.
    """
    img = np.asarray(image, dtype=float)
    sx, sy = (size, size) if np.isscalar(size) else size
    nx, ny = img.shape
    if sx > nx or sy > ny:
        raise ValueError("pattern size exceeds the image")
    windows = np.lib.stride_tricks.sliding_window_view(img, (sx, sy)).reshape(-1, sx * sy)
    windows = windows[~np.isnan(windows).any(axis=1)]
    pats, counts = np.unique(windows, axis=0, return_counts=True)
    total = counts.sum()
    return {tuple(p): c / total for p, c in zip(pats, counts)}


def pattern_histogram_distance(a: Dict[tuple, float], b: Dict[tuple, float]) -> float:
    """Total-variation distance between two pattern histograms (0 = identical, 1 = disjoint)."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def connectivity(image, category, max_lag: int = 30, axis: int = 0) -> np.ndarray:
    """
    Connectivity function: probability that two cells of `category` a lag
    ``h = 1..max_lag`` cells apart along `axis` belong to the same connected
    (4-neighbour) body, given that both are `category`. Long-range values
    near 1 mean connected features such as channels.
    """
    from scipy import ndimage

    img = np.asarray(image)
    labels, _ = ndimage.label(img == category)
    out = np.full(max_lag, np.nan)
    for h in range(1, max_lag + 1):
        if axis == 0:
            a, b = labels[:-h, :], labels[h:, :]
        else:
            a, b = labels[:, :-h], labels[:, h:]
        both = (a > 0) & (b > 0)
        if both.any():
            out[h - 1] = float((a[both] == b[both]).mean())
    return out


def channel_training_image(nx: int = 150, ny: int = 150, n_channels: int = 6,
                           width: float = 4.0, amplitude: float = 12.0,
                           wavelength: float = 60.0,
                           seed: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """
    Synthetic binary training image of sinuous channels (1) in a background
    (0), running along axis 0 (x). Useful for tests and tutorials.
    """
    rng = _get_random_generator(seed)
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    ti = np.zeros((nx, ny))
    for _ in range(n_channels):
        y0 = rng.uniform(0, ny)
        phase = rng.uniform(0, 2 * np.pi)
        wl = wavelength * rng.uniform(0.7, 1.3)
        amp = amplitude * rng.uniform(0.6, 1.4)
        w = width * rng.uniform(0.7, 1.3)
        centre = y0 + amp * np.sin(2 * np.pi * x / wl + phase)
        ti[np.abs(y - centre) <= w / 2] = 1.0
    return ti
