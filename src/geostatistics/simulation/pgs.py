"""
Plurigaussian simulation (PGS) of categorical variables.

Facies are obtained by truncating one or two latent Gaussian random fields
with a *truncation rule* (lithotype rule) that says which combination of
latent values gives which category (Le Loc'h & Galli, 1997; Armstrong et
al., 2003). Unlike indicator simulation, the rule controls which facies may
touch: categories whose regions share an edge in the rule can be in contact,
others cannot. With one field the rule is a set of ordered thresholds
(truncated Gaussian simulation) and every facies only touches its
neighbours in the sequence.

Latent fields
-------------
``Y1 = A`` and ``Y2 = rho * A + sqrt(1 - rho^2) * B``, where ``A`` and ``B``
are independent standard Gaussian fields with variograms `vario1` and
`vario2` (sill 1). They are simulated with `fftsim` on a regular lattice or
`tbsim` otherwise.

Truncation rule
---------------
`TruncationRule` holds rectangles in probability space ``[0, 1]^2``
(``u_k = Phi(y_k)``), each labelled with a category. Proportions follow from
the rule and ``rho`` (`TruncationRule.proportions`). Constructors:
`truncated_gaussian` (one field, ordered categories), `hierarchical` (first
category cut on ``Y1``, the rest stacked along ``Y2``, all matching target
proportions when ``rho = 0``), and `from_flag` (any lithotype-rule image).

Conditioning
------------
Categorical data are turned into latent Gaussian values by a Gibbs sampler
(Armstrong et al., 2003; Emery, 2007): every datum's ``A`` and ``B`` values
are repeatedly redrawn from their exact simple-kriging conditional
distribution given all other data (from the inverse covariance matrix),
truncated to the part of the rule that gives the observed category. Each
realization gets its own Gibbs chain with its own random stream (chains run
vectorized across realizations, sharing only the sweep order), and the
latent fields are then simulated conditionally on those values, so data
categories are reproduced at their locations. Realization ``i`` depends
only on `seed`, not on `nsim`. On a
lattice, data are moved to their nearest node first (a node with data of
different categories takes the most frequent one).
"""

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from .fftsim import FFTSimulator, _infer_lattice
from .lusim import _cov_block
from .sgsim import _get_random_generator, _make_rotation_matrix
from .tbsim import TBSimulator

__all__ = [
    "TruncationRule",
    "PluriGaussianSimulator",
    "pgsim",
    "category_probabilities",
    "category_proportions",
]


# ---------------------------------------------------------------- rule


class TruncationRule:
    """
    Truncation (lithotype) rule: rectangles in probability space mapped to
    categories.

    Parameters
    ----------
    rectangles : sequence of (category, u1_lo, u1_hi, u2_lo, u2_hi)
        Boxes in ``[0, 1]^2`` with ``u_k = Phi(y_k)``. Together they must tile
        the unit square. A category may own several boxes.

    Examples
    --------
    >>> rule = TruncationRule.hierarchical({"sand": 0.3, "silt": 0.5, "shale": 0.2})
    >>> rule.proportions()
    {'sand': 0.3, 'silt': 0.5, 'shale': 0.2}
    """

    def __init__(self, rectangles: Sequence):
        rects = []
        for r in rectangles:
            cat, a, b, c, d = r
            a, b, c, d = map(float, (a, b, c, d))
            if not (0.0 <= a < b <= 1.0 and 0.0 <= c < d <= 1.0):
                raise ValueError(f"rectangle {r} must satisfy 0 <= lo < hi <= 1 on both axes")
            rects.append((cat, a, b, c, d))
        area = sum((b - a) * (d - c) for _, a, b, c, d in rects)
        if abs(area - 1.0) > 1e-9:
            raise ValueError(f"rectangles must tile the unit square (total area {area:.6g})")
        self.rectangles = rects
        self.categories = list(dict.fromkeys(r[0] for r in rects))
        # Gaussian-space boxes, one row per rectangle: code, y1lo, y1hi, y2lo, y2hi
        codes = {c: i for i, c in enumerate(self.categories)}
        self._boxes = np.array(
            [[codes[c], ndtri(a), ndtri(b), ndtri(lo), ndtri(hi)]
             for c, a, b, lo, hi in rects]
        )

    @property
    def n_fields(self) -> int:
        """1 if no rectangle cuts the second axis (truncated Gaussian), else 2."""
        return 1 if all(c == 0.0 and d == 1.0 for _, _, _, c, d in self.rectangles) else 2

    @classmethod
    def truncated_gaussian(cls, proportions: Dict) -> "TruncationRule":
        """One latent field; categories in the given order along ``Y1``."""
        p = _normalise(proportions)
        edges = np.concatenate([[0.0], np.cumsum(list(p.values()))])
        edges[-1] = 1.0
        return cls([(c, edges[i], edges[i + 1], 0.0, 1.0) for i, c in enumerate(p)])

    @classmethod
    def hierarchical(cls, proportions: Dict) -> "TruncationRule":
        """
        First category where ``U1 < p_first``; the others stacked along ``Y2``
        in the remaining strip. The first category can touch all others; the
        others touch only their neighbours. Proportions are exact for
        ``rho = 0``.
        """
        p = _normalise(proportions)
        cats = list(p)
        if len(cats) < 2:
            raise ValueError("hierarchical rule needs at least two categories")
        p0 = p[cats[0]]
        rest = np.array([p[c] for c in cats[1:]]) / (1.0 - p0)
        edges = np.concatenate([[0.0], np.cumsum(rest)])
        edges[-1] = 1.0
        rects = [(cats[0], 0.0, p0, 0.0, 1.0)]
        rects += [(c, p0, 1.0, edges[i], edges[i + 1]) for i, c in enumerate(cats[1:])]
        return cls(rects)

    @classmethod
    def from_flag(cls, flag, categories: Optional[Sequence] = None,
                  u1_edges=None, u2_edges=None) -> "TruncationRule":
        """
        Rule from a lithotype-rule image: ``flag[i, j]`` is the category of
        the cell between ``u1_edges[i:i+2]`` (rows, ``Y1``) and
        ``u2_edges[j:j+2]`` (columns, ``Y2``); equal-width cells by default.
        Integer flags index into `categories` when it is given.
        """
        flag = np.asarray(flag, dtype=object)
        if flag.ndim != 2:
            raise ValueError("flag must be a 2-D array")
        n1, n2 = flag.shape
        e1 = np.linspace(0, 1, n1 + 1) if u1_edges is None else np.asarray(u1_edges, float)
        e2 = np.linspace(0, 1, n2 + 1) if u2_edges is None else np.asarray(u2_edges, float)
        if len(e1) != n1 + 1 or len(e2) != n2 + 1:
            raise ValueError("u1_edges / u2_edges must have one more entry than flag rows / columns")
        rects = []
        for i in range(n1):
            for j in range(n2):
                c = flag[i, j]
                if categories is not None:
                    c = categories[int(c)]
                rects.append((c, e1[i], e1[i + 1], e2[j], e2[j + 1]))
        return cls(rects)

    def proportions(self, rho: float = 0.0) -> Dict:
        """Category proportions for latent correlation `rho`."""
        out = {c: 0.0 for c in self.categories}
        for (c, *_), (_, y1a, y1b, y2a, y2b) in zip(self.rectangles, self._boxes):
            out[c] += _box_probability(y1a, y1b, y2a, y2b, rho)
        return out

    def classify(self, y1: np.ndarray, y2: Optional[np.ndarray] = None) -> np.ndarray:
        """Category codes (index into `categories`) for latent values."""
        y1 = np.asarray(y1, dtype=float)
        y2 = np.zeros_like(y1) if y2 is None else np.asarray(y2, dtype=float)
        codes = np.full(y1.shape, -1, dtype=np.int64)
        for code, a, b, c, d in self._boxes:
            inside = (y1 >= a) & (y1 < b) & (y2 >= c) & (y2 < d)
            codes[inside] = int(code)
        return codes

    def __repr__(self) -> str:
        p = ", ".join(f"{c!r}: {v:.3f}" for c, v in self.proportions().items())
        return f"TruncationRule({len(self.rectangles)} boxes, {self.n_fields} field(s), {{{p}}})"


def _normalise(proportions: Dict) -> Dict:
    p = {k: float(v) for k, v in dict(proportions).items()}
    if any(v <= 0 for v in p.values()):
        raise ValueError("proportions must be positive")
    total = sum(p.values())
    return {k: v / total for k, v in p.items()}


def _box_probability(y1a, y1b, y2a, y2b, rho: float) -> float:
    if rho == 0.0:
        return float((ndtr(y1b) - ndtr(y1a)) * (ndtr(y2b) - ndtr(y2a)))
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    big = 40.0

    def F(x, y):
        return mvn.cdf([np.clip(x, -big, big), np.clip(y, -big, big)])

    return float(max(F(y1b, y2b) - F(y1a, y2b) - F(y1b, y2a) + F(y1a, y2a), 0.0))


# ------------------------------------------------- truncated normal sampling


def _sample_truncated_union(mu, sigma, lo, hi, rng=None, u=None):
    """
    Vectorized draw from N(mu, sigma^2) restricted to a union of intervals.

    mu, sigma : (k,);  lo, hi : (k, n_int) interval bounds (empty if lo >= hi).
    Uniforms come from `u` (shape (2, k): interval pick, position) or `rng`.
    Returns (k,) values, or NaN where the union carries no mass.
    """
    if u is None:
        u = rng.uniform(size=(2, len(mu)))
    a = (lo - mu[:, None]) / sigma[:, None]
    b = (hi - mu[:, None]) / sigma[:, None]
    valid = b > a
    # mass per interval, computed on the side of the tail that keeps precision
    upper = a > 0
    pa = np.where(upper, ndtr(-b), ndtr(a))
    pb = np.where(upper, ndtr(-a), ndtr(b))
    mass = np.where(valid, pb - pa, 0.0)
    total = mass.sum(axis=1)
    cum = np.cumsum(mass, axis=1)
    pick = (cum < (u[0] * total)[:, None]).sum(axis=1)
    pick = np.minimum(pick, mass.shape[1] - 1)
    rows = np.arange(len(mu))
    lo_p, hi_p, up = pa[rows, pick], pb[rows, pick], upper[rows, pick]
    p = lo_p + u[1] * (hi_p - lo_p)
    p = np.clip(p, 1e-300, 1 - 1e-16)
    z = np.where(up, -ndtri(p), ndtri(p))
    z = np.clip(z, a[rows, pick], b[rows, pick])
    out = mu + sigma * z
    out[total <= 0] = np.nan
    return out


# ------------------------------------------------------------- simulator


class PluriGaussianSimulator:
    """
    Plurigaussian simulator of categorical variables.

    Parameters
    ----------
    rule : TruncationRule
        Truncation rule; ``rule.n_fields`` decides whether one or two latent
        fields are simulated.
    vario1, vario2 : sequence
        GSLIB-style variograms of the independent latent fields ``A`` and
        ``B`` (``[azimuth, nugget, major, minor, sill, vtype(, smoothness)]``,
        sill must be 1). `vario2` is ignored for one-field rules.
    rho : float, default 0.0
        Correlation between ``Y1`` and ``Y2``.
    method : {'auto', 'fft', 'tb'}, default 'auto'
        Latent-field simulator: `fftsim` on a regular lattice ('auto' picks
        it when the grid is one), otherwise `tbsim`.
    gibbs_iter : int, default 200
        Gibbs sweeps over the data per realization.
    nlines : int, default 1000
        Turning bands lines (``method='tb'`` only).
    max_cells : int, default 2**24
        Largest FFT embedding (``method='fft'`` only).
    spacing : float or (dx, dy), optional
        Lattice spacing for ``method='fft'`` when it cannot be inferred.

    Attributes
    ----------
    categories_ : list
        Category labels; realizations hold their codes (indices into it).
    latent_data_ : (A, B) arrays (n_data, nsim)
        Gibbs latent values at the data from the last `simulate` call.

    Examples
    --------
    >>> rule = TruncationRule.hierarchical({"sand": 0.3, "silt": 0.5, "shale": 0.2})
    >>> pgs = PluriGaussianSimulator(rule, [0, 0, 60, 30, 1, "gaussian"],
    ...                              [90, 0, 40, 40, 1, "spherical"])
    >>> codes = pgs.fit(grid, df, "X", "Y", "facies").simulate(nsim=20, seed=1)
    >>> labels = np.asarray(pgs.categories_)[codes]
    """

    def __init__(self, rule: TruncationRule, vario1: Sequence, vario2: Optional[Sequence] = None,
                 rho: float = 0.0, method: str = "auto", gibbs_iter: int = 200,
                 nlines: int = 1000, max_cells: int = 2 ** 24, spacing=None):
        if not isinstance(rule, TruncationRule):
            raise TypeError("rule must be a TruncationRule")
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if method not in ("auto", "fft", "tb"):
            raise ValueError("method must be 'auto', 'fft', or 'tb'")
        n_fields = rule.n_fields
        if n_fields == 2 and vario2 is None:
            raise ValueError("this rule uses two latent fields; pass vario2")
        varios = [list(vario1)] + ([list(vario2)] if n_fields == 2 else [])
        for v in varios:
            if abs(float(v[4]) - 1.0) > 1e-9:
                raise ValueError("latent field variograms must have sill 1 (total C(0))")
        self.rule = rule
        self.varios = varios
        self.rho = float(rho) if n_fields == 2 else 0.0
        self.method = method
        self.gibbs_iter = int(gibbs_iter)
        self.nlines = nlines
        self.max_cells = max_cells
        self.spacing = spacing

    # ------------------------------------------------------------------ fit

    def fit(self, prediction_grid: np.ndarray, df: Optional[pd.DataFrame] = None,
            xx: str = "X", yy: str = "Y", cat: str = "facies") -> "PluriGaussianSimulator":
        """Set up latent-field simulators and the Gibbs sampler for data `df[cat]`."""
        grid = np.asarray(prediction_grid, dtype=float).reshape(-1, 2)
        self.categories_ = list(self.rule.categories)
        method = self.method
        if method == "auto":
            try:
                _infer_lattice(grid, self.spacing)
                method = "fft"
            except ValueError:
                method = "tb"
        self.method_ = method
        self.n_nodes_ = len(grid)

        has_data = df is not None and len(df) > 0
        self._has_data = has_data
        if has_data:
            xy, codes = self._prepare_data(grid, df, xx, yy, cat, method)
            self._data_xy, self._data_codes = xy, codes
            dummy = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1], "Z": 0.0})
        else:
            dummy = None

        self._sims = []
        for v in self.varios:
            if method == "fft":
                s = FFTSimulator(v, mean=0.0, max_cells=self.max_cells, spacing=self.spacing)
            else:
                s = TBSimulator(v, mean=0.0, nlines=self.nlines)
            s.fit(grid, dummy, "X", "Y", "Z")
            self._sims.append(s)

        if has_data:
            # align data with each simulator's internal kriging order
            tree = cKDTree(self._data_xy)
            self._orders = [tree.query(s.conditioning_xy_)[1] for s in self._sims]
            # Gibbs: precision matrices of A and B at the data
            self._precision = []
            for v in self.varios:
                R = _make_rotation_matrix(v[0], v[2], v[3])
                C = _cov_block(self._data_xy, self._data_xy, v, R, same=True)
                C[np.diag_indices(len(C))] += 1e-8
                self._precision.append(cho_solve(cho_factor(C), np.eye(len(C))))
            boxes = self.rule._boxes
            self._data_boxes = [boxes[boxes[:, 0] == c, 1:] for c in range(len(self.categories_))]
        return self

    def _prepare_data(self, grid, df, xx, yy, cat, method):
        labels = df[cat].to_numpy()
        unknown = set(labels) - set(self.categories_)
        if unknown:
            raise ValueError(f"data categories {sorted(map(str, unknown))} are not in the rule")
        codes = np.array([self.categories_.index(c) for c in labels])
        xy = df[[xx, yy]].to_numpy(dtype=float)
        if method == "fft":
            x0, y0, dx, dy = _infer_lattice(grid, self.spacing)
            xy = np.column_stack([x0 + np.rint((xy[:, 0] - x0) / dx) * dx,
                                  y0 + np.rint((xy[:, 1] - y0) / dy) * dy])
        keys, inv = np.unique(np.round(xy, 9), axis=0, return_inverse=True)
        inv = inv.ravel()
        if len(keys) < len(xy):
            node_codes = np.empty(len(keys), dtype=np.int64)
            conflicts = 0
            for k in range(len(keys)):
                vals = np.bincount(codes[inv == k], minlength=len(self.categories_))
                node_codes[k] = int(vals.argmax())
                conflicts += int((vals > 0).sum() > 1)
            if conflicts:
                warnings.warn(f"{conflicts} location(s) hold data of different categories; "
                              "kept the most frequent one", stacklevel=3)
            return keys, node_codes
        return xy, codes

    # -------------------------------------------------------------- gibbs

    def _intervals(self, which: int, codes: np.ndarray, other: np.ndarray):
        """Allowed intervals for A (which=0) or B (which=1) at one datum."""
        rho, s = self.rho, np.sqrt(1.0 - self.rho ** 2)
        boxes = self._data_boxes[codes]
        y1a, y1b, y2a, y2b = boxes.T
        k = len(other)
        if self.rule.n_fields == 1:
            return np.broadcast_to(y1a, (k, len(y1a))), np.broadcast_to(y1b, (k, len(y1b)))
        o = other[:, None]
        if which == 0:                                       # A given B
            lo, hi = np.broadcast_to(y1a, (k, len(y1a))).copy(), np.broadcast_to(y1b, (k, len(y1b))).copy()
            if rho > 0:
                lo = np.maximum(lo, (y2a - s * o) / rho)
                hi = np.minimum(hi, (y2b - s * o) / rho)
            elif rho < 0:
                lo = np.maximum(lo, (y2b - s * o) / rho)
                hi = np.minimum(hi, (y2a - s * o) / rho)
            else:
                ok = (o >= y2a) & (o < y2b)
                hi = np.where(ok, hi, lo)
        else:                                                # B given A
            inside = (o >= y1a) & (o < y1b)
            lo = np.where(inside, (y2a - rho * o) / s, 0.0)
            hi = np.where(inside, (y2b - rho * o) / s, 0.0)
        return lo, hi

    def _gibbs(self, nsim: int, rng: np.random.Generator, seeds=None):
        """
        Gibbs latent values at the data, one chain per realization.

        Chain ``r`` draws all its uniforms from its own generator
        ``seeds[r]`` (drawn from `rng` when not given); `rng` only sets the
        sweep order, shared by all chains. Chain ``r`` therefore does not
        depend on `nsim`, while the update stays vectorized across chains.
        """
        n, nf = len(self._data_codes), self.rule.n_fields
        rho, s = self.rho, np.sqrt(1.0 - self.rho ** 2)
        if seeds is None:
            seeds = rng.integers(0, 2**31 - 1, size=nsim)
        gens = [_get_random_generator(int(sd)) for sd in seeds]

        def uniforms(shape):                     # (*shape, nsim), column r from chain r
            return np.stack([g.uniform(size=shape) for g in gens], axis=-1)

        latent = [np.zeros((n, nsim)) for _ in range(nf)]
        # valid start: pick a box of the observed category, A in its Y1 range,
        # then B in the matching Y2 range
        u0 = uniforms((n, 1 + 2 * nf))
        for i in range(n):
            boxes = self._data_boxes[self._data_codes[i]]
            pick = np.minimum((u0[i, 0] * len(boxes)).astype(np.int64), len(boxes) - 1)
            y1a, y1b, y2a, y2b = boxes[pick].T
            ones, zeros = np.ones(nsim), np.zeros(nsim)
            a = _sample_truncated_union(zeros, ones, y1a[:, None], y1b[:, None], u=u0[i, 1:3])
            latent[0][i] = a
            if nf == 2:
                latent[1][i] = _sample_truncated_union(
                    zeros, ones, ((y2a - rho * a) / s)[:, None], ((y2b - rho * a) / s)[:, None],
                    u=u0[i, 3:5])
        Q = self._precision
        for _ in range(self.gibbs_iter):
            u = uniforms((n, nf, 2))
            for i in rng.permutation(n):
                code = self._data_codes[i]
                for f in range(nf):
                    q_ii = Q[f][i, i]
                    mu = -(Q[f][i] @ latent[f] - q_ii * latent[f][i]) / q_ii
                    sigma = np.full(nsim, 1.0 / np.sqrt(q_ii))
                    other = latent[1 - f][i] if nf == 2 else latent[0][i]
                    lo, hi = self._intervals(f, code, other)
                    new = _sample_truncated_union(mu, sigma, lo, hi, u=u[i, f])
                    keep = np.isnan(new)
                    latent[f][i] = np.where(keep, latent[f][i], new)
        return latent

    # ---------------------------------------------------------- simulate

    def simulate(self, nsim: int = 1,
                 seed: Optional[Union[int, np.random.Generator]] = None,
                 return_latent: bool = False):
        """
        Draw `nsim` categorical realizations, shape ``(nsim, n_nodes)``, as
        codes into `categories_`. With `return_latent`, also return the
        latent fields ``(Y1, Y2)`` (``Y2`` is None for one-field rules).
        """
        if not hasattr(self, "_sims"):
            raise RuntimeError("call fit() first")
        if nsim < 1:
            raise ValueError("nsim must be >= 1")
        # Realization r uses Gibbs chain seed chain_seeds[r] and the r-th
        # sub-generator of each latent-field simulator (fftsim / tbsim draw
        # one sub-seed per realization or pair, so their first realizations
        # do not depend on nsim either): realization r is independent of nsim.
        rng = _get_random_generator(seed)
        seeds = rng.integers(0, 2**31 - 1, size=3)
        chain_seeds = rng.integers(0, 2**31 - 1, size=nsim)
        fields = []
        if self._has_data:
            latent = self._gibbs(nsim, _get_random_generator(int(seeds[0])), chain_seeds)
            self.latent_data_ = tuple(latent)
            for f, sim in enumerate(self._sims):
                vals = latent[f][self._orders[f]]
                fields.append(sim.simulate(nsim, int(seeds[f + 1]), data_values=vals))
        else:
            for f, sim in enumerate(self._sims):
                fields.append(sim.simulate(nsim, int(seeds[f + 1])))
        y1 = fields[0]
        y2 = None
        if self.rule.n_fields == 2:
            y2 = self.rho * fields[0] + np.sqrt(1.0 - self.rho ** 2) * fields[1]
        codes = self.rule.classify(y1, y2)
        return (codes, (y1, y2)) if return_latent else codes


def pgsim(
    prediction_grid: np.ndarray,
    df: Optional[pd.DataFrame],
    xx: str,
    yy: str,
    cat: str,
    rule: TruncationRule,
    vario1: Sequence,
    vario2: Optional[Sequence] = None,
    rho: float = 0.0,
    nsim: int = 1,
    method: str = "auto",
    gibbs_iter: int = 200,
    seed: Optional[Union[int, np.random.Generator]] = None,
    labels: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Plurigaussian simulation in one call.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
    df : pandas.DataFrame or None
        Categorical conditioning data (column `cat`); None for unconditional.
    xx, yy, cat : str
        Column names for x, y and the category.
    rule : TruncationRule
    vario1, vario2 : sequence
        Variograms (sill 1) of the latent fields; see `PluriGaussianSimulator`.
    rho : float, default 0.0
    nsim : int, default 1
    method : {'auto', 'fft', 'tb'}, default 'auto'
    gibbs_iter : int, default 200
    seed : int, numpy.random.Generator, or None
    labels : bool, default False
        Return category labels instead of integer codes.
    **kwargs
        Passed to `PluriGaussianSimulator` (``nlines``, ``max_cells``, ``spacing``).

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
        Category codes (index into ``rule.categories``) or labels.
    """
    pgs = PluriGaussianSimulator(rule, vario1, vario2, rho=rho, method=method,
                                 gibbs_iter=gibbs_iter, **kwargs)
    codes = pgs.fit(prediction_grid, df, xx, yy, cat).simulate(nsim, seed)
    return np.asarray(pgs.categories_, dtype=object)[codes] if labels else codes


def category_probabilities(sims: np.ndarray, n_categories: Optional[int] = None) -> np.ndarray:
    """Per-node frequency of each category code over realizations, shape (n_categories, n_nodes)."""
    sims = np.asarray(sims)
    k = int(sims.max()) + 1 if n_categories is None else int(n_categories)
    return np.stack([(sims == c).mean(axis=0) for c in range(k)])


def category_proportions(sims: np.ndarray, n_categories: Optional[int] = None) -> np.ndarray:
    """Global proportion of each category code per realization, shape (nsim, n_categories)."""
    sims = np.atleast_2d(np.asarray(sims))
    k = int(sims.max()) + 1 if n_categories is None else int(n_categories)
    return np.stack([(sims == c).mean(axis=1) for c in range(k)], axis=1)
