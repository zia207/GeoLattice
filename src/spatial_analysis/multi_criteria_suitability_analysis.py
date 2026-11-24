"""
Multi-criteria suitability analysis (MCDA) for rasters and arrays.

Implements the workflows taught in the ``03_05_multi_criteria_suitability_modeling``
tutorials and mirrors ArcGIS Spatial Analyst suitability tools:

**Part I – categorical & weighted (vector / array)**

- ``categorical_and`` / ``categorical_or`` / ``criteria_count`` — Boolean screens
- ``minmax_normalize`` / ``percentile_rank_normalize`` — criterion standardization
- ``weighted_linear_combination`` — WLC scoring
- ``hybrid_suitability`` — hard constraints + soft ranking

**Part II – fuzzy logic**

- Array memberships: ``gaussian``, ``near``, ``small``, ``large``, ``linear``,
  ``ms_large``, ``ms_small`` (ArcGIS Fuzzy Membership)
- Hedges: ``very``, ``somewhat``
- ``overlay`` / ``weighted_fuzzy_overlay`` — AND, OR, Product, Sum, Gamma
- ``mamdani`` / ``triangular_mf`` / ``trapezoidal_mf`` / ``interp_membership``
- File-based C kernels: ``fuzzy_membership``, ``fuzzy_overlay``

**Part III – sensitivity & uncertainty**

- ``perturb_weights``, ``monte_carlo_weights``
- ``jaccard_index``, ``top_quantile_mask``, ``limiting_criterion``

**C-backed raster file API** (``_suitability`` extension)

- ``reclassify``, ``rescale_by_function``, ``weighted_overlay``, ``weighted_sum``
- ``fuzzy_membership``, ``fuzzy_overlay``, ``locate_regions``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.raster.basic_raster_operation import RasterOperationError, _band_nodata
from geolattice.raster.raster_read_write import open_raster_data, write_raster_data

try:
    # Relative import avoids circular import while package __init__ is loading.
    from . import _suitability as _c
except ImportError as exc:  # pragma: no cover - build/install issue
    raise ImportError(
        "The geolattice.spatial_analysis._suitability C extension is required for "
        "suitability analysis. Rebuild the package with "
        '`pip install -e ".[dev]"` or `python setup.py build_ext --inplace`.'
    ) from exc

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional for Series wrapping
    pd = None  # type: ignore[assignment]

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, Sequence[float], float, int]
WeightLike = Union[Sequence[float], Mapping[str, float], np.ndarray]

_FUZZY_MEMBERSHIP_TYPES = {
    "gaussian",
    "near",
    "large",
    "small",
    "mslarge",
    "ms_large",
    "mssmall",
    "ms_small",
    "linear",
}

_FUZZY_OVERLAY_TYPES = {"and", "or", "product", "sum", "gamma"}

_RESCALE_FUNCTIONS = _FUZZY_MEMBERSHIP_TYPES | {
    "exponential",
    "logarithm",
    "log",
    "power",
    "logistic_growth",
    "logisticgrowth",
    "logistic_decay",
    "logisticdecay",
    "symmetric_linear",
    "symmetriclinear",
}

_LOCATE_SHAPES = {"circle", "ellipse", "square"}


class SuitabilityAnalysisError(RasterOperationError):
    """Raised when a suitability-analysis operation cannot be completed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _wrap(out: np.ndarray, like: Any) -> Any:
    """Clip memberships to [0, 1] and preserve pandas Series when available."""
    out = np.clip(out, 0.0, 1.0)
    if pd is not None and isinstance(like, pd.Series):
        return pd.Series(out, index=like.index, name=like.name)
    return out


def _as_bool_stack(criteria: Sequence[ArrayLike]) -> np.ndarray:
    if not criteria:
        raise SuitabilityAnalysisError("At least one criterion flag array is required.")
    return np.stack([np.asarray(c, dtype=bool) for c in criteria], axis=0)


def _normalize_weights(weights: WeightLike, *, n: Optional[int] = None) -> np.ndarray:
    if isinstance(weights, Mapping):
        if n is not None and len(weights) != n:
            raise SuitabilityAnalysisError(
                f"Expected {n} weights, got {len(weights)}."
            )
        w = np.asarray(list(weights.values()), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if n is not None and w.size != n:
            raise SuitabilityAnalysisError(f"Expected {n} weights, got {w.size}.")
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise SuitabilityAnalysisError("Weights must sum to a positive finite value.")
    return w / total


def _read_band(
    source: PathLike,
    band: int = 1,
) -> Tuple[np.ndarray, Any, Tuple[float, ...], Optional[float]]:
    with open_raster_data(source) as dataset:
        array = np.asarray(dataset.read(bands=band), dtype=np.float64)
        return array, dataset.crs, dataset.geotransform, _band_nodata(dataset, band)


def _write_like(
    destination: PathLike,
    array: np.ndarray,
    *,
    crs: Any,
    geotransform: Tuple[float, ...],
    nodata: Optional[float],
    overwrite: bool,
    dtype: Any = np.float32,
) -> str:
    out = np.asarray(array, dtype=dtype)
    return write_raster_data(
        destination,
        out,
        crs=crs,
        geotransform=geotransform,
        nodata=nodata,
        overwrite=overwrite,
    )


def _rules_to_array(mapping: Sequence[Tuple[Any, ...]]) -> np.ndarray:
    if not mapping:
        raise SuitabilityAnalysisError("mapping must contain at least one rule.")
    rows: List[List[float]] = []
    width = None
    for rule in mapping:
        if len(rule) == 2:
            row = [float(rule[0]), float(rule[1])]
        elif len(rule) == 3:
            row = [float(rule[0]), float(rule[1]), float(rule[2])]
        else:
            raise SuitabilityAnalysisError(
                f"Invalid reclassify rule {rule!r}; use (value, new) or (min, max, new)."
            )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise SuitabilityAnalysisError("All reclassify rules must have the same length.")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _load_aligned_layers(
    sources: Sequence[PathLike],
    *,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], Any, Tuple[float, ...], Optional[float]]:
    if not sources:
        raise SuitabilityAnalysisError("At least one source raster is required.")
    arrays: List[np.ndarray] = []
    crs = None
    gt = None
    nodata = None
    shape = None
    for i, source in enumerate(sources):
        band = 1 if bands is None else int(bands[i])
        array, layer_crs, layer_gt, layer_nodata = _read_band(source, band=band)
        if shape is None:
            shape = array.shape
            crs = layer_crs
            gt = layer_gt
            nodata = layer_nodata
        else:
            if array.shape != shape:
                raise SuitabilityAnalysisError(
                    f"Raster {source!r} shape {array.shape} does not match {shape}."
                )
            if layer_gt != gt:
                raise SuitabilityAnalysisError(
                    f"Raster {source!r} geotransform does not match the first layer."
                )
        arrays.append(array)
    assert crs is not None and gt is not None
    return arrays, crs, gt, nodata


# ===========================================================================
# Part I — Categorical (Boolean) suitability
# ===========================================================================


def categorical_and(criteria: Sequence[ArrayLike]) -> np.ndarray:
    """
    Logical AND of Boolean criterion flags (all must pass).

    Parameters
    ----------
    criteria:
        Sequence of Boolean arrays with a common shape.
    """
    stack = _as_bool_stack(criteria)
    return np.all(stack, axis=0)


def categorical_or(criteria: Sequence[ArrayLike]) -> np.ndarray:
    """Logical OR of Boolean criterion flags (any may pass)."""
    stack = _as_bool_stack(criteria)
    return np.any(stack, axis=0)


def criteria_count(criteria: Sequence[ArrayLike]) -> np.ndarray:
    """
    Count how many criteria each location meets (k-of-n ordinal score).

    Useful for near-miss diagnosis between the AND and OR extremes.
    """
    stack = _as_bool_stack(criteria)
    return stack.sum(axis=0).astype(np.int32)


# ===========================================================================
# Part I — Weighted linear combination
# ===========================================================================


def minmax_normalize(
    values: ArrayLike,
    *,
    benefit: bool = True,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> Any:
    """
    Min–max normalize values to ``[0, 1]``.

    For benefit criteria higher raw values score higher; for cost criteria
    (``benefit=False``) the scale is reversed. Optional ``lo`` / ``hi`` fix
    the range (e.g. RUCC codes 1–9) instead of using the observed extrema.
    """
    like = values
    v = _arr(values)
    v_lo = float(np.nanmin(v)) if lo is None else float(lo)
    v_hi = float(np.nanmax(v)) if hi is None else float(hi)
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi == v_lo:
        raise SuitabilityAnalysisError("minmax_normalize requires a finite non-zero range.")
    scaled = (v - v_lo) / (v_hi - v_lo)
    out = scaled if benefit else 1.0 - scaled
    if pd is not None and isinstance(like, pd.Series):
        return pd.Series(out, index=like.index, name=like.name)
    return out


def percentile_rank_normalize(
    values: ArrayLike,
    *,
    benefit: bool = True,
) -> Any:
    """
    Percentile-rank normalize values to ``[0, 1]`` (ties share the average rank).

    More robust to outliers than min–max when a few extreme observations would
    otherwise compress the rest of the distribution.
    """
    like = values
    if pd is not None and isinstance(values, pd.Series):
        r = values.rank(pct=True, method="average")
        r = (r - r.min()) / (r.max() - r.min())
        return r if benefit else 1.0 - r

    v = _arr(values).ravel()
    order = v.argsort(kind="mergesort")
    ranks = np.empty_like(v, dtype=np.float64)
    # average ranks for ties
    sorted_v = v[order]
    i = 0
    n = v.size
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        avg = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg
        i = j
    pct = ranks / max(n - 1, 1)
    pct = (pct - pct.min()) / max(pct.max() - pct.min(), 1e-15)
    out = pct if benefit else 1.0 - pct
    return out.reshape(np.shape(values))


def weighted_linear_combination(
    layers: Union[Sequence[ArrayLike], Mapping[str, ArrayLike], np.ndarray],
    weights: WeightLike,
) -> np.ndarray:
    """
    Weighted linear combination (WLC): ``sum(w_i * x_i)`` with weights
    normalized to sum to 1.

    ``layers`` may be a sequence of arrays, a mapping of name → array (weights
    keyed by the same names), or a stacked array of shape ``(n_criteria, ...)``.
    """
    if isinstance(layers, Mapping):
        if not isinstance(weights, Mapping):
            raise SuitabilityAnalysisError(
                "When layers is a mapping, weights must be a mapping of the same keys."
            )
        keys = list(layers.keys())
        stack = np.stack([_arr(layers[k]) for k in keys], axis=0)
        w = _normalize_weights({k: float(weights[k]) for k in keys})
    elif isinstance(layers, np.ndarray) and layers.ndim >= 2 and not isinstance(layers[0], (float, int)):
        # stacked criteria on axis 0
        stack = np.asarray(layers, dtype=np.float64)
        w = _normalize_weights(weights, n=stack.shape[0])
    else:
        seq = list(layers)  # type: ignore[arg-type]
        stack = np.stack([_arr(x) for x in seq], axis=0)
        w = _normalize_weights(weights, n=stack.shape[0])
    return np.tensordot(w, stack, axes=1)


def hybrid_suitability(
    scores: ArrayLike,
    constraints: ArrayLike,
    *,
    infeasible: float = np.nan,
) -> np.ndarray:
    """
    Apply hard Boolean constraints to a soft suitability score.

    Locations where ``constraints`` is False receive ``infeasible`` (default
    NaN); others keep their score. Matches the hybrid constraint + WLC pattern
    from the weighted-suitability tutorial.
    """
    s = _arr(scores)
    ok = np.asarray(constraints, dtype=bool)
    if ok.shape != s.shape:
        raise SuitabilityAnalysisError("scores and constraints must share the same shape.")
    out = s.copy()
    out[~ok] = infeasible
    return out


def score_contributions(
    layers: Union[Sequence[ArrayLike], Mapping[str, ArrayLike]],
    weights: WeightLike,
) -> Dict[str, np.ndarray]:
    """
    Return per-criterion weighted contributions ``w_i * x_i``.

    Useful for explaining why a location ranks where it does.
    """
    if isinstance(layers, Mapping):
        if not isinstance(weights, Mapping):
            raise SuitabilityAnalysisError(
                "When layers is a mapping, weights must be a mapping of the same keys."
            )
        keys = list(layers.keys())
        w = _normalize_weights({k: float(weights[k]) for k in keys})
        return {k: float(w[i]) * _arr(layers[k]) for i, k in enumerate(keys)}
    seq = list(layers)  # type: ignore[arg-type]
    w = _normalize_weights(weights, n=len(seq))
    return {f"c{i}": float(w[i]) * _arr(x) for i, x in enumerate(seq)}


# ===========================================================================
# Part II — Fuzzy membership functions (array API; ArcGIS formulas)
# ===========================================================================


def gaussian(x: ArrayLike, midpoint: float, spread: float = 0.1) -> Any:
    """``μ = exp(-spread · (x − midpoint)²)``; peak of 1 at the midpoint."""
    v = _arr(x)
    return _wrap(np.exp(-spread * (v - midpoint) ** 2), x)


def near(x: ArrayLike, midpoint: float, spread: float = 0.1) -> Any:
    """``μ = 1 / (1 + spread · (x − midpoint)²)``; sharper peak, longer tails."""
    v = _arr(x)
    return _wrap(1.0 / (1.0 + spread * (v - midpoint) ** 2), x)


def small(x: ArrayLike, midpoint: float, spread: float = 5.0) -> Any:
    """``μ = 1 / (1 + (x / midpoint)^spread)``; small values → 1. Requires x ≥ 0."""
    v = _arr(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / (1.0 + (np.maximum(v, 0.0) / midpoint) ** spread)
    return _wrap(np.where(np.isnan(v), np.nan, out), x)


def large(x: ArrayLike, midpoint: float, spread: float = 5.0) -> Any:
    """``μ = 1 / (1 + (x / midpoint)^(-spread))``; large values → 1. Requires x ≥ 0."""
    v = _arr(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.where(v > 0, 1.0 / (1.0 + (v / midpoint) ** (-spread)), 0.0)
    return _wrap(np.where(np.isnan(v), np.nan, out), x)


def linear(x: ArrayLike, minimum: float, maximum: float) -> Any:
    """Linear ramp: 0 at ``minimum``, 1 at ``maximum`` (decreasing if min > max)."""
    v = _arr(x)
    return _wrap((v - minimum) / (maximum - minimum), x)


def ms_large(
    x: ArrayLike,
    mean_multiplier: float = 1.0,
    std_multiplier: float = 1.0,
    *,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> Any:
    """Mean/std-based increasing membership (ArcGIS MS Large).

    ``a`` / ``b`` are ArcGIS-style aliases for ``mean_multiplier`` /
    ``std_multiplier``.
    """
    v = _arr(x)
    m, s = float(np.nanmean(v)), float(np.nanstd(v))
    aa = float(mean_multiplier if a is None else a)
    bb = float(std_multiplier if b is None else b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(v > aa * m, 1.0 - (bb * s) / (v - aa * m + bb * s), 0.0)
    return _wrap(np.where(np.isnan(v), np.nan, out), x)


def ms_small(
    x: ArrayLike,
    mean_multiplier: float = 1.0,
    std_multiplier: float = 1.0,
    *,
    a: Optional[float] = None,
    b: Optional[float] = None,
) -> Any:
    """Mean/std-based decreasing membership (ArcGIS MS Small).

    ``a`` / ``b`` are ArcGIS-style aliases for ``mean_multiplier`` /
    ``std_multiplier``.
    """
    v = _arr(x)
    m, s = float(np.nanmean(v)), float(np.nanstd(v))
    aa = float(mean_multiplier if a is None else a)
    bb = float(std_multiplier if b is None else b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(v > aa * m, (bb * s) / (v - aa * m + bb * s), 1.0)
    return _wrap(np.where(np.isnan(v), np.nan, out), x)


def spread_for_small(midpoint: float, x_at_09: float) -> float:
    """
    Spread that gives ``μ = 0.9`` at ``x_at_09`` for the Small membership.

    Example: gentle slopes with midpoint 5 and μ(3) = 0.9 →
    ``spread_for_small(5, 3)``.
    """
    if midpoint <= 0 or x_at_09 <= 0:
        raise SuitabilityAnalysisError("midpoint and x_at_09 must be positive.")
    return float(np.log(1.0 / 9.0) / np.log(x_at_09 / midpoint))


def very(mu: ArrayLike) -> Any:
    """Concentration hedge: ``μ²`` (stricter membership)."""
    return _arr(mu) ** 2


def somewhat(mu: ArrayLike) -> Any:
    """Dilation hedge: ``√μ`` (more lenient membership)."""
    return _arr(mu) ** 0.5


def overlay(
    memberships: Sequence[ArrayLike],
    method: str = "gamma",
    gamma: float = 0.9,
) -> Any:
    """
    Combine fuzzy membership arrays or Series.

    ``method``: ``and`` (min), ``or`` (max), ``product``, ``sum``
    (algebraic sum ``1 − Π(1−μ)``), or ``gamma``
    (``Sum^γ · Product^(1−γ)``).
    """
    if not memberships:
        raise SuitabilityAnalysisError("At least one membership layer is required.")
    like = memberships[0]
    M = np.stack([_arr(m) for m in memberships])
    key = str(method).strip().lower()
    if key == "and":
        out = np.min(M, axis=0)
    elif key == "or":
        out = np.max(M, axis=0)
    elif key == "product":
        out = np.prod(M, axis=0)
    elif key == "sum":
        out = 1.0 - np.prod(1.0 - M, axis=0)
    elif key == "gamma":
        fsum = 1.0 - np.prod(1.0 - M, axis=0)
        fprod = np.prod(M, axis=0)
        out = fsum ** float(gamma) * fprod ** (1.0 - float(gamma))
    else:
        raise SuitabilityAnalysisError(
            f"Unknown overlay method {method!r}. Known: and, or, product, sum, gamma."
        )
    return _wrap(out, like)


def weighted_fuzzy_overlay(
    memberships: Sequence[ArrayLike],
    weights: WeightLike,
    *,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Weighted fuzzy Gamma operator.

    Weights are normalized to sum to 1, then scaled by *n* so that equal weights
    reproduce the standard (unweighted) Gamma overlay.
    """
    M = np.stack([_arr(m) for m in memberships])
    w = _normalize_weights(weights, n=M.shape[0]) * M.shape[0]
    # Broadcast weights over remaining axes
    shape = (M.shape[0],) + (1,) * (M.ndim - 1)
    ww = w.reshape(shape)
    wprod = np.prod(M ** ww, axis=0)
    wsum = 1.0 - np.prod((1.0 - M) ** ww, axis=0)
    return wsum ** float(gamma) * wprod ** (1.0 - float(gamma))


def limiting_criterion(
    memberships: Union[Sequence[ArrayLike], Mapping[str, ArrayLike]],
) -> np.ndarray:
    """
    Index (or name) of the weakest membership at each location (fuzzy AND limit).

    For a sequence, returns integer indices along axis 0. For a mapping, returns
    an object array of criterion names.
    """
    if isinstance(memberships, Mapping):
        keys = list(memberships.keys())
        M = np.stack([_arr(memberships[k]) for k in keys], axis=0)
        idx = np.nanargmin(M, axis=0)
        out = np.empty(idx.shape, dtype=object)
        for i, name in enumerate(keys):
            out[idx == i] = name
        return out
    M = np.stack([_arr(m) for m in memberships], axis=0)
    return np.nanargmin(M, axis=0)


# ===========================================================================
# Part II — Rule-based Mamdani FIS helpers
# ===========================================================================


def triangular_mf(x: ArrayLike, abc: Sequence[float]) -> np.ndarray:
    """Triangular membership: ``abc = (a, b, c)`` with peak at ``b``."""
    a, b, c = (float(v) for v in abc)
    v = _arr(x)
    out = np.zeros_like(v, dtype=np.float64)
    left = (a < v) & (v < b)
    right = (b < v) & (v < c)
    if b != a:
        out[left] = (v[left] - a) / (b - a)
    out[v == b] = 1.0
    if c != b:
        out[right] = (c - v[right]) / (c - b)
    return out


def trapezoidal_mf(x: ArrayLike, abcd: Sequence[float]) -> np.ndarray:
    """Trapezoidal membership: ``abcd = (a, b, c, d)`` with plateau ``[b, c]``."""
    a, b, c, d = (float(v) for v in abcd)
    v = _arr(x)
    out = np.zeros_like(v, dtype=np.float64)
    rise = (a < v) & (v < b)
    flat = (b <= v) & (v <= c)
    fall = (c < v) & (v < d)
    if b != a:
        out[rise] = (v[rise] - a) / (b - a)
    out[flat] = 1.0
    if d != c:
        out[fall] = (d - v[fall]) / (d - c)
    return out


def interp_membership(
    universe: ArrayLike,
    mf: ArrayLike,
    x: ArrayLike,
) -> np.ndarray:
    """Interpolate a discrete membership function at points ``x``."""
    u = _arr(universe)
    m = _arr(mf)
    return np.interp(_arr(x), u, m, left=m[0], right=m[-1])


def mamdani(
    rules: Sequence[Tuple[ArrayLike, str]],
    out_terms: Mapping[str, ArrayLike],
    universe: ArrayLike,
) -> np.ndarray:
    """
    Vectorized Mamdani inference with min implication, max aggregation, and
    centroid defuzzification.

    Parameters
    ----------
    rules:
        Sequence of ``(firing_strength, output_term_name)`` where firing
        strength is a 1-D array over evaluation locations.
    out_terms:
        Mapping of linguistic output term → membership samples on ``universe``.
    universe:
        1-D output universe (e.g. suitability 0–100).
    """
    if not rules:
        raise SuitabilityAnalysisError("rules must not be empty.")
    u = _arr(universe)
    terms = {k: _arr(v) for k, v in out_terms.items()}
    for name, mf in terms.items():
        if mf.shape != u.shape:
            raise SuitabilityAnalysisError(
                f"out_terms[{name!r}] shape {mf.shape} does not match universe {u.shape}."
            )
    strength0 = _arr(rules[0][0])
    n = strength0.size
    agg = np.zeros((n, u.size), dtype=np.float64)
    for strength, term in rules:
        s = _arr(strength)
        if s.size != n:
            raise SuitabilityAnalysisError("All rule firing strengths must share length.")
        if term not in terms:
            raise SuitabilityAnalysisError(f"Unknown output term {term!r}.")
        clipped = np.minimum(s[:, None], terms[term][None, :])
        agg = np.maximum(agg, clipped)
    num = (agg * u).sum(axis=1)
    den = agg.sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


# ===========================================================================
# Part III — Sensitivity and uncertainty helpers
# ===========================================================================


def top_quantile_mask(values: ArrayLike, q: float = 0.80, *, mask: Optional[ArrayLike] = None) -> np.ndarray:
    """Boolean mask of cells at or above the ``q`` quantile (optionally within ``mask``)."""
    v = _arr(values).copy()
    if mask is not None:
        v = np.where(np.asarray(mask, dtype=bool), v, np.nan)
    thr = np.nanquantile(v, q)
    return v >= thr


def jaccard_index(a: ArrayLike, b: ArrayLike) -> float:
    """Jaccard similarity of two Boolean masks: ``|A ∩ B| / |A ∪ B|``."""
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(aa, bb).sum() / union)


def perturb_weights(weights: WeightLike, index: int, delta: float) -> np.ndarray:
    """
    One-at-a-time weight perturbation with proportional renormalization.

    Multiplies ``weights[index]`` by ``(1 + delta)`` and rescales the remaining
    weights so the vector still sums to 1.
    """
    w = np.asarray(
        list(weights.values()) if isinstance(weights, Mapping) else weights,
        dtype=np.float64,
    ).copy()
    if index < 0 or index >= w.size:
        raise SuitabilityAnalysisError(f"index {index} out of range for {w.size} weights.")
    if w[index] >= 1.0:
        raise SuitabilityAnalysisError("Cannot perturb a weight that already equals 1.")
    new = w[index] * (1.0 + float(delta))
    if new <= 0 or new >= 1:
        raise SuitabilityAnalysisError("Perturbed weight must stay in (0, 1).")
    w *= (1.0 - new) / (1.0 - w[index])
    w[index] = new
    return w


def monte_carlo_weights(
    membership_stack: np.ndarray,
    baseline_weights: WeightLike,
    *,
    kappa: float = 50.0,
    n: int = 1000,
    batch: int = 100,
    top_q: float = 0.90,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Dirichlet Monte Carlo over weights around a baseline (concentration ``kappa``).

    ``membership_stack`` has shape ``(n_criteria, n_cells)``. Returns mean / SD
    suitability, P(top quantile), mean / SD percentile rank, sampled weights,
    and Jaccard vs the baseline top set.
    """
    M = np.asarray(membership_stack, dtype=np.float64)
    if M.ndim != 2:
        raise SuitabilityAnalysisError("membership_stack must be 2-D (criteria × cells).")
    w0 = _normalize_weights(baseline_weights, n=M.shape[0])
    rng = np.random.default_rng(seed)
    Ws = rng.dirichlet(float(kappa) * w0, int(n))
    base = w0 @ M
    base_top = base >= np.quantile(base, top_q)

    s = np.zeros(M.shape[1])
    s2 = np.zeros_like(s)
    topc = np.zeros_like(s)
    r = np.zeros_like(s)
    r2 = np.zeros_like(s)
    jac: List[float] = []
    n_sim = int(n)
    for b0 in range(0, n_sim, int(batch)):
        Wb = Ws[b0 : b0 + batch]
        V = Wb @ M
        s += V.sum(0)
        s2 += (V ** 2).sum(0)
        thr = np.quantile(V, top_q, axis=1, keepdims=True)
        T = V >= thr
        topc += T.sum(0)
        jac.extend(jaccard_index(t, base_top) for t in T)
        R = V.argsort(axis=1).argsort(axis=1) / max(V.shape[1] - 1, 1) * 100.0
        r += R.sum(0)
        r2 += (R ** 2).sum(0)
    mean = s / n_sim
    rank_mean = r / n_sim
    return {
        "W": Ws,
        "mean": mean,
        "sd": np.sqrt(np.maximum(s2 / n_sim - mean ** 2, 0.0)),
        "p_top": topc / n_sim,
        "rank_mean": rank_mean,
        "rank_sd": np.sqrt(np.maximum(r2 / n_sim - rank_mean ** 2, 0.0)),
        "jaccard": np.asarray(jac, dtype=np.float64),
        "baseline": base,
        "baseline_top": base_top,
    }


def block_mean(array: ArrayLike, factor: int) -> np.ndarray:
    """
    Aggregate an array by non-overlapping ``factor × factor`` block means.

    Used for resolution-sensitivity (MAUP / scale) experiments.
    """
    a = _arr(array)
    if factor < 1:
        raise SuitabilityAnalysisError("factor must be >= 1.")
    if factor == 1:
        return a.copy()
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    b = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.nanmean(np.nanmean(b, axis=3), axis=1)


# ===========================================================================
# C-backed file API (ArcGIS-style raster tools)
# ===========================================================================


def reclassify(
    source: PathLike,
    destination: PathLike,
    mapping: Sequence[Tuple[Any, ...]],
    *,
    band: int = 1,
    default: Optional[float] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Reclassify raster values with the C suitability kernel.

    ``mapping`` entries are ``(value, new_value)`` for exact matches or
    ``(min, max, new_value)`` for half-open ranges ``[min, max)``.
    """
    array, crs, gt, src_nodata = _read_band(source, band=band)
    out_nodata = nodata if nodata is not None else src_nodata
    rules = _rules_to_array(mapping)
    default_value = float("nan") if default is None else float(default)
    result = _c.reclassify(
        array,
        rules,
        default=default_value,
        nodata=None if out_nodata is None else float(out_nodata),
    )
    if out_nodata is not None:
        result = np.where(np.isnan(result), out_nodata, result)
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def rescale_by_function(
    source: PathLike,
    destination: PathLike,
    *,
    function: str = "linear",
    from_scale: float = 1.0,
    to_scale: float = 10.0,
    midpoint: Optional[float] = None,
    spread: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    mean_multiplier: float = 1.0,
    std_multiplier: float = 1.0,
    shift: Optional[float] = None,
    factor: Optional[float] = None,
    y_intercept_percent: float = 99.0,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    band: int = 1,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Rescale continuous raster values with a transformation function.

    Supported ``function`` values include ``gaussian``, ``near``, ``large``,
    ``small``, ``mslarge``, ``mssmall``, ``linear``, ``exponential``,
    ``logarithm``, ``power``, ``logistic_growth``, ``logistic_decay``, and
    ``symmetric_linear``. Output is scaled to ``[from_scale, to_scale]``.
    """
    key = str(function).strip().lower()
    if key not in _RESCALE_FUNCTIONS:
        known = ", ".join(sorted(_RESCALE_FUNCTIONS))
        raise SuitabilityAnalysisError(f"Unknown rescale function {function!r}. Known: {known}")

    array, crs, gt, src_nodata = _read_band(source, band=band)
    out_nodata = nodata if nodata is not None else src_nodata
    result = _c.rescale_by_function(
        array,
        function=key,
        from_scale=float(from_scale),
        to_scale=float(to_scale),
        midpoint=float("nan") if midpoint is None else float(midpoint),
        spread=float("nan") if spread is None else float(spread),
        minimum=float("nan") if minimum is None else float(minimum),
        maximum=float("nan") if maximum is None else float(maximum),
        mean_multiplier=float(mean_multiplier),
        std_multiplier=float(std_multiplier),
        shift=float("nan") if shift is None else float(shift),
        factor=float("nan") if factor is None else float(factor),
        y_intercept_percent=float(y_intercept_percent),
        lower_threshold=float("nan") if lower_threshold is None else float(lower_threshold),
        upper_threshold=float("nan") if upper_threshold is None else float(upper_threshold),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def weighted_sum(
    sources: Sequence[PathLike],
    destination: PathLike,
    weights: Sequence[float],
    *,
    bands: Optional[Sequence[int]] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Weighted sum of rasters: ``sum(raster_i * weight_i)``.

    Layers must share the same shape and geotransform.
    """
    if len(sources) != len(weights):
        raise SuitabilityAnalysisError("sources and weights must have the same length.")
    arrays, crs, gt, src_nodata = _load_aligned_layers(sources, bands=bands)
    out_nodata = nodata if nodata is not None else src_nodata
    result = _c.weighted_sum(
        arrays,
        np.asarray(weights, dtype=np.float64),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def weighted_overlay(
    sources: Sequence[PathLike],
    destination: PathLike,
    weights: Sequence[float],
    *,
    remap: Optional[Sequence[Sequence[Tuple[Any, ...]]]] = None,
    bands: Optional[Sequence[int]] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Weighted overlay of evaluation-scale rasters.

    Each input should already be on a common evaluation scale (e.g. 1–9), or
    pass ``remap`` with one reclassify mapping per layer. Result is
    ``round(sum(layer_i * weight_i) / sum(weights))``.
    """
    if len(sources) != len(weights):
        raise SuitabilityAnalysisError("sources and weights must have the same length.")
    arrays, crs, gt, src_nodata = _load_aligned_layers(sources, bands=bands)
    out_nodata = nodata if nodata is not None else src_nodata

    if remap is not None:
        if len(remap) != len(arrays):
            raise SuitabilityAnalysisError("remap must provide one mapping per source.")
        remapped: List[np.ndarray] = []
        for array, mapping in zip(arrays, remap):
            rules = _rules_to_array(mapping)
            remapped.append(
                _c.reclassify(
                    array,
                    rules,
                    default=float("nan"),
                    nodata=None if out_nodata is None else float(out_nodata),
                )
            )
        arrays = remapped

    result = _c.weighted_overlay(
        arrays,
        np.asarray(weights, dtype=np.float64),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def fuzzy_membership(
    source: PathLike,
    destination: PathLike,
    *,
    membership_type: str = "gaussian",
    midpoint: Optional[float] = None,
    spread: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    mean_multiplier: float = 1.0,
    std_multiplier: float = 1.0,
    band: int = 1,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Transform a continuous raster to a 0–1 fuzzy membership surface.

    ``membership_type``: ``gaussian``, ``near``, ``large``, ``small``,
    ``mslarge`` / ``ms_large``, ``mssmall`` / ``ms_small``, or ``linear``.
    """
    key = str(membership_type).strip().lower()
    if key not in _FUZZY_MEMBERSHIP_TYPES:
        known = ", ".join(sorted(_FUZZY_MEMBERSHIP_TYPES))
        raise SuitabilityAnalysisError(
            f"Unknown membership_type {membership_type!r}. Known: {known}"
        )

    array, crs, gt, src_nodata = _read_band(source, band=band)
    out_nodata = nodata if nodata is not None else src_nodata
    result = _c.fuzzy_membership(
        array,
        membership_type=key,
        midpoint=float("nan") if midpoint is None else float(midpoint),
        spread=float("nan") if spread is None else float(spread),
        minimum=float("nan") if minimum is None else float(minimum),
        maximum=float("nan") if maximum is None else float(maximum),
        mean_multiplier=float(mean_multiplier),
        std_multiplier=float(std_multiplier),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def fuzzy_overlay(
    sources: Sequence[PathLike],
    destination: PathLike,
    *,
    overlay_type: str = "and",
    gamma: float = 0.9,
    bands: Optional[Sequence[int]] = None,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Combine fuzzy membership rasters.

    ``overlay_type``: ``and`` (min), ``or`` (max), ``product``, ``sum``
    (algebraic sum ``1 - Π(1-μ)``), or ``gamma``.
    """
    key = str(overlay_type).strip().lower()
    if key not in _FUZZY_OVERLAY_TYPES:
        known = ", ".join(sorted(_FUZZY_OVERLAY_TYPES))
        raise SuitabilityAnalysisError(f"Unknown overlay_type {overlay_type!r}. Known: {known}")

    arrays, crs, gt, src_nodata = _load_aligned_layers(sources, bands=bands)
    out_nodata = nodata if nodata is not None else src_nodata
    result = _c.fuzzy_overlay(
        arrays,
        overlay_type=key,
        gamma=float(gamma),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=out_nodata,
        overwrite=overwrite,
    )


def locate_regions(
    source: PathLike,
    destination: PathLike,
    *,
    n_regions: int,
    region_size: Optional[int] = None,
    area: Optional[float] = None,
    cell_area: Optional[float] = None,
    shape: str = "circle",
    minimum_distance: float = 0.0,
    band: int = 1,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> str:
    """
    Turn a continuous suitability surface into discrete candidate sites.

    Answers questions like: "Where exactly should the three 50-hectare sites
    go?" Regions grow from the highest-suitability seeds until each reaches
    ``region_size`` cells (or an ``area`` converted with ``cell_area``).

    Parameters
    ----------
    n_regions:
        Number of sites to place.
    region_size:
        Target size of each region in cells.
    area:
        Alternative to ``region_size``: target area in the same units as
        ``cell_area`` (e.g. hectares). Requires ``cell_area``.
    cell_area:
        Area of one cell in the same units as ``area``. If omitted and the
        raster CRS is projected, derived from the geotransform.
    shape:
        Growth bias: ``circle``, ``ellipse``, or ``square``.
    minimum_distance:
        Minimum distance (in cells) between region seed centers.
    """
    shape_key = str(shape).strip().lower()
    if shape_key not in _LOCATE_SHAPES:
        known = ", ".join(sorted(_LOCATE_SHAPES))
        raise SuitabilityAnalysisError(f"Unknown shape {shape!r}. Known: {known}")
    if n_regions < 1:
        raise SuitabilityAnalysisError("n_regions must be >= 1.")

    array, crs, gt, src_nodata = _read_band(source, band=band)
    out_nodata = nodata if nodata is not None else src_nodata

    size = region_size
    if size is None:
        if area is None:
            raise SuitabilityAnalysisError("Provide region_size or area (+ cell_area).")
        ca = cell_area
        if ca is None:
            px = abs(float(gt[1]))
            py = abs(float(gt[5]))
            ca = px * py
            if ca <= 0:
                raise SuitabilityAnalysisError(
                    "Could not derive cell_area from geotransform; pass cell_area explicitly."
                )
        size = max(1, int(math.ceil(float(area) / float(ca))))

    result = _c.locate_regions(
        array,
        int(n_regions),
        int(size),
        shape=shape_key,
        minimum_distance=float(minimum_distance),
        nodata=None if out_nodata is None else float(out_nodata),
    )
    return _write_like(
        destination,
        result,
        crs=crs,
        geotransform=gt,
        nodata=0,
        overwrite=overwrite,
        dtype=np.int32,
    )


__all__ = [
    "SuitabilityAnalysisError",
    # Categorical / weighted (array)
    "categorical_and",
    "categorical_or",
    "criteria_count",
    "minmax_normalize",
    "percentile_rank_normalize",
    "weighted_linear_combination",
    "hybrid_suitability",
    "score_contributions",
    # Fuzzy array API
    "gaussian",
    "near",
    "small",
    "large",
    "linear",
    "ms_large",
    "ms_small",
    "spread_for_small",
    "very",
    "somewhat",
    "overlay",
    "weighted_fuzzy_overlay",
    "limiting_criterion",
    # Mamdani FIS
    "triangular_mf",
    "trapezoidal_mf",
    "interp_membership",
    "mamdani",
    # Sensitivity / uncertainty
    "top_quantile_mask",
    "jaccard_index",
    "perturb_weights",
    "monte_carlo_weights",
    "block_mean",
    # C-backed file API
    "reclassify",
    "rescale_by_function",
    "weighted_overlay",
    "weighted_sum",
    "fuzzy_membership",
    "fuzzy_overlay",
    "locate_regions",
]
