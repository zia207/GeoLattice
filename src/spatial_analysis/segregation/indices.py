"""Aspatial segregation indices: one number summarizing how unevenly groups are
distributed across a set of units. ``groups`` picks the count columns to use (two
columns for the single-group / two-group indices, any number for the multigroup ones).

All formulas follow the standard references (Massey & Denton 1988; James & Taeuber 1985;
Reardon & Firebaugh 2002) and are validated in ``tests/test_segregation.py`` against
hand-worked cases and, where ``segregation`` (PySAL) is installed, against its own values.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import extract_counts, global_proportions, local_proportions, unit_totals


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini coefficient of ``values``, via the Lorenz-curve trapezoid rule --
    ``O(n log n)`` from one sort, not the ``O(n^2)`` pairwise-difference formula it is
    algebraically equal to. Shared by :func:`gini` here and reused for any other
    population-weighted inequality measure in this package."""
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total_w = w.sum()
    total_wv = (w * v).sum()
    if total_w <= 0 or total_wv <= 0:
        return 0.0
    cum_w = np.concatenate([[0.0], np.cumsum(w)]) / total_w
    cum_wv = np.concatenate([[0.0], np.cumsum(w * v)]) / total_wv
    # area under the Lorenz curve via the trapezoid rule; Gini = 1 - 2 * area
    area = np.sum((cum_w[1:] - cum_w[:-1]) * (cum_wv[1:] + cum_wv[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)


def _two_group_counts(gdf, groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(groups) != 2:
        raise ValueError(f"Single-group indices need exactly 2 group columns, got {len(groups)}")
    counts = extract_counts(gdf, groups)
    return counts[:, 0], counts[:, 1]


def dissimilarity(gdf, groups: Sequence[str]) -> float:
    """Duncan & Duncan (1955) index of dissimilarity: the share of either group that
    would have to move for every unit to match the region's overall composition.
    Ranges 0 (no segregation) to 1 (complete segregation); the classic, most-cited index.
    """
    x, y = _two_group_counts(gdf, groups)
    X, Y = x.sum(), y.sum()
    if X <= 0 or Y <= 0:
        return float("nan")
    return float(0.5 * np.sum(np.abs(x / X - y / Y)))


def gini(gdf, groups: Sequence[str]) -> float:
    """Segregation Gini index (James & Taeuber 1985): the population-weighted Gini
    coefficient of each unit's minority share ``p_i = x_i / t_i``. More sensitive than
    dissimilarity to the *size* of units, not just whether they are over/under-represented.
    """
    x, y = _two_group_counts(gdf, groups)
    t = x + y
    P = x.sum() / t.sum() if t.sum() > 0 else float("nan")
    if not (0 < P < 1):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(t > 0, x / t, 0.0)
    g = weighted_gini(p, t)
    # weighted_gini on p already IS the James-Taeuber Gini up to the (1 - P) normalization
    # James & Taeuber use, included here so the result matches the published definition.
    return float(g / (1 - P)) if P < 1 else float("nan")


def isolation(gdf, groups: Sequence[str], group: str) -> float:
    """Isolation index (Bell 1954; Lieberson 1981): the average share of ``group`` in the
    unit a randomly chosen member of ``group`` lives in. High when ``group`` members
    mostly share units with each other."""
    counts = extract_counts(gdf, groups)
    k = list(groups).index(group)
    x = counts[:, k]
    t = unit_totals(counts)
    X = x.sum()
    if X <= 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(t > 0, x / t, 0.0)
    return float(np.sum((x / X) * p))


def exposure(gdf, groups: Sequence[str], group_a: str, group_b: str) -> float:
    """Exposure index (Bell 1954): the average share of ``group_b`` in the unit a
    randomly chosen member of ``group_a`` lives in. ``exposure(a, a)`` equals
    :func:`isolation` for group ``a``; ``exposure(a, b) != exposure(b, a)`` in general
    (it is population-weighted by the group named first)."""
    counts = extract_counts(gdf, groups)
    ka, kb = list(groups).index(group_a), list(groups).index(group_b)
    xa, xb = counts[:, ka], counts[:, kb]
    t = unit_totals(counts)
    Xa = xa.sum()
    if Xa <= 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        p_b = np.where(t > 0, xb / t, 0.0)
    return float(np.sum((xa / Xa) * p_b))


def correlation_ratio(gdf, groups: Sequence[str], group: str) -> float:
    """Correlation ratio / Eta-squared (Bell 1954; White 1986): normalized exposure,
    ``(P* - P) / (1 - P)`` where ``P*`` is ``group``'s isolation index and ``P`` its
    region-wide share. 0 when ``group`` is evenly spread (isolation equals its region-wide
    share); 1 when every member of ``group`` lives with only other members."""
    counts = extract_counts(gdf, groups)
    P = global_proportions(counts)[list(groups).index(group)]
    P_star = isolation(gdf, groups, group)
    if not (0 <= P < 1) or np.isnan(P_star):
        return float("nan")
    return float((P_star - P) / (1 - P))


def theil_h(gdf, groups: Sequence[str]) -> float:
    """Multigroup information-theory index H (Theil & Finizza 1971; Reardon & Firebaugh
    2002): the population-weighted share of the region's overall group-diversity (entropy)
    that is "explained away" by sorting people into units. 0 when every unit has the
    region's own composition; 1 when every unit is a single group.
    """
    counts = extract_counts(gdf, groups)
    t = unit_totals(counts)
    T = t.sum()
    P = global_proportions(counts)
    E = -np.sum(np.where(P > 0, P * np.log(P), 0.0))
    if E <= 0:
        return float("nan")  # every unit already has the same single-group composition
    p = local_proportions(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        E_i = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return float(np.sum(t * (E - E_i)) / (T * E))


def simpson_diversity(gdf, groups: Sequence[str]) -> float:
    """Region-wide Simpson diversity index: the probability two people drawn at random
    from the *whole region* (not a single unit -- see :func:`local.local_simpson_diversity`
    for that) belong to different groups, ``1 - sum_k P_k^2``."""
    counts = extract_counts(gdf, groups)
    P = global_proportions(counts)
    return float(1.0 - np.sum(P**2))


def multigroup_dissimilarity(gdf, groups: Sequence[str]) -> float:
    """Multigroup generalized dissimilarity index (Sakoda 1981): ``sum_i sum_k t_i *
    |p_ik - P_k| / (2 * T * I_s)``, where ``I_s = sum_k P_k*(1-P_k)`` normalizes by the
    region's own group diversity so the index still ranges roughly 0-1. Reduces to
    :func:`dissimilarity` exactly when there are only two groups."""
    counts = extract_counts(gdf, groups)
    t = unit_totals(counts)
    T = t.sum()
    P = global_proportions(counts)
    I_s = np.sum(P * (1 - P))
    if I_s <= 0:
        return float("nan")
    p = local_proportions(counts)
    return float(np.sum(t[:, None] * np.abs(p - P[None, :])) / (2 * T * I_s))
