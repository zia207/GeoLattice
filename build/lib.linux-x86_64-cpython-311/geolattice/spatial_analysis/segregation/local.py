"""Per-unit ("local") segregation measures, the LISA-style counterpart to the single
region-wide numbers in :mod:`indices`. Each function returns one value per row of
``gdf``, ready to map; :func:`local_significance` adds permutation p-values and
:func:`fdr_correction` controls the false-discovery rate across all those per-unit tests
at once, the same multiple-comparisons problem local Moran's I maps face.

Pass a :class:`~geolattice.spatial_analysis.segregation.neighborhoods.Neighborhood` via
``neighborhood=`` to compute a *spatially pooled* version of any local measure here
(composition drawn from a unit and its neighbors, not the unit alone) -- this is what
turns an aspatial local index into a "generalized spatial" one.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .data import extract_counts, global_proportions, local_proportions, unit_totals


def _maybe_pool(counts: np.ndarray, neighborhood: Any | None) -> np.ndarray:
    return neighborhood.pooled_counts(counts) if neighborhood is not None else counts


def local_quotient(gdf, groups: Sequence[str], group: str, *, neighborhood: Any | None = None
                    ) -> np.ndarray:
    """Location quotient: ``LQ_i = p_ik / P_k``, the unit's share of ``group`` relative to
    the region's own share. ``LQ > 1`` means ``group`` is over-represented in that unit
    (locally, or in its pooled neighborhood if ``neighborhood`` is given); ``LQ < 1`` under-
    represented; ``LQ == 1`` exactly matches the regional composition."""
    counts = extract_counts(gdf, groups)
    P_k = global_proportions(counts)[list(groups).index(group)]
    pooled = _maybe_pool(counts, neighborhood)
    p = local_proportions(pooled)[:, list(groups).index(group)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(P_k > 0, p / P_k, np.nan)


def local_entropy(gdf, groups: Sequence[str], *, neighborhood: Any | None = None) -> np.ndarray:
    """Local Shannon entropy ``E_i = -sum_k p_ik * ln(p_ik)``: 0 when a unit (or its
    pooled neighborhood) has only one group present, highest when groups are evenly mixed."""
    counts = extract_counts(gdf, groups)
    pooled = _maybe_pool(counts, neighborhood)
    p = local_proportions(pooled)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def local_simpson_diversity(gdf, groups: Sequence[str], *, neighborhood: Any | None = None
                            ) -> np.ndarray:
    """Local Simpson diversity ``1 - sum_k p_ik^2``: the probability two people drawn at
    random *from this unit* (or its pooled neighborhood) belong to different groups."""
    counts = extract_counts(gdf, groups)
    pooled = _maybe_pool(counts, neighborhood)
    p = local_proportions(pooled)
    return 1.0 - np.sum(p**2, axis=1)


def local_exposure_deviation(gdf, groups: Sequence[str], group: str, *,
                              neighborhood: Any | None = None) -> np.ndarray:
    """How far a unit's share of ``group`` departs from the regional share, ``p_ik - P_k``
    (signed: positive means locally over-represented). Unlike the location quotient this
    stays well-behaved when ``P_k`` is small."""
    counts = extract_counts(gdf, groups)
    P_k = global_proportions(counts)[list(groups).index(group)]
    pooled = _maybe_pool(counts, neighborhood)
    p = local_proportions(pooled)[:, list(groups).index(group)]
    return p - P_k


def local_dissimilarity_contribution(gdf, groups: Sequence[str], *, neighborhood: Any | None = None
                                      ) -> np.ndarray:
    """Each unit's own contribution to the two-group dissimilarity index: ``D = sum_i d_i``,
    with ``d_i = t_i * |p_i - P| / (2*X*(1-P))`` for minority group ``groups[0]`` (so the
    region-wide :func:`indices.dissimilarity` is recovered exactly by summing this)."""
    counts = extract_counts(gdf, groups)
    if counts.shape[1] != 2:
        raise ValueError("local_dissimilarity_contribution needs exactly 2 group columns")
    t = unit_totals(counts)
    X = counts[:, 0].sum()
    T = t.sum()
    P = X / T if T > 0 else float("nan")
    pooled = _maybe_pool(counts, neighborhood)
    t_pool = unit_totals(pooled)
    p = local_proportions(pooled)[:, 0]
    if X <= 0 or P >= 1:
        return np.full(len(t), np.nan)
    return t_pool * np.abs(p - P) / (2 * X * (1 - P))


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction: which of ``p_values`` are
    significant at ``alpha`` once you account for running one test per unit (the same
    multiple-comparisons adjustment LISA cluster maps need -- otherwise ~5% of units show
    up "significant" by chance alone at a 0.05 per-test threshold).
    """
    p = np.asarray(p_values, dtype="float64")
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff_rank = np.max(np.where(passed)[0])
    cutoff = ranked[cutoff_rank]
    significant = np.zeros(n, dtype=bool)
    significant[order] = p[order] <= cutoff
    return significant


def local_significance(gdf, groups: Sequence[str], stat_fn: Callable[..., np.ndarray], *,
                        neighborhood: Any | None = None, permutations: int = 999,
                        alternative: str = "two-sided", alpha: float = 0.05,
                        seed: int | None = None, **stat_kwargs) -> pd.DataFrame:
    """Permutation significance for any local statistic in this module (pass the function
    itself, e.g. ``stat_fn=local_quotient, group="black"``).

    The null model redraws each unit's group composition from a multinomial with the
    region's own proportions ``P_k``, keeping each unit's observed total population ``t_i``
    fixed (the "evenness" null -- see :func:`null.evenness_null`): under this null, every
    unit is expected to look like the region, so a local statistic's departure from that is
    what gets tested. Returns one row per unit with the observed statistic, a p-value, and
    (after Benjamini-Hochberg) whether it stays significant at ``alpha``.
    """
    from .null import evenness_null

    counts = extract_counts(gdf, groups)
    rng = np.random.default_rng(seed)
    observed = stat_fn(gdf, groups, neighborhood=neighborhood, **stat_kwargs)
    totals = unit_totals(counts)
    P = global_proportions(counts)
    sims = evenness_null(totals, P, n_sims=permutations, rng=rng)

    null_stats = np.empty((permutations, len(observed)))
    cols = list(groups)
    for s in range(permutations):
        sim_gdf = gdf.copy()
        sim_gdf[cols] = sims[s]
        null_stats[s] = stat_fn(sim_gdf, groups, neighborhood=neighborhood, **stat_kwargs)

    if alternative == "two-sided":
        more_extreme = np.abs(null_stats) >= np.abs(observed)[None, :]
    elif alternative == "greater":
        more_extreme = null_stats >= observed[None, :]
    elif alternative == "less":
        more_extreme = null_stats <= observed[None, :]
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    p_values = (more_extreme.sum(axis=0) + 1) / (permutations + 1)

    significant = fdr_correction(p_values, alpha=alpha)
    return pd.DataFrame({"statistic": observed, "p_value": p_values, "significant_fdr": significant})
