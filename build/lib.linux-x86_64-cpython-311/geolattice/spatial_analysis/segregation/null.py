"""Null models: simulate what the group-count table would look like under a specific
"no segregation" story, for :mod:`local` and :mod:`inference` to compare observed
statistics against. Each function returns ``(n_sims, n_units, n_groups)`` simulated
count tables -- never a materialized pairwise distance/divergence table, so simulating
thousands of units thousands of times stays a simple array operation.
"""

from __future__ import annotations

import numpy as np


def evenness_null(totals: np.ndarray, global_props: np.ndarray, *, n_sims: int = 999,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """"Evenness" null (the one most indices are compared against): every unit keeps its
    observed total population ``t_i``, but its group composition is redrawn from a
    multinomial with the *region's* proportions ``global_props`` -- i.e. the null world
    where nobody sorts into units by group at all. ``totals`` is rounded to the nearest
    integer (a multinomial draw needs an integer trial count).
    """
    rng = rng or np.random.default_rng()
    t_int = np.rint(totals).astype("int64")
    n, k = len(totals), len(global_props)
    out = np.empty((n_sims, n, k), dtype="float64")
    for s in range(n_sims):
        out[s] = rng.multinomial(t_int, global_props)  # broadcasts n over units in one call
    return out


def systematic_null(counts: np.ndarray, *, n_sims: int = 999,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """"Systematic" (spatial-randomization) null: the exact observed set of unit
    compositions is kept, but which unit gets which composition is reshuffled. Unlike
    :func:`evenness_null`, this preserves the empirical distribution of compositions
    exactly (including however segregated it already is) and tests only whether *where*
    that segregation sits in space matters -- the right null for a spatially pooled local
    statistic, where :func:`evenness_null` would also wash out genuine global segregation.
    """
    rng = rng or np.random.default_rng()
    n = counts.shape[0]
    out = np.empty((n_sims, *counts.shape), dtype="float64")
    for s in range(n_sims):
        out[s] = counts[rng.permutation(n)]
    return out


def spatially_constrained_null(counts: np.ndarray, block_labels: np.ndarray, *, n_sims: int = 999,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Like :func:`systematic_null`, but compositions are only reshuffled *within* each
    spatial block (``block_labels``, e.g. from :func:`inference.spatial_blocks`) --
    respects whatever spatial autocorrelation exists beyond block scale, instead of mixing
    compositions across the whole study area."""
    rng = rng or np.random.default_rng()
    out = np.empty((n_sims, *counts.shape), dtype="float64")
    blocks = {b: np.where(block_labels == b)[0] for b in np.unique(block_labels)}
    for s in range(n_sims):
        sim = counts.copy()
        for idx in blocks.values():
            sim[idx] = counts[idx][rng.permutation(len(idx))]
        out[s] = sim
    return out


def permutation_pvalue(observed: float, null_values: np.ndarray, *, alternative: str = "two-sided"
                        ) -> float:
    """Permutation p-value for a scalar ``observed`` statistic against its simulated null
    distribution ``null_values``, with the standard ``+1`` correction so the p-value can
    never read exactly 0 from a finite number of simulations."""
    n_sims = len(null_values)
    if alternative == "two-sided":
        more_extreme = np.sum(np.abs(null_values) >= abs(observed))
    elif alternative == "greater":
        more_extreme = np.sum(null_values >= observed)
    elif alternative == "less":
        more_extreme = np.sum(null_values <= observed)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return float((more_extreme + 1) / (n_sims + 1))
