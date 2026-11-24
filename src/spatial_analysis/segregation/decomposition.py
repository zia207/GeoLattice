"""Decompose the *change* in a two-group segregation index between two datasets (two
cities, or the same city at two points in time) into how much comes from the overall
group composition shifting versus how much comes from people's spatial arrangement
shifting, holding composition fixed.

Uses the standard two-player Shapley value: build a counterfactual that keeps dataset
A's spatial pattern of minority shares but rescales it to dataset B's regional
composition (and vice versa), then average the two orderings in which "turn on A's
spatial pattern" and "turn on A's composition" could happen. This is exactly additive by
construction (:func:`decompose_change`'s two components always sum to ``index(A) -
index(B)``, checked in the test suite), which is the property that actually matters here
-- not the name of a specific paper's counterfactual recipe.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .data import extract_counts, global_proportions, unit_totals


def _counterfactual(gdf, groups: Sequence[str], target_proportion: float):
    """A copy of ``gdf`` keeping its own unit totals and *relative* spatial pattern of
    minority share, rescaled so the region-wide minority share matches ``target_proportion``."""
    counts = extract_counts(gdf, groups)
    t = unit_totals(counts)
    x = counts[:, 0]
    P = x.sum() / t.sum() if t.sum() > 0 else 0.0
    if P <= 0:
        scale = 0.0
    else:
        scale = target_proportion / P
    x_new = np.clip(x * scale, 0, t)
    out = gdf.copy()
    out[groups[0]] = x_new
    out[groups[1]] = t - x_new
    return out


def decompose_change(gdf_a, gdf_b, groups: Sequence[str],
                      index_fn: Callable[..., float] = None) -> dict:
    """Split ``index_fn(gdf_a, groups) - index_fn(gdf_b, groups)`` into a ``spatial`` and
    a ``composition`` component (exactly 2 group columns; defaults to
    :func:`~geolattice.spatial_analysis.segregation.indices.dissimilarity` if ``index_fn`` is
    not given). Both components are returned in the same units as the index itself and
    sum exactly to the observed difference.
    """
    if index_fn is None:
        from .indices import dissimilarity as index_fn

    if len(groups) != 2:
        raise ValueError("decompose_change needs exactly 2 group columns")

    counts_a, counts_b = extract_counts(gdf_a, groups), extract_counts(gdf_b, groups)
    P_a = global_proportions(counts_a)[0]
    P_b = global_proportions(counts_b)[0]

    d_a_comp_a = float(index_fn(gdf_a, groups))
    d_b_comp_b = float(index_fn(gdf_b, groups))
    d_a_comp_b = float(index_fn(_counterfactual(gdf_a, groups, P_b), groups))
    d_b_comp_a = float(index_fn(_counterfactual(gdf_b, groups, P_a), groups))

    spatial = 0.5 * ((d_a_comp_a - d_b_comp_a) + (d_a_comp_b - d_b_comp_b))
    composition = 0.5 * ((d_a_comp_a - d_a_comp_b) + (d_b_comp_a - d_b_comp_b))

    return {"index_a": d_a_comp_a, "index_b": d_b_comp_b, "total_change": d_a_comp_a - d_b_comp_b,
            "spatial_component": spatial, "composition_component": composition}


def chain_decomposition(gdfs: dict, groups: Sequence[str],
                         index_fn: Callable[..., float] = None) -> pd.DataFrame:
    """Apply :func:`decompose_change` across every pair of *consecutive* periods in
    ``gdfs`` (a ``{period: gdf}`` mapping, e.g. ``{2000: gdf_2000, 2010: gdf_2010, 2020:
    gdf_2020}``), instead of only ever comparing two periods at a time. Returns one row
    per consecutive pair plus running cumulative totals of each component, so a multi-
    decade trend's spatial-vs-composition story can be read off directly.
    """
    periods = sorted(gdfs)
    if len(periods) < 2:
        raise ValueError("chain_decomposition needs at least 2 periods")
    rows = []
    cum_spatial = cum_composition = 0.0
    for p0, p1 in zip(periods[:-1], periods[1:]):
        result = decompose_change(gdfs[p1], gdfs[p0], groups, index_fn)
        cum_spatial += result["spatial_component"]
        cum_composition += result["composition_component"]
        rows.append({"period_from": p0, "period_to": p1, **result,
                     "cumulative_spatial": cum_spatial, "cumulative_composition": cum_composition})
    return pd.DataFrame(rows)
