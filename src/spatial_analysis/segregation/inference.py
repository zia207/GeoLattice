"""Inference for region-wide (not per-unit -- see :mod:`local` for that) segregation
statistics: is an observed index value more extreme than a null model would produce
(:func:`single_value_test`), does it differ between two datasets or two time periods
(:func:`two_value_test`), what is its confidence interval once spatial autocorrelation is
accounted for (:func:`spatial_block_bootstrap`), and is a multi-year panel trending up or
down (:func:`panel_trend_test`).

The plain bootstrap PySAL ships resamples individual units with replacement, which treats
each unit as an independent draw -- exactly the assumption spatial autocorrelation
violates (neighboring units' compositions are not independent, so resampling them one at
a time understates the true uncertainty). :func:`spatial_block_bootstrap` resamples whole
spatial blocks instead, so correlated neighbors move together in each replicate.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .data import extract_counts, global_proportions, unit_totals
from .null import evenness_null, permutation_pvalue, systematic_null


def spatial_blocks(gdf, *, n_blocks: int = 10, method: str = "kmeans",
                    seed: int | None = None) -> np.ndarray:
    """Partition units into ``n_blocks`` contiguous-ish spatial groups: ``"kmeans"``
    clusters centroid coordinates (fast, no extra dependency beyond scikit-learn);
    ``"grid"`` instead cuts the bounding box into a ``sqrt(n_blocks) x sqrt(n_blocks)``
    regular grid (deterministic, no clustering library needed). Returns one integer block
    label per unit, for :func:`spatial_block_bootstrap` or
    :func:`~geolattice.spatial_analysis.segregation.null.spatially_constrained_null`.
    """
    geom = gdf.geometry
    cent = geom.centroid if geom.geom_type.isin(["Polygon", "MultiPolygon"]).any() else geom
    xy = np.c_[cent.x.to_numpy(), cent.y.to_numpy()]
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(xy)
    if method == "grid":
        side = int(np.ceil(np.sqrt(n_blocks)))
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        gx = np.clip(((xy[:, 0] - xmin) / max(xmax - xmin, 1e-9) * side).astype(int), 0, side - 1)
        gy = np.clip(((xy[:, 1] - ymin) / max(ymax - ymin, 1e-9) * side).astype(int), 0, side - 1)
        return gx * side + gy
    raise ValueError("method must be 'kmeans' or 'grid'")


def single_value_test(gdf, groups: Sequence[str], index_fn: Callable[..., float], *,
                       null: str = "evenness", n_sims: int = 999, alternative: str = "greater",
                       seed: int | None = None, **index_kwargs) -> dict:
    """Is the observed ``index_fn(gdf, groups, **index_kwargs)`` more extreme than the
    chosen null model would produce? ``null="evenness"`` (the usual choice) redraws every
    unit's composition from the region's own proportions; ``null="systematic"`` instead
    only reshuffles which unit gets which observed composition (so it keeps however
    segregated the data already is, and only tests whether that pattern's exact spatial
    placement -- not its existence -- could be due to chance; mainly useful for
    spatially-pooled statistics, not the plain aspatial indices in :mod:`indices`).
    """
    rng = np.random.default_rng(seed)
    counts = extract_counts(gdf, groups)
    observed = float(index_fn(gdf, groups, **index_kwargs))

    if null == "evenness":
        totals, P = unit_totals(counts), global_proportions(counts)
        sims = evenness_null(totals, P, n_sims=n_sims, rng=rng)
    elif null == "systematic":
        sims = systematic_null(counts, n_sims=n_sims, rng=rng)
    else:
        raise ValueError("null must be 'evenness' or 'systematic'")

    null_values = np.empty(n_sims)
    sim_gdf = gdf.copy()
    cols = list(groups)
    for s in range(n_sims):
        sim_gdf[cols] = sims[s]
        null_values[s] = index_fn(sim_gdf, groups, **index_kwargs)

    p_value = permutation_pvalue(observed, null_values, alternative=alternative)
    return {"statistic": observed, "p_value": p_value, "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std()), "null_distribution": null_values}


def two_value_test(gdf_a, gdf_b, groups: Sequence[str], index_fn: Callable[..., float], *,
                    n_sims: int = 999, alternative: str = "two-sided", seed: int | None = None,
                    **index_kwargs) -> dict:
    """Does ``index_fn`` differ between two datasets (two cities, or the same city two
    years apart)? Pools both datasets' units, then under the null randomly re-splits the
    pool into two groups of the original sizes, building a null distribution of the
    difference in index values -- the classic two-sample permutation test, applied to a
    segregation index instead of a mean. Only needs the group-count columns (not
    geometry), since it is built for the aspatial, region-wide indices in :mod:`indices`.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = len(gdf_a), len(gdf_b)
    observed = float(index_fn(gdf_a, groups, **index_kwargs) - index_fn(gdf_b, groups, **index_kwargs))

    cols = list(groups)
    pooled = pd.concat([gdf_a[cols], gdf_b[cols]], ignore_index=True)
    n_total = n_a + n_b

    null_values = np.empty(n_sims)
    for s in range(n_sims):
        perm = rng.permutation(n_total)
        sub_a, sub_b = pooled.iloc[perm[:n_a]], pooled.iloc[perm[n_a:]]
        null_values[s] = index_fn(sub_a, groups, **index_kwargs) - index_fn(sub_b, groups, **index_kwargs)

    p_value = permutation_pvalue(observed, null_values, alternative=alternative)
    return {"difference": observed, "p_value": p_value, "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std()), "null_distribution": null_values}


def spatial_block_bootstrap(gdf, groups: Sequence[str], index_fn: Callable[..., float], *,
                             n_boot: int = 999, n_blocks: int = 10, block_method: str = "kmeans",
                             ci: float = 0.95, seed: int | None = None, **index_kwargs) -> dict:
    """Bootstrap confidence interval for ``index_fn``, resampling whole spatial blocks
    (from :func:`spatial_blocks`) with replacement rather than individual units -- so
    neighboring, spatially-correlated units are resampled together instead of as if they
    were independent draws, which is what the naive unit-level bootstrap assumes.
    """
    rng = np.random.default_rng(seed)
    blocks = spatial_blocks(gdf, n_blocks=n_blocks, method=block_method, seed=seed)
    unique_blocks = np.unique(blocks)
    estimate = float(index_fn(gdf, groups, **index_kwargs))

    boot_values = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        rows = np.concatenate([np.where(blocks == c)[0] for c in chosen])
        boot_values[b] = index_fn(gdf.iloc[rows], groups, **index_kwargs)

    alpha = 1 - ci
    lower, upper = np.quantile(boot_values, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "ci_lower": float(lower), "ci_upper": float(upper),
            "ci_level": ci, "boot_distribution": boot_values, "blocks": blocks}


def panel_trend_test(values: Sequence[float], periods: Sequence[float] | None = None) -> dict:
    """Mann-Kendall trend test across a panel's successive period values (3+ periods
    needed), plus an ordinary-least-squares slope for an interpretable rate of change. A
    small, dependency-free, non-parametric alternative to assuming the trend is linear
    and normally distributed, appropriate for a handful of census years.
    """
    v = np.asarray(values, dtype="float64")
    n = len(v)
    if n < 3:
        raise ValueError("panel_trend_test needs at least 3 periods")
    periods = np.arange(n, dtype="float64") if periods is None else np.asarray(periods, dtype="float64")

    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(v[i + 1:] - v[i]))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    from scipy.stats import norm

    p_value = float(2 * (1 - norm.cdf(abs(z))))

    slope_estimates = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = periods[j] - periods[i]
            if dt != 0:
                slope_estimates.append((v[j] - v[i]) / dt)
    sen_slope = float(np.median(slope_estimates)) if slope_estimates else float("nan")

    return {"tau": float(s / (n * (n - 1) / 2)), "z": float(z), "p_value": p_value,
            "trend": "increasing" if s > 0 else ("decreasing" if s < 0 else "no trend"),
            "sen_slope": sen_slope}
