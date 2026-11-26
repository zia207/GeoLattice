"""Static plots for segregation results: choropleth maps of local measures (with
non-significant units hatched out, if a significance mask is given), multiscale profile
curves, decomposition bar charts, and panel trend lines. Thin wrappers around matplotlib
and GeoDataFrame's own ``.plot()`` -- no new plotting dependency.
"""

from __future__ import annotations

from typing import Sequence


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("segregation.viz needs matplotlib: pip install matplotlib") from exc
    return plt


def plot_local_map(gdf, values, *, significant=None, title: str | None = None,
                    cmap: str = "RdYlBu_r", ax=None, legend: bool = True):
    """Choropleth of a local measure. Pass ``significant`` (a boolean array, e.g. from
    :func:`~geolattice.spatial_analysis.segregation.local.local_significance`'s
    ``significant_fdr`` column) to hatch out units that are not significant."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    plot_gdf = gdf.copy()
    plot_gdf["_value"] = values
    plot_gdf.plot(column="_value", cmap=cmap, legend=legend, ax=ax, edgecolor="white", linewidth=0.3)
    if significant is not None:
        import numpy as np

        insignificant = plot_gdf[~np.asarray(significant)]
        if not insignificant.empty:
            insignificant.plot(ax=ax, facecolor="none", edgecolor="black", hatch="///", linewidth=0.3)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def plot_profile(profile, unit_indices: Sequence[int] | None = None, *, ax=None,
                 labels: Sequence[str] | None = None):
    """Divergence-vs-radius curves for selected units from a
    :class:`~geolattice.spatial_analysis.segregation.multiscale.MultiscaleProfile` (defaults
    to the 5 units with the highest ``segregation_scale``, i.e. the most persistently
    segregated ones, if ``unit_indices`` is not given)."""
    plt = _require_matplotlib()
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if unit_indices is None:
        finite_scale = np.where(np.isfinite(profile.segregation_scale), profile.segregation_scale, -1)
        unit_indices = np.argsort(finite_scale)[-5:][::-1]
    for pos, i in enumerate(unit_indices):
        label = labels[pos] if labels is not None else f"unit {i}"
        ax.plot(profile.radii, profile.profile[i], marker="o", markersize=3, label=label)
    ax.set_xlabel("radius")
    ax.set_ylabel("divergence from regional composition")
    ax.legend(fontsize="small")
    return ax


def plot_decomposition(chain_df, *, ax=None):
    """Stacked bar chart of the spatial vs. composition components from
    :func:`~geolattice.spatial_analysis.segregation.decomposition.chain_decomposition`, one
    bar per consecutive period pair."""
    plt = _require_matplotlib()
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{r.period_from}→{r.period_to}" for r in chain_df.itertuples()]
    x = np.arange(len(chain_df))
    spatial = chain_df["spatial_component"].to_numpy()
    composition = chain_df["composition_component"].to_numpy()
    ax.bar(x, spatial, label="spatial")
    ax.bar(x, composition, bottom=np.where(spatial >= 0, spatial, 0) + np.where(spatial < 0, spatial, 0),
           label="composition")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("change in index")
    ax.legend()
    return ax


def plot_trend(panel, *, trend_result: dict | None = None, ax=None):
    """Line plot of a panel's statistic over time (``panel`` is the DataFrame from
    :func:`~geolattice.spatial_analysis.segregation.temporal.compute_panel`, with ``period``
    and ``statistic`` columns), with the Sen slope overlaid if ``trend_result`` (from
    :func:`~geolattice.spatial_analysis.segregation.temporal.trend`) is given."""
    plt = _require_matplotlib()
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(panel["period"], panel["statistic"], marker="o")
    if trend_result is not None and np.isfinite(trend_result.get("sen_slope", np.nan)):
        periods = panel["period"].to_numpy(dtype="float64")
        mid = periods.mean()
        mid_value = np.interp(mid, periods, panel["statistic"].to_numpy())
        fit = mid_value + trend_result["sen_slope"] * (periods - mid)
        ax.plot(periods, fit, linestyle="--", color="gray",
               label=f"Sen slope = {trend_result['sen_slope']:.4g}/period (p={trend_result['p_value']:.3g})")
        ax.legend(fontsize="small")
    ax.set_xlabel("period")
    ax.set_ylabel("statistic")
    return ax
