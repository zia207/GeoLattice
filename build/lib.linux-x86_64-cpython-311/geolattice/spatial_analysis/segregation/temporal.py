"""Panel workflows: compute an index across many years at once, test whether it trends,
and put every year on one consistent set of units first when the census geography itself
changed between them. PySAL's own segregation indices only ever compare two values; this
accepts a ``{period: gdf}`` panel of any length.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd

from .data import areal_interpolate, dasymetric_interpolate
from .inference import panel_trend_test


def harmonize_panel(gdfs: dict, groups: Sequence[str], target_gdf, *, method: str = "areal",
                     raster_paths: dict | None = None) -> dict:
    """Interpolate every period's source geometry in ``gdfs`` onto one common
    ``target_gdf``, so the whole panel ends up on the same units before computing any
    index across it -- the usual problem with multi-decade segregation panels, since
    census tract/block-group boundaries get redrawn almost every census.

    ``method="areal"`` (default) uses plain area-weighted interpolation
    (:func:`~geolattice.spatial_analysis.segregation.data.areal_interpolate`); ``"dasymetric"``
    instead needs ``raster_paths`` (a ``{period: path}`` mapping of a land-cover,
    population-grid, or nighttime-lights raster for that period) and redistributes counts
    by that raster's mass within each source-target overlap piece instead of by bare area.
    """
    out = {}
    for period, gdf in gdfs.items():
        if method == "areal":
            out[period] = areal_interpolate(gdf, target_gdf, list(groups))
        elif method == "dasymetric":
            if not raster_paths or period not in raster_paths:
                raise ValueError(f"method='dasymetric' needs raster_paths[{period!r}]")
            out[period] = dasymetric_interpolate(gdf, target_gdf, list(groups),
                                                 raster_path=raster_paths[period])
        else:
            raise ValueError("method must be 'areal' or 'dasymetric'")
    return out


def compute_panel(gdfs: dict, groups: Sequence[str], index_fn: Callable[..., float] = None,
                   **index_kwargs) -> pd.DataFrame:
    """One row per period: ``index_fn(gdf, groups, **index_kwargs)`` for each ``{period:
    gdf}`` pair in ``gdfs`` (defaults to
    :func:`~geolattice.spatial_analysis.segregation.indices.dissimilarity`)."""
    if index_fn is None:
        from .indices import dissimilarity as index_fn
    rows = [{"period": period, "statistic": float(index_fn(gdf, groups, **index_kwargs))}
            for period, gdf in sorted(gdfs.items())]
    return pd.DataFrame(rows)


def trend(gdfs: dict, groups: Sequence[str], index_fn: Callable[..., float] = None,
          **index_kwargs) -> dict:
    """:func:`compute_panel` followed by
    :func:`~geolattice.spatial_analysis.segregation.inference.panel_trend_test` on the
    resulting series -- the one-call version of "is this index trending across the panel."
    """
    panel = compute_panel(gdfs, groups, index_fn, **index_kwargs)
    result = panel_trend_test(panel["statistic"].to_numpy(), panel["period"].to_numpy())
    result["panel"] = panel
    return result
