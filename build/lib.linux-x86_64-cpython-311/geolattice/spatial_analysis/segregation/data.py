"""Input validation, group-count extraction, and boundary harmonization.

Segregation indices need one row per spatial unit and a set of mutually
exclusive population counts per group. This module turns a user's
GeoDataFrame into that ``(n_units, n_groups)`` array, validates it, and
provides two ways to make counts comparable across years when census
geography changes: plain areal-weighted interpolation, and a dasymetric
variant that redistributes counts using a raster weight layer (land cover,
a population grid, nighttime lights, ...) instead of polygon area alone.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


def _overlay_intersection(gpd, a, b):
    """``gpd.overlay(a, b, how="intersection")``, without the benign warning that fires
    whenever two of the inputs' polygons merely touch along an edge (a degenerate,
    zero-area line intersection, correctly dropped rather than kept)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`keep_geom_type=True`.*", category=UserWarning)
        return gpd.overlay(a, b, how="intersection")


class SegregationDataError(ValueError):
    """Raised when input data cannot be turned into a valid group-count table."""


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geolattice.spatial_analysis.segregation needs geopandas: pip install geopandas"
        ) from exc
    return gpd


def extract_counts(gdf, groups: Sequence[str]) -> np.ndarray:
    """Pull the group-count columns out of ``gdf`` as a float ``(n_units, n_groups)`` array.

    Validates that every requested column exists, every count is finite and
    non-negative, and that at least one unit has a positive total -- the
    checks every index in this package relies on instead of repeating.
    """
    missing = [g for g in groups if g not in gdf.columns]
    if missing:
        raise SegregationDataError(f"Columns not found in input data: {missing}")
    counts = gdf[list(groups)].to_numpy(dtype="float64")
    if not np.all(np.isfinite(counts)):
        raise SegregationDataError("Group-count columns contain NaN or infinite values")
    if np.any(counts < 0):
        raise SegregationDataError("Group counts must be non-negative")
    totals = counts.sum(axis=1)
    if not np.any(totals > 0):
        raise SegregationDataError("Every unit has zero total population across the given groups")
    return counts


def unit_totals(counts: np.ndarray) -> np.ndarray:
    return counts.sum(axis=1)


def global_proportions(counts: np.ndarray) -> np.ndarray:
    """Region-wide share of each group, ``X_k / T``."""
    group_totals = counts.sum(axis=0)
    total = group_totals.sum()
    if total <= 0:
        raise SegregationDataError("Total population across all groups and units is zero")
    return group_totals / total


def local_proportions(counts: np.ndarray) -> np.ndarray:
    """Within-unit share of each group, ``p_ik = x_ik / t_i`` (0 where ``t_i == 0``)."""
    totals = unit_totals(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        props = counts / totals[:, None]
    return np.where(totals[:, None] > 0, props, 0.0)


def wide_to_long(gdf, groups: Sequence[str], *, id_col: str, year: int | str,
                  value_name: str = "count", group_name: str = "group") -> pd.DataFrame:
    """One row per (unit, group) for a single year -- the building block :func:`combine_years`
    and :mod:`temporal` stack into a long panel."""
    long = gdf[[id_col, *groups]].melt(id_vars=id_col, value_vars=list(groups),
                                        var_name=group_name, value_name=value_name)
    long["year"] = year
    return long


def combine_years(frames: dict[int | str, pd.DataFrame]) -> pd.DataFrame:
    """Stack several years' :func:`wide_to_long` outputs into one panel, tagging each row
    with its ``year`` (already present if built by ``wide_to_long``, but reapplied here so
    callers can also pass plain per-year long frames directly)."""
    parts = []
    for year, df in frames.items():
        part = df.copy()
        part["year"] = year
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def areal_interpolate(source_gdf, target_gdf, value_cols: Sequence[str], *,
                       extensive: bool = True):
    """Area-weighted interpolation of ``value_cols`` from ``source_gdf``'s polygons onto
    ``target_gdf``'s polygons -- the standard way to compare counts across geographies that
    do not share boundaries (e.g. two census years' tracts).

    Each source polygon's value is split across the target polygons it overlaps, in
    proportion to the overlapping area's share of the *source* polygon's area
    (``extensive=True``, the right choice for counts such as population: a value is
    redistributed, not duplicated). ``extensive=False`` instead weights by the share of the
    *target* polygon's area, appropriate for intensive quantities such as a rate or density
    that should be area-averaged rather than split. Both GeoDataFrames must share a CRS;
    reproject first if they do not.
    """
    gpd = _require_geopandas()
    if source_gdf.crs != target_gdf.crs:
        raise SegregationDataError(
            "source_gdf and target_gdf must share a CRS; reproject one onto the other first")
    missing = [c for c in value_cols if c not in source_gdf.columns]
    if missing:
        raise SegregationDataError(f"Columns not found in source_gdf: {missing}")

    src = source_gdf.reset_index(drop=True).copy()
    tgt = target_gdf.reset_index(drop=True).copy()
    src["_src_id"] = src.index
    tgt["_tgt_id"] = tgt.index
    src["_src_area"] = src.geometry.area
    tgt["_tgt_area"] = tgt.geometry.area

    pieces = _overlay_intersection(gpd, src[["_src_id", "_src_area", *value_cols, "geometry"]],
                                   tgt[["_tgt_id", "_tgt_area", "geometry"]])
    if pieces.empty:
        raise SegregationDataError("source_gdf and target_gdf do not overlap at all")
    pieces["_piece_area"] = pieces.geometry.area

    if extensive:
        denom = pieces.groupby("_src_id")["_piece_area"].transform("sum")
        # guard against a source polygon whose overlap area underflows to ~0 (sliver geometry)
        src_area = pieces["_src_area"].to_numpy()
        denom = np.where(denom.to_numpy() > 0, denom.to_numpy(), src_area)
        weight = pieces["_piece_area"].to_numpy() / np.maximum(denom, 1e-12)
    else:
        weight = pieces["_piece_area"].to_numpy() / np.maximum(pieces["_tgt_area"].to_numpy(), 1e-12)

    out = pd.DataFrame({"_tgt_id": pieces["_tgt_id"].to_numpy()})
    for col in value_cols:
        out[col] = pieces[col].to_numpy() * weight
    agg = out.groupby("_tgt_id").sum() if extensive else out.groupby("_tgt_id").sum()
    # drop any pre-existing value_cols from the target first: the result always carries the
    # *interpolated* values, never a pandas merge-suffixed duplicate of a same-named column
    # the target happened to already have (e.g. interpolating onto another year's own data).
    tgt_clean = tgt.drop(columns=[*[c for c in value_cols if c in tgt.columns], "_tgt_area"])
    result = tgt_clean.merge(agg, left_on="_tgt_id", right_index=True, how="left")
    for col in value_cols:
        result[col] = result[col].fillna(0.0)
    return result.drop(columns="_tgt_id")


def dasymetric_interpolate(source_gdf, target_gdf, value_cols: Sequence[str], *,
                           raster_path, band: int = 1, nodata: float | None = None):
    """Dasymetric interpolation: like :func:`areal_interpolate`, but each source-target
    overlap piece is weighted by the *sum of a raster layer* inside that piece (a land-cover
    built-up mask, a population or nighttime-lights grid, ...) instead of by its bare area.

    This is what to reach for when population is known to be unevenly distributed within a
    source polygon (e.g. a large rural census tract that is mostly farmland): weighting by a
    built-up-area or population-grid raster keeps the interpolated counts out of land that
    could not have held them. Needs ``rasterio`` (``pip install rasterio`` or
    ``geolattice[io]``). ``value_cols`` must be extensive (count-like) quantities.
    """
    gpd = _require_geopandas()
    try:
        import rasterio
        from rasterio.features import geometry_mask
    except ImportError as exc:
        raise ImportError(
            "dasymetric_interpolate needs rasterio: pip install rasterio (or geolattice[io])") from exc

    if source_gdf.crs != target_gdf.crs:
        raise SegregationDataError(
            "source_gdf and target_gdf must share a CRS; reproject one onto the other first")
    missing = [c for c in value_cols if c not in source_gdf.columns]
    if missing:
        raise SegregationDataError(f"Columns not found in source_gdf: {missing}")

    src = source_gdf.reset_index(drop=True).copy()
    tgt = target_gdf.reset_index(drop=True).copy()
    src["_src_id"] = src.index
    tgt["_tgt_id"] = tgt.index

    pieces = _overlay_intersection(gpd, src[["_src_id", *value_cols, "geometry"]],
                                   tgt[["_tgt_id", "geometry"]])
    if pieces.empty:
        raise SegregationDataError("source_gdf and target_gdf do not overlap at all")

    with rasterio.open(raster_path) as ds:
        if str(ds.crs) != str(src.crs):
            raise SegregationDataError(
                f"Raster CRS ({ds.crs}) does not match source_gdf/target_gdf CRS ({src.crs}); "
                "reproject the raster first")
        raster = ds.read(band).astype("float64")
        raster_nodata = ds.nodata if nodata is None else nodata
        if raster_nodata is not None:
            raster = np.where(raster == raster_nodata, 0.0, raster)
        raster = np.where(np.isfinite(raster), raster, 0.0)
        transform = ds.transform
        out_shape = raster.shape

        def piece_weight(geom):
            if geom.is_empty:
                return 0.0
            mask = ~geometry_mask([geom], out_shape=out_shape, transform=transform, invert=False)
            return float(raster[mask].sum())

    weights = pieces.geometry.apply(piece_weight).to_numpy()
    denom = pd.Series(weights, index=pieces.index).groupby(pieces["_src_id"]).transform("sum").to_numpy()
    src_area_fallback = pieces.geometry.area.to_numpy()
    denom = np.where(denom > 0, denom, src_area_fallback)  # no raster mass -> fall back to area weights
    weights = np.where(denom > 0, weights / np.maximum(denom, 1e-12), 0.0)

    out = pd.DataFrame({"_tgt_id": pieces["_tgt_id"].to_numpy()})
    for col in value_cols:
        out[col] = pieces[col].to_numpy() * weights
    agg = out.groupby("_tgt_id").sum()
    tgt_clean = tgt.drop(columns=[c for c in value_cols if c in tgt.columns])
    result = tgt_clean.merge(agg, left_on="_tgt_id", right_index=True, how="left")
    for col in value_cols:
        result[col] = result[col].fillna(0.0)
    return result.drop(columns="_tgt_id")
