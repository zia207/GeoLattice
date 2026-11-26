"""Spaceborne lidar: GEDI (L2A heights, L2B cover/PAI/FHD, L4A biomass) and ICESat-2 ATL08."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from ..core._utils import map_to_pix

GEDI_BEAMS = ("BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011", "BEAM0101", "BEAM0110",
              "BEAM1000", "BEAM1011")
ATL08_BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")

SHORT_NAMES = {"gedi_l2a": "GEDI02_A", "gedi_l2b": "GEDI02_B", "gedi_l4a": "GEDI_L4A_AGB_Density_V2_1_2056",
               "atl08": "ATL08", "atl03": "ATL03"}


def search(product: str, *, aoi: Any, dates: Any, **kwargs: Any):
    """Search NASA Earthdata for GEDI/ICESat-2 granules (``product`` key from SHORT_NAMES)."""
    from ..access import earthdata_search

    return earthdata_search(SHORT_NAMES.get(product, product), aoi=aoi, dates=dates, **kwargs)


def _gdf(df: pd.DataFrame, lon: str, lat: str):
    import geopandas as gpd

    return gpd.GeoDataFrame(df.drop(columns=[lon, lat]),
                            geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)


def read_gedi_l2a(path: str | PathLike, *, beams: Sequence[str] | None = None,
                  rh: Sequence[int] = (25, 50, 75, 95, 98, 100), quality: bool = True,
                  min_sensitivity: float = 0.9, bbox: tuple[float, float, float, float] | None = None):
    """GEDI L2A footprints → GeoDataFrame with relative heights ``rhNN`` (m), ground elevation,
    sensitivity and quality. ``quality=True`` keeps quality_flag==1, degrade_flag==0 and
    sensitivity >= ``min_sensitivity`` (recommended over forests)."""
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        for b in beams or [k for k in GEDI_BEAMS if k in f]:
            g = f[b]
            d = {"beam": b, "shot_number": g["shot_number"][:],
                 "lat": g["lat_lowestmode"][:], "lon": g["lon_lowestmode"][:],
                 "elev_lowestmode": g["elev_lowestmode"][:],
                 "sensitivity": g["sensitivity"][:], "quality_flag": g["quality_flag"][:],
                 "degrade_flag": g["degrade_flag"][:] if "degrade_flag" in g else 0}
            rhs = g["rh"][:]
            for p in rh:
                d[f"rh{p}"] = rhs[:, p]
            if "delta_time" in g:
                d["delta_time"] = g["delta_time"][:]
            frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if quality and len(df):
        df = df[(df.quality_flag == 1) & (df.degrade_flag == 0) & (df.sensitivity >= min_sensitivity)]
    if bbox is not None and len(df):
        df = df[(df.lon >= bbox[0]) & (df.lon <= bbox[2]) & (df.lat >= bbox[1]) & (df.lat <= bbox[3])]
    return _gdf(df.reset_index(drop=True), "lon", "lat")


def read_gedi_l4a(path: str | PathLike, *, beams: Sequence[str] | None = None,
                  quality: bool = True, bbox: tuple[float, float, float, float] | None = None):
    """GEDI L4A aboveground biomass density (Mg/ha) with standard error per footprint."""
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        for b in beams or [k for k in GEDI_BEAMS if k in f]:
            g = f[b]
            frames.append(pd.DataFrame({
                "beam": b, "shot_number": g["shot_number"][:], "lat": g["lat_lowestmode"][:],
                "lon": g["lon_lowestmode"][:], "agbd": g["agbd"][:], "agbd_se": g["agbd_se"][:],
                "l4_quality_flag": g["l4_quality_flag"][:]}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if quality and len(df):
        df = df[(df.l4_quality_flag == 1) & (df.agbd >= 0)]
    if bbox is not None and len(df):
        df = df[(df.lon >= bbox[0]) & (df.lon <= bbox[2]) & (df.lat >= bbox[1]) & (df.lat <= bbox[3])]
    return _gdf(df.reset_index(drop=True), "lon", "lat")


def read_gedi_l2b(path: str | PathLike, *, beams: Sequence[str] | None = None,
                  quality: bool = True, profiles: bool = False,
                  bbox: tuple[float, float, float, float] | None = None):
    """GEDI L2B canopy cover and vertical profile metrics per footprint.

    Columns: ``cover`` (total canopy cover, 0–1), ``pai`` (plant area index), ``fhd_normal``
    (foliage height diversity), ``rh100`` (m), ``sensitivity`` and quality flags. With
    ``profiles=True`` adds ``pavd_z0 ...`` (plant area volume density per 5 m layer) and
    ``cover_z0 ...`` columns. ``quality=True`` keeps ``l2b_quality_flag == 1`` and drops
    fill values.
    """
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        for b in beams or [k for k in GEDI_BEAMS if k in f]:
            g = f[b]
            geo = g["geolocation"] if "geolocation" in g else g
            lat = geo["lat_lowestmode"][:] if "lat_lowestmode" in geo else geo["latitude_bin0"][:]
            lon = geo["lon_lowestmode"][:] if "lon_lowestmode" in geo else geo["longitude_bin0"][:]
            d = {"beam": b, "shot_number": g["shot_number"][:], "lat": lat, "lon": lon,
                 "cover": g["cover"][:], "pai": g["pai"][:], "fhd_normal": g["fhd_normal"][:],
                 "rh100": g["rh100"][:] / 100.0 if g["rh100"].dtype.kind in "iu" else g["rh100"][:],
                 "sensitivity": g["sensitivity"][:],
                 "l2b_quality_flag": g["l2b_quality_flag"][:],
                 "algorithmrun_flag": g["algorithmrun_flag"][:] if "algorithmrun_flag" in g else 1}
            if profiles:
                for name in ("pavd_z", "cover_z"):
                    if name in g:
                        arr = g[name][:]
                        for j in range(arr.shape[1]):
                            d[f"{name}{j}"] = arr[:, j]
            frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if quality and len(df):
        df = df[(df.l2b_quality_flag == 1) & (df.algorithmrun_flag == 1) & (df.cover >= 0)
                & (df.pai >= 0)]
    if bbox is not None and len(df):
        df = df[(df.lon >= bbox[0]) & (df.lon <= bbox[2]) & (df.lat >= bbox[1]) & (df.lat <= bbox[3])]
    return _gdf(df.reset_index(drop=True), "lon", "lat")


def read_atl08(path: str | PathLike, *, beams: Sequence[str] | None = None,
               strong_only: bool = False, bbox: tuple[float, float, float, float] | None = None):
    """ICESat-2 ATL08 100 m land segments → canopy height (``h_canopy``), terrain
    (``h_te_best_fit``) and canopy cover. Fill values (> 3e38) become NaN."""
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        orient = int(f["orbit_info/sc_orient"][0]) if "orbit_info/sc_orient" in f else None
        for b in beams or [k for k in ATL08_BEAMS if k in f]:
            if strong_only and orient is not None:
                strong = b.endswith("l") if orient == 0 else b.endswith("r")
                if not strong:
                    continue
            ls = f[f"{b}/land_segments"]
            frames.append(pd.DataFrame({
                "beam": b, "lat": ls["latitude"][:], "lon": ls["longitude"][:],
                "h_canopy": ls["canopy/h_canopy"][:], "h_te_best_fit": ls["terrain/h_te_best_fit"][:],
                "canopy_openness": ls["canopy/canopy_openness"][:]
                if "canopy/canopy_openness" in ls else np.nan,
                "night_flag": ls["night_flag"][:] if "night_flag" in ls else np.nan}))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for c in ("h_canopy", "h_te_best_fit", "canopy_openness"):
        if c in df:
            df.loc[df[c] > 3e38, c] = np.nan
    if bbox is not None and len(df):
        df = df[(df.lon >= bbox[0]) & (df.lon <= bbox[2]) & (df.lat >= bbox[1]) & (df.lat <= bbox[3])]
    return _gdf(df.reset_index(drop=True), "lon", "lat")


def footprints_to_raster(points, value: str, template, stat: str = "mean"):
    """Grid footprint values onto a raster template (mean/median/count per cell)."""
    import xarray as xr

    pts = points.to_crs(template.rio.crs)
    cols, rows = map_to_pix(template.rio.transform(), pts.geometry.x.values, pts.geometry.y.values)
    r, c = rows.astype(int), cols.astype(int)
    ok = (r >= 0) & (r < template.shape[-2]) & (c >= 0) & (c < template.shape[-1])
    df = pd.DataFrame({"cell": r[ok] * template.shape[-1] + c[ok], "v": pts[value].values[ok]})
    agg = getattr(df.groupby("cell")["v"], stat)()
    grid = np.full(template.shape[-2] * template.shape[-1], np.nan, dtype="float32")
    grid[agg.index.values] = agg.values
    ydim, xdim = template.dims[-2:]
    da = xr.DataArray(grid.reshape(template.shape[-2:]), dims=(ydim, xdim),
                      coords={ydim: template[ydim], xdim: template[xdim]}, name=value)
    return da.rio.write_crs(template.rio.crs)
