"""Ready-made SAR applications: flood mapping, rice mapping, ship and oil-spill detection.

Flood helpers that sit on top of :func:`flood_map` are :func:`hand` (Height Above
Nearest Drainage from a DEM) and :func:`mask_to_polygons` (vector flood extent).

The :func:`un_spider_flood_mapping` function provides a complete workflow wrapper
that follows the UN-SPIDER recommended practice for Sentinel-1 flood mapping.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from .change import cfar, detections_to_points, log_ratio, threshold
from .sentinel1 import to_db


def _values(obj) -> np.ndarray:
    """Host ndarray from a DataArray or a plain array."""
    obj = to_cpu(obj)
    return np.asarray(getattr(obj, "values", obj))


def _clean(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove objects and fill holes smaller than ``min_pixels`` (8-connectivity)."""
    from ..core.morphology import fill_small_holes, remove_small

    if min_pixels <= 1:
        return mask
    return fill_small_holes(remove_small(mask, min_pixels), min_pixels)


def _split_otsu(db: np.ndarray, tile: int, min_d: float) -> float:
    """Tile-based (split) Otsu: use only bimodal tiles (Ashman D > ``min_d``) containing both
    water and land to estimate a robust global threshold (Martinis et al. 2009 idea)."""
    from skimage.filters import threshold_otsu

    vals = []
    H, W = db.shape
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            t = db[r:r + tile, c:c + tile]
            t = t[np.isfinite(t)]
            if t.size < tile * tile * 0.5:
                continue
            th = threshold_otsu(t)
            a, b = t[t < th], t[t >= th]
            if min(a.size, b.size) < 0.1 * t.size:
                continue
            d = np.sqrt(2) * abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())
            if d > min_d:
                vals.extend([a, b])
    if not vals:
        return float(threshold_otsu(db[np.isfinite(db)]))
    return float(threshold_otsu(np.concatenate(vals)))


def flood_map(img: xr.DataArray, *, method: Literal[
              "otsu", "split_otsu", "change", "fixed", "ratio"] = "split_otsu",
              reference: xr.DataArray | None = None, db_input: bool = False,
              fixed_db: float = -18.0, change_db: float = -3.0, ratio_threshold: float = 1.25,
              hand: xr.DataArray | None = None, hand_max: float = 15.0,
              slope: xr.DataArray | None = None, slope_max: float = 5.0,
              shadow: xr.DataArray | None = None, permanent_water: xr.DataArray | None = None,
              min_pixels: int = 20, tile: int = 256, min_ashman: float = 2.0) -> xr.Dataset:
    """Open-water flood extent from Sentinel-1 (VV or VH).

    Methods
    -------
    otsu        global Otsu threshold on dB backscatter
    split_otsu  Otsu estimated only over bimodal tiles (robust when water is a small fraction)
    fixed       a fixed dB threshold (``fixed_db``)
    change      log-ratio vs a dry ``reference`` image below ``change_db`` *and* dark (threshold)
    ratio       ``after / before`` on dB backscatter above ``ratio_threshold`` (default 1.25,
                the UN-SPIDER Sentinel-1 practice). dB values are negative, so a drop in
                backscatter makes the ratio larger than 1.

    ``hand`` (Height Above Nearest Drainage, metres) removes false positives on high
    ground (> ``hand_max``). ``slope`` drops pixels steeper than ``slope_max``, in the
    units of ``slope`` (degrees or percent). ``shadow`` drops pixels where the mask is
    nonzero, including classes 1 (layover) and 2 (shadow) from
    :func:`geolattice.rsense.sar.layover_shadow_mask`. ``permanent_water`` separates flood
    from normal water. Returns ``water``, ``flood`` (if permanent water is given) and
    the threshold used.
    """
    db = img if db_input else to_db(img)
    a = np.asarray(to_cpu(db).values, dtype="float64")
    if method == "otsu":
        th = threshold(a)
    elif method == "split_otsu":
        th = _split_otsu(a, tile, min_ashman)
    elif method == "fixed":
        th = fixed_db
    elif method == "change":
        if reference is None:
            raise ValueError("method='change' requires reference=")
        ref = reference if db_input else to_db(reference)
        lr = np.asarray(to_cpu((db - ref).values))
        th = _split_otsu(a, tile, min_ashman)
        water = (lr < change_db) & (a < th)
    elif method == "ratio":
        if reference is None:
            raise ValueError("method='ratio' requires reference=")
        ref = reference if db_input else to_db(reference)
        ratio = np.asarray(to_cpu((db / ref).values), dtype="float64")
        th = float(ratio_threshold)
        # dB backscatter is negative, so flooding (a darker after image) raises the ratio.
        water = (ratio > th) & np.isfinite(ratio)
    else:
        raise ValueError(f"Unknown method {method!r}")
    if method not in ("change", "ratio"):
        water = a < th
    water &= np.isfinite(a)
    if hand is not None:
        water &= _values(hand) <= hand_max
    if slope is not None:
        water &= _values(slope) <= slope_max
    if shadow is not None:
        sh = _values(shadow)
        water &= ~(np.isfinite(sh) & (sh != 0))
    water = _clean(water, min_pixels)
    out = xr.Dataset({"water": db.copy(data=water)}, attrs={"threshold_db": float(th),
                                                            "method": method})
    if permanent_water is not None:
        out["flood"] = out["water"] & ~permanent_water.astype(bool)
    return out


def hand(dem: xr.DataArray, *, channel_cells: float = 100.0):
    """Height Above Nearest Drainage (HAND; Nobre et al. 2011), in the units of ``dem``.

    Depressions are filled, then a D8 channel network is every cell whose upslope
    count is at least ``channel_cells`` (the same drainage as
    :func:`geolattice.rsense.lidar.flow_accumulation`). HAND at a cell is its filled
    elevation minus the elevation of the channel cell it drains to. Channel cells
    are 0. Cells that reach the border without meeting a channel are measured
    above that outlet.
    """
    from ..lidar.terrain import _D8, fill_depressions, flow_accumulation, flow_direction

    filled = fill_depressions(dem)
    fd_da = flow_direction(filled)
    acc_da = flow_accumulation(filled, filled=True)
    z = np.asarray(to_cpu(getattr(filled, "values", filled)), dtype="float64")
    fd = np.asarray(to_cpu(getattr(fd_da, "values", fd_da)), dtype=int)
    acc = np.asarray(to_cpu(getattr(acc_da, "values", acc_da)), dtype="float64")
    rows, cols = z.shape
    dr = np.array([d[0] for d in _D8])
    dc = np.array([d[1] for d in _D8])
    fd_safe = np.clip(fd, 0, 7)
    rr = np.arange(rows)[:, None] + dr[fd_safe]
    cc = np.arange(cols)[None, :] + dc[fd_safe]
    inside = (fd >= 0) & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    down = np.where(inside, rr * cols + cc, -1).ravel()
    flat_z = z.ravel()
    flat_stream = (acc >= channel_cells).ravel()
    drainage = np.full(flat_z.size, np.nan)
    order = np.argsort(np.where(np.isfinite(flat_z), flat_z, np.inf))
    for idx in order:
        if not np.isfinite(flat_z[idx]):
            continue
        j = int(down[idx])
        if flat_stream[idx] or j < 0 or not np.isfinite(drainage[j]):
            drainage[idx] = flat_z[idx]
        else:
            drainage[idx] = drainage[j]
    h = (flat_z - drainage).reshape(rows, cols)
    h = np.where(np.isfinite(z), np.maximum(h, 0), np.nan).astype("float32")
    if not isinstance(dem, xr.DataArray):
        return h
    out = dem.copy(data=h).rename("hand")
    out.attrs["units"] = "m"
    out.attrs["channel_cells"] = float(channel_cells)
    return out


def mask_to_polygons(mask: xr.DataArray, *, min_pixels: int = 1):
    """Vectorize a boolean mask into polygons with pixel count and area.

    Connected True regions (8-connectivity) smaller than ``min_pixels`` are
    dropped. ``area_m2`` is square metres when ``mask`` has a projected CRS, or
    after projection to an estimated UTM zone when the CRS is geographic. With
    no CRS it is the polygon area in the raster's coordinate units.
    """
    import geopandas as gpd
    from rasterio.errors import RasterioIOError
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from shapely.geometry import shape

    values = mask.values if isinstance(mask, xr.DataArray) else mask
    m = np.asarray(to_cpu(values), bool)
    if min_pixels > 1:
        from ..core.morphology import remove_small

        m = remove_small(m, min_pixels)
    transform, crs = Affine.identity(), None
    if isinstance(mask, xr.DataArray):
        try:
            transform = mask.rio.transform()
            crs = mask.rio.crs
        except (AttributeError, RasterioIOError, ValueError):
            transform, crs = Affine.identity(), None
    geoms = [shape(geom) for geom, val in shapes(m.astype("uint8"), mask=m, transform=transform,
                                                  connectivity=8) if int(val) == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if gdf.empty:
        gdf["pixels"] = []
        gdf["area_m2"] = []
        gdf["area_km2"] = []
        return gdf
    pixel_area = abs(float(transform.a) * float(transform.e)) or 1.0
    gdf["pixels"] = np.maximum(np.round(gdf.geometry.area / pixel_area), 1).astype(int)
    if crs is not None and crs.is_geographic:
        area = gdf.to_crs(gdf.estimate_utm_crs()).area.to_numpy()
    else:
        area = gdf.geometry.area.to_numpy()
    gdf["area_m2"] = area
    gdf["area_km2"] = area / 1e6
    return gdf


def rice_map(vh: xr.DataArray, *, db_input: bool = False, dim: str = "time",
             flood_db: float = -20.0, rise_db: float = 6.0, window_days: int = 70,
             min_peak_db: float = -16.0) -> xr.Dataset:
    """Paddy rice mapping from a VH time series using the transplanting flood signature.

    A pixel is rice if its backscatter drops below ``flood_db`` (flooded field at
    transplanting) and then rises by at least ``rise_db`` within ``window_days`` as the
    canopy grows, reaching at least ``min_peak_db``. Returns the rice mask and the
    estimated transplanting date (time of the VH minimum).
    """
    db = vh if db_input else to_db(vh)
    db = db.transpose(dim, ...)
    a = np.asarray(to_cpu(db).values, dtype="float64")
    t = db[dim].values.astype("datetime64[D]")
    filled = np.where(np.isfinite(a), a, np.inf)
    imin = np.argmin(filled, axis=0)
    vmin = np.take_along_axis(a, imin[None], 0)[0]
    days = (t[None, :] - t[imin.ravel()][:, None]).astype(int).T.reshape((len(t),) + imin.shape)
    after = (days > 0) & (days <= window_days)
    peak = np.nanmax(np.where(after, a, -np.inf), axis=0)
    rice = (vmin < flood_db) & (peak - vmin >= rise_db) & (peak >= min_peak_db)
    template = db.isel({dim: 0}, drop=True)
    trans = np.where(rice, t[imin], np.datetime64("NaT"))
    return xr.Dataset({"rice": template.copy(data=rice),
                       "transplanting_date": template.copy(data=trans),
                       "min_db": template.copy(data=vmin), "rise_db": template.copy(data=peak - vmin)})


def ship_detection(img: xr.DataArray, *, land_mask: xr.DataArray | None = None,
                   pfa: float = 1e-7, guard: int = 4, background: int = 16,
                   min_pixels: int = 3, max_pixels: int = 2000):
    """CFAR ship detection on linear intensity (VH or HV best). ``land_mask`` True = land.

    Returns a GeoDataFrame of detections (centroid, pixel count, peak intensity).
    """
    det = cfar(img, guard=guard, background=background, pfa=pfa)
    if land_mask is not None:
        det = det & ~land_mask.astype(bool)
    pts = detections_to_points(det, img, min_pixels=min_pixels)
    return pts[pts["pixels"] <= max_pixels].reset_index(drop=True)


def oil_spill_detection(vv: xr.DataArray, *, db_input: bool = False, window: int = 51,
                        contrast_db: float = 3.0, min_pixels: int = 50,
                        land_mask: xr.DataArray | None = None,
                        low_wind_db: float = -22.0) -> xr.Dataset:
    """Dark-spot detection for oil-slick candidates (VV).

    Pixels darker than the local background mean by ``contrast_db`` are flagged, excluding
    areas whose background is itself very dark (< ``low_wind_db``, typical of low wind,
    a common look-alike). Candidates still need verification (look-alike discrimination).
    """
    from ..core._utils import nan_box_mean, run_array

    db = vv if db_input else to_db(vv)
    bg = run_array(db, lambda x, xp: nan_box_mean(x, window, xp))
    cand = (db < bg - contrast_db) & (bg > low_wind_db)
    if land_mask is not None:
        cand = cand & ~land_mask.astype(bool)
    m = _clean(np.asarray(to_cpu(cand).values, bool), min_pixels)
    return xr.Dataset({"dark_spot": db.copy(data=m), "background_db": bg})


def change_map(before: xr.DataArray, after: xr.DataArray, *, threshold_db: float | None = None
               ) -> xr.Dataset:
    """Two-date increase/decrease map from the log-ratio (Otsu on |ratio| if no threshold)."""
    lr = log_ratio(before, after)
    th = threshold_db if threshold_db is not None else threshold(np.abs(to_cpu(lr).values))
    return xr.Dataset({"log_ratio": lr, "increase": lr > th, "decrease": lr < -th},
                      attrs={"threshold_db": float(th)})


def un_spider_flood_mapping(
    pre: xr.DataArray,
    post: xr.DataArray,
    *,
    units: str = "db",
    ratio_threshold: float = 0.80,
    smooth_size: int = 3,
    min_pixels: int = 8,
    permanent_water: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    max_slope_degrees: float = 5.0,
    population: xr.DataArray | None = None,
    landcover: xr.DataArray | None = None,
    cropland_codes: tuple[int, ...] = (12, 14),
    urban_codes: tuple[int, ...] = (13,),
) -> xr.Dataset:
    """Complete UN-SPIDER Sentinel-1 flood mapping workflow.

    This function implements the full local flood mapping workflow from the UN-SPIDER
    recommended practice. It processes pre-flood and post-flood Sentinel-1 backscatter
    images to detect flood extent and estimate exposure.

    Parameters
    ----------
    pre : xr.DataArray
        Pre-flood backscatter image (linear or dB).
    post : xr.DataArray
        Post-flood backscatter image (linear or dB).
    units : str, default "db"
        Units of input backscatter: "db" or "linear".
    ratio_threshold : float, default 0.80
        Post/pre linear ratio threshold. Pixels with ratio < threshold are flagged as flood.
    smooth_size : int, default 3
        Odd median-filter window size in pixels for speckle suppression.
    min_pixels : int, default 8
        Minimum connected component size in pixels (8-neighbour).
    permanent_water : xr.DataArray, optional
        Permanent water mask (nonzero = permanent water). These pixels are excluded.
    slope : xr.DataArray, optional
        Slope raster in degrees. Pixels steeper than max_slope_degrees are excluded.
    max_slope_degrees : float, default 5.0
        Maximum slope threshold in degrees.
    population : xr.DataArray, optional
        Population raster (people per pixel). Used for exposure estimation.
    landcover : xr.DataArray, optional
        Land cover raster with MODIS IGBP class codes.
    cropland_codes : tuple of int, default (12, 14)
        MODIS IGBP codes for cropland classes.
    urban_codes : tuple of int, default (13,)
        MODIS IGBP code for urban class.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - flood: boolean flood mask
        - ratio: post/pre linear backscatter ratio
        - pre_smoothed: smoothed pre-flood backscatter (linear)
        - post_smoothed: smoothed post-flood backscatter (linear)
        - exposed_population: (if population provided) exposed people per pixel
        - affected_cropland: (if landcover provided) cropland flood mask
        - affected_urban: (if landcover provided) urban flood mask
        - flood_area_ha: total flood area in hectares
        - flood_pixel_count: number of flood pixels

    Notes
    -----
    This follows the UN-SPIDER recommended practice:
    https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step

    The workflow:
    1. Convert to linear power if input is dB
    2. Apply median smoothing to suppress speckle
    3. Calculate post/pre backscatter ratio
    4. Flag pixels with ratio < threshold (new open water)
    5. Remove permanent water
    6. Remove steep slopes
    7. Remove small connected components
    8. Calculate exposure statistics (if auxiliary layers provided)
    """
    from scipy.ndimage import median_filter
    from ..core.morphology import remove_small

    # Convert to linear power
    if units.lower() == "db":
        pre_lin = 10 ** (pre / 10)
        post_lin = 10 ** (post / 10)
    else:
        pre_lin = pre.copy()
        post_lin = post.copy()

    # Median smoothing
    if smooth_size < 1 or smooth_size % 2 == 0:
        raise ValueError("smooth_size must be a positive odd integer")

    def smooth_with_nan(arr, size):
        """Median filter that handles NaN values."""
        valid = np.isfinite(arr)
        filled = np.where(valid, arr, 0)
        smoothed = median_filter(filled, size=size, mode="nearest")
        valid_smoothed = median_filter(valid.astype(float), size=size, mode="nearest")
        result = smoothed.astype(float)
        result[valid_smoothed < 0.5] = np.nan
        result[~valid] = np.nan
        return result

    pre_smoothed = smooth_with_nan(pre_lin.values, smooth_size)
    post_smoothed = smooth_with_nan(post_lin.values, smooth_size)

    # Calculate ratio
    ratio = np.divide(
        post_smoothed, pre_smoothed,
        out=np.full_like(post_smoothed, np.nan),
        where=np.isfinite(pre_smoothed) & np.isfinite(post_smoothed) & (pre_smoothed > 0),
    )

    # Initial flood detection (ratio < threshold)
    valid = np.isfinite(ratio)
    flood = valid & (ratio < ratio_threshold)

    # Remove permanent water
    if permanent_water is not None:
        perm_mask = np.asarray(permanent_water.values, bool)
        flood &= ~perm_mask

    # Remove steep slopes
    if slope is not None:
        slope_mask = np.asarray(slope.values, bool)
        flood &= ~slope_mask

    # Remove small components
    if min_pixels > 1:
        flood = remove_small(flood, min_pixels)

    # Calculate pixel area
    transform = pre.rio.transform() if hasattr(pre, "rio") else None
    if transform is not None:
        pixel_area_m2 = abs(float(transform.a) * float(transform.e))
    else:
        pixel_area_m2 = 1.0

    # Build result dataset
    result = xr.Dataset({
        "flood": pre.copy(data=flood),
        "ratio": pre.copy(data=ratio),
        "pre_smoothed": pre.copy(data=pre_smoothed),
        "post_smoothed": pre.copy(data=post_smoothed),
    })

    # Calculate flood area
    flood_pixels = int(np.count_nonzero(flood))
    flood_area_ha = flood_pixels * pixel_area_m2 / 10000.0
    result.attrs["flood_area_ha"] = float(flood_area_ha)
    result.attrs["flood_pixel_count"] = flood_pixels
    result.attrs["ratio_threshold"] = float(ratio_threshold)
    result.attrs["backscatter_input_units"] = units
    result.attrs["smoothing_filter_size_pixels"] = int(smooth_size)
    result.attrs["minimum_connected_component_pixels"] = int(min_pixels)

    # Exposure estimation
    if population is not None:
        pop_values = np.asarray(population.values, float)
        exposed = np.where(flood & np.isfinite(pop_values) & (pop_values >= 0), pop_values, 0)
        result["exposed_population"] = pre.copy(data=exposed)
        result.attrs["estimated_exposed_population"] = float(np.nansum(exposed))

    if landcover is not None:
        lc_values = np.asarray(landcover.values, int)
        lc_valid = np.isfinite(lc_values)

        # Cropland
        crop_mask = flood & lc_valid & np.isin(lc_values, cropland_codes)
        result["affected_cropland"] = pre.copy(data=crop_mask)
        crop_ha = int(np.count_nonzero(crop_mask)) * pixel_area_m2 / 10000.0
        result.attrs["potentially_affected_cropland_ha"] = float(crop_ha)

        # Urban
        urban_mask = flood & lc_valid & np.isin(lc_values, urban_codes)
        result["affected_urban"] = pre.copy(data=urban_mask)
        urban_ha = int(np.count_nonzero(urban_mask)) * pixel_area_m2 / 10000.0
        result.attrs["potentially_affected_urban_ha"] = float(urban_ha)

    return result


