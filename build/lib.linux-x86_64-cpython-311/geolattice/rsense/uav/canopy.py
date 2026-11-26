"""Canopy products from UAV photogrammetry: CHM, canopy cover, plant counting."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..core._utils import pix_to_map


def _same_grid(a: xr.DataArray, b: xr.DataArray) -> bool:
    if a.shape != b.shape:
        return False
    # absolute tolerance: a fraction of a pixel (relative tolerances accept metre-scale
    # offsets on large projected coordinates)
    tol = 1e-3 * abs(float(a.x[1] - a.x[0])) if a.sizes["x"] > 1 else 1e-9
    if a.rio.crs != b.rio.crs:
        return False
    return bool(np.allclose(a.x, b.x, rtol=0, atol=tol) and np.allclose(a.y, b.y, rtol=0, atol=tol))


def chm_from_surfaces(dsm: xr.DataArray, dtm: xr.DataArray, *, min_height: float = 0.0,
                      max_height: float | None = None) -> xr.DataArray:
    """Canopy height = DSM - DTM.

    The DTM is resampled onto the DSM grid whenever the CRS, transform or shape differ
    (not only when the shapes differ), so same-size rasters that are offset are aligned
    before subtracting. ``min_height`` clips small negatives; heights above ``max_height``
    become NaN.
    """
    if "band" in dsm.dims:
        dsm = dsm.squeeze("band", drop=True)
    if "band" in dtm.dims:
        dtm = dtm.squeeze("band", drop=True)
    if not _same_grid(dsm, dtm):
        if dsm.rio.crs is None or dtm.rio.crs is None:
            raise ValueError("DSM and DTM are on different grids and lack a CRS to align them")
        dtm = dtm.rio.reproject_match(dsm)
    d = dtm.assign_coords(x=dsm.x, y=dsm.y)
    h = (dsm - d).clip(min=min_height)
    if max_height is not None:
        h = h.where(h <= max_height)
    if dsm.rio.crs is not None:
        h = h.rio.write_crs(dsm.rio.crs)
    return h.rename("chm").assign_attrs(units="m")


def canopy_cover(veg_index: xr.DataArray, threshold: float | None = None,
                 method: Literal["fixed", "otsu"] = "otsu") -> tuple[xr.DataArray, float]:
    """Binary vegetation mask and fractional cover from a vegetation index (e.g. ExG, NDVI)."""
    a = np.asarray(to_cpu(veg_index).values, dtype="float64")
    if threshold is None:
        if method != "otsu":
            raise ValueError("Give threshold= for method='fixed'")
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(a[np.isfinite(a)]))
    mask = veg_index > threshold
    return mask.rename("vegetation"), float(mask.where(veg_index.notnull()).mean())


def _as_layer(layer, gsd: float | None) -> xr.DataArray:
    if not isinstance(layer, xr.DataArray):
        from .io import grid_coords

        a = np.asarray(layer, dtype="float32")
        layer = xr.DataArray(a, dims=("y", "x"), coords=grid_coords(*a.shape, gsd or 1.0))
    if "band" in layer.dims:
        if layer.sizes["band"] != 1:
            raise ValueError("count_plants needs a single-band layer (e.g. a CHM or one index)")
        layer = layer.squeeze("band", drop=True)
    return layer


def count_plants(layer, *, min_distance_m: float = 0.3, threshold: float | None = None,
                 sigma_m: float = 0.05, method: Literal["peaks", "blobs"] = "peaks",
                 gsd: float | None = None):
    """Detect individual plants (or trees) as local maxima of a CHM or vegetation index.

    ``peaks`` uses non-maximum suppression with ``min_distance_m``; ``blobs`` uses a
    Laplacian-of-Gaussian detector sized by ``min_distance_m``. Distances are in the
    layer's coordinate units: metres for GeoTIFFs and for frames read with a ``gsd``,
    pixels for frames without one (then pass distances in pixels). ``gsd`` overrides the
    pixel size and ``layer`` may also be a plain 2-D array.

    Returns a GeoDataFrame of plant centres with ``value``, ``row`` and ``col`` (CRS kept
    when the layer has one).
    """
    import geopandas as gpd
    from scipy import ndimage as snd

    layer = _as_layer(layer, gsd)
    a = np.nan_to_num(np.asarray(to_cpu(layer).values, dtype="float64"))
    if gsd:
        res = float(gsd)
    else:
        rx = abs(float(layer.x[1] - layer.x[0])) if layer.sizes["x"] > 1 else 1.0
        ry = abs(float(layer.y[1] - layer.y[0])) if layer.sizes["y"] > 1 else rx
        res = (rx + ry) / 2
    if sigma_m:
        a = snd.gaussian_filter(a, sigma_m / res)
    if threshold is None:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(a))
    if method == "peaks":
        from skimage.feature import peak_local_max

        rc = peak_local_max(a, min_distance=max(1, int(round(min_distance_m / res))),
                            threshold_abs=threshold, exclude_border=False)
    else:
        from skimage.feature import blob_log

        s = min_distance_m / res / 2
        blobs = blob_log(np.where(a > threshold, a, 0), min_sigma=max(s / 2, 1), max_sigma=s * 2,
                         threshold=0.02)
        rc = blobs[:, :2].astype(int)
    crs = layer.rio.crs
    if len(rc) == 0:
        return gpd.GeoDataFrame({"value": [], "row": [], "col": []}, geometry=[], crs=crs)
    if gsd and crs is None:  # explicit pixel size on an ungeoreferenced frame
        xs = (rc[:, 1] + 0.5) * res
        ys = -(rc[:, 0] + 0.5) * res
    else:
        T = layer.rio.transform()
        xs, ys = pix_to_map(T, rc[:, 1] + 0.5, rc[:, 0] + 0.5)
    vals = np.asarray(to_cpu(layer).values)[rc[:, 0], rc[:, 1]]
    return gpd.GeoDataFrame({"value": vals, "row": rc[:, 0], "col": rc[:, 1]},
                            geometry=gpd.points_from_xy(xs, ys), crs=crs)


def plants_per_area(points, polygons, area_unit: float = 1.0):
    """Plant density per polygon (count / area in ``area_unit`` m², e.g. 1e4 for per-hectare).

    Polygons in a geographic CRS (degrees) are projected to their UTM zone first so the
    areas are in m². Without any CRS, areas are in the polygons' own units.
    """
    import geopandas as gpd

    polys = polygons
    if polys.crs is not None and polys.crs.is_geographic:
        polys = polys.to_crs(polys.estimate_utm_crs())
    pts = points.to_crs(polys.crs) if points.crs is not None and polys.crs is not None else points
    joined = gpd.sjoin(pts, polys, predicate="within")
    counts = joined.groupby("index_right").size()
    out = polygons.copy()
    out["count"] = counts.reindex(out.index, fill_value=0).values
    out["area"] = polys.geometry.area.values / area_unit
    out["density"] = out["count"] / out["area"]
    return out
