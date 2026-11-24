"""Gridding point clouds to DEM, DSM and CHM rasters (GPU-capable binning)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy import ndimage as snd

from ..backend import get_xp, scatter_reduce, to_cpu
from ..core._utils import to_xp
from ..core.pointcloud import PointCloud


def _grid_spec(pc: PointCloud, res: float, bounds=None):
    minx, miny, maxx, maxy = bounds or pc.bbox
    x0 = np.floor(minx / res) * res
    y1 = np.ceil(maxy / res) * res
    W = int(np.ceil((maxx - x0) / res)) or 1
    H = int(np.ceil((y1 - miny) / res)) or 1
    return x0, y1, H, W


def _to_dataarray(grid: np.ndarray, x0: float, y1: float, res: float, crs: Any, name: str):
    H, W = grid.shape
    da = xr.DataArray(grid, dims=("y", "x"), name=name,
                      coords={"y": y1 - res * (np.arange(H) + 0.5),
                              "x": x0 + res * (np.arange(W) + 0.5)})
    return da.rio.write_crs(crs) if crs is not None else da


def rasterize(pc: PointCloud, res: float = 1.0, *, value: str = "z",
              stat: Literal["min", "max", "mean", "count", "sum"] = "max",
              bounds: tuple[float, float, float, float] | None = None,
              name: str | None = None) -> xr.DataArray:
    """Bin points onto a grid with a statistic, on the active backend (scatter reductions)."""
    xp = get_xp()
    x0, y1, H, W = _grid_spec(pc, res, bounds)
    px, py, v = to_xp(pc.x, xp), to_xp(pc.y, xp), to_xp(pc[value], xp)
    col = xp.floor((px - x0) / res).astype(xp.int64)
    row = xp.floor((y1 - py) / res).astype(xp.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    idx = (row * W + col)[ok]
    vals = v[ok].astype(xp.float64)
    if stat in ("min", "max"):
        g = xp.full(H * W, np.inf if stat == "min" else -np.inf)
        scatter_reduce(g, idx, vals, stat)
        g[xp.isinf(g)] = xp.nan
    else:
        s = xp.zeros(H * W)
        n = xp.zeros(H * W)
        scatter_reduce(n, idx, xp.ones_like(vals), "add")
        if stat == "count":
            g = n
        else:
            scatter_reduce(s, idx, vals, "add")
            g = s if stat == "sum" else xp.where(n > 0, s / xp.maximum(n, 1), xp.nan)
    grid = np.asarray(to_cpu(g)).reshape(H, W).astype("float32")
    return _to_dataarray(grid, x0, y1, res, pc.crs, name or f"{value}_{stat}")


def fill_gaps(grid: xr.DataArray, method: Literal["nearest", "idw", "linear"] = "idw",
              max_distance: float | None = None, power: float = 2.0) -> xr.DataArray:
    """Fill NaN cells: ``nearest``, inverse-distance from the 8 nearest cells (``idw``) or
    Delaunay ``linear`` (TIN) interpolation. ``max_distance`` (map units) limits filling."""
    a = np.asarray(grid.values, dtype="float64")
    nan = np.isnan(a)
    if not nan.any():
        return grid
    res = abs(float(grid.x[1] - grid.x[0])) if grid.sizes["x"] > 1 else 1.0
    if method == "nearest":
        dist, idx = snd.distance_transform_edt(nan, return_indices=True)
        out = a[tuple(idx)]
    elif method == "linear":
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        rr, cc = np.nonzero(~nan)
        f = LinearNDInterpolator(np.column_stack([rr, cc]), a[~nan])
        qr, qc = np.nonzero(nan)
        out = a.copy()
        vals = f(qr, qc)
        miss = np.isnan(vals)
        if miss.any():
            vals[miss] = NearestNDInterpolator(np.column_stack([rr, cc]), a[~nan])(qr[miss], qc[miss])
        out[qr, qc] = vals
        dist = snd.distance_transform_edt(nan)
    elif method == "idw":
        from scipy.spatial import cKDTree

        rr, cc = np.nonzero(~nan)
        tree = cKDTree(np.column_stack([rr, cc]))
        qr, qc = np.nonzero(nan)
        d, i = tree.query(np.column_stack([qr, qc]), k=min(8, len(rr)))
        d = np.atleast_2d(d)
        i = np.atleast_2d(i)
        w = 1.0 / np.maximum(d, 1e-9) ** power
        out = a.copy()
        out[qr, qc] = (w * a[~nan][i]).sum(1) / w.sum(1)
        dist = snd.distance_transform_edt(nan)
    else:
        raise ValueError(f"Unknown method {method!r}")
    if max_distance is not None:
        out = np.where(dist * res > max_distance, np.nan, out)
    return grid.copy(data=out.astype("float32"))


def dem(pc: PointCloud, res: float = 1.0, *, fill: str = "linear",
        bounds: tuple[float, float, float, float] | None = None) -> xr.DataArray:
    """Digital Elevation (terrain) Model from ground points (class 2), gaps interpolated."""
    g = pc.ground()
    if len(g) == 0:
        raise ValueError("No ground points; run lidar.classify_ground first")
    return fill_gaps(rasterize(g, res, stat="mean", bounds=bounds or pc.bbox, name="dem"), fill)


def dsm(pc: PointCloud, res: float = 1.0, *, first_returns: bool = True, fill: str = "idw",
        bounds: tuple[float, float, float, float] | None = None) -> xr.DataArray:
    """Digital Surface Model: highest (first-return) elevation per cell."""
    p = pc.drop_noise()
    p = p.first_returns() if first_returns else p
    return fill_gaps(rasterize(p, res, stat="max", bounds=bounds or pc.bbox, name="dsm"), fill)


def chm(pc_or_dsm: PointCloud | xr.DataArray, dtm: xr.DataArray | None = None, res: float = 1.0,
        *, pit_free: bool = True, pit_size: int = 3, min_height: float = 0.0) -> xr.DataArray:
    """Canopy Height Model = DSM − DEM (clipped at ``min_height``).

    Pass a PointCloud (DEM and DSM are derived) or a DSM DataArray plus ``dtm``. With
    ``pit_free`` a median filter removes canopy pits (inter-crown gaps of single pixels).
    """
    if isinstance(pc_or_dsm, PointCloud):
        d = dem(pc_or_dsm, res)
        s = dsm(pc_or_dsm, res, bounds=pc_or_dsm.bbox)
        s = s.rio.reproject_match(d) if s.shape != d.shape else s
    else:
        if dtm is None:
            raise ValueError("Pass dtm= when giving a DSM raster")
        s, d = pc_or_dsm, dtm.rio.reproject_match(pc_or_dsm)
    h = (s - d.values).clip(min=min_height)
    if pit_free:
        a = np.nan_to_num(h.values)
        filt = snd.median_filter(a, size=pit_size)
        pits = (filt - a) > 1.0
        h = h.copy(data=np.where(pits, filt, h.values))
    return h.rename("chm").assign_attrs(units="m")


def normalize_heights(pc: PointCloud, dtm: xr.DataArray) -> PointCloud:
    """Add ``height`` (z minus terrain, bilinear from ``dtm``) to every point."""
    p = pc.to_cpu()
    res = abs(float(dtm.x[1] - dtm.x[0]))
    x0 = float(dtm.x[0]) - res / 2
    y1 = float(dtm.y[0]) + res / 2
    fr = (y1 - p.y) / res - 0.5
    fc = (p.x - x0) / res - 0.5
    ground = snd.map_coordinates(np.nan_to_num(dtm.values.astype("float64")), [fr, fc],
                                 order=1, mode="nearest")
    return pc.with_field("height", (p.z - ground).astype("float32") if pc.xp is np
                         else pc.xp.asarray((p.z - ground).astype("float32")))


def sample_raster(pc: PointCloud, raster: xr.DataArray, name: str | None = None, *,
                  method: Literal["nearest", "bilinear"] = "nearest") -> PointCloud:
    """Attach raster values (DTM, NDVI, land cover...) to every point as field ``name``."""
    from ..core._utils import map_to_pix

    p = pc.to_cpu()
    r = raster.rio.reproject(pc.crs) if (pc.crs is not None and raster.rio.crs is not None and
                                         raster.rio.crs != _crs_obj(pc.crs)) else raster
    a = np.asarray(to_cpu(r).values, dtype="float64")
    cols, rows = map_to_pix(r.rio.transform(), np.asarray(p.x), np.asarray(p.y))
    if method == "bilinear":
        vals = snd.map_coordinates(a, [rows - 0.5, cols - 0.5], order=1, mode="nearest",
                                   cval=np.nan)
    else:
        ri, ci = np.floor(rows).astype(int), np.floor(cols).astype(int)
        ok = (ri >= 0) & (ri < a.shape[0]) & (ci >= 0) & (ci < a.shape[1])
        vals = np.full(len(p), np.nan)
        vals[ok] = a[ri[ok], ci[ok]]
    return pc.with_field(name or str(raster.name or "value"), vals.astype("float32"))


def _crs_obj(crs):
    from pyproj import CRS

    return CRS.from_user_input(crs)


def colorize(pc: PointCloud, image: xr.Dataset | xr.DataArray, *,
             bands: tuple[str, str, str] | tuple[int, int, int] = ("red", "green", "blue"),
             scale: Literal["auto"] | float = "auto", nir: str | int | None = None) -> PointCloud:
    """Colour points from co-registered imagery (e.g. NAIP orthophotos, Sentinel-2).

    ``image`` is a Dataset with the named ``bands`` or a ``band``-dimension DataArray indexed by
    position. Values are rescaled to the LAS 16-bit range (``auto``: 2nd–98th percentile
    stretch) and written to ``red``, ``green``, ``blue`` (and ``nir`` when given).
    """
    def layer(b):
        if isinstance(image, xr.Dataset):
            return image[b]
        return image.isel(band=b) if isinstance(b, int) else image.sel(band=b)

    names = ["red", "green", "blue"] + (["nir"] if nir is not None else [])
    out = pc
    for nm, b in zip(names, [*bands, nir] if nir is not None else bands):
        v = np.asarray(sample_raster(pc, layer(b), "_v")["_v"], dtype="float64")
        if scale == "auto":
            lo, hi = np.nanpercentile(v, [2, 98]) if np.isfinite(v).any() else (0, 1)
            v = (v - lo) / max(hi - lo, 1e-12) * 65535
        else:
            v = v * float(scale)
        out = out.with_field(nm, np.clip(np.nan_to_num(v), 0, 65535).astype("uint16"))
    return out
