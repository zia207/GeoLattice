"""Optical preprocessing: radiometric calibration, simple atmospheric correction,
topographic correction, pan-sharpening and co-registration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from .bands import as_dataset


# ----------------------------------------------------------------------------- radiometry
def radiance(dn: xr.DataArray, mult: float, add: float) -> xr.DataArray:
    """Level-1 DN to top-of-atmosphere spectral radiance: ``L = M * DN + A``."""
    return (dn.where(dn != 0) * mult + add).astype("float32").assign_attrs(units="W/(m2 sr um)")


def toa_reflectance(dn: xr.DataArray, mult: float, add: float, sun_elevation: float
                    ) -> xr.DataArray:
    """Landsat Level-1 DN to TOA reflectance with sun-angle correction:
    ``rho = (M_rho * DN + A_rho) / sin(sun_elevation)`` (values from the MTL file)."""
    rho = (dn.where(dn != 0) * mult + add) / np.sin(np.radians(sun_elevation))
    return rho.astype("float32").assign_attrs(units="reflectance")


def parse_mtl(path: str) -> dict[str, Any]:
    """Parse a Landsat ``*_MTL.txt`` metadata file into a flat dict of numbers/strings."""
    meta: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            v = v.strip('"')
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v
    return meta


def dos1(obj: xr.Dataset | xr.DataArray, *, percentile: float = 0.01,
         dark_reflectance: float = 0.01) -> xr.Dataset | xr.DataArray:
    """Dark Object Subtraction (DOS1, Chavez 1996) on TOA reflectance.

    For each band the darkest pixels (``percentile`` %) are assumed to have
    ``dark_reflectance`` (1%); the excess is atmospheric path radiance and is removed.
    A quick first-order correction when no surface-reflectance product exists.
    """
    def one(v: xr.DataArray) -> xr.DataArray:
        dark = float(np.nanpercentile(to_cpu(v).values, percentile))
        return (v - (dark - dark_reflectance)).clip(min=0).assign_attrs(v.attrs)

    if isinstance(obj, xr.Dataset):
        return obj.map(lambda v: one(v) if np.issubdtype(v.dtype, np.floating) else v)
    return one(obj)


# ----------------------------------------------------------------------------- topography
def topographic_correction(obj: xr.Dataset | xr.DataArray, dem: xr.DataArray,
                           sun_azimuth: float, sun_elevation: float, *,
                           method: Literal["c", "cosine", "minnaert", "scs+c"] = "c",
                           mask: xr.DataArray | None = None) -> xr.Dataset | xr.DataArray:
    """Remove terrain illumination effects from reflectance.

    Methods: ``cosine`` (Teillet 1982), ``c`` correction (Teillet 1982), ``minnaert``
    (Minnaert 1941) and ``scs+c`` (Soenen et al. 2005). Empirical coefficients are fit per
    band by regression against cos(i), optionally only over ``mask`` (e.g. vegetation).
    The DEM must be on the image grid (use ``dem.rio.reproject_match(image)``).
    """
    from ..lidar.terrain import illumination, slope

    cos_i = illumination(dem, sun_azimuth, sun_elevation)
    cos_z = float(np.cos(np.radians(90 - sun_elevation)))
    s = slope(dem, units="radians")
    ci = np.asarray(to_cpu(cos_i).values, dtype="float64")
    sel = np.isfinite(ci) & (ci > 0.05)
    if mask is not None:
        sel &= np.asarray(to_cpu(mask).values, dtype=bool)

    def one(v: xr.DataArray) -> xr.DataArray:
        if method == "cosine":
            out = v * cos_z / cos_i.clip(min=0.05)
        else:
            y = np.asarray(to_cpu(v).values, dtype="float64")
            ok = sel & np.isfinite(y) & (y > 0)
            if ok.sum() < 100:
                raise ValueError("Too few valid pixels to fit topographic correction coefficients")
            if method == "minnaert":
                xx = np.log(ci[ok] / cos_z)
                k = np.polyfit(xx, np.log(y[ok]), 1)[0]
                out = v * (cos_z / cos_i.clip(min=0.05)) ** k
            else:
                m, b = np.polyfit(ci[ok], y[ok], 1)
                c = b / m if m != 0 else 0.0
                if method == "c":
                    out = v * (cos_z + c) / (cos_i + c)
                elif method == "scs+c":
                    out = v * (np.cos(s) * cos_z + c) / (cos_i + c)
                else:
                    raise ValueError(f"Unknown method {method!r}")
        return out.astype("float32").assign_attrs(v.attrs)

    if isinstance(obj, xr.Dataset):
        return obj.map(lambda v: one(v) if np.issubdtype(v.dtype, np.floating) else v,
                       keep_attrs=True)
    return one(obj)


# ----------------------------------------------------------------------------- pan-sharpening
def pansharpen(ms: xr.Dataset | xr.DataArray, pan: xr.DataArray, *,
               method: Literal["brovey", "ihs", "hpf"] = "brovey",
               bands: tuple[str, ...] = ("red", "green", "blue"), hpf_size: int = 5
               ) -> xr.Dataset:
    """Fuse coarse multispectral bands with a fine panchromatic band.

    The multispectral data are first resampled (bilinear) to the pan grid.
    ``brovey`` scales each band by pan/intensity; ``ihs`` (fast IHS) adds pan - intensity;
    ``hpf`` adds the pan high-pass detail (pan minus its ``hpf_size`` box mean).
    """
    from rasterio.enums import Resampling

    ds = as_dataset(ms)
    names = [b for b in bands if b in ds.data_vars] or list(ds.data_vars)
    up = xr.Dataset({b: to_cpu(ds[b]).rio.reproject_match(pan, resampling=Resampling.bilinear)
                     for b in names})
    p = to_cpu(pan).astype("float32")
    intensity = sum(up[b] for b in names) / len(names)
    if method == "brovey":
        out = xr.Dataset({b: up[b] * p / intensity.where(intensity != 0) for b in names})
    elif method == "ihs":
        out = xr.Dataset({b: up[b] + (p - intensity) for b in names})
    elif method == "hpf":
        from scipy.ndimage import uniform_filter

        detail = p - p.copy(data=uniform_filter(np.nan_to_num(p.values), hpf_size))
        out = xr.Dataset({b: up[b] + detail for b in names})
    else:
        raise ValueError(f"Unknown method {method!r}")
    out = out.astype("float32").rio.write_crs(pan.rio.crs)
    out.attrs = {**ds.attrs, "pyrsense:pansharpen": method}
    return out


# ----------------------------------------------------------------------------- co-registration
def coregister(target: xr.DataArray, reference: xr.DataArray, *, method: str = "phase",
               upsample: int = 20, max_shift: float | None = None,
               apply: bool = True, **kwargs: Any) -> tuple[xr.DataArray | None, Mapping[str, float]]:
    """Estimate (and optionally apply) the translation aligning ``target`` to ``reference``.

    ``phase`` uses sub-pixel phase correlation (scikit-image) on the overlap; ``arosics``
    delegates to AROSICS global co-registration if installed. Both rasters must share
    a grid (use ``reproject_match`` first). Returns ``(shifted, {"dy", "dx", "error"})``
    with shifts in pixels.
    """
    if method == "arosics":
        return _coreg_arosics(target, reference, **kwargs)
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    t = np.asarray(to_cpu(target).values, dtype="float64")
    r = np.asarray(to_cpu(reference).values, dtype="float64")
    valid = np.isfinite(t) & np.isfinite(r)
    tf = np.where(valid, t - np.nanmean(t), 0)
    rf = np.where(valid, r - np.nanmean(r), 0)
    shift, error, _ = phase_cross_correlation(rf, tf, upsample_factor=upsample, normalization=None)
    dy, dx = float(shift[0]), float(shift[1])
    if max_shift is not None and max(abs(dy), abs(dx)) > max_shift:
        raise ValueError(f"Estimated shift ({dy:.2f}, {dx:.2f}) px exceeds max_shift={max_shift}")
    info = {"dy": dy, "dx": dx, "error": float(error)}
    if not apply:
        return None, info
    moved = nd_shift(np.nan_to_num(t, nan=0.0), (dy, dx), order=1, mode="constant", cval=np.nan)
    nanmask = nd_shift(np.isnan(t).astype(float), (dy, dx), order=0, mode="constant", cval=1) > 0.5
    moved[nanmask] = np.nan
    return target.copy(data=moved.astype("float32")), info


def _coreg_arosics(target, reference, **kwargs):
    import os
    import tempfile

    try:
        from arosics import COREG
    except ImportError as e:
        raise ImportError("method='arosics' needs `pip install arosics`") from e
    from ..core.raster import read, write

    with tempfile.TemporaryDirectory() as d:
        tp, rp, op = (os.path.join(d, n) for n in ("t.tif", "r.tif", "o.tif"))
        write(target, tp, cog=False)
        write(reference, rp, cog=False)
        cr = COREG(rp, tp, path_out=op, fmt_out="GTIFF", q=True, **kwargs)
        cr.calculate_spatial_shifts()
        cr.correct_shifts()
        out = read(op).squeeze("band", drop=True).load()
        return out, {"dx": float(cr.x_shift_px), "dy": float(cr.y_shift_px), "error": float("nan")}
