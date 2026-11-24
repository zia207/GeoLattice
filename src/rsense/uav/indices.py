"""Vegetation indices for UAV RGB and RGB + NIR imagery.

The standard formulas (VARI, GLI, ExG, ExR, ExGR, NGRDI, TGI, RGBVI, MGRVI, GCC, NDVI,
GNDVI, SAVI, ...) live in :mod:`geolattice.rsense.optical.indices` and are reused here. This
module adds the variants common in UAV phenotyping that are not band-ratio formulas:
chromatic-coordinate normalisation, CIVE, and the max-normalised TGI and min-max ExG
used in the LUMS rice pipeline. With red and near-infrared digital numbers,
:func:`bare_soil_line` and :func:`cover_layers` estimate green ground cover
(Maas and Rajan 2008).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr
from scipy.stats import linregress

RGB_DEFAULT = ("VARI", "TGI", "ExG", "ExGR", "GLI", "NGRDI")
NIR_DEFAULT = ("NDVI", "GNDVI", "SAVI")
_LOCAL = {"CIVE", "TGI_MAXNORM", "EXG_MINMAX"}


def _bands(img: xr.DataArray | xr.Dataset) -> dict[str, xr.DataArray]:
    if isinstance(img, xr.Dataset):
        return {k: img[k].astype("float32") for k in img.data_vars}
    if "band" not in img.dims:
        raise ValueError("expected a (band, y, x) DataArray or a Dataset of bands")
    return {str(b): img.sel(band=b, drop=True).astype("float32") for b in img.band.values}


def chromatic(img: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """Chromatic coordinates ``r = R/(R+G+B)`` (likewise ``g``, ``b``): removes brightness
    differences between frames and flights before computing RGB indices."""
    b = _bands(img)
    s = b["red"] + b["green"] + b["blue"]
    s = s.where(s > 0)
    return xr.Dataset(
        {k: b[k] / s for k in ("red", "green", "blue")}, attrs=getattr(img, "attrs", {})
    )


def _local(name: str, b: dict[str, xr.DataArray]) -> xr.DataArray:
    R, G, B = b["red"], b["green"], b["blue"]
    if name == "CIVE":  # Kataoka et al. 2003; low values = vegetation
        out = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
        ref = "Kataoka et al. 2003"
    elif name == "TGI_MAXNORM":  # LUMS make_tgi_vari.py: (G - 0.39R - 0.61B) / max(R, G, B)
        mx = np.maximum(np.maximum(R, G), B)
        out = (G - 0.39 * R - 0.61 * B) / mx.where(mx > 0)
        ref = "Hunt et al. 2013 (normalised, Chaudhry et al. 2026)"
    else:  # EXG_MINMAX: ExG rescaled to 0-1 over the image (SAM prompt sampling)
        e = 2 * G - R - B
        lo, hi = float(e.min()), float(e.max())
        out = (e - lo) / (hi - lo if hi > lo else 1.0)
        ref = "Woebbecke et al. 1995 (min-max scaled)"
    return out.assign_attrs(long_name=name, reference=ref)


def vegetation_indices(
    img: xr.DataArray | xr.Dataset,
    names: str | Iterable[str] | None = None,
    *,
    normalize: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Compute UAV vegetation indices from a ``(band, y, x)`` image or a Dataset of bands.

    ``names`` defaults to the RGB set (VARI, TGI, ExG, ExGR, GLI, NGRDI), plus NDVI, GNDVI
    and SAVI when a ``nir`` band is present. Any index in ``rs.optical.list_indices()`` whose
    bands are available works, as do ``CIVE``, ``TGI_MAXNORM`` and ``EXG_MINMAX``.
    ``normalize=True`` first converts R, G, B to chromatic coordinates (NIR is untouched).
    Returns a Dataset, or a DataArray when ``names`` is a single string.
    """
    from ..optical.indices import index

    single = isinstance(names, str)
    b = _bands(img)
    if names is None:
        names = list(RGB_DEFAULT) + (list(NIR_DEFAULT) if "nir" in b else [])
    names = [names] if single else list(names)
    if normalize:
        b.update({k: v for k, v in chromatic(img).data_vars.items()})
    ds = xr.Dataset(b)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in names:
            key = n.upper()
            da = _local(key, b) if key in _LOCAL else index(ds, n, bands={k: k for k in b})
            out[n] = da.where(np.isfinite(da)).rename(n)
    res = xr.Dataset(
        out,
        attrs={
            k: v
            for k, v in getattr(img, "attrs", {}).items()
            if k in ("gsd_m", "units_xy", "source")
        },
    )
    return res[names[0]] if single else res


def _array(band: np.ndarray | xr.DataArray) -> np.ndarray:
    values = band.values if isinstance(band, xr.DataArray) else band
    return np.asarray(values, dtype="float64")


def _major_axis(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Major-axis slope and intercept, as in ``lmodel2::MA.reg``, plus R-squared.

    A perfect line gives the same slope as ordinary least squares. A zero correlation
    falls back to the ordinary slope, because the major-axis formula divides by it.
    """
    if x.size < 2:
        raise ValueError("bare soil line needs at least two red levels")
    ols = linregress(x, y)
    b_ols, r_val = float(ols.slope), float(ols.rvalue)
    rsquare = r_val**2
    if not np.isfinite(b_ols) or abs(rsquare * b_ols) < 1e-15:
        slope = 0.0 if not np.isfinite(b_ols) else b_ols
    else:
        d = (b_ols**2 - rsquare) / (rsquare * b_ols)
        slope = 0.5 * (d + np.sign(r_val) * np.sqrt(d**2 + 4.0))
    intercept = float(y.mean() - slope * x.mean())
    return float(slope), intercept, float(rsquare)


def bare_soil_line(
    red: np.ndarray | xr.DataArray,
    nir: np.ndarray | xr.DataArray,
    *,
    maxval: float = 255,
    ulimit: float = 0.99,
) -> dict[str, float | int]:
    """Fit the minimum bare-soil line to red and near-infrared digital numbers.

    For each integer red level below ``maxval``, the lowest NIR is kept and a major-axis
    line is fit. This is the ``minimum`` method of ``landsat::BSL``, with the major-axis
    slope from ``lmodel2``. The full-canopy point is the pixel, among those whose NIR/red
    ratio is above the ``ulimit`` quantile, closest to high NIR and low red. Digital
    numbers at or above ``maxval`` are left out of the fit. ``red`` and ``nir`` are arrays
    or DataArrays of the same shape.

    Returns ``slope`` (a1), ``intercept`` (a0), ``r_squared``, ``top_red``, ``top_nir``,
    ``n_levels`` and ``n_pixels``.
    """
    r = _array(red).ravel()
    n = _array(nir).ravel()
    if r.shape != n.shape:
        raise ValueError("red and NIR must have the same shape")
    finite = np.isfinite(r) & np.isfinite(n)
    r, n = r[finite], n[finite]
    keep = (r < maxval) & (n < maxval)  # drop saturated digital numbers
    r, n = r[keep], n[keep]
    if r.size == 0:
        raise ValueError("no finite red and NIR values below maxval")

    integer = np.isclose(r, np.rint(r))
    r_int = np.rint(r[integer]).astype(np.int32)
    n_int = n[integer]
    in_range = (r_int >= 1) & (r_int <= int(maxval) - 1)
    r_int, n_int = r_int[in_range], n_int[in_range]
    if r_int.size == 0:
        raise ValueError("no integer red values between 1 and maxval")
    order = np.argsort(r_int, kind="mergesort")
    r_sorted, n_sorted = r_int[order], n_int[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    x = r_sorted[starts].astype("float64")
    y = np.minimum.reduceat(n_sorted, starts).astype("float64")
    slope, intercept, rsquare = _major_axis(x, y)

    positive = r > 0  # a zero red digital number has no NIR/red ratio
    if not np.any(positive):
        raise ValueError("bare soil line needs red digital numbers above 0")
    ratio = n[positive] / r[positive]
    candidates = ratio > np.quantile(ratio, ulimit)
    if not np.any(candidates):
        candidates = np.ones(ratio.shape, dtype=bool)
    rr, nn = r[positive][candidates], n[positive][candidates]
    best = int(np.argmin(rr + (maxval - nn)))
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": rsquare,
        "top_red": float(rr[best]),
        "top_nir": float(nn[best]),
        "n_levels": int(x.size),
        "n_pixels": int(r.size),
    }


def cover_layers(
    img: xr.DataArray | xr.Dataset,
    *,
    maxval: float = 255,
    ulimit: float = 0.99,
    pvi_max: float | None = 100.0,
    red: str = "red",
    nir: str = "nir",
) -> xr.Dataset:
    """Perpendicular vegetation index, ground cover and WDVI from red and NIR.

    Ground cover is the perpendicular vegetation index (PVI) divided by PVI at full
    canopy (Maas and Rajan 2008). PVI below 0 is set to 0. PVI above ``pvi_max`` is set
    to ``pvi_max`` unless ``pvi_max`` is None. WDVI is ``NIR / maxval - slope * Red / maxval``,
    clipped to 0-1. ``img`` is a ``(band, y, x)`` DataArray or a Dataset that has ``red``
    and ``nir`` bands (those names can be changed). Soil-line coefficients and ``pvi_fc``
    are stored on the Dataset ``attrs``. The returned variables are ``PVI``, ``GC`` and ``WDVI``.
    """
    bands = _bands(img)
    missing = [name for name in (red, nir) if name not in bands]
    if missing:
        have = ", ".join(sorted(bands))
        raise ValueError(
            f"cover_layers needs bands {red!r} and {nir!r}; missing {missing} (have {have})"
        )
    red_da, nir_da = bands[red], bands[nir]
    red_dn, nir_dn = _array(red_da), _array(nir_da)
    if red_dn.shape != nir_dn.shape:
        raise ValueError("red and NIR bands have different shapes")
    line = bare_soil_line(red_dn, nir_dn, maxval=maxval, ulimit=ulimit)
    a1, a0 = float(line["slope"]), float(line["intercept"])
    a2 = float(np.sqrt(1.0 + a1**2))
    valid = np.isfinite(red_dn) & np.isfinite(nir_dn)

    pvi = (nir_dn - a1 * red_dn - a0) / a2
    pvi = np.where(pvi < 0.0, 0.0, pvi)
    if pvi_max is not None:
        pvi = np.where(pvi > pvi_max + 0.001, float(pvi_max), pvi)
    pvi_fc = (float(line["top_nir"]) - a1 * float(line["top_red"]) - a0) / a2
    if not np.isfinite(pvi_fc) or pvi_fc <= 0.0:
        raise ValueError("PVI at full cover is not positive")
    gc = np.clip(pvi / pvi_fc, 0.0, 1.0)
    wdvi = np.clip(nir_dn / maxval - (a1 * red_dn / maxval), 0.0, 1.0)
    pvi = np.where(valid, pvi, np.nan)
    gc = np.where(valid, gc, np.nan)
    wdvi = np.where(valid, wdvi, np.nan)

    def layer(values: np.ndarray, name: str, long_name: str) -> xr.DataArray:
        da = red_da.copy(data=np.asarray(values, dtype="float32"))
        da.name = name
        da.attrs = dict(red_da.attrs)
        da.attrs["long_name"] = long_name
        da.attrs["reference"] = "Maas and Rajan 2008"
        return da

    out = xr.Dataset(
        {
            "PVI": layer(pvi, "PVI", "Perpendicular vegetation index"),
            "GC": layer(gc, "GC", "Green ground cover"),
            "WDVI": layer(wdvi, "WDVI", "Weighted difference vegetation index"),
        }
    )
    out.attrs["pvi_fc"] = float(pvi_fc)
    out.attrs.update(line)
    out.attrs["reference"] = "Maas and Rajan 2008"
    return out


def index_summary(
    indices: xr.Dataset,
    mask: xr.DataArray | None = None,
    stats: Sequence[str] = ("mean", "std", "p10", "p50", "p90"),
) -> dict[str, float]:
    """Flat dictionary of index statistics (optionally inside a vegetation ``mask``), handy as
    one row of a multi-date trait table."""
    row = {}
    for n, da in indices.data_vars.items():
        a = np.asarray((da.where(mask) if mask is not None else da).values, float)
        a = a[np.isfinite(a)]
        for s in stats:
            if a.size == 0:
                v = np.nan
            elif s.startswith("p"):
                v = float(np.percentile(a, float(s[1:])))
            else:
                v = float(getattr(np, s)(a))
            row[f"{n}_{s}"] = v
    return row
