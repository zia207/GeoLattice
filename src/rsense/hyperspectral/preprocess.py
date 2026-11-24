"""Hyperspectral preprocessing: bad bands, noise and destriping, quality masks, empirical-line,
flat-field and IAR calibration, smoothing, derivatives, continuum removal, absorption feature
analysis, normalisation and spectral resampling to other sensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu
from .cube import apply_spectral, band_at, band_stats, wavelengths

WATER_VAPOUR_RANGES = ((1340.0, 1455.0), (1790.0, 1960.0), (2450.0, 2600.0))


# ----------------------------------------------------------------------------- bad bands
def bad_bands(cube: xr.DataArray, *, ranges: Sequence[tuple[float, float]] = WATER_VAPOUR_RANGES,
              min_wavelength: float | None = 400.0, min_snr: float | None = None,
              max_nan_fraction: float = 0.5) -> np.ndarray:
    """Boolean mask of bands to drop: atmospheric absorption ``ranges`` (default water vapour
    near 1400, 1900 and >2450 nm), bands below ``min_wavelength``, bands whose estimated
    SNR is below ``min_snr`` and bands that are mostly NaN or constant."""
    wl = wavelengths(cube)
    bad = np.zeros(len(wl), bool)
    for lo, hi in ranges:
        bad |= (wl >= lo) & (wl <= hi)
    if min_wavelength is not None:
        bad |= wl < min_wavelength
    st = band_stats(cube)
    bad |= st.nan_fraction.to_numpy() > max_nan_fraction
    bad |= ~(st["std"].to_numpy() > 0)
    if min_snr is not None:
        bad |= ~(st.snr.to_numpy() >= min_snr)
    if "good" in cube.coords:
        bad |= ~np.asarray(cube["good"].values, bool)
    return bad


def drop_bad_bands(cube: xr.DataArray, **kwargs: Any) -> xr.DataArray:
    """Remove the bands flagged by :func:`bad_bands`."""
    return cube.isel(band=np.flatnonzero(~bad_bands(cube, **kwargs)))


def estimate_noise(cube: xr.DataArray, *, method: Literal["shift", "residual"] = "shift"
                   ) -> np.ndarray:
    """Per-band noise standard deviation.

    ``shift``: horizontal neighbour differences / √2 (assumes neighbours share signal).
    ``residual``: residual of predicting each band from its two spectral neighbours by
    multiple regression (Roger & Arnold 1996), robust to spatial texture.
    """
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    if method == "shift":
        d = np.diff(a, axis=-1).reshape(a.shape[0], -1)
        return np.nanstd(d, axis=1) / np.sqrt(2)
    X = a.reshape(a.shape[0], -1).T
    X = X[np.all(np.isfinite(X), axis=1)]
    out = np.zeros(a.shape[0])
    for b in range(a.shape[0]):
        nb = [i for i in (b - 1, b + 1) if 0 <= i < a.shape[0]]
        A = np.column_stack([X[:, nb], np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, X[:, b], rcond=None)
        out[b] = np.std(X[:, b] - A @ coef)
    return out


def destripe(cube: xr.DataArray, *, axis: Literal["x", "y"] = "x",
             method: Literal["moment", "median"] = "moment") -> xr.DataArray:
    """Remove detector striping in pushbroom data by matching each column's (``axis="x"``)
    mean and standard deviation (moment matching) or median to the band's global value."""
    a = np.array(to_cpu(cube).values, dtype="float64")
    ax = 2 if axis == "x" else 1
    other = 1 if ax == 2 else 2
    if method == "moment":
        m = np.nanmean(a, axis=other, keepdims=True)
        s = np.nanstd(a, axis=other, keepdims=True)
        gm = np.nanmean(a, axis=(1, 2), keepdims=True)
        gs = np.nanstd(a, axis=(1, 2), keepdims=True)
        a = (a - m) / np.where(s > 0, s, 1) * gs + gm
    else:
        med = np.nanmedian(a, axis=other, keepdims=True)
        a = a - med + np.nanmedian(a, axis=(1, 2), keepdims=True)
    return cube.copy(data=a.astype("float32"))


def quality_mask(cube: xr.DataArray, *, cloud_blue: float = 0.25, cirrus: float | None = None,
                 water_ndwi: float = 0.2, shadow_nir: float = 0.04,
                 saturation: float = 1.2) -> xr.Dataset:
    """Simple reflectance-based masks: ``cloud`` (bright at 450 nm and spectrally flat in the
    visible; with ``cirrus`` set, also TOA reflectance above it at 1380 nm, e.g. 0.01 for
    L1 data), ``water`` (NDWI 560/860), ``shadow`` (dark NIR), ``saturated`` and ``valid``.
    Bright white minerals and snow can be flagged as cloud; use the product's own cloud mask
    (EMIT, EnMAP and PRISMA ship one) when available."""
    b = band_at(cube, 450)
    g, r, n = band_at(cube, 560), band_at(cube, 650), band_at(cube, 860)
    wl = wavelengths(cube)
    cloud = (b > cloud_blue) & (abs(b - r) / (b + r + 1e-9) < 0.2)
    if cirrus is not None and np.any(np.abs(wl - 1380) < 15):
        cloud = cloud | (band_at(cube, 1380) > cirrus)
    ndwi = (g - n) / (g + n + 1e-9)
    water = (ndwi > water_ndwi) & ~cloud
    shadow = (n < shadow_nir) & ~water
    sat = (cube > saturation).any("band")
    finite = np.isfinite(cube).all("band")
    valid = finite & ~cloud & ~water & ~shadow & ~sat
    return xr.Dataset({k: v.astype("uint8") for k, v in
                       dict(cloud=cloud, water=water, shadow=shadow, saturated=sat,
                            valid=valid).items()})


# ----------------------------------------------------------------------------- calibration
def empirical_line(cube: xr.DataArray, targets: Mapping[str, tuple[Any, Any]], *,
                   field_wavelengths: Sequence[float] | None = None) -> tuple[xr.DataArray,
                                                                          pd.DataFrame]:
    """Empirical line calibration to reflectance from two or more field targets.

    ``targets`` maps a name to ``(image_selection, field_spectrum)`` where the selection is a
    boolean mask of the spatial shape or ``(row_slice, col_slice)``, and the field spectrum is
    either already on the image bands or on ``field_wavelengths`` (it is then interpolated).
    Returns the calibrated cube and a per-band table of ``gain``, ``offset`` and ``r2``.
    """
    wl = wavelengths(cube)
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    img, fld = [], []
    for sel, spec in targets.values():
        if isinstance(sel, tuple):
            px = a[:, sel[0], sel[1]].reshape(a.shape[0], -1)
        else:
            px = a[:, np.asarray(getattr(sel, "values", sel), bool)]
        img.append(np.nanmean(px, axis=1))
        s = np.asarray(spec, float)
        if field_wavelengths is not None:
            s = np.interp(wl, np.asarray(field_wavelengths, float), s)
        fld.append(s)
    I, F = np.array(img), np.array(fld)                 # (targets, bands)
    if len(I) < 2:
        raise ValueError("Empirical line needs at least two targets (e.g. dark and bright)")
    Im, Fm = I.mean(0), F.mean(0)
    gain = ((I - Im) * (F - Fm)).sum(0) / np.maximum(((I - Im) ** 2).sum(0), 1e-12)
    off = Fm - gain * Im
    pred = gain * I + off
    ss = ((F - Fm) ** 2).sum(0)
    r2 = np.where(ss > 0, 1 - ((F - pred) ** 2).sum(0) / ss, np.nan)
    out = cube.copy(data=(a * gain[:, None, None] + off[:, None, None]).astype("float32"))
    return out.rename("reflectance"), pd.DataFrame({"wavelength": wl, "gain": gain,
                                                     "offset": off, "r2": r2})


def flat_field(cube: xr.DataArray, region: Any) -> xr.DataArray:
    """Flat-field relative reflectance: divide by the mean spectrum of a bright, spectrally
    flat ``region`` (boolean mask or ``(row_slice, col_slice)``)."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    px = a[:, region[0], region[1]].reshape(a.shape[0], -1) if isinstance(region, tuple) \
        else a[:, np.asarray(getattr(region, "values", region), bool)]
    ref = np.nanmean(px, axis=1)
    return cube.copy(data=(a / np.where(ref > 0, ref, np.nan)[:, None, None]).astype("float32"))


def iar_reflectance(cube: xr.DataArray) -> xr.DataArray:
    """Internal Average Relative reflectance: divide every pixel by the scene mean spectrum."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    ref = np.nanmean(a.reshape(a.shape[0], -1), axis=1)
    return cube.copy(data=(a / np.where(ref > 0, ref, np.nan)[:, None, None]).astype("float32"))


# ----------------------------------------------------------------------------- spectral filters
def smooth(cube: xr.DataArray, *, method: Literal["savgol", "moving", "gaussian"] = "savgol",
           window: int = 7, polyorder: int = 2, sigma: float = 1.0) -> xr.DataArray:
    """Smooth spectra along wavelength (Savitzky–Golay, moving average or Gaussian)."""
    from scipy import ndimage as snd
    from scipy.signal import savgol_filter

    if method == "savgol":
        def f(X):
            return savgol_filter(X, window, polyorder, axis=1)
    elif method == "moving":
        def f(X):
            return snd.uniform_filter1d(X, window, axis=1, mode="nearest")
    else:
        def f(X):
            return snd.gaussian_filter1d(X, sigma, axis=1, mode="nearest")
    return apply_spectral(cube, f)


def derivative(cube: xr.DataArray, order: int = 1, *, window: int = 7,
               polyorder: int = 2) -> xr.DataArray:
    """First or second spectral derivative (per nm), Savitzky–Golay smoothed."""
    from scipy.signal import savgol_filter

    wl = wavelengths(cube)
    step = float(np.median(np.diff(wl)))

    def f(X):
        d = savgol_filter(X, window, max(polyorder, order), deriv=order, delta=step, axis=1)
        return d

    return apply_spectral(cube, f, name=f"d{order}")


def snv(cube: xr.DataArray) -> xr.DataArray:
    """Standard normal variate: centre and scale each spectrum (removes brightness/scatter)."""
    return apply_spectral(cube, lambda X: (X - X.mean(1, keepdims=True))
                          / np.maximum(X.std(1, keepdims=True), 1e-12))


def normalize(cube: xr.DataArray, method: Literal["vector", "brightness", "max", "area"]
              = "vector") -> xr.DataArray:
    """Per-spectrum normalisation: unit L2 norm (``vector``), divide by mean (``brightness``),
    by maximum (``max``) or by the area under the curve (``area``)."""
    fn = {"vector": lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12),
          "brightness": lambda X: X / np.maximum(np.abs(X.mean(1, keepdims=True)), 1e-12),
          "max": lambda X: X / np.maximum(X.max(1, keepdims=True), 1e-12),
          "area": lambda X: X / np.maximum(np.abs(X).sum(1, keepdims=True), 1e-12)}[method]
    return apply_spectral(cube, fn)


def _upper_hull(X: np.ndarray, wl: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """Upper convex hull of many spectra at once (vectorised quickhull).

    Each pass adds, in every hull segment of every spectrum, the point lying farthest above
    the segment; spectra whose hull is complete drop out of the active set.
    """
    n, B = X.shape
    vert = np.zeros((n, B), bool)
    vert[:, 0] = vert[:, -1] = True
    idx = np.arange(B)
    hull = np.empty_like(X)
    active = np.arange(n)
    scale = np.maximum(np.abs(X).max(1), 1) * 1e-10
    for _ in range(max_iter or B):
        if active.size == 0:
            break
        Xa, Va = X[active], vert[active]
        rows = np.arange(len(active))[:, None]
        prev = np.maximum.accumulate(np.where(Va, idx, 0), axis=1)
        nxt = np.minimum.accumulate(np.where(Va, idx, B - 1)[:, ::-1], axis=1)[:, ::-1]
        xp, xn = wl[prev], wl[nxt]
        yp, yn = Xa[rows, prev], Xa[rows, nxt]
        t = np.where(xn > xp, (wl[None, :] - xp) / np.where(xn > xp, xn - xp, 1), 0.0)
        h = yp + t * (yn - yp)
        hull[active] = h
        exc = np.where(Va, -np.inf, Xa - h).ravel()
        starts = np.flatnonzero(Va.ravel())               # every segment starts at a vertex
        segmax = np.maximum.reduceat(exc, starts)
        seg = np.cumsum(Va.ravel()) - 1
        add = ((exc > np.repeat(scale[active], B)) & (exc >= segmax[seg])).reshape(Xa.shape)
        if not add.any():
            break
        vert[active] |= add
        active = active[add.any(1)]
    return hull


def continuum_removal(cube: xr.DataArray, *, start: float | None = None,
                      stop: float | None = None, mode: Literal["ratio", "difference"] = "ratio",
                      return_hull: bool = False):
    """Convex-hull continuum removal (Clark & Roush 1984), optionally restricted to
    ``start``–``stop`` nm. ``ratio`` gives values in (0, 1] with absorption features < 1."""
    from .cube import sel_range

    c = sel_range(cube, start or -np.inf, stop or np.inf) if (start or stop) else cube
    wl = wavelengths(c)

    def f(X):
        h = _upper_hull(X, wl)
        return X / np.where(h != 0, h, np.nan) if mode == "ratio" else X - h

    out = apply_spectral(c, f, name="continuum_removed", chunk=20_000)
    if return_hull:
        hull = apply_spectral(c, lambda X: _upper_hull(X, wl), name="continuum", chunk=20_000)
        return out, hull
    return out


def band_depth(cube: xr.DataArray, center: float, left: float, right: float, *,
               feature: bool = True) -> xr.Dataset:
    """Absorption feature analysis on a local continuum between ``left`` and ``right`` nm.

    Returns ``depth`` (1 − R/Rc at the minimum), ``position`` (nm of the minimum),
    ``width`` (FWHM of the feature, nm), ``area`` and ``asymmetry``. ``center`` is only used
    when ``feature=False`` (depth at that fixed wavelength).
    """
    from .cube import sel_range

    c = sel_range(cube, left, right)
    wl = wavelengths(c)
    a = np.asarray(to_cpu(c).values, dtype="float64")
    B = a.shape[0]
    X = a.reshape(B, -1).T
    cont = X[:, :1] + (X[:, -1:] - X[:, :1]) * (wl - wl[0]) / (wl[-1] - wl[0])
    cr = X / np.where(cont != 0, cont, np.nan)
    if feature:
        j = np.nanargmin(np.where(np.isfinite(cr), cr, np.inf), axis=1)
    else:
        j = np.full(len(X), int(np.argmin(np.abs(wl - center))))
    rows = np.arange(len(X))
    depth = 1 - cr[rows, j]
    half = 1 - depth / 2
    below = cr <= half[:, None]
    step = np.gradient(wl)
    width = (below * step).sum(1)
    area = ((1 - cr) * step).sum(1)
    left_area = np.array([((1 - cr[i, :k + 1]) * step[:k + 1]).sum() for i, k in enumerate(j)])
    asym = np.where(area > 0, left_area / area, np.nan)
    shp = a.shape[1:]
    tpl = c.isel(band=0, drop=True)
    tpl = tpl.drop_vars([v for v in ("wavelength", "fwhm", "good") if v in tpl.coords])

    def mk(v):
        return tpl.copy(data=np.asarray(v, "float32").reshape(shp))

    return xr.Dataset({"depth": mk(depth), "position": mk(wl[j]), "width": mk(width),
                       "area": mk(area), "asymmetry": mk(asym)})


# ----------------------------------------------------------------------------- resampling
SENSOR_SRF: dict[str, dict[str, tuple[float, float]]] = {
    # band: (centre nm, FWHM nm); Gaussian approximations of the published SRFs
    "sentinel2": {"coastal": (443, 21), "blue": (492, 66), "green": (560, 36), "red": (665, 31),
                  "rededge1": (704, 15), "rededge2": (740, 15), "rededge3": (783, 20),
                  "nir": (833, 106), "nir08": (865, 21), "watervapor": (945, 20),
                  "swir16": (1614, 91), "swir22": (2202, 175)},
    "landsat89": {"coastal": (443, 16), "blue": (482, 60), "green": (561, 57), "red": (655, 37),
                  "nir08": (865, 28), "swir16": (1609, 85), "swir22": (2201, 187)},
    "modis": {"red": (645, 50), "nir": (858, 35), "blue": (469, 20), "green": (555, 20),
              "nir12": (1240, 20), "swir16": (1640, 24), "swir22": (2130, 50)},
    "planetscope": {"coastal": (443, 20), "blue": (490, 50), "greeni": (531, 36),
                    "green": (565, 36), "yellow": (610, 20), "red": (665, 31),
                    "rededge": (705, 15), "nir": (865, 40)},
}


def _gaussian_matrix(src_wl, src_fwhm, dst_wl, dst_fwhm):
    sig = np.asarray(dst_fwhm, float)[:, None] / 2.3548
    W = np.exp(-0.5 * ((src_wl[None, :] - np.asarray(dst_wl, float)[:, None]) / sig) ** 2)
    if src_fwhm is not None:
        W = W * np.gradient(src_wl)[None, :]
    W[W < 1e-4] = 0
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.where(s > 0, s, 1), 0)


def resample_spectral(cube: xr.DataArray, target_wavelengths: Sequence[float],
                      target_fwhm: Sequence[float] | float | None = None) -> xr.DataArray:
    """Convolve to new bands with Gaussian spectral response functions (FWHM), or linear
    interpolation when ``target_fwhm`` is None. Bands outside the source range are NaN."""
    wl = wavelengths(cube)
    tw = np.asarray(target_wavelengths, float)
    if target_fwhm is None:
        def f(X):
            return np.stack([np.interp(tw, wl, x, left=np.nan, right=np.nan) for x in X])
        out = apply_spectral(cube, f, wavelengths_out=tw)
    else:
        tf = np.broadcast_to(np.asarray(target_fwhm, float), tw.shape)
        src_f = np.asarray(cube["fwhm"].values) if "fwhm" in cube.coords else None
        W = _gaussian_matrix(wl, src_f, tw, tf)
        outside = (tw < wl.min()) | (tw > wl.max())
        W[outside] = np.nan
        out = apply_spectral(cube, lambda X: X @ W.T, wavelengths_out=tw)
        out = out.assign_coords(fwhm=("band", tf))
    return out


def resample_to_sensor(cube: xr.DataArray, sensor: str = "sentinel2", *,
                       bands: Sequence[str] | None = None) -> xr.Dataset:
    """Simulate a multispectral sensor (``sentinel2``, ``landsat89``, ``modis``,
    ``planetscope``) as a Dataset with the canonical band names used by
    :mod:`geolattice.rsense.optical`, so ``ds.rs.index("NDVI")`` and the spectral-index module work."""
    srf = SENSOR_SRF[sensor]
    names = [b for b in (bands or srf) if b in srf]
    wl = wavelengths(cube)
    names = [b for b in names if wl.min() - 5 <= srf[b][0] <= wl.max() + 5]
    r = resample_spectral(cube, [srf[b][0] for b in names], [srf[b][1] for b in names])
    ds = xr.Dataset({b: r.isel(band=i, drop=True).drop_vars(
        [c for c in ("wavelength", "fwhm") if c in r.coords], errors="ignore")
        for i, b in enumerate(names)})
    if cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
    return ds.assign_attrs(sensor=sensor)


def resample_library(spectra: np.ndarray, src_wavelengths: Sequence[float], cube: xr.DataArray
                     ) -> np.ndarray:
    """Resample library spectra ``(n, src_bands)`` to the cube's bands (Gaussian when the cube
    has FWHM, else interpolation)."""
    src = np.asarray(src_wavelengths, float)
    X = np.atleast_2d(np.asarray(spectra, float))
    wl = wavelengths(cube)
    if "fwhm" in cube.coords:
        W = _gaussian_matrix(src, None, wl, np.asarray(cube["fwhm"].values, float))
        W = W * (np.gradient(src)[None, :])
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        good = np.isfinite(X)
        Xf = np.where(good, X, 0)
        num = Xf @ W.T
        den = good.astype(float) @ W.T
        return np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    return np.stack([np.interp(wl, src, x, left=np.nan, right=np.nan) for x in X])
