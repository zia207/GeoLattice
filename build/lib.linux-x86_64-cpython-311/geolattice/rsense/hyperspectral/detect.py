"""Target and anomaly detection: RX (global and local), matched filter, ACE, CEM, OSP, and a
column-wise methane/CO₂ matched filter with plume masking and IME emission rates.

Background statistics are estimated on all valid pixels or on a ``background`` mask; add
``shrinkage`` for numerical stability when there are few pixels per band.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .cube import from_matrix, to_matrix
from .library import as_library


def _stats(X: np.ndarray, shrinkage: float = 1e-6, *, correlation: bool = False):
    mu = np.zeros(X.shape[1]) if correlation else X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / max(len(X) - (0 if correlation else 1), 1)
    C = C + np.eye(len(C)) * shrinkage * np.trace(C) / len(C)
    return mu, np.linalg.inv(C)


def _background(cube, background):
    X, valid = to_matrix(cube)
    if background is None:
        return X, valid, X
    bm = np.asarray(getattr(background, "values", background), bool).ravel()[valid]
    return X, valid, X[bm]


def _target(target: Any, cube: xr.DataArray) -> np.ndarray:
    if isinstance(target, str):
        raise ValueError("Pass a spectrum, or library.to_dict()[name]")
    lib = as_library(np.atleast_2d(np.asarray(target, float)), cube) \
        if not isinstance(target, dict) else as_library(target, cube)
    return lib.spectra[0]


def _out(v, cube, valid, name):
    return from_matrix(v, cube, valid).isel(component=0, drop=True).rename(name)


def rx(cube: xr.DataArray, *, mode: Literal["global", "local"] = "global",
       window: int = 15, guard: int = 3, n_components: int | None = 10,
       background: Any = None, shrinkage: float = 1e-6) -> xr.DataArray:
    """Reed–Xiaoli anomaly detector (Mahalanobis distance to the background).

    ``local`` uses a dual window (``window`` outer, ``guard`` inner, pixels) on the first
    ``n_components`` principal components so it stays fast; ``global`` uses scene statistics.
    """
    if mode == "global":
        X, valid, Bg = _background(cube, background)
        mu, Ci = _stats(Bg, shrinkage)
        D = X - mu
        return _out(np.einsum("ij,jk,ik->i", D, Ci, D), cube, valid, "rx")
    from scipy import ndimage as snd

    from .reduce import pca

    Z = pca(cube, n_components or 10)[0] if n_components else cube
    a = np.nan_to_num(np.asarray(Z.values, dtype="float64"))
    p = a.shape[0]
    no, ni = window**2, guard**2

    def ring_mean(img):
        so = snd.uniform_filter(img, window, mode="reflect") * no
        si = snd.uniform_filter(img, guard, mode="reflect") * ni
        return (so - si) / (no - ni)

    m = np.stack([ring_mean(a[i]) for i in range(p)])
    S = np.empty((p, p) + a.shape[1:])
    for i in range(p):
        for j in range(i, p):
            S[i, j] = S[j, i] = ring_mean(a[i] * a[j]) - m[i] * m[j]
    S = np.moveaxis(S, (0, 1), (-2, -1))
    S = S + np.eye(p) * shrinkage * np.trace(S, axis1=-2, axis2=-1)[..., None, None] / p + \
        np.eye(p) * 1e-12
    d = np.moveaxis(a - m, 0, -1)
    sol = np.linalg.solve(S, d[..., None])[..., 0]
    out = np.einsum("...i,...i->...", d, sol)
    out[~np.isfinite(np.asarray(cube.values)).all(0)] = np.nan
    tpl = cube.isel(band=0, drop=True)
    tpl = tpl.drop_vars([v for v in ("wavelength", "fwhm", "good") if v in tpl.coords])
    return tpl.copy(data=out.astype("float32")).rename("rx_local")


def matched_filter(cube: xr.DataArray, target: Any, *, background: Any = None,
                   shrinkage: float = 1e-6) -> xr.DataArray:
    """Clutter matched filter, normalised so a pure target pixel scores ≈ 1 and background ≈ 0
    (the score approximates the target abundance)."""
    X, valid, Bg = _background(cube, background)
    mu, Ci = _stats(Bg, shrinkage)
    t = _target(target, cube) - mu
    w = Ci @ t / (t @ Ci @ t)
    return _out((X - mu) @ w, cube, valid, "mf")


def ace(cube: xr.DataArray, target: Any, *, background: Any = None,
        shrinkage: float = 1e-6) -> xr.DataArray:
    """Adaptive Cosine/Coherence Estimator (Kraut & Scharf 1999): squared cosine between the
    whitened pixel and whitened target, 0–1, invariant to pixel scaling."""
    X, valid, Bg = _background(cube, background)
    mu, Ci = _stats(Bg, shrinkage)
    t = _target(target, cube) - mu
    D = X - mu
    num = (D @ Ci @ t) ** 2
    den = (t @ Ci @ t) * np.einsum("ij,jk,ik->i", D, Ci, D)
    return _out(num / np.maximum(den, 1e-30), cube, valid, "ace")


def cem(cube: xr.DataArray, target: Any, *, shrinkage: float = 1e-6) -> xr.DataArray:
    """Constrained Energy Minimisation (Harsanyi 1993): FIR filter with unit response to the
    target and minimum output energy, using the sample correlation matrix."""
    X, valid = to_matrix(cube)
    _, Ri = _stats(X, shrinkage, correlation=True)
    d = _target(target, cube)
    w = Ri @ d / (d @ Ri @ d)
    return _out(X @ w, cube, valid, "cem")


def osp(cube: xr.DataArray, target: Any, undesired: Any) -> xr.DataArray:
    """Orthogonal Subspace Projection (Harsanyi & Chang 1994): project out ``undesired``
    signatures (k × bands) and match the target."""
    X, valid = to_matrix(cube)
    U = as_library(undesired, cube).spectra.T
    P = np.eye(U.shape[0]) - U @ np.linalg.pinv(U)
    d = _target(target, cube)
    w = P @ d / (d @ P @ d)
    return _out(X @ w, cube, valid, "osp")


def detection_roc(score: xr.DataArray, truth: Any, *, n: int = 200) -> pd.DataFrame:
    """ROC table (threshold, true positive rate, false alarm rate) and AUC in ``attrs``."""
    s = np.asarray(score.values, float).ravel()
    t = np.asarray(getattr(truth, "values", truth), bool).ravel()
    ok = np.isfinite(s)
    s, t = s[ok], t[ok]
    th = np.unique(np.quantile(s, np.linspace(0, 1, n)))[::-1]
    tpr = np.array([(s[t] >= x).mean() if t.any() else np.nan for x in th])
    far = np.array([(s[~t] >= x).mean() if (~t).any() else np.nan for x in th])
    df = pd.DataFrame({"threshold": th, "tpr": tpr, "far": far})
    x, y = np.r_[0, far, 1], np.r_[0, tpr, 1]
    df.attrs["auc"] = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))
    return df


# ----------------------------------------------------------------------------- gas plumes
def gas_matched_filter(radiance: xr.DataArray, unit_absorption: Any, *,
                       window: tuple[float, float] = (2100.0, 2500.0), columnwise: bool = True,
                       column_dim: str | None = None, shrinkage: float = 1e-4,
                       background: Any = None, min_pixels: int = 50) -> xr.DataArray:
    """Column-wise matched filter for CH₄ or CO₂ enhancement (Thompson et al. 2016; EMIT).

    ``unit_absorption`` is the gas unit-absorption spectrum ``k`` on the cube's bands (per
    ppm·m; the change in log radiance per unit enhancement, negative inside absorption
    bands). The target is ``t = μ ⊙ k``; the estimate is ``α = (x − μ)ᵀΣ⁻¹t / tᵀΣ⁻¹t`` in
    ppm·m. With ``columnwise`` the background mean/covariance are computed per detector
    column (cross-track), as for pushbroom radiance before orthorectification.
    """
    from .cube import wavelengths

    k_full = np.asarray(unit_absorption, float)
    wl_all = wavelengths(radiance)
    keep = (wl_all >= window[0]) & (wl_all <= window[1])
    if "good" in radiance.coords:
        keep &= np.asarray(radiance["good"].values, bool)
    c = radiance.isel(band=np.flatnonzero(keep))
    k = k_full[keep]
    a = np.asarray(c.values, dtype="float64")
    B, H, W = a.shape
    out = np.full((H, W), np.nan)
    bm = None if background is None else np.asarray(getattr(background, "values", background), bool)
    cols = [slice(j, j + 1) for j in range(W)] if columnwise else [slice(0, W)]
    for cs in cols:
        X = a[:, :, cs].reshape(B, -1).T
        ok = np.all(np.isfinite(X), axis=1)
        sel = ok if bm is None else ok & bm[:, cs].ravel()
        if sel.sum() < min_pixels:
            continue
        mu = X[sel].mean(0)
        Cov = np.cov(X[sel] - mu, rowvar=False)
        Cov += np.eye(B) * shrinkage * np.trace(Cov) / B
        Ci = np.linalg.inv(Cov)
        t = mu * k
        res = np.full(len(X), np.nan)
        res[ok] = (X[ok] - mu) @ Ci @ t / (t @ Ci @ t)
        out[:, cs] = res.reshape(H, -1)
    tpl = radiance.isel(band=0, drop=True)
    tpl = tpl.drop_vars([v for v in ("wavelength", "fwhm", "good") if v in tpl.coords])
    return tpl.copy(data=out.astype("float32")).rename("enhancement").assign_attrs(
        units="ppm m")


def plume_mask(enhancement: xr.DataArray, *, threshold: float | None = None,
               sigma: float = 3.0, min_pixels: int = 10, dilate: int = 1) -> tuple[xr.DataArray,
                                                                                  pd.DataFrame]:
    """Segment plumes: pixels above ``threshold`` ppm·m (default: robust median + ``sigma``
    × MAD), connected (8-neighbour), at least ``min_pixels``. Returns labels and a table with
    ``n_pixels``, ``max``, ``mean`` and the peak location."""
    from scipy import ndimage as snd

    e = np.asarray(enhancement.values, dtype="float64")
    fin = e[np.isfinite(e)]
    if threshold is None:
        med = np.median(fin)
        mad = 1.4826 * np.median(np.abs(fin - med))
        threshold = med + sigma * mad
    m = np.isfinite(e) & (e > threshold)
    lab, n = snd.label(m, structure=np.ones((3, 3)))
    sizes = snd.sum(m, lab, np.arange(1, n + 1)) if n else np.array([])
    keep = np.flatnonzero(sizes >= min_pixels) + 1
    lab = np.where(np.isin(lab, keep), lab, 0)
    if dilate:
        grown = snd.grey_dilation(lab, size=(2 * dilate + 1, 2 * dilate + 1))
        lab = np.where((lab == 0) & np.isfinite(e) & (e > threshold / 2), grown, lab)
    _, inv = np.unique(lab, return_inverse=True)
    lab = inv.reshape(lab.shape)
    rows = []
    for i in range(1, lab.max() + 1):
        sel = lab == i
        v = e[sel]
        r, c = np.unravel_index(np.argmax(np.where(sel, e, -np.inf)), e.shape)
        row = {"plume": i, "n_pixels": int(sel.sum()), "max": float(np.nanmax(v)),
               "mean": float(np.nanmean(v)), "row": int(r), "col": int(c)}
        dims = enhancement.dims
        if all(d in enhancement.coords for d in dims):
            row["y"] = float(enhancement[dims[0]].values[r])
            row["x"] = float(enhancement[dims[1]].values[c])
        rows.append(row)
    table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["plume", "n_pixels", "max", "mean", "row", "col"])
    table.attrs["threshold"] = float(threshold)
    return enhancement.copy(data=lab.astype("int32")).rename("plume"), table


GAS_DENSITY_KG_M3 = {"CH4": 0.6785, "CO2": 1.8622}   # at 101325 Pa, 288.15 K


def plume_ime(enhancement: xr.DataArray, labels: xr.DataArray, plume: int = 1, *,
              pixel_area_m2: float | None = None, gas: str = "CH4") -> dict[str, float]:
    """Integrated Mass Enhancement of one plume (kg) from a ppm·m enhancement map:
    ``IME = Σ ΔX · 10⁻⁶ · ρ_gas · A_pixel``. Also returns the plume area and the
    Varon et al. (2018) length scale ``L = √area`` for :func:`emission_rate`."""
    e = np.asarray(enhancement.values, float)
    sel = (np.asarray(labels.values) == plume) & np.isfinite(e)
    if pixel_area_m2 is None:
        dx = abs(float(enhancement[enhancement.dims[-1]].diff(enhancement.dims[-1]).mean()))
        dy = abs(float(enhancement[enhancement.dims[-2]].diff(enhancement.dims[-2]).mean()))
        crs = enhancement.rio.crs
        if crs is not None and crs.is_geographic:
            lat = float(enhancement[enhancement.dims[-2]].mean())
            dx, dy = dx * 111_320 * np.cos(np.radians(lat)), dy * 110_540
        pixel_area_m2 = dx * dy
    ime = float(np.clip(e[sel], 0, None).sum() * 1e-6 * GAS_DENSITY_KG_M3[gas] * pixel_area_m2)
    area = float(sel.sum() * pixel_area_m2)
    return {"ime_kg": ime, "area_m2": area, "length_m": float(np.sqrt(area)),
            "pixel_area_m2": float(pixel_area_m2), "n_pixels": int(sel.sum())}


def emission_rate(ime_kg: float, wind_speed_m_s: float, plume_length_m: float, *,
                  per: Literal["s", "h"] = "h") -> float:
    """Point-source rate ``Q = IME · U_eff / L`` (Varon et al. 2018), in kg/h (or kg/s).
    Same formula as ``geolattice.rsense.atmosphere.sentinel_5p.estimate_emission_rate``."""
    q = float(ime_kg) * float(wind_speed_m_s) / float(plume_length_m)
    return q * 3600.0 if per == "h" else q
