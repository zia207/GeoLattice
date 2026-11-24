"""Polarimetric features for dual-pol (VV/VH or HH/HV) and quad-pol (HH/HV/VH/VV) SAR."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ..core._utils import nan_box_mean, run_array
from ..core._utils import where as _where


def covariance_c2(co: Any, cross: Any, size: int = 7) -> dict[str, Any]:
    """Locally averaged 2×2 covariance matrix from complex SLC channels.

    Returns ``{"c11", "c22", "c12_re", "c12_im"}`` (``c11 = <|co|^2>``, ``c12 = <co cross*>``).
    """
    def avg(a):
        return run_array(a, lambda x, xp: nan_box_mean(x, size, xp))

    c12 = co * np.conj(cross)
    return {"c11": avg(np.abs(co) ** 2), "c22": avg(np.abs(cross) ** 2),
            "c12_re": avg(np.real(c12)), "c12_im": avg(np.imag(c12))}


def h_alpha_dual(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """Dual-pol H/A/α decomposition (Cloude 2007) from C2 elements.

    Eigenvalues of the Hermitian 2×2 matrix are found analytically. Entropy H uses
    log base 2 (0-1), α is in degrees (0-90), anisotropy A = (λ1-λ2)/(λ1+λ2).
    Also returns DpRVI (Mandal et al. 2020), a dual-pol vegetation index.
    """
    tr = c11 + c22
    det = c11 * c22 - (c12_re**2 + c12_im**2)
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    s = l1 + l2
    with np.errstate(divide="ignore", invalid="ignore"):
        p1, p2 = l1 / s, l2 / s
        H = -(_where(p1 > 0, p1 * np.log2(p1), 0) + _where(p2 > 0, p2 * np.log2(p2), 0))
        mag12 = np.sqrt(c12_re**2 + c12_im**2)

        def alpha_i(lam):
            return np.degrees(np.arccos(np.clip(mag12 / np.sqrt(mag12**2 + (lam - c11) ** 2), 0, 1)))

        alpha = p1 * alpha_i(l1) + p2 * alpha_i(l2)
        A = (l1 - l2) / s
        m = np.sqrt(np.maximum(1 - 4 * det / tr**2, 0))
        dprvi = 1 - m * p1
    out = {"entropy": H, "alpha": alpha, "anisotropy": A, "lambda1": l1, "lambda2": l2,
           "dprvi": dprvi}
    if isinstance(c11, xr.DataArray):
        return xr.Dataset(out)
    return out


def dual_pol_features(co: Any, cross: Any) -> xr.Dataset | dict:
    """Intensity-only dual-pol features from GRD/RTC (linear power).

    ``ratio`` cross/co, ``ratio_db``, ``rvi`` = 4·cross/(co+cross) (dual-pol RVI),
    ``nrpb`` = (cross-co)/(cross+co), ``span`` = co+cross, and dB versions of both channels.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "ratio": cross / co, "ratio_db": 10 * np.log10(cross / co),
            "rvi": 4 * cross / (co + cross), "nrpb": (cross - co) / (cross + co),
            "span": co + cross, "co_db": 10 * np.log10(co), "cross_db": 10 * np.log10(cross),
        }
    if isinstance(co, xr.DataArray):
        return xr.Dataset(out)
    return out


def stokes_parameters(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """Stokes-vector parameters from a dual-pol C2 covariance matrix.

    Treats ``{co, cross}`` as the two components of a hybrid-pol-style Stokes
    vector, by direct analogy with the compact-pol architectures the m-χ/m-δ
    framework was built for (Raney 2007): ``s0`` = total power, ``s1`` = co/cross
    power difference, ``s2``/``s3`` = twice the real/imaginary part of the
    co-cross correlation. ``degree_of_polarization`` (m) is 0 for fully
    depolarized (independent channels) and 1 for a single, pure scattering
    mechanism; for a Hermitian 2×2 covariance matrix it equals the Cloude
    anisotropy that ``h_alpha_dual`` returns.
    """
    s0 = c11 + c22
    s1 = c11 - c22
    s2 = 2 * c12_re
    s3 = 2 * c12_im
    with np.errstate(divide="ignore", invalid="ignore"):
        dop = np.clip(np.sqrt(s1**2 + s2**2 + s3**2) / s0, 0, 1)
    out = {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "degree_of_polarization": dop}
    if isinstance(c11, xr.DataArray):
        return xr.Dataset(out)
    return out


def m_delta_decomposition(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """m-δ model-based power decomposition (Raney 2007) of a dual-pol C2 matrix.

    δ is the co/cross relative phase (``atan2(s2, s1)``, degrees). Surface-like
    power ``ps`` grows with ``cos δ`` near 0° and double-bounce-like power ``pd``
    with ``cos δ`` near 180°; the depolarized remainder ``pv`` (volume/diffuse
    scattering) is ``(1-m)·s0``. By construction ``ps + pd + pv == s0`` (span).
    """
    st = stokes_parameters(c11, c22, c12_re, c12_im)
    s0, s1, s2, m = st["s0"], st["s1"], st["s2"], st["degree_of_polarization"]
    delta = np.arctan2(s2, s1)
    ps = s0 * m * (1 + np.cos(delta)) / 2
    pd = s0 * m * (1 - np.cos(delta)) / 2
    pv = s0 * (1 - m)
    out = {"delta": np.degrees(delta), "ps": ps, "pd": pd, "pv": pv}
    if isinstance(c11, xr.DataArray):
        return xr.Dataset(out)
    return out


def m_chi_decomposition(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """m-χ model-based power decomposition (Raney 2007) of a dual-pol C2 matrix.

    χ is the ellipticity-style angle ``0.5·asin(s3 / (m·s0))`` (degrees). Surface-like
    power ``ps`` is largest at χ near -45° and double-bounce-like power ``pd`` at χ
    near +45°; the depolarized remainder ``pv`` is ``(1-m)·s0``. By construction
    ``ps + pd + pv == s0`` (span).
    """
    st = stokes_parameters(c11, c22, c12_re, c12_im)
    s0, s3, m = st["s0"], st["s3"], st["degree_of_polarization"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s3 / (m * s0)
        chi = 0.5 * np.arcsin(np.clip(_where(np.isfinite(ratio), ratio, 0), -1, 1))
    ps = s0 * m * (1 - np.sin(2 * chi)) / 2
    pd = s0 * m * (1 + np.sin(2 * chi)) / 2
    pv = s0 * (1 - m)
    out = {"chi": np.degrees(chi), "ps": ps, "pd": pd, "pv": pv}
    if isinstance(c11, xr.DataArray):
        return xr.Dataset(out)
    return out


def dual_pol_model_decomposition(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """Eigen-decomposition model-based power decomposition for dual-pol data.

    Generalizes model-based decomposition (the Freeman-Durden/van Zyl idea of
    splitting span into surface, double-bounce and volume power) to the 2×2
    dual-pol covariance matrix, following the degree-of-polarization / mean-α
    framework used by dual-pol model-based decompositions such as Mascolo et al.
    (2022): the depolarized part of the span, ``(1-m)·span``, is attributed to
    volume/diffuse scattering (``pv``); the polarized remainder, ``m·span``, is
    split between surface-like (``ps``) and double-bounce-like (``pd``) power by
    the dominant eigenvector's mean α angle from ``h_alpha_dual`` (α < 45°: surface
    scattering dominates; α ≥ 45°: double-bounce/dipole scattering dominates).
    ``ps + pd + pv == c11 + c22`` (span) by construction.
    """
    ha = h_alpha_dual(c11, c22, c12_re, c12_im)
    alpha = ha["alpha"]
    span = c11 + c22
    det = c11 * c22 - (c12_re**2 + c12_im**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.sqrt(np.maximum(1 - 4 * det / span**2, 0))
    polarized = m * span
    is_surface = _where(alpha < 45, 1.0, 0.0)
    ps = polarized * is_surface
    pd = polarized * (1 - is_surface)
    pv = span * (1 - m)
    out = {"alpha": alpha, "ps": ps, "pd": pd, "pv": pv}
    if isinstance(c11, xr.DataArray):
        return xr.Dataset(out)
    return out


def coherency_t3(s_hh: Any, s_hv: Any, s_vh: Any, s_vv: Any, size: int = 7) -> dict[str, Any]:
    """Locally averaged 3x3 Pauli-basis coherency matrix T3 from quad-pol SLC channels.

    The Pauli scattering vector is ``k = [Shh+Svv, Shh-Svv, sqrt(2) Sx] / sqrt(2)``, where
    ``Sx = (Shv+Svh)/2`` -- the cross-pol channels averaged first, assuming reciprocity (the
    usual monostatic-SAR assumption). Real data lets this be checked directly: ``|Shv-Svh|``
    should be small next to ``|Shv|`` wherever the assumption holds.

    Returns ``{"t11", "t22", "t33"}`` (real diagonal: odd-bounce-, double-bounce- and
    cross-pol-like power) and ``{"t12_re", "t12_im", "t13_re", "t13_im", "t23_re", "t23_im"}``
    (the three independent complex off-diagonal terms of the Hermitian matrix). Feeds directly
    into :func:`h_alpha_quad`.
    """
    def avg(a):
        return run_array(a, lambda x, xp: nan_box_mean(x, size, xp))

    s_x = (s_hv + s_vh) / 2
    k1 = (s_hh + s_vv) / np.sqrt(2)
    k2 = (s_hh - s_vv) / np.sqrt(2)
    k3 = np.sqrt(2) * s_x
    ks = (k1, k2, k3)
    out: dict[str, Any] = {}
    for i in range(3):
        out[f"t{i + 1}{i + 1}"] = avg(np.abs(ks[i]) ** 2)
    for i, j in ((1, 2), (1, 3), (2, 3)):
        c = ks[i - 1] * np.conj(ks[j - 1])
        out[f"t{i}{j}_re"] = avg(np.real(c))
        out[f"t{i}{j}_im"] = avg(np.imag(c))
    return out


def h_alpha_quad(t11: Any, t22: Any, t33: Any, t12_re: Any, t12_im: Any, t13_re: Any,
                 t13_im: Any, t23_re: Any, t23_im: Any) -> xr.Dataset | dict:
    """Quad-pol Cloude-Pottier H/A/alpha decomposition (Cloude & Pottier 1997) from T3 elements
    (:func:`coherency_t3`).

    Builds the Hermitian 3x3 coherency matrix per pixel and solves it with
    ``numpy.linalg.eigh`` (vectorised over every pixel at once). ``entropy`` =
    ``-sum_i p_i log_3 p_i`` (log base 3, so entropy is 0-1 for three mechanisms, exactly 1 for
    three equal eigenvalues); ``anisotropy`` = ``(lambda2-lambda3)/(lambda2+lambda3)`` (the two
    smaller eigenvalues -- the usual secondary-mechanism contrast, informative mainly once
    entropy is already high); ``alpha`` is the power-weighted mean of each eigenvector's angle
    from the first (odd-bounce) Pauli axis, ``alpha_i = arccos(|e_i[0]|)`` -- 0 degrees pure
    surface, 90 degrees pure double-bounce *or* pure cross-pol (both orthogonal to the surface
    axis for a single mechanism; entropy is what tells them apart -- the same reason the H-alpha
    plane needs both axes). Also returns the three eigenvalues ``lambda1 >= lambda2 >= lambda3``.
    """
    was_xr = isinstance(t11, xr.DataArray)

    def val(a):
        return a.values if isinstance(a, xr.DataArray) else np.asarray(a)

    t11v, t22v, t33v = val(t11), val(t22), val(t33)
    valid = np.isfinite(t11v) & np.isfinite(t22v) & np.isfinite(t33v)
    shape = t11v.shape
    T = np.zeros(shape + (3, 3), dtype="complex128")
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = t11v, t22v, t33v
    t12 = val(t12_re) + 1j * val(t12_im)
    t13 = val(t13_re) + 1j * val(t13_im)
    t23 = val(t23_re) + 1j * val(t23_im)
    T[..., 0, 1], T[..., 1, 0] = t12, np.conj(t12)
    T[..., 0, 2], T[..., 2, 0] = t13, np.conj(t13)
    T[..., 1, 2], T[..., 2, 1] = t23, np.conj(t23)
    T = np.nan_to_num(T, nan=0.0)

    lam, vec = np.linalg.eigh(T)   # ascending order
    lam, vec = lam[..., ::-1], vec[..., ::-1]   # descending: lambda1 >= lambda2 >= lambda3
    lam = np.clip(lam.real, 0, None)
    total = lam.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, lam / total, 0)
        H = -np.sum(np.where(p > 0, p * np.log(p) / np.log(3), 0), axis=-1)
        alpha_i = np.degrees(np.arccos(np.clip(np.abs(vec[..., 0, :]), 0, 1)))
        alpha = np.sum(p * alpha_i, axis=-1)
        denom = lam[..., 1] + lam[..., 2]
        A = np.where(denom > 0, (lam[..., 1] - lam[..., 2]) / np.where(denom > 0, denom, 1), 0)

    out = {"entropy": H, "alpha": alpha, "anisotropy": A,
           "lambda1": lam[..., 0], "lambda2": lam[..., 1], "lambda3": lam[..., 2]}
    out = {k: np.where(valid, v, np.nan) for k, v in out.items()}
    if was_xr:
        return xr.Dataset({k: (t11.dims, v) for k, v in out.items()}, coords=t11.coords)
    return out


H_ALPHA_ZONES = {
    "Z9 low entropy surface": ((0, 0.5), (0, 42.5)),
    "Z8 low entropy dipole": ((0, 0.5), (42.5, 47.5)),
    "Z7 low entropy multiple": ((0, 0.5), (47.5, 90)),
    "Z6 medium entropy surface": ((0.5, 0.9), (0, 40)),
    "Z5 medium entropy vegetation": ((0.5, 0.9), (40, 50)),
    "Z4 medium entropy multiple": ((0.5, 0.9), (50, 90)),
    "Z2 high entropy vegetation": ((0.9, 1.0), (0, 55)),
    "Z1 high entropy multiple": ((0.9, 1.0), (55, 90)),
}


def h_alpha_classify(entropy: xr.DataArray, alpha: xr.DataArray) -> xr.DataArray:
    """Unsupervised H-α plane classification into Cloude-Pottier zones (codes 1-9)."""
    codes = {"Z9": 9, "Z8": 8, "Z7": 7, "Z6": 6, "Z5": 5, "Z4": 4, "Z2": 2, "Z1": 1}
    out = xr.zeros_like(entropy, dtype="int8")
    for name, ((h0, h1), (a0, a1)) in H_ALPHA_ZONES.items():
        sel = (entropy >= h0) & (entropy <= h1) & (alpha >= a0) & (alpha < a1)
        out = xr.where(sel & (out == 0), codes[name[:2]], out)
    return out.rename("h_alpha_zone")
