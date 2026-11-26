"""Sentinel-1 dual-pol (VV/VH) backscatter indices.

Simple, widely used derived metrics computed directly from calibrated, linear-scale sigma0
VV/VH backscatter (e.g. :func:`geolattice.rsense.sar.sentinel1.read_grd_subset`'s ``"data"``, or
:func:`geolattice.rsense.sar.sentinel1.calibrate`'s output) -- no speckle filtering or multilooking
is applied here, since these are plain per-pixel formulas; apply
:func:`geolattice.rsense.sar.speckle` filters beforehand if a smoother result is wanted. Every
function accepts NumPy arrays or ``xr.DataArray`` and returns the same type. Division is
protected against a zero or negative denominator (which a true calibrated sigma0 should never
have, but a noise-subtracted one occasionally can), returning NaN there instead of raising or
silently producing inf.

See ``07_remote_sensing/07_03_radar/07_03_01_09_sentinel1_indices.ipynb`` for the formulas,
interpretation and literature behind each index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core._utils import where as _where


def _safe_div(num: Any, den: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _where(den > 0, num / den, np.nan)


def cross_pol_ratio(vv: Any, vh: Any) -> Any:
    """Cross-pol ratio ``CR = VH / VV`` (linear). Sensitive to crop structure and growth stage."""
    return _safe_div(vh, vv)


def radar_vegetation_index(vv: Any, vh: Any) -> Any:
    """Dual-pol Radar Vegetation Index ``RVI = 4*VH / (VV+VH)`` (linear inputs)."""
    return _safe_div(4 * vh, vv + vh)


def rvi4s1(vv: Any, vh: Any) -> Any:
    """Sentinel-1 RVI4S1 ``= sqrt(VV/(VV+VH)) * 4*VH/(VV+VH)`` (linear inputs).

    Not the same formula as :func:`radar_vegetation_index`: RVI4S1 adds a co-pol weighting
    term, intended specifically for Sentinel-1 dual-pol (VV+VH) acquisitions.
    """
    s = vv + vh
    with np.errstate(divide="ignore", invalid="ignore"):
        return _where(s > 0, np.sqrt(_safe_div(vv, s)) * _safe_div(4 * vh, s), np.nan)


def radar_forest_degradation_index(vv: Any, vh: Any) -> Any:
    """Radar Forest Degradation Index ``RFDI = (VV-VH) / (VV+VH)`` (linear inputs).

    Algebraically the negative of :func:`sentinel_normalized_index`: ``RFDI = -SNI``. Both are
    kept as separate functions because the remote-sensing literature cites them under different
    names for different applications (forest degradation vs. general polarization contrast),
    not because they compute different quantities.
    """
    return _safe_div(vv - vh, vv + vh)


def sentinel_normalized_index(vv: Any, vh: Any) -> Any:
    """Sentinel Normalized Index ``SNI = (VH-VV) / (VH+VV)`` (linear inputs); ``SNI = -RFDI``
    (see :func:`radar_forest_degradation_index`)."""
    return _safe_div(vh - vv, vh + vv)


def polarimetric_rvi(vv: Any, vh: Any) -> Any:
    """Polarimetric RVI ``PRVI = (1 - VV/(VH+VV)) * VH``, algebraically ``VH^2/(VH+VV)``
    (linear inputs). Computed via the first form, which is better-conditioned than squaring VH
    directly when VH is large."""
    s = vh + vv
    with np.errstate(divide="ignore", invalid="ignore"):
        return _where(s > 0, (1 - _safe_div(vv, s)) * vh, np.nan)


def dpsvi(vv: Any, vh: Any) -> Any:
    """Dual-Polarization SAR Vegetation Index ``DPSVI = (VV+VH) / VV`` (linear inputs).

    This is the simple ratio variant; other published DPSVI formulations use more complex
    polarimetric quantities and are not interchangeable with this one.
    """
    return _safe_div(vv + vh, vv)


def temporal_change(x_t: Any, x_t1: Any) -> Any:
    """Simple temporal difference ``x_t - x_t1`` of the same backscatter channel at two dates."""
    return x_t - x_t1


def temporal_ratio(x_t: Any, x_t1: Any) -> Any:
    """Simple temporal ratio ``x_t / x_t1`` of the same backscatter channel at two dates."""
    return _safe_div(x_t, x_t1)


def dual_pol_indices(vv: Any, vh: Any) -> dict[str, Any]:
    """All single-date VV/VH indices and derived metrics in one call.

    Returns a dict with ``"cross_pol_ratio"``, ``"rvi"``, ``"rvi4s1"``,
    ``"forest_degradation_index"``, ``"normalized_index"``, ``"polarimetric_rvi"``, ``"dpsvi"``,
    ``"vv_minus_vh"``, ``"vv_plus_vh"``, ``"vh_vv_ratio"`` and ``"vv_vh_ratio"`` -- every index
    and metric in this module except the temporal ones, which need a second date.
    """
    return {
        "cross_pol_ratio": cross_pol_ratio(vv, vh),
        "rvi": radar_vegetation_index(vv, vh),
        "rvi4s1": rvi4s1(vv, vh),
        "forest_degradation_index": radar_forest_degradation_index(vv, vh),
        "normalized_index": sentinel_normalized_index(vv, vh),
        "polarimetric_rvi": polarimetric_rvi(vv, vh),
        "dpsvi": dpsvi(vv, vh),
        "vv_minus_vh": vv - vh,
        "vv_plus_vh": vv + vh,
        "vh_vv_ratio": _safe_div(vh, vv),
        "vv_vh_ratio": _safe_div(vv, vh),
    }
