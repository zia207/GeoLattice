"""Spatiotemporal fusion: STARFM (Gao et al. 2006) for blending coarse-frequent and fine-sparse data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..backend import get_xp, is_gpu_array, to_cpu
from ..core._utils import to_xp


def starfm(fine_t1: xr.DataArray, coarse_t1: xr.DataArray, coarse_t2: xr.DataArray, *,
           window: int = 31, n_classes: int = 4, uncertainty: float = 0.002,
           distance_weight: float = 1.0) -> xr.DataArray:
    """Predict a fine-resolution image at t2 from a fine/coarse pair at t1 and coarse at t2.

    Coarse images must already be resampled to the fine grid (e.g. MODIS→Landsat with
    ``reproject_match``). For each pixel, spectrally similar neighbours within ``window``
    (similarity threshold ``2σ/n_classes``) contribute
    ``F1 + (C2 - C1)`` weighted by the inverse of spectral difference, temporal difference and
    relative distance. Vectorized over window offsets on the active backend.
    """
    xp = get_xp()
    F1, C1, C2 = (to_xp(a, xp).astype(xp.float32) for a in (fine_t1, coarse_t1, coarse_t2))
    H, W = F1.shape
    r = window // 2
    sim_thr = 2 * float(xp.nanstd(F1)) / n_classes
    F1p, C1p, C2p = (xp.pad(a, r, mode="reflect") for a in (F1, C1, C2))
    num = xp.zeros((H, W), xp.float32)
    den = xp.zeros((H, W), xp.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sl = (slice(r + dy, r + dy + H), slice(r + dx, r + dx + W))
            f1, c1, c2 = F1p[sl], C1p[sl], C2p[sl]
            similar = xp.abs(f1 - F1) <= sim_thr
            S = xp.abs(f1 - c1) + uncertainty
            T = xp.abs(c2 - c1) + uncertainty
            D = 1 + distance_weight * np.hypot(dy, dx) / r
            w = xp.where(similar & xp.isfinite(f1) & xp.isfinite(c1) & xp.isfinite(c2),
                         1.0 / (S * T * D), 0.0)
            num += w * (f1 + c2 - c1)
            den += w
    pred = xp.where(den > 0, num / xp.where(den > 0, den, 1), F1 + C2 - C1)
    if not is_gpu_array(fine_t1.data):
        pred = to_cpu(pred)
    return fine_t1.copy(data=pred).rename(f"{fine_t1.name or 'band'}_starfm")
