"""Speckle filters for SAR intensity images, GPU-native via the backend dispatch.

All filters expect *linear* intensity (sigma0/gamma0, not dB) and ignore NaN.
Inputs may be 2-D or have leading dims (time, polarization); filtering is
applied over the last two axes. ``enl`` is the equivalent number of looks
(Sentinel-1 IW GRDH ~4.4).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import ndimage
from ..core._utils import as_float, local_mean_var, nan_box_mean, run_array

S1_IW_GRDH_ENL = 4.4


def _per_image(arr: Any, xp: Any, fn) -> Any:
    """Apply a 2-D filter over leading dimensions."""
    if arr.ndim == 2:
        return fn(arr)
    flat = arr.reshape((-1,) + arr.shape[-2:])
    return xp.stack([fn(a) for a in flat]).reshape(arr.shape)


def boxcar(img: Any, size: int = 5):
    """Moving-average (multilook-style) filter."""
    return run_array(img, lambda a, xp: nan_box_mean(as_float(a, xp), size, xp))


def median(img: Any, size: int = 5):
    """Median filter (NaN treated as missing by filling with the local mean first)."""
    def f(a, xp):
        a = as_float(a, xp)
        filled = xp.where(xp.isnan(a), nan_box_mean(a, size, xp), a)
        nd = ndimage(a)
        sz = (1,) * (a.ndim - 2) + (size, size)
        return xp.where(xp.isnan(a), xp.nan, nd.median_filter(filled, size=sz, mode="reflect"))

    return run_array(img, f)


def lee(img: Any, size: int = 7, enl: float = S1_IW_GRDH_ENL):
    """Lee (1980) adaptive filter: ``x = m + b (z - m)``, ``b = var_x / var_z``."""
    cu2 = 1.0 / enl

    def f(a, xp):
        a = as_float(a, xp)
        m, v = local_mean_var(a, size, xp)
        var_x = (v - m * m * cu2) / (1 + cu2)
        b = xp.clip(var_x / xp.where(v > 0, v, xp.nan), 0, 1)
        b = xp.nan_to_num(b, nan=0.0)
        return xp.where(xp.isnan(a), xp.nan, m + b * (a - m))

    return run_array(img, f)


def gamma_map(img: Any, size: int = 7, enl: float = S1_IW_GRDH_ENL):
    """Gamma-MAP filter (Lopes et al. 1990): homogeneous areas → mean, strong scatterers kept,
    textured areas → MAP estimate under a Gamma scene model."""
    L = float(enl)
    cu = 1 / np.sqrt(L)
    cmax = np.sqrt(2) * cu

    def f(a, xp):
        a = as_float(a, xp)
        m, v = local_mean_var(a, size, xp)
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = xp.sqrt(v) / m
            alpha = (1 + cu**2) / (ci**2 - cu**2)
            B = alpha - L - 1
            D = m * m * B * B + 4 * alpha * L * m * a
            r = (B * m + xp.sqrt(xp.maximum(D, 0))) / (2 * alpha)
        out = xp.where(ci <= cu, m, xp.where(ci >= cmax, a, r))
        return xp.where(xp.isnan(a), xp.nan, out)

    return run_array(img, f)


def _refined_lee_kernels(xp: Any) -> Any:
    """Eight 7×7 directional half-window masks (Lee 1981) as an array (8, 7, 7)."""
    i, j = np.mgrid[0:7, 0:7]
    masks = [
        j <= 3,          # 0: left half (vertical edge, darker/brighter side west)
        j >= 3,          # 1: right half
        i <= 3,          # 2: top half (horizontal edge)
        i >= 3,          # 3: bottom half
        i <= j,          # 4: upper-right triangle (diagonal edge \)
        i >= j,          # 5: lower-left triangle
        i + j <= 6,      # 6: upper-left triangle (anti-diagonal edge /)
        i + j >= 6,      # 7: lower-right triangle
    ]
    k = np.stack(masks).astype("float32")
    return xp.asarray(k / k.sum(axis=(1, 2), keepdims=True))


def refined_lee(img: Any, enl: float = S1_IW_GRDH_ENL):
    """Refined Lee filter (Lee 1981) with 7×7 edge-aligned windows.

    Edge direction is found from gradients between 3×3 sub-window means; the local
    statistics are then computed only on the homogeneous side of the edge, which
    preserves boundaries much better than the standard Lee filter.
    """
    cu2 = 1.0 / enl

    def one(a, xp):
        nd = ndimage(a)
        valid = ~xp.isnan(a)
        z = xp.where(valid, a, nan_box_mean(a, 7, xp))
        z = xp.nan_to_num(z, nan=0.0)
        m3 = nd.uniform_filter(z, 3, mode="reflect")
        pad = xp.pad(m3, 2, mode="edge")
        H, W = z.shape

        def s(dy, dx):  # 3x3 mean centred at offset (dy, dx)
            return pad[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]

        grads = xp.stack([
            xp.abs(s(0, -2) - s(0, 2)),    # vertical edge
            xp.abs(s(-2, 0) - s(2, 0)),    # horizontal edge
            xp.abs(s(-2, 2) - s(2, -2)),   # diagonal "\" edge
            xp.abs(s(-2, -2) - s(2, 2)),   # anti-diagonal "/" edge
        ])
        direction = xp.argmax(grads, axis=0)
        c = s(0, 0)
        # choose side whose sub-window mean is closer to the centre
        side = xp.stack([
            xp.abs(s(0, -2) - c) > xp.abs(s(0, 2) - c),
            xp.abs(s(-2, 0) - c) > xp.abs(s(2, 0) - c),
            xp.abs(s(-2, 2) - c) > xp.abs(s(2, -2) - c),
            xp.abs(s(-2, -2) - c) > xp.abs(s(2, 2) - c),
        ])
        # direction d uses mask 2d (first half) or 2d+1 (second half); side==True means the
        # first-compared neighbour differs more from the centre, so use the opposite half
        sel = 2 * direction + xp.take_along_axis(side, direction[None], 0)[0].astype(direction.dtype)
        kernels = _refined_lee_kernels(xp)
        z2 = z * z
        means = xp.stack([nd.convolve(z, k, mode="reflect") for k in kernels])
        sq = xp.stack([nd.convolve(z2, k, mode="reflect") for k in kernels])
        m = xp.take_along_axis(means, sel[None], 0)[0]
        v = xp.maximum(xp.take_along_axis(sq, sel[None], 0)[0] - m * m, 0)
        var_x = (v - m * m * cu2) / (1 + cu2)
        b = xp.nan_to_num(xp.clip(var_x / xp.where(v > 0, v, xp.nan), 0, 1), nan=0.0)
        out = m + b * (z - m)
        return xp.where(valid, out, xp.nan)

    def f(a, xp):
        return _per_image(as_float(a, xp), xp, lambda x: one(x, xp))

    return run_array(img, f)


def multitemporal(stack: Any, *, spatial: str = "boxcar", size: int = 7,
                  enl: float = S1_IW_GRDH_ENL, time_axis: int = 0):
    """Quegan & Yu (2001) multi-temporal filter.

    ``J_k = (E[I_k] / N) * sum_i (I_i / E[I_i])`` where ``E`` is a spatial filter
    (``boxcar``, ``lee`` or ``refined_lee``). Reduces speckle by up to sqrt(N) while
    preserving each date's local mean. ``stack`` has time on ``time_axis``.
    """
    spatial_fn = {"boxcar": lambda a: boxcar(a, size), "lee": lambda a: lee(a, size, enl),
                  "refined_lee": lambda a: refined_lee(a, enl)}[spatial]

    def f(a, xp):
        a = xp.moveaxis(as_float(a, xp), time_axis, 0)
        e = xp.stack([xp.asarray(spatial_fn(a[k])) for k in range(a.shape[0])])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = xp.where(e > 0, a / e, xp.nan)
        n = xp.sum(~xp.isnan(ratio), axis=0)
        mean_ratio = xp.nansum(ratio, axis=0) / xp.where(n > 0, n, xp.nan)
        out = e * mean_ratio[None]
        return xp.moveaxis(xp.where(xp.isnan(a), xp.nan, out), 0, time_axis)

    return run_array(stack, f)


def enl_estimate(img: Any) -> float:
    """Estimate ENL over a homogeneous area: ``mean^2 / var`` of linear intensity."""
    from ..backend import to_cpu

    a = np.asarray(to_cpu(getattr(img, "values", img)), dtype="float64")
    a = a[np.isfinite(a)]
    return float(a.mean() ** 2 / a.var())
