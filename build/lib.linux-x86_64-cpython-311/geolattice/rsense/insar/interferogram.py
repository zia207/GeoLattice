"""Native interferogram formation, coherence estimation and phase filtering (GPU-capable).

Inputs are co-registered complex SLC arrays (NumPy/CuPy/DataArray) on the same grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import fft, ndimage
from ..core._utils import run_array

SENTINEL1_WAVELENGTH = 0.05546576  # m (C-band, 5.405 GHz)


def _box(a: Any, size: tuple[int, int], xp: Any) -> Any:
    nd = ndimage(a)
    if xp.iscomplexobj(a):
        return nd.uniform_filter(a.real, size, mode="reflect") + \
            1j * nd.uniform_filter(a.imag, size, mode="reflect")
    return nd.uniform_filter(a, size, mode="reflect")


def interferogram(reference: Any, secondary: Any, *, window: tuple[int, int] = (5, 5)) -> dict:
    """Form a complex interferogram with coherence.

    Returns ``{"interferogram": <s1 s2*> (complex, window-averaged), "phase": wrapped phase
    (rad), "coherence": |<s1 s2*>| / sqrt(<|s1|^2><|s2|^2>), "amplitude": (<|s1|^2><|s2|^2>)^(1/4)}``
    in the memory space of ``reference`` (DataArrays keep their coordinates).
    """
    from ..backend import get_xp, is_gpu_array, to_cpu
    from ..core._utils import to_xp

    was_gpu = is_gpu_array(getattr(reference, "data", reference))
    xp = get_xp()
    s1, s2 = to_xp(reference, xp), to_xp(secondary, xp)
    if s1.shape != s2.shape:
        raise ValueError("reference and secondary must be co-registered on the same grid")
    prod = _box(s1 * xp.conj(s2), window, xp)
    p1 = _box(xp.abs(s1) ** 2, window, xp)
    p2 = _box(xp.abs(s2) ** 2, window, xp)
    out = {"interferogram": prod, "phase": xp.angle(prod),
           "coherence": xp.clip(xp.abs(prod) / xp.sqrt(p1 * p2 + 1e-30), 0, 1),
           "amplitude": (p1 * p2) ** 0.25}
    if not was_gpu:
        out = {k: to_cpu(v) for k, v in out.items()}
    if hasattr(reference, "dims"):
        out = {k: reference.copy(data=v).rename(k) for k, v in out.items()}
    return out


def multilook_complex(ifg: Any, looks: tuple[int, int] = (4, 1)) -> Any:
    """Complex block averaging (azimuth, range looks) of an interferogram."""
    ly, lx = looks

    def f(a, xp):
        H, W = a.shape[-2] // ly * ly, a.shape[-1] // lx * lx
        b = a[..., :H, :W].reshape(a.shape[:-2] + (H // ly, ly, W // lx, lx))
        return b.mean(axis=(-3, -1))

    data = getattr(ifg, "data", ifg)
    return run_array(data, f)


def goldstein_filter(ifg: Any, *, alpha: float = 0.5, patch: int = 32, overlap: int = 8,
                     smooth: int = 3) -> Any:
    """Goldstein & Werner (1998) adaptive phase filter.

    Each overlapping ``patch`` is filtered in the frequency domain by ``|S|^alpha``
    (with ``|S|`` smoothed by a ``smooth``-pixel box) and blended back with triangular
    weights. ``alpha`` in [0, 1]: higher = stronger smoothing.
    """
    def f(a, xp):
        F = fft(a)
        nd = ndimage(a)
        H, W = a.shape
        step = patch - 2 * overlap
        out = xp.zeros_like(a)
        wsum = xp.zeros(a.shape, dtype=xp.float32)
        w1 = 1 - xp.abs(xp.linspace(-1, 1, patch, dtype=xp.float32))
        w = xp.outer(w1, w1) + 1e-3
        z = xp.nan_to_num(a)
        pad_h, pad_w = (-H) % step + patch, (-W) % step + patch
        zp = xp.pad(z, ((0, pad_h), (0, pad_w)))
        for r in range(0, H, step):
            for c in range(0, W, step):
                blk = zp[r:r + patch, c:c + patch]
                S = F.fft2(blk)
                mag = nd.uniform_filter(xp.abs(S), smooth, mode="wrap")
                mag = mag / (mag.max() + 1e-30)
                filt = F.ifft2(S * mag**alpha)
                rr, cc = min(patch, H - r), min(patch, W - c)
                out[r:r + rr, c:c + cc] += (filt * w)[:rr, :cc]
                wsum[r:r + rr, c:c + cc] += w[:rr, :cc]
        return out / wsum

    data = getattr(ifg, "data", ifg)
    res = run_array(data, f)
    return ifg.copy(data=res) if hasattr(ifg, "dims") else res


def remove_ramp(phase: Any, mask: Any | None = None, order: int = 1) -> Any:
    """Fit and subtract a planar (order 1) or quadratic (order 2) ramp from *unwrapped* phase."""
    def f(p, xp):
        H, W = p.shape
        yy, xx = xp.mgrid[0:H, 0:W]
        yy, xx = yy.astype(p.dtype) / H, xx.astype(p.dtype) / W
        terms = [xp.ones_like(xx), xx, yy] + ([xx * xx, yy * yy, xx * yy] if order == 2 else [])
        A = xp.stack([t.ravel() for t in terms], 1)
        ok = xp.isfinite(p.ravel())
        if mask is not None:
            ok &= xp.asarray(getattr(mask, "data", mask)).ravel().astype(bool)
        coef = xp.linalg.lstsq(A[ok], p.ravel()[ok], rcond=None)[0]
        return p - (A @ coef).reshape(H, W)

    data = getattr(phase, "data", phase)
    res = run_array(data, f)
    return phase.copy(data=res) if hasattr(phase, "dims") else res


def wrap(phase: Any) -> Any:
    """Wrap phase to (-pi, pi]."""
    return np.angle(np.exp(1j * phase))
