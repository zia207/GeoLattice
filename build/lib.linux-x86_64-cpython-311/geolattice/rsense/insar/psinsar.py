"""Persistent Scatterer InSAR (PS-InSAR) on a co-registered single-reference SLC stack.

The steps follow Ferretti et al. (2001) and Hooper et al. (2004, StaMPS):

* :func:`amplitude_dispersion` / :func:`ps_candidates` -- candidate selection by the amplitude
  dispersion index D_A on radiometrically normalised amplitudes;
* :func:`sparse_lowpass` -- the spatially correlated phase of each interferogram (deformation,
  atmosphere, orbit) estimated from neighbouring points;
* :func:`ps_periodogram` -- per point, the DEM (look-angle) error and optionally a linear
  velocity that maximise the temporal coherence of the wrapped phase;
* :func:`coherence_threshold` -- the temporal coherence that random-phase pixels reach with the
  same estimator, for a chosen false-positive rate;
* :func:`unwrap_points` -- unwrapping of the sparse PS phase through a coarse grid (SNAPHU) plus
  each point's wrapped residual;
* :func:`atmospheric_filter` -- the atmospheric phase screen: high-pass in time, low-pass in
  space;
* :func:`point_velocity` -- linear LOS velocity and residual RMSE per point.

The co-registration itself (back-geocoding, deramping, ESD) is
:func:`geolattice.rsense.sar.coregister_stack`, which also provides the per-date height sensitivity
``kz`` the DEM-error term needs. Phase convention: interferograms ``reference * conj(secondary)``,
so a phase ``phi`` is ``-lambda / (4 pi) * phi`` metres of motion toward the satellite
(:func:`geolattice.rsense.insar.phase_to_los`).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def amplitude_dispersion(amplitude: Any, *, normalize: bool = True, min_valid: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude dispersion index ``D_A = sigma_A / mu_A`` and mean amplitude of a stack
    ``(n_images, ...)``.

    For a point dominated by one stable scatterer, D_A approximates the phase standard deviation
    (Ferretti et al. 2001), so a low D_A flags phase-stable candidates without any phase
    information. ``normalize`` divides each image by its own median amplitude first (radiometric
    calibration across dates), so that a brighter or darker acquisition does not inflate D_A.
    Zero (no-data) samples are ignored; pixels with fewer than ``min_valid`` valid samples
    (default: half the stack) are NaN.
    """
    a = np.asarray(amplitude, float)
    a = np.where(a > 0, a, np.nan)
    if normalize:
        med = np.nanmedian(a.reshape(a.shape[0], -1), axis=1)
        a = a / med.reshape((-1,) + (1,) * (a.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(a, axis=0)
        disp = np.nanstd(a, axis=0) / mean
    few = np.sum(np.isfinite(a), axis=0) < (a.shape[0] // 2 if min_valid is None else min_valid)
    disp[few] = np.nan
    mean[few] = np.nan
    return disp.astype("float32"), mean.astype("float32")


def ps_candidates(dispersion: Any, threshold: float = 0.25, *, mean_amplitude: Any | None = None,
                  min_amplitude: float | None = None, peak_window: int | None = None) -> np.ndarray:
    """Boolean mask of PS candidates: ``D_A < threshold`` (Ferretti: 0.25; StaMPS uses a looser
    0.4 for candidates and selects on phase later). Optionally also require a mean (normalised)
    amplitude of at least ``min_amplitude`` and, with ``peak_window``, a local maximum of the mean
    amplitude in a ``peak_window`` x ``peak_window`` box (one candidate per bright point target)."""
    d = np.asarray(dispersion, float)
    ok = np.isfinite(d) & (d < threshold)
    if mean_amplitude is not None:
        m = np.asarray(mean_amplitude, float)
        if min_amplitude is not None:
            ok &= m >= min_amplitude
        if peak_window:
            from scipy import ndimage

            mm = np.nan_to_num(m)
            ok &= mm == ndimage.maximum_filter(mm, peak_window)
    return ok


def _bin_points(values: np.ndarray, x: np.ndarray, y: np.ndarray, cell: float, weights: np.ndarray | None = None):
    """Sum values (n, P) into a regular grid of ``cell``-sized bins; returns sums, counts, indices."""
    x0, y0 = np.nanmin(x), np.nanmin(y)
    ix = ((x - x0) // cell).astype(int)
    iy = ((y - y0) // cell).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    flat = iy * nx + ix
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    vals = np.atleast_2d(values)
    sums = np.zeros((vals.shape[0], ny * nx), dtype=vals.dtype)
    for k in range(vals.shape[0]):
        np.add.at(sums[k], flat, vals[k] * w)
    counts = np.bincount(flat, weights=w, minlength=ny * nx)
    return sums.reshape(vals.shape[0], ny, nx), counts.reshape(ny, nx), (iy, ix)


def sparse_lowpass(values: Any, x: Any, y: Any, *, cell: float = 50.0, sigma: float = 200.0,
                   exclude_self: bool = False) -> np.ndarray:
    """Gaussian low-pass of values known only at scattered points, evaluated at the same points.

    ``values`` is ``(n, P)`` (one row per interferogram; real values, or unit phasors
    ``exp(1j * phase)`` for wrapped phase), ``x``/``y`` the point coordinates in metres. Points
    are summed into ``cell``-metre bins, the sums and counts are smoothed with a Gaussian of
    ``sigma`` metres and divided, and the result is read back at each point -- the spatially
    correlated part of the signal (deformation, atmosphere, orbit), which StaMPS removes before
    estimating each point's own DEM error. ``exclude_self`` removes each point's own contribution
    from its estimate. For phasors the result is complex (its angle is the filtered phase).
    """
    from scipy import ndimage

    v = np.atleast_2d(np.asarray(values))
    x, y = np.asarray(x, float), np.asarray(y, float)
    sums, counts, (iy, ix) = _bin_points(v, x, y, cell)
    s = sigma / cell
    cnt = ndimage.gaussian_filter(counts, s, mode="constant")
    out = np.empty(v.shape, dtype=v.dtype if np.iscomplexobj(v) else float)
    w_self = ndimage.gaussian_filter(np.pad(np.ones((1, 1)), 4 * int(np.ceil(s)) + 1), s, mode="constant").max()
    for k in range(v.shape[0]):
        if np.iscomplexobj(v):
            sm = ndimage.gaussian_filter(sums[k].real, s, mode="constant") + 1j * ndimage.gaussian_filter(sums[k].imag, s, mode="constant")
        else:
            sm = ndimage.gaussian_filter(sums[k], s, mode="constant")
        num = sm[iy, ix]
        den = cnt[iy, ix]
        if exclude_self:
            num = num - w_self * v[k]
            den = den - w_self
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = num / den
    return out if np.ndim(values) > 1 else out[0]


def ps_periodogram(phase: Any, kz_phase: Any, *, dem_error: Any = None, time_years: Any | None = None,
                   velocity_phase: Any | None = None, chunk: int = 4000) -> dict[str, np.ndarray]:
    """Per-point grid search (periodogram) of the DEM error -- and optionally a linear phase rate --
    that maximises the temporal coherence of wrapped phase (Ferretti et al. 2001).

    ``phase`` is ``(n_ifg, P)`` wrapped phase (rad) of each point; ``kz_phase`` the phase per metre
    of height error for each interferogram and point (``4 pi / lambda * kz``, with ``kz`` from
    :func:`geolattice.rsense.sar.coregister_stack`). ``dem_error`` is the grid of trial height errors
    (m; default -40 to 40 m in 0.5 m steps). With ``time_years`` (``(n_ifg,)``) and
    ``velocity_phase`` (trial phase rates, rad/yr) the search is two-dimensional.

    Temporal coherence ``gamma = |mean_k exp(j (phi_k - model_k))|`` is 1 for a perfect fit and
    about ``1 / sqrt(n)`` for random phase (NaN where a point's phase or ``kz_phase`` is not
    finite). Returns ``dem_error``, ``phase_rate`` (if searched),
    ``offset`` (the constant phase of the best fit), ``coherence`` and the wrapped ``residual``
    ``(n_ifg, P)``.
    """
    ph = np.asarray(phase, float)
    kz = np.broadcast_to(np.asarray(kz_phase, float), ph.shape)
    dh = np.arange(-40.0, 40.25, 0.5) if dem_error is None else np.asarray(dem_error, float)
    rates = np.zeros(1) if velocity_phase is None else np.asarray(velocity_phase, float)
    t = np.zeros(ph.shape[0]) if time_years is None else np.asarray(time_years, float)
    n, P = ph.shape
    best_g = np.full(P, -1.0)
    best_dh = np.zeros(P)
    best_r = np.zeros(P)
    best_off = np.zeros(P)
    z = np.exp(1j * ph)
    for p0 in range(0, P, chunk):
        sl = slice(p0, min(p0 + chunk, P))
        zc, kc = z[:, sl], kz[:, sl]
        for r in rates:
            zr = zc * np.exp(-1j * r * t)[:, None]
            # (n, chunk, n_dh): exp(-j kz dh)
            m = np.einsum("np,npd->pd", zr, np.exp(-1j * kc[:, :, None] * dh[None, None, :])) / n
            g = np.abs(m)
            j = np.argmax(g, axis=1)
            gbest = g[np.arange(g.shape[0]), j]
            better = gbest > best_g[sl]
            idx = np.flatnonzero(better) + p0
            best_g[idx] = gbest[better]
            best_dh[idx] = dh[j[better]]
            best_r[idx] = r
            best_off[idx] = np.angle(m[np.arange(m.shape[0]), j])[better]
    bad = ~np.all(np.isfinite(ph) & np.isfinite(kz), axis=0)
    best_g[bad] = np.nan
    best_dh[bad] = np.nan
    model = kz * best_dh[None, :] + best_r[None, :] * t[:, None] + best_off[None, :]
    out = {"dem_error": best_dh, "offset": best_off, "coherence": best_g,
           "residual": np.angle(np.exp(1j * (ph - model)))}
    if velocity_phase is not None:
        out["phase_rate"] = best_r
    return out


def coherence_threshold(kz_phase: Any, *, fpr: float = 0.01, n_sim: int = 20000, dem_error: Any = None,
                        seed: int = 0) -> float:
    """Temporal coherence that pixels with *random* phase reach with :func:`ps_periodogram`, at
    false-positive rate ``fpr`` -- the threshold above which a point is unlikely to be noise.

    ``kz_phase`` is a representative ``(n_ifg,)`` phase-per-metre vector (e.g. the median over
    points); the grid search lets random phase fit some DEM error, so the threshold is higher than
    the ``1 / sqrt(n)`` of pure averaging, and higher still for few interferograms.
    """
    kz = np.asarray(kz_phase, float).reshape(-1, 1)
    if not np.all(np.isfinite(kz)):
        raise ValueError("kz_phase contains NaN; pass e.g. np.nanmedian(kz, axis=1)")
    rng = np.random.default_rng(seed)
    rnd = rng.uniform(-np.pi, np.pi, (kz.shape[0], n_sim))
    g = ps_periodogram(rnd, np.broadcast_to(kz, rnd.shape), dem_error=dem_error)["coherence"]
    return float(np.quantile(g, 1 - fpr))


def unwrap_points(phase: Any, x: Any, y: Any, *, cell: float = 100.0, min_points: int = 1,
                  method: str = "snaphu", cost: str = "smooth", sequential: bool = False) -> np.ndarray:
    """Unwrap the wrapped phase of sparse points ``(n_ifg, P)`` (after DEM-error removal).

    For each interferogram the points' unit phasors are averaged in ``cell``-metre bins, the
    resulting grid (cells with fewer than ``min_points`` points masked) is unwrapped with
    :func:`geolattice.rsense.insar.unwrap` (SNAPHU), and each point gets its cell's unwrapped phase plus
    its own wrapped difference from the cell's mean phase. This keeps the full-resolution
    point-to-cell detail while SNAPHU handles the large-scale integer cycles. Points in masked
    cells are NaN.

    With ``sequential=True`` the rows must be a *time-ordered* series of phases relative to one
    reference date; the differences between consecutive dates are unwrapped instead and summed,
    and the result is relative to the first row. A point that moves differently from its
    neighbours (a stable outcrop in a sinking city) then only needs to differ by less than half a
    cycle between consecutive dates, not over the whole span of each single-reference
    interferogram -- the temporal continuity that StaMPS's 3-D unwrapping exploits.
    """
    from .unwrap import unwrap

    ph = np.atleast_2d(np.asarray(phase, float))
    x, y = np.asarray(x, float), np.asarray(y, float)
    if sequential:
        steps = np.angle(np.exp(1j * np.diff(ph, axis=0)))
        u = unwrap_points(steps, x, y, cell=cell, min_points=min_points, method=method, cost=cost)
        # an unwrapped image is only defined up to a whole number of cycles; consecutive-date steps
        # are mostly within half a cycle, so anchor each step to its wrapped median before summing
        k = np.round(np.nanmedian(u - steps, axis=1, keepdims=True) / (2 * np.pi))
        u = u - 2 * np.pi * k
        return np.vstack([np.zeros((1, ph.shape[1])), np.cumsum(u, axis=0)])
    sums, counts, (iy, ix) = _bin_points(np.exp(1j * ph), x, y, cell)
    valid = counts >= min_points
    out = np.full(ph.shape, np.nan)
    for k in range(ph.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(valid, sums[k] / np.maximum(counts, 1), 0)
        grid = np.where(valid, mean, 0).astype("complex64")
        if method == "snaphu":
            qual = np.clip(np.abs(mean), 0, 1).astype("float32")
            unw = unwrap(grid, qual, method="snaphu", cost=cost, mask=valid,
                         nlooks=max(float(np.median(counts[valid])), 1.0))
        else:
            unw = unwrap(grid, method=method, mask=valid)
        u = np.asarray(unw)[iy, ix]
        cell_phase = np.angle(mean[iy, ix])
        out[k] = u + np.angle(np.exp(1j * (ph[k] - cell_phase)))
    return out


def atmospheric_filter(unwrapped: Any, x: Any, y: Any, time_years: Any, *, spatial_sigma: float = 1000.0,
                       cell: float = 100.0, temporal: str = "linear") -> np.ndarray:
    """Atmospheric phase screen of unwrapped point phase ``(n_dates, P)``: high-pass in time,
    low-pass in space (Ferretti et al. 2001).

    The deformation is assumed smooth in time, so each point's temporal model (``"linear"``: a
    least-squares line in ``time_years``) is removed first; what remains is atmosphere, nonlinear
    motion and noise. The atmosphere is spatially correlated over kilometres but random in time,
    so the remainder is low-pass filtered in space (:func:`sparse_lowpass`, Gaussian of
    ``spatial_sigma`` metres) for each date. Subtract the result from ``unwrapped``.
    """
    u = np.asarray(unwrapped, float)
    t = np.asarray(time_years, float)
    A = np.c_[np.ones_like(t), t]
    ok = np.all(np.isfinite(u), axis=0)
    resid = np.full(u.shape, np.nan)
    coef = np.linalg.lstsq(A, u[:, ok], rcond=None)[0]
    resid[:, ok] = u[:, ok] - A @ coef
    aps = np.full(u.shape, np.nan)
    aps[:, ok] = sparse_lowpass(resid[:, ok], np.asarray(x)[ok], np.asarray(y)[ok], cell=cell, sigma=spatial_sigma)
    return aps


def point_velocity(series: Any, time_years: Any) -> dict[str, np.ndarray]:
    """Least-squares linear rate (units per year) and residual RMSE of point time series
    ``(n_dates, P)``."""
    d = np.asarray(series, float)
    t = np.asarray(time_years, float)
    A = np.c_[np.ones_like(t), t]
    ok = np.all(np.isfinite(d), axis=0)
    rate = np.full(d.shape[1], np.nan)
    rmse = np.full(d.shape[1], np.nan)
    coef = np.linalg.lstsq(A, d[:, ok], rcond=None)[0]
    rate[ok] = coef[1]
    rmse[ok] = np.sqrt(np.mean((d[:, ok] - A @ coef) ** 2, axis=0))
    return {"velocity": rate, "rmse": rmse}
