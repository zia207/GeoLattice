"""Differential InSAR helpers: topographic phase removal, reference points, GNSS comparison.

The Sentinel-1 burst pre-processing (orbits, back-geocoding, ESD, deburst, stack
co-registration with :func:`geolattice.rsense.sar.coregister_stack`) is in
:mod:`geolattice.rsense.sar.sentinel1`; unwrapping, LOS conversion and the SBAS inversion are in the
rest of :mod:`geolattice.rsense.insar`. These functions connect the two:

* :func:`topographic_phase` -- the phase a DEM predicts for a pair (flat Earth plus topography),
  from the slant-range difference that back-geocoding computes;
* :func:`differential_interferogram` -- reference x conj(secondary) with that phase removed,
  leaving deformation, atmosphere, residual orbit/DEM error and noise;
* :func:`select_reference_point` / :func:`reference_to_point` -- pick a stable, coherent pixel
  and make displacements relative to it;
* :func:`enu_to_los` -- project GNSS east/north/up motion onto the radar line of sight.

Phase convention: ``reference * conj(secondary)`` has phase ``4 pi / lambda * (R_sec - R_ref)``,
so :func:`geolattice.rsense.insar.phase_to_los` turns it into displacement *toward* the satellite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def topographic_phase(delta_range: Any, wavelength: float) -> Any:
    """Phase (rad) a DEM predicts for a pair: ``4 pi / lambda * (R_sec - R_ref)`` at the DEM
    surface, i.e. the flat-Earth *and* topographic phase together.

    ``delta_range`` is the secondary-minus-reference slant range at the DEM surface
    (``back_geocode(...)["delta_range"]``, or for two stack members ``i``, ``j`` of
    :func:`geolattice.rsense.sar.coregister_stack`, ``delta_range_j - delta_range_i``). Subtracting
    this phase is the step that turns an interferogram into a *differential* interferogram.
    Compare :func:`geolattice.rsense.insar.flat_earth_phase`, which uses the range difference on the
    ellipsoid and so keeps the topography (for DEM generation).
    """
    return 4 * np.pi / wavelength * np.asarray(delta_range)


def differential_interferogram(reference: Any, secondary: Any, delta_range: Any, wavelength: float) -> np.ndarray:
    """``reference * conj(secondary) * exp(-1j * topographic_phase(delta_range))`` for two
    co-registered SLCs on the same grid. Pixels without data (zero SLC samples or NaN range
    difference) become 0, which multilooking and :func:`geolattice.rsense.insar.unwrap` masks treat
    as no data."""
    m = np.asarray(reference)
    s = np.asarray(secondary)
    z = m * np.conj(s) * np.exp(-1j * topographic_phase(delta_range, wavelength))
    z = np.nan_to_num(z)
    z[(m == 0) | (s == 0)] = 0
    return z.astype("complex64")


def select_reference_point(coherence: Any, *, mask: Any | None = None, window: int = 7,
                           velocity: Any | None = None, max_velocity: float | None = None
                           ) -> tuple[int, int]:
    """Choose a reference pixel ``(row, col)``: the centre of the ``window`` x ``window``
    neighbourhood with the highest mean coherence, among pixels allowed by ``mask`` (e.g. an area
    known to be stable, or the main unwrapped component). With ``velocity`` and
    ``max_velocity``, pixels moving faster than ``max_velocity`` (absolute) are excluded too.

    InSAR displacements are relative; a coherent, stable reference keeps unwrapping errors and
    local motion out of every other pixel's value.
    """
    from scipy import ndimage

    c = np.nan_to_num(np.asarray(getattr(coherence, "values", coherence), float))
    score = ndimage.uniform_filter(c, window)
    ok = np.isfinite(np.asarray(getattr(coherence, "values", coherence), float))
    if mask is not None:
        ok &= np.asarray(getattr(mask, "values", mask), bool)
    if velocity is not None and max_velocity is not None:
        v = np.asarray(getattr(velocity, "values", velocity), float)
        ok &= np.isfinite(v) & (np.abs(v) <= max_velocity)
    # the whole window must be allowed, not just its centre
    ok = ndimage.minimum_filter(ok.astype(np.uint8), window) > 0
    if not ok.any():
        raise ValueError("No pixel satisfies the mask/velocity constraints")
    score = np.where(ok, score, -np.inf)
    r, c_ = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(r), int(c_)


def reference_to_point(data: Any, point: tuple[int, int], *, window: int = 3) -> Any:
    """Subtract the median of a ``window`` x ``window`` box around ``point`` (row, col) from
    every pixel; works on 2-D maps and on stacks with leading dimensions (e.g. ``(time, y, x)``),
    each layer referenced separately."""
    arr = np.asarray(getattr(data, "values", data), float)
    r, c = point
    h = window // 2
    box = arr[..., max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1]
    ref = np.nanmedian(box.reshape(box.shape[:-2] + (-1,)), axis=-1)
    out = arr - ref[..., None, None]
    if isinstance(data, xr.DataArray):
        return data.copy(data=out)
    return out


def enu_to_los(east: Any, north: Any, up: Any, los_enu: Any) -> Any:
    """Project east/north/up displacements (e.g. GNSS) onto the line of sight; ``los_enu`` is the
    ground-to-satellite unit vector ``(e, n, u)`` from :func:`geolattice.rsense.sar.los_unit_vector`.
    Positive = toward the satellite."""
    v = np.asarray(los_enu, float)
    return np.asarray(east) * v[..., 0] + np.asarray(north) * v[..., 1] + np.asarray(up) * v[..., 2]
