"""Topographic InSAR: baselines, height of ambiguity, phase to height and DEM quality control.

Steps 1, 6, 10, 11 and 13 of the classic Sentinel-1 interferometric DEM chain (the burst
pre-processing of Steps 2-5, 7 and 12 lives in :mod:`geolattice.rsense.sar.sentinel1`):

* :func:`perpendicular_baseline` and :func:`height_of_ambiguity` -- how sensitive a pair is to
  topography (pair selection);
* :func:`flat_earth_phase` -- the phase of a flat (ellipsoidal) Earth, removed when the
  interferogram is formed (the topographic phase is kept);
* :func:`unwrap_snaphu_topo` -- SNAPHU in ``TOPO`` statistical-cost mode;
* :func:`phase_to_height` -- unwrapped phase to elevation, tied to a reference height;
* :func:`geoid_to_ellipsoid`, :func:`height_std`, :func:`detrend_heights`,
  :func:`combine_heights`, :func:`dem_error_stats` and :func:`fill_dem_voids` -- datum
  handling, error budget, tropospheric/orbital trend removal, multi-pair averaging, validation
  and gap filling.

Phase convention: Sentinel-1 SLCs carry ``exp(-1j 4 pi R / lambda)``, so an interferogram
``reference * conj(secondary)`` has phase ``4 pi / lambda * (R_secondary - R_reference)``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def perpendicular_baseline(ref_orbit: dict, sec_orbit: dict, lat: Any, lon: Any, height: Any = 0.0
                           ) -> dict[str, np.ndarray]:
    """Perpendicular and parallel baseline (m) of a pair at ground points.

    Each orbit (``read_orbit_file`` layout) is placed at its own zero-Doppler time for the
    point; the baseline vector between the two satellite positions is split into the part along
    the reference line of sight (``bparallel``) and the part perpendicular to it (``bperp``,
    positive when the secondary is on the far-from-Earth side of the reference look direction).
    Also returns the reference ``slant_range`` (m) and ellipsoid ``incidence`` angle (degrees).
    """
    from ..sar.sentinel1 import _Orbit, geo2rdr, llh_to_ecef

    P = llh_to_ecef(lat, lon, height)
    shape = P.shape[:-1]
    P = P.reshape(-1, 3)
    out = {}
    pos = []
    for orbit in (ref_orbit, sec_orbit):
        t, _ = geo2rdr(np.ravel(lat), np.ravel(lon), np.broadcast_to(height, np.shape(lat)).ravel(), orbit)
        orb = _Orbit(orbit)
        pos.append(orb.pos(orb.seconds(t)))
    s1, s2 = pos
    rng = np.linalg.norm(P - s1, axis=1)
    los = (P - s1) / rng[:, None]
    b = s2 - s1
    bpar = np.einsum("ij,ij->i", b, los)
    bperp_vec = b - bpar[:, None] * los
    up = s1 / np.linalg.norm(s1, axis=1, keepdims=True)
    up = up - np.einsum("ij,ij->i", up, los)[:, None] * los
    sign = np.sign(np.einsum("ij,ij->i", bperp_vec, up))
    lat_r, lon_r = np.radians(np.ravel(lat)), np.radians(np.ravel(lon))
    normal = np.c_[np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    inc = np.degrees(np.arccos(np.clip(-np.einsum("ij,ij->i", los, normal), -1, 1)))
    out["bperp"] = (sign * np.linalg.norm(bperp_vec, axis=1)).reshape(shape)
    out["bparallel"] = bpar.reshape(shape)
    out["slant_range"] = rng.reshape(shape)
    out["incidence"] = inc.reshape(shape)
    return out


def height_of_ambiguity(wavelength: float, slant_range: Any, incidence_deg: Any, bperp: Any) -> Any:
    """Height change (m) that produces one full fringe (2 pi): ``lambda R sin(theta) / (2 B_perp)``.

    Small values mean high sensitivity to topography (and to phase noise turned into height
    error); the sign follows ``bperp``.
    """
    return wavelength * np.asarray(slant_range) * np.sin(np.radians(incidence_deg)) / (2 * np.asarray(bperp, float))


def flat_earth_phase(flat_delta_range: Any, wavelength: float) -> Any:
    """Phase (rad) of a flat, ellipsoidal Earth from the secondary-minus-reference slant range
    difference on the ellipsoid (``back_geocode(...)["flat_delta_range"]``). Remove it with
    ``ifg * exp(-1j * phase)`` when forming the interferogram."""
    return 4 * np.pi / wavelength * np.asarray(flat_delta_range)


def phase_to_height(unwrapped_phase: Any, wavelength: float, dheight_ddelta_range: Any, *,
                    reference: Any | None = None, mask: Any | None = None, components: Any | None = None,
                    min_tie_pixels: int = 100) -> tuple[Any, Any]:
    """Convert flat-Earth-removed unwrapped phase (rad) to elevation (m) -- SNAP's
    ``Phase to Elevation``.

    ``dheight_ddelta_range`` is metres of height per metre of range difference
    (``back_geocode(...)["dheight_ddelta_range"]``, multilooked like the phase), so
    ``h = phase * lambda / (4 pi) * dh/dDeltaR``. Unwrapped phase is only known up to a constant
    (an unknown whole number of cycles), so the result is relative; with ``reference`` heights
    (a tie point or a reference DEM on the same grid) the median difference over ``mask`` is
    added. With ``components`` (SNAPHU connected components, 0 = not unwrapped) each component
    is unwrapped independently and so gets its own offset; components with fewer than
    ``min_tie_pixels`` tie pixels, and component 0, become NaN. Returns ``(height, offset)``,
    the offset a float or, with ``components``, a ``{label: offset}`` dict.
    """
    phase = np.asarray(getattr(unwrapped_phase, "values", unwrapped_phase), float)
    h = phase * wavelength / (4 * np.pi) * np.asarray(getattr(dheight_ddelta_range, "values", dheight_ddelta_range), float)
    offset: Any = 0.0
    if reference is not None:
        ref = np.asarray(getattr(reference, "values", reference), float)
        ok = np.isfinite(h) & np.isfinite(ref)
        if mask is not None:
            ok &= np.asarray(getattr(mask, "values", mask), bool)
        if components is None:
            offset = float(np.median(ref[ok] - h[ok]))
            h = h + offset
        else:
            comp = np.asarray(getattr(components, "values", components))
            offset = {}
            out = np.full(h.shape, np.nan)
            for lab in np.unique(comp[comp > 0]):
                sel = ok & (comp == lab)
                if sel.sum() >= min_tie_pixels:
                    offset[int(lab)] = float(np.median(ref[sel] - h[sel]))
                    out[comp == lab] = h[comp == lab] + offset[int(lab)]
            h = out
    if hasattr(unwrapped_phase, "dims"):
        return unwrapped_phase.copy(data=h.astype("float32")).rename("height"), offset
    return h.astype("float32"), offset


def unwrap_snaphu_topo(ifg: Any, coherence: Any, *, wavelength: float, bperp: float, near_range: float,
                       range_spacing: float, azimuth_spacing: float, altitude: float, nlooks_range: int,
                       nlooks_azimuth: int, earth_radius: float = 6_371_000.0, power: Any | None = None,
                       range_resolution: float | None = None, azimuth_resolution: float | None = None,
                       mask: Any | None = None, ntiles: tuple[int, int] = (1, 1), tile_overlap: int = 200,
                       nproc: int = 1, transmit_mode: str = "REPEATPASS", init: str = "MCF",
                       workdir: str | os.PathLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap a topographic interferogram with SNAPHU in ``TOPO`` statistical-cost mode.

    TOPO mode models how phase gradients arise from terrain (including layover, which produces
    discontinuities facing the radar), so it needs the acquisition geometry: ``bperp`` (m),
    ``near_range`` (slant range of column 0, m), the multilooked ``range_spacing`` /
    ``azimuth_spacing`` (m), the satellite ``altitude`` above ``earth_radius`` (m), the number
    of looks and optionally the single-look resolutions. Uses the ``snaphu`` executable bundled
    with ``pip install snaphu``. Returns ``(unwrapped_phase, connected_components)``; masked or
    invalid pixels are NaN / component 0.
    """
    try:
        from snaphu._snaphu import get_snaphu_executable
    except ImportError as e:
        raise ImportError("unwrap_snaphu_topo needs `pip install snaphu`") from e

    z = np.asarray(getattr(ifg, "values", ifg))
    z = z if np.iscomplexobj(z) else np.exp(1j * z)
    corr = np.nan_to_num(np.asarray(getattr(coherence, "values", coherence), "float32"))
    bad = ~np.isfinite(z) | (z == 0)
    if mask is not None:
        bad |= ~np.asarray(getattr(mask, "values", mask), bool)
    z = np.where(bad, 0, z).astype("complex64")
    H, W = z.shape
    pwr = (np.abs(z) ** 2 if power is None else np.nan_to_num(np.asarray(power))).astype("float32")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        z.tofile(tmp / "ifg.c8")
        corr.tofile(tmp / "corr.f4")
        pwr.tofile(tmp / "pwr.f4")
        (~bad).astype("uint8").tofile(tmp / "mask.u1")
        cfg = {
            "INFILE": tmp / "ifg.c8", "INFILEFORMAT": "COMPLEX_DATA", "LINELENGTH": W,
            "OUTFILE": tmp / "unw.f4", "OUTFILEFORMAT": "FLOAT_DATA",
            "CORRFILE": tmp / "corr.f4", "CORRFILEFORMAT": "FLOAT_DATA",
            "PWRFILE": tmp / "pwr.f4", "AMPFILEFORMAT": "FLOAT_DATA",
            "BYTEMASKFILE": tmp / "mask.u1", "CONNCOMPFILE": tmp / "cc.u1",
            "STATCOSTMODE": "TOPO", "INITMETHOD": init, "TRANSMITMODE": transmit_mode,
            "LAMBDA": wavelength, "BPERP": bperp, "NEARRANGE": near_range,
            "DR": range_spacing, "DA": azimuth_spacing, "ALTITUDE": altitude, "EARTHRADIUS": earth_radius,
            "NLOOKSRANGE": nlooks_range, "NLOOKSAZ": nlooks_azimuth,
            "NCORRLOOKS": max(1.0, 0.5 * nlooks_range * nlooks_azimuth),
            "NTILEROW": ntiles[0], "NTILECOL": ntiles[1], "ROWOVRLP": tile_overlap, "COLOVRLP": tile_overlap,
            "NPROC": nproc,
        }
        if range_resolution:
            cfg["RANGERES"] = range_resolution
        if azimuth_resolution:
            cfg["AZRES"] = azimuth_resolution
        if ntiles != (1, 1):
            cfg["SINGLETILEREOPTIMIZE"] = "TRUE"
        (tmp / "snaphu.conf").write_text("".join(f"{k} {v}\n" for k, v in cfg.items()))
        with get_snaphu_executable() as exe:
            res = subprocess.run([str(exe), "-f", str(tmp / "snaphu.conf")], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"snaphu failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        unw = np.fromfile(tmp / "unw.f4", "float32").reshape(H, W)
        cc = np.fromfile(tmp / "cc.u1", "uint8").reshape(H, W)
    unw[bad] = np.nan
    cc[bad] = 0
    return unw, cc


def geoid_to_ellipsoid(dem: xr.DataArray, geoid: str = "EGM96", *, inverse: bool = False,
                       samples: int = 25) -> xr.DataArray:
    """Convert heights above the geoid (orthometric, as in SRTM/NASADEM: ``EGM96``; Copernicus
    DEM: ``EGM2008``) to WGS84 ellipsoidal heights (``inverse=True``: the reverse).

    InSAR geometry needs ellipsoidal heights; the geoid undulation is tens of metres. The
    undulation is computed with PROJ (geoid grids are fetched from the PROJ CDN) on a
    ``samples`` x ``samples`` lattice and interpolated, since it varies smoothly.
    """
    import pyproj
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    vcrs = {"EGM96": 5773, "EGM2008": 3855}[geoid.upper()]
    was = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(True)
    try:
        xs = np.linspace(float(dem.x.min()), float(dem.x.max()), samples)
        ys = np.linspace(float(dem.y.min()), float(dem.y.max()), samples)
        X, Y = np.meshgrid(xs, ys)
        crs = dem.rio.crs or "EPSG:4326"
        lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(X, Y)
        for attempt in range(4):  # geoid grid tiles come from the PROJ CDN: retry transient failures
            to_ell = Transformer.from_crs(f"EPSG:4326+{vcrs}", "EPSG:4979", always_xy=True)
            _, _, n = to_ell.transform(lon, lat, np.zeros_like(lon))
            if np.all(np.isfinite(n)):
                break
            __import__("time").sleep(2 ** attempt)
    finally:
        pyproj.network.set_network_enabled(was)
    if not np.all(np.isfinite(n)):
        raise RuntimeError("Geoid grid unavailable (PROJ network access needed)")
    interp = RegularGridInterpolator((ys, xs), n, bounds_error=False, fill_value=None)
    Xf, Yf = np.meshgrid(dem.x.values, dem.y.values)
    und = interp(np.c_[Yf.ravel(), Xf.ravel()]).reshape(Xf.shape)
    out = dem - und if inverse else dem + und
    return out.assign_attrs(dem.attrs, vertical_datum="WGS84 ellipsoid" if not inverse else geoid,
                            geoid_undulation_mean=float(np.mean(und)))


def height_std(coherence: Any, nlooks: float, height_ambiguity: Any) -> Any:
    """Height standard deviation (m) expected from phase noise: the Cramer-Rao bound
    ``sigma_phi = sqrt(1 - g^2) / (g sqrt(2 N))`` scaled by ``|h_a| / (2 pi)``."""
    g = np.clip(np.asarray(coherence, float), 1e-3, 0.9999)
    sigma_phi = np.sqrt(1 - g**2) / (g * np.sqrt(2 * nlooks))
    return np.abs(np.asarray(height_ambiguity, float)) * sigma_phi / (2 * np.pi)


def combine_heights(heights: list[Any], sigmas: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weighted mean of several height maps (e.g. one per interferometric pair)
    on the same grid, ignoring NaN. Returns ``(height, sigma)``."""
    h = np.stack([np.asarray(getattr(a, "values", a), float) for a in heights])
    s = np.stack([np.asarray(getattr(a, "values", a), float) for a in sigmas])
    w = np.where(np.isfinite(h) & np.isfinite(s) & (s > 0), 1 / s**2, 0.0)
    wsum = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, np.nansum(np.nan_to_num(h) * w, 0) / wsum, np.nan)
        sigma = np.where(wsum > 0, 1 / np.sqrt(wsum), np.nan)
    return mean, sigma


def detrend_heights(height: Any, reference: Any, mask: Any | None = None, *, order: int = 1,
                    elevation: bool = True) -> tuple[Any, dict[str, float]]:
    """Remove long-wavelength errors from an InSAR height map by fitting its difference from a
    reference DEM on the same grid and subtracting the fit.

    The fit is a polynomial surface in the grid coordinates (``order`` 1: a plane, from residual
    orbit errors and large-scale atmospheric gradients; 2: quadratic) plus, with
    ``elevation=True``, a term proportional to the reference height -- the signature of a
    *stratified* troposphere, whose delay difference between the two dates grows with the
    height of the ground. Only pixels in ``mask`` (e.g. coherent, unmasked ground) enter the
    fit. The correction transfers the reference DEM's longest wavelengths (and its own errors at
    those wavelengths) into the result; validate against data independent of ``reference``.
    Returns ``(corrected, coefficients)`` with coefficients per normalised grid unit and, for
    ``elevation``, metres of error per metre of height.
    """
    h = np.asarray(getattr(height, "values", height), float)
    ref = np.asarray(getattr(reference, "values", reference), float)
    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    yy, xx = yy / max(ny - 1, 1), xx / max(nx - 1, 1)
    terms = {"const": np.ones_like(xx), "x": xx, "y": yy}
    if order >= 2:
        terms.update({"x2": xx**2, "y2": yy**2, "xy": xx * yy})
    if elevation:
        terms["elevation"] = np.nan_to_num(ref)
    ok = np.isfinite(h) & np.isfinite(ref)
    if mask is not None:
        ok &= np.asarray(getattr(mask, "values", mask), bool)
    A = np.stack([t[ok] for t in terms.values()], axis=1)
    coef = np.linalg.lstsq(A, h[ok] - ref[ok], rcond=None)[0]
    fit = sum(c * t for c, t in zip(coef, terms.values()))
    out = (h - fit).astype("float32")
    coefs = {k: float(c) for k, c in zip(terms, coef)}
    if hasattr(height, "dims"):
        return height.copy(data=out), coefs
    return out, coefs


def dem_error_stats(dem: Any, reference: Any, mask: Any | None = None) -> dict[str, float]:
    """Accuracy of a DEM against reference heights on the same grid (or at the same points):
    ``n``, ``bias`` (mean error), ``median``, ``std``, ``rmse``, ``nmad`` (1.4826 x median
    absolute deviation: a robust standard deviation) and ``le90`` (90th percentile of the
    absolute error)."""
    e = np.asarray(getattr(dem, "values", dem), float) - np.asarray(getattr(reference, "values", reference), float)
    ok = np.isfinite(e)
    if mask is not None:
        ok &= np.asarray(getattr(mask, "values", mask), bool)
    e = e[ok]
    if e.size == 0:
        return dict(n=0, bias=np.nan, median=np.nan, std=np.nan, rmse=np.nan, nmad=np.nan, le90=np.nan)
    med = float(np.median(e))
    return dict(n=int(e.size), bias=float(e.mean()), median=med, std=float(e.std()),
                rmse=float(np.sqrt(np.mean(e**2))), nmad=float(1.4826 * np.median(np.abs(e - med))),
                le90=float(np.percentile(np.abs(e), 90)))


def fill_dem_voids(dem: xr.DataArray, reference: xr.DataArray | None = None, *,
                   max_search_distance: float = 100, smoothing_iterations: int = 0) -> xr.DataArray:
    """Fill NaN voids of a DEM.

    With a ``reference`` DEM on the same grid, the *delta-surface* method is used: the
    difference ``dem - reference`` is interpolated into the voids (inverse distance, GDAL
    ``FillNodata``) and added back to the reference, so the filled surface keeps the
    reference's terrain shape while matching the InSAR DEM's level at the void edges. Without
    a reference the heights themselves are interpolated (fine for small voids only).
    ``max_search_distance`` is in pixels.
    """
    from rasterio.fill import fillnodata

    base = np.asarray(dem.values, float)
    void = ~np.isfinite(base)
    if reference is None:
        target = base
    else:
        ref = np.asarray(reference.values, float)
        target = base - ref
    filled = fillnodata(np.nan_to_num(target).astype("float32"), mask=(~void).astype("uint8"),
                        max_search_distance=max_search_distance, smoothing_iterations=smoothing_iterations)
    if reference is not None:
        filled = filled + ref
    still = void & (filled == 0) if reference is None else void & ~np.isfinite(filled)
    out = np.where(void, filled, base)
    out[still] = np.nan
    return dem.copy(data=out.astype("float32")).assign_attrs(dem.attrs, filled_fraction=float(void.mean()))
