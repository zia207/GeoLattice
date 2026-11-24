"""Ship detection for Sentinel-1: CFAR on GRD, sub-look coherence on IW SLC, and AIS matching.

A companion to :mod:`geolattice.rsense.sar.sentinel1`. That module is used as is: this one only
imports from it.

Workflow
--------
1. GRD (radar geometry, native 10 m): read the AOI window of VV and VH with
   :func:`sentinel1.read_grd_subset` (``border_noise=True``), build a sea mask with
   :func:`land_mask_for_window`, run :func:`cfar` on VH, then :func:`extract_targets`
   (centroids geolocated with the window's GCPs) and :func:`flag_azimuth_ambiguities`.
   :func:`detect_ships_grd` chains these steps.
2. SLC (same datatake): read the bursts over the AOI with :func:`sentinel1.read_slc_annotation`,
   :func:`sentinel1.apply_orbit_file`, :func:`sentinel1.burst_window` and
   :func:`sentinel1.read_slc_burst`; split each burst into two sub-looks with
   :func:`range_sublooks` (or :func:`azimuth_sublooks`, which deramps with
   :func:`sentinel1.tops_deramp_phase`); compute :func:`sublook_coherence`; sample it at the GRD
   candidates with :func:`coherence_at_points` (:func:`sentinel1.geo2rdr` +
   :func:`sentinel1.burst_coordinates`).
3. Labels: :func:`read_ais_dma` and :func:`ais_at_time` give AIS vessel positions at the SAR
   time; :func:`match_points` and :func:`detection_scores` evaluate the detections.

Do not speckle filter before CFAR: filtering smears small ships into the background.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .sentinel1 import (SPEED_OF_LIGHT, burst_coordinates, geo2rdr, incidence_angle_grid,
                        read_grd_subset, to_db)

EARTH_RADIUS = 6_371_000.0


# ============================================================================= sea mask
def land_mask_for_window(gcps: list, shape: tuple[int, int], land_geoms: Any, *,
                         buffer_m: float = 300.0, pixel_spacing: float = 10.0) -> np.ndarray:
    """Land mask (True = land or coastal buffer) for a GRD window in radar geometry.

    ``gcps`` are the window-local GCPs returned by :func:`sentinel1.read_grd_subset`;
    ``land_geoms`` is an iterable of lon/lat (EPSG:4326) land polygons, for example OSM land
    polygons or GSHHG clipped to the AOI. The polygons are burned in with an affine transform
    fitted to the local GCPs (accurate over a window of tens of km), then dilated by
    ``buffer_m`` so piers, harbours and moored boats do not trigger the detector.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_gcps

    transform = from_gcps(gcps)
    geoms = [(g, 1) for g in land_geoms]
    if not geoms:
        return np.zeros(shape, bool)
    land = rasterize(geoms, out_shape=shape, transform=transform, fill=0, dtype="uint8",
                     all_touched=True).astype(bool)
    iters = int(round(buffer_m / pixel_spacing))
    return ndi.binary_dilation(land, iterations=iters) if iters > 0 else land


# ============================================================================= clutter statistics
def estimate_enl(img: np.ndarray, valid: np.ndarray, *, block: int = 64,
                 max_cv_quantile: float = 0.3) -> float:
    """Equivalent number of looks ``mean^2 / var`` of linear intensity, from the most
    homogeneous sea blocks (lowest coefficient of variation). IW GRDH is about 4.4."""
    H, W = img.shape
    enl, cv = [], []
    for r in range(0, H - block + 1, block):
        for c in range(0, W - block + 1, block):
            m = valid[r:r + block, c:c + block]
            if m.mean() < 0.95:
                continue
            x = img[r:r + block, c:c + block][m]
            mu, var = float(np.mean(x)), float(np.var(x))
            if mu > 0 and var > 0:
                enl.append(mu * mu / var)
                cv.append(np.sqrt(var) / mu)
    if not enl:
        raise ValueError("No homogeneous sea blocks found to estimate the ENL")
    enl, cv = np.asarray(enl), np.asarray(cv)
    return float(np.median(enl[cv <= np.quantile(cv, max_cv_quantile)]))


def _box_sum(a: np.ndarray, size: int) -> np.ndarray:
    return ndi.uniform_filter(a, size=size, mode="constant") * float(size * size)


def ring_stats(img: np.ndarray, valid: np.ndarray, *, outer: int = 101, guard: int = 61
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, standard deviation and pixel count of ``img`` in the ring between a square
    ``guard`` window and a square ``outer`` window around every pixel (valid pixels only)."""
    if guard >= outer:
        raise ValueError("guard must be smaller than outer")
    x = np.where(valid, img, 0.0).astype("float64")
    w = valid.astype("float64")
    n = _box_sum(w, outer) - _box_sum(w, guard)
    s1 = _box_sum(x, outer) - _box_sum(x, guard)
    s2 = _box_sum(x * x, outer) - _box_sum(x * x, guard)
    mean = s1 / np.maximum(n, 1.0)
    var = np.maximum(s2 / np.maximum(n, 1.0) - mean ** 2, 1e-20)
    return mean, np.sqrt(var), n


# ============================================================================= CFAR
def cfar(img: np.ndarray, valid: np.ndarray, *, method: Literal["gamma", "lognormal"] = "gamma",
         pfa: float = 1e-7, enl: float = 4.4, outer: int = 101, guard: int = 61,
         min_ring_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Constant false alarm rate detector on linear intensity (e.g. sigma0 VH).

    ``method="gamma"``: cell-averaging CFAR for multilook intensity with ``enl`` looks; a pixel
    is a detection where ``I > alpha * mean_ring`` and ``alpha`` is the gamma quantile at
    ``1 - pfa``. ``method="lognormal"``: Gaussian CFAR on dB, ``I_dB > mu + t sigma`` with
    ``t`` the normal quantile (heavier tails, more robust in rough sea).

    The ``guard`` window must exceed the largest ship (400 m = 40 px at 10 m); the ring needs
    enough pixels for stable statistics. Pixels whose ring is less than ``min_ring_fraction``
    valid (coast, scene edge) are not tested. Returns ``(detections, background_mean_dB)``.
    """
    from scipy.stats import gamma as gamma_dist
    from scipy.stats import norm

    img = np.asarray(img, "float64")
    full = float(outer * outer - guard * guard)
    if method == "gamma":
        mean, _, n = ring_stats(img, valid, outer=outer, guard=guard)
        alpha = gamma_dist.isf(pfa, a=enl, scale=1.0 / enl)
        det = img > alpha * mean
        bg_db = to_db(mean)
    elif method == "lognormal":
        db = to_db(img)
        mean, sd, n = ring_stats(db, valid, outer=outer, guard=guard)
        det = db > mean + norm.isf(pfa) * sd
        bg_db = mean
    else:
        raise ValueError(f"Unknown CFAR method {method!r}")
    det &= valid & (n >= min_ring_fraction * full)
    return det, bg_db.astype("float32")


def cfar_tiled(img: np.ndarray, valid: np.ndarray, *, tile: int = 2048, **kwargs: Any
               ) -> tuple[np.ndarray, np.ndarray]:
    """:func:`cfar` over a large image in overlapping tiles (overlap = half the outer window),
    keeping only each tile's interior, so memory stays bounded."""
    outer = int(kwargs.get("outer", 101))
    pad = outer // 2 + 1
    H, W = img.shape
    det = np.zeros((H, W), bool)
    bg = np.full((H, W), np.nan, "float32")
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
            R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
            R1, C1 = min(r1 + pad, H), min(c1 + pad, W)
            d, b = cfar(img[R0:R1, C0:C1], valid[R0:R1, C0:C1], **kwargs)
            det[r0:r1, c0:c1] = d[r0 - R0:r1 - R0, c0 - C0:c1 - C0]
            bg[r0:r1, c0:c1] = b[r0 - R0:r1 - R0, c0 - C0:c1 - C0]
    return det, bg


# ============================================================================= targets
def extract_targets(det: np.ndarray, vh: np.ndarray, vv: np.ndarray | None = None, *,
                    gcps: list | None = None, background_db: np.ndarray | None = None,
                    pixel_spacing: float = 10.0, dilate: int = 1, min_px: int = 2,
                    max_px: int = 4000, min_contrast_db: float = 0.0,
                    min_mean_contrast_db: float = 0.0) -> pd.DataFrame:
    """Connected components of a CFAR mask → one row per target.

    Columns: ``row``, ``col`` (intensity-weighted centroid, window pixels), ``lon``/``lat``
    (from ``gcps``, the window-local GCPs of :func:`sentinel1.read_grd_subset`), ``n_px``,
    peak and mean VH (and VV) in dB, ``vh_vv_ratio_db``, ``contrast_db`` (peak VH minus
    background mean), ``mean_contrast_db`` (mean VH power minus background mean),
    ``length_m``/``width_m`` (PCA axes: raw, still blurred by the impulse response; calibrate
    against AIS lengths) and ``orientation_deg`` (major axis vs. azimuth).
    """
    grown = ndi.binary_dilation(det, iterations=dilate) if dilate > 0 else det
    lab, _ = ndi.label(grown, structure=np.ones((3, 3), bool))
    vh_db = to_db(vh)
    vv_db = to_db(vv) if vv is not None else None
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = (lab[sl] == i) & det[sl]
        npx = int(m.sum())
        if npx < min_px or npx > max_px:
            continue
        rr, cc = np.nonzero(m)
        w = vh[sl][m].astype("float64")
        r0 = sl[0].start + float(np.average(rr, weights=w))
        c0 = sl[1].start + float(np.average(cc, weights=w))
        if npx >= 3:
            ev, vec = np.linalg.eigh(np.cov(np.c_[rr, cc].T.astype(float)))
            major = np.sqrt(12 * max(ev[1], 0)) + 1
            minor = np.sqrt(12 * max(ev[0], 0)) + 1
            orient = float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))) % 180
        else:
            major = minor = float(npx)
            orient = np.nan
        peak = float(vh_db[sl][m].max())
        mean_db = float(to_db(np.mean(w)))
        rec = dict(row=r0, col=c0, n_px=npx, vh_peak_db=peak,
                   vh_mean_db=mean_db,
                   length_m=major * pixel_spacing, width_m=minor * pixel_spacing,
                   orientation_deg=orient)
        if vv_db is not None:
            rec["vv_peak_db"] = float(vv_db[sl][m].max())
            rec["vh_vv_ratio_db"] = rec["vh_mean_db"] - float(to_db(np.mean(vv[sl][m])))
        if background_db is not None:
            background = float(background_db[int(round(r0)), int(round(c0))])
            rec["contrast_db"] = peak - background
            rec["mean_contrast_db"] = mean_db - background
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "contrast_db" in df and min_contrast_db > 0:
        df = df[df.contrast_db >= min_contrast_db].reset_index(drop=True)
    if "mean_contrast_db" in df and min_mean_contrast_db > 0:
        df = df[df.mean_contrast_db >= min_mean_contrast_db].reset_index(drop=True)
    if gcps:
        from rasterio.transform import GCPTransformer

        lon, lat = GCPTransformer(gcps).xy(df.row.values, df.col.values, offset="center")
        df["lon"], df["lat"] = np.asarray(lon, float), np.asarray(lat, float)
    return df


# ============================================================================= azimuth ambiguities
def read_prf(annotation_xml: Any) -> list[float]:
    """Pulse repetition frequencies (Hz) listed in a GRD or SLC annotation XML
    (``downlinkInformation/prf``; one per sub-swath in an IW GRD)."""
    root = annotation_xml if isinstance(annotation_xml, ET.Element) else ET.parse(annotation_xml).getroot()
    return sorted({float(e.text) for e in root.iter("prf")})


def flag_azimuth_ambiguities(targets: pd.DataFrame, prfs: list[float], incidence_deg: np.ndarray, *,
                             wavelength: float = SPEED_OF_LIGHT / 5.405e9,
                             altitude: float = 693e3, velocity: float = 7590.0,
                             azimuth_spacing: float = 10.0, tolerance: float = 0.08,
                             col_tolerance_px: float = 15, min_ratio_db: float = 10.0) -> pd.Series:
    """True for targets that look like azimuth ghosts of a brighter target.

    A target at slant range ``R`` produces ghosts displaced along azimuth by
    ``dx = lambda R PRF / (2 V)`` (several km for Sentinel-1 IW). A target is flagged when
    another target ``min_ratio_db`` brighter lies at ``+- dx`` (within ``tolerance``) in rows
    and ``col_tolerance_px`` in columns. ``R`` is approximated as ``altitude / cos(incidence)``,
    good enough for a search window. ``incidence_deg`` is the window's incidence grid
    (:func:`sentinel1.incidence_angle_grid`).
    """
    flag = pd.Series(False, index=targets.index)
    if len(targets) < 2 or not prfs:
        return flag
    r = targets.row.values
    c = targets.col.values
    p = targets.vh_peak_db.values
    inc = incidence_deg[np.clip(np.round(r).astype(int), 0, incidence_deg.shape[0] - 1),
                        np.clip(np.round(c).astype(int), 0, incidence_deg.shape[1] - 1)]
    R = altitude / np.cos(np.radians(inc))
    for i in range(len(r)):
        for prf in prfs:
            dx = wavelength * R[i] * prf / (2 * velocity) / azimuth_spacing  # pixels
            dr = np.abs(np.abs(r - r[i]) - dx)
            near = (dr <= tolerance * dx) & (np.abs(c - c[i]) <= col_tolerance_px) & (p >= p[i] + min_ratio_db)
            if near.any():
                flag.iloc[i] = True
                break
    return flag


# ============================================================================= GRD pipeline
def detect_ships_grd(safe_dir: str | Path, bounds: tuple[float, float, float, float], land_geoms: Any, *,
                     pol: str = "vh", pfa: float = 1e-7, method: str = "gamma", enl: float | None = None,
                     outer: int = 101, guard: int = 61, buffer_m: float = 300.0,
                     min_contrast_db: float = 5.0, remove_ambiguities: bool = True) -> dict:
    """CFAR ship detection on a Sentinel-1 IW GRDH AOI, in native radar geometry (10 m).

    Uses :func:`sentinel1.read_grd_subset` for calibrated sigma0 (thermal and border noise
    removed) of VV and VH over ``bounds`` (lon/lat), so no resampling blurs the targets.
    Returns ``{"targets", "sigma0": {"vv", "vh"}, "detections", "sea", "gcps", "window",
    "enl"}``.
    """
    from ..core.io import read_safe

    idx = read_safe(safe_dir)
    data = {}
    for p in ("vv", "vh"):
        data[p] = read_grd_subset(idx["measurement"][p], idx["calibration"][p], idx["noise"][p],
                                  bounds=bounds, border_noise=True)
    gcps, window = data[pol]["gcps"], data[pol]["window"]
    vv, vh = data["vv"]["data"], data["vh"]["data"]
    img = data[pol]["data"]
    land = land_mask_for_window(gcps, img.shape, land_geoms, buffer_m=buffer_m)
    sea = ~land & np.isfinite(vv) & np.isfinite(vh) & (img > 0)
    enl = enl or estimate_enl(img, sea)
    det, bg = cfar_tiled(np.nan_to_num(img), sea, method=method, pfa=pfa, enl=enl,
                         outer=outer, guard=guard)
    targets = extract_targets(det, np.nan_to_num(vh), np.nan_to_num(vv), gcps=gcps,
                              background_db=bg, min_contrast_db=min_contrast_db)
    if remove_ambiguities and len(targets):
        ann = sorted(Path(safe_dir).glob(f"annotation/*-{pol}-*.xml"))
        if ann:
            inc = incidence_angle_grid(ann[0], img.shape, offset=window[:2])
            targets["ambiguity"] = flag_azimuth_ambiguities(targets, read_prf(ann[0]), inc)
            targets = targets[~targets.ambiguity].reset_index(drop=True)
    return {"targets": targets, "sigma0": {"vv": vv, "vh": vh}, "detections": det, "sea": sea,
            "gcps": gcps, "window": window, "enl": enl}


# ============================================================================= SLC sub-looks
def read_processing_params(annotation_xml: Any) -> dict[str, float]:
    """Range and azimuth processing bandwidth (Hz) and Hamming window coefficients from an
    IW SLC annotation XML (``imageAnnotation/processingInformation/swathProcParamsList``) --
    what :func:`range_sublooks` and :func:`azimuth_sublooks` need and
    :func:`sentinel1.read_slc_annotation` does not return."""
    root = annotation_xml if isinstance(annotation_xml, ET.Element) else ET.parse(annotation_xml).getroot()
    out = {}
    for kind in ("range", "azimuth"):
        el = next(root.iter(f"{kind}Processing"), None)
        if el is None:
            raise ValueError(f"No <{kind}Processing> in the annotation")
        out[f"{kind}_bandwidth"] = float(el.findtext("processingBandwidth"))
        out[f"{kind}_window_coefficient"] = float(el.findtext("windowCoefficient") or 0.75)
        out[f"{kind}_window_type"] = (el.findtext("windowType") or "Hamming").strip()
    return out


def _hamming(n: int, alpha: float) -> np.ndarray:
    k = np.arange(n)
    return alpha - (1 - alpha) * np.cos(2 * np.pi * k / max(n - 1, 1))


def split_spectrum(slc: np.ndarray, axis: int, bandwidth_fraction: float, *, alpha: float = 0.75,
                   gap_fraction: float = 0.05, rewindow: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two sub-look images from the lower and upper halves of an SLC's (baseband) spectrum
    along ``axis``. ``bandwidth_fraction`` = processed bandwidth / sampling rate. The
    processor's Hamming weighting (``alpha``) is divided out first so both halves carry equal
    energy; ``gap_fraction`` of the band is left unused around the centre so the halves do not
    overlap. ``rewindow`` re-applies a Hamming window to each half (lower sidelobes)."""
    s = np.moveaxis(np.asarray(slc, "complex64"), axis, -1)
    N = s.shape[-1]
    S = np.fft.fftshift(np.fft.fft(s, axis=-1), axes=-1)
    nbw = int(round(bandwidth_fraction * N))
    nbw = max(4, min(nbw, N))
    lo = N // 2 - nbw // 2
    hi = lo + nbw
    S[..., lo:hi] /= np.maximum(_hamming(nbw, alpha), 0.1)
    half, gap = nbw // 2, int(round(gap_fraction * nbw / 2))
    out = []
    for a, b in ((lo, lo + half - gap), (lo + half + gap, hi)):
        Sk = np.zeros_like(S)
        Sk[..., a:b] = S[..., a:b] * (_hamming(b - a, alpha) if rewindow else 1.0)
        sk = np.fft.ifft(np.fft.ifftshift(Sk, axes=-1), axis=-1).astype("complex64")
        out.append(np.moveaxis(sk, -1, axis))
    return out[0], out[1]


def range_sublooks(burst: np.ndarray, params: dict, range_sampling_rate: float, **kwargs: Any
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Range sub-looks of an IW SLC burst (or deburst image): the range spectrum is not
    steered, so no deramping is needed. ``params`` from :func:`read_processing_params`,
    ``range_sampling_rate`` from :func:`sentinel1.read_slc_annotation`."""
    return split_spectrum(burst, 1, params["range_bandwidth"] / range_sampling_rate,
                          alpha=params["range_window_coefficient"], **kwargs)


def azimuth_sublooks(burst: np.ndarray, ann: dict, burst_index: int, params: dict, *, col0: int = 0,
                     **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth sub-looks of one IW SLC burst. The burst is first deramped with
    :func:`sentinel1.tops_deramp_phase` (TOPS steering moves the azimuth spectrum by several kHz
    across a burst); reramping is not needed for coherence, because the same phase multiplies
    both sub-looks. Work per burst, before :func:`sentinel1.deburst`."""
    from .sentinel1 import tops_deramp_phase

    lines = np.arange(burst.shape[0], dtype=float)[:, None]
    cols = np.arange(col0, col0 + burst.shape[1], dtype=float)[None, :]
    phi = tops_deramp_phase(ann, burst_index, lines, cols)
    base = (burst * np.exp(1j * phi)).astype("complex64")
    frac = params["azimuth_bandwidth"] * ann["azimuth_time_interval"]
    return split_spectrum(base, 0, frac, alpha=params["azimuth_window_coefficient"], **kwargs)


def sublook_coherence(s1: np.ndarray, s2: np.ndarray, *, window: tuple[int, int] = (3, 9),
                      valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sub-look coherence ``|<s1 s2*>| / sqrt(<|s1|^2><|s2|^2>)`` and the internal Hermitian
    product magnitude ``|<s1 s2*>|`` over a (lines, samples) box ``window``. IW SLC pixels are
    about 14 m x 2.3 m, so a window longer in range keeps the box roughly square on the ground.
    Ships stay coherent between sub-looks (0.6-0.95); sea clutter does not (0.1-0.3)."""
    x = s1 * np.conj(s2)
    num = ndi.uniform_filter(x.real, window) + 1j * ndi.uniform_filter(x.imag, window)
    den = np.sqrt(ndi.uniform_filter(np.abs(s1) ** 2, window) * ndi.uniform_filter(np.abs(s2) ** 2, window))
    coh = np.abs(num) / np.maximum(den, 1e-20)
    ihp = np.abs(num)
    if valid is not None:
        coh = np.where(valid, coh, np.nan)
        ihp = np.where(valid, ihp, np.nan)
    return coh.astype("float32"), ihp.astype("float32")


def coherence_at_points(lat: Any, lon: Any, ann: dict, bursts: list[int], cols: tuple[int, int],
                        coherence: list[np.ndarray], *, sea_height: float = 0.0,
                        search: tuple[int, int] = (2, 8)) -> np.ndarray:
    """Maximum sub-look coherence near each lon/lat point (e.g. GRD CFAR candidates).

    Points are located in each burst with :func:`sentinel1.geo2rdr` and
    :func:`sentinel1.burst_coordinates` on the annotation's (precise) orbit -- no geocoding of
    the coherence image is needed. ``coherence`` holds one array per burst in ``bursts``, all
    cut to range samples ``cols``. ``sea_height`` is the ellipsoidal height of the sea surface
    (the local geoid height; 40 m of error moves a point about 60 m in ground range). The
    maximum within +-``search`` (lines, samples) absorbs small geolocation errors. NaN where a
    point falls in no burst.
    """
    lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
    t, r = geo2rdr(lat, lon, np.full(lat.shape, float(sea_height)), ann["orbit"])
    out = np.full(lat.shape, np.nan, "float32")
    for b, coh in zip(bursts, coherence):
        line, sample = burst_coordinates(ann, b, t, r)
        info = ann["bursts"][b]
        for k in np.flatnonzero(np.isnan(out)):
            li, si = int(round(line[k])), int(round(sample[k])) - cols[0]
            if not (info["first_valid_line"] <= li <= info["last_valid_line"]) or not (0 <= si < coh.shape[1]):
                continue
            win = coh[max(li - search[0], 0):li + search[0] + 1, max(si - search[1], 0):si + search[1] + 1]
            if np.isfinite(win).any():
                out[k] = np.nanmax(win)
    return out


# ============================================================================= AIS labels
def read_ais_dma(path: str | Path, bounds: tuple[float, float, float, float], t0: Any, *,
                 window_min: float = 15.0) -> pd.DataFrame:
    """Read a Danish Maritime Authority daily AIS file (CSV or zipped CSV) and keep class A/B
    vessel reports inside ``bounds`` (lon/lat) within ``window_min`` minutes of ``t0`` (the
    SAR acquisition time, UTC). Returns columns ``time, mmsi, lat, lon, sog, cog, heading,
    length, width, ship_type, name``. Check the column names of your files: the layout has
    changed over the years."""
    usecols = {"# Timestamp": "time", "Type of mobile": "mobile", "MMSI": "mmsi", "Latitude": "lat",
               "Longitude": "lon", "SOG": "sog", "COG": "cog", "Heading": "heading", "Length": "length",
               "Width": "width", "Ship type": "ship_type", "Name": "name"}
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False).rename(columns=usecols)
    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    t0 = pd.Timestamp(t0).tz_localize(None) if pd.Timestamp(t0).tzinfo else pd.Timestamp(t0)
    dt = pd.Timedelta(minutes=window_min)
    west, south, east, north = bounds
    keep = (df.time.between(t0 - dt, t0 + dt) & df.lon.between(west, east) & df.lat.between(south, north)
            & df.mmsi.between(100_000_000, 799_999_999))
    if "mobile" in df:
        keep &= df.mobile.isin(["Class A", "Class B"])
    return df[keep].drop(columns=[c for c in ("mobile",) if c in df]).reset_index(drop=True)


def ais_at_time(ais: pd.DataFrame, t0: Any, *, max_gap_s: float = 180.0) -> pd.DataFrame:
    """One position per vessel at the SAR time ``t0``: linear interpolation between the last
    report before and the first after ``t0`` when both are within ``max_gap_s``, else the
    nearest report within ``max_gap_s`` dead-reckoned with its SOG/COG."""
    t0 = pd.Timestamp(t0).tz_localize(None) if pd.Timestamp(t0).tzinfo else pd.Timestamp(t0)
    rows = []
    for mmsi, g in ais.sort_values("time").groupby("mmsi"):
        dt = (g.time - t0).dt.total_seconds().values
        before, after = g[dt <= 0], g[dt > 0]
        b = before.iloc[-1] if len(before) else None
        a = after.iloc[0] if len(after) else None
        tb = (b.time - t0).total_seconds() if b is not None else None
        ta = (a.time - t0).total_seconds() if a is not None else None
        if b is not None and a is not None and -tb <= max_gap_s and ta <= max_gap_s:
            w = -tb / (ta - tb)
            lat, lon = b.lat + w * (a.lat - b.lat), b.lon + w * (a.lon - b.lon)
            ref = b
        else:
            cands = [(abs(x), r) for x, r in ((tb, b), (ta, a)) if r is not None and abs(x) <= max_gap_s]
            if not cands:
                continue
            gap, ref = min(cands, key=lambda z: z[0])
            sec = (t0 - ref.time).total_seconds()
            v = (ref.sog if np.isfinite(ref.sog) else 0.0) * 0.514444
            cog = np.radians(ref.cog if np.isfinite(ref.cog) else 0.0)
            lat = ref.lat + v * sec * np.cos(cog) / EARTH_RADIUS * 180 / np.pi
            lon = ref.lon + v * sec * np.sin(cog) / (EARTH_RADIUS * np.cos(np.radians(ref.lat))) * 180 / np.pi
        rows.append(dict(mmsi=mmsi, lat=lat, lon=lon, sog=ref.sog, cog=ref.cog,
                         length=ref.get("length", np.nan), ship_type=ref.get("ship_type", None)))
    return pd.DataFrame(rows)


def azimuth_displacement_m(ais: pd.DataFrame, heading_deg: float, incidence_deg: Any, *,
                           altitude: float = 693e3, velocity: float = 7590.0) -> np.ndarray:
    """Expected along-track image displacement (m) of moving vessels: ``R * v_r / V``, with
    ``v_r`` the vessel velocity projected on the line of sight (right-looking, from SOG/COG).
    Moving ships appear offset from their true (AIS) position along the flight direction;
    :func:`match_points` with ``shift_m`` tests both signs, so the sign convention cannot bite."""
    v = np.nan_to_num(ais.sog.values.astype(float)) * 0.514444
    look_az = np.radians(heading_deg + 90)
    v_h = v * np.cos(np.radians(np.nan_to_num(ais.cog.values.astype(float))) - look_az)
    inc = np.radians(np.broadcast_to(np.asarray(incidence_deg, float), v.shape))
    return (altitude / np.cos(inc)) * v_h * np.sin(inc) / velocity


# ============================================================================= evaluation
def haversine_m(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """Pairwise great-circle distances (m), shape ``(len(lat1), len(lat2))``."""
    p1, p2 = np.radians(np.asarray(lat1, float))[:, None], np.radians(np.asarray(lat2, float))[None, :]
    dl = np.radians(np.asarray(lon2, float))[None, :] - np.radians(np.asarray(lon1, float))[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_points(pred: pd.DataFrame, truth: pd.DataFrame, *, max_dist_m: float = 200.0,
                 shift_m: np.ndarray | None = None, heading_deg: float | None = None
                 ) -> tuple[list[tuple[int, int]], np.ndarray]:
    """One-to-one matching (Hungarian) of predictions to reference points within
    ``max_dist_m``. With ``shift_m`` (from :func:`azimuth_displacement_m`) and ``heading_deg``,
    each reference point is also tried at +-``shift_m`` along the flight direction and the
    smallest distance is used. Returns ``(pairs, distance_matrix)``."""
    from scipy.optimize import linear_sum_assignment

    if len(pred) == 0 or len(truth) == 0:
        return [], np.zeros((len(pred), len(truth)))
    D = haversine_m(pred.lat.values, pred.lon.values, truth.lat.values, truth.lon.values)
    if shift_m is not None and heading_deg is not None:
        h = np.radians(heading_deg)
        for sign in (-1, 1):
            dn = sign * shift_m * np.cos(h)
            de = sign * shift_m * np.sin(h)
            lat = truth.lat.values + dn / EARTH_RADIUS * 180 / np.pi
            lon = truth.lon.values + de / (EARTH_RADIUS * np.cos(np.radians(truth.lat.values))) * 180 / np.pi
            D = np.minimum(D, haversine_m(pred.lat.values, pred.lon.values, lat, lon))
    C = np.where(D <= max_dist_m, D, 1e12)
    r, c = linear_sum_assignment(C)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if D[i, j] <= max_dist_m]
    return pairs, D


def detection_scores(n_pred: int, n_truth: int, pairs: list) -> dict[str, float]:
    """Precision, recall and F1 from a :func:`match_points` result."""
    tp = len(pairs)
    fp, fn = n_pred - tp, n_truth - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
