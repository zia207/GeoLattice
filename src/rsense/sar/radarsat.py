"""RADARSAT Constellation Mission (RCM) compact-pol and Level-1 GRD processing.

RCM's national Analysis Ready Data (ARD) product (NRCan/EODMS, distributed via a STAC
catalogue and the AWS Open Data Registry -- see :func:`geolattice.rsense.get_images.get_rcm_ard`
for access) ships compact polarimetry: a right-circular transmit, linear (RH/RV) receive. Its
raster assets are RR and RL backscatter (the two circular receive channels) and RRRL*, their
complex correlation, stored as a two-band (real, imaginary) GeoTIFF -- not the RH/RV basis
directly. This module converts that native RR/RL/RRRL* product into Stokes parameters, the
RH/RV compact-pol covariance matrix (directly usable with :mod:`geolattice.rsense.sar.polsar`'s
generic dual-pol H/alpha tools, since those operate on any 2x2 Hermitian C2 regardless of
basis) and the m-chi (surface/volume/double-bounce) decomposition, following the formulas and
conventions of NRCan/EODMS's own reference notebooks
(https://github.com/eodms-sgdot/radarsat-notebooks, ``mchi_decomposition.ipynb`` and
``rrrl_to_rhrv.ipynb``).

RADARSAT-2 carries no compact-pol hardware, so there is no native RR/RL product from it --
:func:`simulate_compact_pol` instead derives compact-pol-equivalent RH/RV data from a RS2
quad-pol scattering matrix (see its docstring), which :func:`rhrv_to_stokes` then converts into
the same Stokes-parameter form :func:`rr_rl_to_stokes` produces, so :func:`compact_pol_mchi`
applies unchanged to either a real RCM acquisition or a RADARSAT-2 simulation.

It also reads the calibration ``lutSigma``/``lutBeta``/``lutGamma`` and ``noiseLevels`` XML
files that ship beside a raw RCM Level-1 product's imagery (``metadata/calibration/``) and
applies them: ``sigma0 (or beta0, gamma0) = (DN^2 + offset) / gain``, with the NESZ noise
floor optionally subtracted in that same calibrated domain. Multilooking, speckle filtering,
terrain correction and dB conversion for the calibrated result are the existing, sensor-generic
functions in :mod:`geolattice.rsense.sar.sentinel1` and :mod:`geolattice.rsense.sar.speckle` -- RCM's gain
and noise LUTs are the only mission-specific piece.

For a whole Level-1 GRD product folder, :func:`read_rcm_grd_subset` does the full chain for one
polarisation and a lon/lat box: it finds the imagery and calibration files
(:func:`find_rcm_files`), reads only the window covering the box, calibrates it, subtracts the
noise floor, attaches the window's ground control points (from the GeoTIFF, or from
``product.xml``'s tie-point grid via :func:`read_rcm_tie_points`) and the ellipsoid incidence
angle (:func:`rcm_incidence_angle`). The result has the same ``{"data", "gcps", "gcp_crs"}``
shape as :func:`geolattice.rsense.sar.sentinel1.read_grd_subset`, so
:func:`geolattice.rsense.sar.sentinel1.create_stack` puts several RCM dates (even from different
beams or orbits) on one map grid, and :func:`geolattice.rsense.sar.ship.land_mask_for_window` masks
land in radar geometry. :func:`fit_incidence_slope` and :func:`normalize_incidence` then bring
scenes acquired at different incidence angles to a common reference angle.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..core._utils import nan_box_mean, run_array
from ..core._utils import where as _where

_RCM_NS = {"r": "rcmGsProductSchema"}


def simulate_compact_pol(s_hh: Any, s_hv: Any, s_vh: Any, s_vv: Any, size: int = 7) -> dict[str, Any]:
    """Simulate a right-circular-transmit, linear-receive (CTLR) compact-pol acquisition from
    quad-pol SLC channels -- what RCM's (or RISAT-1's) actual compact-pol hardware would have
    measured over the same scene, had it flown instead of a quad-pol sensor. RADARSAT-2 has no
    compact-pol hardware mode, so this is the only way to get compact-pol-equivalent data from
    it, and is the standard technique used to evaluate compact-pol algorithms against quad-pol
    ground truth on the same pixels (e.g. Charbonneau et al. 2010 used exactly this on
    RADARSAT-2 data to validate what became RCM's compact-pol mode).

    Follows Guo et al. (2018, *Sensors* 18(7):2271, "Inversion of Rice Biophysical Parameters
    Using Simulated Compact Polarimetric SAR C-band Data", Eqs. 2-5), who built the same
    simulation from RADARSAT-2 quad-pol data: the transmitted Jones vector for right-hand
    circular polarisation is ``R = (1/2)[1, -j]^T``; applying the Sinclair matrix gives
    ``E_RH = (S_hh - j*S_x) / 2``, ``E_RV = (S_x - j*S_vv) / 2``, with the reciprocity average
    ``S_x = (S_hv + S_vh) / 2``.

    Returns a locally-averaged ``{"c11", "c22", "c12_re", "c12_im"}`` RH/RV covariance matrix
    -- the same naming :func:`rrrl_to_rhrv` and :func:`geolattice.rsense.sar.polsar.covariance_c2`
    use, so it plugs directly into ``h_alpha_dual`` (basis-independent) or, via
    :func:`rhrv_to_stokes`, into :func:`compact_pol_mchi`.
    """
    def avg(a):
        return run_array(a, lambda x, xp: nan_box_mean(x, size, xp))

    s_x = (s_hv + s_vh) / 2
    e_rh = (s_hh - 1j * s_x) / 2
    e_rv = (s_x - 1j * s_vv) / 2
    c12 = e_rh * np.conj(e_rv)
    out = {"c11": avg(np.abs(e_rh) ** 2), "c22": avg(np.abs(e_rv) ** 2),
           "c12_re": avg(np.real(c12)), "c12_im": avg(np.imag(c12))}
    if isinstance(s_hh, xr.DataArray):
        return xr.Dataset(out)
    return out


def rhrv_to_stokes(c11: Any, c22: Any, c12_re: Any, c12_im: Any) -> xr.Dataset | dict:
    """Stokes parameters from an RH/RV compact-pol covariance matrix (Raney 2007), the inverse
    of :func:`rrrl_to_rhrv`'s ``c11 = (s0+s1)/2`` etc. mapping: ``s0 = <|RH|^2> + <|RV|^2>``,
    ``s1 = <|RH|^2> - <|RV|^2>``, ``s2 = 2*Re<RH RV*>``, ``s3 = 2*Im<RH RV*>``. Feeds directly
    into :func:`compact_pol_mchi`, in the same ``s0..s3`` convention :func:`rr_rl_to_stokes`
    produces from RCM's native circular-receive product -- both describe the same
    right-circular-transmit Stokes vector, just measured (or, from :func:`simulate_compact_pol`,
    simulated) in a different receive basis.
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


def split_rrrl(rrrl: Any) -> tuple[Any, Any]:
    """Real and imaginary bands of an RCM-ARD ``rrrl`` asset (the RR/RL* complex correlation).

    The asset is a two-band GeoTIFF, band 0 real and band 1 imaginary. Accepts a ``band``-
    dimensioned ``xr.DataArray`` (as read by ``rioxarray``) or a plain array indexable as
    ``rrrl[0]``/``rrrl[1]``.
    """
    if isinstance(rrrl, xr.DataArray) and "band" in rrrl.dims:
        return rrrl.isel(band=0), rrrl.isel(band=1)
    return rrrl[0], rrrl[1]


def rr_rl_to_stokes(rr: Any, rl: Any, rrrl_re: Any, rrrl_im: Any) -> xr.Dataset | dict:
    """RCM-ARD's circular-basis Stokes parameters from RR, RL and RRRL*.

    ``s0`` = total power, ``s1`` = RH/RV-style power difference (via the RR/RL cross term's
    imaginary part), ``s2``/``s3`` the remaining two Stokes components, and
    ``degree_of_polarization`` (m). These are *not* the same formulas as
    :func:`geolattice.rsense.sar.polsar.stokes_parameters` (which treats a linear co/cross pair by
    analogy): RCM's hardware compact-pol is a true circular-transmit system, so the basis
    rotation from (RR, RL, RRRL*) to Stokes space is fixed by the instrument, following
    NRCan/EODMS's own reference implementation.
    """
    s0 = rl + rr
    s1 = 2 * rrrl_im
    s2 = 2 * rrrl_re
    s3 = rl - rr
    with np.errstate(divide="ignore", invalid="ignore"):
        dop = np.clip(np.sqrt(s1**2 + s2**2 + s3**2) / s0, 0, 1)
    out = {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "degree_of_polarization": dop}
    if isinstance(rr, xr.DataArray):
        return xr.Dataset(out)
    return out


def rrrl_to_rhrv(rr: Any, rl: Any, rrrl_re: Any, rrrl_im: Any) -> xr.Dataset | dict:
    """Losslessly convert RCM-ARD's circular (RR, RL, RRRL*) product to the linear RH/RV
    compact-pol covariance matrix (Charbonneau et al. 2010; NRCan/EODMS ``rrrl_to_rhrv.ipynb``).

    Returns ``{"c11", "c22", "c12_re", "c12_im"}`` -- ``c11 = <|RH|^2>``, ``c22 = <|RV|^2>``,
    ``c12 = <RH RV*>`` -- the same C2 naming :func:`geolattice.rsense.sar.polsar.covariance_c2`
    uses, so the result can be passed straight into any dual-pol function there
    (``h_alpha_dual``, ``stokes_parameters``, the m-delta/m-chi/eigen-model decompositions):
    Cloude's H/alpha eigendecomposition is basis-independent and applies to any 2x2 Hermitian
    covariance matrix, linear or circular.
    """
    st = rr_rl_to_stokes(rr, rl, rrrl_re, rrrl_im)
    s0, s1, s2, s3 = st["s0"], st["s1"], st["s2"], st["s3"]
    out = {"c11": (s0 + s1) / 2, "c22": (s0 - s1) / 2, "c12_re": s2 / 2, "c12_im": s3 / 2}
    if isinstance(rr, xr.DataArray):
        return xr.Dataset(out)
    return out


def compact_pol_mchi(s0: Any, s1: Any, s2: Any, s3: Any) -> xr.Dataset | dict:
    """m-chi decomposition of RCM compact-pol data into surface, volume and double-bounce power.

    Takes Stokes parameters (:func:`rr_rl_to_stokes`) and follows NRCan/EODMS's own
    ``mchi_decomposition.ipynb`` exactly: ``chi = s3 / (m s0)`` (the sine of twice the
    ellipticity angle, used directly rather than via ``arcsin``), ``ps = s0 m (1+chi)/2``
    (surface/odd-bounce), ``pd = s0 m (1-chi)/2`` (double-bounce) and ``pv = s0 (1-m)``
    (volume). ``ps + pd + pv == s0`` by construction.

    This sign convention is specific to RCM's real right-circular-transmit hardware and is
    *not* the same as :func:`geolattice.rsense.sar.polsar.m_chi_decomposition`'s generic linear-pol
    analogy (whose surface/double-bounce assignment runs the other way): RCM's physical
    transmit helicity fixes which sign of ``chi`` is odd- vs even-bounce, and this function
    matches the authoritative reference rather than the generic one.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.clip(np.sqrt(s1**2 + s2**2 + s3**2) / s0, 0, 1)
        ratio = s3 / (m * s0)
        chi = _where(np.isfinite(ratio), ratio, 0)
    ps = s0 * m * (1 + chi) / 2
    pd = s0 * m * (1 - chi) / 2
    pv = s0 * (1 - m)
    out = {"ps": ps, "pd": pd, "pv": pv}
    if isinstance(s0, xr.DataArray):
        return xr.Dataset(out)
    return out


def gamma0_to_sigma0(gamma0: Any, ratio: Any) -> Any:
    """RCM-ARD's own terrain-corrected ``sigma0 = gamma0 * ratio``.

    ``ratio`` is the product's ``gamma_to_sigma_ratio`` asset: a per-pixel, DEM-derived
    gamma0-to-sigma0 conversion factor, precomputed from the local scattering area --
    not the ellipsoid-only ``cos(incidence)`` approximation
    :func:`geolattice.rsense.sar.sentinel1.sigma0_to_gamma0` uses in the other direction.
    """
    return gamma0 * ratio


# ----------------------------------------------------------------------------- Level-1 GRD calibration
def read_rcm_lut(xml_path: Any) -> dict[str, Any]:
    """Parse an RCM Level-1 ``lutSigma``/``lutBeta``/``lutGamma`` calibration XML
    (``metadata/calibration/lutSigma_<pol>.xml`` etc.).

    The gain is a 1-D curve along the range (column) axis only -- constant along every line --
    sampled at ``pixel_first + step * i`` for ``i`` in ``range(n)``. Returns
    ``{"pixel_first", "step", "n", "offset", "gains"}``.
    """
    root = ET.parse(xml_path).getroot()

    def f(tag, cast=float):
        return cast(root.find(f"r:{tag}", _RCM_NS).text)

    gains = np.array(root.find("r:gains", _RCM_NS).text.split(), dtype="float64")
    return {"pixel_first": f("pixelFirstLutValue", int), "step": f("stepSize", int),
            "n": f("numberOfValues", int), "offset": f("offset"), "gains": gains}


def _range_curve(pixel_first: int, step: int, values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Sample columns and values of an RCM range curve, in ascending column order.

    RCM products whose ``pixelTimeOrdering`` is ``Decreasing`` (far range at column 0, common
    for ScanSAR GRD) store their LUT, noise and incidence curves from the last column backwards
    with a negative ``stepSize``; ``np.interp`` needs increasing sample positions."""
    values = np.asarray(values, dtype="float64")
    xp = pixel_first + step * np.arange(values.size)
    order = np.argsort(xp, kind="stable")
    return xp[order], values[order]


def interpolate_rcm_lut(lut: dict[str, Any], width: int, *, col0: int = 0) -> np.ndarray:
    """Interpolate a sparse RCM range-only gain LUT (:func:`read_rcm_lut`) onto ``width``
    columns, holding the edge value beyond the LUT's own sampled range. ``col0`` is the scene
    column of the first output column, for a window cut out of the full image. Negative
    ``stepSize`` (curves stored from the last column backwards) is handled."""
    xp, fp = _range_curve(lut["pixel_first"], lut["step"], lut["gains"][: lut["n"]])
    return np.interp(col0 + np.arange(width), xp, fp)


def read_rcm_noise_levels(xml_path: Any, cal_type: str = "Sigma Nought") -> list[dict[str, Any]]:
    """Parse an RCM Level-1 ``noiseLevels_<pol>.xml`` file's per-beam NESZ segments.

    A ScanSAR product stitches several sub-beams (one ``perBeamReferenceNoiseLevel`` segment
    per beam, each covering its own range of columns); a Stripmap product has just one. Each
    segment is stored for every calibration type the product supports (``"Sigma Nought"``,
    ``"Beta Nought"`` and ``"Gamma"``, written ``"Gamma Nought"`` in some products); pass the one
    matching the gain LUT used for calibration (matching ignores case and a trailing
    ``"Nought"``). When a product has no per-beam segments for that type, its whole-swath
    ``referenceNoiseLevel`` is returned as a single segment (``beam="all"``). Returns a list of
    ``{"beam", "pixel_first", "step", "n", "values_db"}``.
    """
    def norm(t):
        return (t or "").strip().lower().replace(" nought", "").replace("nought", "").strip()

    root = ET.parse(xml_path).getroot()
    want = norm(cal_type)

    def parse(tag):
        out = []
        for seg in root.findall(f"r:{tag}", _RCM_NS):
            if norm(seg.findtext("r:sarCalibrationType", namespaces=_RCM_NS)) != want:
                continue
            values = np.array(seg.find("r:noiseLevelValues", _RCM_NS).text.split(), dtype="float64")
            out.append({
                "beam": seg.findtext("r:beam", default="all", namespaces=_RCM_NS),
                "pixel_first": int(seg.find("r:pixelFirstNoiseValue", _RCM_NS).text),
                "step": int(seg.find("r:stepSize", _RCM_NS).text),
                "n": int(seg.find("r:numberOfValues", _RCM_NS).text),
                "values_db": values,
            })
        return out

    return parse("perBeamReferenceNoiseLevel") or parse("referenceNoiseLevel")


def interpolate_rcm_noise(segments: list[dict[str, Any]], width: int, *, col0: int = 0) -> np.ndarray:
    """Stitch RCM per-beam noise-level segments (:func:`read_rcm_noise_levels`) onto ``width``
    columns (dB). Each segment is interpolated within its own column range; any gap between
    segments (a ScanSAR sub-beam seam) or before/after all of them is filled linearly from the
    nearest segment end points (edge values held). ``col0`` is the scene column of the first
    output column, for a window cut out of the full image."""
    out = np.full(width, np.nan)
    cols = col0 + np.arange(width)
    if not segments:
        return out
    curves = [_range_curve(s["pixel_first"], s["step"], s["values_db"][: s["n"]]) for s in segments]
    ends_x, ends_v = [], []
    for xp, fp in sorted(curves, key=lambda c: c[0][0]):    # later (higher-column) beams win overlaps
        sel = (cols >= xp[0]) & (cols <= xp[-1])
        out[sel] = np.interp(cols[sel], xp, fp)
        ends_x += [xp[0], xp[-1]]
        ends_v += [fp[0], fp[-1]]
    gap = np.isnan(out)
    if np.any(gap):
        order = np.argsort(ends_x, kind="stable")
        out[gap] = np.interp(cols[gap], np.asarray(ends_x)[order], np.asarray(ends_v)[order])
    return out


def calibrate_rcm(dn: Any, gain: Any, *, offset: float = 0.0, noise_db: Any | None = None,
                  floor: float = 1e-6) -> np.ndarray:
    """RCM Level-1 GRD radiometric calibration: ``(DN^2 + offset) / gain``.

    ``gain`` is a range-only (per-column) curve from :func:`interpolate_rcm_lut` -- pass
    ``lutSigma``/``lutBeta``/``lutGamma``'s gain for sigma0/beta0/gamma0 respectively; it
    broadcasts against a ``(lines, samples)`` ``dn``. RCM's LUT gain is used unsquared (its
    values, ~1e7 to ~1e9 depending on beam mode, already play the role of a squared Sentinel-1-style amplitude gain --
    confirmed against a real product: dividing by the squared gain put sigma0 five to six
    orders of magnitude below any plausible backscatter, dividing by the gain directly gives
    the expected -10 to 0 dB range for ordinary land). ``noise_db``
    (:func:`interpolate_rcm_noise`, read with the matching ``cal_type``) is subtracted
    directly in the calibrated linear domain: NESZ (noise-equivalent sigma/beta/gamma zero) is
    itself already expressed in that domain -- unlike :func:`geolattice.rsense.sar.sentinel1.calibrate`,
    whose Sentinel-1 ``noise_lut`` is raw DN-squared and is subtracted before dividing by the
    gain.
    """
    dn = np.asarray(dn)
    out = (dn.astype("float64") ** 2 + offset) / np.asarray(gain, dtype="float64")
    if noise_db is not None:
        out = out - 10 ** (np.asarray(noise_db, dtype="float64") / 10)
    out[dn == 0] = np.nan
    return np.maximum(out, floor, where=np.isfinite(out), out=out).astype("float32")


# ----------------------------------------------------------------------------- Level-1 GRD product reader
_CAL_TYPES = {"sigma": ("lutSigma", "Sigma Nought"), "beta": ("lutBeta", "Beta Nought"),
              "gamma": ("lutGamma", "Gamma")}


def _first(paths: list[Path]) -> Path | None:
    return min(paths) if paths else None


def find_rcm_files(product_dir: Any, pol: str = "HH") -> dict[str, Path | None]:
    """Locate the files of one polarisation inside an unzipped RCM Level-1 product folder.

    RCM products keep one GeoTIFF per polarisation in ``imagery/`` (``*_HH.tif``, ``*_HV.tif``
    ...) and the per-polarisation calibration files in ``metadata/calibration/``
    (``lutSigma_HH.xml``, ``lutBeta_HH.xml``, ``lutGamma_HH.xml``, ``noiseLevels_HH.xml``),
    plus a shared ``incidenceAngles.xml`` and the main ``metadata/product.xml``. Older products
    that store unsuffixed ``lutSigma.xml``/``noiseLevels.xml`` are found as a fallback.

    Returns ``{"image", "product_xml", "lut_sigma", "lut_beta", "lut_gamma", "noise",
    "incidence"}``; a file that is not present is ``None``. Raises ``FileNotFoundError`` when
    the image for ``pol`` is missing.
    """
    root = Path(product_dir)
    pol = pol.upper()
    imagery = root / "imagery"
    image = _first([p for p in imagery.glob("*.tif*") if p.stem.upper().endswith(f"_{pol}")]) \
        if imagery.is_dir() else None
    if image is None:
        raise FileNotFoundError(f"no {pol} GeoTIFF under {imagery}")
    meta = root / "metadata"
    cal = meta / "calibration"

    def cal_file(stem: str) -> Path | None:
        for name in (f"{stem}_{pol}.xml", f"{stem}.xml"):
            if (cal / name).is_file():
                return cal / name
        return None

    product_xml = meta / "product.xml"
    return {"image": image, "product_xml": product_xml if product_xml.is_file() else None,
            "lut_sigma": cal_file("lutSigma"), "lut_beta": cal_file("lutBeta"),
            "lut_gamma": cal_file("lutGamma"), "noise": cal_file("noiseLevels"),
            "incidence": _first(list(cal.glob("incidenceAngles*.xml")) + list(meta.glob("incidenceAngles*.xml")))}


def read_rcm_metadata(product_xml: Any) -> dict[str, Any]:
    """Scene facts from an RCM ``product.xml`` that a workflow should record before processing:
    ``satellite``, ``beam_mode`` (e.g. ``"SCLNA"``), ``product_type`` (``"GRD"``, ``"SLC"``...),
    ``acquisition_start`` (``np.datetime64``), ``pass`` (``"Ascending"``/``"Descending"``),
    ``polarizations``, ``lines``, ``samples``, ``pixel_spacing``/``line_spacing`` (m),
    ``incidence_near``/``incidence_far`` (degrees) and ``pixel_time_ordering``/
    ``line_time_ordering`` (``"Increasing"``/``"Decreasing"``). A tag a product does not carry is
    returned as ``None`` rather than raising, so the same reader works across beam modes and
    product types.
    """
    root = ET.parse(product_xml).getroot()

    def text(tag):
        e = root.find(f".//r:{tag}", _RCM_NS)
        return e.text.strip() if e is not None and e.text else None

    def num(tag, cast=float):
        t = text(tag)
        return cast(t) if t is not None else None

    t0 = text("rawDataStartTime") or text("zeroDopplerTimeFirstLine")
    return {
        "satellite": text("satellite"), "beam_mode": text("beamModeMnemonic"),
        "product_type": text("productType"),
        "acquisition_start": np.datetime64(t0.rstrip("Z"), "ns") if t0 else None,
        "pass": text("passDirection"), "polarizations": text("polarizations"),
        "lines": num("numLines", int), "samples": num("samplesPerLine", int),
        "pixel_spacing": num("sampledPixelSpacing"), "line_spacing": num("sampledLineSpacing"),
        "incidence_near": num("incAngNearRng"), "incidence_far": num("incAngFarRng"),
        "pixel_time_ordering": text("pixelTimeOrdering"), "line_time_ordering": text("lineTimeOrdering"),
    }


def read_rcm_tie_points(product_xml: Any) -> list:
    """Ground control points from ``product.xml``'s geolocation grid (``imageTiePoint``: image
    line/pixel against geodetic latitude/longitude/height) as ``rasterio`` GCPs in EPSG:4326 --
    the fallback :func:`read_rcm_grd_subset` uses when the imagery GeoTIFF carries no GCPs of
    its own."""
    from rasterio.control import GroundControlPoint

    root = ET.parse(product_xml).getroot()
    gcps = []
    for tp in root.findall(".//r:imageTiePoint", _RCM_NS):
        img = tp.find("r:imageCoordinate", _RCM_NS)
        geo = tp.find("r:geodeticCoordinate", _RCM_NS)
        if img is None or geo is None:
            continue

        def v(el, tag, default=0.0):
            e = el.find(f"r:{tag}", _RCM_NS)
            return float(e.text) if e is not None else default

        gcps.append(GroundControlPoint(row=v(img, "line"), col=v(img, "pixel"),
                                       x=v(geo, "longitude"), y=v(geo, "latitude"), z=v(geo, "height")))
    if not gcps:
        raise ValueError(f"No <imageTiePoint> found in {product_xml}")
    return gcps


def read_rcm_incidence(xml_path: Any) -> dict[str, Any]:
    """Parse RCM's ``incidenceAngles.xml``: a range-only (per-column) ellipsoid incidence angle
    curve sampled at ``pixel_first + step * i``, in the same layout as the gain LUTs (``step``
    may be negative). Real products write one ``<angles>`` element per value; a single element
    holding a space-separated list is read too. Returns ``{"pixel_first", "step", "n",
    "angles"}`` (degrees)."""
    root = ET.parse(xml_path).getroot()

    def f(tag, cast=float):
        e = root.find(f".//r:{tag}", _RCM_NS)
        if e is None:
            raise ValueError(f"<{tag}> not found in {xml_path}")
        return cast(e.text)

    texts = [e.text for e in root.findall(".//r:angles", _RCM_NS) if e.text]
    if not texts:
        raise ValueError(f"<angles> not found in {xml_path}")
    angles = np.array(" ".join(texts).split(), dtype="float64")
    return {"pixel_first": f("pixelFirstAnglesValue", int), "step": f("stepSize", int),
            "n": f("numberOfValues", int), "angles": angles}


def rcm_incidence_angle(source: Any, width: int, *, col0: int = 0) -> np.ndarray:
    """Per-column ellipsoid incidence angle (degrees) for ``width`` columns starting at scene
    column ``col0``.

    ``source`` is either an ``incidenceAngles.xml`` path / :func:`read_rcm_incidence` dict
    (interpolated, edge values held), or a :func:`read_rcm_metadata` dict, in which case the
    angle is taken as linear between ``incidence_near`` and ``incidence_far`` across the
    scene's ``samples`` columns (reversed when ``pixel_time_ordering`` is ``"Decreasing"``,
    i.e. far range at column 0). The linear fallback ignores Earth curvature and is off by up to
    about 2 degrees across a 350 km ScanSAR swath (1.3 to 1.8 degrees on real SCLN products), so
    prefer ``incidenceAngles.xml`` whenever the product has it.
    """
    cols = col0 + np.arange(width)
    if isinstance(source, dict) and "incidence_near" in source:
        near, far, n = source["incidence_near"], source["incidence_far"], source["samples"]
        if near is None or far is None or not n:
            raise ValueError("metadata has no incidence_near/incidence_far/samples")
        if (source.get("pixel_time_ordering") or "").lower() == "decreasing":
            near, far = far, near
        return np.interp(cols, [0, n - 1], [near, far])
    inc = source if isinstance(source, dict) else read_rcm_incidence(source)
    xp, fp = _range_curve(inc["pixel_first"], inc["step"], inc["angles"][: inc["n"]])
    return np.interp(cols, xp, fp)


def _window_from_gcps(gcps: list, bounds: tuple[float, float, float, float],
                      shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """(row0, col0, row1, col1) covering a lon/lat box: corners mapped through an affine fit
    to all GCPs, widened by the image extent of every GCP inside the box (a single affine over
    a 350 km ScanSAR swath can be off by a few km locally), plus ``margin`` pixels."""
    from rasterio.transform import from_gcps

    west, south, east, north = bounds
    inverse = ~from_gcps(gcps)
    cols, rows = zip(*[inverse * (x, y) for x, y in
                       [(west, south), (east, south), (west, north), (east, north)]])
    rows, cols = list(rows), list(cols)
    inside = [g for g in gcps if west <= g.x <= east and south <= g.y <= north]
    rows += [g.row for g in inside]
    cols += [g.col for g in inside]
    h, w = shape
    row0 = max(0, int(np.floor(min(rows))) - margin)
    col0 = max(0, int(np.floor(min(cols))) - margin)
    row1 = min(h, int(np.ceil(max(rows))) + margin)
    col1 = min(w, int(np.ceil(max(cols))) + margin)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("bounds do not intersect the scene")
    return row0, col0, row1, col1


def read_rcm_grd_subset(product_dir: Any, pol: str = "HH", *, bounds: tuple[float, float, float, float],
                        kind: str = "sigma", noise: bool = True, margin: int = 100,
                        incidence: bool = True) -> dict[str, Any]:
    """Calibrated, noise-corrected backscatter for a lon/lat box cut out of one polarisation of
    an RCM Level-1 GRD product folder (ScanSAR or Stripmap), still in radar geometry.

    Only the window covering ``bounds`` (west, south, east, north, degrees) plus ``margin``
    pixels is read. The window is calibrated with the product's own gain LUT for ``kind``
    (``"sigma"``, ``"beta"`` or ``"gamma"``) and, with ``noise=True``, the matching NESZ curve
    from ``noiseLevels_<pol>.xml`` is subtracted (:func:`calibrate_rcm`); a product without a
    noise file is calibrated without noise removal. Ground control points come from the
    GeoTIFF when it has them, otherwise from ``product.xml``'s tie-point grid; only those around
    the window are kept and shifted to window coordinates, so geocoding fits the local geometry.

    Returns ``{"data" (float32, linear), "gcps", "gcp_crs", "window": (row0, col0, height,
    width), "incidence" (float32 degrees, same shape as ``data``, or ``None``), "pol",
    "meta"}`` (``meta`` from :func:`read_rcm_metadata`, or ``{}`` without a ``product.xml``).
    The dict plugs straight into :func:`geolattice.rsense.sar.sentinel1.create_stack` (pass
    ``{"data": out["incidence"], "gcps": out["gcps"]}`` to geocode the incidence angle onto the
    same grid) and :func:`geolattice.rsense.sar.ship.land_mask_for_window`. Speckle filtering is
    best done on ``data`` before geocoding, while pixels are still independent looks.
    """
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.windows import Window

    from .sentinel1 import _gcps_near

    kind = kind.lower()
    if kind not in _CAL_TYPES:
        raise ValueError(f"kind must be one of {sorted(_CAL_TYPES)}")
    files = find_rcm_files(product_dir, pol)
    lut_key = "lut_" + kind
    if files[lut_key] is None:
        raise FileNotFoundError(f"{_CAL_TYPES[kind][0]} LUT for {pol} not found in {product_dir}")
    meta = read_rcm_metadata(files["product_xml"]) if files["product_xml"] else {}

    with warnings.catch_warnings():
        # GeoTIFFs without GCPs are expected here (tie points come from product.xml instead)
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        src = rasterio.open(str(files["image"]))
    with src:
        gcps, gcp_crs = src.gcps
        if not gcps:
            if files["product_xml"] is None:
                raise ValueError(f"{files['image']} has no GCPs and no product.xml was found")
            gcps, gcp_crs = read_rcm_tie_points(files["product_xml"]), "EPSG:4326"
        row0, col0, row1, col1 = _window_from_gcps(gcps, bounds, (src.height, src.width), margin)
        dn = src.read(1, window=Window(col0, row0, col1 - col0, row1 - row0))

    width = dn.shape[1]
    lut = read_rcm_lut(files[lut_key])
    gain = interpolate_rcm_lut(lut, width, col0=col0)
    noise_db = None
    if noise and files["noise"] is not None:
        segs = read_rcm_noise_levels(files["noise"], cal_type=_CAL_TYPES[kind][1])
        if segs:
            noise_db = interpolate_rcm_noise(segs, width, col0=col0)
    data = calibrate_rcm(dn, gain, offset=lut["offset"], noise_db=noise_db)

    inc = None
    if incidence:
        src_inc = files["incidence"] or (meta if meta.get("incidence_near") is not None else None)
        if src_inc is not None:
            inc = np.broadcast_to(rcm_incidence_angle(src_inc, width, col0=col0), dn.shape).astype("float32")

    near = _gcps_near(gcps, row0, col0, row1, col1)
    local = [GroundControlPoint(row=g.row - row0, col=g.col - col0, x=g.x, y=g.y, z=g.z) for g in near]
    return {"data": data, "gcps": local, "gcp_crs": gcp_crs or "EPSG:4326",
            "window": (row0, col0, dn.shape[0], dn.shape[1]), "incidence": inc,
            "pol": pol.upper(), "meta": meta}


# ----------------------------------------------------------------------------- incidence-angle normalisation
def fit_incidence_slope(sigma0_db: Any, incidence_deg: Any, mask: Any | None = None, *,
                        min_count: int = 100) -> dict[str, float]:
    """Least-squares slope of backscatter (dB) against incidence angle (degrees) over the
    pixels in ``mask`` (default: every finite pixel) -- pick a large area of one surface type,
    e.g. uniform first-year ice, for the ``slope`` that :func:`normalize_incidence` needs.

    Returns ``{"slope" (dB/degree), "intercept" (dB at 0 degrees), "r2", "n"}``. Raises
    ``ValueError`` with fewer than ``min_count`` usable pixels or no spread in incidence angle.
    """
    s = np.asarray(getattr(sigma0_db, "values", sigma0_db), dtype="float64").ravel()
    t = np.asarray(getattr(incidence_deg, "values", incidence_deg), dtype="float64")
    t = np.broadcast_to(t, np.shape(getattr(sigma0_db, "values", sigma0_db))).ravel()
    ok = np.isfinite(s) & np.isfinite(t)
    if mask is not None:
        ok &= np.asarray(getattr(mask, "values", mask), bool).ravel()
    s, t = s[ok], t[ok]
    if s.size < min_count:
        raise ValueError(f"only {s.size} usable pixels (min_count={min_count})")
    if np.ptp(t) < 1e-6:
        raise ValueError("incidence angle has no spread over the selected pixels")
    slope, intercept = np.polyfit(t, s, 1)
    resid = s - (slope * t + intercept)
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(s.size)}


def normalize_incidence(sigma0_db: Any, incidence_deg: Any, slope: float, ref_deg: float = 35.0) -> Any:
    """Bring backscatter (dB) to a reference incidence angle with a linear slope:
    ``sigma0_norm = sigma0 - slope * (theta - ref_deg)``.

    ``slope`` is in dB/degree and negative for most surfaces (HH over first-year sea ice is
    typically about -0.2 to -0.25); estimate it with :func:`fit_incidence_slope`. Needed before
    comparing scenes from different beams, or the near and far range of one wide swath.
    Works on NumPy arrays and ``xr.DataArray`` (coordinates and attributes kept).
    """
    out = sigma0_db - slope * (incidence_deg - ref_deg)
    if isinstance(out, xr.DataArray):
        out.attrs.update({"incidence_reference_deg": ref_deg, "incidence_slope_db_per_deg": slope})
    return out


# ----------------------------------------------------------------------------- Stripmap SLC / InSAR
def read_rcm_annotation(xml_path: Any) -> dict[str, Any]:
    """Parse what InSAR coregistration needs from an RCM Level-1 **Stripmap** SLC
    ``product.xml`` (one continuous image, not TOPS bursts -- beams such as ``3M``, ``5M``,
    ``16M``, ``QP`` or the quad-pol ``FQ``/``SQ`` family; a ScanSAR beam, e.g. ``SC30MCP*``,
    packs several sub-beams the way this reader does not handle).

    Returns an annotation dict in the same shape
    :func:`geolattice.rsense.sar.sentinel1.read_slc_annotation` produces for one Sentinel-1 burst --
    ``wavelength``, ``range_sampling_rate``, ``slant_range_time``, ``azimuth_time_interval``,
    ``lines_per_burst``, a single-entry ``bursts`` list covering the whole image, and ``orbit``
    (RCM's own precise state vectors; no separate orbit-file download is needed, unlike
    Sentinel-1) -- so :func:`geolattice.rsense.sar.sentinel1.back_geocode` and
    :func:`geolattice.rsense.sar.sentinel1.resample_slc` (called with ``ann=None``, since Stripmap
    has no TOPS burst steering to deramp) work unchanged for coregistration, and the rest of
    the InSAR chain (``geolattice.rsense.insar``) is sensor-agnostic already.
    """
    from .sentinel1 import SPEED_OF_LIGHT

    root = ET.parse(xml_path).getroot()

    def text(path):
        e = root.find(path, _RCM_NS)
        return e.text if e is not None else None

    def num(path):
        return float(text(path))

    def rcm_time(path):
        return np.datetime64(text(path).rstrip("Z"), "ns")

    freq = num(".//r:radarCenterFrequency")
    n_lines = int(num(".//r:numLines"))
    n_samples = int(num(".//r:samplesPerLine"))
    t_first = rcm_time(".//r:zeroDopplerTimeFirstLine")
    t_last = rcm_time(".//r:zeroDopplerTimeLastLine")
    dt = ((t_last - t_first) / np.timedelta64(1, "s") / (n_lines - 1)) if n_lines > 1 else 0.0
    range_spacing = num(".//r:sampledPixelSpacing")

    times, pos, vel = [], [], []
    for sv in root.findall(".//r:orbitInformation/r:stateVector", _RCM_NS):
        times.append(np.datetime64(sv.find("r:timeStamp", _RCM_NS).text.rstrip("Z"), "ns"))
        pos.append([float(sv.find(f"r:{c}Position", _RCM_NS).text) for c in "xyz"])
        vel.append([float(sv.find(f"r:{c}Velocity", _RCM_NS).text) for c in "xyz"])

    return {
        "mission": text(".//r:satellite"), "beam": text(".//r:beamModeMnemonic"),
        "pass": text(".//r:passDirection"), "polarisation": text(".//r:polarizations"),
        "radar_frequency": freq, "wavelength": SPEED_OF_LIGHT / freq,
        "range_sampling_rate": SPEED_OF_LIGHT / (2 * range_spacing),
        "slant_range_time": 2 * num(".//r:slantRangeNearEdge") / SPEED_OF_LIGHT,
        "azimuth_time_interval": dt,
        "range_pixel_spacing": range_spacing, "azimuth_pixel_spacing": num(".//r:sampledLineSpacing"),
        "incidence_near": num(".//r:incAngNearRng"), "incidence_far": num(".//r:incAngFarRng"),
        "lines_per_burst": n_lines, "samples_per_burst": n_samples,
        "bursts": [{"azimuth_time": t_first, "first_valid_sample": np.zeros(n_lines, int),
                   "last_valid_sample": np.full(n_lines, n_samples - 1), "first_valid_line": 0,
                   "last_valid_line": n_lines - 1}],
        "orbit": {"time": np.asarray(times), "position": np.asarray(pos), "velocity": np.asarray(vel)},
    }
