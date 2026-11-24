"""Sentinel-1 GRD processing, and SLC / TOPS burst processing for interferometry.

``process_grd(..., backend=...)`` offers four routes:

* ``"native"``  - pure Python: GRD border-noise removal (:func:`remove_border_noise`),
  thermal noise removal, radiometric calibration (sigma0/beta0/gamma0 LUTs), multilooking
  and GCP-based geocoding. Fast and dependency-light, but *ellipsoid* geocoding: no DEM
  terrain correction (pair :func:`local_incidence_angle` and :func:`terrain_flatten` with
  your own DEM for a native terrain-flattened gamma0 instead).
* ``"sarsen"``  - terrain-corrected, radiometrically terrain-flattened gamma0 (RTC)
  with a DEM, via B-Open's sarsen.
* ``"snap"``    - ESA SNAP graph via pyroSAR (requires a SNAP installation).
* ``"hyp3"``    - submit an on-demand RTC job to ASF HyP3 (cloud; Earthdata login).

For analysis-ready RTC without local processing, :func:`load_rtc` reads
Microsoft Planetary Computer's global ``sentinel-1-rtc`` collection.

Multitemporal stacks: :func:`read_grd_subset` calibrates just the AOI window of a (remote)
GRD scene, :func:`create_stack` co-registers many scenes onto one map grid (SNAP's
``CreateStack``), :func:`estimate_offset` / :func:`apply_offset` register the stack to a
terrain-corrected reference, and :func:`multitemporal_speckle_filter` applies the Quegan & Yu
filter (SNAP's ``Multi-Temporal-Speckle-Filter``) to the stack.

Supporting readers parse the product's own XML metadata for what the functions above need:
precise/restituted orbit state vectors (:func:`read_orbit_file`, :func:`interpolate_orbit`,
the ``Apply-Orbit-File`` step), and per-pixel incidence angle and platform heading from the
annotation file's geolocation grid (:func:`read_geolocation_grid`, :func:`incidence_angle_grid`,
:func:`read_platform_heading`). VH/VV ratio and the dual-pol radar vegetation index live in
:func:`geolattice.rsense.sar.polsar.dual_pol_features`, not here.

SLC / TOPS bursts (interferometry): :func:`download_orbit`, :func:`crop_orbit` and
:func:`apply_orbit_file` (precise orbits), :func:`read_slc_annotation` and :func:`read_slc_burst`
(TOPS Split), :func:`geo2rdr` (range-Doppler geometry), :func:`back_geocode` (DEM-assisted
co-registration), :func:`tops_deramp_phase` and :func:`resample_slc` (deramp, interpolate,
reramp), :func:`esd_azimuth_shift` (Enhanced Spectral Diversity), :func:`deburst` and
:func:`terrain_correct` (Range-Doppler terrain correction). Interferogram formation, unwrapping
and phase-to-height are in :mod:`geolattice.rsense.insar`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------- units
def to_db(x: Any, floor: float = 1e-10):
    """Linear power → decibels."""
    return 10 * np.log10(np.maximum(x, floor) if not isinstance(x, xr.DataArray) else x.clip(min=floor))


def from_db(x: Any):
    """Decibels → linear power."""
    return 10 ** (x / 10)


def sigma0_to_gamma0(sigma0: Any, incidence_deg: Any):
    """Ellipsoid normalisation ``gamma0 = sigma0 / cos(theta_inc)`` (no DEM)."""
    return sigma0 / np.cos(np.radians(incidence_deg))


# ----------------------------------------------------------------------------- LUT parsing
def _xml_root(xml_path: Any):
    """Parse an XML file from a local path, an http(s) URL (e.g. a signed Planetary Computer
    ``schema-calibration-*`` asset) or an already parsed root element."""
    if isinstance(xml_path, ET.Element):
        return xml_path
    src = str(xml_path)
    if src.startswith(("http://", "https://")):
        import io
        import urllib.request

        with urllib.request.urlopen(src, timeout=60) as resp:
            return ET.parse(io.BytesIO(resp.read())).getroot()
    return ET.parse(xml_path).getroot()


def _vectors(xml_path: Any, tag: str, value_tag: str):
    root = _xml_root(xml_path)
    lines, pixels, values = [], None, []
    for vec in root.iter(tag):
        lines.append(int(vec.find("line").text))
        pix = np.array(vec.find("pixel").text.split(), dtype=float)
        val = np.array(vec.find(value_tag).text.split(), dtype=float)
        if pixels is None:
            pixels = pix
        elif len(pix) != len(pixels) or not np.allclose(pix, pixels):
            val = np.interp(pixels, pix, val)  # harmonise irregular vectors
        values.append(val)
    if not lines:
        raise ValueError(f"No <{tag}> vectors found in {xml_path if not isinstance(xml_path, ET.Element) else root.tag}")
    return np.asarray(lines, float), pixels, np.vstack(values)


def read_calibration_lut(xml_path: str | PathLike,
                         kind: Literal["sigmaNought", "betaNought", "gamma", "dn"] = "sigmaNought"):
    """Parse a ``calibration-*.xml`` file (local path or URL) → (lines, pixels, values)."""
    return _vectors(xml_path, "calibrationVector", kind)


def read_noise_lut(xml_path: str | PathLike):
    """Parse range noise vectors from ``noise-*.xml`` (local path or URL; IPF >= 2.9 and older layouts)."""
    root = _xml_root(xml_path)
    if root.find(".//noiseRangeVector") is not None:
        return _vectors(root, "noiseRangeVector", "noiseRangeLut")
    return _vectors(root, "noiseVector", "noiseLut")


def interpolate_lut(lines: np.ndarray, pixels: np.ndarray, values: np.ndarray,
                    shape: tuple[int, int], *, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Bilinear interpolation of a sparse (line, pixel) LUT onto an image grid.

    ``offset`` is ``(row0, col0)`` of that grid in the LUT's own (full-scene) line/pixel
    numbering -- pass it when ``shape`` is a window read from the measurement GeoTIFF (e.g.
    cropped to an AOI with ``rasterio``) rather than the whole scene, so the LUT is sampled at
    the window's true position, not as if it started at the scene's own line/pixel 0.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((lines, pixels), values, bounds_error=False, fill_value=None)
    row0, col0 = offset
    rr = row0 + np.arange(shape[0], dtype=float)
    cc = col0 + np.arange(shape[1], dtype=float)
    # evaluate separably for memory efficiency: rows first, then columns
    out = np.empty(shape, dtype="float32")
    step = max(1, 2_000_000 // shape[1])
    for r0 in range(0, shape[0], step):
        r = rr[r0:r0 + step]
        g = np.stack(np.meshgrid(r, cc, indexing="ij"), -1)
        out[r0:r0 + step] = interp(g)
    return out


def calibrate(dn: np.ndarray, cal_lut: np.ndarray, noise_lut: np.ndarray | None = None
              ) -> np.ndarray:
    """``sigma0 = (DN^2 - noise) / A^2``, clipped at a small positive floor."""
    dn2 = dn.astype("float64") ** 2
    if noise_lut is not None:
        dn2 = dn2 - noise_lut
    out = dn2 / cal_lut.astype("float64") ** 2
    out[dn == 0] = np.nan
    return np.maximum(out, 1e-6, where=np.isfinite(out), out=out).astype("float32")


def multilook(img: Any, looks: tuple[int, int] = (1, 1)):
    """Block-average by (azimuth, range) looks."""
    ly, lx = looks
    if ly == lx == 1:
        return img
    if isinstance(img, xr.DataArray):
        ydim, xdim = img.dims[-2:]
        return img.coarsen({ydim: ly, xdim: lx}, boundary="trim").mean()
    H, W = img.shape[-2] // ly * ly, img.shape[-1] // lx * lx
    a = img[..., :H, :W]
    return np.nanmean(a.reshape(a.shape[:-2] + (H // ly, ly, W // lx, lx)), axis=(-3, -1))


# ----------------------------------------------------------------------------- native chain
def _utm_from_gcps(gcps) -> str:
    """UTM zone EPSG code for the centroid of a GCP list (lon/lat, EPSG:4326)."""
    lon = sum(gcp.x for gcp in gcps) / len(gcps)
    lat = sum(gcp.y for gcp in gcps) / len(gcps)
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _process_native(safe_dir, polarizations, kind, looks, dst_crs, resolution, noise, border_noise):
    import rasterio
    from rasterio.enums import Resampling

    from ..core.io import read_safe

    idx = read_safe(safe_dir)
    layers = {}
    for pol in polarizations:
        pol = pol.lower()
        if pol not in idx["measurement"]:
            raise KeyError(f"Polarization {pol} not in product ({list(idx['measurement'])})")
        with rasterio.open(idx["measurement"][pol]) as src:
            dn = src.read(1)
            gcps, gcp_crs = src.gcps
        if border_noise:
            dn = remove_border_noise(dn)
        cal = interpolate_lut(*read_calibration_lut(idx["calibration"][pol], kind), dn.shape)
        nz = interpolate_lut(*read_noise_lut(idx["noise"][pol]), dn.shape) if noise else None
        s0 = calibrate(dn, cal, nz)
        da = xr.DataArray(s0, dims=("y", "x"),
                          coords={"y": np.arange(s0.shape[0]), "x": np.arange(s0.shape[1])})
        da = da.rio.write_crs(gcp_crs)
        # GCPs are geographic (EPSG:4326); `resolution` is a distance (metres, 20 m default),
        # so reprojecting to that geographic CRS unless a projected `dst_crs` is given would
        # silently ask for e.g. 20-degree pixels -- auto-pick the UTM zone under the scene
        # instead of defaulting to gcp_crs.
        target_crs = dst_crs or _utm_from_gcps(gcps)
        da = da.rio.reproject(target_crs, gcps=gcps, resolution=resolution,
                              resampling=Resampling.bilinear, nodata=np.nan)
        layers[pol] = multilook(da, looks)
    ds = xr.Dataset(layers)
    ds.attrs.update({"pyrsense:sensor": "sentinel1", "calibration": kind, "units": "linear",
                     "terrain_corrected": False})
    return ds


def process_grd(safe_dir: str | PathLike | None = None, *, polarizations: tuple[str, ...] = ("vv", "vh"),
                backend: Literal["native", "sarsen", "snap", "hyp3"] = "native",
                kind: str = "sigmaNought", looks: tuple[int, int] = (1, 1), dst_crs: Any = None,
                resolution: float | None = 20.0, noise: bool = True, border_noise: bool = True,
                dem: str | None = None, out_dir: str | PathLike = "s1_processed",
                granule: str | None = None, **kwargs: Any):
    """Process a Sentinel-1 GRD product to calibrated backscatter (linear units).

    Parameters
    ----------
    safe_dir : unzipped ``.SAFE`` directory (native/sarsen/snap) – or use ``granule`` for HyP3.
    backend : see module docstring.
    kind : calibration LUT for native: ``sigmaNought`` (default), ``betaNought`` or ``gamma``.
    border_noise : native only – zero likely GRD border-noise pixels before calibration
        (:func:`remove_border_noise`), SNAP's ``Remove-GRD-Border-Noise`` step.
    dem : DEM path for ``sarsen`` (any GDAL raster), ignored by native.
    """
    if backend == "native":
        return _process_native(safe_dir, polarizations, kind, looks, dst_crs, resolution, noise,
                               border_noise)
    if backend == "sarsen":
        return _process_sarsen(safe_dir, polarizations, dem, out_dir, **kwargs)
    if backend == "snap":
        return _process_snap(safe_dir, polarizations, out_dir, resolution, dst_crs, **kwargs)
    if backend == "hyp3":
        from ..insar.hyp3 import submit_rtc

        return submit_rtc(granule or Path(str(safe_dir)).stem, resolution=int(resolution or 30),
                          **kwargs)
    raise ValueError(f"Unknown backend {backend!r}")


def _process_sarsen(safe_dir, polarizations, dem, out_dir, **kwargs):
    try:
        import sarsen
    except ImportError as e:
        raise ImportError("backend='sarsen' needs `pip install sarsen`") from e
    if dem is None:
        raise ValueError("backend='sarsen' requires dem=<path to DEM>")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    swath = "IW"
    layers = {}
    for pol in polarizations:
        target = out / f"{Path(str(safe_dir)).stem}_{pol}_gamma0.tif"
        # sarsen >= 0.9 takes a pre-built SarProduct, not a bare path, as its first argument
        product = sarsen.Sentinel1SarProduct(str(safe_dir), measurement_group=f"{swath}/{pol.upper()}")
        da = sarsen.terrain_correction(product, dem_urlpath=dem, output_urlpath=str(target),
                                       correct_radiometry="gamma_bilinear", **kwargs)
        layers[pol.lower()] = da
    ds = xr.Dataset(layers)
    ds.attrs.update({"pyrsense:sensor": "sentinel1", "calibration": "gamma0_rtc",
                     "terrain_corrected": True})
    return ds


def _process_snap(safe_dir, polarizations, out_dir, resolution, dst_crs, **kwargs):
    try:
        from pyroSAR.snap import geocode
    except ImportError as e:
        raise ImportError("backend='snap' needs pyroSAR and an ESA SNAP installation") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geocode(infile=str(safe_dir), outdir=str(out), spacing=resolution or 20,
            polarizations=[p.upper() for p in polarizations],
            t_srs=dst_crs or 4326, scaling="linear", refarea="gamma0",
            removeS1ThermalNoise=True, terrainFlattening=True, **kwargs)
    from ..core.raster import read

    layers = {}
    for p in polarizations:
        hits = sorted(out.glob(f"*{p.upper()}*gamma0*.tif"))
        if hits:
            layers[p.lower()] = read(hits[-1]).squeeze("band", drop=True)
    return xr.Dataset(layers)


# ----------------------------------------------------------------------------- multitemporal stacks
def read_grd_subset(measurement: str | PathLike, calibration: str | PathLike,
                    noise: str | PathLike | None = None, *, bounds: tuple[float, float, float, float],
                    margin: int = 50, kind: Literal["sigmaNought", "betaNought", "gamma"] = "sigmaNought",
                    border_noise: bool = False) -> dict:
    """Calibrated backscatter for a lon/lat box cut out of one GRD measurement image.

    Only the window covering ``bounds`` (west, south, east, north, degrees) plus ``margin``
    pixels is read, so a remote Cloud-Optimized GeoTIFF (Planetary Computer ``sentinel-1-grd``)
    costs a few megabytes instead of the whole ~600 MB scene. The window is located with an
    affine transform fitted to the product's own GCPs; the calibration and noise LUTs (local
    paths or URLs) are interpolated at the window's true position in the scene. Only the GCPs
    surrounding the window are returned, so geocoding fits the local geometry.

    Returns ``{"data": sigma0 (float32, linear), "gcps": GCPs shifted to the window,
    "gcp_crs", "window": (row0, col0, height, width)}`` for :func:`create_stack`.
    """
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.transform import from_gcps
    from rasterio.windows import Window

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(str(measurement)) as src:
            scene_width = src.width
            gcps, gcp_crs = src.gcps
            if not gcps:
                raise ValueError(f"{measurement} has no GCPs; is it a GRD measurement image?")
            inverse = ~from_gcps(gcps)
            west, south, east, north = bounds
            cols, rows = zip(*[inverse * (x, y) for x, y in
                               [(west, south), (east, south), (west, north), (east, north)]])
            row0 = max(0, int(np.floor(min(rows))) - margin)
            col0 = max(0, int(np.floor(min(cols))) - margin)
            row1 = min(src.height, int(np.ceil(max(rows))) + margin)
            col1 = min(src.width, int(np.ceil(max(cols))) + margin)
            if row1 <= row0 or col1 <= col0:
                raise ValueError("bounds do not intersect the scene")
            dn = src.read(1, window=Window(col0, row0, col1 - col0, row1 - row0))
    if border_noise:
        dn = remove_border_noise(dn, col_offset=col0, scene_width=scene_width)
    offset = (row0, col0)
    cal = interpolate_lut(*read_calibration_lut(calibration, kind), dn.shape, offset=offset)
    nz = interpolate_lut(*read_noise_lut(noise), dn.shape, offset=offset) if noise is not None else None
    # Geocode with the GCPs around the window only: a first-order polynomial fitted to all of a
    # 250 km swath's GCPs is off by hundreds of metres locally (Earth curvature, terrain).
    near = _gcps_near(gcps, row0, col0, row0 + dn.shape[0], col0 + dn.shape[1])
    local = [GroundControlPoint(row=g.row - row0, col=g.col - col0, x=g.x, y=g.y, z=g.z) for g in near]
    return {"data": calibrate(dn, cal, nz), "gcps": local, "gcp_crs": gcp_crs,
            "window": (row0, col0, dn.shape[0], dn.shape[1])}


def _gcps_near(gcps: list, row0: int, col0: int, row1: int, col1: int, min_points: int = 9) -> list:
    """GCPs inside the window grown by one GCP-grid spacing on every side (at least ``min_points``)."""
    rows = np.unique([g.row for g in gcps])
    cols = np.unique([g.col for g in gcps])
    dr = float(np.median(np.diff(rows))) if len(rows) > 1 else 0.0
    dc = float(np.median(np.diff(cols))) if len(cols) > 1 else 0.0
    for grow in (1.0, 2.0, 3.0):
        near = [g for g in gcps if row0 - grow * dr <= g.row <= row1 + grow * dr
                and col0 - grow * dc <= g.col <= col1 + grow * dc]
        if len(near) >= min_points:
            return near
    return list(gcps)


def common_grid(bounds: tuple[float, float, float, float], crs: Any, resolution: float):
    """A north-up target grid ``(transform, width, height)`` in ``crs`` covering a lon/lat box,
    snapped to whole multiples of ``resolution`` so separate runs share pixel edges."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bounds, densify_pts=21)
    left = np.floor(left / resolution) * resolution
    top = np.ceil(top / resolution) * resolution
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))
    return from_origin(left, top, resolution, resolution), width, height


def _naive_utc(times: Any) -> Any:
    """Timestamps as timezone-naive UTC (STAC datetimes are tz-aware; netCDF and Zarr need naive)."""
    import pandas as pd

    t = pd.to_datetime(list(times), utc=True)
    return t.tz_convert(None)


def create_stack(scenes: list, times: Any, *, bounds: tuple[float, float, float, float],
                 crs: Any = None, resolution: float = 20.0, resampling: str = "bilinear",
                 name: str = "sigma0") -> xr.DataArray:
    """Co-register GRD scenes into one ``(time, y, x)`` stack on a common map grid.

    The counterpart of SNAP's ``CreateStack`` for a single track: every scene (an output of
    :func:`read_grd_subset`, or ``{"data", "gcps", "gcp_crs"}``) is geocoded with its own GCPs
    directly onto the same target grid (``bounds`` in lon/lat, ``crs``, ``resolution`` in CRS
    units; ``crs`` defaults to the UTM zone of the first scene). Because all scenes land on
    identical pixels, the stack is co-registered by construction and ready for
    :func:`multitemporal_speckle_filter` and per-pixel time-series analysis. Geocoding uses the
    products' GCPs (ellipsoid geolocation), not a DEM; for scenes from one relative orbit the
    residual terrain displacement is the same on every date, so it does not create false change.
    Values stay in linear units; pixels a scene does not cover are NaN.
    """
    import pandas as pd
    import rasterio.warp as warp
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from rasterio.enums import Resampling

    if len(scenes) != len(times):
        raise ValueError("scenes and times must have the same length")
    if not scenes:
        raise ValueError("no scenes to stack")
    crs = crs or _utm_from_gcps(scenes[0]["gcps"])
    transform, width, height = common_grid(bounds, crs, resolution)
    out = np.full((len(scenes), height, width), np.nan, dtype="float32")
    for k, scene in enumerate(scenes):
        src = np.asarray(scene["data"], dtype="float32")
        warp.reproject(src, out[k], gcps=scene["gcps"], src_crs=scene.get("gcp_crs") or "EPSG:4326",
                       dst_transform=transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
                       resampling=Resampling[resampling])
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    da = xr.DataArray(out, dims=("time", "y", "x"), name=name,
                      coords={"time": _naive_utc(times), "y": ys, "x": xs})
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    da.attrs.update({"units": "linear", "geocoding": "GCP (ellipsoid), common grid",
                     "resolution": resolution})
    return da


def estimate_offset(moving: Any, reference: Any, *, max_shift: int = 30, smooth: int = 5,
                    border: int | None = None) -> tuple[int, int, float]:
    """Integer pixel offset ``(dy, dx)`` that best aligns ``moving`` with ``reference``.

    Both are 2-D images on the same grid (e.g. a stack's temporal mean in dB and a
    terrain-corrected reference such as Planetary Computer RTC, or an optical index). The offset
    maximises the Pearson correlation of the two over shifts up to ``max_shift`` pixels; ``moving``
    is box-smoothed by ``smooth`` pixels first to suppress speckle. Apply the result with
    :func:`apply_offset`. Returns ``(dy, dx, correlation)``; positive ``dy``/``dx`` move ``moving``
    down/right (south/east on a north-up grid). Use it to register a GCP-geocoded stack to a
    terrain-corrected product when orbit and annotation data for range-Doppler terrain correction
    are not available; it removes the bulk offset, not terrain-dependent local displacement.
    """
    from scipy.ndimage import uniform_filter

    def prep(a):
        a = np.asarray(getattr(a, "values", a), dtype="float64")
        return np.where(np.isfinite(a), a, np.nanmean(a))

    m, r = prep(moving), prep(reference)
    if m.shape != r.shape:
        raise ValueError(f"shapes differ: {m.shape} vs {r.shape}")
    if smooth > 1:
        m = uniform_filter(m, smooth)
    b = border if border is not None else max_shift + 1
    if 2 * b >= min(m.shape):
        raise ValueError("image too small for this max_shift")
    rc = r[b:-b, b:-b].ravel()
    rc = (rc - rc.mean()) / rc.std()
    best = (-np.inf, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        rolled = np.roll(m, dy, axis=0)
        for dx in range(-max_shift, max_shift + 1):
            mc = np.roll(rolled, dx, axis=1)[b:-b, b:-b].ravel()
            sd = mc.std()
            if sd == 0:
                continue
            corr = float(np.dot((mc - mc.mean()) / sd, rc) / rc.size)
            if corr > best[0]:
                best = (corr, dy, dx)
    return best[1], best[2], best[0]


def apply_offset(data: Any, dy: int, dx: int) -> Any:
    """Shift an image or ``(..., y, x)`` stack by whole pixels on its own grid (NaN where uncovered).

    Use with :func:`estimate_offset`. For an ``xarray`` object the coordinates, CRS and attributes
    are kept and the applied offset is recorded in ``attrs["offset_pixels"]``.
    """
    def shift(a):
        a = np.asarray(a, dtype="float32")
        out = np.full_like(a, np.nan)
        ys = slice(max(dy, 0), a.shape[-2] + min(dy, 0))
        yd = slice(max(-dy, 0), a.shape[-2] + min(-dy, 0))
        xs = slice(max(dx, 0), a.shape[-1] + min(dx, 0))
        xd = slice(max(-dx, 0), a.shape[-1] + min(-dx, 0))
        out[..., ys, xs] = a[..., yd, xd]
        return out

    if isinstance(data, xr.Dataset):
        out = data.copy()
        for var in data.data_vars:
            out[var] = apply_offset(data[var], dy, dx)
        return out
    if isinstance(data, xr.DataArray):
        out = data.copy(data=shift(data.values))
        out.attrs["offset_pixels"] = (int(dy), int(dx))
        return out
    return shift(data)


def multitemporal_speckle_filter(stack: Any, *, spatial: Literal["boxcar", "lee", "refined_lee"] = "boxcar",
                                 size: int = 7, enl: float = 4.4, time_dim: str = "time") -> Any:
    """Quegan & Yu (2001) multi-temporal speckle filter for a co-registered stack.

    SNAP's ``Multi-Temporal-Speckle-Filter``: each date keeps its own local mean while the
    speckle is averaged across all N dates,
    ``J_k = E[I_k] / N * sum_i I_i / E[I_i]``, with ``E`` a spatial filter (``boxcar``, ``lee``
    or ``refined_lee`` of ``size``). Speckle falls by up to sqrt(N) without the spatial blurring
    of a single-image filter of the same strength. Accepts an ``xarray.DataArray`` with a time
    dimension (coordinates and CRS are kept), an ``xarray.Dataset`` of such stacks (filtered per
    polarisation) or a NumPy array with time on axis 0. Input must be linear intensity.
    """
    from .speckle import multitemporal

    if isinstance(stack, xr.Dataset):
        out = stack.copy()
        for var in stack.data_vars:
            out[var] = multitemporal_speckle_filter(stack[var], spatial=spatial, size=size, enl=enl,
                                                    time_dim=time_dim)
        return out
    if isinstance(stack, xr.DataArray):
        if time_dim not in stack.dims:
            raise ValueError(f"stack has no {time_dim!r} dimension")
        axis = stack.dims.index(time_dim)
        filtered = np.asarray(multitemporal(stack.values, spatial=spatial, size=size, enl=enl, time_axis=axis))
        out = stack.copy(data=filtered.astype(stack.dtype if stack.dtype.kind == "f" else "float32"))
        out.attrs.update({"speckle_filter": f"Quegan multitemporal ({spatial}, {size}x{size})",
                          "units": stack.attrs.get("units", "linear")})
        return out
    return multitemporal(np.asarray(stack), spatial=spatial, size=size, enl=enl, time_axis=0)


# ----------------------------------------------------------------------------- ARD loader
def load_rtc(aoi: Any, dates: Any, *, polarizations: tuple[str, ...] = ("vv", "vh"),
             orbit: Literal["ascending", "descending"] | None = None,
             resolution: float = 10, crs: Any = None, chunks: Any = None,
             **kwargs: Any) -> xr.Dataset:
    """Load Planetary Computer ``sentinel-1-rtc`` (gamma0, terrain-corrected, linear power).

    ``orbit`` filters by ``sat:orbit_state`` so a time series has consistent geometry.
    """
    from ..access import stac_load, stac_search

    query = {"sat:orbit_state": {"eq": orbit}} if orbit else None
    items = stac_search("sentinel-1-rtc", aoi=aoi, dates=dates, query=query, **kwargs)
    ds = stac_load(items, bands=list(polarizations), aoi=aoi, resolution=resolution, crs=crs,
                   chunks=chunks, sensor="sentinel1")
    ds = ds.where(ds > 0)
    ds.attrs.update({"calibration": "gamma0_rtc", "units": "linear", "terrain_corrected": True})
    return ds


# ----------------------------------------------------------------------------- border noise
def remove_border_noise(dn: np.ndarray, *, threshold: float = 0.0, margin: int = 400,
                        col_offset: int = 0, scene_width: int | None = None) -> np.ndarray:
    """Mask likely GRD border-noise pixels near the left/right scene edges.

    A lightweight approximation of SNAP's ``Remove-GRD-Border-Noise`` operator: within
    ``margin`` pixels of each edge (where border noise actually occurs for IW/EW GRD
    products), zero out any run of pixels starting at the edge until a value first exceeds
    ``threshold``. The default ``threshold=0.0`` only removes the zero-padded invalid region
    most products already carry at their border; raise it to also trim low-level thermal noise
    that SNAP's own per-subswath noise-vector logic would otherwise catch. When processing a
    cropped window, pass its ``col_offset`` and the full ``scene_width`` so only actual scene
    edges are considered. Call this on raw DN *before* calibration.
    """
    out = np.array(dn, copy=True)
    H, W = out.shape
    if col_offset < 0 or (scene_width is not None
                          and (scene_width <= 0 or col_offset + W > scene_width)):
        raise ValueError("The window column offset and width must fit inside the scene")
    m = min(margin, W // 2)
    if m <= 0:
        return out

    def _edge_run(block: np.ndarray) -> np.ndarray:
        """True where a column should be zeroed: a contiguous run from column 0 of ``block``
        (the true scene edge) up to the first value exceeding ``threshold``."""
        valid = block > threshold
        first = np.where(valid.any(axis=1), np.argmax(valid, axis=1), block.shape[1])
        cols = np.arange(block.shape[1])
        return cols[None, :] < first[:, None]

    if scene_width is None or col_offset == 0:
        out[:, :m][_edge_run(out[:, :m])] = 0
    if scene_width is None or col_offset + W == scene_width:
        out[:, -m:][_edge_run(out[:, -m:][:, ::-1])[:, ::-1]] = 0
    return out


# ----------------------------------------------------------------------------- orbit files
def read_orbit_file(eof_path: str | PathLike) -> dict[str, np.ndarray]:
    """Parse a Sentinel-1 precise (POEORB) or restituted (RESORB) orbit file (``.EOF``) into
    its state vectors: UTC ``time``, ECEF ``position`` (m) and ``velocity`` (m/s), one row per
    orbit state vector (OSV, spaced about every 10 s).

    This is the data ``Apply-Orbit-File`` substitutes for a GRD/SLC product's own (less
    precise) orbit metadata before geocoding. A matching ``.EOF`` file can be downloaded from
    the Copernicus Data Space Ecosystem or ASF by sensing time and orbit number -- POEORB
    becomes available about 20 days after acquisition (the best accuracy); RESORB is published
    within hours, for near-real-time use. The native backend here geocodes from the GRD
    product's own embedded GCPs rather than reprocessing with a refined orbit (a full
    reprocessing is what the ``sarsen``/``snap``/``hyp3`` backends already do); these state
    vectors are for independent uses such as baseline estimation or QC, via
    :func:`interpolate_orbit`.
    """
    root = _xml_root(eof_path)
    times, pos, vel = [], [], []
    for osv in root.iter("OSV"):
        utc = osv.find("UTC").text.split("=", 1)[-1]
        times.append(np.datetime64(utc))
        pos.append([float(osv.find(c).text) for c in ("X", "Y", "Z")])
        vel.append([float(osv.find(c).text) for c in ("VX", "VY", "VZ")])
    if not times:
        raise ValueError(f"No <OSV> state vectors found in {eof_path}")
    order = np.argsort(times)
    return {"time": np.asarray(times)[order], "position": np.asarray(pos)[order],
           "velocity": np.asarray(vel)[order]}


def interpolate_orbit(orbit: dict[str, np.ndarray], times: Any) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ``read_orbit_file``'s position/velocity to arbitrary ``times``.

    Fits a cubic spline per axis: state vectors are spaced about every 10 s, dense enough that
    a spline reproduces the true (physically smooth) orbit well without a full force-model
    propagator. Returns ``(position, velocity)``, each shape ``(len(times), 3)``, metres and
    metres/second in the orbit file's ECEF frame.
    """
    from scipy.interpolate import CubicSpline

    t0 = orbit["time"].astype("datetime64[ns]").astype("int64")
    tq = np.asarray(times, dtype="datetime64[ns]").astype("int64")
    pos = np.column_stack([CubicSpline(t0, orbit["position"][:, k])(tq) for k in range(3)])
    vel = np.column_stack([CubicSpline(t0, orbit["velocity"][:, k])(tq) for k in range(3)])
    return pos, vel


# ----------------------------------------------------------------------------- geometry
def read_geolocation_grid(xml_path: str | PathLike
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Parse an annotation XML's sparse ``geolocationGridPointList`` (an evenly spaced
    line x pixel grid, typically 21 x 10 points for IW GRDH) into line/pixel axes plus 2D
    ``{"latitude", "longitude", "height", "incidenceAngle", "elevationAngle"}`` arrays.

    Feed any of these through :func:`interpolate_lut` to get a full-resolution raster, for
    example the ``incidence_deg`` that :func:`sigma0_to_gamma0` and
    :func:`local_incidence_angle` need (see :func:`incidence_angle_grid`).
    """
    root = _xml_root(xml_path)
    fields = ("latitude", "longitude", "height", "incidenceAngle", "elevationAngle")
    rows = [{tag: float(pt.find(tag).text) for tag in ("line", "pixel") + fields}
           for pt in root.iter("geolocationGridPoint")]
    if not rows:
        raise ValueError(f"No <geolocationGridPoint> found in {xml_path}")
    lines = np.asarray(sorted({r["line"] for r in rows}))
    pixels = np.asarray(sorted({r["pixel"] for r in rows}))
    li = {v: i for i, v in enumerate(lines)}
    pi = {v: i for i, v in enumerate(pixels)}
    grids = {f: np.full((lines.size, pixels.size), np.nan) for f in fields}
    for r in rows:
        i, j = li[r["line"]], pi[r["pixel"]]
        for f in fields:
            grids[f][i, j] = r[f]
    return lines, pixels, grids


def incidence_angle_grid(xml_path: str | PathLike, shape: tuple[int, int], *,
                         offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Full-resolution ellipsoid incidence angle (degrees), bilinearly interpolated from the
    annotation XML's sparse geolocation grid -- the ``incidence_deg`` :func:`sigma0_to_gamma0`
    and :func:`local_incidence_angle` need. ``offset`` is ``(row0, col0)`` of ``shape`` in the
    scene's own line/pixel numbering; see :func:`interpolate_lut`."""
    lines, pixels, grids = read_geolocation_grid(xml_path)
    return interpolate_lut(lines, pixels, grids["incidenceAngle"], shape, offset=offset)


def read_platform_heading(xml_path: str | PathLike) -> tuple[float, str | None]:
    """Platform flight-direction heading (degrees clockwise from north, 0-360) and the pass
    direction (``"ASCENDING"``/``"DESCENDING"``) from an annotation XML. Feed the heading into
    :func:`local_incidence_angle`."""
    root = _xml_root(xml_path)
    heading_el = root.find(".//platformHeading")
    if heading_el is None:
        raise ValueError(f"No <platformHeading> found in {xml_path}")
    pass_el = root.find(".//pass")
    return float(heading_el.text) % 360, (pass_el.text if pass_el is not None else None)


# ----------------------------------------------------------------------------- terrain flattening
def local_incidence_angle(dem: np.ndarray, resolution: float, heading_deg: float,
                          incidence_deg: Any, *, look: Literal["right", "left"] = "right"
                          ) -> np.ndarray:
    """Local incidence angle (degrees) from a DEM already resampled onto the SAR image grid.

    Models the ground as locally planar from the DEM's own gradients -- a surface normal in a
    local east-north-up frame, so ``dem`` must be on a projected, equal-spacing (metres) grid
    -- and compares it against the radar look vector implied by ``heading_deg`` (platform
    flight direction, degrees clockwise from north; see :func:`read_platform_heading`) and
    ``incidence_deg`` (ellipsoid incidence angle, scalar or per-pixel; see
    :func:`incidence_angle_grid`). Sentinel-1 only ever flies right-looking
    (``look="right"``): an ascending pass then looks east and a descending pass looks west,
    both handled automatically from ``heading_deg`` alone. Follows the surface-normal /
    look-vector dot-product formulation (Small, 2011, "Flattening Gamma").

    Pass the result to :func:`terrain_flatten` to turn a beta0 image into a true
    terrain-flattened gamma0 -- the DEM-aware counterpart of :func:`sigma0_to_gamma0`'s
    ellipsoid-only approximation.
    """
    dem = np.asarray(dem, dtype="float64")
    dzdy, dzdx = np.gradient(dem, resolution)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(dem)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    look_azimuth = np.radians(heading_deg + (90 if look == "right" else -90))
    theta = np.broadcast_to(np.radians(incidence_deg), dem.shape).astype("float64")
    look_vec = np.stack([np.sin(theta) * np.sin(look_azimuth),
                         np.sin(theta) * np.cos(look_azimuth),
                         np.cos(theta)], axis=-1)
    cos_lia = np.clip(np.sum(normal * look_vec, axis=-1), -1, 1)
    return np.degrees(np.arccos(cos_lia))


def terrain_flatten(beta0: Any, local_incidence_deg: Any) -> Any:
    """Beta0 -> terrain-flattened gamma0: ``gamma0 = beta0 / cos(local incidence angle)``,
    using the DEM-derived local incidence angle from :func:`local_incidence_angle` rather than
    :func:`sigma0_to_gamma0`'s ellipsoid-only approximation. Calibrate to beta0 first
    (``process_grd(..., kind="betaNought")``), not sigma0, before flattening."""
    return beta0 / np.cos(np.radians(local_incidence_deg))


# ============================================================================= SLC / TOPS bursts
# The functions below take Sentinel-1 IW SLC bursts through the interferometric pre-processing
# chain of SNAP's TOPS InSAR graph: Apply-Orbit-File, TOPS-Split, Back-Geocoding (with TOPS
# deramping), Enhanced Spectral Diversity and TOPS-Deburst, plus the Range-Doppler geometry
# (geo2rdr) that back-geocoding and terrain correction share. Interferogram formation,
# filtering, unwrapping and phase-to-height live in :mod:`geolattice.rsense.insar`.
SPEED_OF_LIGHT = 299_792_458.0
_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141317e-3


def _annotation_product(root: ET.Element, swath: str | None = None, polarisation: str | None = None) -> ET.Element:
    """The ``<product>`` annotation element of an SLC: a SAFE ``annotation/*.xml`` file, or one
    of the annotations embedded in an ASF burst-extractor metadata XML (which carries one per
    sub-swath; ``swath``/``polarisation`` pick it)."""
    found = []
    for el in root.iter():
        if el.find("swathTiming") is not None and el.find("generalAnnotation") is not None:
            head = el.find("adsHeader")
            sw = head.findtext("swath") if head is not None else None
            po = head.findtext("polarisation") if head is not None else None
            if (swath is None or (sw or "").upper() == swath.upper()) and \
               (polarisation is None or (po or "").upper() == polarisation.upper()):
                found.append(el)
    if not found:
        raise ValueError(f"No SLC annotation (<swathTiming>) for swath={swath}, polarisation={polarisation}")
    if len(found) > 1:
        raise ValueError(f"{len(found)} SLC annotations found; pass swath= (and polarisation=) to choose one")
    return found[0]


def _f(el: ET.Element, path: str, cast=float):
    found = el.find(path)
    if found is None:
        raise ValueError(f"<{path}> not found in the annotation")
    return cast(found.text)


def _t64(text: str) -> np.datetime64:
    return np.datetime64(text.strip(), "ns")


def _polys(prod: ET.Element, item: str, poly: str) -> list[tuple[np.datetime64, float, np.ndarray]]:
    """(azimuth time, t0, coefficients) for each azimuth FM rate / Doppler centroid estimate."""
    out = []
    for el in prod.iter(item):
        coeffs = el.find(poly)
        if coeffs is not None:
            c = np.array(coeffs.text.split(), float)
        else:  # IPF < 2.80: c0, c1, c2 tags
            c = np.array([float(el.find(k).text) for k in ("c0", "c1", "c2")])
        out.append((_t64(el.find("azimuthTime").text), float(el.find("t0").text), c))
    return out


def read_slc_annotation(xml_path: Any, *, swath: str | None = None, polarisation: str | None = None
                        ) -> dict[str, Any]:
    """Parse what TOPS interferometry needs from a Sentinel-1 IW SLC annotation XML (a SAFE
    ``annotation/*.xml`` file or an ASF burst-extractor metadata file; local path, URL or parsed
    element).

    Returns a dict with the radar parameters (``wavelength``, ``radar_frequency``,
    ``range_sampling_rate``, ``slant_range_time`` of the first sample, ``azimuth_time_interval``,
    ``azimuth_steering_rate`` in rad/s, pixel spacings, ``lines_per_burst``,
    ``samples_per_burst``), the ``bursts`` list (``azimuth_time`` of the first line, per-line
    ``first_valid_sample``/``last_valid_sample`` and the derived ``first_valid_line`` /
    ``last_valid_line``), the azimuth FM-rate (``fm_rate``) and Doppler-centroid (``doppler``)
    polynomials used for deramping, the annotation ``orbit`` (same layout as
    :func:`read_orbit_file`), the ``geolocation`` grid (as :func:`read_geolocation_grid`) and
    ``mission``/``swath``/``polarisation``/``pass``/``heading``. An ASF burst-extractor XML
    holds the annotations of every sub-swath: choose one with ``swath`` (``"IW1"``-``"IW3"``) and,
    if needed, ``polarisation``.
    """
    root = _xml_root(xml_path)
    prod = _annotation_product(root, swath, polarisation)
    gen, img = prod.find("generalAnnotation"), prod.find("imageAnnotation/imageInformation")
    info = gen.find("productInformation")
    freq = _f(info, "radarFrequency")
    lpb = _f(prod, "swathTiming/linesPerBurst", int)
    bursts = []
    for b in prod.find("swathTiming/burstList"):
        fvs = np.array(b.find("firstValidSample").text.split(), int)
        lvs = np.array(b.find("lastValidSample").text.split(), int)
        valid = np.flatnonzero(fvs >= 0)
        bursts.append({"azimuth_time": _t64(b.find("azimuthTime").text),
                       "first_valid_sample": fvs, "last_valid_sample": lvs,
                       "first_valid_line": int(valid[0]) if valid.size else 0,
                       "last_valid_line": int(valid[-1]) if valid.size else lpb - 1})
    times, pos, vel = [], [], []
    for o in gen.find("orbitList"):
        times.append(_t64(o.find("time").text))
        pos.append([float(o.find(f"position/{c}").text) for c in "xyz"])
        vel.append([float(o.find(f"velocity/{c}").text) for c in "xyz"])
    header = prod.find("adsHeader")
    heading = prod.find(".//platformHeading")
    pas = prod.find(".//pass")
    return {
        "mission": header.find("missionId").text if header is not None else None,
        "swath": header.find("swath").text if header is not None else None,
        "polarisation": header.find("polarisation").text if header is not None else None,
        "pass": pas.text if pas is not None else None,
        "heading": float(heading.text) % 360 if heading is not None else None,
        "radar_frequency": freq,
        "wavelength": SPEED_OF_LIGHT / freq,
        "range_sampling_rate": _f(info, "rangeSamplingRate"),
        "azimuth_steering_rate": np.radians(_f(info, "azimuthSteeringRate")),
        "slant_range_time": _f(img, "slantRangeTime"),
        "azimuth_time_interval": _f(img, "azimuthTimeInterval"),
        "range_pixel_spacing": _f(img, "rangePixelSpacing"),
        "azimuth_pixel_spacing": _f(img, "azimuthPixelSpacing"),
        "incidence_mid": _f(img, "incidenceAngleMidSwath"),
        "lines_per_burst": lpb,
        "samples_per_burst": _f(prod, "swathTiming/samplesPerBurst", int),
        "bursts": bursts,
        "fm_rate": _polys(gen, "azimuthFmRate", "azimuthFmRatePolynomial"),
        "doppler": _polys(prod.find("dopplerCentroid"), "dcEstimate", "dataDcPolynomial"),
        "orbit": {"time": np.asarray(times), "position": np.asarray(pos), "velocity": np.asarray(vel)},
        "geolocation": read_geolocation_grid(prod),
    }


def read_slc_burst(path: str | PathLike, ann: dict | None = None, burst: int = 0, *,
                   cols: tuple[int, int] | None = None) -> np.ndarray:
    """Read one burst of an IW SLC measurement GeoTIFF as ``complex64`` (TOPS-Split).

    ``path`` is either a single-burst file (ASF burst extractor; ``burst`` is then ignored) or
    a full sub-swath measurement file, from which lines ``burst * linesPerBurst`` onwards are
    read. ``cols=(start, stop)`` crops the range samples to an AOI. Lines outside the burst's
    valid range and samples outside each line's valid range are set to 0.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        lpb = ann["lines_per_burst"] if ann else src.height
        row0 = 0 if src.height <= lpb else burst * lpb
        c0, c1 = cols if cols else (0, src.width)
        data = src.read(1, window=Window(c0, row0, c1 - c0, lpb)).astype("complex64")
    if ann is not None:
        b = ann["bursts"][burst]
        fvs, lvs = b["first_valid_sample"], b["last_valid_sample"]
        cc = np.arange(c0, c1)
        valid = (fvs[:, None] >= 0) & (cc[None, :] >= fvs[:, None]) & (cc[None, :] <= lvs[:, None])
        data[~valid[:data.shape[0]]] = 0
    return data


# ----------------------------------------------------------------------------- orbits (Step 2)
def _fetch_text(url: str, retries: int = 5) -> str:
    """GET a URL as text, retrying transient network errors with exponential backoff."""
    import time as _time
    import urllib.request

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read().decode()
        except OSError:
            if attempt == retries - 1:
                raise
            _time.sleep(2 ** attempt)
    return ""


def download_orbit(mission: str, time: Any, out_dir: str | PathLike = ".", *,
                   kinds: tuple[str, ...] = ("AUX_POEORB", "AUX_RESORB"), margin_s: float = 120.0
                   ) -> Path:
    """Download the Sentinel-1 orbit file covering ``time`` from the open AWS ``s1-orbits``
    bucket (no login): precise POEORB first (available about 20 days after acquisition), then
    restituted RESORB. ``mission`` is ``"S1A"``, ``"S1B"``, ``"S1C"`` or ``"S1D"`` (a
    ``SENTINEL-1A`` style name works too). A matching file already in ``out_dir`` is reused
    without network access. Returns the local ``.EOF`` path."""
    import re
    import urllib.request

    import pandas as pd

    sat = "S1" + mission.strip().upper()[-1]
    ts = pd.Timestamp(time)
    t = np.datetime64((ts.tz_convert(None) if ts.tzinfo else ts).to_datetime64(), "s")
    base = "https://s1-orbits.s3.us-west-2.amazonaws.com"
    out_dir = Path(out_dir)

    def covers(name):
        m = re.search(r"_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$", name)
        if not m:
            return False
        start, stop = (np.datetime64(f"{v[:4]}-{v[4:6]}-{v[6:8]}T{v[9:11]}:{v[11:13]}:{v[13:]}") for v in m.groups())
        pad = np.timedelta64(int(margin_s), "s")
        return start <= t - pad and stop >= t + pad

    for kind in kinds:  # a matching file already downloaded: no network needed
        local = sorted(p for p in out_dir.glob(f"{sat}_OPER_{kind}_*.EOF") if covers(p.name))
        if local:
            return local[-1]
    for kind in kinds:
        offsets = (0, 10, 20, 30) if kind == "AUX_POEORB" else (-1, 0, 1)
        months = sorted({str((t + np.timedelta64(d, "D")).astype("datetime64[M]")).replace("-", "")
                         for d in offsets})
        found = []
        for month in months:
            url = f"{base}/?list-type=2&prefix={kind}/{sat}_OPER_{kind}_OPOD_{month}"
            keys = re.findall(r"<Key>([^<]+)</Key>", _fetch_text(url))
            found += [key for key in keys if covers(key)]
        if found:
            key = sorted(found)[-1]  # latest production
            dst = out_dir / Path(key).name
            if not dst.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                for attempt in range(5):
                    try:
                        urllib.request.urlretrieve(f"{base}/{key}", dst)
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        __import__("time").sleep(2 ** attempt)
            return dst
    raise FileNotFoundError(f"No orbit file for {sat} at {t} in {kinds}")


def crop_orbit(orbit: dict | str | PathLike, start: Any, stop: Any | None = None, *, margin_s: float = 100.0) -> dict:
    """State vectors of an orbit (``read_orbit_file`` dict or ``.EOF`` path) between ``start``
    and ``stop`` (default ``start``) plus ``margin_s`` seconds either side. A POEORB file spans
    26 hours and passes near any point on Earth several times; geometry functions need only the
    minutes around one acquisition."""
    import pandas as pd

    if not isinstance(orbit, dict):
        orbit = read_orbit_file(orbit)

    def naive(t):
        ts = pd.Timestamp(t)
        return np.datetime64((ts.tz_convert(None) if ts.tzinfo else ts).to_datetime64(), "ns")

    t0 = naive(start) - np.timedelta64(int(margin_s * 1e9), "ns")
    t1 = naive(stop if stop is not None else start) + np.timedelta64(int(margin_s * 1e9), "ns")
    keep = (orbit["time"] >= t0) & (orbit["time"] <= t1)
    if keep.sum() < 4:
        raise ValueError("The orbit does not cover the requested time")
    return {k: np.asarray(v)[keep] for k, v in orbit.items()}


def apply_orbit_file(ann: dict, orbit: dict | str | PathLike, *, margin_s: float = 100.0) -> dict:
    """``Apply-Orbit-File``: return a copy of ``ann`` (:func:`read_slc_annotation`) whose orbit
    is replaced by the state vectors of a precise orbit (``read_orbit_file`` dict or ``.EOF``
    path) within ``margin_s`` seconds of the product's bursts. The annotation orbit is kept as
    ``ann["annotation_orbit"]`` for comparison."""
    times = [b["azimuth_time"] for b in ann["bursts"]]
    stop = max(times) + np.timedelta64(int(ann["lines_per_burst"] * ann["azimuth_time_interval"] * 1e9), "ns")
    out = dict(ann)
    out["annotation_orbit"] = ann.get("annotation_orbit", ann["orbit"])
    out["orbit"] = crop_orbit(orbit, min(times), stop, margin_s=margin_s)
    return out


class _Orbit:
    """Hermite interpolation of orbit state vectors (positions and velocities together)."""

    def __init__(self, orbit: dict):
        from scipy.interpolate import CubicHermiteSpline

        t = np.asarray(orbit["time"], "datetime64[ns]")
        self.epoch = t[0]
        s = (t - self.epoch) / np.timedelta64(1, "s")
        self.pos = CubicHermiteSpline(s, np.asarray(orbit["position"], float),
                                      np.asarray(orbit["velocity"], float), axis=0)
        self.vel = self.pos.derivative()
        self.acc = self.vel.derivative()
        self.s = s
        self.positions = np.asarray(orbit["position"], float)

    def seconds(self, t: Any) -> np.ndarray:
        return (np.asarray(t, "datetime64[ns]") - self.epoch) / np.timedelta64(1, "s")

    def time(self, s: Any) -> np.ndarray:
        return self.epoch + np.round(np.asarray(s) * 1e9).astype("int64").astype("timedelta64[ns]")


# ----------------------------------------------------------------------------- geometry
def llh_to_ecef(lat: Any, lon: Any, height: Any) -> np.ndarray:
    """WGS84 latitude/longitude (degrees) and ellipsoidal height (m) to ECEF XYZ (m), shape (..., 3)."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    h = np.asarray(height, float)
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - _WGS84_E2) + h) * np.sin(lat)], axis=-1)


def geo2rdr(lat: Any, lon: Any, height: Any, orbit: dict, *, iterations: int = 12
            ) -> tuple[np.ndarray, np.ndarray]:
    """Range-Doppler geocoding of ground points: the zero-Doppler azimuth time and slant range
    (m) at which the satellite on ``orbit`` sees each WGS84 point (lat, lon in degrees,
    ``height`` *ellipsoidal* in metres).

    Solves ``(P - S(t)) . V(t) = 0`` by Newton iteration on a Hermite-interpolated orbit -- the
    geometry behind SNAP's Back-Geocoding and Range-Doppler Terrain Correction. Returns
    ``(azimuth_time datetime64[ns], slant_range)`` with the shape of the inputs.
    """
    P = llh_to_ecef(lat, lon, height)
    shape = P.shape[:-1]
    P = P.reshape(-1, 3)
    orb = _Orbit(orbit)
    centre = np.nanmean(P, axis=0)
    s = np.full(len(P), orb.s[np.argmin(np.linalg.norm(orb.positions - centre, axis=1))])
    for _ in range(iterations):
        S, V, A = orb.pos(s), orb.vel(s), orb.acc(s)
        d = P - S
        f = np.einsum("ij,ij->i", d, V)
        fp = np.einsum("ij,ij->i", d, A) - np.einsum("ij,ij->i", V, V)
        s = s - f / fp
    R = np.linalg.norm(P - orb.pos(s), axis=1)
    return orb.time(s).reshape(shape), R.reshape(shape)


def burst_coordinates(ann: dict, burst: int, azimuth_time: Any, slant_range: Any
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (line, sample) in a burst for zero-Doppler times and slant ranges from :func:`geo2rdr`."""
    t0 = ann["bursts"][burst]["azimuth_time"]
    line = (np.asarray(azimuth_time, "datetime64[ns]") - t0) / np.timedelta64(1, "s") / ann["azimuth_time_interval"]
    sample = (2 * np.asarray(slant_range) / SPEED_OF_LIGHT - ann["slant_range_time"]) * ann["range_sampling_rate"]
    return line, sample


def burst_window(ann: dict, burst: int, bounds: tuple[float, float, float, float], *,
                 margin: int = 200) -> tuple[int, int]:
    """Range-sample window ``(start, stop)`` of a burst covering lon/lat ``bounds`` (TOPS-Split
    to an AOI), from the annotation geolocation grid and orbit."""
    minx, miny, maxx, maxy = bounds
    lon, lat = np.meshgrid(np.linspace(minx, maxx, 9), np.linspace(miny, maxy, 9))
    _, _, grids = ann["geolocation"]
    t, r = geo2rdr(lat, lon, np.full(lat.shape, float(np.nanmean(grids["height"]))), ann["orbit"])
    _, sample = burst_coordinates(ann, burst, t, r)
    start = max(int(np.floor(sample.min())) - margin, 0)
    stop = min(int(np.ceil(sample.max())) + margin, ann["samples_per_burst"])
    return start, stop


def _scatter_to_grid(line: np.ndarray, sample: np.ndarray, values: np.ndarray, shape: tuple[int, int],
                     *, col0: int = 0, step: tuple[int, int] = (8, 32)) -> np.ndarray:
    """Linear interpolation of scattered (line, sample) values onto a burst window
    (``shape`` lines from line 0, columns from ``col0``), via a coarse grid then bilinear upsampling."""
    from scipy import ndimage
    from scipy.interpolate import LinearNDInterpolator

    rr = np.arange(0, shape[0] + step[0], step[0], dtype=float)
    cc = np.arange(col0, col0 + shape[1] + step[1], step[1], dtype=float)
    ok = np.isfinite(line) & np.isfinite(sample) & np.all(np.isfinite(values), axis=-1)
    interp = LinearNDInterpolator(np.c_[line[ok], sample[ok]], values[ok])
    coarse = interp(*np.meshgrid(rr, cc, indexing="ij"))  # (nr, nc, k)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype("float32")
    coords = [yy / step[0], xx / step[1]]
    return np.stack([ndimage.map_coordinates(coarse[..., k], coords, order=1, mode="nearest", cval=np.nan)
                     for k in range(values.shape[-1])], axis=-1)


def back_geocode(ref_ann: dict, sec_ann: dict, ref_burst: int, sec_burst: int, dem: xr.DataArray, *,
                 cols: tuple[int, int], heights: tuple[float, float] = (0.0, 20.0),
                 step: tuple[int, int] = (8, 32)) -> dict[str, np.ndarray]:
    """DEM-assisted co-registration (SNAP ``Back-Geocoding``) of a secondary burst to a
    reference burst window, plus the geometry the interferogram needs.

    Every DEM cell (``dem``: *ellipsoidal* heights with a CRS, covering the window) is located
    in both bursts with :func:`geo2rdr` on each product's (precise) orbit; the secondary
    positions are then interpolated onto every reference pixel of lines ``0..linesPerBurst``
    and samples ``cols``. Two extra passes at constant ``heights`` give the range difference
    for a flat Earth and its change with height.

    Returns ``sec_line``/``sec_sample`` (where to sample the secondary burst),
    ``dem_height`` (DEM height under each reference pixel), ``delta_range`` (secondary minus
    reference slant range at the DEM surface), ``flat_delta_range`` (same on the ellipsoid,
    for flat-Earth phase removal), ``dheight_ddelta_range`` (metres of height per metre of
    range difference: the phase-to-height factor) and ``slant_range``/``incidence`` (deg).
    """
    from pyproj import Transformer

    shape = (ref_ann["lines_per_burst"], cols[1] - cols[0])
    xx, yy = np.meshgrid(dem.x.values, dem.y.values)
    crs = dem.rio.crs
    lon, lat = (Transformer.from_crs(crs, 4326, always_xy=True).transform(xx, yy)
                if crs and not crs.is_geographic else (xx, yy))
    h = np.asarray(dem.values, float)
    ok = np.isfinite(h)
    lat, lon, h = lat[ok], lon[ok], h[ok]

    def project(height):
        tr, rr = geo2rdr(lat, lon, height, ref_ann["orbit"])
        ts, rs = geo2rdr(lat, lon, height, sec_ann["orbit"])
        lr, sr = burst_coordinates(ref_ann, ref_burst, tr, rr)
        ls, ss = burst_coordinates(sec_ann, sec_burst, ts, rs)
        return lr, sr, ls, ss, rr, rs

    lr, sr, ls, ss, rr, rs = project(h)
    # keep DEM points near the window so the triangulation stays small
    near = (lr > -2 * step[0]) & (lr < shape[0] + 2 * step[0]) & (sr > cols[0] - 2 * step[1]) & (sr < cols[1] + 2 * step[1])
    if not near.any():
        raise ValueError("The DEM does not cover the burst window")
    up = (lat[near], lon[near], h[near])
    surf = _scatter_to_grid(lr[near], sr[near],
                            np.c_[ls[near] - lr[near], ss[near] - sr[near], up[2], rs[near] - rr[near], rr[near]],
                            shape, col0=cols[0], step=step)
    flat = []
    for hh in heights:
        tr, r1 =geo2rdr(up[0], up[1], np.full(up[0].shape, hh), ref_ann["orbit"])
        _, r2 = geo2rdr(up[0], up[1], np.full(up[0].shape, hh), sec_ann["orbit"])
        l1, s1 = burst_coordinates(ref_ann, ref_burst, tr, r1)
        flat.append(_scatter_to_grid(l1, s1, (r2 - r1)[:, None], shape, col0=cols[0], step=step)[..., 0])
    lines = np.arange(shape[0], dtype=float)[:, None]
    samples = np.arange(cols[0], cols[1], dtype=float)[None, :]
    slant = surf[..., 4]
    # ellipsoid incidence angle from the satellite height above the local Earth radius
    s_mid = _Orbit(ref_ann["orbit"]).pos(_Orbit(ref_ann["orbit"]).seconds(ref_ann["bursts"][ref_burst]["azimuth_time"]))
    rs_ = np.linalg.norm(s_mid)
    lat0 = np.radians(np.nanmean(up[0]))
    re_ = np.linalg.norm(llh_to_ecef(np.degrees(lat0), np.nanmean(up[1]), 0.0))
    cos_look = (rs_**2 + slant**2 - re_**2) / (2 * rs_ * slant)
    look = np.arccos(np.clip(cos_look, -1, 1))
    incidence = np.degrees(np.arcsin(np.clip(rs_ / re_ * np.sin(look), -1, 1)))
    return {
        "sec_line": lines + surf[..., 0].astype("float64"),
        "sec_sample": samples + surf[..., 1].astype("float64"),
        "dem_height": surf[..., 2].astype("float32"),
        "delta_range": surf[..., 3],
        "flat_delta_range": flat[0],
        "dheight_ddelta_range": ((heights[1] - heights[0]) / (flat[1] - flat[0])).astype("float32"),
        "slant_range": slant.astype("float32"),
        "incidence": incidence.astype("float32"),
    }


# ----------------------------------------------------------------------------- TOPS deramping
def _nearest_poly(polys: list, t: np.datetime64):
    return polys[int(np.argmin([abs((p[0] - t) / np.timedelta64(1, "s")) for p in polys]))]


def _tops_terms(ann: dict, burst: int, samples: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Doppler-centroid rate ``kt``, Doppler centroid ``fdc`` and reference time ``eta_ref`` at
    range ``samples`` for one burst (Sentinel-1 TOPS deramping, ESA S1-TN-MDA-52-7445)."""
    b = ann["bursts"][burst]
    dt, lpb, fs = ann["azimuth_time_interval"], ann["lines_per_burst"], ann["range_sampling_rate"]
    t_mid = b["azimuth_time"] + np.timedelta64(int(lpb / 2 * dt * 1e9), "ns")
    _, fm_t0, fm_c = _nearest_poly(ann["fm_rate"], t_mid)
    _, dc_t0, dc_c = _nearest_poly(ann["doppler"], t_mid)
    orb = _Orbit(ann["orbit"])
    vs = float(np.linalg.norm(orb.vel(orb.seconds(t_mid))))
    ks = 2 * vs / SPEED_OF_LIGHT * ann["radar_frequency"] * ann["azimuth_steering_rate"]

    def at(sample):
        tau = ann["slant_range_time"] + np.asarray(sample, float) / fs
        ka = np.polynomial.polynomial.polyval(tau - fm_t0, fm_c)
        fdc = np.polynomial.polynomial.polyval(tau - dc_t0, dc_c)
        return ka, fdc

    ka, fdc = at(samples)
    ka_mid, fdc_mid = at(ann["samples_per_burst"] / 2)
    kt = ka * ks / (ka - ks)
    eta_ref = -fdc / ka + fdc_mid / ka_mid
    return kt, fdc, eta_ref


def tops_deramp_phase(ann: dict, burst: int, lines: Any, samples: Any) -> np.ndarray:
    """TOPS azimuth deramping phase ``phi(eta, tau)`` (radians) at burst ``lines``/``samples``.

    The TOPS beam sweeps forward during each burst, so the azimuth spectrum of an SLC drifts
    by several kHz across the burst -- far beyond the ~486 Hz azimuth sampling rate. The signal
    carries ``exp(-1j * phi)`` (instantaneous frequency = the Doppler centroid), so multiplying
    by ``exp(+1j * phi)`` brings the spectrum to baseband for interpolation (co-registration) and
    ``exp(-1j * phi)`` at the new positions restores it:
    ``phi = -pi kt (eta - eta_ref)^2 - 2 pi fdc (eta - eta_ref)``, with ``eta`` the azimuth time
    from mid-burst, ``kt`` the Doppler-centroid rate, ``fdc`` the Doppler centroid and
    ``eta_ref`` the beam-centre reference time (ESA S1-TN-MDA-52-7445).
    """
    kt, fdc, eta_ref = _tops_terms(ann, burst, samples)
    eta = (np.asarray(lines, float) - ann["lines_per_burst"] / 2) * ann["azimuth_time_interval"]
    d = eta - eta_ref
    return -np.pi * kt * d**2 - 2 * np.pi * fdc * d


def tops_doppler_centroid(ann: dict, burst: int, lines: Any, samples: Any) -> np.ndarray:
    """Doppler centroid (Hz) of each burst pixel: ``fdc + kt (eta - eta_ref)`` -- the frequency
    that :func:`esd_azimuth_shift` uses to turn an overlap phase into an azimuth shift."""
    kt, fdc, eta_ref = _tops_terms(ann, burst, samples)
    eta = (np.asarray(lines, float) - ann["lines_per_burst"] / 2) * ann["azimuth_time_interval"]
    return fdc + kt * (eta - eta_ref)


def estimate_doppler_centroid(slc: np.ndarray, azimuth_time_interval: float, *, block: int = 50
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Measured azimuth Doppler centroid (Hz) of an SLC in blocks of ``block`` lines, from the
    phase of the lag-one azimuth correlation ``sum(s[i+1] conj(s[i]))`` (Madsen 1989). Aliased to
    +-PRF/2: a raw TOPS burst sweeps through several kHz, a deramped one stays near 0.
    Returns ``(block centre line, doppler)``."""
    s = np.asarray(slc)
    centres, f = [], []
    for r0 in range(0, s.shape[0] - block, block):
        a = s[r0:r0 + block]
        c = np.sum(a[1:] * np.conj(a[:-1]))
        if c != 0:
            centres.append(r0 + block / 2)
            f.append(np.angle(c) / (2 * np.pi * azimuth_time_interval))
    return np.asarray(centres), np.asarray(f)


def resample_slc(slc: np.ndarray, line: np.ndarray, sample: np.ndarray, *, ann: dict | None = None,
                 burst: int = 0, col0: int = 0, order: int = 3) -> np.ndarray:
    """Resample a secondary burst ``slc`` (lines from 0, columns from ``col0``) at fractional
    ``line``/``sample`` positions (from :func:`back_geocode`), with TOPS deramping/reramping when
    ``ann`` is given. Spline interpolation (``order`` 3) of the real and imaginary parts;
    positions outside the burst become 0."""
    from scipy import ndimage

    data = np.array(slc, dtype="complex64", copy=True)  # deramped in place below: never touch the caller's array
    if ann is not None:
        cc = np.arange(col0, col0 + data.shape[1], dtype=float)
        for r0 in range(0, data.shape[0], 256):
            rr = np.arange(r0, min(r0 + 256, data.shape[0]), dtype=float)[:, None]
            data[r0:r0 + 256] *= np.exp(1j * tops_deramp_phase(ann, burst, rr, cc[None, :])).astype("complex64")
    coords = np.stack([np.asarray(line, float), np.asarray(sample, float) - col0])
    out = np.empty(coords.shape[1:], "complex64")
    out.real = ndimage.map_coordinates(data.real, coords, order=order, mode="constant", cval=np.nan)
    out.imag = ndimage.map_coordinates(data.imag, coords, order=order, mode="constant", cval=np.nan)
    if ann is not None:
        out *= np.exp(-1j * tops_deramp_phase(ann, burst, line, sample)).astype("complex64")
    # invalid (zero-filled) secondary samples
    valid = ndimage.map_coordinates((np.asarray(slc) != 0).astype("float32"), coords, order=0,
                                    mode="constant", cval=0) > 0
    out[~valid | ~np.isfinite(out)] = 0
    return out


# ----------------------------------------------------------------------------- ESD (Step 5)
def _overlap_rows(ann: dict, burst_a: int, burst_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Line indices of the azimuth overlap between consecutive bursts ``a`` and ``b`` (``a``
    first), matched to the nearest line in each burst."""
    dt = ann["azimuth_time_interval"]
    ba, bb = ann["bursts"][burst_a], ann["bursts"][burst_b]
    shift = int(round((bb["azimuth_time"] - ba["azimuth_time"]) / np.timedelta64(1, "s") / dt))
    rows_a = np.arange(max(bb["first_valid_line"] + shift, ba["first_valid_line"]), ba["last_valid_line"] + 1)
    return rows_a, rows_a - shift


def esd_azimuth_shift(ref_bursts: list[np.ndarray], sec_bursts: list[np.ndarray], ann: dict,
                      bursts: list[int], *, col0: int = 0, coherence_threshold: float = 0.3,
                      window: tuple[int, int] = (3, 15), anns: list[dict] | None = None) -> dict[str, Any]:
    """Enhanced Spectral Diversity: the residual azimuth misregistration of the resampled
    secondary, from the burst overlaps.

    In each overlap the same ground is imaged twice, once looking forward (end of burst n) and
    once looking backward (start of burst n+1), with Doppler centroids several kHz apart. An
    azimuth misregistration ``delta`` (seconds) adds a phase ``-2 pi f_DC delta`` to each
    interferogram, so the double difference ``arg(ifg_n conj(ifg_n+1))`` equals
    ``-2 pi (f_n - f_n+1) delta``: a 0.001-pixel error already gives a few degrees.

    ``ref_bursts``/``sec_bursts`` are the reference and *resampled* secondary bursts (same
    window, columns from ``col0``); ``ann`` is the reference annotation (``anns`` one per burst
    when the bursts come from separate files). Returns ``shift_lines`` (mean misregistration in
    lines: re-resample the secondary at ``sec_line - shift_lines``), ``shift_seconds``,
    ``phase`` and ``doppler_difference`` per overlap, and ``coherent_pixels``.
    """
    from scipy import ndimage

    anns = anns or [ann] * len(bursts)
    out = {"phase": [], "doppler_difference": [], "shift_seconds": [], "coherent_pixels": []}
    for k in range(len(bursts) - 1):
        a, b = bursts[k], bursts[k + 1]
        ra, rb = _overlap_rows(_merge_burst_ann(anns[k], anns[k + 1], a, b), a, b)
        if ra.size == 0:
            continue
        cols = np.arange(col0, col0 + ref_bursts[k].shape[1], dtype=float)
        m0, s0 = ref_bursts[k][ra], sec_bursts[k][ra]
        m1, s1 = ref_bursts[k + 1][rb], sec_bursts[k + 1][rb]
        i0, i1 = m0 * np.conj(s0), m1 * np.conj(s1)
        dd = i0 * np.conj(i1)

        def box(x):
            return ndimage.uniform_filter(x.real, window) + 1j * ndimage.uniform_filter(x.imag, window)

        num = np.abs(box(i0)) * np.abs(box(i1))
        den = np.sqrt(ndimage.uniform_filter(np.abs(m0) ** 2, window) * ndimage.uniform_filter(np.abs(s0) ** 2, window)
                      * ndimage.uniform_filter(np.abs(m1) ** 2, window) * ndimage.uniform_filter(np.abs(s1) ** 2, window))
        coh = np.sqrt(num / np.maximum(den, 1e-30))
        good = (coh > coherence_threshold) & (m0 != 0) & (s0 != 0) & (m1 != 0) & (s1 != 0)
        f0 = tops_doppler_centroid(anns[k], a, ra[:, None].astype(float), cols[None, :])
        f1 = tops_doppler_centroid(anns[k + 1], b, rb[:, None].astype(float), cols[None, :])
        df = float(np.mean((f0 - f1)[good])) if good.any() else float(np.mean(f0 - f1))
        phi = float(np.angle(np.sum(box(dd)[good]))) if good.any() else np.nan
        out["phase"].append(phi)
        out["doppler_difference"].append(df)
        out["shift_seconds"].append(-phi / (2 * np.pi * df))
        out["coherent_pixels"].append(int(good.sum()))
    w = np.asarray(out["coherent_pixels"], float)
    sh = np.asarray(out["shift_seconds"], float)
    mean = float(np.nansum(sh * w) / np.nansum(w)) if w.sum() else np.nan
    out["shift_seconds_mean"] = mean
    out["shift_lines"] = mean / ann["azimuth_time_interval"]
    return out


def _merge_burst_ann(ann_a: dict, ann_b: dict, a: int, b: int) -> dict:
    """An annotation whose burst list holds both bursts (for bursts read from separate files)."""
    if ann_a is ann_b:
        return ann_a
    merged = dict(ann_a)
    bl = list(ann_a["bursts"])
    while len(bl) <= max(a, b):
        bl.append(None)
    bl[a], bl[b] = ann_a["bursts"][a], ann_b["bursts"][b]
    merged["bursts"] = bl
    return merged


# ----------------------------------------------------------------------------- deburst (Step 7)
def deburst(bursts: list[np.ndarray], ann: dict, burst_ids: list[int], *,
            anns: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``TOPS-Deburst``: join consecutive bursts (arrays on the same column window, lines from
    0) into one continuous image, cutting each azimuth overlap at its middle and dropping
    invalid lines. Works on SLCs, interferograms or any per-pixel layer.

    Returns ``(image, azimuth_time)`` with the zero-Doppler time of every output line
    (consecutive bursts' line grids differ by a fraction of a line; the times keep this exact
    for geocoding).
    """
    anns = anns or [ann] * len(bursts)
    dt = np.timedelta64(int(round(ann["azimuth_time_interval"] * 1e9)), "ns")
    pieces, times = [], []
    cut_prev = None
    for k, (arr, b) in enumerate(zip(bursts, burst_ids)):
        info = anns[k]["bursts"][b]
        n = arr.shape[0]
        t = info["azimuth_time"] + np.arange(n) * dt
        keep = (np.arange(n) >= info["first_valid_line"]) & (np.arange(n) <= info["last_valid_line"])
        if cut_prev is not None:
            keep &= t >= cut_prev
        if k + 1 < len(bursts):
            nxt = anns[k + 1]["bursts"][burst_ids[k + 1]]
            start_next = nxt["azimuth_time"] + nxt["first_valid_line"] * dt
            end_this = info["azimuth_time"] + info["last_valid_line"] * dt
            cut = start_next + (end_this - start_next) / 2
            keep &= t < cut
            cut_prev = cut
        pieces.append(arr[keep])
        times.append(t[keep])
    return np.concatenate(pieces, axis=0), np.concatenate(times)


# ----------------------------------------------------------------------------- terrain correction (Step 12)
def terrain_correct(layers: dict[str, np.ndarray] | np.ndarray, azimuth_time: np.ndarray,
                    near_range: float, range_spacing: float, orbit: dict, dem: xr.DataArray, *,
                    order: int = 1) -> xr.Dataset | xr.DataArray:
    """``Range-Doppler Terrain Correction``: resample radar-geometry layers onto a DEM's map grid.

    Every DEM cell (``dem``: *ellipsoidal* heights, any CRS) is located in radar geometry with
    :func:`geo2rdr` -- its zero-Doppler time gives the row (``azimuth_time``: time of each
    row, e.g. from :func:`deburst` after multilooking) and its slant range the column
    (``near_range`` is the slant range of column 0, ``range_spacing`` the column spacing in
    metres) -- and the layers are sampled there. Terrain displacement (foreshortening, layover)
    is removed because the DEM's own heights are used. Returns a Dataset (or a DataArray for a
    single array) on the DEM grid, with the DEM's CRS.
    """
    from pyproj import Transformer
    from scipy import ndimage

    xx, yy = np.meshgrid(dem.x.values, dem.y.values)
    crs = dem.rio.crs
    lon, lat = (Transformer.from_crs(crs, 4326, always_xy=True).transform(xx, yy)
                if crs and not crs.is_geographic else (xx, yy))
    h = np.asarray(dem.values, float)
    t, r = geo2rdr(lat, lon, np.nan_to_num(h, nan=float(np.nanmedian(h))), orbit)
    tsec = (np.asarray(azimuth_time, "datetime64[ns]") - azimuth_time[0]) / np.timedelta64(1, "s")
    q = (t - azimuth_time[0]) / np.timedelta64(1, "s")
    idx = np.arange(tsec.size)
    if tsec.size > 1 and tsec[1] < tsec[0]:  # row time decreasing (e.g. some RCM products): np.interp needs ascending xp
        tsec, idx = tsec[::-1], idx[::-1]
    row = np.interp(q.ravel(), tsec, idx, left=np.nan, right=np.nan).reshape(q.shape)
    col = (r - near_range) / range_spacing
    single = not isinstance(layers, dict)
    layers = {"value": layers} if single else layers
    out = {}
    for name, arr in layers.items():
        arr = np.asarray(arr)
        coords = np.stack([np.nan_to_num(row, nan=-1e6), np.nan_to_num(col, nan=-1e6)])
        if np.iscomplexobj(arr):
            v = (ndimage.map_coordinates(arr.real, coords, order=order, cval=np.nan)
                 + 1j * ndimage.map_coordinates(arr.imag, coords, order=order, cval=np.nan))
        else:
            v = ndimage.map_coordinates(np.asarray(arr, float), coords, order=order, cval=np.nan)
        v[~np.isfinite(h)] = np.nan
        out[name] = (("y", "x"), v)
    ds = xr.Dataset(out, coords={"y": dem.y, "x": dem.x})
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    ds["radar_row"] = (("y", "x"), row.astype("float32"))
    ds["radar_col"] = (("y", "x"), col.astype("float32"))
    return ds["value"] if single else ds


def layover_shadow_mask(dem: np.ndarray, resolution: float, heading_deg: float, incidence_deg: Any,
                        *, look: Literal["right", "left"] = "right") -> np.ndarray:
    """Local layover/shadow classes on a projected DEM grid: 0 = fine, 1 = layover (terrain
    facing the radar steeper than the incidence angle), 2 = shadow (terrain facing away steeper
    than 90 deg minus the incidence angle). A per-cell slope test; distant shadows cast by
    ridges further along the range line are not traced."""
    dem = np.asarray(dem, float)
    dzdy, dzdx = np.gradient(dem, resolution)
    dzdy = -dzdy  # rows run north to south on a north-up grid
    look_az = np.radians(heading_deg + (90 if look == "right" else -90))
    # slope along the ground-range direction, positive where the terrain rises away from the radar
    rise = dzdx * np.sin(look_az) + dzdy * np.cos(look_az)
    slope_towards = np.degrees(np.arctan(rise))
    inc = np.broadcast_to(np.asarray(incidence_deg, float), dem.shape)
    out = np.zeros(dem.shape, "uint8")
    out[slope_towards > inc] = 1
    out[-slope_towards > 90 - inc] = 2
    return out


# ----------------------------------------------------------------------------- stacks (DInSAR / SBAS)
def los_unit_vector(lat: Any, lon: Any, height: Any, orbit: dict) -> np.ndarray:
    """Unit vector from ground points to the satellite (at each point's zero-Doppler time), in
    local east/north/up components, shape ``(..., 3)``.

    Project a 3-D displacement onto it to compare GNSS with InSAR: ``d_los = d_e e + d_n n +
    d_u u`` is positive for motion *toward* the satellite (the convention of
    :func:`geolattice.rsense.insar.phase_to_los`).
    """
    lat_a, lon_a, h_a = (np.asarray(v, float) for v in (lat, lon, height))
    lat_a, lon_a, h_a = np.broadcast_arrays(lat_a, lon_a, h_a)
    t, _ = geo2rdr(lat_a, lon_a, h_a, orbit)
    orb = _Orbit(orbit)
    S = orb.pos(orb.seconds(t).ravel()).reshape(lat_a.shape + (3,))
    P = llh_to_ecef(lat_a, lon_a, h_a)
    los = (S - P) / np.linalg.norm(S - P, axis=-1, keepdims=True)
    la, lo = np.radians(lat_a), np.radians(lon_a)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    return np.stack([np.sum(los * east, -1), np.sum(los * north, -1), np.sum(los * up, -1)], -1)


def coregister_stack(reference: dict, secondaries: dict, dem: xr.DataArray, *, esd: bool = True,
                     order: int = 3) -> dict[str, Any]:
    """Co-register a stack of SLC acquisitions to one reference date (SNAP's back-geocoding of a
    stack, ISCE's ``topsStack``), with Enhanced Spectral Diversity for each secondary.

    ``reference`` and every value of ``secondaries`` (keyed by date or name) describe the same
    consecutive bursts as dicts of per-burst lists: ``"slc"`` (complex arrays, e.g. from
    :func:`read_slc_burst`), ``"ann"`` (annotations with precise orbits, see
    :func:`apply_orbit_file`), ``"burst"`` (burst index in each annotation) and ``"col0"``
    (first range sample of each array). Every burst of every secondary is back-geocoded onto the
    reference window with :func:`back_geocode` and resampled with :func:`resample_slc`; with
    ``esd`` and two or more bursts, :func:`esd_azimuth_shift` measures the residual azimuth
    misregistration and the secondary is resampled again with the correction.

    Returns ``{"geometry": per-burst lists of the reference geometry ("dem_height",
    "dheight_ddelta_range", "slant_range", "incidence"), "stack": {key: {"slc": [...],
    "delta_range": [...] (float32, secondary minus reference slant range at the DEM surface:
    the input of :func:`geolattice.rsense.insar.topographic_phase`), "flat_delta_range": [...],
    "kz": [...] (metres of range difference per metre of height, the DEM-error sensitivity that
    PS-InSAR estimates; see :func:`geolattice.rsense.insar.ps_periodogram`), "esd": ESD result or None}}}``. Any two stack members ``i``, ``j`` then form a differential
    interferogram with ``delta_range_j - delta_range_i`` (the reference itself has 0).
    """
    nb = len(reference["slc"])
    cols = [(int(reference["col0"][k]), int(reference["col0"][k]) + reference["slc"][k].shape[1]) for k in range(nb)]
    out: dict[str, Any] = {"geometry": None, "stack": {}}
    for key, sec in secondaries.items():
        geoms, res = [], []
        for k in range(nb):
            g = back_geocode(reference["ann"][k], sec["ann"][k], reference["burst"][k], sec["burst"][k], dem,
                             cols=cols[k])
            geoms.append(g)
            res.append(resample_slc(sec["slc"][k], g["sec_line"], g["sec_sample"], ann=sec["ann"][k],
                                    burst=sec["burst"][k], col0=int(sec["col0"][k]), order=order))
        e = None
        if esd and nb > 1:
            e = esd_azimuth_shift(reference["slc"], res, reference["ann"][0], list(reference["burst"]),
                                  col0=cols[0][0], anns=list(reference["ann"]))
            if np.isfinite(e["shift_lines"]):
                res = [resample_slc(sec["slc"][k], geoms[k]["sec_line"] - e["shift_lines"], geoms[k]["sec_sample"],
                                    ann=sec["ann"][k], burst=sec["burst"][k], col0=int(sec["col0"][k]), order=order)
                       for k in range(nb)]
        if out["geometry"] is None:
            out["geometry"] = {name: [g[name] for g in geoms]
                               for name in ("dem_height", "dheight_ddelta_range", "slant_range", "incidence")}
        out["stack"][key] = {"slc": res,
                             "delta_range": [g["delta_range"].astype("float32") for g in geoms],
                             "flat_delta_range": [g["flat_delta_range"].astype("float32") for g in geoms],
                             "kz": [(1.0 / g["dheight_ddelta_range"]).astype("float32") for g in geoms],
                             "esd": e}
    return out


def geocode_points(rows: Any, cols: Any, tc: xr.Dataset, *, height_error: Any | None = None,
                   incidence_deg: Any | None = None, heading_deg: float | None = None,
                   look: Literal["right", "left"] = "right") -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates of radar-geometry points (e.g. persistent scatterers), from the
    ``radar_row``/``radar_col`` layers of a :func:`terrain_correct` result ``tc``: the map
    position of each point is interpolated (linearly, between map cells) from where every map
    cell lies in the radar image. ``rows``/``cols`` must use the same (multilooked or full)
    radar grid that ``tc`` was computed for.

    A scatterer that is ``height_error`` metres above the DEM (a rooftop, a mast) has the slant
    range of a DEM point nearer the radar, so plain geocoding places it too close to the sensor;
    with ``height_error``, ``incidence_deg`` and ``heading_deg`` each point is moved
    ``height_error / tan(incidence)`` along the look direction, which is how PS-InSAR's DEM-error
    estimate improves geolocation. Returns ``(x, y)`` in ``tc``'s CRS.
    """
    from scipy.interpolate import LinearNDInterpolator

    rr = np.asarray(tc["radar_row"].values, float)
    cc = np.asarray(tc["radar_col"].values, float)
    X, Y = np.meshgrid(tc.x.values, tc.y.values)
    ok = np.isfinite(rr) & np.isfinite(cc) & (rr > -1e5) & (cc > -1e5)
    interp = LinearNDInterpolator(np.c_[rr[ok], cc[ok]], np.c_[X[ok], Y[ok]])
    xy = interp(np.c_[np.ravel(rows), np.ravel(cols)])
    x, y = xy[:, 0], xy[:, 1]
    if height_error is not None:
        if incidence_deg is None or heading_deg is None:
            raise ValueError("height_error needs incidence_deg and heading_deg")
        shift = np.ravel(height_error) / np.tan(np.radians(np.ravel(incidence_deg) * np.ones_like(x)))
        look_az = np.radians(heading_deg + (90 if look == "right" else -90))
        x = x + shift * np.sin(look_az)
        y = y + shift * np.cos(look_az)
    return x.reshape(np.shape(rows)), y.reshape(np.shape(rows))
