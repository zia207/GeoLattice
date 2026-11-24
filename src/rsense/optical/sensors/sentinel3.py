"""Sentinel-3: OLCI, SLSTR, SRAL and Synergy product processing.

Sentinel-3 does not fit the lazy-STAC-raster loader the other sensor modules share
(:func:`~geolattice.rsense.optical.sensors.load_landsat` and friends): its products are NetCDF-4
files in satellite swath geometry, not georeferenced COGs, and each instrument's product
family (OLCI, SLSTR, SRAL, Synergy) has its own files, flags and physical units. This module
instead gives you one reader and mask function per product type, following the standard
workflow: read, mask by quality flags, convert units, reproject from swath to a map grid,
then composite.

Input is an extracted ``.SEN3`` product folder (what you get after unzipping a Copernicus
Data Space Ecosystem or EUMETSAT download; :func:`geolattice.rsense.get_images.get_images` writes
these for ``source="cdse"``) or, for the single-file products (SST, FRP, Synergy VGT), the
NetCDF file itself. Reading NetCDF needs the optional ``netCDF4`` or ``h5netcdf`` package,
installed with ``pip install geolattice[rsense-sentinel3]``.

Product collection names, matching :data:`geolattice.rsense.get_images.CDSE_TYPES`:
OL_1_EFR/ERR (OLCI L1 radiance), OL_2_LFR/LRR (OLCI L2 land), OL_2_WFR/WRR (OLCI L2 water),
SL_1_RBT (SLSTR L1 radiance/BT), SL_2_LST (land surface temperature),
SL_2_WST (sea surface temperature), SL_2_FRP (fire radiative power),
SR_2_WAT/LAN (SRAL altimetry), SY_2_SYN (synergy reflectance),
SY_2_VGP/VG1/V10 (synergy vegetation, already gridded).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "OLCI_BANDS",
    "OLCI_L1_DEFAULT_EXCLUDE",
    "OLCI_LAND_VARS",
    "OLCI_LFR_DEFAULT_EXCLUDE",
    "OLCI_WATER_VARS",
    "OLCI_WFR_DEFAULT_EXCLUDE",
    "SLSTR_GRID_SUFFIXES",
    "aggregate_frp",
    "decode_flags",
    "flag_mask",
    "frp_to_points",
    "kelvin_to_celsius",
    "log_to_linear",
    "olci_l1_mask",
    "olci_land_mask",
    "olci_toa_reflectance",
    "olci_water_mask",
    "read_manifest",
    "read_olci_l1",
    "read_olci_land",
    "read_olci_water",
    "read_sral_marine",
    "read_sral_virtual_station",
    "read_slstr_frp",
    "read_slstr_l1",
    "read_slstr_lst",
    "read_slstr_sst",
    "read_synergy",
    "read_synergy_vgt",
    "slstr_flag_mask",
    "slstr_lst_mask",
    "slstr_reflectance",
    "sral_marine_mask",
    "sst_bias_correct",
    "sst_quality_mask",
    "swath_to_grid",
    "synergy_mask",
]

# Centre wavelength (nm) of each OLCI band. Datasets built by this module use the lowercase
# "oa01".."oa21" variable names (dropping the file's "_radiance"/"_reflectance" suffix), which
# bands.SENSOR_BANDS["sentinel3_olci"] maps onto the shared canonical names (red, nir, ...).
OLCI_BANDS = {
    "oa01": 400.0, "oa02": 412.5, "oa03": 442.5, "oa04": 490.0, "oa05": 510.0, "oa06": 560.0,
    "oa07": 620.0, "oa08": 665.0, "oa09": 673.75, "oa10": 681.25, "oa11": 708.75, "oa12": 753.75,
    "oa13": 761.25, "oa14": 764.375, "oa15": 767.5, "oa16": 778.75, "oa17": 865.0, "oa18": 885.0,
    "oa19": 900.0, "oa20": 940.0, "oa21": 1020.0,
}

# SLSTR file-name grid suffixes (section 5 of the processing guide).
SLSTR_GRID_SUFFIXES = {
    "an": "500 m nadir, VIS/SWIR stripe A (S1-S6)", "bn": "500 m nadir, VIS/SWIR stripe B (S1-S6)",
    "cn": "500 m nadir, VIS/SWIR stripe TDI (S1-S6)", "in": "1 km nadir, thermal (S7-S9) and fire",
    "ao": "500 m oblique, VIS/SWIR stripe A", "bo": "500 m oblique, VIS/SWIR stripe B",
    "io": "1 km oblique, thermal", "fn": "1 km nadir fire channels (F1, F2)",
    "fo": "1 km oblique fire channels",
}

# Recommended flag exclusions per product (guide sections 2-8). Pass to the matching
# ``*_mask`` function's ``exclude=`` to override, or inspect with :func:`decode_flags`.
OLCI_L1_DEFAULT_EXCLUDE = ("invalid", "saturated", "bright")
OLCI_WFR_DEFAULT_EXCLUDE = ("LAND", "CLOUD", "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "INVALID",
                           "SNOW_ICE", "SUSPECT", "HISOLZEN", "SATURATED", "HIGHGLINT",
                           "WHITECAPS", "AC_FAIL")
OLCI_WFR_OC4ME_EXCLUDE = (*OLCI_WFR_DEFAULT_EXCLUDE, "OC4ME_FAIL")
OLCI_LFR_DEFAULT_EXCLUDE = ("CLOUD", "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LAND", "WATER",
                           "SNOW_ICE", "INVALID")
SLSTR_L1_DEFAULT_EXCLUDE = ("cloud", "summary_cloud")
SLSTR_LST_DEFAULT_EXCLUDE = ("cloud", "summary_cloud")


# --------------------------------------------------------------------------- common workflow
def read_manifest(folder: str | PathLike) -> dict[str, str]:
    """Processing baseline and key attributes from a ``.SEN3`` product's ``xfdumanifest.xml``."""
    import xml.etree.ElementTree as ET

    path = Path(folder) / "xfdumanifest.xml"
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("processingBaseline", "productName", "startTime", "stopTime",
                  "productType", "creationTime", "platform", "nssdcIdentifier") and el.text:
            out[tag] = el.text.strip()
        for key in ("processingLevel", "timeliness"):
            if key in el.attrib:
                out[key] = el.attrib[key]
    return out


def _open(paths: Iterable[str | PathLike], **kwargs: Any) -> xr.Dataset:
    paths = [Path(p) for p in paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Sentinel-3 file(s): {missing}")
    datasets = [xr.open_dataset(p, decode_cf=True, mask_and_scale=True, **kwargs) for p in paths]
    return xr.merge(datasets, compat="override", join="override")


def decode_flags(flag: xr.DataArray) -> dict[str, xr.DataArray]:
    """Every named bit of a Sentinel-3 flag variable, read from its own metadata.

    Uses the variable's ``flag_masks`` and ``flag_meanings`` attributes (CF convention),
    never a hardcoded bit position, since the bit layout differs by product and baseline.
    """
    masks = flag.attrs.get("flag_masks")
    meanings = flag.attrs.get("flag_meanings")
    if masks is None or meanings is None:
        raise ValueError(f"{flag.name!r} has no flag_masks/flag_meanings attributes; "
                         "it may not be a flag variable, or xarray stripped them on open "
                         "(pass decode_cf=False when opening).")
    names = str(meanings).split()
    raw = flag.values if np.issubdtype(flag.dtype, np.integer) else flag.fillna(0).astype("int64")
    return {name: xr.DataArray((raw & int(m)).astype(bool), dims=flag.dims, coords=flag.coords)
           for name, m in zip(names, np.atleast_1d(masks))}


def flag_mask(flag: xr.DataArray, exclude: Iterable[str], *, match: str = "exact") -> xr.DataArray:
    """Boolean validity mask (``True`` = usable) excluding the named flags.

    ``match="exact"`` requires the flag meaning to equal an entry in ``exclude``;
    ``match="contains"`` excludes a flag if any entry in ``exclude`` is a substring of its
    name (handy for SLSTR, whose per-channel flags are named e.g. ``cloud_in``).
    """
    flags = decode_flags(flag)
    exclude = list(exclude)
    if match == "contains":
        hit = [name for name in flags if any(e.lower() in name.lower() for e in exclude)]
    else:
        wanted = {e.lower() for e in exclude}
        hit = [name for name in flags if name.lower() in wanted]
    valid = xr.ones_like(flag, dtype=bool)
    for name in hit:
        valid &= ~flags[name]
    return valid.rename("valid")


def swath_to_grid(obj: xr.Dataset | xr.DataArray, lon: xr.DataArray, lat: xr.DataArray, *,
                  resolution: float, bounds: tuple[float, float, float, float] | None = None,
                  crs: Any = 4326, method: str = "nearest",
                  max_distance: float | None = None) -> xr.Dataset | xr.DataArray:
    """Resample swath data (per-pixel ``lon``/``lat``) onto a regular, georeferenced grid.

    Run this once, at the end of your processing, after masking and unit conversion — not
    before, so you never interpolate across a cloud edge or resample the same pixel twice.

    ``resolution`` is in the units of ``crs`` (degrees for EPSG:4326). ``bounds`` is
    ``(minx, miny, maxx, maxy)``; by default it is taken from ``lon``/``lat``.
    ``method="nearest"`` uses a KD-tree (fast, works for any swath size); ``"linear"`` uses
    Delaunay triangulation (:func:`scipy.interpolate.griddata`, smoother but slower and
    memory-heavy on a full OLCI/SLSTR scene — subset first).
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    lon_v, lat_v = np.asarray(lon.values).ravel(), np.asarray(lat.values).ravel()
    ok = np.isfinite(lon_v) & np.isfinite(lat_v)
    lon_v, lat_v = lon_v[ok], lat_v[ok]
    if bounds is None:
        bounds = (float(np.nanmin(lon_v)), float(np.nanmin(lat_v)),
                 float(np.nanmax(lon_v)), float(np.nanmax(lat_v)))
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx + resolution / 2, maxx, resolution)
    ys = np.arange(maxy - resolution / 2, miny, -resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    if method == "nearest":
        tree = cKDTree(np.column_stack([lon_v, lat_v]))
        dist, idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
        if max_distance is not None:
            idx = np.where(dist <= max_distance, idx, -1)

    def _one(da: xr.DataArray) -> xr.DataArray:
        v = np.asarray(da.values).ravel()[ok]
        if method == "nearest":
            out = np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], np.nan).reshape(grid_x.shape)
        else:
            out = griddata((lon_v, lat_v), v, (grid_x, grid_y), method=method)
        return xr.DataArray(out.astype(da.dtype if np.issubdtype(da.dtype, np.floating) else "float32"),
                           dims=("y", "x"), coords={"y": ys, "x": xs}, name=da.name, attrs=da.attrs)

    if isinstance(obj, xr.DataArray):
        out = _one(obj)
    else:
        out = xr.Dataset({name: _one(da) for name, da in obj.data_vars.items()
                          if set(da.dims) >= set(lon.dims)}, attrs=obj.attrs)
    return out.rio.write_crs(crs)


def log_to_linear(da: xr.DataArray) -> xr.DataArray:
    """Back-transform a log10-scaled variable (OLCI CHL, TSM): ``10**da``.

    Average the *linear* values, not the stored log values; they give a different (biased)
    mean. Apply this before compositing or statistics.
    """
    # "back_transform", not "transform": rioxarray reserves the "transform" attribute name
    # for a GDAL-style affine geotransform and crashes on write_crs() if it holds anything else.
    return (10.0 ** da).astype("float32").assign_attrs({**da.attrs, "back_transform": "10**x"})


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Kelvin to degrees Celsius, keeping other attributes."""
    return (da - 273.15).astype("float32").assign_attrs({**da.attrs, "units": "degC"})


def olci_toa_reflectance(radiance: xr.DataArray, solar_flux: xr.DataArray,
                         solar_zenith: xr.DataArray) -> xr.DataArray:
    """OLCI L1 top-of-atmosphere reflectance: ``rho = pi * L / (F0 * cos(theta_s))``.

    ``solar_zenith`` must already be at full resolution (see :func:`read_olci_l1`, which
    interpolates it from the coarser tie-point grid).
    """
    rho = np.pi * radiance / (solar_flux * np.cos(np.radians(solar_zenith)))
    return rho.astype("float32").rename(radiance.name).assign_attrs(units="reflectance")


def _oa_name(b: str) -> str:
    """``"oa01"`` -> ``"Oa01"``, matching OLCI's file and variable naming."""
    return b.capitalize()


def _interp_tie_point(tie: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Upsample a coarser tie-point array (e.g. sun/view geometry) onto ``like``'s grid.

    Tie-point angle fields vary smoothly across a scene, so linear interpolation (what ESA's
    own baseline processing uses) is safe here, unlike for reflectance or flags.
    """
    from scipy.ndimage import zoom

    dims = like.dims[-2:]
    shape = (like.sizes[dims[0]], like.sizes[dims[1]])
    if tie.shape == shape:
        return tie.rename({tie.dims[-2]: dims[0], tie.dims[-1]: dims[1]})
    factors = (shape[0] / tie.shape[-2], shape[1] / tie.shape[-1])
    out = zoom(np.asarray(tie.values, dtype="float64"), factors, order=1)
    return xr.DataArray(out[: shape[0], : shape[1]], dims=dims)


# --------------------------------------------------------------------------- OLCI L1 (EFR/ERR)
def read_olci_l1(folder: str | PathLike, bands: Sequence[str] | None = None) -> xr.Dataset:
    """Read OLCI Level-1B radiances, solar flux and the full-resolution solar zenith angle.

    Returns a Dataset with ``oa01``..``oa21`` radiance (W m-2 sr-1 um-1), ``sza`` (solar
    zenith, interpolated to full resolution) and ``quality_flags``. The raw per-detector
    ``solar_flux`` array from ``instrument_data.nc`` is included as-is (its exact band/detector
    layout varies by baseline, so index into it yourself rather than trust a guessed mapping)
    for :func:`olci_toa_reflectance`.
    """
    folder = Path(folder)
    wanted = [b.lower() for b in (bands or OLCI_BANDS)]
    files = [folder / f"{_oa_name(b)}_radiance.nc" for b in wanted]
    ds = _open(files)
    ds = ds.rename({f"{_oa_name(b)}_radiance": b for b in wanted
                    if f"{_oa_name(b)}_radiance" in ds.data_vars})
    inst = xr.open_dataset(folder / "instrument_data.nc")
    tie = xr.open_dataset(folder / "tie_geometries.nc")
    qf = xr.open_dataset(folder / "qualityFlags.nc")
    ds["sza"] = _interp_tie_point(tie["SZA"], ds[wanted[0]])
    if "solar_flux" in inst.data_vars:
        ds["solar_flux"] = inst["solar_flux"]
    ds["quality_flags"] = qf["quality_flags"] if "quality_flags" in qf else qf[list(qf.data_vars)[0]]
    return ds


def olci_l1_mask(quality_flags: xr.DataArray, *,
                 exclude: Iterable[str] = OLCI_L1_DEFAULT_EXCLUDE) -> xr.DataArray:
    """Mask from OLCI L1 ``quality_flags`` (invalid, saturated, bright/cloud by default)."""
    return flag_mask(quality_flags, exclude, match="contains")


# --------------------------------------------------------------------------- OLCI L2 water (WFR/WRR)
# File stem -> its real (uppercase) main variable name inside that file. The file stem and
# the variable name only agree by coincidence for some water products (chl_oc4me.nc really
# does hold CHL_OC4ME) and not others (trsp.nc holds KD490_M07; iop_nn.nc holds ADG443_NN) --
# confirmed against real downloaded Planetary Computer files, not assumed from the product spec.
OLCI_WATER_VARS = {"chl_oc4me": "CHL_OC4ME", "chl_nn": "CHL_NN", "tsm_nn": "TSM_NN",
                   "trsp": "KD490_M07", "iop_nn": "ADG443_NN"}


def read_olci_water(folder: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read OLCI L2 water (ocean colour) variables: chlorophyll, TSM, Kd490, CDOM, wqsf.

    ``variables`` picks from ``chl_oc4me``, ``chl_nn``, ``tsm_nn``, ``trsp`` (Kd490),
    ``iop_nn`` (CDOM absorption); omit for all of them. Each comes back under its own
    lowercase name (plus ``{name}_err`` for its per-pixel uncertainty), not the file's
    original uppercase variable name. Reflectance bands (``Oa##_reflectance``) are not read
    by default — pass them explicitly, e.g. ``variables=["chl_nn", "oa08"]``.
    """
    folder = Path(folder)
    default = list(OLCI_WATER_VARS)
    wanted = list(variables) if variables is not None else default
    water_vars, reflectance_bands = [], []
    for v in wanted:
        if v.lower().startswith("oa") and v[2:4].isdigit():
            reflectance_bands.append(v.lower())
        else:
            water_vars.append(v)
    ds = xr.Dataset()
    for v in water_vars:
        real = OLCI_WATER_VARS[v]
        one = xr.open_dataset(folder / f"{v}.nc")
        rename = {real: v}
        if f"{real}_err" in one.data_vars:
            rename[f"{real}_err"] = f"{v}_err"
        ds = ds.merge(one[list(rename)].rename(rename))
    if reflectance_bands:
        refl_files = [folder / f"{_oa_name(b)}_reflectance.nc" for b in reflectance_bands]
        refl = _open(refl_files)
        ds = ds.merge(refl.rename({f"{_oa_name(b)}_reflectance": b for b in reflectance_bands}))
    ds["wqsf"] = xr.open_dataset(folder / "wqsf.nc")["WQSF"]
    geo = xr.open_dataset(folder / "geo_coordinates.nc")
    ds = ds.assign_coords(longitude=geo["longitude"], latitude=geo["latitude"])
    return ds


def olci_water_mask(wqsf: xr.DataArray, *, algorithm: str = "nn",
                    exclude: Iterable[str] | None = None) -> xr.DataArray:
    """Mask from OLCI L2 water ``WQSF``. ``algorithm`` is ``"nn"`` (neural-net products,
    coastal/inland) or ``"oc4me"`` (open ocean), which adds the matching *_FAIL flag."""
    if exclude is not None:
        return flag_mask(wqsf, exclude)
    base = OLCI_WFR_OC4ME_EXCLUDE if algorithm == "oc4me" else OLCI_WFR_DEFAULT_EXCLUDE
    return flag_mask(wqsf, base)


# --------------------------------------------------------------------------- OLCI L2 land (LFR/LRR)
# gifapar.nc, otci.nc and iwv.nc each hold one main variable matching their own uppercase
# name; rc_gifapar.nc is the exception, holding two bands (RC681, RC865) instead of one
# "RC_GIFAPAR" -- confirmed against real downloaded Planetary Computer files.
OLCI_LAND_VARS = {"gifapar": ["GIFAPAR"], "otci": ["OTCI"], "iwv": ["IWV"],
                  "rc_gifapar": ["RC681", "RC865"]}


def read_olci_land(folder: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read OLCI L2 land variables: FAPAR (``gifapar``), canopy chlorophyll (``otci``),
    integrated water vapour (``iwv``), rectified reflectance at 681/865 nm (``rc681``,
    ``rc865``, both read from ``rc_gifapar.nc``) and ``lqsf``."""
    folder = Path(folder)
    default = list(OLCI_LAND_VARS)
    wanted = list(variables) if variables is not None else default
    ds = xr.Dataset()
    for v in wanted:
        real_names = OLCI_LAND_VARS[v]
        one = xr.open_dataset(folder / f"{v}.nc")
        rename = {}
        for real in real_names:
            lower = real.lower()
            rename[real] = lower
            if f"{real}_unc" in one.data_vars:
                rename[f"{real}_unc"] = f"{lower}_unc"
        ds = ds.merge(one[list(rename)].rename(rename))
    ds["lqsf"] = xr.open_dataset(folder / "lqsf.nc")["LQSF"]
    geo = xr.open_dataset(folder / "geo_coordinates.nc")
    return ds.assign_coords(longitude=geo["longitude"], latitude=geo["latitude"])


def olci_land_mask(lqsf: xr.DataArray,
                   exclude: Iterable[str] = OLCI_LFR_DEFAULT_EXCLUDE) -> xr.DataArray:
    """Mask from OLCI L2 land ``LQSF`` (cloud, water, snow/ice, invalid by default)."""
    return flag_mask(lqsf, exclude, match="contains")


# --------------------------------------------------------------------------- SLSTR L1 (RBT)
def read_slstr_l1(folder: str | PathLike, bands: Sequence[str] | None = None, *,
                  view: str = "n") -> xr.Dataset:
    """Read SLSTR L1 radiance (S1-S6) and brightness temperature (S7-S9, F1-F2).

    ``view`` is ``"n"`` (nadir) or ``"o"`` (oblique); VIS/SWIR bands come on the 500 m
    ``_an``/``_bn`` grids and thermal/fire bands on the 1 km ``_in``/``_fn`` grids (see
    :data:`SLSTR_GRID_SUFFIXES`), so this returns two sub-resolutions, not one Dataset —
    co-register them yourself with :meth:`xarray.DataArray.interp_like` before combining.
    """
    folder = Path(folder)
    wanted = [b.upper() for b in (bands or [f"S{i}" for i in range(1, 10)])]
    vis = [b for b in wanted if b in {f"S{i}" for i in range(1, 7)}]
    thermal = [b for b in wanted if b in {"S7", "S8", "S9"}]
    out: dict[str, xr.DataArray] = {}
    if vis:
        vis_ds = _open([folder / f"{b}_radiance_a{view}.nc" for b in vis])
        for b in vis:
            out[b.lower()] = vis_ds[f"{b}_radiance_a{view}"]
            quality_path = folder / f"{b}_quality_a{view}.nc"
            irr = f"{b}_solar_irradiance_a{view}"
            if quality_path.exists():
                quality = xr.open_dataset(quality_path)
                if irr in quality:
                    out[b.lower()].attrs["solar_irradiance"] = quality[irr]
    if thermal:
        th_ds = _open([folder / f"{b}_BT_i{view}.nc" for b in thermal])
        for b in thermal:
            out[b.lower()] = th_ds[f"{b}_BT_i{view}"]
    flags_grid = "in" if thermal and not vis else f"a{view}"
    flags = xr.open_dataset(folder / f"flags_{flags_grid}.nc")
    flag_var = next((v for v in flags.data_vars if "flag" in v.lower()), None)
    ds = xr.Dataset(out)
    if flag_var:
        ds["flags"] = flags[flag_var]
    geo_file = f"geodetic_i{view}.nc" if (thermal and not vis) else f"geodetic_a{view}.nc"
    geo_path = folder / geo_file
    if geo_path.exists():
        geo = xr.open_dataset(geo_path)
        lat = next((v for v in geo.data_vars if "latitude" in v), None)
        lon = next((v for v in geo.data_vars if "longitude" in v), None)
        if lat and lon:
            ds = ds.assign_coords(longitude=geo[lon], latitude=geo[lat])
    return ds


def slstr_reflectance(radiance: xr.DataArray, solar_zenith: xr.DataArray) -> xr.DataArray:
    """SLSTR VIS/SWIR radiance to reflectance, using the band's own ``solar_irradiance``
    attribute (set by :func:`read_slstr_l1`) and the matching solar zenith angle."""
    irradiance = radiance.attrs.get("solar_irradiance")
    if irradiance is None:
        raise ValueError("radiance has no solar_irradiance attribute; read it with read_slstr_l1")
    rho = np.pi * radiance / (irradiance * np.cos(np.radians(solar_zenith)))
    return rho.astype("float32").rename(radiance.name).assign_attrs(units="reflectance")


def slstr_flag_mask(flags: xr.DataArray,
                    exclude: Iterable[str] = SLSTR_L1_DEFAULT_EXCLUDE) -> xr.DataArray:
    """Mask from SLSTR ``flags_an``/``flags_in`` (cloud by default; also carries
    confidence, land/water and day/night bits accessible via :func:`decode_flags`)."""
    return flag_mask(flags, exclude, match="contains")


# --------------------------------------------------------------------------- SLSTR L2: LST
def read_slstr_lst(folder: str | PathLike) -> xr.Dataset:
    """Read SLSTR Level-2 Land Surface Temperature: ``LST``, its uncertainty and flags."""
    folder = Path(folder)
    ds = _open([folder / "LST_in.nc"])
    flags = xr.open_dataset(folder / "flags_in.nc")
    flag_var = next((v for v in flags.data_vars if "flag" in v.lower()), None)
    if flag_var:
        ds["flags"] = flags[flag_var]
    geo = xr.open_dataset(folder / "geodetic_in.nc")
    lat = next((v for v in geo.data_vars if "latitude" in v), None)
    lon = next((v for v in geo.data_vars if "longitude" in v), None)
    if lat and lon:
        ds = ds.assign_coords(longitude=geo[lon], latitude=geo[lat])
    return ds


def slstr_lst_mask(ds: xr.Dataset, *, max_uncertainty: float | None = None,
                   exclude: Iterable[str] = SLSTR_LST_DEFAULT_EXCLUDE) -> xr.DataArray:
    """Mask from SLSTR LST flags, optionally also dropping pixels above
    ``max_uncertainty`` K (from the ``LST_uncertainty`` variable, when present)."""
    valid = flag_mask(ds["flags"], exclude, match="contains") if "flags" in ds else \
        xr.ones_like(ds["LST"], dtype=bool)
    if max_uncertainty is not None and "LST_uncertainty" in ds:
        valid &= ds["LST_uncertainty"] <= max_uncertainty
    return valid.rename("valid")


# --------------------------------------------------------------------------- SLSTR L2: SST
def read_slstr_sst(path: str | PathLike) -> xr.Dataset:
    """Read a SLSTR Level-2 SST product (one GHRSST L2P NetCDF file)."""
    return xr.open_dataset(path, decode_cf=True, mask_and_scale=True)


def sst_quality_mask(ds: xr.Dataset, *, min_level: int = 4) -> xr.DataArray:
    """Mask from GHRSST ``quality_level`` (4 = acceptable, 5 = best; default keeps both)."""
    return (ds["quality_level"] >= min_level).rename("valid")


def sst_bias_correct(ds: xr.Dataset) -> xr.DataArray:
    """Bias-corrected SST: ``sea_surface_temperature - sses_bias``."""
    return (ds["sea_surface_temperature"] - ds["sses_bias"]).astype("float32") \
        .rename("sst").assign_attrs(units="K")


# --------------------------------------------------------------------------- SLSTR L2: FRP
def read_slstr_frp(path: str | PathLike) -> xr.Dataset:
    """Read SLSTR Level-2 Fire Radiative Power: one record per detected fire pixel."""
    path = Path(path)
    nc = path / "FRP_in.nc" if path.is_dir() else path
    return xr.open_dataset(nc, decode_cf=True, mask_and_scale=True)


def frp_to_points(ds: xr.Dataset, *, min_confidence: float | None = None,
                  exclude_mask: Any | None = None):
    """FRP fire pixels as a :class:`geopandas.GeoDataFrame` of points.

    ``min_confidence`` filters on the ``FRP_MWIR`` confidence variable when present.
    ``exclude_mask`` is an optional boolean array (same length as the record dimension,
    ``True`` = drop) for removing known static hot sources (gas flares, volcanoes) that
    this function does not look up for you — build it from a static-source inventory you
    trust for your study area.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    df = ds.to_dataframe().reset_index()
    if min_confidence is not None and "FRP_MWIR_confidence" in df:
        df = df[df["FRP_MWIR_confidence"] >= min_confidence]
    if exclude_mask is not None:
        df = df[~np.asarray(exclude_mask)[: len(df)]]
    geom = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs=4326)


def aggregate_frp(points, *, resolution: float, agg: str = "sum"):
    """Aggregate :func:`frp_to_points` output onto a regular grid cell FRP sum (default)
    or another statistic, for emissions estimates or hotspot density maps."""
    df = points.copy()
    df["grid_x"] = (df.geometry.x // resolution) * resolution
    df["grid_y"] = (df.geometry.y // resolution) * resolution
    return df.groupby(["grid_y", "grid_x"])["FRP_MWIR"].agg(agg)


# --------------------------------------------------------------------------- SRAL/MWR altimetry
def read_sral_marine(path: str | PathLike, *, resolution: str = "01") -> xr.Dataset:
    """Read SRAL marine altimetry: ``standard_measurement.nc`` (1 Hz, ``resolution="01"``)
    or ``enhanced_measurement.nc`` (20 Hz, ``resolution="20"``, needed for coastal work)."""
    path = Path(path)
    name = "standard_measurement.nc" if resolution == "01" else "enhanced_measurement.nc"
    nc = path / name if path.is_dir() else path
    return xr.open_dataset(nc, decode_cf=True, mask_and_scale=True)


def sral_marine_mask(ds: xr.Dataset, *, surface_type_ocean: int = 0) -> xr.DataArray:
    """SRAL marine editing criteria: ocean surface type, no rain flag, range/SWH flagged
    valid, and SSHA within a plausible range (|SSHA| < 2 m)."""
    valid = xr.ones_like(ds[next(v for v in ds.data_vars if "ssha" in v)], dtype=bool)
    if "surf_type_01" in ds:
        valid &= ds["surf_type_01"] == surface_type_ocean
    for flag in ("range_ocean_qual_01_ku", "swh_ocean_qual_01_ku", "rain_flag_01_ku"):
        if flag in ds:
            valid &= ds[flag] == 0
    ssha_var = next((v for v in ds.data_vars if "ssha" in v), None)
    if ssha_var:
        valid &= np.abs(ds[ssha_var]) < 2.0
    return valid.rename("valid")


def read_sral_virtual_station(paths: Sequence[str | PathLike], point: tuple[float, float], *,
                              buffer_m: float = 500.0, geoid_correction: float = 0.0) -> Any:
    """Water-level time series at a virtual station: every pass's 20 Hz heights within
    ``buffer_m`` of ``point`` (lon, lat), across the multiple files in ``paths`` (one per
    satellite overpass, repeating roughly every 27 days).

    Returns a :class:`pandas.Series` of median height per pass, after a median/MAD outlier
    filter. Heights are ellipsoidal; pass a ``geoid_correction`` (metres, positive up) for
    your station to convert to an approximate orthometric height — this function does not
    look one up, since that needs a geoid model such as EGM2008 you supply separately.
    """
    import pandas as pd
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    lon0, lat0 = point
    out = {}
    for p in paths:
        ds = read_sral_marine(p, resolution="20")
        lat_var = next((v for v in ds.data_vars if "lat" in v.lower()), "lat_20_ku")
        lon_var = next((v for v in ds.data_vars if "lon" in v.lower()), "lon_20_ku")
        height_var = next((v for v in ds.data_vars if "height" in v.lower() or "elevation" in v.lower()),
                          None)
        if height_var is None or lat_var not in ds or lon_var not in ds:
            continue
        _, _, dist = geod.inv(np.full(ds[lon_var].shape, lon0), np.full(ds[lat_var].shape, lat0),
                              ds[lon_var].values, ds[lat_var].values)
        near = ds[height_var].values[dist <= buffer_m]
        near = near[np.isfinite(near)]
        if near.size < 3:
            continue
        med = np.median(near)
        mad = np.median(np.abs(near - med)) * 1.4826
        clean = near[np.abs(near - med) <= 3 * mad] if mad > 0 else near
        if clean.size:
            t = ds.attrs.get("start_time") or ds.attrs.get("sensing_start") or str(Path(p).stem)
            out[t] = float(np.median(clean)) + geoid_correction
    return pd.Series(out).sort_index().rename("water_level_m")


# --------------------------------------------------------------------------- Synergy
def read_synergy(folder: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read SY_2_SYN surface reflectances (OLCI ``Syn_Oa##_reflectance`` bands plus the
    matching SLSTR bands) and ``Syn_AOT550``."""
    folder = Path(folder)
    wanted = list(variables) if variables is not None else list(OLCI_BANDS)
    candidates = {b: folder / f"Syn_{_oa_name(b)}_reflectance.nc" for b in wanted}
    files = [p for p in candidates.values() if p.exists()]
    ds = _open(files) if files else xr.Dataset()
    ds = ds.rename({f"Syn_{_oa_name(b)}_reflectance": b for b, p in candidates.items() if p.exists()})
    aot_path = folder / "Syn_AOT550.nc"
    if aot_path.exists():
        ds["aot550"] = xr.open_dataset(aot_path)["T550"]
    flags_path = folder / "flags.nc"
    if flags_path.exists():
        flags = xr.open_dataset(flags_path)
        flag_var = next((v for v in flags.data_vars if "flag" in v.lower()), None)
        if flag_var:
            ds["flags"] = flags[flag_var]
    geo = xr.open_dataset(folder / "geo_coordinates.nc")
    return ds.assign_coords(longitude=geo["longitude"], latitude=geo["latitude"])


def synergy_mask(flags: xr.DataArray, exclude: Iterable[str] = ("cloud", "snow")) -> xr.DataArray:
    """Mask from SY_2_SYN ``flags`` (cloud and snow by default)."""
    return flag_mask(flags, exclude, match="contains")


def read_synergy_vgt(path: str | PathLike, variable: str | None = None) -> xr.Dataset | xr.DataArray:
    """Read SY_2_VGP/VG1/V10: already on a regular lat/lon grid, so no swath resampling is
    needed. Thin wrapper on :func:`geolattice.rsense.core.read_netcdf`; this is the simplest
    Sentinel-3 product to start with."""
    from ...core.io import read_netcdf

    return read_netcdf(path, variable, crs=4326)
