"""Satellite soil moisture: physics, products, SAR retrieval, downscaling, validation, applications.

Status: the soil water (section 1) and microwave physics (section 2) functions are
implemented and tested; ``load_soilgrids`` and sections 3 onward are an outline with their
final signatures and docstrings (method, units, reference, tutorial module and the existing
GeoLattice helper each builds on), whose bodies raise ``NotImplementedError`` for now. The
module backs the 26-module tutorial "Satellite-Based Soil Moisture with Python".

Sections (tutorial modules in brackets):

* constants, product registry and EASE-Grid 2.0 grids                         [3, 4]
* soil water and hydraulic properties, pedotransfer functions                  [1]
* dielectric mixing models and microwave forward models (tau-omega, Fresnel)   [2]
* product discovery, download and readers: SMAP, SMOS, AMSR2 LPRM, ASCAT,
  ESA CCI, ERA5-Land, SPL2SMAP_S, CLMS SSM, NISAR                              [3 to 9, 13, 14]
* quality masking, daily cubes, climatologies and long-term trends             [5, 9]
* optical and thermal proxies: OPTRAM, TVDI, soil evaporative efficiency       [2, 15]
* Sentinel-1 analysis-ready cubes, change detection, Water Cloud Model and
  surface scattering (Oh, Dubois, IEM look-up tables)                         [10 to 12]
* downscaling: predictor stacks, DisPATCh, machine learning, U-Net             [15 to 17]
* gap filling and multi-product fusion                                         [18]
* in situ data (ISMN) and point-to-pixel matching                              [19]
* validation: temporal matching, rescaling, metrics, triple collocation        [20, 21]
* root zone: exponential filter (SWI), bucket model, Ensemble Kalman Filter    [22]
* drought indices, irrigation detection, crop yield features                    [23, 24]
* hydrology and hazards: antecedent moisture and event response                [25]

Units: volumetric soil moisture in m3 m-3, degree of saturation and indices in 0 to 1 (or %
where stated), backscatter in dB unless ``db_input=False``, brightness temperature in K,
frequency in GHz, angles in degrees, depth in cm. Functions accept NumPy arrays or xarray
objects (``time, y, x`` cubes or ``time`` series) and return the same kind of object.

Quick start (implemented physics)::

    import geolattice.rsense.soil_moisture as sm

    soil = sm.saxton_rawls(sand=0.40, clay=0.20)            # wilting point, field capacity, ksat
    eps = sm.mironov_dielectric(0.25, clay=0.20)             # complex permittivity at 1.41 GHz
    tb = sm.tau_omega(295.0, eps, 40.0, tau=0.3, pol="v")    # L-band brightness temperature (K)
    theta = sm.invert_tau_omega(tb, 295.0, 40.0, clay=0.20, tau=0.3)   # back to 0.25 m3 m-3
    table = sm.tb_sensitivity()                              # dTb/dtheta under rising VOD

Planned workflow::

    import geolattice.rsense.soil_moisture as sm

    smap = sm.open_product("SMAP_L3_E", aoi="aoi.geojson", dates=("2025-01-01", "2025-12-31"))
    smap = sm.mask_quality(smap, level="recommended")
    s1 = sm.s1_soil_moisture_cube("aoi.geojson", ("2025-01-01", "2025-12-31"))
    rel = sm.change_detection(s1.vv)
    vsm = sm.relative_to_volumetric(rel, porosity=sm.porosity_from_texture(sand, clay))
    ref = sm.read_ismn("ISMN.zip", aoi="aoi.geojson")
    print(sm.validation_metrics(*sm.temporal_match(ref, smap)))

Optional dependencies (planned extra ``geolattice[rsense-soilmoisture]``): ``earthaccess``,
``asf_search``, ``pystac-client``, ``odc-stac``, ``h5py``, ``netCDF4``, ``pytesmo``, ``ismn``,
``ascat``, ``pyresample``, ``cdsapi``, ``scikit-learn``, ``xgboost``, ``lightgbm``, ``shap``,
``torch`` and ``filterpy``. They are imported only inside the functions that need them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import rioxarray  # registers the .rio accessor
import xarray as xr

# ----------------------------------------------------------------------------- constants
EPS_WATER_STATIC = 80.1  # static dielectric constant of free water at 20 C
EPS_ICE = 3.15  # dielectric constant of ice
EPS_SOLID = 4.7  # dielectric constant of soil solids (Dobson et al. 1985)
RHO_SOLID = 2.66  # particle density of mineral soil, g cm-3
LIGHT_SPEED = 299_792_458.0  # m s-1
SMAP_UBRMSE_TARGET = 0.04  # SMAP mission accuracy requirement, m3 m-3

BANDS_GHZ = {"L": 1.41, "C": 5.405, "X": 10.65, "S": 3.2}  # nominal centre frequencies

# Product registry: one entry per product the readers understand. Fields: provider short name,
# access route, grid, nominal resolution, soil moisture variable, quality variable and units.
# Resolutions are nominal; re-check versions before publishing. NISAR is still maturing: the
# provisional collection (data from 28 October 2025) follows NISAR_L3_SME2_BETA_V1 (from 1 October 2025).
PRODUCTS: dict[str, dict[str, Any]] = {
    "SMAP_L3": {"short_name": "SPL3SMP", "access": "earthdata", "grid": "EASE2_M36km",
                "res_m": 36_000, "var": "soil_moisture", "qc": "retrieval_qual_flag",
                "units": "m3 m-3", "kind": "passive"},
    "SMAP_L3_E": {"short_name": "SPL3SMP_E", "access": "earthdata", "grid": "EASE2_M09km",
                  "res_m": 9_000, "var": "soil_moisture", "qc": "retrieval_qual_flag",
                  "units": "m3 m-3", "kind": "passive"},
    "SMAP_L4": {"short_name": "SPL4SMGP", "access": "earthdata", "grid": "EASE2_M09km",
                "res_m": 9_000, "var": ("sm_surface", "sm_rootzone"), "qc": None,
                "units": "m3 m-3", "kind": "model"},
    "SPL2SMAP_S": {"short_name": "SPL2SMAP_S", "access": "earthdata", "grid": "EASE2_M03km",
                   "res_m": 3_000, "var": "soil_moisture_3km", "qc": "retrieval_qual_flag_3km",
                   "units": "m3 m-3", "kind": "active_passive"},
    "SMOS_L3": {"short_name": "SMOS_L3_CATDS", "access": "catds", "grid": "EASE2_M25km",
                "res_m": 25_000, "var": "Soil_Moisture", "qc": "Science_Flags",
                "units": "m3 m-3", "kind": "passive"},
    "SMOS_SMAP_L3": {"short_name": "NSIDC-0800", "access": "earthdata", "grid": "EASE2_M09km",
                     "res_m": 9_000, "var": "soil_moisture", "qc": "retrieval_qual_flag",
                     "units": "m3 m-3", "kind": "passive"},
    "AMSR2_LPRM": {"short_name": "LPRM_AMSR2_DS_A_SOILM3", "access": "earthdata",
                   "grid": "regular_0.1deg", "res_m": 10_000, "var": "soil_moisture_c1",
                   "qc": "mask", "units": "m3 m-3", "kind": "passive"},
    "ASCAT_SSM": {"short_name": "H SAF SSM", "access": "hsaf", "grid": "WARP5_12.5km",
                  "res_m": 12_500, "var": "sm", "qc": "proc_flag", "units": "%",
                  "kind": "active"},
    "ESA_CCI": {"short_name": "ESACCI-SOILMOISTURE-L3S", "access": "cci", "grid": "regular_0.25deg",
                "res_m": 25_000, "var": "sm", "qc": "flag", "units": "m3 m-3", "kind": "merged"},
    "ERA5_LAND": {"short_name": "reanalysis-era5-land", "access": "cds", "grid": "regular_0.1deg",
                  "res_m": 9_000, "var": "swvl1", "qc": None, "units": "m3 m-3",
                  "kind": "reanalysis"},
    "CLMS_SSM": {"short_name": "clms_ssm_1km", "access": "clms", "grid": "regular_1km",
                 "res_m": 1_000, "var": "ssm", "qc": "ssm_noise", "units": "%",
                 "kind": "active"},
    "NISAR_SM": {"short_name": "NISAR_L3_SME2_PROVISIONAL_V1", "access": "asf", "grid": "EASE2_200m",
                 "res_m": 200, "var": "soilMoisture", "qc": "retrievalQualityFlag",
                 "units": "m3 m-3", "kind": "active"},
}

# EASE-Grid 2.0 global (M) grid definitions: EPSG:6933, cell size in m, shape (rows, cols)
EASE2_GRIDS = {
    "EASE2_M36km": (36_032.220840584, (406, 964)),
    "EASE2_M25km": (25_025.2600000, (584, 1388)),
    "EASE2_M09km": (9_008.055210146, (1624, 3856)),
    "EASE2_M03km": (3_002.6850700487, (4872, 11568)),
    "EASE2_M01km": (1_000.89502334956, (14616, 34704)),
}


# ----------------------------------------------------------------------------- helpers
def _todo(module: int | str) -> None:
    """Raise the placeholder error used by every unimplemented function in this outline."""
    raise NotImplementedError(f"soil_moisture outline: planned for tutorial module {module}")


def _is_xr(*objs) -> bool:
    return any(isinstance(o, (xr.DataArray, xr.Dataset)) for o in objs)


def _where(cond, a, b):
    """``np.where`` that keeps xarray objects as xarray."""
    if _is_xr(cond, a, b):
        return xr.where(cond, a, b)
    return np.where(cond, a, b)


def _named(out, name: str, units: str | None = None):
    """Name a DataArray result and record its units; pass NumPy results through."""
    if isinstance(out, xr.DataArray):
        out = out.rename(name)
        if units is not None:
            out.attrs["units"] = units
    return out


def _fraction(x) -> Any:
    """Texture as a fraction: values in % are divided by 100 and SoilGrids g kg-1 by 1000."""
    x = x.astype("float64") if hasattr(x, "astype") else np.asarray(x, dtype="float64")
    top = float(np.nanmax(np.asarray(getattr(x, "values", x))))
    if top > 100.0:
        return x / 1000.0
    if top > 1.0:
        return x / 100.0
    return x


def _percent_to_fraction(x) -> Any:
    """Degree of saturation as 0 to 1: values above 1.5 are taken as %."""
    top = float(np.nanmax(np.asarray(getattr(x, "values", x), dtype="float64")))
    return x / 100.0 if top > 1.5 else x


# ============================================================================= 1. soil water
def volumetric_from_gravimetric(w, bulk_density, *, water_density: float = 1.0) -> Any:
    """Volumetric water content ``theta = w rho_b / rho_w`` (m3 m-3).  [Module 1]

    ``w`` is gravimetric water content (g g-1) and densities are in g cm-3.
    """
    return _named(w * bulk_density / water_density, "theta", "m3 m-3")


def porosity_from_bulk_density(bulk_density, *, particle_density: float = RHO_SOLID) -> Any:
    """Total porosity ``1 - rho_b / rho_s`` (m3 m-3).  [Module 1]"""
    return _named(1.0 - bulk_density / particle_density, "porosity", "m3 m-3")


def saxton_rawls(sand, clay, *, organic_matter=2.5, density_factor: float = 1.0) -> xr.Dataset | dict:
    """Saxton and Rawls (2006) pedotransfer functions.  [Module 1]

    ``sand`` and ``clay`` are fractions (0 to 1), % or SoilGrids g kg-1; ``organic_matter`` is
    in % by weight (2.5 % is the value of the published tables), capped at the 8 % of the fitting data. ``density_factor`` (0.9 to
    1.3) compacts (> 1) or loosens (< 1) the soil relative to its normal density.

    Returns ``wilting_point`` (-1500 kPa), ``field_capacity`` (-33 kPa), ``saturation`` (m3 m-3),
    ``plant_available_water`` (m3 m-3), ``bulk_density`` (g cm-3), the pore size distribution
    index ``lambda_`` and saturated hydraulic conductivity ``ksat`` (mm h-1), as a Dataset for
    xarray input or a dict otherwise.
    """
    s, c = _fraction(sand), _fraction(clay)
    om = np.clip(organic_matter, 0.0, 8.0)  # the regressions were fitted for up to 8 % organic matter
    t1500 = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    wp = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    fc = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    ts33 = 0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om - 0.584 * s * c + 0.078
    s33 = ts33 + (0.636 * ts33 - 0.107)
    sat = fc + s33 - 0.097 * s + 0.043
    rho_n = (1.0 - sat) * 2.65
    if density_factor != 1.0:
        rho = rho_n * density_factor
        sat_df = 1.0 - rho / 2.65
        fc = fc - 0.2 * (sat - sat_df)
        sat = sat_df
    else:
        rho = rho_n
    b = (np.log(1500.0) - np.log(fc)) / (np.log(fc) - np.log(wp))
    lam = 1.0 / b
    ksat = 1930.0 * (sat - fc) ** (3.0 - lam)
    out = {"wilting_point": wp, "field_capacity": fc, "saturation": sat,
           "plant_available_water": fc - wp, "bulk_density": rho, "lambda_": lam, "ksat": ksat}
    out = {k: _where(np.isfinite(v), v, np.nan) for k, v in out.items()}  # outside the fitted range
    if _is_xr(sand, clay, organic_matter):
        units = {"bulk_density": "g cm-3", "lambda_": "1", "ksat": "mm h-1"}
        ds = xr.Dataset({k: xr.DataArray(v) for k, v in out.items()})
        for k in ds:
            ds[k].attrs["units"] = units.get(k, "m3 m-3")
        return ds
    return out


def porosity_from_texture(sand, clay, *, organic_matter=2.5, density_factor: float = 1.0) -> Any:
    """Saturated water content (m3 m-3) from texture with Saxton and Rawls (2006).  [Modules 1, 8, 11]

    ``sand`` and ``clay`` are fractions (0 to 1), % or SoilGrids g kg-1.
    """
    sat = saxton_rawls(sand, clay, organic_matter=organic_matter,
                       density_factor=density_factor)["saturation"]
    return _named(sat, "porosity", "m3 m-3")


def degree_of_saturation(theta, porosity) -> Any:
    """Degree of saturation ``theta / porosity`` (0 to 1).  [Modules 1, 8]"""
    return _named(theta / porosity, "degree_of_saturation", "1")


def saturation_to_volumetric(saturation, porosity, *, residual=0.0) -> Any:
    """Degree of saturation (0 to 1, or % above 1.5) to m3 m-3, e.g. ASCAT SSM.  [Modules 8, 11]

    ``theta = residual + s (porosity - residual)``.
    """
    s = _percent_to_fraction(saturation)
    return _named(residual + s * (porosity - residual), "theta", "m3 m-3")


def soil_water_status(theta, wilting_point, field_capacity) -> Any:
    """Relative available water ``(theta - WP) / (FC - WP)``, clipped to 0 to 1.  [Modules 1, 23]"""
    raw = (theta - wilting_point) / (field_capacity - wilting_point)
    return _named(np.clip(raw, 0.0, 1.0), "relative_available_water", "1")


SOILGRIDS = "https://files.isric.org/soilgrids/latest/data"
SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
# conversion from the mapped integer units to usual units
SOILGRIDS_SCALE = {"sand": 1e-3, "silt": 1e-3, "clay": 1e-3, "soc": 0.1, "bdod": 0.01, "cfvo": 0.1, "phh2o": 0.1,
                   "nitrogen": 0.01, "cec": 0.1, "ocd": 0.1}
SOILGRIDS_UNITS = {"sand": "fraction", "silt": "fraction", "clay": "fraction", "soc": "g kg-1", "bdod": "g cm-3",
                   "cfvo": "%", "phh2o": "pH", "nitrogen": "g kg-1", "cec": "cmol(c) kg-1", "ocd": "kg m-3"}


def load_soilgrids(aoi: Any, *, properties: Sequence[str] = ("sand", "clay", "soc", "bdod"),
                   depths: Sequence[str] = ("0-5cm",), stat: str = "mean", crs: Any = "EPSG:4326",
                   resolution: float | None = None, overview_level: int | None = None) -> xr.Dataset:
    """Read SoilGrids 2.0 properties for an AOI from the ISRIC cloud-optimized mosaics (no account).  [Modules 1, 11, 15]

    Only the AOI window is read (HTTP range requests on the VRT/COG files, 250 m, Homolosine
    projection). Values are converted from the mapped integers to fractions (sand, silt, clay),
    g kg-1 (soil organic carbon) and g cm-3 (bulk density ``bdod``). The result has a ``depth`` dimension
    and is reprojected to ``crs`` (``resolution`` in its units; ``None`` keeps about 250 m), or left
    in Homolosine with ``crs=None``. ``aoi`` can also be a point ``(lon, lat)``, which returns profiles.
    ``overview_level`` (0, 1, ...) reads the files' coarser internal overviews for large AOIs (level 1 is
    about 4 km for the SoilGrids mosaics).
    """
    point = isinstance(aoi, (tuple, list)) and len(aoi) == 2
    bbox = (aoi[0] - 0.01, aoi[1] - 0.01, aoi[0] + 0.01, aoi[1] + 0.01) if point else _bbox(aoi)
    out, ref = {}, None
    for prop in properties:
        layers = []
        for d in depths:
            url = f"/vsicurl/{SOILGRIDS}/{prop}/{prop}_{d}_{stat}.vrt"
            kw = {} if overview_level is None else {"overview_level": overview_level}
            da = rioxarray.open_rasterio(url, **kw).squeeze("band", drop=True)
            da = da.rio.clip_box(*bbox, crs="EPSG:4326").load()
            if ref is None:
                ref = da
            elif da.shape == ref.shape:  # same window: share the coordinates exactly
                da = da.assign_coords(x=ref.x, y=ref.y)
            else:
                da = da.rio.reproject_match(ref)
            nod = da.rio.nodata
            da = da.where(da != nod) if nod is not None else da
            layers.append(da.astype("float32") * SOILGRIDS_SCALE.get(prop, 1.0))
        v = xr.concat(layers, "depth").assign_coords(depth=list(depths))
        v.attrs["units"] = SOILGRIDS_UNITS.get(prop, "")
        out[prop] = v
    ds = xr.Dataset(out)
    ds = ds.rio.write_crs(layers[0].rio.crs)
    if point:
        from pyproj import Transformer

        x, y = Transformer.from_crs(4326, ds.rio.crs, always_xy=True).transform(aoi[0], aoi[1])
        prof = ds.sel(x=x, y=y, method="nearest").drop_vars(["x", "y", "spatial_ref"], errors="ignore")
        return prof
    if crs is not None:
        kw = {"resolution": resolution} if resolution else {}
        names = list(ds.data_vars)
        first = ds[names[0]].rio.reproject(crs, nodata=np.nan, **kw)  # every property on the grid of the first
        ds = xr.Dataset({k: first if k == names[0] else
                         ds[k].rio.reproject_match(first, nodata=np.nan).assign_coords(x=first.x, y=first.y)
                         for k in names})  # identical coordinates, so the variables share one grid
    return ds


# ============================================================================= 2. microwave physics
# Permittivities follow the engineering convention eps = eps' - j eps'' (Ulaby and Long 2014),
# so the loss factor is the negative imaginary part.
EPS_0 = 8.854187817e-12  # vacuum permittivity, F m-1
_EPS_W_INF = 4.9  # high frequency limit of water permittivity


def _water_relaxation(temperature_c) -> tuple[Any, Any]:
    """Static permittivity and relaxation time (s) of pure water (Ulaby and Long 2014, eq. 4.15 to 4.17)."""
    t = temperature_c
    eps_static = 88.045 - 0.4147 * t + 6.295e-4 * t ** 2 + 1.075e-5 * t ** 3
    two_pi_tau = 1.1109e-10 - 3.824e-12 * t + 6.938e-14 * t ** 2 - 5.096e-16 * t ** 3
    return eps_static, two_pi_tau / (2.0 * np.pi)


def free_water_dielectric(freq_ghz, temperature_c=20.0, *, conductivity=0.0) -> Any:
    """Complex permittivity of free water from the Debye model (Ulaby and Long 2014).  [Module 2]

    ``conductivity`` (S m-1) adds the ionic loss ``sigma / (2 pi f eps_0)``. Relaxation is at
    about 17 GHz at 20 C, so the loss factor grows from L- to X-band.
    """
    f = freq_ghz * 1e9
    eps_s, tau = _water_relaxation(temperature_c)
    x = 2.0 * np.pi * f * tau
    real = _EPS_W_INF + (eps_s - _EPS_W_INF) / (1.0 + x ** 2)
    loss = x * (eps_s - _EPS_W_INF) / (1.0 + x ** 2) + conductivity / (2.0 * np.pi * f * EPS_0)
    return real - 1j * loss


def dobson_dielectric(theta, sand, clay, *, freq_ghz=1.41, temperature_c=20.0,
                      bulk_density=1.3, particle_density: float = RHO_SOLID) -> Any:
    """Complex soil permittivity from the Dobson et al. (1985) semi-empirical mixing model.  [Module 2]

    Uses the Peplinski et al. (1995) effective conductivity and real part correction below
    1.4 GHz. ``sand`` and ``clay`` are fractions, % or g kg-1; ``theta`` in m3 m-3.
    """
    s, c = _fraction(sand), _fraction(clay)
    mv = np.clip(theta, 1e-6, None)
    alpha = 0.65
    eps_solid = (1.01 + 0.44 * particle_density) ** 2 - 0.062
    beta_r = 1.2748 - 0.519 * s - 0.152 * c
    beta_i = 1.33797 - 0.603 * s - 0.166 * c
    low = np.asarray(freq_ghz) < 1.4
    sigma = _where(low, 0.0467 + 0.2204 * bulk_density - 0.4111 * s + 0.6614 * c,
                   -1.645 + 1.939 * bulk_density - 2.25622 * s + 1.594 * c)
    f = freq_ghz * 1e9
    eps_s, tau = _water_relaxation(temperature_c)
    x = 2.0 * np.pi * f * tau
    fw_r = _EPS_W_INF + (eps_s - _EPS_W_INF) / (1.0 + x ** 2)
    fw_i = (x * (eps_s - _EPS_W_INF) / (1.0 + x ** 2)
            + sigma / (2.0 * np.pi * EPS_0 * f) * (particle_density - bulk_density) / (particle_density * mv))
    real = (1.0 + bulk_density / particle_density * (eps_solid ** alpha - 1.0)
            + mv ** beta_r * fw_r ** alpha - mv) ** (1.0 / alpha)
    real = _where(low, 1.15 * real - 0.68, real)
    loss = (mv ** beta_i * fw_i ** alpha) ** (1.0 / alpha)
    return _named(real - 1j * loss, "eps_soil")


def mironov_dielectric(theta, clay, *, freq_ghz=1.41) -> Any:
    """Complex soil permittivity from the Mironov et al. (2009) clay-based model (SMAP L2).  [Module 2]

    The generalized refractive mixing model separates bound and free soil water at the
    transition moisture ``mvt``. ``clay`` is a fraction, % or g kg-1; valid from 1.4 to 18 GHz
    and about 0 to 76 % clay, and used by SMAP at 1.41 GHz.
    """
    cl = _fraction(clay) * 100.0
    mv = np.clip(theta, 0.0, None)
    nd = 1.634 - 0.539e-2 * cl + 0.2748e-4 * cl ** 2
    kd = 0.03952 - 0.04038e-2 * cl
    mvt = 0.02863 + 0.30673e-2 * cl
    f = freq_ghz * 1e9

    def _nk(e0, tau, sigma):
        x = 2.0 * np.pi * f * tau
        er = _EPS_W_INF + (e0 - _EPS_W_INF) / (1.0 + x ** 2)
        ei = (e0 - _EPS_W_INF) * x / (1.0 + x ** 2) + sigma / (2.0 * np.pi * EPS_0 * f)
        mod = np.sqrt(er ** 2 + ei ** 2)
        return np.sqrt((mod + er) / 2.0), np.sqrt((mod - er) / 2.0)

    nb, kb = _nk(79.8 - 85.4e-2 * cl + 32.7e-4 * cl ** 2, 1.062e-11 + 3.450e-12 * 1e-2 * cl,
                 0.3112 + 0.467e-2 * cl)
    nu, ku = _nk(100.0, 8.5e-12, 0.3631 + 1.217e-2 * cl)
    bound = mv <= mvt
    n = _where(bound, nd + (nb - 1.0) * mv, nd + (nb - 1.0) * mvt + (nu - 1.0) * (mv - mvt))
    k = _where(bound, kd + kb * mv, kd + kb * mvt + ku * (mv - mvt))
    return _named((n ** 2 - k ** 2) - 1j * (2.0 * n * k), "eps_soil")


def topp(eps_real) -> Any:
    """Topp et al. (1980) theta (m3 m-3) from apparent permittivity (TDR calibration).  [Modules 1, 2]"""
    e = eps_real
    return _named(-5.3e-2 + 2.92e-2 * e - 5.5e-4 * e ** 2 + 4.3e-6 * e ** 3, "theta", "m3 m-3")


def topp_inverse(theta, *, iterations: int = 30) -> Any:
    """Apparent permittivity from theta, inverse of :func:`topp` (Newton's method).  [Module 2]

    The Topp cubic increases monotonically, so the root is unique.
    """
    e = 1.0 + 40.0 * np.clip(theta, 0.0, None)
    for _ in range(iterations):
        g = -5.3e-2 + 2.92e-2 * e - 5.5e-4 * e ** 2 + 4.3e-6 * e ** 3 - theta
        e = e - g / (2.92e-2 - 1.1e-3 * e + 1.29e-5 * e ** 2)
    return _named(e, "eps_apparent")


def penetration_depth(eps, freq_ghz) -> Any:
    """Power penetration depth (cm) ``1 / (2 alpha)`` of a lossy medium.  [Module 2]

    ``alpha = (2 pi / lambda_0) |Im(sqrt(eps))|``. Shows why L-band senses deeper than C- or
    X-band and dry soil deeper than wet soil.
    """
    wavelength = LIGHT_SPEED / (freq_ghz * 1e9)
    alpha = 2.0 * np.pi / wavelength * np.abs(np.imag(np.sqrt(eps)))
    return _named(100.0 / (2.0 * alpha), "penetration_depth", "cm")


def fresnel_reflectivity(eps, incidence_deg, *, pol: Literal["h", "v"] = "h") -> Any:
    """Smooth surface Fresnel power reflectivity (0 to 1) at horizontal or vertical polarization.  [Module 2]"""
    th = np.deg2rad(incidence_deg)
    cos, root = np.cos(th), np.sqrt(eps - np.sin(th) ** 2)
    if pol == "h":
        r = (cos - root) / (cos + root)
    elif pol == "v":
        r = (eps * cos - root) / (eps * cos + root)
    else:
        raise ValueError("pol must be 'h' or 'v'")
    return _named(np.abs(r) ** 2, f"reflectivity_{pol}", "1")


def rough_reflectivity(r_smooth, incidence_deg, *, h: float = 0.1, q: float = 0.0, n: float = 2.0,
                       r_other=None) -> Any:
    """H-Q-N roughness correction (Wang and Choudhury 1981).  [Module 2]

    ``r_p = ((1 - Q) r_p0 + Q r_q0) exp(-h cos^N theta)``, where ``r_other`` is the smooth
    reflectivity at the other polarization (needed when ``q > 0``). SMAP uses ``Q = 0`` and
    ``N = 2``.
    """
    if q and r_other is None:
        raise ValueError("r_other (the other polarization) is required when q > 0")
    mixed = r_smooth if not q else (1.0 - q) * r_smooth + q * r_other
    return mixed * np.exp(-h * np.cos(np.deg2rad(incidence_deg)) ** n)


def vod_from_vwc(vwc, *, b: float = 0.11) -> Any:
    """Nadir vegetation optical depth ``tau = b VWC``; VWC in kg m-2 (Jackson and Schmugge 1991).  [Modules 2, 7]"""
    return _named(b * vwc, "vod", "1")


def vwc_from_ndvi(ndvi, *, stem_factor: float = 0.0, ndvi_max=None, ndvi_min: float = 0.1) -> Any:
    """Vegetation water content (kg m-2) from NDVI, as in the SMAP ancillary (Chan et al. 2013).  [Modules 2, 12]

    Foliage ``1.9134 NDVI^2 - 0.3215 NDVI`` plus stems
    ``stem_factor (NDVI_max - NDVI_min) / (1 - NDVI_min)``; ``ndvi_max`` (the annual maximum)
    defaults to ``ndvi``. Stem factors are land cover specific (0 for grass, up to 15 or more
    for forests).
    """
    foliage = np.clip(1.9134 * ndvi ** 2 - 0.3215 * ndvi, 0.0, None)
    peak = ndvi if ndvi_max is None else ndvi_max
    stems = stem_factor * np.clip(peak - ndvi_min, 0.0, None) / (1.0 - ndvi_min)
    return _named(foliage + stems, "vwc", "kg m-2")


def tau_omega(soil_temp_k, eps, incidence_deg, *, tau=0.0, omega=0.05, h=0.1, q: float = 0.0,
              n: float = 2.0, canopy_temp_k=None, pol: Literal["h", "v"] = "h") -> Any:
    """Zeroth-order tau-omega brightness temperature Tb (K) of a vegetated soil.  [Module 2]

    ``Tb = Ts e Γ + Tc (1 - ω)(1 - Γ)(1 + r Γ)`` with soil reflectivity ``r`` (Fresnel plus
    H-Q-N roughness), emissivity ``e = 1 - r`` and canopy transmissivity
    ``Γ = exp(-τ / cos θ)``, where ``tau`` is nadir VOD and ``omega`` the single scattering
    albedo (Mo et al. 1982). ``canopy_temp_k`` defaults to the soil temperature.
    """
    other = "v" if pol == "h" else "h"
    r0 = fresnel_reflectivity(eps, incidence_deg, pol=pol)
    r_other = fresnel_reflectivity(eps, incidence_deg, pol=other) if q else None
    r = rough_reflectivity(r0, incidence_deg, h=h, q=q, n=n, r_other=r_other)
    gamma = np.exp(-tau / np.cos(np.deg2rad(incidence_deg)))
    tc = soil_temp_k if canopy_temp_k is None else canopy_temp_k
    tb = soil_temp_k * (1.0 - r) * gamma + tc * (1.0 - omega) * (1.0 - gamma) * (1.0 + r * gamma)
    return _named(tb, f"tb_{pol}", "K")


def _soil_eps(theta, *, dielectric: str, clay, sand, freq_ghz, temperature_c, bulk_density):
    if dielectric == "mironov":
        return mironov_dielectric(theta, clay, freq_ghz=freq_ghz)
    if dielectric == "dobson":
        if sand is None:
            raise ValueError("dielectric='dobson' needs sand")
        return dobson_dielectric(theta, sand, clay, freq_ghz=freq_ghz, temperature_c=temperature_c,
                                 bulk_density=bulk_density)
    raise ValueError("dielectric must be 'mironov' or 'dobson'")


def invert_tau_omega(tb, soil_temp_k, incidence_deg, *, clay, tau=0.0, omega=0.05, h=0.1,
                     pol: Literal["h", "v"] = "v", dielectric: Literal["mironov", "dobson"] = "mironov",
                     sand=None, freq_ghz: float = BANDS_GHZ["L"], bulk_density: float = 1.3,
                     theta_range: tuple[float, float] = (0.02, 0.6), iterations: int = 40) -> Any:
    """Single channel algorithm: retrieve theta (m3 m-3) by inverting :func:`tau_omega`.  [Module 2]

    Tb falls monotonically as soil moisture rises, so each pixel is solved by bisection on
    ``theta_range``. Pixels whose Tb lies outside the Tb range of ``theta_range`` (for example
    open water or RFI) are NaN. The soil temperature also sets the water temperature of the
    dielectric model.
    """
    clay_f = _fraction(clay)
    sand_f = None if sand is None else _fraction(sand)

    def forward(theta):
        eps = _soil_eps(theta, dielectric=dielectric, clay=clay_f, sand=sand_f, freq_ghz=freq_ghz,
                        temperature_c=soil_temp_k - 273.15, bulk_density=bulk_density)
        return tau_omega(soil_temp_k, eps, incidence_deg, tau=tau, omega=omega, h=h, pol=pol)

    lo_t, hi_t = theta_range
    lo = xr.zeros_like(tb, dtype="float64") + lo_t if _is_xr(tb) else np.full(np.shape(tb), lo_t)
    hi = lo - lo_t + hi_t
    tb_dry, tb_wet = forward(lo), forward(hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        wetter = forward(mid) > tb  # model too warm: the soil is wetter than mid
        lo, hi = _where(wetter, mid, lo), _where(wetter, hi, mid)
    theta = 0.5 * (lo + hi)
    valid = (tb <= tb_dry) & (tb >= tb_wet)
    theta = _where(valid, theta, np.nan)
    if np.ndim(theta) == 0:
        theta = float(theta)
    return _named(theta, "theta", "m3 m-3")


def tb_sensitivity(theta_values: Sequence[float] = tuple(np.round(np.arange(0.02, 0.51, 0.02), 2)),
                   tau_values: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), *, clay=0.2,
                   soil_temp_k: float = 295.0, incidence_deg: float = 40.0,
                   freq_ghz: float = BANDS_GHZ["L"], pol: Literal["h", "v"] = "v", omega: float = 0.05,
                   h: float = 0.1, dielectric: Literal["mironov", "dobson"] = "mironov",
                   sand=None, bulk_density: float = 1.3) -> pd.DataFrame:
    """Tidy table of Tb and dTb/dtheta under rising VOD, for the Notebook 2.1 plot.  [Module 2]

    Columns ``tau``, ``theta``, ``tb`` (K) and ``dtb_dtheta`` (K per m3 m-3). The sensitivity
    shrinks as ``tau`` grows, which is why dense vegetation masks soil moisture.
    """
    theta = np.asarray(theta_values, dtype="float64")
    eps = _soil_eps(theta, dielectric=dielectric, clay=_fraction(clay),
                    sand=None if sand is None else _fraction(sand), freq_ghz=freq_ghz,
                    temperature_c=soil_temp_k - 273.15, bulk_density=bulk_density)
    rows = []
    for t in tau_values:
        tb = np.asarray(tau_omega(soil_temp_k, eps, incidence_deg, tau=t, omega=omega, h=h, pol=pol))
        rows.append(pd.DataFrame({"tau": t, "theta": theta, "tb": tb,
                                  "dtb_dtheta": np.gradient(tb, theta)}))
    return pd.concat(rows, ignore_index=True)


# ============================================================================= 3. products and access
CMR = "https://cmr.earthdata.nasa.gov/search"
HARMONY = "https://harmony.earthdata.nasa.gov"
CCI_THREDDS = "https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/soil_moisture/data/daily_files"
CCI_VERSION = "v09.2"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
# collection versions used by the readers (checked in CMR on 25 November 2026)
VERSIONS = {"SMAP_L3": "009", "SMAP_L3_E": "006", "SMAP_L4": "008", "SPL2SMAP_S": "003",
            "SMOS_SMAP_L3": "2", "AMSR2_LPRM": "001", "NISAR_SM": "1"}
SMAP_L3_VARIABLES = ("soil_moisture", "retrieval_qual_flag", "surface_flag", "vegetation_water_content",
                     "vegetation_opacity", "surface_temperature", "tb_v_corrected", "tb_h_corrected")


def _match(src, tmpl, resampling=None):
    """Reproject ``src`` onto the grid of ``tmpl`` and copy its coordinates exactly, so later arithmetic aligns."""
    from rasterio.enums import Resampling

    src = _spatial(src)
    ydim, xdim = src.rio.y_dim, src.rio.x_dim
    extra = [d for d in src.dims if d not in (ydim, xdim)]
    src = src.transpose(*extra, ydim, xdim)  # the order rasterio expects
    if np.issubdtype(src.dtype, np.floating):
        src = src.rio.write_nodata(np.nan, encoded=False)  # NaN = no data
        nodata = np.nan
    else:  # integer classes keep their own no-data value (or 0)
        nodata = src.rio.nodata if src.rio.nodata is not None else 0
    t = _spatial(tmpl)
    out = src.rio.reproject_match(t, resampling=resampling or Resampling.nearest, nodata=nodata)
    return out.assign_coords({t.rio.x_dim: t[t.rio.x_dim].values, t.rio.y_dim: t[t.rio.y_dim].values})


def regrid(src: xr.DataArray, template: xr.DataArray, *, method: Literal["nearest", "average", "bilinear", "mode"] = "average"
           ) -> xr.DataArray:
    """Reproject ``src`` onto the grid of ``template`` (both with a CRS), treating NaN as no data.  [Modules 8, 12, 16]

    The result has exactly the template's coordinates, so it can be combined with the template's
    data without alignment surprises. Grids named ``lat``/``lon`` are handled.
    """
    from rasterio.enums import Resampling

    rs = {"nearest": Resampling.nearest, "average": Resampling.average, "bilinear": Resampling.bilinear,
          "mode": Resampling.mode}[method]
    ren = {"lat": "y", "lon": "x"}
    s_ = src.rename({k: v for k, v in ren.items() if k in src.dims})
    t_ = template.rename({k: v for k, v in ren.items() if k in template.dims})
    if s_.rio.crs is None and src.rio.crs is not None:
        s_ = s_.rio.write_crs(src.rio.crs)
    if t_.rio.crs is None and template.rio.crs is not None:
        t_ = t_.rio.write_crs(template.rio.crs)
    t_ = t_.isel({d: 0 for d in t_.dims if d not in ("y", "x")}, drop=True)
    out = _match(s_, t_, rs)
    back = {v: k for k, v in ren.items() if k in template.dims}
    return out.rename(back) if back else out


def _spatial(da):
    """Drop auxiliary coordinates (row, col, lat, lon on projected grids) that a reprojection cannot carry."""
    keep = set(da.dims) | {"spatial_ref"}
    return da.drop_vars([c for c in da.coords if c not in keep])


def _bbox(aoi) -> tuple[float, float, float, float] | None:
    """``(west, south, east, north)`` in degrees from a bbox, a GeoDataFrame or a vector file."""
    if aoi is None:
        return None
    if isinstance(aoi, (tuple, list, np.ndarray)) and len(aoi) == 4:
        return tuple(float(v) for v in aoi)
    import geopandas as gpd

    g = aoi if hasattr(aoi, "total_bounds") else gpd.read_file(aoi)
    if g.crs is not None:
        g = g.to_crs(4326)
    return tuple(float(v) for v in g.total_bounds)


def _dates(dates) -> tuple[pd.Timestamp, pd.Timestamp]:
    if dates is None:
        raise ValueError("dates=(start, end) is required")
    if isinstance(dates, str):
        dates = (dates, dates)
    return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])


def _ease2_params(name: str):
    try:
        cell, (rows, cols) = EASE2_GRIDS[name]
    except KeyError:
        raise ValueError(f"unknown EASE-Grid 2.0 grid {name!r}; one of {sorted(EASE2_GRIDS)}") from None
    return cell, rows, cols, -cols / 2.0 * cell, rows / 2.0 * cell


def product_info(name: str) -> dict[str, Any]:
    """Registry entry for a product in :data:`PRODUCTS` (case-insensitive name), with its version.  [Module 3]"""
    key = {k.lower(): k for k in PRODUCTS}.get(name.lower())
    if key is None:
        raise KeyError(f"unknown product {name!r}; one of {sorted(PRODUCTS)}")
    return {"name": key, "version": VERSIONS.get(key), **PRODUCTS[key]}


def product_table(names: Sequence[str] | None = None, *, from_cmr: bool = False) -> pd.DataFrame:
    """Comparison table of products: band type, grid, resolution, access and units.  [Module 3]

    With ``from_cmr=True`` the NASA collections (Earthdata and ASF) get their title, data centre,
    first and last date and concept id from CMR collection metadata (public, no login).
    """
    rows = [product_info(n) for n in (names or PRODUCTS)]
    df = pd.DataFrame(rows).set_index("name")
    df["var"] = df["var"].map(lambda v: ", ".join(v) if isinstance(v, (tuple, list)) else v)
    if from_cmr:
        import requests

        extra = {}
        for name, row in df.iterrows():
            if row["access"] not in ("earthdata", "asf"):
                continue
            params = {"short_name": row["short_name"], "page_size": 5}
            if row.get("version"):
                params["version"] = row["version"]
            entries = requests.get(f"{CMR}/collections.json", params=params, timeout=60).json()["feed"]["entry"]
            if entries:
                e = entries[0]
                extra[name] = {"title": e.get("title"), "data_center": e.get("data_center"),
                               "time_start": (e.get("time_start") or "")[:10],
                               "time_end": (e.get("time_end") or "")[:10] or "present",
                               "concept_id": e.get("id")}
        df = df.join(pd.DataFrame(extra).T)
    return df


def ease2_index(lat, lon, grid: str = "EASE2_M09km") -> tuple[Any, Any]:
    """Row and column of the EASE-Grid 2.0 cells containing the given points.  [Modules 4, 19]"""
    from pyproj import Transformer

    cell, _, _, x0, y0 = _ease2_params(grid)
    x, y = Transformer.from_crs(4326, 6933, always_xy=True).transform(np.asarray(lon, float),
                                                                       np.asarray(lat, float))
    return (np.floor((y0 - np.asarray(y)) / cell).astype(int),
            np.floor((np.asarray(x) - x0) / cell).astype(int))


def _ease2_window(bbox, grid: str) -> tuple[int, int, int, int]:
    """Rows ``r0:r1`` and columns ``c0:c1`` (end exclusive) of the cells covering a bbox."""
    w, s, e, n = bbox
    lon = np.linspace(w, e, 9)
    lat = np.linspace(s, n, 9)
    lo, la = np.meshgrid(lon, lat)
    r, c = ease2_index(la.ravel(), lo.ravel(), grid)
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def ease2_grid(name: str = "EASE2_M09km", *, aoi: Any = None) -> xr.Dataset:
    """EASE-Grid 2.0 cell centres (``x``, ``y`` in EPSG:6933, ``lat``, ``lon``, ``row``, ``col``).  [Module 4]

    With ``aoi`` only the cells covering it are returned (the whole global grid of the 1 km
    grid has 500 million cells).
    """
    from pyproj import Transformer

    cell, rows, cols, x0, y0 = _ease2_params(name)
    r0, r1, c0, c1 = (0, rows, 0, cols) if aoi is None else _ease2_window(_bbox(aoi), name)
    x = x0 + (np.arange(c0, c1) + 0.5) * cell
    y = y0 - (np.arange(r0, r1) + 0.5) * cell
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(6933, 4326, always_xy=True).transform(xx, yy)
    ds = xr.Dataset(coords={"y": y, "x": x, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon),
                            "row": ("y", np.arange(r0, r1)), "col": ("x", np.arange(c0, c1))})
    ds.attrs.update(grid=name, cell_m=cell)
    return ds.rio.write_crs("EPSG:6933")


def _earthdata_login():
    import earthaccess

    earthaccess.login(strategy="environment" if __import__("os").environ.get("EARTHDATA_USERNAME") else "netrc")
    return earthaccess


def search_product(name: str, aoi: Any = None, dates: Any = None, **kwargs) -> list:
    """Find granules of a registry product with the right backend.  [Module 4]

    NASA products (SMAP, SMOS with the SMAP algorithm, AMSR2 LPRM, NISAR) are searched in CMR with
    ``earthaccess``; ESA CCI returns one THREDDS URL per day. ERA5-Land has no granules: read it with
    :func:`read_era5_land`. Extra keyword arguments go to ``earthaccess.search_data``.
    """
    info = product_info(name)
    start, end = _dates(dates)
    if info["access"] in ("earthdata", "asf"):
        ea = _earthdata_login()
        kw = {"short_name": info["short_name"], "temporal": (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%dT23:59:59")),
              "count": -1, **kwargs}
        if info.get("version"):
            kw["version"] = info["version"]
        if aoi is not None:
            kw["bounding_box"] = _bbox(aoi)
        return list(ea.search_data(**kw))
    if info["access"] == "cci":
        product = kwargs.get("product", "COMBINED").upper()
        return [_cci_url(d, product) for d in pd.date_range(start, end, freq="D")]
    raise ValueError(f"{info['name']} has no granule search; use open_product or its reader")


def download_product(granules: Sequence[Any], out_dir: str = "data/soil_moisture", *, threads: int = 8) -> list[str]:
    """Download granules into ``out_dir``, skipping files already there (``earthaccess.download``).  [Module 4]"""
    from pathlib import Path

    ea = _earthdata_login()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return [str(p) for p in ea.download(list(granules), str(out_dir), threads=threads, show_progress=False)]


def _granule_date(g) -> pd.Timestamp:
    import re

    name = g["umm"]["GranuleUR"]
    m = re.search(r"_(\d{8})(?:T(\d{6}))?", name)
    return pd.Timestamp(m.group(1) + ("T" + m.group(2) if m.group(2) else ""))


def _latest_per_time(granules) -> list:
    """One granule per acquisition time (the last reprocessing when CMR lists several)."""
    best = {}
    for g in sorted(granules, key=lambda g: g["umm"]["GranuleUR"]):
        best[_granule_date(g)] = g
    return [best[k] for k in sorted(best)]


def _retry(fn, *args, tries: int = 4, wait: float = 5.0):
    import time

    for i in range(tries):
        try:
            return fn(*args)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def _smap_l3_day(session, url: str, window, variables, grid_shape=None) -> dict[str, np.ndarray]:
    """Subset one SMAP L3 granule over OPeNDAP (DAP4): AM and PM groups for ``variables``."""
    import io

    import h5py

    r0, r1, c0, c1 = window
    paths = []
    for v in variables:
        paths.append(f"/Soil_Moisture_Retrieval_Data_AM/{v}")
        paths.append(f"/Soil_Moisture_Retrieval_Data_PM/{v}_pm")
    ce = ";".join(f"{p}[{r0}:{r1 - 1}][{c0}:{c1 - 1}]" for p in paths)
    resp = session.get(url + ".dap.nc4", params={"dap4.ce": ce}, timeout=300)
    resp.raise_for_status()
    out = {}
    with h5py.File(io.BytesIO(resp.content), "r") as h:
        for v in variables:
            out[v] = np.stack([h[f"Soil_Moisture_Retrieval_Data_AM/{v}"][:],
                               h[f"Soil_Moisture_Retrieval_Data_PM/{v}_pm"][:]])
    return out


def _to_float(a: np.ndarray, fill: float = -9999.0) -> np.ndarray:
    a = a.astype("float32")
    a[(a <= fill + 1) | (a > 1e30)] = np.nan
    return a


def _open_smap_l3(name: str, bbox, dates, variables, threads: int) -> xr.Dataset:
    from concurrent.futures import ThreadPoolExecutor

    info = product_info(name)
    window = _ease2_window(bbox, info["grid"])
    grid = ease2_grid(info["grid"], aoi=bbox)
    granules = _latest_per_time(search_product(name, None, dates))
    ea = _earthdata_login()
    session = ea.get_requests_https_session()

    def one(g):
        url = next(u["URL"] for u in g["umm"]["RelatedUrls"] if "opendap" in u["URL"])
        return _granule_date(g).normalize(), _retry(_smap_l3_day, session, url, window, variables)

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(one, granules))
    if not results:
        raise RuntimeError(f"no {name} granules for {dates}")
    times = [t for t, _ in results]
    data = {}
    for v in variables:
        arr = np.stack([d[v] for _, d in results])
        if v in ("retrieval_qual_flag", "surface_flag"):
            data[v] = (("time", "overpass", "y", "x"), arr.astype("uint16"))
        else:
            data[v] = (("time", "overpass", "y", "x"), _to_float(arr))
    ds = xr.Dataset(data, coords={"time": times, "overpass": ["AM", "PM"], "y": grid.y, "x": grid.x,
                                  "lat": grid.lat, "lon": grid.lon})
    ds = ds.rename({"soil_moisture": "sm"}) if "soil_moisture" in ds else ds
    ds.attrs.update(product=info["name"], short_name=info["short_name"], version=info["version"],
                    grid=info["grid"], units="m3 m-3")
    return ds.rio.write_crs("EPSG:6933")


def _harmony_job(session, collection_id: str, variables: Sequence[str], bbox, granule_ids: Sequence[str],
                 *, poll: float = 10.0, timeout: float = 7200.0) -> list[str]:
    """Run one Harmony subsetting job and return the URLs of its output files."""
    import time
    from urllib.parse import quote

    w, s, e, n = bbox
    coll = ",".join(quote(v.strip("/"), safe="") for v in variables)
    url = f"{HARMONY}/{collection_id}/ogc-api-coverages/1.0.0/collections/{coll}/coverage/rangeset"
    form = [("subset", f"lat({s}:{n})"), ("subset", f"lon({w}:{e})"),
            ("granuleId", ",".join(granule_ids)), ("forceAsync", "true"), ("skipPreview", "true")]
    resp = session.post(url, files=[(k, (None, v)) for k, v in form], timeout=300)
    resp.raise_for_status()
    job_id, t0 = resp.json()["jobID"], time.time()
    while True:
        job = session.get(f"{HARMONY}/jobs/{job_id}", timeout=120).json()
        if job["status"] == "paused":  # large jobs pause after a preview: resume them
            session.get(f"{HARMONY}/jobs/{job_id}/resume", timeout=120)
            time.sleep(poll)
            continue
        if job["status"] not in ("accepted", "running", "running_with_errors", "previewing"):
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Harmony job {job_id} still {job['status']}")
        time.sleep(poll)
    if job["status"] not in ("successful", "complete_with_errors"):
        raise RuntimeError(f"Harmony job {job_id} {job['status']}: {job.get('message')}")
    links = [lk["href"] for lk in job.get("links", []) if lk.get("rel") == "data"]
    while "next" in {lk.get("rel") for lk in job.get("links", [])}:  # paged results
        job = session.get(next(lk["href"] for lk in job["links"] if lk.get("rel") == "next"), timeout=120).json()
        links += [lk["href"] for lk in job.get("links", []) if lk.get("rel") == "data"]
    return links


def _open_smap_l4(bbox, dates, variables, threads: int, hours: Sequence[str] = ("103000",),
                  batch: int = 400) -> xr.Dataset:
    import io
    from concurrent.futures import ThreadPoolExecutor

    import h5py

    info = product_info("SMAP_L4")
    variables = tuple(variables or ("sm_surface", "sm_rootzone"))
    granules = [g for g in _latest_per_time(search_product("SMAP_L4", None, dates))
                if _granule_date(g).strftime("%H%M%S") in hours]
    ea = _earthdata_login()
    session = ea.get_requests_https_session()
    cid = granules[0]["meta"]["collection-concept-id"]
    links = []
    for i in range(0, len(granules), batch):
        ids = [g["meta"]["concept-id"] for g in granules[i:i + batch]]
        links += _harmony_job(session, cid, [f"Geophysical_Data/{v}" for v in variables], bbox, ids)

    def one(url):
        content = _retry(lambda u: session.get(u, timeout=300).content, url)
        with h5py.File(io.BytesIO(content), "r") as h:
            arrays = {v: _to_float(h[f"Geophysical_Data/{v}"][:]) for v in variables}
            return arrays, h["x"][:], h["y"][:]

    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(one, links))
    import re

    times = [pd.Timestamp(re.search(r"_(\d{8}T\d{6})_", u).group(1)) for u in links]
    order = np.argsort(times)
    x, y = parts[0][1], parts[0][2]
    grid = ease2_grid(info["grid"], aoi=bbox)
    if grid.sizes["x"] == len(x) and grid.sizes["y"] == len(y):  # snap to the exact cell centres
        x, y = grid.x.values, grid.y.values
    data = {v: (("time", "y", "x"), np.stack([parts[i][0][v] for i in order])) for v in variables}
    ds = xr.Dataset(data, coords={"time": [times[i] for i in order], "y": y, "x": x})
    ds.attrs.update(product="SMAP_L4", short_name=info["short_name"], version=info["version"],
                    grid=info["grid"], units="m3 m-3")
    return ds.rio.write_crs("EPSG:6933")


def _open_nsidc0800(bbox, dates, variables, threads: int, retrieval: str = "SMOS_Based") -> xr.Dataset:
    from concurrent.futures import ThreadPoolExecutor

    import h5py

    info = product_info("SMOS_SMAP_L3")
    variables = tuple(variables or ("soil_moisture", "retrieval_qual_flag", "vegetation_opacity"))
    r0, r1, c0, c1 = _ease2_window(bbox, info["grid"])
    grid = ease2_grid(info["grid"], aoi=bbox)
    granules = _latest_per_time(search_product("SMOS_SMAP_L3", None, dates))
    ea = _earthdata_login()

    def one(g):
        def read():
            f = ea.open([g], show_progress=False)[0]
            with h5py.File(f, "r") as h:
                out = {}
                for v in variables:
                    am = h[f"{retrieval}/Soil_Moisture_Retrieval_Data_AM/{v}_am"][r0:r1, c0:c1]
                    pm = h[f"{retrieval}/Soil_Moisture_Retrieval_Data_PM/{v}_pm"][r0:r1, c0:c1]
                    out[v] = np.stack([am, pm])
            return out
        return _granule_date(g).normalize(), _retry(read)

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(one, granules))
    data = {}
    for v in variables:
        arr = np.stack([d[v] for _, d in results])
        data[v] = (("time", "overpass", "y", "x"),
                   arr.astype("uint16") if "flag" in v else _to_float(arr))
    ds = xr.Dataset(data, coords={"time": [t for t, _ in results], "overpass": ["AM", "PM"], "y": grid.y,
                                  "x": grid.x, "lat": grid.lat, "lon": grid.lon})
    ds = ds.rename({"soil_moisture": "sm"}) if "soil_moisture" in ds else ds
    ds.attrs.update(product="SMOS_SMAP_L3", short_name=info["short_name"], version=info["version"],
                    grid=info["grid"], retrieval=retrieval, units="m3 m-3")
    return ds.rio.write_crs("EPSG:6933")


def _cci_url(day, product: str = "COMBINED", version: str = CCI_VERSION) -> str:
    day = pd.Timestamp(day)
    kind = "SSMS" if product == "ACTIVE" else "SSMV"
    return (f"{CCI_THREDDS}/{product}/{version}/{day:%Y}/ESACCI-SOILMOISTURE-L3S-{kind}-{product}-"
            f"{day:%Y%m%d}000000-f{version}.nc")


def _cci_window(bbox):
    w, s, e, n = bbox
    lat = 89.875 - 0.25 * np.arange(720)
    lon = -179.875 + 0.25 * np.arange(1440)
    iy = np.flatnonzero((lat >= s - 0.125) & (lat <= n + 0.125))
    ix = np.flatnonzero((lon >= w - 0.125) & (lon <= e + 0.125))
    return iy[0], iy[-1], ix[0], ix[-1], lat[iy], lon[ix]


def _parse_dap2_ascii(text: str, name: str, shape) -> np.ndarray:
    """Values of one array from a DAP2 ``.ascii`` response."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith((f"{name}.{name}", f"{name}[")))
    vals = []
    for ln in lines[start + 1:]:
        if not ln.startswith("["):
            if vals:
                break
            continue
        vals.extend(float(v) for v in ln.split(",")[1:])
    return np.asarray(vals, dtype="float32").reshape(shape)


def _open_cci(bbox, dates, product: str, variables, threads: int) -> xr.Dataset:
    from concurrent.futures import ThreadPoolExecutor

    import requests

    y0, y1, x0, x1, lat, lon = _cci_window(bbox)
    variables = tuple(variables or ("sm", "flag", "sm_uncertainty"))
    shape = (y1 - y0 + 1, x1 - x0 + 1)
    session = requests.Session()
    start, end = _dates(dates)

    def one(day):
        ce = ",".join(f"{v}[0:0][{y0}:{y1}][{x0}:{x1}]" for v in variables)

        def get():
            r = session.get(_cci_url(day, product) + ".ascii?" + ce, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return {v: _parse_dap2_ascii(r.text, v, shape) for v in variables}
        return day, _retry(get)

    with ThreadPoolExecutor(threads) as pool:
        results = [(d, a) for d, a in pool.map(one, pd.date_range(start, end, freq="D")) if a is not None]
    data = {}
    for v in variables:
        arr = np.stack([a[v] for _, a in results])
        if v == "flag":
            data[v] = (("time", "lat", "lon"), arr.astype("int16"))
        else:
            arr[arr <= -9998] = np.nan
            data[v] = (("time", "lat", "lon"), arr)
    ds = xr.Dataset(data, coords={"time": [d for d, _ in results], "lat": lat, "lon": lon})
    ds.attrs.update(product=f"ESA_CCI_{product}", version=CCI_VERSION,
                    units="%" if product == "ACTIVE" else "m3 m-3")
    return ds.rio.write_crs("EPSG:4326")


def open_product(name: str, aoi: Any = None, dates: Any = None, *, files: Sequence[str] | None = None,
                 variables: Sequence[str] | None = None, cache: str | None = None, threads: int = 8,
                 **kwargs) -> xr.Dataset:
    """Search, subset and read a product as a tidy cube in one call.  [Module 4]

    Only the AOI is transferred: SMAP L3 and L3 Enhanced through NASA OPeNDAP (DAP4), SMAP L4
    through the Harmony subsetter (one granule per day, ``hours=("103000",)`` by default), SMOS
    with the SMAP algorithm (``SMOS_SMAP_L3``, its SMOS-based group) by HTTP range reads of its HDF5 files, ESA CCI
    (``product="ACTIVE" | "PASSIVE" | "COMBINED"``) through the CEDA THREDDS server and ERA5-Land
    through the Open-Meteo archive. Swath and tile products (``SPL2SMAP_S``, ``NISAR_SM``,
    ``AMSR2_LPRM``) are downloaded and read with their readers; ``files`` reads local files instead.
    SMAP and SMOS cubes have dims ``time, overpass, y, x`` (EPSG:6933) with soil moisture as ``sm``;
    use :func:`daily_cube` to merge the overpasses. With ``cache`` the cube is written to that NetCDF
    path on the first call and read from it afterwards.
    """
    from pathlib import Path

    if cache is not None and Path(cache).exists():
        return open_cached(cache)
    info = product_info(name)
    key = info["name"]
    if files is not None:
        reader = {"SMAP_L3": read_smap, "SMAP_L3_E": read_smap, "SMAP_L4": read_smap, "SPL2SMAP_S": read_smap,
                  "NISAR_SM": read_nisar_sm, "AMSR2_LPRM": read_amsr2_lprm}[key]
        parts = [reader(f, aoi=aoi) for f in files]
        ds = xr.concat(parts, "time").sortby("time") if "time" in parts[0].dims else parts
    else:
        bbox = _bbox(aoi)
        if bbox is None:
            raise ValueError("aoi is required")
        if key in ("SMAP_L3", "SMAP_L3_E"):
            ds = _open_smap_l3(key, bbox, dates, tuple(variables or SMAP_L3_VARIABLES), threads)
        elif key == "SMAP_L4":
            ds = _open_smap_l4(bbox, dates, variables, threads, **kwargs)
        elif key == "SMOS_SMAP_L3":
            ds = _open_nsidc0800(bbox, dates, variables, threads, **kwargs)
        elif key == "ESA_CCI":
            ds = _open_cci(bbox, dates, kwargs.get("product", "COMBINED").upper(), variables, threads)
        elif key == "ERA5_LAND":
            ds = read_era5_land(bbox, dates, **kwargs)
        else:
            out_dir = kwargs.get("out_dir", "data/soil_moisture")
            paths = download_product(search_product(key, bbox, dates), out_dir, threads=threads)
            return open_product(key, aoi, dates, files=sorted(paths), cache=cache)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        to_cache(ds, cache, fmt="netcdf")
    return ds


def open_cached(path: str) -> xr.Dataset:
    """Open a cube written by :func:`to_cache` into memory, with its CRS on every variable."""
    ds = xr.open_dataset(path, decode_coords="all").load()
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    crs = ds.rio.crs
    if crs is None and "spatial_ref" in ds.coords and "crs_wkt" in ds.spatial_ref.attrs:
        crs = ds.spatial_ref.attrs["crs_wkt"]
    if crs is not None:
        for v in ds.data_vars:
            if {"x", "y"} <= set(ds[v].dims) or {"lat", "lon"} <= set(ds[v].dims):
                ds[v] = ds[v].rio.write_crs(crs)
    return ds


def to_cache(ds: xr.Dataset, path: str, *, fmt: Literal["netcdf", "zarr", "cog"] = "netcdf") -> str:
    """Write a cube to NetCDF (compressed), Zarr or one Cloud-Optimized GeoTIFF per variable.  [Module 4]"""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ds.rio.crs is not None:  # record the CRS on every gridded variable
        crs = ds.rio.crs
        ds = ds.copy()
        for v in ds.data_vars:
            if {"x", "y"} <= set(ds[v].dims) or {"lat", "lon"} <= set(ds[v].dims):
                ds[v] = ds[v].rio.write_crs(crs)
    if fmt == "netcdf":
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(path, encoding=enc)
    elif fmt == "zarr":
        ds.to_zarr(path, mode="w")
    elif fmt == "cog":
        path.mkdir(parents=True, exist_ok=True)
        for v in ds.data_vars:
            da = ds[v]
            if "time" in da.dims:
                da = da.isel(time=-1)
            da.rio.to_raster(path / f"{v}.tif", driver="COG")
    else:
        raise ValueError("fmt must be 'netcdf', 'zarr' or 'cog'")
    return str(path)


# ----------------------------------------------------------------------------- readers
def read_smap(path: str, *, product: Literal["L3", "L3_E", "L4", "SPL2SMAP_S"] | None = None,
              variables: Sequence[str] | None = None, aoi: Any = None) -> xr.Dataset:
    """Read a SMAP HDF5 granule (or an OPeNDAP or Harmony subset of one) onto its EASE-Grid 2.0 grid.  [Modules 5, 13]

    ``product`` is guessed from the file name. L3 files give ``time, overpass, y, x`` (AM and PM
    groups); L4 gives ``sm_surface`` and ``sm_rootzone``; SPL2SMAP_S tiles give the 3 km and 1 km
    soil moisture on their own grids (``soil_moisture_3km``, ``soil_moisture_1km``).
    """
    import re
    from pathlib import Path

    import h5py

    name = Path(path).name
    product = product or ("L4" if "_L4_" in name else "SPL2SMAP_S" if "_SM_SP_" in name
                          else "L3_E" if "_P_E_" in name else "L3")
    when = pd.Timestamp(re.search(r"_(\d{8})(?:T(\d{6}))?", name).group(1))
    with h5py.File(path, "r") as h:
        if product in ("L3", "L3_E"):
            ds = _read_smap_l3_h5(h, variables or SMAP_L3_VARIABLES, "EASE2_M09km" if product == "L3_E" else "EASE2_M36km")
            ds = ds.expand_dims(time=[when])
        elif product == "L4":
            g = h["Geophysical_Data"]
            vs = variables or ("sm_surface", "sm_rootzone")
            when = pd.Timestamp(re.search(r"_(\d{8}T\d{6})_", name).group(1))
            arrs = {v: (("y", "x"), _to_float(g[v][:])) for v in vs}
            if "x" in h and "y" in h:
                coords = {"y": h["y"][:], "x": h["x"][:]}
            else:
                grid = ease2_grid("EASE2_M09km")
                coords = {"y": grid.y.values, "x": grid.x.values}
            ds = xr.Dataset(arrs, coords=coords).expand_dims(time=[when])
        else:
            out = {}
            for res, grp in (("3km", "Soil_Moisture_Retrieval_Data_3km"), ("1km", "Soil_Moisture_Retrieval_Data_1km")):
                if grp not in h:
                    continue
                g = h[grp]
                sm = _to_float(g[f"soil_moisture_{res}"][:])
                lat = g[f"latitude_{res}"][:]
                lon = g[f"longitude_{res}"][:]
                qc = g[f"retrieval_qual_flag_{res}"][:].astype("uint16")
                out[res] = xr.Dataset({"sm": (("y", "x"), sm), "retrieval_qual_flag": (("y", "x"), qc)},
                                      coords={"lat": (("y", "x"), _to_float(lat)), "lon": (("y", "x"), _to_float(lon))})
            ds = xr.Dataset({f"sm_{k}": v.sm.rename({"y": f"y_{k}", "x": f"x_{k}"}) for k, v in out.items()})
            for k, v in out.items():
                ds[f"retrieval_qual_flag_{k}"] = v.retrieval_qual_flag.rename({"y": f"y_{k}", "x": f"x_{k}"})
                ds[f"lat_{k}"] = v.lat.rename({"y": f"y_{k}", "x": f"x_{k}"})
                ds[f"lon_{k}"] = v.lon.rename({"y": f"y_{k}", "x": f"x_{k}"})
            ds = ds.expand_dims(time=[pd.Timestamp(re.search(r"_(\d{8}T\d{6})_", name).group(1))])
            ds.attrs["tile"] = re.search(r"_(\d{3}[EW]\d{2}[NS])_", name).group(1)
    ds.attrs.update(product=f"SMAP_{product}", source=name)
    if aoi is not None and "x" in ds.dims:
        w, s, e, n = _bbox(aoi)
        from pyproj import Transformer

        xs, ys = Transformer.from_crs(4326, 6933, always_xy=True).transform([w, e], [s, n])
        ds = ds.sel(x=slice(xs[0], xs[1]), y=slice(ys[1], ys[0]))
    return ds.rio.write_crs("EPSG:6933") if "x" in ds.dims else ds


def _read_smap_l3_h5(h, variables, grid: str) -> xr.Dataset:
    data, rows = {}, None
    for v in variables:
        am = h[f"Soil_Moisture_Retrieval_Data_AM/{v}"][:]
        pm = h[f"Soil_Moisture_Retrieval_Data_PM/{v}_pm"][:]
        arr = np.stack([am, pm])
        data[v] = (("overpass", "y", "x"), arr.astype("uint16") if "flag" in v else _to_float(arr))
        rows = arr.shape[1:]
    cell, nrow, ncol, x0, y0 = _ease2_params(grid)
    if rows == (nrow, ncol):
        y = y0 - (np.arange(nrow) + 0.5) * cell
        x = x0 + (np.arange(ncol) + 0.5) * cell
    else:  # a subset without its row and column offsets
        y, x = np.arange(rows[0]), np.arange(rows[1])
    ds = xr.Dataset(data, coords={"overpass": ["AM", "PM"], "y": y, "x": x})
    return ds.rename({"soil_moisture": "sm"}) if "soil_moisture" in ds else ds


def read_smos(path: str, *, retrieval: Literal["SMOS_Based", "SMAP_Based"] = "SMOS_Based",
              variables: Sequence[str] = ("soil_moisture", "retrieval_qual_flag"), aoi: Any = None) -> xr.Dataset:
    """Read an NSIDC-0800 SMOS-SMAP L3 file: SMOS and SMAP retrievals made with the same SMAP algorithm.  [Module 6]

    ``retrieval="SMOS_Based"`` (default) reads the retrieval from SMOS brightness temperatures,
    ``"SMAP_Based"`` the retrieval from SMAP brightness temperatures stored alongside it.
    """
    import re
    from pathlib import Path

    import h5py

    info = product_info("SMOS_SMAP_L3")
    window = None if aoi is None else _ease2_window(_bbox(aoi), info["grid"])
    with h5py.File(path, "r") as h:
        data = {}
        for v in variables:
            parts = []
            for op in ("AM", "PM"):
                d = h[f"{retrieval}/Soil_Moisture_Retrieval_Data_{op}/{v}_{op.lower()}"]
                parts.append(d[:] if window is None else d[window[0]:window[1], window[2]:window[3]])
            arr = np.stack(parts)
            data[v] = (("overpass", "y", "x"), arr.astype("uint16") if "flag" in v else _to_float(arr))
    grid = ease2_grid(info["grid"], aoi=aoi)
    ds = xr.Dataset(data, coords={"overpass": ["AM", "PM"], "y": grid.y, "x": grid.x})
    when = pd.Timestamp(re.search(r"_(\d{8})_", Path(path).name).group(1))
    ds = ds.rename({"soil_moisture": "sm"}).expand_dims(time=[when])
    return ds.rio.write_crs("EPSG:6933")


def read_amsr2_lprm(path: str, *, band: Literal["c1", "c2", "x"] = "c1", aoi: Any = None) -> xr.Dataset:
    """Read an AMSR2 LPRM L3 NetCDF (GES DISC) for one band: soil moisture and VOD.  [Module 7]"""
    import re
    from pathlib import Path

    raw = xr.open_dataset(path)
    names = {"sm": f"soil_moisture_{band}", "vod": f"opt_depth_{band}",
             "sm_error": f"soil_moisture_{band}_error"}
    ds = xr.Dataset({k: raw[v] for k, v in names.items() if v in raw})
    ds = ds.rename({d: {"Longitude": "lon", "Latitude": "lat"}.get(d, d) for d in ds.dims})
    ds = ds.transpose(..., "lat", "lon").sortby("lat", ascending=False)
    if "sm" in ds and float(ds.sm.max()) > 1.5:
        ds["sm"] = ds.sm / 100.0
    if aoi is not None:
        w, s, e, n = _bbox(aoi)
        ds = ds.sel(lat=slice(n, s), lon=slice(w, e))
    m = re.search(r"_(\d{8})", Path(path).name)
    if m and "time" not in ds.dims:
        ds = ds.expand_dims(time=[pd.Timestamp(m.group(1))])
    return ds.where(ds > -1)


def read_ascat(source: str, *, lon=None, lat=None, product: Literal["ssm", "swi", "cci_active"] = "cci_active",
               dates: Any = None, porosity=None, **kwargs) -> pd.DataFrame | xr.Dataset:
    """ASCAT surface soil moisture: ESA CCI ACTIVE by default, or H SAF files with ``ascat``.  [Modules 8, 22]

    ESA CCI ACTIVE is the ASCAT (and ERS) record in degree of saturation (%), served without an
    account; ``source`` is then a bbox and ``dates`` the period. ``product="ssm"`` or ``"swi"``
    read H SAF files with the ``ascat`` package. With ``porosity`` the result also has volumetric
    soil moisture ``theta`` from :func:`saturation_to_volumetric`.
    """
    if product == "cci_active":
        ds = open_product("ESA_CCI", source, dates, product="ACTIVE", **kwargs)
        if lon is not None and lat is not None:
            ds = ds.sel(lon=lon, lat=lat, method="nearest")
        if porosity is not None:
            ds["theta"] = saturation_to_volumetric(ds.sm, porosity)
        return ds
    try:
        from ascat.read_native.cdr import AscatGriddedNcTs  # noqa: F401
    except ImportError as err:
        raise ImportError("H SAF files need the ascat package: pip install ascat") from err
    raise NotImplementedError("H SAF readers: use the ascat package directly (see notebook 08)")


def read_esa_cci(path_or_dir: str | Sequence[str] | None = None, *, product: Literal["ACTIVE", "PASSIVE", "COMBINED"] = "COMBINED",
                 aoi: Any = None, dates: Any = None, **kwargs) -> xr.Dataset:
    """ESA CCI SM daily files: local NetCDF files, or the CEDA server when no path is given.  [Module 9]

    Returns ``sm`` (m3 m-3, or % saturation for ACTIVE), ``flag`` and ``sm_uncertainty`` on the
    0.25 degree grid.
    """
    if path_or_dir is None:
        return open_product("ESA_CCI", aoi, dates, product=product, **kwargs)
    from pathlib import Path

    paths = path_or_dir
    if isinstance(path_or_dir, (str, Path)):
        p = Path(path_or_dir)
        paths = sorted(p.rglob(f"*{product}*.nc")) if p.is_dir() else [p]
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")[["sm", "flag", "sm_uncertainty"]]
    if aoi is not None:
        w, s, e, n = _bbox(aoi)
        ds = ds.sel(lat=slice(n, s), lon=slice(w, e))
    if dates is not None:
        ds = ds.sel(time=slice(*_dates(dates)))
    return ds.load()


def read_era5_land(aoi: Any, dates: Any, *, layers: Sequence[int] = (1, 2, 3), path: str | None = None,
                   freq: Literal["hourly", "daily"] = "daily", resolution: float = 0.25,
                   points: Sequence[tuple[float, float]] | None = None, batch: int = 40) -> xr.Dataset:
    """ERA5-Land volumetric soil water from the Open-Meteo archive (no key) or a local file.  [Modules 9, 21, 24]

    Layers 1, 2 and 3 are 0 to 7, 7 to 28 and 28 to 100 cm (``swvl1`` to ``swvl3``, m3 m-3).
    Values are daily means (or hourly) at grid points every ``resolution`` degrees over the AOI,
    or at ``points`` (a list of ``(lon, lat)``), which gives a ``point`` dimension instead.
    """
    import requests

    if path is not None:
        ds = xr.open_dataset(path)
        keep = [f"swvl{i}" for i in layers if f"swvl{i}" in ds]
        return ds[keep]
    names = {1: "soil_moisture_0_to_7cm", 2: "soil_moisture_7_to_28cm", 3: "soil_moisture_28_to_100cm"}
    start, end = _dates(dates)
    if points is None:
        w, s, e, n = _bbox(aoi)
        lats = np.round(np.arange(n, s - 1e-9, -resolution), 4)
        lons = np.round(np.arange(w, e + 1e-9, resolution), 4)
        la, lo = np.meshgrid(lats, lons, indexing="ij")
        pts = list(zip(lo.ravel(), la.ravel()))
    else:
        pts = [tuple(p) for p in points]
    hourly = freq == "hourly"
    var_list = [names[i] + ("" if hourly else "_mean") for i in layers]
    rows = []
    for i in range(0, len(pts), batch):
        chunk = pts[i:i + batch]
        params = {"latitude": ",".join(f"{p[1]:.4f}" for p in chunk), "longitude": ",".join(f"{p[0]:.4f}" for p in chunk),
                  "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}", "models": "era5_land",
                  ("hourly" if hourly else "daily"): ",".join(var_list), "timezone": "GMT"}

        def get(params=params):
            r = requests.get(OPEN_METEO, params=params, timeout=600)
            if r.status_code == 429:
                raise RuntimeError("Open-Meteo rate limit")
            r.raise_for_status()
            j = r.json()
            return j if isinstance(j, list) else [j]
        rows.extend(_retry(get, tries=6, wait=60.0))
    key = "hourly" if hourly else "daily"
    time = pd.to_datetime(rows[0][key]["time"])
    arrays = {f"swvl{i}": np.array([r[key][v] for r in rows], dtype="float32").T
              for i, v in zip(layers, var_list)}
    if points is None:
        shp = (len(time), len(lats), len(lons))
        ds = xr.Dataset({k: (("time", "lat", "lon"), a.reshape(shp)) for k, a in arrays.items()},
                        coords={"time": time, "lat": lats, "lon": lons}).rio.write_crs("EPSG:4326")
    else:
        ds = xr.Dataset({k: (("time", "point"), a) for k, a in arrays.items()},
                        coords={"time": time, "lon": ("point", [p[0] for p in pts]), "lat": ("point", [p[1] for p in pts])})
    for k in ds.data_vars:
        ds[k].attrs["units"] = "m3 m-3"
    ds.attrs.update(product="ERA5_LAND", source="Open-Meteo archive (ERA5-Land)")
    return ds


def read_clms_ssm(path_or_dir: str, *, aoi: Any = None) -> xr.Dataset:
    """Read Copernicus Land 1 km surface soil moisture GeoTIFF or NetCDF files (Sentinel-1, Europe).  [Module 13]

    Download the files from the CLMS portal (free EU Login). ``ssm`` is relative saturation (%).
    """
    import re
    from pathlib import Path

    p = Path(path_or_dir)
    paths = sorted(p.rglob("*SSM*")) if p.is_dir() else [p]
    parts = []
    for f in paths:
        da = xr.open_dataarray(f, engine="rasterio").squeeze(drop=True) if f.suffix.lower() in (".tif", ".tiff") \
            else xr.open_dataset(f)["ssm"].squeeze(drop=True)
        m = re.search(r"(\d{8})", f.name)
        parts.append(da.expand_dims(time=[pd.Timestamp(m.group(1))]) if m else da)
    ds = xr.concat(parts, "time").to_dataset(name="ssm") if len(parts) > 1 else parts[0].to_dataset(name="ssm")
    if aoi is not None:
        w, s, e, n = _bbox(aoi)
        ds = ds.rio.clip_box(w, s, e, n, crs="EPSG:4326")
    return ds


NISAR_SM_GROUP = "science/LSAR/SME2/grids"


def read_nisar_sm(path: Any, *, aoi: Any = None, flags: bool = True) -> xr.Dataset:
    """Read a NISAR L3 soil moisture (SME2) HDF5 file, or a remote file object, with its quality layers.  [Module 14]

    Reads ``/science/LSAR/SME2/grids/soilMoisture`` (the combined retrieval; the per-algorithm
    candidates are in ``algorithmCandidates``) on the 200 m EASE-Grid 2.0 (EPSG:6933). With ``aoi``
    only that window is read, which also works on ``earthaccess.open`` file objects. With ``flags``
    the result has ``uncertainty``, ``retrieval_qual_flag`` (bit 0 set: not recommended),
    ``surface_qual_flag``, ``water_fraction``, ``land_cover`` and ``local_incidence``.
    """
    import re
    from pathlib import Path

    import h5py

    name = getattr(path, "path", None) or getattr(path, "full_name", None) or str(path)
    with h5py.File(path, "r") as h:
        g = h[NISAR_SM_GROUP]
        x, y = g["xCoordinates"][:], g["yCoordinates"][:]
        if aoi is not None:
            from pyproj import Transformer

            w, s_, e, n = _bbox(aoi)
            xs, ys = Transformer.from_crs(4326, 6933, always_xy=True).transform([w, e, w, e], [s_, s_, n, n])
            ix = np.flatnonzero((x >= min(xs)) & (x <= max(xs)))
            iy = np.flatnonzero((y >= min(ys)) & (y <= max(ys)))
            if len(ix) == 0 or len(iy) == 0:
                return xr.Dataset()
            sl = (slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))
        else:
            sl = (slice(None), slice(None))

        def read(ds, fill=-9999.0, kind="float"):
            a = ds[sl]
            if kind == "float":
                a = a.astype("float32")
                a[a <= fill + 1] = np.nan
            return a
        data = {"sm": (("y", "x"), read(g["soilMoisture"]))}
        if flags:
            anc = g["ancillaryData"]
            data.update({
                "uncertainty": (("y", "x"), read(g["soilMoistureUncertainty"])),
                "retrieval_qual_flag": (("y", "x"), read(g["retrievalQualityFlag"], kind="int")),
                "surface_qual_flag": (("y", "x"), read(g["surfaceQualityFlag"], kind="int")),
                "water_fraction": (("y", "x"), read(anc["waterbodyFraction"])),
                "land_cover": (("y", "x"), read(anc["landCover"], kind="int")),
                "local_incidence": (("y", "x"), read(anc["localIncidenceAngle"])),
            })
        coords = {"y": y[sl[0]], "x": x[sl[1]]}
    ds = xr.Dataset(data, coords=coords)
    ds["sm"] = ds.sm.where((ds.sm >= 0) & (ds.sm <= 1))
    m = re.search(r"_(\d{8}T\d{6})_", Path(name).name)
    if m:
        ds = ds.expand_dims(time=[pd.Timestamp(m.group(1))])
    ds.attrs.update(product="NISAR_SM", source=Path(name).name, grid="EASE2 200 m")
    return ds.rio.write_crs("EPSG:6933")


# ----------------------------------------------------------------------------- masking, cubes, trends
# SMAP L2/L3 bit definitions (SMAP L3 product specification); a set bit means "not" or "present"
SMAP_QUAL_BITS = {"not_recommended": 0, "retrieval_not_attempted": 1, "retrieval_failed": 2,
                  "freeze_thaw_failed": 3}
SMAP_SURFACE_BITS = {"static_water": 0, "radar_water": 1, "coastal": 2, "urban": 3, "precipitation": 4,
                     "snow_ice": 5, "permanent_snow_ice": 6, "frozen_radiometer": 7, "frozen_model": 8,
                     "mountainous": 9, "dense_vegetation": 10, "nadir": 11}
CCI_FLAG_BITS = {"snow_or_temperature_below_zero": 0, "dense_vegetation": 1, "others_no_convergence": 2,
                 "soil_moisture_out_of_range": 3, "barren_ground": 4, "frozen_era5": 5, "water_body": 6}


def decode_flags(flags: xr.DataArray, product: str = "SMAP") -> xr.Dataset:
    """Expand a bit-packed quality layer into named boolean layers.  [Modules 5, 14]

    ``product`` is ``"SMAP"`` (``retrieval_qual_flag``), ``"SMAP_SURFACE"`` (``surface_flag``) or
    ``"CCI"`` (ESA CCI ``flag``). Fill values (65534 and above) are False everywhere.
    """
    bits = {"SMAP": SMAP_QUAL_BITS, "SMAP_SURFACE": SMAP_SURFACE_BITS, "CCI": CCI_FLAG_BITS}[product.upper()]
    f = flags.astype("int64")
    valid = f < 65534
    return xr.Dataset({k: ((f // (2 ** b)) % 2 == 1) & valid for k, b in bits.items()})


def mask_quality(ds: xr.Dataset, *, level: Literal["recommended", "all", "strict"] = "recommended",
                 product: str | None = None, frozen: bool = True, water: bool = True,
                 rfi: bool = True, dense_vegetation: bool = False, var: str = "sm") -> xr.Dataset:
    """Mask retrievals with product quality bits and surface flags.  [Modules 5, 6, 14]

    SMAP and SMOS-SMAP: ``recommended`` keeps retrievals whose ``retrieval_qual_flag`` bit 0 is
    clear; ``all`` keeps every successful retrieval; ``strict`` also drops precipitation, frozen
    ground, coastal and urban cells from ``surface_flag``. ESA CCI: keeps ``flag == 0`` cells (or
    only drops frozen and water cells with ``level="all"``). The quality layers are kept.
    """
    out = ds.copy()
    product = (product or ds.attrs.get("product", "")).upper()
    keep = xr.ones_like(out[var], dtype=bool)
    if product.startswith("ESA_CCI") or "flag" in ds and "retrieval_qual_flag" not in ds:
        fl = decode_flags(ds["flag"], "CCI")
        if level == "all":
            keep = ~(fl.water_body | fl.snow_or_temperature_below_zero | fl.frozen_era5)
        else:
            keep = ds["flag"] == 0
        if dense_vegetation:
            keep = keep & ~fl.dense_vegetation
    else:
        if "retrieval_qual_flag" in ds:
            q = decode_flags(ds["retrieval_qual_flag"], "SMAP")
            keep = ~(q.retrieval_failed | q.retrieval_not_attempted) & (ds["retrieval_qual_flag"] < 65534)
            if level in ("recommended", "strict"):
                keep = keep & ~q.not_recommended
        if "surface_flag" in ds:
            s = decode_flags(ds["surface_flag"], "SMAP_SURFACE")
            if frozen:
                keep = keep & ~(s.frozen_model | s.frozen_radiometer | s.snow_ice | s.permanent_snow_ice)
            if water:
                keep = keep & ~s.static_water
            if dense_vegetation:
                keep = keep & ~s.dense_vegetation
            if level == "strict":
                keep = keep & ~(s.precipitation | s.coastal | s.urban)
    for v in out.data_vars:
        if "flag" not in v and set(out[v].dims) >= set(keep.dims) and np.issubdtype(out[v].dtype, np.floating):
            out[v] = out[v].where(keep)
    out.attrs["quality_mask"] = level
    return out


def daily_cube(datasets: Sequence[xr.Dataset] | xr.Dataset, *,
               combine: Literal["am_first", "mean", "pm_first", "am", "pm"] = "am_first") -> xr.Dataset:
    """Merge AM and PM overpasses into one regular daily ``time, y, x`` cube.  [Module 5]

    ``am_first`` uses the morning (descending) retrieval and fills gaps with the evening one, the
    usual choice because near-surface temperature gradients are smallest at 6 am.
    """
    ds = xr.concat(list(datasets), "time").sortby("time") if not isinstance(datasets, xr.Dataset) else datasets
    if "overpass" in ds.dims:
        am, pm = ds.sel(overpass="AM", drop=True), ds.sel(overpass="PM", drop=True)
        floats = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)]
        if combine == "am":
            ds = am
        elif combine == "pm":
            ds = pm
        elif combine == "mean":
            ds = xr.merge([xr.concat([am[floats], pm[floats]], "o").mean("o"), am.drop_vars(floats)])
        else:
            first, second = (am, pm) if combine == "am_first" else (pm, am)
            ds = first.copy()
            for v in floats:
                ds[v] = first[v].fillna(second[v])
            ds["overpass_used"] = xr.where(first["sm"].notnull(), 0, xr.where(second["sm"].notnull(), 1, -1)).astype("int8")
    ds = ds.resample(time="1D").mean() if ds.indexes["time"].has_duplicates else ds
    full = pd.date_range(ds.time.values.min(), ds.time.values.max(), freq="D")
    return ds.reindex(time=full)


def harmonize(cubes: Mapping[str, xr.Dataset | xr.DataArray], *, target: str | xr.Dataset | xr.DataArray = "EASE2_M09km",
              method: Literal["nearest", "average", "bilinear"] = "average", var: str = "sm",
              time_freq: str | None = "1D") -> xr.Dataset:
    """Put several products on one grid and calendar, one variable per product.  [Modules 4, 6, 9]

    ``target`` is an EASE-Grid 2.0 name (the cells covering the first cube) or a cube whose grid
    is used. Each input is reprojected with ``rioxarray`` (``average`` resampling aggregates fine
    data, ``nearest`` keeps values) and averaged to ``time_freq``.
    """
    from rasterio.enums import Resampling

    items = {k: (v[var] if isinstance(v, xr.Dataset) else v) for k, v in cubes.items()}
    first = next(iter(items.values()))
    if isinstance(target, str):
        b = first.rio.transform_bounds("EPSG:4326") if first.rio.crs else None
        tgt = ease2_grid(target, aoi=b)
        template = xr.DataArray(np.zeros((tgt.sizes["y"], tgt.sizes["x"]), "float32"), coords={"y": tgt.y, "x": tgt.x},
                                dims=("y", "x")).rio.write_crs("EPSG:6933")
    else:
        template = target[var] if isinstance(target, xr.Dataset) else target
        template = template.isel({d: 0 for d in template.dims if d not in ("y", "x", "lat", "lon")})
    rs = {"nearest": Resampling.nearest, "average": Resampling.average, "bilinear": Resampling.bilinear}[method]
    out = {}
    for k, da in items.items():
        if "lat" in da.dims:
            da = da.rename({"lat": "y", "lon": "x"})
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        tmpl = template.rename({"lat": "y", "lon": "x"}) if "lat" in template.dims else template
        dims = [d for d in da.dims if d not in ("y", "x")]
        if dims:
            da = da.transpose(*dims, "y", "x")
        r = _match(da.astype("float32"), tmpl, rs)
        if time_freq and "time" in r.dims:
            r = r.resample(time=time_freq).mean()
        out[k] = r
    ds = xr.Dataset(out)
    return ds.rio.write_crs(template.rio.crs) if template.rio.crs else ds


def sm_climatology(cube: xr.DataArray, *, freq: Literal["month", "doy"] = "doy", window: int = 31,
                   period: tuple[Any, Any] | None = None, min_years: int = 3) -> xr.DataArray:
    """Smoothed day-of-year (or monthly) climatology of a soil moisture cube.  [Modules 5, 9]

    The day-of-year mean is smoothed with a circular moving window of ``window`` days, so gaps of a
    few days do not show; values with fewer than ``min_years`` observations are NaN.
    """
    da = cube.sel(time=slice(*period)) if period else cube
    if freq == "month":
        return da.groupby("time.month").mean("time")
    doy = da.time.dt.dayofyear.where(da.time.dt.dayofyear < 366, 365)
    mean = da.groupby(doy.rename("doy")).mean("time")
    count = da.notnull().groupby(doy.rename("doy")).sum("time")
    mean = mean.reindex(doy=np.arange(1, 366))
    count = count.reindex(doy=np.arange(1, 366)).fillna(0)
    half = window // 2
    pad_m = xr.concat([mean.isel(doy=slice(-half, None)), mean, mean.isel(doy=slice(0, half))], "doy")
    pad_c = xr.concat([count.isel(doy=slice(-half, None)), count, count.isel(doy=slice(0, half))], "doy")
    w_sum = (pad_m.fillna(0) * pad_c).rolling(doy=window, center=True).sum()
    c_sum = pad_c.rolling(doy=window, center=True).sum()
    clim = (w_sum / c_sum).isel(doy=slice(half, half + 365))
    years = len(np.unique(da.time.dt.year))
    clim = clim.where(c_sum.isel(doy=slice(half, half + 365)) >= min_years * min(window, 30) / 30 * (years > 0))
    return clim.assign_coords(doy=np.arange(1, 366)).rename("climatology")


def sm_anomaly(cube: xr.DataArray, *, clim: xr.DataArray | None = None,
               kind: Literal["absolute", "standardized"] = "absolute", window: int = 31, **kwargs) -> xr.DataArray:
    """Anomaly from a day-of-year climatology: absolute (m3 m-3) or standardized (z-score).  [Modules 9, 20]"""
    clim = sm_climatology(cube, window=window, **kwargs) if clim is None else clim
    doy = cube.time.dt.dayofyear.where(cube.time.dt.dayofyear < 366, 365)
    base = clim.sel(doy=doy).drop_vars("doy")
    anom = cube - base
    if kind == "standardized":
        sd = anom.groupby(doy.rename("doy")).std("time")
        sd = sd.reindex(doy=np.arange(1, 366)).rolling(doy=window, center=True, min_periods=1).mean()
        anom = anom / sd.sel(doy=doy).drop_vars("doy")
    return anom.rename("anomaly")


def _mann_kendall(y: np.ndarray, alpha: float):
    from scipy import stats

    y = y[np.isfinite(y)]
    n = len(y)
    if n < 6:
        return np.nan, np.nan, np.nan
    s = np.sign(y[None, :] - y[:, None])[np.triu_indices(n, 1)].sum()
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return p, tau, z


def trend(cube: xr.DataArray, *, freq: str = "YS", alpha: float = 0.05, min_count: int = 1) -> xr.Dataset:
    """Mann-Kendall test and Sen's slope per pixel on values aggregated to ``freq``.  [Module 9]

    Returns ``slope`` (change per year, Sen 1968), ``p_value``, Kendall's ``tau`` and a
    ``significant`` mask at ``alpha``. Aggregate periods with fewer than ``min_count`` daily values
    are dropped.
    """
    from scipy import stats

    agg = cube.resample(time=freq).mean()
    cnt = cube.notnull().resample(time=freq).sum()
    agg = agg.where(cnt >= min_count)
    years = (agg.time.dt.year + (agg.time.dt.dayofyear - 1) / 365.25).values

    def per_pixel(y):
        ok = np.isfinite(y)
        if ok.sum() < 6:
            return np.array([np.nan, np.nan, np.nan])
        slope = stats.theilslopes(y[ok], years[ok])[0]
        p, tau, _ = _mann_kendall(y, alpha)
        return np.array([slope, p, tau])

    res = xr.apply_ufunc(per_pixel, agg, input_core_dims=[["time"]], output_core_dims=[["stat"]],
                         vectorize=True, output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"stat": 3}})
    out = xr.Dataset({"slope": res.isel(stat=0), "p_value": res.isel(stat=1), "tau": res.isel(stat=2)})
    out["significant"] = out.p_value < alpha
    out.slope.attrs["units"] = f"{cube.attrs.get('units', '')} per year".strip()
    return out


# ============================================================================= 4. optical and thermal
def str_transform(swir) -> Any:
    """SWIR transformed reflectance ``STR = (1 - R)^2 / (2 R)`` (Sadeghi et al. 2017).  [Modules 2, 15]

    ``swir`` is surface reflectance (0 to 1) near 2.2 um (Sentinel-2 B12, Landsat band 7, MODIS
    band 7); values are clipped to 0.001 to 1.
    """
    r = np.clip(swir, 1e-3, 1.0)
    return _named((1.0 - r) ** 2 / (2.0 * r), "str", "1")


def _edge_fit(x, y, *, bins: int, q: float, x_range, min_count: int = 20):
    """Straight lines through the low and high ``q`` quantiles of ``y`` in bins of ``x``."""
    xv = np.asarray(getattr(x, "values", x), dtype="float64").ravel()
    yv = np.asarray(getattr(y, "values", y), dtype="float64").ravel()
    ok = np.isfinite(xv) & np.isfinite(yv) & (xv >= x_range[0]) & (xv <= x_range[1])
    xv, yv = xv[ok], yv[ok]
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    mid, lo, hi = [], [], []
    for a, b in itertools.pairwise(edges):
        sel = (xv >= a) & (xv < b)
        if sel.sum() >= min_count:
            mid.append(0.5 * (a + b))
            lo.append(np.quantile(yv[sel], q))
            hi.append(np.quantile(yv[sel], 1.0 - q))
    if len(mid) < 3:
        raise ValueError("too few populated bins to fit the edges")
    mid = np.asarray(mid)
    b_lo, a_lo = np.polyfit(mid, lo, 1)
    b_hi, a_hi = np.polyfit(mid, hi, 1)
    return (a_lo, b_lo), (a_hi, b_hi)


def optram_edges(str_, ndvi, *, ndvi_bins: int = 50, quantile: float = 0.01,
                 ndvi_range: tuple[float, float] = (0.05, 0.85)) -> dict[str, float]:
    """Fit the OPTRAM dry and wet edges ``STR = i + s NDVI`` (Sadeghi et al. 2017).  [Module 15]

    The dry edge runs through the low ``quantile`` of STR in NDVI bins, the wet edge through the
    high one. Returns ``i_d``, ``s_d`` (dry) and ``i_w``, ``s_w`` (wet).
    """
    (i_d, s_d), (i_w, s_w) = _edge_fit(ndvi, str_, bins=ndvi_bins, q=quantile, x_range=ndvi_range)
    return {"i_d": float(i_d), "s_d": float(s_d), "i_w": float(i_w), "s_w": float(s_w)}


def optram(str_, ndvi, edges: Mapping[str, float]) -> Any:
    """OPTRAM normalized moisture ``W = (i_d + s_d NDVI - STR) / (i_d - i_w + (s_d - s_w) NDVI)``, 0 to 1.  [Modules 2, 15]"""
    dry = edges["i_d"] + edges["s_d"] * ndvi
    wet = edges["i_w"] + edges["s_w"] * ndvi
    w = (str_ - dry) / (wet - dry)
    return _named(np.clip(w, 0.0, 1.0), "optram_w", "1")


def ts_vi_edges(lst, ndvi, *, ndvi_bins: int = 50, quantile: float = 0.01,
                ndvi_range: tuple[float, float] = (0.1, 0.85)) -> dict[str, float]:
    """Dry and wet edges ``Ts = a + b NDVI`` of the LST-NDVI triangle (Sandholt et al. 2002).  [Module 15]

    Returns ``a_dry``, ``b_dry`` (warm edge, high quantile of LST) and ``a_wet``, ``b_wet``.
    """
    (a_w, b_w), (a_d, b_d) = _edge_fit(ndvi, lst, bins=ndvi_bins, q=quantile, x_range=ndvi_range)
    return {"a_dry": float(a_d), "b_dry": float(b_d), "a_wet": float(a_w), "b_wet": float(b_w)}


def tvdi(lst, ndvi, edges: Mapping[str, float] | None = None) -> Any:
    """Temperature-Vegetation Dryness Index ``(Ts - Ts_wet) / (Ts_dry - Ts_wet)``, 0 (wet) to 1 (dry).  [Modules 2, 15]"""
    e = ts_vi_edges(lst, ndvi) if edges is None else edges
    t_dry = e["a_dry"] + e["b_dry"] * ndvi
    t_wet = e["a_wet"] + e["b_wet"] * ndvi
    return _named(np.clip((lst - t_wet) / (t_dry - t_wet), 0.0, 1.0), "tvdi", "1")


def soil_evaporative_efficiency(lst, ndvi, *, t_veg=None, ndvi_soil: float = 0.15,
                                ndvi_veg: float = 0.9, edges: Mapping[str, float] | None = None) -> Any:
    """Soil evaporative efficiency from the soil temperature end-members, as in DisPATCh.  [Modules 15, 16]

    The soil temperature is separated from LST with the vegetation fraction
    ``fv = (NDVI - NDVIs) / (NDVIv - NDVIs)``: ``Ts = (LST - fv Tv) / (1 - fv)``, and
    ``SEE = (Ts_max - Ts) / (Ts_max - Ts_min)`` with the dry and wet soil end-members taken from
    the LST-NDVI edges at bare soil (Merlin et al. 2012). ``t_veg`` defaults to the mean of the two
    edges at full cover. Pixels with ``fv > 0.9`` are NaN (too little soil to see).
    """
    e = ts_vi_edges(lst, ndvi) if edges is None else edges
    fv = np.clip((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil), 0.0, 1.0)
    tv = 0.5 * (e["a_dry"] + e["b_dry"] * ndvi_veg + e["a_wet"] + e["b_wet"] * ndvi_veg) if t_veg is None else t_veg
    ts_soil = (lst - fv * tv) / (1.0 - fv)
    t_max = e["a_dry"] + e["b_dry"] * ndvi_soil
    t_min = e["a_wet"] + e["b_wet"] * ndvi_soil
    see = np.clip((t_max - ts_soil) / (t_max - t_min), 0.0, 1.0)
    see = _where(fv <= 0.9, see, np.nan)
    return _named(see, "see", "1")


# ============================================================================= 5. SAR retrieval
def s1_soil_moisture_cube(aoi: Any, dates: Any, *, orbit: Literal["ascending", "descending"] | None = None,
                          relative_orbit: int | None = None, resolution: float = 20, crs: Any = None,
                          speckle: Literal["multitemporal", "refined_lee"] | None = "multitemporal",
                          size: int = 5) -> xr.Dataset:
    """Analysis-ready Sentinel-1 VV/VH cube (dB) for soil moisture from Planetary Computer RTC.  [Module 10]

    Loads ``sentinel-1-rtc`` (gamma0, terrain flattened), keeps one relative orbit (the most
    frequent one when ``relative_orbit`` is None) so every date has the same geometry, filters
    speckle (``rsense.sar.speckle.multitemporal``, Quegan and Yu 2001, or a refined Lee filter
    per date) and returns ``vv`` and ``vh`` in dB with ``relative_orbit`` and ``orbit_state`` in
    the attributes. ``crs`` defaults to the UTM zone of the AOI centre.
    """
    from collections import Counter

    from .access import stac_load, stac_search

    bbox = _bbox(aoi)
    query = {"sat:orbit_state": {"eq": orbit}} if orbit else None
    items = list(stac_search("sentinel-1-rtc", aoi=bbox, dates=_date_str(dates), query=query))
    items = [i for i in items if "vv" in i.assets and "vh" in i.assets]  # dual-polarization scenes only
    if not items:
        raise RuntimeError(f"no dual-polarization Sentinel-1 RTC scenes for {dates}")
    if relative_orbit is None:
        relative_orbit = Counter(i.properties["sat:relative_orbit"] for i in items).most_common(1)[0][0]
    items = [i for i in items if i.properties["sat:relative_orbit"] == relative_orbit]
    if crs is None:
        lon, lat = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        crs = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
    ds = stac_load(items, bands=["vv", "vh"], aoi=bbox, resolution=resolution, crs=crs,
                   chunks={"x": 1024, "y": 1024}, sensor="sentinel1").load()
    ds = ds.where(ds > 0)
    if speckle == "multitemporal":
        from .sar.speckle import multitemporal

        ds = xr.Dataset({v: multitemporal(ds[v], size=size) for v in ("vv", "vh")})
    elif speckle == "refined_lee":
        from .sar.speckle import refined_lee

        ds = xr.Dataset({v: xr.concat([refined_lee(ds[v].isel(time=t)) for t in range(ds.sizes["time"])], "time")
                         for v in ("vv", "vh")})
    out = 10.0 * np.log10(ds.where(ds > 0))
    out = out.sortby("time")
    out.attrs.update(relative_orbit=int(relative_orbit), orbit_state=items[0].properties["sat:orbit_state"],
                     units="dB", calibration="gamma0 RTC", source="Planetary Computer sentinel-1-rtc")
    for v in ("vv", "vh"):
        out[v].attrs["units"] = "dB"
    return out.rio.write_crs(crs)


def _date_str(dates) -> str:
    start, end = _dates(dates)
    return f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"


def sar_masks(cube: xr.Dataset, *, dem: xr.DataArray | None = None, landcover: xr.DataArray | None = None,
              max_slope: float = 10.0, water_db: float = -18.0, temperature=None,
              water_classes: Sequence[int] = (80,), urban_classes: Sequence[int] = (50,),
              var: str = "vv") -> xr.Dataset:
    """Boolean masks for open water, urban areas, steep slopes and frozen dates.  [Module 10]

    ``water`` is a median backscatter below ``water_db`` or a land cover class in ``water_classes``
    (ESA WorldCover 80); ``urban`` is WorldCover 50 (strong double bounce); ``steep`` is a slope above
    ``max_slope`` degrees from ``dem``; ``frozen`` (per date) is an air or soil temperature at or
    below 0 C (``temperature``, a series or cube aligned on ``time``). ``valid`` combines the static
    masks (no water, urban or steep pixel); combine it with ``~frozen`` for each date.
    """
    ref = cube[var]
    static = ref.isel(time=0, drop=True) if "time" in ref.dims else ref
    out = xr.Dataset()
    med = ref.median("time") if "time" in ref.dims else ref
    water = med < water_db
    if landcover is not None:
        lc = _match(landcover, static) if landcover.rio.crs else landcover
        water = water | lc.isin(list(water_classes))
        out["urban"] = lc.isin(list(urban_classes))
    out["water"] = water
    if dem is not None:
        d = _match(dem.astype("float32"), static, 1)
        res = abs(float(d.x[1] - d.x[0]))
        gy, gx = np.gradient(d.values.astype("float64"), res)
        out["steep"] = (("y", "x"), np.degrees(np.arctan(np.hypot(gx, gy))) > max_slope)
    valid = ~out["water"]
    for k in ("urban", "steep"):
        if k in out:
            valid = valid & ~out[k]
    if temperature is not None and "time" in ref.dims:
        t = temperature
        if isinstance(t, pd.Series):
            t = xr.DataArray(t.values, coords={"time": t.index.values}, dims="time")
        frozen = t.reindex(time=ref.time, method="nearest", tolerance=np.timedelta64(1, "D")) <= 0.0
        out["frozen"] = frozen.fillna(False).astype(bool)
    out["valid"] = valid
    return out


def normalize_incidence(sigma0_db, incidence_deg, *, ref_deg: float = 40.0, slope: float | None = None,
                        per_pixel: bool = False) -> Any:
    """Normalize backscatter (dB) to a reference angle: ``s_ref = s - b (theta - ref)``.  [Module 10]

    The slope ``b`` (dB per degree) is fitted by least squares over all valid values when not
    given, or per pixel along time with ``per_pixel=True`` (needs varying angles). The fitted slope
    is stored in ``attrs["incidence_slope"]``.
    """
    if slope is None:
        if per_pixel:
            x = incidence_deg - incidence_deg.mean("time")
            yv = sigma0_db - sigma0_db.mean("time")
            slope = (x * yv).sum("time") / (x * x).sum("time")
        else:
            xv = np.asarray(getattr(incidence_deg + 0 * sigma0_db, "values", incidence_deg)).ravel()
            yv = np.asarray(getattr(sigma0_db, "values", sigma0_db)).ravel()
            ok = np.isfinite(xv) & np.isfinite(yv)
            slope = float(np.polyfit(xv[ok], yv[ok], 1)[0])
    out = sigma0_db - slope * (incidence_deg - ref_deg)
    if isinstance(out, xr.DataArray):
        out.attrs.update(incidence_slope=slope if np.ndim(slope) == 0 else "per pixel", ref_deg=ref_deg)
    return out


def change_detection_references(sigma0_db: xr.DataArray, *, dry_quantile: float = 0.05,
                                wet_quantile: float = 0.95, min_obs: int = 20,
                                dim: str = "time") -> xr.Dataset:
    """Per-pixel dry and wet reference backscatter for TU Wien change detection.  [Module 11]

    The references are low and high quantiles of the time series (more robust than the minimum and
    maximum); ``sensitivity`` is their difference (dB). Pixels with fewer than ``min_obs`` dates are NaN.
    """
    n = sigma0_db.notnull().sum(dim)
    dry = sigma0_db.quantile(dry_quantile, dim).drop_vars("quantile")
    wet = sigma0_db.quantile(wet_quantile, dim).drop_vars("quantile")
    ok = n >= min_obs
    return xr.Dataset({"dry": dry.where(ok), "wet": wet.where(ok), "sensitivity": (wet - dry).where(ok), "n": n})


def change_detection(sigma0_db: xr.DataArray, *, references: xr.Dataset | None = None,
                     dim: str = "time", **kwargs) -> xr.DataArray:
    """Relative surface soil moisture (0 to 1) ``(sigma0 - dry) / (wet - dry)`` (Wagner et al. 1999).  [Module 11]"""
    ref = change_detection_references(sigma0_db, dim=dim, **kwargs) if references is None else references
    rel = (sigma0_db - ref["dry"]) / ref["sensitivity"]
    return _named(rel.clip(0.0, 1.0), "relative_sm", "1")


def short_term_change_detection(sigma0_db: xr.DataArray, coarse_sm: xr.DataArray, *,
                                min_obs: int = 10, dim: str = "time") -> xr.Dataset:
    """Disaggregate coarse soil moisture with fine backscatter, the SMAP active-passive baseline.  [Modules 11, 13]

    ``theta_fine = theta_coarse + beta (sigma_fine - sigma_coarse)`` (Das et al. 2014), where
    ``sigma_coarse`` is the mean backscatter (dB) of the fine pixels in each coarse cell and ``beta``
    (m3 m-3 per dB) is the slope of the coarse soil moisture against ``sigma_coarse`` over time,
    fitted per coarse cell. ``coarse_sm`` must have a CRS and share the dates of ``sigma0_db``
    (match them first). Returns the fine soil moisture and ``beta``.
    """
    from rasterio.enums import Resampling

    sig = sigma0_db.transpose(dim, "y", "x")
    coarse = coarse_sm.transpose(dim, "y", "x")
    sig, coarse = _spatial(sig), _spatial(coarse)
    sig_c = _match(sig, coarse, Resampling.average)  # mean backscatter of each coarse cell
    sig_c = sig_c.assign_coords({dim: coarse[dim]})
    xa = sig_c - sig_c.mean(dim)
    ya = coarse - coarse.mean(dim)
    ok = xa.notnull() & ya.notnull()
    beta = (xa * ya).where(ok).sum(dim) / (xa * xa).where(ok).sum(dim)
    beta = beta.where(ok.sum(dim) >= min_obs).rio.write_crs(coarse.rio.crs)
    up = {k: _match(v.rio.write_crs(coarse.rio.crs), sig.isel({dim: 0}), Resampling.nearest)
          for k, v in {"theta": coarse, "sig_c": sig_c}.items()}
    beta_f = _match(beta, sig.isel({dim: 0}), Resampling.nearest)
    up = {k: v.assign_coords({dim: sig[dim]}) for k, v in up.items()}
    fine = (up["theta"] + beta_f * (sig - up["sig_c"])).clip(0.02, 0.6)
    return xr.Dataset({"sm": fine.rename("sm"), "beta": beta_f})


def relative_to_volumetric(rel, *, porosity, residual=0.0) -> Any:
    """Scale relative moisture (0 to 1) to m3 m-3: ``theta = residual + rel (porosity - residual)``.  [Module 11]"""
    return _named(residual + rel * (porosity - residual), "theta", "m3 m-3")


def water_cloud_model(theta, veg, incidence_deg, *, A: float, B: float, C: float, D: float,
                      E: float = 1.0, db: bool = True) -> Any:
    """Forward Water Cloud Model (Attema and Ulaby 1978) with a soil term linear in dB.  [Module 12]

    ``sigma = A V^E cos(t) (1 - tau2) + tau2 10^((C + D theta) / 10)``, ``tau2 = exp(-2 B V / cos(t))``,
    where ``V`` is the vegetation descriptor (NDVI, LAI or VWC). Returns dB with ``db=True``.
    """
    cos = np.cos(np.deg2rad(incidence_deg))
    v = np.clip(veg, 0.0, None)
    tau2 = np.exp(-2.0 * B * v / cos)
    sig = A * v ** E * cos * (1.0 - tau2) + tau2 * 10.0 ** ((C + D * theta) / 10.0)
    return _named(10.0 * np.log10(sig) if db else sig, "sigma0", "dB" if db else "1")


def calibrate_wcm(sigma0_db, theta, veg, incidence_deg, *, groups=None,
                  bounds: Mapping[str, tuple[float, float]] | None = None, E: float = 1.0) -> pd.DataFrame:
    """Fit WCM parameters A, B, C and D by least squares in dB, optionally per group (crop, stage).  [Module 12]

    Inputs are 1-D arrays of matched samples. Returns one row per group with the parameters, the
    calibration RMSE (dB), the soil moisture RMSE of the inversion (m3 m-3) and ``n``.
    """
    from scipy.optimize import least_squares

    b = {"A": (0.0, 1.0), "B": (0.0, 2.0), "C": (-40.0, 0.0), "D": (0.0, 80.0), **(bounds or {})}
    lo = [b[k][0] for k in "ABCD"]
    hi = [b[k][1] for k in "ABCD"]
    df = pd.DataFrame({"s": np.asarray(sigma0_db, float).ravel(), "t": np.asarray(theta, float).ravel(),
                       "v": np.asarray(veg, float).ravel(), "i": np.broadcast_to(np.asarray(incidence_deg, float),
                                                                                  np.shape(np.asarray(sigma0_db).ravel())).ravel(),
                       "g": "all" if groups is None else np.asarray(groups).ravel()}).dropna()
    rows = {}
    for g, d in df.groupby("g"):
        if len(d) < 8:
            continue

        def res(p, d=d):
            return water_cloud_model(d.t.values, d.v.values, d.i.values, A=p[0], B=p[1], C=p[2], D=p[3], E=E) - d.s.values
        x0 = [0.1, 0.2, -18.0, 25.0]
        fit = least_squares(res, x0, bounds=(lo, hi))
        p = dict(zip("ABCD", fit.x))
        inv = invert_wcm(d.s.values, d.v.values, d.i.values, {**p, "E": E})
        rows[g] = {**p, "rmse_db": float(np.sqrt(np.mean(fit.fun ** 2))),
                   "rmse_theta": float(np.sqrt(np.nanmean((inv - d.t.values) ** 2))), "n": len(d)}
    return pd.DataFrame(rows).T.rename_axis("group")


def invert_wcm(sigma0_db, veg, incidence_deg, params: Mapping[str, float] | pd.DataFrame, *,
               groups=None, theta_range: tuple[float, float] = (0.02, 0.55)) -> Any:
    """Invert the calibrated WCM analytically to soil moisture (m3 m-3).  [Module 12]

    The vegetation term is removed and the attenuated soil term solved for theta. With a parameter
    table (from :func:`calibrate_wcm`) and ``groups`` each sample uses its group's parameters.
    Values outside ``theta_range`` are clipped; samples where the vegetation term exceeds the signal
    are NaN.
    """
    if isinstance(params, pd.DataFrame):
        if groups is None:
            params = params.iloc[0].to_dict()
        else:
            gv = np.asarray(groups)
            out = np.full(np.shape(sigma0_db), np.nan)
            for g, row in params.iterrows():
                sel = gv == g
                out[sel] = invert_wcm(np.asarray(sigma0_db)[sel], np.asarray(veg)[sel],
                                      np.broadcast_to(incidence_deg, np.shape(sigma0_db))[sel], row.to_dict(),
                                      theta_range=theta_range)
            return out
    A, B, C, D = (float(params[k]) for k in "ABCD")
    E = float(params.get("E", 1.0))
    cos = np.cos(np.deg2rad(incidence_deg))
    v = np.clip(veg, 0.0, None)
    tau2 = np.exp(-2.0 * B * v / cos)
    s_lin = 10.0 ** (np.asarray(sigma0_db) / 10.0) if not isinstance(sigma0_db, xr.DataArray) else 10.0 ** (sigma0_db / 10.0)
    soil = (s_lin - A * v ** E * cos * (1.0 - tau2)) / tau2
    soil = _where(soil > 0, soil, np.nan)
    theta = (10.0 * np.log10(soil) - C) / D
    return _named(np.clip(theta, *theta_range), "theta", "m3 m-3")


def oh_model(theta, rms_height_cm: float, incidence_deg, *, freq_ghz: float = 5.405,
             version: Literal[1992, 2004] = 2004, clay: float = 0.2) -> xr.Dataset | dict:
    """Oh semi-empirical bare-soil backscatter (dB) for VV, HH and HV.  [Module 12]

    Version 2004 (Oh 2004) uses soil moisture directly; version 1992 (Oh et al. 1992) uses the
    Fresnel reflectivity from the Mironov permittivity (``clay``). Valid for ks 0.1 to 6 and theta
    0.04 to 0.29 m3 m-3 (2004).
    """
    th = np.deg2rad(incidence_deg)
    k = 2.0 * np.pi * freq_ghz * 1e9 / LIGHT_SPEED / 100.0  # rad per cm
    ks = k * rms_height_cm
    mv = np.clip(theta, 1e-3, None)
    if version == 2004:
        p = 1.0 - (incidence_deg / 90.0) ** (0.35 * mv ** -0.65) * np.exp(-0.4 * ks ** 1.4)
        q = 0.095 * (0.13 + np.sin(1.5 * th)) ** 1.4 * (1.0 - np.exp(-1.3 * ks ** 0.9))
        hv = 0.11 * mv ** 0.7 * np.cos(th) ** 2.2 * (1.0 - np.exp(-0.32 * ks ** 1.8))
        vv = hv / q
        hh = p * vv
    else:
        eps = mironov_dielectric(theta, clay, freq_ghz=freq_ghz)
        g0 = np.abs((1.0 - np.sqrt(eps)) / (1.0 + np.sqrt(eps))) ** 2
        rv, rh = fresnel_reflectivity(eps, incidence_deg, pol="v"), fresnel_reflectivity(eps, incidence_deg, pol="h")
        p = (1.0 - (2.0 * th / np.pi) ** (1.0 / (3.0 * g0)) * np.exp(-ks)) ** 2
        q = 0.23 * np.sqrt(g0) * (1.0 - np.exp(-ks))
        g = 0.7 * (1.0 - np.exp(-0.65 * ks ** 1.8))
        vv = g * np.cos(th) ** 3 / np.sqrt(p) * (rv + rh)
        hh = p * vv
        hv = q * vv
    out = {"vv": 10 * np.log10(vv), "hh": 10 * np.log10(hh), "hv": 10 * np.log10(hv)}
    return xr.Dataset(out) if _is_xr(theta, incidence_deg) else out


def dubois_model(eps, rms_height_cm: float, incidence_deg, *, freq_ghz: float = 5.405) -> xr.Dataset | dict:
    """Dubois et al. (1995) VV and HH bare-soil backscatter (dB); valid for ks < 2.5 and theta > 30 degrees.  [Module 12]"""
    th = np.deg2rad(incidence_deg)
    lam = LIGHT_SPEED / (freq_ghz * 1e9) * 100.0  # cm
    ks = 2.0 * np.pi / lam * rms_height_cm
    er = np.real(eps)
    hh = 10 ** -2.75 * np.cos(th) ** 1.5 / np.sin(th) ** 5 * 10 ** (0.028 * er * np.tan(th)) * (ks * np.sin(th)) ** 1.4 * lam ** 0.7
    vv = 10 ** -2.35 * np.cos(th) ** 3 / np.sin(th) ** 3 * 10 ** (0.046 * er * np.tan(th)) * (ks * np.sin(th)) ** 1.1 * lam ** 0.7
    out = {"vv": 10 * np.log10(vv), "hh": 10 * np.log10(hh)}
    return xr.Dataset(out) if _is_xr(eps, incidence_deg) else out


def _iem(eps, ks, kl, th, *, correlation: str = "exponential", terms: int = 15):
    """IEM single-scattering backscatter (linear) for VV and HH (Fung et al. 1992)."""
    from math import factorial

    cos, sin = np.cos(th), np.sin(th)
    root = np.sqrt(eps - sin ** 2)
    rv = (eps * cos - root) / (eps * cos + root)
    rh = (cos - root) / (cos + root)
    ksz = ks * cos  # k s cos(theta), dimensionless
    fvv, fhh = 2.0 * rv / cos, -2.0 * rh / cos
    Fvv = 2.0 * sin ** 2 * (1 + rv) ** 2 / cos * ((1 - 1 / eps) + (eps - sin ** 2 - eps * cos ** 2) / (eps ** 2 * cos ** 2))
    Fhh = -2.0 * sin ** 2 * (1 + rh) ** 2 / cos * ((eps - sin ** 2 - cos ** 2) / cos ** 2)
    K = 2.0 * kl * sin  # 2 k sin times l
    out = {}
    for pol, f, F in (("vv", fvv, Fvv), ("hh", fhh, Fhh)):
        total = 0.0
        for n in range(1, terms + 1):
            inner = (2 * ksz) ** n * f * np.exp(-ksz ** 2) + ksz ** n * F / 2.0
            if correlation == "exponential":
                w = (kl / n) ** 2 * (1 + (K / n) ** 2) ** -1.5
            else:
                w = kl ** 2 / (2 * n) * np.exp(-K ** 2 / (4 * n))
            total = total + np.abs(inner) ** 2 * w / factorial(n)
        out[pol] = 0.5 * np.exp(-2 * ksz ** 2) * total / 1.0  # k^2 absorbed in the dimensionless kl
    return out


def iem_lut(theta: Sequence[float], rms_height_cm: Sequence[float], corr_length_cm: Sequence[float],
            incidence_deg: Sequence[float], *, freq_ghz: float = 5.405, pol: str = "vv",
            clay: float = 0.2, correlation: Literal["exponential", "gaussian"] = "exponential") -> xr.DataArray:
    """Look-up table of IEM backscatter (dB) over soil moisture, roughness and incidence.  [Module 12]

    Single-scattering IEM of Fung et al. (1992) with an exponential (or Gaussian) surface
    correlation function and the Mironov permittivity; valid for ks below about 3.
    """
    k = 2.0 * np.pi * freq_ghz * 1e9 / LIGHT_SPEED / 100.0
    t = np.asarray(theta, float)
    eps = np.asarray(mironov_dielectric(t, clay, freq_ghz=freq_ghz))
    out = np.empty((len(t), len(rms_height_cm), len(corr_length_cm), len(incidence_deg)))
    for j, s in enumerate(rms_height_cm):
        for m, ell in enumerate(corr_length_cm):
            for i, inc in enumerate(incidence_deg):
                sig = _iem(eps, k * s, k * ell, np.deg2rad(inc), correlation=correlation)[pol]
                out[:, j, m, i] = 10 * np.log10(sig)
    return xr.DataArray(out, dims=("theta", "rms_height_cm", "corr_length_cm", "incidence_deg"),
                        coords={"theta": t, "rms_height_cm": list(rms_height_cm), "corr_length_cm": list(corr_length_cm),
                                "incidence_deg": list(incidence_deg)}, name=f"sigma0_{pol}", attrs={"units": "dB"})


def invert_lut(sigma0_db, lut: xr.DataArray | xr.Dataset, *, incidence_deg=None, veg=None,
               cost: Literal["l2", "l1"] = "l2") -> Any:
    """Invert backscatter (dB) against a look-up table: the theta of the lowest cost entry.  [Module 12]

    ``lut`` comes from :func:`iem_lut` (a Dataset with one variable per polarization uses them all,
    with ``sigma0_db`` a dict or Dataset of the same names). The incidence angle is matched to the
    nearest LUT angle; roughness is a free parameter of the search.
    """
    luts = lut if isinstance(lut, xr.Dataset) else xr.Dataset({"s": lut})
    obs = sigma0_db if isinstance(sigma0_db, (dict, xr.Dataset)) else {"s": sigma0_db}
    names = list(luts.data_vars)
    first = np.asarray(obs[names[0]], float)
    shape = first.shape
    inc = np.broadcast_to(np.asarray(incidence_deg if incidence_deg is not None
                                     else luts.incidence_deg.values[0], float), shape).ravel()
    angles = luts.incidence_deg.values
    idx = np.abs(inc[:, None] - angles[None, :]).argmin(1)
    out = np.full(first.size, np.nan)
    thetas = luts.theta.values
    for a in np.unique(idx):
        sel = np.flatnonzero(idx == a)
        tabs = np.stack([luts[n].isel(incidence_deg=a).values.reshape(len(thetas), -1) for n in names])  # pol, theta, rough
        o = np.stack([np.asarray(obs[n], float).ravel()[sel] for n in names])  # pol, pixel
        d = tabs[:, None, :, :] - o[:, :, None, None]
        c = (d ** 2 if cost == "l2" else np.abs(d)).sum(0)  # pixel, theta, rough
        flat = c.reshape(len(sel), -1)
        best = np.nanargmin(np.where(np.isfinite(flat), flat, np.inf), axis=1)
        out[sel] = thetas[best // tabs.shape[-1]]
        out[sel[~np.isfinite(o).all(0)]] = np.nan
    res = out.reshape(shape)
    if isinstance(obs[names[0]], xr.DataArray):
        return _named(obs[names[0]].copy(data=res), "theta", "m3 m-3")
    return res


# ============================================================================= 6. downscaling
def nested_upsample(coarse: xr.DataArray, factor: int, *, method: Literal["nearest", "bilinear"] = "nearest") -> xr.DataArray:
    """Split every coarse cell into ``factor x factor`` sub-cells (nested grids such as EASE 9 km and 1 km).  [Modules 15 to 17]

    ``nearest`` copies each coarse value to its sub-cells; ``bilinear`` interpolates between coarse
    cell centres (edges by nearest). The sub-cell centres are exact, so a 9 km EASE-Grid 2.0 cube
    lands on the 1 km EASE-Grid 2.0 cells inside it.
    """
    xd, yd = coarse.rio.x_dim, coarse.rio.y_dim
    dx = float(coarse[xd][1] - coarse[xd][0])
    dy = float(coarse[yd][1] - coarse[yd][0])
    off = (np.arange(factor) - (factor - 1) / 2.0) / factor
    xf = (coarse[xd].values[:, None] + off[None, :] * dx).ravel()
    yf = (coarse[yd].values[:, None] + off[None, :] * dy).ravel()
    c = _spatial(coarse)
    if method == "nearest":
        out = c.isel({xd: np.repeat(np.arange(c.sizes[xd]), factor), yd: np.repeat(np.arange(c.sizes[yd]), factor)})
        out = out.assign_coords({xd: xf, yd: yf})
    else:
        out = c.interp({xd: xf, yd: yf}, method="linear")
        out = out.fillna(c.reindex({xd: xf, yd: yf}, method="nearest"))
    return out.rio.write_crs(coarse.rio.crs) if coarse.rio.crs else out


def block_mean(fine: xr.DataArray, factor: int) -> xr.DataArray:
    """Mean of every ``factor x factor`` block of a nested fine grid (NaN ignored): the coarse view of a fine map.  [Modules 15 to 17]"""
    xd, yd = fine.rio.x_dim, fine.rio.y_dim
    out = _spatial(fine).coarsen({xd: factor, yd: factor}, boundary="trim").mean()
    return out.rio.write_crs(fine.rio.crs) if fine.rio.crs else out


def pycnophylactic(coarse: xr.DataArray, factor: int, *, iterations: int = 30, additive: bool = True) -> xr.DataArray:
    """Smooth, mass-preserving interpolation of coarse values to nested sub-cells (Tobler 1979).  [Module 15]

    Starts from the nearest-neighbour copy, then repeatedly smooths with a 3 x 3 mean and restores
    each coarse cell's mean (``additive=True`` shifts the block, otherwise rescales it), so the
    surface is smooth across cell edges and every block averages exactly to its coarse value. A
    simple alternative to area-to-point kriging for spreading residuals.
    """
    xd, yd = coarse.rio.x_dim, coarse.rio.y_dim
    up = nested_upsample(coarse, factor)
    dims = [d for d in up.dims if d not in (xd, yd)]
    up = up.transpose(*dims, yd, xd)
    target = np.repeat(np.repeat(_spatial(coarse).transpose(*dims, yd, xd).values, factor, -2), factor, -1)
    z = up.values.astype("float64").copy()
    ok = np.isfinite(target)
    for _ in range(iterations):
        pad = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
        stack = np.stack([pad[..., i:i + z.shape[-2], j:j + z.shape[-1]] for i in range(3) for j in range(3)])
        z = np.nanmean(stack, axis=0)
        shp = z.shape[:-2] + (z.shape[-2] // factor, factor, z.shape[-1] // factor, factor)
        bm = np.nanmean(z.reshape(shp), axis=(-3, -1))
        bm = np.repeat(np.repeat(bm, factor, -2), factor, -1)
        z = z + (target - bm) if additive else z * np.where(bm != 0, target / bm, 1.0)
        z[~ok] = np.nan
    return up.copy(data=z.astype("float32"))


def predictor_stack(layers: Mapping[str, xr.DataArray], *, target: str | xr.DataArray = "EASE2_M01km",
                    aoi: Any = None, method: Literal["average", "bilinear", "nearest"] = "average",
                    categorical: Sequence[str] = ()) -> xr.Dataset:
    """Align predictor layers (LST, NDVI, OPTRAM, texture, terrain, land cover) on one grid.  [Module 15]

    ``target`` is an EASE-Grid 2.0 name (cells covering ``aoi``) or a DataArray whose grid is used.
    Continuous layers are aggregated with ``method``; layers named in ``categorical`` use the mode.
    Layers with a ``time`` dimension keep it.
    """
    from rasterio.enums import Resampling

    if isinstance(target, str):
        g = ease2_grid(target, aoi=aoi)
        tmpl = xr.DataArray(np.zeros((g.sizes["y"], g.sizes["x"]), "float32"), coords={"y": g.y, "x": g.x},
                            dims=("y", "x")).rio.write_crs("EPSG:6933")
    else:
        tmpl = target.isel({d: 0 for d in target.dims if d not in ("y", "x")}, drop=True)
    rs = {"average": Resampling.average, "bilinear": Resampling.bilinear, "nearest": Resampling.nearest}
    out = {}
    for name, da in layers.items():
        r = Resampling.mode if name in categorical else rs[method]
        da = da.astype("float32")
        dims = [d for d in da.dims if d not in ("y", "x")]
        if dims:
            da = da.transpose(*dims, "y", "x")
        out[name] = _match(da, tmpl, r)
    return xr.Dataset(out).rio.write_crs(tmpl.rio.crs)


def _to_fine(coarse: xr.DataArray, fine: xr.DataArray, resampling=None) -> xr.DataArray:
    from rasterio.enums import Resampling

    tmpl = fine.isel({d: 0 for d in fine.dims if d not in ("y", "x")}, drop=True)
    return _match(coarse, tmpl, resampling or Resampling.nearest)


def _to_coarse(fine: xr.DataArray, coarse: xr.DataArray) -> xr.DataArray:
    from rasterio.enums import Resampling

    tmpl = coarse.isel({d: 0 for d in coarse.dims if d not in ("y", "x")}, drop=True)
    return _match(fine, tmpl, Resampling.average)  # masked (NaN) pixels are left out of the average


def dispatch(coarse_sm: xr.DataArray, see: xr.DataArray, *, model: Literal["linear", "nonlinear"] = "linear",
             sm_p=None) -> xr.DataArray:
    """DisPATCh downscaling of coarse soil moisture with fine soil evaporative efficiency (Merlin et al. 2012).  [Module 16]

    ``SM_fine = SM_coarse + dSM/dSEE (SEE_fine - SEE_coarse)``, where ``SEE_coarse`` is the mean
    SEE of the fine pixels in each coarse cell and the derivative comes from the SEE model:
    linear ``SEE = SM / SMp`` gives ``dSM/dSEE = SMp``; nonlinear ``SEE = 0.5 - 0.5 cos(pi SM / SMp)``
    gives ``SMp / (pi sqrt(SEE (1 - SEE)))``. ``SMp`` (soil moisture parameter) defaults to
    ``SM_coarse / SEE_coarse`` per coarse cell (linear) and is clipped to 0.1 to 0.6. Both inputs
    need a CRS; the result is on the ``see`` grid.
    """
    see_c = _to_coarse(see, coarse_sm)
    if sm_p is None:
        if model == "linear":
            sm_p = (coarse_sm / see_c).clip(0.1, 0.6)
        else:
            sm_p = (np.pi * coarse_sm / np.arccos(1 - 2 * see_c.clip(0.01, 0.99))).clip(0.1, 0.6)
    sm_p = sm_p if np.ndim(sm_p) == 0 else _to_fine(sm_p.rio.write_crs(coarse_sm.rio.crs), see)
    sm_c_f = _to_fine(coarse_sm, see)
    see_c_f = _to_fine(see_c.rio.write_crs(coarse_sm.rio.crs), see)
    if model == "linear":
        deriv = sm_p
    else:
        s = see_c_f.clip(0.02, 0.98)
        deriv = sm_p / (np.pi * np.sqrt(s * (1 - s)))
    out = sm_c_f + deriv * (see - see_c_f)
    return _named(out.clip(0.01, 0.6).rio.write_crs(see.rio.crs), "sm_dispatch", "m3 m-3")


def thermal_inertia_downscale(coarse_sm: xr.DataArray, lst_diurnal: xr.DataArray, ndvi: xr.DataArray
                              ) -> xr.DataArray:
    """Downscale with the diurnal LST amplitude (thermal inertia) and NDVI by a coarse-scale regression.  [Module 16]

    ``SM = a + b dT + c NDVI`` is fitted across coarse cells (dT and NDVI aggregated to the coarse
    grid), and the fine values add ``b (dT_fine - dT_coarse) + c (NDVI_fine - NDVI_coarse)`` to the
    coarse soil moisture, so each coarse cell keeps its mean. Coefficients are in ``attrs``.
    """
    dt_c = _to_coarse(lst_diurnal, coarse_sm)
    nd_c = _to_coarse(ndvi, coarse_sm)
    df = pd.DataFrame({"sm": coarse_sm.values.ravel(), "dt": dt_c.values.ravel(), "nd": nd_c.values.ravel()}).dropna()
    X = np.column_stack([np.ones(len(df)), df.dt, df.nd])
    a, b, c = np.linalg.lstsq(X, df.sm.values, rcond=None)[0]
    sm_f = _to_fine(coarse_sm, ndvi)
    dt_cf = _to_fine(dt_c.rio.write_crs(coarse_sm.rio.crs), ndvi)
    nd_cf = _to_fine(nd_c.rio.write_crs(coarse_sm.rio.crs), ndvi)
    out = (sm_f + b * (lst_diurnal - dt_cf) + c * (ndvi - nd_cf)).clip(0.01, 0.6)
    out.attrs.update(a=float(a), b=float(b), c=float(c), n_cells=len(df))
    return _named(out, "sm_thermal_inertia", "m3 m-3")


def spatial_blocks(template: xr.DataArray, *, block_size_m: float = 50_000, n_folds: int = 5,
                   seed: int = 0) -> xr.DataArray:
    """Fold labels (0 to ``n_folds - 1``) on square spatial blocks, for leakage-free cross-validation.  [Modules 17, 24]

    Pixels in the same ``block_size_m`` square share a fold, so a model is never tested on
    neighbours of its training pixels. The template's ``x`` and ``y`` must be in metres.
    """
    bx = np.floor((template.x.values - template.x.values.min()) / block_size_m).astype(int)
    by = np.floor((template.y.values.max() - template.y.values) / block_size_m).astype(int)
    ids = by[:, None] * (bx.max() + 1) + bx[None, :]
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    folds = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds))
    lab = np.vectorize(folds.get)(ids)
    return xr.DataArray(lab, coords={"y": template.y, "x": template.x}, dims=("y", "x"), name="fold")


def _make_model(model: str, params: Mapping[str, Any] | None):
    params = dict(params or {})
    if model == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**{"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
                               "colsample_bytree": 0.8, "n_jobs": 4, **params})
    if model == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**{"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, **params})
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**{"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1, **params})


def _stack_samples(target: xr.DataArray | None, predictors: xr.Dataset) -> pd.DataFrame:
    ds = predictors.copy()
    if target is not None:
        ds["__y"] = target
    df = ds.to_dataframe().reset_index()
    return df


def ml_downscale(coarse_sm: xr.DataArray, predictors: xr.Dataset, *,
                 model: Literal["rf", "xgboost", "lightgbm"] = "xgboost", residual: bool = True,
                 folds: xr.DataArray | None = None, params: Mapping[str, Any] | None = None
                 ) -> tuple[xr.DataArray, dict[str, Any]]:
    """Train on coarse-aggregated predictors and predict at the fine scale.  [Module 17]

    The predictors (fine grid, with a CRS) are averaged to the coarse cells; a random forest,
    XGBoost or LightGBM model learns ``SM_coarse`` from them; spatial cross-validation by ``folds``
    (labels on the coarse grid, from :func:`spatial_blocks`) reports R2 and RMSE per fold; the
    model then predicts at fine resolution. With ``residual=True`` the coarse residual is added back
    so every coarse cell keeps its observed mean. Time dimensions are stacked as extra samples.
    Returns the fine map and a report (``model``, ``cv`` table, ``features``).
    """
    from sklearn.metrics import mean_squared_error, r2_score

    feats = list(predictors.data_vars)
    pc = xr.Dataset({k: _to_coarse(predictors[k], coarse_sm) for k in feats})
    pc = pc.assign_coords({d: coarse_sm[d] for d in coarse_sm.dims if d not in ("y", "x")})
    df = _stack_samples(coarse_sm.rename("__y"), pc)
    if folds is not None:
        f = folds.reindex_like(coarse_sm.isel({d: 0 for d in coarse_sm.dims if d not in ("y", "x")}, drop=True),
                               method="nearest")
        df = df.merge(f.to_dataframe(name="fold").reset_index()[["y", "x", "fold"]], on=["y", "x"], how="left")
    df = df.dropna(subset=feats + ["__y"])
    cv = []
    if folds is not None:
        for k in sorted(df.fold.dropna().unique()):
            tr, te = df[df.fold != k], df[df.fold == k]
            if len(te) < 5 or len(tr) < 20:
                continue
            m = _make_model(model, params).fit(tr[feats], tr["__y"])
            p = m.predict(te[feats])
            cv.append({"fold": int(k), "n_test": len(te), "r2": r2_score(te["__y"], p),
                       "rmse": float(np.sqrt(mean_squared_error(te["__y"], p)))})
    m = _make_model(model, params).fit(df[feats], df["__y"])
    fine_df = _stack_samples(None, predictors)
    ok = fine_df[feats].notnull().all(1)
    pred = np.full(len(fine_df), np.nan)
    pred[ok.values] = m.predict(fine_df.loc[ok, feats])
    tmpl = predictors[feats[0]]
    dims = list(tmpl.dims)
    placed = pd.Series(pred, index=pd.MultiIndex.from_frame(fine_df[dims])).to_xarray()  # place by coordinates, not by row order
    fine = placed.reindex({d: tmpl[d] for d in dims}).transpose(*dims).rio.write_crs(tmpl.rio.crs)
    if residual:
        pc_pred = _to_coarse(fine, coarse_sm)
        pc_pred = pc_pred.assign_coords({d: coarse_sm[d] for d in coarse_sm.dims if d not in ("y", "x")})
        res = (coarse_sm - pc_pred).rio.write_crs(coarse_sm.rio.crs)
        fine = fine + _to_fine(res, tmpl).assign_coords({d: tmpl[d] for d in tmpl.dims if d not in ("y", "x")}).fillna(0)
    fine = _named(fine.clip(0.01, 0.6), "sm_ml", "m3 m-3")
    return fine, {"model": m, "cv": pd.DataFrame(cv), "features": feats, "n_train": len(df)}


def explain_downscaling(model: Any, predictors: xr.Dataset | pd.DataFrame, *, sample: int = 5_000,
                        seed: int = 0) -> dict[str, Any]:
    """SHAP values and partial dependence of a fitted downscaling model.  [Module 17]

    Returns ``importance`` (mean absolute SHAP per feature), ``shap_values`` and the sample ``X``
    they explain, and ``partial_dependence`` (feature grid and mean prediction).
    """
    import shap

    X = predictors.to_dataframe()[list(predictors.data_vars)] if isinstance(predictors, xr.Dataset) else predictors
    X = X.dropna()
    X = X.sample(min(sample, len(X)), random_state=seed)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    pdp = {}
    for c in X.columns:
        grid = np.quantile(X[c], np.linspace(0.02, 0.98, 25))
        vals = []
        for g in grid:
            Xg = X.copy()
            Xg[c] = g
            vals.append(float(np.mean(model.predict(Xg))))
        pdp[c] = pd.DataFrame({c: grid, "prediction": vals})
    return {"importance": imp, "shap_values": sv, "X": X, "partial_dependence": pdp}


def _unet(in_ch: int, base: int = 32):
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(in_ch, base), block(base, 2 * base), block(2 * base, 4 * base)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
            self.d2 = block(4 * base, 2 * base)
            self.u1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
            self.d1 = block(2 * base, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return x[:, :1] + self.out(d1)  # residual: correction of the upsampled coarse field

    return UNet()


def unet_downscale(pairs: xr.Dataset, *, epochs: int = 50, patch: int = 64, batch: int = 16,
                   device: str | None = None, checkpoint: str | None = None, lr: float = 1e-3,
                   val_fraction: float = 0.2, seed: int = 0) -> tuple[Any, pd.DataFrame]:
    """Train a small residual U-Net that sharpens upsampled coarse soil moisture (GPU, CPU fallback).  [Module 17]

    ``pairs`` has ``coarse`` (coarse soil moisture upsampled to the fine grid), any number of fine
    covariates and ``target`` (fine soil moisture), each ``time, y, x``. The network learns the
    correction ``target - coarse`` from the covariates on random patches; the last
    ``val_fraction`` of the dates is held out. Returns the model and a loss history table.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = ["coarse"] + [v for v in pairs.data_vars if v not in ("coarse", "target")]
    X = np.stack([pairs[v].transpose("time", "y", "x").values for v in inputs], 1).astype("float32")
    Y = pairs["target"].transpose("time", "y", "x").values[:, None].astype("float32")
    stats = {"mean": np.nanmean(X, axis=(0, 2, 3)), "std": np.nanstd(X, axis=(0, 2, 3)) + 1e-6}
    Xn = (X - stats["mean"][None, :, None, None]) / stats["std"][None, :, None, None]
    Xn[:, 0] = X[:, 0]  # keep the coarse field in m3 m-3 for the residual connection
    mask = np.isfinite(Y) & np.isfinite(X).all(1, keepdims=True)
    Xn, Yf = np.nan_to_num(Xn), np.nan_to_num(Y)
    nt = X.shape[0]
    n_val = max(1, round(val_fraction * nt))
    train_t, val_t = np.arange(nt - n_val), np.arange(nt - n_val, nt)
    net = _unet(len(inputs)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    H, W = X.shape[2], X.shape[3]
    ps = min(patch, H - H % 4, W - W % 4)

    def batch_from(ts):
        t = rng.choice(ts, batch)
        i = rng.integers(0, H - ps + 1, batch)
        j = rng.integers(0, W - ps + 1, batch)
        xb = np.stack([Xn[a, :, b:b + ps, c:c + ps] for a, b, c in zip(t, i, j)])
        yb = np.stack([Yf[a, :, b:b + ps, c:c + ps] for a, b, c in zip(t, i, j)])
        mb = np.stack([mask[a, :, b:b + ps, c:c + ps] for a, b, c in zip(t, i, j)])
        return (torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev), torch.from_numpy(mb).to(dev))

    hist = []
    for ep in range(epochs):
        net.train()
        losses = []
        for _ in range(20):
            xb, yb, mb = batch_from(train_t)
            loss = ((net(xb) - yb) ** 2)[mb].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        net.eval()
        with torch.no_grad():
            xb, yb, mb = batch_from(val_t)
            vl = float(((net(xb) - yb) ** 2)[mb].mean())
        hist.append({"epoch": ep + 1, "train_rmse": float(np.sqrt(np.mean(losses))), "val_rmse": float(np.sqrt(vl))})
    net.stats, net.inputs = stats, inputs
    if checkpoint:
        torch.save({"state": net.state_dict(), "stats": stats, "inputs": inputs}, checkpoint)
    return net, pd.DataFrame(hist)


def predict_unet(model: Any, coarse_sm: xr.DataArray, predictors: xr.Dataset | None = None,
                 *, tile: int = 256) -> xr.DataArray:
    """Apply a trained U-Net to an upsampled coarse field (``time, y, x`` or ``y, x``) and its covariates.  [Module 17]"""
    import torch

    dev = next(model.parameters()).device
    single = "time" not in coarse_sm.dims
    c = coarse_sm.expand_dims(time=[0]) if single else coarse_sm
    layers = [c] + [(predictors[v].expand_dims(time=[0]) if single else predictors[v]) for v in model.inputs[1:]]
    X = np.stack([la.transpose("time", "y", "x").values for la in layers], 1).astype("float32")
    Xn = (X - model.stats["mean"][None, :, None, None]) / model.stats["std"][None, :, None, None]
    Xn[:, 0] = X[:, 0]
    Xn = np.nan_to_num(Xn)
    H, W = X.shape[2:]
    Hp, Wp = int(np.ceil(H / 4) * 4), int(np.ceil(W / 4) * 4)
    Xp = np.pad(Xn, ((0, 0), (0, 0), (0, Hp - H), (0, Wp - W)), mode="edge")
    model.eval()
    with torch.no_grad():
        out = np.concatenate([model(torch.from_numpy(Xp[t:t + 1]).to(dev)).cpu().numpy() for t in range(len(Xp))])
    out = out[:, 0, :H, :W]
    res = c.transpose("time", "y", "x").copy(data=out).where(np.isfinite(X).all(1))
    res = res.isel(time=0, drop=True) if single else res
    return _named(res.clip(0.01, 0.6), "sm_unet", "m3 m-3")


# ============================================================================= 7. gap filling and fusion
def gap_fraction(cube: xr.DataArray, *, dim: str = "time") -> xr.Dataset:
    """Share of missing dates per pixel (``per_pixel``) and missing pixels per date (``per_time``).  [Module 18]

    Pixels that are never observed (water, outside the swath) are left out of ``per_time``.
    """
    miss = cube.isnull()
    land = cube.notnull().any(dim)
    space = [d for d in cube.dims if d != dim]
    return xr.Dataset({"per_pixel": miss.mean(dim).where(land), "per_time": miss.where(land).mean(space)})


def _interp_gaps(da: xr.DataArray, max_gap: float) -> xr.DataArray:
    """Linear interpolation along ``time`` of gaps up to ``max_gap`` days (NumPy, no bottleneck)."""
    d = da.transpose("time", ...)
    t = (d.time.values - d.time.values[0]) / np.timedelta64(1, "D")
    vals = d.values.reshape(d.shape[0], -1).copy()
    for j in range(vals.shape[1]):
        v = vals[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 2 or ok.all():
            continue
        tv = t[ok]
        filled = np.interp(t, tv, v[ok], left=np.nan, right=np.nan)
        nxt = np.searchsorted(tv, t)  # gap length: next valid minus previous valid time
        prev_t = tv[np.clip(nxt - 1, 0, len(tv) - 1)]
        next_t = tv[np.clip(nxt, 0, len(tv) - 1)]
        span = next_t - prev_t
        step = np.median(np.diff(tv)) if len(tv) > 1 else 1.0
        v[~ok] = np.where(span[~ok] - step <= max_gap, filled[~ok], np.nan)  # missing days in the gap
    return d.copy(data=vals.reshape(d.shape)).transpose(*da.dims)


def fill_climatology_anomaly(cube: xr.DataArray, *, max_gap: int = 10, method: str = "linear",
                             window: int = 31) -> xr.DataArray:
    """Fill gaps by interpolating anomalies in time and adding the climatology back.  [Module 18]

    Gaps longer than ``max_gap`` days stay NaN. The climatology (:func:`sm_climatology`) carries the
    seasonal cycle across the gap, so only the anomaly is interpolated.
    """
    clim = sm_climatology(cube, window=window, min_years=1)
    doy = cube.time.dt.dayofyear.where(cube.time.dt.dayofyear < 366, 365)
    base = clim.sel(doy=doy).drop_vars("doy")
    anom = _interp_gaps(cube - base, max_gap)
    return _named(base + anom, "sm_filled", "m3 m-3")


def fill_dineof(cube: xr.DataArray, *, max_modes: int = 20, tol: float = 1e-4, cv_fraction: float = 0.02,
                max_iter: int = 100, seed: int = 0) -> xr.DataArray:
    """DINEOF reconstruction of missing values by iterated truncated EOFs (Beckers and Rixen 2003).  [Module 18]

    The anomaly matrix (time x pixel) has its gaps set to 0; for k = 1, 2, ... modes the gaps are
    replaced by the rank-k reconstruction until they converge. A random ``cv_fraction`` of the valid
    values is held out to choose k (the one with the lowest cross-validation RMSE, stored in
    ``attrs``). Pixels never observed stay NaN.
    """
    da = cube.transpose("time", ...)
    shape = da.shape
    M = da.values.reshape(shape[0], -1).astype("float64")
    land = np.isfinite(M).any(0)
    A = M[:, land]
    mean = np.nanmean(A, 0)
    std = np.nanstd(A) or 1.0
    Z = (A - mean) / std
    valid = np.isfinite(Z)
    rng = np.random.default_rng(seed)
    vi = np.flatnonzero(valid.ravel())
    hold = rng.choice(vi, max(1, int(cv_fraction * len(vi))), replace=False)
    miss = ~valid
    miss_cv = miss.copy().ravel()
    miss_cv[hold] = True
    miss_cv = miss_cv.reshape(Z.shape)
    truth = Z.ravel()[hold]

    def reconstruct(mask, k_max):
        X = np.where(mask, 0.0, np.nan_to_num(Z))
        errs = []
        for k in range(1, k_max + 1):
            for _ in range(max_iter):
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                R = (U[:, :k] * s[:k]) @ Vt[:k]
                delta = np.sqrt(np.mean((R[mask] - X[mask]) ** 2)) if mask.any() else 0.0
                X[mask] = R[mask]
                if delta < tol:
                    break
            errs.append(X.copy())
        return errs

    k_max = min(max_modes, min(Z.shape) - 1)
    cvs = reconstruct(miss_cv, k_max)
    rmse = [float(np.sqrt(np.mean((X.ravel()[hold] - truth) ** 2))) for X in cvs]
    best = int(np.argmin(rmse)) + 1
    X = reconstruct(miss, best)[-1]
    out = np.full(M.shape, np.nan)
    out[:, land] = X * std + mean
    res = da.copy(data=out.reshape(shape))
    res.attrs.update(n_modes=best, cv_rmse=rmse[best - 1] * std, units="m3 m-3")
    return res.rename("sm_dineof")


def fill_gp(cube: xr.DataArray, *, kernel: str = "matern", max_points: int = 800, seed: int = 0) -> xr.Dataset:
    """Gaussian process gap filling per date in space, with the predictive standard deviation.  [Module 18]

    For each date a GP (Matern or RBF kernel plus white noise, fitted on up to ``max_points``
    observed pixels) predicts the missing pixels of the land mask. Coordinates are scaled to km.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    rng = np.random.default_rng(seed)
    da = cube.transpose("time", "y", "x")
    land = da.notnull().any("time").values
    yy, xx = np.meshgrid(da.y.values / 1000.0, da.x.values / 1000.0, indexing="ij")
    P = np.column_stack([yy.ravel(), xx.ravel()])
    filled = da.values.copy()
    std = np.zeros_like(filled)
    base = Matern(length_scale=30.0, nu=1.5) if kernel == "matern" else RBF(length_scale=30.0)
    for t in range(da.sizes["time"]):
        v = filled[t].ravel()
        ok = np.isfinite(v)
        need = (~ok) & land.ravel()
        if ok.sum() < 10 or not need.any():
            continue
        idx = np.flatnonzero(ok)
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        gp = GaussianProcessRegressor(ConstantKernel(0.005) * base + WhiteKernel(1e-4), normalize_y=True)
        gp.fit(P[idx], v[idx])
        mu, sd = gp.predict(P[need], return_std=True)
        v[need] = mu
        s = std[t].ravel()
        s[need] = sd
        filled[t] = v.reshape(filled[t].shape)
        std[t] = s.reshape(filled[t].shape)
    return xr.Dataset({"sm": da.copy(data=filled), "std": da.copy(data=np.where(np.isfinite(filled), std, np.nan))})


def merge_products(cubes: Mapping[str, xr.DataArray], *, weights: Mapping[str, Any] | None = None,
                   errors: Mapping[str, Any] | None = None, rescale_to: str | None = None,
                   rescale: bool = True) -> xr.Dataset:
    """Weighted merge of products on one grid, each rescaled to a reference (mean and std per pixel).  [Modules 18, 21]

    Weights are given, or ``1 / error variance`` from ``errors`` (for example triple collocation
    error standard deviations), or equal. At each pixel and date the weights of the products present
    are renormalized, so the merge has no gaps where any product has data. ``rescale=False`` skips the
    rescaling (for products that already share a level, such as downscaled maps of one coarse product). Returns ``sm``,
    ``n_products`` and the effective ``weight`` of each product.
    """
    names = list(cubes)
    ref = rescale_to or names[0]
    r = cubes[ref]
    scaled = {}
    for k, da in cubes.items():
        if k == ref or not rescale:  # rescale=False keeps products that already share a level
            scaled[k] = da
        else:
            scaled[k] = (da - da.mean("time")) / da.std("time") * r.std("time") + r.mean("time")
    if weights is None:
        if errors is not None:
            weights = {k: 1.0 / (errors[k] ** 2) for k in names}
        else:
            weights = {k: 1.0 for k in names}
    num, den = 0.0, 0.0
    n = 0
    for k in names:
        w = xr.where(scaled[k].notnull(), weights[k], 0.0)
        num = num + scaled[k].fillna(0.0) * w
        den = den + w
        n = n + scaled[k].notnull()
    merged = (num / den).where(den > 0)
    out = xr.Dataset({"sm": merged.rename(None), "n_products": n})
    for k in names:
        out[f"weight_{k}"] = (xr.where(scaled[k].notnull(), weights[k], 0.0) / den).where(den > 0)
    return out


# ============================================================================= 8. in situ data
def read_ismn(path: str, *, aoi: Any = None, variable: str = "soil_moisture",
              depth: tuple[float, float] = (0.0, 0.05), good_flags_only: bool = True,
              networks: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ISMN downloads with the ``ismn`` reader (``pip install ismn``).  [Module 19]

    Returns station metadata (``network``, ``station``, ``lon``, ``lat``, depths) and a tidy
    ``station, time, sm`` table, keeping only ISMN quality flag ``G`` values when ``good_flags_only``.
    """
    try:
        from ismn.interface import ISMN_Interface
    except ImportError as err:
        raise ImportError("read_ismn needs the ismn package: pip install ismn") from err
    ds = ISMN_Interface(path, network=list(networks) if networks else None)
    bbox = _bbox(aoi)
    meta, rows = [], []
    for net, stat, sens in ds.collection.iter_sensors(variable=variable, depth=list(depth)):
        if bbox and not (bbox[0] <= stat.lon <= bbox[2] and bbox[1] <= stat.lat <= bbox[3]):
            continue
        d = sens.read_data()
        if good_flags_only and f"{variable}_flag" in d:
            d = d[d[f"{variable}_flag"] == "G"]
        meta.append({"network": net.name, "station": stat.name, "lon": stat.lon, "lat": stat.lat,
                     "depth_from": sens.depth.start, "depth_to": sens.depth.end, "sensor": sens.name})
        rows.append(pd.DataFrame({"station": stat.name, "time": d.index, "sm": d[variable].values}))
    return pd.DataFrame(meta), (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["station", "time", "sm"]))


def read_uscrn(station: str, years: Sequence[int]) -> pd.DataFrame:
    """Hourly USCRN data of one station (NOAA NCEI, no account): soil moisture at 5 to 100 cm (m3 m-3), soil and air temperature, precipitation.  [Module 19]

    ``station`` is the file name stem, e.g. ``"IA_Des_Moines_17_E"``. The index is the start of the
    hour in UTC.
    """
    import io

    import requests

    frames = []
    for y in years:
        url = f"https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{y}/CRNH0203-{y}-{station}.txt"
        r = requests.get(url, timeout=300)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None, dtype=str))
    if not frames:
        raise FileNotFoundError(f"no USCRN files for {station} in {list(years)}")
    raw = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(raw[1] + raw[2], format="%Y%m%d%H%M")
    cols = {"p_mm": 12, "t_air": 9, "sm_5": 28, "sm_10": 29, "sm_20": 30, "sm_50": 31, "sm_100": 32,
            "ts_5": 33, "ts_10": 34, "ts_20": 35, "ts_50": 36, "ts_100": 37}
    df = pd.DataFrame({k: pd.to_numeric(raw[i], errors="coerce") for k, i in cols.items()})
    df[df <= -99] = np.nan
    df.index = pd.DatetimeIndex(t - pd.Timedelta(hours=1), name="time_utc")
    df.attrs.update(station=station, network="USCRN", lon=float(pd.to_numeric(raw[6]).iloc[0]),
                    lat=float(pd.to_numeric(raw[7]).iloc[0]))
    return df


def read_scan(triplet: str, start: str, end: str) -> pd.DataFrame:
    """Daily SCAN data of one station (NRCS AWDB REST service, no account).  [Module 19]

    Soil moisture ``sm_<cm>`` at the 2, 4, 8, 20 and 40 inch sensors (5, 10, 20, 51, 102 cm) in
    m3 m-3, soil temperature ``ts_<cm>`` (C) and precipitation ``p_mm``.
    """
    import requests

    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
    r = requests.get(url, params={"stationTriplets": triplet, "elements": "SMS:*,PRCP,STO:*",
                                  "duration": "DAILY", "beginDate": start, "endDate": end}, timeout=600)
    r.raise_for_status()
    cols = {}
    for series in r.json()[0]["data"]:
        el = series["stationElement"]
        depth = el.get("heightDepth")
        cm = None if depth is None else round(abs(depth) * 2.54)
        key = {"SMS": f"sm_{cm}", "STO": f"ts_{cm}", "PRCP": "p_mm"}[el["elementCode"]]
        vals = pd.Series({pd.Timestamp(v["date"]): v.get("value") for v in series["values"]}, dtype=float)
        vals = {"SMS": vals / 100.0, "PRCP": vals * 25.4, "STO": (vals - 32.0) / 1.8}[el["elementCode"]]
        cols[key] = vals
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df.attrs.update(station=triplet, network="SCAN")
    return df


def read_network(name: Literal["USCRN", "SCAN"], *, aoi: Any = None, dates: Any = None,
                 depth_cm: float = 5.0, stations: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stations of a public US network in an AOI and their daily soil moisture at one depth.  [Module 19]

    USCRN (5 to 100 cm) and SCAN (5 to 102 cm) need no account. Returns station metadata
    (``station``, ``network``, ``lon``, ``lat``, ``depth_cm``) and a tidy ``station, time, sm``
    table of daily means (m3 m-3) at the sensor nearest ``depth_cm``.
    """
    import io

    import requests

    start, end = _dates(dates)
    bbox = _bbox(aoi)
    meta, rows = [], []
    if name.upper() == "USCRN":
        tab = pd.read_csv(io.StringIO(requests.get("https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv",
                                                   timeout=120).text), sep="\t")
        tab["stem"] = tab.STATE + "_" + tab.LOCATION.str.replace(" ", "_") + "_" + tab.VECTOR.str.replace(" ", "_")
        if bbox:
            tab = tab[(tab.LONGITUDE >= bbox[0]) & (tab.LONGITUDE <= bbox[2]) & (tab.LATITUDE >= bbox[1]) & (tab.LATITUDE <= bbox[3])]
        if stations:
            tab = tab[tab.stem.isin(stations)]
        for _, s in tab.iterrows():
            try:
                h = read_uscrn(s.stem, range(start.year, end.year + 1))
            except FileNotFoundError:
                continue
            depths = [5, 10, 20, 50, 100]
            d = min(depths, key=lambda v: abs(v - depth_cm))
            daily = h[f"sm_{d}"].resample("1D").mean().loc[start:end]
            meta.append({"station": s.stem, "network": "USCRN", "lon": s.LONGITUDE, "lat": s.LATITUDE, "depth_cm": d})
            rows.append(pd.DataFrame({"station": s.stem, "time": daily.index, "sm": daily.values}))
    elif name.upper() == "SCAN":
        r = requests.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations",
                         params={"networkCds": "SCAN", "returnForecastPointMetadata": "false"}, timeout=120)
        tab = pd.DataFrame(r.json())
        tab = tab[tab.stationTriplet.str.endswith(":SCAN")]
        if bbox:
            tab = tab[(tab.longitude >= bbox[0]) & (tab.longitude <= bbox[2]) & (tab.latitude >= bbox[1]) & (tab.latitude <= bbox[3])]
        if stations:
            tab = tab[tab.stationTriplet.isin(stations)]
        for _, s in tab.iterrows():
            d = read_scan(s.stationTriplet, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
            sms = [c for c in d if c.startswith("sm_")]
            if not sms:
                continue
            col = min(sms, key=lambda c: abs(int(c[3:]) - depth_cm))
            meta.append({"station": s.stationTriplet, "network": "SCAN", "lon": s.longitude, "lat": s.latitude,
                         "depth_cm": int(col[3:])})
            rows.append(pd.DataFrame({"station": s.stationTriplet, "time": d.index, "sm": d[col].values}))
    else:
        raise ValueError("name must be 'USCRN' or 'SCAN' (use read_ismn for other networks)")
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["station", "time", "sm"])
    return pd.DataFrame(meta), data


def match_stations(stations: pd.DataFrame, cubes: Mapping[str, xr.Dataset | xr.DataArray], *,
                   method: Literal["nearest", "contain"] = "contain") -> pd.DataFrame:
    """Map each station (``lon``, ``lat`` columns) to the pixel of each product that contains it.  [Module 19]

    Works for EASE-Grid 2.0 cubes (``x``, ``y`` in EPSG:6933), projected rasters with a CRS and
    regular lat/lon grids. Returns one row per station and product with the pixel coordinates, its
    centre in lon/lat and the station's distance from the centre (m).
    """
    from pyproj import Geod, Transformer

    geod = Geod(ellps="WGS84")
    rows = []
    for name, cube in cubes.items():
        da = cube if isinstance(cube, xr.DataArray) else cube[next(iter(cube.data_vars))]
        latlon = "lat" in da.dims
        crs = None if latlon else (da.rio.crs or "EPSG:6933")
        for st, s in stations.iterrows():
            if latlon:
                py, px = float(s.lat), float(s.lon)
            else:
                px, py = Transformer.from_crs(4326, crs, always_xy=True).transform(s.lon, s.lat)
            xd, yd = ("lon", "lat") if latlon else ("x", "y")
            xs, ys = da[xd].values, da[yd].values
            ix, iy = int(np.abs(xs - px).argmin()), int(np.abs(ys - py).argmin())
            cx, cy = xs[ix], ys[iy]
            if latlon:
                clon, clat = cx, cy
            else:
                clon, clat = Transformer.from_crs(crs, 4326, always_xy=True).transform(cx, cy)
            dx = abs(xs[1] - xs[0]) if len(xs) > 1 else np.inf
            dy = abs(ys[1] - ys[0]) if len(ys) > 1 else np.inf
            inside = abs(px - cx) <= dx / 2 + 1e-9 and abs(py - cy) <= dy / 2 + 1e-9
            if method == "contain" and not inside:
                continue
            dist = geod.inv(s.lon, s.lat, clon, clat)[2]
            rows.append({"station": st, "product": name, xd: cx, yd: cy, "ix": ix, "iy": iy,
                         "pixel_lon": clon, "pixel_lat": clat, "distance_m": dist})
    return pd.DataFrame(rows)


def extract_at_stations(cube: xr.Dataset | xr.DataArray, matches: pd.DataFrame, *, var: str = "sm",
                        product: str | None = None) -> pd.DataFrame:
    """Pixel time series at matched stations, as a tidy ``station, time, value`` table.  [Module 19]"""
    da = cube[var] if isinstance(cube, xr.Dataset) else cube
    m = matches if product is None else matches[matches["product"] == product]
    xd, yd = ("lon", "lat") if "lat" in da.dims else ("x", "y")
    rows = []
    for _, r in m.iterrows():
        s = da.isel({xd: int(r.ix), yd: int(r.iy)}).to_series()
        rows.append(pd.DataFrame({"station": r.station, "time": s.index, "value": s.values}))
    return pd.concat(rows, ignore_index=True)


def representativeness(stations: pd.DataFrame, fine_sm: xr.DataArray, coarse_grid: str | xr.DataArray = "EASE2_M09km"
                       ) -> pd.DataFrame:
    """How well a point represents its coarse cell, from a fine soil moisture cube (``time, y, x``).  [Module 19]

    For each station: the fine pixel series at the station against the mean of the fine pixels in
    its coarse cell. Returns the spatial standard deviation within the cell (mean over dates), the
    bias and ubRMSE of point minus cell mean, and their correlation.
    """
    from pyproj import Transformer

    if isinstance(coarse_grid, str):
        b = fine_sm.rio.transform_bounds("EPSG:4326")
        g = ease2_grid(coarse_grid, aoi=b)
        coarse = xr.DataArray(np.zeros((g.sizes["y"], g.sizes["x"])), coords={"y": g.y, "x": g.x}, dims=("y", "x")).rio.write_crs(6933)
    else:
        coarse = coarse_grid
    cell = coarse.copy(data=np.arange(coarse.size).reshape(coarse.shape).astype("float64"))
    cell_f = _to_fine(cell, fine_sm)
    rows = []
    for st, s in stations.iterrows():
        px, py = Transformer.from_crs(4326, fine_sm.rio.crs, always_xy=True).transform(s.lon, s.lat)
        pt = fine_sm.sel(x=px, y=py, method="nearest")
        cid = float(cell_f.sel(x=px, y=py, method="nearest"))
        inside = fine_sm.where(cell_f == cid)
        mean = inside.mean(("y", "x"))
        sd = float(inside.std(("y", "x")).mean())
        d = (pt - mean).to_series().dropna()
        a = pd.concat([pt.to_series(), mean.to_series()], axis=1).dropna()
        rows.append({"station": st, "within_cell_std": sd, "bias": float(d.mean()), "ubrmse": float(d.std()),
                     "r": float(a.corr().iloc[0, 1]) if len(a) > 2 else np.nan, "n": len(a)})
    return pd.DataFrame(rows).set_index("station")


# ============================================================================= 9. validation
def _series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, xr.DataArray):
        return x.to_series()
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    raise TypeError("expected a pandas Series or a 1-D DataArray with a time index")


def temporal_match(reference: pd.DataFrame | pd.Series, *others: Any, window_hours: float = 12.0,
                   names: Sequence[str] | None = None, dropna: bool = True) -> pd.DataFrame:
    """Nearest-in-time matching of other series to a reference, within ``window_hours``.  [Module 20]

    Returns a table on the reference times with columns ``reference`` and one per other series
    (``names`` or ``s1``, ``s2``, ...); with ``dropna`` only complete rows are kept.
    """
    ref = _series(reference).dropna().sort_index()
    out = pd.DataFrame({"reference": ref})
    names = list(names) if names else [f"s{i + 1}" for i in range(len(others))]
    tol = pd.Timedelta(hours=window_hours)
    for nm, o in zip(names, others):
        s = _series(o).dropna().sort_index()
        m = pd.merge_asof(out[[]].reset_index().rename(columns={out.index.name or "index": "t"}),
                          s.rename(nm).reset_index().rename(columns={s.index.name or "index": "t"}),
                          on="t", direction="nearest", tolerance=tol)
        out[nm] = m[nm].values
    return out.dropna() if dropna else out


def rescale(src, ref, *, method: Literal["mean_std", "min_max", "cdf", "linreg"] = "mean_std",
            n_bins: int = 100):
    """Rescale ``src`` to the climatology of ``ref`` (paired samples), e.g. CDF matching.  [Modules 18, 20]

    ``mean_std`` matches mean and standard deviation, ``min_max`` the range, ``linreg`` regresses
    ``ref`` on ``src``, ``cdf`` maps percentiles (``n_bins`` of them, linear in between, linear
    extrapolation beyond). The parameters come from times where both are valid; every value of
    ``src`` is transformed. Returns the same type as ``src``.
    """
    s = np.asarray(getattr(src, "values", src), float)
    r = np.asarray(getattr(ref, "values", ref), float)
    ok = np.isfinite(s) & np.isfinite(r)
    a, b = s[ok], r[ok]
    if method == "mean_std":
        out = (s - a.mean()) / a.std() * b.std() + b.mean()
    elif method == "min_max":
        out = (s - a.min()) / (a.max() - a.min()) * (b.max() - b.min()) + b.min()
    elif method == "linreg":
        k, c = np.polyfit(a, b, 1)
        out = k * s + c
    elif method == "cdf":
        q = np.linspace(0, 100, n_bins + 1)
        ps, pr = np.percentile(a, q), np.percentile(b, q)
        ps, idx = np.unique(ps, return_index=True)
        pr = pr[idx]
        out = np.interp(s, ps, pr)
        lo, hi = s < ps[0], s > ps[-1]
        if len(ps) > 1:
            out[lo] = pr[0] + (s[lo] - ps[0]) * (pr[1] - pr[0]) / (ps[1] - ps[0])
            out[hi] = pr[-1] + (s[hi] - ps[-1]) * (pr[-1] - pr[-2]) / (ps[-1] - ps[-2])
    else:
        raise ValueError("method must be 'mean_std', 'min_max', 'linreg' or 'cdf'")
    out[~np.isfinite(s)] = np.nan
    if isinstance(src, pd.Series):
        return pd.Series(out, index=src.index, name=src.name)
    if isinstance(src, xr.DataArray):
        return src.copy(data=out)
    return out


def _anomaly_series(s: pd.Series, window: int) -> pd.Series:
    doy = np.minimum(np.asarray(s.index.dayofyear), 365)
    clim = s.groupby(doy).mean().reindex(range(1, 366))
    ext = pd.concat([clim.iloc[-window:], clim, clim.iloc[:window]])
    sm_ = ext.rolling(window, center=True, min_periods=1).mean().iloc[window:window + 365]
    sm_.index = range(1, 366)
    return s - sm_.reindex(doy).values


def validation_metrics(reference, estimate, *, anomalies: bool = False, clim_window: int = 35,
                       bootstrap: int = 0, ci: float = 0.95, seed: int = 0) -> pd.Series | pd.DataFrame:
    """bias, RMSE, ubRMSE, Pearson and Spearman R, anomaly R and n of paired samples.  [Module 20]

    ``bias`` is estimate minus reference; ``ubrmse = sqrt(RMSE^2 - bias^2)``; ``meets_target`` says
    whether ubRMSE is at most the SMAP requirement of 0.04 m3 m-3. With ``anomalies`` (needs time
    indexes) the correlation of anomalies from a ``clim_window``-day climatology is added. With
    ``bootstrap > 0`` a table with ``value``, ``lo`` and ``hi`` (percentile intervals) is returned.
    """
    from scipy import stats

    r_ = _series(reference) if not isinstance(reference, np.ndarray) else pd.Series(reference)
    e_ = _series(estimate) if not isinstance(estimate, np.ndarray) else pd.Series(estimate, index=r_.index)
    df = pd.concat([r_.rename("r"), e_.rename("e")], axis=1).dropna()

    def metrics(d):
        diff = d.e - d.r
        bias = diff.mean()
        rmse = np.sqrt((diff ** 2).mean())
        out = {"n": len(d), "bias": bias, "rmse": rmse, "ubrmse": np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)),
               "pearson_r": stats.pearsonr(d.r, d.e)[0] if len(d) > 2 else np.nan,
               "spearman_r": stats.spearmanr(d.r, d.e)[0] if len(d) > 2 else np.nan}
        if anomalies:
            a = pd.concat([_anomaly_series(d.r, clim_window), _anomaly_series(d.e, clim_window)], axis=1).dropna()
            out["anomaly_r"] = stats.pearsonr(a.iloc[:, 0], a.iloc[:, 1])[0] if len(a) > 2 else np.nan
        return out

    base = metrics(df)
    base["meets_target"] = bool(base["ubrmse"] <= SMAP_UBRMSE_TARGET)
    if bootstrap <= 0:
        return pd.Series(base)
    rng = np.random.default_rng(seed)
    boots = pd.DataFrame([metrics(df.iloc[rng.integers(0, len(df), len(df))].sort_index()) for _ in range(bootstrap)])
    a = (1 - ci) / 2
    tab = pd.DataFrame({"value": pd.Series(base).drop("meets_target"), "lo": boots.quantile(a), "hi": boots.quantile(1 - a)})
    tab.loc["n", ["lo", "hi"]] = np.nan
    return tab


def validation_report(pairs: pd.DataFrame, *, by: Sequence[str] = ("product",), reference: str = "reference",
                      estimate: str = "estimate", **kwargs) -> pd.DataFrame:
    """Metrics of :func:`validation_metrics` per group of a tidy table of matched pairs.  [Module 20]

    ``pairs`` has the ``reference`` and ``estimate`` columns, the grouping columns in ``by``
    (product, station, land cover, season, ...) and a ``time`` column or index when anomaly metrics
    are wanted.
    """
    rows = {}
    for key, d in pairs.groupby(list(by)):
        d = d.set_index("time") if "time" in d.columns else d
        m = validation_metrics(d[reference], d[estimate], **kwargs)
        rows[key if isinstance(key, tuple) else (key,)] = m if isinstance(m, pd.Series) else m["value"]
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=list(by)) if len(by) > 1 else out.index.get_level_values(0)
    out.index.name = by[0] if len(by) == 1 else out.index.name
    return out


def triple_collocation(x, y, z, *, method: Literal["classical", "extended"] = "classical",
                       ref: int = 0, min_n: int = 100) -> dict[str, Any]:
    """Error standard deviations, SNR and data-truth correlations of three collocated products.  [Module 21]

    Covariance notation (McColl et al. 2014; Gruber et al. 2016): the error variance of ``x`` is
    ``Qxx - Qxy Qxz / Qyz`` (likewise for y and z) in each product's own units; ``err_std_ref`` gives
    them in the units of the ``ref`` product (0, 1 or 2), ``snr_db`` the signal-to-noise ratio and
    ``rho2`` the squared correlation with the unknown truth (extended TC). The errors must be
    independent of each other and of the truth (for example a passive, an active and a model product).
    """
    a = np.column_stack([np.asarray(getattr(v, "values", v), float).ravel() for v in (x, y, z)])
    a = a[np.isfinite(a).all(1)]
    n = len(a)
    nan3 = np.full(3, np.nan)
    if n < min_n:
        return {"n": n, "err_std": nan3, "err_std_ref": nan3, "snr_db": nan3, "rho2": nan3, "beta": nan3}
    Q = np.cov(a.T)
    i, j, k = 0, 1, 2
    sig = np.array([Q[i, j] * Q[i, k] / Q[j, k], Q[i, j] * Q[j, k] / Q[i, k], Q[i, k] * Q[j, k] / Q[i, j]])
    err = np.diag(Q) - sig
    rho2 = sig / np.diag(Q)
    # scaling of each product to the reference: beta_m = signal sensitivity relative to ref
    sens = np.sqrt(np.clip(sig, 0, None))
    beta = sens[ref] / sens
    err_std = np.sqrt(np.clip(err, 0, None))
    return {"n": n, "err_std": err_std, "err_std_ref": err_std * beta, "snr_db": 10 * np.log10(sig / err),
            "rho2": rho2, "beta": beta, "method": method}


def triple_collocation_map(x: xr.DataArray, y: xr.DataArray, z: xr.DataArray, *, anomalies: bool = True,
                           names: Sequence[str] = ("x", "y", "z"), min_n: int = 100, ref: int = 0,
                           window: int = 31) -> xr.Dataset:
    """Per-pixel triple collocation over time for three cubes on one grid (``time, y, x``).  [Module 21]

    With ``anomalies`` the seasonal cycle (:func:`sm_climatology`) is removed first, since the
    shared seasonal signal violates the independence assumptions less than the raw series do.
    Returns ``err_std_<name>`` (in ``ref`` units), ``rho2_<name>``, ``snr_db_<name>`` and ``n``.
    """
    cubes = [x, y, z]
    if anomalies:
        cubes = [sm_anomaly(c, window=window) for c in cubes]
    cubes = xr.align(*cubes, join="inner")

    def per_pixel(a, b, c):
        r = triple_collocation(a, b, c, min_n=min_n, ref=ref)
        return np.concatenate([r["err_std_ref"], r["rho2"], r["snr_db"], [r["n"]]])

    res = xr.apply_ufunc(per_pixel, *cubes, input_core_dims=[["time"]] * 3, output_core_dims=[["stat"]],
                         vectorize=True, output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"stat": 10}})
    out = xr.Dataset()
    for i, nm in enumerate(names):
        out[f"err_std_{nm}"] = res.isel(stat=i)
        out[f"rho2_{nm}"] = res.isel(stat=3 + i)
        out[f"snr_db_{nm}"] = res.isel(stat=6 + i)
    out["n"] = res.isel(stat=9)
    return out


def tc_weights(errors: Mapping[str, Any]) -> dict[str, Any]:
    """Least-squares merging weights ``w_i = (1 / s_i^2) / sum_j (1 / s_j^2)`` from error standard deviations in common units.  [Modules 18, 21]"""
    inv = {k: 1.0 / (v ** 2) for k, v in errors.items()}
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()}


def plot_validation(pairs: pd.DataFrame, *, kind: Literal["timeseries", "scatter", "taylor"] = "timeseries",
                    ax=None, reference: str = "reference", by: str = "product", **kwargs):
    """Time series, scatter with a 1:1 line, or a Taylor diagram of matched pairs.  [Module 20]"""
    import matplotlib.pyplot as plt

    groups = list(pairs.groupby(by)) if by in pairs else [("estimate", pairs)]
    if kind == "taylor":
        fig = plt.figure(figsize=kwargs.get("figsize", (6, 6))) if ax is None else ax.figure
        ax = fig.add_subplot(111, projection="polar") if ax is None else ax
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ref_sd = None
        for name, d in groups:
            r = d[reference]
            e = d["estimate"]
            ref_sd = r.std()
            cor = np.corrcoef(r, e)[0, 1]
            ax.plot(np.arccos(np.clip(cor, -1, 1)), e.std() / ref_sd, "o", label=str(name))
        ax.plot(0, 1.0, "k*", ms=12, label="reference")
        ax.set_xlabel("normalized standard deviation")
        ax.set_title("Taylor diagram (angle = correlation)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        return ax
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 4) if kind == "timeseries" else (5, 5)))
    if kind == "timeseries":
        first = groups[0][1]
        t = first.get("time", first.index)
        ax.plot(t, first[reference], "k.", ms=3, label="in situ")
        for name, d in groups:
            ax.plot(d.get("time", d.index), d["estimate"], ".", ms=3, label=str(name))
        ax.set_ylabel("soil moisture (m3 m-3)")
    else:
        lo = float(np.nanmin(pairs[[reference, "estimate"]].values))
        hi = float(np.nanmax(pairs[[reference, "estimate"]].values))
        for name, d in groups:
            ax.scatter(d[reference], d["estimate"], s=6, alpha=0.5, label=str(name))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("in situ (m3 m-3)")
        ax.set_ylabel("satellite (m3 m-3)")
    ax.legend(fontsize=8)
    return ax


# ============================================================================= 10. root zone
def _expfilter_np(values: np.ndarray, days: np.ndarray, t: float) -> np.ndarray:
    """Recursive exponential filter along axis 0 with irregular sampling and gaps."""
    out = np.full(values.shape, np.nan)
    swi = np.full(values.shape[1:], np.nan)
    gain = np.ones(values.shape[1:])
    last = np.full(values.shape[1:], np.nan)
    for n in range(values.shape[0]):
        v = values[n]
        ok = np.isfinite(v)
        first = ok & ~np.isfinite(swi)
        swi = np.where(first, v, swi)
        gain = np.where(first, 1.0, gain)
        last = np.where(first, days[n], last)
        upd = ok & ~first
        dt = days[n] - last
        g = gain / (gain + np.exp(-dt / t))
        swi = np.where(upd, swi + g * (v - swi), swi)
        gain = np.where(upd, g, gain)
        last = np.where(upd, days[n], last)
        out[n] = swi
    return out


def exponential_filter(surface_sm, *, t: float = 10.0, dim: str = "time") -> Any:
    """Soil Water Index by the recursive exponential filter (Wagner et al. 1999; Albergel et al. 2008).  [Modules 8, 22]

    ``SWI_n = SWI_{n-1} + K_n (ms_n - SWI_{n-1})`` with gain ``K_n = K_{n-1} / (K_{n-1} + exp(-dt / T))``,
    so irregular sampling and gaps are handled. ``t`` is the characteristic time in days. Accepts a
    Series with a DatetimeIndex or a DataArray with a ``time`` dimension; the filtered value is
    carried forward between observations.
    """
    if isinstance(surface_sm, pd.Series):
        s = surface_sm.sort_index()
        days = (s.index - s.index[0]).total_seconds().values / 86400.0
        return pd.Series(_expfilter_np(s.values[:, None].astype(float), days, t)[:, 0], index=s.index, name=f"swi_t{t:g}")
    da = surface_sm.transpose(dim, ...)
    days = (da[dim].values - da[dim].values[0]) / np.timedelta64(1, "D")
    vals = da.values.reshape(da.shape[0], -1).astype(float)
    out = _expfilter_np(vals, days.astype(float), t).reshape(da.shape)
    return da.copy(data=out).rename(f"swi_t{t:g}")


def optimize_t(surface_sm, rootzone_ref, *, t_values: Sequence[float] = tuple(range(1, 61)),
               metric: Literal["r", "ubrmse", "nse"] = "r") -> pd.DataFrame:
    """Score the exponential filter for each characteristic time against a root-zone reference.  [Module 22]

    Both are Series; the filtered series is matched to the reference dates. Returns ``T``, ``r``,
    ``ubrmse`` (after mean and std matching) and ``nse``, with the best T (by ``metric``) in
    ``attrs["best_T"]``.
    """
    ref = _series(rootzone_ref).dropna()
    rows = []
    for t in t_values:
        swi = exponential_filter(_series(surface_sm).dropna(), t=t)
        swi = swi.reindex(ref.index, method="ffill")
        d = pd.concat([ref.rename("r"), swi.rename("s")], axis=1).dropna()
        sc = rescale(d.s, d.r, method="mean_std")
        err = sc - d.r
        nse = 1 - (err ** 2).sum() / ((d.r - d.r.mean()) ** 2).sum()
        rows.append({"T": t, "r": d.r.corr(d.s), "ubrmse": float(np.sqrt(((err - err.mean()) ** 2).mean())), "nse": nse, "n": len(d)})
    tab = pd.DataFrame(rows)
    best = tab.loc[tab[metric].idxmin() if metric == "ubrmse" else tab[metric].idxmax(), "T"]
    tab.attrs["best_T"] = float(best)
    return tab


def bucket_model(precip, pet, *, capacity: float = 150.0, initial: float | None = None,
                 drainage: float = 0.05, et_threshold: float = 0.7) -> pd.DataFrame:
    """Single-layer bucket water balance (mm) for root-zone soil moisture (Manabe 1969 style).  [Module 22]

    Each day: storage gains precipitation, loses ``ET = PET min(1, S / (et_threshold capacity))`` and
    drainage ``drainage S``, and spills runoff above ``capacity``. Returns ``storage``, ``rel``
    (storage / capacity), ``et``, ``drainage`` and ``runoff`` per day. Missing inputs count as 0.
    """
    p = _series(precip).fillna(0.0)
    e = _series(pet).reindex(p.index).fillna(0.0)
    s = 0.5 * capacity if initial is None else initial
    rows = []
    for pv, ev in zip(p.values, e.values):
        s = s + pv
        et = ev * min(1.0, s / (et_threshold * capacity))
        s = max(s - et, 0.0)
        dr = drainage * s
        s -= dr
        ro = max(s - capacity, 0.0)
        s -= ro
        rows.append((s, s / capacity, et, dr, ro))
    return pd.DataFrame(rows, index=p.index, columns=["storage", "rel", "et", "drainage", "runoff"])


def enkf_assimilate(precip, pet, observations, *, capacity: float = 150.0, drainage: float = 0.05,
                    n_ensemble: int = 50, obs_error: float = 0.05, precip_error: float = 0.3,
                    model_error: float = 0.01, rescale_obs: bool = True, seed: int = 0) -> pd.DataFrame:
    """Ensemble Kalman Filter assimilation of surface soil moisture into the bucket model (NumPy).  [Module 22]

    The state is the relative storage ``S / capacity``. Each member gets multiplicative precipitation
    noise (lognormal, ``precip_error``) and additive state noise (``model_error``); on observation
    days (``observations``, a Series in m3 m-3 or relative units) the ensemble is updated with
    perturbed observations of error ``obs_error`` (in state units after rescaling). With
    ``rescale_obs`` the observations are first CDF matched to the open-loop climatology. Returns
    ``open_loop``, ``analysis`` (ensemble mean), ``spread``, ``obs`` and ``innovation`` per day.
    """
    rng = np.random.default_rng(seed)
    p = _series(precip).fillna(0.0)
    e = _series(pet).reindex(p.index).fillna(0.0)
    ol = bucket_model(p, e, capacity=capacity, drainage=drainage)["rel"]
    obs = _series(observations).reindex(p.index)
    if rescale_obs:
        obs = rescale(obs, ol.where(obs.notnull()), method="cdf")
    ens = np.full(n_ensemble, 0.5)
    rows = []
    for t, (pv, ev) in enumerate(zip(p.values, e.values)):
        s = ens * capacity + pv * rng.lognormal(-0.5 * precip_error ** 2, precip_error, n_ensemble)
        et = ev * np.minimum(1.0, s / (0.7 * capacity))
        s = np.maximum(s - et, 0.0)
        s = s - drainage * s
        s = np.minimum(s, capacity)
        ens = np.clip(s / capacity + rng.normal(0, model_error, n_ensemble), 0, 1)
        o = obs.iloc[t]
        innov = np.nan
        if np.isfinite(o):
            var = ens.var(ddof=1)
            k = var / (var + obs_error ** 2)
            pert = o + rng.normal(0, obs_error, n_ensemble)
            innov = o - ens.mean()
            ens = np.clip(ens + k * (pert - ens), 0, 1)
        rows.append((ol.iloc[t], ens.mean(), ens.std(ddof=1), o, innov))
    return pd.DataFrame(rows, index=p.index, columns=["open_loop", "analysis", "spread", "obs", "innovation"])


# ============================================================================= 11. drought and agriculture
def sm_percentile(cube: xr.DataArray, *, window: int = 31, period: tuple[Any, Any] | None = None,
                  min_samples: int = 30) -> xr.DataArray:
    """Day-of-year percentile (0 to 100) of soil moisture against its own record.  [Module 23]

    For each day of year the sample is every value within ``window`` days of that date in the
    ``period`` years (all years by default); the percentile is the Weibull plotting position
    ``100 rank / (n + 1)``. Cells with fewer than ``min_samples`` values are NaN.
    """
    da = cube.transpose("time", ...)
    ref = da.sel(time=slice(*period)) if period else da
    doy = da.time.dt.dayofyear.values.clip(max=365)
    rdoy = ref.time.dt.dayofyear.values.clip(max=365)
    vals, rv = da.values, ref.values
    out = np.full(vals.shape, np.nan)
    half = window // 2
    for d in np.unique(doy):
        dist = np.abs(((rdoy - d) + 182) % 365 - 182)
        sample = rv[dist <= half]
        n = np.isfinite(sample).sum(0)
        tgt = np.flatnonzero(doy == d)
        for i in tgt:
            v = vals[i]
            less = (sample < v[None]).sum(0) + 0.5 * (sample == v[None]).sum(0)
            pct = 100.0 * (less + 0.5) / (n + 1)
            out[i] = np.where((n >= min_samples) & np.isfinite(v), pct, np.nan)
    return da.copy(data=out).rename("percentile").assign_attrs(units="%")


def ssi(cube: xr.DataArray, *, scale: int = 1, freq: str = "MS",
        distribution: Literal["empirical", "beta", "gamma"] = "empirical") -> xr.DataArray:
    """Standardized Soil moisture Index on monthly means accumulated over ``scale`` months (Hao and AghaKouchak 2013).  [Module 23]

    Each calendar month is standardized separately: the empirical CDF (Gringorten plotting
    positions) or a fitted beta or gamma distribution is mapped to the standard normal.
    """
    from scipy import stats

    m = cube.resample(time=freq).mean()
    if scale > 1:
        m = m.rolling(time=scale, min_periods=scale).mean()

    def standardize(x):
        out = np.full(x.shape, np.nan)
        ok = np.isfinite(x)
        if ok.sum() < 5:
            return out
        v = x[ok]
        if distribution == "empirical":
            ranks = stats.rankdata(v)
            p = (ranks - 0.44) / (len(v) + 0.12)
        elif distribution == "beta":
            a, b, loc, sc = stats.beta.fit(np.clip(v, 1e-4, 1 - 1e-4), floc=0, fscale=1)
            p = stats.beta.cdf(v, a, b, loc, sc)
        else:
            a, loc, sc = stats.gamma.fit(v, floc=0)
            p = stats.gamma.cdf(v, a, loc, sc)
        out[ok] = stats.norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))
        return out

    res = m.copy()
    for month in range(1, 13):
        sel = m.time.dt.month == month
        if sel.any():
            sub = m.sel(time=sel)
            res.loc[{"time": sub.time}] = xr.apply_ufunc(standardize, sub, input_core_dims=[["time"]],
                                                         output_core_dims=[["time"]], vectorize=True).transpose(*sub.dims)
    return res.rename("ssi").assign_attrs(units="1", scale_months=scale)


def swdi(theta, field_capacity, wilting_point) -> Any:
    """Soil Water Deficit Index ``10 (theta - FC) / (FC - WP)`` (Martinez-Fernandez et al. 2015).  [Module 23]

    0 or more is no deficit; -10 is wilting point. Drought classes: mild 0 to -2, moderate -2 to
    -5, severe -5 to -10, extreme below -10.
    """
    return _named(10.0 * (theta - field_capacity) / (field_capacity - wilting_point), "swdi", "1")


def smapi(cube: xr.DataArray, *, clim: xr.DataArray | None = None, window: int = 31) -> xr.DataArray:
    """Soil Moisture Anomaly Percentage Index ``100 (theta - mean) / mean`` with a day-of-year mean.  [Module 23]"""
    clim = sm_climatology(cube, window=window) if clim is None else clim
    doy = cube.time.dt.dayofyear.where(cube.time.dt.dayofyear < 366, 365)
    base = clim.sel(doy=doy).drop_vars("doy")
    return (100.0 * (cube - base) / base).rename("smapi").assign_attrs(units="%")


def drought_events(index: xr.DataArray | pd.Series, *, threshold: float = 20.0, min_duration: int = 2,
                   below: bool = True) -> pd.DataFrame:
    """Run-theory drought events of a 1-D index (Yevjevich 1967): onset, end, duration, severity, peak.  [Module 23]

    An event is a run of at least ``min_duration`` consecutive time steps below (or above, with
    ``below=False``) ``threshold``. Severity is the summed departure from the threshold, intensity
    the severity per step and peak the most extreme value. Average a cube over an area first.
    """
    s = _series(index).dropna()
    flag = (s < threshold) if below else (s > threshold)
    runs = (flag != flag.shift()).cumsum()
    rows = []
    for _, d in s[flag].groupby(runs[flag]):
        if len(d) < min_duration:
            continue
        dep = (threshold - d) if below else (d - threshold)
        rows.append({"onset": d.index[0], "end": d.index[-1], "duration": len(d), "severity": float(dep.sum()),
                     "intensity": float(dep.mean()), "peak": float(d.min() if below else d.max()),
                     "peak_date": d.idxmin() if below else d.idxmax()})
    return pd.DataFrame(rows)


def drought_class(percentile: Any, *, scheme: Literal["usdm"] = "usdm") -> Any:
    """US Drought Monitor style classes from percentiles: -1 none, 0 (D0, at most 30) to 4 (D4, at most 2).  [Module 23]"""
    p = percentile
    cls = _where(p <= 2, 4, _where(p <= 5, 3, _where(p <= 10, 2, _where(p <= 20, 1, _where(p <= 30, 0, -1)))))
    return _where(np.isfinite(p) if not _is_xr(p) else p.notnull(), cls, np.nan)


def lagged_correlation(driver, response, *, max_lag: int = 6, dim: str = "time",
                       method: Literal["pearson", "spearman"] = "pearson") -> xr.Dataset:
    """Correlation of ``response`` with ``driver`` at lags 0 to ``max_lag`` (response after driver), and the best lag.  [Module 23]

    Works for Series (returned as a Dataset with a ``lag`` dimension) and DataArrays on one grid.
    """
    d = driver.to_xarray() if isinstance(driver, pd.Series) else driver
    r = response.to_xarray() if isinstance(response, pd.Series) else response
    d, r = xr.align(d, r, join="inner")
    if method == "spearman":
        d = d.rank(dim)
        r = r.rank(dim)
    cors = [xr.corr(d, r.shift({dim: -k}), dim) for k in range(max_lag + 1)]
    c = xr.concat(cors, "lag").assign_coords(lag=np.arange(max_lag + 1))
    best = c.fillna(-9).argmax("lag")
    return xr.Dataset({"r": c, "best_lag": best, "best_r": c.max("lag")})


def irrigation_signal(satellite_sm: xr.DataArray, model_sm: xr.DataArray, *, season: Sequence[int] = (6, 7, 8),
                      rescale_method: Literal["cdf", "mean_std"] = "mean_std",
                      reference_months: Sequence[int] = (4, 5, 10)) -> xr.Dataset:
    """Satellite minus model soil moisture in the irrigation season (Zaussinger et al. 2019 idea).  [Module 24]

    Models without irrigation miss the water applied to fields; satellites see it. The satellite
    series is first rescaled to the model per pixel using only ``reference_months`` (outside the
    main irrigation season), so a constant offset is removed; ``diff`` is then rescaled satellite
    minus model and ``score`` its mean in the ``season`` months of each year (``time`` = year).
    """
    sat, mod = xr.align(satellite_sm, model_sm, join="inner")
    refm = sat.time.dt.month.isin(list(reference_months))

    def fit(s, m):
        ok = np.isfinite(s) & np.isfinite(m) & refm.values
        out = np.full(s.shape, np.nan)
        if ok.sum() < 10:
            return out
        if rescale_method == "mean_std":
            a, b = s[ok], m[ok]
            out = (s - a.mean()) / a.std() * b.std() + b.mean()
        else:
            out = rescale(np.where(ok, s, np.nan), np.where(ok, m, np.nan), method="cdf")
            out = np.interp(s, np.sort(s[ok]), np.sort(out[ok]))
        return out

    scaled = xr.apply_ufunc(fit, sat, mod, input_core_dims=[["time"], ["time"]], output_core_dims=[["time"]],
                            vectorize=True).transpose(*sat.dims)
    diff = scaled - mod
    season_mask = diff.time.dt.month.isin(list(season))
    score = diff.where(season_mask).groupby("time.year").mean("time")
    return xr.Dataset({"diff": diff, "score": score})


def irrigation_events(sm_series: xr.DataArray | pd.Series, precip: xr.DataArray | pd.Series, *, rise: float = 0.05,
                      max_precip_mm: float = 2.0) -> pd.DataFrame | xr.Dataset:
    """Moisture jumps between consecutive observations without matching rainfall: candidate irrigation.  [Module 24]

    A step counts when soil moisture rises by more than ``rise`` (m3 m-3) and the precipitation
    summed over the interval is at most ``max_precip_mm``. Series give a table of events; cubes
    (``time, y, x`` with precipitation on the same grid or as one series) give ``event`` and ``count``.
    """
    if isinstance(sm_series, pd.Series):
        s = sm_series.dropna()
        p = _series(precip).fillna(0.0)
        rows = []
        for t0, t1 in zip(s.index[:-1], s.index[1:]):
            dp = p.loc[(p.index > t0) & (p.index <= t1)].sum()
            d = s.loc[t1] - s.loc[t0]
            if d > rise and dp <= max_precip_mm:
                rows.append({"start": t0, "end": t1, "rise": float(d), "precip_mm": float(dp)})
        return pd.DataFrame(rows)
    da = sm_series
    pr = precip.to_xarray() if isinstance(precip, pd.Series) else precip
    obs_t = da.time.values
    pc = pr.fillna(0).cumsum("time")
    pc_at = pc.reindex(time=obs_t, method="ffill")
    dp = pc_at.diff("time")
    vals = da.transpose("time", ...).values.copy()
    for i in range(1, len(vals)):  # carry the last observation forward
        vals[i] = np.where(np.isfinite(vals[i]), vals[i], vals[i - 1])
    ds = da.transpose("time", ...).copy(data=vals).diff("time")
    ev = (ds > rise) & (dp <= max_precip_mm)
    return xr.Dataset({"event": ev, "count": ev.sum("time")})


def crop_stage_features(cube: xr.DataArray, zones: Any, calendar: pd.DataFrame, *,
                        stats: Sequence[str] = ("mean", "min", "anomaly"), id_column: str | None = None,
                        mask: xr.DataArray | None = None) -> pd.DataFrame:
    """Zonal soil moisture features per zone, year and crop stage, for yield models.  [Module 24]

    ``zones`` is a GeoDataFrame (rasterized onto the cube grid, ``id_column`` names the zones) or
    a label DataArray; ``calendar`` has ``stage``, ``start`` and ``end`` as ``MM-DD`` strings.
    ``anomaly`` is the stage mean minus the all-year mean of that stage. ``mask`` (e.g. a crop
    mask) limits the pixels. Returns one row per zone and year with ``<stage>_<stat>`` columns.
    """
    tmpl = cube.isel(time=0, drop=True)
    if isinstance(zones, xr.DataArray):
        lab = _match(zones, tmpl) if zones.rio.crs else zones
        names = None
    else:
        from rasterio import features

        z = zones.to_crs(tmpl.rio.crs).reset_index(drop=True)
        names = z[id_column] if id_column else pd.Series(z.index)
        arr = features.rasterize(((g, i + 1) for i, g in enumerate(z.geometry)), out_shape=tmpl.shape,
                                 transform=tmpl.rio.transform(), fill=0, all_touched=True)
        lab = tmpl.copy(data=arr.astype("float64")).where(arr > 0)
    if mask is not None:
        lab = lab.where(_match(mask, tmpl) > 0 if mask.rio.crs else mask > 0)
    rows = []
    for year in np.unique(cube.time.dt.year):
        for _, st in calendar.iterrows():
            t0, t1 = pd.Timestamp(f"{year}-{st['start']}"), pd.Timestamp(f"{year}-{st['end']}")
            seg = cube.sel(time=slice(t0, t1))
            if seg.sizes["time"] == 0:
                continue
            mean_t = seg.mean("time")
            min_t = seg.min("time")
            for zid in np.unique(lab.values[np.isfinite(lab.values)]):
                sel = lab == zid
                rows.append({"zone": names.iloc[int(zid) - 1] if names is not None else int(zid), "year": int(year),
                             "stage": st["stage"], "mean": float(mean_t.where(sel).mean()), "min": float(min_t.where(sel).mean())})
    df = pd.DataFrame(rows)
    if "anomaly" in stats:
        df["anomaly"] = df["mean"] - df.groupby(["zone", "stage"])["mean"].transform("mean")
    keep = [s for s in stats if s in df]
    wide = df.pivot_table(index=["zone", "year"], columns="stage", values=keep)
    wide.columns = [f"{st}_{s}" for s, st in wide.columns]
    return wide.reset_index()


# ============================================================================= 12. hydrology and hazards
def antecedent_moisture(cube: xr.DataArray | pd.Series, events: pd.DataFrame | Sequence[Any], *,
                        days_before: int = 3, column: str = "start") -> pd.DataFrame:
    """Mean soil moisture in the ``days_before`` days before each event start (area mean for cubes).  [Module 25]"""
    starts = pd.to_datetime(events[column] if isinstance(events, pd.DataFrame) else list(events))
    s = cube if isinstance(cube, pd.Series) else cube.mean([d for d in cube.dims if d != "time"]).to_series()
    rows = []
    for t in starts:
        w = s.loc[t - pd.Timedelta(days=days_before):t - pd.Timedelta(days=1)]
        rows.append({"start": t, "antecedent_sm": float(w.mean()) if len(w) else np.nan, "n_days": int(w.notnull().sum())})
    return pd.DataFrame(rows)


def rain_events(precip: pd.Series, *, min_total_mm: float = 25.0, dry_days: int = 3) -> pd.DataFrame:
    """Storm events from a daily precipitation series: runs of wet days separated by ``dry_days`` dry days.  [Module 25]"""
    p = _series(precip).fillna(0.0)
    wet = p > 1.0
    rows, start, last, total = [], None, None, 0.0
    for t, v in p.items():
        if wet[t]:
            if start is None:
                start, total = t, 0.0
            total += v
            last = t
        elif start is not None and (t - last).days >= dry_days:
            if total >= min_total_mm:
                rows.append({"start": start, "end": last, "precip_mm": total})
            start = None
    return pd.DataFrame(rows)


def event_response(sm_pre: pd.DataFrame, flow: pd.Series, precip: pd.Series | None = None, *,
                   window_days: int = 5, area_km2: float | None = None) -> pd.DataFrame:
    """Peak and volume of streamflow after each event, and the runoff ratio with ``area_km2``.  [Module 25]

    ``sm_pre`` comes from :func:`antecedent_moisture` (one row per event start). ``flow`` is daily
    discharge (m3 s-1). The baseflow is the flow on the day before the event; quickflow volume is
    the flow above it summed over ``window_days`` days; the runoff ratio is quickflow depth over
    event precipitation (from ``precip`` over the same window).
    """
    q = _series(flow)
    rows = []
    for _, ev in sm_pre.iterrows():
        t = pd.Timestamp(ev["start"])
        win = q.loc[t:t + pd.Timedelta(days=window_days)]
        base = q.get(t - pd.Timedelta(days=1), np.nan)
        if win.empty or not np.isfinite(base):
            continue
        quick = (win - base).clip(lower=0).sum() * 86400.0  # m3
        row = {**ev.to_dict(), "base_m3s": base, "peak_m3s": float(win.max()), "peak_rise_m3s": float(win.max() - base),
               "quickflow_m3": float(quick)}
        if precip is not None:
            p = _series(precip).loc[t:t + pd.Timedelta(days=window_days)].sum()
            row["precip_mm"] = float(p)
            if area_km2:
                row["runoff_ratio"] = float(quick / (area_km2 * 1e6) * 1000.0 / p) if p > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hazard_indicator(sm: xr.DataArray, *, kind: Literal["flood", "landslide", "fire"] = "flood",
                     static: xr.Dataset | None = None, percentile: xr.DataArray | None = None) -> xr.DataArray:
    """Simple soil moisture hazard indicators (0 to 1).  [Module 25]

    ``flood``: degree of saturation above field capacity, ``(theta - FC) / (sat - FC)`` (``static``
    needs ``field_capacity`` and ``saturation``); ``landslide``: that wetness times the slope factor
    ``sin(slope)`` (``static.slope`` in degrees); ``fire``: dryness ``1 - percentile / 100`` (needs
    ``percentile`` from :func:`sm_percentile`).
    """
    if kind == "fire":
        if percentile is None:
            raise ValueError("kind='fire' needs percentile=")
        return (1 - percentile / 100.0).clip(0, 1).rename("fire_dryness")
    if static is None:
        raise ValueError(f"kind={kind!r} needs static= with field_capacity and saturation")
    wet = ((sm - static.field_capacity) / (static.saturation - static.field_capacity)).clip(0, 1)
    if kind == "flood":
        return wet.rename("flood_potential")
    return (wet * np.sin(np.deg2rad(static.slope))).clip(0, 1).rename("landslide_index")


__all__ = [
    "BANDS_GHZ",
    "CCI_FLAG_BITS",
    "EASE2_GRIDS",
    "NISAR_SM_GROUP",
    "PRODUCTS",
    "SMAP_QUAL_BITS",
    "SMAP_SURFACE_BITS",
    "SMAP_UBRMSE_TARGET",
    "VERSIONS",
    "antecedent_moisture",
    "block_mean",
    "bucket_model",
    "calibrate_wcm",
    "change_detection",
    "change_detection_references",
    "crop_stage_features",
    "daily_cube",
    "decode_flags",
    "degree_of_saturation",
    "dispatch",
    "dobson_dielectric",
    "download_product",
    "drought_class",
    "drought_events",
    "dubois_model",
    "ease2_grid",
    "ease2_index",
    "enkf_assimilate",
    "event_response",
    "explain_downscaling",
    "exponential_filter",
    "extract_at_stations",
    "fill_climatology_anomaly",
    "fill_dineof",
    "fill_gp",
    "free_water_dielectric",
    "fresnel_reflectivity",
    "gap_fraction",
    "harmonize",
    "hazard_indicator",
    "iem_lut",
    "invert_lut",
    "invert_tau_omega",
    "invert_wcm",
    "irrigation_events",
    "irrigation_signal",
    "lagged_correlation",
    "load_soilgrids",
    "mask_quality",
    "match_stations",
    "merge_products",
    "mironov_dielectric",
    "ml_downscale",
    "nested_upsample",
    "normalize_incidence",
    "oh_model",
    "open_cached",
    "open_product",
    "optimize_t",
    "optram",
    "optram_edges",
    "penetration_depth",
    "plot_validation",
    "porosity_from_bulk_density",
    "porosity_from_texture",
    "predict_unet",
    "predictor_stack",
    "product_info",
    "product_table",
    "pycnophylactic",
    "rain_events",
    "read_amsr2_lprm",
    "read_ascat",
    "read_clms_ssm",
    "read_era5_land",
    "read_esa_cci",
    "read_ismn",
    "read_network",
    "read_nisar_sm",
    "read_scan",
    "read_smap",
    "read_smos",
    "read_uscrn",
    "regrid",
    "relative_to_volumetric",
    "representativeness",
    "rescale",
    "rough_reflectivity",
    "s1_soil_moisture_cube",
    "sar_masks",
    "saturation_to_volumetric",
    "saxton_rawls",
    "search_product",
    "short_term_change_detection",
    "sm_anomaly",
    "sm_climatology",
    "sm_percentile",
    "smapi",
    "soil_evaporative_efficiency",
    "soil_water_status",
    "spatial_blocks",
    "ssi",
    "str_transform",
    "swdi",
    "tau_omega",
    "tb_sensitivity",
    "tc_weights",
    "temporal_match",
    "thermal_inertia_downscale",
    "to_cache",
    "topp",
    "topp_inverse",
    "trend",
    "triple_collocation",
    "triple_collocation_map",
    "ts_vi_edges",
    "tvdi",
    "unet_downscale",
    "validation_metrics",
    "validation_report",
    "vod_from_vwc",
    "volumetric_from_gravimetric",
    "vwc_from_ndvi",
    "water_cloud_model",
]
