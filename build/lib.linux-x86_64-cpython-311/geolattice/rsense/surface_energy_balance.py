"""Surface energy balance (SEB) models of evapotranspiration from satellite images.

The surface energy balance ``Rn = G + H + LE`` is solved for the latent heat flux ``LE`` (the
energy used by evapotranspiration, ET) from optical reflectance, land surface temperature (LST)
and a little weather. This module has the parts every model needs and the models themselves:

* constants, solar geometry and atmosphere (pressure, vapour pressure, air density,
  broadband transmissivity) and Open-Meteo weather access
* the radiation balance (broadband albedo, emissivity, LAI, incoming and outgoing short and
  longwave, net radiation), soil heat flux, roughness, aerodynamic resistance and
  Monin-Obukhov stability
* ASCE standardized and FAO-56 Penman-Monteith reference ET (hourly and daily)
* hot and cold anchor pixel selection (Allen et al. 2013 percentile method)
* one-source residual models SEBAL (Bastiaanssen et al. 1998) and METRIC (Allen et al. 2007)
* simplified and contextual models SSEBop (Senay et al. 2013), S-SEBI (Roerink et al. 2000),
  the Ts-vegetation triangle (Jiang and Islam 2001) and PT-JPL (Fisher et al. 2008)
* the two-source energy balance with Priestley-Taylor start (TSEB-PT, Norman et al. 1995)
* thermal sharpening (data mining sharpener, Gao et al. 2012)
* daily and seasonal upscaling, per zone water use, energy closure, accuracy metrics,
  sensitivity and anchor Monte Carlo

Units: temperatures in kelvin for energy balance functions and degrees Celsius for the
reference ET functions (as in ASCE 2005 and FAO-56), fluxes in W m-2, pressure and vapour
pressure in kPa, wind in m s-1, resistances in s m-1, ET in mm. Every function accepts NumPy
arrays or xarray DataArrays (2-D ``y, x`` rasters) and returns the same kind of object.

Quick start::

    import geolattice.rsense.surface_energy_balance as seb

    alb = seb.albedo(sr, sensor="landsat8")                 # sr: Dataset of surface reflectance
    lai = seb.leaf_area_index(seb.savi(sr.red, sr.nir))
    e0 = seb.surface_emissivity(lai, ndvi)
    rn = seb.net_radiation(rs_in, alb, rl_in, seb.outgoing_longwave(lst, e0), e0)
    g = seb.soil_heat_flux(rn, lst, alb, ndvi)
    anchors = seb.select_anchors(ndvi, lst, albedo=alb)
    out = seb.run_metric(lst, rn, g, seb.momentum_roughness(lai=lai), u200, pressure,
                         etr_inst=0.85, anchors=anchors, etr_daily=9.1)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .backend import to_cpu

# ----------------------------------------------------------------------------- constants
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W m-2 K-4
VON_KARMAN = 0.41  # von Karman constant
CP = 1004.0  # specific heat of air at constant pressure, J kg-1 K-1
GRAVITY = 9.81  # m s-2
R_DRY = 287.05  # gas constant of dry air, J kg-1 K-1
GSC = 1367.0  # solar constant, W m-2
GSC_MJ_MIN = 0.0820  # solar constant, MJ m-2 min-1 (FAO-56)
LAMBDA_DEFAULT = 2.45e6  # latent heat of vaporization at about 20 C, J kg-1

# Narrow to broadband albedo weights for surface reflectance, in band order
# (blue, green, red, nir, swir1, swir2) plus an offset.
ALBEDO_WEIGHTS = {
    # Liang (2001) as adapted for OLI by Smith (2010): no green band
    "landsat8": ({"blue": 0.356, "red": 0.130, "nir": 0.373, "swir16": 0.085, "swir22": 0.072},
                 -0.0018),
    "landsat9": ({"blue": 0.356, "red": 0.130, "nir": 0.373, "swir16": 0.085, "swir22": 0.072},
                 -0.0018),
    # Tasumi et al. (2008) at-surface weights for TM and ETM+ (bands 1 to 5 and 7)
    "tasumi": ({"blue": 0.254, "green": 0.149, "red": 0.147, "nir": 0.311, "swir16": 0.103,
                "swir22": 0.036}, 0.0),
    # Bonafoni and Sekertekin (2020) for Sentinel-2 MSI (B2, B3, B4, B8, B11, B12)
    "sentinel2": ({"blue": 0.2266, "green": 0.1236, "red": 0.1573, "nir": 0.3417,
                   "swir16": 0.1170, "swir22": 0.0338}, 0.0),
    # Liang (2001) for MODIS bands 1, 2, 3, 4, 5 and 7
    "modis": ({"red": 0.160, "nir": 0.291, "blue": 0.243, "green": 0.116, "swir12": 0.112,
               "swir22": 0.081}, -0.0015),
}
ALBEDO_WEIGHTS["landsat"] = ALBEDO_WEIGHTS["landsat8"]

# Band name aliases accepted by :func:`albedo` (GeoLattice and common product names)
_BAND_ALIASES = {
    "blue": ("blue", "B2", "B02", "SR_B2", "sur_refl_b03"),
    "green": ("green", "B3", "B03", "SR_B3", "sur_refl_b04"),
    "red": ("red", "B4", "B04", "SR_B4", "sur_refl_b01"),
    "nir": ("nir", "nir08", "B5", "B8", "B08", "SR_B5", "sur_refl_b02"),
    "swir12": ("swir12", "sur_refl_b05"),
    "swir16": ("swir16", "swir1", "B6", "B11", "SR_B6", "sur_refl_b06"),
    "swir22": ("swir22", "swir2", "B7", "B12", "SR_B7", "sur_refl_b07"),
}


# ----------------------------------------------------------------------------- helpers
def _is_xr(*objs) -> bool:
    return any(isinstance(o, (xr.DataArray, xr.Dataset)) for o in objs)


def _where(cond, a, b):
    """``np.where`` that keeps xarray objects as xarray."""
    if _is_xr(cond, a, b):
        return xr.where(cond, a, b)
    return np.where(cond, a, b)


def _np(a) -> np.ndarray:
    """Host NumPy copy of an array or DataArray."""
    a = to_cpu(a) if _is_xr(a) else a
    return np.asarray(getattr(a, "values", a), dtype="float64")


def _named(out, name: str, units: str | None = None, **attrs):
    """Name a DataArray result and record its units; pass NumPy results through."""
    if isinstance(out, xr.DataArray):
        out = out.rename(name)
        if units is not None:
            out.attrs["units"] = units
        out.attrs.update(attrs)
    return out


def _pixel(a, iy: int, ix: int) -> float:
    """Value of a 2-D array at (iy, ix), or the value itself for a scalar."""
    if np.ndim(a) == 0:
        return float(a)
    return float(_np(a)[iy, ix])


def _anchor_index(anchors: Mapping, key: str) -> tuple[int, int]:
    a = anchors[key]
    if isinstance(a, Mapping):
        return int(a["iy"]), int(a["ix"])
    return int(a[0]), int(a[1])


def _as_timestamp(when) -> pd.Timestamp:
    ts = pd.Timestamp(when)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def latent_heat_of_vaporization(t_k) -> Any:
    """Latent heat of vaporization (J kg-1) at temperature ``t_k`` (K):
    ``lambda = (2.501 - 0.00236 (T - 273.15)) 1e6``."""
    return (2.501 - 0.00236 * (t_k - 273.15)) * 1e6


# ----------------------------------------------------------------------------- solar geometry
def day_of_year(when) -> int:
    """Day of the year (1 to 366) of a date, datetime or string."""
    return int(_as_timestamp(when).day_of_year)


def solar_declination(doy) -> Any:
    """Solar declination (rad), FAO-56 eq. 24: ``0.409 sin(2 pi J / 365 - 1.39)``."""
    return 0.409 * np.sin(2 * np.pi * np.asarray(doy) / 365 - 1.39)


def inverse_relative_distance(doy) -> Any:
    """Inverse relative Earth-Sun distance ``dr = 1 + 0.033 cos(2 pi J / 365)`` (FAO-56 eq. 23)."""
    return 1 + 0.033 * np.cos(2 * np.pi * np.asarray(doy) / 365)


def equation_of_time(doy) -> Any:
    """Seasonal correction for solar time (hours), FAO-56 eq. 32."""
    b = 2 * np.pi * (np.asarray(doy) - 81) / 364
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)


def hour_angle(when, lon) -> Any:
    """Solar hour angle (rad) at UTC time ``when`` and longitude ``lon`` (degrees, east positive).

    Negative before solar noon, zero at solar noon, positive after.
    """
    ts = _as_timestamp(when)
    t = ts.hour + ts.minute / 60 + ts.second / 3600
    solar_time = t + np.asarray(lon) / 15 + equation_of_time(ts.day_of_year)
    return np.pi / 12 * (solar_time - 12)


def solar_position(lat, lon, when) -> dict[str, Any]:
    """Sun zenith and azimuth (degrees, azimuth clockwise from north) at UTC time ``when``.

    ``lat`` and ``lon`` may be scalars or arrays in degrees. Accurate to a few tenths of a
    degree, which is enough for energy balance work; use the scene metadata when you have it.
    """
    doy = day_of_year(when)
    dec = solar_declination(doy)
    phi = np.radians(lat)
    w = hour_angle(when, lon)
    cos_z = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(w)
    zenith = np.degrees(np.arccos(np.clip(cos_z, -1, 1)))
    az_south = np.arctan2(np.sin(w), np.cos(w) * np.sin(phi) - np.tan(dec) * np.cos(phi))
    azimuth = (np.degrees(az_south) + 180) % 360
    return {"zenith": zenith, "azimuth": azimuth, "elevation": 90 - zenith,
            "declination": dec, "hour_angle": w, "doy": doy}


def scene_geometry(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Overpass time and sun angles from a Landsat MTL dict or a STAC item's properties.

    Returns ``datetime`` (UTC), ``doy``, ``sun_elevation``, ``sun_zenith``, ``sun_azimuth``
    (degrees) and ``dr`` (inverse relative Earth-Sun distance, from ``EARTH_SUN_DISTANCE``
    when present).
    """
    if "SUN_ELEVATION" in meta:
        when = _as_timestamp(f"{meta['DATE_ACQUIRED']} {str(meta['SCENE_CENTER_TIME'])[:15]}")
        elev, azim = float(meta["SUN_ELEVATION"]), float(meta["SUN_AZIMUTH"])
        d = meta.get("EARTH_SUN_DISTANCE")
    else:
        when = _as_timestamp(meta["datetime"])
        elev = float(meta.get("view:sun_elevation", np.nan))
        azim = float(meta.get("view:sun_azimuth", np.nan))
        d = None
    doy = int(when.day_of_year)
    dr = 1 / float(d) ** 2 if d else float(inverse_relative_distance(doy))
    return {"datetime": when, "doy": doy, "sun_elevation": elev, "sun_zenith": 90 - elev,
            "sun_azimuth": azim, "dr": dr}


def slope_aspect(dem: xr.DataArray, *, res: float | None = None) -> tuple[Any, Any]:
    """Slope and aspect (degrees, aspect clockwise from north) of a projected DEM in metres.

    ``dem`` is a 2-D DataArray with ``y`` and ``x`` coordinates (or a NumPy array with ``res``).
    """
    z = _np(dem)
    if isinstance(dem, xr.DataArray) and res is None:
        dzdy, dzdx = np.gradient(z, dem["y"].values, dem["x"].values)
    else:
        if res is None:
            raise ValueError("pass res= (pixel size in metres) for a NumPy DEM")
        dzdy, dzdx = np.gradient(z, -res, res)  # rows run north to south
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360
    if isinstance(dem, xr.DataArray):
        slope = dem.copy(data=slope.astype("float32")).rename("slope").assign_attrs(units="deg")
        aspect = dem.copy(data=aspect.astype("float32")).rename("aspect").assign_attrs(units="deg")
    return slope, aspect


def cos_incidence(lat, when, lon, slope=0.0, aspect=180.0) -> Any:
    """Cosine of the solar incidence angle on a sloping surface (Duffie and Beckman; Allen 2006).

    ``slope`` and ``aspect`` in degrees, aspect clockwise from north. On flat ground this is
    the cosine of the sun zenith angle.
    """
    doy = day_of_year(when)
    dec = solar_declination(doy)
    phi = np.radians(lat)
    w = hour_angle(when, lon)
    s = np.radians(slope)
    gam = np.radians(aspect - 180.0)  # from south, east negative
    cos_t = (np.sin(dec) * np.sin(phi) * np.cos(s)
             - np.sin(dec) * np.cos(phi) * np.sin(s) * np.cos(gam)
             + np.cos(dec) * np.cos(phi) * np.cos(s) * np.cos(w)
             + np.cos(dec) * np.sin(phi) * np.sin(s) * np.cos(gam) * np.cos(w)
             + np.cos(dec) * np.sin(gam) * np.sin(s) * np.sin(w))
    return _named(cos_t, "cos_theta")


def sunset_hour_angle(lat, doy) -> Any:
    """Sunset hour angle ``ws = arccos(-tan(lat) tan(dec))`` (rad), FAO-56 eq. 25."""
    return np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(solar_declination(doy)), -1, 1))


def extraterrestrial_radiation_daily(lat, doy) -> Any:
    """Daily extraterrestrial radiation Ra (MJ m-2 d-1), FAO-56 eq. 21."""
    phi = np.radians(lat)
    dec = solar_declination(doy)
    ws = sunset_hour_angle(lat, doy)
    return (24 * 60 / np.pi * GSC_MJ_MIN * inverse_relative_distance(doy)
            * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)))


def extraterrestrial_radiation_hourly(lat, lon, when, *, period: float = 1.0) -> Any:
    """Extraterrestrial radiation Ra (MJ m-2 per period) for a period centred on UTC ``when``,
    FAO-56 eq. 28. ``period`` is the length in hours (1 for hourly data)."""
    doy = day_of_year(when)
    phi = np.radians(lat)
    dec = solar_declination(doy)
    w = hour_angle(when, lon)
    ws = sunset_hour_angle(lat, doy)
    w1 = np.clip(w - np.pi * period / 24, -ws, ws)
    w2 = np.clip(w + np.pi * period / 24, -ws, ws)
    ra = (12 * 60 / np.pi * GSC_MJ_MIN * inverse_relative_distance(doy)
          * ((w2 - w1) * np.sin(phi) * np.sin(dec)
             + np.cos(phi) * np.cos(dec) * (np.sin(w2) - np.sin(w1))))
    return np.maximum(ra, 0.0)


# ----------------------------------------------------------------------------- atmosphere
def pressure_from_elevation(z) -> Any:
    """Atmospheric pressure (kPa) from elevation (m): ``101.3 ((293 - 0.0065 z) / 293)^5.26``."""
    return _named(101.3 * ((293 - 0.0065 * z) / 293) ** 5.26, "pressure", "kPa")


def psychrometric_constant(pressure) -> Any:
    """Psychrometric constant (kPa C-1): ``0.000665 P``."""
    return 0.000665 * pressure


def saturation_vapour_pressure(t_c) -> Any:
    """Saturation vapour pressure (kPa) at air temperature ``t_c`` (C), FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def slope_vapour_pressure_curve(t_c) -> Any:
    """Slope of the saturation vapour pressure curve (kPa C-1), FAO-56 eq. 13."""
    return 4098 * saturation_vapour_pressure(t_c) / (t_c + 237.3) ** 2


def actual_vapour_pressure(*, t_c=None, rh=None, t_dew_c=None) -> Any:
    """Actual vapour pressure (kPa) from dew point (C), or from air temperature (C) and relative
    humidity (percent)."""
    if t_dew_c is not None:
        return saturation_vapour_pressure(t_dew_c)
    if t_c is None or rh is None:
        raise ValueError("pass t_dew_c, or t_c and rh")
    return saturation_vapour_pressure(t_c) * rh / 100.0


def air_density(pressure, t_k, ea=None) -> Any:
    """Air density (kg m-3) from pressure (kPa) and air temperature (K).

    With vapour pressure ``ea`` (kPa) the virtual temperature is used; without it the
    METRIC approximation ``rho = 1000 P / (1.01 T R)`` is applied.
    """
    if ea is None:
        return 1000 * pressure / (1.01 * t_k * R_DRY)
    tv = t_k / (1 - 0.378 * ea / pressure)
    return 1000 * pressure / (tv * R_DRY)


def precipitable_water(ea, pressure) -> Any:
    """Precipitable water in the atmosphere (mm), ASCE (2005): ``W = 0.14 ea P + 2.1``."""
    return 0.14 * ea * pressure + 2.1


def broadband_transmissivity(cos_theta, *, pressure=None, ea=None, z=None,
                             kt: float = 1.0, method: str = "asce") -> Any:
    """One-way broadband shortwave transmissivity of the atmosphere for clear sky.

    ``method="asce"``: ASCE (2005) / METRIC, ``0.35 + 0.627 exp(-0.00146 P / (Kt cos) -
    0.075 (W / cos)^0.4)``, needs ``pressure`` and ``ea`` (``kt`` is a turbidity coefficient,
    1 for clean air, 0.5 for very turbid air). ``method="elevation"``: ``0.75 + 2e-5 z``.
    """
    if method == "elevation":
        if z is None:
            raise ValueError("method='elevation' needs z")
        return 0.75 + 2e-5 * z
    if pressure is None or ea is None:
        raise ValueError("method='asce' needs pressure and ea")
    c = np.maximum(cos_theta, 0.05)
    w = precipitable_water(ea, pressure)
    tau = 0.35 + 0.627 * np.exp(-0.00146 * pressure / (kt * c) - 0.075 * (w / c) ** 0.4)
    return _named(tau, "tau_sw")


def weather_open_meteo(lat: float, lon: float, start, end, *,
                       hourly: Sequence[str] = ("temperature_2m", "relative_humidity_2m",
                                                "dew_point_2m", "wind_speed_10m",
                                                "shortwave_radiation", "surface_pressure"),
                       daily: Sequence[str] | None = None, timeout: float = 60
                       ) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Hourly (and optional daily) ERA5 based weather from the free Open-Meteo archive API.

    No key is needed. Times are UTC. Wind is converted to m s-1. With ``daily`` a dict with
    ``"hourly"`` and ``"daily"`` DataFrames is returned. Daily variables include
    ``et0_fao_evapotranspiration``, ``temperature_2m_max`` and ``shortwave_radiation_sum``.
    """
    import requests

    params = {"latitude": lat, "longitude": lon,
              "start_date": str(pd.Timestamp(start).date()),
              "end_date": str(pd.Timestamp(end).date()),
              "hourly": ",".join(hourly), "timezone": "UTC", "wind_speed_unit": "ms"}
    if daily:
        params["daily"] = ",".join(daily)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params,
                     timeout=timeout)
    r.raise_for_status()
    js = r.json()
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")
    h.attrs.update(latitude=js.get("latitude"), longitude=js.get("longitude"),
                   elevation=js.get("elevation"))
    if not daily:
        return h
    d = pd.DataFrame(js["daily"])
    d["time"] = pd.to_datetime(d["time"])
    return {"hourly": h, "daily": d.set_index("time")}


def interpolate_weather(df: pd.DataFrame, when) -> pd.Series:
    """Linearly interpolate an hourly weather table (DatetimeIndex, UTC) to the overpass time."""
    ts = _as_timestamp(when)
    num = df.select_dtypes("number")
    idx = num.index.union(pd.DatetimeIndex([ts]))
    return num.reindex(idx).interpolate(method="time").loc[ts]


# ----------------------------------------------------------------------------- vegetation
def ndvi(red, nir) -> Any:
    """Normalized difference vegetation index ``(nir - red) / (nir + red)``."""
    return _named((nir - red) / (nir + red), "ndvi")


def savi(red, nir, L: float = 0.5) -> Any:
    """Soil adjusted vegetation index ``(1 + L)(nir - red) / (L + nir + red)``.

    METRIC was calibrated in Idaho with ``L = 0.1``; 0.5 is the usual general value.
    """
    return _named((1 + L) * (nir - red) / (L + nir + red), "savi")


def leaf_area_index(savi_, *, max_lai: float = 6.0) -> Any:
    """Leaf area index from SAVI (Allen et al. 2002, METRIC):
    ``LAI = -ln((0.69 - SAVI) / 0.59) / 0.91``, capped at ``max_lai``, zero below SAVI 0.1."""
    s = np.clip(savi_, -1.0, 0.687)
    lai = -np.log((0.69 - s) / 0.59) / 0.91
    lai = _where(savi_ >= 0.687, max_lai, _where(lai < 0, 0.0, lai))
    return _named(np.minimum(lai, max_lai), "lai", "m2/m2")


def fractional_cover(ndvi_, ndvi_soil: float = 0.15, ndvi_veg: float = 0.85,
                     power: float = 1.0) -> Any:
    """Fractional vegetation cover ``((NDVI - NDVIs) / (NDVIv - NDVIs))^power``, clipped 0 to 1."""
    return _named(np.clip((ndvi_ - ndvi_soil) / (ndvi_veg - ndvi_soil), 0, 1) ** power, "fc")


# ----------------------------------------------------------------------------- radiation balance
def albedo(sr: xr.Dataset | Mapping[str, Any], *, sensor: str = "landsat8",
           weights: Mapping[str, float] | None = None, offset: float | None = None) -> Any:
    """Broadband surface albedo from surface reflectance bands (0 to 1).

    ``sr`` is a Dataset or a dict of bands named ``blue, green, red, nir, swir16, swir22``
    (aliases such as ``SR_B2`` or ``B04`` are recognised). ``sensor`` picks the weights:
    ``landsat8``/``landsat9`` (Liang 2001 for OLI), ``tasumi`` (Tasumi et al. 2008, TM and
    ETM+), ``sentinel2`` (Bonafoni and Sekertekin 2020) or ``modis`` (Liang 2001). Pass
    ``weights`` and ``offset`` to use your own.
    """
    if weights is None:
        if sensor not in ALBEDO_WEIGHTS:
            raise ValueError(f"unknown sensor {sensor!r}; choose from {sorted(ALBEDO_WEIGHTS)}")
        weights, default_offset = ALBEDO_WEIGHTS[sensor]
        offset = default_offset if offset is None else offset
    names = list(sr.data_vars) if isinstance(sr, xr.Dataset) else list(sr)
    total = 0.0
    for band, w in weights.items():
        key = next((a for a in _BAND_ALIASES.get(band, (band,)) if a in names), None)
        if key is None:
            raise KeyError(f"band {band!r} not found; available: {names}")
        total = total + w * sr[key]
    return _named(np.clip(total + (offset or 0.0), 0, 1), "albedo", "1", sensor=sensor)


def albedo_from_toa(toa_albedo, tau_sw, *, path_albedo: float = 0.03) -> Any:
    """Surface albedo from top of atmosphere broadband albedo (SEBAL):
    ``(alpha_toa - alpha_path) / tau_sw^2``."""
    return _named((toa_albedo - path_albedo) / tau_sw ** 2, "albedo", "1")


def surface_emissivity(lai, ndvi_=None, *, kind: str = "broadband", water: float = 0.985,
                       snow_albedo=None, albedo_=None) -> Any:
    """Surface emissivity from LAI (Tasumi 2003, METRIC).

    ``kind="broadband"`` (epsilon_0, for longwave emission): ``0.95 + 0.01 LAI`` for LAI up
    to 3, else 0.98. ``kind="narrowband"`` (epsilon_NB, for the thermal band):
    ``0.97 + 0.0033 LAI``, else 0.98. Water (NDVI < 0) gets ``water``; pass ``albedo_`` and
    ``snow_albedo`` to give snow (NDVI < 0 and albedo above the threshold) 0.99.
    """
    if kind == "broadband":
        e = 0.95 + 0.01 * lai
    elif kind == "narrowband":
        e = 0.97 + 0.0033 * lai
    else:
        raise ValueError("kind must be 'broadband' or 'narrowband'")
    e = _where(lai > 3, 0.98, e)
    if ndvi_ is not None:
        wet = water
        if snow_albedo is not None and albedo_ is not None:
            wet = _where(albedo_ > snow_albedo, 0.99, water)
        e = _where(ndvi_ < 0, wet, e)
    return _named(e, "emissivity", "1", kind=kind)


def incoming_shortwave(cos_theta, tau_sw, *, dr: float = 1.0, gsc: float = GSC) -> Any:
    """Incoming shortwave radiation at the surface (W m-2) for clear sky:
    ``Rs_in = Gsc cos(theta) dr tau_sw``."""
    return _named(np.maximum(gsc * cos_theta * dr * tau_sw, 0.0), "rs_in", "W/m2")


def air_emissivity(*, tau_sw=None, ea=None, t_air_k=None) -> Any:
    """Effective atmospheric emissivity.

    From transmissivity (Bastiaanssen 1995, SEBAL and METRIC): ``0.85 (-ln tau)^0.09``; or
    from vapour pressure (kPa) and air temperature (K) after Brutsaert (1975):
    ``1.24 (10 ea / Ta)^(1/7)``.
    """
    if tau_sw is not None:
        return 0.85 * (-np.log(tau_sw)) ** 0.09
    if ea is None or t_air_k is None:
        raise ValueError("pass tau_sw, or ea and t_air_k")
    return 1.24 * (10 * ea / t_air_k) ** (1 / 7)


def incoming_longwave(t_air_k, eps_air) -> Any:
    """Incoming longwave radiation (W m-2): ``eps_a sigma Ta^4``. In SEBAL and METRIC ``Ta`` is
    the surface temperature at the cold anchor pixel."""
    return _named(eps_air * SIGMA * t_air_k ** 4, "rl_in", "W/m2")


def outgoing_longwave(lst_k, eps0) -> Any:
    """Outgoing longwave radiation (W m-2): ``eps_0 sigma Ts^4``."""
    return _named(eps0 * SIGMA * lst_k ** 4, "rl_out", "W/m2")


def net_radiation(rs_in, albedo_, rl_in, rl_out, eps0) -> Any:
    """All wave net radiation (W m-2):
    ``Rn = (1 - albedo) Rs_in + RL_in - RL_out - (1 - eps_0) RL_in``."""
    rn = (1 - albedo_) * rs_in + rl_in - rl_out - (1 - eps0) * rl_in
    return _named(rn, "rn", "W/m2")


def daily_net_radiation(rs24, albedo_, *, ra24=None, tau24=None, a: float = 110.0) -> Any:
    """Daily (24 h mean) net radiation (W m-2) from daily shortwave ``rs24`` (W m-2) with the
    de Bruin (1987) / Bastiaanssen (2000) longwave term: ``(1 - albedo) Rs24 - a tau24``.

    ``tau24 = Rs24 / Ra24``; pass it or the daily extraterrestrial radiation ``ra24`` (W m-2).
    """
    if tau24 is None:
        if ra24 is None:
            raise ValueError("pass tau24 or ra24")
        tau24 = rs24 / ra24
    return _named((1 - albedo_) * rs24 - a * tau24, "rn24", "W/m2")


def datum_temperature(lst_k, dem, *, lapse_rate: float = 0.0065, z_ref: float | None = None
                      ) -> Any:
    """Surface temperature moved to a common elevation (METRIC ``Ts_datum``):
    ``Ts + lapse_rate (z - z_ref)``. ``z_ref`` defaults to the median elevation."""
    if z_ref is None:
        z_ref = float(np.nanmedian(_np(dem)))
    return _named(lst_k + lapse_rate * (dem - z_ref), "lst_datum", "K", z_ref=z_ref)


# ----------------------------------------------------------------------------- soil heat flux
def soil_heat_flux(rn, lst_k, albedo_=None, ndvi_=None, *, lai=None, method: str = "bastiaanssen",
                   water_mask=None, water_ratio: float = 0.5) -> Any:
    """Soil heat flux G (W m-2).

    ``method="bastiaanssen"`` (SEBAL): ``G/Rn = (Ts - 273.15) (0.0038 alpha + 0.0074 alpha^2)
    (1 - 0.98 NDVI^4) / alpha``. ``method="metric"`` (Tasumi 2003): ``G/Rn = 0.05 + 0.18
    exp(-0.521 LAI)`` for LAI of at least 0.5, else ``1.80 (Ts - 273.15) / Rn + 0.084``.
    Water pixels (``water_mask`` or NDVI < 0) use ``G = water_ratio Rn``.
    """
    if method == "bastiaanssen":
        if albedo_ is None or ndvi_ is None:
            raise ValueError("method='bastiaanssen' needs albedo_ and ndvi_")
        ratio = ((lst_k - 273.15) * (0.0038 * albedo_ + 0.0074 * albedo_ ** 2)
                 * (1 - 0.98 * ndvi_ ** 4) / albedo_)
    elif method == "metric":
        if lai is None:
            raise ValueError("method='metric' needs lai")
        ratio = _where(lai >= 0.5, 0.05 + 0.18 * np.exp(-0.521 * lai),
                       1.80 * (lst_k - 273.15) / rn + 0.084)
    else:
        raise ValueError("method must be 'bastiaanssen' or 'metric'")
    if water_mask is None and ndvi_ is not None:
        water_mask = ndvi_ < 0
    if water_mask is not None:
        ratio = _where(water_mask, water_ratio, ratio)
    return _named(ratio * rn, "g", "W/m2", method=method)


# ----------------------------------------------------------------------------- aerodynamics
def momentum_roughness(*, lai=None, ndvi_=None, albedo_=None, height=None, landcover=None,
                       table: Mapping[int, float] | None = None, method: str | None = None,
                       a: float = 5.62, b: float = -5.809, min_zom: float = 0.005) -> Any:
    """Momentum roughness length zom (m).

    * ``method="lai"`` (METRIC, agriculture): ``0.018 LAI``
    * ``method="ndvi"`` (SEBAL / geeSEBAL): ``exp(a NDVI + b)``; with ``albedo_`` the
      Bastiaanssen form ``exp(a NDVI / albedo + b)`` is used and ``a, b`` should be fitted
    * ``method="height"``: ``0.123 h`` from canopy height
    * ``method="table"``: a lookup from land cover classes (e.g. CDL codes) to zom

    The method is inferred from the arguments when not given. Values below ``min_zom`` are
    raised to it.
    """
    if method is None:
        method = ("table" if table is not None else "lai" if lai is not None
                  else "height" if height is not None else "ndvi")
    if method == "lai":
        z = 0.018 * lai
    elif method == "ndvi":
        x = ndvi_ / albedo_ if albedo_ is not None else ndvi_
        z = np.exp(a * x + b)
    elif method == "height":
        z = 0.123 * height
    elif method == "table":
        lc = _np(landcover)
        z = np.full(lc.shape, np.nan)
        for k, v in table.items():
            z[lc == k] = v
        if isinstance(landcover, xr.DataArray):
            z = landcover.copy(data=z)
    else:
        raise ValueError("method must be 'lai', 'ndvi', 'height' or 'table'")
    return _named(np.maximum(z, min_zom), "zom", "m", method=method)


def wind_at_blending_height(u_z, *, z: float = 2.0, station_height: float = 0.12,
                            blend: float = 200.0) -> Any:
    """Wind speed at the blending height (m s-1) from a weather station measurement.

    The station is assumed to be over grass of height ``station_height`` with
    ``zom = 0.12 h``; the log profile is extended to ``blend`` metres (200 in METRIC).
    """
    zom_w = 0.12 * station_height
    u_star_w = VON_KARMAN * u_z / np.log(z / zom_w)
    return u_star_w * np.log(blend / zom_w) / VON_KARMAN


def wind_2m(u_z, z: float) -> Any:
    """Wind speed at 2 m from wind at height ``z`` (FAO-56 eq. 47):
    ``u2 = uz 4.87 / ln(67.8 z - 5.42)``."""
    return u_z * 4.87 / np.log(67.8 * z - 5.42)


def friction_velocity(u, z, zom, psi_m=0.0) -> Any:
    """Friction velocity (m s-1): ``u* = k u / (ln(z / zom) - psi_m)``."""
    return _named(VON_KARMAN * u / (np.log(z / zom) - psi_m), "u_star", "m/s")


def aerodynamic_resistance(u_star, *, z1: float = 0.1, z2: float = 2.0, psi_h_z2=0.0,
                           psi_h_z1=0.0) -> Any:
    """Aerodynamic resistance to heat transport between heights ``z1`` and ``z2`` (s m-1):
    ``rah = (ln(z2 / z1) - psi_h(z2) + psi_h(z1)) / (u* k)``."""
    return _named((np.log(z2 / z1) - psi_h_z2 + psi_h_z1) / (u_star * VON_KARMAN), "rah", "s/m")


def monin_obukhov_length(rho, u_star, t_k, h) -> Any:
    """Monin-Obukhov length (m): ``L = -rho cp u*^3 T / (k g H)``. Infinite when H is zero
    (neutral); negative for unstable (H > 0) and positive for stable (H < 0) conditions."""
    h_safe = _where(np.abs(h) < 1e-6, np.nan, h)
    L = -rho * CP * u_star ** 3 * t_k / (VON_KARMAN * GRAVITY * h_safe)
    return _where(np.isnan(L), np.inf, L)


def stability_corrections(z, L, *, max_unstable: float = 5.0, max_stable: float = 1.0
                          ) -> tuple[Any, Any]:
    """Stability corrections ``(psi_m, psi_h)`` at height ``z`` for Monin-Obukhov length ``L``.

    Unstable (``L < 0``): Paulson (1970), ``x = (1 - 16 z / L)^0.25``,
    ``psi_m = 2 ln((1 + x)/2) + ln((1 + x^2)/2) - 2 atan(x) + pi/2``,
    ``psi_h = 2 ln((1 + x^2)/2)``. Stable (``L > 0``): Webb (1970), ``psi = -5 z / L``.
    ``z / L`` is limited to ``-max_unstable`` and ``max_stable`` to keep the iteration stable.
    """
    zl = _where(np.isinf(L), 0.0, z / L)
    zl = np.clip(zl, -max_unstable, max_stable)
    x = (1 - 16 * np.minimum(zl, 0.0)) ** 0.25
    psi_m_u = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
    psi_h_u = 2 * np.log((1 + x ** 2) / 2)
    psi_s = -5 * zl
    psi_m = _where(zl < 0, psi_m_u, psi_s)
    psi_h = _where(zl < 0, psi_h_u, psi_s)
    return psi_m, psi_h


# ----------------------------------------------------------------------------- reference ET
_REF = {  # Cn, Cd (daily); Cn, Cd day, Cd night, G/Rn day, G/Rn night (hourly)
    "short": {"cn_d": 900, "cd_d": 0.34, "cn_h": 37, "cd_day": 0.24, "cd_night": 0.96,
              "g_day": 0.1, "g_night": 0.5},
    "tall": {"cn_d": 1600, "cd_d": 0.38, "cn_h": 66, "cd_day": 0.25, "cd_night": 1.7,
             "g_day": 0.04, "g_night": 0.2},
}


def asce_pm_daily(tmax_c, tmin_c, ea, rs, u2, z, lat, doy, *, reference: str = "short",
                  albedo_: float = 0.23) -> Any:
    """ASCE standardized daily reference ET (mm d-1), ASCE-EWRI (2005).

    ``reference="short"`` is grass (ETo, the same as FAO-56), ``"tall"`` is alfalfa (ETr).
    ``rs`` is daily shortwave radiation (MJ m-2 d-1), ``u2`` wind at 2 m (m s-1), ``ea``
    actual vapour pressure (kPa), ``z`` elevation (m), ``lat`` degrees, ``doy`` day of year.
    """
    c = _REF[reference]
    t = (tmax_c + tmin_c) / 2
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = psychrometric_constant(p)
    delta = slope_vapour_pressure_curve(t)
    es = (saturation_vapour_pressure(tmax_c) + saturation_vapour_pressure(tmin_c)) / 2
    ra = extraterrestrial_radiation_daily(lat, doy)
    rso = (0.75 + 2e-5 * z) * ra
    fcd = np.clip(1.35 * np.minimum(rs / rso, 1.0) - 0.35, 0.05, 1.0)
    rnl = (4.901e-9 * fcd * (0.34 - 0.14 * np.sqrt(ea))
           * ((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2)
    rn = (1 - albedo_) * rs - rnl
    num = 0.408 * delta * rn + gamma * c["cn_d"] / (t + 273) * u2 * (es - ea)
    return num / (delta + gamma * (1 + c["cd_d"] * u2))


def fao56_eto(tmax_c, tmin_c, ea, rs, u2, z, lat, doy) -> Any:
    """FAO-56 Penman-Monteith grass reference ET (mm d-1); the same as the ASCE short reference."""
    return asce_pm_daily(tmax_c, tmin_c, ea, rs, u2, z, lat, doy, reference="short")


def asce_pm_hourly(t_c, ea, rs, u2, z, lat, lon, when, *, reference: str = "short",
                   cd: float | None = None, g_ratio: float | None = None,
                   fcd_night: float = 0.7, albedo_: float = 0.23) -> Any:
    """ASCE standardized hourly reference ET (mm h-1), ASCE-EWRI (2005).

    ``when`` is the UTC midpoint of the hour, ``rs`` the shortwave radiation for the hour
    (MJ m-2 h-1), ``u2`` wind at 2 m (m s-1), ``ea`` vapour pressure (kPa). Daytime and
    nighttime constants follow ASCE; ``cd`` and ``g_ratio`` override them (FAO-56 uses
    ``cd=0.34``). At night the cloudiness function ``fcd_night`` is used; set it to the value
    from the hours before sunset when you have them. Multiply by about 1 for the instantaneous
    rate in mm h-1 at the satellite overpass.
    """
    c = _REF[reference]
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = psychrometric_constant(p)
    delta = slope_vapour_pressure_curve(t_c)
    es = saturation_vapour_pressure(t_c)
    ra = extraterrestrial_radiation_hourly(lat, lon, when)
    rso = (0.75 + 2e-5 * z) * ra
    day = ra > 1e-6
    ratio = np.where(day, rs / np.where(day, rso, 1.0), 0.0)
    fcd = np.where(day, np.clip(1.35 * np.minimum(ratio, 1.0) - 0.35, 0.05, 1.0), fcd_night)
    rnl = 2.042e-10 * fcd * (0.34 - 0.14 * np.sqrt(ea)) * (t_c + 273.16) ** 4
    rn = (1 - albedo_) * rs - rnl
    g = rn * (g_ratio if g_ratio is not None else np.where(day, c["g_day"], c["g_night"]))
    cd_ = cd if cd is not None else np.where(day, c["cd_day"], c["cd_night"])
    num = 0.408 * delta * (rn - g) + gamma * c["cn_h"] / (t_c + 273) * u2 * (es - ea)
    return num / (delta + gamma * (1 + cd_ * u2))


# ----------------------------------------------------------------------------- anchor pixels
def candidate_anchors(ndvi_, lst_k, *, albedo_=None, landcover=None, mask=None,
                      cold_ndvi_pct: float = 95.0, cold_lst_pct: float = 20.0,
                      hot_ndvi_pct: float = 10.0, hot_lst_pct: float = 80.0,
                      cold_albedo: tuple[float, float] | None = (0.18, 0.25),
                      hot_ndvi_range: tuple[float, float] = (0.10, 0.28),
                      cold_classes: Iterable[int] | None = None,
                      hot_classes: Iterable[int] | None = None) -> dict[str, Any]:
    """Candidate cold and hot anchor pixels (Allen et al. 2013 percentile method).

    Cold: NDVI in the top ``100 - cold_ndvi_pct`` percent of the image, and of those the
    coolest ``cold_lst_pct`` percent; albedo within ``cold_albedo``. Hot: NDVI in the bottom
    ``hot_ndvi_pct`` percent and within ``hot_ndvi_range`` (bare, dry soil, not water), and
    of those the hottest ``100 - hot_lst_pct`` percent. ``landcover`` with ``cold_classes``
    (e.g. CDL alfalfa 36) or ``hot_classes`` (e.g. fallow 61) narrows each pool. When no dense
    pixel passes the albedo screen, or no low-NDVI pixel lies in ``hot_ndvi_range``, that screen is
    skipped with a ``RuntimeWarning``. ``mask``
    (True = usable) removes clouds and edges. Returns boolean masks ``{"cold", "hot"}``
    of the same kind as the inputs.
    """
    n, t = _np(ndvi_), _np(lst_k)
    ok = np.isfinite(n) & np.isfinite(t) & (n > 0)
    if mask is not None:
        ok &= _np(mask).astype(bool)
    a = _np(albedo_) if albedo_ is not None else None
    lc = _np(landcover) if landcover is not None else None
    if a is not None:
        ok &= np.isfinite(a)

    cold = ok & (n >= np.nanpercentile(n[ok], cold_ndvi_pct))
    if a is not None and cold_albedo is not None:
        screened = cold & (a >= cold_albedo[0]) & (a <= cold_albedo[1])
        if screened.any():
            cold = screened
        else:  # e.g. dense spring cereals darker than 0.18: keep the NDVI pool
            import warnings

            warnings.warn(f"no dense pixel has albedo within {cold_albedo}; the cold pool "
                          "ignores the albedo screen", RuntimeWarning, stacklevel=2)
    if lc is not None and cold_classes is not None:
        cold &= np.isin(lc, list(cold_classes))
    if cold.any():
        cold &= t <= np.nanpercentile(t[cold], cold_lst_pct)

    hot = ok & (n <= np.nanpercentile(n[ok], hot_ndvi_pct))
    ranged = hot & (n >= hot_ndvi_range[0]) & (n <= hot_ndvi_range[1])
    if ranged.any():
        hot = ranged
    else:  # e.g. Sentinel-2 bare soil with NDVI just below 0.10: keep the percentile pool
        import warnings

        warnings.warn(f"no low-NDVI pixel lies within {hot_ndvi_range}; the hot pool ignores "
                      "the NDVI range", RuntimeWarning, stacklevel=2)
    if lc is not None and hot_classes is not None:
        hot &= np.isin(lc, list(hot_classes))
    if hot.any():
        hot &= t >= np.nanpercentile(t[hot], hot_lst_pct)

    if isinstance(ndvi_, xr.DataArray):
        cold = ndvi_.copy(data=cold).rename("cold_candidates")
        hot = ndvi_.copy(data=hot).rename("hot_candidates")
    return {"cold": cold, "hot": hot}


def _local_std(a: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    filled = np.where(np.isfinite(a), a, np.nanmean(a))
    m = uniform_filter(filled, size)
    m2 = uniform_filter(filled ** 2, size)
    return np.sqrt(np.maximum(m2 - m ** 2, 0))


def _anchor_record(iy, ix, ref, layers: Mapping[str, Any], n: int) -> dict[str, Any]:
    rec: dict[str, Any] = {"iy": int(iy), "ix": int(ix), "n_candidates": int(n)}
    if isinstance(ref, xr.DataArray) and "x" in ref.coords and "y" in ref.coords:
        rec["x"] = float(ref["x"].values[ix])
        rec["y"] = float(ref["y"].values[iy])
    for k, v in layers.items():
        if v is not None:
            rec[k] = _pixel(v, iy, ix)
    return rec


def select_anchors(ndvi_, lst_k, *, albedo_=None, landcover=None, mask=None,
                   candidates: Mapping[str, Any] | None = None, homogeneity: int | None = 3,
                   **candidate_kwargs) -> dict[str, dict[str, Any]]:
    """Pick one cold and one hot anchor pixel.

    Candidates come from :func:`candidate_anchors` (or ``candidates``). With
    ``homogeneity`` (a window size in pixels) only candidates whose local LST standard
    deviation is at or below the median of the pool are kept, which avoids field edges. The
    pixel whose LST is closest to the pool mean is chosen. Returns
    ``{"cold": {...}, "hot": {...}}`` with the row ``iy``, column ``ix``, map ``x, y``, NDVI,
    LST, albedo and the number of candidates.
    """
    if candidates is None:
        candidates = candidate_anchors(ndvi_, lst_k, albedo_=albedo_, landcover=landcover,
                                       mask=mask, **candidate_kwargs)
    t = _np(lst_k)
    sd = _local_std(t, homogeneity) if homogeneity else None
    out = {}
    for key in ("cold", "hot"):
        pool = _np(candidates[key]).astype(bool)
        if not pool.any():
            raise ValueError(f"no {key} anchor candidates; relax the percentiles or screens")
        n = int(pool.sum())
        if sd is not None and n > 5:
            pool = pool & (sd <= np.nanmedian(sd[pool]))
        idx = np.argwhere(pool)
        vals = t[pool]
        iy, ix = idx[np.nanargmin(np.abs(vals - np.nanmean(vals)))]
        out[key] = _anchor_record(iy, ix, ndvi_, {"ndvi": ndvi_, "lst": lst_k,
                                                  "albedo": albedo_}, n)
    return out


def anchor_at(ref, x: float, y: float, **layers) -> dict[str, Any]:
    """Anchor record for the pixel nearest to map coordinates ``(x, y)`` of DataArray ``ref``.

    Use it to set an anchor by hand; extra keyword layers (``ndvi=``, ``lst=``) are sampled.
    """
    ix = int(np.abs(ref["x"].values - x).argmin())
    iy = int(np.abs(ref["y"].values - y).argmin())
    return _anchor_record(iy, ix, ref, layers, 1)


def anchor_report(anchors: Mapping[str, Mapping[str, Any]], **layers) -> pd.DataFrame:
    """Table of anchor pixels with any extra layers sampled at them (``rn=``, ``g=``, ...)."""
    rows = []
    for key, a in anchors.items():
        iy, ix = _anchor_index(anchors, key)
        row = {"anchor": key, **(dict(a) if isinstance(a, Mapping) else {"iy": iy, "ix": ix})}
        for k, v in layers.items():
            row[k] = _pixel(v, iy, ix)
        rows.append(row)
    return pd.DataFrame(rows).set_index("anchor")


# ----------------------------------------------------------------------------- SEBAL and METRIC
def calibrate_dt(lst_hot: float, lst_cold: float, dt_hot: float, dt_cold: float = 0.0
                 ) -> tuple[float, float]:
    """Coefficients of the near-surface temperature difference ``dT = a + b Ts`` from the
    anchors: ``b = (dT_hot - dT_cold) / (Ts_hot - Ts_cold)``, ``a = dT_hot - b Ts_hot``."""
    b = (dt_hot - dt_cold) / (lst_hot - lst_cold)
    return dt_hot - b * lst_hot, b


def sensible_heat_iterative(lst_k, rn, g, zom, u_blend, pressure, anchors: Mapping, *,
                            le_hot: float | None = None, le_cold: float | None = None,
                            lst_dt=None, blend: float = 200.0, z1: float = 0.1,
                            z2: float = 2.0, max_iter: int = 30, tol: float = 0.005
                            ) -> dict[str, Any]:
    """Sensible heat flux H by the SEBAL / METRIC internal calibration with stability iteration.

    ``u_blend`` is wind at ``blend`` metres (see :func:`wind_at_blending_height`) and
    ``pressure`` is in kPa (scalar or raster). ``anchors`` comes from :func:`select_anchors`
    or is ``{"cold": (iy, ix), "hot": (iy, ix)}``. At the hot pixel
    ``H = Rn - G - le_hot`` (SEBAL: ``le_hot = 0``) and at the cold pixel
    ``H = Rn - G - le_cold`` (SEBAL: ``le_cold = Rn - G``, so H is zero). ``lst_dt`` is the
    temperature used in ``dT = a + b T`` (METRIC uses the datum temperature,
    :func:`datum_temperature`); it defaults to ``lst_k``.

    Each iteration recomputes ``u*``, ``rah``, the anchor ``dT``, ``a`` and ``b``, then ``H``
    for every pixel, the Monin-Obukhov length and the corrections ``psi_m(200)``, ``psi_h(2)``
    and ``psi_h(0.1)`` (for stable air ``psi_m`` at the blending height is evaluated at 2 m,
    as in the METRIC manual). It stops when the hot pixel ``rah`` changes by less than
    ``tol`` (relative).

    Returns a dict with ``h``, ``rah``, ``u_star``, ``dt``, ``L``, ``a``, ``b``,
    ``iterations``, ``converged`` and ``history`` (a DataFrame of the anchor values).
    """
    lst_dt = lst_k if lst_dt is None else lst_dt
    (hy, hx), (cy, cx) = _anchor_index(anchors, "hot"), _anchor_index(anchors, "cold")
    avail_hot = _pixel(rn, hy, hx) - _pixel(g, hy, hx)
    avail_cold = _pixel(rn, cy, cx) - _pixel(g, cy, cx)
    h_hot = avail_hot - (0.0 if le_hot is None else le_hot)
    h_cold = 0.0 if le_cold is None else avail_cold - le_cold
    t_hot, t_cold = _pixel(lst_dt, hy, hx), _pixel(lst_dt, cy, cx)
    ts_hot, ts_cold = _pixel(lst_k, hy, hx), _pixel(lst_k, cy, cx)

    psi_m200 = psi_h2 = psi_h01 = 0.0
    rah_prev = np.nan
    hist = []
    converged = False
    a = b = np.nan
    for it in range(1, max_iter + 1):
        u_star = friction_velocity(u_blend, blend, zom, psi_m200)
        rah = aerodynamic_resistance(u_star, z1=z1, z2=z2, psi_h_z2=psi_h2, psi_h_z1=psi_h01)
        rah_h, rah_c = _pixel(rah, hy, hx), _pixel(rah, cy, cx)
        # air density at the anchors uses the air temperature Ts - dT from the last step
        dt_h0 = h_hot * rah_h / (1.2 * CP) if it == 1 else _pixel(dt, hy, hx)
        dt_c0 = h_cold * rah_c / (1.2 * CP) if it == 1 else _pixel(dt, cy, cx)
        p_h, p_c = _pixel(pressure, hy, hx), _pixel(pressure, cy, cx)
        rho_h = float(air_density(p_h, ts_hot - dt_h0))
        rho_c = float(air_density(p_c, ts_cold - dt_c0))
        dt_hot = h_hot * rah_h / (rho_h * CP)
        dt_cold = h_cold * rah_c / (rho_c * CP)
        a, b = calibrate_dt(t_hot, t_cold, dt_hot, dt_cold)
        dt = a + b * lst_dt
        rho = air_density(pressure, lst_k - dt)
        h = rho * CP * dt / rah
        L = monin_obukhov_length(rho, u_star, lst_k, h)
        psi_m_200u, _ = stability_corrections(blend, L)
        psi_m_2s, psi_h2 = stability_corrections(z2, L)
        _, psi_h01 = stability_corrections(z1, L)
        psi_m200 = _where(L < 0, psi_m_200u, psi_m_2s)
        hist.append({"iteration": it, "rah_hot": rah_h, "rah_cold": rah_c,
                     "u_star_hot": _pixel(u_star, hy, hx), "dt_hot": dt_hot,
                     "dt_cold": dt_cold, "a": a, "b": b, "h_hot": _pixel(h, hy, hx),
                     "L_hot": _pixel(L, hy, hx)})
        if np.isfinite(rah_prev) and abs(rah_h - rah_prev) / rah_prev < tol:
            converged = True
            break
        rah_prev = rah_h
    return {"h": _named(h, "h", "W/m2"), "rah": _named(rah, "rah", "s/m"),
            "u_star": _named(u_star, "u_star", "m/s"), "dt": _named(dt, "dt", "K"),
            "L": _named(L, "L", "m"), "a": float(a), "b": float(b), "iterations": it,
            "converged": converged, "h_hot": h_hot, "h_cold": h_cold,
            "history": pd.DataFrame(hist).set_index("iteration")}


def latent_heat_flux(rn, g, h) -> Any:
    """Latent heat flux as the energy balance residual (W m-2): ``LE = Rn - G - H``."""
    return _named(rn - g - h, "le", "W/m2")


def evaporative_fraction(le, rn, g) -> Any:
    """Evaporative fraction ``EF = LE / (Rn - G)``."""
    return _named(le / (rn - g), "ef", "1")


def instantaneous_et(le, lst_k=None) -> Any:
    """Instantaneous ET (mm h-1) from LE (W m-2): ``3600 LE / lambda``; ``lambda`` from
    ``lst_k`` when given, else 2.45 MJ kg-1."""
    lam = LAMBDA_DEFAULT if lst_k is None else latent_heat_of_vaporization(lst_k)
    return _named(3600 * le / lam, "et_inst", "mm/h")


def _jsonable(obj):
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _to_dataset(fields: Mapping[str, Any], ref, attrs: Mapping[str, Any]) -> xr.Dataset | dict:
    """Pack model outputs into a Dataset on the grid of ``ref`` (a dict for NumPy inputs).

    Scalar attributes are kept as they are; the calibration history (a DataFrame) and the
    anchors are stored as JSON strings so the Dataset can be written to NetCDF. Read them
    back with :func:`calibration_history` and ``json.loads(ds.attrs["anchors"])``.
    """
    if isinstance(ref, xr.DataArray):
        import json

        ds = xr.Dataset({k: (v if isinstance(v, xr.DataArray) else ref.copy(data=np.asarray(v)))
                         .rename(k) for k, v in fields.items()})
        for k, v in attrs.items():
            if isinstance(v, (int, float, str, bool, np.number)):
                ds.attrs[k] = v
            elif isinstance(v, pd.DataFrame):
                ds.attrs[k] = v.reset_index().to_json(orient="records")
            elif isinstance(v, Mapping):
                ds.attrs[k] = json.dumps(_jsonable(v))
        if ref.rio.crs is not None:
            ds = ds.rio.write_crs(ref.rio.crs)
        return ds
    return {**fields, **attrs}


def calibration_history(result: xr.Dataset | Mapping) -> pd.DataFrame:
    """Iteration table of a SEBAL or METRIC run (anchor rah, dT, a, b, H, L per iteration)."""
    h = result.attrs["history"] if isinstance(result, xr.Dataset) else result["history"]
    if isinstance(h, pd.DataFrame):
        return h
    from io import StringIO

    return pd.read_json(StringIO(h), orient="records").set_index("iteration")


def run_sebal(lst_k, rn, g, zom, u_blend, pressure, *, anchors: Mapping | None = None,
              ndvi_=None, albedo_=None, rn24=None, clip_ef: bool = True,
              **iter_kwargs) -> xr.Dataset | dict:
    """SEBAL (Bastiaanssen et al. 1998): H from hot and cold anchors (no ET at the hot pixel,
    no H at the cold pixel), LE as the residual, the evaporative fraction and, with daily net
    radiation ``rn24`` (W m-2, :func:`daily_net_radiation`), daily ET with constant EF:
    ``ET24 = 86400 EF Rn24 / lambda``.

    Anchors are selected automatically from ``ndvi_`` and ``lst_k`` when not given. Returns an
    ``xr.Dataset`` (``h, le, ef, et_inst, rah, dt`` and ``et24``) for DataArray inputs, else a
    dict; the calibration ``a, b`` and iteration count are attributes.
    """
    if anchors is None:
        if ndvi_ is None:
            raise ValueError("pass anchors or ndvi_ to select them")
        anchors = select_anchors(ndvi_, lst_k, albedo_=albedo_)
    sh = sensible_heat_iterative(lst_k, rn, g, zom, u_blend, pressure, anchors, **iter_kwargs)
    le = latent_heat_flux(rn, g, sh["h"])
    ef = evaporative_fraction(le, rn, g)
    if clip_ef:
        ef = np.clip(ef, 0, 1)
    fields = {"h": sh["h"], "le": le, "ef": ef, "et_inst": instantaneous_et(le, lst_k),
              "rah": sh["rah"], "dt": sh["dt"]}
    if rn24 is not None:
        fields["et24"] = 86400 * ef * rn24 / LAMBDA_DEFAULT
    attrs = {"model": "SEBAL", "a": sh["a"], "b": sh["b"], "iterations": sh["iterations"],
             "converged": int(sh["converged"]), "history": sh["history"], "anchors": anchors}
    return _to_dataset(fields, lst_k, attrs)


def run_metric(lst_k, rn, g, zom, u_blend, pressure, etr_inst, *, anchors: Mapping | None = None,
               ndvi_=None, albedo_=None, etrf_cold: float = 1.05, etrf_hot: float = 0.0,
               etr_daily=None, lst_dt=None, clip_etrf: tuple[float, float] | None = (0.0, 1.3),
               **iter_kwargs) -> xr.Dataset | dict:
    """METRIC (Allen et al. 2007): SEBAL style H calibrated so that the cold anchor has
    ``ET = etrf_cold ETr`` and the hot anchor ``ET = etrf_hot ETr``.

    ``etr_inst`` is the alfalfa reference ET at overpass (mm h-1, :func:`asce_pm_hourly`
    with ``reference="tall"``). ``etrf_hot`` comes from a daily soil water balance of the
    hot pixel (FAO-56 evaporation), 0 after a long dry spell. The reference ET fraction
    ``ETrF = ET_inst / ETr_inst`` is assumed constant over the day, so with ``etr_daily``
    (mm d-1) ``ET24 = ETrF ETr24``. ``lst_dt`` is the datum temperature for mountainous
    scenes. Returns ``h, le, et_inst, etrf, rah, dt`` and ``et24``.
    """
    if anchors is None:
        if ndvi_ is None:
            raise ValueError("pass anchors or ndvi_ to select them")
        anchors = select_anchors(ndvi_, lst_k, albedo_=albedo_)
    (hy, hx), (cy, cx) = _anchor_index(anchors, "hot"), _anchor_index(anchors, "cold")
    lam_h = float(latent_heat_of_vaporization(_pixel(lst_k, hy, hx)))
    lam_c = float(latent_heat_of_vaporization(_pixel(lst_k, cy, cx)))
    etr_h, etr_c = _pixel(etr_inst, hy, hx), _pixel(etr_inst, cy, cx)
    le_hot = etrf_hot * etr_h * lam_h / 3600
    le_cold = etrf_cold * etr_c * lam_c / 3600
    sh = sensible_heat_iterative(lst_k, rn, g, zom, u_blend, pressure, anchors,
                                 le_hot=le_hot, le_cold=le_cold, lst_dt=lst_dt, **iter_kwargs)
    le = latent_heat_flux(rn, g, sh["h"])
    et_inst = instantaneous_et(le, lst_k)
    etrf = et_inst / etr_inst
    if clip_etrf is not None:
        etrf = np.clip(etrf, *clip_etrf)
    fields = {"h": sh["h"], "le": le, "et_inst": et_inst, "etrf": _named(etrf, "etrf", "1"),
              "rah": sh["rah"], "dt": sh["dt"]}
    if etr_daily is not None:
        fields["et24"] = etrf * etr_daily
    attrs = {"model": "METRIC", "a": sh["a"], "b": sh["b"], "iterations": sh["iterations"],
             "converged": int(sh["converged"]), "etrf_cold": etrf_cold, "etrf_hot": etrf_hot,
             "history": sh["history"], "anchors": anchors}
    return _to_dataset(fields, lst_k, attrs)


def etrf(et_inst, etr_inst) -> Any:
    """Reference ET fraction ``ETrF = ET_inst / ETr_inst`` (METRIC), the satellite crop
    coefficient relative to alfalfa."""
    return _named(et_inst / etr_inst, "etrf", "1")


# ----------------------------------------------------------------------------- SSEBop
def cold_reference_temperature(lst_k, ndvi_, t_max_k, *, ndvi_min: float = 0.7,
                               c_factor: float | None = None, mask=None) -> tuple[Any, float]:
    """SSEBop cold boundary temperature ``Tc = c Tmax`` (Senay et al. 2013).

    The ``c`` factor is the median ratio ``Ts / Tmax`` over well watered vegetation
    (NDVI of at least ``ndvi_min``) unless given. Returns ``(Tc, c)``.
    """
    if c_factor is None:
        n, t, ta = _np(ndvi_), _np(lst_k), np.broadcast_to(_np(t_max_k), _np(lst_k).shape)
        sel = np.isfinite(n) & np.isfinite(t) & (n >= ndvi_min)
        if mask is not None:
            sel &= _np(mask).astype(bool)
        if not sel.any():
            raise ValueError("no pixels with NDVI >= ndvi_min; lower it or pass c_factor")
        c_factor = float(np.nanmedian(t[sel] / ta[sel]))
    return _named(c_factor * t_max_k, "tc", "K"), c_factor


def dt_from_climate(rn_clear_sky, t_air_k, pressure, *, rah: float = 110.0) -> Any:
    """SSEBop temperature difference between the hot and cold boundaries (K):
    ``dT = Rn rah / (rho cp)``, with clear-sky daily net radiation (W m-2) and a fixed
    aerodynamic resistance of 110 s m-1 for bare dry soil."""
    return _named(rn_clear_sky * rah / (air_density(pressure, t_air_k) * CP), "dt", "K")


def run_ssebop(lst_k, tc, dt, eto, *, k: float = 1.0, max_etf: float = 1.05
               ) -> xr.Dataset | dict:
    """SSEBop (Senay et al. 2013): ``ETf = (Th - Ts) / dT`` with ``Th = Tc + dT``, clipped to
    0 and ``max_etf``, and actual ET ``ETa = ETf k ETo`` (``eto`` in mm d-1; ``k`` scales the
    reference, 1 for the reference used to calibrate, about 1.25 to move grass to alfalfa)."""
    th = tc + dt
    etf = np.clip((th - lst_k) / dt, 0.0, max_etf)
    eta = etf * k * eto
    return _to_dataset({"etf": _named(etf, "etf", "1"), "eta": _named(eta, "eta", "mm/d")},
                       lst_k, {"model": "SSEBop", "k": k})


# ----------------------------------------------------------------------------- simple models
def _edge_fit(x: np.ndarray, y: np.ndarray, n_bins: int, q_hi: float, q_lo: float,
              min_count: int) -> pd.DataFrame:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= min_count:
            rows.append({"x": (lo + hi) / 2, "y_max": np.percentile(y[s], q_hi),
                         "y_min": np.percentile(y[s], q_lo), "n": int(s.sum())})
    if len(rows) < 3:
        raise ValueError("too few populated bins to fit the edges")
    return pd.DataFrame(rows)


def ssebi(albedo_, lst_k, rn=None, g=None, *, n_bins: int = 20, min_count: int = 20
          ) -> xr.Dataset | dict:
    """S-SEBI (Roerink et al. 2000): evaporative fraction from the albedo and LST scatter.

    The dry edge ``T_H = a_H + b_H albedo`` is fitted to the 99th percentile LST of each albedo
    bin above the albedo of the hottest bin; the wet edge ``T_LE = a_LE + b_LE albedo`` to the
    1st percentile of every bin below it. ``EF = (T_H - Ts) / (T_H - T_LE)``, clipped 0 to 1;
    with ``rn`` and ``g`` also ``LE = EF (Rn - G)``. Edge coefficients are in the attributes.
    """
    bins = _edge_fit(_np(albedo_).ravel(), _np(lst_k).ravel(), n_bins, 99, 1, min_count)
    peak = bins["x"].iloc[int(bins["y_max"].values.argmax())]
    dry, wet = bins[bins["x"] >= peak], bins[bins["x"] <= peak]
    if len(dry) < 2 or len(wet) < 2:
        dry = wet = bins
    b_h, a_h = np.polyfit(dry["x"], dry["y_max"], 1)
    b_le, a_le = np.polyfit(wet["x"], wet["y_min"], 1)
    t_h, t_le = a_h + b_h * albedo_, a_le + b_le * albedo_
    ef = np.clip((t_h - lst_k) / (t_h - t_le), 0, 1)
    fields = {"ef": _named(ef, "ef", "1")}
    if rn is not None and g is not None:
        fields["le"] = _named(ef * (rn - g), "le", "W/m2")
    return _to_dataset(fields, lst_k, {"model": "S-SEBI", "a_dry": a_h, "b_dry": b_h,
                                       "a_wet": a_le, "b_wet": b_le, "bins": bins})


def triangle_method(fc, lst_k, rn, g, t_air_k, pressure, *, phi_max: float = 1.26,
                    n_bins: int = 20, min_count: int = 20, t_wet: float | None = None
                    ) -> xr.Dataset | dict:
    """Ts and vegetation triangle (Jiang and Islam 2001) with a Priestley-Taylor parameter.

    The dry edge ``Tmax(fc) = a + b fc`` is fitted to the 99th percentile LST of each
    fractional cover bin; the wet limit ``t_wet`` defaults to the 1st percentile LST of the
    scene. ``phi`` runs from ``phi_max fc`` at the dry edge to ``phi_max`` at the wet limit,
    ``EF = phi Delta / (Delta + gamma)`` and ``LE = EF (Rn - G)``.
    """
    bins = _edge_fit(_np(fc).ravel(), _np(lst_k).ravel(), n_bins, 99, 1, min_count)
    b, a = np.polyfit(bins["x"], bins["y_max"], 1)
    if t_wet is None:
        t_wet = float(np.nanpercentile(_np(lst_k), 1))
    t_dry = a + b * fc
    phi_min = phi_max * fc
    rel = np.clip((t_dry - lst_k) / (t_dry - t_wet), 0, 1)
    phi = phi_min + (phi_max - phi_min) * rel
    ta_c = t_air_k - 273.15
    delta = slope_vapour_pressure_curve(ta_c)
    gamma = psychrometric_constant(pressure)
    ef = np.clip(phi * delta / (delta + gamma), 0, 1)
    return _to_dataset({"phi": _named(phi, "phi", "1"), "ef": _named(ef, "ef", "1"),
                        "le": _named(ef * (rn - g), "le", "W/m2")},
                       lst_k, {"model": "triangle", "a_dry": a, "b_dry": b, "t_wet": t_wet,
                               "bins": bins})


def priestley_taylor_jpl(rn, ndvi_, savi_, t_air_c, ea, pressure, *, g=0.0, topt_c=25.0,
                         fapar_max=None, alpha: float = 1.26, beta: float = 1.0,
                         k_rn: float = 0.6, k_par: float = 0.5) -> xr.Dataset | dict:
    """PT-JPL (Fisher et al. 2008): Priestley-Taylor ET reduced by ecophysiological constraints.

    ``fwet = RH^4``, green fraction ``fg = fAPAR / fIPAR``, temperature constraint
    ``fT = exp(-((Ta - Topt) / Topt)^2)``, plant moisture ``fM = fAPAR / fAPARmax`` and soil
    moisture ``fSM = RH^(VPD / beta)``; ``fAPAR = 1.3632 SAVI - 0.048``,
    ``fIPAR = NDVI - 0.05``, ``LAI = -ln(1 - fIPAR) / k_par`` and soil net radiation
    ``Rns = Rn exp(-k_rn LAI)``. Returns canopy transpiration ``le_c``, soil evaporation
    ``le_s``, interception ``le_i`` and the total ``le`` (W m-2).
    """
    es = saturation_vapour_pressure(t_air_c)
    rh = np.clip(ea / es, 0, 1)
    vpd = np.maximum(es - ea, 0)
    fapar = np.clip(1.3632 * savi_ - 0.048, 0, 1)
    fipar = np.clip(ndvi_ - 0.05, 0, 0.999)
    fg = np.clip(fapar / _where(fipar > 0, fipar, np.nan), 0, 1)
    fg = _where(np.isnan(fg), 0.0, fg)
    if fapar_max is None:
        fapar_max = float(np.nanmax(_np(fapar)))
    fm = np.clip(fapar / fapar_max, 0, 1)
    ft = np.exp(-(((t_air_c - topt_c) / topt_c) ** 2))
    fwet = rh ** 4
    fsm = rh ** (vpd / beta)
    lai = -np.log(1 - fipar) / k_par
    rns = rn * np.exp(-k_rn * lai)
    rnc = rn - rns
    delta = slope_vapour_pressure_curve(t_air_c)
    pt = alpha * delta / (delta + psychrometric_constant(pressure))
    le_c = (1 - fwet) * fg * ft * fm * pt * rnc
    le_s = (fwet + fsm * (1 - fwet)) * pt * (rns - g)
    le_i = fwet * pt * rnc
    fields = {"le_c": _named(le_c, "le_c", "W/m2"), "le_s": _named(le_s, "le_s", "W/m2"),
              "le_i": _named(le_i, "le_i", "W/m2"),
              "le": _named(le_c + le_s + le_i, "le", "W/m2")}
    return _to_dataset(fields, rn if isinstance(rn, xr.DataArray) else ndvi_,
                       {"model": "PT-JPL", "fapar_max": fapar_max})


# ----------------------------------------------------------------------------- two-source model
def tseb_pt(lst_k, lai, t_air_k, u, ea, pressure, rn, sza, *, hc=0.5, z_u: float = 10.0,
            z_t: float = 2.0, fg=1.0, vza: float = 0.0, clumping=1.0, leaf_width: float = 0.05,
            alpha_pt: float = 1.26, g_ratio: float = 0.35, kappa: float = 0.45,
            bare_lai: float = 0.05, zom_soil: float = 0.01, n_alpha: int = 13,
            n_stability: int = 3) -> xr.Dataset | dict:
    """Two-source energy balance with the Priestley-Taylor start (TSEB-PT, Norman et al. 1995),
    parallel resistance network.

    Radiometric temperature ``lst_k`` is split into soil and canopy parts through
    Wind (``z_u``) and air temperature (``z_t``) heights below ``hc + 2`` m are raised to it, so
    reanalysis 2 m air temperature over tall orchards is treated as air above the canopy.

    ``Trad^4 = f_theta Tc^4 + (1 - f_theta) Ts^4`` with ``f_theta = 1 - exp(-0.5 clumping LAI /
    cos(vza))``. Net radiation ``rn`` is split with ``Rn_s = Rn exp(-kappa clumping LAI /
    sqrt(2 cos(sza)))``. The canopy starts with Priestley-Taylor transpiration
    ``LE_c = alpha fg Delta / (Delta + gamma) Rn_c``, its temperature follows from
    ``H_c = rho cp (Tc - Ta) / rah``, the soil temperature from the radiometric split, and
    ``H_s = rho cp (Ts - Ta) / (rah + rs)`` with the Kustas and Norman (1999) soil
    resistance ``rs = 1 / (0.0025 |Ts - Tc|^(1/3) + 0.012 us)``. Where soil evaporation comes
    out negative alpha is lowered in steps of 0.1; if it reaches zero both latent fluxes are
    set to zero. Pixels with LAI below ``bare_lai`` are bare soil and use a one-source
    balance, ``H = rho cp (Trad - Ta) / rah`` with roughness ``zom_soil`` (as pyTSEB does). ``u`` is wind at ``z_u`` (m),
    ``hc`` canopy height (m), angles in degrees. Monin-Obukhov stability is updated
    ``n_stability`` times from the total H.

    Returns ``rn_s, rn_c, g, h_s, h_c, le_s, le_c, le, t_soil, t_canopy`` and ``t_et``
    (transpiration fraction).
    """
    hc = np.maximum(hc, 0.05)
    # forcing heights must be above the canopy: 2 m air temperature over a 4 m orchard is taken
    # as the air 2 m above the tree tops (a blending assumption, as in pyTSEB applications)
    z_u = np.maximum(z_u, hc + 2.0)
    z_t = np.maximum(z_t, hc + 2.0)
    bare = _np(lai) < bare_lai
    d = _where(bare, 0.0, 0.65 * hc)
    zom = _where(bare, zom_soil, 0.125 * hc)
    zoh = zom * np.exp(-2.0)  # kB-1 of 2
    f_theta = np.clip(1 - np.exp(-0.5 * clumping * lai / np.cos(np.radians(vza))), 0, 0.95)
    rn_s = rn * np.exp(-kappa * clumping * lai / np.sqrt(2 * np.cos(np.radians(sza))))
    rn_s = _where(bare, rn, rn_s)  # bare soil takes all the net radiation
    rn_c = rn - rn_s
    g = g_ratio * rn_s
    ta_c = t_air_k - 273.15
    delta = slope_vapour_pressure_curve(ta_c)
    gamma = psychrometric_constant(pressure)
    rho = air_density(pressure, t_air_k, ea)
    # wind below the canopy and soil surface resistance (Kustas and Norman 1999)
    a_w = 0.28 * np.maximum(lai, 0.01) ** (2 / 3) * hc ** (1 / 3) * leaf_width ** (-1 / 3)
    psi_m = psi_h = 0.0
    for _ in range(max(1, n_stability)):
        u_star = VON_KARMAN * u / (np.log((z_u - d) / zom) - psi_m)
        rah = (np.log((z_t - d) / zoh) - psi_h) / (VON_KARMAN * u_star)
        u_c = u * np.log(np.maximum(hc - d, 1.01 * zom) / zom) / np.log((z_u - d) / zom)
        u_s = u_c * np.exp(-a_w * (1 - 0.05 / hc))
        alpha = alpha_pt + 0 * _np(lst_k)
        for _ in range(n_alpha):
            le_c = np.maximum(alpha, 0) * fg * delta / (delta + gamma) * rn_c
            h_c = rn_c - le_c
            t_c = t_air_k + h_c * rah / (rho * CP)
            soil4 = (lst_k ** 4 - f_theta * t_c ** 4) / (1 - f_theta)
            t_s = np.maximum(soil4, (t_air_k - 10) ** 4) ** 0.25
            r_s = 1 / (0.0025 * np.abs(t_s - t_c) ** (1 / 3) + 0.012 * u_s)
            h_s = rho * CP * (t_s - t_air_k) / (rah + r_s)
            # bare soil: one-source balance on the radiometric temperature
            h_s = _where(bare, rho * CP * (lst_k - t_air_k) / rah, h_s)
            t_s = _where(bare, lst_k, t_s)
            le_c = _where(bare, 0.0, le_c)
            h_c = _where(bare, 0.0, h_c)
            le_s = rn_s - g - h_s
            bad = (_np(le_s) < 0) & ~bare
            if not bad.any():
                break
            alpha = np.where(bad, alpha - 0.1, alpha)
        # no evaporation anywhere it stayed negative: soil and canopy keep their energy as H
        neg = _np(le_s) < 0
        if neg.any():
            le_s = _where(neg, 0.0, le_s)
            h_s = _where(neg, rn_s - g, h_s)
            gone = neg & (np.asarray(alpha) <= 0) & ~bare
            le_c = _where(gone, 0.0, le_c)
            h_c = _where(gone, rn_c, h_c)
        L = monin_obukhov_length(rho, u_star, t_air_k, h_c + h_s)
        psi_m, _ = stability_corrections(z_u - d, L)
        _, psi_h = stability_corrections(z_t - d, L)
    le = le_c + le_s
    t_et = _where(le > 0, le_c / _where(le > 0, le, 1.0), np.nan)
    fields = {"rn_s": rn_s, "rn_c": rn_c, "g": g, "h_s": h_s, "h_c": h_c, "le_s": le_s,
              "le_c": le_c, "le": le, "t_soil": t_s, "t_canopy": t_c, "t_et": t_et}
    return _to_dataset(fields, lst_k, {"model": "TSEB-PT"})


def partition_et(le_c, le_s) -> Any:
    """Transpiration fraction ``T / ET = LE_c / (LE_c + LE_s)``."""
    tot = le_c + le_s
    return _named(_where(tot > 0, le_c / _where(tot > 0, tot, 1.0), np.nan), "t_et", "1")


# ----------------------------------------------------------------------------- thermal sharpening
def _block(a: np.ndarray, f: int, func=np.nanmean) -> np.ndarray:
    ny, nx = a.shape[-2] // f, a.shape[-1] // f
    a = a[..., : ny * f, : nx * f]
    shp = a.shape[:-2] + (ny, f, nx, f)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(a.reshape(shp), axis=(-3, -1))


def aggregate_lst(lst_k, factor: int, *, emissive: bool = True) -> Any:
    """Aggregate LST by an integer ``factor`` (for example 30 m Landsat to 990 m).

    With ``emissive=True`` the mean is taken in emitted power, ``(mean(T^4))^(1/4)``, which is
    what a coarse thermal sensor sees.
    """
    if isinstance(lst_k, xr.DataArray):
        x = lst_k ** 4 if emissive else lst_k
        out = x.coarsen(y=factor, x=factor, boundary="trim").mean()
        return (out ** 0.25 if emissive else out).rename("lst").assign_attrs(lst_k.attrs)
    x = np.asarray(lst_k, "float64")
    out = _block(x ** 4 if emissive else x, factor)
    return out ** 0.25 if emissive else out


def thermal_sharpen(lst_coarse, predictors_fine, factor: int | None = None, *,
                    model: str = "tree", homogeneous_fraction: float = 0.8,
                    residual_correction: bool = True, smooth_residuals: bool = True,
                    random_state: int = 0, **model_kwargs) -> Any:
    """Sharpen coarse LST with fine resolution reflectance (data mining sharpener,
    Gao et al. 2012).

    The fine predictors (``(band, y, x)`` array, a DataArray with a ``band`` dimension, or a
    Dataset of bands) are averaged to the coarse grid, a regression tree (``model="tree"``),
    random forest (``"forest"``) or gradient boosting (``"boosting"``) model is fitted to the
    most homogeneous ``homogeneous_fraction`` of coarse pixels, and applied at the fine
    resolution. With ``residual_correction`` the coarse residual (observed minus aggregated
    prediction) is added back. Unsmoothed (``smooth_residuals=False``) the result averages
    exactly back to the observed coarse LST; bilinear smoothing of the residual removes block
    edges at the cost of small departures from that.

    The grids must nest: fine shape equals ``factor`` times the coarse shape (extra fine rows
    and columns are dropped). Reproject the coarse LST onto a grid aligned with the fine one
    first (``rio.reproject``). ``factor`` is inferred from the resolutions of DataArrays.
    """
    if isinstance(predictors_fine, xr.Dataset):
        predictors_fine = predictors_fine.to_dataarray("band")
    ref = predictors_fine.isel(band=0) if isinstance(predictors_fine, xr.DataArray) else None
    if factor is None:
        if ref is None or not isinstance(lst_coarse, xr.DataArray):
            raise ValueError("pass factor for NumPy inputs")
        factor = int(round(abs(lst_coarse.rio.resolution()[0]) / abs(ref.rio.resolution()[0])))
    p = _np(predictors_fine)
    if p.ndim == 2:
        p = p[None]
    t = _np(lst_coarse)
    ny, nx = t.shape
    p = p[:, : ny * factor, : nx * factor]
    ny, nx = p.shape[1] // factor, p.shape[2] // factor
    t = t[:ny, :nx]
    pc = _block(p, factor)
    cv = np.nanmean(_block(p, factor, np.nanstd) / np.abs(pc), axis=0)
    X = pc.reshape(p.shape[0], -1).T
    y = t.ravel()
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(cv.ravel())
    if ok.sum() < 10:
        raise ValueError("fewer than 10 usable coarse pixels")
    ok &= cv.ravel() <= np.nanquantile(cv.ravel()[ok], homogeneous_fraction)

    if model == "tree":
        from sklearn.tree import DecisionTreeRegressor

        reg = DecisionTreeRegressor(random_state=random_state,
                                    **({"min_samples_leaf": 5} | model_kwargs))
    elif model == "forest":
        from sklearn.ensemble import RandomForestRegressor

        reg = RandomForestRegressor(random_state=random_state, n_jobs=-1,
                                    **({"n_estimators": 100, "min_samples_leaf": 3}
                                       | model_kwargs))
    elif model == "boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        reg = HistGradientBoostingRegressor(random_state=random_state, **model_kwargs)
    else:
        raise ValueError("model must be 'tree', 'forest' or 'boosting'")
    reg.fit(X[ok], y[ok])

    Xf = p.reshape(p.shape[0], -1).T
    good = np.isfinite(Xf).all(axis=1)
    pred = np.full(Xf.shape[0], np.nan)
    pred[good] = reg.predict(Xf[good])
    pred = pred.reshape(p.shape[1:])
    if residual_correction:
        resid = t - _block(pred, factor)
        resid = np.where(np.isfinite(resid), resid, 0.0)
        if smooth_residuals:
            from scipy.ndimage import zoom

            up = zoom(resid, factor, order=1, mode="nearest")
        else:
            up = np.kron(resid, np.ones((factor, factor)))
        pred = pred + up[: pred.shape[0], : pred.shape[1]]
    if ref is not None:
        out = ref.isel(y=slice(0, pred.shape[0]), x=slice(0, pred.shape[1])).copy(data=pred)
        return out.rename("lst_sharpened").assign_attrs(units="K", factor=factor, model=model)
    return pred


# ----------------------------------------------------------------------------- upscaling
def instantaneous_to_daily(fraction, *, method: str = "etrf", etr24=None, rn24=None,
                           lam: float = LAMBDA_DEFAULT) -> Any:
    """Daily ET (mm d-1) from an instantaneous fraction held constant over the day.

    ``method="etrf"`` (METRIC): ``ET24 = ETrF ETr24``. ``method="ef"`` (SEBAL):
    ``ET24 = 86400 EF Rn24 / lambda`` with ``rn24`` the 24 h mean net radiation (W m-2).
    """
    if method == "etrf":
        if etr24 is None:
            raise ValueError("method='etrf' needs etr24")
        return _named(fraction * etr24, "et24", "mm/d")
    if method == "ef":
        if rn24 is None:
            raise ValueError("method='ef' needs rn24")
        return _named(86400 * fraction * rn24 / lam, "et24", "mm/d")
    raise ValueError("method must be 'etrf' or 'ef'")


def _interp_gaps(a: np.ndarray, t: np.ndarray, max_gap: float | None) -> np.ndarray:
    """Linear interpolation of NaNs along the first axis at times ``t``, vectorized over the
    other axes; gaps between valid values longer than ``max_gap`` stay NaN."""
    a = np.array(a, dtype="float64")
    n = a.shape[0]
    flat = a.reshape(n, -1)
    valid = np.isfinite(flat)
    idx = np.arange(n)[:, None]
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(valid, idx, n)[::-1], axis=0)[::-1]
    inside = (prev >= 0) & (nxt < n)
    p, q = np.clip(prev, 0, n - 1), np.clip(nxt, 0, n - 1)
    vp = np.take_along_axis(flat, p, 0)
    vq = np.take_along_axis(flat, q, 0)
    tp, tq, tt = t[p], t[q], np.broadcast_to(t[:, None], flat.shape)
    w = np.where(tq > tp, (tt - tp) / np.where(tq > tp, tq - tp, 1), 0.0)
    filled = vp + (vq - vp) * w
    ok = inside & ((max_gap is None) | ((tq - tp) <= (max_gap if max_gap is not None else 0)))
    out = np.where(valid, flat, np.where(ok, filled, np.nan))
    return out.reshape(a.shape)


def _hold_ends(a: np.ndarray, axis: int, limit: int | None) -> np.ndarray:
    """Hold the first and last valid values of ``a`` along ``axis`` over leading and trailing
    NaNs, for at most ``limit`` steps (no bottleneck needed)."""
    a = np.moveaxis(np.array(a, dtype="float64"), axis, 0)
    n = a.shape[0]
    valid = np.isfinite(a)
    has = valid.any(axis=0)
    first = np.where(has, valid.argmax(axis=0), 0)
    last = np.where(has, n - 1 - valid[::-1].argmax(axis=0), n - 1)
    first_v = np.take_along_axis(a, first[None], 0)[0]
    last_v = np.take_along_axis(a, last[None], 0)[0]
    t = np.arange(n).reshape((n,) + (1,) * (a.ndim - 1))
    lim = n if limit is None else limit
    lead = has & (t < first) & (first - t <= lim)
    trail = has & (t > last) & (t - last <= lim)
    a = np.where(lead, first_v, np.where(trail, last_v, a))
    return np.moveaxis(a, 0, axis)


def interpolate_etrf(etrf_stack: xr.DataArray, *, start=None, end=None,
                     max_gap_days: int | None = 32, extrapolate: bool = True,
                     clip: tuple[float, float] | None = (0.0, 1.3)) -> xr.DataArray:
    """Daily ETrF (or ETf) between clear satellite dates by linear interpolation in time.

    ``etrf_stack`` has a ``time`` dimension (cloudy pixels NaN). Gaps longer than
    ``max_gap_days`` stay NaN. With ``extrapolate`` the first and last values are held to
    ``start`` and ``end``. Interpolate the fraction, not ET, because the fraction changes
    slowly with crop growth while ET follows the weather (Allen et al. 2007).
    """
    t = pd.DatetimeIndex(etrf_stack["time"].values)
    start = t.min().normalize() if start is None else pd.Timestamp(start)
    end = t.max().normalize() if end is None else pd.Timestamp(end)
    days = pd.date_range(start, end, freq="D")
    s = etrf_stack.assign_coords(time=t.normalize()).groupby("time").mean()
    full = s.reindex(time=days.union(pd.DatetimeIndex(s["time"].values))).transpose("time", ...)
    tday = ((pd.DatetimeIndex(full["time"].values) - days[0]) / pd.Timedelta(days=1)).values
    vals = _interp_gaps(full.values, tday, max_gap_days)
    if extrapolate:
        vals = _hold_ends(vals, 0, max_gap_days)
    out = full.copy(data=vals).sel(time=days)
    if clip is not None:
        out = out.clip(*clip)
    return out.rename(etrf_stack.name or "etrf")


def seasonal_et(fraction_daily: xr.DataArray, reference_daily, *, start=None, end=None,
                freq: str | None = None) -> xr.DataArray:
    """Seasonal (or monthly with ``freq="MS"``) ET totals (mm): the sum over days of the daily
    fraction times daily reference ET.

    ``reference_daily`` is a pandas Series with a DatetimeIndex, a DataArray over ``time``,
    or a ``(time, y, x)`` DataArray on the same grid.
    """
    ref = reference_daily
    if isinstance(ref, pd.Series):
        ref = xr.DataArray(ref.values, coords={"time": pd.DatetimeIndex(ref.index).normalize()},
                           dims="time")
    et = fraction_daily * ref
    if start is not None or end is not None:
        et = et.sel(time=slice(start, end))
    if freq is None:
        return et.sum("time", min_count=1).rename("et_total").assign_attrs(units="mm")
    return et.resample(time=freq).sum(min_count=1).rename("et").assign_attrs(units="mm")


def zonal_et(et, zones, *, names: Mapping[int, str] | None = None, pixel_area: float | None = None,
             min_count: int = 1) -> pd.DataFrame:
    """ET statistics per zone (mm) and water volume (m3).

    ``zones`` is an integer raster on the same grid (for example CDL crop codes or field
    ids) or a GeoDataFrame of polygons (passed to :func:`geolattice.rsense.zonal_stats`).
    ``names`` maps zone codes to labels. The volume uses ``pixel_area`` (m2), read from the
    raster resolution when not given.
    """
    if hasattr(zones, "geometry"):
        from .core import zonal_stats

        return zonal_stats(et, zones, stats=("mean", "median", "std", "count"))
    v, z = _np(et).ravel(), _np(zones).ravel()
    ok = np.isfinite(v) & np.isfinite(z)
    df = pd.DataFrame({"zone": z[ok].astype(int), "et": v[ok]})
    out = df.groupby("zone")["et"].agg(count="count", mean_mm="mean", median_mm="median",
                                       std_mm="std", p10_mm=lambda s: s.quantile(0.1),
                                       p90_mm=lambda s: s.quantile(0.9))
    out = out[out["count"] >= min_count]
    if pixel_area is None and isinstance(et, xr.DataArray):
        try:
            rx, ry = et.rio.resolution()
            pixel_area = abs(rx * ry)
        except Exception:
            pixel_area = None
    if pixel_area is not None:
        out["area_ha"] = out["count"] * pixel_area / 1e4
        out["volume_m3"] = out["mean_mm"] / 1000 * out["count"] * pixel_area
    if names is not None:
        out.insert(0, "name", [names.get(int(k), str(k)) for k in out.index])
    return out.sort_values("count", ascending=False)


# ----------------------------------------------------------------------------- validation
def energy_closure(rn, g, h, le) -> dict[str, float]:
    """Energy balance closure of fluxes (tower or model): ratio ``sum(H + LE) / sum(Rn - G)``,
    mean residual ``Rn - G - H - LE`` (W m-2) and the slope of ``H + LE`` against ``Rn - G``."""
    a = (_np(rn) - _np(g)).ravel()
    t = (_np(h) + _np(le)).ravel()
    ok = np.isfinite(a) & np.isfinite(t)
    a, t = a[ok], t[ok]
    slope = float(np.sum(a * t) / np.sum(a * a)) if a.size else np.nan
    return {"n": int(a.size), "closure_ratio": float(t.sum() / a.sum()),
            "mean_residual": float(np.mean(a - t)), "slope_through_origin": slope}


def compare_to_tower(predicted, observed) -> pd.Series:
    """Accuracy of predicted against observed values (ET in mm, or fluxes): count, mean of
    each, bias, mean absolute error, RMSE, relative RMSE, R2 (squared Pearson correlation),
    Nash-Sutcliffe efficiency and the ordinary least squares slope and intercept."""
    p, o = _np(predicted).ravel(), _np(observed).ravel()
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if p.size < 2:
        raise ValueError("need at least two paired values")
    err = p - o
    slope, intercept = np.polyfit(o, p, 1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return pd.Series({"n": int(p.size), "mean_obs": o.mean(), "mean_pred": p.mean(),
                      "bias": err.mean(), "mae": np.abs(err).mean(), "rmse": rmse,
                      "rel_rmse": rmse / o.mean(), "r2": np.corrcoef(o, p)[0, 1] ** 2,
                      "nse": 1 - np.sum(err ** 2) / np.sum((o - o.mean()) ** 2),
                      "slope": slope, "intercept": intercept})


def sensitivity(func: Callable[..., Any], inputs: Mapping[str, Any],
                perturbations: Mapping[str, Sequence[float]], *, mode: str = "relative",
                summary: Callable[[Any], float] | None = None) -> pd.DataFrame:
    """One at a time sensitivity of a model output.

    ``func(**inputs)`` returns an array or scalar, reduced by ``summary`` (default the nan mean).
    Each input in ``perturbations`` is changed by each value in turn, as a fraction
    (``mode="relative"``, 0.1 is plus 10 percent) or added (``"absolute"``, e.g. 2 K). Returns
    the summary and its percent change from the base run for every case.
    """
    summary = summary or (lambda out: float(np.nanmean(_np(out))))
    base = summary(func(**inputs))
    rows = [{"input": "base", "change": 0.0, "output": base, "change_pct": 0.0}]
    for name, deltas in perturbations.items():
        for dv in deltas:
            kw = dict(inputs)
            kw[name] = inputs[name] * (1 + dv) if mode == "relative" else inputs[name] + dv
            out = summary(func(**kw))
            rows.append({"input": name, "change": dv, "output": out,
                         "change_pct": 100 * (out - base) / base})
    return pd.DataFrame(rows)


def monte_carlo_anchors(run: Callable[[dict], Any], candidates: Mapping[str, Any], *,
                        n: int = 50, seed: int = 0) -> tuple[Any, pd.DataFrame]:
    """Spread of a model result over many anchor pairs drawn from the candidate pools.

    ``run(anchors)`` runs the model for ``{"cold": (iy, ix), "hot": (iy, ix)}`` and returns a
    DataArray or array (for example daily ET). ``candidates`` are the masks from
    :func:`candidate_anchors`. Returns the results stacked along a new ``draw`` dimension and
    a table of the anchors used; take ``.std("draw")`` for the anchor uncertainty map.
    """
    rng = np.random.default_rng(seed)
    pools = {k: np.argwhere(_np(candidates[k]).astype(bool)) for k in ("cold", "hot")}
    for k, v in pools.items():
        if not len(v):
            raise ValueError(f"empty {k} candidate pool")
    outs, rows = [], []
    for i in range(n):
        a = {k: tuple(int(x) for x in pools[k][rng.integers(len(pools[k]))])
             for k in ("cold", "hot")}
        try:
            outs.append(run(a))
        except Exception as err:  # a bad pair (e.g. hot cooler than cold) is skipped
            rows.append({"draw": i, "cold": a["cold"], "hot": a["hot"], "ok": False,
                         "error": str(err)})
            continue
        rows.append({"draw": i, "cold": a["cold"], "hot": a["hot"], "ok": True, "error": ""})
    if not outs:
        raise RuntimeError("every draw failed")
    if isinstance(outs[0], xr.DataArray):
        stack = xr.concat(outs, dim="draw")
    else:
        stack = np.stack([np.asarray(o) for o in outs])
    return stack, pd.DataFrame(rows).set_index("draw")


__all__ = [
    "SIGMA", "VON_KARMAN", "CP", "GRAVITY", "R_DRY", "GSC", "LAMBDA_DEFAULT", "ALBEDO_WEIGHTS",
    "latent_heat_of_vaporization",
    # solar geometry
    "day_of_year", "solar_declination", "inverse_relative_distance", "equation_of_time",
    "hour_angle", "solar_position", "scene_geometry", "slope_aspect", "cos_incidence",
    "sunset_hour_angle", "extraterrestrial_radiation_daily", "extraterrestrial_radiation_hourly",
    # atmosphere and weather
    "pressure_from_elevation", "psychrometric_constant", "saturation_vapour_pressure",
    "slope_vapour_pressure_curve", "actual_vapour_pressure", "air_density",
    "precipitable_water", "broadband_transmissivity", "weather_open_meteo",
    "interpolate_weather",
    # vegetation and radiation
    "ndvi", "savi", "leaf_area_index", "fractional_cover", "albedo", "albedo_from_toa",
    "surface_emissivity", "incoming_shortwave", "air_emissivity", "incoming_longwave",
    "outgoing_longwave", "net_radiation", "daily_net_radiation", "datum_temperature",
    # soil heat and aerodynamics
    "soil_heat_flux", "momentum_roughness", "wind_at_blending_height", "wind_2m",
    "friction_velocity", "aerodynamic_resistance", "monin_obukhov_length",
    "stability_corrections",
    # reference ET
    "asce_pm_daily", "asce_pm_hourly", "fao56_eto",
    # anchors
    "candidate_anchors", "select_anchors", "anchor_at", "anchor_report",
    # models
    "calibrate_dt", "sensible_heat_iterative", "latent_heat_flux", "evaporative_fraction",
    "instantaneous_et", "calibration_history", "run_sebal", "run_metric", "etrf",
    "cold_reference_temperature",
    "dt_from_climate", "run_ssebop", "ssebi", "triangle_method", "priestley_taylor_jpl",
    "tseb_pt", "partition_et",
    # sharpening, upscaling, validation
    "aggregate_lst", "thermal_sharpen", "instantaneous_to_daily", "interpolate_etrf",
    "seasonal_et", "zonal_et", "energy_closure", "compare_to_tower", "sensitivity",
    "monte_carlo_anchors",
]
