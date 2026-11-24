"""Sensor-aware spectral indices for optical imagery.

Formulas live in :mod:`geolattice.rsense.optical.indices`. This module decides which
of those indices a given satellite can compute from its band set, and provides
a thin compute API with sensor validation.

Supported sensors
-----------------
* **MODIS** — Terra/Aqua surface reflectance (MOD09 / MYD09)
* **Landsat 8/9** — OLI/TIRS Collection 2 Level-2, including the USGS
  Surface Reflectance-derived indices (NDVI, EVI, SAVI, MSAVI, NDMI, NBR,
  NBR2, NDSI)
* **Sentinel-2** — MSI L2A (red-edge and SWIR indices included)
* **PlanetScope** — 4-band PS2 / PS2.SD and 8-band SuperDove (PSB.SD)
* **WorldView-2 / WorldView-3** — Maxar VNIR (WV-3 also maps SWIR)
* **NAIP** — USDA 4-band aerial (RGB + NIR)

Band mappings reuse :data:`geolattice.rsense.optical.bands.SENSOR_BANDS`. Index
catalog and references follow
`Awesome Spectral Indices <https://awesome-spectral-indices.github.io/awesome-spectral-indices/>`_
and sensor pages such as
`Geopera Landsat 8/9 <https://docs.geopera.com/spectral-indices/sensors/landsat-8-9>`_,
`Sentinel-2 <https://docs.geopera.com/spectral-indices/sensors/sentinel-2>`_,
`WorldView-2 <https://docs.geopera.com/spectral-indices/sensors/worldview-2>`_,
and `NAIP <https://docs.geopera.com/spectral-indices/sensors/naip>`_.

USGS Landsat SR product page:
https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices

Example
-------
::

    import geolattice.rsense as rs

    s2 = rs.optical.load_sentinel2("aoi.geojson", ("2026-06-01", "2026-06-30"))
    s2 = s2.rs.scale().rs.mask_clouds()

    rs.spectral_index.available("sentinel2")          # names this sensor can compute
    ndvi = rs.spectral_index.compute(s2, "NDVI")      # sensor guessed from bands
    usgs = rs.spectral_index.compute(s2, rs.spectral_index.USGS_LANDSAT_SR,
                                     sensor="landsat89")
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import xarray as xr

from .optical.bands import SENSOR_BANDS, SENSORS, guess_sensor
from .optical.indices import ALIASES, INDICES, SpectralIndex, index as _index

# ---------------------------------------------------------------------------
# USGS Landsat Collection 2 Surface Reflectance-derived spectral indices
# https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices
# ---------------------------------------------------------------------------
USGS_LANDSAT_SR: tuple[str, ...] = (
    "NDVI",   # Normalized Difference Vegetation Index
    "EVI",    # Enhanced Vegetation Index
    "SAVI",   # Soil Adjusted Vegetation Index
    "MSAVI",  # Modified Soil Adjusted Vegetation Index
    "NDMI",   # Normalized Difference Moisture Index
    "NBR",    # Normalized Burn Ratio
    "NBR2",   # Normalized Burn Ratio 2
    "NDSI",   # Normalized Difference Snow Index
)

# Friendly aliases accepted by :func:`normalize_sensor`
_SENSOR_ALIASES: dict[str, str] = {
    "l8": "landsat89",
    "l9": "landsat89",
    "landsat8": "landsat89",
    "landsat9": "landsat89",
    "landsat-8": "landsat89",
    "landsat-9": "landsat89",
    "oli": "landsat89",
    "tm": "landsat_tm",
    "etm": "landsat_tm",
    "etm+": "landsat_tm",
    "landsat7": "landsat_tm",
    "s2": "sentinel2",
    "sentinel-2": "sentinel2",
    "msi": "sentinel2",
    "planet": "planetscope",
    "ps2": "planetscope",
    "psb": "planetscope8",
    "superdove": "planetscope8",
    "planetscope_sd": "planetscope8",
    "wv2": "worldview2",
    "worldview-2": "worldview2",
    "wv3": "worldview3",
    "worldview-3": "worldview3",
    "mod09": "modis",
    "myd09": "modis",
}


@dataclass(frozen=True)
class SensorProfile:
    """Static description of a sensor for spectral-index work."""

    key: str
    name: str
    provider: str
    platform: str
    bands: tuple[str, ...]
    wavelengths_nm: Mapping[str, tuple[float, float] | float] = field(default_factory=dict)
    notes: str = ""
    references: tuple[str, ...] = ()


# Nominal center / range wavelengths used for documentation and band checks.
# Values follow USGS / ESA / Maxar / USDA product docs and Geopera sensor pages.
SENSOR_PROFILES: dict[str, SensorProfile] = {
    "modis": SensorProfile(
        key="modis",
        name="MODIS Terra/Aqua Surface Reflectance",
        provider="NASA",
        platform="Terra / Aqua",
        bands=("red", "nir", "blue", "green", "swir12", "swir16", "swir22"),
        wavelengths_nm={
            "red": (620, 670), "nir": (841, 876), "blue": (459, 479),
            "green": (545, 565), "swir12": (1230, 1250), "swir16": (1628, 1652),
            "swir22": (2105, 2155),
        },
        notes="MOD09GA/MYD09GA and 8-day composites; thermal from LST products when present.",
        references=(
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
            "https://github.com/awesome-spectral-indices/awesome-spectral-indices",
        ),
    ),
    "landsat89": SensorProfile(
        key="landsat89",
        name="Landsat 8/9 OLI/TIRS Collection 2 Level-2",
        provider="USGS/NASA",
        platform="Landsat 8 / Landsat 9",
        bands=("coastal", "blue", "green", "red", "nir", "swir16", "swir22", "thermal"),
        wavelengths_nm={
            "coastal": (435, 451), "blue": (452, 512), "green": (533, 590),
            "red": (636, 673), "nir": (851, 879), "swir16": (1566, 1651),
            "swir22": (2107, 2294), "thermal": (10600, 11190),
        },
        notes=(
            "USGS ESPA can deliver Surface Reflectance-derived indices on demand: "
            + ", ".join(USGS_LANDSAT_SR) + "."
        ),
        references=(
            "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices",
            "https://docs.geopera.com/spectral-indices/sensors/landsat-8-9",
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "landsat": SensorProfile(
        key="landsat",
        name="Landsat 8/9 OLI/TIRS (alias of landsat89)",
        provider="USGS/NASA",
        platform="Landsat 8 / Landsat 9",
        bands=("coastal", "blue", "green", "red", "nir", "swir16", "swir22", "thermal"),
        wavelengths_nm={
            "coastal": (435, 451), "blue": (452, 512), "green": (533, 590),
            "red": (636, 673), "nir": (851, 879), "swir16": (1566, 1651),
            "swir22": (2107, 2294), "thermal": (10600, 11190),
        },
        notes="Same band map as landsat89; preferred key for Collection 2 loaders.",
        references=(
            "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices",
            "https://docs.geopera.com/spectral-indices/sensors/landsat-8-9",
        ),
    ),
    "landsat_tm": SensorProfile(
        key="landsat_tm",
        name="Landsat 4–7 TM/ETM+ Collection 2 Level-2",
        provider="USGS/NASA",
        platform="Landsat 4 / 5 / 7",
        bands=("blue", "green", "red", "nir", "swir16", "swir22", "thermal"),
        wavelengths_nm={
            "blue": (450, 520), "green": (520, 600), "red": (630, 690),
            "nir": (770, 900), "swir16": (1550, 1750), "swir22": (2080, 2350),
            "thermal": (10400, 12500),
        },
        notes="LEDAPS surface reflectance; same USGS SR-derived index set as Landsat 8/9.",
        references=(
            "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices",
        ),
    ),
    "sentinel2": SensorProfile(
        key="sentinel2",
        name="Sentinel-2 MSI L2A",
        provider="ESA",
        platform="Sentinel-2A / 2B / 2C",
        bands=("coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
               "nir", "nir08", "swir16", "swir22"),
        wavelengths_nm={
            "coastal": 443, "blue": 490, "green": 560, "red": 665,
            "rededge1": 705, "rededge2": 740, "rededge3": 783, "nir": 842,
            "nir08": 865, "swir16": 1610, "swir22": 2190,
        },
        notes="Red-edge indices (NDRE, MTCI, IRECI, S2REP, …) require B05–B07 / B8A.",
        references=(
            "https://docs.geopera.com/spectral-indices/sensors/sentinel-2",
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "planetscope": SensorProfile(
        key="planetscope",
        name="PlanetScope 4-band (PS2 / PS2.SD)",
        provider="Planet",
        platform="Dove Classic / Dove-R",
        bands=("blue", "green", "red", "nir"),
        wavelengths_nm={
            "blue": (455, 515), "green": (500, 590), "red": (590, 670),
            "nir": (780, 860),
        },
        notes="No SWIR or red-edge; vegetation/water RGB–NIR indices only.",
        references=(
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
            "https://github.com/awesome-spectral-indices/awesome-spectral-indices",
        ),
    ),
    "planetscope8": SensorProfile(
        key="planetscope8",
        name="PlanetScope SuperDove 8-band (PSB.SD)",
        provider="Planet",
        platform="SuperDove",
        bands=("coastal", "blue", "green", "yellow", "red", "rededge1", "nir"),
        wavelengths_nm={
            "coastal": (431, 452), "blue": (465, 515), "green": (513, 549),
            "yellow": (547, 583), "red": (650, 680), "rededge1": (697, 713),
            "nir": (845, 885),
        },
        notes="Green I (≈531 nm) is not aliased to green; use bands= to map it if needed.",
        references=(
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "worldview2": SensorProfile(
        key="worldview2",
        name="WorldView-2",
        provider="Maxar",
        platform="WorldView-2",
        bands=("coastal", "blue", "green", "yellow", "red", "rededge1", "nir", "nir08"),
        wavelengths_nm={
            "coastal": (400, 450), "blue": (450, 510), "green": (510, 580),
            "yellow": (585, 625), "red": (630, 690), "rededge1": (705, 745),
            "nir": (770, 895), "nir08": (860, 1040),
        },
        notes="NIR1 → nir, NIR2 → nir08. No onboard SWIR.",
        references=(
            "https://docs.geopera.com/spectral-indices/sensors/worldview-2",
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "worldview3": SensorProfile(
        key="worldview3",
        name="WorldView-3",
        provider="Maxar",
        platform="WorldView-3",
        bands=("coastal", "blue", "green", "yellow", "red", "rededge1", "nir", "nir08",
               "swir12", "swir16", "swir22"),
        wavelengths_nm={
            "coastal": (400, 450), "blue": (450, 510), "green": (510, 580),
            "yellow": (585, 625), "red": (630, 690), "rededge1": (705, 745),
            "nir": (770, 895), "nir08": (860, 1040),
            "swir12": (1195, 1225), "swir16": (1550, 1590), "swir22": (2145, 2365),
        },
        notes="VNIR matches WorldView-2; SWIR-1/2/5 stand in for swir12/16/22.",
        references=(
            "https://docs.geopera.com/spectral-indices/sensors/worldview-2",
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "naip": SensorProfile(
        key="naip",
        name="NAIP 4-band Aerial",
        provider="USDA",
        platform="NAIP",
        bands=("red", "green", "blue", "nir"),
        wavelengths_nm={
            "red": (604, 664), "green": (533, 587), "blue": (420, 492),
            "nir": (833, 920),
        },
        notes="COG band order is typically Red, Green, Blue, NIR (not Blue-first).",
        references=(
            "https://docs.geopera.com/spectral-indices/sensors/naip",
            "https://awesome-spectral-indices.github.io/awesome-spectral-indices/",
        ),
    ),
    "hls_l30": SensorProfile(
        key="hls_l30",
        name="Harmonized Landsat Sentinel-2 (HLSL30)",
        provider="NASA",
        platform="HLS L30",
        bands=("coastal", "blue", "green", "red", "nir", "swir16", "swir22", "thermal"),
        notes="Landsat-derived HLS; same index set as Landsat 8/9.",
        references=("https://awesome-spectral-indices.github.io/awesome-spectral-indices/",),
    ),
    "hls_s30": SensorProfile(
        key="hls_s30",
        name="Harmonized Landsat Sentinel-2 (HLSS30)",
        provider="NASA",
        platform="HLS S30",
        bands=("coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
               "nir", "nir08", "swir16", "swir22"),
        notes="Sentinel-2-derived HLS; red-edge indices available.",
        references=("https://awesome-spectral-indices.github.io/awesome-spectral-indices/",),
    ),
}


def normalize_sensor(sensor: str) -> str:
    """Return the canonical sensor key (e.g. ``'s2'`` → ``'sentinel2'``)."""
    key = sensor.strip().lower().replace(" ", "")
    key = _SENSOR_ALIASES.get(key, key)
    if key not in SENSOR_BANDS:
        raise ValueError(
            f"Unknown sensor {sensor!r}. Choose from {SENSORS} "
            f"or aliases {sorted(_SENSOR_ALIASES)}."
        )
    return key


def sensor_bands(sensor: str) -> frozenset[str]:
    """Canonical optical bands available for ``sensor`` (excludes QA)."""
    key = normalize_sensor(sensor)
    return frozenset(b for b in SENSOR_BANDS[key] if b != "qa")


def _band_available(band: str, available: frozenset[str]) -> bool:
    if band in available:
        return True
    # optical.get_band falls back to nir08 when looking up nir
    return band == "nir" and "nir08" in available


def index_supported(name: str, sensor: str) -> bool:
    """Return True if all bands required by ``name`` exist on ``sensor``."""
    key = ALIASES.get(name.upper(), name.upper())
    if key not in INDICES:
        raise KeyError(f"Unknown index {name!r}. See geolattice.rsense.optical.list_indices().")
    bands = sensor_bands(sensor)
    return all(_band_available(b, bands) for b in INDICES[key].bands)


def available(sensor: str, *, domain: str | None = None) -> list[str]:
    """Names of registered indices computable for ``sensor``.

    Parameters
    ----------
    sensor : str
        Sensor key or alias (``landsat89``, ``sentinel2``, ``modis``, …).
    domain : str, optional
        Filter by index domain (``vegetation``, ``water``, ``burn``, …).
    """
    bands = sensor_bands(sensor)
    out = []
    for spec in INDICES.values():
        if domain is not None and spec.domain != domain:
            continue
        if all(_band_available(b, bands) for b in spec.bands):
            out.append(spec.name)
    return out


def missing_bands(name: str, sensor: str) -> list[str]:
    """Canonical bands required by ``name`` that ``sensor`` does not provide."""
    key = ALIASES.get(name.upper(), name.upper())
    if key not in INDICES:
        raise KeyError(f"Unknown index {name!r}.")
    bands = sensor_bands(sensor)
    return [b for b in INDICES[key].bands if not _band_available(b, bands)]


def list_for_sensor(sensor: str, *, domain: str | None = None):
    """DataFrame of indices available for ``sensor`` (same columns as ``list_indices``)."""
    import pandas as pd

    names = set(available(sensor, domain=domain))
    rows = [{"name": i.name, "long_name": i.long_name, "bands": ", ".join(i.bands),
             "domain": i.domain, "reference": i.reference}
            for i in INDICES.values() if i.name in names]
    return pd.DataFrame(rows)


def describe(sensor: str) -> SensorProfile:
    """Return the :class:`SensorProfile` for ``sensor``."""
    key = normalize_sensor(sensor)
    if key in SENSOR_PROFILES:
        return SENSOR_PROFILES[key]
    # Fallback for keys present in SENSOR_BANDS but without a rich profile
    bands = tuple(sorted(sensor_bands(key)))
    return SensorProfile(key=key, name=key, provider="", platform=key, bands=bands)


def sensors() -> tuple[str, ...]:
    """Canonical sensor keys that have band maps (same as ``optical.bands.SENSORS``)."""
    return SENSORS


def resolve_sensor(obj: xr.Dataset | xr.DataArray, sensor: str | None = None) -> str:
    """Normalize ``sensor`` or guess it from ``obj``; raise if neither works."""
    if sensor is not None:
        return normalize_sensor(sensor)
    guessed = guess_sensor(obj)
    if guessed is None:
        raise ValueError(
            "Could not guess sensor from band names; pass sensor= explicitly "
            f"(one of {SENSORS})."
        )
    return guessed


def compute(
    obj: xr.Dataset | xr.DataArray,
    name: str | Sequence[str],
    *,
    sensor: str | None = None,
    bands: Mapping[str, str] | None = None,
    check: bool = True,
    **params: float,
) -> xr.DataArray | xr.Dataset:
    """Compute one or more spectral indices with optional sensor validation.

    Parameters
    ----------
    obj :
        Dataset with one variable per band, or DataArray with a ``band`` dim.
        Values should be scaled reflectance (see :func:`geolattice.rsense.optical.scale`).
    name :
        Index name, a sequence of names, or :data:`USGS_LANDSAT_SR`.
    sensor :
        Sensor key/alias. Guessed from attributes or band names when omitted.
    bands :
        Explicit canonical → variable mapping, e.g. ``{"nir": "B8A"}``.
    check :
        If True (default), raise when the sensor lacks a required band.
    params :
        Override formula constants (e.g. ``L=0.25`` for SAVI).

    Returns
    -------
    DataArray for a single name, Dataset when several names are requested.
    """
    sensor_key = resolve_sensor(obj, sensor)
    names: Iterable[str] = (name,) if isinstance(name, str) else name
    if check:
        bad = {n: missing_bands(n, sensor_key) for n in names}
        bad = {n: m for n, m in bad.items() if m}
        if bad:
            detail = "; ".join(f"{n} needs {m}" for n, m in bad.items())
            raise ValueError(
                f"Sensor {sensor_key!r} cannot compute requested index(es): {detail}. "
                f"See rs.spectral_index.available({sensor_key!r})."
            )
    return _index(obj, name, sensor=sensor_key, bands=bands, **params)


def usgs_landsat_sr(
    obj: xr.Dataset | xr.DataArray,
    *,
    sensor: str | None = "landsat89",
    bands: Mapping[str, str] | None = None,
    which: Sequence[str] | None = None,
    **params: float,
) -> xr.Dataset:
    """Compute the USGS Landsat Collection 2 Surface Reflectance-derived indices.

    By default returns all eight products listed by USGS (NDVI, EVI, SAVI,
    MSAVI, NDMI, NBR, NBR2, NDSI). Pass ``which=`` to subset. ``sensor``
    defaults to ``landsat89``; pass ``None`` to guess from the Dataset.

    See
    https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices
    """
    names = tuple(which) if which is not None else USGS_LANDSAT_SR
    bad = [n for n in names if ALIASES.get(n.upper(), n.upper()) not in INDICES]
    if bad:
        raise KeyError(f"Unknown USGS index name(s): {bad}. Valid: {USGS_LANDSAT_SR}")
    out = compute(obj, names, sensor=sensor, bands=bands, **params)
    assert isinstance(out, xr.Dataset)
    out.attrs["pyrsense:product"] = "usgs_landsat_sr"
    out.attrs["pyrsense:sensor"] = resolve_sensor(obj, sensor)
    return out


def get_index(name: str) -> SpectralIndex:
    """Return the registered :class:`~geolattice.rsense.optical.indices.SpectralIndex`."""
    key = ALIASES.get(name.upper(), name.upper())
    if key not in INDICES:
        raise KeyError(f"Unknown index {name!r}.")
    return INDICES[key]


__all__ = [
    "USGS_LANDSAT_SR",
    "SENSOR_PROFILES",
    "SensorProfile",
    "available",
    "compute",
    "describe",
    "get_index",
    "index_supported",
    "list_for_sensor",
    "missing_bands",
    "normalize_sensor",
    "resolve_sensor",
    "sensor_bands",
    "sensors",
    "usgs_landsat_sr",
]
