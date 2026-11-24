"""Sensor band definitions and name resolution.

Index formulas are written against *canonical* band names (``blue``, ``red``,
``nir``, ``swir16``...). Data arrives under many names: STAC asset keys
(``nir08``, ``B08``), raw product names (``SR_B5``, ``sur_refl_b02``) or
user-chosen labels. :func:`get_band` finds the right variable for each sensor.
"""

from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

CANONICAL = ("coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
             "nir", "nir08", "swir12", "swir16", "swir22", "thermal", "yellow")

# canonical -> candidate variable names, in priority order
SENSOR_BANDS: dict[str, dict[str, tuple[str, ...]]] = {
    # Landsat 8/9 OLI/TIRS, Collection 2 Level-2 (STAC asset keys, then raw names)
    "landsat": {
        "coastal": ("coastal", "SR_B1"), "blue": ("blue", "SR_B2"), "green": ("green", "SR_B3"),
        "red": ("red", "SR_B4"), "nir": ("nir08", "nir", "SR_B5"), "swir16": ("swir16", "SR_B6"),
        "swir22": ("swir22", "SR_B7"), "thermal": ("lwir11", "lwir", "ST_B10"),
        "qa": ("qa_pixel", "QA_PIXEL"),
    },
    # Landsat 4-7 TM/ETM+, Collection 2 Level-2 (raw band numbering differs from OLI)
    "landsat_tm": {
        "blue": ("blue", "SR_B1"), "green": ("green", "SR_B2"), "red": ("red", "SR_B3"),
        "nir": ("nir08", "nir", "SR_B4"), "swir16": ("swir16", "SR_B5"),
        "swir22": ("swir22", "SR_B7"), "thermal": ("lwir", "lwir11", "ST_B6"),
        "qa": ("qa_pixel", "QA_PIXEL"),
    },
    # Sentinel-2 MSI L2A (Earth Search keys, Planetary Computer keys)
    "sentinel2": {
        "coastal": ("coastal", "B01"), "blue": ("blue", "B02"), "green": ("green", "B03"),
        "red": ("red", "B04"), "rededge1": ("rededge1", "B05"), "rededge2": ("rededge2", "B06"),
        "rededge3": ("rededge3", "B07"), "nir": ("nir", "B08"), "nir08": ("nir08", "B8A"),
        "swir16": ("swir16", "B11"), "swir22": ("swir22", "B12"), "qa": ("scl", "SCL"),
    },
    # Harmonized Landsat Sentinel-2 v2.0, Landsat-derived (HLSL30)
    "hls_l30": {
        "coastal": ("B01",), "blue": ("B02",), "green": ("B03",), "red": ("B04",),
        "nir": ("B05",), "swir16": ("B06",), "swir22": ("B07",), "thermal": ("B10",),
        "qa": ("Fmask",),
    },
    # Harmonized Landsat Sentinel-2 v2.0, Sentinel-2-derived (HLSS30)
    "hls_s30": {
        "coastal": ("B01",), "blue": ("B02",), "green": ("B03",), "red": ("B04",),
        "rededge1": ("B05",), "rededge2": ("B06",), "rededge3": ("B07",), "nir": ("B08",),
        "nir08": ("B8A",), "swir16": ("B11",), "swir22": ("B12",), "qa": ("Fmask",),
    },
    # MODIS Terra/Aqua surface reflectance (MOD09GA/MOD09A1 and MYD equivalents)
    "modis": {
        "red": ("sur_refl_b01",), "nir": ("sur_refl_b02",), "blue": ("sur_refl_b03",),
        "green": ("sur_refl_b04",), "swir12": ("sur_refl_b05",), "swir16": ("sur_refl_b06",),
        "swir22": ("sur_refl_b07",),
        "qa": ("sur_refl_state_500m", "state_1km", "sur_refl_state_1km"),
        "thermal": ("b31", "Band31", "LST_Day_1km"),
    },
    # PlanetScope 4-band (PS2 / PS2.SD). Band order is Blue, Green, Red, NIR.
    "planetscope": {
        "blue": ("blue", "B1"), "green": ("green", "B2"), "red": ("red", "B3"),
        "nir": ("nir", "B4"),
    },
    # PlanetScope SuperDove (PSB.SD), 8-band. Green I is not aliased to green.
    "planetscope8": {
        "coastal": ("coastal_blue", "coastal", "B1"), "blue": ("blue", "B2"),
        "green": ("green", "B4"), "yellow": ("yellow", "B5"), "red": ("red", "B6"),
        "rededge1": ("rededge", "red_edge", "B7"), "nir": ("nir", "B8"),
    },
    # WorldView-2. NIR1 is the vegetation NIR; NIR2 is the broader second NIR.
    "worldview2": {
        "coastal": ("coastal", "B1"), "blue": ("blue", "B2"), "green": ("green", "B3"),
        "yellow": ("yellow", "B4"), "red": ("red", "B5"),
        "rededge1": ("red_edge", "rededge", "B6"), "nir": ("nir1", "nir", "B7"),
        "nir08": ("nir2", "B8"),
    },
    # WorldView-3 VNIR matches WorldView-2. SWIR-1/2/5 stand in for 1.2, 1.6 and 2.2 µm.
    "worldview3": {
        "coastal": ("coastal", "B1"), "blue": ("blue", "B2"), "green": ("green", "B3"),
        "yellow": ("yellow", "B4"), "red": ("red", "B5"),
        "rededge1": ("red_edge", "rededge", "B6"), "nir": ("nir1", "nir", "B7"),
        "nir08": ("nir2", "B8"), "swir12": ("swir1", "SWIR-1"),
        "swir16": ("swir2", "SWIR-2"), "swir22": ("swir5", "SWIR-5"),
    },
    # NAIP 4-band. COG band order is Red, Green, Blue, NIR (not Blue-first).
    "naip": {
        "red": ("red", "Red", "B1"), "green": ("green", "Green", "B2"),
        "blue": ("blue", "Blue", "B3"), "nir": ("nir", "NIR", "B4"),
    },
    # Sentinel-3 OLCI, 21 bands with no exact match to Landsat/Sentinel-2 wavelengths;
    # these are the nearest OLCI band to each canonical slot (see sensors.sentinel3.OLCI_BANDS
    # for every band's centre wavelength). Datasets built by sensors.sentinel3 use lowercase
    # "oa01".."oa21" variable names, which this table maps onto the generic index catalogue.
    "sentinel3_olci": {
        "coastal": ("oa02",), "blue": ("oa04",), "green": ("oa06",), "red": ("oa08",),
        "rededge1": ("oa11",), "nir": ("oa17",), "nir08": ("oa21",),
        "qa": ("wqsf", "lqsf", "quality_flags"),
    },
}

# Landsat 8/9 is the OLI/TIRS map. Kept as its own key for sensor="landsat89".
SENSOR_BANDS["landsat89"] = SENSOR_BANDS["landsat"]

SENSORS = tuple(SENSOR_BANDS)


def _names(obj: xr.Dataset | xr.DataArray) -> list[str]:
    if isinstance(obj, xr.Dataset):
        return [str(v) for v in obj.data_vars]
    if "band" in obj.dims:
        return [str(b) for b in obj["band"].values]
    return [str(obj.name)] if obj.name is not None else []


def guess_sensor(obj: xr.Dataset | xr.DataArray) -> str | None:
    """Guess the sensor from attributes or band names; ``None`` if ambiguous."""
    attr = obj.attrs.get("pyrsense:sensor")
    if attr in SENSOR_BANDS:
        return attr
    names = {n.lower() for n in _names(obj)}
    if "fmask" in names:
        return "hls_s30" if names & {"b8a", "b11", "b12"} else "hls_l30"
    if names & {"scl", "b8a", "rededge1", "b05"}:
        return "sentinel2"
    if names & {"qa_pixel", "nir08", "sr_b5"} or any(n.startswith("sr_b") for n in names):
        return "landsat"
    if any(n.startswith("sur_refl_b") for n in names):
        return "modis"
    if names & {"coastal_blue", "green_i"}:
        return "planetscope8"
    if any(n.startswith("oa") and n[2:4].isdigit() for n in names):
        return "sentinel3_olci"
    if "yellow" in names and names & {"nir1", "nir2", "red_edge"}:
        return "worldview3" if names & {"swir1", "swir2", "swir5", "swir-1"} else "worldview2"
    return None


def resolve_band_name(obj: xr.Dataset | xr.DataArray, band: str, *, sensor: str | None = None,
                      bands: Mapping[str, str] | None = None) -> str:
    """Return the actual variable/band label in ``obj`` for canonical ``band``."""
    available = _names(obj)
    lower = {n.lower(): n for n in available}
    if bands and band in bands:
        if bands[band] not in available:
            raise KeyError(f"Mapped band {bands[band]!r} for {band!r} not found in {available}")
        return bands[band]
    candidates: list[str] = [band]
    sensor = sensor or guess_sensor(obj)
    if sensor:
        if sensor not in SENSOR_BANDS:
            raise ValueError(f"Unknown sensor {sensor!r}; choose from {SENSORS}")
        candidates = list(SENSOR_BANDS[sensor].get(band, ())) + candidates
    if band == "nir":
        candidates.append("nir08")
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(
        f"Band {band!r} not found (sensor={sensor!r}). Available: {available}. "
        f"Pass bands={{'{band}': '<name>'}} to map it explicitly."
    )


def get_band(obj: xr.Dataset | xr.DataArray, band: str, *, sensor: str | None = None,
             bands: Mapping[str, str] | None = None) -> xr.DataArray:
    """Fetch canonical ``band`` from a Dataset (by variable) or DataArray (by ``band`` coord)."""
    name = resolve_band_name(obj, band, sensor=sensor, bands=bands)
    if isinstance(obj, xr.Dataset):
        return obj[name]
    if "band" in obj.dims:
        return obj.sel(band=name, drop=True)
    return obj


def as_dataset(obj: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Convert a DataArray with a ``band`` dim to a Dataset with one variable per band."""
    if isinstance(obj, xr.Dataset):
        return obj
    if "band" not in obj.dims:
        return obj.to_dataset(name=obj.name or "value")
    ds = obj.to_dataset(dim="band")
    ds.attrs.update(obj.attrs)
    return ds
