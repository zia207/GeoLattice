"""Download Sentinel, Landsat, RCM, NAIP, NOAA, and USGS lidar from Python.

Same catalogues, collections, band names, and product types as PyGIS Explorer.
The Streamlit page stays in ``PyGIS-Explorer/app.py``. This module calls
``PyGIS-Explorer/image_downloader.py`` and writes the files into a folder you choose.

Area of interest
    ``bbox=(minlon, minlat, maxlon, maxlat)`` in EPSG:4326, or ``aoi=`` a shapefile,
    GeoJSON, GeoPackage, GeoDataFrame, or shapely geometry. A file with a CRS is
    reprojected to longitude/latitude.

Example::

    from geolattice.rsense.get_images import get_images, search_images

    found = search_images(
        bbox=(-76.6, 42.4, -76.4, 42.5),
        dates=("2024-03-01", "2024-03-31"),
        source="planetary",
        collection="sentinel-2-l2a",
        cloud=10,
    )

    saved = get_images(
        "downloads/s2_ithaca",
        aoi="field.shp",
        dates=("2024-03-01", "2024-03-31"),
        source="earth-search",
        collection="sentinel-2-l2a",
        cloud=10,
        bands=["red", "green", "blue", "nir", "scl"],
        stack=True,
        clearest=True,
    )

``source`` is ``earth-search``, ``planetary``, ``cdse``, ``rcm-ard``, ``noaa``, or ``usgs``.
``collection`` is a name from :data:`COLLECTIONS`. Landsat Level-1 products use
``landsat_product`` (``l2``, ``toa``, ``l1tp``, ``l1gt``). Copernicus product types
are in :data:`CDSE_TYPES` (set ``CDSE_USER`` and ``CDSE_PASSWORD``, or pass
``cdse_user`` and ``cdse_password``). Sentinel-1 accepts ``s1_mode``,
``s1_polarization``, and ``s1_orbit``. Sentinel-5P accepts ``s5p_product``.

``mode="clip"`` reads the area out of each cloud-optimized GeoTIFF.
``mode="full"`` saves the whole asset. Copernicus, NOAA, USGS lidar, and NetCDF
products are always whole files. NAIP is always a clip of red, green, blue, and
near infrared, using the newest year that covers the area.

RADARSAT Constellation Mission (RCM) CEOS Analysis Ready Data
    :func:`search_rcm_ard` and :func:`get_rcm_ard` are thin, pre-configured wrappers around
    :func:`search_images`/:func:`get_images` (``source="rcm-ard", collection="rcm-ard"`` also
    works directly) for Canada's RCM-ARD archive
    (https://registry.opendata.aws/rcm-ceos-ard/): systematic compact-polarization, 30 m,
    C-band coverage of Canada, served as cloud-optimized GeoTIFFs from a STAC catalogue at
    Natural Resources Canada's EODMS and mirrored, open and unsigned, in the AWS
    ``rcm-ceos-ard`` S3 bucket (``ca-central-1``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

SOURCES = ("earth-search", "planetary", "cdse", "rcm-ard", "noaa", "usgs")

SOURCE_LABELS = {
    "earth-search": "Earth Search (AWS)",
    "planetary": "Planetary Computer",
    "cdse": "Copernicus (CDSE)",
    "rcm-ard": "RCM CEOS ARD (EODMS / AWS)",
    "noaa": "NOAA Digital Coast",
    "usgs": "USGS 3DEP Lidar",
}

DEFAULT_COLLECTION = {
    "earth-search": "sentinel-2-l2a",
    "planetary": "sentinel-2-l2a",
    "cdse": "SENTINEL-2",
    "rcm-ard": "rcm-ard",
    "noaa": "noaa-rgb8",
    "usgs": "lidar-lpc",
}

COLLECTIONS = {
    "earth-search": ["sentinel-2-l2a", "sentinel-2-l1c", "landsat-c2-l2", "naip"],
    "planetary": [
        "sentinel-2-l2a",
        "sentinel-1-rtc",
        "sentinel-1-grd",
        "landsat-c2-l2",
        "modis-17A2H-061",
        "modis-15A2H-061",
        "modis-09Q1-061",
        "modis-13Q1-061",
        "hls2-s30",
        "hls2-l30",
        "sentinel-3-synergy-vg1-l2-netcdf",
        "sentinel-3-olci-lfr-l2-netcdf",
        "sentinel-3-slstr-lst-l2-netcdf",
        "sentinel-3-olci-wfr-l2-netcdf",
        "sentinel-3-slstr-wst-l2-netcdf",
        "sentinel-3-slstr-frp-l2-netcdf",
        "sentinel-3-synergy-syn-l2-netcdf",
        "sentinel-3-synergy-aod-l2-netcdf",
        "sentinel-5p-l2-netcdf",
        "jrc-gsw",
    ],
    "cdse": ["SENTINEL-2", "SENTINEL-1", "SENTINEL-3", "SENTINEL-5P", "LANDSAT-8", "LANDSAT-9"],
    "rcm-ard": ["rcm-ard"],
    "noaa": ["noaa-rgb8", "noaa-rgb16", "noaa-rgbn8", "noaa-cir8", "noaa-naip"],
    "usgs": ["lidar-lpc", "lidar-projects"],
}

COLLECTION_LABELS = {
    "sentinel-2-l2a": "Sentinel-2 L2A",
    "sentinel-2-l1c": "Sentinel-2 L1C",
    "sentinel-1-rtc": "Sentinel-1 RTC",
    "sentinel-1-grd": "Sentinel-1 GRD",
    "landsat-c2-l2": "Landsat Collection 2",
    "landsat-c2-l1": "Landsat Collection 2 Level-1",
    "naip": "USDA NAIP (latest)",
    "modis-17A2H-061": "MODIS Gross Primary Productivity 8-Day",
    "modis-15A2H-061": "MODIS Leaf Area Index/FPAR 8-Day",
    "modis-09Q1-061": "MODIS Surface Reflectance 8-Day (250m)",
    "modis-13Q1-061": "MODIS Vegetation Indices 16-Day (250m)",
    "hls2-s30": "HLS v2.0, Sentinel-2",
    "hls2-l30": "HLS v2.0, Landsat",
    "sentinel-3-synergy-vg1-l2-netcdf": "Sentinel-3 1-Day Surface Reflectance and NDVI",
    "sentinel-3-olci-lfr-l2-netcdf": "Sentinel-3 Land (Full Resolution)",
    "sentinel-3-slstr-lst-l2-netcdf": "Sentinel-3 Land Surface Temperature",
    "sentinel-3-olci-wfr-l2-netcdf": "Sentinel-3 Water (Full Resolution)",
    "sentinel-3-slstr-wst-l2-netcdf": "Sentinel-3 sea surface temperature",
    "sentinel-3-slstr-frp-l2-netcdf": "Sentinel-3 fire radiative power",
    "sentinel-3-synergy-syn-l2-netcdf": "Sentinel-3 synergy reflectance",
    "sentinel-3-synergy-aod-l2-netcdf": "Sentinel-3 synergy aerosol",
    "sentinel-5p-l2-netcdf": "Sentinel-5P Level-2",
    "jrc-gsw": "JRC Global Surface Water",
    "SENTINEL-2": "Sentinel-2",
    "SENTINEL-1": "Sentinel-1",
    "SENTINEL-3": "Sentinel-3",
    "SENTINEL-5P": "Sentinel-5P",
    "LANDSAT-8": "Landsat 8",
    "LANDSAT-9": "Landsat 9",
    "rcm-ard": "RCM CEOS Analysis Ready Data",
    "noaa-rgb8": "Natural color, 8-bit",
    "noaa-rgb16": "Natural color, 16-bit",
    "noaa-rgbn8": "Color and near infrared, 8-bit",
    "noaa-cir8": "Color infrared, 8-bit",
    "noaa-naip": "USDA NAIP (latest)",
    "lidar-lpc": "Lidar point cloud (LAZ)",
    "lidar-projects": "Lidar projects",
}

S2_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir22",
    "scl",
    "coastal",
    "visual",
]
LS_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "coastal", "lwir11", "qa_pixel"]
LS_L1_BANDS = ["red", "green", "nir08", "nir09", "qa_pixel"]

# RCM-ARD's raster assets, keyed by their native STAC asset id (same name downloads under).
# "rr"/"rl" (RR/RL circular-polarization backscatter) and "data_mask" are the defaults;
# "rrrl" is the full normalized compact-pol covariance matrix.
RCM_ARD_BANDS = [
    "rr", "rl", "rrrl", "gamma_to_sigma_ratio", "local_inc_angle",
    "local_contributing_area", "data_mask",
]
RCM_ARD_BAND_LABELS = {
    "rr": "Backscatter, RR polarization",
    "rl": "Backscatter, RL polarization",
    "rrrl": "Normalized polarimetric covariance matrix (CovMat)",
    "gamma_to_sigma_ratio": "Gamma0-to-sigma0 ratio",
    "local_inc_angle": "Local incidence angle",
    "local_contributing_area": "Scattering (local contributing) area",
    "data_mask": "Data/QA mask",
}
RCM_ARD_DEFAULT_BANDS = ["rr", "rl", "data_mask"]

S1_MODES = ("SM", "IW", "EW")
S1_POLARIZATIONS = {
    "VV": ["vv"],
    "VV+VH": ["vv", "vh"],
    "HH": ["hh"],
    "HH+HV": ["hh", "hv"],
}
S1_CDSE_POL = {"VV": "VV", "VV+VH": "VV&VH", "HH": "HH", "HH+HV": "HH&HV"}

LS_PLATFORM_CHOICES = {
    "Landsat 8 and 9": "landsat-8,landsat-9",
    "Landsat 8": "landsat-8",
    "Landsat 9": "landsat-9",
    "Landsat 4–9": "landsat-4,landsat-5,landsat-7,landsat-8,landsat-9",
}
LS_L1_PLATFORMS = {
    "Landsat 1–5": "landsat-1,landsat-2,landsat-3,landsat-4,landsat-5",
    "Landsat 5": "landsat-5",
    "Landsat 4": "landsat-4",
    "Landsat 1–3": "landsat-1,landsat-2,landsat-3",
}
LS_PRODUCTS = ("l2", "toa", "l1tp", "l1gt")
LS_L1_CORRECTION = {"l1tp": "L1TP", "l1gt": "L1GT"}

S3_VARIABLES = {
    "sentinel-3-olci-lfr-l2-netcdf": ["otci", "gifapar", "iwv", "rc-gifapar", "lqsf"],
    "sentinel-3-olci-wfr-l2-netcdf": [
        "chl-oc4me",
        "chl-nn",
        "tsm-nn",
        "iop-nn",
        "iwv",
        "par",
        "wqsf",
        "oa04-reflectance",
        "oa06-reflectance",
        "oa08-reflectance",
    ],
    "sentinel-3-slstr-lst-l2-netcdf": ["lst-in", "slstr-flags-in"],
    "sentinel-3-slstr-wst-l2-netcdf": ["l2p"],
    "sentinel-3-slstr-frp-l2-netcdf": ["frp-in"],
    "sentinel-3-synergy-aod-l2-netcdf": ["ntc-aod"],
    "sentinel-3-synergy-syn-l2-netcdf": [
        "syn-oa04-reflectance",
        "syn-oa06-reflectance",
        "syn-oa08-reflectance",
        "syn-oa17-reflectance",
        "syn-s5n-reflectance",
        "syn-s6n-reflectance",
        "syn-ato550",
    ],
    "sentinel-3-synergy-vg1-l2-netcdf": ["ndvi", "b0", "b2", "b3", "mir"],
}

S5P_PRODUCTS = {
    "L2__NO2___": ("Nitrogen dioxide", "no2"),
    "L2__O3____": ("Ozone", "o3"),
    "L2__SO2___": ("Sulfur dioxide", "so2"),
    "L2__CO____": ("Carbon monoxide", "co"),
    "L2__CH4___": ("Methane", "ch4"),
    "L2__HCHO__": ("Formaldehyde", "hcho"),
    "L2__AER_AI": ("Aerosol index", "aer-ai"),
    "L2__AER_LH": ("Aerosol layer height", "aer-lh"),
    "L2__CLOUD_": ("Cloud", "cloud"),
}

CDSE_TYPES = {
    "SENTINEL-2": ["S2MSI2A", "S2MSI1C"],
    "SENTINEL-1": ["GRD", "SLC"],
    "SENTINEL-3": [
        "OL_2_LFR___",
        "OL_2_WFR___",
        "OL_1_EFR___",
        "SL_2_LST___",
        "SL_2_WST___",
        "SL_2_FRP___",
        "SL_1_RBT___",
        "SY_2_SYN___",
        "SY_2_AOD___",
        "SY_2_VGP___",
        "SY_2_VG1___",
        "SY_2_V10___",
    ],
    "SENTINEL-5P": [
        "L2__NO2___",
        "L2__O3____",
        "L2__SO2___",
        "L2__CO____",
        "L2__CH4___",
        "L2__HCHO__",
        "L2__AER_AI",
        "L2__AER_LH",
        "L2__CLOUD_",
        "L2B_SIF___",
        "L2__BRO___",
    ],
    "LANDSAT-8": ["L1TP", "L1GT"],
    "LANDSAT-9": ["L1TP", "L1GT"],
}

PLANETARY_BANDS = {
    "modis-17A2H-061": {
        "gpp": "Gpp_500m",
        "net_photosynthesis": "PsnNet_500m",
        "qc": "Psn_QC_500m",
    },
    "modis-15A2H-061": {
        "lai": "Lai_500m",
        "fpar": "Fpar_500m",
        "lai_sd": "LaiStdDev_500m",
        "fpar_sd": "FparStdDev_500m",
        "qc": "FparLai_QC",
    },
    "modis-09Q1-061": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "qc": "sur_refl_qc_250m",
        "state": "sur_refl_state_250m",
    },
    "modis-13Q1-061": {
        "ndvi": "250m_16_days_NDVI",
        "evi": "250m_16_days_EVI",
        "red": "250m_16_days_red_reflectance",
        "nir": "250m_16_days_NIR_reflectance",
        "blue": "250m_16_days_blue_reflectance",
        "mir": "250m_16_days_MIR_reflectance",
        "quality": "250m_16_days_VI_Quality",
        "reliability": "250m_16_days_pixel_reliability",
    },
    "hls2-s30": {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "rededge1": "B05",
        "rededge2": "B06",
        "rededge3": "B07",
        "nir": "B08",
        "nir08": "B8A",
        "swir16": "B11",
        "swir22": "B12",
        "fmask": "Fmask",
    },
    "hls2-l30": {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir": "B05",
        "swir16": "B06",
        "swir22": "B07",
        "cirrus": "B09",
        "lwir": "B10",
        "lwir2": "B11",
        "fmask": "Fmask",
    },
    "jrc-gsw": {
        "occurrence": "occurrence",
        "change": "change",
        "seasonality": "seasonality",
        "recurrence": "recurrence",
        "transitions": "transitions",
        "extent": "extent",
    },
}
PLANETARY_DEFAULT_BANDS = {
    "modis-17A2H-061": ["gpp", "net_photosynthesis"],
    "modis-15A2H-061": ["lai", "fpar"],
    "modis-09Q1-061": ["red", "nir"],
    "modis-13Q1-061": ["ndvi", "evi", "red", "nir"],
    "hls2-s30": ["red", "green", "blue", "nir"],
    "hls2-l30": ["red", "green", "blue", "nir"],
    "jrc-gsw": ["occurrence", "extent"],
}

_EXPLORER = None


@dataclass
class Area:
    """Area of interest in EPSG:4326."""

    bbox: tuple[float, float, float, float]
    geometry: dict

    def wkt(self) -> str:
        from shapely.geometry import shape

        return shape(self.geometry).wkt


@dataclass
class SearchResult:
    """Scenes returned by :func:`search_images`, before any files are written."""

    source: str
    collection: str
    start: str | None
    end: str | None
    area: Area
    limit: int = 50
    scenes: list[dict] = field(default_factory=list)
    items: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SearchResult({self.source}/{self.collection}, " f"{len(self.scenes)} scenes)"


@dataclass
class ImageResult:
    """Files written by :func:`get_images`."""

    out: Path
    source: str
    collection: str
    mode: str
    area: Area
    scenes: list[dict] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ImageResult({self.source}/{self.collection}, {len(self.scenes)} scenes, "
            f"{len(self.files)} files, {len(self.failures)} failures, out={self.out})"
        )


def list_collections(source: str | None = None):
    """Collection ids for one source, or a dict of every source."""
    if source is None:
        return {name: list(cols) for name, cols in COLLECTIONS.items()}
    _check_source(source)
    return list(COLLECTIONS[source])


def bands_for(collection: str, source: str | None = None) -> list[str]:
    """Band or variable names the Explorer offers for this collection."""
    del source  # names follow the collection, which already implies the source
    if collection in PLANETARY_BANDS:
        return list(PLANETARY_BANDS[collection])
    if collection in S3_VARIABLES:
        return list(S3_VARIABLES[collection])
    if collection == "rcm-ard":
        return list(RCM_ARD_BANDS)
    lowered = (collection or "").lower()
    if "sentinel-5p" in lowered or collection == "SENTINEL-5P":
        return [asset for _label, asset in S5P_PRODUCTS.values()]
    if "sentinel-1" in lowered or collection == "SENTINEL-1":
        return ["vv", "vh", "hh", "hv"]
    if collection == "landsat-c2-l1":
        return list(LS_L1_BANDS)
    if "landsat" in lowered:
        return list(LS_BANDS)
    if collection in ("naip", "noaa-naip", "noaa-rgbn8"):
        return ["red", "green", "blue", "nir"]
    if lowered.startswith("noaa") or lowered.startswith("lidar"):
        return []
    return list(S2_BANDS)


def default_bands(
    collection: str,
    source: str = "earth-search",
    *,
    s1_polarization: str | None = None,
    s5p_product: str | None = None,
    landsat_product: str = "l2",
) -> list[str]:
    """Bands :func:`get_images` downloads when ``bands`` is omitted."""
    lowered = (collection or "").lower()
    if collection == "naip":
        return ["red", "green", "blue", "nir"]
    if (
        source in ("cdse", "noaa", "usgs")
        or lowered.startswith("noaa")
        or lowered.startswith("lidar")
    ):
        return []
    if collection == "landsat-c2-l1" or (
        landsat_product != "l2" and "landsat" in lowered and source == "planetary"
    ):
        return list(LS_L1_BANDS)
    if collection in PLANETARY_DEFAULT_BANDS:
        return list(PLANETARY_DEFAULT_BANDS[collection])
    if collection == "rcm-ard":
        return list(RCM_ARD_DEFAULT_BANDS)
    if "sentinel-1" in lowered or collection == "SENTINEL-1":
        key = _s1_polarization(s1_polarization or "VV+VH")
        return list(S1_POLARIZATIONS[key])
    if s5p_product:
        return [S5P_PRODUCTS[_s5p_code(s5p_product)][1]]
    if "sentinel-5p" in lowered or collection == "SENTINEL-5P":
        return [S5P_PRODUCTS["L2__NO2___"][1]]
    if collection in S3_VARIABLES:
        opts = S3_VARIABLES[collection]
        return opts[: min(3, len(opts))]
    if "landsat" in lowered:
        return ["red", "green", "blue", "nir"]
    if lowered.startswith("sentinel-2") or collection == "SENTINEL-2":
        return ["red", "green", "blue", "nir", "scl"]
    return ["red", "green", "blue", "nir"]


def load_aoi(bbox=None, aoi=None) -> Area:
    """Build an EPSG:4326 area from a bounding box or a vector dataset.

    ``bbox`` is ``(minlon, minlat, maxlon, maxlat)`` or the same four numbers
    separated by commas. ``aoi`` is a shapefile, GeoJSON, GeoPackage, a
    GeoDataFrame, a GeoJSON dict, or a shapely geometry in longitude/latitude.
    """
    if bbox is not None and aoi is not None:
        raise ValueError("Pass bbox or aoi, not both.")
    if bbox is None and aoi is None:
        raise ValueError(
            "Pass bbox=(minlon, minlat, maxlon, maxlat) or aoi='area.shp' / 'area.geojson'."
        )
    if bbox is not None:
        return _area_from_bounds(_bbox_tuple(bbox))
    if isinstance(aoi, (str, PathLike)):
        return _from_file(Path(aoi))
    if _is_bbox(aoi):
        return _area_from_bounds(_bbox_tuple(aoi))
    if hasattr(aoi, "to_crs") and hasattr(aoi, "geometry"):
        return _from_gdf(aoi)
    if isinstance(aoi, dict):
        return _from_geojson_obj(aoi)
    if hasattr(aoi, "bounds") and hasattr(aoi, "__geo_interface__"):
        geom = json.loads(json.dumps(aoi.__geo_interface__))
        return _area_from_bounds(tuple(float(v) for v in aoi.bounds), geom)
    raise TypeError(f"Unsupported AOI type: {type(aoi).__name__}")


def naip_coverage() -> list[dict]:
    """USDA NAIP years on file for each state, the same table the Explorer draws."""
    return _call(lambda: _explorer().naip_coverage())


def search_images(
    *,
    bbox=None,
    aoi=None,
    dates=None,
    start=None,
    end=None,
    source: str = "earth-search",
    collection: str | None = None,
    sensor: str | None = None,
    cloud: float | None = None,
    max_cloud: float | None = None,
    limit: int = 50,
    product_type: str | None = None,
    platforms=None,
    landsat_product: str = "l2",
    s1_mode: str | None = None,
    s1_polarization: str | None = None,
    s1_orbit: str | None = None,
    s5p_product: str | None = None,
    ids: Sequence[str] | None = None,
    clearest: bool = False,
    cdse_user: str | None = None,
    cdse_password: str | None = None,
    quiet: bool = False,
) -> SearchResult:
    """List scenes for an area. Nothing is downloaded.

    ``cloud`` and ``max_cloud`` are the same limit (percent). ``sensor`` is
    another name for ``collection``. ``clearest=True`` keeps the scene with the
    lowest cloud cover. ``ids`` keeps only those scene ids.
    """
    req = _prepare(
        bbox=bbox,
        aoi=aoi,
        dates=dates,
        start=start,
        end=end,
        source=source,
        collection=collection,
        sensor=sensor,
        cloud=cloud,
        max_cloud=max_cloud,
        limit=limit,
        product_type=product_type,
        platforms=platforms,
        landsat_product=landsat_product,
        s1_mode=s1_mode,
        s1_polarization=s1_polarization,
        s1_orbit=s1_orbit,
        s5p_product=s5p_product,
        cdse_user=cdse_user,
        cdse_password=cdse_password,
    )
    items = _run_search(req)
    items, notes = _select(items, ids, clearest, req["source"])
    scenes = [_summarize(item, req["source"]) for item in items]
    if not scenes:
        notes = [*notes, _empty_message(req)]
    _say(quiet, f"{len(scenes)} scene(s) from {req['source']} / {req['collection']}")
    for note in notes:
        _say(quiet, note)
    return SearchResult(
        source=req["source"],
        collection=req["collection"],
        start=req["start"],
        end=req["end"],
        area=req["area"],
        limit=req["limit"],
        scenes=scenes,
        items=items,
        notes=notes,
    )


def get_images(
    out,
    *,
    bbox=None,
    aoi=None,
    dates=None,
    start=None,
    end=None,
    source: str = "earth-search",
    collection: str | None = None,
    sensor: str | None = None,
    cloud: float | None = None,
    max_cloud: float | None = None,
    bands: Sequence[str] | None = None,
    mode: str = "clip",
    stack=None,
    limit: int = 50,
    workers: int = 4,
    product_type: str | None = None,
    platforms=None,
    landsat_product: str = "l2",
    s1_mode: str | None = None,
    s1_polarization: str | None = None,
    s1_orbit: str | None = None,
    s5p_product: str | None = None,
    ids: Sequence[str] | None = None,
    clearest: bool = False,
    search: SearchResult | None = None,
    cdse_user: str | None = None,
    cdse_password: str | None = None,
    quiet: bool = False,
) -> ImageResult:
    """Search and download into ``out``.

    Pass ``search=`` a :class:`SearchResult` to download scenes you already
    listed. Otherwise pass ``bbox`` or ``aoi`` and ``dates``, the same arguments
    as :func:`search_images`.

    ``bands`` defaults to the Explorer sidebar (red, green, blue, near infrared,
    and SCL for Sentinel-2). ``stack=True`` also writes ``stack.tif`` from red,
    green, blue, and near infrared when ``mode`` is ``clip``. ``clearest=True``
    downloads only the least cloudy scene.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _apply_cdse_login(cdse_user, cdse_password)
    if search is None:
        found = search_images(
            bbox=bbox,
            aoi=aoi,
            dates=dates,
            start=start,
            end=end,
            source=source,
            collection=collection,
            sensor=sensor,
            cloud=cloud,
            max_cloud=max_cloud,
            limit=limit,
            product_type=product_type,
            platforms=platforms,
            landsat_product=landsat_product,
            s1_mode=s1_mode,
            s1_polarization=s1_polarization,
            s1_orbit=s1_orbit,
            s5p_product=s5p_product,
            ids=ids,
            clearest=clearest,
            quiet=quiet,
        )
    else:
        if not isinstance(search, SearchResult):
            raise TypeError("search must be a SearchResult from search_images()")
        items, notes = _select(search.items, ids, clearest, search.source)
        found = SearchResult(
            source=search.source,
            collection=search.collection,
            start=search.start,
            end=search.end,
            area=search.area,
            limit=search.limit,
            scenes=[_summarize(item, search.source) for item in items],
            items=items,
            notes=[*search.notes, *notes],
        )
        landsat_product = "l2"
    mode = _mode(mode, found.source, found.collection)
    if bands is None:
        bands = default_bands(
            found.collection,
            found.source,
            s1_polarization=s1_polarization,
            s5p_product=s5p_product,
            landsat_product=landsat_product,
        )
    else:
        bands = list(bands)
    stack_names = _stack_names(stack, bands, mode)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "aoi.geojson").write_text(
        json.dumps({"type": "Feature", "geometry": found.area.geometry, "properties": {}}, indent=2)
    )
    failures: list[dict] = []
    notes = list(found.notes)
    if found.items:
        _say(quiet, f"Downloading {len(found.items)} scene(s) → {out_path}")
        failures, download_notes = _download(
            found,
            out_path,
            bands,
            mode,
            stack_names,
            workers,
        )
        notes.extend(download_notes)
    files = _list_files(out_path)
    summary = {
        "source": found.source,
        "collection": found.collection,
        "mode": mode,
        "bbox": list(found.area.bbox),
        "bands": list(bands),
        "scenes": found.scenes,
        "failures": failures,
        "notes": notes,
    }
    (out_path / "request.json").write_text(json.dumps(summary, indent=2))
    if failures:
        _say(quiet, f"{len(failures)} download(s) failed — see {out_path / 'request.json'}")
    else:
        _say(quiet, f"Done → {out_path.resolve()} ({len(files)} files)")
    return ImageResult(
        out=out_path,
        source=found.source,
        collection=found.collection,
        mode=mode,
        area=found.area,
        scenes=found.scenes,
        files=files,
        failures=failures,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# RCM CEOS Analysis Ready Data (RADARSAT Constellation Mission)
# --------------------------------------------------------------------------- #
def _rcm_orbit(orbit: str | None) -> str | None:
    if orbit is None:
        return None
    text = str(orbit).strip().lower()
    if text not in ("ascending", "descending"):
        raise ValueError("orbit must be 'ascending' or 'descending'.")
    return text


def search_rcm_ard(*, bbox=None, aoi=None, dates=None, start=None, end=None,
                   orbit: str | None = None, limit: int = 50, ids: Sequence[str] | None = None,
                   clearest: bool = False, quiet: bool = False) -> SearchResult:
    """List RADARSAT Constellation Mission (RCM) CEOS Analysis Ready Data scenes.

    RCM-ARD (https://registry.opendata.aws/rcm-ceos-ard/) is systematic compact-polarization
    (CH/CV/XC), 30 m, C-band coverage of Canada with a 12-day revisit. It is served from a
    STAC catalogue at Natural Resources Canada's EODMS and mirrored, open and unsigned, in the
    ``rcm-ceos-ard`` S3 bucket (``ca-central-1``). ``orbit`` keeps only ``"ascending"`` or
    ``"descending"`` passes (``sat:orbit_state``), useful for a consistent-geometry time
    series -- filtered here in Python rather than as a server-side STAC query, since EODMS's
    catalogue does not advertise the STAC ``query`` extension that would otherwise need (a
    server-side ``s1_orbit``-style filter would be silently ignored, not an error). Nothing is
    downloaded here; pass the result to :func:`get_rcm_ard` via ``search=``, or call
    :func:`get_rcm_ard` directly.

    ``clearest`` has no cloud-cover meaning for SAR; with no ``eo:cloud_cover`` field, it keeps
    the earliest scene in range when more than one is found, same degrade-gracefully behaviour
    :func:`search_images` already has for any cloud-free collection.

    The EODMS STAC API rate-limits large page sizes -- keep ``limit`` modest (50, the default,
    is comfortably within it) rather than paging thousands of items in one call.
    """
    orbit = _rcm_orbit(orbit)
    found = search_images(bbox=bbox, aoi=aoi, dates=dates, start=start, end=end,
                          source="rcm-ard", collection="rcm-ard", limit=limit, ids=ids,
                          clearest=False, quiet=True)
    items, notes = found.items, list(found.notes)
    if orbit:
        wanted = orbit.capitalize()  # STAC sat:orbit_state values are "Ascending"/"Descending"
        items = [it for it in items if it.properties.get("sat:orbit_state") == wanted]
        if not items:
            notes.append(f"No RCM-ARD scenes on the {orbit} orbit in this area and date range.")
    if clearest and len(items) > 1:
        items = [min(items, key=lambda it: it.datetime.isoformat() if it.datetime else "")]
    scenes = [_summarize(it, "rcm-ard") for it in items]
    if not scenes and not notes:
        notes.append(_empty_message({"source": "rcm-ard", "collection": "rcm-ard"}))
    result = SearchResult(source="rcm-ard", collection="rcm-ard", start=found.start,
                          end=found.end, area=found.area, limit=found.limit,
                          scenes=scenes, items=items, notes=notes)
    _say(quiet, f"{len(scenes)} scene(s) from rcm-ard / rcm-ard" + (f" ({orbit})" if orbit else ""))
    for note in notes:
        _say(quiet, note)
    return result


def get_rcm_ard(out, *, bbox=None, aoi=None, dates=None, start=None, end=None,
                orbit: str | None = None, bands: Sequence[str] | None = None,
                mode: str = "clip", limit: int = 50, workers: int = 4,
                ids: Sequence[str] | None = None, clearest: bool = False,
                search: SearchResult | None = None, quiet: bool = False) -> ImageResult:
    """Download RADARSAT Constellation Mission (RCM) CEOS Analysis Ready Data.

    See :func:`search_rcm_ard` for the dataset and ``orbit``. ``bands`` defaults to
    :data:`RCM_ARD_DEFAULT_BANDS` (``["rr", "rl", "data_mask"]``: RR/RL backscatter, linear
    amplitude, plus the data mask); pass e.g. ``bands=["rrrl", "local_inc_angle"]`` for the
    full compact-pol covariance matrix and local incidence angle too -- :data:`RCM_ARD_BANDS`
    lists every raster asset, :data:`RCM_ARD_BAND_LABELS` what each one is.

    ``mode="clip"`` (the default) reads only the AOI window out of each cloud-optimized
    GeoTIFF, which matters here: a whole RCM-ARD scene can run close to 2 GB per band.
    ``mode="full"`` saves the whole asset instead.

    Example::

        from geolattice.rsense.get_images import get_rcm_ard

        saved = get_rcm_ard(
            "downloads/rcm_ottawa",
            bbox=(-76.0, 45.2, -75.5, 45.5),
            dates=("2025-01-01", "2025-05-31"),
            orbit="ascending",
            clearest=True,
        )
        print(saved.files)
    """
    if search is None:
        search = search_rcm_ard(bbox=bbox, aoi=aoi, dates=dates, start=start, end=end,
                                orbit=orbit, limit=limit, clearest=clearest, quiet=quiet)
    return get_images(out, source="rcm-ard", collection="rcm-ard", bands=bands, mode=mode,
                      limit=limit, workers=workers, ids=ids, clearest=False, search=search,
                      quiet=quiet)


# --------------------------------------------------------------------------- #
# Explorer loader
# --------------------------------------------------------------------------- #
def _explorer_path() -> Path:
    env = os.environ.get("GEOLATTICE_EXPLORER")
    candidates = []
    if env:
        env_path = Path(env).expanduser()
        candidates.append(
            env_path if env_path.name == "image_downloader.py" else env_path / "image_downloader.py"
        )
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "PyGIS-Explorer" / "image_downloader.py")
    candidates.append(Path.cwd() / "PyGIS-Explorer" / "image_downloader.py")
    # Shipped with the wheel so an install works without the Explorer folder.
    candidates.append(here.with_name("_image_downloader.py"))
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    raise ImportError(
        "Could not find the image downloader. It ships as geolattice.rsense._image_downloader. "
        "Set GEOLATTICE_EXPLORER to a PyGIS-Explorer folder to use a different copy."
    )


def _explorer():
    """Import ``PyGIS-Explorer/image_downloader.py`` without adding it to ``sys.path``."""
    global _EXPLORER
    if _EXPLORER is not None:
        return _EXPLORER
    path = _explorer_path()
    spec = importlib.util.spec_from_file_location("geolattice_explorer_image_downloader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's annotations.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(spec.name, None)
        raise ImportError(
            "PyGIS Explorer's downloader needs requests, pystac-client, rasterio, "
            f"and shapely. Original error: {exc}"
        ) from exc
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    _EXPLORER = module
    return module


def _call(fn):
    """Run a downloader call. The script uses ``sys.exit`` for missing credentials."""
    try:
        return fn()
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) and exc.code else "download stopped"
        raise RuntimeError(message) from exc


def _say(quiet: bool, *parts: object) -> None:
    if not quiet:
        print(*parts)


# --------------------------------------------------------------------------- #
# Request preparation
# --------------------------------------------------------------------------- #
def _prepare(
    *,
    bbox,
    aoi,
    dates,
    start,
    end,
    source,
    collection,
    sensor,
    cloud,
    max_cloud,
    limit,
    product_type,
    platforms,
    landsat_product,
    s1_mode,
    s1_polarization,
    s1_orbit,
    s5p_product,
    cdse_user,
    cdse_password,
) -> dict:
    _check_source(source)
    _apply_cdse_login(cdse_user, cdse_password)
    if landsat_product not in LS_PRODUCTS:
        raise ValueError(f"landsat_product must be one of: {', '.join(LS_PRODUCTS)}")
    collection = collection or sensor or DEFAULT_COLLECTION[source]
    collection = _resolve_collection(source, collection, landsat_product)
    start_day, end_day = _dates(dates, start, end, collection)
    if cloud is None:
        cloud = max_cloud
    if limit < 1:
        raise ValueError("limit must be at least 1")
    area = load_aoi(bbox=bbox, aoi=aoi)
    mode_name, polarization, orbit = _s1_filters(s1_mode, s1_polarization, s1_orbit)
    product = _product_type(source, collection, product_type)
    s5p = _s5p_code(s5p_product) if s5p_product else None
    return {
        "source": source,
        "collection": collection,
        "start": start_day,
        "end": end_day,
        "cloud": None if cloud is None else float(cloud),
        "limit": int(limit),
        "area": area,
        "platforms": _platforms(platforms, collection),
        "extra": _extra_query(mode_name, polarization, orbit, landsat_product, s5p, collection),
        "product_type": product,
        "s1_mode": mode_name,
        "s1_polarisation": S1_CDSE_POL.get(polarization) if polarization else None,
        "s1_orbit": orbit.upper() if orbit else None,
        "landsat_product": landsat_product,
        "s5p_product": s5p,
    }


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of: {', '.join(SOURCES)}")


def _resolve_collection(source: str, collection: str, landsat_product: str) -> str:
    known = COLLECTIONS[source]
    landsat = "landsat" in collection.lower()
    if source == "earth-search" and (
        collection == "landsat-c2-l1"
        or (
            landsat_product != "l2"
            and (collection in ("landsat-c2-l2", "landsat-c2-l1") or landsat)
        )
    ):
        raise ValueError(
            "Earth Search publishes Landsat Collection 2 Level-2. "
            "Use source='planetary' for TOA, L1TP, and L1GT."
        )
    if (
        landsat_product != "l2"
        and source == "planetary"
        and (collection in ("landsat-c2-l2", "landsat-c2-l1") or landsat)
    ):
        collection = "landsat-c2-l1"
    if collection not in known and collection != "landsat-c2-l1":
        if source in ("cdse", "noaa", "usgs"):
            raise ValueError(
                f"Unknown {source} collection {collection!r}. " f"Choose one of: {', '.join(known)}"
            )
    if source == "cdse" and collection not in known:
        raise ValueError(
            f"Unknown cdse collection {collection!r}. Choose one of: {', '.join(known)}"
        )
    return collection


def _dates(dates, start, end, collection: str) -> tuple[str | None, str | None]:
    if dates is None and start is not None and end is not None:
        dates = (start, end)
    if dates is None:
        if collection == "naip":
            return "2010-01-01", date.today().isoformat()
        if collection == "jrc-gsw":
            return "1984-01-01", "2021-12-31"
        raise ValueError("dates=(start, end) is required, as YYYY-MM-DD strings.")
    if isinstance(dates, str):
        if "/" not in dates:
            raise ValueError("dates must be (start, end) or 'YYYY-MM-DD/YYYY-MM-DD'.")
        start_text, end_text = dates.split("/", 1)
    else:
        start_text, end_text = dates
    start_day, end_day = _day(start_text), _day(end_text)
    if start_day > end_day:
        raise ValueError("dates start is after end.")
    return start_day, end_day


def _day(value) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    text = text[:10]
    date.fromisoformat(text)
    return text


def _product_type(source: str, collection: str, product_type: str | None) -> str | None:
    if not product_type or source != "cdse":
        return product_type
    allowed = CDSE_TYPES.get(collection)
    if allowed and product_type not in allowed:
        raise ValueError(
            f"Unknown product type {product_type!r} for {collection}. "
            f"Choose one of: {', '.join(allowed)}"
        )
    return product_type


def _s1_filters(mode, polarization, orbit):
    if mode is not None and mode not in S1_MODES:
        raise ValueError(f"s1_mode must be one of: {', '.join(S1_MODES)}")
    pol = _s1_polarization(polarization) if polarization else None
    if orbit is not None:
        orbit = str(orbit).strip().lower()
        if orbit not in ("ascending", "descending"):
            raise ValueError("s1_orbit must be 'ascending' or 'descending'.")
    return mode, pol, orbit


def _s1_polarization(value: str) -> str:
    text = str(value).upper().replace(" ", "").replace("&", "+").replace(",", "+")
    text = {"VVVH": "VV+VH", "HHHV": "HH+HV"}.get(text, text)
    if text not in S1_POLARIZATIONS:
        raise ValueError("s1_polarization must be one of: " + ", ".join(S1_POLARIZATIONS))
    return text


def _s5p_code(value: str) -> str:
    if value in S5P_PRODUCTS:
        return value
    for code, (label, _asset) in S5P_PRODUCTS.items():
        if str(value).lower() == label.lower():
            return code
    raise ValueError(
        "s5p_product must be a Sentinel-5P code such as 'L2__NO2___' "
        "or a name such as 'Nitrogen dioxide'."
    )


def _platforms(value, collection: str):
    if value is None and collection == "landsat-c2-l1":
        value = LS_L1_PLATFORMS["Landsat 1–5"]
    if value is None:
        return None
    if isinstance(value, str):
        if value in LS_PLATFORM_CHOICES:
            value = LS_PLATFORM_CHOICES[value]
        elif value in LS_L1_PLATFORMS:
            value = LS_L1_PLATFORMS[value]
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value]


def _extra_query(s1_mode, s1_polarization, s1_orbit, landsat_product, s5p, collection):
    extra = {}
    if s5p and ("sentinel-5p" in collection or collection == "SENTINEL-5P"):
        extra["s5p:product_type"] = {"eq": s5p}
    if s1_mode:
        extra["sar:instrument_mode"] = {"eq": s1_mode}
    if s1_polarization:
        pols = [band.upper() for band in S1_POLARIZATIONS[s1_polarization]]
        extra["sar:polarizations"] = {"eq": pols}
    if s1_orbit:
        extra["sat:orbit_state"] = {"eq": s1_orbit}
    correction = LS_L1_CORRECTION.get(landsat_product)
    if correction and collection == "landsat-c2-l1":
        extra["landsat:correction"] = {"eq": correction}
    return extra or None


def _apply_cdse_login(user, password) -> None:
    if user:
        os.environ["CDSE_USER"] = user
    if password:
        os.environ["CDSE_PASSWORD"] = password


def _mode(mode: str, source: str, collection: str) -> str:
    if mode not in ("clip", "full"):
        raise ValueError("mode must be 'clip' or 'full'.")
    if source in ("cdse", "noaa", "usgs") or "netcdf" in collection:
        return "full"
    if collection == "naip":
        return "clip"
    return mode


def _stack_names(stack, bands: list[str], mode: str):
    if not stack or mode != "clip":
        return None
    if stack is True:
        if not bands:
            return None
        preferred = [band for band in ("red", "green", "blue", "nir") if band in bands]
        return preferred or list(bands)
    return list(stack)


def _empty_message(req: dict) -> str:
    source, collection = req["source"], req["collection"]
    if collection == "naip":
        return "No USDA NAIP imagery covers this area."
    if collection == "noaa-naip":
        return "No USDA NAIP in the NOAA coastal catalogue covers this area."
    if source == "noaa":
        return "No NOAA coastal imagery intersects this area and year range."
    if source == "usgs":
        return "No USGS 3DEP lidar meets this area and date range."
    if collection == "jrc-gsw":
        return "No JRC Global Surface Water tile covers this area."
    if collection == "rcm-ard":
        return ("No RCM-ARD scenes found. Coverage is Canada only, on a 12-day revisit -- "
                "try a wider date range or drop the orbit filter.")
    if "sentinel-1" in collection.lower():
        return (
            "No scenes found for this mode, polarization, and orbit. "
            "Try the other orbit or a wider date range."
        )
    return "No scenes found. Widen the dates or raise the cloud limit."


# --------------------------------------------------------------------------- #
# Area of interest
# --------------------------------------------------------------------------- #
def _is_bbox(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 4
        and all(isinstance(part, (int, float)) for part in value)
    )


def _bbox_tuple(bbox) -> tuple[float, float, float, float]:
    if isinstance(bbox, str):
        bbox = [part for part in bbox.replace(" ", "").split(",") if part]
    if len(bbox) != 4:
        raise ValueError("bbox must be four numbers: minlon, minlat, maxlon, maxlat.")
    try:
        values = tuple(float(part) for part in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must be four numbers: minlon, minlat, maxlon, maxlat.") from exc
    return values


def _area_from_bounds(bounds, geometry: dict | None = None) -> Area:
    minx, miny, maxx, maxy = bounds
    if not (minx < maxx and miny < maxy):
        raise ValueError("bbox must be minlon, minlat, maxlon, maxlat with a positive area.")
    if not (
        -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90
    ):
        raise ValueError(
            "AOI is not in longitude/latitude (EPSG:4326). "
            "Give the file a CRS so it can be reprojected, or pass bbox in degrees."
        )
    if geometry is None:
        geometry = {
            "type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
        }
    return Area((minx, miny, maxx, maxy), geometry)


def _from_file(path: Path) -> Area:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import geopandas as gpd
    except ImportError:
        gpd = None
    if gpd is not None:
        return _from_gdf(gpd.read_file(path))
    if path.suffix.lower() in {".json", ".geojson"}:
        return _from_geojson_obj(json.loads(path.read_text()))
    raise ImportError(
        f"Reading {path.name} needs geopandas. Install it with `pip install geolattice[rsense]`."
    )


def _from_gdf(gdf) -> Area:
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    if len(gdf) == 0:
        raise ValueError("AOI file has no features.")
    if getattr(gdf, "crs", None) is not None:
        gdf = gdf.to_crs(4326)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError("AOI file has no geometry.")
    union = unary_union(geoms)
    return _area_from_bounds(tuple(float(v) for v in union.bounds), mapping(union))


def _from_geojson_obj(data: dict) -> Area:
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    kind = data.get("type")
    if kind == "FeatureCollection":
        geoms = [shape(feature["geometry"]) for feature in data["features"]]
    elif kind == "Feature":
        geoms = [shape(data["geometry"])]
    else:
        geoms = [shape(data)]
    union = unary_union(geoms)
    return _area_from_bounds(tuple(float(v) for v in union.bounds), mapping(union))


# --------------------------------------------------------------------------- #
# Search and download
# --------------------------------------------------------------------------- #
def _run_search(req: dict):
    sd = _explorer()
    source, collection = req["source"], req["collection"]
    area, start, end = req["area"], req["start"], req["end"]
    if source == "cdse":
        return _call(
            lambda: sd.CDSE().search(
                collection,
                area,
                start,
                end,
                req["cloud"],
                req["product_type"],
                req["limit"],
                s1_mode=req["s1_mode"],
                s1_polarisation=req["s1_polarisation"],
                s1_orbit=req["s1_orbit"],
            )
        )
    if source == "noaa":
        return _call(lambda: sd.NOAA().search(collection, area, start, end, req["limit"]))
    if source == "usgs":
        return _call(lambda: sd.USGS().search(collection, area, start, end, req["limit"]))
    if source == "earth-search" and collection == "naip":
        return _call(lambda: sd.naip_search(area, req["limit"]))
    return _call(
        lambda: sd.stac_search(
            source,
            collection,
            area,
            start,
            end,
            req["cloud"],
            req["limit"],
            req["platforms"],
            req["extra"],
        )
    )


def _select(items, ids, clearest: bool, source: str):
    notes: list[str] = []
    chosen = list(items or [])
    if ids:
        wanted = {str(item_id) for item_id in ids}
        kept = [item for item in chosen if str(_item_id(item, source)) in wanted]
        seen = {str(_item_id(item, source)) for item in kept}
        missing = [item_id for item_id in ids if str(item_id) not in seen]
        if missing:
            notes.append("No scene with id: " + ", ".join(str(item_id) for item_id in missing))
        chosen = kept
    if clearest and len(chosen) > 1:
        chosen = [min(chosen, key=lambda item: _cloud_value(item, source))]
    return chosen, notes


def _item_id(item, source: str):
    if source == "cdse" and isinstance(item, dict):
        return item.get("Name") or item.get("Id")
    if isinstance(item, dict):
        return item.get("id")
    return getattr(item, "id", None)


def _cloud_value(item, source: str) -> float:
    if source == "cdse" and isinstance(item, dict):
        value = _cdse_attr(item, "cloudCover")
    elif isinstance(item, dict):
        value = (item.get("properties") or {}).get("eo:cloud_cover")
    else:
        value = (getattr(item, "properties", None) or {}).get("eo:cloud_cover")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 100.0


def _cdse_attr(product: dict, name: str):
    for attr in product.get("Attributes") or []:
        if attr.get("Name") == name:
            return attr.get("Value")
    return None


def _summarize(item, source: str) -> dict:
    if source == "cdse" and isinstance(item, dict):
        return {
            "id": item.get("Name"),
            "datetime": (item.get("ContentDate") or {}).get("Start"),
            "cloud": _cdse_attr(item, "cloudCover"),
            "platform": None,
            "title": item.get("Name"),
        }
    if isinstance(item, dict):
        props = item.get("properties") or {}
        return {
            "id": item.get("id"),
            "datetime": props.get("datetime"),
            "cloud": props.get("eo:cloud_cover"),
            "platform": props.get("platform"),
            "title": props.get("title") or item.get("id"),
            "tile": props.get("s2:mgrs_tile"),
        }
    props = getattr(item, "properties", None) or {}
    when = item.datetime.isoformat() if getattr(item, "datetime", None) else props.get("datetime")
    return {
        "id": item.id,
        "datetime": when,
        "cloud": props.get("eo:cloud_cover"),
        "platform": props.get("platform"),
        "title": props.get("title") or item.id,
        "tile": props.get("s2:mgrs_tile"),
    }


def _download(found: SearchResult, out: Path, bands, mode: str, stack, workers: int):
    sd = _explorer()
    source, collection = found.source, found.collection
    failures: list[dict] = []
    notes: list[str] = []
    if source == "cdse":
        client = sd.CDSE()
        for product in found.items:
            name = product.get("Name") or product.get("Id")
            try:
                _call(
                    lambda product=product: sd.retry(
                        lambda product=product: client.download(product, out), what=name
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": name, "error": str(exc)})
        (out / "products.json").write_text(json.dumps(found.items, indent=2))
        return failures, notes
    if source == "noaa":
        client = sd.NOAA()
        for scene in found.items:
            label = (scene.get("properties") or {}).get("title") or scene.get("id")
            try:
                _path, note = _call(
                    lambda scene=scene: sd.retry(
                        lambda scene=scene: client.download(scene, found.area, out), what=label
                    )
                )
                if note:
                    notes.append(note)
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": scene.get("id"), "error": str(exc)})
        return failures, notes
    if source == "usgs":
        client = sd.USGS()
        for scene in found.items:
            label = (scene.get("properties") or {}).get("title") or scene.get("id")
            try:
                _paths, note = _call(
                    lambda scene=scene: sd.retry(
                        lambda scene=scene: client.download(scene, found.area, out, found.limit),
                        what=label,
                    )
                )
                if note:
                    notes.append(note)
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": scene.get("id"), "error": str(exc)})
        return failures, notes
    if source == "earth-search" and collection == "naip":
        raw = _call(lambda: sd.naip_download(found.items, found.area, out))
        for item_id, error in raw or []:
            failures.append({"id": item_id, "error": error})
        return failures, notes
    raw = _call(
        lambda: sd.stac_download(
            found.items, source, collection, found.area, bands, mode, out, stack, workers
        )
    )
    for item_id, band, error in raw or []:
        failures.append({"id": item_id, "band": band, "error": error})
    return failures, notes


def _list_files(out: Path) -> list[Path]:
    return sorted(
        path for path in out.rglob("*") if path.is_file() and not path.name.endswith(".part")
    )


__all__ = [
    "Area",
    "CDSE_TYPES",
    "COLLECTION_LABELS",
    "COLLECTIONS",
    "DEFAULT_COLLECTION",
    "ImageResult",
    "LS_BANDS",
    "LS_L1_BANDS",
    "LS_L1_PLATFORMS",
    "LS_PLATFORM_CHOICES",
    "LS_PRODUCTS",
    "PLANETARY_BANDS",
    "PLANETARY_DEFAULT_BANDS",
    "RCM_ARD_BANDS",
    "RCM_ARD_BAND_LABELS",
    "RCM_ARD_DEFAULT_BANDS",
    "S1_MODES",
    "S1_POLARIZATIONS",
    "S2_BANDS",
    "S3_VARIABLES",
    "S5P_PRODUCTS",
    "SOURCES",
    "SOURCE_LABELS",
    "SearchResult",
    "bands_for",
    "default_bands",
    "get_images",
    "get_rcm_ard",
    "list_collections",
    "load_aoi",
    "naip_coverage",
    "search_images",
    "search_rcm_ard",
]
