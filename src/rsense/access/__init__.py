"""Data discovery and access: STAC catalogs, NASA Earthdata, Copernicus CDSE, ASF, Earth Engine."""

from .asf import asf_download, asf_search, baseline_stack
from .cdse import cdse_download, cdse_search, cdse_token
from .earthdata import earthdata_download, earthdata_search
from .gee import gee_image_to_xarray, gee_init, gee_to_xarray
from .stac import CATALOGS, aoi_to_geojson, stac_load, stac_search

__all__ = [
    "CATALOGS",
    "aoi_to_geojson",
    "asf_download",
    "asf_search",
    "baseline_stack",
    "cdse_download",
    "cdse_search",
    "cdse_token",
    "earthdata_download",
    "earthdata_search",
    "gee_image_to_xarray",
    "gee_init",
    "gee_to_xarray",
    "stac_load",
    "stac_search",
]
