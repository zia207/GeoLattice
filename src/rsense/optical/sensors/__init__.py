"""Sensor-specific loaders, constants, scale factors and QA decoding.

Each submodule (``landsat``, ``sentinel2``, ``modis``, ``hls``) exposes ``load``,
``qa_mask``, band maps and product constants. ``sentinel3`` is different: its NetCDF swath
products need their own readers and mask functions rather than a lazy STAC loader, so it
exposes those directly instead of a ``load``/``qa_mask`` pair — see its module docstring.
"""

from . import hls, landsat, modis, sentinel2, sentinel2_burn_indices, sentinel3
from ._loaders import ASSET_KEYS, COLLECTIONS, load_hls, load_landsat, load_modis, load_sentinel2

__all__ = [
           "ASSET_KEYS",
           "COLLECTIONS",
           "hls",
           "landsat",
           "load_hls",
           "load_landsat",
           "load_modis",
           "load_sentinel2",
           "modis",
           "sentinel2",
           "sentinel2_burn_indices",
           "sentinel3",
]
