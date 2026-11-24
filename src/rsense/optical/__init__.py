"""Optical remote sensing: MODIS, Landsat, Sentinel-2, Sentinel-3 and HLS.

Sentinel-3 (OLCI, SLSTR, SRAL, Synergy) has its own reader and mask functions under
``sensors.sentinel3``, since its NetCDF swath products do not fit the lazy STAC-raster
loaders the other sensors share. ``sentinel2_biophysical`` estimates LAI, FAPAR,
FCOVER, CCC and CWC through the Copernicus Data Space BioPAR openEO service.
"""

from . import lst, preprocess, sensors, sentinel2_biophysical
from .bands import SENSOR_BANDS, SENSORS, as_dataset, get_band, guess_sensor
from .composite import composite
from .indices import ALIASES, INDICES, index, list_indices, register_index
from .lst import (
                  brightness_temperature,
                  emissivity_ndvi,
                  fractional_vegetation,
                  landsat_lst,
                  lst_single_channel,
)
from .preprocess import (
                  coregister,
                  dos1,
                  pansharpen,
                  parse_mtl,
                  radiance,
                  toa_reflectance,
                  topographic_correction,
)
from .qa import (
                  cloud_mask,
                  decode_bits,
                  hls_fmask_mask,
                  landsat_qa_mask,
                  mask_clouds,
                  modis_state_mask,
                  modis_vi_mask,
                  s2_scl_mask,
)
from .scaling import scale
from .sensors import load_hls, load_landsat, load_modis, load_sentinel2
from .unmix import endmembers_from_samples, unmix

__all__ = [
                  "ALIASES",
                  "INDICES",
                  "SENSORS",
                  "SENSOR_BANDS",
                  "as_dataset",
                  "brightness_temperature",
                  "cloud_mask",
                  "composite",
                  "coregister",
                  "decode_bits",
                  "dos1",
                  "emissivity_ndvi",
                  "endmembers_from_samples",
                  "fractional_vegetation",
                  "get_band",
                  "guess_sensor",
                  "hls_fmask_mask",
                  "index",
                  "landsat_lst",
                  "landsat_qa_mask",
                  "list_indices",
                  "load_hls",
                  "load_landsat",
                  "load_modis",
                  "load_sentinel2",
                  "lst",
                  "lst_single_channel",
                  "mask_clouds",
                  "modis_state_mask",
                  "modis_vi_mask",
                  "pansharpen",
                  "parse_mtl",
                  "preprocess",
                  "radiance",
                  "register_index",
                  "s2_scl_mask",
                  "scale",
                  "sensors",
                  "sentinel2_biophysical",
                  "toa_reflectance",
                  "topographic_correction",
                  "unmix",
]
