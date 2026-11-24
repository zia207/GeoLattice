"""Harmonized Landsat Sentinel-2 (HLS v2.0): L30 and S30 products."""

from ..bands import SENSOR_BANDS
from ..qa import HLS_FMASK_BITS, hls_fmask_mask
from ._loaders import load_hls as load

BANDS_L30 = SENSOR_BANDS["hls_l30"]
BANDS_S30 = SENSOR_BANDS["hls_s30"]
FMASK_BITS = HLS_FMASK_BITS
qa_mask = hls_fmask_mask

__all__ = ["BANDS_L30", "BANDS_S30", "FMASK_BITS", "load", "qa_mask"]
