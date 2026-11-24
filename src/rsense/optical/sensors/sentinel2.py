"""Sentinel-2 MSI L2A: band maps, scale factors (incl. BOA offset) and SCL masking."""

from functools import partial

from ..bands import SENSOR_BANDS
from ..qa import S2_SCL_CLASSES, s2_scl_mask
from ..scaling import S2_OFFSET_START
from ..scaling import scale as _scale
from ._loaders import load_sentinel2 as load

BANDS = SENSOR_BANDS["sentinel2"]
RESOLUTION = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20, "B06": 20, "B07": 20,
              "B8A": 20, "B11": 20, "B12": 20, "B01": 60, "B09": 60, "SCL": 20}
QUANTIFICATION = 10000
BOA_ADD_OFFSET = -1000
qa_mask = s2_scl_mask
scale = partial(_scale, sensor="sentinel2")
SCL_CLASSES = S2_SCL_CLASSES

__all__ = [
              "BANDS",
              "BOA_ADD_OFFSET",
              "QUANTIFICATION",
              "RESOLUTION",
              "S2_OFFSET_START",
              "SCL_CLASSES",
              "load",
              "qa_mask",
              "scale",
]
