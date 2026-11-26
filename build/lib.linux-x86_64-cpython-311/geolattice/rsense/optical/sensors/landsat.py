"""Landsat 4-9 Collection 2: band maps, scale factors, QA and thermal constants."""

from functools import partial

from ..bands import SENSOR_BANDS
from ..qa import LANDSAT_QA_BITS, landsat_qa_mask
from ..scaling import scale as _scale
from ._loaders import load_landsat as load

BANDS = SENSOR_BANDS["landsat"]
BANDS_TM = SENSOR_BANDS["landsat_tm"]
SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
# Thermal conversion constants (K1 W/(m2 sr um), K2 K) for Level-1 brightness temperature
THERMAL_K = {"landsat-8": {"B10": (774.8853, 1321.0789), "B11": (480.8883, 1201.1442)},
             "landsat-9": {"B10": (799.0284, 1329.2405), "B11": (475.6581, 1198.3494)},
             "landsat-7": {"B6": (666.09, 1282.71)}, "landsat-5": {"B6": (607.76, 1260.56)}}
qa_mask = landsat_qa_mask
scale = partial(_scale, sensor="landsat")
QA_BITS = LANDSAT_QA_BITS

__all__ = [
             "BANDS",
             "BANDS_TM",
             "QA_BITS",
             "SR_OFFSET",
             "SR_SCALE",
             "ST_OFFSET",
             "ST_SCALE",
             "THERMAL_K",
             "load",
             "qa_mask",
             "scale",
]
