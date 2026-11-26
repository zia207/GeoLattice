"""MODIS Terra/Aqua: surface reflectance, vegetation index and LST products."""

from functools import partial

import xarray as xr

from ..bands import SENSOR_BANDS
from ..qa import modis_state_mask, modis_vi_mask
from ..scaling import scale as _scale
from ._loaders import load_modis as load

BANDS = SENSOR_BANDS["modis"]
# Scale factors for common products (value = DN * scale + offset)
PRODUCT_SCALES = {
    "MOD09": {"sur_refl": (1e-4, 0.0)},
    "MOD13": {"NDVI": (1e-4, 0.0), "EVI": (1e-4, 0.0)},
    "MOD11": {"LST_Day_1km": (0.02, 0.0), "LST_Night_1km": (0.02, 0.0),
              "Emis_31": (0.002, 0.49), "Emis_32": (0.002, 0.49)},
    "MOD15": {"Lai_500m": (0.1, 0.0), "Fpar_500m": (0.01, 0.0)},
    "MOD17": {"Gpp_500m": (1e-4, 0.0)},
}
SINUSOIDAL_WKT = ('PROJCS["MODIS Sinusoidal",GEOGCS["Custom",DATUM["Custom",SPHEROID["Custom",'
                  '6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                  'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
                  'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')

state_mask = modis_state_mask
vi_mask = modis_vi_mask
scale = partial(_scale, sensor="modis")


def scale_product(da: xr.DataArray, product: str, layer: str, fill: float | None = 0) -> xr.DataArray:
    """Apply a product scale factor, e.g. ``scale_product(lst, "MOD11", "LST_Day_1km")``."""
    key = next((k for k in PRODUCT_SCALES[product] if layer.startswith(k)), None)
    if key is None:
        raise KeyError(f"No scale known for {product}/{layer}")
    s, o = PRODUCT_SCALES[product][key]
    if fill is not None:
        da = da.where(da != fill)
    return (da * s + o).astype("float32")


__all__ = [
    "BANDS",
    "PRODUCT_SCALES",
    "SINUSOIDAL_WKT",
    "load",
    "scale",
    "scale_product",
    "state_mask",
    "vi_mask",
]
