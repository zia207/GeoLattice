"""Land surface temperature (LST) retrieval from thermal infrared bands."""

from __future__ import annotations

import numpy as np
import xarray as xr

RHO = 14388.0  # h*c/k_B in um*K

# Effective wavelengths (um) of common thermal bands
WAVELENGTH = {"landsat8_b10": 10.895, "landsat9_b10": 10.904, "landsat7_b6": 11.335,
              "landsat5_b6": 11.457}


def brightness_temperature(radiance: xr.DataArray, k1: float, k2: float) -> xr.DataArray:
    """At-sensor brightness temperature (K): ``BT = K2 / ln(K1 / L + 1)``."""
    return (k2 / np.log(k1 / radiance + 1)).astype("float32").assign_attrs(units="K")


def fractional_vegetation(ndvi: xr.DataArray, ndvi_soil: float = 0.2,
                          ndvi_veg: float = 0.5) -> xr.DataArray:
    """Proportion of vegetation ``Pv = ((NDVI - NDVIs) / (NDVIv - NDVIs))^2``, clipped 0-1."""
    return (((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)).clip(0, 1) ** 2).rename("pv")


def emissivity_ndvi(ndvi: xr.DataArray, *, ndvi_soil: float = 0.2, ndvi_veg: float = 0.5,
                    e_soil: float = 0.97, e_veg: float = 0.99, e_water: float = 0.991,
                    cavity: float = 0.005) -> xr.DataArray:
    """NDVI threshold method (Sobrino et al. 2004) for land surface emissivity.

    Water/soil/vegetation regimes: NDVI < 0 water, < NDVIs bare soil, > NDVIv full
    vegetation; in between a mixture with a cavity term ``4 * d * Pv * (1 - Pv)``.
    """
    pv = fractional_vegetation(ndvi, ndvi_soil, ndvi_veg)
    mixed = e_veg * pv + e_soil * (1 - pv) + 4 * cavity * pv * (1 - pv)
    e = xr.where(ndvi < 0, e_water, xr.where(ndvi < ndvi_soil, e_soil,
                                            xr.where(ndvi > ndvi_veg, e_veg + cavity, mixed)))
    return e.astype("float32").rename("emissivity")


def lst_single_channel(bt: xr.DataArray, emissivity: xr.DataArray | float, *,
                       wavelength: float = WAVELENGTH["landsat8_b10"],
                       celsius: bool = False) -> xr.DataArray:
    """Single-channel emissivity correction (Artis & Carnahan 1982):
    ``LST = BT / (1 + (lambda * BT / rho) * ln(e))``."""
    lst = bt / (1 + (wavelength * bt / RHO) * np.log(emissivity))
    if celsius:
        lst = lst - 273.15
    return lst.astype("float32").rename("lst").assign_attrs(units="degC" if celsius else "K")


def landsat_lst(thermal_dn: xr.DataArray, ndvi: xr.DataArray, *, mult: float, add: float,
                k1: float, k2: float, wavelength: float = WAVELENGTH["landsat8_b10"],
                celsius: bool = True, **emissivity_kwargs) -> xr.DataArray:
    """End-to-end LST from a Landsat Level-1 thermal band and NDVI (MTL values for M, A, K1, K2).

    For Collection 2 Level-2 products the ``ST_B10`` band is already LST; use
    :func:`geolattice.rsense.optical.scale` instead.
    """
    from .preprocess import radiance

    bt = brightness_temperature(radiance(thermal_dn, mult, add), k1, k2)
    e = emissivity_ndvi(ndvi, **emissivity_kwargs)
    return lst_single_channel(bt, e, wavelength=wavelength, celsius=celsius)


def surface_urban_heat_island(lst: xr.DataArray, urban: xr.DataArray) -> float:
    """SUHI intensity: mean LST over urban pixels minus mean over non-urban pixels."""
    u = urban.astype(bool)
    return float(lst.where(u).mean() - lst.where(~u).mean())
