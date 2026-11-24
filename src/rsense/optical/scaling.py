"""Convert stored digital numbers to physical units (reflectance, Kelvin)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from .bands import SENSOR_BANDS, guess_sensor

# Sentinel-2 processing baseline 04.00 (25 Jan 2022) added BOA_ADD_OFFSET = -1000
S2_OFFSET_START = np.datetime64("2022-01-25")


def _band_role(sensor: str, name: str) -> str | None:
    """Return the canonical band a variable name corresponds to for this sensor."""
    for canon, cands in SENSOR_BANDS[sensor].items():
        if name.lower() in (c.lower() for c in cands) or name.lower() == canon:
            return canon
    return None


def _scale_landsat(v: xr.DataArray, role: str) -> xr.DataArray:
    v = v.where(v != 0)  # C2 L2 fill value
    if role == "thermal":
        return (v * 0.00341802 + 149.0).astype("float32").assign_attrs(units="K")
    return (v * 2.75e-5 - 0.2).astype("float32").assign_attrs(units="reflectance")


def _scale_modis(v: xr.DataArray, role: str) -> xr.DataArray:
    v = v.where((v != -28672) & (v >= -100) & (v <= 16000))
    return (v * 1e-4).astype("float32").assign_attrs(units="reflectance")


def _scale_hls(v: xr.DataArray, role: str) -> xr.DataArray:
    v = v.where(v != -9999)
    if role == "thermal":  # HLS L30 thermal is stored in 0.01 degC
        return (v * 0.01 + 273.15).astype("float32").assign_attrs(units="K")
    return (v * 1e-4).astype("float32").assign_attrs(units="reflectance")


def _scale_s2(v: xr.DataArray, offset) -> xr.DataArray:
    v = v.where(v != 0)
    return ((v + offset) / 10000.0).astype("float32").assign_attrs(units="reflectance")


def _s2_offset(ds: xr.Dataset | xr.DataArray, boa_offset, harmonized: bool):
    if boa_offset is not None:
        return boa_offset
    if harmonized:
        return 0
    attr = ds.attrs.get("BOA_ADD_OFFSET")
    if attr is not None:
        return float(attr)
    if "time" in ds.coords:
        t = ds["time"].values.astype("datetime64[ns]")
        off = np.where(t >= S2_OFFSET_START, -1000.0, 0.0)
        return xr.DataArray(off, coords={"time": ds["time"]}, dims="time")
    warnings.warn(
        "Sentinel-2 offset unknown (no time coordinate or BOA_ADD_OFFSET attribute); "
        "assuming no offset. Pass boa_offset=-1000 for scenes processed with baseline >= 04.00, "
        "or harmonized=True for collections that already removed it.",
        UserWarning, stacklevel=3,
    )
    return 0


def scale(obj: xr.Dataset | xr.DataArray, *, sensor: str | None = None,
          boa_offset: float | None = None, harmonized: bool = False,
          bands: Mapping[str, str] | None = None) -> xr.Dataset | xr.DataArray:
    """Scale surface-reflectance products to reflectance (0-1) and thermal bands to Kelvin.

    * **Landsat C2 L2:** ``DN * 2.75e-5 - 0.2``; surface temperature ``DN * 0.00341802 + 149``.
    * **Sentinel-2 L2A:** ``(DN + BOA_ADD_OFFSET) / 10000``. The offset is taken from
      ``boa_offset``, then the ``BOA_ADD_OFFSET`` attribute, then inferred per scene from
      acquisition date (-1000 from 2022-01-25). Use ``harmonized=True`` for collections
      that already removed it.
    * **HLS v2.0 (L30/S30):** ``DN * 1e-4``; L30 thermal ``DN * 0.01 + 273.15`` K.
    * **MODIS MOD09:** ``DN * 1e-4`` with fill and out-of-range values masked.

    QA bands are left untouched. Fill values become NaN.
    """
    sensor = sensor or guess_sensor(obj)
    if sensor not in SENSOR_BANDS:
        raise ValueError(f"Cannot determine sensor (got {sensor!r}); pass sensor=...")
    reverse = {v: k for k, v in (bands or {}).items()}

    def one(v: xr.DataArray, name: str) -> xr.DataArray:
        role = reverse.get(name) or _band_role(sensor, name)
        if role in (None, "qa"):
            return v
        if sensor.startswith("landsat"):
            return _scale_landsat(v, role)
        if sensor == "modis":
            return _scale_modis(v, role)
        if sensor.startswith("hls"):
            return _scale_hls(v, role)
        return _scale_s2(v, _s2_offset(obj, boa_offset, harmonized))

    if isinstance(obj, xr.Dataset):
        out = xr.Dataset({n: one(obj[n], str(n)) for n in obj.data_vars}, attrs=obj.attrs)
    elif "band" in obj.dims:
        out = xr.concat([one(obj.sel(band=b), str(b)) for b in obj["band"].values], dim="band")
        out.attrs = obj.attrs
    else:
        out = one(obj, str(obj.name))
    out.attrs["pyrsense:sensor"] = sensor
    out.attrs["pyrsense:scaled"] = True
    return out
