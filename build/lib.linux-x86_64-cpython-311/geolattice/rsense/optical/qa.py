"""Quality-band decoding and cloud masking.

All mask functions return a boolean DataArray where ``True`` means the pixel
is *usable*. :func:`mask_clouds` applies the right mask for a sensor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import xarray as xr

from .bands import get_band, guess_sensor, resolve_band_name


def _as_uint(qa: xr.DataArray) -> xr.DataArray:
    """QA bands read with ``masked=True`` arrive as float with NaN; restore integers."""
    if np.issubdtype(qa.dtype, np.integer):
        return qa
    return qa.fillna(0).astype("uint32")


def decode_bits(qa: xr.DataArray, start: int, length: int = 1) -> xr.DataArray:
    """Extract an integer field of ``length`` bits starting at bit ``start``."""
    return (_as_uint(qa) >> start) & ((1 << length) - 1)


# ----------------------------------------------------------------------------- Landsat
LANDSAT_QA_BITS = {"fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3, "shadow": 4,
                   "snow": 5, "clear": 6, "water": 7}


def landsat_qa_mask(qa: xr.DataArray, *, cloud: bool = True, dilated_cloud: bool = True,
                    shadow: bool = True, cirrus: bool = True, snow: bool = False,
                    water: bool = False) -> xr.DataArray:
    """Mask from Landsat Collection 2 ``QA_PIXEL``. Flags set to ``True`` are excluded.

    Fill pixels are always excluded.
    """
    flags = {"fill": True, "cloud": cloud, "dilated_cloud": dilated_cloud, "shadow": shadow,
             "cirrus": cirrus, "snow": snow, "water": water}
    bad = sum((1 << LANDSAT_QA_BITS[k]) for k, on in flags.items() if on)
    valid = (_as_uint(qa) & bad) == 0
    return valid.where(qa.notnull(), False).astype(bool).rename("valid")


# ----------------------------------------------------------------------------- Sentinel-2
S2_SCL_CLASSES = {
    0: "no_data", 1: "saturated_defective", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation",
    5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium_probability",
    9: "cloud_high_probability", 10: "thin_cirrus", 11: "snow_ice",
}
S2_SCL_DEFAULT_EXCLUDE = (0, 1, 3, 8, 9, 10)


def s2_scl_mask(scl: xr.DataArray, *, exclude: Iterable[int] = S2_SCL_DEFAULT_EXCLUDE,
                snow: bool = False, dark: bool = False) -> xr.DataArray:
    """Mask from the Sentinel-2 L2A Scene Classification Layer.

    By default excludes no-data, saturated, cloud shadow, medium/high cloud and cirrus.
    Set ``snow=True`` / ``dark=True`` to also exclude snow (11) or dark areas (2).
    """
    classes = set(exclude) | ({11} if snow else set()) | ({2} if dark else set())
    s = _as_uint(scl)
    return (~s.isin(sorted(classes))).where(scl.notnull(), False).astype(bool).rename("valid")


# ----------------------------------------------------------------------------- MODIS
def modis_state_mask(state: xr.DataArray, *, cloud: bool = True, shadow: bool = True,
                     cirrus: bool = True, snow: bool = False, adjacent: bool = False,
                     allow_unset: bool = True) -> xr.DataArray:
    """Mask from MOD09 ``state_1km`` / ``sur_refl_state_500m``.

    Bits: 0-1 cloud state (0 clear, 1 cloudy, 2 mixed, 3 not set), 2 cloud shadow,
    8-9 cirrus, 10 internal cloud, 13 adjacent to cloud, 15 internal snow.
    """
    ok = xr.ones_like(state, dtype=bool)
    if cloud:
        cs = decode_bits(state, 0, 2)
        ok &= (cs == 0) | ((cs == 3) if allow_unset else False)
        ok &= decode_bits(state, 10) == 0
    if shadow:
        ok &= decode_bits(state, 2) == 0
    if cirrus:
        ok &= decode_bits(state, 8, 2) == 0
    if adjacent:
        ok &= decode_bits(state, 13) == 0
    if snow:
        ok &= decode_bits(state, 15) == 0
    return ok.where(state.notnull(), False).astype(bool).rename("valid")


def modis_vi_mask(vi_quality: xr.DataArray, *, max_level: int = 1) -> xr.DataArray:
    """Mask from MOD13 ``VI Quality`` bits 0-1 (0 good, 1 check QA, 2 cloudy, 3 not produced)."""
    return (decode_bits(vi_quality, 0, 2) <= max_level).where(vi_quality.notnull(), False) \
        .astype(bool).rename("valid")


# ----------------------------------------------------------------------------- HLS
HLS_FMASK_BITS = {"cirrus": 0, "cloud": 1, "adjacent": 2, "shadow": 3, "snow": 4, "water": 5}


def hls_fmask_mask(fmask: xr.DataArray, *, cloud: bool = True, adjacent: bool = True,
                   shadow: bool = True, cirrus: bool = False, snow: bool = False,
                   water: bool = False, max_aerosol: int = 3) -> xr.DataArray:
    """Mask from the HLS v2.0 ``Fmask`` layer (bits 0-5 flags, bits 6-7 aerosol level 0-3)."""
    flags = {"cloud": cloud, "adjacent": adjacent, "shadow": shadow, "cirrus": cirrus,
             "snow": snow, "water": water}
    bad = sum((1 << HLS_FMASK_BITS[k]) for k, on in flags.items() if on)
    f = _as_uint(fmask)
    valid = ((f & bad) == 0) & (decode_bits(fmask, 6, 2) <= max_aerosol) & (f != 255)
    return valid.where(fmask.notnull(), False).astype(bool).rename("valid")


# ----------------------------------------------------------------------------- dispatcher
_MASKERS = {"landsat": landsat_qa_mask, "landsat_tm": landsat_qa_mask,
            "sentinel2": s2_scl_mask, "modis": modis_state_mask,
            "hls_l30": hls_fmask_mask, "hls_s30": hls_fmask_mask}


def cloud_mask(obj: xr.Dataset | xr.DataArray, *, sensor: str | None = None,
               bands: Mapping[str, str] | None = None, **kwargs) -> xr.DataArray:
    """Return the validity mask for ``obj`` using its sensor's QA band."""
    sensor = sensor or guess_sensor(obj)
    if sensor not in _MASKERS:
        raise ValueError(f"Cannot determine sensor (got {sensor!r}); pass sensor=...")
    qa = get_band(obj, "qa", sensor=sensor, bands=bands)
    return _MASKERS[sensor](qa, **kwargs)


def mask_clouds(obj: xr.Dataset | xr.DataArray, *, sensor: str | None = None,
                bands: Mapping[str, str] | None = None, drop_qa: bool = False,
                **kwargs) -> xr.Dataset | xr.DataArray:
    """Set cloudy/invalid pixels to NaN in every band.

    ``kwargs`` pass through to the sensor's mask function
    (e.g. ``snow=True`` or ``exclude=[...]``). The QA band itself is kept unmasked
    unless ``drop_qa=True`` removes it.
    """
    sensor = sensor or guess_sensor(obj)
    valid = cloud_mask(obj, sensor=sensor, bands=bands, **kwargs)
    qa_name = resolve_band_name(obj, "qa", sensor=sensor, bands=bands)
    if isinstance(obj, xr.Dataset):
        out = obj.copy()
        for v in out.data_vars:
            if v != qa_name:
                out[v] = out[v].where(valid)
        if drop_qa:
            out = out.drop_vars(qa_name)
        return out
    out = obj.where(valid | (obj["band"] == qa_name))
    return out.drop_sel(band=qa_name) if drop_qa else out
