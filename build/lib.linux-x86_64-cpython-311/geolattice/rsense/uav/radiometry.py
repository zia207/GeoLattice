"""Radiometric calibration of multispectral UAV imagery with reflectance panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from ..backend import to_cpu


def panel_calibration(obj: xr.Dataset, panel_dn: Mapping[str, float],
                      panel_reflectance: Mapping[str, float]) -> xr.Dataset:
    """Single-panel calibration: ``reflectance = DN * (rho_panel / DN_panel)`` per band.

    ``panel_dn``: mean DN of the panel per band (measured in a calibration image);
    ``panel_reflectance``: the panel's certified reflectance per band (0-1).
    """
    out = {}
    for b in obj.data_vars:
        if b in panel_dn:
            out[b] = (obj[b].astype("float32") * (panel_reflectance[b] / panel_dn[b])) \
                .assign_attrs(units="reflectance")
        else:
            out[b] = obj[b]
    return xr.Dataset(out, attrs=obj.attrs)


def empirical_line(obj: xr.Dataset, targets_dn: Mapping[str, Sequence[float]],
                   targets_reflectance: Mapping[str, Sequence[float]]) -> tuple[xr.Dataset, dict]:
    """Empirical line method with 2+ targets of known reflectance (linear fit per band).

    Returns the calibrated Dataset and ``{band: (gain, offset, r2)}``.
    """
    out, fits = {}, {}
    for b in obj.data_vars:
        if b not in targets_dn:
            out[b] = obj[b]
            continue
        x = np.asarray(targets_dn[b], float)
        y = np.asarray(targets_reflectance[b], float)
        if x.size < 2:
            raise ValueError(f"Band {b}: need at least two targets")
        gain, offset = np.polyfit(x, y, 1)
        pred = gain * x + offset
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        fits[b] = (float(gain), float(offset), float(r2))
        out[b] = (obj[b].astype("float32") * gain + offset).assign_attrs(units="reflectance")
    return xr.Dataset(out, attrs=obj.attrs), fits


def panel_dn_from_image(img: xr.Dataset, box: tuple[int, int, int, int]) -> dict[str, float]:
    """Mean DN per band inside a pixel box (row0, row1, col0, col1) covering the panel."""
    r0, r1, c0, c1 = box
    return {b: float(np.nanmean(to_cpu(img[b].values[..., r0:r1, c0:c1]))) for b in img.data_vars}


def sunshine_normalize(obj: xr.Dataset, irradiance: Mapping[str, float],
                       reference: Mapping[str, float]) -> xr.Dataset:
    """Normalize for changing illumination using a downwelling light sensor (DLS):
    ``DN * (E_ref / E_image)`` per band."""
    return xr.Dataset({b: obj[b] * (reference[b] / irradiance[b]) if b in irradiance else obj[b]
                       for b in obj.data_vars}, attrs=obj.attrs)
