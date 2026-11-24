"""Thermal UAV analytics: canopy temperature and the Crop Water Stress Index (CWSI)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu


def cwsi(canopy_temp: xr.DataArray, *, method: Literal["statistical", "reference", "empirical"]
         = "statistical", t_wet: float | None = None, t_dry: float | None = None,
         vegetation: xr.DataArray | None = None, percentiles: tuple[float, float] = (5, 95),
         air_temp: float | None = None, vpd: float | None = None,
         nwsb: tuple[float, float] = (3.38, -3.25), dry_offset: float = 5.0) -> xr.DataArray:
    """Crop Water Stress Index ``(Tc - Twet) / (Tdry - Twet)``, clipped 0-1.

    Methods
    -------
    statistical  Twet/Tdry from low/high ``percentiles`` of canopy pixels (Rud et al. 2014 style)
    reference    measured wet/dry reference temperatures (``t_wet``, ``t_dry``)
    empirical    Idso et al. (1981): non-water-stressed baseline ``Tc - Ta = a + b*VPD``
                 (``nwsb`` = (a, b) in °C and °C/kPa; ``b`` is negative because a transpiring
                 canopy cools further below air as the air dries). The default (3.38, -3.25)
                 is Idso's (1982) baseline for wheat before heading; use one for your crop.
                 Tdry = Ta + ``dry_offset``
    ``vegetation`` (bool) restricts statistics to canopy pixels (exclude soil).
    """
    tc = canopy_temp.astype("float32")
    if vegetation is not None:
        tc = tc.where(vegetation.astype(bool))
    if method == "statistical":
        a = np.asarray(to_cpu(tc).values, dtype="float64")
        t_wet, t_dry = np.nanpercentile(a, percentiles)
    elif method == "reference":
        if t_wet is None or t_dry is None:
            raise ValueError("method='reference' needs t_wet and t_dry")
    elif method == "empirical":
        if air_temp is None or vpd is None:
            raise ValueError("method='empirical' needs air_temp and vpd (kPa)")
        a_, b_ = nwsb
        t_wet = air_temp + a_ + b_ * vpd
        t_dry = air_temp + dry_offset
        if t_dry <= t_wet:
            raise ValueError(f"non-stressed baseline gives Twet {t_wet:.1f} >= Tdry {t_dry:.1f}; "
                             "check nwsb (slope b should be negative) and dry_offset")
    else:
        raise ValueError(f"Unknown method {method!r}")
    out = ((tc - t_wet) / (t_dry - t_wet)).clip(0, 1)
    return out.rename("cwsi").assign_attrs(t_wet=float(t_wet), t_dry=float(t_dry), method=method)


def raw_to_celsius(raw: xr.DataArray, *, gain: float = 0.04, offset: float = -273.15) -> xr.DataArray:
    """Radiometric thermal camera raw counts to °C (FLIR/DJI R-JPEG style ``raw*gain + offset``)."""
    return (raw.astype("float32") * gain + offset).rename("temperature").assign_attrs(units="degC")
