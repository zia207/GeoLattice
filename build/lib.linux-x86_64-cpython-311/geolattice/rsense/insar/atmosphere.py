"""Tropospheric corrections: GACOS zenith delay maps and ERA5 via PyAPS."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .interferogram import SENTINEL1_WAVELENGTH


def read_gacos(path: str | PathLike) -> xr.DataArray:
    """Read a GACOS ``.ztd`` binary grid (with its ``.rsc`` header) as a DataArray in metres."""
    path = Path(path)
    rsc = path.with_suffix(path.suffix + ".rsc")
    if not rsc.exists():
        rsc = path.with_suffix(".rsc")
    meta = {}
    for line in rsc.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meta[parts[0]] = parts[1]
    W, H = int(meta["WIDTH"]), int(meta["FILE_LENGTH"])
    x0, y0 = float(meta["X_FIRST"]), float(meta["Y_FIRST"])
    dx, dy = float(meta["X_STEP"]), float(meta["Y_STEP"])
    data = np.fromfile(path, dtype="<f4").reshape(H, W)
    da = xr.DataArray(data, dims=("y", "x"),
                      coords={"y": y0 + dy * (np.arange(H) + 0.5),
                              "x": x0 + dx * (np.arange(W) + 0.5)}, name="ztd",
                      attrs={"units": "m"})
    return da.rio.write_crs(4326)


def gacos_correction(unwrapped_phase: xr.DataArray, ztd_reference: xr.DataArray,
                     ztd_secondary: xr.DataArray, incidence_deg: Any, *,
                     wavelength: float = SENTINEL1_WAVELENGTH,
                     reference_point: tuple[float, float] | None = None) -> xr.DataArray:
    """Subtract the GACOS tropospheric delay difference from unwrapped phase.

    The zenith delay difference is mapped to slant range by ``1/cos(inc)`` and converted to
    phase ``4 pi / lambda * delay`` (same sign convention as the interferogram,
    reference minus secondary). Delays are resampled to the interferogram grid. If
    ``reference_point`` (x, y) is given, the correction is zeroed there.
    """
    from rasterio.enums import Resampling

    dz = (ztd_secondary - ztd_reference).rio.write_crs(ztd_reference.rio.crs)
    dz = dz.rio.reproject_match(unwrapped_phase, resampling=Resampling.bilinear)
    corr = 4 * np.pi / wavelength * dz / np.cos(np.radians(incidence_deg))
    if reference_point is not None:
        corr = corr - corr.sel(x=reference_point[0], y=reference_point[1], method="nearest")
    out = unwrapped_phase - corr.values
    return out.rename("phase_corrected").assign_attrs(unwrapped_phase.attrs)


def era5_delay(date_time: str, dem: xr.DataArray, incidence_deg: Any, *,
               grib_dir: str | PathLike = "era5", wavelength: float = SENTINEL1_WAVELENGTH
               ) -> xr.DataArray:
    """Slant tropospheric delay (as phase) from ERA5 using PyAPS (``pip install pyaps3``).

    ``date_time`` like ``"2026-07-01T05:30"`` (acquisition UTC time). Requires a CDS API key.
    Returns phase (rad) on the DEM grid; difference two dates to correct an interferogram.
    """
    try:
        import pyaps3 as pa
    except ImportError as e:
        raise ImportError("ERA5 corrections need `pip install pyaps3` and a CDS API key") from e
    import pandas as pd

    from ..backend import to_cpu

    t = pd.Timestamp(date_time)
    Path(grib_dir).mkdir(parents=True, exist_ok=True)
    hour = f"{int(round(t.hour + t.minute / 60)) % 24:02d}"
    files = pa.ECMWFdload([t.strftime("%Y%m%d")], hour, str(grib_dir), model="ERA5", snwe=None)
    lat = np.broadcast_to(dem["y"].values[:, None], dem.shape)
    lon = np.broadcast_to(dem["x"].values[None, :], dem.shape)
    aps = pa.PyAPS(files[0], dem=np.asarray(to_cpu(dem).values), inc=incidence_deg, lat=lat,
                   lon=lon, grib="ERA5")
    phs = np.zeros(dem.shape, dtype="float32")
    aps.getdelay(phs)  # slant delay in metres
    return dem.copy(data=(4 * np.pi / wavelength * phs).astype("float32")).rename("era5_phase")
