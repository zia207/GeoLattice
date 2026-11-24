"""MintPy ``smallbaselineApp.py`` wrapper for InSAR time-series analysis."""

from __future__ import annotations

import shutil
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any


def write_config(workdir: str | PathLike, *, processor: str = "hyp3",
                 unw_files: str = "*/*unw_phase.tif", cor_files: str = "*/*corr.tif",
                 dem_file: str = "*/*dem.tif", inc_angle_file: str = "*/*lv_theta.tif",
                 az_angle_file: str = "*/*lv_phi.tif", water_mask_file: str | None = None,
                 reference_lalo: tuple[float, float] | None = None,
                 extra: dict[str, Any] | None = None, name: str = "smallbaselineApp.cfg") -> Path:
    """Write a MintPy config for interferograms from HyP3 (default), ISCE2 or others."""
    lines = [
        f"mintpy.load.processor = {processor}",
        f"mintpy.load.unwFile = {unw_files}",
        f"mintpy.load.corFile = {cor_files}",
        f"mintpy.load.demFile = {dem_file}",
        f"mintpy.load.incAngleFile = {inc_angle_file}",
        f"mintpy.load.azAngleFile = {az_angle_file}",
    ]
    if water_mask_file:
        lines.append(f"mintpy.load.waterMaskFile = {water_mask_file}")
    if reference_lalo:
        lines.append(f"mintpy.reference.lalo = {reference_lalo[0]},{reference_lalo[1]}")
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {v}")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    path = wd / name
    path.write_text("\n".join(lines) + "\n")
    return path


def run(config: str | PathLike, *, dostep: str | None = None) -> None:
    """Run ``smallbaselineApp.py`` in the config's directory."""
    exe = shutil.which("smallbaselineApp.py")
    if exe is None:
        raise RuntimeError("smallbaselineApp.py not found; `pip install mintpy`")
    cmd = [exe, Path(config).name] + (["--dostep", dostep] if dostep else [])
    subprocess.run(cmd, cwd=Path(config).parent, check=True)


def read_velocity(workdir: str | PathLike, name: str = "velocity.h5"):
    """Read MintPy's velocity (m/yr) as a DataArray (requires h5py)."""
    import h5py
    import numpy as np
    import xarray as xr

    with h5py.File(Path(workdir) / name, "r") as f:
        vel = f["velocity"][:]
        a = dict(f.attrs)
    H, W = vel.shape

    def num(k):
        v = a.get(k)
        return float(v.decode() if isinstance(v, bytes) else v) if v is not None else None

    x0, y0, dx, dy = num("X_FIRST"), num("Y_FIRST"), num("X_STEP"), num("Y_STEP")
    coords = ({"y": y0 + dy * (np.arange(H) + 0.5), "x": x0 + dx * (np.arange(W) + 0.5)}
              if x0 is not None else {})
    da = xr.DataArray(vel, dims=("y", "x"), coords=coords, name="velocity", attrs={"units": "m/yr"})
    epsg = a.get("EPSG")
    if epsg is not None:
        da = da.rio.write_crs(int(epsg.decode() if isinstance(epsg, bytes) else epsg))
    elif x0 is not None and abs(x0) <= 180:
        da = da.rio.write_crs(4326)
    return da


def read_timeseries(workdir: str | PathLike, name: str = "timeseries.h5"):
    """Read MintPy's displacement time series (m) as a DataArray (time, y, x)."""
    import h5py
    import pandas as pd
    import xarray as xr

    with h5py.File(Path(workdir) / name, "r") as f:
        ts = f["timeseries"][:]
        dates = pd.to_datetime([d.decode() for d in f["date"][:]])
    return xr.DataArray(ts, dims=("time", "y", "x"), coords={"time": dates}, name="displacement",
                        attrs={"units": "m"})
