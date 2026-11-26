"""RasterCube: a light, time-aware wrapper around an ``xarray.Dataset`` (time, y, x)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any

import pandas as pd
import xarray as xr


class RasterCube:
    """A georeferenced image time series with one variable per band.

    Thin by design: ``cube.ds`` is always a plain Dataset, so every xarray and
    PyRSense function still applies. The class adds construction from files,
    period selection, temporal resampling and persistence.
    """

    def __init__(self, ds: xr.Dataset, sensor: str | None = None):
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[pd.Timestamp("1970-01-01")])
        self.ds = ds.sortby("time")
        if sensor:
            self.ds.attrs["pyrsense:sensor"] = sensor

    # -- construction
    @classmethod
    def from_files(cls, paths: Sequence[str | PathLike], times: Sequence[Any],
                   band_names: Sequence[str], *, chunks: Any = None,
                   sensor: str | None = None) -> RasterCube:
        """Stack single-date multiband rasters (all on the same grid) into a cube."""
        from .raster import read

        if len(paths) != len(times):
            raise ValueError("paths and times must have equal length")
        dss = [read(p, band_names=band_names, as_dataset=True, chunks=chunks) for p in paths]
        ds = xr.concat(dss, dim=pd.Index(pd.to_datetime(list(times)), name="time"))
        return cls(ds, sensor=sensor)

    @classmethod
    def from_dataarray(cls, da: xr.DataArray, sensor: str | None = None) -> RasterCube:
        from ..optical.bands import as_dataset

        return cls(as_dataset(da), sensor=sensor)

    # -- properties
    @property
    def bands(self) -> list[str]:
        return [str(v) for v in self.ds.data_vars]

    @property
    def times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ds["time"].values)

    @property
    def crs(self):
        return self.ds.rio.crs

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ds.sizes["time"], self.ds.sizes[self.ds.rio.y_dim],
                self.ds.sizes[self.ds.rio.x_dim])

    # -- operations
    def sel_period(self, start: Any = None, end: Any = None,
                   months: Sequence[int] | None = None) -> RasterCube:
        ds = self.ds.sel(time=slice(start, end))
        if months:
            ds = ds.sel(time=ds["time.month"].isin(list(months)))
        return RasterCube(ds)

    def resample(self, freq: str = "1MS", how: str = "median") -> RasterCube:
        """Temporal aggregation, e.g. monthly medians: ``cube.resample("1MS", "median")``."""
        r = self.ds.resample(time=freq)
        return RasterCube(getattr(r, how)(skipna=True) if how != "count" else r.count())

    def apply(self, func: Callable[[xr.Dataset], xr.Dataset]) -> RasterCube:
        return RasterCube(func(self.ds))

    def valid_fraction(self) -> xr.DataArray:
        """Per-date fraction of non-NaN pixels (first band)."""
        v = self.ds[self.bands[0]]
        return v.notnull().mean([d for d in v.dims if d != "time"])

    def to_zarr(self, store: str | PathLike, **kwargs: Any) -> None:
        from .io import write_zarr

        write_zarr(self.ds, store, **kwargs)

    @classmethod
    def from_zarr(cls, store: str | PathLike, **kwargs: Any) -> RasterCube:
        from .io import read_zarr

        return cls(read_zarr(store, **kwargs))

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.ds[key]

    def __repr__(self) -> str:
        t = self.times
        span = f"{t.min():%Y-%m-%d} → {t.max():%Y-%m-%d}" if len(t) else "empty"
        return (f"<RasterCube {self.shape[0]} dates × {self.shape[1]}×{self.shape[2]} px, "
                f"bands={self.bands}, {span}, crs={self.crs}>")

    def _repr_html_(self) -> str:  # notebooks
        return f"<pre>{self!r}</pre>" + self.ds._repr_html_()


def stack_times(arrays: Sequence[xr.DataArray], times: Sequence[Any]) -> xr.DataArray:
    """Concatenate same-grid DataArrays along a new ``time`` dimension."""
    return xr.concat(list(arrays), dim=pd.Index(pd.to_datetime(list(times)), name="time"))


