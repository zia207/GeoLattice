"""The ``.rs`` xarray accessor, for method-chaining on Datasets and DataArrays::

    ndvi = ds.rs.scale().rs.mask_clouds().rs.composite("median").rs.index("NDVI")
"""

from __future__ import annotations

from typing import Any

import xarray as xr


class _RSBase:
    def __init__(self, obj: xr.Dataset | xr.DataArray):
        self._obj = obj

    @property
    def sensor(self) -> str | None:
        """Sensor recorded in attributes, or guessed from band names."""
        from .optical.bands import guess_sensor

        return guess_sensor(self._obj)

    def with_sensor(self, sensor: str):
        """Record the sensor so later calls don't need ``sensor=``."""
        out = self._obj.copy()
        out.attrs["pyrsense:sensor"] = sensor
        return out

    # optical
    def scale(self, **kw: Any):
        from .optical import scale

        return scale(self._obj, **kw)

    def mask_clouds(self, **kw: Any):
        from .optical import mask_clouds

        return mask_clouds(self._obj, **kw)

    def cloud_mask(self, **kw: Any):
        from .optical import cloud_mask

        return cloud_mask(self._obj, **kw)

    def index(self, name, **kw: Any):
        from .optical import index

        return index(self._obj, name, **kw)

    def composite(self, method: str = "median", **kw: Any):
        from .optical import composite

        return composite(self._obj, method, **kw)

    # core
    def clip(self, geometries, **kw: Any):
        from .core import clip

        return clip(self._obj, geometries, **kw)

    def reproject(self, dst_crs, **kw: Any):
        from .core import reproject

        return reproject(self._obj, dst_crs, **kw)

    def write(self, path, **kw: Any) -> None:
        from .core import write

        write(self._obj, path, **kw)

    # backend
    def to_device(self, spec: str | None = None):
        from .backend import to_device

        return to_device(self._obj, spec)

    def to_cpu(self):
        from .backend import to_cpu

        return to_cpu(self._obj)


@xr.register_dataset_accessor("rs")
class RSDatasetAccessor(_RSBase):
    """PyRSense methods for ``xarray.Dataset``."""


@xr.register_dataarray_accessor("rs")
class RSDataArrayAccessor(_RSBase):
    """PyRSense methods for ``xarray.DataArray``."""

    def apply_tiled(self, func, **kw: Any):
        from .core import apply_tiled

        return apply_tiled(self._obj, func, **kw)

    def zonal_stats(self, zones, **kw: Any):
        from .core import zonal_stats

        return zonal_stats(self._obj, zones, **kw)
