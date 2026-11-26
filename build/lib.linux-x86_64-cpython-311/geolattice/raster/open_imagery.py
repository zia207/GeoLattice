"""Read a remote raster window and compute NDVI.

``read_raster`` here returns an xarray object (rioxarray), which is what a
Sentinel or Landsat COG needs. The GDAL dataset reader remains
``read_raster_data``. ``from geolattice import read_raster`` still returns
``RasterData``.
"""

from __future__ import annotations

from typing import Any


def read_raster(path, **kwargs) -> Any:
    """Open a GeoTIFF, COG, or raster URL as an xarray DataArray.

    A single-band file is squeezed so the array is ``y`` by ``x``.
    """
    import xarray as xr

    from geolattice.rsense.core.raster import read

    data = read(path, **kwargs)
    if isinstance(data, xr.DataArray) and data.sizes.get("band") == 1:
        data = data.squeeze("band", drop=True)
    return data


def clip_to_bbox(raster, bbox):
    """Clip ``raster`` to a :class:`~geolattice.vector.bbox.VectorBBox` or a bounds tuple.

    A box in another CRS is reprojected onto the raster CRS before the window
    is read.
    """
    from geolattice.vector.bbox import VectorBBox, to_crs

    if isinstance(bbox, VectorBBox):
        crs = getattr(getattr(raster, "rio", None), "crs", None)
        if crs is not None and bbox.crs is not None:
            bbox = to_crs(bbox, crs)
        bounds = bbox.bounds
    elif hasattr(bbox, "total_bounds"):
        bounds = tuple(bbox.total_bounds)
    else:
        bounds = tuple(bbox)
    return raster.rio.clip_box(*bounds)


def raster_ndvi(nir, red):
    """Normalized difference vegetation index, ``(nir - red) / (nir + red)``.

    Pixels where ``red`` is null stay null.
    """
    ndvi = (nir - red) / (nir + red)
    if hasattr(red, "notnull"):
        ndvi = ndvi.where(red.notnull())
    return ndvi


__all__ = ["clip_to_bbox", "raster_ndvi", "read_raster"]
