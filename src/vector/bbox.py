"""Axis-aligned bounds that can move between coordinate reference systems."""

from __future__ import annotations

from typing import Any


class VectorBBox:
    """Bounding box ``(minx, miny, maxx, maxy)`` with a CRS.

    ``bounds`` and ``total_bounds`` are the same four numbers. ``to_crs`` returns
    a new box in the requested CRS.
    """

    def __init__(self, bounds, crs):
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        if not (minx < maxx and miny < maxy):
            raise ValueError("bounds must be minx, miny, maxx, maxy with a positive area")
        self.bounds = (minx, miny, maxx, maxy)
        self.crs = crs

    total_bounds = property(lambda self: self.bounds)

    @classmethod
    def from_raster(cls, raster) -> "VectorBBox":
        """Box of a raster that exposes ``.rio.bounds()`` and ``.rio.crs``."""
        return cls(tuple(raster.rio.bounds()), raster.rio.crs)

    @classmethod
    def from_vector(cls, features) -> "VectorBBox":
        """Box of a GeoDataFrame or any object with ``total_bounds`` and ``crs``."""
        return cls(tuple(features.total_bounds), features.crs)

    def to_crs(self, crs) -> "VectorBBox":
        return to_crs(self, crs)

    def __iter__(self):
        return iter(self.bounds)

    def __repr__(self) -> str:
        bounds = ", ".join(f"{v:.4f}" for v in self.bounds)
        return f"VectorBBox(({bounds}), crs={self.crs!r})"


def to_crs(obj: Any, crs) -> Any:
    """Reproject a :class:`VectorBBox` or a GeoDataFrame/GeoSeries to ``crs``."""
    if isinstance(obj, VectorBBox):
        if _same_crs(obj.crs, crs):
            return VectorBBox(obj.bounds, crs)
        from pyproj import Transformer

        transformer = Transformer.from_crs(obj.crs, crs, always_xy=True)
        minx, miny, maxx, maxy = transformer.transform_bounds(*obj.bounds)
        return VectorBBox((minx, miny, maxx, maxy), crs)
    if hasattr(obj, "to_crs"):
        return obj.to_crs(crs)
    raise TypeError(f"Cannot reproject {type(obj).__name__}. Pass a VectorBBox or a GeoDataFrame.")


def _same_crs(left, right) -> bool:
    from pyproj import CRS

    try:
        return CRS.from_user_input(left) == CRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)
