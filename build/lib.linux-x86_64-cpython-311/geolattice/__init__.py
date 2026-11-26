"""
GeoLattice: Comprehensive geospatial analysis package with multi-core CPU and GPU support.

This package provides a unified interface for vector, raster, and point cloud data processing
with native support for multi-core CPU and GPU acceleration.
"""

__version__ = "0.3.49"
__author__ = "GeoLattice Contributors"
__license__ = "MIT"

# Import main API components
from geolattice.api import (
    read_vector,
    read_raster,
    read_pointcloud,
    gpu_backend,
    cpu_backend,
)

# Import configuration
from geolattice.api.config import config

# Subpackages are loaded on first attribute access, so ``import geolattice`` stays
# light while ``geolattice.raster``, ``geolattice.geostatistics``, ... still work.
_SUBMODULES = (
    "analysis",
    "api",
    "compute",
    "core",
    "crs",
    "distributed",
    "geostatistics",
    "io",
    "network",
    "pointcloud",
    "geolatticedb",
    "raster",
    "rsense",
    "spatial_analysis",
    "spatial_bd",
    "spatial_statistics",
    "spatialcausal",
    "stac",
    "utils",
    "vector",
    "viz",
    "web",
    "whitebox",
)


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"geolattice.{name}")
    raise AttributeError(f"module 'geolattice' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))


__all__ = [
    "read_vector",
    "read_raster",
    "read_pointcloud",
    "gpu_backend",
    "cpu_backend",
    "config",
]
