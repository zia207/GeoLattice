"""
Analysis module for GeoLattice.

Alias of ``geolattice.spatial_analysis`` for shorter imports, plus handles to
geostatistics and spatial statistics::

    import geolattice.analysis as analysis
    from geolattice.analysis import slope, watershed
    from geolattice.analysis import geostatistics, spatial_statistics
"""

from geolattice import geostatistics, spatial_statistics
from geolattice.spatial_analysis import *  # noqa: F403
from geolattice.spatial_analysis import __all__ as _spatial_all

__all__ = list(_spatial_all) + ["geostatistics", "spatial_statistics"]
