"""
geolattice.spatial_statistics.core
=================
Core infrastructure: spatial data containers, result classes, CRS & distance.
"""

from .data import SpatialData
from .result import SpatialResult
from .crs import (
    validate_crs,
    to_crs,
    ensure_projected,
    is_geographic,
    is_projected,
)
from .distance import (
    coords_from_geometry,
    euclidean_distance_matrix,
    euclidean_distances,
    haversine_distance_matrix,
    pairwise_distances,
    normalize_coords,
)

__all__ = [
    "SpatialData",
    "SpatialResult",
    "validate_crs",
    "to_crs",
    "ensure_projected",
    "is_geographic",
    "is_projected",
    "coords_from_geometry",
    "euclidean_distance_matrix",
    "euclidean_distances",
    "haversine_distance_matrix",
    "pairwise_distances",
    "normalize_coords",
]
