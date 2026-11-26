"""
Core module for GeoLattice.

This module provides base classes, data structures, and fundamental abstractions
for all spatial data types.
"""

from geolattice.core.base import SpatialData, VectorData, RasterData, PointCloudData
from geolattice.crs import (
    CRS,
    CRSCatalogEntry,
    CRSPromptRequired,
    CRSSelector,
    CRSSettings,
    CustomProjectionRegistry,
    DatumTransformationDialog,
    DatumTransformationRequired,
    Layer,
    LayoutMapGrid,
    MissingCRSBehavior,
    Project,
    SpatialReference,
    crs_settings,
    custom_projections,
)
from geolattice.core.bounds import Bounds
from geolattice.core.metadata import Metadata

__all__ = [
    "SpatialData",
    "VectorData",
    "RasterData",
    "PointCloudData",
    "CRS",
    "CRSCatalogEntry",
    "CRSPromptRequired",
    "CRSSelector",
    "CRSSettings",
    "CustomProjectionRegistry",
    "DatumTransformationDialog",
    "DatumTransformationRequired",
    "Layer",
    "LayoutMapGrid",
    "MissingCRSBehavior",
    "Project",
    "SpatialReference",
    "crs_settings",
    "custom_projections",
    "Bounds",
    "Metadata",
]
