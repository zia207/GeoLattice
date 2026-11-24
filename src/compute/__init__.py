"""
Compute module for GeoLattice.

This module provides CPU and GPU compute backends for spatial operations.
"""

from geolattice.compute.backend import (
    ComputeBackend,
    CPUBackend,
    GPUBackend,
    BackendManager,
    get_backend_manager,
)

__all__ = [
    "ComputeBackend",
    "CPUBackend",
    "GPUBackend",
    "BackendManager",
    "get_backend_manager",
]
