"""Distributed vector and raster processing on CPU and GPU.

One API for spatial data that runs on all local cores, on a Dask cluster (local, multi-GPU
or HPC), or on CUDA GPUs. Every operation takes ``compute="auto" | "cpu" | "gpu"`` and
returns the same container type and memory space it was given; Dask input stays lazy.

>>> import geolattice.distributed as pdist
>>> with pdist.start("cuda"):                      # one worker per GPU
...     dem = pdist.read_raster("dem.tif")         # lazy, chunked
...     s = pdist.raster.slope(dem, compute="gpu")  # CuPy chunks with 1-pixel halo
...     pdist.write_raster(s, "slope.zarr")
>>> hits = pdist.vector.sjoin(points, parcels, predicate="within")

Submodules
----------
``raster``  map algebra, focal filters, terrain (CPU and GPU).
``vector``  per-feature geometry ops, clip, partition-pruned spatial join, k nearest neighbours;
            GPU kernels (CuPy, cuDF) for area, length, bounds, point-in-polygon join, knn.
``bridge``  rasterize, zonal statistics, point extraction, polygonize.
``io``      parallel GeoParquet/OGR, raster, Zarr and STAC readers and writers.
``analysis``  spatial weights, Moran's I, Geary's C, LISA, Getis-Ord G*, DBSCAN/HDBSCAN/k-means, KDE.
``network``   graphs from lines, shortest paths, centrality, components (cuGraph or SciPy/NetworkX).
``hydrology`` chunked D8 flow direction; in-memory depression filling and flow accumulation.
``sedona``    Apache Sedona (Spark) engine for spatial joins.
``ml``        model fitting, per-pixel raster prediction, multi-GPU tiled inference, DDP training.
``diagnostics`` benchmark, CPU-vs-GPU equivalence, dispatch explanations, memory, profiling.
"""

try:
    import dask  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("geolattice.distributed needs Dask: pip install 'geolattice[distributed]'") from e

from . import analysis, bridge, diagnostics, hydrology, io, ml, network, raster, sedona, vector
from ._backend import (
    capabilities,
    compute,
    get_defaults,
    is_gpu,
    is_lazy,
    persist,
    resolve_compute,
    set_defaults,
    to_cpu,
    to_gpu,
)
from ._partition import partition_bounds, plan_chunks, spatial_partition
from ._registry import dispatch, list_ops, register
from ._session import Session, dashboard_link, get_session, shutdown, start, status
from .bridge import extract_values, polygonize, rasterize, zonal_stats
from .diagnostics import benchmark, check_equivalence, explain, memory_report, profile
from .io import read_raster, read_stac, read_vector, read_zarr, write_raster, write_vector

# Submodules (raster, vector, io, ...) stay attributes but are not star-exported, so
# ``from geolattice.distributed import *`` cannot shadow names such as the stdlib ``io``.
__submodules__ = ["raster", "vector", "bridge", "analysis", "network", "hydrology", "ml", "io",
                  "sedona", "diagnostics"]
__all__ = [
    "Session",
    "benchmark",
    "capabilities",
    "check_equivalence",
    "compute",
    "dashboard_link",
    "dispatch",
    "explain",
    "extract_values",
    "get_defaults",
    "get_session",
    "is_gpu",
    "is_lazy",
    "list_ops",
    "memory_report",
    "partition_bounds",
    "persist",
    "plan_chunks",
    "polygonize",
    "profile",
    "rasterize",
    "read_raster",
    "read_stac",
    "read_vector",
    "read_zarr",
    "register",
    "resolve_compute",
    "set_defaults",
    "shutdown",
    "spatial_partition",
    "start",
    "status",
    "to_cpu",
    "to_gpu",
    "write_raster",
    "write_vector",
    "zonal_stats",
]
