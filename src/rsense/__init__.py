"""Remote sensing for GeoLattice (``geolattice.rsense``), on CPU or GPU.

Ported from PyRSense. Install the dependencies with ``pip install geolattice[rsense]``; optional
feature groups are ``geolattice[rsense-access]``, ``rsense-biopar``, ``rsense-sar``, ``rsense-lidar``, ``rsense-uav``,
``rsense-dl``, ``rsense-xai``, ``rsense-gee``, ``rsense-viz``, ``rsense-gpu`` and ``rsense-all``.

Quick start::

    import geolattice.rsense as rs

    rs.set_backend("auto")                         # GPU if available, else CPU
    s2 = rs.optical.load_sentinel2("aoi.geojson", ("2026-05-01", "2026-09-30"))
    ndvi = s2.rs.composite("median").rs.index("NDVI")

Submodules: ``backend``, ``core``, ``access``, ``optical``, ``atmosphere``, ``spectral_index``,
``sar``, ``insar``, ``uav``, ``lidar``, ``hyperspectral``, ``timeseries``, ``classify``, ``dl``,
``fusion``, ``xai``, ``viz``, ``scale``, ``apps``, ``get_images``, ``super_resolution``,
``texture``, ``image_processing``, ``foundation_models``, ``surface_energy_balance``,
``classify.image_classification`` (alias ``image_classification``), ``edge_detectors``. Heavy
third-party dependencies are imported only when a function needs them. ``get_images`` downloads
the same catalogues as PyGIS Explorer. ``spectral_index`` computes indices for MODIS, Landsat
8/9, Sentinel-2, PlanetScope, WorldView and NAIP. ``texture`` computes statistical, structural,
transform-based, local-pattern, model-based and deep-learning texture features for optical
and SAR imagery. ``classify.image_classification`` runs end-to-end image learning workflows:
leakage-safe spatial splits, pixel/object and supervised/unsupervised/semi-/self-supervised
classification, CNN/transformer training, segmentation, detection, instance delineation, change
detection, satellite time series, optical-SAR fusion, foundation models, SAM/CLIP and evaluation.
``foundation_models`` catalogs remote sensing foundation models (Prithvi, Clay, DOFA, CROMA,
Copernicus-FM, Panopticon, TerraMind, AnySat, SSL4EO, Satlas, DINOv3-sat, RemoteCLIP, AlphaEarth,
weather emulators and more), loads them through one ``load_model`` call, turns rasters into
embedding maps and probes, clusters or compares those embeddings. ``surface_energy_balance``
maps evapotranspiration from satellite images: the radiation balance, soil heat flux,
aerodynamic resistance, ASCE reference ET, anchor pixels, SEBAL, METRIC, SSEBop, S-SEBI, the
triangle method, PT-JPL, TSEB-PT, thermal sharpening, seasonal upscaling and validation.
"""

import importlib as _importlib

import rioxarray as _rioxarray  # noqa: F401  (registers the .rio accessor used throughout)

from . import accessor as _accessor  # noqa: F401  (registers the .rs accessor)
from . import backend, core
from ._version import __version__
from .backend import (
    autocast,
    available_backends,
    device,
    get_backend,
    info,
    set_backend,
    to_cpu,
    to_device,
)
from .core import (
    PointCloud,
    RasterCube,
    apply_tiled,
    clip,
    mosaic,
    read,
    reproject,
    write,
    zonal_stats,
)

_SUBMODULES = (
    "access",
    "optical",
    "atmosphere",
    "spectral_index",
    "sar",
    "insar",
    "uav",
    "lidar",
    "hyperspectral",
    "timeseries",
    "classify",
    "dl",
    "fusion",
    "xai",
    "viz",
    "scale",
    "apps",
    "get_images",
    "super_resolution",
    "texture",
    "image_processing",
    "edge_detectors",
    "foundation_models",
    "surface_energy_balance",
)

# moved modules kept importable under their old attribute name
_ALIASES = {"image_classification": "classify.image_classification"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        mod = _importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _ALIASES:
        mod = _importlib.import_module(f".{_ALIASES[name]}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES) | set(_ALIASES))


__all__ = [  # noqa: F822  (submodules are resolved lazily by __getattr__)
    "__version__",
    "backend",
    "core",
    "access",
    "optical",
    "atmosphere",
    "spectral_index",
    "sar",
    "insar",
    "uav",
    "lidar",
    "hyperspectral",
    "timeseries",
    "classify",
    "dl",
    "fusion",
    "xai",
    "viz",
    "scale",
    "apps",
    "get_images",
    "super_resolution",
    "texture",
    "image_processing",
    "edge_detectors",
    "foundation_models",
    "surface_energy_balance",
    "image_classification",
    "autocast",
    "available_backends",
    "device",
    "get_backend",
    "info",
    "set_backend",
    "to_cpu",
    "to_device",
    "PointCloud",
    "RasterCube",
    "apply_tiled",
    "clip",
    "mosaic",
    "read",
    "reproject",
    "write",
    "zonal_stats",
]
