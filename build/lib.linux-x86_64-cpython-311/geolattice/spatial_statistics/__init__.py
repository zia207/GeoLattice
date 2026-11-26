"""
Spatial statistics and geographically weighted modeling for GeoLattice.

Re-exports the public API so tutorials and applications can use::

    from geolattice.spatial_statistics import Queen, Moran, GWR
    import geolattice.spatial_statistics as sstat

Subpackages
-----------
core         : SpatialData, SpatialResult, CRS and distance utilities
weights      : Contiguity, distance, and graph-based spatial weights
esda         : Global and local spatial autocorrelation
viz          : Choropleth, Moran scatter, LISA maps
datasets     : Example and teaching datasets
gwmodel      : Geographically weighted models (GWR, MGWR, GW ML, GNNWR, ...)
pointpattern : Intensity, Ripley K/L/g, CSR envelopes
regression   : OLS diagnostics, LM tests, spatial lag / error / Durbin models
cluster      : Hot spots (Gi*, LISA), scan statistic, DBSCAN, regionalisation
bayes        : Empirical Bayes and BYM disease mapping
interpolate  : Deterministic surfaces, kriging family, areal / dasymetric transfer
sampling     : Random, stratified, GRTS, cLHS designs and estimators
spacetime    : Placeholder for space–time methods
"""

from __future__ import annotations

from . import (
    bayes,
    cluster,
    core,
    datasets,
    esda,
    gwmodel,
    interpolate,
    pointpattern,
    regression,
    sampling,
    spacetime,
    viz,
    weights,
)
from .core import SpatialData, SpatialResult
from .esda import Geary, GeneralG, Moran, Moran_Local
from .gwmodel import (
    ComputeConfig,
    GNNWR,
    GTNNWR,
    GTWR,
    GWGLM,
    GWPCA,
    GWR,
    GWLightGBM,
    GWRandomForest,
    GWSS,
    GWXGBoost,
    MGWR,
    NetworkGWR,
    get_compute_config,
    set_compute_options,
)
from .weights import KNN, Queen, Rook, W

__version__ = "0.3.49"
__author__ = "GeoLattice Contributors"
__license__ = "MIT"

__all__ = [
    "__version__",
    # Core
    "SpatialData",
    "SpatialResult",
    # Weights
    "W",
    "Queen",
    "Rook",
    "KNN",
    # ESDA
    "Moran",
    "Moran_Local",
    "Geary",
    "GeneralG",
    # GW models
    "GWR",
    "MGWR",
    "GWSS",
    "GWGLM",
    "GTWR",
    "GWRandomForest",
    "GWXGBoost",
    "GWLightGBM",
    "GNNWR",
    "GTNNWR",
    "GWPCA",
    "NetworkGWR",
    "set_compute_options",
    "get_compute_config",
    "ComputeConfig",
    # Subpackages
    "core",
    "weights",
    "esda",
    "viz",
    "datasets",
    "gwmodel",
    "pointpattern",
    "interpolate",
    "regression",
    "bayes",
    "spacetime",
    "sampling",
    "cluster",
]
