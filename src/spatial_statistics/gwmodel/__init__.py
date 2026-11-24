"""
Geographically Weighted Modeling (``geolattice.spatial_statistics.gwmodel``).

Modules
-------
core           : Kernels, bandwidth selection, base classes
linear         : GWR, MGWR, GWGLM, GWSS
spatiotemporal : GTWR, Multiscale GTWR
ensemble       : GW Random Forest, GW XGBoost, GW LightGBM
deep           : GNNWR, GTNNWR (PyTorch-based neural GW models)
multivariate   : GWPCA, GW Discriminant Analysis
network        : NetworkGWR (network distance-based GWR)
diagnostics    : Collinearity, heterogeneity tests, Moran's I
visualization  : Coefficient maps, importance maps, diagnostic plots
datasets       : Bundled example datasets
utils          : Distance, stats, parallel helpers
"""

from . import (
    core,
    datasets,
    deep,
    diagnostics,
    ensemble,
    linear,
    multivariate,
    network,
    spatiotemporal,
    utils,
    visualization,
)
from .deep import GNNWR, GTNNWR
from .ensemble import GWLightGBM, GWRandomForest, GWXGBoost
from .linear import GWGLM, GWR, GWSS, MGWR
from .multivariate import GWPCA
from .network import NetworkGWR
from .spatiotemporal import GTWR
from .utils.compute import ComputeConfig, get_compute_config, set_compute_options

__version__ = "0.1.0"
__author__ = "GeoLattice Contributors"
__license__ = "MIT"

__all__ = [
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
    "core",
    "linear",
    "spatiotemporal",
    "ensemble",
    "deep",
    "multivariate",
    "network",
    "diagnostics",
    "visualization",
    "datasets",
    "utils",
]
