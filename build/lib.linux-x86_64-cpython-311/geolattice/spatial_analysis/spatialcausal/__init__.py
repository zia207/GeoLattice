"""Causal inference for spatially dependent data (ported from PySpatialCausal 0.3.1).

Interference and spillover estimators, spatial-confounding adjustment,
spatial double machine learning, local (geographically weighted) effects,
quasi-experiments (geographic RDD, spatial DiD, synthetic control),
Bayesian spatial causal models, experimental design and spatial causal
discovery. Built on :mod:`geolattice.spatial_statistics` (spatial weights, ESDA,
GW kernels, ICAR / BYM); PyCausalML estimators are reachable through
:class:`PyCausalMLAdapter` (``pip install geolattice[spatialcausal-causalml]``).

Usage::

    import geolattice.spatial_analysis.spatialcausal as sc
    data = sc.datasets.simulate_spatial_confounding(n=400, random_state=0)
"""
from geolattice.spatial_analysis.spatialcausal._version import __version__
from geolattice.spatial_analysis.spatialcausal.core import EffectEstimate, SpatialCausalData, SpatialCausalResult, SpatialPanel
from geolattice.spatial_analysis.spatialcausal.crossfit import SpatialBlockKFold, make_folds
from geolattice.spatial_analysis.spatialcausal.exposure import (
    BernoulliDesign,
    BipartiteExposure,
    ClusterDesign,
    CompleteDesign,
    DistanceDecayExposure,
    ExposureMapping,
    exposure_probabilities,
    kernel_weights,
)
from geolattice.spatial_analysis.spatialcausal.interference import InterferenceEstimator
from geolattice.spatial_analysis.spatialcausal.confounding import (
    SpatialBasis,
    SpatialPlus,
    compare_adjustments,
    diagnose_spatial_confounding,
    moran_eigenvectors,
    rbf_basis,
)
from geolattice.spatial_analysis.spatialcausal.ml import PyCausalMLAdapter, SpatialDML, SpatialDRLearner, spatial_features
from geolattice.spatial_analysis.spatialcausal.inference import (
    block_permutation_test,
    conley_variance,
    spatial_block_bootstrap,
    standard_error,
    suggest_cutoff,
)
from geolattice.spatial_analysis.spatialcausal.local import GWCausalForest, GWDML
from geolattice.spatial_analysis.spatialcausal.quasi import GeographicRDD, SpatialDiD, SpatialSyntheticControl, ik_bandwidth
from geolattice.spatial_analysis.spatialcausal.bayes import BayesianSpatialCausal
from geolattice.spatial_analysis.spatialcausal.spacetime import StaggeredSpatialDiD
from geolattice.spatial_analysis.spatialcausal.design import (
    ClusterRandomizedDesign,
    SaturationDesign,
    balanced_allocation,
    design_diagnostics,
    power_analysis,
    power_curve,
    rerandomize,
)
from geolattice.spatial_analysis.spatialcausal.discovery import GCCM, SpatialGranger, SpatialNOTEARS, notears_linear
from geolattice.spatial_analysis.spatialcausal import datasets

__all__ = [
    "__version__", "SpatialCausalData", "SpatialPanel", "SpatialCausalResult", "EffectEstimate",
    "SpatialBlockKFold", "make_folds",
    "ExposureMapping", "DistanceDecayExposure", "BipartiteExposure", "kernel_weights",
    "BernoulliDesign", "CompleteDesign", "ClusterDesign", "exposure_probabilities",
    "InterferenceEstimator",
    "SpatialBasis", "SpatialPlus", "compare_adjustments", "diagnose_spatial_confounding",
    "moran_eigenvectors", "rbf_basis",
    "SpatialDML", "SpatialDRLearner", "PyCausalMLAdapter", "spatial_features",
    "standard_error", "conley_variance", "spatial_block_bootstrap", "block_permutation_test",
    "suggest_cutoff", "datasets",
    "GWDML", "GWCausalForest",
    "GeographicRDD", "SpatialDiD", "SpatialSyntheticControl", "ik_bandwidth",
    "BayesianSpatialCausal", "StaggeredSpatialDiD",
    "ClusterRandomizedDesign", "SaturationDesign", "balanced_allocation", "rerandomize",
    "design_diagnostics", "power_analysis", "power_curve",
    "SpatialGranger", "GCCM", "SpatialNOTEARS", "notears_linear",
]
