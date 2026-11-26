"""Segregation analysis for GeoLattice.

A lighter, faster, time-series-aware alternative to PySAL's ``segregation`` package.
Single-group and multigroup indices are validated against PySAL's own values in
``tests/test_segregation.py``; the pairwise-distance-heavy pieces PySAL struggles with
past a few thousand units (its KL-divergence spatial profile, its unit-level bootstrap)
are rewritten here around a ``scipy.spatial.cKDTree`` and spatial-block resampling
instead of a dense ``n x n`` table.

Quick start::

    from geolattice.spatial_analysis import segregation as seg

    result = seg.compute(tracts, index="dissimilarity", groups=["white", "black"], ci=True)
    print(result.summary())

    sig = seg.local_significance(tracts, ["white", "black"], seg.local_quotient, group="black")
    seg.plot_local_map(tracts, sig.statistic, significant=sig.significant_fdr)

    profile = seg.compute_profile(tracts, ["white", "black"], radii=[500, 1000, 2000, 4000])
    tracts["segregation_scale"] = profile.segregation_scale  # ready to map or export

Submodules: ``backend`` (NumPy/CuPy switch), ``data`` (validation, areal/dasymetric
boundary harmonization), ``neighborhoods`` (kernel/KNN/network weights),
``indices`` (region-wide single- and multi-group measures), ``local`` (per-unit measures
and permutation significance), ``multiscale`` (KD-tree spatial segregation profiles),
``null`` (null-model simulators), ``inference`` (permutation tests, spatial block
bootstrap, panel trend test), ``decomposition`` (spatial-vs-composition Shapley
decomposition, chained across periods), ``temporal`` (panel workflows), ``viz`` (static
plots), ``report`` (the ``compute()`` entry point and the ``export_for_causal`` hook into
``geolattice.spatial_analysis.spatialcausal`` / PyCausalML).
"""

from . import (
    backend,
    data,
    decomposition,
    indices,
    inference,
    local,
    multiscale,
    neighborhoods,
    null,
    report,
    temporal,
    viz,
)
from .data import (
    SegregationDataError,
    areal_interpolate,
    combine_years,
    dasymetric_interpolate,
    extract_counts,
    global_proportions,
    local_proportions,
    unit_totals,
    wide_to_long,
)
from .decomposition import chain_decomposition, decompose_change
from .indices import (
    correlation_ratio,
    dissimilarity,
    exposure,
    gini,
    isolation,
    multigroup_dissimilarity,
    simpson_diversity,
    theil_h,
    weighted_gini,
)
from .inference import (
    panel_trend_test,
    single_value_test,
    spatial_block_bootstrap,
    spatial_blocks,
    two_value_test,
)
from .local import (
    fdr_correction,
    local_dissimilarity_contribution,
    local_entropy,
    local_exposure_deviation,
    local_quotient,
    local_significance,
    local_simpson_diversity,
)
from .multiscale import MultiscaleProfile, compute_profile
from .neighborhoods import (
    Neighborhood,
    distance_band_weights,
    kernel_weights,
    knn_weights,
    network_weights,
)
from .null import evenness_null, permutation_pvalue, spatially_constrained_null, systematic_null
from .report import SegregationResult, compute, export_for_causal
from .temporal import compute_panel, harmonize_panel, trend
from .viz import plot_decomposition, plot_local_map, plot_profile, plot_trend

__all__ = [
    # Submodules
    "backend", "data", "neighborhoods", "indices", "local", "multiscale", "null",
    "inference", "decomposition", "temporal", "viz", "report",
    # data
    "SegregationDataError", "extract_counts", "unit_totals", "global_proportions",
    "local_proportions", "wide_to_long", "combine_years", "areal_interpolate",
    "dasymetric_interpolate",
    # neighborhoods
    "Neighborhood", "kernel_weights", "distance_band_weights", "knn_weights", "network_weights",
    # indices
    "weighted_gini", "dissimilarity", "gini", "isolation", "exposure", "correlation_ratio",
    "theil_h", "simpson_diversity", "multigroup_dissimilarity",
    # local
    "local_quotient", "local_entropy", "local_simpson_diversity", "local_exposure_deviation",
    "local_dissimilarity_contribution", "fdr_correction", "local_significance",
    # multiscale
    "MultiscaleProfile", "compute_profile",
    # null
    "evenness_null", "systematic_null", "spatially_constrained_null", "permutation_pvalue",
    # inference
    "spatial_blocks", "single_value_test", "two_value_test", "spatial_block_bootstrap",
    "panel_trend_test",
    # decomposition
    "decompose_change", "chain_decomposition",
    # temporal
    "harmonize_panel", "compute_panel", "trend",
    # viz
    "plot_local_map", "plot_profile", "plot_decomposition", "plot_trend",
    # report
    "SegregationResult", "compute", "export_for_causal",
]
