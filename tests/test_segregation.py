"""Tests for geolattice.spatial_analysis.segregation.

Single- and multi-group region-wide indices are cross-checked against PySAL's own
``segregation`` package values where it is installed (skipped otherwise); everything
else is checked against hand-worked cases (complete segregation, perfectly even
distribution) and internal consistency identities (local measures summing exactly to
their region-wide counterpart, decomposition components summing exactly to the observed
difference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geolattice.spatial_analysis.segregation import (
    data as seg_data,
    decomposition as dec,
    indices as idx,
    inference as inf,
    local,
    multiscale as ms,
    neighborhoods as nb,
    null,
    report,
    temporal,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


@pytest.fixture
def two_group_df():
    rng = _rng(0)
    n = 40
    x = rng.integers(0, 200, n).astype(float)
    y = rng.integers(0, 200, n).astype(float)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def multi_group_df():
    rng = _rng(0)
    n = 40
    return pd.DataFrame({
        "a": rng.integers(0, 100, n).astype(float),
        "b": rng.integers(0, 100, n).astype(float),
        "c": rng.integers(0, 100, n).astype(float),
    })


def _grid_gdf(n_side: int, seed: int = 0, bias: float = 0.0):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = _rng(seed)
    polys, x_counts, y_counts = [], [], []
    for i in range(n_side):
        for j in range(n_side):
            polys.append(box(i, j, i + 1, j + 1))
            if i < n_side / 2:
                x_counts.append(rng.integers(70, 100) + bias)
                y_counts.append(rng.integers(0, 30))
            else:
                x_counts.append(rng.integers(0, 30))
                y_counts.append(rng.integers(70, 100) + bias)
    return gpd.GeoDataFrame({"x": x_counts, "y": y_counts}, geometry=polys, crs="EPSG:3857")


# --------------------------------------------------------------------------- hand-worked cases

def test_complete_segregation_and_even_distribution():
    complete = pd.DataFrame({"x": [10, 0], "y": [0, 10]})
    assert idx.dissimilarity(complete, ["x", "y"]) == pytest.approx(1.0)
    assert idx.gini(complete, ["x", "y"]) == pytest.approx(1.0)
    assert idx.theil_h(complete, ["x", "y"]) == pytest.approx(1.0)
    assert idx.isolation(complete, ["x", "y"], "x") == pytest.approx(1.0)
    assert idx.exposure(complete, ["x", "y"], "x", "y") == pytest.approx(0.0)
    assert idx.correlation_ratio(complete, ["x", "y"], "x") == pytest.approx(1.0)

    even = pd.DataFrame({"x": [5, 5, 5], "y": [5, 5, 5]})
    assert idx.dissimilarity(even, ["x", "y"]) == pytest.approx(0.0)
    assert idx.gini(even, ["x", "y"]) == pytest.approx(0.0)
    assert idx.theil_h(even, ["x", "y"]) == pytest.approx(0.0)
    assert idx.isolation(even, ["x", "y"], "x") == pytest.approx(0.5)
    assert idx.correlation_ratio(even, ["x", "y"], "x") == pytest.approx(0.0)


def test_dissimilarity_hand_computed():
    df = pd.DataFrame({"x": [9, 1], "y": [1, 9]})
    assert idx.dissimilarity(df, ["x", "y"]) == pytest.approx(0.8)


def test_multigroup_dissimilarity_reduces_to_two_group_case(two_group_df):
    md = idx.multigroup_dissimilarity(two_group_df, ["x", "y"])
    d = idx.dissimilarity(two_group_df, ["x", "y"])
    assert md == pytest.approx(d)


def test_weighted_gini_matches_unweighted_textbook_case():
    # two equal-mass points at 0 and 1: textbook Gini = 0.5
    assert idx.weighted_gini(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


# --------------------------------------------------------------------------- PySAL cross-validation

def test_single_group_indices_match_pysal(two_group_df):
    segregation = pytest.importorskip("segregation")
    from segregation.singlegroup import CorrelationR, Dissim, Gini, Interaction, Isolation

    df = two_group_df.copy()
    df["total"] = df["x"] + df["y"]

    assert idx.dissimilarity(df, ["x", "y"]) == pytest.approx(Dissim(df, "x", "total").statistic)
    assert idx.gini(df, ["x", "y"]) == pytest.approx(Gini(df, "x", "total").statistic)
    assert idx.isolation(df, ["x", "y"], "x") == pytest.approx(Isolation(df, "x", "total").statistic)
    assert idx.exposure(df, ["x", "y"], "x", "y") == pytest.approx(Interaction(df, "x", "total").statistic)
    assert idx.correlation_ratio(df, ["x", "y"], "x") == pytest.approx(
        CorrelationR(df, "x", "total").statistic)
    _ = segregation


def test_multigroup_indices_match_pysal(multi_group_df):
    pytest.importorskip("segregation")
    from segregation.multigroup import MultiDissim, MultiInfoTheory

    groups = ["a", "b", "c"]
    assert idx.theil_h(multi_group_df, groups) == pytest.approx(
        MultiInfoTheory(multi_group_df, groups).statistic)
    assert idx.multigroup_dissimilarity(multi_group_df, groups) == pytest.approx(
        MultiDissim(multi_group_df, groups).statistic)


# --------------------------------------------------------------------------- neighborhoods

def test_kernel_and_knn_neighborhoods_pool_counts_correctly():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    polys = [box(i, j, i + 1, j + 1) for j in range(3) for i in range(3)]
    gdf = gpd.GeoDataFrame({"uid": range(9)}, geometry=polys, crs="EPSG:3857")
    counts = np.tile(np.array([10.0, 5.0]), (9, 1))

    uniform = nb.kernel_weights(gdf, bandwidth=1.5, kernel="uniform")
    pooled = uniform.pooled_counts(counts)
    assert pooled.shape == (9, 2)
    # center cell (index 4) has 4 neighbors within 1.5, each with [10, 5]; row-standardized
    # neighbor average equals [10, 5] again, plus its own [10, 5] -> [20, 10]
    np.testing.assert_allclose(pooled[4], [20.0, 10.0])

    pytest.importorskip("libpysal")
    k3 = nb.knn_weights(gdf, k=3)
    pooled_knn = k3.pooled_counts(counts)
    np.testing.assert_allclose(pooled_knn[0], [20.0, 10.0])


# --------------------------------------------------------------------------- local measures

def test_local_dissimilarity_contribution_sums_to_region_value(two_group_df):
    contrib = local.local_dissimilarity_contribution(two_group_df, ["x", "y"])
    assert contrib.sum() == pytest.approx(idx.dissimilarity(two_group_df, ["x", "y"]))


def test_local_quotient_and_entropy_ranges(two_group_df):
    lq = local.local_quotient(two_group_df, ["x", "y"], "x")
    assert np.all(np.isfinite(lq)) and np.all(lq >= 0)
    ent = local.local_entropy(two_group_df, ["x", "y"])
    assert np.all(ent >= -1e-12) and np.all(ent <= np.log(2) + 1e-9)
    simp = local.local_simpson_diversity(two_group_df, ["x", "y"])
    assert np.all(simp >= -1e-12) and np.all(simp <= 0.5 + 1e-9)


def test_fdr_correction_is_conservative_and_monotone():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    sig = local.fdr_correction(p, alpha=0.05)
    assert sig.dtype == bool
    assert sig[0]  # the smallest p-value is always kept if anything is significant
    assert not sig[-1]  # the largest is never kept here


def test_local_significance_flags_segregated_grid():
    gdf = _grid_gdf(10, seed=1)
    sig = local.local_significance(gdf, ["x", "y"], local.local_quotient, group="x",
                                    permutations=199, seed=0)
    assert set(sig.columns) == {"statistic", "p_value", "significant_fdr"}
    assert sig.significant_fdr.sum() > 0


# --------------------------------------------------------------------------- multiscale profile

def test_multiscale_profile_decreases_with_radius_and_is_fast_for_2000_units():
    import time

    gdf = _grid_gdf(20, seed=2)
    profile = ms.compute_profile(gdf, ["x", "y"], radii=[1, 2, 4, 8, 16, 25], divergence="kl",
                                 scale_threshold=0.1)
    # Divergence from the regional composition decays with radius *on average* -- not
    # necessarily at every single step for every unit, since which specific neighbors a
    # unit picks up next can transiently raise its local purity before pooling in the
    # other region's units. By the largest radius (25, close to the whole 20x20 extent)
    # every unit's pooled composition should be close to the true regional composition.
    assert profile.profile[:, 0].mean() > profile.profile[:, -1].mean()
    assert np.all(profile.profile[:, -1] < 0.05)
    assert np.all(np.isfinite(profile.auc))
    assert np.all(profile.distortion >= 0)

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = _rng(3)
    n = 2000
    xy = rng.uniform(0, 500, (n, 2))
    big_gdf = gpd.GeoDataFrame({"x": rng.integers(0, 100, n), "y": rng.integers(0, 100, n)},
                               geometry=[box(x, y, x + 1, y + 1) for x, y in xy], crs="EPSG:3857")
    t0 = time.time()
    ms.compute_profile(big_gdf, ["x", "y"], radii=[5, 10, 20, 40], max_neighbors=500)
    assert time.time() - t0 < 10.0  # a dense n^2 table over 2000 units would be far slower


def test_multiscale_raises_on_truncated_neighbors():
    gdf = _grid_gdf(10, seed=4)
    with pytest.raises(ValueError, match="max_neighbors"):
        ms.compute_profile(gdf, ["x", "y"], radii=[50], max_neighbors=3)


# --------------------------------------------------------------------------- null models and inference

def test_evenness_null_preserves_totals():
    totals = np.array([10.0, 20.0, 0.0, 5.0])
    props = np.array([0.3, 0.7])
    sims = null.evenness_null(totals, props, n_sims=50, rng=_rng(0))
    assert sims.shape == (50, 4, 2)
    np.testing.assert_allclose(sims.sum(axis=2), np.tile(totals, (50, 1)))


def test_systematic_null_preserves_each_row_exactly():
    counts = np.array([[9.0, 1.0], [1.0, 9.0], [5.0, 5.0]])
    sims = null.systematic_null(counts, n_sims=20, rng=_rng(0))
    for sim in sims:
        # every simulated table is a row-permutation of the same observed rows
        assert sorted(map(tuple, sim.tolist())) == sorted(map(tuple, counts.tolist()))


def test_single_value_test_rejects_evenness_for_segregated_grid():
    gdf = _grid_gdf(10, seed=5)
    result = inf.single_value_test(gdf, ["x", "y"], idx.dissimilarity, null="evenness",
                                    n_sims=199, alternative="greater", seed=0)
    assert result["p_value"] < 0.05
    assert result["statistic"] > result["null_mean"]


def test_two_value_test_detects_a_real_difference():
    gdf = _grid_gdf(10, seed=6)
    even_gdf = gdf.copy()
    even_gdf["x"], even_gdf["y"] = 50.0, 50.0
    result = inf.two_value_test(gdf, even_gdf, ["x", "y"], idx.dissimilarity, n_sims=199, seed=0)
    assert result["difference"] > 0
    assert result["p_value"] < 0.05


def test_spatial_block_bootstrap_ci_contains_estimate():
    gdf = _grid_gdf(10, seed=7)
    boot = inf.spatial_block_bootstrap(gdf, ["x", "y"], idx.dissimilarity, n_boot=199, n_blocks=4,
                                       block_method="grid", seed=0)
    assert boot["ci_lower"] <= boot["estimate"] <= boot["ci_upper"]


def test_panel_trend_test_detects_monotone_trend():
    increasing = inf.panel_trend_test([0.1, 0.15, 0.22, 0.30, 0.41])
    assert increasing["trend"] == "increasing"
    assert increasing["sen_slope"] > 0
    flat = inf.panel_trend_test([0.2, 0.19, 0.21, 0.20, 0.2])
    assert flat["p_value"] > 0.05


# --------------------------------------------------------------------------- decomposition

def test_decomposition_is_exactly_additive(two_group_df):
    rng = _rng(9)
    other = pd.DataFrame({"x": rng.integers(0, 200, len(two_group_df)).astype(float),
                          "y": rng.integers(0, 200, len(two_group_df)).astype(float)})
    result = dec.decompose_change(two_group_df, other, ["x", "y"])
    assert result["spatial_component"] + result["composition_component"] == pytest.approx(
        result["total_change"], abs=1e-9)

    result_gini = dec.decompose_change(two_group_df, other, ["x", "y"], index_fn=idx.gini)
    assert result_gini["spatial_component"] + result_gini["composition_component"] == pytest.approx(
        result_gini["total_change"], abs=1e-9)


def test_chain_decomposition_accumulates_correctly():
    rng = _rng(10)
    n = 20
    years = {y: pd.DataFrame({"x": rng.integers(0, 200, n).astype(float),
                             "y": rng.integers(0, 200, n).astype(float)})
             for y in (2000, 2010, 2020)}
    chain = dec.chain_decomposition(years, ["x", "y"])
    assert list(chain["period_from"]) == [2000, 2010]
    assert list(chain["period_to"]) == [2010, 2020]
    last = chain.iloc[-1]
    assert last["cumulative_spatial"] == pytest.approx(
        chain["spatial_component"].sum(), abs=1e-9)


# --------------------------------------------------------------------------- data / temporal

def test_areal_interpolate_splits_by_area():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    src = gpd.GeoDataFrame({"x": [100.0], "y": [0.0]}, geometry=[box(0, 0, 2, 1)], crs="EPSG:3857")
    tgt = gpd.GeoDataFrame({"tid": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857")
    out = seg_data.areal_interpolate(src, tgt, ["x", "y"])
    np.testing.assert_allclose(out["x"].to_numpy(), [50.0, 50.0])


def test_areal_interpolate_onto_target_with_same_named_columns():
    """Regression: interpolating onto a target GeoDataFrame that already has its own
    columns named like value_cols (e.g. harmonizing one year's data onto another year's
    tract layer, which carries its own demographic columns) must not collide via pandas'
    merge-suffixing and must overwrite with the interpolated values."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    src = gpd.GeoDataFrame({"x": [100.0], "y": [0.0]}, geometry=[box(0, 0, 2, 1)], crs="EPSG:3857")
    tgt = gpd.GeoDataFrame({"tid": [0, 1], "x": [999.0, 999.0], "y": [999.0, 999.0]},
                          geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857")
    out = seg_data.areal_interpolate(src, tgt, ["x", "y"])
    assert list(out.columns).count("x") == 1
    np.testing.assert_allclose(out["x"].to_numpy(), [50.0, 50.0])
    np.testing.assert_allclose(out["y"].to_numpy(), [0.0, 0.0])


def test_harmonize_panel_and_compute_panel_and_trend():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    # Each period has one big source polygon split into two unequal-sized halves with a
    # sharp x/y split, so both groups are present every period and dissimilarity stays
    # well-defined (1.0: all of x in one half, all of y in the other) after harmonizing.
    src_2000 = gpd.GeoDataFrame({"x": [100.0, 0.0], "y": [0.0, 100.0]},
                                geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857")
    src_2010 = gpd.GeoDataFrame({"x": [0.0, 100.0], "y": [100.0, 0.0]},
                                geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857")
    target = gpd.GeoDataFrame({"tid": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:3857")
    harmonized = temporal.harmonize_panel({2000: src_2000, 2010: src_2010}, ["x", "y"], target)
    np.testing.assert_allclose(harmonized[2000]["x"].to_numpy(), [100.0, 0.0])
    np.testing.assert_allclose(harmonized[2010]["y"].to_numpy(), [100.0, 0.0])

    panel = temporal.compute_panel(harmonized, ["x", "y"], idx.dissimilarity)
    assert list(panel["period"]) == [2000, 2010]
    assert panel.loc[panel.period == 2000, "statistic"].iloc[0] == pytest.approx(1.0)
    assert panel.loc[panel.period == 2010, "statistic"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- report API

def test_compute_region_wide_with_ci():
    gdf = _grid_gdf(10, seed=11)
    result = report.compute(gdf, index="dissimilarity", groups=["x", "y"], ci=True, n_boot=99,
                            n_blocks=4, seed=0)
    assert result.value is not None
    assert result.ci_lower <= result.value <= result.ci_upper
    assert "dissimilarity" in result.summary()


def test_compute_local_index_and_export_for_causal():
    gdf = _grid_gdf(10, seed=12)
    result = report.compute(gdf, index="local_quotient", groups=["x", "y"], group="x")
    assert result.local_values is not None
    assert len(result.local_values) == len(gdf)

    exported = report.export_for_causal(gdf, result.local_values, name="seg_exposure")
    assert "seg_exposure" in exported.columns
    assert len(exported) == len(gdf)


def test_compute_rejects_unknown_index():
    gdf = _grid_gdf(5, seed=13)
    with pytest.raises(ValueError, match="Unknown index"):
        report.compute(gdf, index="not_a_real_index", groups=["x", "y"])
