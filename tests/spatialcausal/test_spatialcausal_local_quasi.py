"""Tests for geolattice.spatial_analysis.spatialcausal tests (from PySpatialCausal v0.2): GW-DML, GW causal forest, geographic RDD, spatial DiD, spatial synthetic control."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, box
from geolattice.spatial_statistics.gwmodel.core.kernels import SpatialKernel

import geolattice.spatial_analysis.spatialcausal as sc
from geolattice.spatial_analysis.spatialcausal.datasets import (
    simulate_geographic_rdd,
    simulate_spatial_confounding,
    simulate_spatial_did,
    simulate_spatial_heterogeneity,
    simulate_synthetic_control,
)
from geolattice.spatial_analysis.spatialcausal.local import _kernel_rows
from geolattice.spatial_analysis.spatialcausal.quasi import _sc_weights, _twfe_demean

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------- kernels
@pytest.mark.parametrize("kernel", ["bisquare", "gaussian", "tricube", "exponential", "boxcar"])
@pytest.mark.parametrize("fixed,bw", [(False, 20), (True, 0.25)])
def test_kernel_rows_match_spatialstats(kernel, fixed, bw):
    c = np.random.default_rng(0).uniform(size=(150, 2))
    D = cdist(c, c)
    ours = _kernel_rows(D[:5], kernel, bw, fixed)
    ref = np.vstack([SpatialKernel(kernel, fixed, bw)(D[i]) for i in range(5)])
    assert np.allclose(ours, ref)


# ---------------------------------------------------------------- GW-DML
@pytest.fixture(scope="module")
def hetero():
    return simulate_spatial_heterogeneity(n=1000, random_state=21)


@pytest.fixture(scope="module")
def gwdml(hetero):
    return sc.GWDML(model_y="gbr", model_t="logistic", n_estimators=80).fit(hetero)


def test_gwdml_recovers_surface(gwdml, hetero):
    th = gwdml.local_effects_["theta"].to_numpy()
    assert np.corrcoef(th, hetero.truth["cate"])[0, 1] > 0.75
    assert abs(gwdml.effect() - hetero.truth["ate"]) < 4 * gwdml.result_["ate"].se + 0.1
    assert set(gwdml.local_effects_.columns) >= {"theta", "se", "ci_lower", "ci_upper", "p_value", "eff_n"}
    assert gwdml.bandwidth_ >= 30


def test_gwdml_stationarity_detects_variation(gwdml):
    stat, p, null = gwdml.stationarity_test(n_perm=39)
    assert p < 0.1 and stat > np.median(null)


def test_gwdml_stationarity_constant_effect():
    d = simulate_spatial_confounding(n=600, confounding_strength=0.0, random_state=22)
    g = sc.GWDML(model_y="ridge", model_t="ridge").fit(d)
    _, p, _ = g.stationarity_test(n_perm=39)
    assert p > 0.05


def test_gwdml_fixed_bandwidth_and_predict(hetero):
    g = sc.GWDML(model_y="ridge", model_t="logistic", bandwidth=0.25, fixed=True, kernel="gaussian").fit(hetero)
    assert g.bandwidth_ == 0.25
    new = np.array([[0.25, 0.25], [0.75, 0.25]])
    pred = g.predict(new)
    assert pred.shape == (2,) and pred[0] > pred[1]  # tau = 1 + sin(2πx)cos(πy)


# ---------------------------------------------------------------- GW causal forest
@pytest.fixture(scope="module")
def gwcf(hetero):
    return sc.GWCausalForest(n_estimators=80, n_estimators_local=60, n_knots=25).fit(hetero)


def test_gwcf_out_of_bag_cate(gwcf, hetero):
    assert np.corrcoef(gwcf.cate_, hetero.truth["cate"])[0, 1] > 0.5
    assert np.corrcoef(gwcf.local_ate_["ate"], hetero.truth["cate"])[0, 1] > 0.6
    assert 0.0 <= gwcf.local_weight_ <= 1.0
    assert gwcf.local_importance_.shape == (25, hetero.X.shape[1])


def test_gwcf_predict_and_ate(gwcf, hetero):
    assert gwcf.predict(hetero.X[:7], hetero.coords[:7]).shape == (7,)
    assert abs(gwcf.effect() - hetero.truth["ate"]) < 4 * gwcf.result_["ate"].se


def test_gwcf_rejects_continuous():
    with pytest.raises(ValueError):
        sc.GWCausalForest().fit(simulate_spatial_confounding(n=200))


def test_gwcf_spatialstats_engine_small():
    d = simulate_spatial_heterogeneity(n=200, random_state=23)
    f = sc.GWCausalForest(engine="spatialstats", bandwidth=60, n_estimators=40, n_estimators_local=20,
                          local_weight=0.7, n_splits=3).fit(d)
    assert np.all(np.isfinite(f.cate_)) and f.local_weight_ == 0.7


# ---------------------------------------------------------------- geographic RDD
@pytest.fixture(scope="module")
def rdd_data():
    return simulate_geographic_rdd(n=2000, effect=1.0, random_state=24)


def test_rdd_recovers_jump(rdd_data):
    r = sc.GeographicRDD(rdd_data.truth["boundary"]).fit(rdd_data)
    e = r.result_["effect_at_boundary"]
    assert abs(e.estimate - 1.0) < 4 * e.se
    naive = rdd_data.y[rdd_data.treatment == 1].mean() - rdd_data.y[rdd_data.treatment == 0].mean()
    assert abs(naive - 1.0) > abs(e.estimate - 1.0)


def test_rdd_running_variable_sign(rdd_data):
    r = sc.GeographicRDD(rdd_data.truth["boundary"])
    rv = r.running_variable(rdd_data.coords, rdd_data.treatment)
    assert np.all(rv[rdd_data.treatment == 1] >= 0) and np.all(rv[rdd_data.treatment == 0] <= 0)


def test_rdd_options_and_diagnostics(rdd_data):
    r = sc.GeographicRDD(rdd_data.truth["boundary"], n_segments=5, use_covariates=True, polynomial=2,
                         se="conley", cutoff=0.05, donut=0.005).fit(rdd_data)
    assert abs(r.effect() - 1.0) < 4 * r.result_["effect_at_boundary"].se
    bal = r.balance(names=["x0", "x1"])
    assert (bal["p_value"] > 0.001).all()
    sens = r.bandwidth_sensitivity()
    assert len(sens) == 5 and sens["n_in_bandwidth"].is_monotonic_increasing
    ab = r.along_boundary(6)
    assert len(ab) == 6 and ab["estimate"].notna().sum() >= 4


def test_rdd_heterogeneity_along_boundary():
    d = simulate_geographic_rdd(n=3000, effect=1.0, heterogeneity=0.6, random_state=25)
    ab = sc.GeographicRDD(d.truth["boundary"]).fit(d).along_boundary(8)
    truth = d.truth["effect_at"](ab["x"])
    assert np.corrcoef(ab["estimate"], truth)[0, 1] > 0.6


def test_rdd_polygon_and_array_boundary(rdd_data):
    arr = np.asarray(rdd_data.truth["boundary"].coords)
    a = sc.GeographicRDD(arr, bandwidth=0.15).fit(rdd_data).effect()
    b = sc.GeographicRDD(LineString(arr), bandwidth=0.15).fit(rdd_data).effect()
    assert np.isclose(a, b)
    poly = box(0.2, 0.2, 0.8, 0.8)
    inside = np.array([poly.contains(__import__("shapely").geometry.Point(p)) for p in rdd_data.coords]).astype(int)
    r = sc.GeographicRDD(poly, bandwidth=0.1).fit(rdd_data.X, inside, rdd_data.y, rdd_data.coords)
    assert np.isfinite(r.effect())


def test_ik_bandwidth_reasonable():
    rng = np.random.default_rng(0)
    r = rng.uniform(-1, 1, 3000)
    y = 0.5 * r + 1.0 * (r >= 0) + rng.normal(scale=0.3, size=3000)
    h = sc.ik_bandwidth(r, y)
    assert 0.05 < h < 1.5


# ---------------------------------------------------------------- panel container
def test_panel_from_long_and_wide():
    p = simulate_spatial_did(n_units=50, n_periods=4, treat_time=3, random_state=26)
    df = p.to_frame()
    q = sc.SpatialPanel.from_long(df, "unit", "time", "y", coords=("coord_x", "coord_y"),
                                  treated="treated", post="post")
    assert q.is_balanced and q.wide().shape == (50, 4)
    assert np.array_equal(q.treated, p.treated)
    assert np.allclose(q.unit_coords, p.unit_coords)


def test_twfe_demean_unbalanced():
    rng = np.random.default_rng(1)
    uid = np.repeat(np.arange(30), 5)
    tid = np.tile(np.arange(5), 30)
    keep = rng.random(150) > 0.2
    uid, tid = uid[keep], tid[keep]
    z = rng.normal(size=uid.size)
    zt = _twfe_demean(z, uid, tid)[:, 0]
    assert np.allclose(np.bincount(uid, zt), 0, atol=1e-7)
    assert np.allclose(np.bincount(tid, zt), 0, atol=1e-7)


# ---------------------------------------------------------------- spatial DiD
@pytest.fixture(scope="module")
def did_panel():
    return simulate_spatial_did(random_state=27)


def test_did_rings_recover_truth(did_panel):
    did = sc.SpatialDiD(rings=did_panel.truth["rings"] + (0.20,)).fit(did_panel)
    r = did.result_
    assert abs(r["att"].estimate - 2.0) < 4 * r["att"].se
    for k, t in zip(range(1, 4), did_panel.truth["spillovers"]):
        e = r[f"spillover_ring_{k}"]
        assert abs(e.estimate - t) < 4 * e.se
    assert r["att_naive"].estimate < r["att"].estimate  # naive is biased down by contaminated controls


def test_did_event_study_and_pretrends(did_panel):
    did = sc.SpatialDiD(rings=(0, 0.05, 0.10, 0.15), event_study=True, cluster="spatial").fit(did_panel)
    es = did.event_study_
    assert set(es["group"]) == {"treated", "ring_1", "ring_2", "ring_3"}
    assert did.pretrend_test_["p_value"] > 0.01
    post = es[(es.group == "treated") & (es.rel_time >= 0)]
    assert abs(post["estimate"].mean() - 2.0) < 0.3


def test_did_validation(did_panel):
    with pytest.raises(ValueError):
        sc.SpatialDiD(rings=(0.1, 0.2))
    with pytest.raises(ValueError):
        sc.SpatialDiD(rings=(0, 5.0)).fit(did_panel)  # no controls left


def test_did_sources(did_panel):
    src = did_panel.unit_coords[did_panel.treated == 1]
    a = sc.SpatialDiD(rings=(0, 0.05, 0.1, 0.15)).fit(did_panel).effect("spillover_ring_1")
    b = sc.SpatialDiD(rings=(0, 0.05, 0.1, 0.15), sources=src).fit(did_panel).effect("spillover_ring_1")
    assert np.isclose(a, b)


# ---------------------------------------------------------------- synthetic control
def test_sc_weights_simplex():
    rng = np.random.default_rng(0)
    Y0 = rng.normal(size=(20, 6))
    w_true = np.array([0.6, 0.4, 0, 0, 0, 0])
    w = _sc_weights(Y0 @ w_true, Y0)
    assert np.isclose(w.sum(), 1) and np.all(w >= 0)
    assert np.allclose(w, w_true, atol=1e-3)


@pytest.fixture(scope="module")
def sc_panel():
    return simulate_synthetic_control(random_state=28)


def test_sc_fit_and_placebos(sc_panel):
    m = sc.SpatialSyntheticControl(buffer=0.25).fit(sc_panel)
    assert m.att_ < -2
    assert 0 < m.p_value_ < 0.2
    assert np.isclose(m.weights_.sum(), 1)
    near_ids = set(np.flatnonzero(sc_panel.truth["spillover_by_unit"] != 0))
    assert near_ids.isdisjoint(set(m.donors_))


def test_sc_spillovers_and_buffer(sc_panel):
    m = sc.SpatialSyntheticControl(buffer=0.25, placebo=False).fit(sc_panel)
    sp = m.spillovers()
    truth = sc_panel.truth["spillover_by_unit"][sp["unit"].astype(int)]
    assert len(sp) > 3 and np.mean(sp["spillover"]) < 0 and np.corrcoef(sp["spillover"], truth)[0, 1] > 0.3
    bs = m.buffer_sensitivity([0.0, 0.1, 0.25, 0.4])
    assert bs["n_donors"].is_monotonic_decreasing


def test_sc_needs_balanced(sc_panel):
    df = sc_panel.to_frame().iloc[:-3]
    p = sc.SpatialPanel.from_long(df, "unit", "time", "y", coords=("coord_x", "coord_y"), treated="treated",
                                  treat_time=21)
    with pytest.raises(ValueError):
        sc.SpatialSyntheticControl().fit(p)


# ---------------------------------------------------------------- plots
def test_v02_plots(gwdml, hetero, rdd_data, did_panel, sc_panel):
    import matplotlib
    matplotlib.use("Agg")
    from geolattice.spatial_analysis.spatialcausal import viz
    viz.plot_local_effects(hetero.coords, gwdml.local_effects_)
    viz.plot_rdd(sc.GeographicRDD(rdd_data.truth["boundary"]).fit(rdd_data))
    viz.plot_event_study(sc.SpatialDiD(rings=(0, 0.05, 0.1, 0.15), event_study=True).fit(did_panel))
    viz.plot_synthetic_control(sc.SpatialSyntheticControl(buffer=0.25).fit(sc_panel))
