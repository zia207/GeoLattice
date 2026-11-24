"""geolattice.spatial_analysis.spatialcausal tests (from PySpatialCausal v0.3): Bayesian ICAR/BYM causal models, staggered spatial DiD, design, discovery."""
import warnings

import numpy as np
import pytest
from geolattice.spatial_statistics.weights import KNN

import geolattice.spatial_analysis.spatialcausal as sc
from geolattice.spatial_analysis.spatialcausal.bayes import _icar_eigen
from geolattice.spatial_analysis.spatialcausal.datasets import (
    simulate_areal_causal,
    simulate_gccm,
    simulate_spatial_dag,
    simulate_spatial_granger,
    simulate_staggered_spillover,
)
from geolattice.spatial_analysis.spatialcausal.discovery import spatial_lag_rings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------- Bayesian
def test_icar_eigen_null_space():
    d = simulate_areal_causal(nrow=6, ncol=6)
    V, lam, n_null = _icar_eigen(d.w, d.n)
    assert n_null == 1 and np.all(lam > 0)
    assert np.allclose(V.sum(0), 0, atol=1e-8)  # constrained field sums to zero


@pytest.fixture(scope="module")
def areal():
    return simulate_areal_causal(effect=0.5, spillover=0.4, random_state=1)


def test_bayes_gaussian_recovers_effects(areal):
    m = sc.BayesianSpatialCausal(spillover=True, draws=800, tune=400).fit(areal)
    for k in ("direct", "spillover"):
        lo, hi = m.result_._stats["credible_95"][k]
        assert lo <= areal.truth[k] <= hi
    assert m.diagnostics_.loc["tau", "rhat"] < 1.05 and m.diagnostics_.loc["tau", "ess"] > 400
    assert m.diagnostics_.loc["sigma", "rhat"] < 1.1
    assert np.corrcoef(m.field_, areal.truth["field"])[0, 1] > 0.6
    assert 0 <= m.probability("direct") <= 1 and m.effect_draws("total").size == 2 * 800


def test_bayes_spatial_confounding_adjustments():
    d = simulate_areal_causal(effect=0.5, confounding=1.5, random_state=2)
    naive = sc.BayesianSpatialCausal(spatial="none", draws=400, tune=200).fit(d).effect()
    icar = sc.BayesianSpatialCausal(draws=400, tune=200).fit(d).effect()
    splus = sc.BayesianSpatialCausal(adjust="spatial+", draws=400, tune=200).fit(d).effect()
    assert abs(naive - 0.5) > 0.4
    assert abs(icar - 0.5) < abs(naive - 0.5) and abs(splus - 0.5) < abs(naive - 0.5)


def test_bayes_poisson_uses_spatialstats_bym():
    d = simulate_areal_causal(nrow=10, ncol=10, family="poisson", effect=0.3, random_state=3)
    m = sc.BayesianSpatialCausal(family="poisson", draws=300, tune=300, thin=5).fit(
        d, expected=d.truth["expected"])
    assert hasattr(m, "bym_")
    assert abs(m.effect() - 0.3) < 4 * m.result_["direct"].se + 0.05
    assert "rate_ratio" in m.result_._stats
    with pytest.raises(ValueError):
        sc.BayesianSpatialCausal(family="poisson").fit(d)


def test_bayes_validation():
    with pytest.raises(ValueError):
        sc.BayesianSpatialCausal(chains=1)
    with pytest.raises(ValueError):
        sc.BayesianSpatialCausal(adjust="rsr")


# ---------------------------------------------------------------- staggered DiD
@pytest.fixture(scope="module")
def stag():
    return simulate_staggered_spillover(random_state=7)


def test_staggered_recovers_event_time_and_spillover(stag):
    m = sc.StaggeredSpatialDiD(exposure_distance=0.05, n_boot=99).fit(stag)
    es = m.event_study_
    d = es[(es.kind == "direct") & (es.rel_time >= 0)]
    truth = stag.truth["event_time"](d.rel_time.to_numpy())
    assert np.mean(np.abs(d.att.to_numpy() - truth) < 3 * d.se.to_numpy() + 0.05) >= 0.8
    pre = es[(es.kind == "direct") & (es.rel_time < -1)]
    assert np.all(np.abs(pre.att) < 4 * pre.se + 0.05)
    sp = m.result_["spillover"]
    assert abs(sp.estimate - 0.8) < 4 * sp.se
    assert m.pretrend_test_["p_value"] > 0.01


def test_staggered_naive_twfe_biased(stag):
    m = sc.StaggeredSpatialDiD(exposure_distance=0.05, n_boot=20).fit(stag)
    c = m.cells_
    post = c[(c.kind == "direct") & (c.rel_time >= 0)]
    true_att = np.average(1 + 0.3 * post.rel_time, weights=post["size"])
    assert abs(m.effect("att") - true_att) < abs(m.effect("att_naive_twfe") - true_att)


def test_staggered_exposure_times(stag):
    m = sc.StaggeredSpatialDiD(exposure_distance=0.05, n_boot=5)
    adopt = stag.adoption_times()
    adopt = np.where(np.isnan(adopt), np.inf, adopt)
    e = m.exposure_times(stag, adopt)
    assert np.isfinite(e[np.isinf(adopt)]).sum() > 20
    # check one exposed never-treated unit against a direct computation
    i = int(np.flatnonzero(np.isinf(adopt) & np.isfinite(e))[0])
    d = np.sqrt(((stag.unit_coords - stag.unit_coords[i]) ** 2).sum(1))
    first = min(adopt[(d <= 0.05) & np.isfinite(adopt)])
    assert e[i] == first


def test_staggered_never_control_and_spatial_cluster(stag):
    m = sc.StaggeredSpatialDiD(exposure_distance=0.05, control="never", cluster="spatial", n_boot=30).fit(stag)
    assert np.isfinite(m.effect("att"))


# ---------------------------------------------------------------- design
@pytest.fixture(scope="module")
def pts():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(600, 2))
    X = rng.normal(size=(600, 2))
    X[:, 0] += 2 * coords[:, 0]
    return coords, X, KNN(coords, k=6, transform="row")


def test_cluster_design_buffer(pts):
    coords, _, _ = pts
    d = sc.ClusterRandomizedDesign(coords, n_clusters=20, buffer=0.04)
    t = d.sample(rng=1)
    assert set(np.unique(t)) == {0, 1} and abs(t.mean() - 0.5) < 0.2
    m = d.analysis_mask(t)
    assert 0.3 < m.mean() < 1.0
    for c in np.unique(d.clusters):  # whole clusters share assignment
        assert len(np.unique(t[d.clusters == c])) == 1


def test_saturation_design(pts):
    coords, _, _ = pts
    d = sc.SaturationDesign(coords, n_clusters=30, saturations=(0.0, 0.5, 1.0))
    t, sat = d.unit_saturation(rng=2)
    assert set(np.unique(sat)) == {0.0, 0.5, 1.0}
    assert t[sat == 0].sum() == 0 and t[sat == 1].mean() == 1


def test_grts_allocation_disperses_treatment(pts):
    coords, X, w = pts
    t = sc.balanced_allocation(coords, 150, seed=1)
    assert t.sum() == 150
    assert sc.design_diagnostics(t, coords, w=w)["moran"]["I"] < 0


def test_rerandomize_meets_criteria(pts):
    coords, X, w = pts
    t, info = sc.rerandomize(coords, X, 200, max_smd=0.05, max_moran=0.02, n_draws=500, w=w)
    diag = sc.design_diagnostics(t, coords, X, w)
    assert diag["balance"].max() <= 0.05 and info["acceptance_rate"] > 0


def test_power_analysis_size_and_power(pts):
    coords, _, w = pts
    d = sc.ClusterRandomizedDesign(coords, n_clusters=30)
    r = sc.power_analysis(d, coords, direct=0.6, estimand="total", n_sims=80, spatial_sd=0.3)
    assert r["power"] > 0.5 and r["size"] < 0.15
    null = sc.power_analysis(d, coords, direct=0.0, estimand="total", n_sims=40, spatial_sd=0.3)
    assert null["power"] < 0.25
    exp = sc.ExposureMapping(w, "share", 0.5)
    sp = sc.power_analysis(sc.SaturationDesign(coords, n_clusters=30), coords, spillover=0.5, exposure=exp,
                           estimand="spillover", spatial_sd=0.0, n_sims=60)
    assert sp["power"] > 0.5
    with pytest.raises(ValueError):
        sc.power_analysis(d, coords, estimand="spillover", n_sims=2)


def test_power_curve_increases(pts):
    coords, _, _ = pts
    pc = sc.power_curve(lambda g: sc.ClusterRandomizedDesign(coords, n_clusters=g), coords, [8, 60],
                        direct=0.4, spatial_sd=0.6, estimand="total", n_sims=60)
    assert pc.loc[60, "mean_se"] < pc.loc[8, "mean_se"]


def test_designs_plug_into_exposure_probabilities(pts):
    coords, _, w = pts
    exp = sc.ExposureMapping(w, "share", 0.5)
    pi = sc.exposure_probabilities(exp, sc.ClusterRandomizedDesign(coords, n_clusters=20), 600, n_sims=200)
    assert np.allclose(pi.sum(1), 1)


# ---------------------------------------------------------------- discovery
def test_spatial_granger_direction():
    p = simulate_spatial_granger(n_units=200, n_periods=24, random_state=1)
    fwd = sc.SpatialGranger().fit(p, cause=0)
    assert fwd.tests_["local"]["p_value"] < 1e-4 and fwd.tests_["spillover"]["p_value"] < 1e-4
    assert abs(fwd.coef_.loc["x_lag1", "coef"] - 0.5) < 0.1
    rev = sc.SpatialGranger().fit(p, cause=p.y, effect=p.X[:, 0])
    assert rev.tests_["joint"]["p_value"] > 0.01


def test_granger_jackknife_reduces_nickell_bias():
    p = simulate_spatial_granger(n_units=200, n_periods=20, random_state=2)
    raw = sc.SpatialGranger(bias_correction=None).fit(p, cause=0).coef_.loc["y_lag1", "coef"]
    jk = sc.SpatialGranger().fit(p, cause=0).coef_.loc["y_lag1", "coef"]
    assert abs(jk - 0.4) < abs(raw - 0.4)


def test_lag_rings_are_disjoint():
    rng = np.random.default_rng(0)
    c = rng.uniform(size=(200, 2))
    R1, R2 = spatial_lag_rings(KNN(c, k=4), 200, 2)
    assert np.allclose(np.asarray(R1.sum(1)).ravel(), 1)
    overlap = (R1 > 0).multiply(R2 > 0)
    assert overlap.nnz == 0


def test_gccm_direction_by_convergence():
    x, y, s, w = simulate_gccm(n=900, beta_yx=0.3, random_state=4)
    c = sc.GCCM(E=3, n_rep=10).fit(x, y, w=w)
    t = c.tests_
    assert t["x_causes_y"]["p_convergence"] < 0.05
    assert t["y_causes_x"]["p_convergence"] > 0.05
    assert c.skill_.shape[1] == 2


def test_gccm_exclusion_removes_autocorrelation_skill():
    x, y, s, w = simulate_gccm(n=700, beta_yx=0.0, beta_xy=0.0, random_state=4)
    leak = sc.GCCM(E=3, n_rep=5, exclusion=0).fit(x, y, w=w).tests_["x_causes_y"]["skill"]
    clean = sc.GCCM(E=3, n_rep=5).fit(x, y, w=w).tests_["x_causes_y"]["skill"]
    assert clean < leak - 0.15


def test_notears_recovers_dag():
    rng = np.random.default_rng(0)
    n = 1000
    a = rng.normal(size=n)
    b = 1.5 * a + 0.5 * rng.normal(size=n)
    c = -1.2 * b + 0.5 * rng.normal(size=n)
    W = sc.notears_linear(np.column_stack([a, b, c]))
    assert W[0, 1] != 0 and W[1, 2] != 0
    assert W[1, 0] == 0 and W[2, 1] == 0 and W[0, 2] == 0


def test_spatial_notears_spillover_edge():
    X, s, w, truth = simulate_spatial_dag(random_state=2)
    m = sc.SpatialNOTEARS().fit(X, w=w)
    found_s = {(r.source, r.target) for r in m.edges_.itertuples() if r.type == "spillover"}
    found_l = {(r.source, r.target) for r in m.edges_.itertuples() if r.type == "local"}
    assert truth["spillover"] <= found_s
    assert len(found_l & truth["local"]) >= 3
    assert (m.spillover_.index.str.startswith("W.")).all()
    plain = sc.SpatialNOTEARS(spatial=False).fit(X)
    assert plain.spillover_ is None
