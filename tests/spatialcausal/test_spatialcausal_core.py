import warnings

import geopandas as gpd
import numpy as np
import pytest
from scipy.stats import binom
from geolattice.spatial_statistics.weights import KNN, Queen

import geolattice.spatial_analysis.spatialcausal as sc
from geolattice.spatial_analysis.spatialcausal._utils import as_sparse_weights, make_learner
from geolattice.spatial_analysis.spatialcausal.datasets import (
    simulate_interference,
    simulate_spatial_confounding,
    simulate_spatial_heterogeneity,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="module")
def grid():
    xs, ys = np.meshgrid(np.arange(10.0), np.arange(10.0))
    return np.column_stack([xs.ravel(), ys.ravel()])


# ---------------------------------------------------------------- core / utils
def test_data_container_and_default_weights(grid):
    rng = np.random.default_rng(0)
    d = sc.SpatialCausalData(rng.normal(size=(100, 2)), rng.integers(0, 2, 100), rng.normal(size=100), grid)
    assert d.n == 100 and d.is_binary
    assert d.w_sparse().shape == (100, 100)
    assert np.allclose(np.asarray(d.w_sparse().sum(1)).ravel(), 1.0)


def test_from_geodataframe(grid):
    rng = np.random.default_rng(1)
    gdf = gpd.GeoDataFrame({"a": rng.normal(size=100), "t": rng.integers(0, 2, 100), "y": rng.normal(size=100)},
                           geometry=gpd.points_from_xy(grid[:, 0], grid[:, 1]))
    d = sc.SpatialCausalData.from_geodataframe(gdf, ["a"], "t", "y")
    assert d.feature_names == ["a"] and np.allclose(d.coords, grid)


def test_weights_interop_zero_diagonal(grid):
    W = as_sparse_weights(np.ones((5, 5)))
    assert W.diagonal().sum() == 0
    with pytest.raises(ValueError):
        as_sparse_weights(np.ones((5, 4)))


def test_bad_inputs():
    with pytest.raises(ValueError):
        sc.SpatialCausalData(np.ones((3, 1)), [0, 1, 0], [1, 2, np.nan], np.zeros((3, 2)))
    with pytest.raises(ValueError):
        sc.SpatialCausalData(np.ones((3, 1)), [0, 1], [1, 2, 3], np.zeros((3, 2)))
    with pytest.raises(ValueError):
        make_learner("nope")


# ---------------------------------------------------------------- exposure
def test_exposure_conditions_share(grid):
    w = KNN(grid, k=4, transform="row")
    t = np.zeros(100, int)
    t[:50] = 1
    m = sc.ExposureMapping(w, kind="share", threshold=0.5)
    g = m.neighbor_exposure(t)
    assert g.min() >= 0 and g.max() <= 1
    c = m.conditions(t)
    assert set(np.unique(c)) <= {0, 1, 2, 3}
    assert np.all((c >= 2) == (t == 1))


def test_exposure_any_and_count(grid):
    w = Queen(_grid_polys())
    t = np.zeros(100, int)
    t[55] = 1
    any_ = sc.ExposureMapping(w, kind="any").neighbor_exposure(t)
    cnt = sc.ExposureMapping(w, kind="count").neighbor_exposure(t)
    assert any_.sum() == cnt.sum() == 8  # queen neighbours of an interior cell


def _grid_polys():
    from shapely.geometry import box
    return gpd.GeoDataFrame(geometry=[box(i, j, i + 1, j + 1) for j in range(10) for i in range(10)])


def test_kernel_weights_and_bipartite(grid):
    K = sc.kernel_weights(grid, bandwidth=1.5, kernel="bisquare")
    assert K.diagonal().sum() == 0 and K.nnz > 0
    src = np.array([[0.0, 0.0], [9.0, 9.0]])
    be = sc.BipartiteExposure(src, grid, bandwidth=2.0, kernel="gaussian")
    g = be.neighbor_exposure([1, 0])
    assert g[0] > g[-1]
    assert be.key_source()[0] == 0 and be.key_source()[-1] == 1


def test_exposure_probabilities_match_binomial(grid):
    k, p = 4, 0.3
    w = KNN(grid, k=k, transform="row")
    m = sc.ExposureMapping(w, kind="share", threshold=0.5)
    pi = sc.exposure_probabilities(m, sc.BernoulliDesign(p), 100, n_sims=6000, random_state=3)
    p_high = binom.sf(1, k, p)  # >= 2 of 4 treated
    expected_c11 = p * p_high
    assert abs(pi[:, 3].mean() - expected_c11) < 0.01
    assert np.allclose(pi.sum(1), 1)


# ---------------------------------------------------------------- cross-fitting
def test_spatial_block_folds_partition(grid):
    folds = list(sc.SpatialBlockKFold(n_splits=4, n_blocks=12).split(grid))
    test_all = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(test_all), np.arange(100))
    for tr, te in folds:
        assert np.intersect1d(tr, te).size == 0


def test_buffer_removes_neighbours(grid):
    for tr, te in sc.SpatialBlockKFold(n_splits=3, n_blocks=6, buffer=1.5).split(grid):
        d = np.sqrt(((grid[tr][:, None] - grid[te][None]) ** 2).sum(-1))
        assert d.min() > 1.5


def test_grid_blocks(grid):
    b = sc.SpatialBlockKFold(n_splits=2, n_blocks=9, method="grid").make_blocks(grid)
    assert len(np.unique(b)) == 9


# ---------------------------------------------------------------- inference
def test_conley_reduces_to_iid_with_tiny_cutoff(grid):
    psi = np.random.default_rng(0).normal(size=100)
    iid = sc.standard_error(psi, method="iid")
    con = sc.standard_error(psi, grid, method="conley", cutoff=0.5)
    assert np.isclose(iid, con)


def test_conley_larger_for_positively_correlated_scores(grid):
    psi = np.sin(grid[:, 0] / 3) + 0.1 * np.random.default_rng(0).normal(size=100)
    assert sc.standard_error(psi, grid, "conley", cutoff=4) > sc.standard_error(psi, method="iid")


def test_bootstrap_and_permutation(grid):
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    _, se = sc.spatial_block_bootstrap(lambda idx: y[idx].mean(), grid, n_boot=100, n_blocks=20)
    assert 0 < se < 1
    t = rng.integers(0, 2, 100)
    obs, p, null = sc.block_permutation_test(lambda tt: y[tt == 1].mean() - y[tt == 0].mean(), t, grid, n_perm=99)
    assert 0 < p <= 1 and len(null) == 99


def test_suggest_cutoff_positive(grid):
    r = np.sin(grid[:, 0] / 2)
    assert sc.suggest_cutoff(r, grid) > 0


# ---------------------------------------------------------------- interference
@pytest.fixture(scope="module")
def interf():
    return simulate_interference(n=1200, random_state=11)


def test_aipw_interference_recovers_truth(interf):
    exp = sc.ExposureMapping(interf.w, "share", 0.5)
    est = sc.InterferenceEstimator(exp, learner="ridge", n_estimators=50).fit(interf)
    for k in ("direct", "spillover", "total"):
        e = est.result_[k]
        assert abs(e.estimate - interf.truth[k]) < 4 * e.se, k


def test_hajek_interference_recovers_truth(interf):
    exp = sc.ExposureMapping(interf.w, "share", 0.5)
    est = sc.InterferenceEstimator(exp, method="hajek", design=sc.BernoulliDesign(0.5), n_sims=800).fit(interf)
    e = est.result_["direct"]
    assert abs(e.estimate - interf.truth["direct"]) < 4 * e.se


def test_interference_confounded_aipw():
    d = simulate_interference(n=1500, confounded=True, random_state=4)
    exp = sc.ExposureMapping(d.w, "share", 0.5)
    est = sc.InterferenceEstimator(exp, learner="gbr", n_estimators=60, se="conley", cutoff=0.05).fit(d)
    assert abs(est.effect("direct") - d.truth["direct"]) < 4 * est.result_["direct"].se


def test_hajek_requires_design(interf):
    with pytest.raises(ValueError):
        sc.InterferenceEstimator(sc.ExposureMapping(interf.w), method="hajek")


# ---------------------------------------------------------------- confounding
@pytest.fixture(scope="module")
def conf():
    return simulate_spatial_confounding(n=600, confounder_range=0.2, random_state=5)


def test_diagnostics_flag_confounding(conf):
    df = sc.diagnose_spatial_confounding(conf, permutations=99)
    assert df.loc["treatment | X residual", "p_value"] < 0.05
    assert "risk" in df.attrs["interpretation"]


def test_moran_eigenvectors_orthonormal(conf):
    E = sc.moran_eigenvectors(conf.w, k=10)
    assert E.shape == (600, 10)
    assert np.allclose(E.T @ E, np.eye(10), atol=1e-6)


def test_spatial_plus_beats_naive(conf):
    naive = sc.compare_adjustments(conf, include_dml=False)
    sp = sc.SpatialPlus().fit(conf)
    assert abs(sp.effect() - 1) < abs(naive.loc["OLS (X only)", "estimate"] - 1)
    assert abs(sp.effect() - 1) < abs(naive.loc["OLS + spatial smooth", "estimate"] - 1) + 0.1


def test_spatial_plus_is_not_fwl_identical(conf):
    tbl = sc.compare_adjustments(conf, include_dml=False)
    assert not np.isclose(tbl.loc["spatial+", "estimate"], tbl.loc["OLS + spatial smooth", "estimate"])


def test_bases(conf):
    assert sc.SpatialBasis("rbf", k=15).build(conf.coords).shape == (600, 15)
    assert sc.SpatialBasis("coords").build(conf.coords).shape == (600, 5)


# ---------------------------------------------------------------- spatial ML
def test_spatial_dml_unconfounded_recovers_effect():
    d = simulate_spatial_confounding(n=800, confounding_strength=0.0, random_state=6)
    est = sc.SpatialDML(model_y="rf", model_t="rf", n_estimators=100).fit(d)
    e = est.result_["ate"]
    assert abs(e.estimate - 1.0) < 4 * e.se + 0.05


def test_spatial_dml_conley_and_block_se():
    d = simulate_spatial_confounding(n=500, random_state=7, noise_range=0.2)
    a = sc.SpatialDML(model_y="ridge", model_t="ridge", se="iid").fit(d).result_["ate"].se
    b = sc.SpatialDML(model_y="ridge", model_t="ridge", se="conley", cutoff=0.15).fit(d).result_["ate"].se
    c = sc.SpatialDML(model_y="ridge", model_t="ridge", se="block").fit(d).result_["ate"].se
    assert a > 0 and b > 0 and c > 0


def test_spatial_dml_binary_treatment():
    d = simulate_spatial_confounding(n=600, treatment="binary", confounding_strength=0.0, random_state=8)
    e = sc.SpatialDML(model_y="gbr", model_t="logistic", n_estimators=80).fit(d).result_["ate"]
    assert abs(e.estimate - 1.0) < 4 * e.se + 0.1


def test_dr_learner_ate_and_cate():
    d = simulate_spatial_heterogeneity(n=1200, random_state=9)
    dr = sc.SpatialDRLearner(n_estimators=80, cate_features="coords").fit(d)
    e = dr.result_["ate"]
    assert abs(e.estimate - d.truth["ate"]) < 4 * e.se
    assert np.corrcoef(dr.cate_, d.truth["cate"])[0, 1] > 0.4  # out-of-fold
    assert dr.predict(d.X[:5], d.coords[:5]).shape == (5,)
    with pytest.raises(ValueError):
        sc.SpatialDRLearner().fit(simulate_spatial_confounding(n=100))


def test_spatial_features_options(conf):
    F = sc.spatial_features(conf.X, conf.coords, conf.w, ("coords", "lag", "eigen", "rbf"), k_eigen=5, k_rbf=7)
    assert F.shape == (600, 3 + 2 + 3 + 5 + 7)


# ---------------------------------------------------------------- PyCausalML adapter
class _MockEstimator:
    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, treatment, y):
        self.p_ = X.shape[1]
        return self

    def predict(self, X):
        return np.full(len(X), self.scale)

    def average_treatment_effect(self):
        return self.scale


def test_adapter_with_instance(conf):
    ad = sc.PyCausalMLAdapter(_MockEstimator(2.0), spatial_features=("coords", "lag")).fit(conf)
    assert ad.p_ == 3 + 2 + 3
    assert ad.average_treatment_effect() == 2.0
    new = conf.coords[:20] + 0.001
    assert ad.predict(conf.X[:20], new).shape == (20,)


def test_adapter_rejects_eigen_out_of_sample(conf):
    ad = sc.PyCausalMLAdapter(_MockEstimator(), spatial_features=("eigen",)).fit(conf)
    with pytest.raises(ValueError):
        ad.predict(conf.X[:10], conf.coords[:10])


def test_adapter_missing_pycausalml_gives_clear_error(conf):
    try:
        import pycausalml  # noqa: F401
        pytest.skip("pycausalml importable here")
    except Exception:
        pass
    with pytest.raises(ImportError, match="geolattice\\[spatialcausal-causalml\\]"):
        sc.PyCausalMLAdapter("CausalForest").fit(conf)


def test_result_table(interf):
    est = sc.InterferenceEstimator(sc.ExposureMapping(interf.w), learner="ridge").fit(interf)
    df = est.result_.to_frame()
    assert {"estimate", "se", "ci_lower", "ci_upper", "p_value"} <= set(df.columns)
    assert "Spatial interference" in est.summary()
