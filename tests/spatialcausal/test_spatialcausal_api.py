"""Direct tests for geolattice.spatial_analysis.spatialcausal helpers, designs, results and plots.

The other suites exercise these mostly through the estimators; here each
public function is called on its own.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import geolattice
import geolattice.spatial_analysis.spatialcausal as sc
from geolattice.spatial_analysis.spatialcausal import _utils as U
from geolattice.spatial_analysis.spatialcausal.crossfit import cross_fit_predict
from geolattice.spatial_analysis.spatialcausal.datasets import simulate_interference, simulate_spatial_confounding
from geolattice.spatial_analysis.spatialcausal.exposure import CONDITIONS
from geolattice.spatial_analysis.spatialcausal.inference import block_variance
from geolattice.spatial_analysis.spatialcausal.ml import aipw_pseudo_outcomes, dml_residuals
from geolattice.spatial_statistics.weights import W

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="module")
def grid():
    xs, ys = np.meshgrid(np.arange(10.0), np.arange(10.0))
    return np.column_stack([xs.ravel(), ys.ravel()])


@pytest.fixture(scope="module")
def binary_data():
    return simulate_spatial_confounding(n=300, treatment="binary", random_state=3)


# ---------------------------------------------------------------- package wiring
def test_lazy_subpackage_and_exports():
    assert geolattice.spatial_analysis.spatialcausal is sc
    assert "spatialcausal" in dir(geolattice)
    missing = [name for name in sc.__all__ if not hasattr(sc, name)]
    assert missing == []


def test_result_base_is_geolattice_spatial_result():
    from geolattice.spatial_statistics.core.result import SpatialResult

    assert issubclass(sc.SpatialCausalResult, SpatialResult)
    assert isinstance(sc.datasets.simulate_interference(n=50).w, W)


# ---------------------------------------------------------------- _utils
def test_array_validators():
    assert U.as_2d([1.0, 2.0]).shape == (2, 1)
    with pytest.raises(ValueError):
        U.as_2d(np.zeros((2, 2, 2)))
    assert U.as_1d([[1], [2]]).shape == (2,)
    with pytest.raises(ValueError, match="NaN"):
        U.as_1d([1.0, np.nan])
    assert U.as_binary([0, 1, 1]).dtype.kind == "i"
    with pytest.raises(ValueError, match="binary"):
        U.as_binary([0, 2])
    assert U.check_lengths(a=[1, 2], b=np.zeros(2), c=None) == 2
    with pytest.raises(ValueError, match="Inconsistent"):
        U.check_lengths(a=[1, 2], b=[1])


def test_as_coords_from_geodataframe(grid):
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid[:, 0], grid[:, 1]))
    assert np.allclose(U.as_coords(gdf), grid)
    assert np.allclose(U.as_coords(grid), grid)


def test_default_weights_lag_and_clip(grid):
    w = U.default_weights(grid, k=4)
    assert isinstance(w, W)
    m = U.as_sparse_weights(w, len(grid), row_standardize=True)
    assert sparse.issparse(m) and np.allclose(np.asarray(m.sum(1)).ravel(), 1.0)
    x = np.arange(len(grid), dtype=float)
    lag = U.spatial_lag(m, x)
    assert lag.shape == x.shape and lag.min() >= x.min() and lag.max() <= x.max()
    assert np.allclose(U.clip_propensity(np.array([0.0, 0.5, 1.0]), 0.05), [0.05, 0.5, 0.95])
    assert U.default_weights(grid[:3]).n == 3  # k is capped at n - 1


# ---------------------------------------------------------------- confounding bases
def test_rbf_basis(grid):
    B = sc.rbf_basis(grid, k=12, random_state=0)
    assert B.shape == (100, 12)
    assert np.allclose(B.mean(0), 0.0)  # columns are centred Gaussian bumps
    assert np.all(B.max(0) - B.min(0) > 0.5) and np.all(B.max(0) <= 1)
    assert np.allclose(B, sc.rbf_basis(grid, k=12, random_state=0))
    assert sc.rbf_basis(grid[:5], k=30).shape == (5, 5)  # k capped at n


# ---------------------------------------------------------------- cross-fitting
def test_make_folds_partition(grid):
    for cv in ("spatial_block", "random"):
        folds = sc.make_folds(grid, cv=cv, n_splits=5)
        assert len(folds) == 5
        test_idx = np.concatenate([te for _, te in folds])
        assert np.array_equal(np.sort(test_idx), np.arange(100))
        for tr, te in folds:
            assert np.intersect1d(tr, te).size == 0
    pre = [(np.arange(50), np.arange(50, 100)), (np.arange(50, 100), np.arange(50))]
    assert sc.make_folds(grid, cv=pre) == pre


def test_cross_fit_predict_out_of_fold(grid):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = X @ [1.0, -2.0] + rng.normal(0, 0.1, 100)
    folds = sc.make_folds(grid, n_splits=4)
    pred = cross_fit_predict(LinearRegression, X, y, folds)
    assert np.all(np.isfinite(pred)) and np.corrcoef(pred, y)[0, 1] > 0.99
    t = (X[:, 0] > 0).astype(int)
    p = cross_fit_predict(LogisticRegression, X, t, folds, method="predict_proba")
    assert np.all((p >= 0) & (p <= 1)) and np.mean((p > 0.5) == t) > 0.9


# ---------------------------------------------------------------- nuisance stages
def test_dml_residuals_and_aipw(binary_data):
    d = binary_data
    F = np.column_stack([d.X, d.coords])
    folds = sc.make_folds(d.coords, n_splits=3)
    u, v, l_hat, m_hat = dml_residuals(F, d.treatment, d.y, folds, "ridge", "logistic", n_estimators=30)
    assert np.allclose(u, d.y - l_hat) and np.allclose(v, d.treatment - m_hat)
    assert np.all((m_hat >= 0.01) & (m_hat <= 0.99))
    gamma, e, mu0, mu1 = aipw_pseudo_outcomes(F, d.treatment, d.y, folds, "ridge", "logistic", n_estimators=30)
    for a in (gamma, e, mu0, mu1):
        assert a.shape == (d.n,) and np.all(np.isfinite(a))
    t = d.treatment.astype(int)
    expected = mu1 - mu0 + t * (d.y - mu1) / e - (1 - t) * (d.y - mu0) / (1 - e)
    assert np.allclose(gamma, expected)


# ---------------------------------------------------------------- exposure and designs
def test_distance_decay_exposure(grid):
    t = np.zeros(100)
    t[0] = 1
    share = sc.DistanceDecayExposure(grid, bandwidth=1.5, normalize=True).neighbor_exposure(t)
    raw = sc.DistanceDecayExposure(grid, bandwidth=1.5, normalize=False).neighbor_exposure(t)
    assert np.all((share >= 0) & (share <= 1))
    assert raw[1] > raw[50] == 0.0  # neighbour of unit 0 exposed, far unit not
    assert np.argmax(raw[1:]) + 1 in (1, 10)


def test_complete_and_cluster_designs(grid):
    rng = np.random.default_rng(0)
    t = sc.CompleteDesign(30).sample(100, rng)
    assert t.sum() == 30 and set(np.unique(t)) <= {0, 1}
    clusters = (grid[:, 0] // 5) * 2 + grid[:, 1] // 5
    for _ in range(5):
        t = sc.ClusterDesign(clusters, p=0.5).sample(100, rng)
        for c in np.unique(clusters):
            assert len(np.unique(t[clusters == c])) == 1  # whole clusters switch together
    mapping = sc.ExposureMapping(U.default_weights(grid, 4), kind="share")
    pi = sc.exposure_probabilities(mapping, sc.CompleteDesign(50), 100, n_sims=300)
    assert pi.shape == (100, len(CONDITIONS)) and np.allclose(pi.sum(1), 1.0)
    assert np.allclose(pi[:, 2:].sum(1), 0.5, atol=0.12)  # P(treated) = 50 / 100


# ---------------------------------------------------------------- variance estimators
def test_conley_and_block_variance(grid):
    rng = np.random.default_rng(1)
    psi = rng.normal(size=100)
    psi -= psi.mean()
    iid = np.sum(psi ** 2) / 100 ** 2
    assert np.isclose(sc.conley_variance(psi, grid, cutoff=0.5), iid)  # no pairs inside the cutoff
    smooth = np.repeat(rng.normal(size=10), 10)  # positively correlated within rows of the grid
    smooth -= smooth.mean()
    assert sc.conley_variance(smooth, grid, cutoff=3.0) > np.sum(smooth ** 2) / 100 ** 2
    blocks = np.arange(100) // 10
    expected = 10 / 9 * np.sum(np.bincount(blocks, weights=smooth) ** 2) / 100 ** 2
    assert np.isclose(block_variance(smooth, blocks), expected)
    assert np.isclose(sc.standard_error(smooth, method="block", blocks=blocks), np.sqrt(expected))
    with pytest.raises(ValueError):
        sc.standard_error(psi, method="conley")


# ---------------------------------------------------------------- results
def test_effect_estimate_and_result_table():
    e = sc.EffectEstimate("ate", 1.0, 0.5)
    lo, hi = e.ci
    assert np.isclose(lo, 1.0 - 1.959964 * 0.5) and np.isclose(hi, 1.0 + 1.959964 * 0.5)
    assert np.isclose(e.z, 2.0) and np.isclose(e.p_value, 0.0455, atol=1e-4)
    assert np.isnan(sc.EffectEstimate("x", 1.0, 0.0).p_value)
    r = sc.SpatialCausalResult("demo", {"ate": e, "spill": sc.EffectEstimate("spill", -0.2, 0.1)},
                               se_method="conley", n=10, cutoff=2.0)
    df = r.to_frame()
    assert isinstance(df, pd.DataFrame) and list(df.index) == ["ate", "spill"]
    assert r["spill"].estimate == -0.2
    text = r.summary()
    assert "demo" in text and "conley" in text and "cutoff: 2.0" in text
    assert repr(r) == text


# ---------------------------------------------------------------- plots
@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_effect_map_and_conditions(plt, grid):
    from geolattice.spatial_analysis.spatialcausal import viz

    ax = viz.plot_effect_map(grid, grid[:, 0] - 4.5, center=0.0, title="cate")
    assert ax.get_title() == "cate" and len(ax.collections) == 1
    d = simulate_interference(n=150, random_state=0)
    cond = sc.ExposureMapping(d.w).conditions(d.treatment)
    ax = viz.plot_exposure_conditions(d.coords, cond)
    assert len(ax.collections) == 4 and ax.get_legend() is not None


def test_plot_effects_and_cutoff_sensitivity(plt):
    from geolattice.spatial_analysis.spatialcausal import viz

    d = simulate_spatial_confounding(n=200, random_state=0)
    est = sc.SpatialDML(model_y="ridge", model_t="ridge").fit(d)
    ax = viz.plot_effects(est.result_, truth={"ate": d.truth.get("ate", 1.0)})
    assert [t.get_text() for t in ax.get_yticklabels()] == ["ate"]
    cutoffs = [0.05, 0.1, 0.2]
    ax = viz.plot_cutoff_sensitivity(
        lambda c: sc.SpatialDML(model_y="ridge", model_t="ridge", se="conley", cutoff=c), d, cutoffs)
    x, est_line = ax.lines[0].get_data()
    assert list(x) == cutoffs and np.allclose(est_line, est_line[0])  # point estimate does not depend on SE


def test_old_import_path_is_an_alias():
    """``geolattice.spatialcausal`` still works, warns, and returns the same objects."""
    import importlib
    import sys
    import warnings

    for name in [m for m in sys.modules if m == "geolattice.spatialcausal" or m.startswith("geolattice.spatialcausal.")]:
        del sys.modules[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module("geolattice.spatialcausal")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert old is sc
    from geolattice.spatialcausal.quasi import GeographicRDD as OldRDD
    from geolattice.spatial_analysis.spatialcausal.quasi import GeographicRDD
    assert OldRDD is GeographicRDD
