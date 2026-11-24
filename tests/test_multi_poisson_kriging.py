"""Tests for geolattice.geostatistics.kriging.multi_poisson_kriging (scaling, k-NN paths, nugget LMC)."""
import numpy as np
import pytest

from geolattice.geostatistics.kriging import multi_poisson_kriging as M


@pytest.fixture(scope="module")
def toy():
    rs = np.random.default_rng(1)
    N = 120
    xy = rs.uniform(0, 1000, (N, 2))
    n = rs.integers(500, 20000, N).astype(float)
    z = rs.normal(size=N)
    y1 = rs.poisson(n * 0.10 * np.exp(0.2 * z)).astype(float)
    y2 = rs.poisson(n * 0.30 * np.exp(0.15 * z + 0.1 * rs.normal(size=N))).astype(float)
    ds = [dict(x=xy[:, 0], y=xy[:, 1], cases=y1, pop=n), dict(x=xy[:, 0], y=xy[:, 1], cases=y2, pop=n)]
    lmc = M.LMC([[4e-4, 3e-4], [3e-4, 5e-4]], 200, "Exp", [[1e-5, 0], [0, 1e-5]])
    return xy, n, y1, y2, ds, lmc


def test_variogram_constant_risk_has_no_spurious_sill():
    rs = np.random.default_rng(0)
    xy = rs.uniform(0, 1000, (400, 2)); n = rs.integers(2000, 50000, 400).astype(float)
    y = rs.poisson(n * 0.1).astype(float)
    g1 = M.pck_variogram(xy, y, n, pop_rate=1, nbins=5)[1]
    g100 = M.pck_variogram(xy, y, n, pop_rate=100, nbins=5)[1]
    assert np.allclose(g100, 1e4 * g1)
    assert g100.max() < 0.02          # noise variance in %^2 is ~ 1e4 * 0.1 / 2e4 = 0.05 before correction


def test_pop_rate_scaling(toy):
    xy, n, y1, y2, ds, lmc = toy
    a = M.poisson_cokrige(ds, 0, lmc, pop_rate=1, shared_mean={(0, 1): 0.05}, smooth=True)
    lmc100 = M.LMC(lmc.B * 1e4, lmc.range, "Exp", lmc.B0 * 1e4)
    b = M.poisson_cokrige(ds, 0, lmc100, pop_rate=100, shared_mean={(0, 1): 5.0}, smooth=True)
    assert np.allclose(b["rate_pred"], 100 * a["rate_pred"])
    assert np.allclose(b["rate_var"], 1e4 * a["rate_var"])


def test_knn_prediction_equals_global_with_all_neighbours(toy):
    xy, n, y1, y2, ds, lmc = toy
    rs = np.random.default_rng(2)
    obs = np.sort(rs.choice(len(n), 50, replace=False)); mis = np.setdiff1d(np.arange(len(n)), obs)
    dt = dict(x=xy[obs, 0], y=xy[obs, 1], cases=y1[obs], pop=n[obs])
    g = M.poisson_cokrige([dt, ds[1]], 0, lmc, pop_rate=1, shared_mean={(0, 1): 0.05}, coords_pred=xy[mis])
    k = M.poisson_cokrige([dt, ds[1]], 0, lmc, pop_rate=1, shared_mean={(0, 1): 0.05}, coords_pred=xy[mis],
                          number_of_neighbors=[len(obs), len(n)], mean_rate="global", use_gpu=False)
    assert np.allclose(g["pred"], k["pred"]) and np.allclose(g["var"], k["var"])


def test_knn_loo_smoothing_close_to_global(toy):
    xy, n, y1, y2, ds, lmc = toy
    N = len(n)
    a = M.poisson_cokrige(ds, 0, lmc, pop_rate=1, shared_mean={(0, 1): 0.05}, smooth=True)
    c = M.poisson_cokrige(ds, 0, lmc, pop_rate=1, shared_mean={(0, 1): 0.05}, smooth=True,
                          number_of_neighbors=[N - 1, N], mean_rate="global", use_gpu=False)
    # differ only through m*, which the global path recomputes without the left-out area
    assert np.max(np.abs(a["rate_pred"] - c["rate_pred"])) < 1e-4


def test_poisson_krige_knn_matches_p1_cokriging(toy):
    xy, n, y1, y2, ds, lmc = toy
    u = M.poisson_krige(ds[0], 4e-4, 200, nugget=1e-5, pop_rate=1, smooth=True, number_of_neighbors=20, use_gpu=False)
    v = M.poisson_cokrige([ds[0]], 0, M.LMC([[4e-4]], 200, "Exp", [[1e-5]]), pop_rate=1, smooth=True,
                          number_of_neighbors=20, use_gpu=False)
    assert np.allclose(u["rate_pred"], v["rate_pred"])


def test_fit_lmc_nugget_recovers_structure():
    lags = np.linspace(25, 775, 16); npv = np.full(16, 1000.0)
    true = M.LMC([[2.0, 1.0], [1.0, 3.0]], 300, "Exp", [[1.0, 0.5], [0.5, 2.0]])
    dv = [(lags, true.gamma(l, l, lags), npv) for l in range(2)]
    cv = [(lags, true.gamma(0, 1, lags), npv)]
    fit = M.fit_lmc(dv, cv, [(0, 1)], 2, 150, fit_range=True, nugget=True, range_grid=np.arange(100, 601, 25))
    assert fit.range == pytest.approx(300)
    assert np.allclose(fit.B, true.B, atol=1e-6) and np.allclose(fit.B0, true.B0, atol=1e-6)
