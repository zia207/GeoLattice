"""Tests for geolattice.spatial_statistics.

Each subpackage gets known-answer checks on small synthetic data: regular
lattices with exact neighbour counts, regressions with planted coefficients,
interpolators that must honour the data, and CSR point patterns whose test
statistics must sit near their null values.
"""

from __future__ import annotations

import importlib
import pkgutil

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("sklearn")

import geolattice.spatial_statistics as ss  # noqa: E402
from geolattice.spatial_statistics import (  # noqa: E402
    bayes,
    cluster,
    core,
    datasets,
    esda,
    gwmodel,
    interpolate as ip,
    pointpattern as pp,
    regression,
    sampling,
    weights,
)

# Optional heavy backends whose modules may fail to import without the extra.
_OPTIONAL_PREFIXES = (
    "geolattice.spatial_statistics.gwmodel.deep",
    "geolattice.spatial_statistics.bayes.pymc_backend",
    "geolattice.spatial_statistics.viz",
    "geolattice.spatial_statistics.gwmodel.visualization",
)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def lattice():
    gdf, col = datasets.load_lattice(n=8, seed=0)
    return gdf, gdf[col].to_numpy()


@pytest.fixture(scope="module")
def grid3():
    """3 x 3 lattice of unit squares; cell 4 is the centre."""
    gdf, _ = datasets.load_lattice(n=3, seed=0)
    return gdf


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(42)
    xy = rng.uniform(0, 10, size=(120, 2))
    return rng, xy


@pytest.fixture(scope="module")
def linear_data(points):
    """y = 1 + 2 x1 - x2 + small noise, stationary in space."""
    rng, xy = points
    X = rng.normal(size=(len(xy), 2))
    y = 1.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0, 0.05, len(xy))
    return X, y, xy


# ---------------------------------------------------------- package layout


def test_top_level_exports():
    for name in ss.__all__:
        assert hasattr(ss, name), name
    assert ss.__version__ == importlib.import_module("geolattice").__version__


def test_reachable_from_geolattice_namespace():
    import geolattice

    assert geolattice.spatial_statistics is ss
    assert geolattice.analysis.spatial_statistics is ss


def test_every_submodule_imports():
    failed = []
    for info in pkgutil.walk_packages(ss.__path__, ss.__name__ + "."):
        try:
            importlib.import_module(info.name)
        except ImportError as exc:
            if not info.name.startswith(_OPTIONAL_PREFIXES):
                failed.append(f"{info.name}: {exc}")
    assert not failed, failed


@pytest.mark.parametrize(
    "sub",
    ["core", "weights", "esda", "regression", "pointpattern", "interpolate",
     "cluster", "sampling", "bayes", "viz", "datasets"],
)
def test_subpackage_all_resolves(sub):
    mod = importlib.import_module(f"geolattice.spatial_statistics.{sub}")
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, missing


# -------------------------------------------------------------------- core


def test_distance_helpers():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    d = core.euclidean_distance_matrix(pts)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0)
    np.testing.assert_allclose(d[0, 1], 5.0)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(core.euclidean_distances(pts[:1], pts), [[0.0, 5.0, 10.0]])
    np.testing.assert_allclose(core.pairwise_distances(pts), d)


def test_haversine_one_degree_of_latitude():
    d = core.haversine_distance_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    # ~111.2 km per degree; accept metres or kilometres.
    val = d[0, 1]
    assert 110 < val < 112.5 or 110_000 < val < 112_500


def test_crs_predicates():
    assert core.is_geographic("EPSG:4326")
    assert not core.is_projected("EPSG:4326")
    assert core.is_projected("EPSG:3857")


def test_ensure_projected_reprojects_geographic():
    from shapely.geometry import Point

    g = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    out = core.ensure_projected(g)
    assert core.is_projected(out.crs)


def test_coords_from_geometry(grid3):
    c = core.coords_from_geometry(grid3.geometry)
    assert c.shape == (9, 2)
    np.testing.assert_allclose(c[4], [1.5, 1.5])


# ----------------------------------------------------------------- weights


def test_queen_and_rook_neighbour_counts(grid3):
    q = weights.Queen(grid3, transform="binary")
    r = weights.Rook(grid3, transform="binary")
    qc = np.array([len(q.neighbors[i]) for i in q.id_order])
    rc = np.array([len(r.neighbors[i]) for i in r.id_order])
    # corners / edges / centre on a 3 x 3 lattice
    assert qc.tolist() == [3, 5, 3, 5, 8, 5, 3, 5, 3]
    assert rc.tolist() == [2, 3, 2, 3, 4, 3, 2, 3, 2]
    assert q.is_symmetric() if callable(q.is_symmetric) else q.is_symmetric


def test_row_standardisation_sums_to_one(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf)  # row by default
    rows = np.asarray(w.sparse.sum(axis=1)).ravel()
    np.testing.assert_allclose(rows, 1.0)


def test_lag_of_constant_is_constant(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf)
    np.testing.assert_allclose(w.lag(np.full(w.n, 7.0)), 7.0)


def test_knn_has_exactly_k_neighbours(points):
    _, xy = points
    w = weights.KNN(xy, k=6)
    assert w.n == len(xy)
    assert all(len(v) == 6 for v in w.neighbors.values())
    assert all(i not in w.neighbors[i] for i in w.id_order)


def test_distance_band_matches_brute_force(points):
    _, xy = points
    thr = 1.5
    w = weights.DistanceBand(xy, threshold=thr)
    d = core.euclidean_distance_matrix(xy)
    expected = ((d <= thr) & (d > 0)).sum(axis=1)
    got = np.array([len(w.neighbors[i]) for i in w.id_order])
    np.testing.assert_array_equal(got, expected)


def test_kernel_and_graph_weights(points):
    _, xy = points
    kw = weights.Kernel(xy, k=8)
    assert kw.n == len(xy)
    dl = weights.Delaunay(xy)
    gb = weights.Gabriel(xy)
    rn = weights.RelativeNeighborhood(xy)
    # RNG ⊆ Gabriel ⊆ Delaunay
    for i in dl.id_order:
        assert set(rn.neighbors[i]) <= set(gb.neighbors[i]) <= set(dl.neighbors[i])
    assert dl.n_components() == 1


def test_higher_order(grid3):
    w = weights.Rook(grid3, transform="binary")
    w2 = weights.higher_order(w, k=2)
    # second-order rook neighbours of the centre are the 4 corners
    assert sorted(w2.neighbors[w2.id_order[4]]) == sorted(
        [w2.id_order[i] for i in (0, 2, 6, 8)]
    )


def test_w_from_dict_and_islands():
    w = weights.W({0: [1], 1: [0], 2: []})
    assert w.n == 3
    assert w.islands() == [2]


# -------------------------------------------------------------------- esda


def test_moran_detects_autocorrelation(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    m = esda.Moran(y, w, permutations=199, seed=0)
    assert m.I > 0.3
    assert m.p_value < 0.05


def test_moran_on_noise_is_near_expectation(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf)
    noise = np.random.default_rng(3).normal(size=w.n)
    m = esda.Moran(noise, w, permutations=199, seed=0)
    assert abs(m.I - (-1 / (w.n - 1))) < 0.2
    assert m.p_value > 0.01


def test_geary_and_general_g(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    g = esda.Geary(y, w, permutations=99, seed=0)
    assert g._stats  # populated
    C = g.get("C") if g.get("C") is not None else next(iter(g._stats.values()))
    assert C < 1  # positive autocorrelation -> C below 1
    gg = esda.GeneralG(y - y.min() + 1, weights.Queen(gdf, transform="binary"),
                       permutations=99, seed=0)
    assert gg._stats


def test_bivariate_and_diff_moran(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    bv = esda.MoranBV(y, y, w, permutations=99, seed=0)
    assert bv.I > 0.3
    diff = esda.Moran_Diff(y + 1.0, y, w, permutations=99, seed=0)
    assert np.isfinite(diff.I)


def test_local_moran_shapes_and_quadrants(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    lm = esda.Moran_Local(y, w, permutations=99, seed=0)
    assert lm.Is.shape == (w.n,)
    # 0 = not significant, 1..4 = HH / LH / LL / HL
    assert set(np.unique(lm.q)) <= {0, 1, 2, 3, 4}
    assert set(lm.labels[lm.q == 1]) <= {"HH"} and set(lm.labels[lm.q == 0]) <= {"NS"}
    assert np.all((lm.p_sim > 0) & (lm.p_sim <= 1))
    # mean of local Is (row-standardised) equals global I up to n/(n-1) style scaling
    gI = esda.Moran(y, w, permutations=0).I
    assert np.sign(lm.Is.mean()) == np.sign(gI)


def test_other_local_statistics(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    assert esda.Moran_Local_BV(y, y, w, permutations=49, seed=0).Is.shape == (w.n,)
    assert esda.Geary_Local(y, w, permutations=49, seed=0).local_C.shape == (w.n,)
    gl = esda.G_Local(y - y.min() + 1, w, star=True, permutations=49, seed=0)
    assert gl.Gs.shape == (w.n,)


def test_join_counts(lattice):
    gdf, _ = lattice
    b = gdf["binary"].to_numpy()
    jc = esda.Join_Counts(b, weights.Queen(gdf, transform="binary"), permutations=99, seed=0)
    assert jc._stats


def test_fdr_bh_matches_manual():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    reject, adj = esda.fdr_bh(p, alpha=0.05)
    m = len(p)
    manual = np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1]
    np.testing.assert_allclose(adj, np.minimum(manual, 1.0))
    assert reject.tolist() == [True, True] + [False] * 8


# -------------------------------------------------------------- regression


def test_ols_recovers_coefficients(linear_data):
    X, y, xy = linear_data
    res = regression.OLS(y, X)
    np.testing.assert_allclose(np.ravel(res.beta), [1.0, 2.0, -1.0], atol=0.03)
    assert res.r2 > 0.99
    assert np.allclose(np.ravel(res.predy) + np.ravel(res.residuals), y)


def test_ols_diagnostics(linear_data):
    X, y, xy = linear_data
    w = weights.KNN(xy, k=6)
    res = regression.OLS(y, X, w=w)
    lm = regression.lm_tests(res, w)
    assert isinstance(lm.table, pd.DataFrame) and len(lm.table) >= 2
    assert regression.jarque_bera(np.ravel(res.residuals), k=3)
    assert regression.breusch_pagan(np.ravel(res.residuals), X)
    assert regression.condition_number(np.column_stack([np.ones(len(X)), X])) >= 1
    mr = regression.moran_residuals(res, w)
    assert isinstance(mr, dict)


def test_spatial_lag_recovers_rho(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf)
    rng = np.random.default_rng(7)
    n = w.n
    X = rng.normal(size=(n, 1))
    rho, beta = 0.5, np.array([1.0, 2.0])
    A = np.eye(n) - rho * w.sparse.toarray()
    y = np.linalg.solve(A, beta[0] + beta[1] * X[:, 0] + rng.normal(0, 0.1, n))
    lag = regression.SpatialLag(y, X, w)
    est_rho = lag.get("rho") if lag.get("rho") is not None else np.ravel(lag.beta)[-1]
    assert abs(float(est_rho) - rho) < 0.15
    err = regression.SpatialError(y, X, w)
    assert np.isfinite(err.loglik)
    assert np.isfinite(regression.SpatialDurbin(y, X, w).loglik)
    assert np.isfinite(regression.SLX(y, X, w).r2)
    ols = regression.OLS(y, X)
    lr = regression.lr_test(ols, lag)
    assert isinstance(lr, dict)
    tab = regression.compare_models({"ols": ols, "lag": lag, "err": err})
    assert len(tab) == 3


# ---------------------------------------------------------- point patterns


@pytest.fixture(scope="module")
def csr():
    return pp.simulate_csr(n=200, seed=1)


def test_csr_tests_do_not_reject(csr):
    ce = pp.clark_evans(csr)
    assert 0.85 < ce.get("R") < 1.15
    assert pp.quadrat_test(csr, nx=4, ny=4).get("p_value") > 0.01


def test_ripley_l_close_to_r_under_csr(csr):
    L = pp.ripley_l(csr)
    r = np.asarray(L.r)
    est = np.asarray(L.estimate)
    mask = (r > 0) & (r < 0.15 * np.ptp(csr.coords[:, 0]))
    assert np.max(np.abs(est[mask] - r[mask])) < 0.05 * np.ptp(csr.coords[:, 0])


def test_distance_functions_are_monotone_cdfs(csr):
    for fn in (pp.g_function, pp.f_function):
        res = fn(csr)
        est = np.asarray(res.estimate)
        est = est[np.isfinite(est)]
        assert est.min() >= -1e-9 and est.max() <= 1 + 1e-9
        assert np.all(np.diff(est) >= -1e-9)
    assert np.isfinite(np.asarray(pp.j_function(csr).r)).all()
    assert len(pp.nearest_neighbor_distances(csr)) == csr.coords.shape[0]


def test_clustered_and_regular_patterns_are_detected():
    thomas = pp.simulate_thomas(kappa=15, sigma=0.02, mu=12, seed=3)
    matern = pp.simulate_matern_inhibition(intensity=400, radius=0.04, seed=3)
    assert pp.clark_evans(thomas).get("R") < 0.8
    assert pp.clark_evans(matern).get("R") > 1.1


def test_kde_integrates_to_n(csr):
    k = pp.kde(csr, grid_size=64)
    dens = np.where(k.mask, k.density, 0.0) if k.mask is not None else k.density
    dx = np.diff(np.asarray(k.x)).mean()
    dy = np.diff(np.asarray(k.y)).mean()
    total = np.nansum(dens) * dx * dy
    assert total == pytest.approx(csr.coords.shape[0], rel=0.15)
    assert pp.bandwidth_scott(csr) > 0
    assert len(pp.intensity_at_points(csr)) == csr.coords.shape[0]


def test_envelope_contains_observed_for_csr(csr):
    env = pp.envelope(csr, statistic="L", nsim=39, seed=0)
    inside = (env.observed >= env.lo - 1e-9) & (env.observed <= env.hi + 1e-9)
    assert inside.mean() > 0.8


def test_marked_cross_k_and_relative_risk():
    a = pp.simulate_csr(n=80, seed=5)
    b = pp.simulate_csr(n=80, seed=6)
    coords = np.vstack([a.coords, b.coords])
    marks = np.array(["a"] * 80 + ["b"] * 80)
    marked = pp.PointPattern(coords, marks=marks)
    ck = pp.cross_k(marked, "a", "b")
    assert len(ck.r) == len(ck.estimate)
    rr = pp.relative_risk(a, b, grid_size=32)
    assert rr.density.shape == (32, 32) or rr.density.size == 32 * 32


# ------------------------------------------------------------- interpolate


@pytest.fixture(scope="module")
def surface(points):
    _, xy = points
    z = np.sin(xy[:, 0] / 3) + np.cos(xy[:, 1] / 3)
    return xy, z


@pytest.mark.parametrize(
    "model",
    [ip.NearestNeighbor(), ip.IDW(), ip.TIN(), ip.RBF(), ip.ThinPlateSpline()],
    ids=["nn", "idw", "tin", "rbf", "tps"],
)
def test_exact_interpolators_honour_data(surface, model):
    xy, z = surface
    pred = model.fit(xy, z).predict(xy)
    np.testing.assert_allclose(pred, z, atol=1e-6)


def test_trend_surface_recovers_plane(points):
    _, xy = points
    z = 3.0 + 0.5 * xy[:, 0] - 0.25 * xy[:, 1]
    pred = ip.TrendSurface(degree=1).fit(xy, z).predict(np.array([[0.0, 0.0], [2.0, 4.0]]))
    np.testing.assert_allclose(pred, [3.0, 3.0], atol=1e-8)


def test_make_grid_and_thiessen(surface):
    xy, z = surface
    grid = ip.make_grid(xy, n_cells=20)
    assert grid.coords.shape[1] == 2 and len(grid.coords) > 0
    polys = ip.thiessen_polygons(xy, values=z)
    assert len(polys) == len(xy)


def test_variogram_fit_and_kriging(surface):
    xy, z = surface
    emp = ip.empirical_variogram(xy, z, n_lags=10)
    assert {"lag", "gamma"} <= set(emp.columns) or emp.shape[1] >= 2
    vm = ip.fit_variogram(emp, model="spherical")
    assert vm.range > 0 and vm.psill > 0
    auto_vm, _ = ip.fit_variogram_auto(xy, z)
    assert auto_vm.model in {"spherical", "exponential", "gaussian"}
    new = np.array([[5.0, 5.0], [2.0, 8.0]])
    truth = np.sin(new[:, 0] / 3) + np.cos(new[:, 1] / 3)
    for kr in (ip.OrdinaryKriging(vm), ip.SimpleKriging(vm), ip.UniversalKriging(vm)):
        out = kr.fit(xy, z).predict(new)
        pred = out[0] if isinstance(out, tuple) else out
        np.testing.assert_allclose(np.ravel(pred), truth, atol=0.15)


def test_indicator_kriging_probabilities(surface):
    xy, z = surface
    thr = float(np.median(z))
    ind = ip.indicator_transform(z, thr)
    assert set(np.unique(ind)) <= {0, 1}
    vm = ip.fit_indicator_variogram(xy, z, thr)
    assert vm is not None
    probs = ip.IndicatorKriging().fit(xy, z).predict(np.array([[5.0, 5.0], [0.5, 9.5]]),
                                                     thresholds=[thr])
    prob = np.asarray(probs[thr])
    assert prob.shape == (2,)
    assert np.all((prob >= -1e-9) & (prob <= 1 + 1e-9))


def test_cross_validate_and_compare(surface):
    xy, z = surface
    cv = ip.cross_validate(ip.IDW(), xy, z, method="kfold", k=5, seed=0)
    assert cv.get("rmse") < np.std(z)
    tab = ip.compare_methods({"idw": ip.IDW(), "nn": ip.NearestNeighbor()}, xy, z,
                             method="kfold", k=5, seed=0)
    assert len(tab) == 2


def test_areal_weighting_conserves_extensive_totals(grid3):
    from shapely.geometry import box

    src = grid3[["geometry"]].copy()
    src["pop"] = np.arange(1, 10, dtype=float)
    target = gpd.GeoDataFrame(geometry=[box(0, 0, 1.5, 3), box(1.5, 0, 3, 3)], crs=src.crs)
    out = ip.areal_weighting(src, target, extensive=["pop"])
    assert out["pop"].sum() == pytest.approx(src["pop"].sum())


# ----------------------------------------------------------------- cluster


def test_hotspot_finds_planted_cluster(lattice):
    gdf, _ = lattice
    y = np.zeros(len(gdf))
    hot = (gdf["row"] < 3) & (gdf["col"] < 3)
    y[hot.to_numpy()] = 10.0
    y += np.random.default_rng(0).normal(0, 0.1, len(y))
    res = cluster.getis_ord_gi(y, weights.Queen(gdf), seed=0)
    assert np.asarray(res.hot)[hot.to_numpy()].mean() > 0.5
    assert not np.asarray(res.hot)[~hot.to_numpy()].any() or \
        np.asarray(res.hot)[~hot.to_numpy()].mean() < 0.1


def test_local_moran_clusters(lattice):
    gdf, y = lattice
    res = cluster.local_moran_clusters(y, weights.Queen(gdf), permutations=99, seed=0)
    assert len(res.labels) == len(y)


def test_dbscan_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (40, 2))
    b = rng.normal(5, 0.1, (40, 2))
    res = cluster.dbscan_clusters(np.vstack([a, b]), eps=0.5, min_samples=4)
    labels = np.asarray(res.labels)
    assert len(set(labels[labels >= 0])) == 2
    assert len(cluster.k_distance(np.vstack([a, b]), k=4)) == 80


def test_regionalisation(lattice):
    gdf, y = lattice
    w = weights.Queen(gdf)
    data = np.column_stack([y, gdf["x"], gdf["y"]])
    for res in (cluster.skater(data, w, n_clusters=4), cluster.agglomerative(data, w, n_clusters=4)):
        labels = np.asarray(res.labels)
        assert len(np.unique(labels)) == 4
    ev = cluster.evaluate_regions(data, labels)
    assert isinstance(ev, dict)


def test_agreement_metrics():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 0, 0], bool)
    assert cluster.jaccard(a, b) == pytest.approx(0.5)
    assert cluster.partition_agreement([0, 0, 1, 1], [1, 1, 0, 0])
    adj = cluster.adjust_pvalues(np.array([0.01, 0.02, 0.5]))
    assert np.all(adj >= np.array([0.01, 0.02, 0.5]) - 1e-12)


def test_spatial_scan_finds_hotspot(points):
    _, xy = points
    pop = np.full(len(xy), 1000.0)
    near = np.linalg.norm(xy - [2, 2], axis=1) < 1.5
    rng = np.random.default_rng(1)
    cases = rng.poisson(np.where(near, 50, 5))
    res = cluster.spatial_scan(xy, cases, population=pop, n_clusters=1, seed=0)
    members = set(np.asarray(res.members[0]).tolist())
    assert len(members & set(np.flatnonzero(near).tolist())) >= 0.5 * near.sum()


# ---------------------------------------------------------------- sampling


def test_simple_random_and_systematic(points):
    _, xy = points
    s = sampling.simple_random(xy, 20, seed=0)
    assert s.n == 20 and len(set(np.asarray(s.indices))) == 20
    np.testing.assert_allclose(s.pi, 20 / len(xy))
    sy = sampling.systematic(xy, 20, seed=0)
    assert sy.n == 20


def test_stratified_respects_allocation(points):
    _, xy = points
    strata = np.where(xy[:, 0] < 5, "west", "east")
    s = sampling.stratified(xy, strata, n=20, seed=0)
    picked = strata[np.asarray(s.indices)]
    assert set(picked) == {"west", "east"}
    assert s.n == 20


def test_spatially_balanced_designs(points):
    _, xy = points
    for s in (sampling.grts(xy, 15, seed=0), sampling.space_filling(xy, 15, seed=0)):
        assert s.n == 15
        assert np.isfinite(sampling.spatial_balance(s))
    assert isinstance(sampling.coverage_metrics(s), dict)


def test_estimate_mean_is_unbiased_for_srs(points):
    _, xy = points
    y = xy[:, 0] + xy[:, 1]
    ests = [sampling.estimate_mean(y, sampling.simple_random(xy, 30, seed=s)).get("mean")
            for s in range(60)]
    assert np.mean(ests) == pytest.approx(y.mean(), rel=0.05)


def test_sample_size_and_allocation():
    assert sampling.sample_size_mean(sd=10, margin=2) == 97  # ceil((1.96*10/2)^2)
    alloc = sampling.neyman_allocation([100, 100], [1.0, 3.0], n=40)
    assert alloc.sum() == 40 and alloc[1] > alloc[0]
    assert sampling.effective_sample_size(100, 0.0) == pytest.approx(100)
    assert sampling.design_effect(2.0, 1.0) == pytest.approx(2.0)


def test_clhs_and_grid_points(points):
    _, xy = points
    cov = pd.DataFrame({"a": xy[:, 0], "b": xy[:, 1]})
    s = sampling.clhs(cov, 10, iterations=500, seed=0)
    assert s.n == 10
    pts = sampling.grid_points((0, 0, 10, 10), n=25)
    assert len(pts) >= 16


# ------------------------------------------------------------------- bayes


def test_smr_and_expected_counts():
    pop = np.array([1000, 2000, 3000])
    obs = np.array([10, 30, 20])
    e = bayes.expected_counts(pop, observed=obs)
    assert e.sum() == pytest.approx(obs.sum())
    np.testing.assert_allclose(bayes.smr(obs, e), obs / e)
    ci = bayes.smr_confidence_interval(obs, e)
    assert len(ci) == 3
    assert np.all(bayes.excess_pvalue(obs, e) <= 1)
    assert len(bayes.funnel_limits(e)) == 3


def test_eb_shrinks_toward_prior():
    rng = np.random.default_rng(0)
    e = rng.uniform(1, 50, 200)
    obs = rng.poisson(e)
    res = bayes.eb_gamma_poisson(obs, e)
    raw_dev = np.abs(np.asarray(res.raw) - res.prior_mean)
    est_dev = np.abs(np.asarray(res.estimate) - res.prior_mean)
    assert np.all(est_dev <= raw_dev + 1e-9)
    js = bayes.eb_james_stein(np.asarray(res.raw), 1.0 / e)
    assert len(js.estimate) == 200


def test_eb_local_and_icar(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf, transform="binary")
    e = np.full(w.n, 10.0)
    obs = np.random.default_rng(0).poisson(e)
    assert len(bayes.eb_local(obs, e, w).estimate) == w.n
    st = bayes.icar_structure(w)
    assert bayes.icar_scaling_factor(st) > 0


def test_simulation_and_mcmc_diagnostics(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf, transform="binary")
    theta = bayes.simulate_relative_risk(w, seed=0)
    counts = bayes.simulate_counts(theta, np.full(w.n, 20.0), seed=0)
    assert theta.shape == counts.shape == (w.n,)
    chains = np.random.default_rng(0).normal(size=(4, 500))
    assert bayes.rhat(chains) == pytest.approx(1.0, abs=0.05)
    assert bayes.ess(chains) > 500


def test_bym_gibbs_runs(lattice):
    gdf, _ = lattice
    w = weights.Queen(gdf, transform="binary")
    e = np.full(w.n, 20.0)
    obs = bayes.simulate_counts(bayes.simulate_relative_risk(w, seed=1), e, seed=1)
    fit = bayes.BYM(obs, e, w, draws=200, tune=200, chains=2, thin=2, seed=0)
    rr = fit.get("relative_risk") if hasattr(fit, "get") else None
    assert fit is not None and (rr is None or len(rr) == w.n)


# ----------------------------------------------------------------- gwmodel


def test_gwr_recovers_stationary_coefficients(linear_data):
    X, y, xy = linear_data
    g = gwmodel.GWR(bandwidth=60).fit(X, y, xy)
    coef = np.asarray(g.coef_)
    assert coef.shape == (len(y), 3)
    np.testing.assert_allclose(coef.mean(axis=0), [1.0, 2.0, -1.0], atol=0.05)
    assert np.isfinite(g.aicc_)


def test_gwr_bandwidth_selection_and_predict(linear_data):
    X, y, xy = linear_data
    g = gwmodel.GWR().fit(X, y, xy)
    assert g.bandwidth_ is not None and g.bandwidth_ > 0
    pred = g.predict(X[:5], xy[:5])
    np.testing.assert_allclose(np.ravel(pred), y[:5], atol=0.3)


def test_gwr_detects_varying_coefficient(points):
    rng, xy = points
    X = rng.normal(size=(len(xy), 1))
    beta = xy[:, 0] / 5.0  # 0 at west edge, 2 at east edge
    y = beta * X[:, 0] + rng.normal(0, 0.05, len(xy))
    g = gwmodel.GWR(bandwidth=25).fit(X, y, xy)
    corr = np.corrcoef(np.asarray(g.coef_)[:, 1], beta)[0, 1]
    assert corr > 0.8


def test_mgwr_runs(linear_data):
    X, y, xy = linear_data
    m = gwmodel.MGWR().fit(X, y, xy)
    np.testing.assert_allclose(np.asarray(m.coef_).mean(axis=0), [1.0, 2.0, -1.0], atol=0.05)


def test_gwss_local_mean(points):
    _, xy = points
    X = np.column_stack([np.full(len(xy), 4.0), xy[:, 0]])
    s = gwmodel.GWSS(bandwidth=30).fit(X, xy)
    np.testing.assert_allclose(np.asarray(s.local_mean_)[:, 0], 4.0)
    np.testing.assert_allclose(np.asarray(s.local_var_)[:, 0], 0.0, atol=1e-10)


@pytest.mark.parametrize("family", ["poisson", "binomial", "gaussian"])
def test_gwglm_matches_global_glm_at_infinite_bandwidth(points, family):
    """With uniform kernel weights every local IRLS fit is the global MLE."""
    sm = pytest.importorskip("statsmodels.api")
    rng, xy = points
    X = rng.normal(size=(len(xy), 2))
    eta = 0.3 + 0.8 * X[:, 0] - 0.4 * X[:, 1]
    y, fam = {
        "poisson": (rng.poisson(np.exp(eta)), sm.families.Poisson()),
        "binomial": (rng.binomial(1, 1 / (1 + np.exp(-eta))), sm.families.Binomial()),
        "gaussian": (eta + rng.normal(0, 0.1, len(xy)), sm.families.Gaussian()),
    }[family]
    ref = sm.GLM(y, sm.add_constant(X), family=fam).fit().params
    g = gwmodel.GWGLM(family=family, bandwidth=1e9, fixed=True).fit(X, y, xy)
    np.testing.assert_allclose(g.coef_, np.broadcast_to(ref, g.coef_.shape), atol=1e-6)


def test_gwglm_poisson_local_fits_are_stable(points):
    rng, xy = points
    X = rng.normal(size=(len(xy), 1))
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 0]))
    g = gwmodel.GWGLM(family="poisson", bandwidth=40).fit(X, y, xy)
    assert np.all(np.isfinite(g.coef_)) and np.abs(g.coef_).max() < 5


def test_gwpca_runs(points):
    rng, xy = points
    X = rng.normal(size=(len(xy), 3))
    p = gwmodel.GWPCA(n_components=2, bandwidth=40).fit(X, xy)
    assert p is not None


def test_gw_random_forest_runs(linear_data):
    X, y, xy = linear_data
    rf = gwmodel.GWRandomForest(n_estimators=20, bandwidth=40).fit(X, y, xy)
    pred = rf.predict(X[:10], xy[:10])
    assert np.corrcoef(np.ravel(pred), y[:10])[0, 1] > 0.7


def test_compute_config_roundtrip():
    before = gwmodel.get_compute_config()
    try:
        cfg = gwmodel.set_compute_options(n_jobs=1)
        assert cfg.n_jobs == 1 and gwmodel.get_compute_config().n_jobs == 1
    finally:
        gwmodel.set_compute_options(n_jobs=before.n_jobs)


# ---------------------------------------------------------------- datasets


@pytest.mark.parametrize("loader", ["load_georgia", "load_londonhouse",
                                    "load_beijing_pm25", "load_ncovid"])
def test_bundled_datasets_load(loader):
    try:
        data = getattr(datasets, loader)()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{loader} data not available: {exc}")
    data = data[0] if isinstance(data, tuple) else data
    assert len(data) > 0


def test_synthetic_datasets():
    gdf, col = datasets.load_lattice(n=5)
    assert len(gdf) == 25 and col in gdf
    pts, _ = datasets.load_points_csr(n=50)
    assert len(pts) == 50


# --------------------------------------------------------------------- viz


def test_viz_functions_draw(lattice):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from geolattice.spatial_statistics import viz

    gdf, y = lattice
    w = weights.Queen(gdf)
    viz.moran_scatter(y=y, w=w)
    lm = esda.Moran_Local(y, w, permutations=49, seed=0)
    viz.lisa_cluster_map(lm, gdf=gdf)
    try:
        viz.choropleth(gdf, "z", k=4)
    except ImportError:
        pass  # mapclassify optional
    plt.close("all")
