"""Smoke tests for geolattice.geostatistics (pygstat) public API."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

# Optional backends; skip only when the extra is absent.
_OPTIONAL_MODULES = {
    "geolattice.geostatistics.ml.regression_kriging_h2o": "h2o",
    "geolattice.geostatistics.validation.spatial_cv_automl": "h2o",
    "geolattice.geostatistics.validation.spatial_cv_h2o_rk": "h2o",
    "geolattice.geostatistics.jax_variogram_kriging": "jax",
    # TensorFlow is opt-in (geolattice[geostatistics-tensorflow]) and has no Python 3.14 wheel
    "geolattice.geostatistics.ml.regression_kriging_tf": "tensorflow",
}


def test_geostatistics_public_exports():
    from geolattice import geostatistics as gstat

    required = [
        "Variogram",
        "OrdinaryKriging",
        "SimpleKriging",
        "UniversalKriging",
        "Cokriging",
        "IndicatorKriging",
        "DisjunctiveKriging",
        "EmpiricalBayesianKriging",
        "PoissonKriging",
        "RegressionKriging",
        "SoftKriging",
        "FactorialKriging",
        "STKriging",
        "STPoissonKriging",
        "sgsim",
        "sisim",
        "lusim",
        "lusim_unconditional",
        "LUSimulator",
        "tbsim",
        "tbsim_unconditional",
        "TBSimulator",
        "fftsim",
        "fftsim_unconditional",
        "FFTSimulator",
        "pgsim",
        "PluriGaussianSimulator",
        "TruncationRule",
        "dssim",
        "DirectSampling",
        "snesim",
        "SNESIM",
        "tau_model",
        "sicosim",
        "IndicatorCoSimulator",
        "SoftCalibration",
        "sgcosim",
        "sgcosim_hierarchical",
        "GaussianCoSimulator",
        "sgsim_gpu",
        "dssim_direct",
        "DirectSequentialSimulator",
        "GlobalDistribution",
        "sasim",
        "SimulatedAnnealing",
        "kt3d",
        "loo_cv",
        "kfold_cv",
        "from_geodataframe",
        "nscore_forward",
        "declus_cell",
        "spherical",
        "MODEL_FUNCS",
        "JaxVariogram",
        "JaxOrdinaryKriging",
        "JaxSimpleKriging",
        "JaxUniversalKriging",
        "JaxCokriging",
        "poisson_cokrige",
        "fit_poisson_lmc",
        "pygstat",
    ]
    missing = [name for name in required if not hasattr(gstat, name)]
    assert not missing, f"Missing geostatistics exports: {missing}"
    assert "Variogram" in gstat.__all__
    assert gstat.pygstat.Variogram is gstat.Variogram
    assert len(gstat.__all__) >= 100


def test_geostatistics_submodules_importable():
    import geolattice.geostatistics as gstat

    failures = []
    skipped = []
    prefix = gstat.__name__ + "."
    for module_info in pkgutil.walk_packages(gstat.__path__, prefix):
        name = module_info.name
        extra = _OPTIONAL_MODULES.get(name)
        if extra is not None:
            try:
                importlib.import_module(extra)
            except ImportError:
                skipped.append(name)
                continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - surfaced in assertion
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Submodule import failures:\n" + "\n".join(failures)
    # Optional H2O backends are skipped when h2o is not installed.
    assert len(skipped) <= len(_OPTIONAL_MODULES)


def test_variogram_and_ordinary_kriging_roundtrip():
    from geolattice.geostatistics import OrdinaryKriging, Variogram

    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(40, 2))
    values = (
        10
        + 0.05 * coords[:, 0]
        + 0.03 * coords[:, 1]
        + rng.normal(0, 0.5, size=40)
    )

    variogram = Variogram(coords, values, model="spherical", n_lags=8)
    variogram.fit()

    ok = OrdinaryKriging(variogram, use_gpu=False)
    ok.fit(coords, values)
    pred = ok.predict(np.array([[25.0, 25.0], [75.0, 75.0]]))

    pred_arr = np.asarray(pred, dtype=float)
    assert pred_arr.shape[0] == 2
    assert np.all(np.isfinite(pred_arr))


def test_analysis_geostatistics_alias():
    from geolattice.analysis import geostatistics as via_analysis
    from geolattice.geostatistics import Variogram

    assert via_analysis.Variogram is Variogram


def test_legacy_flat_module_aliases():
    """Former flat paths are the same module objects as the subpackage paths."""
    import sys

    import geolattice.geostatistics as gstat

    # H2O modules import h2o at import time. The TensorFlow module is checked
    # in test_tf_module_import_does_not_break_cupy_cublas so this process does
    # not load a second CUDA stack.
    skip_unless = {
        "regression_kriging_h2o": "h2o",
        "spatial_cv_automl": "h2o",
        "spatial_cv_h2o_rk": "h2o",
    }
    checked = []
    for old, new in gstat._LEGACY_MODULES.items():
        if old == "regression_kriging_tf":
            assert new == "ml.regression_kriging_tf"
            continue
        dep = skip_unless.get(old)
        if dep is not None and importlib.util.find_spec(dep) is None:
            continue
        legacy = importlib.import_module(f"geolattice.geostatistics.{old}")
        current = importlib.import_module(f"geolattice.geostatistics.{new}")
        assert legacy is current
        assert legacy.__name__ == f"geolattice.geostatistics.{new}"
        assert sys.modules[f"geolattice.geostatistics.{old}"] is current
        checked.append(old)

    for old in (
        "lusim",
        "sgsim",
        "tbsim",
        "fftsim",
        "pgs",
        "sicosim",
        "sgcosim",
        "direct_sequential",
        "sgs_gpu",
        "sis_gpu",
        "dss_gpu",
        "tbsim_gpu",
    ):
        assert old in checked


# ------------------------------------------------- LU (Cholesky) simulation


def _lu_grid(step=10.0, size=60.0):
    xs = np.arange(0.0, size, step)
    return np.array([(x, y) for x in xs for y in xs])


def _lu_cov(grid, vario):
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix

    R = _make_rotation_matrix(vario[0], vario[2], vario[3])
    return _cov_block(grid, grid, vario, R, same=True)


@pytest.fixture(scope="module")
def lu_data():
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.uniform(0, 50, 15), "Y": rng.uniform(0, 50, 15)})
    df["Z"] = rng.normal(2.0, 1.0, 15)
    df.loc[0, ["X", "Y"]] = [20.0, 30.0]  # sits exactly on a grid node
    return df


@pytest.mark.parametrize("vtype", ["spherical", "exponential", "gaussian"])
def test_lusim_unconditional_reproduces_covariance(vtype):
    from geolattice.geostatistics import lusim_unconditional

    grid = _lu_grid()
    vario = [30, 0.1, 50, 25, 1.0, vtype]
    sims = lusim_unconditional(grid, vario, nsim=8000, mean=5.0, seed=1)
    assert sims.shape == (8000, len(grid))
    assert abs(sims.mean() - 5.0) < 0.05
    np.testing.assert_allclose(np.cov(sims.T), _lu_cov(grid, vario), atol=0.07)


@pytest.mark.parametrize("kriging_type", ["simple", "ordinary"])
def test_lusim_conditional_moments_match_kriging(lu_data, kriging_type):
    from geolattice.geostatistics import LUSimulator

    grid = _lu_grid()
    sim = LUSimulator([0, 0.1, 40, 40, 1.0, "spherical"], kriging_type=kriging_type)
    sims = sim.fit(grid, lu_data).simulate(nsim=8000, seed=2)
    np.testing.assert_allclose(sims.mean(axis=0), sim.conditional_mean_, atol=0.05)
    np.testing.assert_allclose(sims.std(axis=0), sim.conditional_std_, atol=0.05)
    assert np.all(sim.conditional_std_ <= 1.0 + 1e-9)


def test_lusim_simple_kriging_mean_matches_direct_solve(lu_data):
    from geolattice.geostatistics import LUSimulator
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix

    grid = _lu_grid()
    vario = [45, 0.0, 60, 20, 2.0, "exponential"]
    sim = LUSimulator(vario, mean=1.5).fit(grid, lu_data)
    R = _make_rotation_matrix(45, 60, 20)
    xy, z = lu_data[["X", "Y"]].to_numpy(), lu_data["Z"].to_numpy()
    C11 = _cov_block(xy, xy, vario, R, same=True)
    C12 = _cov_block(xy, grid, vario, R)
    lam = np.linalg.solve(C11, C12)
    np.testing.assert_allclose(sim.conditional_mean_, 1.5 + lam.T @ (z - 1.5), atol=1e-6)
    np.testing.assert_allclose(
        sim.conditional_std_ ** 2, 2.0 - np.einsum("ij,ij->j", lam, C12), atol=1e-6
    )


@pytest.mark.parametrize("kriging_type", ["simple", "ordinary"])
def test_lusim_honours_data_without_nugget(lu_data, kriging_type):
    from geolattice.geostatistics import lusim

    grid = _lu_grid()
    node = np.flatnonzero((grid == [20.0, 30.0]).all(axis=1))[0]
    sims = lusim(grid, lu_data, "X", "Y", "Z", [0, 0.0, 40, 40, 1.0, "spherical"],
                 kriging_type=kriging_type, nsim=5, seed=3)
    np.testing.assert_allclose(sims[:, node], lu_data.loc[0, "Z"], atol=1e-4)


def test_lusim_itrans_seed_and_bounds(lu_data):
    from geolattice.geostatistics import lusim

    grid = _lu_grid()
    df = lu_data.assign(Z=np.exp(lu_data["Z"]))
    vario = [0, 0.0, 50, 50, 1.0, "matern", 1.5]
    a = lusim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=20, seed=9, zmin=0.0)
    b = lusim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=20, seed=9, zmin=0.0)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0
    # mostly inside the data range: back-transformed through the data histogram
    inside = (a >= df["Z"].min()) & (a <= df["Z"].max())
    assert inside.mean() > 0.8


def test_lusim_rejects_bad_input():
    from geolattice.geostatistics import LUSimulator, lusim_unconditional

    with pytest.raises(ValueError):
        LUSimulator([0, 0, 10, 10, 1.0, "spherical"], kriging_type="universal")
    with pytest.raises(ValueError, match="max_nodes"):
        lusim_unconditional(_lu_grid(step=1.0), [0, 0, 10, 10, 1.0, "spherical"], max_nodes=100)
    with pytest.raises(RuntimeError):
        LUSimulator([0, 0, 10, 10, 1.0, "spherical"]).simulate()


# ------------------------------------------------------ turning bands (TBSIM)


@pytest.mark.parametrize(
    "vario",
    [
        [30, 0.1, 50, 25, 1.0, "spherical"],
        [30, 0.1, 50, 25, 1.0, "exponential"],
        [30, 0.0, 50, 25, 1.0, "gaussian"],
        [0, 0.0, 40, 40, 1.0, "matern", 1.5],
    ],
    ids=lambda v: v[5],
)
def test_tbsim_unconditional_reproduces_covariance(vario):
    from geolattice.geostatistics import tbsim_unconditional

    grid = _lu_grid()
    sims = tbsim_unconditional(grid, vario, nsim=5000, mean=3.0, nlines=200, seed=1)
    assert sims.shape == (5000, len(grid))
    assert abs(sims.mean() - 3.0) < 0.05
    np.testing.assert_allclose(np.cov(sims.T), _lu_cov(grid, vario), atol=0.08)


def test_tbsim_spherical_sawtooth_lines_give_spherical_covariance():
    """Sawtooth lines (1-D covariance 1 - 3h + 2h^3) sum to C3 = 1 - 1.5h + 0.5h^3."""
    from geolattice.geostatistics.simulation.tbsim import _unconditional_standard

    rng = np.random.default_rng(0)
    h = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.2])
    pts = np.column_stack([h, np.zeros_like(h), np.zeros_like(h)])
    reals = np.array([_unconditional_standard(pts, "spherical", None, 50, rng)
                      for _ in range(6000)])
    c3 = np.where(h < 1, 1 - 1.5 * h + 0.5 * h ** 3, 0.0)
    np.testing.assert_allclose((reals * reals[:, :1]).mean(axis=0), c3, atol=0.04)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_tbsim_matern_spectrum_matches_covariance(nu):
    from geolattice.geostatistics.simulation.sgsim import _covar
    from geolattice.geostatistics.simulation.tbsim import _spectral_frequencies

    rng = np.random.default_rng(0)
    h = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
    k = _spectral_frequencies("matern", nu, 400_000, rng)
    u = rng.standard_normal((400_000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    mc = np.array([np.mean(np.cos(k * u[:, 0] * hh)) for hh in h])
    np.testing.assert_allclose(mc, _covar(h, 1.0, 0.0, "matern", nu), atol=0.01)


@pytest.mark.parametrize("kriging_type", ["simple", "ordinary"])
def test_tbsim_conditioning_matches_lu_kriging(lu_data, kriging_type):
    from geolattice.geostatistics import LUSimulator, TBSimulator

    grid = _lu_grid()
    vario = [20, 0.1, 50, 25, 1.0, "exponential"]
    tb = TBSimulator(vario, kriging_type=kriging_type, nlines=200).fit(grid, lu_data)
    lu = LUSimulator(vario, kriging_type=kriging_type).fit(grid, lu_data)
    np.testing.assert_allclose(tb.conditional_mean_, lu.conditional_mean_, atol=1e-6)
    np.testing.assert_allclose(tb.conditional_std_, lu.conditional_std_, atol=1e-4)
    sims = tb.simulate(nsim=4000, seed=2)
    np.testing.assert_allclose(sims.mean(axis=0), tb.conditional_mean_, atol=0.06)
    np.testing.assert_allclose(sims.std(axis=0), tb.conditional_std_, atol=0.06)


@pytest.mark.parametrize("vtype", ["spherical", "gaussian"])
def test_tbsim_honours_data_without_nugget(lu_data, vtype):
    from geolattice.geostatistics import tbsim

    grid = _lu_grid()
    node = np.flatnonzero((grid == [20.0, 30.0]).all(axis=1))[0]
    sims = tbsim(grid, lu_data, "X", "Y", "Z", [0, 0.0, 40, 40, 1.0, vtype],
                 nsim=4, nlines=100, seed=3)
    np.testing.assert_allclose(sims[:, node], lu_data.loc[0, "Z"], atol=1e-4)


def test_tbsim_large_grid_single_realization_variance():
    from geolattice.geostatistics import tbsim_unconditional

    xs = np.arange(0.0, 1000.0, 5.0)
    grid = np.array([(x, y) for x in xs for y in xs])        # 40 000 nodes
    sim = tbsim_unconditional(grid, [0, 0.0, 60, 60, 2.0, "spherical"], nlines=400, seed=4)
    assert sim.shape == (1, len(grid))
    # field is much larger than the range, so one realization is near-ergodic
    assert sim.var() == pytest.approx(2.0, rel=0.15)
    assert abs(sim.mean()) < 0.2


def test_tbsim_itrans_seed_bounds_and_validation(lu_data):
    from geolattice.geostatistics import TBSimulator, tbsim

    grid = _lu_grid()
    df = lu_data.assign(Z=np.exp(lu_data["Z"]))
    vario = [0, 0.0, 50, 50, 1.0, "spherical"]
    a = tbsim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=5, nlines=100, seed=9, zmin=0)
    b = tbsim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=5, nlines=100, seed=9, zmin=0)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0
    with pytest.raises(ValueError):
        TBSimulator([0, 0, 10, 10, 1.0, "cubic"])
    with pytest.raises(ValueError):
        TBSimulator([0, 0, 10, 10, 1.0, "matern"])
    with pytest.raises(ValueError, match="max_data"):
        TBSimulator(vario, max_data=5).fit(grid, lu_data)
    with pytest.raises(RuntimeError):
        TBSimulator(vario).simulate()


# --------------------------------------------- FFT spectral (circulant embedding)


@pytest.mark.parametrize(
    "vario",
    [
        [30, 0.1, 50, 25, 1.0, "spherical"],
        [30, 0.1, 50, 25, 1.0, "exponential"],
        [30, 0.0, 50, 25, 1.0, "gaussian"],
        [0, 0.0, 40, 40, 1.0, "matern", 1.5],
    ],
    ids=lambda v: v[5],
)
def test_fftsim_unconditional_is_exact(vario):
    from geolattice.geostatistics import FFTSimulator

    grid = _lu_grid()
    grid = grid[np.random.default_rng(5).permutation(len(grid))]   # any node order
    sim = FFTSimulator(vario, mean=3.0).fit(grid)
    assert sim.negative_eigen_fraction_ == 0.0
    sims = sim.simulate(nsim=6000, seed=1)
    assert sims.shape == (6000, len(grid))
    assert abs(sims.mean() - 3.0) < 0.05
    np.testing.assert_allclose(np.cov(sims.T), _lu_cov(grid, vario), atol=0.07)


def test_fftsim_embedding_eigenvalues_reproduce_lattice_covariance():
    """sqrt(lambda) squared and inverse-FFT'd gives back the torus covariance."""
    from scipy import fft as sfft

    from geolattice.geostatistics import FFTSimulator
    from geolattice.geostatistics.simulation.fftsim import _torus_covariance

    vario = [30, 0.1, 50, 25, 1.0, "spherical"]
    sim = FFTSimulator(vario).fit(_lu_grid())
    Mx, My = sim.embedding_shape_
    c = sfft.ifft2(sim._sqrt_lam ** 2 * Mx * My).real
    np.testing.assert_allclose(c, _torus_covariance(Mx, My, 10.0, 10.0, vario, sim._R), atol=1e-10)


def test_fftsim_real_and_imaginary_realizations_are_independent():
    from geolattice.geostatistics import FFTSimulator

    sim = FFTSimulator([0, 0, 30, 30, 1.0, "spherical"]).fit(_lu_grid())
    lat = sim.simulate_lattice(nsim=4000, seed=2)
    assert lat.shape == (4000, 6, 6)
    assert abs(np.corrcoef(lat[0::2].ravel(), lat[1::2].ravel())[0, 1]) < 0.02


@pytest.fixture(scope="module")
def lattice_data():
    import pandas as pd

    rng = np.random.default_rng(5)
    xs = np.arange(0.0, 60.0, 10.0)
    df = pd.DataFrame({"X": rng.choice(xs, 12), "Y": rng.choice(xs, 12),
                       "Z": rng.normal(2.0, 1.0, 12)})
    return df.drop_duplicates(["X", "Y"]).reset_index(drop=True)


@pytest.mark.parametrize("kriging_type", ["simple", "ordinary"])
@pytest.mark.parametrize("nugget", [0.0, 0.1])
def test_fftsim_conditional_moments_and_data(lattice_data, kriging_type, nugget):
    from geolattice.geostatistics import FFTSimulator, LUSimulator

    grid = _lu_grid()
    vario = [20, nugget, 50, 25, 1.0, "exponential"]
    sim = FFTSimulator(vario, kriging_type=kriging_type).fit(grid, lattice_data)
    assert sim.max_data_shift_ == 0.0
    sims = sim.simulate(nsim=5000, seed=3)
    np.testing.assert_allclose(sims.mean(axis=0), sim.conditional_mean_, atol=0.06)
    np.testing.assert_allclose(sims.std(axis=0), sim.conditional_std_, atol=0.06)
    # a datum on a node is that node's value, nugget included
    nodes = [np.flatnonzero((grid == xy).all(axis=1))[0]
             for xy in lattice_data[["X", "Y"]].to_numpy()]
    np.testing.assert_allclose(sims[:, nodes], np.broadcast_to(lattice_data["Z"], (5000, len(nodes))),
                               atol=1e-5)
    if nugget == 0.0:
        lu = LUSimulator(vario, kriging_type=kriging_type).fit(grid, lattice_data)
        np.testing.assert_allclose(sim.conditional_mean_, lu.conditional_mean_, atol=1e-6)


def test_fftsim_off_lattice_data_and_lattice_extension():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator

    grid = _lu_grid()
    df = pd.DataFrame({"X": [12.3, 75.0, 11.0], "Y": [18.9, -14.0, 21.0], "Z": [1.0, 2.0, 3.0]})
    sim = FFTSimulator([0, 0, 40, 40, 1.0, "spherical"]).fit(grid, df)
    assert sim.lattice_["nx"] == 9 and sim.lattice_["ny"] == 7     # extended to cover (80, -10)
    assert sim.max_data_shift_ == pytest.approx(np.hypot(5.0, 4.0))
    assert sim.n_data_ == 2                                          # two data share node (10, 20)
    node = np.flatnonzero((grid == [10.0, 20.0]).all(axis=1))[0]
    np.testing.assert_allclose(sim.simulate(3, seed=0)[:, node], 2.0, atol=1e-6)


def test_fftsim_large_grid_is_fast_and_has_target_variance():
    from geolattice.geostatistics import fftsim_unconditional

    xs = np.arange(0.0, 1000.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T   # 250 000 nodes
    sims = fftsim_unconditional(grid, [30, 0, 100, 50, 2.0, "spherical"], nsim=2, seed=4)
    assert sims.shape == (2, len(grid))
    assert sims.var() == pytest.approx(2.0, rel=0.1)


def test_fftsim_warns_when_embedding_cannot_be_made_valid():
    from geolattice.geostatistics import FFTSimulator

    with pytest.warns(UserWarning, match="circulant embedding"):
        sim = FFTSimulator([0, 0, 500, 500, 1.0, "gaussian"], max_cells=2 ** 12).fit(_lu_grid())
    assert 0.0 < sim.negative_eigen_fraction_ < 0.5


def test_fftsim_validation_itrans_and_seed(lu_data):
    from geolattice.geostatistics import FFTSimulator, fftsim

    grid = _lu_grid()
    vario = [0, 0.0, 50, 50, 1.0, "spherical"]
    with pytest.raises(ValueError, match="regular lattice"):
        FFTSimulator(vario).fit(np.array([[0.0, 0.0], [1.0, 0.0], [2.5, 0.0]]))
    with pytest.raises(ValueError, match="max_cells"):
        FFTSimulator(vario, max_cells=10).fit(grid)
    with pytest.raises(RuntimeError):
        FFTSimulator(vario).simulate()
    row = FFTSimulator(vario, spacing=10.0).fit(np.column_stack([np.arange(0, 60, 10.0), np.zeros(6)]))
    assert row.simulate(2, seed=0).shape == (2, 6)
    df = lu_data.assign(Z=np.exp(lu_data["Z"]))
    a = fftsim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=3, seed=9, zmin=0.0)
    b = fftsim(grid, df, "X", "Y", "Z", vario, itrans=True, nsim=3, seed=9, zmin=0.0)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0


# ------------------------------------------------------ plurigaussian (PGS)

_PGS_P = {"sand": 0.3, "silt": 0.5, "shale": 0.2}
_PGS_V1 = [0, 0, 60, 30, 1.0, "gaussian"]
_PGS_V2 = [90, 0, 40, 40, 1.0, "spherical"]


def test_truncation_rule_constructors_and_proportions():
    from geolattice.geostatistics import TruncationRule

    tg = TruncationRule.truncated_gaussian(_PGS_P)
    hi = TruncationRule.hierarchical(_PGS_P)
    fl = TruncationRule.from_flag([[0, 0, 1], [2, 1, 1]], categories=["a", "b", "c"])
    assert tg.n_fields == 1 and hi.n_fields == 2 and fl.n_fields == 2
    for rule in (tg, hi):
        assert rule.categories == list(_PGS_P)
        np.testing.assert_allclose(list(rule.proportions().values()), list(_PGS_P.values()))
    np.testing.assert_allclose(list(fl.proportions().values()), [2 / 6, 3 / 6, 1 / 6])
    # one-field rules do not depend on rho
    np.testing.assert_allclose(list(tg.proportions(0.7).values()), list(_PGS_P.values()))
    with pytest.raises(ValueError, match="tile"):
        TruncationRule([("a", 0, 0.5, 0, 1)])
    with pytest.raises(ValueError):
        TruncationRule.hierarchical({"a": 1.0})


@pytest.mark.parametrize("rho", [0.0, 0.6, -0.5])
def test_truncation_rule_proportions_match_monte_carlo(rho):
    from geolattice.geostatistics import TruncationRule

    rule = TruncationRule.from_flag([[0, 1, 1], [2, 2, 0], [1, 0, 2]])
    rng = np.random.default_rng(0)
    y = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=400_000)
    codes = rule.classify(y[:, 0], y[:, 1])
    assert (codes >= 0).all()
    mc = np.bincount(codes, minlength=3) / len(codes)
    np.testing.assert_allclose(mc, list(rule.proportions(rho).values()), atol=0.005)


def test_truncated_union_sampler_moments_and_tails():
    from scipy.stats import truncnorm

    from geolattice.geostatistics.simulation.pgs import _sample_truncated_union

    rng = np.random.default_rng(0)
    k = 200_000
    mu, sd = np.full(k, 0.5), np.full(k, 2.0)
    # single interval, including a far upper tail
    for lo, hi in ((-1.0, 0.3), (12.0, np.inf), (-np.inf, -9.0)):
        x = _sample_truncated_union(mu, sd, np.full((k, 1), lo), np.full((k, 1), hi), rng)
        ref = truncnorm((lo - 0.5) / 2.0, (hi - 0.5) / 2.0, loc=0.5, scale=2.0)
        assert np.all((x >= lo) & (x <= hi))
        assert x.mean() == pytest.approx(ref.mean(), abs=0.01)
        assert x.std() == pytest.approx(ref.std(), rel=0.02)
    # union of two intervals: mass split by their normal probabilities
    lo = np.tile([-np.inf, 1.0], (k, 1))
    hi = np.tile([-1.0, 2.0], (k, 1))
    x = _sample_truncated_union(np.zeros(k), np.ones(k), lo, hi, rng)
    from scipy.special import ndtr

    w = ndtr(-1.0) / (ndtr(-1.0) + ndtr(2.0) - ndtr(1.0))
    assert (x < 0).mean() == pytest.approx(w, abs=0.005)
    # empty union -> NaN
    assert np.isnan(_sample_truncated_union(np.zeros(3), np.ones(3), np.ones((3, 1)), np.zeros((3, 1)), rng)).all()


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_pgs_gibbs_matches_rejection_sampling(rho):
    import pandas as pd

    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix

    v1, v2 = [0, 0, 60, 60, 1.0, "spherical"], [0, 0, 40, 40, 1.0, "exponential"]
    rule = TruncationRule.hierarchical(_PGS_P)
    df = pd.DataFrame({"X": [20.0, 40, 40], "Y": [20.0, 20, 50],
                       "facies": ["sand", "silt", "shale"]})
    pgs = PluriGaussianSimulator(rule, v1, v2, rho=rho, gibbs_iter=30)
    pgs.fit(_lu_grid(size=100.0), df, "X", "Y", "facies")
    A, B = pgs._gibbs(8000, np.random.default_rng(0))

    xy, codes = pgs._data_xy, pgs._data_codes
    rng = np.random.default_rng(1)
    a = rng.multivariate_normal(np.zeros(3), _cov_block(xy, xy, v1, _make_rotation_matrix(0, 60, 60), True), 1_000_000)
    b = rng.multivariate_normal(np.zeros(3), _cov_block(xy, xy, v2, _make_rotation_matrix(0, 40, 40), True), 1_000_000)
    ok = (rule.classify(a, rho * a + np.sqrt(1 - rho ** 2) * b) == codes).all(axis=1)
    a, b = a[ok], b[ok]
    np.testing.assert_allclose(A.mean(axis=1), a.mean(axis=0), atol=0.04)
    np.testing.assert_allclose(B.mean(axis=1), b.mean(axis=0), atol=0.04)
    np.testing.assert_allclose(A.std(axis=1), a.std(axis=0), atol=0.03)
    # every Gibbs state reproduces the data categories
    assert (rule.classify(A.T, (rho * A + np.sqrt(1 - rho ** 2) * B).T) == codes).all()


@pytest.mark.parametrize("method", ["fft", "tb"])
@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_pgs_unconditional_reproduces_rule_proportions(method, rho):
    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule, category_proportions

    rule = TruncationRule.hierarchical(_PGS_P)
    grid = _lu_grid(step=10.0, size=200.0)
    pgs = PluriGaussianSimulator(rule, _PGS_V1, _PGS_V2, rho=rho, method=method, nlines=200)
    codes = pgs.fit(grid).simulate(nsim=300, seed=1)
    assert pgs.method_ == method
    np.testing.assert_allclose(category_proportions(codes, 3).mean(axis=0),
                               list(rule.proportions(rho).values()), atol=0.03)


def test_pgs_truncated_gaussian_respects_contacts():
    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule

    xs = np.arange(0.0, 400.0, 4.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rule = TruncationRule.truncated_gaussian(_PGS_P)
    c = PluriGaussianSimulator(rule, [0, 0, 60, 60, 1.0, "gaussian"]).fit(grid).simulate(3, seed=2)
    c = c.reshape(3, 100, 100)
    pairs = [(c[:, 1:, :], c[:, :-1, :]), (c[:, :, 1:], c[:, :, :-1])]
    sand_shale = sum(((p == 0) & (q == 2)).sum() + ((p == 2) & (q == 0)).sum() for p, q in pairs)
    assert sand_shale == 0
    sand_silt = sum(((p == 0) & (q == 1)).sum() for p, q in pairs)
    assert sand_silt > 0


@pytest.mark.parametrize("method", ["fft", "tb"])
@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_pgs_conditional_honours_data(method, rho):
    import pandas as pd

    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule, category_probabilities

    rng = np.random.default_rng(0)
    grid = _lu_grid(step=10.0, size=200.0)
    pick = rng.choice(len(grid), 40, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1],
                       "facies": rng.choice(list(_PGS_P), 40, p=list(_PGS_P.values()))})
    pgs = PluriGaussianSimulator(TruncationRule.hierarchical(_PGS_P), _PGS_V1, _PGS_V2,
                                 rho=rho, method=method, nlines=200, gibbs_iter=50)
    codes = pgs.fit(grid, df, "X", "Y", "facies").simulate(nsim=10, seed=3)
    expected = np.array([pgs.categories_.index(f) for f in df["facies"]])
    assert (codes[:, pick] == expected).all()
    prob = category_probabilities(codes, 3)
    np.testing.assert_allclose(prob[expected, pick], 1.0)
    np.testing.assert_allclose(prob.sum(axis=0), 1.0)


@pytest.mark.parametrize("method", ["fft", "tb"])
@pytest.mark.parametrize("conditional", [True, False])
def test_pgs_realizations_do_not_depend_on_nsim(method, conditional):
    import pandas as pd

    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule

    rng = np.random.default_rng(5)
    grid = _lu_grid(step=10.0, size=200.0)
    pick = rng.choice(len(grid), 30, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1],
                       "facies": rng.choice(list(_PGS_P), 30, p=list(_PGS_P.values()))})
    pgs = PluriGaussianSimulator(TruncationRule.hierarchical(_PGS_P), _PGS_V1, _PGS_V2,
                                 rho=0.4, method=method, nlines=100, gibbs_iter=20)
    pgs.fit(grid, df if conditional else None, "X", "Y", "facies")
    c8, (y8, _) = pgs.simulate(nsim=8, seed=11, return_latent=True)
    lat8 = pgs.latent_data_ if conditional else None
    c3, (y3, _) = pgs.simulate(nsim=3, seed=11, return_latent=True)
    np.testing.assert_array_equal(c3, c8[:3])
    np.testing.assert_allclose(y3, y8[:3], rtol=0, atol=1e-12)
    if conditional:
        for a3, a8 in zip(pgs.latent_data_, lat8):      # equal up to BLAS rounding
            np.testing.assert_allclose(a3, a8[:, :3], rtol=0, atol=1e-10)
        A, B = pgs.latent_data_
        assert (pgs.rule.classify(A.T, (0.4 * A + np.sqrt(1 - 0.4 ** 2) * B).T) == pgs._data_codes).all()


def test_pgsim_labels_off_lattice_data_and_validation():
    import pandas as pd

    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule, pgsim

    grid = _lu_grid()
    rule = TruncationRule.hierarchical(_PGS_P)
    df = pd.DataFrame({"X": [11.0, 9.0, 31.0], "Y": [19.0, 21.0, 40.0],
                       "facies": ["sand", "shale", "silt"]})
    with pytest.warns(UserWarning, match="different categories"):
        out = pgsim(grid, df, "X", "Y", "facies", rule, _PGS_V1, _PGS_V2, nsim=2,
                    gibbs_iter=10, seed=0, labels=True)
    assert out.shape == (2, len(grid)) and set(np.unique(out)) <= set(_PGS_P)
    node = np.flatnonzero((grid == [30.0, 40.0]).all(axis=1))[0]
    assert (out[:, node] == "silt").all()
    with pytest.raises(ValueError, match="not in the rule"):
        PluriGaussianSimulator(rule, _PGS_V1, _PGS_V2).fit(grid, df.assign(facies="gravel"),
                                                          "X", "Y", "facies")
    with pytest.raises(ValueError, match="sill 1"):
        PluriGaussianSimulator(rule, [0, 0, 10, 10, 2.0, "spherical"], _PGS_V2)
    with pytest.raises(ValueError, match="vario2"):
        PluriGaussianSimulator(rule, _PGS_V1)
    with pytest.raises(RuntimeError):
        PluriGaussianSimulator(rule, _PGS_V1, _PGS_V2).simulate()


def test_simulators_recondition_on_new_data_values(lu_data):
    from geolattice.geostatistics import FFTSimulator, TBSimulator

    grid = _lu_grid()
    df = lu_data.assign(X=np.round(lu_data["X"] / 10) * 10, Y=np.round(lu_data["Y"] / 10) * 10)
    df = df.drop_duplicates(["X", "Y"]).reset_index(drop=True)
    vario = [0, 0.0, 40, 40, 1.0, "spherical"]
    for cls in (FFTSimulator, TBSimulator):
        sim = cls(vario).fit(grid, df)
        xy = sim.conditioning_xy_
        new = np.arange(len(xy), dtype=float)
        out = sim.simulate(3, seed=1, data_values=new)
        nodes = [np.flatnonzero((grid == p).all(axis=1))[0] for p in xy]
        np.testing.assert_allclose(out[:, nodes], np.broadcast_to(new, (3, len(xy))), atol=1e-5)
        with pytest.raises(ValueError, match="data_values"):
            sim.simulate(1, seed=1, data_values=np.zeros(len(xy) + 1))


# ------------------------------------------- multiple-point (Direct Sampling)


def _lattice(n):
    xs = np.arange(float(n))
    return np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T


@pytest.fixture(scope="module")
def channel_ti():
    from geolattice.geostatistics import channel_training_image

    return channel_training_image(120, 120, seed=0)


def test_channel_training_image_and_kind_detection(channel_ti):
    from geolattice.geostatistics import DirectSampling, connectivity

    assert channel_ti.shape == (120, 120)
    assert set(np.unique(channel_ti)) == {0.0, 1.0}
    assert 0.05 < channel_ti.mean() < 0.4
    # channels run along x: connected far along axis 0, not along axis 1
    assert connectivity(channel_ti, 1, 30, axis=0)[-1] > 0.6
    assert DirectSampling(channel_ti).kind == "categorical"
    assert DirectSampling(np.random.default_rng(0).normal(size=(20, 20))).kind == "continuous"


def test_pattern_histogram_and_connectivity_on_known_images():
    from geolattice.geostatistics import connectivity, pattern_histogram, pattern_histogram_distance

    cb = (np.add.outer(np.arange(10), np.arange(10)) % 2).astype(float)
    h = pattern_histogram(cb, 2)
    assert set(h) == {(0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)}
    assert sum(h.values()) == pytest.approx(1.0)
    assert pattern_histogram_distance(h, h) == 0.0
    stripes = np.tile([0.0, 1.0], (10, 5))
    assert pattern_histogram_distance(h, pattern_histogram(stripes, 2)) == 1.0
    # a straight full-length stripe is connected at every lag along it
    img = np.zeros((20, 9))
    img[:, 4] = 1
    np.testing.assert_allclose(connectivity(img, 1, 10, axis=0), 1.0)
    assert np.isnan(connectivity(img, 1, 3, axis=1)).all()
    with pytest.raises(ValueError):
        pattern_histogram(cb, 11)


def test_dssim_reproduces_deterministic_checkerboard():
    from geolattice.geostatistics import DirectSampling

    cb = (np.add.outer(np.arange(40), np.arange(40)) % 2).astype(float)
    lat = DirectSampling(cb, n_neighbors=8, threshold=0.0).fit(_lattice(30)).simulate_lattice(2, seed=0)
    differ = ((lat[:, 1:] != lat[:, :-1]).mean() + (lat[:, :, 1:] != lat[:, :, :-1]).mean()) / 2
    # only boundaries between independently started phase domains can break the pattern
    assert differ > 0.95


def test_dssim_reproduces_channel_patterns(channel_ti):
    from geolattice.geostatistics import (
        DirectSampling, connectivity, pattern_histogram, pattern_histogram_distance,
    )

    lat = DirectSampling(channel_ti).fit(_lattice(70)).simulate_lattice(2, seed=1)
    ref = pattern_histogram(channel_ti, 3)
    rng = np.random.default_rng(0)
    for real in lat:
        assert set(np.unique(real)) <= {0.0, 1.0}
        shuffled = rng.permutation(real.ravel()).reshape(real.shape)
        d_real = pattern_histogram_distance(ref, pattern_histogram(real, 3))
        d_shuf = pattern_histogram_distance(ref, pattern_histogram(shuffled, 3))
        assert d_real < 0.2 and d_real < d_shuf / 3
        assert connectivity(shuffled, 1, 20, axis=0)[-1] < 0.05
    conn = [connectivity(real, 1, 20, axis=0)[-1] for real in lat]
    assert np.mean(conn) > 0.5 and min(conn) > 0.3


def _gamma(img, h, axis):
    a, b = (img[:-h], img[h:]) if axis == 0 else (img[:, :-h], img[:, h:])
    return 0.5 * np.mean((a - b) ** 2)


@pytest.fixture(scope="module")
def continuous_ti():
    from geolattice.geostatistics import FFTSimulator

    return FFTSimulator([0, 0, 40, 10, 1.0, "exponential"]).fit(_lattice(80)).simulate_lattice(1, seed=0)[0]


def test_dssim_continuous_copies_ti_values_and_variogram(continuous_ti):
    from geolattice.geostatistics import DirectSampling

    ti = continuous_ti
    lat = DirectSampling(ti, threshold=0.02).fit(_lattice(50)).simulate_lattice(2, seed=1)
    assert np.isin(lat, ti).all()
    for img in (ti, *lat):                      # longer continuity along x, as in the TI
        assert _gamma(img, 4, 0) < 0.5 * _gamma(img, 4, 1)
    for h in (4, 8):
        real = np.mean([_gamma(l, h, 0) for l in lat])
        assert real == pytest.approx(_gamma(ti, h, 0), rel=0.3)
    # Direct Sampling leaves some extra roughness at the finest lag (documented)
    lag1 = np.mean([_gamma(l, 1, 0) for l in lat])
    assert 0.5 * _gamma(ti, 1, 0) < lag1 < 2.0 * _gamma(ti, 1, 0)


def test_dssim_postprocessing_reduces_fine_scale_noise(continuous_ti):
    from geolattice.geostatistics import DirectSampling

    grid = _lattice(40)
    noise = []
    for n_post in (0, 1):
        lat = DirectSampling(continuous_ti, threshold=0.02, n_postprocess=n_post).fit(grid)
        noise.append(np.mean([_gamma(l, 1, 0) for l in lat.simulate_lattice(3, seed=4)]))
    assert noise[1] < 0.85 * noise[0]
    with pytest.raises(ValueError):
        DirectSampling(continuous_ti, n_postprocess=-1)


@pytest.mark.parametrize("kind", ["categorical", "continuous"])
def test_dssim_honours_hard_data(channel_ti, kind):
    import pandas as pd

    from geolattice.geostatistics import dssim

    grid = _lattice(40)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 40, replace=False)
    if kind == "categorical":
        ti, z = channel_ti, rng.choice([0.0, 1.0], 40)
    else:
        ti, z = rng.normal(size=(60, 60)), rng.normal(size=40)
    df = pd.DataFrame({"X": grid[pick, 0] + 0.3, "Y": grid[pick, 1] - 0.4, "Z": z})
    sims = dssim(grid, df, "X", "Y", "Z", ti, nsim=2, kind=kind, seed=2)
    assert sims.shape == (2, len(grid))
    np.testing.assert_allclose(sims[:, pick], np.broadcast_to(z, (2, 40)))


def test_dssim_seed_lattice_extension_and_validation(channel_ti):
    import pandas as pd

    from geolattice.geostatistics import DirectSampling

    grid = _lattice(20)
    a = DirectSampling(channel_ti).fit(grid).simulate(2, seed=5)
    b = DirectSampling(channel_ti).fit(grid).simulate(2, seed=5)
    np.testing.assert_array_equal(a, b)
    ds = DirectSampling(channel_ti).fit(grid, pd.DataFrame({"X": [25.0], "Y": [3.0], "Z": [1.0]}))
    assert ds.lattice_["nx"] == 26 and ds.simulate(1, seed=0).shape == (1, len(grid))
    with pytest.raises(ValueError, match="not categories"):
        DirectSampling(channel_ti).fit(grid, pd.DataFrame({"X": [1.0], "Y": [1.0], "Z": [7.0]}))
    with pytest.raises(ValueError):
        DirectSampling(np.zeros(10))
    with pytest.raises(ValueError):
        DirectSampling(channel_ti, threshold=1.5)
    with pytest.raises(RuntimeError):
        DirectSampling(channel_ti).simulate()
    # NaN cells of the TI are never pasted
    ti = channel_ti.copy()
    ti[:, :60] = np.nan
    assert not np.isnan(DirectSampling(ti).fit(grid).simulate(1, seed=1)).any()


# ------------------------------------------------------------------- SNESIM


def test_snesim_pattern_tables_count_every_ti_cell(channel_ti):
    from geolattice.geostatistics import SNESIM

    sim = SNESIM(channel_ti, n_template=20, n_multigrid=3)
    assert len(sim._template) == 20
    d = np.hypot(*sim._template.T)
    assert np.all(np.diff(d) >= 0)                       # nearest first
    for pats, counts in sim._levels:
        assert counts.sum() == channel_ti.size           # one replicate per TI centre
        assert len(pats) == len(np.unique(pats, axis=0))
    np.testing.assert_allclose(sim.target_proportions_, [1 - channel_ti.mean(), channel_ti.mean()])


def test_snesim_conditional_probabilities_from_stripes():
    from geolattice.geostatistics import SNESIM

    stripes = np.tile([0.0, 1.0], (12, 6))               # value = iy % 2
    sim = SNESIM(stripes, n_template=4, n_multigrid=1, min_replicates=1)
    col = int(np.flatnonzero((sim._template == [0, 1]).all(axis=1))[0])
    # neighbour at +1 in y is 1  ->  centre must be 0
    np.testing.assert_allclose(sim._probabilities(0, np.array([1], np.uint8), np.array([col])), [1.0, 0.0])
    np.testing.assert_allclose(sim._probabilities(0, np.array([0], np.uint8), np.array([col])), [0.0, 1.0])
    # no data event -> target proportions
    np.testing.assert_allclose(sim._probabilities(0, np.array([], np.uint8), np.array([], int)), [0.5, 0.5])


def test_snesim_reproduces_checkerboard_with_multigrid():
    from geolattice.geostatistics import SNESIM

    cb = (np.add.outer(np.arange(40), np.arange(40)) % 2).astype(float)
    lat = SNESIM(cb, n_template=40, n_multigrid=2, servo=0.0).fit(_lattice(30)).simulate_lattice(2, seed=0)
    differ = ((lat[:, 1:] != lat[:, :-1]).mean() + (lat[:, :, 1:] != lat[:, :, :-1]).mean()) / 2
    assert differ > 0.93


def test_snesim_reproduces_channel_statistics(channel_ti):
    from geolattice.geostatistics import SNESIM, connectivity, pattern_histogram, pattern_histogram_distance

    lat = SNESIM(channel_ti).fit(_lattice(80)).simulate_lattice(4, seed=1)
    assert set(np.unique(lat)) <= {0.0, 1.0}
    # the servosystem keeps proportions on target
    np.testing.assert_allclose(lat.mean(axis=(1, 2)), channel_ti.mean(), atol=0.025)
    ref = pattern_histogram(channel_ti, 3)
    rng = np.random.default_rng(0)
    for real in lat:
        shuffled = rng.permutation(real.ravel()).reshape(real.shape)
        d = pattern_histogram_distance(ref, pattern_histogram(real, 3))
        assert d < 0.1 and d < pattern_histogram_distance(ref, pattern_histogram(shuffled, 3)) / 3
    assert np.mean([connectivity(real, 1, 20, axis=0)[-1] for real in lat]) > 0.5


def test_snesim_servosystem_follows_target_proportions(channel_ti):
    from geolattice.geostatistics import SNESIM

    grid = _lattice(50)
    base = SNESIM(channel_ti).fit(grid).simulate_lattice(2, seed=3).mean()
    pushed = SNESIM(channel_ti, servo=0.9, target_proportions={0.0: 0.6, 1.0: 0.4})
    assert pushed.fit(grid).simulate_lattice(2, seed=3).mean() > base + 0.05
    np.testing.assert_allclose(pushed.target_proportions_, [0.6, 0.4])


@pytest.fixture(scope="module")
def ordered_ti():
    """3 ordered classes from a truncated Gaussian field: class 0 never touches class 2."""
    from geolattice.geostatistics import FFTSimulator

    z = FFTSimulator([0, 0, 25, 25, 1, "gaussian"]).fit(_lattice(200)).simulate(1, seed=0)[0]
    ti = np.digitize(z.reshape(200, 200), [-0.5, 0.5]).astype(float)
    assert _forbidden_contacts(ti) == 0
    return ti


def _forbidden_contacts(a):
    return int((np.abs(np.diff(a, axis=0)) == 2).sum() + (np.abs(np.diff(a, axis=1)) == 2).sum())


def test_snesim_servosystem_never_revives_impossible_categories(ordered_ti):
    from geolattice.geostatistics import SNESIM

    grid = _lattice(100)
    lat = SNESIM(ordered_ti, servo=0.5).fit(grid).simulate_lattice(2, seed=1)
    assert [_forbidden_contacts(x) for x in lat] == [0, 0]
    # still pulled toward the TI proportions (servo=0 drifts away from them)
    ti_prop = np.array([(ordered_ti == c).mean() for c in range(3)])
    prop = np.array([(lat == c).mean() for c in range(3)])
    free = SNESIM(ordered_ti, servo=0.0).fit(grid).simulate_lattice(2, seed=1)
    free_prop = np.array([(free == c).mean() for c in range(3)])
    err, free_err = np.abs(prop - ti_prop).max(), np.abs(free_prop - ti_prop).max()
    assert err < 0.05 and err < free_err / 2

    target = {0.0: 0.6, 1.0: 0.2, 2.0: 0.2}
    pushed = SNESIM(ordered_ti, servo=0.9, target_proportions=target).fit(_lattice(60))
    lat = pushed.simulate_lattice(2, seed=1)
    assert [_forbidden_contacts(x) for x in lat] == [0, 0]
    np.testing.assert_allclose([(lat == c).mean() for c in range(3)], [0.6, 0.2, 0.2], atol=0.03)


def test_snesim_honours_hard_data_including_relocation(channel_ti):
    import pandas as pd

    from geolattice.geostatistics import snesim

    grid = _lattice(40)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 60, replace=False)      # most are off the coarse grids
    z = rng.choice([0.0, 1.0], 60)
    df = pd.DataFrame({"X": grid[pick, 0] + 0.2, "Y": grid[pick, 1] - 0.3, "facies": z})
    sims = snesim(grid, df, "X", "Y", "facies", channel_ti, nsim=3, seed=2)
    assert sims.shape == (3, len(grid))
    np.testing.assert_array_equal(sims[:, pick], np.broadcast_to(z, (3, 60)))


def test_snesim_seed_nan_ti_and_validation(channel_ti):
    import pandas as pd

    from geolattice.geostatistics import SNESIM

    grid = _lattice(25)
    a = SNESIM(channel_ti).fit(grid).simulate(2, seed=5)
    np.testing.assert_array_equal(a, SNESIM(channel_ti).fit(grid).simulate(2, seed=5))
    ti = channel_ti.copy()
    ti[:40, :] = np.nan
    out = SNESIM(ti).fit(grid).simulate(1, seed=1)
    assert set(np.unique(out)) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="not categories"):
        SNESIM(channel_ti).fit(grid, pd.DataFrame({"X": [1.0], "Y": [1.0], "facies": [3.0]}))
    with pytest.raises(ValueError, match="categorical"):
        SNESIM(np.random.default_rng(0).normal(size=(30, 30)))
    with pytest.raises(ValueError):
        SNESIM(channel_ti, servo=1.0)
    with pytest.raises(RuntimeError):
        SNESIM(channel_ti).simulate()


# ------------------------------------------- sequential indicator co-simulation


def test_markov_bayes_cokriging_limits():
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix
    from geolattice.geostatistics.simulation.sicosim import _cokrige

    v = [0, 0, 30, 30, 0.25, "spherical"]
    R = _make_rotation_matrix(0, 30, 30)
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 15.0]])
    ind = np.array([1.0, 0.0, 1.0])
    c_ii = _cov_block(xy, xy, v, R, same=True)
    c_i0 = _cov_block(xy, np.array([[5.0, 5.0]]), v, R)[:, 0]
    w = np.linalg.solve(c_ii, c_i0)
    sk = 0.4 + w @ (ind - 0.4)
    assert _cokrige(c_ii, c_i0, ind, 0.25, 0.4, None, 0.0) == pytest.approx(sk, abs=1e-8)
    assert _cokrige(c_ii, c_i0, ind, 0.25, 0.4, 0.9, 0.0) == pytest.approx(sk, abs=1e-8)   # B=0: soft ignored
    assert _cokrige(c_ii, c_i0, ind, 0.25, 0.4, 0.9, 1.0) == pytest.approx(0.9, abs=1e-6)  # B=1: soft is hard
    mid = _cokrige(c_ii, c_i0, ind, 0.25, 0.4, 0.9, 0.5)
    assert sk < mid < 0.9
    # no primary neighbours: the collocated soft datum alone (weight 1)
    empty = np.empty((0, 0)), np.empty(0), np.empty(0)
    assert _cokrige(*empty, 0.25, 0.4, 0.7, 0.3) == pytest.approx(0.7)
    assert _cokrige(*empty, 0.25, 0.4, None, 0.0) == pytest.approx(0.4)


def test_soft_calibration_table_and_b():
    from geolattice.geostatistics import SoftCalibration

    rng = np.random.default_rng(0)
    cls = rng.integers(0, 2, 400)
    ind = np.column_stack([cls == 0, cls == 1]).astype(float)
    sec = cls + rng.uniform(0, 0.5, 400)          # separates the classes perfectly
    perfect = SoftCalibration(n_bins=10, smoothing=0.0).fit(ind, sec)
    # only the quantile bin straddling the class gap mixes them: B loses ~1/n_bins
    assert perfect.B_.min() > 1.0 - 1.5 / 10
    np.testing.assert_allclose(perfect.transform([0.01, 1.49]), [[1, 0], [0, 1]])
    assert SoftCalibration(n_bins=4, smoothing=0.0).fit(ind, sec).B_.min() < perfect.B_.min()
    useless = SoftCalibration(n_bins=5).fit(ind, rng.normal(size=400))
    assert useless.B_.max() < 0.15
    np.testing.assert_allclose(useless.table_.sum(axis=1), 1.0)
    assert np.isnan(useless.transform([np.nan])).all()
    with pytest.raises(ValueError):
        SoftCalibration(n_bins=0)


@pytest.fixture(scope="module")
def facies_truth():
    from geolattice.geostatistics import PluriGaussianSimulator, TruncationRule

    xs = np.arange(0.0, 60.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rule = TruncationRule.truncated_gaussian({0: 0.4, 1: 0.35, 2: 0.25})
    truth = PluriGaussianSimulator(rule, [0, 0, 30, 15, 1.0, "spherical"]).fit(grid).simulate(1, seed=4)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 50, replace=False)
    return grid, truth.astype(float), pick, rng


def test_sicosim_accuracy_grows_with_secondary_quality(facies_truth):
    import pandas as pd

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, truth, pick, rng = facies_truth
    vario = [0, 0, 30, 15, 0.24, "spherical"]
    acc, B = [], []
    for noise in (None, 1.5, 0.15):
        df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "facies": truth[pick]})
        kw = {}
        if noise is not None:
            sec = truth + np.random.default_rng(1).normal(0, noise, len(grid))
            df["seis"] = sec[pick]
            kw = dict(secondary=sec, secondary_col="seis")
        sim = IndicatorCoSimulator([0, 1, 2], vario, radius=40).fit(grid, df, "X", "Y", "facies", **kw)
        reals = sim.simulate(8, seed=2)
        assert set(np.unique(reals)) <= {0.0, 1.0, 2.0}
        np.testing.assert_array_equal(reals[:, pick], np.broadcast_to(truth[pick], (8, len(pick))))
        acc.append((reals == truth).mean())
        B.append(sim.B_.mean())
    assert B[0] == 0.0 and B[1] < B[2]
    assert acc[0] < acc[2] - 0.1 and acc[1] < acc[2]




def test_sis_local_means_reproducible():
    """SIS with local prior means is seed-reproducible and stays in [zmin, zmax]."""
    from geolattice.geostatistics import sis_local_means, sklm_cdf

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 5000, size=(28, 2))
    z = 40 + 0.008 * xy[:, 0] + rng.normal(0, 4, size=28)
    thr = np.quantile(z, [0.25, 0.5, 0.75])
    ind = (z[:, None] <= thr[None, :]).astype(float)
    prior = np.tile(ind.mean(axis=0), (len(z), 1))
    models = [{"model": "spherical", "nugget": 0.25, "psill": 0.75, "range": 1800.0}] * 3
    var = ind.var(axis=0)
    nodes = rng.uniform(0, 5000, size=(35, 2))
    prior_n = np.tile(ind.mean(axis=0), (len(nodes), 1))
    kw = dict(
        zmin=float(z.min()),
        zmax=float(z.max()),
        radius=2500.0,
        k_nn=12,
        nsim=3,
        seed=11,
    )
    a = sis_local_means(nodes, prior_n, xy, ind, prior, thr, models, var, **kw)
    b = sis_local_means(nodes, prior_n, xy, ind, prior, thr, models, var, **kw)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 35)
    assert a.min() >= z.min() - 1e-9 and a.max() <= z.max() + 1e-9
    F = sklm_cdf(nodes[:8], prior_n[:8], xy, ind - prior, models, var, radius=2500.0)
    assert F.shape == (8, 3)
    assert np.all((F >= 0.0) & (F <= 1.0))
    assert np.all(np.diff(F, axis=1) >= -1e-12)


def test_sicosim_plain_sis_reproduces_proportions(facies_truth):
    import pandas as pd

    from geolattice.geostatistics import sicosim

    grid, truth, pick, _ = facies_truth
    df = pd.DataFrame({"X": grid[pick, 0] + 0.5, "Y": grid[pick, 1] + 0.5, "facies": truth[pick]})
    props = np.array([(truth[pick] == k).mean() for k in range(3)])
    reals = sicosim(grid, df, "X", "Y", "facies", [0, 1, 2], [0, 0, 30, 15, 0.24, "spherical"],
                    radius=40, nsim=8, seed=3)
    np.testing.assert_allclose([(reals == k).mean() for k in range(3)], props, atol=0.08)


def test_sicosim_soft_probabilities_given_directly(facies_truth):
    import pandas as pd

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, truth, pick, _ = facies_truth
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "facies": truth[pick]})
    soft = np.full((len(grid), 3), 0.05)
    soft[np.arange(len(grid)), truth.astype(int)] = 0.9
    sim = IndicatorCoSimulator([0, 1, 2], [0, 0, 30, 15, 0.24, "spherical"], radius=40)
    reals = sim.fit(grid, df, "X", "Y", "facies", soft=soft, B=[1.0, 1.0, 1.0]).simulate(5, seed=4)
    assert (reals == truth).mean() > 0.8
    sim.fit(grid, df, "X", "Y", "facies", soft=soft, soft_data=soft[pick])
    np.testing.assert_allclose(sim.B_, 0.85)
    with pytest.raises(ValueError, match="B or soft_data"):
        sim.fit(grid, df, "X", "Y", "facies", soft=soft)


def test_sicosim_continuous_with_secondary():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator, IndicatorCoSimulator

    xs = np.arange(0.0, 60.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    gauss = FFTSimulator([0, 0, 30, 15, 1.0, "spherical"]).fit(grid).simulate(1, seed=4)[0]
    truth = np.exp(gauss)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 60, replace=False)
    th = np.quantile(truth[pick], [0.2, 0.4, 0.6, 0.8])
    sec = gauss + rng.normal(0, 0.3, len(grid))
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick], "s": sec[pick]})
    corr = []
    for kw in ({}, dict(secondary=sec, secondary_col="s")):
        sim = IndicatorCoSimulator(th, [0, 0, 30, 15, 0.16, "spherical"], variable="continuous", radius=40)
        reals = sim.fit(grid, df, "X", "Y", "Z", **kw).simulate(6, seed=1)
        assert reals.min() >= truth[pick].min() - 1e-9 and reals.max() <= truth[pick].max() + 1e-9
        np.testing.assert_allclose(reals[:, pick], np.broadcast_to(truth[pick], (6, 60)))
        np.testing.assert_allclose([(reals <= t).mean() for t in th], [0.2, 0.4, 0.6, 0.8], atol=0.1)
        corr.append(np.mean([np.corrcoef(np.log(r), gauss)[0, 1] for r in reals]))
    assert corr[1] > corr[0] + 0.1


def test_sicosim_validation_and_seed(facies_truth):
    import pandas as pd

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, truth, pick, _ = facies_truth
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "facies": truth[pick]})
    vario = [0, 0, 30, 15, 0.24, "spherical"]
    a = IndicatorCoSimulator([0, 1, 2], vario).fit(grid, df, "X", "Y", "facies").simulate(2, seed=7)
    b = IndicatorCoSimulator([0, 1, 2], vario).fit(grid, df, "X", "Y", "facies").simulate(2, seed=7)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="not in classes"):
        IndicatorCoSimulator([0, 1], vario).fit(grid, df, "X", "Y", "facies")
    with pytest.raises(ValueError, match="ascending"):
        IndicatorCoSimulator([2.0, 1.0], vario, variable="continuous")
    with pytest.raises(ValueError, match="one value per grid node"):
        IndicatorCoSimulator([0, 1, 2], vario).fit(grid, df, "X", "Y", "facies", secondary=np.zeros(3))
    with pytest.raises(RuntimeError):
        IndicatorCoSimulator([0, 1, 2], vario).simulate()


# ------------------------------------------- sequential Gaussian co-simulation


def _sgco_module():
    import sys

    import geolattice.geostatistics  # noqa: F401

    return sys.modules["geolattice.geostatistics.simulation.sgcosim"]


def test_collocated_cokriging_limits():
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix

    M = _sgco_module()
    v = [0, 0, 30, 30, 2.0, "spherical"]
    R = _make_rotation_matrix(0, 30, 30)
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 15.0]])
    z = np.array([1.0, -0.5, 0.3])
    t = np.array([5.0, 5.0])
    c_zz = _cov_block(xy, xy, v, R, same=True)
    c_z0 = _cov_block(xy, t[None, :], v, R)[:, 0]
    w = np.linalg.solve(c_zz, c_z0)
    np.testing.assert_allclose(M._collocated_cokrige(c_zz, c_z0, z, 2.0, 1.3, 0.0),
                               [w @ z, 2.0 - w @ c_z0], atol=1e-8)          # rho=0: SK
    est, var = M._collocated_cokrige(np.empty((0, 0)), np.empty(0), np.empty(0), 2.0, 1.3, 0.6)
    assert est == pytest.approx(0.6 * np.sqrt(2.0) * 1.3) and var == pytest.approx(2.0 * (1 - 0.36))
    est, var = M._collocated_cokrige(c_zz, c_z0, z, 2.0, 1.3, 1.0 - 1e-9)
    assert est == pytest.approx(np.sqrt(2.0) * 1.3, abs=1e-3) and var == pytest.approx(0.0, abs=1e-6)
    # the general MM1 system with only the collocated secondary is the same system
    np.testing.assert_allclose(M._mm1_cokrige(xy, z, t[None, :], np.array([1.3]), t, v, R, 0.6),
                               M._collocated_cokrige(c_zz, c_z0, z, 2.0, 1.3, 0.6), atol=1e-8)


@pytest.fixture(scope="module")
def sgco_fields():
    from geolattice.geostatistics import FFTSimulator

    xs = np.arange(0.0, 90.0, 3.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    v = [0, 0, 25, 25, 1.0, "spherical"]
    Z = FFTSimulator(v).fit(grid).simulate(3, seed=3)
    return grid, v, Z


@pytest.mark.parametrize("kind", ["smooth", "noisy"])
def test_sgcosim_reproduces_rho_in_its_regime(sgco_fields, kind):
    """Collocated (m=1) suits smooth secondaries, multi-collocated (m>1) noisy MM1 ones."""
    from geolattice.geostatistics import GaussianCoSimulator

    grid, v, Zs = sgco_fields
    rng = np.random.default_rng(0)
    rho = 0.7
    m = 1 if kind == "smooth" else 25
    corr, ratio = [], []
    for k, Z in enumerate(Zs):
        Y = Z if kind == "smooth" else rho * Z + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(Z))
        sim = GaussianCoSimulator(v, rho=rho, itrans=False, radius=50, secondary_neighbors=m)
        reals = sim.fit(grid, None, secondary=Y).simulate(3, seed=k)
        corr += [np.corrcoef(r, sim.secondary_scores_)[0, 1] for r in reals]
        ratio.append(sim.mm1_nugget_ratio_)
        assert reals.var(axis=1).mean() == pytest.approx(1.0, abs=0.25)
    assert np.mean(corr) == pytest.approx(rho, abs=0.07)
    # the diagnostic tells the two regimes apart
    if kind == "smooth":
        assert np.mean(ratio) < 0.4
    else:
        assert np.mean(ratio) > 0.8


def test_sgcosim_collocated_underreproduces_noisy_secondary(sgco_fields):
    """Documented limitation of the GSLIB collocated scheme on a noisy secondary."""
    from geolattice.geostatistics import GaussianCoSimulator

    grid, v, Zs = sgco_fields
    rng = np.random.default_rng(1)
    Y = 0.7 * Zs[0] + np.sqrt(1 - 0.49) * rng.standard_normal(len(grid))
    corr = {}
    for m in (1, 25):
        sim = GaussianCoSimulator(v, rho=0.7, itrans=False, radius=50, secondary_neighbors=m)
        reals = sim.fit(grid, None, secondary=Y).simulate(3, seed=2)
        corr[m] = np.mean([np.corrcoef(r, sim.secondary_scores_)[0, 1] for r in reals])
    assert corr[1] < 0.65 < corr[25]


def test_sgcosim_conditional_improves_on_sgs():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator, GaussianCoSimulator, sgcosim

    xs = np.arange(0.0, 60.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    v = [0, 0, 30, 15, 1.0, "spherical"]
    Z = FFTSimulator(v).fit(grid).simulate(1, seed=4)[0]
    sec = 0.8 * Z + 0.6 * FFTSimulator(v).fit(grid).simulate(1, seed=9)[0]
    truth = np.exp(Z)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 40, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick], "ai": sec[pick]})
    plain = GaussianCoSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z").simulate(6, seed=1)
    cosim = GaussianCoSimulator(v, radius=40, zmin=0.0).fit(grid, df, "X", "Y", "Z", secondary=sec,
                                                           secondary_col="ai")
    assert 0.6 < cosim.rho_ < 1.0
    reals = cosim.simulate(6, seed=1)
    assert reals.min() >= 0.0
    np.testing.assert_allclose(reals[:, pick], np.broadcast_to(truth[pick], (6, 40)))

    def rank_corr(a, b):
        from scipy.stats import spearmanr

        return spearmanr(a, b)[0]

    assert (np.mean([rank_corr(r, Z) for r in reals])
            > np.mean([rank_corr(r, Z) for r in plain]) + 0.15)
    one = sgcosim(grid, df, "X", "Y", "Z", v, sec, secondary_col="ai", radius=40, nsim=2, seed=3)
    assert one.shape == (2, len(grid))


def test_sgcosim_hierarchical_joint_correlation():
    import pandas as pd
    from scipy.stats import spearmanr

    from geolattice.geostatistics import FFTSimulator, sgcosim_hierarchical

    xs = np.arange(0.0, 60.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    v = [0, 0, 30, 15, 1.0, "spherical"]
    Z = FFTSimulator(v).fit(grid).simulate(1, seed=4)[0]
    S = 0.8 * Z + 0.6 * FFTSimulator(v).fit(grid).simulate(1, seed=9)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 60, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1],
                       "P": np.exp(Z[pick]), "S": np.exp(S[pick])})
    P, Sec = sgcosim_hierarchical(grid, df, "X", "Y", "P", "S", v, v, nsim=4, radius=40, seed=2)
    assert P.shape == Sec.shape == (4, len(grid))
    np.testing.assert_allclose(P[:, pick], np.broadcast_to(df["P"], (4, 60)))
    np.testing.assert_allclose(Sec[:, pick], np.broadcast_to(df["S"], (4, 60)))
    rho_data = spearmanr(df["P"], df["S"])[0]
    realized = [spearmanr(p, s)[0] for p, s in zip(P, Sec)]
    assert np.mean(realized) == pytest.approx(rho_data, abs=0.12)


def test_sgcosim_validation_and_seed(sgco_fields):
    import pandas as pd

    from geolattice.geostatistics import GaussianCoSimulator

    grid, v, Zs = sgco_fields
    with pytest.raises(ValueError, match="rho"):
        GaussianCoSimulator(v, rho=1.5)
    with pytest.raises(ValueError, match="secondary_neighbors"):
        GaussianCoSimulator(v, secondary_neighbors=0)
    with pytest.raises(ValueError, match="hard data"):
        GaussianCoSimulator(v).fit(grid, None, secondary=Zs[0])
    with pytest.raises(ValueError, match="pass rho"):
        GaussianCoSimulator(v, itrans=False).fit(grid, None, secondary=Zs[0])
    with pytest.raises(ValueError, match="no secondary"):
        GaussianCoSimulator(v, rho=0.5, itrans=False).fit(grid, None)
    with pytest.raises(RuntimeError):
        GaussianCoSimulator(v).simulate()
    sim = GaussianCoSimulator(v, rho=0.5, itrans=False).fit(grid, None, secondary=Zs[0])
    np.testing.assert_array_equal(sim.simulate(2, seed=4), sim.simulate(2, seed=4))
    with pytest.raises(ValueError, match="secondary_scores"):
        sim.simulate(2, seed=0, secondary_scores=np.zeros((3, len(grid))))


# ------------------------------------------------ direct sequential simulation


def test_global_distribution_transforms():
    from geolattice.geostatistics import GlobalDistribution

    z = np.array([1.0, 2.0, 3.0, 4.0])
    dist = GlobalDistribution(z, zmin=0.0, zmax=5.0)
    np.testing.assert_allclose(dist.cdf(z), [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(dist.cdf([0.0, 5.0]), [0.0, 1.0])
    np.testing.assert_allclose(dist.from_gaussian(dist.to_gaussian(z)), z, atol=1e-9)
    assert dist.mean_ == pytest.approx(2.5) and dist.var_ == pytest.approx(1.25)
    # declustering weights move the mean and the cdf
    w = GlobalDistribution(z, weights=[3, 1, 1, 1])
    assert w.mean_ == pytest.approx((3 + 2 + 3 + 4) / 6)
    assert w.cdf(1.0) > dist.cdf(1.0)
    with pytest.raises(ValueError, match="bracket"):
        GlobalDistribution(z, zmin=2.0)


def test_dss_lookup_windows_match_targets():
    import sys

    import geolattice.geostatistics  # noqa: F401
    from geolattice.geostatistics import GlobalDistribution

    M = sys.modules["geolattice.geostatistics.simulation.direct_sequential"]
    rng = np.random.default_rng(0)
    dist = GlobalDistribution(np.exp(rng.normal(size=500)))
    table = M._WindowTable(dist)
    for mean, sd in ((1.0, 0.5), (2.0, 1.2), (0.6, 0.2)):
        ym, ys = table.window(mean, sd)
        draws = dist.from_gaussian(ym + ys * rng.standard_normal(200_000))
        assert draws.mean() == pytest.approx(mean, rel=0.05)
        assert draws.std() == pytest.approx(sd, rel=0.08)


@pytest.fixture(scope="module")
def dss_case():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator

    xs = np.arange(0.0, 90.0, 3.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    v1 = [0, 0, 30, 30, 1.0, "spherical"]
    gauss = FFTSimulator(v1).fit(grid).simulate(1, seed=4)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 80, replace=False)
    return grid, xs, gauss, pick


def _gamma_x(field, n, h):
    f = field.reshape(n, n)
    return 0.5 * np.mean((f[h:] - f[:-h]) ** 2)


def test_dss_lookup_reproduces_original_unit_variogram_and_mean(dss_case):
    import pandas as pd

    from geolattice.geostatistics import DirectSequentialSimulator, GaussianCoSimulator
    from geolattice.geostatistics.simulation.lusim import _cov_block
    from geolattice.geostatistics.simulation.sgsim import _make_rotation_matrix

    grid, xs, gauss, pick = dss_case
    truth = np.exp(gauss)                              # strongly skewed
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick]})
    var = df["Z"].var(ddof=0)
    v = [0, 0, 30, 30, var, "exponential"]
    reals = DirectSequentialSimulator(v, radius=60).fit(grid, df, "X", "Y", "Z").simulate(6, seed=1)
    np.testing.assert_allclose(reals[:, pick], np.broadcast_to(truth[pick], (6, 80)))
    assert reals.min() >= truth[pick].min() - 1e-9 and reals.max() <= truth[pick].max() + 1e-9
    assert reals.mean() == pytest.approx(df["Z"].mean(), rel=0.1)
    R = _make_rotation_matrix(0, 30, 30)
    for h in (1, 3, 6):
        model = var - _cov_block(np.zeros((1, 2)), np.array([[3.0 * h, 0.0]]), v, R)[0, 0]
        assert np.mean([_gamma_x(r, len(xs), h) for r in reals]) == pytest.approx(model, rel=0.25)
    # SGS on normal scores does not reproduce the original-unit variogram this well
    sgs = GaussianCoSimulator([0, 0, 30, 30, 1.0, "exponential"], radius=60)
    sg = sgs.fit(grid, df, "X", "Y", "Z").simulate(6, seed=1)
    model1 = var - _cov_block(np.zeros((1, 2)), np.array([[3.0, 0.0]]), v, R)[0, 0]
    dss_err = abs(np.mean([_gamma_x(r, len(xs), 1) for r in reals]) - model1)
    sgs_err = abs(np.mean([_gamma_x(r, len(xs), 1) for r in sg]) - model1)
    assert dss_err < sgs_err


def test_dss_soares_reproduces_symmetric_data_and_warns_on_skew(dss_case):
    import pandas as pd

    from geolattice.geostatistics import DirectSequentialSimulator

    grid, xs, gauss, pick = dss_case
    sym = 10 + 2 * gauss
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": sym[pick]})
    v = [0, 0, 30, 30, df["Z"].var(ddof=0), "exponential"]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no bias warning for near-symmetric data
        sim = DirectSequentialSimulator(v, method="soares", radius=60).fit(grid, df, "X", "Y", "Z")
    reals = sim.simulate(4, seed=1)
    assert reals.mean() == pytest.approx(df["Z"].mean(), rel=0.05)
    # every Soares value is a quantile of the global distribution
    q = np.quantile(reals, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(q, np.quantile(df["Z"], [0.1, 0.5, 0.9]), atol=0.6)
    skewed = df.assign(Z=np.exp(gauss[pick]))
    with pytest.warns(UserWarning, match="biased for skewed"):
        DirectSequentialSimulator(v, method="soares").fit(grid, skewed, "X", "Y", "Z")


def test_dss_cosimulation_improves_with_secondary(dss_case):
    import pandas as pd
    from scipy.stats import spearmanr

    from geolattice.geostatistics import FFTSimulator, dssim_direct

    grid, xs, gauss, pick = dss_case
    truth = np.exp(0.6 * gauss)
    sec = 0.8 * gauss + 0.6 * FFTSimulator([0, 0, 30, 30, 1.0, "spherical"]).fit(grid).simulate(1, seed=9)[0]
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick], "s": sec[pick]})
    v = [0, 0, 30, 30, df["Z"].var(ddof=0), "spherical"]
    plain = dssim_direct(grid, df, "X", "Y", "Z", v, radius=60, nsim=5, seed=1)
    co = dssim_direct(grid, df, "X", "Y", "Z", v, radius=60, nsim=5, seed=1,
                      secondary=sec, secondary_col="s")
    rc = lambda sims: np.mean([spearmanr(r, truth)[0] for r in sims])  # noqa: E731
    assert rc(co) > rc(plain) + 0.1
    assert co.mean() == pytest.approx(df["Z"].mean(), rel=0.1)


def test_dss_validation_and_seed(dss_case):
    import pandas as pd

    from geolattice.geostatistics import DirectSequentialSimulator

    grid, xs, gauss, pick = dss_case
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": np.exp(gauss[pick])})
    v = [0, 0, 30, 30, 1.0, "spherical"]
    with pytest.raises(ValueError, match="method"):
        DirectSequentialSimulator(v, method="sgs")
    with pytest.raises(ValueError, match="no secondary"):
        DirectSequentialSimulator(v, rho=0.5).fit(grid, df, "X", "Y", "Z")
    with pytest.raises(ValueError, match="constant"):
        DirectSequentialSimulator(v).fit(grid, df, "X", "Y", "Z", secondary=np.ones(len(grid)))
    with pytest.raises(RuntimeError):
        DirectSequentialSimulator(v).simulate()
    sim = DirectSequentialSimulator(v, radius=60).fit(grid, df, "X", "Y", "Z")
    np.testing.assert_array_equal(sim.simulate(2, seed=3), sim.simulate(2, seed=3))


# ------------------------------------------------------- simulated annealing


@pytest.fixture(scope="module")
def sa_case():
    from geolattice.geostatistics import FFTSimulator

    grid = _lattice(30)
    truth = FFTSimulator([0, 0, 12, 12, 1.0, "spherical"]).fit(grid).simulate(1, seed=2)[0]
    return grid, truth


def test_sasim_incremental_objective_matches_recomputation(sa_case):
    from geolattice.geostatistics import SimulatedAnnealing

    grid, truth = sa_case
    rng = np.random.default_rng(0)
    sec = 0.7 * truth + 0.71 * rng.normal(size=len(grid))
    sa = SimulatedAnnealing(vario=[0, 0, 12, 12, 1.0, "spherical"], n_lags=6, target_values=truth,
                            target_correlation=0.7, max_swaps=20_000)
    lat = sa.fit(grid, secondary=sec).simulate_lattice(1, seed=1)[0]
    sums, szy = sa._tracked
    flat, idx = sa._setup_state(lat)
    np.testing.assert_allclose(sums, sa._gamma_sums(flat, idx), rtol=1e-10)
    assert szy == pytest.approx((lat.ravel() * sa._y.ravel()).sum(), rel=1e-10)


def test_sasim_matches_target_variogram_and_keeps_histogram(sa_case):
    from geolattice.geostatistics import SimulatedAnnealing

    grid, truth = sa_case
    target = np.exp(truth)
    v = [0, 0, 12, 12, target.var(), "spherical"]
    sa = SimulatedAnnealing(vario=v, n_lags=6, target_values=target).fit(grid)
    lat = sa.simulate_lattice(1, seed=1)[0]
    assert sa.objective_[0] <= 1e-3 + 1e-12
    hist = sa.history_[0]
    assert hist[0] == 1.0 and hist[-1] < 0.01
    assert np.isin(lat, target).all()
    exp, tgt = sa.experimental_variogram(lat)[0], sa.target_variogram()[0]
    first_lags = np.arange(0, sa._lags.shape[0], 6)              # lag 1 in each direction
    np.testing.assert_allclose(exp[first_lags], tgt[first_lags], rtol=0.3)
    random_img = np.random.default_rng(0).permutation(lat.ravel()).reshape(lat.shape)
    assert sa.experimental_variogram(random_img)[0][0] > 3 * exp[0]


def test_sasim_correlation_and_hard_data(sa_case):
    import pandas as pd

    from geolattice.geostatistics import sasim

    grid, truth = sa_case
    rng = np.random.default_rng(1)
    sec = 0.6 * truth + 0.8 * rng.normal(size=len(grid))
    pick = rng.choice(len(grid), 25, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick]})
    out = sasim(grid, df, "X", "Y", "Z", vario=[0, 0, 12, 12, 1.0, "spherical"], n_lags=6,
                target_values=truth, secondary=sec, target_correlation=0.6, nsim=2, seed=3)
    assert out.shape == (2, len(grid))
    np.testing.assert_allclose(out[:, pick], np.broadcast_to(truth[pick], (2, 25)))
    for r in out:
        assert np.corrcoef(r, sec)[0, 1] == pytest.approx(0.6, abs=0.05)


def test_sasim_categorical_indicator_variograms(sa_case):
    from geolattice.geostatistics import SimulatedAnnealing

    grid, truth = sa_case
    cats = (truth > 0.5).astype(float) + (truth > -0.3)
    iv = {c: [0, 0, 12, 12, (cats == c).mean() * (1 - (cats == c).mean()), "spherical"]
          for c in (0.0, 1.0, 2.0)}
    sa = SimulatedAnnealing(indicator_varios=iv, n_lags=6, target_values=cats).fit(grid)
    lat = sa.simulate_lattice(1, seed=1)[0]
    assert set(np.unique(lat)) <= {0.0, 1.0, 2.0}
    np.testing.assert_allclose([(lat == c).mean() for c in (0, 1, 2)],
                               [(cats == c).mean() for c in (0, 1, 2)], atol=0.002)
    exp, tgt = sa.experimental_variogram(lat), sa.target_variogram()
    np.testing.assert_allclose(exp[:, 0], tgt[:, 0], rtol=0.3)
    sums, _ = sa._tracked
    flat, idx = sa._setup_state(lat)
    np.testing.assert_allclose(sums, sa._gamma_sums(flat, idx), atol=1e-9)


def test_sasim_validation_and_seed(sa_case):
    from geolattice.geostatistics import SimulatedAnnealing

    grid, truth = sa_case
    v = [0, 0, 12, 12, 1.0, "spherical"]
    with pytest.raises(ValueError, match="exactly one"):
        SimulatedAnnealing()
    with pytest.raises(ValueError, match="exactly one"):
        SimulatedAnnealing(vario=v, indicator_varios={0: v})
    with pytest.raises(ValueError, match="target histogram"):
        SimulatedAnnealing(vario=v).fit(grid)
    with pytest.raises(ValueError, match="secondary"):
        SimulatedAnnealing(vario=v, target_values=truth, target_correlation=0.5).fit(grid)
    with pytest.raises(ValueError, match="longer than the grid"):
        SimulatedAnnealing(vario=v, n_lags=40, target_values=truth).fit(grid).simulate(1, seed=0)
    with pytest.raises(RuntimeError):
        SimulatedAnnealing(vario=v).simulate()
    sa = SimulatedAnnealing(vario=v, n_lags=4, target_values=truth, max_swaps=3000).fit(grid)
    np.testing.assert_array_equal(sa.simulate(1, seed=5), sa.simulate(1, seed=5))


# ----------------------------------------------- multiple-point co-simulation


@pytest.fixture(scope="module")
def mps_cosim_case():
    from scipy.ndimage import gaussian_filter

    from geolattice.geostatistics import channel_training_image

    rng = np.random.default_rng(0)

    def seismic(img):
        return gaussian_filter(img, 2.0) + rng.normal(0, 0.08, img.shape)

    ti = channel_training_image(120, 120, seed=0)
    truth = channel_training_image(60, 60, seed=7)
    grid = _lattice(60)
    gi = grid.astype(int)
    return ti, seismic(ti), grid, truth[gi[:, 0], gi[:, 1]], seismic(truth)[gi[:, 0], gi[:, 1]]


def _recall(reals, truth):
    return (reals[:, truth == 1] == 1).mean()


def test_tau_model_limits():
    from geolattice.geostatistics import tau_model

    prior = np.array([0.7, 0.3])
    p_ti = np.array([0.4, 0.6])
    p_soft = np.array([0.2, 0.8])
    np.testing.assert_allclose(tau_model(prior, p_ti, p_soft, (1.0, 0.0)), p_ti, atol=1e-6)
    np.testing.assert_allclose(tau_model(prior, p_ti, p_soft, (0.0, 1.0)), p_soft, atol=1e-6)
    np.testing.assert_allclose(tau_model(prior, p_ti, prior), p_ti, atol=1e-6)   # uninformative soft
    both = tau_model(prior, p_ti, p_soft)
    assert both[1] > max(p_ti[1], p_soft[1])          # two agreeing sources reinforce
    np.testing.assert_allclose(both.sum(), 1.0)
    assert tau_model(prior, p_ti, np.array([0.0, 1.0]))[1] > 0.999


def test_direct_sampling_cosimulation_beats_plain(mps_cosim_case):
    from geolattice.geostatistics import DirectSampling

    ti, sti, grid, truth, sec = mps_cosim_case
    plain = DirectSampling(ti).fit(grid).simulate(2, seed=1)
    co = DirectSampling(ti, secondary_ti=sti, secondary_weight=0.5).fit(grid, secondary=sec)
    reals = co.simulate(2, seed=1)
    assert co.secondary_kind == "continuous"
    assert set(np.unique(reals)) <= {0.0, 1.0}
    assert (reals == truth).mean() > (plain == truth).mean() + 0.15
    assert _recall(reals, truth) > 0.7 > _recall(plain, truth)


def test_direct_sampling_cosimulation_honours_data_and_validates(mps_cosim_case):
    import pandas as pd

    from geolattice.geostatistics import DirectSampling, dssim

    ti, sti, grid, truth, sec = mps_cosim_case
    rng = np.random.default_rng(2)
    pick = rng.choice(len(grid), 30, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": truth[pick]})
    out = dssim(grid, df, "X", "Y", "Z", ti, nsim=2, seed=3, secondary=sec, secondary_ti=sti)
    np.testing.assert_array_equal(out[:, pick], np.broadcast_to(truth[pick], (2, 30)))
    with pytest.raises(ValueError, match="same shape"):
        DirectSampling(ti, secondary_ti=sti[:10])
    with pytest.raises(ValueError, match="pass the exhaustive secondary"):
        DirectSampling(ti, secondary_ti=sti).fit(grid)
    with pytest.raises(ValueError, match="needs a secondary_ti"):
        DirectSampling(ti).fit(grid, secondary=sec)
    with pytest.raises(ValueError, match="secondary_weight"):
        DirectSampling(ti, secondary_ti=sti, secondary_weight=1.5)


def test_snesim_tau_cosimulation_beats_plain(mps_cosim_case):
    from geolattice.geostatistics import SNESIM, SoftCalibration

    ti, sti, grid, truth, sec = mps_cosim_case
    ind = np.column_stack([ti.ravel() == 0, ti.ravel() == 1]).astype(float)
    soft = SoftCalibration(n_bins=15).fit(ind, sti.ravel()).transform(sec)
    plain = SNESIM(ti).fit(grid).simulate(2, seed=1)
    sim = SNESIM(ti)
    reals = sim.fit(grid, soft=soft).simulate(2, seed=1)
    # without user target proportions, the soft mean becomes the target
    np.testing.assert_allclose(sim.target_proportions_, soft.mean(axis=0))
    assert (reals == truth).mean() > (plain == truth).mean() + 0.1
    assert _recall(reals, truth) > _recall(plain, truth) + 0.2
    assert reals.mean() == pytest.approx(soft[:, 1].mean(), abs=0.05)
    # refitting without soft data restores the TI proportions
    sim.fit(grid)
    np.testing.assert_allclose(sim.target_proportions_, [1 - ti.mean(), ti.mean()])


def test_snesim_soft_validation(mps_cosim_case):
    from geolattice.geostatistics import SNESIM

    ti, sti, grid, truth, sec = mps_cosim_case
    with pytest.raises(ValueError, match="shape"):
        SNESIM(ti).fit(grid, soft=np.full((5, 2), 0.5))
    with pytest.raises(ValueError, match="summing to 1"):
        SNESIM(ti).fit(grid, soft=np.full((len(grid), 2), 0.7))
    with pytest.raises(ValueError, match="tau"):
        SNESIM(ti, tau=(1.0, -1.0))


# ------------------------------------------------------- MPS GPU acceleration


def _gpu_usable():
    try:
        from geolattice.geostatistics.utils.backend import cupy_gpu_usable

        return cupy_gpu_usable()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_usable(), reason="no usable CUDA GPU / CuPy")


@pytest.fixture(scope="module")
def gpu_case(channel_ti):
    grid = _lattice(40)
    return channel_ti, grid, np.arange(5) + 10


def test_batched_snesim_reference_equals_sequential(gpu_case):
    """The vectorized (NumPy) batched engine draws exactly what the sequential one does."""
    from geolattice.geostatistics import SNESIM
    from geolattice.geostatistics import mps_gpu as G

    ti, _, _ = gpu_case
    grid, seeds = _lattice(24), np.arange(3) + 10
    sn = SNESIM(ti, n_template=24, n_multigrid=3).fit(grid)
    seq = np.stack([sn._simulate_one(np.random.default_rng(int(s))) for s in seeds])
    np.testing.assert_array_equal(G.snesim_simulate_batched(sn, seeds, np), seq)


def test_batched_direct_sampling_reference_matches_sequential_statistics(gpu_case):
    from geolattice.geostatistics import DirectSampling, pattern_histogram, pattern_histogram_distance
    from geolattice.geostatistics import mps_gpu as G

    ti, grid, seeds = gpu_case
    ds = DirectSampling(ti).fit(grid)
    seq = np.stack([ds._simulate_one(np.random.default_rng(int(s))) for s in seeds])
    bat = G.ds_simulate_batched(ds, seeds, np)
    ref = pattern_histogram(ti, 3)
    d_seq = np.mean([pattern_histogram_distance(ref, pattern_histogram(x, 3)) for x in seq])
    d_bat = np.mean([pattern_histogram_distance(ref, pattern_histogram(x, 3)) for x in bat])
    assert abs(d_seq - d_bat) < 0.05
    assert abs(seq.mean() - bat.mean()) < 0.05
    assert np.isin(bat, ti).all()


@needs_gpu
def test_gpu_snesim_identical_to_cpu(gpu_case):
    from geolattice.geostatistics import SNESIM

    ti, grid, _ = gpu_case
    cpu = SNESIM(ti).fit(grid).simulate(6, seed=3)
    gpu = SNESIM(ti, use_gpu=True).fit(grid).simulate(6, seed=3)
    np.testing.assert_array_equal(gpu, cpu)


def test_batched_snesim_reference_masked_servo_equals_sequential(ordered_ti):
    from geolattice.geostatistics import SNESIM
    from geolattice.geostatistics import mps_gpu as G

    seeds = np.arange(3) + 10
    sn = SNESIM(ordered_ti, n_template=24, n_multigrid=3, servo=0.5).fit(_lattice(30))
    seq = np.stack([sn._simulate_one(np.random.default_rng(int(s))) for s in seeds])
    np.testing.assert_array_equal(G.snesim_simulate_batched(sn, seeds, np), seq)


@needs_gpu
def test_gpu_snesim_masked_servo_identical_to_cpu(ordered_ti):
    from geolattice.geostatistics import SNESIM

    grid = _lattice(50)
    cpu = SNESIM(ordered_ti, servo=0.5).fit(grid).simulate(4, seed=1)
    gpu = SNESIM(ordered_ti, servo=0.5, use_gpu=True).fit(grid).simulate(4, seed=1)
    np.testing.assert_array_equal(gpu, cpu)
    assert all(_forbidden_contacts(x.reshape(50, 50)) == 0 for x in gpu)


@needs_gpu
def test_gpu_direct_sampling_kernel_matches_reference(gpu_case):
    import cupy

    from geolattice.geostatistics import DirectSampling
    from geolattice.geostatistics import mps_gpu as G

    ti, grid, seeds = gpu_case
    ds = DirectSampling(ti).fit(grid)
    ref = G.ds_simulate_batched(ds, seeds, np)
    ker = G.ds_simulate_batched(ds, seeds, cupy)
    # only floating-point summation order can flip a near-threshold choice
    assert (ker == ref).mean() > 0.99


@needs_gpu
@pytest.mark.parametrize("kind", ["categorical", "continuous"])
def test_gpu_direct_sampling_honours_data(gpu_case, kind):
    import pandas as pd

    from geolattice.geostatistics import DirectSampling

    ti, grid, _ = gpu_case
    rng = np.random.default_rng(0)
    pick = rng.choice(len(grid), 30, replace=False)
    if kind == "continuous":
        ti = rng.normal(size=(60, 60))
        z = rng.normal(size=30)
    else:
        z = rng.choice([0.0, 1.0], 30)
    df = pd.DataFrame({"X": grid[pick, 0], "Y": grid[pick, 1], "Z": z})
    out = DirectSampling(ti, kind=kind, use_gpu=True).fit(grid, df).simulate(3, seed=1)
    np.testing.assert_allclose(out[:, pick], np.broadcast_to(z, (3, 30)))
    assert np.isin(out, np.concatenate([ti.ravel(), z])).all()


@needs_gpu
def test_gpu_cosimulation_matches_cpu_quality(mps_cosim_case):
    from geolattice.geostatistics import SNESIM, DirectSampling, SoftCalibration

    ti, sti, grid, truth, sec = mps_cosim_case
    acc = {}
    for gpu in (False, True):
        ds = DirectSampling(ti, secondary_ti=sti, use_gpu=gpu).fit(grid, secondary=sec)
        acc[gpu] = (ds.simulate(3, seed=1) == truth).mean()
    assert abs(acc[True] - acc[False]) < 0.03
    ind = np.column_stack([ti.ravel() == 0, ti.ravel() == 1]).astype(float)
    soft = SoftCalibration(n_bins=15).fit(ind, sti.ravel()).transform(sec)
    cpu = SNESIM(ti).fit(grid, soft=soft).simulate(3, seed=2)
    gpu = SNESIM(ti, use_gpu=True).fit(grid, soft=soft).simulate(3, seed=2)
    np.testing.assert_array_equal(gpu, cpu)


@needs_gpu
def test_gpu_is_faster_for_many_realizations(gpu_case):
    import time

    from geolattice.geostatistics import SNESIM, DirectSampling

    ti, grid, _ = gpu_case
    for cls in (SNESIM, DirectSampling):
        cpu = cls(ti).fit(grid)
        t0 = time.perf_counter()
        cpu.simulate(4, seed=1)
        t_cpu4 = time.perf_counter() - t0
        gpu = cls(ti, use_gpu=True).fit(grid)
        gpu.simulate(2, seed=0)                         # compile / warm up
        t0 = time.perf_counter()
        gpu.simulate(32, seed=1)
        assert time.perf_counter() - t0 < t_cpu4        # 32 on GPU beat 4 on CPU


@needs_gpu
def test_gpu_limits_are_checked(gpu_case):
    from geolattice.geostatistics import SNESIM, DirectSampling

    ti, grid, _ = gpu_case
    with pytest.raises(ValueError, match="n_neighbors"):
        DirectSampling(ti, n_neighbors=200, use_gpu=True).fit(grid).simulate(1, seed=0)
    with pytest.raises(ValueError, match="n_template"):
        SNESIM(ti, n_template=80, search_radius=6, n_multigrid=1, use_gpu=True).fit(grid).simulate(1, seed=0)


def test_use_gpu_falls_back_without_cuda(gpu_case, monkeypatch):
    from geolattice.geostatistics import SNESIM
    from geolattice.geostatistics.utils import backend

    ti, grid, _ = gpu_case
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: False)
    with pytest.warns(UserWarning, match="falling back to CPU"):
        out = SNESIM(ti, use_gpu=True).fit(grid).simulate(1, seed=0)
    np.testing.assert_array_equal(out, SNESIM(ti).fit(grid).simulate(1, seed=0))
    assert SNESIM(ti, use_gpu="auto").fit(grid).simulate(1, seed=0).shape == out.shape


# ----------------------------------------------- GPU sequential Gaussian sim


@pytest.fixture(scope="module")
def sgs_gpu_case():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator

    xs = np.arange(0.0, 50.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.uniform(0, 48, 30), "Y": rng.uniform(0, 48, 30),
                       "Z": np.exp(rng.normal(size=30))})
    df.loc[0, ["X", "Y"]] = [10.0, 20.0]                 # one datum on a node
    sec = FFTSimulator([0, 0, 30, 15, 1.0, "spherical"]).fit(grid).simulate(1, seed=4)[0]
    return grid, df, sec


def test_sgs_ordinary_kriging_cpu(sgs_gpu_case):
    from geolattice.geostatistics import GaussianCoSimulator

    grid, df, sec = sgs_gpu_case
    v = [0, 0.0, 30, 15, 1.0, "spherical"]
    reals = GaussianCoSimulator(v, kriging_type="ordinary", radius=40).fit(
        grid, df, "X", "Y", "Z").simulate(4, seed=1)
    node = np.flatnonzero((grid == [10.0, 20.0]).all(axis=1))[0]
    np.testing.assert_allclose(reals[:, node], df.loc[0, "Z"])
    with pytest.raises(ValueError, match="ordinary"):
        GaussianCoSimulator(v, kriging_type="ordinary", rho=0.5).fit(
            grid, df, "X", "Y", "Z", secondary=sec)
    with pytest.raises(ValueError, match="kriging_type"):
        GaussianCoSimulator(v, kriging_type="universal")


@needs_gpu
@pytest.mark.parametrize("vtype", ["spherical", "exponential", "gaussian"])
@pytest.mark.parametrize("mode", ["simple", "ordinary", "collocated", "per_realization"])
def test_gpu_sgs_equals_cpu(sgs_gpu_case, vtype, mode):
    from geolattice.geostatistics import GaussianCoSimulator

    grid, df, sec = sgs_gpu_case
    v = [20, 0.0 if vtype == "gaussian" else 0.1, 30, 15, 1.0, vtype]
    kw, fk, sim_kw = {}, {}, {}
    if mode == "ordinary":
        kw = dict(kriging_type="ordinary")
    elif mode in ("collocated", "per_realization"):
        kw, fk = dict(rho=0.6), dict(secondary=sec)
    sim = GaussianCoSimulator(v, radius=40, **kw).fit(grid, df, "X", "Y", "Z", **fk)
    if mode == "per_realization":
        sim_kw = dict(secondary_scores=np.random.default_rng(3).normal(size=(4, len(grid))))
    cpu = sim.simulate(4, seed=1, **sim_kw)
    sim.use_gpu = True
    gpu = sim.simulate(4, seed=1, **sim_kw)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-7 * np.ptp(cpu))


@needs_gpu
def test_gpu_sgs_batches_and_hierarchical(sgs_gpu_case, monkeypatch):
    import cupy

    from geolattice.geostatistics import GaussianCoSimulator, sgsim_gpu

    grid, df, _ = sgs_gpu_case
    v = [0, 0, 30, 15, 1.0, "spherical"]
    sim = GaussianCoSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z")
    cpu = sim.simulate(5, seed=2)
    # pretend only enough memory for ~2 realizations per batch
    real = cupy.cuda.runtime.memGetInfo
    n_loc = len(sim._loc)
    per = 8 * n_loc + 12 * len(grid) + 8 * 50 ** 2
    monkeypatch.setattr(cupy.cuda.runtime, "memGetInfo", lambda: (int(2.2 * per / 0.6), real()[1]))
    sim.use_gpu = True
    np.testing.assert_allclose(sim.simulate(5, seed=2), cpu, atol=1e-7 * np.ptp(cpu))
    monkeypatch.undo()
    out = sgsim_gpu(grid, df, "X", "Y", "Z", v, radius=40, nsim=3, seed=4)
    np.testing.assert_allclose(out, GaussianCoSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z").simulate(3, seed=4),
                               atol=1e-7 * np.ptp(out))


@needs_gpu
def test_gpu_sgs_limits(sgs_gpu_case):
    from geolattice.geostatistics import GaussianCoSimulator

    grid, df, sec = sgs_gpu_case
    with pytest.raises(ValueError, match="num_points"):
        GaussianCoSimulator([0, 0, 30, 15, 1.0, "spherical"], num_points=60, use_gpu=True).fit(
            grid, df, "X", "Y", "Z").simulate(1, seed=0)
    with pytest.raises(ValueError, match="variograms"):
        GaussianCoSimulator([0, 0, 30, 15, 1.0, "matern", 1.5], use_gpu=True).fit(
            grid, df, "X", "Y", "Z").simulate(1, seed=0)


def test_sgs_gpu_out_of_memory_falls_back_to_cpu(sgs_gpu_case, monkeypatch):
    import sys

    from geolattice.geostatistics import GaussianCoSimulator

    grid, df, _ = sgs_gpu_case
    v = [0, 0, 30, 15, 1.0, "spherical"]
    cpu = GaussianCoSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z").simulate(2, seed=1)
    import geolattice.geostatistics.gpu.sgs_gpu as _sgs_gpu_mod  # noqa: F401
    mod = sys.modules["geolattice.geostatistics.gpu.sgs_gpu"]
    backend = sys.modules["geolattice.geostatistics.utils.backend"]

    def boom(*a, **k):
        raise MemoryError("out of memory allocating 81,920,000 bytes")

    monkeypatch.setattr(mod, "sgs_simulate_gpu", boom)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)
    sim = GaussianCoSimulator(v, radius=40, use_gpu=True).fit(grid, df, "X", "Y", "Z")
    with pytest.warns(UserWarning, match="running on the CPU"):
        out = sim.simulate(2, seed=1)
    np.testing.assert_array_equal(out, cpu)


# ----------------------------------------------- GPU sequential indicator sim


@pytest.fixture(scope="module")
def sis_gpu_case():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator

    xs = np.arange(0.0, 44.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(0)
    truth = FFTSimulator([0, 0, 30, 15, 1.0, "spherical"]).fit(grid).simulate(1, seed=4)[0]
    cats = (truth > 0.5).astype(float) + (truth > -0.3)
    pick = rng.choice(len(grid), 35, replace=False)
    df = pd.DataFrame({"X": grid[pick, 0] + 0.4, "Y": grid[pick, 1] - 0.3, "facies": cats[pick],
                       "Z": np.exp(truth[pick])})
    df.loc[0, ["X", "Y"]] = grid[pick[0]]                # one datum on a node
    sec = cats + rng.normal(0, 0.6, len(grid))
    df["seis"] = sec[pick]
    return grid, df, sec, truth


_SIS_IV = {0.0: [0, 0.02, 30, 15, 0.24, "spherical"], 1.0: [30, 0.0, 25, 20, 0.22, "exponential"],
           2.0: [0, 0.0, 30, 15, 0.16, "gaussian"]}


def _sis_cases(df, sec, truth, grid):
    th = np.quantile(df["Z"], [0.2, 0.4, 0.6, 0.8])
    rng = np.random.default_rng(1)
    return {
        "categorical": (dict(classes=[0, 1, 2], vario=_SIS_IV), dict(zz="facies")),
        "categorical_soft": (dict(classes=[0, 1, 2], vario=_SIS_IV),
                             dict(zz="facies", secondary=sec, secondary_col="seis")),
        "continuous": (dict(classes=th, vario=[0, 0, 30, 15, 0.16, "spherical"], variable="continuous"),
                       dict(zz="Z")),
        "continuous_tails": (dict(classes=th, vario=[0, 0, 30, 15, 0.16, "exponential"],
                                  variable="continuous", ltail="power", ltpar=2.0, middle="power",
                                  mpar=1.5, utail="hyperbolic", utpar=1.5, zmin=0.0), dict(zz="Z")),
        "continuous_soft": (dict(classes=th, vario=[0, 0, 30, 15, 0.16, "spherical"], variable="continuous"),
                            dict(zz="Z", secondary=truth + rng.normal(0, 0.3, len(grid)))),
    }


@needs_gpu
@pytest.mark.parametrize("case", ["categorical", "categorical_soft", "continuous",
                                  "continuous_tails", "continuous_soft"])
def test_gpu_sis_equals_cpu(sis_gpu_case, case):
    from geolattice.geostatistics import IndicatorCoSimulator

    grid, df, sec, truth = sis_gpu_case
    ck, fk = _sis_cases(df, sec, truth, grid)[case]
    fk = dict(fk)
    zz = fk.pop("zz")
    sim = IndicatorCoSimulator(radius=40, **ck).fit(grid, df, "X", "Y", zz, **fk)
    cpu = sim.simulate(4, seed=1)
    sim.use_gpu = True
    gpu = sim.simulate(4, seed=1)
    if case.startswith("categorical"):
        np.testing.assert_array_equal(gpu, cpu)
    else:
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-9 * np.ptp(cpu))
    node = np.flatnonzero((grid == grid[np.argmin(np.abs(grid - df.loc[0, ["X", "Y"]].to_numpy(float)).sum(1))]).all(1))[0]
    np.testing.assert_allclose(gpu[:, node], df.loc[0, zz])


@needs_gpu
def test_gpu_sis_batching_and_limits(sis_gpu_case, monkeypatch):
    import cupy

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, df, sec, truth = sis_gpu_case
    sim = IndicatorCoSimulator([0, 1, 2], _SIS_IV, radius=40).fit(grid, df, "X", "Y", "facies")
    cpu = sim.simulate(5, seed=2)
    real = cupy.cuda.runtime.memGetInfo
    per = 8 * len(sim._loc) + 12 * len(grid) + 8 * 49 ** 2
    monkeypatch.setattr(cupy.cuda.runtime, "memGetInfo", lambda: (int(2.2 * per / 0.6), real()[1]))
    sim.use_gpu = True
    np.testing.assert_array_equal(sim.simulate(5, seed=2), cpu)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="num_points"):
        IndicatorCoSimulator([0, 1, 2], _SIS_IV, num_points=60, use_gpu=True).fit(
            grid, df, "X", "Y", "facies").simulate(1, seed=0)
    with pytest.raises(ValueError, match="variograms"):
        IndicatorCoSimulator([0, 1, 2], [0, 0, 30, 15, 0.2, "matern", 1.5], use_gpu=True).fit(
            grid, df, "X", "Y", "facies").simulate(1, seed=0)


@needs_gpu
def test_gpu_sis_faster_for_many_realizations(sis_gpu_case):
    import time

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, df, _, _ = sis_gpu_case
    cpu = IndicatorCoSimulator([0, 1, 2], _SIS_IV, radius=40).fit(grid, df, "X", "Y", "facies")
    t0 = time.perf_counter()
    cpu.simulate(4, seed=1)
    t_cpu4 = time.perf_counter() - t0
    gpu = IndicatorCoSimulator([0, 1, 2], _SIS_IV, radius=40, use_gpu=True).fit(grid, df, "X", "Y", "facies")
    gpu.simulate(2, seed=0)
    t0 = time.perf_counter()
    gpu.simulate(128, seed=1)
    assert time.perf_counter() - t0 < 4 * t_cpu4         # 128 on GPU well under 16 on CPU


def test_sis_gpu_out_of_memory_falls_back_to_cpu(sis_gpu_case, monkeypatch):
    import sys

    from geolattice.geostatistics import IndicatorCoSimulator

    grid, df, _, _ = sis_gpu_case
    cpu = IndicatorCoSimulator([0, 1, 2], _SIS_IV, radius=40).fit(grid, df, "X", "Y", "facies").simulate(2, seed=1)
    import geolattice.geostatistics.gpu.sis_gpu  # noqa: F401
    mod = sys.modules["geolattice.geostatistics.gpu.sis_gpu"]
    backend = sys.modules["geolattice.geostatistics.utils.backend"]

    def boom(*a, **k):
        raise MemoryError("out of memory allocating 81,920,000 bytes")

    monkeypatch.setattr(mod, "sis_simulate_gpu", boom)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)
    sim = IndicatorCoSimulator([0, 1, 2], _SIS_IV, radius=40, use_gpu=True).fit(grid, df, "X", "Y", "facies")
    with pytest.warns(UserWarning, match="running on the CPU"):
        out = sim.simulate(2, seed=1)
    np.testing.assert_array_equal(out, cpu)


# ----------------------------------------------- GPU direct sequential sim


@pytest.fixture(scope="module")
def dss_gpu_case():
    import pandas as pd

    from geolattice.geostatistics import FFTSimulator

    xs = np.arange(0.0, 44.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(0)
    truth = FFTSimulator([0, 0, 30, 15, 1.0, "spherical"]).fit(grid).simulate(1, seed=4)[0]
    pick = rng.choice(len(grid), 35, replace=False)
    sec = 0.8 * truth + 0.6 * rng.normal(size=len(grid))
    df = pd.DataFrame({"X": grid[pick, 0] + 0.4, "Y": grid[pick, 1] - 0.3,
                       "Z": np.exp(truth[pick]), "s": sec[pick]})
    df.loc[0, ["X", "Y"]] = grid[pick[0]]                # one datum on a node
    return grid, df, sec, pick


@needs_gpu
@pytest.mark.parametrize("method", ["lookup", "soares"])
@pytest.mark.parametrize("vtype", ["spherical", "exponential", "gaussian"])
@pytest.mark.parametrize("cosim", [False, True])
def test_gpu_dss_equals_cpu(dss_gpu_case, method, vtype, cosim):
    import warnings

    from geolattice.geostatistics import DirectSequentialSimulator

    grid, df, sec, pick = dss_gpu_case
    var = df["Z"].var(ddof=0)
    v = [20, 0.0 if vtype == "gaussian" else 0.05 * var, 30, 15, var, vtype]
    kw = dict(secondary=sec, secondary_col="s") if cosim else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # soares skewness warning
        sim = DirectSequentialSimulator(v, method=method, radius=40).fit(grid, df, "X", "Y", "Z", **kw)
    cpu = sim.simulate(4, seed=1)
    sim.use_gpu = True
    gpu = sim.simulate(4, seed=1)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-8 * np.ptp(cpu))
    node = np.flatnonzero((grid == grid[pick[0]]).all(axis=1))[0]
    np.testing.assert_allclose(gpu[:, node], df.loc[0, "Z"])


@needs_gpu
def test_gpu_dss_batching_speed_and_limits(dss_gpu_case, monkeypatch):
    import time

    import cupy

    from geolattice.geostatistics import DirectSequentialSimulator

    grid, df, _, _ = dss_gpu_case
    var = df["Z"].var(ddof=0)
    v = [20, 0, 30, 15, var, "spherical"]
    sim = DirectSequentialSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z")
    t0 = time.perf_counter()
    cpu = sim.simulate(5, seed=2)
    t_cpu5 = time.perf_counter() - t0
    real = cupy.cuda.runtime.memGetInfo
    per = 8 * len(sim._loc) + 12 * len(grid) + 8 * 50 ** 2
    monkeypatch.setattr(cupy.cuda.runtime, "memGetInfo", lambda: (int(2.2 * per / 0.6), real()[1]))
    sim.use_gpu = True
    np.testing.assert_allclose(sim.simulate(5, seed=2), cpu, atol=1e-8 * np.ptp(cpu))
    monkeypatch.undo()
    sim.simulate(2, seed=0)
    t0 = time.perf_counter()
    sim.simulate(128, seed=1)
    assert time.perf_counter() - t0 < 10 * t_cpu5       # 128 on GPU well under 50 on CPU
    with pytest.raises(ValueError, match="num_points"):
        DirectSequentialSimulator(v, num_points=60, use_gpu=True).fit(grid, df, "X", "Y", "Z").simulate(1, seed=0)
    with pytest.raises(ValueError, match="variograms"):
        DirectSequentialSimulator([0, 0, 30, 15, var, "matern", 1.5], use_gpu=True).fit(
            grid, df, "X", "Y", "Z").simulate(1, seed=0)


def test_dss_gpu_out_of_memory_falls_back_to_cpu(dss_gpu_case, monkeypatch):
    import sys

    from geolattice.geostatistics import DirectSequentialSimulator

    grid, df, _, _ = dss_gpu_case
    v = [20, 0, 30, 15, df["Z"].var(ddof=0), "spherical"]
    cpu = DirectSequentialSimulator(v, radius=40).fit(grid, df, "X", "Y", "Z").simulate(2, seed=1)
    import geolattice.geostatistics.gpu.dss_gpu  # noqa: F401
    mod = sys.modules["geolattice.geostatistics.gpu.dss_gpu"]
    backend = sys.modules["geolattice.geostatistics.utils.backend"]

    def boom(*a, **k):
        raise MemoryError("out of memory allocating 81,920,000 bytes")

    monkeypatch.setattr(mod, "dss_simulate_gpu", boom)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)
    sim = DirectSequentialSimulator(v, radius=40, use_gpu=True).fit(grid, df, "X", "Y", "Z")
    with pytest.warns(UserWarning, match="running on the CPU"):
        out = sim.simulate(2, seed=1)
    np.testing.assert_array_equal(out, cpu)


# ------------------------------------------------- GPU turning bands


@pytest.fixture(scope="module")
def tb_gpu_case():
    import pandas as pd

    xs = np.arange(0.0, 50.0, 2.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"X": rng.uniform(0, 48, 25), "Y": rng.uniform(0, 48, 25),
                       "Z": np.exp(rng.normal(size=25))})
    return grid, df


@needs_gpu
@pytest.mark.parametrize("vario", [
    [20, 0.1, 40, 20, 1.0, "spherical"],
    [20, 0.1, 40, 20, 1.0, "exponential"],
    [20, 0.0, 40, 20, 1.0, "gaussian"],
    [20, 0.1, 40, 20, 1.0, "matern", 1.5],
], ids=lambda v: v[5])
@pytest.mark.parametrize("mode", ["unconditional", "simple", "ordinary", "itrans"])
def test_gpu_tbsim_equals_cpu(tb_gpu_case, vario, mode):
    from geolattice.geostatistics import TBSimulator

    grid, df = tb_gpu_case
    kw = {"ordinary": dict(kriging_type="ordinary"), "itrans": dict(itrans=True, zmin=0.0)}.get(mode, {})
    sim = TBSimulator(vario, nlines=200, **kw).fit(grid, None if mode == "unconditional" else df)
    cpu = sim.simulate(3, seed=1)
    sim.use_gpu = True
    gpu = sim.simulate(3, seed=1)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10 * np.ptp(cpu))


@needs_gpu
def test_gpu_tbsim_data_values_and_pgs(tb_gpu_case):
    from geolattice.geostatistics import PluriGaussianSimulator, TBSimulator, TruncationRule

    grid, df = tb_gpu_case
    sim = TBSimulator([0, 0, 40, 20, 1.0, "exponential"], nlines=200).fit(grid, df)
    dv = np.arange(len(df), dtype=float)
    cpu = sim.simulate(2, seed=2, data_values=dv)
    sim.use_gpu = True
    np.testing.assert_allclose(sim.simulate(2, seed=2, data_values=dv), cpu, atol=1e-10 * np.ptp(cpu))
    with pytest.raises(ValueError, match="data_values"):
        sim.simulate(1, seed=0, data_values=np.zeros(3))


@needs_gpu
def test_gpu_tbsim_is_fast_even_for_one_realization(tb_gpu_case):
    import time

    import pandas as pd

    from geolattice.geostatistics import TBSimulator

    xs = np.arange(0.0, 120.0, 1.0)
    grid = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"X": rng.uniform(0, 119, 100), "Y": rng.uniform(0, 119, 100), "Z": rng.normal(size=100)})
    cpu = TBSimulator([0, 0, 40, 20, 1.0, "exponential"], nlines=500).fit(grid, df)
    t0 = time.perf_counter()
    cpu.simulate(1, seed=1)
    t_cpu = time.perf_counter() - t0
    gpu = TBSimulator([0, 0, 40, 20, 1.0, "exponential"], nlines=500, use_gpu=True).fit(grid, df)
    gpu.simulate(1, seed=0)
    t0 = time.perf_counter()
    gpu.simulate(1, seed=1)
    assert time.perf_counter() - t0 < t_cpu / 3


def test_tbsim_gpu_out_of_memory_falls_back_to_cpu(tb_gpu_case, monkeypatch):
    import sys

    from geolattice.geostatistics import TBSimulator

    grid, df = tb_gpu_case
    v = [0, 0, 40, 20, 1.0, "spherical"]
    cpu = TBSimulator(v, nlines=100).fit(grid, df).simulate(2, seed=1)
    import geolattice.geostatistics.gpu.tbsim_gpu  # noqa: F401
    mod = sys.modules["geolattice.geostatistics.gpu.tbsim_gpu"]
    backend = sys.modules["geolattice.geostatistics.utils.backend"]

    def boom(*a, **k):
        raise MemoryError("out of memory allocating 81,920,000 bytes")

    monkeypatch.setattr(mod, "tb_simulate_gpu", boom)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)
    sim = TBSimulator(v, nlines=100, use_gpu=True).fit(grid, df)
    with pytest.warns(UserWarning, match="running on the CPU"):
        out = sim.simulate(2, seed=1)
    np.testing.assert_array_equal(out, cpu)


@needs_gpu
def test_tf_module_import_does_not_break_cupy_cublas():
    """Importing the TensorFlow backend must not map a second cuBLAS build
    (it used to, and the next CuPy matmul segfaulted). Run in a subprocess so
    a crash fails this test instead of killing the session."""
    import subprocess
    import sys

    pytest.importorskip("tensorflow")
    code = (
        "import geolattice.geostatistics.ml.regression_kriging_tf as real\n"
        "import geolattice.geostatistics.regression_kriging_tf as alias\n"
        "assert alias is real\n"
        "import cupy as cp\n"
        "a = cp.ones((10, 3))\n"
        "print(float((a @ a.T).sum()))\n"
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr[-2000:]}"
    assert "300.0" in r.stdout
