"""JAX variogram and kriging against the NumPy core, on CPU and GPU."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

pytest.importorskip("scipy")
jax = pytest.importorskip("jax")

from geolattice.geostatistics.core.indicator_variogram import fit_indicator_variogram as ind_cpu
from geolattice.geostatistics.core.kriging import OrdinaryKriging as OKCPU
from geolattice.geostatistics.core.kriging import SimpleKriging as SKCPU
from geolattice.geostatistics.core.variogram import Variogram as VariogramCPU
from geolattice.geostatistics.core.variogram_cloud import compute_variogram_cloud as cloud_cpu
from geolattice.geostatistics.core.variogram_models import (
    covariance_from_variogram as cov_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    cubic as cubic_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    exponential as exp_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    gaussian as gauss_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    matern as matern_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    spherical as sph_cpu,
)
from geolattice.geostatistics.core.variogram_models import (
    stable as stable_cpu,
)
from geolattice.geostatistics.utils.anisotropy import transform_aniso as aniso_cpu

import geolattice.geostatistics.jax_variogram_kriging as jx


def _gpus():
    return [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]


def _sample(n=36, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    values = np.sin(coords[:, 0] * 0.4) + 0.2 * coords[:, 1] + rng.normal(0, 0.02, n)
    return coords, values


def _close(a, b, rtol=1e-6, atol=1e-8):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rtol, atol=atol, equal_nan=True)


def test_models_match_numpy():
    h = np.linspace(0, 8, 17)
    cases = [
        (sph_cpu, jx.spherical, {}),
        (exp_cpu, jx.exponential, {}),
        (gauss_cpu, jx.gaussian, {}),
        (cubic_cpu, jx.cubic, {}),
        (stable_cpu, jx.stable, {"alpha": 1.4}),
        (matern_cpu, jx.matern, {"nu": 1.5}),
        (matern_cpu, jx.matern, {"nu": 0.7}),
    ]
    for cpu_fn, jax_fn, kwargs in cases:
        _close(cpu_fn(h, 0.2, 1.3, 4.0, **kwargs), jax_fn(h, 0.2, 1.3, 4.0, **kwargs), rtol=1e-6, atol=1e-8)
    _close(cov_cpu(h, 0.2, 1.3, 4.0), jx.covariance_from_variogram(h, 0.2, 1.3, 4.0))
    c0 = jx.covariance_from_variogram(np.array([0.0, 1.0]), 0.2, 1.3, 4.0)
    assert np.asarray(c0)[0] == pytest.approx(1.5)


def test_anisotropy_matches_numpy():
    coords, _ = _sample()
    _close(
        aniso_cpu(coords, angle=30, ratio=0.5),
        jx.transform_aniso(coords, angle=30, ratio=0.5),
    )


@pytest.mark.parametrize("estimator", ["matheron", "cressie", "dowd"])
@pytest.mark.parametrize("use_gpu", [False, True])
def test_empirical_and_kriging_match_numpy(estimator, use_gpu):
    if use_gpu and not _gpus():
        pytest.skip("JAX has no GPU device")
    coords, values = _sample()
    pred = np.array([[2.0, 3.0], [7.5, 1.5], [5.0, 5.0]])
    cpu = VariogramCPU(
        coords, values, model="exponential", estimator=estimator, maxlag=6.0, n_lags=8, use_gpu=False
    ).fit()
    gpu = jx.Variogram(
        coords, values, model="exponential", estimator=estimator, maxlag=6.0, n_lags=8, use_gpu=use_gpu
    ).fit()
    _close(cpu.lags, gpu.lags)
    _close(cpu.experimental, gpu.experimental, rtol=1e-6, atol=1e-8)
    if not use_gpu:
        _close(cpu.fitted_params, gpu.fitted_params, rtol=1e-4, atol=1e-5)
    # Kriging uses the NumPy fit so a GPU optimizer step cannot drift the check.
    gpu.set_params(*[float(v) for v in cpu.fitted_params])
    _close(cpu(np.linspace(0, 6, 9)), gpu(np.linspace(0, 6, 9)))

    ok_cpu = OKCPU(cpu, use_gpu=False).fit(coords, values)
    ok_jax = jx.OrdinaryKriging(gpu, use_gpu=use_gpu).fit(coords, values)
    pa, sa = ok_cpu.predict(pred, return_variance=True)
    pb, sb = ok_jax.predict(pred, return_variance=True)
    tol = dict(rtol=1e-6, atol=1e-6) if not use_gpu else dict(rtol=1e-5, atol=1e-6)
    _close(pa, pb, **tol)
    _close(sa, sb, **tol)

    sk_cpu = SKCPU(cpu, use_gpu=False).fit(coords, values)
    sk_jax = jx.SimpleKriging(gpu, use_gpu=use_gpu).fit(coords, values)
    pa, sa = sk_cpu.predict(pred, return_variance=True)
    pb, sb = sk_jax.predict(pred, return_variance=True)
    _close(pa, pb, **tol)
    _close(sa, sb, **tol)

    assert gpu.use_gpu is bool(use_gpu and _gpus())


def test_indicator_cloud_and_block_match_numpy():
    coords, values = _sample()
    threshold = float(np.median(values))
    lags_c, gamma_c = ind_cpu(coords, values, threshold=threshold, maxlag=6, n_lags=8)
    lags_j, gamma_j = jx.fit_indicator_variogram(
        coords, values, threshold=threshold, maxlag=6, n_lags=8, use_gpu=False
    )
    _close(lags_c, lags_j)
    _close(gamma_c, gamma_j)

    cloud_c = cloud_cpu(coords, values, maxlag=4)
    cloud_j = jx.compute_variogram_cloud(coords, values, maxlag=4, use_gpu=False)
    for left, right in zip(cloud_c, cloud_j):
        _close(left, right)

    cpu = VariogramCPU(coords, values, model="spherical", maxlag=6, n_lags=6, use_gpu=False).fit()
    jax_v = jx.Variogram(coords, values, model="spherical", maxlag=6, n_lags=6, use_gpu=False)
    jax_v.set_params(*cpu.fitted_params)
    centers = np.array([[3.0, 3.0], [8.0, 2.0]])
    ok_cpu = OKCPU(cpu, use_gpu=False).fit(coords, values)
    ok_jax = jx.OrdinaryKriging(jax_v, use_gpu=False).fit(coords, values)
    np.random.seed(2)
    block_c = ok_cpu.predict(centers, block_size=1.0, return_variance=True)
    np.random.seed(2)
    block_j = ok_jax.predict(centers, block_size=1.0, return_variance=True)
    _close(block_c[0], block_j[0], rtol=1e-6, atol=1e-6)
    _close(block_c[1], block_j[1], rtol=1e-6, atol=1e-6)


def test_jax_array_call_stays_on_device():
    coords, values = _sample(n=12)
    variogram = jx.Variogram(coords, values, model="gaussian", maxlag=5, n_lags=5, use_gpu=False).fit()
    import jax.numpy as jnp

    out = variogram(jnp.array([0.0, 1.0, 2.0]))
    assert isinstance(out, jax.Array)
    assert np.all(np.isfinite(np.asarray(out)))


def test_aliases_do_not_replace_numpy_classes():
    import geolattice.geostatistics as gstat
    from geolattice.geostatistics.core.variogram import Variogram as CoreV

    assert gstat.Variogram is CoreV
    assert gstat.JaxVariogram is jx.Variogram
    assert gstat.JaxOrdinaryKriging is jx.OrdinaryKriging
    assert gstat.JaxSimpleKriging is jx.SimpleKriging
    assert "JaxVariogram" in gstat.__all__


def test_universal_and_cokriging_match_numpy():
    from geolattice.geostatistics.core.variogram import Variogram as VariogramCPU
    from geolattice.geostatistics.kriging.cokriging import Cokriging as CKCPU
    from geolattice.geostatistics.kriging.cokriging import CrossVariogram as CVCPU
    from geolattice.geostatistics.kriging.universal_kriging import UniversalKriging as UKCPU

    rng = np.random.default_rng(1)
    coords, primary = _sample(n=28, seed=1)
    secondary = 0.6 * primary + rng.normal(0, 0.05, len(primary))
    pred = rng.uniform(0, 10, size=(6, 2))

    cpu = VariogramCPU(coords, primary, model="exponential", maxlag=6, n_lags=6, use_gpu=False).fit()
    jax_v = jx.Variogram(coords, primary, model="exponential", maxlag=6, n_lags=6, use_gpu=False)
    jax_v.set_params(*[float(v) for v in cpu.fitted_params])

    uk_cpu = UKCPU(cpu, degree=1, use_gpu=False).fit(coords, primary)
    uk_jax = jx.UniversalKriging(jax_v, degree=1, use_gpu=False).fit(coords, primary)
    pa, sa = uk_cpu.predict(pred, return_variance=True)
    pb, sb = uk_jax.predict(pred, return_variance=True)
    _close(pa, pb, rtol=1e-5, atol=1e-6)
    # Variances that sit on zero clip to a few ulps on one backend and not the other.
    _close(sa, sb, rtol=1e-5, atol=1e-3)

    sec_cpu = VariogramCPU(coords, secondary, model="exponential", maxlag=6, n_lags=6, use_gpu=False).fit()
    sec_jax = jx.Variogram(coords, secondary, model="exponential", maxlag=6, n_lags=6, use_gpu=False)
    sec_jax.set_params(*[float(v) for v in sec_cpu.fitted_params])
    cross_cpu = CVCPU(coords, primary, secondary, maxlag=6, n_lags=6, model="exponential", use_gpu=False).fit()
    cross_jax = jx.CrossVariogram(
        coords, primary, secondary, maxlag=6, n_lags=6, model="exponential", use_gpu=False
    ).fit()
    _close(cross_cpu.lags, cross_jax.lags)
    _close(cross_cpu.experimental, cross_jax.experimental, rtol=1e-6, atol=1e-8)
    cross_cpu.fit_model()
    cross_jax.set_params(*[float(v) for v in cross_cpu.fitted_params])

    ck_cpu = CKCPU(cpu, sec_cpu, cross_cpu, use_gpu=False).fit(coords, primary, coords, secondary)
    ck_jax = jx.Cokriging(jax_v, sec_jax, cross_jax, use_gpu=False).fit(coords, primary, coords, secondary)
    pa, sa = ck_cpu.predict(pred, return_variance=True)
    pb, sb = ck_jax.predict(pred, return_variance=True)
    _close(pa, pb, rtol=1e-5, atol=1e-6)
    _close(sa, sb, rtol=1e-5, atol=1e-3)


def test_validation_errors():
    coords, values = _sample(n=8)
    with pytest.raises(ValueError, match="model must be one of"):
        jx.Variogram(coords, values, model="linear")
    with pytest.raises(ValueError, match="at least 2"):
        jx.Variogram(coords[:1], values[:1])
    variogram = jx.Variogram(coords, values, model="matern", use_gpu=False)
    with pytest.raises(ValueError, match="nu"):
        variogram.set_params(0.0, 1.0, 2.0)
    variogram.set_params(0.1, 1.0, 2.0, nu=1.5)
    ok = jx.OrdinaryKriging(variogram, use_gpu=False)
    with pytest.raises(RuntimeError, match="not fitted"):
        ok.predict(coords[:2])
    ok.fit(coords, values)
    with pytest.raises(ValueError, match="same length"):
        ok.fit(coords, values[:-1])


# ---------------------------------------------------------------------------
# Factored prediction path (Cholesky in fit, compiled kernels in predict)
# ---------------------------------------------------------------------------

def _fitted_variogram(coords, values, use_gpu=False, params=(0.05, 1.0, 4.0)):
    vgm = jx.Variogram(coords, values, model="spherical", n_lags=8, use_gpu=use_gpu)
    vgm.set_params(*params)
    return vgm


@pytest.mark.parametrize("use_gpu", [False, True])
def test_fast_path_matches_numpy_for_every_batch_shape(use_gpu):
    if use_gpu and not _gpus():
        pytest.skip("no JAX GPU")
    coords, values = _sample(60)
    vgm = _fitted_variogram(coords, values, use_gpu)
    core = VariogramCPU(coords, values, model="spherical", n_lags=8, use_gpu=False)
    core.set_params(0.05, 1.0, 4.0)
    ok = jx.OrdinaryKriging(vgm, use_gpu=use_gpu).fit(coords, values)
    assert ok._factor is not None
    ok_cpu = OKCPU(core, use_gpu=False).fit(coords, values)
    pred = np.random.default_rng(3).uniform(0, 10, size=(700, 2))
    for k in (1, 5, 256, 257, 700):  # single point, padded and multi-batch calls
        p, s = ok.predict(pred[:k], return_variance=True)
        q, t = ok_cpu.predict(pred[:k], return_variance=True)
        _close(p, q, rtol=1e-7, atol=1e-9)
        _close(s, t, rtol=1e-6, atol=1e-8)


def test_singular_system_falls_back_to_bordered_solve():
    """Duplicate locations and no nugget: Cholesky fails, predictions stay finite."""
    coords, values = _sample(30)
    coords = np.vstack([coords, coords[:4]])
    values = np.concatenate([values, values[:4]])
    vgm = jx.Variogram(coords, values, model="gaussian", n_lags=8, use_gpu=False)
    vgm.set_params(0.0, 1.0, 3.0)
    ok = jx.OrdinaryKriging(vgm, use_gpu=False).fit(coords, values)
    assert ok._factor is None
    pred = ok.predict(np.array([[2.0, 3.0], [7.5, 1.0]]))
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("cls,kw", [(jx.OrdinaryKriging, {}), (jx.SimpleKriging, {"mean": 0.5}),
                                    (jx.UniversalKriging, {"degree": 1})])
def test_float32_is_close_to_float64(cls, kw):
    coords, values = _sample(80)
    use_gpu = bool(_gpus())
    vgm = _fitted_variogram(coords * 1000.0 + 5e5, values, use_gpu, params=(0.05, 1.0, 4000.0))
    x = coords * 1000.0 + 5e5  # metre-scale coordinates far from the origin
    pred = np.random.default_rng(4).uniform(0, 10, size=(300, 2)) * 1000.0 + 5e5
    m64 = cls(vgm, use_gpu=use_gpu, **kw).fit(x, values)
    m32 = cls(vgm, use_gpu=use_gpu, dtype="float32", **kw).fit(x, values)
    assert m32._factor is not None
    p64, s64 = m64.predict(pred, return_variance=True)
    p32, s32 = m32.predict(pred, return_variance=True)
    assert p32.dtype == np.float32
    _close(p32, p64, rtol=0, atol=1e-3)
    _close(s32, s64, rtol=0, atol=1e-3)


def test_dtype_must_be_float32_or_float64():
    coords, values = _sample()
    with pytest.raises(ValueError, match="dtype"):
        jx.OrdinaryKriging(_fitted_variogram(coords, values), use_gpu=False, dtype="float16")
