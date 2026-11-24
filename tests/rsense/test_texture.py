import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import texture as tex


def _grid(ny=60, nx=60, res=10.0):
    y = 4_700_000 - res * (np.arange(ny) + 0.5)
    x = 400_000 + res * (np.arange(nx) + 0.5)
    return y, x


@pytest.fixture
def flat():
    """A perfectly uniform surface: every texture measure should read "textureless" here."""
    return np.full((60, 60), 0.3)


@pytest.fixture
def noisy():
    """Strong, unstructured speckle-like noise: textured in every sense."""
    rng = np.random.default_rng(0)
    return rng.gamma(4.0, 0.3 / 4.0, size=(60, 60))


@pytest.fixture
def edge_scene():
    """A flat image with a sharp vertical edge at column 30."""
    img = np.full((60, 60), 0.1)
    img[:, 30:] = 0.9
    return img


@pytest.fixture
def periodic():
    """Vertical stripes (period 6 px): a textbook case for Fourier/Gabor orientation."""
    _, x = _grid(60, 60)
    xx = np.arange(60)
    return np.tile(np.sin(2 * np.pi * xx / 6.0), (60, 1))


@pytest.fixture
def ramp():
    """A smooth, low-roughness linear gradient."""
    xx = np.linspace(0, 1, 60)
    return np.tile(xx, (60, 1))


@pytest.fixture
def dual_pol_da(noisy, flat):
    """A georeferenced (pol, y, x) DataArray, like a VV/VH SAR stack, to exercise leading-dim
    broadcasting and coordinate/CRS preservation."""
    y, x = _grid(60, 60)
    data = np.stack([noisy, flat])
    da = xr.DataArray(data, dims=("pol", "y", "x"),
                      coords={"pol": ["VV", "VH"], "y": y, "x": x}, name="sigma0")
    return da.rio.write_crs(32646)


def _assert_leading_dims_preserved(result, da):
    if isinstance(result, xr.Dataset):
        for var in result.data_vars.values():
            assert var.shape == da.shape
            assert var.rio.crs is not None
            assert list(var.pol.values) == list(da.pol.values)
    else:
        assert result.shape == da.shape
        assert result.rio.crs is not None
        assert list(result.pol.values) == list(da.pol.values)


# --------------------------------------------------------------------------- 1. statistical

def test_first_order_stats_flat_vs_noisy(flat, noisy):
    flat_out = tex.first_order_stats(flat, size=9, stats=("std", "entropy"), entropy_bins=16)
    noisy_out = tex.first_order_stats(noisy, size=9, stats=("std", "entropy"), entropy_bins=16)
    interior = (slice(10, 50), slice(10, 50))
    assert np.nanmax(flat_out["std"][interior]) < 1e-9
    assert np.nanmean(noisy_out["std"][interior]) > np.nanmean(flat_out["std"][interior])
    assert np.nanmean(noisy_out["entropy"][interior]) > np.nanmean(flat_out["entropy"][interior])


def test_glcm_texture_flat_vs_noisy(flat, noisy):
    flat_out = tex.glcm_texture(flat, size=9, step=4, levels=16, props=("contrast", "homogeneity"))
    noisy_out = tex.glcm_texture(noisy, size=9, step=4, levels=16, props=("contrast", "homogeneity"))
    interior = (slice(10, 50), slice(10, 50))
    assert np.nanmean(flat_out["homogeneity"][interior]) > np.nanmean(noisy_out["homogeneity"][interior])
    assert np.nanmean(flat_out["contrast"][interior]) < np.nanmean(noisy_out["contrast"][interior])


def test_glcm_texture_dataset_on_dataarray(dual_pol_da):
    out = tex.glcm_texture(dual_pol_da, size=7, step=3, levels=16, props=("contrast", "homogeneity"))
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"contrast", "homogeneity"}
    _assert_leading_dims_preserved(out, dual_pol_da)


def test_glrlm_texture_directional(periodic, noisy):
    """Vertical stripes give long vertical runs (low RP) but short horizontal runs across
    the stripe boundaries; pure noise gives short runs in every direction (RP near 1)."""
    out_periodic = tex.glrlm_texture(periodic, size=18, step=9, levels=8,
                                     directions=("vertical",), features=("RP",))
    out_noisy = tex.glrlm_texture(noisy, size=18, step=9, levels=8,
                                  directions=("vertical",), features=("RP",))
    interior = (slice(15, 45), slice(15, 45))
    assert np.nanmean(out_periodic["RP"][interior]) < np.nanmean(out_noisy["RP"][interior])


# --------------------------------------------------------------------------- 2. structural

def test_edge_density_marks_the_edge(edge_scene):
    out = np.asarray(tex.edge_density(edge_scene, size=9))
    near_edge = out[:, 27:33].mean()
    away = np.concatenate([out[:, 5:15].ravel(), out[:, 45:55].ravel()]).mean()
    assert near_edge > away


def test_lacunarity_uniform_vs_clumped(flat):
    rng = np.random.default_rng(1)
    clumped = np.zeros((60, 60))
    for _ in range(4):
        r, c = rng.integers(10, 50, 2)
        clumped[r:r + 8, c:c + 8] = 1.0

    out_flat = tex.lacunarity(flat, box_sizes=(4,), size=40)
    out_clumped = tex.lacunarity(clumped, box_sizes=(4,), size=40)
    interior = (slice(20, 40), slice(20, 40))
    assert np.nanmean(out_flat["lac_r4"][interior]) == pytest.approx(1.0, abs=0.05)
    assert np.nanmean(out_clumped["lac_r4"][interior]) > np.nanmean(out_flat["lac_r4"][interior])


# --------------------------------------------------------------------------- 3. transform-based

def test_gabor_texture_matches_stripe_orientation(periodic):
    out = tex.gabor_texture(periodic, frequencies=(1.0 / 6.0,), thetas=(0, 90), size=12)
    interior = (slice(15, 45), slice(15, 45))
    # skimage's gabor theta=0 means the filter's harmonic varies along x, matching these
    # vertical stripes (which vary along x, constant along y); theta=90 is rotated
    # perpendicular to that and should respond far more weakly.
    e0 = np.nanmean(out["gabor_f0p166667_t0"][interior])
    e90 = np.nanmean(out["gabor_f0p166667_t90"][interior])
    assert e0 > e90


def test_fourier_texture_sums_to_one(noisy):
    out = tex.fourier_texture(noisy, size=16, step=8, n_rings=4)
    interior = (slice(8, 52), slice(8, 52))
    total = sum(np.nanmean(v[interior]) for v in out.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_wavelet_texture_flat_vs_checkerboard(flat):
    yy, xx = np.mgrid[0:60, 0:60]
    checker = ((yy // 3 + xx // 3) % 2).astype("float64")
    out_flat = tex.wavelet_texture(flat, level=1, size=9)
    out_checker = tex.wavelet_texture(checker, level=1, size=9)
    interior = (slice(10, 50), slice(10, 50))
    for band in ("LH_L1", "HL_L1", "HH_L1"):
        assert np.nanmax(out_flat[band][interior]) < 1e-9
        assert np.nanmean(out_checker[band][interior]) > 0


# --------------------------------------------------------------------------- 4. local pattern

def test_lbp_texture_uniform_region_has_low_entropy(flat, noisy):
    out_flat = tex.lbp_texture(flat, size=9, step=3)
    out_noisy = tex.lbp_texture(noisy, size=9, step=3)
    interior = (slice(10, 50), slice(10, 50))
    assert np.nanmean(out_flat["lbp_entropy"][interior]) < np.nanmean(out_noisy["lbp_entropy"][interior])
    assert np.nanmean(out_flat["lbp_uniformity"][interior]) > np.nanmean(out_noisy["lbp_uniformity"][interior])


# --------------------------------------------------------------------------- 5. model-based

def test_mrf_texture_energy_flat_vs_noisy(flat, noisy):
    out_flat = np.asarray(tex.mrf_texture_energy(flat, size=9))
    out_noisy = np.asarray(tex.mrf_texture_energy(noisy, size=9))
    interior = (slice(10, 50), slice(10, 50))
    assert np.nanmax(out_flat[interior]) < 1e-9
    assert np.nanmean(out_noisy[interior]) > 0


def test_fractal_dimension_smooth_vs_rough(ramp, noisy):
    out_ramp = np.asarray(tex.fractal_dimension(ramp, size=16, step=8, box_sizes=(2, 4, 8, 16)))
    out_noisy = np.asarray(tex.fractal_dimension(noisy, size=16, step=8, box_sizes=(2, 4, 8, 16)))
    interior = (slice(8, 52), slice(8, 52))
    assert np.nanmean(out_noisy[interior]) > np.nanmean(out_ramp[interior])


def test_autoregressive_texture_smooth_vs_noisy(ramp, noisy):
    out_ramp = tex.autoregressive_texture(ramp, size=9)
    out_noisy = tex.autoregressive_texture(noisy, size=9)
    interior = (slice(10, 50), slice(10, 50))
    # a smooth ramp is far better explained by its neighbour average than pure noise is
    assert np.nanmean(out_ramp["rho"][interior]) > np.nanmean(out_noisy["rho"][interior])
    assert np.nanmean(out_ramp["residual"][interior]) < np.nanmean(out_noisy["residual"][interior])


# --------------------------------------------------------------------------- 6. deep learning

def test_deep_texture_features():
    try:
        import timm  # noqa: F401
    except Exception:
        pytest.skip("timm (or its torchvision dependency) is not usable in this environment")

    rng = np.random.default_rng(2)
    img = rng.normal(100, 10, (48, 48)).astype("float32")
    out = tex.deep_texture_features(img, model_name="resnet18", layer=1, reduce="map")
    assert isinstance(out, dict) and len(out) > 0
    for v in out.values():
        assert v.shape == img.shape

    out_mean = tex.deep_texture_features(img, model_name="resnet18", layer=1, reduce="mean")
    assert all(isinstance(v, float) for v in out_mean.values())


# --------------------------------------------------------------------------- shared contract

@pytest.mark.parametrize("fn, kwargs", [
    (tex.first_order_stats, dict(size=7)),
    (tex.glcm_texture, dict(size=7, step=3, levels=8)),
    (tex.glrlm_texture, dict(size=12, step=6, levels=8)),
    (tex.edge_density, dict(size=7)),
    (tex.lacunarity, dict(box_sizes=(2, 4))),
    (tex.gabor_texture, dict(frequencies=(0.2,), thetas=(0, 90), size=7)),
    (tex.fourier_texture, dict(size=16, step=8, n_rings=4)),
    (tex.wavelet_texture, dict(level=2, size=7)),
    (tex.lbp_texture, dict(size=7, step=3)),
    (tex.mrf_texture_energy, dict(size=7)),
    (tex.fractal_dimension, dict(size=16, step=8, box_sizes=(2, 4, 8))),
    (tex.autoregressive_texture, dict(size=7)),
])
def test_output_shape_and_georeferencing_preserved(dual_pol_da, fn, kwargs):
    """Every texture function returns the exact input shape and, for a DataArray input with
    leading dims, keeps the CRS and non-spatial coordinates (e.g. polarization) intact."""
    result = fn(dual_pol_da, **kwargs)
    _assert_leading_dims_preserved(result, dual_pol_da)


@pytest.mark.parametrize("fn, kwargs", [
    (tex.first_order_stats, dict(size=7)),
    (tex.edge_density, dict(size=7)),
    (tex.mrf_texture_energy, dict(size=7)),
])
def test_nan_input_stays_nan(fn, kwargs):
    """A missing pixel must stay NaN in the output, not silently become a valid value.

    This exact-pixel contract only holds for the per-pixel box-filter functions tested
    here. The step-based grid functions (GLCM/GLRLM/Fourier/fractal/LBP) nearest-neighbour
    resample a coarser grid back to full resolution -- including across the window margin
    their ``pad`` always excludes from sampling -- so an isolated single-pixel NaN is
    approximated from nearby valid windows rather than reproduced exactly at that pixel;
    see the module docstring.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(0.1, 0.02, (40, 40))
    a[5, 5] = np.nan
    out = fn(a, **kwargs)
    if isinstance(out, dict):
        for v in out.values():
            assert np.isnan(v[5, 5])
    else:
        assert np.isnan(out[5, 5])
