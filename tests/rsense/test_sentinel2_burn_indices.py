"""Tests for Sentinel-2 burn indices."""

import numpy as np
import pytest
import xarray as xr

from geolattice.rsense.optical.sensors.sentinel2_burn_indices import (
    bai,
    bais2,
    dnbr,
    mirbi,
    nbr,
    nbr2,
)


@pytest.fixture
def sample_bands():
    """Create sample Sentinel-2 band data for testing."""
    # Create a 10x10 grid of sample data
    size = 10
    data = {
        "B4": xr.DataArray(
            np.random.uniform(0.02, 0.30, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "Red"}
        ),
        "B6": xr.DataArray(
            np.random.uniform(0.05, 0.40, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "Red Edge 1"}
        ),
        "B7": xr.DataArray(
            np.random.uniform(0.08, 0.45, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "Red Edge 2"}
        ),
        "B8": xr.DataArray(
            np.random.uniform(0.10, 0.50, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "NIR"}
        ),
        "B8A": xr.DataArray(
            np.random.uniform(0.12, 0.48, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "Narrow NIR"}
        ),
        "B11": xr.DataArray(
            np.random.uniform(0.05, 0.35, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "SWIR1"}
        ),
        "B12": xr.DataArray(
            np.random.uniform(0.03, 0.30, (size, size)),
            dims=["y", "x"],
            attrs={"long_name": "SWIR2"}
        ),
    }
    return data


def test_nbr(sample_bands):
    """Test Normalized Burn Ratio calculation."""
    b8 = sample_bands["B8"]
    b12 = sample_bands["B12"]

    result = nbr(b8, b12)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == b8.shape

    # Check that values are in reasonable range (approximately -1 to 1)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)

    # Check that the formula is correct: (B8 - B12) / (B8 + B12)
    expected = (b8 - b12) / (b8 + b12)
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_dnbr(sample_bands):
    """Test Differenced Normalized Burn Ratio calculation."""
    b8 = sample_bands["B8"]
    b12 = sample_bands["B12"]

    nbr_pre = nbr(b8, b12)
    # Simulate post-fire with reduced NIR and increased SWIR
    nbr_post = nbr(b8 * 0.7, b12 * 1.3)

    result = dnbr(nbr_pre, nbr_post)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == nbr_pre.shape

    # Check formula: NBR_pre - NBR_post
    expected = nbr_pre - nbr_post
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_nbr2(sample_bands):
    """Test Normalized Burn Ratio 2 calculation."""
    b11 = sample_bands["B11"]
    b12 = sample_bands["B12"]

    result = nbr2(b11, b12)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == b11.shape

    # Check that values are in reasonable range (approximately -1 to 1)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)

    # Check formula: (B11 - B12) / (B11 + B12)
    expected = (b11 - b12) / (b11 + b12)
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_bai(sample_bands):
    """Test Burned Area Index calculation."""
    b4 = sample_bands["B4"]
    b8 = sample_bands["B8"]

    result = bai(b4, b8)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == b4.shape

    # Check that values are positive
    assert np.all(result > 0)

    # Check formula with default reference values
    ref_red = 0.1
    ref_nir = 0.06
    expected = 1.0 / ((ref_red - b4) ** 2 + (ref_nir - b8) ** 2)
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_bai_custom_refs(sample_bands):
    """Test BAI with custom reference values."""
    b4 = sample_bands["B4"]
    b8 = sample_bands["B8"]

    ref_red = 0.15
    ref_nir = 0.08
    result = bai(b4, b8, ref_red=ref_red, ref_nir=ref_nir)

    expected = 1.0 / ((ref_red - b4) ** 2 + (ref_nir - b8) ** 2)
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_bais2(sample_bands):
    """Test Burned Area Index for Sentinel-2 calculation."""
    b4 = sample_bands["B4"]
    b6 = sample_bands["B6"]
    b7 = sample_bands["B7"]
    b8a = sample_bands["B8A"]
    b12 = sample_bands["B12"]

    result = bais2(b4, b6, b7, b8a, b12)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == b4.shape

    # Check formula: (1 - sqrt(B6 * B7 * B8A / B4)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)
    term1 = 1.0 - np.sqrt((b6 * b7 * b8a) / b4)
    term2 = (b12 - b8a) / np.sqrt(b12 + b8a) + 1.0
    expected = term1 * term2
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_mirbi(sample_bands):
    """Test Mid-Infrared Burn Index calculation."""
    b12 = sample_bands["B12"]
    b11 = sample_bands["B11"]

    result = mirbi(b12, b11)

    # Check that result is an xarray DataArray
    assert isinstance(result, xr.DataArray)

    # Check shape
    assert result.shape == b12.shape

    # Check formula: 10 * B12 - 9.8 * B11 + 2
    expected = 10.0 * b12 - 9.8 * b11 + 2.0
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_module_exports():
    """Test that all expected functions are exported."""
    from geolattice.rsense.optical.sensors import sentinel2_burn_indices

    assert hasattr(sentinel2_burn_indices, "nbr")
    assert hasattr(sentinel2_burn_indices, "dnbr")
    assert hasattr(sentinel2_burn_indices, "nbr2")
    assert hasattr(sentinel2_burn_indices, "bai")
    assert hasattr(sentinel2_burn_indices, "bais2")
    assert hasattr(sentinel2_burn_indices, "mirbi")

    # Check __all__
    assert "nbr" in sentinel2_burn_indices.__all__
    assert "dnbr" in sentinel2_burn_indices.__all__
    assert "nbr2" in sentinel2_burn_indices.__all__
    assert "bai" in sentinel2_burn_indices.__all__
    assert "bais2" in sentinel2_burn_indices.__all__
    assert "mirbi" in sentinel2_burn_indices.__all__


def test_sensor_module_import():
    """Test that sentinel2_burn_indices can be imported from sensors package."""
    from geolattice.rsense.optical.sensors import sentinel2_burn_indices

    assert sentinel2_burn_indices is not None
