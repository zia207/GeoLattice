"""
Tests for the geolattice.rsense.spectral_index module.

These tests cover:
- Sensor name normalization and alias handling
- Reporting which spectral indices are available for a given sensor and its bands
- Coverage of the USGS Landsat Surface Reflectance index set and its presence in the catalog
- Required band checks and error flow when computing indices with missing bands
- Correct computation of indices and handling of band name priorities
- Listing and describing available indices and bands for each supported sensor
- Ensuring every indexed sensor has a profile and returned indices
"""

import numpy as np
import pytest

import geolattice.rsense as rs
from geolattice.rsense import spectral_index as si
from geolattice.rsense.optical import INDICES


def test_normalize_and_aliases():
    """Test normalization of sensor names and aliases."""
    assert si.normalize_sensor("s2") == "sentinel2"
    assert si.normalize_sensor("Landsat-8") == "landsat89"
    assert si.normalize_sensor("wv2") == "worldview2"
    assert si.normalize_sensor("NAIP") == "naip"
    with pytest.raises(ValueError, match="Unknown sensor"):
        si.normalize_sensor("not-a-sensor")


def test_available_respects_bands():
    """Test index reporting for each sensor covers only indices available with its bands."""
    s2 = set(si.available("sentinel2"))
    ls = set(si.available("landsat89"))
    naip = set(si.available("naip"))
    assert "NDVI" in s2 & ls & naip
    assert "NDRE" in s2          # red-edge available only for S2
    assert "NDRE" not in ls
    assert "NDRE" not in naip
    assert "NBR" in ls           # requires swir22, available in Landsat
    assert "NBR" not in naip
    assert "MNDWI" in ls         # requires swir16, available in Landsat
    assert "MNDWI" not in set(si.available("planetscope"))


def test_usgs_landsat_set():
    """Test the USGS Landsat SR index group is present and supported by Landsat, TM, and MODIS."""
    assert si.USGS_LANDSAT_SR == (
        "NDVI", "EVI", "SAVI", "MSAVI", "NDMI", "NBR", "NBR2", "NDSI")
    for name in si.USGS_LANDSAT_SR:
        assert si.index_supported(name, "landsat89")
        assert si.index_supported(name, "landsat_tm")
        assert si.index_supported(name, "modis")


def test_missing_bands_and_check(reflectance):
    """Test missing band detection, and check=False delegate behavior."""
    assert si.missing_bands("NDRE", "landsat89") == ["rededge1"]
    with pytest.raises(ValueError, match="cannot compute"):
        si.compute(reflectance.drop_vars("rededge1"), "NDRE", sensor="landsat89")
    # check=False delegates to optical.index (may still error if bands missing)
    out = si.compute(reflectance, "NDVI", sensor="sentinel2", check=False)
    assert out.name == "NDVI"


def test_compute_and_usgs_product(reflectance):
    """Test correct computation of indices and USGS Landsat SR group."""
    # Landsat band map: nir08 takes precedence over nir if both present
    ndvi = rs.to_cpu(si.compute(reflectance, "NDVI", sensor="landsat89"))
    np.testing.assert_allclose(
        ndvi, rs.to_cpu(rs.optical.index(reflectance, "NDVI", sensor="landsat89")), rtol=1e-6)

    ls = reflectance.drop_vars("nir08")
    ndvi_ls = rs.to_cpu(si.compute(ls, "NDVI", sensor="landsat89"))
    exp = (ls.nir - ls.red) / (ls.nir + ls.red)
    np.testing.assert_allclose(ndvi_ls, exp, rtol=1e-6)

    usgs = rs.to_cpu(si.usgs_landsat_sr(ls, sensor="landsat89"))
    assert set(usgs.data_vars) == set(si.USGS_LANDSAT_SR)
    assert usgs.attrs["pyrsense:product"] == "usgs_landsat_sr"


def test_list_for_sensor_and_describe():
    """Test listing and describing indices and sensor metadata."""
    df = si.list_for_sensor("naip", domain="vegetation")
    assert "NDVI" in set(df["name"])
    assert (df["domain"] == "vegetation").all()
    profile = si.describe("worldview2")
    assert profile.provider == "Maxar"
    assert "nir" in profile.bands
    assert any("worldview-2" in r for r in profile.references)


def test_every_profile_sensor_has_band_map():
    """Test every sensor profile is present and every key is describable."""
    for key in si.SENSOR_PROFILES:
        assert key in si.sensors()
    # every SENSOR_BANDS key is either profiled or still listable via describe()
    for key in si.sensors():
        p = si.describe(key)
        assert p.key == key
        assert len(si.available(key)) > 0


def test_index_catalog_covers_usgs():
    """Test the master index catalog covers all USGS SR indices."""
    for name in si.USGS_LANDSAT_SR:
        assert name in INDICES
