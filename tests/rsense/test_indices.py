import numpy as np
import pytest

import geolattice.rsense as rs
from geolattice.rsense.optical import INDICES, index, list_indices, register_index


def test_ndvi_matches_formula(backend, reflectance):
    ndvi = rs.to_cpu(index(reflectance, "NDVI"))
    exp = (reflectance.nir - reflectance.red) / (reflectance.nir + reflectance.red)
    np.testing.assert_allclose(ndvi, exp, rtol=1e-6)
    assert ndvi.attrs["pyrsense:index"] == "NDVI"


@pytest.mark.parametrize("name", sorted(INDICES))
def test_every_index_runs(backend, reflectance, name):
    out = rs.to_cpu(index(reflectance, name))
    assert out.shape == reflectance.red.shape
    assert np.isfinite(out.values).mean() > 0.99


def test_savi_param_override(reflectance):
    a = index(reflectance, "SAVI")
    b = index(reflectance, "SAVI", L=0.25)
    assert not np.allclose(a, b)
    with pytest.raises(TypeError):
        index(reflectance, "SAVI", foo=1)


def test_multiple_indices_dataset(reflectance):
    ds = index(reflectance, ["NDVI", "NBR"])
    assert set(ds.data_vars) == {"NDVI", "NBR"}


def test_sensor_band_names(reflectance):
    s2 = reflectance.rename({"red": "B04", "nir": "B08"})
    np.testing.assert_allclose(index(s2, "NDVI", sensor="sentinel2"), index(reflectance, "NDVI"))
    ls = reflectance.drop_vars("nir08").rename({"red": "SR_B4", "nir": "SR_B5"})
    np.testing.assert_allclose(index(ls, "NDVI", sensor="landsat"), index(reflectance, "NDVI"))


def test_explicit_band_mapping(reflectance):
    ds = reflectance.drop_vars("nir08").rename({"nir": "my_nir"})
    out = index(ds, "NDVI", bands={"nir": "my_nir"})
    assert out.notnull().all()
    with pytest.raises(KeyError, match="bands="):
        index(ds, "NDVI")


def test_division_by_zero_is_nan(reflectance):
    ds = reflectance.copy()
    ds["nir"][0, 0, 0] = 0.0
    ds["red"][0, 0, 0] = 0.0
    assert np.isnan(index(ds, "NDVI")[0, 0, 0])


def test_dataarray_with_band_dim(reflectance):
    da = reflectance.to_array("band")
    np.testing.assert_allclose(index(da, "NDVI"), index(reflectance, "NDVI"))


def test_register_custom_index(reflectance):
    register_index("SR_TEST", "Simple ratio", ["nir", "red"], lambda nir, red: nir / red,
                   domain="vegetation", overwrite=True)
    np.testing.assert_allclose(index(reflectance, "sr_test"), reflectance.nir / reflectance.red)
    assert "SR_TEST" in list(list_indices("vegetation")["name"])


def test_lazy_dask(reflectance):
    out = index(reflectance.chunk({"time": 1}), "NDVI")
    assert hasattr(out.data, "dask")
