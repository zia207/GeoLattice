import numpy as np
import pytest

from geolattice.rsense.optical import composite, index


def test_median(reflectance):
    out = composite(reflectance, "median")
    np.testing.assert_allclose(out.red, reflectance.red.median("time"))
    assert "time" not in out.dims


def test_percentile(reflectance):
    out = composite(reflectance, "percentile", q=90)
    np.testing.assert_allclose(out.nir, reflectance.nir.quantile(0.9, "time"), rtol=1e-5)


def test_max_index_picks_consistent_date(reflectance):
    out = composite(reflectance, "max_index", index="NDVI")
    ndvi = index(reflectance, "NDVI")
    best = ndvi.argmax("time")
    np.testing.assert_allclose(out.red, reflectance.red.isel(time=best))
    np.testing.assert_allclose(out.nir, reflectance.nir.isel(time=best))
    assert (out.source_time == reflectance.time.isel(time=best)).all()


def test_max_index_handles_clouds(reflectance):
    ds = reflectance.copy(deep=True)
    ds["red"][:, 0, 0] = np.nan        # fully cloudy pixel
    ds["red"][1:, 1, 1] = np.nan       # only first date clear
    out = composite(ds, "max_index")
    assert np.isnan(out.red[0, 0]) and np.isnat(out.source_time[0, 0].values)
    assert out.red[1, 1] == ds.red[0, 1, 1]


def test_median_nearest(reflectance):
    ds = reflectance.copy(deep=True)
    for v in ds.data_vars:              # date 2 is a bright cloud everywhere
        ds[v][2] = 0.9
    out = composite(ds, "median_nearest")
    assert (out.source_time != ds.time[2]).all()


def test_unknown_method(reflectance):
    with pytest.raises(ValueError):
        composite(reflectance, "mode")


def test_accessor_chain(reflectance):
    ndvi = reflectance.rs.composite("median").rs.index("NDVI")
    assert ndvi.dims == ("y", "x")
