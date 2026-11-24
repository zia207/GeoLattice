import numpy as np
import pytest
import xarray as xr

import geolattice.rsense as rs
from geolattice.rsense import optical as op


def test_unmix_recovers_fractions(reflectance):
    E = {"veg": [0.03, 0.08, 0.04, 0.45], "soil": [0.10, 0.15, 0.20, 0.25],
         "water": [0.05, 0.04, 0.02, 0.01]}
    rng = np.random.default_rng(0)
    f = rng.dirichlet([1, 1, 1], size=(40, 50)).transpose(2, 0, 1)
    M = np.array(list(E.values())).T  # bands x 3
    img = np.einsum("bk,kyx->byx", M, f).astype("float32")
    tpl = reflectance.red.isel(time=0)
    ds = xr.Dataset({b: tpl.copy(data=img[i]) for i, b in enumerate(["blue", "green", "red", "nir"])})
    for method in ("ls", "scls", "fcls"):
        out = op.unmix(ds, E, method=method)
        np.testing.assert_allclose(out["veg"].values, f[0], atol=1e-3)
        assert float(out["rmse"].max()) < 1e-3


def test_lst_chain():
    bt = xr.DataArray(np.full((2, 2), 300.0, "float32"), dims=("y", "x"))
    ndvi = xr.DataArray(np.array([[-0.1, 0.1], [0.35, 0.8]]), dims=("y", "x"))
    e = op.emissivity_ndvi(ndvi)
    assert e.values[0, 0] == pytest.approx(0.991) and e.values[0, 1] == pytest.approx(0.97)
    lst = op.lst_single_channel(bt, e)
    assert (lst > bt).all() and float(lst.max()) < 303
    L = xr.DataArray([10.0], dims="x")
    assert float(op.brightness_temperature(L, 774.8853, 1321.0789)[0]) == pytest.approx(
        1321.0789 / np.log(774.8853 / 10 + 1), rel=1e-6)


def test_toa_and_dos(reflectance):
    dn = xr.DataArray(np.array([0, 10000, 20000], "uint16"), dims="x")
    toa = op.toa_reflectance(dn, 2e-5, -0.1, 30.0)
    assert np.isnan(toa[0]) and toa[1] == pytest.approx((0.2 - 0.1) / 0.5, rel=1e-5)
    hazy = reflectance.isel(time=0)[["red", "nir"]] + 0.05
    out = op.dos1(hazy)
    assert float(out.red.min()) < float(hazy.red.min())


def test_topographic_correction_removes_illumination(single_band):
    yy, xx = np.mgrid[0:64, 0:80]
    dem = single_band.copy(data=(30 * np.sin(xx / 6.0) + 20 * np.cos(yy / 5.0)).astype("float32"))
    cos_i = rs.lidar.illumination(dem, 135, 35).values
    refl = single_band.copy(data=(0.1 + 0.2 * cos_i).astype("float32"))
    for method in ("c", "scs+c", "minnaert", "cosine"):
        corr = op.topographic_correction(refl, dem, 135, 35, method=method)
        r_before = abs(np.corrcoef(cos_i.ravel(), refl.values.ravel())[0, 1])
        r_after = abs(np.corrcoef(cos_i.ravel(), corr.values.ravel())[0, 1])
        assert r_after < r_before


def test_coregister_recovers_shift(single_band):
    from scipy.ndimage import gaussian_filter, shift

    ref = single_band.copy(data=gaussian_filter(single_band.values, 2))
    moved = ref.copy(data=shift(ref.values, (2.0, -3.0), order=1, mode="nearest"))
    out, info = op.coregister(moved, ref)
    assert info["dy"] == pytest.approx(-2.0, abs=0.2) and info["dx"] == pytest.approx(3.0, abs=0.2)
    inner = (slice(5, -5), slice(5, -5))
    assert np.nanmean(np.abs(out.values[inner] - ref.values[inner])) < 0.05


def test_pansharpen(reflectance):
    ms = reflectance.isel(time=0)[["red", "green", "blue"]].coarsen(x=2, y=2).mean()
    ms = ms.rio.write_crs(32618)
    pan = reflectance.isel(time=0)[["red", "green", "blue"]].to_array().mean("variable")
    pan = pan.rio.write_crs(32618)
    for m in ("brovey", "ihs", "hpf"):
        out = op.pansharpen(ms, pan, method=m)
        assert out.red.shape == pan.shape


def test_bap_composite(reflectance):
    ds = reflectance.copy(deep=True)
    ds["red"][0, :20] = np.nan
    out = op.composite(ds, "bap", target_doy=165)
    assert out.red.notnull().all()
    assert "source_time" in out


def test_hls_mask_and_scale():
    f = xr.DataArray(np.array([0, 2, 8, 64, 255], "uint8"), dims="x")
    assert op.hls_fmask_mask(f).values.tolist() == [True, False, False, True, False]
    ds = xr.Dataset({"B04": ("x", np.array([1000, -9999], "int16")),
                     "Fmask": ("x", np.array([0, 0], "uint8"))})
    assert op.guess_sensor(ds) == "hls_l30"
    out = op.scale(ds)
    assert out.B04[0] == pytest.approx(0.1) and np.isnan(out.B04[1])


def test_index_aliases_and_count(reflectance):
    assert len(op.INDICES) >= 100
    np.testing.assert_allclose(op.index(reflectance, "LSWI"), op.index(reflectance, "NDMI"))


def test_sensor_modules():
    assert op.sensors.landsat.SR_SCALE == 2.75e-5
    assert op.sensors.sentinel2.BOA_ADD_OFFSET == -1000
    da = xr.DataArray(np.array([15000, 0]), dims="x")
    assert float(op.sensors.modis.scale_product(da, "MOD11", "LST_Day_1km")[0]) == 300.0
