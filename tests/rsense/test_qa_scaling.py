import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense.optical import (
    decode_bits,
    landsat_qa_mask,
    mask_clouds,
    modis_state_mask,
    modis_vi_mask,
    s2_scl_mask,
    scale,
)


def _da(vals, dtype="uint16"):
    return xr.DataArray(np.array(vals, dtype=dtype), dims="x")


def test_decode_bits():
    qa = _da([0b1011_0100])
    assert int(decode_bits(qa, 2)[0]) == 1
    assert int(decode_bits(qa, 4, 4)[0]) == 0b1011


def test_landsat_qa():
    clear = 1 << 6                    # clear bit
    cloud = (1 << 3) | (1 << 1)       # cloud + dilated
    shadow = 1 << 4
    snow = 1 << 5
    fill = 1
    m = landsat_qa_mask(_da([clear, cloud, shadow, snow, fill]))
    assert m.values.tolist() == [True, False, False, True, False]
    assert landsat_qa_mask(_da([snow]), snow=True).values.tolist() == [False]


def test_landsat_qa_float_nan():
    qa = xr.DataArray(np.array([64.0, np.nan]), dims="x")
    assert landsat_qa_mask(qa).values.tolist() == [True, False]


def test_s2_scl():
    m = s2_scl_mask(_da(list(range(12)), "uint8"))
    assert m.values.tolist() == [False, False, True, False, True, True, True, True,
                                 False, False, False, True]
    assert not bool(s2_scl_mask(_da([11], "uint8"), snow=True)[0])


def test_modis_state():
    clear, cloudy, shadow, cirrus = 0, 1, 1 << 2, 1 << 8
    m = modis_state_mask(_da([clear, cloudy, shadow, cirrus, 3]))
    assert m.values.tolist() == [True, False, False, False, True]
    assert modis_state_mask(_da([3]), allow_unset=False).values.tolist() == [False]


def test_modis_vi():
    assert modis_vi_mask(_da([0, 1, 2, 3])).values.tolist() == [True, True, False, False]


def test_mask_clouds_dataset():
    ds = xr.Dataset({"B04": ("x", np.array([1000, 2000, 3000], "uint16")),
                     "SCL": ("x", np.array([4, 9, 0], "uint8"))})
    out = mask_clouds(ds)  # sensor guessed from SCL
    assert np.isnan(out["B04"].values[1:]).all() and out["B04"][0] == 1000
    assert "SCL" in out and "SCL" not in mask_clouds(ds, drop_qa=True)


def test_scale_landsat():
    ds = xr.Dataset({"SR_B4": ("x", np.array([0, 10000], "uint16")),
                     "ST_B10": ("x", np.array([0, 44000], "uint16")),
                     "QA_PIXEL": ("x", np.array([1, 64], "uint16"))})
    out = scale(ds, sensor="landsat")
    assert np.isnan(out.SR_B4[0])
    assert out.SR_B4[1] == pytest.approx(10000 * 2.75e-5 - 0.2, rel=1e-6)
    assert out.ST_B10[1] == pytest.approx(44000 * 0.00341802 + 149, rel=1e-6)
    assert out.QA_PIXEL.dtype == np.uint16  # QA untouched


def test_scale_s2_offset_by_date():
    t = pd.to_datetime(["2021-06-01", "2023-06-01"])
    ds = xr.Dataset({"B04": (("time", "x"), np.array([[2000], [3000]], "uint16"))},
                    coords={"time": t})
    out = scale(ds, sensor="sentinel2")
    np.testing.assert_allclose(out.B04.values.ravel(), [0.2, 0.2], rtol=1e-6)
    np.testing.assert_allclose(scale(ds, sensor="sentinel2", harmonized=True).B04.values.ravel(),
                               [0.2, 0.3], rtol=1e-6)


def test_scale_s2_warns_without_time():
    ds = xr.Dataset({"B04": ("x", np.array([1000], "uint16"))})
    with pytest.warns(UserWarning, match="offset unknown"):
        scale(ds, sensor="sentinel2")


def test_scale_modis_fill():
    ds = xr.Dataset({"sur_refl_b01": ("x", np.array([-28672, 500, 20000], "int16"))})
    out = scale(ds)
    assert np.isnan(out.sur_refl_b01[0]) and np.isnan(out.sur_refl_b01[2])
    assert out.sur_refl_b01[1] == pytest.approx(0.05)
