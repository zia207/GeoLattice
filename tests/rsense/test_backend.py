import numpy as np
import pytest
import xarray as xr

import geolattice.rsense as rs


def test_default_is_cpu():
    assert rs.get_backend().name == "cpu"
    assert "cpu" in rs.available_backends()


def test_auto_resolves_to_available():
    b = rs.set_backend("auto")
    assert b.name in rs.available_backends()


@pytest.mark.skipif("cuda" in rs.available_backends(), reason="GPU present")
def test_cuda_falls_back_with_warning():
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        b = rs.set_backend("cuda")
    assert b.name == "cpu"


def test_invalid_backend():
    with pytest.raises(ValueError):
        rs.set_backend("tpu")


def test_device_context_restores():
    rs.set_backend("cpu")
    with rs.device("auto"):
        pass
    assert rs.get_backend().name == "cpu"


def test_roundtrip_xarray(backend, reflectance):
    moved = rs.to_device(reflectance)
    back = rs.to_cpu(moved)
    assert isinstance(back["red"].data, np.ndarray)
    xr.testing.assert_identical(back, reflectance)


def test_dask_stays_lazy(backend, reflectance):
    lazy = reflectance.chunk({"x": 25})
    moved = rs.to_device(lazy)
    assert hasattr(moved["red"].data, "dask")
    np.testing.assert_allclose(rs.to_cpu(moved)["red"].compute(), reflectance["red"])


def test_info_keys():
    info = rs.info()
    assert {"pyrsense_backend", "available_backends", "numpy", "gpus"} <= set(info)
