import numpy as np
import pytest

import geolattice.rsense as rs
from geolattice.rsense.backend import (
    as_compute_dtype,
    autocast,
    compute_dtype,
    ndimage,
    recommend_chunks,
    scatter_reduce,
    set_precision,
)


def test_precision_policy():
    assert compute_dtype() == np.float32
    set_precision("float64")
    try:
        assert as_compute_dtype(np.arange(3)).dtype == np.float64
    finally:
        set_precision("float32")
    with pytest.raises(ValueError):
        set_precision("int8")


def test_autocast_noop_without_torch():
    with autocast():
        pass


def test_recommend_chunks():
    ch = recommend_chunks((10, 20000, 20000), "float32", target_mb=64)
    assert ch[0] == 10 and ch[1] == ch[2] and ch[1] % 256 == 0


def test_ndimage_dispatch(backend):
    import scipy.ndimage as snd

    if not backend.array_gpu:
        assert ndimage() is snd


def test_scatter_reduce():
    t = np.full(3, np.inf)
    scatter_reduce(t, np.array([0, 0, 2]), np.array([5.0, 3.0, 1.0]), "min")
    assert t.tolist() == [3.0, np.inf, 1.0]
    a = np.zeros(2)
    scatter_reduce(a, np.array([1, 1]), np.array([1.0, 2.0]), "add")
    assert a.tolist() == [0.0, 3.0]


def test_memory_report():
    rep = rs.backend.memory_report()
    assert "backend" in rep


def test_host_only_functions_accept_device_arrays(backend, reflectance):
    # Host-side steps (smoothers, OBIA, plots, ...) pull data back with ``to_cpu``; calling
    # ``.values`` on a CuPy-backed DataArray first would raise before the copy happens.
    ndvi = rs.to_device(rs.optical.index(reflectance, "NDVI"))
    out = rs.timeseries.whittaker(ndvi, lmbda=10)
    assert out.shape == ndvi.shape
    assert np.isfinite(np.asarray(rs.to_cpu(out).values)).all()
