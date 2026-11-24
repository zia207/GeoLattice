import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

import geolattice.rsense as rs

ARRAY_BACKENDS = [b for b in rs.available_backends() if b in ("cpu", "cuda")]


@pytest.fixture(params=ARRAY_BACKENDS)
def backend(request):
    with rs.device(request.param) as b:
        yield b


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    rs.set_backend("cpu")


def _grid(ny=40, nx=50, res=10.0):
    y = 4_700_000 - res * (np.arange(ny) + 0.5)
    x = 400_000 + res * (np.arange(nx) + 0.5)
    return y, x


@pytest.fixture
def reflectance():
    """Scaled reflectance Dataset with canonical band names, 4 dates, UTM 18N."""
    rng = np.random.default_rng(0)
    y, x = _grid()
    t = pd.date_range("2026-06-01", periods=4, freq="10D")
    shape = (4, y.size, x.size)
    data = {
        "blue": rng.uniform(0.02, 0.08, shape), "green": rng.uniform(0.04, 0.12, shape),
        "red": rng.uniform(0.03, 0.15, shape), "rededge1": rng.uniform(0.1, 0.2, shape),
        "rededge2": rng.uniform(0.2, 0.3, shape), "rededge3": rng.uniform(0.25, 0.4, shape),
        "nir08": rng.uniform(0.25, 0.5, shape), "thermal": rng.uniform(290, 310, shape),
        "nir": rng.uniform(0.2, 0.5, shape), "swir16": rng.uniform(0.1, 0.3, shape),
        "swir22": rng.uniform(0.05, 0.2, shape),
        # Sentinel-3 OLCI bands 10/11/12 (681/709/754 nm), needed by MCI and OTCI.
        "oa10": rng.uniform(0.05, 0.15, shape), "oa11": rng.uniform(0.1, 0.25, shape),
        "oa12": rng.uniform(0.08, 0.2, shape),
    }
    ds = xr.Dataset({k: (("time", "y", "x"), v.astype("float32")) for k, v in data.items()},
                    coords={"time": t, "y": y, "x": x})
    return ds.rio.write_crs(32618)


@pytest.fixture
def single_band():
    rng = np.random.default_rng(1)
    y, x = _grid(64, 80)
    da = xr.DataArray(rng.normal(size=(64, 80)).astype("float32"),
                      coords={"y": y, "x": x}, dims=("y", "x"), name="value")
    return da.rio.write_crs(32618)
