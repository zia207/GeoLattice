import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter
from shapely.geometry import box

import geolattice.rsense as rs
from geolattice.rsense.core import iter_windows


def test_windows_cover_exactly():
    seen = np.zeros((101, 77), int)
    for w in iter_windows(101, 77, tile=32, overlap=5):
        seen[w.row, w.col] += 1
    assert (seen == 1).all()


@pytest.mark.parametrize("tile", [16, 25, 1000])
def test_apply_tiled_matches_full(backend, single_band, tile):
    size = 5
    full = uniform_filter(single_band.values, size, mode="reflect")
    if backend.array_gpu:
        import cupyx.scipy.ndimage as ndi

        def f(b):
            return ndi.uniform_filter(b, size, mode="reflect")
    else:
        def f(b):
            return uniform_filter(b, size, mode="reflect")
    out = rs.apply_tiled(single_band, f, tile=tile, overlap=size // 2)
    # interior matches exactly; image borders use the same reflect boundary
    np.testing.assert_allclose(out.values, full, rtol=1e-5, atol=1e-6)
    assert out.rio.crs == single_band.rio.crs


def test_apply_tiled_dask(single_band):
    lazy = single_band.chunk({"x": 20, "y": 20})
    out = rs.apply_tiled(lazy, lambda b: uniform_filter(b, 3, mode="reflect"), overlap=1)
    expected = uniform_filter(single_band.values, 3, mode="reflect")
    np.testing.assert_allclose(out.compute().values, expected, rtol=1e-5, atol=1e-6)


def test_apply_tiled_multiband(reflectance):
    da = reflectance.to_array("band").isel(time=0)
    out = rs.apply_tiled(da, lambda b: b * 2, tile=16)
    np.testing.assert_allclose(out, da * 2)


def test_read_write_roundtrip(tmp_path, reflectance):
    da = reflectance.isel(time=0)[["red", "nir"]].to_array("band")
    path = tmp_path / "x.tif"
    rs.write(da, path)
    back = rs.read(path, band_names=["red", "nir"])
    np.testing.assert_allclose(back.values, da.values, rtol=1e-6)
    assert back.rio.crs.to_epsg() == 32618
    ds = rs.read(path, band_names=["red", "nir"], as_dataset=True)
    assert set(ds.data_vars) == {"red", "nir"}


def test_write_requires_crs(tmp_path, single_band):
    # Build a CRS-free copy directly; ``rio.write_crs(None)`` raises MissingCRS in rioxarray>=0.19.
    da = xr.DataArray(single_band.values, coords={"y": single_band.y.values, "x": single_band.x.values},
                      dims=single_band.dims, name=single_band.name)
    assert da.rio.crs is None
    with pytest.raises(ValueError, match="CRS"):
        rs.write(da, tmp_path / "bad.tif")


def test_zonal_stats(single_band):
    x0, y1 = float(single_band.x[0]) - 5, float(single_band.y[0]) + 5
    zones = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[
        box(x0, y1 - 100, x0 + 100, y1),          # top-left 10x10 pixels
        box(x0 + 2000, y1 - 100, x0 + 2100, y1),  # outside raster
    ], crs=32618)
    out = rs.zonal_stats(single_band, zones, stats=["mean", "count", "p50"])
    block = single_band.values[:10, :10]
    assert out.loc[0, "value_count"] == 100
    assert out.loc[0, "value_mean"] == pytest.approx(block.mean(), rel=1e-5)
    assert out.loc[1, "value_count"] == 0 and np.isnan(out.loc[1, "value_mean"])


def test_clip(single_band):
    x0, y1 = float(single_band.x[0]) - 5, float(single_band.y[0]) + 5
    g = gpd.GeoDataFrame(geometry=[box(x0, y1 - 200, x0 + 300, y1)], crs=32618)
    assert rs.clip(single_band, g).shape == (20, 30)
