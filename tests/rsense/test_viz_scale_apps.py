import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import geolattice.rsense as rs
from geolattice.rsense import apps, scale, viz
from geolattice.rsense.core import PointCloud


def test_stretch_methods():
    a = np.array([[0, 1, 2], [3, 4, np.nan]], float)
    for m in ("percentile", "minmax", "std", "equalize"):
        s = viz.stretch(a, m)
        assert np.nanmin(s) >= 0 and np.nanmax(s) <= 1 and np.isnan(s[1, 2])


def test_rgb_and_plots(reflectance):
    img = viz.rgb(reflectance.isel(time=0), "false_color")
    assert img.dims[-1] == "rgb" and img.shape[-1] == 3
    u8 = viz.rgb(reflectance.isel(time=0), as_uint8=True)
    assert u8.dtype == np.uint8
    assert viz.plot_rgb(reflectance.isel(time=0)) is not None
    ndvi = rs.optical.index(reflectance.isel(time=0), "NDVI")
    assert viz.plot_raster(ndvi) is not None
    cls = (ndvi > 0.5).astype(int)
    assert viz.plot_raster(cls, categorical={0: ("low", "tan"), 1: ("high", "green")}) is not None
    ts = rs.optical.index(reflectance, "NDVI")
    assert viz.plot_timeseries(ts) is not None
    pc = PointCloud.from_arrays(*np.random.default_rng(0).random((3, 500)))
    assert viz.plot_pointcloud(pc, backend="matplotlib") is not None


def test_zarr_cube(tmp_path, reflectance):
    scale.to_zarr_cube(reflectance, tmp_path / "c.zarr")
    back = scale.open_cube(tmp_path / "c.zarr")
    np.testing.assert_allclose(back.red.values, reflectance.red.values)
    assert back.rio.crs.to_epsg() == 32618
    more = reflectance.assign_coords(time=reflectance.time + pd.Timedelta("100D"))
    scale.to_zarr_cube(more, tmp_path / "c.zarr", append_dim="time")
    assert scale.open_cube(tmp_path / "c.zarr").sizes["time"] == 8
    rc = scale.rechunk_for(reflectance)
    assert rc.red.chunks[0] == (4,)


def test_stac_export(tmp_path, reflectance):
    pytest.importorskip("pystac")         # geolattice[rsense-access]
    files = []
    for i in range(2):
        p = tmp_path / f"ndvi_{i}.tif"
        rs.write(rs.optical.index(reflectance.isel(time=i), "NDVI"), p)
        files.append(p)
    cat = scale.export_stac(files, tmp_path / "stac", collection_id="ndvi")
    import pystac

    c = pystac.Catalog.from_file(str(cat))
    items = list(c.get_items(recursive=True))
    assert len(items) == 2
    props = items[0].properties
    assert props.get("proj:epsg") == 32618 or props.get("proj:code") == "EPSG:32618"
    assert -77 < items[0].bbox[0] < -75


def test_apps(tmp_path):
    assert set(apps.list_templates()) == {"index_explorer", "flood_mapper", "change_viewer"}
    p = apps.create_app("index_explorer", tmp_path)
    assert p.exists() and "streamlit" in p.read_text()
    with pytest.raises(FileExistsError):
        apps.create_app("index_explorer", tmp_path)
    compile(p.read_text(), str(p), "exec")


def test_lazy_submodules():
    for name in rs._SUBMODULES:
        assert getattr(rs, name).__name__ == f"geolattice.rsense.{name}"
    with pytest.raises(AttributeError):
        rs.not_a_module
