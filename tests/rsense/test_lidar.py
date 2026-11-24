import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import lidar
from geolattice.rsense.core import PointCloud


def _scene(seed=0):
    """Gently sloping terrain + 6 conical trees + one flat-roofed building."""
    rng = np.random.default_rng(seed)
    n = 40_000
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    ground_z = 100 + 0.05 * x + 0.02 * y
    z = ground_z.copy()
    cls_true = np.full(n, 2)
    trees = [(20, 20, 15), (25, 70, 12), (60, 30, 18), (70, 75, 10), (45, 55, 14), (85, 50, 16)]
    for tx, ty, h in trees:
        r = np.hypot(x - tx, y - ty)
        crown = r < h / 3
        canopy = (rng.random(n) < 0.8) & crown
        z[canopy] = ground_z[canopy] + h * (1 - r[canopy] / (h / 3)) + 1
        cls_true[canopy] = 5
    bld = (x > 75) & (x < 90) & (y > 5) & (y < 20)
    z[bld] = ground_z[bld] + 8
    cls_true[bld] = 6
    rn = np.where(cls_true == 2, 1, rng.choice([1, 2], n))
    pc = PointCloud.from_arrays(x + 400000, y + 4700000, z, crs=32618, return_number=rn)
    return pc, cls_true, trees


@pytest.mark.parametrize("method", ["smrf", "csf"])
def test_ground_classification(method):
    pc, truth, _ = _scene()
    out = lidar.classify_ground(pc, method)
    pred = out["classification"] == 2
    acc = (pred == (truth == 2)).mean()
    assert acc > 0.9
    assert pred[truth == 6].mean() < 0.2      # building roof not ground


def test_dem_chm_trees_crowns():
    pc, truth, trees = _scene()
    pc = lidar.smrf(pc)
    dtm = lidar.dem(pc, 1.0)
    xs = dtm.x.values - 400000
    expected = 100 + 0.05 * xs[None, :] + 0.02 * (dtm.y.values[:, None] - 4700000)
    assert np.nanmean(np.abs(dtm.values - expected)) < 0.3
    chm = lidar.chm(pc, res=1.0)
    assert float(chm.max()) == pytest.approx(18, abs=3)
    tops = lidar.detect_trees(chm, min_height=9, window=6)   # above the 8 m roof
    assert 5 <= len(tops) <= 7
    labels, crowns = lidar.segment_crowns(chm, tops, min_height=3)
    assert len(crowns) == len(tops) and (crowns.area_m2 > 5).all()
    pcn = lidar.normalize_heights(pc, dtm)
    m = lidar.grid_metrics(pcn, res=25)
    assert float(m.hmax.max()) == pytest.approx(18, abs=3)
    assert set(m.data_vars) >= {"p95", "cover", "density"}
    pct = lidar.assign_points_to_crowns(pc, labels)
    assert (pct["tree_id"] > 0).any()


def test_rasterize_stats():
    pc = PointCloud.from_arrays([0.5, 0.6, 1.5], [0.5, 0.5, 0.5], [1.0, 3.0, 5.0])
    for stat, exp in [("min", 1.0), ("max", 3.0), ("mean", 2.0), ("count", 2.0), ("sum", 4.0)]:
        g = lidar.rasterize(pc, 1.0, stat=stat, bounds=(0, 0, 2, 1))
        assert float(g.values[0, 0]) == exp


def test_terrain_on_plane():
    y = 1000 - 10 * (np.arange(50) + 0.5)
    x = 10 * (np.arange(60) + 0.5)
    X, Y = np.meshgrid(x, y)
    dem = xr.DataArray(0.1 * X, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    s = lidar.slope(dem)
    assert float(s[10, 10]) == pytest.approx(np.degrees(np.arctan(0.1)), rel=1e-6)
    a = lidar.aspect(dem)
    assert float(a[10, 10]) == pytest.approx(270.0)          # rises east → faces west
    hs = lidar.hillshade(dem)
    assert 0 <= float(hs.min()) and float(hs.max()) <= 255
    assert abs(float(lidar.curvature(dem)[10, 10])) < 1e-9
    assert float(lidar.tri(dem)[10, 10]) > 0
    north = xr.DataArray(0.1 * Y, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    assert float(lidar.aspect(north)[10, 10]) == pytest.approx(180.0)   # rises north → faces south


def test_hydrology_and_twi():
    y = 500 - 10 * (np.arange(30) + 0.5)
    x = 10 * (np.arange(30) + 0.5)
    X, Y = np.meshgrid(x, y)
    z = np.abs(X - 150) * 0.1 + (500 - Y) * 0.05          # V-shaped valley draining south
    z[10, 10] -= 2                                         # a pit
    dem = xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    filled = lidar.fill_depressions(dem)
    assert float(filled[10, 10]) > z[10, 10] + 0.4   # raised to the spill level
    acc = lidar.flow_accumulation(dem)
    assert float(acc.max()) > 100 and int(np.argmax(acc.values.max(0))) in (14, 15)
    t = lidar.twi(dem)
    assert float(t.values[:, 15].mean()) > float(t.values[:, 2].mean())


def test_geomorphons():
    y = 500 - 10 * (np.arange(41) + 0.5)
    x = 10 * (np.arange(41) + 0.5)
    X, Y = np.meshgrid(x, y)
    hill = 50 * np.exp(-((X - 205) ** 2 + (Y - 295) ** 2) / 8000)
    dem = xr.DataArray(hill, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    g = lidar.geomorphons(dem, search=8)
    assert int(g[20, 20]) == 2                              # peak
    pit = lidar.geomorphons(-dem, search=8)
    assert int(pit[20, 20]) == 10


def test_las_roundtrip(tmp_path):
    pytest.importorskip("laspy")          # geolattice[rsense-lidar]
    pc, _, _ = _scene()
    pc = lidar.smrf(pc)
    p = tmp_path / "t.laz"
    lidar.write_las(pc, p)
    back = lidar.read_las(p)
    assert len(back) == len(pc)
    np.testing.assert_allclose(back.z, pc.z, atol=1e-3)
    assert (back["classification"] == pc["classification"]).all()
    assert back.crs == 32618
    sub = lidar.read_las(p, bounds=(400000, 4700000, 400050, 4700050))
    assert 0 < len(sub) < len(pc)


def test_gedi_reader(tmp_path):
    import h5py

    p = tmp_path / "GEDI02_A.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("BEAM0101")
        n = 5
        g["shot_number"] = np.arange(n)
        g["lat_lowestmode"] = np.linspace(42.4, 42.5, n)
        g["lon_lowestmode"] = np.linspace(-76.6, -76.5, n)
        g["elev_lowestmode"] = np.full(n, 300.0)
        g["sensitivity"] = np.array([0.95, 0.99, 0.5, 0.97, 0.98])
        g["quality_flag"] = np.array([1, 1, 1, 0, 1], "uint8")
        g["degrade_flag"] = np.zeros(n, "uint8")
        g["rh"] = np.tile(np.linspace(0, 30, 101), (n, 1))
    gdf = lidar.read_gedi_l2a(p)
    assert len(gdf) == 3 and gdf.rh98.iloc[0] == pytest.approx(29.4)
    assert gdf.crs.to_epsg() == 4326
