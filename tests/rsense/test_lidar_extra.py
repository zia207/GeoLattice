"""Tests for the lidar tutorial additions: filters, tiles, features, urban, change,
registration, hydrology, I/O extras, spaceborne L2B and lidar plots."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import lidar
from geolattice.rsense.core import PointCloud

X0, Y0 = 400_000.0, 4_700_000.0


def _scene(seed=0, n=40_000):
    """Sloping terrain + 6 conical trees + one flat-roofed 15×15 m building (8 m)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    ground_z = 100 + 0.05 * x + 0.02 * y
    z = ground_z.copy()
    truth = np.full(n, 2)
    trees = [(20, 20, 15), (25, 70, 12), (60, 30, 18), (70, 75, 10), (45, 55, 14), (85, 50, 16)]
    for tx, ty, h in trees:
        r = np.hypot(x - tx, y - ty)
        canopy = (rng.random(n) < 0.8) & (r < h / 3)
        z[canopy] = ground_z[canopy] + h * (1 - r[canopy] / (h / 3)) + 1
        truth[canopy] = 5
    bld = (x > 75) & (x < 90) & (y > 5) & (y < 20)
    z[bld] = ground_z[bld] + 8
    truth[bld] = 6
    rn = np.where(truth == 2, 1, rng.choice([1, 2], n))
    pc = PointCloud.from_arrays(x + X0, y + Y0, z, crs=32618, return_number=rn,
                                intensity=rng.integers(0, 400, n).astype("uint16"),
                                number_of_returns=np.full(n, 2, "uint8"))
    return pc, truth, trees, ground_z


@pytest.fixture(scope="module")
def scene():
    pc, truth, trees, gz = _scene()
    g = lidar.smrf(pc)
    dtm = lidar.dem(g, 1.0)
    pcn = lidar.normalize_heights(g, dtm)
    return {"pc": pc, "truth": truth, "trees": trees, "ground": g, "dtm": dtm, "pcn": pcn,
            "gz": gz}


def _plane(n=5000, seed=0, z0=0.0, noise=0.01):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 20, n), rng.uniform(0, 20, n)
    return PointCloud.from_arrays(x + X0, y + Y0, z0 + rng.normal(0, noise, n), crs=32618)


# ----------------------------------------------------------------------------- filters
@pytest.mark.parametrize("method", ["statistical", "radius"])
def test_outlier_removal(scene, method):
    pc = scene["pc"]
    rng = np.random.default_rng(3)
    k = 200
    nx, ny = rng.uniform(0, 100, k) + X0, rng.uniform(0, 100, k) + Y0
    noisy = PointCloud.from_arrays(np.r_[pc.x, nx], np.r_[pc.y, ny],
                                   np.r_[pc.z, rng.uniform(160, 220, k)], crs=32618)
    kw = {"k": 8, "std_ratio": 2.0} if method == "statistical" else {"radius": 2.0,
                                                                     "min_neighbors": 3}
    out = lidar.remove_outliers(noisy, method, action="mask", **kw)
    bad = np.asarray(out["outlier"])
    assert bad[-k:].mean() > 0.95 and bad[:-k].mean() < 0.02
    cls = lidar.remove_outliers(noisy, method, action="classify", **kw)["classification"]
    assert (cls[-k:] == 18).mean() > 0.9
    assert len(lidar.remove_outliers(noisy, method, **kw)) < len(noisy)


def test_noise_breaks_tree_detection_until_filtered(scene):
    pc = scene["pc"]
    rng = np.random.default_rng(4)
    k = 150
    noisy = PointCloud.from_arrays(np.r_[pc.x, rng.uniform(0, 100, k) + X0],
                                   np.r_[pc.y, rng.uniform(0, 100, k) + Y0],
                                   np.r_[pc.z, rng.uniform(150, 190, k)], crs=32618,
                                   return_number=np.r_[pc["return_number"], np.ones(k, "uint8")])
    clean = lidar.statistical_outlier_removal(noisy)
    g = lidar.smrf(clean)
    tops = lidar.detect_trees(lidar.chm(g, res=1.0), min_height=9, window=6)
    assert 5 <= len(tops) <= 7


def test_pmf_ground(scene):
    out = lidar.classify_ground(scene["pc"], "pmf")
    pred = out["classification"] == 2
    truth = scene["truth"]
    assert (pred == (truth == 2)).mean() > 0.9
    assert pred[truth == 6].mean() < 0.2


def test_thin_reproject_clip_summary(scene):
    pc = scene["pc"]
    for m in ("voxel", "grid_min", "grid_max"):
        t = lidar.thin(pc, 2.0, method=m)
        assert 0 < len(t) <= len(pc) / 5
    assert len(lidar.thin(pc, 2.0, method="grid_min")) <= 2600
    lo = lidar.thin(pc, 5.0, method="grid_min")
    hi = lidar.thin(pc, 5.0, method="grid_max")
    assert np.mean(lo.z) < np.mean(hi.z)
    assert len(lidar.thin(pc, method="random", fraction=0.1)) == len(pc) // 10
    geo = lidar.reproject(pc, 4326)
    assert geo.crs == 4326 and -77 < float(np.mean(geo.x)) < -75
    back = lidar.reproject(geo, 32618)
    np.testing.assert_allclose(back.x, pc.x, atol=1e-5)
    from shapely.geometry import box

    sub = lidar.clip_polygon(pc, box(X0, Y0, X0 + 50, Y0 + 50))
    assert 0.2 < len(sub) / len(pc) < 0.3
    out = lidar.clip_polygon(pc, box(X0, Y0, X0 + 50, Y0 + 50), invert=True)
    assert len(sub) + len(out) == len(pc)
    s = lidar.summary(scene["ground"])
    assert s["points"] == len(pc) and s["density_pts_m2"] == pytest.approx(4.0, rel=0.05)
    assert set(s["classes"]["class"]) >= {1, 2} and "returns" in s
    d = lidar.density_map(pc, 10.0)
    assert float(d.mean()) == pytest.approx(4.0, rel=0.1)


def test_vertical_accuracy_and_swath(scene):
    import geopandas as gpd

    xs = np.array([10.5, 30.5, 50.5, 80.5]) + X0
    ys = np.array([90.5, 10.5, 90.5, 40.5]) + Y0
    z = 100 + 0.05 * (xs - X0) + 0.02 * (ys - Y0)
    cp = gpd.GeoDataFrame({"z": z}, geometry=gpd.points_from_xy(xs, ys), crs=32618)
    va = lidar.vertical_accuracy(scene["dtm"], cp)
    assert va["n"] == 4 and va["rmse_z"] < 0.2
    assert va["nva_95"] == pytest.approx(1.96 * va["rmse_z"])
    p = _plane(8000)
    line = np.r_[np.zeros(4000, "uint16"), np.ones(4000, "uint16")]
    z2 = np.asarray(p.z).copy()
    z2[4000:] += 0.3
    p2 = PointCloud.from_arrays(p.x, p.y, z2, crs=32618, point_source_id=line)
    ov = lidar.swath_overlap(p2, 4.0, min_points=3)
    assert float(np.nanmedian(ov.dz)) == pytest.approx(0.3, abs=0.08)
    assert int(ov.n_lines.max()) == 2


# ----------------------------------------------------------------------------- tiles
def test_tiles_process_and_merge(scene, tmp_path):
    pc = scene["pc"]
    tiles = lidar.tile_grid(pc.bbox, 50, buffer=10)
    assert len(tiles) == 4 and tiles[0].buffered[0] == tiles[0].core[0] - 10
    out = lidar.process_tiles(pc, lambda sub: lidar.smrf(sub), size=50, buffer=10, n_jobs=2)
    assert len(out) == len(pc)
    whole = lidar.smrf(pc)
    agree = np.mean(np.sort(out["classification"]) == np.sort(whole["classification"]))
    assert agree > 0.95
    g = lidar.process_tiles(scene["ground"], lambda s: lidar.dem(s, 1.0), size=50, buffer=5)
    assert abs(g.shape[0] - scene["dtm"].shape[0]) <= 1
    assert float(np.nanmean(np.abs(g.values - scene["dtm"].reindex_like(g).values))) < 0.3
    tops = lidar.process_tiles(scene["ground"], lambda s: lidar.detect_trees(
        lidar.chm(s, res=1.0), min_height=9, window=6), size=50, buffer=15)
    assert 5 <= len(tops) <= 7
    paths = lidar.write_tiles(pc, tmp_path, 50)
    assert len(paths) == 4
    counts = lidar.process_files(paths, len, n_jobs=2)
    assert sum(counts) == len(pc)
    m = lidar.merge_pointclouds([lidar.read_las(p) for p in paths])
    assert len(m) == len(pc)


# ----------------------------------------------------------------------------- features
def test_eigen_features_and_clusters():
    p = _plane(3000)
    f = lidar.eigen_features(p, 16)
    assert float(np.median(f["scattering"])) < 0.01
    assert float(np.median(np.asarray(f["planarity"]) + np.asarray(f["linearity"]))) > 0.98
    assert float(np.median(f["nz"])) > 0.99 and float(np.median(f["verticality"])) < 0.05
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 30, 2000)
    line = PointCloud.from_arrays(t + X0, Y0 + 0.01 * rng.normal(size=2000),
                                  10 + 0.01 * rng.normal(size=2000))
    assert float(np.median(lidar.eigen_features(line, 16)["linearity"])) > 0.9
    nrm = lidar.normals(p)
    assert set(nrm.fields) >= {"nx", "ny", "nz"}
    a = rng.normal(0, 0.3, (300, 3))
    b = rng.normal(0, 0.3, (200, 3)) + [10, 0, 0]
    lab = lidar.euclidean_clusters(np.r_[a, b, [[50, 50, 50]]], 1.0, min_points=5)
    assert set(lab[:300]) == {0} and set(lab[300:500]) == {1} and lab[-1] == -1


def test_vertical_structure(scene):
    prof = lidar.height_profile(scene["pcn"], 1.0)
    assert {"lad", "gap_fraction", "fraction"} <= set(prof.columns)
    assert (prof.lad >= 0).all() and prof.gap_fraction.is_monotonic_increasing
    assert prof["count"].sum() == len(scene["pcn"])
    fhd = lidar.foliage_height_diversity(scene["pcn"])
    assert fhd > 1.0
    sm = lidar.structure_metrics(scene["pcn"], 25)
    assert set(sm.data_vars) == {"fhd", "lai", "crr"}
    assert float(sm.lai.max()) > float(sm.lai.min()) >= 0


def test_plot_metrics(scene):
    import geopandas as gpd

    plots = gpd.GeoDataFrame({"plot": ["tree", "open"]},
                             geometry=gpd.points_from_xy([60 + X0, 10 + X0], [30 + Y0, 90 + Y0]),
                             crs=32618)
    m = lidar.plot_metrics(scene["pcn"], plots, radius=6.0)
    assert m.loc[0, "hmax"] == pytest.approx(19, abs=2.5)
    assert m.loc[0, "cover"] > m.loc[1, "cover"]
    assert m.loc[1, "n_points"] > 300 and m.loc[0, "density"] == pytest.approx(4, rel=0.25)
    assert {"fhd", "lai", "crr", "p95"} <= set(m.columns)


def test_point_classifier(scene):
    pc = scene["pcn"].with_field("label", scene["truth"].astype("uint8"))
    fit = lidar.train_point_classifier(pc, "label", k=16, n_estimators=50, sample=20_000)
    assert fit["report"]["accuracy"] > 0.9
    tbl = lidar.point_feature_table(pc, scales=(8, 24))
    assert "k8_planarity" in tbl and "k24_verticality" in tbl and "height" in tbl
    pred = lidar.predict_point_classes(pc, fit, field="pred")
    assert (np.asarray(pred["pred"]) == scene["truth"]).mean() > 0.9


# ----------------------------------------------------------------------------- forest
def test_grid_metrics_matches_reference(scene):
    pcn = scene["pcn"]
    m = lidar.grid_metrics(pcn, 25, metrics=("hmax", "hmean", "p50", "cover", "density", "cv"))
    h = np.asarray(pcn["height"])
    x = np.asarray(pcn.x) - float(m.x[0] - 12.5)
    y = float(m.y[0] + 12.5) - np.asarray(pcn.y)
    sel = (x < 25) & (y < 25)
    veg = h[sel & (h > 0)]
    assert float(m.hmax[0, 0]) == pytest.approx(veg.max(), rel=1e-5)
    assert float(m.p50[0, 0]) == pytest.approx(np.percentile(veg, 50), rel=1e-5)
    first = np.asarray(pcn["return_number"]) == 1
    assert float(m.cover[0, 0]) == pytest.approx(np.mean(h[sel & first] > 2), rel=1e-5)
    assert float(m.density[0, 0]) == pytest.approx(sel.sum() / 625, rel=1e-5)


def test_segment_trees_points(scene):
    chm = lidar.chm(scene["ground"], res=1.0)
    tops = lidar.detect_trees(chm, min_height=9, window=6)
    seg = lidar.segment_trees_points(scene["pcn"], tops, min_height=2, crown_ratio=0.4)
    tid = np.asarray(seg["tree_id"])
    assert len(np.unique(tid[tid > 0])) == len(tops)
    canopy = (scene["truth"] == 5) & (np.asarray(scene["pcn"]["height"]) > 2)
    assert (tid[canopy] > 0).mean() > 0.85 and (tid[scene["truth"] == 2] == 0).all()


# ----------------------------------------------------------------------------- urban
def test_buildings_vegetation_footprints(scene):
    out = lidar.classify_buildings(scene["pcn"], min_points=30)
    c = np.asarray(out["classification"])
    truth = scene["truth"]
    assert (c[truth == 6] == 6).mean() > 0.8
    assert (c[truth == 5] == 6).mean() < 0.05
    assert (np.isin(c[truth == 5], [3, 4, 5])).mean() > 0.8
    fp = lidar.building_footprints(out, 1.0, min_area=50)
    assert len(fp) == 1
    assert fp.area_m2.iloc[0] == pytest.approx(225, rel=0.2)
    assert fp.height_m.iloc[0] == pytest.approx(8, abs=0.5)
    veg = lidar.classify_vegetation(scene["pcn"])
    assert set(np.unique(veg["classification"])) >= {3, 4, 5}


def _gable(n=6000, slope_deg=30, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 12, n), rng.uniform(0, 10, n)
    t = np.tan(np.radians(slope_deg))
    z = 110 - t * np.abs(y - 5) + rng.normal(0, 0.02, n)
    return PointCloud.from_arrays(x + X0, y + Y0, z, crs=32618)


def test_roof_planes_and_ransac():
    roof = _gable()
    plane, inl = lidar.ransac_plane(np.column_stack([roof.x - X0, roof.y - Y0, roof.z]))
    assert 0.4 < inl.mean() < 0.6
    seg, planes = lidar.segment_roof_planes(roof, min_points=200)
    assert len(planes) == 2
    assert planes.slope_deg.to_numpy() == pytest.approx([30, 30], abs=2)
    asp = sorted(planes.aspect_deg.round())
    assert asp[0] == pytest.approx(0, abs=3) or asp[1] == pytest.approx(360, abs=3)
    assert any(abs(a - 180) < 3 for a in asp)
    assert (np.asarray(seg["plane_id"]) >= 0).mean() > 0.9


def test_power_lines():
    rng = np.random.default_rng(0)
    g = _plane(4000, noise=0.02)
    t = rng.uniform(0, 20, 400)
    wx, wy = X0 + t, Y0 + 10 + 0.02 * rng.normal(size=400)
    wz = 12 + 0.002 * (t - 10) ** 2 + 0.02 * rng.normal(size=400)
    pc = PointCloud.from_arrays(np.r_[g.x, wx], np.r_[g.y, wy], np.r_[g.z, wz], crs=32618,
                                height=np.r_[np.asarray(g.z), wz].astype("float32"),
                                classification=np.r_[np.full(4000, 2),
                                                     np.ones(400)].astype("uint8"))
    out, lines = lidar.detect_power_lines(pc, min_length=10)
    assert len(lines) == 1 and lines.length_m.iloc[0] == pytest.approx(20, abs=1.5)
    assert (np.asarray(out["classification"])[-400:] == 14).mean() > 0.8


def test_solar_potential():
    sp = lidar.sun_positions(42.4)
    assert (sp.elevation > 0).all() and sp.elevation.max() < 72
    y = Y0 + 100 - (np.arange(60) + 0.5)
    x = X0 + np.arange(60) + 0.5
    X, Y = np.meshgrid(x, y)
    south = 0.5 * (Y - Y0)          # rises north → faces south
    dsm = xr.DataArray(south, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    s = lidar.solar_potential(dsm, shading=False, hours=[9, 12, 15])
    n = lidar.solar_potential(-dsm, shading=False, hours=[9, 12, 15])
    assert float(s.mean()) > float(n.mean()) + 0.2
    flat = xr.DataArray(np.zeros((40, 40)), dims=("y", "x"),
                        coords={"y": y[:40], "x": x[:40]}).rio.write_crs(32618)
    flat.values[20:25, 18:22] = 15         # a 15 m block casts a shadow north of it
    sh = lidar.solar_potential(flat, latitude=42.4, hours=[12], days=[355])
    assert float(sh[15, 20]) < 0.1 < float(sh[35, 20])
    import geopandas as gpd
    from shapely.geometry import box

    fp = gpd.GeoDataFrame(geometry=[box(X0 + 19, Y0 + 76, X0 + 21, Y0 + 79)], crs=32618)
    res = lidar.solar_potential(flat, latitude=42.4, footprints=fp, hours=[12], days=[172])
    assert res.solar_mean.iloc[0] > 0.5 and res.suitable_area_m2.iloc[0] > 0


# ----------------------------------------------------------------------------- change
def test_dem_of_difference_and_canopy_change(scene):
    before = scene["dtm"]
    after = before.copy(data=before.values.copy())
    after.values[10:20, 10:20] -= 1.0          # 100 m² eroded 1 m
    after.values[50:55, 50:60] += 0.5          # 50 m² deposited 0.5 m
    after.values += np.random.default_rng(0).normal(0, 0.03, after.shape)
    d = lidar.dem_of_difference(before, after, sigma_before=0.05, sigma_after=0.05)
    assert d["erosion_m3"] == pytest.approx(100, rel=0.05)
    assert d["deposition_m3"] == pytest.approx(25, rel=0.1)
    assert d["lod"] == pytest.approx(1.96 * np.hypot(0.05, 0.05))
    chm = lidar.chm(scene["ground"], res=1.0)
    cut = chm.copy(data=chm.values.copy())
    cut.values[(cut.values > 5) & (cut.x.values[None, :] < X0 + 40)] = 0
    cc = lidar.canopy_change(chm, cut, min_area=5)
    assert int(cc.loss.sum()) > 50 and int(cc.loss.where(cc.x > X0 + 45).sum()) == 0


def test_c2c_and_m3c2():
    a = _plane(6000, seed=1)
    b = _plane(6000, seed=2, z0=0.5)
    c = lidar.c2c_distance(a, b, signed=True)
    assert float(np.median(c["c2c"])) == pytest.approx(0.5, abs=0.1)
    c_down = lidar.c2c_distance(b, a, signed=True)
    assert float(np.median(c_down["c2c"])) < 0
    cores = lidar.thin(a, 4.0, method="voxel")
    m = lidar.m3c2(a, b, core_points=cores, normal_scale=3.0, projection_scale=2.0)
    v = np.asarray(m["m3c2"])
    assert np.nanmedian(v) == pytest.approx(0.5, abs=0.02)
    assert np.asarray(m["significant"])[np.isfinite(v)].mean() > 0.95
    same = lidar.m3c2(a, _plane(6000, seed=3), core_points=cores, normal_scale=3.0,
                      projection_scale=2.0)
    assert np.asarray(same["significant"]).mean() < 0.15


# ----------------------------------------------------------------------------- registration
def test_icp_recovers_transform():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 30, 5000), rng.uniform(0, 30, 5000)
    z = 2 * np.sin(x / 5) + np.cos(y / 4) + 0.5 * np.sin((x + y) / 3)
    tgt = PointCloud.from_arrays(x + X0, y + Y0, z + 100, crs=32618)
    T = lidar.rigid_transform((0, 0, 3), (0.8, -0.5, 0.3))
    src = lidar.apply_transform(tgt, T, origin=(X0 + 15, Y0 + 15, 100))
    res = lidar.icp(src, tgt, max_iterations=80)
    assert res["rmse"] < 0.02
    np.testing.assert_allclose(res["aligned"].x, tgt.x, atol=0.05)
    np.testing.assert_allclose(res["aligned"].z, tgt.z, atol=0.05)


def test_circle_and_stems():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 200)
    cx, cy, r, rmse = lidar.fit_circle(np.c_[5 + 0.2 * np.cos(a), 3 + 0.2 * np.sin(a)])
    assert (cx, cy, r) == pytest.approx((5, 3, 0.2), abs=1e-6) and rmse < 1e-9
    stems = [(10, 10, 0.15), (20, 15, 0.25), (15, 25, 0.10)]
    xs, ys, hs = [], [], []
    for sx, sy, sr in stems:
        ang = rng.uniform(0, 1.6 * np.pi, 600)            # scanned from one side only
        h = rng.uniform(0, 3, 600)
        xs.append(sx + sr * np.cos(ang) + rng.normal(0, 0.005, 600))
        ys.append(sy + sr * np.sin(ang) + rng.normal(0, 0.005, 600))
        hs.append(h)
    gx, gy = rng.uniform(0, 30, 3000), rng.uniform(0, 30, 3000)
    x, y = np.r_[np.concatenate(xs), gx], np.r_[np.concatenate(ys), gy]
    h = np.r_[np.concatenate(hs), np.zeros(3000)]
    pc = PointCloud.from_arrays(x + X0, y + Y0, h + 100, crs=32618, height=h.astype("float32"))
    out = lidar.detect_stems(pc).sort_values("dbh_cm").reset_index(drop=True)
    assert len(out) == 3
    np.testing.assert_allclose(out.dbh_cm, [20, 30, 50], atol=1.5)


# ----------------------------------------------------------------------------- hydrology
def _valley():
    y = 1000 - 10 * (np.arange(40) + 0.5)
    x = 10 * (np.arange(40) + 0.5)
    X, Y = np.meshgrid(x, y)
    z = np.abs(X - 200) * 0.1 + (1000 - Y) * 0.02 + 50
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)


def test_flow_accumulation_sparse_and_weights():
    dem = _valley()
    acc = lidar.flow_accumulation(dem)
    assert float(acc.sum()) > dem.size and float(acc.max()) >= dem.size * 0.45   # two channels
    w = lidar.flow_accumulation(dem, weights=np.full(dem.shape, 2.0))
    np.testing.assert_allclose(w.values, 2 * acc.values, rtol=1e-6)


def test_hand_and_inundation():
    dem = _valley()
    hd = lidar.hand(dem, threshold=50)
    col = int(np.argmin(np.abs(dem.x.values - 200)))      # channel cells: columns 19 and 20
    assert float(np.nanmax(np.abs(hd.values[:30, 19:21]))) < 0.6
    assert float(hd.values[10, 18]) == pytest.approx(1.0, abs=0.3)
    assert float(hd.values[10, 2]) == pytest.approx(0.1 * (195 - float(dem.x[2])), abs=1.5)
    fl = lidar.inundation(hd, 2.0)
    assert set(fl.data_vars) == {"depth", "extent"}
    wet_cols = np.flatnonzero(fl.extent.values[20])
    assert wet_cols.min() >= col - 3 and wet_cols.max() <= col + 3
    bath = lidar.inundation(dem, 52.0)
    assert int(bath.extent.sum()) > 0 and float(bath.depth.max()) <= 2.0 + 1e-6


def test_flatten_water():
    from shapely.geometry import box

    dem = _valley()
    noisy = dem.copy(data=dem.values + np.random.default_rng(0).normal(0, 0.5, dem.shape))
    lake = box(150, 700, 250, 800)
    flat = lidar.flatten_water(noisy, [lake])
    inside = (dem.x.values[None, :] > 150) & (dem.x.values[None, :] < 250) & \
        (dem.y.values[:, None] > 700) & (dem.y.values[:, None] < 800)
    assert np.ptp(flat.values[inside]) == 0
    assert np.array_equal(flat.values[~inside], noisy.values[~inside])


# ----------------------------------------------------------------------------- I/O, sampling
def test_las_info_chunks_ply(scene, tmp_path):
    pytest.importorskip("laspy")
    p = tmp_path / "s.laz"
    lidar.write_las(scene["ground"], p)
    info = lidar.las_info(p)
    assert info["point_count"] == len(scene["pc"]) and info["crs"] == 32618
    assert "classification" in info["dimensions"] and info["compressed"]
    chunks = list(lidar.read_las_chunks(p, 15_000))
    assert len(chunks) == 3 and sum(map(len, chunks)) == len(scene["pc"])
    ply = tmp_path / "s.ply"
    lidar.write_ply(scene["ground"], ply, fields=["classification", "intensity"])
    head = ply.read_bytes()[:400].decode(errors="ignore")
    assert f"element vertex {len(scene['pc'])}" in head and "property int classification" in head
    lidar.write_ply(lidar.thin(scene["pc"], method="random", fraction=0.01), tmp_path / "a.ply",
                    binary=False)
    assert (tmp_path / "a.ply").stat().st_size > 1000


def test_pdal_helpers_optional(scene):
    pytest.importorskip("pdal")
    out = lidar.pdal_pipeline([{"type": "filters.range", "limits": "Z[105:]"}], scene["pc"])
    assert 0 < len(out) < len(scene["pc"])


def test_sample_raster_and_colorize(scene):
    dtm = scene["dtm"]
    s = lidar.sample_raster(scene["pc"], dtm, "dtm_z")
    gz = scene["gz"]
    assert np.nanmean(np.abs(np.asarray(s["dtm_z"]) - gz)) < 0.3
    b = lidar.sample_raster(scene["pc"], dtm, "dtm_b", method="bilinear")
    assert np.nanmean(np.abs(np.asarray(b["dtm_b"]) - gz)) < 0.2
    img = xr.Dataset({k: dtm * f for k, f in (("red", 1.0), ("green", 0.5), ("blue", 0.2),
                                                ("nir", 2.0))})
    col = lidar.colorize(scene["pc"], img, nir="nir")
    assert {"red", "green", "blue", "nir"} <= set(col.fields)
    assert col["red"].dtype == np.uint16 and int(np.max(col["red"])) == 65535


def test_gedi_l2b_reader(tmp_path):
    import h5py

    p = tmp_path / "GEDI02_B.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("BEAM0110")
        geo = g.create_group("geolocation")
        n = 4
        g["shot_number"] = np.arange(n)
        geo["lat_lowestmode"] = np.linspace(42.4, 42.5, n)
        geo["lon_lowestmode"] = np.linspace(-76.6, -76.5, n)
        g["cover"] = np.array([0.8, 0.5, -9999, 0.3], "f4")
        g["pai"] = np.array([3.0, 1.5, -9999, 0.6], "f4")
        g["fhd_normal"] = np.array([2.5, 2.0, -9999, 1.0], "f4")
        g["rh100"] = np.array([2500, 1800, 0, 900], "int16")
        g["sensitivity"] = np.full(n, 0.97, "f4")
        g["l2b_quality_flag"] = np.array([1, 1, 1, 0], "uint8")
        g["algorithmrun_flag"] = np.ones(n, "uint8")
        g["pavd_z"] = np.tile(np.linspace(0.2, 0, 30, dtype="f4"), (n, 1))
    gdf = lidar.read_gedi_l2b(p, profiles=True)
    assert len(gdf) == 2 and gdf.rh100.tolist() == [25.0, 18.0]
    assert "pavd_z0" in gdf and gdf.crs.to_epsg() == 4326
    assert len(lidar.read_gedi_l2b(p, quality=False)) == 4


# ----------------------------------------------------------------------------- plots
def test_cross_section_and_plots(scene):
    import matplotlib

    matplotlib.use("Agg")
    sec = lidar.cross_section(scene["ground"], (X0, Y0 + 30), (X0 + 100, Y0 + 30), width=2)
    assert 650 < len(sec) < 950                              # 2 m × 100 m × 4 pts/m²
    assert float(np.max(sec["distance"])) <= 100 and float(np.max(np.abs(sec["offset"]))) <= 1
    ax = lidar.plot_cross_section(scene["ground"], (X0, Y0 + 30), (X0 + 100, Y0 + 30))
    assert ax.get_xlabel().startswith("Distance")
    lidar.plot_height_profile(lidar.height_profile(scene["pcn"]))
    lidar.plot_classification(scene["ground"])
    with pytest.raises(ValueError):
        lidar.cross_section(scene["pc"], (X0, Y0), (X0, Y0))


def test_process_tiles_gdf_and_list(scene):
    res = lidar.process_tiles(scene["pc"], len, size=50, buffer=5, merge=False)
    assert len(res) == 4 and all(isinstance(r[1], int) for r in res)
    assert isinstance(lidar.process_tiles(scene["pc"], lambda s: {"n": len(s)}, size=50), list)
    assert isinstance(lidar.summary(scene["pc"])["bounds"], dict)
    assert isinstance(pd.DataFrame(lidar.process_tiles(scene["pc"], lambda s: {"n": len(s)},
                                                       size=50)), pd.DataFrame)
