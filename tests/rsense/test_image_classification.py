"""Tests for geolattice.rsense.classify.image_classification (classical parts always; deep parts with torch)."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point, box

from geolattice.rsense.classify import image_classification as ic


# ---------------------------------------------------------------------------------- fixtures
def _grid(ny=48, nx=60, res=10.0):
    y = 4_700_000 - res * (np.arange(ny) + 0.5)
    x = 400_000 + res * (np.arange(nx) + 0.5)
    return y, x


@pytest.fixture
def scene():
    """4-band (band, y, x) image with 3 spectrally distinct vertical stripes + a nodata hole."""
    rng = np.random.default_rng(0)
    y, x = _grid()
    cls = np.zeros((y.size, x.size), int)
    cls[:, 20:40] = 1
    cls[:, 40:] = 2
    means = np.array([[0.05, 0.08, 0.04, 0.30], [0.10, 0.12, 0.15, 0.20], [0.03, 0.04, 0.03, 0.02]])
    img = means[cls].transpose(2, 0, 1) + rng.normal(0, 0.005, (4, y.size, x.size))
    img = img.astype("float32")
    img[:, :3, :3] = np.nan
    da = xr.DataArray(img, dims=("band", "y", "x"),
                      coords={"band": ["blue", "green", "red", "nir"], "y": y, "x": x})
    return da.rio.write_crs(32618), cls


def _points(da, cls, n=300, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.integers(3, da.sizes["y"], n)
    c = rng.integers(3, da.sizes["x"], n)
    return gpd.GeoDataFrame({"cls": cls[r, c]},
                            geometry=[Point(da.x.values[j], da.y.values[i]) for i, j in zip(r, c)],
                            crs=32618)


# ---------------------------------------------------------------------- 0/1. config and data
def test_run_config_validation():
    cfg = ic.RunConfig(task="change-detection", learning_mode="unsupervised", spatial_unit="object")
    assert cfg.to_dict()["task"] == "change-detection"
    with pytest.raises(ValueError):
        ic.RunConfig(task="nope")


def test_load_stack_folder_metadata_and_nodata(tmp_path, scene):
    da, _ = scene
    for i, b in enumerate(["B2", "B3", "B11", "B8A"]):
        band = da.isel(band=i).fillna(-9999).rio.write_nodata(-9999)
        band.rio.to_raster(tmp_path / f"{b}.tif")
    da.isel(band=0).rio.to_raster(tmp_path / "multi_skip.tif")
    out = ic.load_stack(tmp_path, sensor="test", acquisition_dates=["2026-06-01"])
    assert list(out.band.values) == ["B2", "B3", "B8A", "B11"]  # natural sort, multi skipped
    assert out.rio.crs.to_epsg() == 32618 and np.isnan(out.values[0, 0, 0])
    meta = ic.RasterMeta.from_dataarray(out)
    assert meta.nodata == -9999 and meta.sensor == "test" and meta.shape == (48, 60)
    json.dumps(meta.to_dict())


def test_load_stack_assigns_missing_crs_and_rejects_grid_mismatch(tmp_path, scene):
    da, _ = scene
    da.isel(band=0).rio.to_raster(tmp_path / "a.tif")
    da.isel(band=1, x=slice(0, 30)).rio.to_raster(tmp_path / "b.tif")
    with pytest.raises(ValueError, match="grid"):
        ic.load_stack([tmp_path / "a.tif", tmp_path / "b.tif"])
    one = ic.load_stack(tmp_path / "a.tif", crs="EPSG:4326")  # file CRS wins
    assert one.rio.crs.to_epsg() == 32618


def test_spatial_splits_do_not_leak():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 1000, (500, 2))
    y = (coords[:, 0] > 500).astype(int)
    s = ic.spatial_block_split(coords, block_size=200, test_size=0.3, val_size=0.1, stratify=y)
    s.check()
    gtr, gte = set(s.groups[s.train]), set(s.groups[s.test])
    assert not gtr & gte and len(s.val) > 0
    folds = list(ic.spatial_kfold(coords, block_size=200, n_splits=4))
    assert len(folds) == 4 and sum(len(te) for _, te in folds) == 500
    t = ic.temporal_split(pd.date_range("2026-01-01", periods=10, freq="MS"),
                          train_end="2026-05-01", val_end="2026-07-01")
    assert t.train.max() < t.val.min() <= t.val.max() < t.test.min()
    g = ic.group_split(np.repeat(np.arange(10), 5), test_size=0.2)
    assert len(g.test) == 10


def test_normalizer_fit_on_train_only_roundtrip(scene):
    da, _ = scene
    n = ic.Normalizer("zscore").fit(da)
    z = n.transform(da)
    assert abs(float(np.nanmean(z.values[0]))) < 1e-4
    n2 = ic.Normalizer.from_dict(json.loads(json.dumps(n.to_dict())))
    np.testing.assert_allclose(n2.transform(da.values), z.values, rtol=1e-5)
    p = ic.Normalizer("percentile").fit(np.random.default_rng(0).normal(size=(100, 3)))
    assert p.transform(np.zeros((5, 3))).shape == (5, 3)
    with pytest.raises(RuntimeError):
        ic.Normalizer().transform(np.zeros((2, 2)))


# --------------------------------------------------------------------- 2. classical baselines
def test_supervised_pixel_workflow(scene, tmp_path):
    da, cls = scene
    pts = _points(da, cls)
    X, y, coords = ic.sample_pixels(da, pts, "cls")
    assert X.shape[1] == 4 and len(y) == len(coords)
    split = ic.spatial_block_split(coords, block_size=150, test_size=0.3, stratify=y)
    clf, cv = ic.train_pixel_classifier(X.iloc[split.train], y[split.train], model="rf",
                                        coords=coords[split.train], block_size=150, n_splits=3,
                                        n_estimators=50)
    assert len(cv) == 3 and cv["accuracy"].mean() > 0.9
    rep = ic.classification_report(y[split.test], clf.predict(X.iloc[split.test]), n_boot=50)
    assert rep["overall_accuracy"] > 0.95 and "ci" in rep and "f1_ci_low" in rep["per_class"]
    m = ic.predict_map(clf, da)
    assert m.shape == cls.shape and m.values[0, 0] == -1
    assert (m.values[3:, 3:] == cls[3:, 3:]).mean() > 0.97
    path = ic.write_class_map(m, tmp_path / "lc.tif", class_names=["a", "b", "c"],
                              colormap={0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)},
                              metadata={"model": "rf"})
    import rasterio

    with rasterio.open(path) as src:
        assert src.nodata == 255 and src.crs.to_epsg() == 32618
        assert json.loads(src.tags()["class_names"])["1"] == "b"
        assert src.read(1)[0, 0] == 255 and src.colormap(1)[1][:3] == (0, 255, 0)
    named = pts.assign(name=np.array(["x", "y", "z"])[pts.cls])
    lab, codes = ic.rasterize_labels(named, da, "name")
    assert set(codes) == {"x", "y", "z"} and (lab.values >= 0).sum() > 0
    # pandas string dtype (the default for text columns in pandas 3) must work the same way
    lab_s, codes_s = ic.rasterize_labels(named.astype({"name": "string"}), da, "name")
    assert codes_s == codes and np.array_equal(lab_s.values, lab.values)
    X2, y2, _ = ic.sample_pixels(da, lab)
    assert len(X2) == len(y2) > 0
    em = ic.error_map(m, lab)
    assert np.nanmean(em.values) > 0.95
    base = ic.compare_with_baselines(y[split.train], y[split.test],
                                     {"rf": clf.predict(X.iloc[split.test])})
    assert base.loc["rf", "overall_accuracy"] > base.loc["majority_baseline", "overall_accuracy"]


@pytest.mark.parametrize("method", ["kmeans", "minibatch_kmeans", "gmm"])
def test_unsupervised(scene, method):
    da, cls = scene
    m, model = ic.unsupervised_classify(da, 3, method=method, sample=2000)
    v = m.values[3:, 3:]
    # each true class maps to a single cluster
    purity = np.mean([np.bincount(v[cls[3:, 3:] == k]).max() / (cls[3:, 3:] == k).sum()
                      for k in range(3)])
    assert purity > 0.95 and m.values[0, 0] == -1 and len(model.cluster_means_) == 3


def test_object_based(scene):
    da, cls = scene
    res = ic.object_based_classify(da, _points(da, cls, 200), "cls", n_segments=150,
                                   compactness=0.1, n_estimators=50)
    v = res["map"].values
    ok = v >= 0
    assert (v[ok] == cls[ok]).mean() > 0.9 and len(res["features"]) > 20


def test_semi_supervised_and_active_learning(scene):
    from geolattice.rsense.classify import Classifier

    da, cls = scene
    X, y, _ = ic.sample_pixels(da, _points(da, cls, 400), "cls")
    Xa = X.to_numpy("float32")
    st = ic.self_training(lambda: Classifier("rf", n_estimators=30, random_state=0), Xa[:30],
                          y[:30], Xa[30:300], threshold=0.8, X_val=Xa[300:], y_val=y[300:])
    assert st["history"]["val_accuracy"].iloc[-1] > 0.9 and len(st["pseudo_labeled"]) > 0
    al = ic.active_learning_loop(lambda: Classifier("rf", n_estimators=30, random_state=0),
                                 Xa[:20], y[:20], Xa[20:300], oracle=lambda i: y[20:300][i],
                                 rounds=2, batch=15, X_val=Xa[300:], y_val=y[300:])
    assert al["history"]["n_labeled"].tolist() == [20, 35, 50] and len(al["queried"]) == 30
    p = np.array([[0.98, 0.01, 0.01], [0.4, 0.35, 0.25], [0.5, 0.5, 0.0]])
    assert ic.select_for_labeling(p, 1, strategy="entropy")[0] == 1
    assert ic.select_for_labeling(p, 1, strategy="margin")[0] == 2
    idx, lab = ic.pseudo_label(p, 0.9, classes=["a", "b", "c"])
    assert idx.tolist() == [0] and lab.tolist() == ["a"]


def test_domain_adaptation_helpers(scene):
    rng = np.random.default_rng(0)
    src = rng.normal(0, 1, (500, 3))
    tgt = rng.normal(5, 3, (500, 3))
    out = ic.coral(src, tgt, eps=1e-3)
    np.testing.assert_allclose(out.mean(0), tgt.mean(0), atol=0.05)
    np.testing.assert_allclose(np.cov(out, rowvar=False), np.cov(tgt, rowvar=False), rtol=0.05,
                               atol=0.05)
    da, _ = scene
    hm = ic.histogram_match(da * 2 + 1, da)
    np.testing.assert_allclose(np.nanmean(hm.values), np.nanmean(da.values), rtol=1e-3)
    codes = np.zeros((48, 60), int) + 30
    codes[:, :20], codes[:, 20:40] = 10, 20
    prod = da.isel(band=0, drop=True).copy(data=codes)
    wl = ic.weak_labels_from_product(prod, {10: 0, 20: 1}, erode=2)
    assert set(np.unique(wl.values)) == {-100, 0, 1}
    assert wl.values[10, 40] == -100 and wl.values[10, 19] == -100 and wl.values[10, 10] == 0
    assert ic.check_split_isolation(train=[1, 2], adapt=[3], test=[4])
    with pytest.raises(ValueError):
        ic.check_split_isolation(train=[1, 2], test=[2])


# ------------------------------------------------------------------- 6/7. detection, vectors
def test_box_metrics_nms_map():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    iou = ic.box_iou(a, a)
    assert iou[0, 0] == 1 and abs(iou[0, 1] - 25 / 175) < 1e-9
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]])
    keep = ic.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]
    gt = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "labels": np.array([1, 2])}]
    perfect = [{"boxes": gt[0]["boxes"], "scores": np.array([0.9, 0.8]),
                "labels": np.array([1, 2])}]
    assert ic.mean_average_precision(perfect, gt)["map"] == pytest.approx(1.0)
    half = [{"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), "scores": np.array([0.9, 0.95]),
             "labels": np.array([1, 2])}]
    r = ic.mean_average_precision(half, gt, iou_thresholds=np.arange(0.5, 0.96, 0.05))
    assert r["map"] == pytest.approx(0.5)
    p = ic.obb_to_polygon(0, 0, 4, 2, 90)
    assert ic.polygon_iou([p], [ic.obb_to_polygon(0, 0, 2, 4, 0)])[0, 0] == pytest.approx(1.0)
    tiles = ic.detection_tiles(1000, 800, tile=256, overlap=0.25, scales=(1.0, 2.0))
    assert {t[2] for t in tiles} == {256, 128}


def test_merge_detections_and_count():
    g = gpd.GeoDataFrame({"label": ["tree", "tree", "tree", "cow"], "score": [0.9, 0.8, 0.7, 0.6]},
                         geometry=[box(0, 0, 10, 10), box(1, 1, 10, 10), box(50, 50, 60, 60),
                                   box(1, 1, 10, 10)], crs=32618)
    m = ic.merge_detections(g, iou_threshold=0.5)
    assert len(m) == 3
    zones = gpd.GeoDataFrame({"name": ["a", "b"]},
                             geometry=[box(-5, -5, 30, 30), box(40, 40, 70, 70)], crs=32618)
    z = ic.count_objects(m, zones, label="tree")
    assert z["count"].tolist() == [1, 1]


def test_instances_polygonize_export(scene, tmp_path):
    da, _ = scene
    lab = np.zeros((48, 60), int)
    lab[5:20, 5:20] = 1
    lab[5:20, 21:40] = 1   # touching-ish second field (1 px gap)
    lab[30:45, 10:50] = 2
    lab[0, 59] = 1          # speck
    cm = da.isel(band=0, drop=True).copy(data=lab)
    clean = ic.clean_mask(cm > 0, min_size=5, hole_size=5)
    assert clean.values[0, 59] == 0
    inst, tab = ic.instances_from_semantic(cm, min_size=5)
    assert len(tab) == 3 and set(tab["class"]) == {1, 2}
    ws, wtab = ic.instances_from_semantic(cm, method="watershed", min_distance=4, min_size=5)
    assert len(wtab) >= 3
    w = ic.boundary_weight_map(np.asarray(inst.values), w0=10, sigma=3)
    assert w[12, 20] > w[12, 10]  # gap between the two fields is up-weighted
    assert ic.mask_boundaries(np.asarray(inst.values)).sum() > 0
    conf = cm.copy(data=np.full(lab.shape, 0.8))
    polys = ic.polygonize(inst, class_map=cm, confidence=conf, class_names={1: "wheat", 2: "rice"},
                          regularize="auto", simplify=1.0)
    assert len(polys) == 3 and polys.crs.to_epsg() == 32618
    assert set(polys["class_name"]) == {"wheat", "rice"} and polys["confidence"].iloc[0] == 0.8
    assert polys["area"].max() == pytest.approx(15 * 40 * 100, rel=0.05)
    for ext in (".gpkg", ".geojson"):
        p = ic.export_vectors(polys, tmp_path / f"f{ext}")
        back = gpd.read_file(p)
        assert len(back) == 3 and back.crs.to_epsg() == 32618
    m = ic.segmentation_metrics(cm, cm, num_classes=3)
    assert m["miou"] == pytest.approx(1.0)
    off = cm.copy(data=np.where(lab == 2, 1, lab))
    m2 = ic.segmentation_metrics(off, cm, num_classes=3)
    assert m2["per_class"].loc[2, "recall"] == 0


# ----------------------------------------------------------------------- 8/9. change, fusion
def test_change_detection_classical(scene):
    da, _ = scene
    after = da.copy(data=da.values.copy())
    after.values[:, 10:20, 10:20] += 0.2
    ref = np.zeros((48, 60), bool)
    ref[10:20, 10:20] = True
    cva = ic.change_vector_analysis(da, after)
    assert ic.change_metrics(cva["change"], ref)["f1"] > 0.95
    pc = ic.pca_change(da, after, method="difference")
    assert ic.change_metrics(pc["change"], ref)["iou"] > 0.9
    st = ic.pca_change(da, after, method="stacked")
    assert len(st.attrs["explained_variance_ratio"]) == 8
    r = ic.image_ratio(da.isel(band=3), after.isel(band=3))
    assert bool(r["increase"].values[15, 15]) and not bool(r["increase"].values[30, 30])
    d = ic.image_difference(da.isel(band=3), after.isel(band=3))
    assert bool(d["increase"].values[15, 15])
    shifted = da.copy(data=np.roll(np.nan_to_num(da.values), 2, axis=2))
    with pytest.warns(UserWarning, match="misregistered"):
        s = ic.check_coregistration(da.fillna(0), shifted, band="nir")
    assert abs(abs(s["dx"]) - 2) < 0.2


def test_fusion_gap_fill_pansharpen(scene):
    da, _ = scene
    opt = da.isel(band=slice(0, 3))
    sar = da.isel(band=[3]).assign_coords(band=["vv"])
    f = ic.fuse_early(opt, sar)
    assert f.sizes["band"] == 4 and str(f.band.values[-1]) == "sar_vv"
    with pytest.raises(ValueError):
        ic.fuse_early(opt, sar.isel(x=slice(0, 30)))
    p1 = np.full((5, 3), 0.2)
    p2 = np.tile([1.0, 0, 0], (5, 1))
    np.testing.assert_allclose(ic.fuse_late([p1, p2], weights=[1, 3])[0], [0.8, 0.05, 0.05])
    cloudy = ic.as_band_dataset(opt.copy(data=opt.values.copy()))
    cloudy["red"].values[20:30, 20:30] = np.nan
    filled, scores = ic.sar_assisted_gap_fill(cloudy, ic.as_band_dataset(sar), n_estimators=20)
    assert int(filled["reconstructed"].sum()) == 100
    assert np.isfinite(filled["red"].values[20:30, 20:30]).all()
    ms = np.random.default_rng(0).uniform(0.1, 0.3, (3, 32, 32)).astype("float32")
    pan = ms.mean(0)
    out = ic.pansharpen(ms, pan, method="brovey")
    q = ic.pansharpening_quality(out, ms)
    assert q["sam_deg"] < 1 and q["cc"] > 0.99


def test_embeddings_foundation_registry():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 8))
    idx, sc = ic.embedding_search(emb, emb[[3]], k=3)
    assert idx[0, 0] == 3 and sc[0, 0] == pytest.approx(1.0)
    idx2, d = ic.embedding_search(emb, emb[[7]], k=2, metric="euclidean")
    assert idx2[0, 0] == 7 and d[0, 0] == pytest.approx(0)
    protos = np.repeat(np.eye(3) * 5, 4, axis=0) + rng.normal(0, 0.1, (12, 3))
    labels = np.repeat(["a", "b", "c"], 4)
    pred, _ = ic.few_shot_prototypes(protos, labels, np.eye(3))
    assert pred.tolist() == ["a", "b", "c"]
    info = ic.foundation_model_info()
    assert {"prithvi-eo-2.0", "clay-v1.5", "satlaspretrain", "dofa", "skysense"} <= set(info.index)
    cfg = ic.terratorch_config("segmentation", num_classes=3)
    assert cfg["model"]["init_args"]["model_args"]["num_classes"] == 3
    with pytest.raises(KeyError):
        ic.load_foundation_encoder("nope")
    split = ic.geographic_holdout(rng.uniform(0, 100, (100, 2)), (0, 0, 50, 50))
    ov = ic.check_geographic_overlap(rng.uniform(0, 10, (20, 2)), rng.uniform(90, 100, (5, 2)),
                                     min_distance=10)
    assert len(split.test) > 0 and ov["fraction_within"] == 0


def test_save_load_run(tmp_path, scene):
    da, _ = scene
    n = ic.Normalizer().fit(da)
    s = ic.Split(train=np.arange(5), test=np.arange(5, 8))
    p = ic.save_run(tmp_path / "run.json", config=ic.RunConfig(), normalizer=n,
                    class_names={0: "water"}, raster_meta=ic.RasterMeta.from_dataarray(da), split=s,
                    metrics={"oa": np.float64(0.9), "table": pd.DataFrame({"a": [1]})})
    rec = ic.load_run(p)
    assert rec["normalizer"].center.shape == (4,) and rec["split"]["test"] == [5, 6, 7]
    assert "numpy" in rec["versions"] and rec["raster_meta"]["crs"]


def test_cli_end_to_end(tmp_path, scene, capsys):
    da, cls = scene
    da.rio.to_raster(tmp_path / "img.tif")
    pts = _points(da, cls, 300)
    pd.DataFrame({"x": pts.geometry.x, "y": pts.geometry.y, "cls": pts.cls}).to_csv(
        tmp_path / "pts.csv", index=False)
    out = tmp_path / "out"
    res = ic.main(["--image", str(tmp_path / "img.tif"), "--labels", str(tmp_path / "pts.csv"),
                   "--label-col", "cls", "--block-size", "150", "--out", str(out)])
    assert res["report"]["overall_accuracy"] > 0.9
    for f in ("classification.tif", "run.json", "per_class.csv", "confusion_matrix.csv", "cv.csv"):
        assert (out / f).exists()
    ic.main(["--image", str(tmp_path / "img.tif"), "--learning-mode", "unsupervised",
             "--n-classes", "3", "--out", str(tmp_path / "u")])
    assert (tmp_path / "u" / "clusters.tif").exists()
    ic.main(["--image", str(tmp_path / "img.tif"), "--labels", str(tmp_path / "pts.csv"),
             "--label-col", "cls", "--spatial-unit", "object", "--n-segments", "100",
             "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "object_classes.tif").exists()
    r = ic.main(["--image", str(tmp_path / "img.tif"), "--labels", str(tmp_path / "pts.csv"),
                 "--label-col", "cls", "--learning-mode", "semi-supervised", "--block-size", "150"])
    assert r["report"]["overall_accuracy"] > 0.9
    v2 = da.values.copy()
    v2[:, 10:20, 10:20] += 0.2
    da2 = da.copy(data=v2)
    da2.rio.to_raster(tmp_path / "img2.tif")
    r = ic.main(["--task", "change-detection", "--image", str(tmp_path / "img.tif"),
                 "--image2", str(tmp_path / "img2.tif"), "--out", str(tmp_path / "c")])
    assert (tmp_path / "c" / "change.tif").exists()
    assert ic.change_metrics(r["change"]["change"], np.pad(np.ones((10, 10), bool),
                                                           ((10, 28), (10, 40))))["f1"] > 0.9
    flat = ic.change_vector_analysis(da.fillna(0), da.fillna(0) + 0.1)
    assert int(flat["change"].sum()) == 0
    ic.main([])
    assert "nothing to run" in capsys.readouterr().out


def test_lazy_submodule_registered():
    import geolattice.rsense as rs

    assert "image_classification" in dir(rs)
    assert rs.image_classification.train_model is ic.train_model


# ============================================================================ deep learning
try:
    import torch
except ImportError:  # deep-learning tests are skipped without PyTorch
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs PyTorch (geolattice[rsense-dl])")


def _chips(n=48, size=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    x = rng.normal(0, 0.3, (n, 4, size, size)).astype("float32")
    x[np.arange(n), y] += 2.0  # class k bright in band k
    return x, y


@needs_torch
def test_chip_dataset_cnns_and_training(tmp_path):
    x, y = _chips()
    norm = ic.Normalizer().fit(x[:32], band_axis=1)
    aug = ic.OverheadAugment(brightness=0.05, contrast=0.1, noise=0.01, crop=0.8)
    train = ic.ChipDataset(x[:32], y[:32], normalizer=norm, transform=aug)
    val = ic.ChipDataset(x[32:], y[32:], normalizer=norm)
    assert train[0]["image"].shape == (4, 16, 16) and "label" in train[0]
    for arch in ("lenet", "vgg_small", "simple_cnn", "resnet18"):
        m = ic.build_cnn(arch, in_channels=4, num_classes=3)
        assert m(torch.zeros(2, 4, 16, 16)).shape == (2, 3)
    model = ic.build_cnn("simple_cnn", in_channels=4, num_classes=3)
    rf = ic.receptive_field(model, input_size=16)
    assert rf["rf"] > 5 and rf["jump"] == 4
    ck = tmp_path / "best.pt"
    hist = ic.train_model(model, ic.as_loader(train, 8), ic.as_loader(val, 8, shuffle=False),
                          epochs=6, lr=3e-3, loss="focal", early_stopping=4, checkpoint=ck,
                          cutmix_prob=0.3, scheduler="onecycle", verbose=False,
                          logger=lambda e, m: None)
    assert max(hist["val_score"]) > 0.8 and ck.exists()
    m2, _ = ic.load_checkpoint(ic.build_cnn("simple_cnn", in_channels=4, num_classes=3), ck)
    p = ic.predict_proba_batches(m2, norm.transform(x[32:], 1), tta=True)
    assert p.shape == (16, 3) and np.allclose(p.sum(1), 1, atol=1e-5)


@needs_torch
@pytest.mark.parametrize("name", ["ce", "weighted_ce", "focal", "dice", "lovasz", "ce+dice",
                                  "gce", "sce"])
def test_losses_dense_and_flat(name):
    w = ic.class_weights(np.array([0, 0, 0, 1, 2]), num_classes=3, method="effective")
    loss = ic.get_loss(name, class_weights=w)
    lg = torch.randn(2, 3, 8, 8, requires_grad=True)
    t = torch.randint(0, 3, (2, 8, 8))
    t[0, 0, 0] = -100
    v = loss(lg, t)
    v.backward()
    assert torch.isfinite(v) and lg.grad is not None
    assert torch.isfinite(loss(torch.randn(5, 3), torch.randint(0, 3, (5,))))


@needs_torch
def test_adapt_input_channels_and_transfer_modes():
    m = ic.build_cnn("resnet18", in_channels=6, num_classes=4, first_conv="random")
    assert m.conv1.in_channels == 6 and m.fc.out_features == 4
    c = ic.set_trainable(m, "linear_probe")
    assert c["trainable"] == 512 * 4 + 4
    c2 = ic.set_trainable(m, "partial", n_blocks=1)
    assert c["trainable"] < c2["trainable"] < c2["total"]
    assert ic.set_trainable(m, "full")["trainable"] == c2["total"]
    enc = ic.make_encoder("resnet18", in_channels=4)
    assert enc(torch.zeros(2, 4, 32, 32)).shape == (2, 512)
    x, y = _chips(32)
    ds = ic.ChipDataset(x, y)
    tab = ic.compare_transfer_modes(lambda: ic.build_cnn("simple_cnn", in_channels=4,
                                                         num_classes=3),
                                    ic.as_loader(ds, 8), ic.as_loader(ds, 8, shuffle=False),
                                    epochs=1, modes=("linear_probe", "full"))
    assert list(tab["mode"]) == ["linear_probe", "full"] and tab["n_train"].iloc[0] == 32


@needs_torch
def test_segmentation_models_and_sliding_window(scene, tmp_path):
    da, cls = scene
    img = da.fillna(0)
    for arch in ("unet_basic", "fcn_resnet50", "lraspp_mobilenet"):
        m = ic.build_segmenter(arch, in_channels=4, num_classes=3)
        assert m.eval()(torch.zeros(1, 4, 32, 32)).shape == (1, 3, 32, 32)
    m = ic.build_segmenter("unet_basic", in_channels=4, num_classes=3, base=8, depth=2)
    arr = np.asarray(img.values)
    chips = np.stack([arr[:, r:r + 16, c:c + 16] for r in range(0, 32, 8) for c in range(0, 44, 8)])
    masks = np.stack([cls[r:r + 16, c:c + 16] for r in range(0, 32, 8) for c in range(0, 44, 8)])
    norm = ic.Normalizer().fit(chips, band_axis=1)
    ds = ic.ChipDataset(chips, masks, normalizer=norm, transform=ic.OverheadAugment())
    assert ds[0]["mask"].shape == (16, 16)
    hist = ic.train_model(m, ic.as_loader(ds, 8), ic.as_loader(ds, 8, shuffle=False),
                          task="segmentation", epochs=8, lr=1e-2, loss="ce+dice", verbose=False)
    assert max(hist["val_score"]) > 0.6
    pred = ic.sliding_window_predict(m, img, tile=16, overlap=4, normalizer=norm)
    assert pred.shape == cls.shape and pred.rio.crs.to_epsg() == 32618
    assert ic.segmentation_metrics(pred, cls, num_classes=3)["pixel_accuracy"] > 0.7
    ic.write_class_map(pred, tmp_path / "seg.tif")
    cm = ic.predict_chip_map(ic.build_cnn("simple_cnn", in_channels=4, num_classes=3), img,
                             size=8, stride=4)
    assert cm.shape == cls.shape


@needs_torch
def test_change_models_temporal_fusion_ssl():
    t1, t2 = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    for arch in ("siamese_diff", "siamese_concat", "early_fusion", "bit"):
        m = ic.build_change_model(arch, in_channels=3, num_classes=2)
        assert m(t1, t2).shape == (2, 2, 32, 32)
    before = np.random.default_rng(0).normal(size=(8, 3, 16, 16)).astype("float32")
    after = before.copy()
    after[:, :, 4:12, 4:12] += 3
    mask = np.zeros((8, 16, 16), int)
    mask[:, 4:12, 4:12] = 1
    ds = ic.PairChipDataset(before, after, mask, transform=ic.OverheadAugment())
    m = ic.build_change_model("siamese_diff", in_channels=3, widths=(8, 16))
    h = ic.train_model(m, ic.as_loader(ds, 4), ic.as_loader(ds, 4, shuffle=False),
                       task="segmentation", epochs=6, lr=1e-2, verbose=False)
    assert max(h["val_score"]) > 0.5
    x = torch.randn(4, 6, 5)
    for arch, inp in (("tempcnn", x), ("lstm", x), ("ltae", x),
                      ("psetae", torch.randn(4, 6, 5, 10)),
                      ("convlstm", torch.randn(2, 6, 5, 16, 16)),
                      ("utae", torch.randn(2, 6, 5, 16, 16))):
        m = ic.build_temporal_model(arch, in_channels=5, num_classes=3, n_timesteps=6)
        out = m(inp)
        assert out.shape[:2] == (inp.shape[0], 3)
    fm = ic.build_fusion_model(4, 2, 3)
    assert fm(torch.randn(2, 4, 16, 16), torch.randn(2, 2, 16, 16)).shape == (2, 3)
    z = torch.randn(8, 16)
    assert ic.nt_xent_loss(z, z) < ic.nt_xent_loss(z, torch.randn(8, 16))
    chips = np.random.default_rng(0).normal(size=(16, 3, 16, 16)).astype("float32")
    loader = ic.as_loader(ic.ChipDataset(chips), 8)
    for method in ("simclr", "moco"):
        enc = ic.build_cnn("simple_cnn", in_channels=3, num_classes=0)
        enc, hist = ic.pretrain_contrastive(enc, loader, in_channels=3, method=method, epochs=1,
                                            queue_size=64, verbose=False)
        assert np.isfinite(hist["loss"][0])
    mae = ic.build_mae(3, width=8, patch=4)
    mae, hist = ic.pretrain_masked(mae, loader, epochs=2, verbose=False)
    emb = ic.extract_embeddings(mae, chips)
    assert emb.shape == (16, 32)
    lp = ic.linear_probe(emb[:12], np.arange(12) % 2, emb[12:], np.arange(4) % 2)
    assert "report" in lp


@needs_torch
def test_detector_fixmatch_tent():
    det = ic.build_object_detector("fcos", num_classes=3, pretrained=False, in_channels=4)
    det.eval()
    out = det([torch.rand(4, 64, 64)])
    assert {"boxes", "scores", "labels"} <= set(out[0])
    x, y = _chips(32)
    lab = ic.ChipDataset(x[:16], y[:16])
    unl = ic.ChipDataset(x[16:])
    model = ic.build_cnn("simple_cnn", in_channels=4, num_classes=3)
    h = ic.train_fixmatch(model, ic.as_loader(lab, 8), ic.as_loader(unl, 8),
                          ic.as_loader(lab, 8, shuffle=False), epochs=2, threshold=0.5,
                          verbose=False)
    assert len(h["loss"]) == 2 and 0 <= h["mask_rate"][-1] <= 1
    model, ent = ic.tent_adapt(model, ic.as_loader(unl, 8), steps=2)
    assert len(ent) == 4


@needs_torch
def test_vit_attention_rollout():
    timm = pytest.importorskip("timm")  # noqa: F841
    m = ic.build_cnn("vit_tiny_patch16_224", in_channels=4, num_classes=3, img_size=64)
    att = ic.vit_attention_rollout(m, torch.randn(2, 4, 64, 64))
    assert att.shape == (2, 4, 4) and np.isclose(att.max(), 1)
    sw = ic.build_cnn("swin_t", in_channels=4, num_classes=3)
    assert sw(torch.zeros(1, 4, 64, 64)).shape == (1, 3)
