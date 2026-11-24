import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from geolattice.rsense import classify


@pytest.fixture
def scene(reflectance):
    """Two-class scene: left half 'water' (dark NIR), right half 'veg' (bright NIR)."""
    ds = reflectance.isel(time=0)[["red", "nir", "swir16"]].copy(deep=True)
    ds["nir"][:, :25] = 0.02
    ds["nir"][:, 25:] = 0.45
    labels = np.where(np.arange(50) < 25, 0, 1)
    return ds, labels


def _points(ds, n=80, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.choice(ds.x.values, n)
    ys = rng.choice(ds.y.values, n)
    cls = np.where(xs < ds.x.values[25], "water", "veg")
    return gpd.GeoDataFrame({"cls": cls}, geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=32618)


@pytest.mark.parametrize("model", ["rf", "knn", "logistic", "gbm"])
def test_classifier_points(scene, model):
    ds, _ = scene
    X, y = classify.extract_samples(ds, _points(ds), "cls")
    assert X.attrs["coords"].shape == (len(X), 2)
    clf = classify.Classifier(model).fit(X, y)
    m = clf.predict_raster(ds)
    w, v = m.attrs["classes"].index("water"), m.attrs["classes"].index("veg")
    assert (m.values[:, :20] == w).mean() > 0.95 and (m.values[:, 30:] == v).mean() > 0.95
    proba = clf.predict_raster(ds, proba=True)
    assert set(proba.data_vars) == {"p_veg", "p_water"}


def test_classifier_polygons_and_importance(scene):
    ds, _ = scene
    x, y = ds.x.values, ds.y.values
    polys = gpd.GeoDataFrame({"cls": [0, 1]}, geometry=[
        box(x[2], y[30], x[10], y[5]), box(x[35], y[30], x[45], y[5])], crs=32618)
    X, lab = classify.extract_samples(ds, polys, "cls", max_per_polygon=50)
    assert len(X) == 100 and "groups" in X.attrs
    clf = classify.Classifier("rf", n_estimators=50, random_state=0).fit(X, lab)
    assert clf.feature_importance().index[0] == "nir"
    assert clf.predict_raster(ds).dtype.kind == "i"


def test_cluster(scene):
    ds, truth = scene
    lab = classify.cluster(ds[["nir"]], 2)
    agree = (lab.values == lab.values[0, 0]) == (truth[None, :] == truth[0])
    assert agree.mean() > 0.99


def test_accuracy_and_olofsson():
    r = classify.accuracy_report([0, 0, 1, 1, 1], [0, 1, 1, 1, 1])
    assert r["overall_accuracy"] == pytest.approx(0.8)
    assert r["per_class"].loc[0, "users_accuracy"] == 1.0
    # Olofsson et al. (2014) worked example
    cm = pd.DataFrame([[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]],
                      index=list("ABCD"), columns=list("ABCD"))
    o = classify.olofsson(cm, dict(A=200000, B=150000, C=3200000, D=6450000), pixel_area=0.09)
    t = o["table"]
    assert t.loc["A", "adjusted_area"] == pytest.approx(21158, rel=1e-3)
    assert t.loc["B", "adjusted_area"] == pytest.approx(11686, rel=1e-3)
    assert t.loc["C", "adjusted_area"] == pytest.approx(285770, rel=1e-3)
    assert t.loc["D", "adjusted_area"] == pytest.approx(581386, rel=1e-3)
    assert o["overall_accuracy"] == pytest.approx(0.947, abs=1e-3)
    assert t.loc["A", "producers_accuracy"] == pytest.approx(0.75, abs=0.01)


def test_obia(scene):
    ds, _ = scene
    seg = classify.segment(ds, method="slic", n_segments=40)
    assert seg.max() > 5
    feats = classify.segment_features(ds, seg)
    assert {"nir_mean", "area_px", "compactness"} <= set(feats.columns)
    back = classify.segments_to_raster(feats["nir_mean"], seg)
    assert back.shape == seg.shape
    fz = classify.segment(ds, method="felzenszwalb", scale=50)
    ws = classify.segment(ds, method="watershed", n_segments=40)
    assert fz.max() > 1 and ws.max() > 1
    pts = _points(ds)
    lab = classify.segment_labels(seg, pts, "cls")
    assert set(lab.unique()) <= {"water", "veg"}


def test_change_tools(scene):
    ds, _ = scene
    after = ds.copy(deep=True)
    after["nir"][:20, :] = 0.02
    c = classify.cva(ds, after, bands=["red", "nir"])
    assert float(c.change[:20, 30:].mean()) > 0.9 and float(c.change[25:, :].mean()) < 0.1
    d = classify.image_difference(ds.nir, after.nir, k=1.0)
    assert float(d.decrease[:20, 30:].mean()) == 1.0
    m1 = xr.DataArray(np.zeros((4, 4), int), dims=("y", "x"))
    m2 = m1.copy(data=np.r_[np.zeros(8), np.ones(8)].reshape(4, 4).astype(int))
    codes, mat = classify.post_classification(m1, m2, labels={0: "forest", 1: "crop"})
    assert mat.loc["forest", "crop"] == 8 and int(codes.max()) == 1
