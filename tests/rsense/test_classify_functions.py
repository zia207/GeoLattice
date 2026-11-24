"""Direct tests for every public function in geolattice.rsense.classify not covered elsewhere.

Functions that download external models (TorchGeo weights, SAM, RemoteCLIP) are tested with
small stand-in modules injected through ``sys.modules``, so their own logic is exercised
without network access.
"""

import subprocess
import sys
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point

from geolattice.rsense import classify
from geolattice.rsense.classify import image_classification as ic

try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs PyTorch (geolattice[rsense-dl])")


def _image(ny=40, nx=50, bands=("b1", "b2", "b3"), seed=0):
    rng = np.random.default_rng(seed)
    y = 4_700_000 - 10.0 * (np.arange(ny) + 0.5)
    x = 400_000 + 10.0 * (np.arange(nx) + 0.5)
    da = xr.DataArray(rng.random((len(bands), ny, nx)).astype("float32"), dims=("band", "y", "x"),
                      coords={"band": list(bands), "y": y, "x": x})
    return da.rio.write_crs(32618)


# ------------------------------------------------------------------------- package layout
def test_module_location_and_alias():
    import importlib

    import geolattice.rsense as rs

    assert classify.image_classification is ic
    assert ic.__name__ == "geolattice.rsense.classify.image_classification"
    alias = importlib.import_module("geolattice.rsense.image_classification")
    assert alias is ic and rs.image_classification is ic
    assert {"Classifier", "image_classification", "olofsson"} <= set(classify.__all__)


@pytest.mark.parametrize("mod", ["geolattice.rsense.classify.image_classification",
                                 "geolattice.rsense.image_classification"])
def test_python_dash_m_entry_points(mod):
    r = subprocess.run([sys.executable, "-m", mod], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "nothing to run" in r.stdout
    h = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True, text=True,
                       timeout=300)
    assert "--learning-mode" in h.stdout


# ------------------------------------------------------------------------------ estimators
@pytest.mark.parametrize("model", ["rf", "knn", "logistic", "gbm", "mlp", "svm"])
def test_classifier_predict_proba_and_raster(model):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": np.r_[rng.normal(0, 0.1, 50), rng.normal(1, 0.1, 50)],
                      "b": rng.normal(0, 0.1, 100)})
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    clf = classify.Classifier(model).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert (clf.predict(X) == y).mean() > 0.95
    ds = xr.Dataset({"a": (("y", "x"), np.r_[np.zeros((5, 4)), np.ones((5, 4))]),
                     "b": (("y", "x"), np.zeros((10, 4)))})
    ds["a"].values[0, 0] = np.nan
    pr = clf.predict_raster(ds, proba=True)
    assert set(pr.data_vars) == {"p_0", "p_1"} and np.isnan(pr["p_0"].values[0, 0])
    cls = clf.predict_raster(ds, chunk_rows=3)
    assert cls.values[0, 0] == -1 and cls.values[9, 0] == 1
    assert "Classifier(" in repr(clf)


def test_classifier_regression_kmeans_and_string_labels():
    rng = np.random.default_rng(1)
    X = rng.random((80, 2)).astype("float32")
    reg = classify.Classifier("rf", task="regression", n_estimators=20).fit(X, X[:, 0] * 2)
    assert np.corrcoef(reg.predict(X), X[:, 0])[0, 1] > 0.9
    km = classify.Classifier("kmeans", n_clusters=3, random_state=0).fit(X)
    assert set(np.unique(km.model.labels_)) == {0, 1, 2}
    ys = np.where(X[:, 0] > 0.5, "veg", "water")
    s = classify.Classifier("rf", n_estimators=20).fit(pd.DataFrame(X, columns=["a", "b"]), ys)
    ds = xr.Dataset({"a": (("y", "x"), np.array([[0.9, 0.1]])), "b": (("y", "x"), [[0.5, 0.5]])})
    m = s.predict_raster(ds)
    assert m.attrs["classes"] == ["veg", "water"] and m.values.tolist() == [[0, 1]]


# ------------------------------------------------------------------- image_classification
def test_band_conversions_labels_table_and_blocks(tmp_path):
    da = _image()
    ds = ic.as_band_dataset(da)
    back = ic.as_band_array(ds)
    assert list(ds.data_vars) == ["b1", "b2", "b3"] and back.shape == da.shape
    assert ic.as_band_array(da.isel(band=0, drop=True).rename("ndvi")).sizes["band"] == 1
    pd.DataFrame({"x": [400_005.0], "y": [4_699_995.0], "c": ["a"]}).to_csv(tmp_path / "p.csv",
                                                                             index=False)
    g = ic.labels_from_table(tmp_path / "p.csv", label="c", crs=32618)
    assert g.crs.to_epsg() == 32618 and g.geometry.iloc[0].x == 400_005.0
    with pytest.raises(KeyError):
        ic.labels_from_table(tmp_path / "p.csv", label="missing")
    ids = ic.spatial_blocks(np.array([[0, 0], [5, 5], [15, 0], [0, 25]]), 10)
    assert ids[0] == ids[1] and len(set(ids)) == 3


def test_normalizer_fit_transform_and_spatial_cv():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, (200, 3))
    z = ic.Normalizer("minmax").fit_transform(X)
    assert z.min() == pytest.approx(0) and z.max() == pytest.approx(1)
    coords = rng.uniform(0, 1000, (200, 2))
    y = (coords[:, 0] > 500).astype(int)
    from sklearn.neighbors import KNeighborsClassifier

    cv = ic.spatial_cross_validate(lambda: KNeighborsClassifier(3), coords, y, coords,
                                   block_size=250, n_splits=4)
    assert len(cv) == 4 and {"accuracy", "macro_f1", "kappa"} <= set(cv.columns)


def test_set_seed_reproducible():
    ic.set_seed(7)
    a = np.random.rand(3)
    ic.set_seed(7)
    assert np.array_equal(a, np.random.rand(3))


def test_extract_chips_and_masks(tmp_path):
    da = _image()
    pts = gpd.GeoDataFrame({"c": [1, 2, 3]},
                           geometry=[Point(da.x.values[20], da.y.values[20]),
                                     Point(da.x.values[0], da.y.values[0]),       # edge: dropped
                                     Point(da.x.values[30], da.y.values[10])], crs=32618)
    chips, lab, xy = ic.extract_chips(da, pts, "c", size=8)
    assert chips.shape == (2, 3, 8, 8) and lab.tolist() == [1, 3] and xy.shape == (2, 2)
    np.testing.assert_allclose(chips[0][:, 4, 4], da.values[:, 20, 20])
    one = np.array([[da.x.values[20], da.y.values[20]]])
    arr_chips, none, _ = ic.extract_chips(da, one, size=6)
    assert arr_chips.shape == (1, 3, 6, 6) and none is None
    colors = {0: (0, 255, 255), 1: (255, 255, 0), 2: (0, 0, 255)}
    rgb = np.array([[[0, 250, 250], [252, 251, 3]], [[1, 2, 254], [10, 10, 10]]], "uint8")
    assert ic.rgb_mask_to_index(rgb, colors).tolist() == [[0, 1], [2, -100]]
    (tmp_path / "cd.csv").write_text("name,r,g,b\nurban,0,255,255\nagri,255,255,0\n")
    names, cols = ic.read_class_dict(tmp_path / "cd.csv")
    assert names == ["urban", "agri"] and cols[1] == (255, 255, 0)


def test_chips_from_folder(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    for k in range(2):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype="uint8")).save(
            tmp_path / f"{k}_sat.jpg")
        m = np.zeros((64, 64, 3), "uint8")
        m[:, :32] = (255, 255, 0)
        m[:, 32:] = (0, 255, 255)
        Image.fromarray(m).save(tmp_path / f"{k}_mask.png")
    X, Y, src = ic.chips_from_folder(tmp_path, colors={0: (0, 255, 255), 1: (255, 255, 0)},
                                     size=16, stride=16, downsample=2)
    assert X.shape == (8, 3, 16, 16) and Y.shape == (8, 16, 16) and len(src) == 8
    assert 0 <= X.min() and X.max() <= 1 and set(np.unique(Y)) == {0, 1}
    X1, _, _ = ic.chips_from_folder(tmp_path, size=32, max_images=1)
    assert len(X1) == 4


def test_pixel_time_series():
    da = _image(bands=("red", "nir"))
    cube = xr.concat([da, da * 2, da * 3], dim="time").assign_coords(
        time=pd.date_range("2026-05-01", periods=3, freq="MS")).rio.write_crs(32618)
    pts = gpd.GeoDataFrame({"c": [1, 2]}, geometry=[Point(da.x.values[3], da.y.values[4]),
                                                    Point(da.x.values[10], da.y.values[2])],
                           crs=32618)
    X, lab, dates = ic.pixel_time_series(cube, pts, "c")
    assert X.shape == (2, 3, 2) and lab.tolist() == [1, 2] and len(dates) == 3
    np.testing.assert_allclose(X[0, 2], 3 * da.values[:, 4, 3], rtol=1e-6)
    Xd, _, _ = ic.pixel_time_series(ic.as_band_dataset(da).expand_dims(
        time=pd.date_range("2026-05-01", periods=1)).rio.write_crs(32618), pts)
    assert Xd.shape == (2, 1, 2)


def test_uncertainty_scores():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    for s in ("entropy", "margin", "least_confidence"):
        u = ic.uncertainty_scores(p, s)
        assert u[1] > u[0]
    with pytest.raises(ValueError):
        ic.uncertainty_scores(p, "nope")


def test_parse_args_and_run_workflow(tmp_path):
    cfg = ic.parse_args(["--task", "classification", "--learning-mode", "unsupervised",
                         "--spatial-unit", "pixel", "--bands", "b1", "b2", "b3", "--seed", "3",
                         "--image", "x.tif", "--n-classes", "4"])
    assert cfg.bands == ("b1", "b2", "b3") and cfg.seed == 3 and cfg.n_classes == 4
    da = _image()
    da.rio.to_raster(tmp_path / "img.tif")
    cfg = ic.RunConfig(learning_mode="unsupervised", image=str(tmp_path / "img.tif"), n_classes=3,
                       model="gmm", out=str(tmp_path / "o"))
    res = ic.run_workflow(cfg)
    assert res["map"].shape == (40, 50) and (tmp_path / "o" / "clusters.tif").exists()
    with pytest.raises(ValueError):
        ic.run_workflow(ic.RunConfig())
    with pytest.raises(NotImplementedError):
        ic.run_workflow(ic.RunConfig(task="detection", image=str(tmp_path / "img.tif")))
    with pytest.raises(ValueError, match="labels"):
        ic.run_workflow(ic.RunConfig(image=str(tmp_path / "img.tif")))


# ----------------------------------------------- external-model wrappers with stand-in modules
def test_sam_and_text_prompt_segment_with_stub(monkeypatch, tmp_path):
    calls = {}

    class SamGeo:
        def __init__(self, model_type=None, automatic=False, **kw):
            calls["init"] = (model_type, automatic)

        def set_image(self, path):
            calls["image"] = path

        def predict(self, **kw):
            calls["predict"] = kw

        def generate(self, src, out):
            calls["generate"] = (src, out)

    samgeo = types.ModuleType("samgeo")
    samgeo.SamGeo = SamGeo
    samgeo.common = types.SimpleNamespace(raster_to_vector=lambda a, b: calls.setdefault("vec", b))
    text_sam = types.ModuleType("samgeo.text_sam")

    class LangSAM:
        def predict(self, img, text, **kw):
            calls["text"] = (text, kw["box_threshold"])

        def raster_to_vector(self, a, b):
            calls["tvec"] = b

    text_sam.LangSAM = LangSAM
    monkeypatch.setitem(sys.modules, "samgeo", samgeo)
    monkeypatch.setitem(sys.modules, "samgeo.text_sam", text_sam)
    r = ic.sam_segment("img.tif", tmp_path / "m.tif", points=[[1, 2]], point_labels=[1],
                       vectorize=tmp_path / "m.gpkg")
    assert r["verified"] is False and calls["predict"]["point_coords"] == [[1, 2]]
    assert calls["init"] == ("vit_h", False) and "vec" in calls
    ic.sam_segment("img.tif", tmp_path / "b.tif", boxes=[(0, 0, 5, 5)])
    assert calls["predict"]["boxes"] == [[0, 0, 5, 5]]
    ic.sam_segment("img.tif", tmp_path / "a.tif", automatic=True)
    assert calls["generate"][0] == "img.tif"
    t = ic.text_prompt_segment("img.tif", "building", tmp_path / "t.tif",
                               vectorize=tmp_path / "t.gpkg")
    assert calls["text"] == ("building", 0.24) and t["prompt"] == "building" and "tvec" in calls


@needs_torch
def test_load_rs_backbone_with_stub(monkeypatch):
    nn = torch.nn

    class W:
        meta = {"in_chans": 13, "model": "resnet18"}

    models = types.ModuleType("torchgeo.models")
    models.get_weight = lambda enum: W()
    models.get_model = lambda arch, weights=None: nn.Sequential(nn.Conv2d(13, 4, 3),
                                                                nn.AdaptiveAvgPool2d(1),
                                                                nn.Flatten(), nn.Linear(4, 9))
    monkeypatch.setitem(sys.modules, "torchgeo", types.ModuleType("torchgeo"))
    monkeypatch.setitem(sys.modules, "torchgeo.models", models)
    m = ic.load_rs_backbone("ssl4eo_s2_resnet18_moco", num_classes=5)
    assert m.rs_weights_meta["in_chans"] == 13 and m[-1].out_features == 5
    assert ic.load_rs_backbone("torchgeo:Any.ENUM")[-1].out_features == 9
    with pytest.raises(KeyError):
        ic.load_rs_backbone("unknown")
    assert ic.load_foundation_encoder("ssl4eo_s2_resnet18_moco")[-1].out_features == 9


@needs_torch
def test_load_rs_clip_and_zero_shot_with_stub(monkeypatch, tmp_path):
    nn = torch.nn

    class Clip(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def encode_image(self, x):  # mean color -> 3-d embedding
            return x.mean((-2, -1))

        def encode_text(self, tokens):
            return torch.eye(3)[tokens]

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = lambda arch: (Clip(), None, lambda im: torch.from_numpy(
        np.asarray(im, "float32").transpose(2, 0, 1) / 255))
    open_clip.get_tokenizer = lambda arch: (lambda texts: torch.tensor(
        [["red", "green", "blue"].index(t.split()[-1].rstrip(".")) for t in texts]))
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": {"w": torch.ones(1)}}, ck)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, fname: str(ck)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    model, pre, tok = ic.load_rs_clip("remoteclip-vit-b-32", device="cpu")
    assert float(model.w) == 1.0
    imgs = [np.full((4, 4, 3), c, "uint8") for c in ([255, 0, 0], [0, 0, 255])]
    df = ic.zero_shot_classify(imgs, ["red", "green", "blue"], device="cpu")
    assert df["prediction"].tolist() == ["red", "blue"] and not df["verified"].any()


@needs_torch
def test_detect_objects_and_instances_from_detector():
    da = _image(ny=64, nx=64)
    det = ic.build_object_detector("fcos", num_classes=3, pretrained=False, in_channels=3,
                                   min_size=64, max_size=64)
    gdf = ic.detect_objects(det, da, tile=64, overlap=0, score_threshold=0.0,
                            class_names={1: "tree"}, normalizer=ic.Normalizer().fit(da))
    assert gdf.crs.to_epsg() == 32618 and {"label", "score"} <= set(gdf.columns)
    masks = np.zeros((2, 1, 64, 64), "float32")
    masks[0, 0, 5:15, 5:20] = 0.9
    masks[1, 0, 30:40, 30:40] = 0.9
    out = {"masks": torch.from_numpy(masks), "labels": torch.tensor([1, 2]),
           "scores": torch.tensor([0.9, 0.2])}
    g = ic.instances_from_detector(out, da, score_threshold=0.5, class_names={1: "building"})
    assert len(g) == 1 and g.class_name.iloc[0] == "building"
    assert g.area.iloc[0] == pytest.approx(10 * 15 * 100)


@needs_torch
def test_torch_helpers_cutmix_fixmatch_channels_params():
    x = torch.rand(4, 3, 16, 16)
    y = torch.tensor([0, 1, 2, 0])
    mixed, ya, yb, lam = ic.cutmix(x, y, rng=np.random.default_rng(0))
    assert mixed.shape == x.shape and yb is not None and 0 <= lam <= 1
    m = torch.randint(0, 3, (4, 16, 16))
    mm, mt, none, one = ic.cutmix(x, m, rng=np.random.default_rng(1))
    assert none is None and one == 1.0 and mt.shape == m.shape
    model = ic.build_cnn("simple_cnn", in_channels=3, num_classes=3)
    loss, info = ic.fixmatch_loss(model, x, y, x, x + 0.01, threshold=0.0)
    loss.backward()
    assert info["mask_rate"] == 1.0 and torch.isfinite(loss)
    seq = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU())
    ic.adapt_input_channels(seq, 5, method="repeat")
    assert seq[0].in_channels == 5
    ic.adapt_input_channels(seq, 7, method="random")
    assert seq[0].in_channels == 7
    with pytest.raises(ValueError):
        ic.adapt_input_channels(torch.nn.Linear(2, 2), 3)
    assert ic.count_parameters(seq) == ic.count_parameters(seq, trainable_only=False) > 0
