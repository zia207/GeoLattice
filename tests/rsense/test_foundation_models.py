"""Tests for geolattice.rsense.foundation_models.

Catalog, preprocessing, embedding utilities and the AlphaEarth helpers (with a fake Earth
Engine module) run everywhere. Real-weight tests run only for checkpoints already in the
local TorchGeo / Hugging Face caches, so the suite never downloads weights.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geolattice.rsense as rs
import geolattice.rsense.foundation_models as fm


# ---------------------------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------------------------

EXPECTED = [
    "prithvi-eo-1.0-100m", "prithvi-eo-2.0-300m", "prithvi-eo-2.0-600m", "satmae", "scale-mae",
    "ssl4eo-s12-resnet18-moco", "satlas-swint-s2-ms", "spectralgpt", "clay-v1.5", "dofa-base",
    "croma-base", "anysat", "galileo", "terramind-v1-base", "copernicus-fm", "olmoearth",
    "panopticon", "alphaearth", "presto", "tessera", "dinov3-sat-vitl16", "skysense", "sam-vit-h",
    "sam2-hiera-large", "remoteclip-vit-b-32", "georsclip-vit-b-32", "skyclip", "geochat",
    "earthgpt", "lhrs-bot", "earthdial", "prithvi-wxc", "aurora", "graphcast", "gencast",
    "fourcastnet",
]


def test_lazy_submodule():
    assert rs.foundation_models is fm
    assert "foundation_models" in dir(rs)


def test_catalog_covers_models():
    missing = [n for n in EXPECTED if n not in fm.MODELS]
    assert not missing
    assert {m.category for m in fm.MODELS.values()} == set(fm.CATEGORIES)
    assert all(m.loader in fm.LOADERS for m in fm.MODELS.values())
    for m in fm.MODELS.values():
        if m.loader in ("torchgeo", "terratorch", "open_clip", "torch.hub", "transformers",
                        "earthengine"):
            assert m.target, m.name


def test_list_models_filters():
    df = fm.list_models()
    assert len(df) == len(fm.MODELS)
    assert {"category", "loader", "loadable", "embed_dim"} <= set(df.columns)
    mm = fm.list_models("multimodal", modality="sar")
    assert len(mm) and (mm["category"] == "multimodal").all()
    assert all("sar" in s for s in mm["modalities"])
    assert fm.list_models(loadable=True)["loadable"].all()
    assert not fm.list_models("weather", loadable=True).shape[0]
    with pytest.raises(ValueError):
        fm.list_models("nope")


def test_spec_aliases_and_info():
    assert fm.get_model_spec("DOFA").name == "dofa-base"
    assert fm.get_model_spec("prithvi").name == "prithvi-eo-2.0-300m"
    info = fm.model_info("alphaearth")
    assert info["embed_dim"] == 64 and info["loader"] == "earthengine" and not info["loadable"]
    with pytest.raises(KeyError, match="unknown foundation model"):
        fm.get_model_spec("not-a-model")


def test_recommend():
    r = fm.recommend("soil")
    assert r.iloc[0]["name"] == "alphaearth"
    sar = fm.recommend("multisensor", sensor="sentinel-1")
    assert len(sar) and all("sar" in fm.MODELS[n].modalities for n in sar["name"])
    lo = fm.recommend("timeseries", loadable_only=True)
    assert all(fm.MODELS[n].loadable or fm.MODELS[n].loader == "earthengine" for n in lo["name"])
    with pytest.raises(ValueError):
        fm.recommend("astrology")


@pytest.mark.parametrize("name", ["alphaearth", "satmae", "galileo"])
def test_load_model_rejects_non_loadable(name):
    with pytest.raises((ValueError, NotImplementedError)):
        fm.load_model(name)


# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------

def test_band_wavelengths():
    assert fm.band_wavelengths(["red", "B08", "swir16"]) == [665.0, 842.0, 1610.0]
    assert fm.band_wavelengths(["blue"], unit="um") == [0.49]
    with pytest.raises(KeyError):
        fm.band_wavelengths(["ultraviolet"])


def test_normalize_methods():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 5000, (4, 3, 8, 8))
    z = fm.normalize(x)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z.mean((0, 2, 3)), 0, atol=1e-4)
    np.testing.assert_allclose(z.std((0, 2, 3)), 1, atol=1e-3)
    s = fm.normalize(x, method="scale", scale=10000, clip=(0, 1))
    assert s.max() <= 1 and s.min() >= 0
    m = fm.normalize(x, method="minmax")
    np.testing.assert_allclose(m.min((0, 2, 3)), 0, atol=1e-6)
    given = fm.normalize(x, mean=[1, 2, 3], std=[2, 2, 2])
    np.testing.assert_allclose(given[:, 1], (x[:, 1] - 2) / 2, rtol=1e-5)
    x[0, 0, 0, 0] = np.nan
    assert np.isfinite(fm.normalize(x)).all()
    with pytest.raises(ValueError):
        fm.normalize(x, method="zscore")


def _raster(nb=4, ny=50, nx=70, res=10.0, seed=0):
    rng = np.random.default_rng(seed)
    y = 4_700_000 - res * (np.arange(ny) + 0.5)
    x = 400_000 + res * (np.arange(nx) + 0.5)
    da = xr.DataArray(rng.uniform(0, 1, (nb, ny, nx)).astype("float32"),
                      dims=("band", "y", "x"),
                      coords={"band": ["blue", "green", "red", "nir"][:nb], "y": y, "x": x})
    return da.rio.write_crs("EPSG:32618")


def test_chip_raster_shapes_and_centres():
    da = _raster()
    chips, ys, xs = fm.chip_raster(da, chip_size=32)
    assert chips.shape == (len(ys) * len(xs), 4, 32, 32)
    assert len(ys) == 2 and len(xs) == 3  # 50 -> 64, 70 -> 96 after padding
    assert xs[0] == pytest.approx(float(da.x[0]) + 10 * 15.5)
    assert ys[0] == pytest.approx(float(da.y[0]) - 10 * 15.5)
    np.testing.assert_array_equal(chips[0], da.values[:, :32, :32])
    chips2, ys2, xs2 = fm.chip_raster(da, chip_size=32, stride=16)
    assert len(xs2) == 4 and len(ys2) == 3
    small = _raster(ny=10, nx=12)
    c, _, _ = fm.chip_raster(small, chip_size=32)
    assert c.shape == (1, 4, 32, 32)
    with pytest.raises(ValueError):
        fm.chip_raster(small, chip_size=32, pad=False)


# ---------------------------------------------------------------------------------------------
# encoder wrapper with a tiny model (no downloads)
# ---------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")

#: Devices to run model tests on: the CPU always, CUDA when PyTorch sees a GPU.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_pool_features():
    t2, t3, t4 = torch.ones(2, 5), torch.arange(24.).reshape(2, 3, 4), torch.ones(2, 6, 3, 3)
    assert fm.pool_features(t2).shape == (2, 5)
    assert torch.equal(fm.pool_features(t3, pool="cls"), t3[:, 0])
    assert torch.allclose(fm.pool_features(t3), t3.mean(1))
    assert fm.pool_features(t4).shape == (2, 6)
    assert fm.pool_features([t4, t3]).shape == (2, 4)
    assert fm.pool_features({"optical_GAP": t2, "x": t3}).shape == (2, 5)
    assert fm.pool_features(torch.ones(2, 7, 1, 3, 3)).shape == (2, 7)
    assert fm.pool_features(t3, pool="none").shape == (2, 3, 4)


class _Tiny(torch.nn.Module):
    def __init__(self, c=4, d=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, d, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _encoder(device="cpu", **kw):
    spec = fm.FoundationModel(
        name="tiny", category="optical", provider="test", year=2026, architecture="conv",
        modalities=("optical",), inputs="4 bands", temporal=False, embed_dim=8, params="1k",
        weights="none", loader="torchgeo", defaults={"img_size": 32})
    torch.manual_seed(0)
    dev = torch.device(device)
    return fm.FoundationEncoder(spec, _Tiny().eval().to(dev), device=dev, **kw)


@pytest.mark.parametrize("device", DEVICES)
def test_encoder_embed_batches(device):
    enc = _encoder(device)
    x = np.random.default_rng(1).normal(size=(5, 4, 32, 32)).astype("float32")
    e1 = enc.embed(x, batch_size=2)
    e2 = enc.embed(x, batch_size=5)
    assert e1.shape == (5, 8) and e1.dtype == np.float32
    np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-6)
    en = enc.embed(x, normalize_embeddings=True)
    np.testing.assert_allclose(np.linalg.norm(en, axis=1), 1, rtol=1e-5)
    assert enc.embed(x[0]).shape == (1, 8)
    assert "tiny" in repr(enc)


@pytest.mark.parametrize("device", DEVICES)
def test_embed_raster_grid_and_crs(device):
    enc = _encoder(device)
    da = _raster()
    emb = enc.embed_raster(da, chip_size=32, batch_size=4)
    assert emb.dims == ("embedding", "y", "x")
    assert emb.shape == (8, 2, 3)
    assert emb.rio.crs.to_epsg() == 32618
    assert enc.bands == ["blue", "green", "red", "nir"]
    assert emb.attrs["chip_size"] == 32
    # chips at another size are resized to the model input
    emb16 = enc.embed_raster(da, chip_size=16)
    assert emb16.shape == (8, 4, 5)


def test_dofa_forward_needs_wavelengths():
    enc = _encoder()

    class M:
        def forward_features(self, x, wl):
            return torch.full((x.shape[0], 3), float(sum(wl)))

    with pytest.raises(ValueError, match="wavelengths"):
        fm._fw_dofa(M(), torch.zeros(1, 4, 8, 8), enc=enc)
    out = fm._fw_dofa(M(), torch.zeros(1, 2, 8, 8), enc=enc, wavelengths=[665, 842])
    assert out[0, 0].item() == pytest.approx(0.665 + 0.842)


def test_copernicusfm_and_panopticon_forward_args():
    enc = _encoder(bands=["red", "vv"])
    seen = {}

    class P(torch.nn.Module):
        def forward(self, d):
            seen["ids"] = d["chn_ids"]
            return d["imgs"].mean((2, 3))

    fm._fw_panopticon(P(), torch.zeros(2, 2, 4, 4), enc=enc)
    assert seen["ids"].shape == (2, 2) and seen["ids"][0].tolist() == [665.0, -1.0]

    class C:
        def forward_features(self, x, meta, wl, bw):
            seen.update(meta=meta, wl=wl, bw=bw)
            return x.mean((2, 3))

    enc2 = _encoder(bands=["red", "nir"])
    fm._fw_copernicusfm(C(), torch.zeros(3, 2, 4, 4), enc=enc2)
    assert seen["wl"] == [665.0, 842.0] and seen["bw"] == [30.0, 115.0]
    assert seen["meta"].shape == (3, 4) and torch.isnan(seen["meta"]).all()


# ---------------------------------------------------------------------------------------------
# embedding analysis
# ---------------------------------------------------------------------------------------------

def _emb_map(seed=0, d=6, ny=5, nx=7):
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.normal(size=(d, ny, nx)), dims=("embedding", "y", "x"),
                        coords={"embedding": np.arange(d), "y": np.arange(ny)[::-1] * 10.0,
                                "x": np.arange(nx) * 10.0}).rio.write_crs("EPSG:32618")


def test_similarity_map():
    emb = _emb_map()
    q = emb.isel(y=2, x=3)
    sim = fm.similarity_map(emb, q)
    assert sim.dims == ("y", "x") and sim.rio.crs.to_epsg() == 32618
    assert sim.values[2, 3] == pytest.approx(1.0, abs=1e-6)
    assert sim.values.max() <= 1 + 1e-6 and sim.values.min() >= -1 - 1e-6
    d = fm.similarity_map(emb, q.values, metric="euclidean")
    assert d.values[2, 3] == pytest.approx(0.0, abs=1e-9)
    two = fm.similarity_map(emb, np.stack([q.values, q.values]))
    np.testing.assert_allclose(two.values, sim.values, rtol=1e-6)
    with pytest.raises(ValueError):
        fm.similarity_map(emb, q, metric="manhattan")


def test_embedding_change():
    a = _emb_map(0)
    assert np.allclose(fm.embedding_change(a, a).values, 0, atol=1e-6)
    ch = fm.embedding_change(a, -a)
    assert np.allclose(ch.values, 2, atol=1e-6)
    with pytest.raises(ValueError):
        fm.embedding_change(a, _emb_map(0, ny=4))


def test_cluster_and_rgb_and_frame():
    rng = np.random.default_rng(3)
    centres = np.array([[5, 0, 0, 0], [0, 5, 0, 0]])
    lab = rng.integers(0, 2, (6, 8))
    a = centres[lab].transpose(2, 0, 1) + rng.normal(0, 0.1, (4, 6, 8))
    a[:, 0, 0] = np.nan
    emb = xr.DataArray(a, dims=("embedding", "y", "x"),
                       coords={"embedding": np.arange(4), "y": np.arange(6), "x": np.arange(8)})
    cl = fm.cluster_embeddings(emb, 2)
    assert cl.dims == ("y", "x") and cl.values[0, 0] == -1
    ok = cl.values >= 0
    agree = (cl.values[ok] == lab[ok]).mean()
    assert agree in (0.0, 1.0) or max(agree, 1 - agree) == 1.0
    assert len(fm.cluster_embeddings(np.nan_to_num(a.reshape(4, -1).T), 2, method="gmm")) == 48
    rgb = fm.embedding_rgb(emb)
    assert rgb.shape == (3, 6, 8) and np.nanmax(rgb.values) <= 1 and np.nanmin(rgb.values) >= 0
    df = fm.embeddings_to_frame(emb)
    assert len(df) == 47 and list(df.columns[:3]) == ["y", "x", "e000"]


def test_linear_and_knn_probes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y_cls = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)
    y_reg = 3 * X[:, 1] - X[:, 2] + 0.1 * rng.normal(size=200)
    c = fm.linear_probe(X, y_cls, cv=4)
    assert c["task"] == "classification" and c["mean"]["accuracy"] > 0.9
    r = fm.linear_probe(X, y_reg, cv=4)
    assert r["task"] == "regression" and r["mean"]["r2"] > 0.95 and r["mean"]["rmse"] > 0
    g = fm.linear_probe(X, y_reg, cv=4, groups=np.repeat(np.arange(8), 25))
    assert g["cv"] == "group" and r["estimator"].predict(X[:3]).shape == (3,)
    k = fm.knn_probe(X[:, :2], y_cls, k=5, cv=4, metric="euclidean")
    assert k["metric"] == "accuracy" and k["mean"] > 0.8


def test_embedding_search():
    X = np.eye(5)
    idx, score = fm.embedding_search(X, X[[2]], k=2)
    assert idx[0, 0] == 2 and score[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# AlphaEarth with a fake Earth Engine module
# ---------------------------------------------------------------------------------------------

class _Obj:
    def __init__(self, calls, payload=None):
        self.calls, self.payload = calls, payload

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a, k))
            if name == "getInfo":
                return self.payload
            return self
        return f


@pytest.fixture
def fake_ee(monkeypatch):
    calls = []
    ee = types.ModuleType("ee")

    class Geometry:
        def __init__(self, geo=None):
            self.geo = geo

        @staticmethod
        def Point(xy):
            return Geometry({"type": "Point", "coordinates": xy})

        @staticmethod
        def Rectangle(b):
            x0, y0, x1, y1 = b
            return Geometry({"type": "Polygon",
                             "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})

        def bounds(self, **k):
            return _Obj(calls, self.geo if self.geo["type"] == "Polygon" else None)

    class FeatureCollection:
        def __init__(self, feats):
            self.feats = feats

    class Image(_Obj):
        def sampleRegions(self, collection, properties, scale, geometries):
            calls.append(("sampleRegions", scale))
            key = properties[0]
            feats = [{"properties": {key: f.props[key],
                                     **{b: float(i) / 100 for i, b in enumerate(fm.ALPHAEARTH_BANDS)}}}
                     for f in collection.feats]
            return _Obj(calls, {"features": feats})

    class Feature:
        def __init__(self, geom, props):
            self.geom, self.props = geom, props

    def ImageCollection(cid):
        calls.append(("ImageCollection", cid))
        return Image(calls)

    def compute_pixels(req):
        calls.append(("computePixels", req))
        g = req["grid"]["dimensions"]
        sel = [c for c in calls if c[0] == "select"][-1][1][0]
        return {b: np.full((g["height"], g["width"]), i, "float32") for i, b in enumerate(sel)}

    ee.Geometry, ee.FeatureCollection, ee.Feature = Geometry, FeatureCollection, Feature
    ee.ImageCollection = ImageCollection
    ee.Number = lambda v: _Obj(calls, v)
    ee.Initialize = lambda **k: calls.append(("Initialize", k))
    ee.data = types.SimpleNamespace(computePixels=compute_pixels)
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setattr(fm, "_EE_READY", False)
    return ee, calls


def test_alphaearth_image(fake_ee):
    ee, calls = fake_ee
    fm.alphaearth_image((90.0, 22.0, 90.1, 22.1), 2023)
    names = [c[0] for c in calls]
    assert ("ImageCollection", fm.ALPHAEARTH_COLLECTION) in calls
    assert ("filterDate", ("2023-01-01", "2024-01-01"), {}) in calls
    assert {"filterBounds", "mosaic", "select", "clip"} <= set(names)
    sel = [c for c in calls if c[0] == "select"][0]
    assert sel[1][0] == fm.ALPHAEARTH_BANDS and len(fm.ALPHAEARTH_BANDS) == 64
    fm.alphaearth_image(None, 2023, project="my-project")
    assert ("Initialize", {"project": "my-project"}) in calls


def test_sample_alphaearth(fake_ee):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    pts = gpd.GeoDataFrame({"soc": [1.0, 2.0, 3.0]},
                           geometry=[Point(400_000 + 10 * i, 4_700_000) for i in range(3)],
                           crs="EPSG:32618")
    out = fm.sample_alphaearth(pts, 2024, chunk=2, source="earthengine")
    _, calls = fake_ee
    assert sum(c[0] == "sampleRegions" for c in calls) == 2
    assert list(out["soc"]) == [1.0, 2.0, 3.0]
    assert set(fm.ALPHAEARTH_BANDS) <= set(out.columns) and "_pt" not in out.columns
    assert out["A01"].iloc[0] == pytest.approx(0.01)


def test_alphaearth_to_xarray(fake_ee):
    da = fm.alphaearth_to_xarray((90.0, 22.0, 90.001, 22.0005), 2024, scale=10,
                                 bands=["A00", "A05"], source="earthengine")
    assert da.dims == ("embedding", "y", "x") and list(da.embedding.values) == ["A00", "A05"]
    assert da.sizes["x"] == 12 and da.sizes["y"] == 6
    assert (da.sel(embedding="A05").values == 1).all()
    assert da.y.values[0] > da.y.values[-1]
    with pytest.raises(ValueError, match="too large"):
        fm.alphaearth_to_xarray((0, 0, 1, 1), 2024, scale=10, source="earthengine")


# ---------------------------------------------------------------------------------------------
# real weights, only when already cached locally (never downloads)
# ---------------------------------------------------------------------------------------------

_TORCH_HUB = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
_HF = Path.home() / ".cache" / "huggingface" / "hub"

CACHED = [
    ("dofa-base", _TORCH_HUB / "dofa_base_patch16_224-a0275954.pth", 4, {"bands": ["blue", "green", "red", "nir"]}),
    ("ssl4eo-s12-resnet18-moco", _TORCH_HUB / "resnet18_sentinel2_all_moco-59bfdff9.pth", 13, {}),
    ("satlas-swint-s2-ms", _TORCH_HUB / "sentinel2_swint_si_ms-bc68e396.pth", 9, {}),
    ("remoteclip-vit-b-32", _HF / "models--chendelong--RemoteCLIP", 3, {}),
    ("georsclip-vit-b-32", _HF / "models--Zilun--GeoRSCLIP", 3, {}),
    ("prithvi-eo-2.0-300m", _HF / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M", 6, {}),
    ("prithvi-eo-1.0-100m", _HF / "models--ibm-nasa-geospatial--Prithvi-EO-1.0-100M", 6, {}),
    ("ssl4eo-s1-resnet50-moco", _TORCH_HUB / "resnet50_sentinel1_all_moco-906e4356.pth", 2, {}),
    ("ssl4eo-s12-vits16-dino", _TORCH_HUB / "vit_small_patch16_224_sentinel2_all_dino-36bcc127.pth", 13, {}),
    ("scale-mae", _TORCH_HUB / "vit_large_patch16_224_fmow_rgb_scalemae-98ed9821.pth", 3, {}),
    ("croma-base", _TORCH_HUB / "CROMA_base-0238d814.pt", 12, {}),
    ("copernicus-fm", _TORCH_HUB / "CopernicusFM_ViT_base_varlang-085350e4.pth", 4, {"bands": ["blue", "green", "red", "nir"]}),
    ("panopticon", _TORCH_HUB / "panopticon_vitb14_teacher.pth", 4, {"bands": ["blue", "green", "red", "nir"]}),
    ("terramind-v1-base", _HF / "models--ibm-esa-geospatial--TerraMind-1.0-base", 12, {}),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,path,bands,kw", CACHED, ids=[c[0] for c in CACHED])
def test_cached_weights_embed(name, path, bands, kw, device):
    if not path.exists():
        pytest.skip(f"{name} weights not cached")
    pytest.importorskip({"open_clip": "open_clip"}.get(fm.MODELS[name].loader, "torchgeo"))
    if fm.MODELS[name].loader == "terratorch":
        pytest.importorskip("terratorch")
    enc = fm.load_model(name, device=device, **kw)
    assert next(enc.model.parameters()).device.type == device
    s = enc.img_size
    x = torch.randn(2, bands, s, s, generator=torch.Generator().manual_seed(0))
    if name.startswith("prithvi"):
        x = x[:, :, None]
    e = enc.encode(x)
    assert tuple(e.shape) == (2, fm.MODELS[name].embed_dim)
    assert torch.isfinite(e).all()
    # deterministic in eval mode
    assert torch.allclose(e, enc.encode(x), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_cached_dofa_raster_pipeline(device):
    if not CACHED[0][1].exists():
        pytest.skip("DOFA weights not cached")
    pytest.importorskip("torchgeo")
    enc = fm.load_model("dofa-base", device=device)
    da = _raster(ny=120, nx=150)
    emb = enc.embed_raster(da, chip_size=64)
    assert enc.wavelengths == [490.0, 560.0, 665.0, 842.0]
    assert emb.shape == (768, 2, 3) and np.isfinite(emb.values).all()
    sim = fm.similarity_map(emb, emb.isel(y=0, x=0))
    assert sim.values[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_segmentation_model_default_necks(monkeypatch):
    captured = {}

    class Factory:
        def build_model(self, **kw):
            captured.update(kw)
            return "model"

    mod = types.ModuleType("terratorch.models")
    mod.EncoderDecoderFactory = Factory
    monkeypatch.setitem(sys.modules, "terratorch", types.ModuleType("terratorch"))
    monkeypatch.setitem(sys.modules, "terratorch.models", mod)
    assert fm.segmentation_model(4, backbone="prithvi-eo-2.0-300m") == "model"
    assert captured["backbone"] == "prithvi_eo_v2_300" and captured["num_classes"] == 4
    assert captured["necks"][0]["indices"] == [5, 11, 17, 23]
    assert captured["backbone_bands"][0] == "BLUE"
    fm.segmentation_model(2, backbone="prithvi-eo-1.0-100m", necks=[])
    assert captured["necks"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("name,path,bands,kw", CACHED, ids=[c[0] for c in CACHED])
def test_cached_weights_gpu_matches_cpu(name, path, bands, kw):
    """The same weights give the same embeddings on the GPU as on the CPU."""
    if not path.exists():
        pytest.skip(f"{name} weights not cached")
    pytest.importorskip({"open_clip": "open_clip"}.get(fm.MODELS[name].loader, "torchgeo"))
    if fm.MODELS[name].loader == "terratorch":
        pytest.importorskip("terratorch")
    enc = fm.load_model(name, device="cpu", **kw)
    s = enc.img_size
    x = torch.randn(2, bands, s, s, generator=torch.Generator().manual_seed(1))
    if name.startswith("prithvi"):
        x = x[:, :, None]
    e_cpu = enc.encode(x).float()
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        e_gpu = enc.to("cuda").encode(x).float().cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        enc.to("cpu")
    cos = torch.nn.functional.cosine_similarity(e_cpu, e_gpu, dim=1)
    assert (cos > 0.999).all(), cos


# ---------------------------------------------------------------------------------------------
# time series chips, dense features, chip metadata, benchmark helper
# ---------------------------------------------------------------------------------------------

def _series(nt=3):
    da = _raster()
    t = pd.Index(pd.to_datetime(["2024-05-01", "2024-07-01", "2024-09-01"][:nt]), name="time")
    return xr.concat([da * (i + 1) for i in range(nt)], dim=t)


def test_chip_raster_time_series_layout():
    ts = _series()
    chips, ys, xs = fm.chip_raster(ts, chip_size=32)
    assert chips.shape == (6, 4, 3, 32, 32)  # (n, band, time, h, w)
    np.testing.assert_allclose(chips[0, :, 1], 2 * ts.values[0, :, :32, :32])
    single, _, _ = fm.chip_raster(ts.isel(time=0), chip_size=32)
    assert single.shape == (6, 4, 32, 32)


def test_tokens_to_grid():
    g = fm._tokens_to_grid(torch.ones(2, 197, 8))  # CLS + 14 x 14
    assert g.shape == (2, 8, 14, 14)
    g2 = fm._tokens_to_grid(torch.arange(2 * 393 * 4.).reshape(2, 393, 4))  # CLS + 2 x 14 x 14
    assert g2.shape == (2, 4, 14, 14)
    assert fm._tokens_to_grid(torch.ones(1, 201, 3)).shape == (1, 3, 14, 14)  # CLS + 4 registers
    assert fm._tokens_to_grid(torch.ones(1, 6, 5, 5)).shape == (1, 6, 5, 5)
    with pytest.raises(ValueError, match="pooled"):
        fm._tokens_to_grid(torch.ones(2, 8))


def test_dense_embed_raster_coordinates():
    enc = _encoder()
    da = _raster()
    d = enc.embed_raster(da, chip_size=32, dense=True)
    assert d.dims == ("embedding", "y", "x") and d.sizes["embedding"] == 8
    assert d.shape[1:] == (50, 70)  # a conv keeps full resolution, padding cropped
    np.testing.assert_allclose(d.x.values, da.x.values)
    np.testing.assert_allclose(d.y.values, da.y.values)
    assert d.attrs["dense"] == 1 and d.rio.crs.to_epsg() == 32618
    with pytest.raises(ValueError, match="stride"):
        enc.embed_raster(da, chip_size=32, stride=16, dense=True)


def test_chip_metadata():
    m = fm.chip_metadata([4_700_000.0, 4_690_000.0], [400_000.0], "EPSG:32618", time="2024-07-01",
                         chip_size=224, resolution=10)
    assert list(m.columns) == ["lon", "lat", "days", "year", "doy", "area_km2"]
    assert len(m) == 2 and m["lat"].iloc[0] > m["lat"].iloc[1]
    assert m["year"].iloc[0] == 2024 and m["doy"].iloc[0] == 183
    assert m["days"].iloc[0] == pytest.approx(19905.0)
    assert m["area_km2"].iloc[0] == pytest.approx(5.0176)
    assert np.isnan(fm.chip_metadata([4.7e6], [4e5], 32618)["days"].iloc[0])


def test_slice_batch_kwargs():
    meta = np.arange(10)
    kw = fm._slice_batch_kwargs({"metadata": meta, "wavelengths": [1, 2]}, 4, 4, 10)
    assert list(kw["metadata"]) == [4, 5, 6, 7] and kw["wavelengths"] == [1, 2]


def test_benchmark_encoders_with_embeddings():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 40)
    good = np.c_[y + 0.1 * rng.normal(size=80), rng.normal(size=80)]
    bad = rng.normal(size=(80, 2))
    res = fm.benchmark_encoders({"good": {"embeddings": good}, "noise": {"embeddings": bad}}, y, cv=4)
    assert res.loc["good", "linear_accuracy"] > 0.95 > res.loc["noise", "linear_accuracy"]
    assert {"embed_dim", "knn_accuracy", "params_m"} <= set(res.columns)


def test_benchmark_encoders_loads_models(monkeypatch):
    monkeypatch.setattr(fm, "load_model", lambda name, device=None, **kw: _encoder())
    x = np.random.default_rng(0).normal(size=(40, 4, 32, 32)).astype("float32")
    res = fm.benchmark_encoders({"tiny": {"x": x}}, np.repeat([0, 1], 20), cv=4, probes=("linear",))
    assert res.loc["tiny", "embed_dim"] == 8 and res.loc["tiny", "params_m"] > 0
    assert res.loc["tiny", "s_per_1000"] > 0


def test_sample_alphaearth_years(fake_ee):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    pts = gpd.GeoDataFrame({"v": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs=4326)
    out = fm.sample_alphaearth(pts, [2020, 2024], source="earthengine")
    assert len(out) == 4 and sorted(out["year"].unique()) == [2020, 2024]


@pytest.mark.parametrize("device", DEVICES)
def test_cached_prithvi_time_series_and_dense(device):
    path = _HF / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M"
    if not path.exists():
        pytest.skip("Prithvi weights not cached")
    pytest.importorskip("terratorch")
    enc = fm.load_model("prithvi-eo-2.0-300m", device=device)
    bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    rng = np.random.default_rng(0)
    y = 4_700_000 - 30 * (np.arange(230) + 0.5)
    x = 400_000 + 30 * (np.arange(230) + 0.5)
    da = xr.DataArray(rng.random((6, 230, 230), dtype="float32"), dims=("band", "y", "x"),
                      coords={"band": bands, "y": y, "x": x}).rio.write_crs(32615)
    ts = xr.concat([da, da], dim=pd.Index(pd.to_datetime(["2024-05-01", "2024-08-01"]), name="time"))
    e = enc.embed_raster(ts)
    assert e.shape == (1024, 2, 2) and np.isfinite(e.values).all()
    d = enc.embed_raster(da.isel(y=slice(0, 224), x=slice(0, 224)), dense=True)
    assert d.shape == (1024, 14, 14)


# ---------------------------------------------------------------------------------------------
# model input contracts
# ---------------------------------------------------------------------------------------------

def _s2(units="reflectance"):
    bands = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16",
             "swir22"]
    v = np.linspace(0.05, 0.4, 10, dtype="float32")[:, None, None] * np.ones((10, 8, 9), "float32")
    if units == "dn":
        v = v * 10000
    return xr.DataArray(v, dims=("band", "y", "x"),
                        coords={"band": bands, "y": np.arange(8.)[::-1], "x": np.arange(9.)})


def test_input_spec_lookup():
    assert fm.input_spec("prithvi-eo-2.0-300m-tl")["mean"][0] == 1087.0
    assert fm.input_spec("prithvi-eo-1.0-100m")["mean"][0] == 775.0
    assert len(fm.input_spec("ssl4eo-s12-vits16-dino")["bands"]) == 13
    assert fm.input_spec("satlas-swinb-s2-ms")["img_size"] == 256
    assert fm.input_spec("dofa-large")["bands"] is None
    with pytest.raises(KeyError):
        fm.input_spec("aurora")


def test_prepare_prithvi_units_and_order():
    a = fm.prepare("prithvi-eo-2.0-300m", _s2())
    b = fm.prepare("prithvi-eo-2.0-300m", _s2("dn"))
    np.testing.assert_allclose(a.values, b.values, rtol=1e-5)  # reflectance and DN agree
    assert list(a.band.values) == ["blue", "green", "red", "nir08", "swir16", "swir22"]
    assert float(a.sel(band="blue")[0, 0]) == pytest.approx((500 - 1087) / 2248, rel=1e-4)
    assert a.attrs["input_units"] == "reflectance"


def test_prepare_fills_missing_bands():
    x = fm.prepare("ssl4eo-s12-resnet18-moco", _s2())
    assert x.sizes["band"] == 13 and set(x.attrs["filled_bands"].split(",")) == {
        "coastal", "watervapor", "cirrus"}
    np.testing.assert_allclose(x.sel(band="coastal"), x.sel(band="blue"))  # nearest wavelength
    z = fm.prepare("ssl4eo-s12-resnet18-moco", _s2(), missing="zero")
    assert float(np.abs(z.sel(band="cirrus")).max()) == 0
    with pytest.raises(KeyError, match="coastal"):
        fm.prepare("ssl4eo-s12-resnet18-moco", _s2(), missing="error")


def test_prepare_standardizes_without_statistics():
    x = fm.prepare("dofa-base", _s2() + np.random.default_rng(0).normal(0, 0.01, (10, 8, 9)))
    np.testing.assert_allclose(x.mean(("y", "x")), 0, atol=1e-5)
    s1 = xr.DataArray(np.full((2, 4, 4), -12.599, "float32"), dims=("band", "y", "x"),
                      coords={"band": ["vv", "vh"], "y": np.arange(4.), "x": np.arange(4.)})
    t = fm.prepare("terramind-v1-base", s1, modality="S1GRD")
    assert float(t.sel(band="vv")[0, 0]) == pytest.approx(0.0, abs=1e-5)


def test_embed_raster_model_norm():
    enc = _encoder()
    enc.name = "dofa-base"  # use DOFA's contract: any bands, standardized
    da = _raster()
    e = enc.embed_raster(da, chip_size=32, norm="model")
    assert e.shape == (8, 2, 3)
    with pytest.raises(ValueError, match="norm"):
        enc.embed_raster(da, chip_size=32, norm="imagenet")


def test_tokens_to_grid_with_known_steps():
    tok = torch.arange(1 * 785 * 2.).reshape(1, 785, 2)  # CLS + 4 frames x 14 x 14
    assert fm._tokens_to_grid(tok, steps=4).shape == (1, 2, 14, 14)
    assert fm._tokens_to_grid(tok).shape == (1, 2, 28, 28)  # ambiguous without the hint


def test_alphaearth_to_xarray_tiles(fake_ee):
    _, calls = fake_ee
    da = fm.alphaearth_to_xarray((90.0, 22.0, 90.001, 22.0005), 2024, scale=10, bands=["A00", "A05"], tile=5,
                                 source="earthengine")
    assert da.shape == (2, 6, 12) and np.isfinite(da.values).all()
    assert sum(c[0] == "computePixels" for c in calls) == 2 * 3  # 6 x 12 pixels in 5 x 5 tiles
    assert (da.sel(embedding="A05").values == 1).all()


# ---------------------------------------------------------------------------------------------
# AlphaEarth from the public Zarr mosaic (a fake in-memory array, no network)
# ---------------------------------------------------------------------------------------------

def test_alphaearth_dequantize():
    v = np.array([-128, -127, 0, 64, 127], dtype="int8")
    d = fm.alphaearth_dequantize(v)
    assert np.isnan(d[0]) and d[2] == 0
    np.testing.assert_allclose(d[[1, 3, 4]], [-(127 / 127.5) ** 2, (64 / 127.5) ** 2, (127 / 127.5) ** 2], rtol=1e-6)


@pytest.fixture
def fake_aef(monkeypatch):
    rng = np.random.default_rng(0)
    # a small mosaic around lon -94, lat 42.9: rows/cols relative to the real origin
    r0 = int((fm._AEF_Y0 - 42.95) / fm._AEF_GSD); c0 = int((-94.05 - fm._AEF_X0) / fm._AEF_GSD)
    block = rng.integers(-127, 128, (2, 64, 1200, 1200), dtype="int8")

    class Fake:
        shape = (2, 64, 10**7, 10**7)

        def __getitem__(self, key):
            t, b, r, c = key
            r = slice(r.start - r0, r.stop - r0) if isinstance(r, slice) else r - r0
            c = slice(c.start - c0, c.stop - c0) if isinstance(c, slice) else c - c0
            return block[t, b, r, c]

    monkeypatch.setattr(fm, "_AEF_ARRAY", Fake())
    monkeypatch.setattr(fm, "_AEF_FIRST_YEAR", 2023)
    return block, r0, c0


def test_sample_alphaearth_zarr(fake_aef):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    block, r0, c0 = fake_aef
    pts = gpd.GeoDataFrame({"v": [1, 2]}, geometry=[Point(-94.0, 42.9), Point(-93.99, 42.91)], crs=4326)
    out = fm.sample_alphaearth(pts, 2024, workers=2)
    assert list(out["v"]) == [1, 2] and set(fm.ALPHAEARTH_BANDS) <= set(out.columns)
    r = int((fm._AEF_Y0 - 42.9) / fm._AEF_GSD) - r0; c = int((-94.0 - fm._AEF_X0) / fm._AEF_GSD) - c0
    np.testing.assert_allclose(out.loc[0, fm.ALPHAEARTH_BANDS].to_numpy(float), fm.alphaearth_dequantize(block[1, :, r, c]))
    long = fm.sample_alphaearth(pts, [2023, 2024], workers=2)
    assert len(long) == 4 and sorted(long["year"].unique()) == [2023, 2024]
    with pytest.raises(ValueError, match="years"):
        fm.sample_alphaearth(pts, 2030)


def test_alphaearth_to_xarray_zarr_like(fake_aef):
    y = np.arange(4_752_000, 4_751_000, -100.0) - 50
    x = np.arange(410_000, 411_000, 100.0) + 50
    grid = xr.DataArray(np.zeros((10, 10)), dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32615)
    da = fm.alphaearth_to_xarray(None, 2024, like=grid, source="zarr", bands=["A00", "A01"])
    assert da.dims == ("embedding", "y", "x") and da.shape == (2, 10, 10)
    np.testing.assert_allclose(da.x, grid.x); assert da.rio.crs.to_epsg() == 32615
    assert np.isfinite(da.values).all() and np.abs(da.values).max() <= 1.0
