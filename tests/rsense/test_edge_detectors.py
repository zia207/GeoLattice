"""Tests for FieldEdge-compatible boundary detection helpers."""

import numpy as np
import pytest

from geolattice.rsense import edge_detectors as edge


def test_rgb_normalization_and_range_validation():
    image = np.full((8, 9, 3), 255, dtype=np.uint8)
    assert np.all(edge._as_rgb01(image) == 1)
    assert np.all(edge._as_rgb01(np.ones((8, 9, 3), dtype=np.float32)) == 1)
    assert np.allclose(edge._as_rgb01(np.full((8, 9, 3), 50.0), (0, 100)), 0.5)
    with pytest.raises(ValueError, match="image_range"):
        edge._as_rgb01(np.full((8, 9, 3), 1000.0, dtype=np.float32))


def test_boundary_to_instances_splits_fields():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[15:17, :] = 1
    labels = edge.boundary_to_instances(
        probability, threshold=0.5, min_area_pixels=1, dilate_pixels=0
    )
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.all(labels[15:17] == 0)


def test_tiled_probability_blending_on_odd_sized_image():
    torch = pytest.importorskip("torch")

    class ConstantBoundary(torch.nn.Module):
        def forward(self, image):
            logits = torch.zeros((image.shape[0], 2, *image.shape[-2:]), device=image.device)
            logits[:, 1] = 1
            return logits

    model = ConstantBoundary()
    image = np.zeros((37, 45, 3), dtype=np.uint8)
    probability = edge.predict_boundary_probability(
        image, model, tile_size=32, overlap=8, batch_size=2, device="cpu"
    )
    assert probability.shape == image.shape[:2]
    assert np.allclose(probability, 1 / (1 + np.exp(-1)), atol=1e-6)


def test_fieldedge_model_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    model = edge.build_edge_model("FC-DenseNet56", device="cpu")
    checkpoint = tmp_path / "fieldedge.pt"
    torch.save(
        {
            "preset": "FC-DenseNet56",
            "num_classes": 2,
            "threshold": np.float64(0.4),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    loaded = edge.load_edge_model(checkpoint, device="cpu")
    assert loaded.edge_threshold == pytest.approx(0.4)
    with torch.inference_mode():
        logits = loaded(torch.zeros(1, 3, 32, 32))
    assert logits.shape == (1, 2, 32, 32)


def test_field_polygons_keep_map_crs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")

    class CrossBoundary(torch.nn.Module):
        def forward(self, image):
            boundary = (image[:, :1] > 0.5).float() * 8
            return torch.cat([torch.full_like(boundary, 8) - boundary, boundary], dim=1)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[15:17, :, 0] = 255
    image[:, 15:17, 0] = 255
    from rasterio.transform import from_origin

    result = edge.detect_field_edges(
        image,
        CrossBoundary(),
        transform=from_origin(500000, 4600000, 1, 1),
        crs="EPSG:32615",
        threshold=0.5,
        min_area_m2=0,
        min_area_pixels=1,
        dilate_pixels=0,
        tile_size=32,
        overlap=0,
        device="cpu",
    )
    assert result.fields.crs.to_epsg() == 32615
    assert len(result.fields) == 4
    assert result.boundaries.crs.to_epsg() == 32615
    assert result.probability.shape == (32, 32)


def test_paper_networks_take_four_bands():
    torch = pytest.importorskip("torch")
    for name in ("SegNet", "FCN-DKConv6", "FC-DenseNet56"):
        model = edge.build_edge_model(name, in_channels=4, device="cpu").eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 4, 64, 64))
        assert logits.shape == (2, 2, 64, 64), name
    assert {"SegNet", "FCN-DKConv6"} <= set(edge.available_models())


def test_four_band_tiled_inference():
    torch = pytest.importorskip("torch")

    class NirBoundary(torch.nn.Module):
        def forward(self, image):
            b = image[:, 3:4] * 10
            return torch.cat([torch.zeros_like(b), b], dim=1)

    image = np.zeros((40, 50, 4), dtype=np.float32)
    image[:, 20:22, 3] = 1.0
    p = edge.predict_boundary_probability(image, NirBoundary(), tile_size=32, overlap=8, device="cpu")
    assert p.shape == (40, 50)
    assert p[:, 20:22].min() > 0.99 and p[:, :10].max() == pytest.approx(0.5)


def test_rasterize_tile_and_augment():
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from shapely.geometry import box

    fields = [box(0, 0, 10, 20), box(10, 0, 20, 20)]
    lab, ok = edge.rasterize_boundaries(fields, transform=from_origin(0, 20, 1, 1), shape=(20, 20),
                                        buffer=0.5, return_valid=True)
    assert lab[:, 9:11].all() and not lab[5:15, 3:7].any() and ok.all()
    image = np.random.default_rng(0).random((20, 20, 4)).astype("float32")
    tiles = edge.tile_pairs(image, lab, size=8, valid=ok)
    assert tiles["images"].shape == (4, 4, 8, 8) and tiles["labels"].shape == (4, 8, 8)
    assert np.array_equal(tiles["images"][1], image[0:8, 8:16].transpose(2, 0, 1))
    xi, yl, vm = edge.dihedral_augment(tiles["images"], tiles["labels"], extra=[tiles["valid"]])
    assert xi.shape[0] == yl.shape[0] == vm.shape[0] == 16
    assert np.array_equal(yl[4], np.rot90(tiles["labels"][0]))


def test_classical_edges_and_metrics():
    pytest.importorskip("skimage")
    pytest.importorskip("sklearn")
    image = np.zeros((40, 40, 3), dtype=np.float32)
    image[:, 20:] = 1.0
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[:, 19:21] = 1
    for method in ("roberts", "sobel", "prewitt", "scharr", "log", "canny"):
        score = edge.classical_edges(image, method, sigma=1.0)
        assert score.shape == truth.shape and 0 <= score.min() and score.max() <= 1
        assert score[:, 18:22].mean() > score[:, :10].mean(), method
    perfect = edge.edge_metrics(truth.astype(float), truth)
    assert perfect["boundary_f1"] == pytest.approx(1) and perfect["auc"] == pytest.approx(1)
    shifted = np.roll(truth, 1, axis=1).astype(float)
    assert edge.edge_metrics(shifted, truth)["boundary_f1"] < 0.6
    assert edge.edge_metrics(shifted, truth, tolerance=1)["boundary_f1"] == pytest.approx(1)
    best, table = edge.select_threshold(truth * 0.7, truth)
    assert best <= 0.7 and table["boundary_f1"].max() == pytest.approx(1)


def test_ftw_registry_and_local_cache(tmp_path):
    table = edge.ftw_models()
    for name in ("FTW_PRUE_EFNET_B3", "FTW_PRUE_EFNET_B5", "FTW_PRUE_EFNET_B7",
                 "FTW_PRUE_EFNET_B5_CCBY"):
        assert name in table.index
    assert table.loc["FTW_PRUE_EFNET_B5_CCBY", "license"] == "CC-BY-4.0"
    with pytest.raises(FileNotFoundError, match="prue_efnet5_checkpoint.ckpt"):
        edge.ftw_checkpoint("FTW_PRUE_EFNET_B5", cache_dir=tmp_path)  # no implicit download
    cached = tmp_path / "prue_efnet3_checkpoint.ckpt"
    cached.write_bytes(b"x")
    assert edge.ftw_checkpoint("FTW_PRUE_EFNET_B3", cache_dir=tmp_path) == cached
    with pytest.raises(ValueError, match="Unknown FTW model"):
        edge.ftw_checkpoint("not_a_model", cache_dir=tmp_path)


def test_ftw_lightning_checkpoint_loads_into_build_segmenter(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from geolattice.rsense.classify.image_classification import build_segmenter

    net = build_segmenter("unet", in_channels=8, num_classes=3, encoder="resnet18", weights=None)
    ckpt = tmp_path / "ftw_like.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in net.state_dict().items()},
                "hyper_parameters": {"model": "unet", "backbone": "resnet18", "in_channels": 8,
                                     "num_classes": 3}}, ckpt)
    model = edge.load_ftw_model(ckpt, device="cpu")
    assert model.ftw_metadata["backbone"] == "resnet18"
    assert model.ftw_metadata["normalization"] == 3000.0
    image = np.random.default_rng(0).integers(0, 4000, (8, 40, 50)).astype("float32")
    proba = edge.predict_ftw(image, model, tile_size=32, overlap=8, resize_factor=2, device="cpu")
    assert proba.shape == (3, 40, 50)
    assert np.allclose(proba.sum(0), 1, atol=1e-4)
    with pytest.raises(ValueError, match="takes 8 bands"):
        edge.predict_ftw(image[:4], model, tile_size=32, overlap=8, device="cpu")


def test_ftw_stack_detect_fields_and_fiboa():
    torch = pytest.importorskip("torch")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(1)
    names = ["blue", "green", "red", "nir"]
    coords = {"band": names, "y": 4_800_000 - 10 * np.arange(64) - 5, "x": 400_000 + 10 * np.arange(64) + 5}
    win = [xr.DataArray(rng.random((4, 64, 64), dtype="float32") * 3000, dims=("band", "y", "x"),
                        coords=coords).rio.write_crs("EPSG:32615") for _ in range(2)]
    stack = edge.ftw_stack(*win)
    assert list(stack.band.values) == ["B04_A", "B03_A", "B02_A", "B08_A", "B04_B", "B03_B", "B02_B", "B08_B"]
    assert np.array_equal(stack.values[0], win[0].sel(band="red").values)
    assert stack.rio.transform() == from_origin(400_000, 4_800_000, 10, 10)

    class TwoFields(torch.nn.Module):          # field everywhere except a boundary column
        def forward(self, x):
            logits = torch.zeros((x.shape[0], 3, *x.shape[-2:]))
            logits[:, 1] = 5
            w = x.shape[-1]
            logits[:, 2, :, w // 2 - 4: w // 2 + 4] = 10
            return logits

    res = edge.detect_fields_ftw(stack, TwoFields(), tile_size=64, overlap=0, resize_factor=1,
                                 min_area_m2=1000, simplify=0, device="cpu")
    assert len(res.fields) == 2
    assert res.class_probability.shape == (3, 64, 64)
    assert res.fields.crs.to_epsg() == 32615
    fb = edge.to_fiboa(res.fields, collection="test", determination_datetime="2025-10-10",
                       determination_details="unit test")
    for col in ("id", "collection", "metrics:area", "metrics:perimeter", "determination:method",
                "determination:datetime"):
        assert col in fb.columns
    assert (fb["determination:method"] == "auto-imagery").all()
    assert fb["id"].is_unique and fb.attrs["schemas"]["test"][0].endswith("schema.yaml")
