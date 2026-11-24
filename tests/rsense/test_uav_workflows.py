"""Tests for the UAV workflow modules: io, register, indices, tiling, dem, counting, detect,
health, phenology, weights, and the canopy/odm/segment updates."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import uav

cv2 = pytest.importorskip("cv2")


def _png(path, arr):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, np.uint8)).save(path)
    return path


def _texture(h=240, w=320, seed=0):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    t = gaussian_filter(rng.random((h, w)), 2.0)
    return ((t - t.min()) / (t.max() - t.min()) * 255).astype(np.uint8)


# ------------------------------------------------------------------ tiling
def test_sliding_windows_cover_every_pixel():
    cov = np.zeros((1000, 1300), int)
    for r0, c0, h, w in uav.sliding_windows(1000, 1300, 512, 64):
        assert (h, w) == (512, 512)
        cov[r0 : r0 + h, c0 : c0 + w] += 1
    assert cov.min() >= 1
    assert list(uav.sliding_windows(100, 80, 512, 64)) == [(0, 0, 100, 80)]
    with pytest.raises(ValueError):
        list(uav.sliding_windows(10, 10, 8, 8))


@pytest.mark.parametrize("blend", ["cosine", "mean", "max"])
def test_predict_tiled_identity(blend):
    a = np.random.default_rng(1).random((300, 410)).astype(np.float32)
    out = uav.predict_tiled(a, lambda t: t, tile=128, overlap=32, blend=blend)
    np.testing.assert_allclose(out, a, atol=1e-5)


def test_merge_points_and_boxes():
    pts = pd.DataFrame({"x": [10, 12, 50, 51.5], "y": [10, 11, 50, 50], "s": [1, 3, 5, 7]})
    m = uav.merge_points(pts, radius=5)
    assert len(m) == 2 and sorted(m.n_merged) == [2, 2]
    assert m.loc[m.x < 20, "s"].item() == pytest.approx(2)
    boxes = pd.DataFrame(
        {
            "x1": [0, 2, 100],
            "y1": [0, 1, 100],
            "x2": [20, 21, 120],
            "y2": [20, 21, 120],
            "confidence": [0.9, 0.8, 0.7],
            "class": [0, 0, 0],
        }
    )
    kept = uav.merge_boxes(boxes, iou=0.4)
    assert len(kept) == 2 and kept.confidence.max() == 0.9
    other_class = boxes.assign(**{"class": [0, 1, 0]})
    assert len(uav.merge_boxes(other_class, iou=0.4)) == 3


# ------------------------------------------------------------------ io
def test_read_frame_gsd_and_pixels(tmp_path):
    rgb = np.dstack([_texture(seed=i) for i in range(3)])
    p = _png(tmp_path / "f.png", rgb)
    px = uav.read_frame(p)
    assert px.dims == ("band", "y", "x") and list(px.band.values) == ["red", "green", "blue"]
    assert px.attrs["units_xy"] == "px" and float(px.max()) <= 1.0
    m = uav.read_frame(p, gsd=0.002)
    assert float(m.x[1] - m.x[0]) == pytest.approx(0.002) and m.attrs["gsd_m"] == 0.002
    small = uav.read_frame(p, gsd=0.002, max_dim=160)
    assert small.sizes["x"] == 160 and small.attrs["gsd_m"] == pytest.approx(0.004)
    assert uav.to_rgb_array(m, uint8=True).dtype == np.uint8


def test_catalog_layouts(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    for d in ("canola_30072018", "canola-26072019"):
        for b in ("RGB", "NIR"):
            for f in ("0", "1"):
                _png(tmp_path / "Canola dataset" / d / b / f"{f}.png", img)
    _png(tmp_path / "Drybean-dataset" / "drybean-04072020" / "RGB-1" / "5.png", img)
    _png(tmp_path / "Drybean-dataset" / "drybean-04072020" / "NIR-1" / "5.png", img)
    _png(tmp_path / "Drybean-dataset" / "drybean-04072020" / "RGB-1" / "6.png", img)  # unpaired
    # a second camera rig in the same session reuses frame numbers
    _png(tmp_path / "Drybean-dataset" / "drybean-04072020" / "RGB-2" / "5.png", img)
    _png(tmp_path / "Drybean-dataset" / "drybean-04072020" / "NIR-2" / "5.png", img)
    _png(tmp_path / "infrastructure-free-uav-phenotyping" / "2025-08-22" / "frame130.png", img)
    cat = uav.build_catalog(tmp_path)
    assert len(cat) == 14
    lums = cat[cat.dataset.str.startswith("infra")].iloc[0]
    assert lums.date == pd.Timestamp("2025-08-22") and lums.band == "RGB"
    can = cat[cat.session == "canola_30072018"].iloc[0]
    assert can.crop == "canola" and can.date == pd.Timestamp("2018-07-30")
    assert set(cat.band) == {"RGB", "NIR"}
    pairs = uav.pair_catalog(cat)
    assert len(pairs) == 6 and {"rgb_path", "nir_path"} <= set(pairs.columns)
    bean = pairs[pairs.crop == "drybean"]
    assert sorted(bean.rig) == ["1", "2"]
    for _, r in bean.iterrows():  # never pair RGB-1 with NIR-2
        assert f"RGB-{r.rig}" in r.rgb_path and f"NIR-{r.rig}" in r.nir_path
    assert uav.parse_date("drybean-25062020") == pd.Timestamp("2020-06-25").date()
    assert uav.parse_date("no date") is None


def test_quality_metrics():
    sharp = _texture()
    from scipy.ndimage import gaussian_filter

    blurry = gaussian_filter(sharp.astype(float), 4)
    assert uav.blur_score(sharp) > 3 * uav.blur_score(blurry)
    st = uav.exposure_stats(np.full((10, 10), 255, np.uint8))
    assert st["saturated_frac"] == 1.0


def test_extract_frames(tmp_path):
    vid = tmp_path / "v.avi"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 48))
    for i in range(30):
        wr.write(np.full((48, 64, 3), i * 8, np.uint8))
    wr.release()
    df = uav.extract_frames(vid, tmp_path / "frames", step=6)
    assert list(df.frame) == [0, 6, 12, 18, 24]
    assert df.time_s.iloc[1] == pytest.approx(0.25)


# ------------------------------------------------------------------ register
def test_coregister_recovers_shift(tmp_path):
    ref = _texture(300, 400, seed=3)
    M = np.float32([[1, 0, 7.0], [0, 1, -5.0]])
    moving = cv2.warpAffine(ref, M, (400, 300), borderMode=cv2.BORDER_REFLECT)
    out, info = uav.coregister(moving, ref, method="orb")
    inner = (slice(20, -20), slice(20, -20))
    assert info["correlation"] > 0.97
    assert np.corrcoef(out[inner].ravel(), ref[inner].ravel())[0, 1] > 0.97
    out2, info2 = uav.coregister(moving, ref, method="ecc", motion="translation")
    assert np.corrcoef(out2[inner].ravel(), ref[inner].ravel())[0, 1] > 0.97
    # through load_pair: NIR shifted against RGB
    rgb = np.dstack([ref, ref, ref])
    _png(tmp_path / "rgb.png", rgb)
    _png(tmp_path / "nir.png", moving)
    pair = uav.load_pair(tmp_path / "rgb.png", tmp_path / "nir.png", register=True)
    assert list(pair.band.values) == ["red", "green", "blue", "nir"]
    g, n = pair.sel(band="green").values[inner], pair.sel(band="nir").values[inner]
    ok = np.isfinite(n)
    assert np.corrcoef(g[ok], n[ok])[0, 1] > 0.97


def test_coregister_validation_rejects_bad_solutions():
    ref = _texture(300, 400, seed=3)
    M = np.float32([[1, 0, 7.0], [0, 1, -5.0]])
    moving = cv2.warpAffine(ref, M, (400, 300), borderMode=cv2.BORDER_REFLECT)
    _, info = uav.coregister(moving, ref, method="orb")
    assert info["accepted"] and info["edge_after"] > info["edge_before"]
    assert 5 < info["shift_px"] < 12
    # an impossible inlier requirement forces the identity fallback
    out2, info2 = uav.coregister(moving, ref, method="orb", min_inliers=10**6)
    assert not info2["accepted"] and "inliers" in info2["reason"]
    assert np.allclose(info2["H"], np.eye(3))
    assert np.allclose(out2, moving.astype(np.float32))
    # a shift limit below the true offset is also rejected
    _, info3 = uav.coregister(moving, ref, method="orb", max_shift=2)
    assert not info3["accepted"] and "shift" in info3["reason"]
    # validation off keeps the raw solution
    _, info4 = uav.coregister(moving, ref, method="orb", min_inliers=10**6, validate=False)
    assert info4["accepted"] and not np.allclose(info4["H"], np.eye(3))
    assert uav.edge_agreement(ref, ref) == pytest.approx(1.0)


# ------------------------------------------------------------------ indices
def test_vegetation_indices():
    img = uav.as_image(
        np.dstack(
            [np.full((4, 4), 0.2), np.full((4, 4), 0.5), np.full((4, 4), 0.1), np.full((4, 4), 0.6)]
        ).astype(np.float32),
        bands=["red", "green", "blue", "nir"],
    )
    vi = uav.vegetation_indices(img)
    assert {"VARI", "TGI", "ExG", "NDVI", "GNDVI"} <= set(vi.data_vars)
    assert float(vi.ExG.mean()) == pytest.approx(2 * 0.5 - 0.2 - 0.1)
    assert float(vi.NDVI.mean()) == pytest.approx((0.6 - 0.2) / 0.8)
    t = uav.vegetation_indices(img, "TGI_MAXNORM")
    assert float(t.mean()) == pytest.approx((0.5 - 0.39 * 0.2 - 0.61 * 0.1) / 0.5)
    cive = uav.vegetation_indices(img, ["CIVE"])
    assert float(cive.CIVE.mean()) == pytest.approx(
        0.441 * 0.2 - 0.811 * 0.5 + 0.385 * 0.1 + 18.78745
    )
    n = uav.vegetation_indices(img, ["ExG"], normalize=True)
    assert float(n.ExG.mean()) == pytest.approx((2 * 0.5 - 0.2 - 0.1) / 0.8)
    row = uav.index_summary(vi[["ExG"]])
    assert row["ExG_mean"] == pytest.approx(0.7)


def test_bare_soil_line_and_cover_layers():
    # Soil on NIR = Red + 10 (exact in float32). One canopy pixel, one below the line.
    red = np.arange(5, 80, dtype="float64")
    nir = red + 10
    nir = np.vstack([nir, nir, nir + 30])
    red = np.vstack([red, red, red])
    nir[2, 0] = 180  # red 5: the full-canopy point
    red[1, 10] = 15.5  # not an integer level, so it does not move the soil line
    nir[1, 10] = 5  # below the line
    nir[0, 20] = np.nan
    red[0, 21] = 255  # saturated digital number, left out of the fit

    line = uav.bare_soil_line(red, nir)
    assert line["slope"] == pytest.approx(1.0, abs=1e-6)
    assert line["intercept"] == pytest.approx(10, abs=1e-6)
    assert line["r_squared"] == pytest.approx(1.0)
    assert line["top_red"] == 5 and line["top_nir"] == 180
    assert line["n_levels"] == 75

    img = uav.as_image(
        np.dstack([red, nir, np.zeros_like(red)]).astype("float32"),
        bands=["red", "nir", "green"],
    )
    layers = uav.cover_layers(img)
    assert set(layers.data_vars) == {"PVI", "GC", "WDVI"}
    assert layers.attrs["pvi_fc"] == pytest.approx((180 - 5 - 10) / np.sqrt(2))
    # Soil pixels have PVI 0. The canopy point is past the PVI clip, so GC stays below 1.
    assert float(layers.GC.values[0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert float(layers.PVI.values[1, 10]) == 0.0
    assert float(layers.GC.values[2, 0]) == pytest.approx(100 / layers.attrs["pvi_fc"], abs=1e-4)
    assert float(layers.GC.values[2, 0]) < 1
    assert np.isnan(layers.GC.values[0, 20])
    open_gc = uav.cover_layers(img, pvi_max=None)
    assert float(open_gc.GC.values[2, 0]) == pytest.approx(1.0, abs=1e-4)
    wdvi = nir[0, 0] / 255 - red[0, 0] / 255
    assert float(layers.WDVI.values[0, 0]) == pytest.approx(wdvi, abs=1e-5)

    rgb = uav.as_image(np.zeros((2, 2, 3), np.float32), bands=["red", "green", "blue"])
    with pytest.raises(ValueError, match="nir"):
        uav.cover_layers(rgb)
    with pytest.raises(ValueError, match="two red levels"):
        uav.bare_soil_line(np.array([4.0, 4.0]), np.array([8.0, 9.0]))


def test_cover_layers_barisal_wheat():
    """The 12 Feb 2015 Barisal mosaic matches the published soil-line numbers."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "Data/08_remote_sensing/08_04_uav/UAV_BD/multi_bwh_mc_20150212.tif"
    )
    if not path.exists():
        pytest.skip("Barisal UAV mosaic is not in the checkout")
    import rioxarray  # noqa: F401  (registers the CRS accessor)
    from geolattice.rsense import read

    stack = read(path, band_names=["b1", "red", "green", "b4", "nir"], masked=True)
    stack = stack.rio.write_crs("EPSG:32646")
    layers = uav.cover_layers(stack)
    assert layers.attrs["intercept"] == pytest.approx(11.9969, abs=1e-3)
    assert layers.attrs["slope"] == pytest.approx(0.9365, abs=1e-3)
    assert layers.attrs["top_red"] == 6 and layers.attrs["top_nir"] == 178
    assert layers.attrs["pvi_fc"] == pytest.approx(117.06, abs=0.02)
    assert float(layers.GC.mean()) == pytest.approx(0.480, abs=1e-3)
    assert layers.GC.rio.crs.to_epsg() == 32646


# ------------------------------------------------------------------ dem
def test_ransac_flatten_removes_dome():
    pytest.importorskip("sklearn")
    rr, cc = np.mgrid[0:200, 0:300]
    dome = -2e-4 * ((rr - 100) ** 2 + (cc - 150) ** 2) + 0.002 * cc
    rng = np.random.default_rng(0)
    plants = np.zeros_like(dome)
    plants[rng.random(dome.shape) < 0.15] = 0.6
    d1 = 100 + dome + rng.normal(0, 0.01, dome.shape)
    d2 = 100 + dome + plants + rng.normal(0, 0.01, dome.shape)
    d1[:3] = -32767  # nodata strip
    res = uav.ransac_flatten([d1, d2], degree=2)
    soil = plants == 0
    soil[:3] = False
    assert np.nanstd(res.dems[0].values[soil]) < 0.03
    assert np.nanstd(dome[soil]) > 1.0
    h = res.dems[1].values
    assert np.nanmedian(h[plants > 0]) - np.nanmedian(h[soil]) == pytest.approx(0.6, abs=0.05)
    assert np.isnan(res.dems[0].values[:3]).all() and res.ground is not None
    assert res.info["trend_range"] > 1.0
    lums = uav.ransac_flatten([d1, d2], normalize="lums")
    assert np.nanmax(np.concatenate([d.values.ravel() for d in lums.dems])) == pytest.approx(5.0)
    assert uav.doming_profile(res.dems[0]).sizes["x"] <= 300


# ------------------------------------------------------------------ canopy updates
def test_count_plants_plain_array_and_gsd():
    yy, xx = np.mgrid[0:200, 0:200]
    centers = [(r, c) for r in range(20, 200, 40) for c in range(20, 200, 40)]
    a = sum(np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / 30.0) for r, c in centers)
    pts = uav.count_plants(a, min_distance_m=0.5, threshold=0.5, gsd=0.05)
    assert len(pts) == len(centers) and {"row", "col"} <= set(pts.columns)
    assert pts.geometry.x.min() == pytest.approx(20.5 * 0.05, abs=0.06)
    pts_px = uav.count_plants(
        uav.as_image(a.astype(np.float32), bands=["exg"]),
        min_distance_m=10,
        threshold=0.5,
        sigma_m=1,
    )
    assert len(pts_px) == len(centers)


def _grid(a, res=0.05, x0=400_000.0):
    H, W = a.shape
    y = 4_700_000 - res * (np.arange(H) + 0.5)
    x = x0 + res * (np.arange(W) + 0.5)
    return xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)


def test_chm_aligns_offset_grids_of_same_shape():
    ramp = np.tile(np.arange(20, dtype=float), (10, 1))
    dsm = _grid(100 + ramp + 2.0)
    dtm = _grid(100 + np.tile(np.arange(-2, 18, dtype=float), (10, 1)), x0=400_000 - 0.1)
    chm = uav.chm_from_surfaces(dsm, dtm)
    # dtm shifted 2 px west: after alignment the inner columns show 2 m everywhere
    assert np.nanmax(np.abs(chm.values[:, 3:-3] - 2.0)) < 1e-6
    assert chm.rio.crs is not None


def test_plants_per_area_projects_geographic():
    import geopandas as gpd
    from shapely.geometry import Point, box

    poly = gpd.GeoDataFrame(geometry=[box(-76.5, 42.44, -76.4999, 42.4401)], crs=4326)
    pts = gpd.GeoDataFrame(geometry=[Point(-76.49995, 42.44005)] * 10, crs=4326)
    out = uav.plants_per_area(pts, poly)
    assert out["count"].item() == 10
    assert 50 < out["area"].item() < 150  # m², not square degrees


# ------------------------------------------------------------------ odm / segment
def test_flight_plan_presets_and_video():
    p = uav.flight_plan(100, camera="rededge_mx")
    assert p["gsd_cm"] == pytest.approx(4.8 * 100 / (5.4 * 1280) * 100)
    v = uav.flight_plan(6, camera="mini4pro_4k_video", speed_ms=1.0, fps=24, frame_step=6)
    assert v["frame_spacing_m"] == pytest.approx(0.25) and 0.9 < v["video_front_overlap"] < 1
    assert uav.flight_plan(100) == uav.flight_plan(100, camera="phantom4pro")


def test_load_outputs_explicit_paths(tmp_path):
    da = _grid(np.ones((5, 5), np.float32))
    da.rio.to_raster(tmp_path / "dsm.tif")
    out = uav.load_outputs(dsm=tmp_path / "dsm.tif", chunks=None)
    assert set(out) == {"dsm"}


def test_sam_segment_validates_prompts():
    with pytest.raises(ValueError, match="points"):
        uav.sam_segment("x.tif", mode="points")
    with pytest.raises(TypeError, match="sam_plant_mask"):
        uav.sam_segment(uav.as_image(np.zeros((4, 4, 3), np.float32)))


# ------------------------------------------------------------------ counting
def test_peaks_from_density():
    d = np.zeros((50, 50), np.float32)
    d[10, 10] = d[30, 40] = 1.0
    d[40:42, 5:7] = 0.5  # plateau counts once
    pk = uav.peaks_from_density(d, threshold=0.1)
    assert len(pk) == 3


def test_density_count_with_stub_model():
    torch = pytest.importorskip("torch")
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    H, W, gsd = 600, 900, 0.002
    truth = [(rng.integers(30, H - 30), rng.integers(30, W - 30)) for _ in range(25)]
    dens_full = np.zeros((H, W), np.float32)
    for r, c in truth:
        dens_full[r, c] = 1
    dens_full = gaussian_filter(dens_full, 6)
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 1] = np.clip(dens_full / dens_full.max() * 255, 0, 255)
    image = uav.as_image(img, gsd=gsd)

    class Stub(torch.nn.Module):
        """Returns the green channel (de-normalised) as a density map at input resolution."""

        def forward(self, x):
            g = x[:, 1:2] * 0.224 + 0.456
            return g / 50.0, g

    pts, dens = uav.density_count(
        image,
        Stub(),
        tile=256,
        overlap=64,
        input_size=128,
        peak_threshold=0.002,
        border=4,
        padding=10,
        merge_radius=12,
        device="cpu",
    )
    assert abs(len(pts) - len(truth)) <= 2
    assert dens.shape == (H, W) and {"x", "y", "size_px", "size_m"} <= set(pts.columns)
    assert pts.size_m.notna().all()
    assert set(uav.size_classes(pts)) <= {"small", "medium", "large"}


def test_ricenet_shapes():
    torch = pytest.importorskip("torch")
    net = uav.build_ricenet().eval()
    with torch.no_grad():
        d, a = net(torch.zeros(1, 3, 64, 64))
    assert d.shape == (1, 1, 32, 32) and a.shape == d.shape
    size_net = uav.build_ricenet(gap=True).eval()
    with torch.no_grad():
        assert size_net(torch.zeros(1, 3, 64, 64)).shape == (1, 1)


# ------------------------------------------------------------------ detect
def _square_detector(tile, conf):
    from scipy.ndimage import find_objects, label

    lab, _ = label(tile[..., 0] > 128)
    out = [[s[1].start, s[0].start, s[1].stop, s[0].stop, 0.95, 0] for s in find_objects(lab)]
    return np.array(out, np.float32).reshape(-1, 6)


def test_tiled_detect_merges_across_windows():
    img = np.zeros((700, 900, 3), np.uint8)
    truth = []
    for r in range(30, 680, 90):
        for c in range(30, 880, 110):
            img[r : r + 20, c : c + 20] = 255
            truth.append([c, r, c + 20, r + 20])
    image = uav.as_image(img, gsd=0.01)
    det = uav.tiled_detect(image, _square_detector, window=256, stride=128, conf=0.5, iou=0.3)
    gt = pd.DataFrame(truth, columns=["x1", "y1", "x2", "y2"])
    sc = uav.detection_scores(det, gt, iou=0.5)
    assert sc["recall"] == 1.0 and sc["count_agreement"] > 0.95
    assert {"x", "y"} <= set(det.columns)
    s = uav.count_summary(det, image)
    assert s["per_m2"] == pytest.approx(len(det) / (7.0 * 9.0), rel=1e-6)
    gdf = uav.boxes_to_gdf(det.head(1), image)
    assert gdf.geometry.iloc[0].area == pytest.approx(0.2 * 0.2, rel=0.25)


# ------------------------------------------------------------------ health
def _health_image():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:] = (120, 100, 80)  # soil
    img[40:100, 40:140] = (40, 170, 40)  # green plant
    img[120:180, 160:260] = (220, 200, 40)  # senescent (yellow)
    img[120:180, 40:120] = (60, 60, 60)  # shadow
    return img


def test_classify_senescence_rule_and_summary():
    img = _health_image()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:100, 40:140] = mask[120:180, 160:260] = mask[120:180, 40:120] = 1
    lab, p = uav.classify_senescence(img, mask)
    assert (lab[60, 80], lab[150, 200]) == (1, 2)
    filt = uav.hsv_filter(img, lab)
    assert filt[150, 80] == 0 and filt[150, 200] == 2
    s = uav.health_summary(filt)
    assert s.senescent_frac.item() == pytest.approx(6000 / 12000, rel=0.01)


def test_health_map_exg_and_segveg_model():
    sk = pytest.importorskip("sklearn.linear_model")
    img = _health_image()
    hm = uav.health_map(uav.as_image(img), tile=128, overlap=32)
    assert hm.dtype == np.uint8 and int(hm.values[60, 80]) == 1
    f = uav.segveg_features(img)
    assert f.shape == (200, 300, 14) and np.isfinite(f).all()
    X = np.vstack(
        [
            uav.segveg_features(np.array([[40, 170, 40]] * 20, np.uint8)),
            uav.segveg_features(np.array([[220, 200, 40]] * 20, np.uint8)),
        ]
    )
    y = np.r_[np.ones(20), np.zeros(20)]
    model = sk.LogisticRegression(max_iter=1000).fit(X, y)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:100, 40:140] = mask[120:180, 160:260] = 1
    lab, _ = uav.classify_senescence(img, mask, model=model, threshold=0.5)
    assert (lab[60, 80], lab[150, 200], lab[5, 5]) == (1, 2, 0)


# ------------------------------------------------------------------ phenology
def test_phenology(tmp_path):
    t = pd.date_range("2025-08-01", periods=10, freq="7D")
    days = (t - t[0]).days.to_numpy()
    y = 0.1 + 0.8 / (1 + np.exp(-0.15 * (days - 30)))
    fit = uav.fit_logistic(t, y)
    assert fit["r2"] > 0.999
    assert abs(fit["mid_date"] - pd.Timestamp("2025-08-31")) < pd.Timedelta(minutes=1)
    assert fit["max_rate"] == pytest.approx(0.8 * 0.15 / 4, rel=1e-3)
    assert np.allclose(fit["predict"](t), y, atol=1e-3)
    for i, d in enumerate(("2025-08-22", "2025-08-28")):
        _png(tmp_path / "lums" / d / "frame1.png", np.full((8, 8, 3), 50 * (i + 1), np.uint8))
    cat = uav.build_catalog(tmp_path)
    tab = uav.trait_table(cat, lambda im: {"bright": float(im.mean())})
    assert len(tab) == 2 and tab.bright.iloc[1] > tab.bright.iloc[0]
    summ = uav.summarize_dates(tab, ["bright"])
    assert "bright_mean" in summ
    gr = uav.growth_rates(tab, "bright")
    assert gr["bright_per_day"].iloc[1] == pytest.approx((100 - 50) / 255 / 6, rel=1e-3)


# ------------------------------------------------------------------ weights
def test_weights_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOLATTICE_UAV_WEIGHTS", str(tmp_path))
    tab = uav.list_weights()
    assert {"sam_vit_b", "ricenet_count", "yolov5_rice_panicle"} <= set(tab.name)
    assert not tab.cached.any()
    (tmp_path / "ricenet_count" / "sub").mkdir(parents=True)
    (tmp_path / "ricenet_count" / "sub" / "checkpoint_counting.pth").write_bytes(b"x")
    assert uav.fetch_weights("ricenet_count").name == "checkpoint_counting.pth"
    with pytest.raises(KeyError):
        uav.fetch_weights("nope")


def test_tiled_detect_edge_margin_zero_keeps_seam_fragments():
    img = np.zeros((300, 300, 3), np.uint8)
    img[120:140, 120:140] = 255  # whole in the first window
    img[250:270, 30:50] = 255  # cut by the first window's bottom edge (row 256)
    image = uav.as_image(img)
    kept = uav.tiled_detect(
        image, _square_detector, window=256, stride=128, conf=0.5, edge_margin=0
    )
    clean = uav.tiled_detect(image, _square_detector, window=256, stride=128, conf=0.5)
    assert len(clean) == 2 and len(kept) > len(clean)


# ------------------------------------------------------------------ synthetic DEM, stages, traits
def test_synthetic_dem_flatten_recovers_heights():
    dsm, chm = uav.synthetic_dem((120, 180), gsd=0.05, doming=1.5, canopy_height=0.6, seed=1)
    assert dsm.shape == chm.shape == (120, 180)
    assert float(dsm.max() - dsm.min()) > 1.0  # doming dominates the raw DSM
    res = uav.ransac_flatten([dsm], degree=2, ground_percentile=5)
    flat = res.dems[0].values
    plant = chm.values > 0
    assert abs(np.nanmedian(flat[plant]) - np.median(chm.values[plant])) < 0.15


def test_detect_stages_orders_and_matches_fit():
    dates = pd.date_range("2025-08-01", periods=10, freq="7D")
    t = np.arange(10) * 7.0
    y = 0.1 + 0.8 / (1 + np.exp(-0.15 * (t - 30)))
    fit = uav.fit_logistic(dates, y)
    st = uav.detect_stages(fit)
    assert list(st.stage) == ["onset", "mid", "plateau"]
    assert abs(st.loc[st.stage == "mid", "day"].item() - 30) < 0.5
    with pytest.raises(ValueError):
        uav.detect_stages(fit, levels={"bad": 1.0})


def test_frame_traits_on_synthetic_frame():
    rgb = np.zeros((80, 100, 3), np.uint8)
    rgb[...] = (120, 90, 60)  # soil
    rgb[20:60, 30:70] = (40, 160, 40)  # green canopy
    rgb[20:30, 30:70] = (200, 180, 40)  # yellow leaves
    img = uav.as_image(rgb / 255.0)
    tr = uav.frame_traits(img, count=False)
    # Otsu on a three-mode ExG (soil, yellow, green) may cut out the yellow strip
    assert 0.1 < tr["canopy_cover"] <= 0.2
    assert {"ExG_veg_mean", "VARI_veg_mean", "TGI_veg_mean", "senescent_frac"} <= tr.keys()
    fixed = uav.frame_traits(img, threshold=0.1)  # canopy is 40 x 40 of 80 x 100 pixels
    assert fixed["canopy_cover"] == pytest.approx(0.2)
    assert fixed["senescent_frac"] == pytest.approx(0.25)


def test_ransac_flatten_match_ground_keeps_growth():
    rng = np.random.default_rng(0)
    dems, truth = [], []
    for h in (0.3, 0.9):
        dsm, chm = uav.synthetic_dem((120, 180), canopy_height=h, doming=1.0, noise=0.0, seed=2)
        dems.append(dsm + rng.normal(0, 0.01, dsm.shape).astype("float32"))
        truth.append(chm)
    plant = truth[0].values > 0
    growth = float(truth[1].values[plant].mean() - truth[0].values[plant].mean())

    def est(res):
        return float(np.nanmean(res.dems[1].values[plant]) - np.nanmean(res.dems[0].values[plant]))

    by_mean = uav.ransac_flatten(dems, match="mean", ground_percentile=5)
    by_ground = uav.ransac_flatten(dems, match="ground", ground_percentile=5)
    assert by_ground.info["match"] == "ground"
    assert abs(est(by_ground) - growth) < 0.05
    assert est(by_mean) < growth - 0.1  # mean matching pulls the taller canopy down
    with pytest.raises(ValueError):
        uav.ransac_flatten(dems, match="median")


def test_cosine_weight_is_centred_and_symmetric():
    w = uav.cosine_weight(101, 61)
    assert w.shape == (101, 61)
    assert np.unravel_index(w.argmax(), w.shape) == (50, 30)
    assert w.max() == pytest.approx(1.0)
    assert np.allclose(w, w[::-1, :]) and np.allclose(w, w[:, ::-1])
    assert w[0, 0] == pytest.approx(1e-3) and w[-1, -1] == pytest.approx(1e-3)


def test_classify_senescence_hue_max():
    rgb = np.zeros((10, 20, 3), np.uint8)
    rgb[:, :10] = (120, 140, 50)  # olive green, hue about 70 degrees
    rgb[:, 10:] = (190, 160, 40)  # straw yellow, hue about 48 degrees
    strict, _ = uav.classify_senescence(rgb, None)  # default 75 degrees
    tuned, _ = uav.classify_senescence(rgb, None, hue_max=60)
    assert (strict[:, :10] == 2).all()  # olive called senescent at 75
    assert (tuned[:, :10] == 1).all() and (tuned[:, 10:] == 2).all()
    hm = uav.health_map(uav.as_image(rgb / 255.0), mask_method="none", hue_max=60, hsv=False, tile=16, overlap=4)
    assert (hm.values[:, :10] == 1).all()


def test_cwsi_empirical_baseline_physics():
    import xarray as xr

    ta, vpd = 32.0, 2.5
    t_wet = ta + 3.38 - 3.25 * vpd  # default baseline: canopy below air
    tc = xr.DataArray(np.array([[t_wet, (t_wet + ta + 5) / 2, ta + 5]]), dims=("y", "x"))
    c = uav.cwsi(tc, method="empirical", air_temp=ta, vpd=vpd)
    assert c.attrs["t_wet"] < ta < c.attrs["t_dry"]
    assert np.allclose(c.values, [[0.0, 0.5, 1.0]])
    drier = uav.cwsi(tc, method="empirical", air_temp=ta, vpd=vpd + 1)
    assert drier.attrs["t_wet"] < c.attrs["t_wet"]  # drier air, cooler unstressed canopy
    with pytest.raises(ValueError):
        uav.cwsi(tc, method="empirical", air_temp=ta, vpd=vpd, nwsb=(-1.9, 3.4))


def test_fit_logistic_bounds_rate_by_sampling():
    rng = np.random.default_rng(3)
    dates = np.repeat(pd.date_range("2025-08-01", periods=6, freq="7D"), 10)
    t = (dates - dates[0]).days.to_numpy().astype(float)
    y = 0.7 + 0.15 * (t > 14) + rng.normal(0, 0.08, t.size)  # a noisy step between two flights
    fit = uav.fit_logistic(dates, y)
    assert 4.39 / fit["rate"] >= 14 - 1e-6  # transition no shorter than two flight gaps
    st = uav.detect_stages(fit)
    assert (st.day.diff().dropna() > 3).all()
    free = uav.fit_logistic(dates, y, min_transition_days=1e-3)
    assert free["rate"] > fit["rate"]


def test_fit_logistic_limits_keep_fraction_in_range():
    rng = np.random.default_rng(5)
    dates = np.repeat(pd.date_range("2025-08-20", periods=7, freq="7D"), 8)
    t = (dates - dates[0]).days.to_numpy().astype(float)
    y = np.clip(0.9 - 0.4 * (t >= 42) + rng.normal(0, 0.05, t.size), 0, 1)  # decline only begins at the end
    fit = uav.fit_logistic(dates, y, decreasing=True, limits=(0, 1))
    end = fit["base"] + fit["amp"]
    assert 0 <= end <= 1 and 0 <= fit["base"] <= 1
    future = fit["predict"](pd.date_range("2025-10-01", periods=60))
    assert np.nanmin(future) >= -1e-9
