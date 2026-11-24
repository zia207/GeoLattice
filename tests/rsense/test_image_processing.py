"""Tests for geolattice.rsense.image_processing."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("skimage")
pytest.importorskip("rioxarray")

from geolattice.rsense import image_processing as ip  # noqa: E402


@pytest.fixture
def rgb():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (3, 40, 60)).astype("uint8")
    da = xr.DataArray(
        data, dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": 4000000.0 - 10 * (np.arange(40) + 0.5),
                "x": 500000.0 + 10 * (np.arange(60) + 0.5)},
    )
    return da.rio.write_crs("EPSG:32648")


@pytest.fixture
def blocks():
    img = np.zeros((64, 64), dtype="float32")
    img[16:48, 16:48] = 1.0
    return img


def test_lazy_registration():
    from geolattice import rsense

    assert rsense.image_processing is ip


def test_info_and_vector(rgb):
    info = ip.image_info(rgb)
    assert (info["bands"], info["height"], info["width"]) == (3, 40, 60)
    assert info["georeferenced"]
    assert ip.to_vector(rgb).shape == (2400, 3)


def test_io_roundtrip(rgb, tmp_path):
    path = ip.write_image(rgb, tmp_path / "a.tif")
    back = ip.read_image(path)
    assert back.shape == rgb.shape
    assert back.rio.crs == rgb.rio.crs
    np.testing.assert_array_equal(back.values, rgb.values)
    assert ip.read_image(path, as_dataarray=False).shape == rgb.shape


def test_read_image_window(rgb, tmp_path):
    path = ip.write_image(rgb, tmp_path / "w.tif")
    win = ip.read_image(path, window=(5, 15, 10, 30), bands=[1, 3])
    assert win.shape == (2, 10, 20)
    np.testing.assert_array_equal(win.values, rgb.values[[0, 2], 5:15, 10:30])
    assert float(win.x[0]) == float(rgb.x[10])


def test_color_conversions(rgb):
    g = ip.to_gray(rgb)
    assert g.shape == (1, 40, 60) or g.shape == (40, 60)
    hsv = ip.rgb_to_hsv(rgb)
    back = ip.hsv_to_rgb(hsv)
    np.testing.assert_allclose(back.values, rgb.values / 255.0, atol=1e-4)
    lab = ip.rgb_to_lab(rgb)
    assert 0 <= float(lab[0].min()) and float(lab[0].max()) <= 100
    assert ip.to_uint8(np.linspace(0, 1, 10).reshape(1, 2, 5)).dtype == np.uint8


def test_resize_keeps_georeferencing(rgb):
    out = ip.resize(rgb, (20, 30))
    assert out.shape == (3, 20, 30) and out.dtype == rgb.dtype
    assert out.rio.crs == rgb.rio.crs
    # extent unchanged: pixel size doubles
    assert out.rio.resolution() == pytest.approx((20.0, -20.0))
    np.testing.assert_allclose(out.rio.bounds(), rgb.rio.bounds())


def test_resize_keep_aspect_and_rescale(blocks):
    assert ip.resize(blocks, (32, None)).shape == (32, 32)
    assert ip.rescale(blocks, 0.25).shape == (16, 16)
    assert ip.rescale(blocks, 2).shape == (128, 128)
    with pytest.raises(ValueError):
        ip.resize(blocks)


def test_downscale_local_mean(blocks):
    out = ip.downscale_local_mean(blocks, (4, 4))
    assert out.shape == (16, 16)
    assert out[8, 8] == pytest.approx(1.0)


def test_rotate_shift_flip_crop(rgb, blocks):
    r = ip.rotate(blocks, 90)
    np.testing.assert_allclose(r[16:48, 16:48], 1.0, atol=1e-5)
    assert ip.rotate(blocks, 45, resize_output=True).shape[0] > 64
    s = ip.shift(blocks, dx=5, dy=0, cval=0)
    assert s[30, 21] == pytest.approx(1.0) and s[30, 20] == pytest.approx(0.0, abs=1e-6) or s[30, 20] < 1
    f = ip.flip(rgb, "horizontal")
    np.testing.assert_array_equal(f.values, rgb.values[:, :, ::-1])
    assert float(f.x[0]) == float(rgb.x[-1])
    c = ip.crop(rgb, (5, 15, 10, 30))
    assert c.shape == (3, 10, 20)
    assert float(c.x[0]) == float(rgb.x[10])
    bx = ip.crop(rgb, bounds=(500000, 3999800, 500300, 3999900))
    assert bx.shape[1] > 0


def test_affine_and_perspective(blocks):
    ident = ip.affine_transform(blocks, [[1, 0, 0], [0, 1, 0]], cval=0)
    np.testing.assert_allclose(ident, blocks, atol=1e-5)
    pts = [(0, 0), (63, 0), (63, 63), (0, 63)]
    same = ip.perspective_transform(blocks, pts, pts, order=0, cval=0)
    np.testing.assert_allclose(same, blocks, atol=1e-5)
    with pytest.raises(ValueError):
        ip.perspective_transform(blocks, pts[:3], pts[:3])


def test_radiometric(rgb):
    s = ip.stretch(rgb)
    assert float(s.min()) >= 0 and float(s.max()) <= 1
    dark = ip.adjust_gamma(rgb, 0.5)
    bright = ip.adjust_gamma(rgb, 2.0)
    assert float(dark.mean()) > float(bright.mean())
    bc = ip.adjust_brightness_contrast(rgb, 1.0, 0.2)
    assert float(bc.mean()) > float(ip.adjust_brightness_contrast(rgb).mean())
    h = ip.histogram(rgb, bins=16)
    assert h["counts"].shape == (3, 16) and h["counts"].sum() == rgb.size
    eq = ip.equalize_hist(rgb)
    assert 0 <= float(eq.min()) and float(eq.max()) <= 1
    cl = ip.equalize_adapthist(rgb, kernel_size=16)
    assert cl.shape == rgb.shape


def test_match_histograms():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 5, (2, 50, 50)).astype("float32")
    b = rng.normal(30, 2, (2, 50, 50)).astype("float32")
    m = ip.match_histograms(a, b)
    assert abs(m.mean() - 30) < 1
    assert abs(ip.match_histograms(a, b[:1]).mean() - 30) < 1


def test_threshold(blocks):
    mask, t = ip.threshold(blocks, "otsu", return_value=True)
    assert mask.dtype == np.uint8 and mask.sum() == 32 * 32 and 0 <= t <= 1
    assert ip.threshold(blocks, 0.5).sum() == 32 * 32
    assert ip.threshold(blocks, "local", block_size=11).shape == blocks.shape
    with pytest.raises(ValueError):
        ip.threshold(blocks, "nope")


def test_filters_reduce_noise(blocks):
    blocks = 0.25 + 0.5 * blocks  # mid-range so clipping at 0/1 does not bias the noise
    noisy = ip.add_noise(blocks, "gaussian", 0.1, seed=0)
    assert not np.allclose(noisy, blocks)
    for fn in (lambda x: ip.mean_blur(x, 5), lambda x: ip.gaussian_blur(x, 2),
               lambda x: ip.median_blur(x, 5), ip.bilateral_denoise, ip.tv_denoise):
        out = fn(noisy)
        assert np.abs(out - blocks).mean() < np.abs(noisy - blocks).mean()
    for mode in ("salt", "pepper", "s&p", "speckle"):
        assert ip.add_noise(blocks, mode, 0.05, seed=1).shape == blocks.shape


def test_convolve_and_sharpen(blocks):
    np.testing.assert_allclose(ip.convolve(blocks, "identity"), blocks, atol=1e-6)
    np.testing.assert_allclose(ip.convolve(blocks, np.full((3, 3), 1 / 9)),
                               ip.mean_blur(blocks, 3), atol=1e-5)
    assert ip.sharpen(blocks, 0).shape == blocks.shape
    np.testing.assert_allclose(ip.sharpen(blocks, 0), blocks, atol=1e-6)
    us = ip.unsharp_mask(ip.gaussian_blur(blocks, 2), 2, 1.0)
    assert np.abs(us - blocks).mean() < np.abs(ip.gaussian_blur(blocks, 2) - blocks).mean()


def test_nan_preserved(blocks):
    x = blocks.copy()
    x[5, 5] = np.nan
    out = ip.gaussian_blur(x, 1)
    assert np.isnan(out[5, 5]) and np.isfinite(out[6, 6])


def test_edges(blocks):
    for m in ("sobel", "roberts", "prewitt", "scharr", "laplacian"):
        e = ip.edges(blocks, m)
        assert e[32, 32] == pytest.approx(0, abs=1e-6) and e.max() > 0
    c = ip.edges(blocks, "canny", sigma=1.0)
    assert c.dtype == np.uint8 and c.sum() > 0 and c[32, 32] == 0
    assert np.abs(ip.sobel_h(blocks)).max() > 0 and np.abs(ip.sobel_v(blocks)).max() > 0
    with pytest.raises(ValueError):
        ip.edges(blocks, "nope")


def test_morphology(blocks):
    er = ip.morphology(blocks, "erode", 3)
    di = ip.morphology(blocks, "dilate", 3)
    assert er.sum() < blocks.sum() < di.sum()
    spot = blocks.copy()
    spot[2, 2] = 1
    assert ip.morphology(spot, "open", 3)[2, 2] == 0
    for op in ("close", "gradient", "tophat", "blackhat"):
        for shape in ("square", "disk", "cross"):
            assert ip.morphology(blocks, op, 3, shape).shape == blocks.shape
    with pytest.raises(ValueError):
        ip.morphology(blocks, "bad")
    cleaned = ip.remove_small_objects(spot > 0, min_size=10)
    assert cleaned[2, 2] == 0 and cleaned[30, 30] == 1


def test_features(blocks):
    corners = ip.harris_corners(blocks)
    assert len(corners) >= 4
    hog = ip.hog_features(blocks)
    assert hog.ndim == 1 and hog.size > 0
    vec = ip.to_vector(np.where(blocks > 0, blocks, np.nan), valid_only=True)
    assert vec.shape == (32 * 32, 1)


def test_numpy_and_2d_dataarray_inputs(rgb):
    arr = rgb.values.astype("float32")
    assert isinstance(ip.gaussian_blur(arr, 1), np.ndarray)
    assert ip.gaussian_blur(arr[0], 1).shape == (40, 60)
    d2 = ip.gaussian_blur(rgb.isel(band=0, drop=True), 1)
    assert isinstance(d2, xr.DataArray) and d2.shape == (40, 60)
    with pytest.raises(ValueError):
        ip.gaussian_blur(np.zeros(5))
    with pytest.raises(ValueError):
        ip.to_gray(arr[:2])


def test_real_sentinel2():
    from pathlib import Path

    d = (Path(__file__).resolve().parents[2] / "Tutorials/notebooks/07_remote_sensing/"
         "07_08_image_classification/data/Sentinel_2_UB/Images")
    if not d.exists():
        pytest.skip("tutorial data not available")
    bands = [ip.read_image(d / f"{b}.tif") for b in ("B4", "B3", "B2")]
    img = xr.concat(bands, dim="band")
    rgb = ip.stretch(img)
    assert rgb.shape[0] == 3 and float(np.nanmax(rgb)) <= 1
    assert ip.edges(ip.to_gray(rgb), "canny").sum() > 0
    assert ip.resize(img, scale=0.5).rio.crs == img.rio.crs
