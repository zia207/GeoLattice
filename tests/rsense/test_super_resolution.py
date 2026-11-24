"""Tests for classical and deep-learning super-resolution."""

import numpy as np
import pytest

from geolattice.rsense import super_resolution as sr


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "lanczos"])
def test_upsample_array_shapes_and_constant_values(method):
    image = np.full((2, 5, 7), 0.25, dtype="float32")
    result = sr.upsample(image, 2, method=method)
    assert result.shape == (2, 10, 14)
    np.testing.assert_allclose(result, 0.25, atol=1e-6)


def test_upsample_preserves_georeferencing(reflectance):
    pytest.importorskip("torch")
    image = reflectance.isel(time=0)["red"]
    result = sr.upsample(image, 2, method="bilinear")
    assert result.shape == (image.shape[0] * 2, image.shape[1] * 2)
    assert result.rio.crs == image.rio.crs
    assert result.rio.resolution() == pytest.approx(
        (image.rio.resolution()[0] / 2, image.rio.resolution()[1] / 2))
    model = sr.build_sr_model(1, scale=2, features=4, blocks=1)
    predicted = sr.predict_super_resolution(model, image.expand_dims("band"), tile=32, overlap=8)
    assert predicted.shape == result.expand_dims("band").shape
    assert predicted.rio.crs == image.rio.crs


def test_back_projection_and_pan_sharpening_shapes():
    low = np.full((3, 8, 10), 0.2, dtype="float32")
    reconstructed = sr.iterative_back_projection(low, scale=2, iterations=2, clip=(0, 1))
    assert reconstructed.shape == (3, 16, 20)
    np.testing.assert_allclose(reconstructed, 0.2, atol=1e-5)
    pan = np.full((16, 20), 0.3, dtype="float32")
    for method in ("highpass", "intensity", "brovey"):
        fused = sr.pansharpen(low, pan, method=method)
        assert fused.shape == (3, 16, 20)
        assert np.isfinite(fused).all()


def test_invalid_inputs_are_reported():
    with pytest.raises(ValueError, match="positive"):
        sr.upsample(np.ones((4, 4)), 0)
    with pytest.raises(ValueError, match="finite"):
        sr.iterative_back_projection(np.array([[np.nan]]))


def test_deep_learning_model_inference_and_loss():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    model = sr.build_sr_model(2, scale=2, features=8, blocks=1)
    low = torch.rand(1, 2, 8, 10)
    high = torch.rand(1, 2, 16, 20)
    prediction = model(low)
    assert prediction.shape == high.shape
    loss = sr.super_resolution_loss(prediction, high, low_res=low)
    assert torch.isfinite(loss)
    history = sr.train_super_resolution(
        model, DataLoader(TensorDataset(low, high), batch_size=1), epochs=1)
    assert len(history["train_loss"]) == 1
    estimate = sr.predict_super_resolution(model, low[0].numpy(), tile=8, overlap=2)
    assert estimate.shape == (2, 16, 20)


@pytest.mark.parametrize("architecture", ["srcnn", "vdsr", "edsr", "rcan"])
def test_supported_sr_architectures(architecture):
    torch = pytest.importorskip("torch")
    model = sr.build_sr_model(3, scale=1.5, features=8, blocks=2,
                              architecture=architecture)
    result = model(torch.rand(1, 3, 8, 10))
    assert result.shape == (1, 3, 12, 15)


def test_gan_training():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    low = torch.rand(2, 2, 16, 16)
    high = torch.rand(2, 2, 32, 32)
    generator, discriminator = sr.build_sr_gan(2, features=8, blocks=1)
    history = sr.train_super_resolution_gan(
        generator, discriminator, DataLoader(TensorDataset(low, high), batch_size=2), epochs=1)
    assert len(history["generator_loss"]) == len(history["discriminator_loss"]) == 1
    assert np.isfinite(history["generator_loss"][0])
    assert np.isfinite(history["discriminator_loss"][0])


def test_conditional_diffusion_training_and_sampling():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    low = torch.rand(2, 2, 8, 8)
    high = torch.rand(2, 2, 16, 16)
    model = sr.build_diffusion_model(2, scale=2, features=8, timesteps=4)
    history = sr.train_super_resolution_diffusion(
        model, DataLoader(TensorDataset(low, high), batch_size=2), epochs=1)
    sample = model.sample(low, steps=2)
    assert len(history["train_loss"]) == 1
    assert np.isfinite(history["train_loss"][0])
    assert sample.shape == high.shape
    assert torch.isfinite(sample).all()
