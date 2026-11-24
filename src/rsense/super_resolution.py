"""Classical and deep-learning super-resolution for multispectral imagery.

Classical interpolation and reconstruction do not recover information removed by
the sensor. Learned methods add a training-data prior and should be checked for
spectral and degradation consistency before being used for measurements.

Array inputs use ``(y, x)`` or ``(band, y, x)`` order. ``upsample`` also accepts
georeferenced xarray objects and preserves their CRS and spatial metadata.
PyTorch is imported only by the deep-learning functions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, zoom

from .dl._torch import require_torch, torch_device

_RESAMPLERS = {"nearest": 0, "bilinear": 1, "bicubic": 3}


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    return scale


def _lanczos_axis(image: np.ndarray, size: int, axis: int, radius: int = 3) -> np.ndarray:
    """Resize one axis using a normalized windowed-sinc (Lanczos) kernel."""
    old_size = image.shape[axis]
    centers = (np.arange(size, dtype="float64") + 0.5) * old_size / size - 0.5
    offsets = np.arange(-radius + 1, radius + 1)
    source = np.floor(centers).astype("int64")[:, None] + offsets[None, :]
    distance = centers[:, None] - source
    weights = np.sinc(distance) * np.sinc(distance / radius)
    weights[np.abs(distance) >= radius] = 0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo("float64").eps)
    source = np.clip(source, 0, old_size - 1)
    moved = np.moveaxis(image, axis, -1)
    sampled = np.take(moved, source, axis=-1)
    result = np.sum(sampled * weights.reshape((1,) * (sampled.ndim - 2) + weights.shape),
                    axis=-1)
    return np.moveaxis(result, -1, axis)


def _resize_array(image: np.ndarray, shape: tuple[int, int], method: str) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (y, x) or (band, y, x)")
    if method not in {*_RESAMPLERS, "lanczos"}:
        raise ValueError(f"method must be one of {', '.join([*_RESAMPLERS, 'lanczos'])}")
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("output dimensions must be positive")
    y_axis, x_axis = image.ndim - 2, image.ndim - 1
    if method == "lanczos":
        return _lanczos_axis(_lanczos_axis(image, width, x_axis), height, y_axis)
    factors = [height / image.shape[-2], width / image.shape[-1]]
    factors = ([1.0, *factors] if image.ndim == 3 else factors)
    return zoom(image, factors, order=_RESAMPLERS[method], mode="nearest",
                prefilter=method == "bicubic")


def _spatial_dims(image: xr.DataArray) -> tuple[str, str]:
    try:
        return image.rio.y_dim, image.rio.x_dim
    except (AttributeError, ValueError):
        if image.ndim >= 2:
            return image.dims[-2], image.dims[-1]
        raise ValueError("image must have two spatial dimensions")


def _resize_xarray(image: xr.DataArray, shape: tuple[int, int], method: str) -> xr.DataArray:
    ydim, xdim = _spatial_dims(image)
    ordered = image.transpose(..., ydim, xdim)
    values = _resize_array(np.asarray(ordered.values), shape, method)
    height, width = shape
    coords = {name: coord for name, coord in ordered.coords.items()
              if ydim not in coord.dims and xdim not in coord.dims}
    transform = None
    try:
        transform = image.rio.transform()
        from affine import Affine

        transform = transform * Affine.scale(image.sizes[xdim] / width,
                                               image.sizes[ydim] / height)
        xcoords = transform.c + transform.a * (np.arange(width) + 0.5)
        ycoords = transform.f + transform.e * (np.arange(height) + 0.5)
    except (AttributeError, ImportError, ValueError):
        xold = np.asarray(image[xdim].values, dtype="float64")
        yold = np.asarray(image[ydim].values, dtype="float64")
        xstep = float(np.median(np.diff(xold))) if xold.size > 1 else 1.0
        ystep = float(np.median(np.diff(yold))) if yold.size > 1 else 1.0
        xcoords = xold[0] - xstep / 2 + (np.arange(width) + 0.5) * xstep * xold.size / width
        ycoords = yold[0] - ystep / 2 + (np.arange(height) + 0.5) * ystep * yold.size / height
    coords[ydim] = ycoords
    coords[xdim] = xcoords
    result = xr.DataArray(values, dims=ordered.dims, coords=coords, name=image.name,
                          attrs=image.attrs)
    try:
        crs = image.rio.crs
        if crs is not None:
            result = result.rio.write_crs(crs)
        if transform is not None:
            result = result.rio.write_transform(transform)
        nodata = image.rio.nodata
        if nodata is not None:
            result = result.rio.write_nodata(nodata)
    except (AttributeError, ValueError):
        pass
    return result.transpose(*image.dims)


def upsample(
    image: np.ndarray | xr.DataArray | xr.Dataset,
    scale: float = 2,
    *,
    method: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bicubic",
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """Upsample a 2-D/3-D array or georeferenced xarray object.

    ``scale`` may be non-integer. Methods are interpolation baselines; they do not
    recover detail that was not observed. Dataset variables without both spatial
    dimensions are retained unchanged.
    """
    scale = _validate_scale(scale)
    if isinstance(image, xr.Dataset):
        resized = {
            name: upsample(value, scale, method=method)
            if value.ndim >= 2 and _spatial_dims(value) == (image.rio.y_dim, image.rio.x_dim)
            else value
            for name, value in image.data_vars.items()
        }
        result = xr.Dataset(resized, attrs=image.attrs)
        for key in ("spatial_ref",):
            if key in image:
                result[key] = image[key]
        return result
    if isinstance(image, xr.DataArray):
        ydim, xdim = _spatial_dims(image)
        shape = (max(1, round(image.sizes[ydim] * scale)),
                 max(1, round(image.sizes[xdim] * scale)))
        return _resize_xarray(image, shape, method)
    arr = np.asarray(image)
    shape = (max(1, round(arr.shape[-2] * scale)),
             max(1, round(arr.shape[-1] * scale)))
    return _resize_array(arr, shape, method)


def unsharp_mask(
    image: np.ndarray,
    *,
    amount: float = 1.0,
    sigma: float = 1.0,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply classical unsharp masking to a ``(y, x)`` or ``(band, y, x)`` array."""
    arr = np.asarray(image, dtype="float32")
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (y, x) or (band, y, x)")
    if not np.isfinite(amount) or not np.isfinite(sigma) or amount < 0 or sigma <= 0:
        raise ValueError("amount must be non-negative and sigma must be positive")
    axes_sigma = (sigma, sigma) if arr.ndim == 2 else (0, sigma, sigma)
    smooth = gaussian_filter(arr, sigma=axes_sigma)
    result = arr + amount * (arr - smooth)
    if clip is not None:
        if clip[0] >= clip[1]:
            raise ValueError("clip must be an increasing (minimum, maximum) pair")
        result = np.clip(result, *clip)
    return result


def iterative_back_projection(
    low_res: np.ndarray,
    *,
    scale: int = 2,
    iterations: int = 20,
    blur_sigma: float = 1.0,
    step: float = 0.5,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """Classical iterative back-projection with a Gaussian blur/block-mean sensor model.

    The reconstruction is constrained to reproduce the input after the specified
    degradation. Its high-frequency detail is determined by the initialization and
    forward model, not recovered uniquely from the coarse observation.
    """
    observed = np.asarray(low_res, dtype="float32")
    if observed.ndim not in (2, 3):
        raise ValueError("low_res must have shape (y, x) or (band, y, x)")
    if not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if (not isinstance(iterations, (int, np.integer)) or iterations < 0
            or not np.isfinite(blur_sigma) or blur_sigma < 0 or not 0 < step <= 1):
        raise ValueError("iterations, blur_sigma, or step is out of range")
    if not np.isfinite(observed).all():
        raise ValueError("low_res must contain only finite values")
    height, width = observed.shape[-2:]
    current = _resize_array(observed, (height * scale, width * scale), "bicubic")
    sigma = (0, blur_sigma, blur_sigma) if observed.ndim == 3 else blur_sigma
    for _ in range(iterations):
        blurred = gaussian_filter(current, sigma=sigma)
        if observed.ndim == 3:
            predicted = blurred.reshape(observed.shape[0], height, scale, width, scale).mean(
                axis=(2, 4))
        else:
            predicted = blurred.reshape(height, scale, width, scale).mean(axis=(1, 3))
        residual = observed - predicted
        current += step * _resize_array(
            residual, (height * scale, width * scale), "bilinear")
        if clip is not None:
            current = np.clip(current, *clip)
    return current


def pansharpen(
    multispectral: np.ndarray,
    panchromatic: np.ndarray,
    *,
    method: Literal["highpass", "intensity", "brovey"] = "highpass",
    gain: float = 0.25,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse a coarse multispectral cube with a higher-resolution pan band.

    Returns ``(band, y, x)`` at the panchromatic grid. ``highpass`` injects
    normalized pan detail, ``intensity`` performs component substitution, and
    ``brovey`` scales bands by pan-to-intensity ratio. Spectral distortion varies
    by method; validate the result against Wald-style reduced-resolution tests.
    """
    ms = np.asarray(multispectral, dtype="float32")
    pan = np.asarray(panchromatic, dtype="float32")
    if ms.ndim != 3 or pan.ndim != 2 or ms.shape[0] < 1:
        raise ValueError("multispectral must be (band, y, x) and panchromatic must be (y, x)")
    if not np.isfinite(ms).all() or not np.isfinite(pan).all():
        raise ValueError("inputs must contain only finite values")
    if method not in {"highpass", "intensity", "brovey"}:
        raise ValueError("method must be 'highpass', 'intensity', or 'brovey'")
    if gain < 0:
        raise ValueError("gain must be non-negative")
    up = _resize_array(ms, pan.shape, "bicubic")
    if weights is None:
        w = np.full(ms.shape[0], 1 / ms.shape[0], dtype="float32")
    else:
        w = np.asarray(weights, dtype="float32")
        if w.shape != (ms.shape[0],) or not np.isfinite(w).all() or np.sum(w) == 0:
            raise ValueError("weights must be one finite, non-zero-sum value per band")
        w = w / w.sum()
    intensity = np.einsum("b,byx->yx", w, up)
    if method == "brovey":
        ratio = pan / np.maximum(intensity, 1e-6)
        return up * ratio[None, ...]
    pan_mean = float(np.mean(pan))
    pan_std = float(np.std(pan))
    if method == "intensity":
        matched = (pan - pan_mean) * (float(np.std(intensity)) /
                                      max(pan_std, 1e-6)) + float(np.mean(intensity))
        detail = matched - intensity
    else:
        detail = pan - gaussian_filter(pan, sigma=1.5)
        detail *= float(np.std(intensity)) / max(pan_std, 1e-6)
    return up + gain * detail[None, ...]


def build_sr_model(
    in_channels: int,
    *,
    out_channels: int | None = None,
    scale: float = 2,
    features: int = 64,
    blocks: int = 8,
    architecture: Literal["srcnn", "vdsr", "edsr", "rcan"] = "edsr",
):
    """Build an SRCNN-, VDSR-, EDSR-, or RCAN-style multispectral CNN.

    The returned model needs PyTorch; use ``pip install geolattice[rsense-dl]``.
    The networks use an arbitrary-scale bicubic skip input and can be trained on
    synthetic degradation pairs or co-registered cross-sensor pairs. Adversarial,
    perceptual, transformer, and diffusion models can be supplied directly to
    :func:`train_super_resolution` and :func:`predict_super_resolution`.
    """
    torch = require_torch()
    scale = _validate_scale(scale)
    out_channels = in_channels if out_channels is None else out_channels
    if in_channels < 1 or out_channels < 1 or features < 1 or blocks < 0:
        raise ValueError("channel counts and features must be positive; blocks cannot be negative")
    if out_channels != in_channels:
        raise ValueError("satellite super-resolution must preserve the input band count")
    if architecture not in {"srcnn", "vdsr", "edsr", "rcan"}:
        raise ValueError("architecture must be 'srcnn', 'vdsr', 'edsr', or 'rcan'")

    class ResidualBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(features, features, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(features, features, 3, padding=1),
            )
            self.attention = (
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Conv2d(features, max(1, features // 16), 1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(max(1, features // 16), features, 1),
                    torch.nn.Sigmoid(),
                )
                if architecture == "rcan" else None
            )

        def forward(self, x):
            residual = self.layers(x)
            if self.attention is not None:
                residual = residual * self.attention(residual)
            return x + 0.1 * residual

    class SuperResolutionNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.architecture = architecture
            self.head = torch.nn.Conv2d(in_channels, features, 3, padding=1)
            if architecture == "srcnn":
                self.body = torch.nn.Sequential(
                    torch.nn.Conv2d(features, features, 5, padding=2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(features, features, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
            elif architecture == "vdsr":
                self.body = torch.nn.Sequential(*[
                    layer
                    for _ in range(max(1, blocks))
                    for layer in (
                        torch.nn.Conv2d(features, features, 3, padding=1),
                        torch.nn.ReLU(inplace=True),
                    )
                ])
            else:
                self.body = torch.nn.Sequential(*(ResidualBlock() for _ in range(blocks)))
            self.tail = torch.nn.Conv2d(features, out_channels, 3, padding=1)

        def forward(self, x):
            size = (max(1, round(x.shape[-2] * self.scale)),
                    max(1, round(x.shape[-1] * self.scale)))
            base = torch.nn.functional.interpolate(
                x, size=size, mode="bicubic", align_corners=False)
            residual = self.tail(self.body(self.head(base)))
            return base + residual

    return SuperResolutionNet()


def super_resolution_loss(
    prediction,
    target,
    *,
    low_res=None,
    sam_weight: float = 0.1,
    consistency_weight: float = 0.1,
):
    """L1 + spectral-angle + optional area-downsample consistency loss.

    ``prediction`` and ``target`` are ``(batch, band, y, x)`` torch tensors.
    The optional consistency term asks the estimate, reduced to the observed grid,
    to match ``low_res``. This term does not by itself identify a unique fine image.
    """
    torch = require_torch()
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching (batch, band, y, x) shapes")
    if sam_weight < 0 or consistency_weight < 0:
        raise ValueError("loss weights must be non-negative")
    loss = torch.nn.functional.l1_loss(prediction, target)
    dot = (prediction * target).sum(dim=1)
    norms = torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1)
    angle = torch.acos(torch.clamp(dot / norms.clamp_min(1e-8), -1 + 1e-7, 1 - 1e-7))
    spectral = angle.mean()
    loss = loss + sam_weight * spectral
    if low_res is not None:
        if low_res.ndim != 4 or low_res.shape[:2] != prediction.shape[:2]:
            raise ValueError("low_res must be (batch, band, y, x) with matching batch and bands")
        degraded = torch.nn.functional.interpolate(
            prediction, size=low_res.shape[-2:], mode="area")
        loss = loss + consistency_weight * torch.nn.functional.l1_loss(degraded, low_res)
    return loss


def train_super_resolution(
    model,
    train_loader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str | None = None,
    sam_weight: float = 0.1,
    consistency_weight: float = 0.1,
) -> dict[str, list[float]]:
    """Train on ``(low_res, high_res)`` batches or dicts with those named tensors."""
    torch = require_torch()
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive")
    dev = torch_device(device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": []}
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            if isinstance(batch, dict):
                low_res = batch.get("low_res", batch.get("image"))
                high_res = batch.get("high_res", batch.get("target"))
            else:
                low_res, high_res = batch[:2]
            if low_res is None or high_res is None:
                raise ValueError("each batch must provide low_res/image and high_res/target")
            low_res, high_res = low_res.to(dev).float(), high_res.to(dev).float()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(low_res)
            loss = super_resolution_loss(
                prediction, high_res, low_res=low_res, sam_weight=sam_weight,
                consistency_weight=consistency_weight)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * low_res.shape[0]
            count += low_res.shape[0]
        history["train_loss"].append(total / max(count, 1))
    return history


def build_sr_gan(
    in_channels: int,
    *,
    scale: float = 2,
    features: int = 64,
    blocks: int = 8,
    architecture: Literal["srcnn", "vdsr", "edsr", "rcan"] = "edsr",
):
    """Build an SRGAN pair: a spectral-preserving generator and patch discriminator.

    Returns ``(generator, discriminator)``. The adversarial objective can create
    plausible texture that is not supported by the observation; use modest
    adversarial weight and report the product as model-generated.
    """
    torch = require_torch()
    generator = build_sr_model(in_channels, scale=scale, features=features, blocks=blocks,
                               architecture=architecture)
    discriminator = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, features, 3, padding=1),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(features, features, 4, stride=2, padding=1),
        torch.nn.BatchNorm2d(features),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(features, features * 2, 3, padding=1),
        torch.nn.BatchNorm2d(features * 2),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(features * 2, features * 2, 4, stride=2, padding=1),
        torch.nn.BatchNorm2d(features * 2),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(features * 2, 1, 3, padding=1),
    )
    return generator, discriminator


def train_super_resolution_gan(
    generator,
    discriminator,
    train_loader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    adversarial_weight: float = 1e-3,
    sam_weight: float = 0.1,
    consistency_weight: float = 0.1,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Train an SRGAN pair on ``(low_res, high_res)`` batches.

    The generator loss combines L1, spectral angle, degradation consistency and
    least-squares adversarial loss. ``adversarial_weight`` defaults low because
    visual sharpness can come at the cost of reflectance fidelity.
    """
    torch = require_torch()
    if epochs < 1 or lr <= 0 or adversarial_weight < 0:
        raise ValueError("epochs and lr must be positive; adversarial_weight cannot be negative")
    dev = torch_device(device)
    generator.to(dev)
    discriminator.to(dev)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    history = {"generator_loss": [], "discriminator_loss": []}
    for _ in range(epochs):
        generator.train()
        discriminator.train()
        gen_total = disc_total = 0.0
        count = 0
        for batch in train_loader:
            if isinstance(batch, dict):
                low_res = batch.get("low_res", batch.get("image"))
                high_res = batch.get("high_res", batch.get("target"))
            else:
                low_res, high_res = batch[:2]
            if low_res is None or high_res is None:
                raise ValueError("each batch must provide low_res/image and high_res/target")
            low_res, high_res = low_res.to(dev).float(), high_res.to(dev).float()
            generated = generator(low_res)

            disc_optimizer.zero_grad(set_to_none=True)
            real_score = discriminator(high_res)
            fake_score = discriminator(generated.detach())
            disc_loss = 0.5 * (
                torch.nn.functional.mse_loss(real_score, torch.ones_like(real_score))
                + torch.nn.functional.mse_loss(fake_score, torch.zeros_like(fake_score))
            )
            disc_loss.backward()
            disc_optimizer.step()

            gen_optimizer.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            gen_loss = super_resolution_loss(
                generated, high_res, low_res=low_res, sam_weight=sam_weight,
                consistency_weight=consistency_weight)
            fake_score = discriminator(generated)
            gen_loss = gen_loss + adversarial_weight * torch.nn.functional.mse_loss(
                fake_score, torch.ones_like(fake_score))
            gen_loss.backward()
            gen_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            n = low_res.shape[0]
            gen_total += float(gen_loss.detach()) * n
            disc_total += float(disc_loss.detach()) * n
            count += n
        history["generator_loss"].append(gen_total / max(count, 1))
        history["discriminator_loss"].append(disc_total / max(count, 1))
    return history


def build_diffusion_model(
    in_channels: int,
    *,
    scale: float = 2,
    features: int = 64,
    timesteps: int = 100,
):
    """Build a conditional DDPM for super-resolution.

    The model learns to denoise a high-resolution reflectance cube conditioned
    on its bicubic-upsampled low-resolution observation. Train it with
    :func:`train_super_resolution_diffusion`; use ``model.sample(low_res)`` for
    stochastic inference. Sampling adds a learned prior, not measured detail.
    """
    torch = require_torch()
    scale = _validate_scale(scale)
    if in_channels < 1 or features < 4 or timesteps < 2:
        raise ValueError("in_channels and features must be positive; timesteps must be at least 2")
    betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Conv2d(in_channels * 2 + 1, features, 3, padding=1)
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(features, features, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(features, features, 3, padding=1),
                torch.nn.SiLU(),
            )
            self.output = torch.nn.Conv2d(features, in_channels, 3, padding=1)
            self.scale = scale
            self.in_channels = in_channels
            self.timesteps = timesteps
            self.register_buffer("betas", betas)
            alphas = 1.0 - betas
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

        def forward(self, noisy, timestep, condition):
            t = timestep.to(noisy.dtype).view(-1, 1, 1, 1)
            t = t.expand(-1, 1, noisy.shape[-2], noisy.shape[-1]) / max(self.timesteps - 1, 1)
            return self.output(self.down(self.input(torch.cat([noisy, condition, t], dim=1))))

        @torch.no_grad()
        def sample(self, low_res, *, steps: int | None = None, clip: bool = True):
            self.eval()
            steps = self.timesteps if steps is None else steps
            if not isinstance(steps, int) or not 1 <= steps <= self.timesteps:
                raise ValueError("steps must be between 1 and timesteps")
            low_res = low_res.to(self.betas.device)
            size = (max(1, round(low_res.shape[-2] * self.scale)),
                    max(1, round(low_res.shape[-1] * self.scale)))
            condition = torch.nn.functional.interpolate(
                low_res, size=size, mode="bicubic", align_corners=False)
            x = torch.randn_like(condition)
            schedule = torch.linspace(self.timesteps - 1, 0, steps, device=x.device).long()
            for i, t in enumerate(schedule):
                ts = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
                alpha_bar = self.alpha_bars[t]
                predicted_noise = self(x, ts, condition)
                clean_estimate = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(
                    alpha_bar)
                previous = (self.alpha_bars[schedule[i + 1]]
                            if i + 1 < len(schedule) else torch.ones_like(alpha_bar))
                x = torch.sqrt(previous) * clean_estimate + torch.sqrt(1 - previous) * predicted_noise
            if clip:
                x = x.clamp(0, 1)
            return x

    return Denoiser()


def train_super_resolution_diffusion(
    model,
    train_loader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Train the conditional DDPM on ``(low_res, high_res)`` image pairs."""
    torch = require_torch()
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive")
    dev = torch_device(device)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": []}
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            if isinstance(batch, dict):
                low_res = batch.get("low_res", batch.get("image"))
                high_res = batch.get("high_res", batch.get("target"))
            else:
                low_res, high_res = batch[:2]
            if low_res is None or high_res is None:
                raise ValueError("each batch must provide low_res/image and high_res/target")
            low_res, high_res = low_res.to(dev).float(), high_res.to(dev).float()
            condition = torch.nn.functional.interpolate(
                low_res, size=high_res.shape[-2:], mode="bicubic", align_corners=False)
            timestep = torch.randint(model.timesteps, (high_res.shape[0],), device=dev)
            noise = torch.randn_like(high_res)
            alpha_bar = model.alpha_bars[timestep].view(-1, 1, 1, 1)
            noisy = alpha_bar.sqrt() * high_res + (1 - alpha_bar).sqrt() * noise
            predicted_noise = model(noisy, timestep, condition)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = low_res.shape[0]
            total += float(loss.detach()) * n
            count += n
        history["train_loss"].append(total / max(count, 1))
    return history


def predict_super_resolution(
    model,
    image: np.ndarray | xr.DataArray,
    *,
    tile: int = 128,
    overlap: int = 16,
    batch_size: int = 4,
    device: str | None = None,
) -> np.ndarray | xr.DataArray:
    """Run tiled, overlap-blended inference, returning data on the finer grid.

    The model must expose a numeric ``scale`` attribute (as models from
    :func:`build_sr_model` do). For xarray input the output retains CRS and spatial
    coordinates; tiles avoid loading a full-scene feature tensor onto the device.
    """
    torch = require_torch()
    scale = _validate_scale(getattr(model, "scale", 0))
    if tile < 1 or not 0 <= overlap < tile or batch_size < 1:
        raise ValueError("tile and batch_size must be positive; overlap must be in [0, tile)")
    is_xarray = isinstance(image, xr.DataArray)
    template = upsample(image, scale) if is_xarray else None
    if is_xarray:
        ydim, xdim = _spatial_dims(image)
        ordered = image.transpose(..., ydim, xdim)
        arr = np.asarray(ordered.values, dtype="float32")
    else:
        arr = np.asarray(image, dtype="float32")
    squeeze_band = arr.ndim == 2
    if squeeze_band:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("image must have shape (y, x) or (band, y, x)")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    bands, height, width = arr.shape
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    padded_h, padded_w = arr.shape[-2:]
    stride = tile - overlap
    rows = list(range(0, max(padded_h - tile, 0) + 1, stride))
    cols = list(range(0, max(padded_w - tile, 0) + 1, stride))
    if not rows or rows[-1] + tile < padded_h:
        rows.append(max(0, padded_h - tile))
    if not cols or cols[-1] + tile < padded_w:
        cols.append(max(0, padded_w - tile))
    windows = [(row, col) for row in rows for col in cols]
    device_obj = torch_device(device)
    model.to(device_obj).eval()
    output_h, output_w = round(padded_h * scale), round(padded_w * scale)
    accumulated = weights_sum = None
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start:start + batch_size]
            inputs = torch.from_numpy(np.stack([
                arr[:, r:r + tile, c:c + tile] for r, c in batch_windows
            ])).to(device_obj)
            predictions = model(inputs).float().cpu().numpy()
            if accumulated is None:
                out_bands = predictions.shape[1]
                accumulated = np.zeros((out_bands, output_h, output_w), dtype="float32")
                weights_sum = np.zeros((output_h, output_w), dtype="float32")
            for (row, col), prediction in zip(batch_windows, predictions):
                r, c = round(row * scale), round(col * scale)
                h, w = prediction.shape[-2:]
                wy = np.hanning(h + 2)[1:-1] if h > 2 else np.ones(h)
                wx = np.hanning(w + 2)[1:-1] if w > 2 else np.ones(w)
                weight = np.maximum(np.outer(wy, wx), 1e-3).astype("float32")
                accumulated[:, r:r + h, c:c + w] += prediction * weight
                weights_sum[r:r + h, c:c + w] += weight
    result = accumulated / np.maximum(weights_sum, 1e-6)
    result = result[:, :round(height * scale), :round(width * scale)]
    if is_xarray:
        values = result[0] if squeeze_band else result
        return template.copy(data=values).transpose(*template.dims)
    return result[0] if squeeze_band else result


__all__ = [
    "build_sr_model",
    "build_sr_gan",
    "build_diffusion_model",
    "iterative_back_projection",
    "pansharpen",
    "predict_super_resolution",
    "super_resolution_loss",
    "train_super_resolution",
    "train_super_resolution_diffusion",
    "train_super_resolution_gan",
    "unsharp_mask",
    "upsample",
]
