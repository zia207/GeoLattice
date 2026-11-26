"""Datasets and samplers turning georeferenced rasters into PyTorch training tiles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ._torch import require_torch


def band_stats(image: xr.DataArray, sample: int = 1_000_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std (from a random pixel sample) for normalization. ``image``: (band, y, x)."""
    a = np.asarray(to_cpu(image).values, dtype="float64").reshape(image.shape[0], -1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.shape[1], min(sample, a.shape[1]), replace=False)
    s = a[:, idx]
    return np.nanmean(s, 1).astype("float32"), np.nanstd(s, 1).astype("float32") + 1e-6


def grid_windows(height: int, width: int, size: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Top-left corners of a regular tile grid covering the image (last tiles snapped to the edge)."""
    stride = stride or size
    rows = list(range(0, max(height - size, 0) + 1, stride))
    cols = list(range(0, max(width - size, 0) + 1, stride))
    if rows[-1] + size < height:
        rows.append(height - size)
    if cols[-1] + size < width:
        cols.append(width - size)
    return [(r, c) for r in rows for c in cols]


class RasterTileDataset:
    """Tiles from an image (band, y, x) and optional label raster (y, x) as a torch Dataset.

    Parameters
    ----------
    image : DataArray (band, y, x) or Dataset (variables become bands).
    labels : class raster on the same grid (int; ``ignore_index`` for unlabeled) or None.
    size : tile edge in pixels.
    sampler : ``grid`` (regular tiles, ``stride`` overlap) or ``random`` (``length`` random
        tiles, optionally only where labels exist).
    normalize : ``(mean, std)`` arrays, ``"auto"`` (computed), or None.
    transform : callable on (image_tensor, mask_tensor) for augmentation.
    """

    def __init__(self, image: xr.DataArray | xr.Dataset, labels: xr.DataArray | None = None, *,
                 size: int = 256, sampler: Literal["grid", "random"] = "grid",
                 stride: int | None = None, length: int = 1000, normalize: Any = "auto",
                 ignore_index: int = -100, transform: Callable | None = None,
                 min_labeled: float = 0.01, seed: int = 0):
        if isinstance(image, xr.Dataset):
            image = image.to_array("band")
        self.image = np.asarray(to_cpu(image.transpose("band", ...).values), dtype="float32")
        self.labels = (np.asarray(to_cpu(labels).values).astype("int64") if labels is not None
                       else None)
        self.size, self.ignore_index, self.transform = size, ignore_index, transform
        _, H, W = self.image.shape
        if normalize == "auto":
            self.mean, self.std = band_stats(image)
        elif normalize is None:
            self.mean, self.std = None, None
        else:
            self.mean, self.std = (np.asarray(v, "float32") for v in normalize)
        rng = np.random.default_rng(seed)
        if sampler == "grid":
            self.windows = grid_windows(H, W, size, stride)
        else:
            wins = []
            tries = 0
            while len(wins) < length and tries < length * 50:
                r, c = rng.integers(0, H - size + 1), rng.integers(0, W - size + 1)
                tries += 1
                if self.labels is not None:
                    lab = self.labels[r:r + size, c:c + size]
                    if np.mean(lab != ignore_index) < min_labeled:
                        continue
                wins.append((int(r), int(c)))
            self.windows = wins

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        torch = require_torch()
        r, c = self.windows[i]
        s = self.size
        x = self.image[:, r:r + s, c:c + s].copy()
        if self.mean is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        x = np.nan_to_num(x, nan=0.0)
        sample = {"image": torch.from_numpy(x), "window": torch.tensor([r, c])}
        if self.labels is not None:
            sample["mask"] = torch.from_numpy(self.labels[r:r + s, c:c + s].copy())
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def random_flip_rotate(sample: dict) -> dict:
    """Simple geometric augmentation: random flips and 90° rotations (image and mask together)."""
    torch = require_torch()
    k = int(torch.randint(0, 4, (1,)))
    flip = bool(torch.randint(0, 2, (1,)))
    for key in ("image", "mask"):
        if key in sample:
            t = torch.rot90(sample[key], k, dims=(-2, -1))
            sample[key] = torch.flip(t, dims=(-1,)) if flip else t
    return sample


def labels_from_vectors(gdf, template: xr.DataArray, label_col: str, *, ignore_index: int = -100,
                        all_touched: bool = False) -> tuple[xr.DataArray, dict]:
    """Rasterize labelled polygons onto the image grid → (label raster, {class_name: code})."""
    import pandas as pd
    from rasterio.features import rasterize

    g = gdf.to_crs(template.rio.crs)
    codes, names = pd.factorize(g[label_col])
    ydim, xdim = template.dims[-2:]
    shape = (template.sizes[ydim], template.sizes[xdim])
    arr = rasterize(((geom, int(c)) for geom, c in zip(g.geometry, codes)), out_shape=shape,
                    transform=template.rio.transform(), fill=ignore_index, dtype="int32",
                    all_touched=all_touched)
    da = xr.DataArray(arr, dims=(ydim, xdim), coords={ydim: template[ydim], xdim: template[xdim]},
                      name="labels").rio.write_crs(template.rio.crs)
    return da, {str(n): i for i, n in enumerate(names)}


def from_torchgeo(name: str, root: str = "data", download: bool = True, **kwargs: Any):
    """Instantiate a TorchGeo benchmark dataset by class name, e.g. ``from_torchgeo("EuroSAT")``."""
    try:
        import torchgeo.datasets as tgd
    except ImportError as e:
        raise ImportError("from_torchgeo needs `pip install torchgeo`") from e
    cls = getattr(tgd, name)
    return cls(root=root, download=download, **kwargs)


def make_loader(dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 0, **kw):
    torch = require_torch()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, **kw)


