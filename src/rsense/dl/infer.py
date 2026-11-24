"""Tiled sliding-window inference over large georeferenced rasters, with seam blending."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..core._utils import pix_to_map
from ._torch import require_torch, torch_device
from .datasets import grid_windows


def _blend_window(size: int, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones((size, size), "float32")
    w = np.hanning(size + 2)[1:-1] if kind == "hann" else 1 - np.abs(np.linspace(-1, 1, size))
    w = np.maximum(w, 1e-3)
    return np.outer(w, w).astype("float32")


def predict_segmentation(model, image: xr.DataArray | xr.Dataset, *, tile: int = 256,
                         overlap: int = 64, batch_size: int = 8, normalize: Any = None,
                         blend: Literal["hann", "triangle", "none"] = "hann",
                         return_proba: bool = False, device: str | None = None,
                         precision: Literal["32", "16", "bf16"] = "32") -> xr.DataArray | xr.Dataset:
    """Run a segmentation model over an arbitrarily large image (band, y, x).

    Overlapping tiles are predicted in batches; logits are softmaxed and blended with a
    ``hann``/``triangle`` weight window so tile seams disappear. Returns a georeferenced
    class map (or per-class probabilities with ``return_proba``).
    """
    torch = require_torch()
    if isinstance(image, xr.Dataset):
        image = image.to_array("band")
    image = image.transpose("band", ...)
    arr = np.asarray(to_cpu(image).values, dtype="float32")
    if normalize is not None:
        mean, std = (np.asarray(v, "float32") for v in normalize)
        arr = (arr - mean[:, None, None]) / std[:, None, None]
    valid = np.all(np.isfinite(arr), 0)
    arr = np.nan_to_num(arr)
    C, H, W = arr.shape
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = arr.shape[1:]
    wins = grid_windows(Hp, Wp, tile, tile - overlap)
    weight = _blend_window(tile, blend)
    dev = torch_device(device)
    model.to(dev).eval()
    amp = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    acc = wsum = None
    with torch.no_grad():
        for i in range(0, len(wins), batch_size):
            batch = wins[i:i + batch_size]
            x = torch.from_numpy(np.stack([arr[:, r:r + tile, c:c + tile] for r, c in batch])).to(dev)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                logits = model(x)
            p = torch.softmax(logits.float(), 1).cpu().numpy()
            if acc is None:
                acc = np.zeros((p.shape[1], Hp, Wp), "float32")
                wsum = np.zeros((Hp, Wp), "float32")
            for (r, c), pi in zip(batch, p):
                acc[:, r:r + tile, c:c + tile] += pi * weight
                wsum[r:r + tile, c:c + tile] += weight
    prob = (acc / np.maximum(wsum, 1e-6))[:, :H, :W]
    ydim, xdim = image.dims[-2:]
    coords = {ydim: image[ydim], xdim: image[xdim]}
    if return_proba:
        ds = xr.Dataset({f"p_{k}": xr.DataArray(np.where(valid, prob[k], np.nan), dims=(ydim, xdim),
                                                coords=coords) for k in range(prob.shape[0])})
        return ds.rio.write_crs(image.rio.crs)
    cls = np.where(valid, prob.argmax(0), -1).astype("int16")
    return xr.DataArray(cls, dims=(ydim, xdim), coords=coords, name="class").rio.write_crs(image.rio.crs)


def predict_detection(model, image: xr.DataArray | xr.Dataset, *, tile: int = 640,
                      overlap: int = 128, score_threshold: float = 0.3, iou_threshold: float = 0.5,
                      batch_size: int = 4, normalize: Any = None, class_names: dict | None = None,
                      device: str | None = None):
    """Sliced detection over a large image with global NMS → GeoDataFrame of boxes in map units.

    Columns: ``label``, ``score`` and polygon geometry (box). Pixel boxes are offset by
    tile position, merged across tiles with class-wise NMS, then georeferenced.
    """
    torch = require_torch()
    import geopandas as gpd
    from shapely.geometry import box as sbox
    from torchvision.ops import batched_nms

    if isinstance(image, xr.Dataset):
        image = image.to_array("band")
    image = image.transpose("band", ...)
    arr = np.nan_to_num(np.asarray(to_cpu(image).values, dtype="float32"))
    if normalize is not None:
        mean, std = (np.asarray(v, "float32") for v in normalize)
        arr = (arr - mean[:, None, None]) / std[:, None, None]
    C, H, W = arr.shape
    wins = grid_windows(max(H, tile), max(W, tile), tile, tile - overlap)
    dev = torch_device(device)
    model.to(dev).eval()
    boxes, scores, labels = [], [], []
    with torch.no_grad():
        for i in range(0, len(wins), batch_size):
            batch = wins[i:i + batch_size]
            imgs = [torch.from_numpy(arr[:, r:r + tile, c:c + tile]).to(dev) for r, c in batch]
            for (r, c), out in zip(batch, model(imgs)):
                keep = out["scores"] >= score_threshold
                b = out["boxes"][keep].cpu()
                b[:, [0, 2]] += c
                b[:, [1, 3]] += r
                boxes.append(b)
                scores.append(out["scores"][keep].cpu())
                labels.append(out["labels"][keep].cpu())
    if not boxes or sum(len(b) for b in boxes) == 0:
        return gpd.GeoDataFrame({"label": [], "score": []}, geometry=[], crs=image.rio.crs)
    B, S, L = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    keep = batched_nms(B, S, L, iou_threshold)
    B, S, L = B[keep].numpy(), S[keep].numpy(), L[keep].numpy()
    T = image.rio.transform()
    geoms = []
    for x0, y0, x1, y1 in B:
        X0, Y0 = pix_to_map(T, x0, y0)
        X1, Y1 = pix_to_map(T, x1, y1)
        geoms.append(sbox(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1)))
    names = [class_names.get(int(v), int(v)) if class_names else int(v) for v in L]
    return gpd.GeoDataFrame({"label": names, "score": S}, geometry=geoms, crs=image.rio.crs)


def export_onnx(model, path: str, *, in_channels: int, tile: int = 256, opset: int = 17) -> str:
    """Export a segmentation model to ONNX (dynamic batch/height/width) for CPU inference."""
    torch = require_torch()
    model.eval().cpu()
    dummy = torch.zeros(1, in_channels, tile, tile)
    torch.onnx.export(model, dummy, path, opset_version=opset, input_names=["image"],
                      output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"},
                                    "logits": {0: "batch", 2: "h", 3: "w"}})
    return path
