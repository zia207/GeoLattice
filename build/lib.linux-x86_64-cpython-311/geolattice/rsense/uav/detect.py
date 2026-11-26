"""Object detection (panicles, heads, fruit, weeds) on large UAV images with sliding windows.

A detector trained on ~1000 px chips misses small objects when a whole orthomosaic is
squeezed into one input. :func:`tiled_detect` runs the detector on overlapping windows,
moves boxes back to image coordinates and removes duplicates with cross-tile NMS (LUMS
panicle defaults: 1024 px windows, stride 512, confidence 0.7, IoU 0.4).

Backends: ``ultralytics`` (YOLOv5u/v8/v11 ``.pt``), ``yolov5`` (original YOLOv5 checkpoints
such as the LUMS rice-panicle model, loaded through ``torch.hub``), or any callable.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

BOX_COLUMNS = ["x1", "y1", "x2", "y2", "confidence", "class"]


def load_detector(
    weights: str | PathLike,
    *,
    backend: Literal["ultralytics", "yolov5"] = "ultralytics",
    device: str | None = None,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return ``predict(tile_rgb_uint8, conf) -> (n, 6) [x1, y1, x2, y2, conf, cls]``."""
    if backend == "ultralytics":
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("backend='ultralytics' needs `pip install ultralytics`") from e
        model = YOLO(str(weights))

        def predict(tile: np.ndarray, conf: float) -> np.ndarray:
            r = model.predict(tile, conf=conf, verbose=False, device=device)[0].boxes
            if r is None or len(r) == 0:
                return np.zeros((0, 6), np.float32)
            return np.column_stack(
                [r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy()]
            ).astype(np.float32)

        return predict
    if backend == "yolov5":
        import torch

        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights), verbose=False)
        if device:
            model.to(device)

        def predict(tile: np.ndarray, conf: float) -> np.ndarray:
            model.conf = conf
            return model(tile).xyxy[0].cpu().numpy().astype(np.float32)

        return predict
    raise ValueError(f"unknown backend {backend!r}")


def detection_windows(
    height: int, width: int, window: int = 1024, stride: int = 512
) -> list[tuple[int, int, int, int]]:
    """Windows ``(row0, col0, rows, cols)`` with ``stride``; the last row/column of windows is
    moved back to the image edge so borders are covered (the original script skipped them)."""
    from .tiling import sliding_windows

    return list(sliding_windows(height, width, window, max(0, window - stride)))


def tiled_detect(
    image: Any,
    detector: Any,
    *,
    window: int = 1024,
    stride: int = 512,
    conf: float = 0.7,
    iou: float = 0.4,
    edge_margin: int = 2,
    backend: Literal["ultralytics", "yolov5"] = "ultralytics",
    device: str | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Detect objects on a large RGB image window by window and merge the results.

    Parameters
    ----------
    image : ``(band, y, x)`` DataArray (:func:`geolattice.rsense.uav.read_frame`) or an
        ``(rows, cols, 3)`` array.
    detector : weights path (loaded with ``backend``) or a callable ``(tile, conf) -> (n, 6)``.
    window, stride : window edge and step in pixels (stride = window/2 gives 50 % overlap).
    conf, iou : confidence threshold and cross-window NMS IoU threshold.
    edge_margin : drop boxes within this many pixels of an interior window edge before NMS.
        Such boxes are objects cut by the seam; with overlapping windows the whole object
        is found in a neighbouring window, and the partial box would otherwise survive NMS
        (low IoU with the full box) and inflate the count. Set 0 to keep them, e.g. when
        objects are larger than the overlap (``window - stride``).

    Returns a DataFrame with ``x1, y1, x2, y2`` (pixels), ``confidence``, ``class``,
    ``cx, cy`` and, when the image has coordinates, box centres ``x, y`` in image units.
    """
    from .io import to_rgb_array
    from .tiling import merge_boxes

    predict = (
        detector if callable(detector) else load_detector(detector, backend=backend, device=device)
    )
    rgb = to_rgb_array(image, uint8=True)
    H, W = rgb.shape[:2]
    wins = detection_windows(H, W, window, stride)
    if progress:
        from tqdm.auto import tqdm

        wins = tqdm(wins, desc="detect")
    parts = []
    for r0, c0, h, w in wins:
        b = np.asarray(
            predict(np.ascontiguousarray(rgb[r0 : r0 + h, c0 : c0 + w]), conf), np.float32
        )
        if b.size == 0:
            continue
        b = b.reshape(-1, 6)
        if edge_margin:
            m = edge_margin
            cut = (
                ((b[:, 0] < m) & (c0 > 0))
                | ((b[:, 1] < m) & (r0 > 0))
                | ((b[:, 2] > w - m) & (c0 + w < W))
                | ((b[:, 3] > h - m) & (r0 + h < H))
            )
            b = b[~cut]
        b[:, [0, 2]] += c0
        b[:, [1, 3]] += r0
        parts.append(b)
    df = pd.DataFrame(np.vstack(parts) if parts else np.zeros((0, 6)), columns=BOX_COLUMNS)
    df["class"] = df["class"].astype(int)
    df = df[df["confidence"] >= conf]
    df = merge_boxes(df, iou=iou)
    df["cx"] = (df.x1 + df.x2) / 2
    df["cy"] = (df.y1 + df.y2) / 2
    if isinstance(image, xr.DataArray) and len(df):
        df["x"] = np.interp(df.cx - 0.5, np.arange(W), image.x.values)
        df["y"] = np.interp(df.cy - 0.5, np.arange(H), image.y.values)
    return df


def boxes_to_gdf(boxes: pd.DataFrame, image: xr.DataArray):
    """Box polygons in the image's coordinates (CRS kept for GeoTIFFs)."""
    import geopandas as gpd
    from shapely.geometry import box

    xs, ys = image.x.values, image.y.values
    rx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    ry = float(ys[1] - ys[0]) if ys.size > 1 else -1.0
    x0, y0 = xs[0] - rx / 2, ys[0] - ry / 2
    geoms = [
        box(x0 + r.x1 * rx, y0 + r.y2 * ry, x0 + r.x2 * rx, y0 + r.y1 * ry)
        for r in boxes.itertuples()
    ]
    return gpd.GeoDataFrame(boxes.copy(), geometry=geoms, crs=image.rio.crs)


def count_summary(
    boxes: pd.DataFrame, image: xr.DataArray | None = None, *, area_m2: float | None = None
) -> dict[str, float]:
    """Count, mean confidence and (with a GSD/CRS image or ``area_m2``) objects per m²."""
    out = {
        "count": int(len(boxes)),
        "mean_confidence": float(boxes["confidence"].mean()) if len(boxes) else np.nan,
    }
    if area_m2 is None and image is not None and image.attrs.get("units_xy") != "px":
        res = abs(float(image.x[1] - image.x[0]))
        area_m2 = image.sizes["x"] * image.sizes["y"] * res * res
    if area_m2:
        out["area_m2"] = float(area_m2)
        out["per_m2"] = out["count"] / float(area_m2)
    return out


def detection_scores(pred: pd.DataFrame, truth: pd.DataFrame, iou: float = 0.5) -> dict[str, float]:
    """Precision, recall, F1 and count agreement of predicted vs reference boxes (greedy
    one-to-one matching at ``iou``)."""
    from .tiling import box_iou

    p = pred.sort_values("confidence", ascending=False) if "confidence" in pred else pred
    tb = truth[["x1", "y1", "x2", "y2"]].to_numpy(float)
    used = np.zeros(len(tb), bool)
    tp = 0
    for b in p[["x1", "y1", "x2", "y2"]].to_numpy(float):
        if not len(tb):
            break
        ious = np.where(used, -1, box_iou(b, tb))
        j = int(np.argmax(ious))
        if ious[j] >= iou:
            used[j] = True
            tp += 1
    fp, fn = len(p) - tp, len(tb) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    agree = 1 - abs(len(p) - len(tb)) / len(tb) if len(tb) else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "count_agreement": agree,
    }
