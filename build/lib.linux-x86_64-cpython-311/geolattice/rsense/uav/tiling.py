"""Overlapping tiles for very large UAV orthomosaics and merging of per-tile results.

Deep models (density counting, SAM, YOLO) see one fixed-size tile at a time. These helpers
cut an image into overlapping tiles that always reach the far edges, blend per-tile
rasters back with a raised-cosine weight (so tile seams disappear), and merge duplicate
point or box detections found twice in the overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
import pandas as pd


def sliding_windows(
    height: int,
    width: int,
    tile: int = 1024,
    overlap: int = 128,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(row0, col0, rows, cols)`` windows of size ``tile`` with ``overlap`` pixels.

    The last window in each direction is shifted back so it ends exactly on the image edge,
    so every pixel is covered and no window is smaller than ``tile`` unless the image is.
    """
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("need tile > 0 and 0 <= overlap < tile")
    step = tile - overlap

    def starts(n: int) -> list[int]:
        s = list(range(0, max(1, n - tile + 1), step))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s

    for r0 in starts(height):
        for c0 in starts(width):
            yield r0, c0, min(tile, height - r0), min(tile, width - c0)


def cosine_weight(rows: int, cols: int, floor: float = 1e-3) -> np.ndarray:
    """2-D raised-cosine (Hann) weight peaking at 1 in the tile centre.

    ``floor`` keeps edge pixels slightly above zero so image borders, which only one tile
    covers, still get a value. (The LUMS ``predict_orthomosaic.py`` scripts evaluate the
    cosine over 0..pi, which gives a ramp peaking at the bottom-right corner rather than a
    centred window; this uses the full 0..2pi period.)
    """
    wy = (1 - np.cos(np.linspace(0, 2 * np.pi, rows, dtype=np.float32))) / 2
    wx = (1 - np.cos(np.linspace(0, 2 * np.pi, cols, dtype=np.float32))) / 2
    w = np.outer(wy, wx)
    w /= w.max() if w.max() > 0 else 1.0
    return np.maximum(w, floor).astype(np.float32)


def predict_tiled(
    image: Any,
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    tile: int = 1024,
    overlap: int = 128,
    blend: Literal["cosine", "mean", "max"] = "cosine",
    dtype: Any = np.float32,
    progress: bool = False,
) -> np.ndarray:
    """Run ``fn`` on overlapping tiles of an ``(rows, cols, ...)`` image and stitch the output.

    ``fn`` gets a tile (channels last) and returns a 2-D array of the tile's ``(rows, cols)``
    shape. ``blend``: ``cosine`` (weighted average, seam-free), ``mean`` (plain average) or
    ``max`` (union of masks or class codes).
    """
    arr = np.asarray(getattr(image, "values", image))
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)  # (band, y, x) -> (y, x, band)
    H, W = arr.shape[:2]
    out = np.zeros((H, W), dtype=np.float32 if blend != "max" else dtype)
    wsum = np.zeros((H, W), dtype=np.float32) if blend != "max" else None
    wins = list(sliding_windows(H, W, tile, overlap))
    it = wins
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(wins, desc="tiles")
    for r0, c0, h, w in it:
        res = np.asarray(fn(arr[r0 : r0 + h, c0 : c0 + w]))
        if res.shape[:2] != (h, w):
            raise ValueError(f"fn returned shape {res.shape[:2]} for a {(h, w)} tile")
        if blend == "max":
            out[r0 : r0 + h, c0 : c0 + w] = np.maximum(out[r0 : r0 + h, c0 : c0 + w], res)
        else:
            wt = cosine_weight(h, w) if blend == "cosine" else np.ones((h, w), np.float32)
            out[r0 : r0 + h, c0 : c0 + w] += res * wt
            wsum[r0 : r0 + h, c0 : c0 + w] += wt
    if blend != "max":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(wsum > 0, out / wsum, np.nan)
    return out.astype(dtype, copy=False)


def merge_points(
    points: pd.DataFrame | np.ndarray, radius: float, *, x: str = "x", y: str = "y"
) -> pd.DataFrame:
    """Merge points closer than ``radius`` (e.g. the same plant found in two tiles).

    Each cluster is replaced by its mean position (and the mean of any other numeric
    column); a ``n_merged`` column records the cluster size.
    """
    from scipy.spatial import cKDTree

    df = pd.DataFrame(points, columns=[x, y]) if isinstance(points, np.ndarray) else points
    df = df.reset_index(drop=True)
    if df.empty:
        return df.assign(n_merged=pd.Series(dtype=int))
    xy = df[[x, y]].to_numpy(float)
    tree = cKDTree(xy)
    seen = np.zeros(len(df), bool)
    rows = []
    num = df.select_dtypes("number").columns
    for i in range(len(df)):
        if seen[i]:
            continue
        idx = [j for j in tree.query_ball_point(xy[i], r=radius) if not seen[j]]
        seen[idx] = True
        row = df.loc[idx, num].mean().to_dict()
        row["n_merged"] = len(idx)
        rows.append(row)
    return pd.DataFrame(rows)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one ``(x1, y1, x2, y2)`` box with an ``(n, 4)`` array of boxes."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    union = area(box) + area(boxes) - inter
    return np.where(union > 0, inter / union, 0.0)


def merge_boxes(
    boxes: pd.DataFrame, iou: float = 0.4, *, score: str = "confidence", by_class: bool = True
) -> pd.DataFrame:
    """Greedy non-maximum suppression across tiles on columns ``x1, y1, x2, y2`` and ``score``."""
    if boxes.empty:
        return boxes.copy()
    groups = boxes.groupby("class") if by_class and "class" in boxes else [(None, boxes)]
    keep = []
    for _, g in groups:
        g = g.sort_values(score, ascending=False)
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if not alive[i]:
                continue
            keep.append(g.index[i])
            later = np.arange(i + 1, len(g))
            later = later[alive[later]]
            if later.size:
                alive[later[box_iou(b[i], b[later]) > iou]] = False
    return boxes.loc[keep].sort_values(score, ascending=False).reset_index(drop=True)
