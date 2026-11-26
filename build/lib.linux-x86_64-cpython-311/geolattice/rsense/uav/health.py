"""Crop health from RGB: plant masks with SAM, then green vs senescent (yellow/brown) pixels.

Pipeline of the LUMS rice study (after SegVeg, Serouart et al. 2022):

1. :func:`sam_plant_mask` separates plants from soil or flood water. Positive point prompts
   come from high Excess-Green pixels, negatives from low-ExG, grey or bright pixels and
   the tile corners; the SAM mask that best fits the prompts is kept and cleaned.
2. :func:`classify_senescence` labels plant pixels green (1) or senescent (2) with a
   pixel classifier on 14 colour features (:func:`segveg_features`; SegVeg SVM/XGBoost
   pickles via :func:`load_segveg`), or with a simple colour rule when no model is given.
3. :func:`hsv_filter` sends dark or grey "senescent" pixels (shadow, water) back to background.
4. :func:`health_map` runs 1-3 on overlapping tiles of a large orthomosaic;
   :func:`health_summary` gives green/senescent fractions overall or per plot.

Labels: 0 background, 1 green, 2 senescent.
"""

from __future__ import annotations

import pickle
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

BACKGROUND, GREEN, SENESCENT = 0, 1, 2


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError("uav.health needs `pip install opencv-python`") from e
    return cv2


def _rgb8(image: Any) -> np.ndarray:
    from .io import to_rgb_array

    return np.ascontiguousarray(to_rgb_array(image, uint8=True))


def excess_green_norm(rgb: np.ndarray) -> np.ndarray:
    """ExG = 2G - R - B rescaled to 0-1 over the image."""
    a = np.asarray(rgb, np.float32)
    e = 2 * a[..., 1] - a[..., 0] - a[..., 2]
    e -= e.min()
    return e / (e.max() if e.max() > 0 else 1.0)


def sample_points_spread(
    mask: np.ndarray, k: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Up to ``k`` ``(x, y)`` points inside ``mask``, spread on a grid of cells."""
    rng = rng or np.random.default_rng(0)
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 2), np.int32)
    g = int(np.sqrt(k)) + 1
    cy = np.minimum(ys * g // H, g - 1)
    cx = np.minimum(xs * g // W, g - 1)
    cell = cy * g + cx
    pts = []
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        j = rng.choice(idx)
        pts.append((xs[j], ys[j]))
        if len(pts) >= k:
            break
    if len(pts) < k:
        extra = rng.choice(xs.size, size=min(k - len(pts), xs.size), replace=False)
        pts += [(xs[j], ys[j]) for j in extra]
    return np.array(pts, np.int32)


def clean_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Morphological open (3x3) and close (7x7), then drop components smaller than ``min_area``."""
    cv2 = _cv2()
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m
    area = np.bincount(lab.ravel())
    keep = np.flatnonzero(area >= min_area)
    keep = keep[keep != 0]
    return np.isin(lab, keep).astype(np.uint8)


_SAM_CACHE: dict[tuple, Any] = {}


def _sam_predictor(checkpoint: str, model_type: str, device: str | None):
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        raise ImportError("sam_plant_mask needs `pip install segment-anything torch`") from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(checkpoint), model_type, dev)
    if key not in _SAM_CACHE:
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device=dev)
        _SAM_CACHE[key] = SamPredictor(sam)
    return _SAM_CACHE[key]


def prompt_points(
    rgb: np.ndarray,
    *,
    n_pos: int = 80,
    n_neg: int = 24,
    pos_q: float = 0.95,
    neg_q: float = 0.20,
    sat_min: float = 0.18,
    bright_q: float = 0.70,
    corner_pad: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive (vegetation) and negative (soil, water, glare) SAM prompts from colour alone."""
    cv2 = _cv2()
    rng = np.random.default_rng(seed)
    exg = excess_green_norm(rgb)
    pos = exg > np.quantile(exg, pos_q)
    S = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    B = rgb[..., 2].astype(np.float32)
    neg = (exg < np.quantile(exg, neg_q)) | (S < sat_min) | (B > np.quantile(B, bright_q))
    p = sample_points_spread(pos, n_pos, rng)
    n = sample_points_spread(neg, n_neg, rng)
    H, W = rgb.shape[:2]
    corners = np.array(
        [
            [corner_pad, corner_pad],
            [W - corner_pad, corner_pad],
            [corner_pad, H - corner_pad],
            [W - corner_pad, H - corner_pad],
        ],
        np.int32,
    )
    return p, np.vstack([n, corners])


def _best_mask(masks: np.ndarray, pos: np.ndarray, neg: np.ndarray, max_area: float = 0.9) -> int:
    def hits(m: np.ndarray, p: np.ndarray) -> int:
        if not len(p):
            return 0
        H, W = m.shape
        return int(m[np.clip(p[:, 1], 0, H - 1), np.clip(p[:, 0], 0, W - 1)].sum())

    best, score_best = 0, -np.inf
    for i, m in enumerate(masks.astype(bool)):
        s = hits(m, pos) - 2 * hits(m, neg) - (10 if m.mean() > max_area else 0)
        if s > score_best:
            best, score_best = i, s
    return best


def sam_plant_mask(
    image: Any,
    checkpoint: str | PathLike,
    *,
    model_type: str = "vit_b",
    max_dim: int = 1024,
    min_area: int = 500,
    device: str | None = None,
    seed: int = 0,
    **prompt_kw: Any,
) -> np.ndarray:
    """Binary plant mask (``uint8`` 0/1) of an RGB tile with Segment Anything and ExG prompts.

    The tile is downsized to ``max_dim`` for SAM and the mask resized back. ``checkpoint``
    is a SAM ``.pth`` file (ViT-B in the LUMS study; see :func:`geolattice.rsense.uav.weights.fetch`).
    """
    cv2 = _cv2()
    rgb = _rgb8(image)
    H, W = rgb.shape[:2]
    s = min(1.0, max_dim / max(H, W))
    small = (
        cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR) if s < 1 else rgb
    )
    pos, neg = prompt_points(small, seed=seed, **prompt_kw)
    pred = _sam_predictor(str(checkpoint), model_type, device)
    pred.set_image(small)
    pts = np.vstack([pos, neg]).astype(np.float32)
    lab = np.r_[np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)]
    masks, _, _ = pred.predict(point_coords=pts, point_labels=lab, multimask_output=True)
    m = clean_mask(masks[_best_mask(masks, pos, neg)], min_area=int(min_area * s * s) or 1)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if s < 1 else m


def exg_plant_mask(
    image: Any, *, threshold: float | None = None, min_area: int = 500
) -> np.ndarray:
    """Model-free plant mask: Otsu threshold on ExG, then :func:`clean_mask`."""
    from skimage.filters import threshold_otsu

    rgb = _rgb8(image).astype(np.float32)
    e = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    t = threshold_otsu(e) if threshold is None else threshold
    return clean_mask(e > t, min_area)


# ------------------------------------------------------------------ senescence
def segveg_features(
    rgb: np.ndarray, *, k_mode: Literal["original", "pixel"] = "original"
) -> np.ndarray:
    """The 14 SegVeg colour features per pixel for an ``(n, 3)`` or ``(rows, cols, 3)`` uint8 RGB.

    R, G, B, H (deg), S (%), a* (CIELAB), b (ICC Lab), grey, magenta, yellow, Cb, Cr, I, Q.
    ``k_mode="original"`` reproduces the SegVeg/LUMS code, where the CMY black term is one
    minimum over all pixels passed in, so the published models see the features they were
    trained on; ``"pixel"`` uses the textbook per-pixel black term.
    """
    from PIL import Image, ImageCms
    from skimage import color

    a = np.asarray(rgb, np.uint8)
    shape = a.shape[:-1]
    px = a.reshape(1, -1, 3)
    pil = Image.fromarray(px)
    hsv = np.asarray(pil.convert("HSV"), np.float32)
    to_lab = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB"
    )
    lab_icc = np.asarray(ImageCms.applyTransform(pil, to_lab), np.float32)
    ycc = np.asarray(pil.convert("YCbCr"), np.float32)
    lab = color.rgb2lab(px)
    r, g, b = (px[..., i].astype(np.float32) for i in range(3))
    k = np.minimum(np.minimum(1 - r / 255, 1 - g / 255), 1 - b / 255)
    if k_mode == "original":
        k = np.full_like(k, k.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (1 - g / 255 - k) / (1 - k) * 100
        ye = (1 - b / 255 - k) / (1 - k) * 100
    feats = np.stack(
        [
            r,
            g,
            b,
            hsv[..., 0] * 360 / 255,
            hsv[..., 1] / 2.55,
            lab[..., 1],
            lab_icc[..., 2],
            (r + g + b) / 3,
            m,
            ye,
            ycc[..., 1],
            ycc[..., 2],
            0.596 * r - 0.275 * g - 0.321 * b,
            0.212 * r - 0.523 * g + 0.311 * b,
        ],
        -1,
    )
    return np.nan_to_num(feats[0]).reshape(*shape, 14)


def load_segveg(path: str | PathLike, *, patch_xgboost: bool = True) -> Any:
    """Load a SegVeg pickle (``model_scikit``, ``XGBoost``, ``Necrosis.pkl``).

    ``patch_xgboost`` adds attributes newer xgboost versions expect on old pickles. Only
    load pickles from sources you trust: unpickling can run arbitrary code.
    """
    with Path(path).open("rb") as f:
        model = pickle.load(f)  # noqa: S301
    if patch_xgboost and type(model).__module__.startswith("xgboost"):
        for attr in (
            "grow_policy",
            "max_bin",
            "eval_metric",
            "callbacks",
            "early_stopping_rounds",
            "max_cat_to_onehot",
            "max_leaves",
            "sampling_method",
            "enable_categorical",
            "feature_types",
            "max_cat_threshold",
            "predictor",
        ):
            if not hasattr(model, attr):
                setattr(model, attr, None)
    return model


def colour_rule(rgb: np.ndarray, *, hue_max: float = 75.0, exg_min: float = 0.0) -> np.ndarray:
    """Model-free green probability per pixel: hue above ``hue_max`` degrees (green side of
    yellow) and positive ExG mean green; returns 0-1 scores."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * 360 / 255
    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / 255
    return ((hue > hue_max) & (exg > exg_min)).astype(np.float32)


def classify_senescence(
    image: Any,
    plant_mask: np.ndarray | None = None,
    *,
    model: Any = None,
    threshold: float = 0.8,
    chunk: int | None = None,
    hue_max: float = 75.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Label plant pixels green (1) or senescent (2); background stays 0.

    ``model`` is a classifier with ``predict_proba`` on :func:`segveg_features` (class 1 =
    green), e.g. a SegVeg pickle from :func:`load_segveg`; without one, :func:`colour_rule`
    is used with ``hue_max`` (degrees). Olive-green canopies can sit below the 75° default;
    check the hue histogram of plant pixels and lower it (about 60° for the LUMS rice). ``chunk`` splits very large pixel sets (note: with SegVeg's original features
    the black term then varies per chunk). Returns ``(labels, p_green)``.
    """
    rgb = _rgb8(image)
    H, W = rgb.shape[:2]
    veg = np.ones((H, W), bool) if plant_mask is None else np.asarray(plant_mask).astype(bool)
    prob = np.zeros((H, W), np.float32)
    if model is None:
        prob[veg] = colour_rule(rgb, hue_max=hue_max)[veg]
    else:
        px = rgb[veg]
        out = np.empty(len(px), np.float32)
        step = chunk or max(len(px), 1)
        for i in range(0, len(px), step):
            out[i : i + step] = model.predict_proba(segveg_features(px[i : i + step]))[:, 1]
        prob[veg] = out
    lab = np.zeros((H, W), np.uint8)
    lab[veg] = np.where(prob[veg] >= threshold, GREEN, SENESCENT)
    return lab, prob


def hsv_filter(image: Any, labels: np.ndarray, *, s_min: int = 50, v_min: int = 160) -> np.ndarray:
    """Send "senescent" pixels that are grey (S < ``s_min``) or dark (V < ``v_min``) to
    background (shadow and water rather than yellow leaves)."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(_rgb8(image), cv2.COLOR_RGB2HSV)
    dark = (hsv[..., 1] < s_min) | (hsv[..., 2] < v_min)
    out = labels.copy()
    out[(labels == SENESCENT) & dark] = BACKGROUND
    return out


def health_map(
    image: Any,
    *,
    sam_checkpoint: str | PathLike | None = None,
    model: Any = None,
    threshold: float = 0.8,
    tile: int = 2048,
    overlap: int = 256,
    mask_method: Literal["sam", "exg", "none"] | None = None,
    hsv: bool = True,
    progress: bool = False,
    hue_max: float = 75.0,
    **sam_kw: Any,
) -> xr.DataArray:
    """Green / senescent map of a large RGB image, tile by tile (LUMS: 2048 px, 256 px overlap).

    ``mask_method`` defaults to ``"sam"`` when ``sam_checkpoint`` is given, else ``"exg"``.
    Overlapping tiles are combined with a pixel-wise maximum, so senescent wins at seams.
    ``hue_max`` is passed to the colour rule when no ``model`` is given.
    Returns a ``uint8`` DataArray (0 background, 1 green, 2 senescent) on the image grid.
    """
    from .io import as_image
    from .tiling import predict_tiled

    method = mask_method or ("sam" if sam_checkpoint else "exg")
    rgb = _rgb8(image)

    def one(t: np.ndarray) -> np.ndarray:
        if method == "sam":
            m = sam_plant_mask(t, sam_checkpoint, **sam_kw)
        elif method == "exg":
            m = exg_plant_mask(t)
        else:
            m = None
        lab, _ = classify_senescence(t, m, model=model, threshold=threshold, hue_max=hue_max)
        return hsv_filter(t, lab) if hsv else lab

    lab = predict_tiled(
        rgb, one, tile=tile, overlap=overlap, blend="max", dtype=np.uint8, progress=progress
    )
    tmpl = image if isinstance(image, xr.DataArray) else as_image(rgb)
    out = xr.DataArray(
        lab,
        dims=("y", "x"),
        coords={"y": tmpl.y, "x": tmpl.x},
        name="health",
        attrs={"classes": "0 background, 1 green, 2 senescent", "mask": method},
    )
    return out.rio.write_crs(tmpl.rio.crs) if tmpl.rio.crs is not None else out


def health_summary(labels: Any, zones: Any = None) -> pd.DataFrame:
    """Plant cover and green/senescent fractions, for the whole image or per zone label.

    ``zones`` is an integer array/DataArray of the same shape (0 = outside any plot), e.g.
    rasterised plot polygons.
    """
    lab = np.asarray(getattr(labels, "values", labels))
    z = (
        np.ones_like(lab, np.int32)
        if zones is None
        else np.asarray(getattr(zones, "values", zones))
    )
    rows = []
    for zid in np.unique(z[z > 0]):
        sel = lab[z == zid]
        plant = int(np.isin(sel, (GREEN, SENESCENT)).sum())
        sen = int((sel == SENESCENT).sum())
        rows.append(
            {
                "zone": int(zid),
                "pixels": int(sel.size),
                "plant_cover": plant / sel.size,
                "green_frac": (plant - sen) / plant if plant else np.nan,
                "senescent_frac": sen / plant if plant else np.nan,
            }
        )
    return pd.DataFrame(rows)
