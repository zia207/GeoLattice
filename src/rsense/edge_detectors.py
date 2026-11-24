"""Agricultural field-boundary and field-instance detection.

The module accepts an RGB or multispectral image (or a georeferenced raster) and a
caller-supplied checkpoint/model. It does not download imagery or weights automatically.
FC-DenseNet checkpoints created by Fieldedge-Detectors are supported directly; SegNet and
FCN-DKConv6 reproduce the other networks of Yang et al. (2020); other semantic segmenters
are built with the existing GeoLattice model builders.

The workflow helpers follow Yang et al. (2020): boundary labels rasterised from digitised
field polygons (:func:`rasterize_boundaries`), image tiles (:func:`tile_pairs`), rotation
augmentation (:func:`dihedral_augment`), the classical detectors used as baselines
(:func:`classical_edges`) and the boundary/field precision, recall, F1 and ROC AUC used to
compare methods (:func:`edge_metrics`, :func:`select_threshold`).

Fields of The World (FTW, https://fieldsofthe.world) U-Net checkpoints are rebuilt with
:func:`geolattice.rsense.classify.image_classification.build_segmenter` from the settings stored
in the checkpoint (:func:`load_ftw_model`). Local checkpoint files are used as they are;
registry names (:func:`ftw_models`) are looked up in a local cache and downloaded only when
``download=True`` is passed. :func:`ftw_stack`, :func:`predict_ftw` and
:func:`detect_fields_ftw` reproduce the FTW inference (two Sentinel-2 windows, B04-B03-B02-B08,
divided by 3000, up-sampled by ``resize_factor``) and :func:`to_fiboa` adds the fiboa core
properties to the field polygons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_FCDENSENET_NAMES = {
    "fcdensenet56": "FC-DenseNet56",
    "fc-densenet56": "FC-DenseNet56",
    "fcdensenet67": "FC-DenseNet67",
    "fc-densenet67": "FC-DenseNet67",
    "fcdensenet103": "FC-DenseNet103",
    "fc-densenet103": "FC-DenseNet103",
}

_PAPER_NAMES = {
    "segnet": "SegNet",
    "fcn_dkconv6": "FCN-DKConv6",
    "fcn-dkconv6": "FCN-DKConv6",
    "fcndkconv6": "FCN-DKConv6",
}

_SEGMENTER_NAMES = (
    "unet",
    "unet_basic",
    "unetplusplus",
    "deeplabv3plus",
    "deeplabv3",
    "fcn_resnet50",
    "fcn_resnet101",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet",
    "lraspp_mobilenet",
    "fpn",
    "manet",
    "pspnet",
    "segformer",
    "upernet",
)


@dataclass
class FieldEdgeResult:
    """Field polygons, their boundaries, the boundary-probability raster, and instance labels.

    ``class_probability`` holds the per-class softmax ``(classes, H, W)`` when the model
    predicts several classes (e.g. the FTW background / field / boundary models).
    """

    fields: Any
    boundaries: Any
    probability: np.ndarray
    labels: np.ndarray
    class_probability: np.ndarray | None = None


def available_models() -> list[str]:
    """Return FC-DenseNet checkpoints and GeoLattice segmenter architectures supported here."""
    return [*dict.fromkeys(_FCDENSENET_NAMES.values()), *dict.fromkeys(_PAPER_NAMES.values()),
            *_SEGMENTER_NAMES]


def _torch():
    from .dl._torch import require_torch

    return require_torch()


def _resolve_device(device: str | None = None):
    torch = _torch()
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_edge_model(
    model: str = "FC-DenseNet103",
    *,
    num_classes: int = 2,
    device: str | None = None,
    **kwargs: Any,
):
    """Build a field-edge segmentation model.

    ``FC-DenseNet56/67/103`` use the checkpoint-compatible FieldEdge architectures;
    ``SegNet`` and ``FCN-DKConv6`` are the other networks of Yang et al. (2020). All three
    take ``in_channels`` (default 3; 4 for blue-green-red-NIR). Other supported names delegate
    to :func:`geolattice.rsense.classify.image_classification.build_segmenter`.
    """
    torch = _torch()
    preset = _FCDENSENET_NAMES.get(model.lower())
    paper = _PAPER_NAMES.get(model.lower())
    if preset:
        from .dl.models.fieldedge import build_fieldedge_fcdensenet

        network = build_fieldedge_fcdensenet(preset, num_classes=num_classes, **kwargs)
    elif paper:
        from .dl.models.fieldedge import FCNDKConv6, SegNet

        network = (SegNet if paper == "SegNet" else FCNDKConv6)(num_classes=num_classes, **kwargs)
    else:
        arch = model.lower()
        if arch not in _SEGMENTER_NAMES:
            raise ValueError(f"Unknown edge model {model!r}; choose from {available_models()}")
        from .classify.image_classification import build_segmenter

        kwargs.setdefault("in_channels", 3)
        network = build_segmenter(arch, num_classes=num_classes, **kwargs)
    network.to(_resolve_device(device))
    return network


def load_edge_model(
    checkpoint: str | Path,
    *,
    model: str | None = None,
    num_classes: int = 2,
    device: str | None = None,
    strict: bool = True,
    **model_kwargs: Any,
):
    """Load a local FieldEdge FC-DenseNet or GeoLattice segmenter checkpoint.

    FieldEdge FC-DenseNet checkpoints store ``preset``, ``state_dict`` and an optional
    validation-selected ``threshold``. For plain GeoLattice state dictionaries, pass ``model``.
    """
    torch = _torch()
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    scalar_type = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    safe_types = [scalar_type, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_types):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and "state_dict" in saved:
        state = saved["state_dict"]
        model_name = model or saved.get("preset") or saved.get("name")
        num_classes = int(saved.get("num_classes", num_classes))
    else:
        state = saved
        model_name = model
    if not model_name:
        raise ValueError(
            "model is required when the checkpoint does not contain its architecture name"
        )
    network = build_edge_model(model_name, num_classes=num_classes, device="cpu", **model_kwargs)
    network.load_state_dict(state, strict=strict)
    network.to(_resolve_device(device)).eval()
    if isinstance(saved, dict):
        network.edge_threshold = float(saved.get("threshold", 0.5))
        network.edge_checkpoint_metadata = {
            key: saved[key] for key in ("preset", "threshold", "valid_f1", "epoch") if key in saved
        }
    return network


def load_delineate_anything(checkpoint: str | Path, *, device: str | None = None):
    """Load a local Ultralytics segmentation checkpoint (e.g. FieldEdge fine-tuned DA V2).

    Requires the optional ``ultralytics`` package. Checkpoint downloads are never implicit.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Delineate Anything inference requires `pip install geolattice[rsense-vision]`"
        ) from exc
    model = YOLO(str(path))
    return model.to(str(_resolve_device(device)))


def load_sam2(model_id: str = "facebook/sam2.1-hiera-large", *, device: str | None = None):
    """Load a SAM 2 checkpoint from its model hub ID.

    Requires the separately installed ``sam2`` package and its normal model-hub access.
    """
    try:
        from sam2.build_sam import build_sam2_hf
    except ImportError as exc:
        raise ImportError(
            "SAM 2 inference requires the `sam2` package from facebookresearch/sam2"
        ) from exc
    return build_sam2_hf(model_id, device=str(_resolve_device(device)))


def _as_rgb01(image: np.ndarray, image_range: tuple[float, float] | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError("image must be an H×W×3 RGB array")
    if arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"image must have exactly three RGB channels; got {arr.shape}")
    return _as_image01(arr, image_range)


def _as_image01(image: np.ndarray, image_range: tuple[float, float] | None = None) -> np.ndarray:
    """H×W×C image (any channel count; C×H×W is accepted when C <= 16) scaled to [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError("image must be an H×W×C array")
    if arr.shape[0] <= 16 and arr.shape[-1] > 16:
        arr = np.moveaxis(arr, 0, -1)
    arr = arr.astype(np.float32, copy=False)
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("image contains no finite RGB values")
    if image_range is not None:
        lo, hi = map(float, image_range)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("image_range must be a finite increasing (low, high) pair")
        arr = (arr - lo) / (hi - lo)
    elif np.issubdtype(np.asarray(image).dtype, np.integer) or float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    if np.nanmin(arr) < -1e-5 or np.nanmax(arr) > 1.00001:
        raise ValueError(
            "image values must be in [0, 1] or [0, 255]; pass image_range=(low, high) to scale other data"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 1)


def _window_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    return sorted(set([*range(0, length - tile + 1, stride), length - tile]))


def _hann_weight(tile: int) -> np.ndarray:
    win = np.maximum(np.hanning(tile + 2)[1:-1], 1e-3).astype(np.float32)
    return np.outer(win, win)


def _model_logits(model, batch):
    output = model(batch)
    if isinstance(output, dict):
        output = output.get("out", output.get("logits"))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output is None or not hasattr(output, "ndim") or output.ndim != 4:
        raise ValueError("segmentation model must return logits shaped (N, classes, height, width)")
    return output


def predict_boundary_probability(
    image: np.ndarray,
    model,
    *,
    boundary_class: int = 1,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    device: str | None = None,
    precision: str = "32",
    image_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Run tiled, overlap-blended inference and return the boundary probability in ``[0, 1]``.

    The model is expected to return per-pixel logits. ``image`` is H×W×C with as many
    channels as the model takes (3 for RGB, 4 for blue-green-red-NIR); it is scaled to
    ``[0, 1]`` like FieldEdge FC-DenseNet training (``image_range`` for other data ranges). ``precision`` may be ``"32"``, ``"16"``
    or ``"bf16"``; reduced precision autocast is enabled only on CUDA.
    """
    if tile_size < 32 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            "tile_size must be >= 32 and overlap must satisfy 0 <= overlap < tile_size"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if precision not in {"32", "16", "bf16"}:
        raise ValueError("precision must be '32', '16', or 'bf16'")
    torch = _torch()
    dev = _resolve_device(device)
    x = _as_image01(image, image_range)
    height, width = x.shape[:2]
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    pad_mode = "reflect" if height > 1 and width > 1 else "edge"
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
    padded_h, padded_w = x.shape[:2]
    rows = _window_starts(padded_h, tile_size, overlap)
    cols = _window_starts(padded_w, tile_size, overlap)
    windows = [(row, col) for row in rows for col in cols]
    weight = _hann_weight(tile_size)
    accumulated = None
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)
    amp_dtype = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    model = model.to(dev).eval()

    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            batch = np.stack(
                [
                    x[r : r + tile_size, c : c + tile_size].transpose(2, 0, 1)
                    for r, c in batch_windows
                ]
            )
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(dev)
            with torch.autocast(
                dev.type, dtype=amp_dtype, enabled=amp_dtype is not None and dev.type == "cuda"
            ):
                logits = _model_logits(model, tensor)
            if logits.shape[-2:] != (tile_size, tile_size):
                logits = torch.nn.functional.interpolate(
                    logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False
                )
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
            else:
                if not 0 <= boundary_class < logits.shape[1]:
                    raise ValueError(
                        f"boundary_class={boundary_class} outside model output classes"
                    )
                probs = torch.softmax(logits.float(), dim=1)[:, boundary_class].cpu().numpy()
            if accumulated is None:
                accumulated = np.zeros((padded_h, padded_w), dtype=np.float32)
            for (row, col), probability in zip(batch_windows, probs):
                accumulated[row : row + tile_size, col : col + tile_size] += probability * weight
                weights[row : row + tile_size, col : col + tile_size] += weight
    if accumulated is None:
        raise RuntimeError("no image windows were processed")
    result = accumulated / np.maximum(weights, 1e-8)
    return np.clip(result[:height, :width], 0, 1)


def boundary_to_instances(
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area_pixels: int = 1,
    dilate_pixels: int = 1,
) -> np.ndarray:
    """Convert a predicted boundary-probability image to connected field-instance IDs."""
    from scipy import ndimage

    p = np.asarray(probability, dtype=np.float32)
    if p.ndim != 2 or not np.isfinite(p).any():
        raise ValueError("probability must be a finite 2-D array")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if min_area_pixels < 1 or dilate_pixels < 0:
        raise ValueError("min_area_pixels must be positive and dilate_pixels non-negative")
    barrier = np.isfinite(p) & (p >= threshold)
    if dilate_pixels:
        barrier = ndimage.binary_dilation(barrier, iterations=dilate_pixels)
    barrier[[0, -1], :] = True
    barrier[:, [0, -1]] = True
    labels, count = ndimage.label(~barrier)
    if count == 0:
        return np.zeros(p.shape, dtype=np.int32)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_area_pixels
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    compact = np.zeros_like(labels, dtype=np.int32)
    for next_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        compact[labels == old_id] = next_id
    return compact


def _empty_gdf(crs=None):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"field_id": [], "score": [], "area_m2": [], "perimeter_m": []},
        geometry=[],
        crs=crs,
    )


def _field_geodataframes(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    transform=None,
    crs=None,
    min_area_m2: float = 20.0,
) -> tuple[Any, Any]:
    import geopandas as gpd
    from pyproj import CRS
    from rasterio.transform import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape

    labels = np.asarray(labels, dtype=np.int32)
    affine = Affine.identity() if transform is None else transform
    records = []
    for geometry, value in shapes(labels, mask=labels > 0, transform=affine, connectivity=4):
        geom = shape(geometry)
        if geom.is_empty:
            continue
        field_id = int(value)
        pixel_mask = labels == field_id
        mean_boundary = float(np.nanmean(probability[pixel_mask]))
        records.append(
            {
                "field_id": field_id,
                "score": float(np.clip(1.0 - mean_boundary, 0, 1)),
                "geometry": geom,
            }
        )
    if not records:
        empty = _empty_gdf(crs)
        return empty, empty.copy()

    fields = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    if crs is not None:
        crs_obj = CRS.from_user_input(crs)
        if crs_obj.is_geographic:
            metric_crs = fields.estimate_utm_crs()
            measured = fields.to_crs(metric_crs)
            fields["area_m2"] = measured.area
            fields["perimeter_m"] = measured.length
            fields["geometry"] = measured.to_crs(crs).geometry
        else:
            axes = crs_obj.axis_info
            factor = axes[0].unit_conversion_factor if axes else 1.0
            fields["area_m2"] = fields.geometry.area * factor**2
            fields["perimeter_m"] = fields.geometry.length * factor
        fields = fields.loc[fields["area_m2"] >= float(min_area_m2)].copy()
    else:
        fields["area_pixels"] = fields.geometry.area
        fields["perimeter_pixels"] = fields.geometry.length

    if not len(fields):
        empty = _empty_gdf(crs)
        return empty, empty.copy()
    fields = fields.reset_index(drop=True)
    boundaries = gpd.GeoDataFrame(
        {"field_id": fields["field_id"].to_numpy()},
        geometry=fields.geometry.boundary.values,
        crs=crs,
    )
    return fields, boundaries


def detect_field_edges(
    image: np.ndarray,
    model,
    *,
    transform=None,
    crs=None,
    boundary_class: int = 1,
    threshold: float | None = None,
    min_area_m2: float = 20.0,
    min_area_pixels: int = 1,
    dilate_pixels: int = 1,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    device: str | None = None,
    precision: str = "32",
    image_range: tuple[float, float] | None = None,
) -> FieldEdgeResult:
    """Infer field-boundary probabilities and return fields and boundary geometries.

    ``image`` is H×W×C (RGB, or as many bands as the model takes) in ``uint8`` [0, 255],
    float [0, 1], or float [0, 255]; ``image_range`` rescales other ranges.
    A georeferenced affine ``transform`` and ``crs`` make the output geometries map-based;
    otherwise the output coordinates are image pixels. Model weights are not downloaded.
    """
    probability = predict_boundary_probability(
        image,
        model,
        boundary_class=boundary_class,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        precision=precision,
        image_range=image_range,
    )
    selected_threshold = float(
        getattr(model, "edge_threshold", 0.5) if threshold is None else threshold
    )
    labels = boundary_to_instances(
        probability,
        threshold=selected_threshold,
        min_area_pixels=min_area_pixels,
        dilate_pixels=dilate_pixels,
    )
    fields, boundaries = _field_geodataframes(
        labels, probability, transform=transform, crs=crs, min_area_m2=min_area_m2
    )
    return FieldEdgeResult(fields, boundaries, probability, labels)


def detect_field_edges_from_raster(
    path: str | Path,
    model,
    *,
    bands: Sequence[int] = (1, 2, 3),
    **kwargs: Any,
) -> FieldEdgeResult:
    """Read selected one-based bands (RGB by default, e.g. ``(2, 3, 5, 8)`` for WorldView
    blue-green-red-NIR2 with a 4-band model) from a raster and detect field boundaries."""
    import rasterio

    with rasterio.open(path) as src:
        if len(bands) < 1 or any(int(b) < 1 or int(b) > src.count for b in bands):
            raise ValueError(f"bands must be one-based indices in [1, {src.count}]")
        image = np.moveaxis(src.read(list(bands)), 0, -1)
        if src.nodata is not None:
            image = np.where(image == src.nodata, 0, image)
        return detect_field_edges(image, model, transform=src.transform, crs=src.crs, **kwargs)


def detect_delineate_anything(
    image: np.ndarray,
    model,
    *,
    tile_size: int = 768,
    overlap: int | None = None,
    confidence: float = 0.25,
    min_area_pixels: int = 320,
    transform=None,
    crs=None,
    image_range: tuple[float, float] | None = None,
) -> FieldEdgeResult:
    """Run an Ultralytics instance-segmentation checkpoint over overlapping RGB windows.

    ``model`` is typically returned by :func:`load_delineate_anything`. The result polygons
    are merged by overlap/NMS before conversion to ``FieldEdgeResult``.
    """
    from shapely.geometry import Polygon

    rgb01 = _as_rgb01(image, image_range)
    rgb = np.round(rgb01 * 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    overlap = tile_size // 2 if overlap is None else int(overlap)
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError(
            "tile_size must be >= 32 and overlap must satisfy 0 <= overlap < tile_size"
        )
    found = []
    for row in _window_starts(height, tile_size, overlap):
        for col in _window_starts(width, tile_size, overlap):
            crop = rgb[row : min(row + tile_size, height), col : min(col + tile_size, width)]
            result = model.predict(
                source=crop,
                conf=confidence,
                iou=0.3,
                max_det=300,
                imgsz=tile_size,
                verbose=False,
            )[0]
            if result.masks is None or result.boxes is None:
                continue
            scores = result.boxes.conf.detach().cpu().numpy()
            for contour, score in zip(result.masks.xy, scores):
                points = np.asarray(contour, dtype=np.float64)
                if points.ndim == 2 and points.shape[0] >= 3:
                    found.append((Polygon(points + [col, row]), float(score)))
    polygons, scores = _polygon_nms(
        found, iou_threshold=0.3, containment_threshold=0.6, min_area_pixels=min_area_pixels
    )
    return _result_from_polygons(
        polygons,
        scores,
        (height, width),
        transform=transform,
        crs=crs,
        min_area_pixels=min_area_pixels,
    )


def detect_sam2(
    image: np.ndarray,
    model,
    *,
    tile_size: int = 1024,
    overlap: int = 192,
    min_area_pixels: int = 320,
    max_area_fraction: float = 0.25,
    generator_kwargs: dict[str, Any] | None = None,
    transform=None,
    crs=None,
    image_range: tuple[float, float] | None = None,
) -> FieldEdgeResult:
    """Generate SAM 2 field-instance masks in overlapping RGB windows.

    SAM 2 is a prompt-free mask proposal model here, not a field-specific classifier.
    Review its polygons before treating them as agricultural fields.
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "SAM 2 inference requires the `sam2` package from facebookresearch/sam2"
        ) from exc
    import rasterio.features
    from rasterio.transform import Affine
    from shapely.geometry import shape

    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError(
            "tile_size must be >= 32 and overlap must satisfy 0 <= overlap < tile_size"
        )
    if min_area_pixels < 1 or not 0 < max_area_fraction <= 1:
        raise ValueError("min_area_pixels must be positive and max_area_fraction in (0, 1]")
    rgb01 = _as_rgb01(image, image_range)
    rgb = np.round(rgb01 * 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    settings = {"min_mask_region_area": min_area_pixels, **(generator_kwargs or {})}
    generator = SAM2AutomaticMaskGenerator(model, **settings)
    candidates = []
    for row in _window_starts(height, tile_size, overlap):
        for col in _window_starts(width, tile_size, overlap):
            crop = np.ascontiguousarray(
                rgb[row : min(row + tile_size, height), col : min(col + tile_size, width)]
            )
            for mask in generator.generate(crop):
                if mask["area"] < min_area_pixels:
                    continue
                if mask["area"] > max_area_fraction * crop.shape[0] * crop.shape[1]:
                    continue
                mask_array = np.asarray(mask["segmentation"], dtype=np.uint8)
                shifted = Affine.translation(col, row)
                for geometry, _ in rasterio.features.shapes(
                    mask_array, mask=mask_array > 0, transform=shifted, connectivity=8
                ):
                    polygon = shape(geometry)
                    if not polygon.is_empty and polygon.area >= min_area_pixels:
                        candidates.append((polygon, float(mask.get("predicted_iou", 0.0))))
    polygons, scores = _polygon_nms(
        candidates,
        iou_threshold=0.3,
        containment_threshold=0.6,
        min_area_pixels=min_area_pixels,
    )
    return _result_from_polygons(
        polygons,
        scores,
        (height, width),
        transform=transform,
        crs=crs,
        min_area_pixels=min_area_pixels,
    )


def _polygon_nms(
    candidates,
    *,
    iou_threshold: float,
    containment_threshold: float,
    min_area_pixels: int,
):
    """Greedy polygon NMS for detections repeated in overlapping image windows."""
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept, scores = [], []
    for geom, score in ordered:
        if geom.is_empty or not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.area < min_area_pixels:
            continue
        duplicate = False
        remainder = geom
        for existing in kept:
            if not geom.intersects(existing):
                continue
            intersection = geom.intersection(existing).area
            union = geom.area + existing.area - intersection
            if union > 0 and intersection / union >= iou_threshold:
                duplicate = True
                break
            if intersection / min(geom.area, existing.area) >= containment_threshold:
                duplicate = True
                break
            remainder = remainder.difference(existing)
        if not duplicate:
            if remainder.is_empty or remainder.area < min_area_pixels:
                continue
            kept.append(remainder)
            scores.append(float(score))
    return kept, scores


def _result_from_polygons(
    polygons,
    scores,
    shape_hw,
    *,
    transform=None,
    crs=None,
    min_area_pixels=1,
) -> FieldEdgeResult:
    import rasterio.features
    from rasterio.transform import Affine

    height, width = shape_hw
    affine = Affine.identity() if transform is None else transform
    shapes = [(geom, index + 1) for index, geom in enumerate(polygons)]
    labels = (
        rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=Affine.identity(),
            fill=0,
            dtype="int32",
        )
        if shapes
        else np.zeros((height, width), dtype=np.int32)
    )
    # Scores are used directly because instance-segmentation models have no boundary raster.
    probability = np.zeros((height, width), dtype=np.float32)
    fields, boundaries = _field_geodataframes(
        labels,
        probability,
        transform=affine,
        crs=crs,
        min_area_m2=0,
    )
    if len(fields):
        for index, row in fields.iterrows():
            field_id = int(row.field_id)
            fields.loc[index, "score"] = scores[field_id - 1] if field_id <= len(scores) else np.nan
        if crs is None and min_area_pixels > 1:
            keep_ids = np.flatnonzero(np.bincount(labels.ravel()) >= min_area_pixels)
            keep_ids = set(keep_ids.tolist()) - {0}
            fields = fields[fields.field_id.isin(keep_ids)].reset_index(drop=True)
            boundaries = gpd_boundary_frame(fields, crs)
    return FieldEdgeResult(fields, boundaries, probability, labels)

# ----------------------------------------------------------------------------- training data
def rasterize_boundaries(
    fields,
    *,
    transform,
    shape: tuple[int, int],
    buffer: float = 0.5,
    all_touched: bool = True,
    return_valid: bool = False,
):
    """Boundary label raster from digitised field polygons (1 = boundary, 0 = field interior).

    Following Yang et al. (2020), each polygon outline is buffered by ``buffer`` map units
    (0.5 m gives a 1 m wide boundary band) and burnt into a grid of ``shape`` with the affine
    ``transform`` (polygons must be in the raster CRS). With ``return_valid=True`` it also
    returns a mask of pixels inside a digitised field or its boundary band, so areas that
    nobody labelled (villages, water) can be left out of training and scoring.
    """
    from rasterio.features import rasterize

    geoms = [g for g in getattr(fields, "geometry", fields) if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("fields contains no polygons")
    if buffer <= 0:
        raise ValueError("buffer must be positive")
    lines = [g.boundary.buffer(buffer) for g in geoms]
    boundary = rasterize(((g, 1) for g in lines), out_shape=shape, transform=transform,
                         fill=0, all_touched=all_touched, dtype="uint8")
    if not return_valid:
        return boundary
    inside = rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                       fill=0, all_touched=False, dtype="uint8")
    return boundary, (inside.astype(bool) | boundary.astype(bool))


def tile_pairs(
    image: np.ndarray,
    label: np.ndarray,
    *,
    size: int = 192,
    stride: int | None = None,
    valid: np.ndarray | None = None,
    min_valid: float = 0.5,
) -> dict[str, np.ndarray]:
    """Cut an H×W×C image and its H×W label into ``size``×``size`` tiles (Yang et al. used 192).

    Tiles start every ``stride`` pixels (default ``size``: no overlap) and only whole tiles are
    kept. Tiles with less than ``min_valid`` of their pixels inside ``valid`` are dropped.
    Returns ``images`` (N, C, size, size), ``labels`` (N, size, size), ``valid`` (N, size, size)
    and ``origins`` (N, 2) row/column of each tile's top-left pixel.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    lab = np.asarray(label)
    if img.shape[:2] != lab.shape:
        raise ValueError(f"image {img.shape[:2]} and label {lab.shape} grids differ")
    ok = np.ones(lab.shape, bool) if valid is None else np.asarray(valid, bool)
    stride = size if stride is None else int(stride)
    if size < 1 or stride < 1:
        raise ValueError("size and stride must be positive")
    rows = range(0, lab.shape[0] - size + 1, stride)
    cols = range(0, lab.shape[1] - size + 1, stride)
    keep = [(r, c) for r in rows for c in cols if ok[r:r + size, c:c + size].mean() >= min_valid]
    if not keep:
        raise ValueError("no tile meets min_valid; lower it or check the valid mask")
    return {
        "images": np.stack([img[r:r + size, c:c + size].transpose(2, 0, 1) for r, c in keep]),
        "labels": np.stack([lab[r:r + size, c:c + size] for r, c in keep]),
        "valid": np.stack([ok[r:r + size, c:c + size] for r, c in keep]),
        "origins": np.asarray(keep, dtype=np.int64),
    }


def dihedral_augment(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    rotations: Sequence[int] = (0, 1, 2, 3),
    flips: bool = False,
    extra: Sequence[np.ndarray] = (),
) -> tuple[np.ndarray, ...]:
    """Rotate (and optionally flip) tiles: ``rotations`` are multiples of 90 degrees.

    The default reproduces Yang et al. (2020): the original tile plus 90-, 180- and
    270-degree rotations, i.e. four copies of each tile. ``images`` is (N, C, H, W),
    ``labels`` and every array in ``extra`` (e.g. valid masks) are (N, H, W). Augment the
    training tiles only, after the split, so rotated copies of a test tile never reach training.
    """
    images, labels = np.asarray(images), np.asarray(labels)
    out_i, out_l, out_e = [], [], [[] for _ in extra]
    for k in rotations:
        for flip in ((False, True) if flips else (False,)):
            def tf(a):
                a = np.rot90(a, int(k), axes=(-2, -1))
                return np.flip(a, -1) if flip else a
            out_i.append(tf(images))
            out_l.append(tf(labels))
            for store, arr in zip(out_e, extra):
                store.append(tf(np.asarray(arr)))
    result = [np.ascontiguousarray(np.concatenate(out_i)), np.ascontiguousarray(np.concatenate(out_l))]
    result += [np.ascontiguousarray(np.concatenate(e)) for e in out_e]
    return tuple(result)


# ----------------------------------------------------------------------------- baselines
_CLASSICAL = ("roberts", "sobel", "prewitt", "scharr", "log", "canny")


def classical_edges(
    image: np.ndarray,
    method: str = "sobel",
    *,
    sigma: float = 1.0,
    low: float | None = None,
    high: float | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Classical edge detectors as a boundary score in ``[0, 1]`` (Roberts, Sobel, Prewitt,
    Scharr, Laplacian of Gaussian, Canny) for comparison with the networks.

    Each band of the H×W×C ``image`` is stretched to ``[0, 1]`` between its ``percentiles``,
    filtered on its own, and the band responses are combined by their maximum (a
    multispectral gradient). Gradient magnitudes are scaled by their 99.5th percentile.
    ``log`` is the magnitude of the Laplacian of Gaussian with ``sigma``; ``canny`` returns
    a binary edge map (``sigma`` smoothing, hysteresis ``low``/``high`` thresholds), so it
    gives a single operating point rather than a score.
    """
    from scipy import ndimage
    from skimage import feature, filters

    m = method.lower()
    if m not in _CLASSICAL:
        raise ValueError(f"method must be one of {_CLASSICAL}")
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    responses = []
    for b in range(img.shape[-1]):
        band = img[..., b]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.percentile(band[finite], percentiles)
        band = np.clip((np.where(finite, band, lo) - lo) / max(hi - lo, 1e-12), 0, 1)
        if m == "canny":
            responses.append(feature.canny(band, sigma=sigma, low_threshold=low,
                                           high_threshold=high).astype(np.float32))
            continue
        smooth = ndimage.gaussian_filter(band, sigma) if sigma and m != "log" else band
        if m == "log":
            responses.append(np.abs(ndimage.gaussian_laplace(band, sigma)))
        else:
            responses.append(getattr(filters, m)(smooth).astype(np.float32))
    if not responses:
        raise ValueError("image contains no finite values")
    score = np.max(np.stack(responses), axis=0)
    if m != "canny":
        score = np.clip(score / max(float(np.percentile(score, 99.5)), 1e-12), 0, 1)
    return score.astype(np.float32)


# ----------------------------------------------------------------------------- evaluation
def _tolerant_hits(pred: np.ndarray, truth: np.ndarray, tolerance: int):
    if tolerance <= 0:
        return pred & truth, pred & truth
    from scipy import ndimage

    st = ndimage.generate_binary_structure(2, 2)
    near_truth = ndimage.binary_dilation(truth, st, iterations=tolerance)
    near_pred = ndimage.binary_dilation(pred, st, iterations=tolerance)
    return pred & near_truth, truth & near_pred


def edge_metrics(
    score: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float = 0.5,
    valid: np.ndarray | None = None,
    tolerance: int = 0,
    max_auc_pixels: int = 500_000,
    seed: int = 0,
) -> dict[str, float]:
    """Boundary and field (non-boundary) precision, recall and F1, plus ROC AUC.

    These are the measures of Yang et al. (2020), Tables 7-8 and Fig. 10. ``score`` is a
    boundary probability or edge strength (2-D, or a stack of tiles), ``truth`` the 0/1
    boundary label, ``valid`` the pixels to score. ``tolerance`` (pixels) counts a predicted
    boundary pixel as correct when a true boundary lies within that distance, and a true
    boundary pixel as found when a prediction lies within it (0 = exact pixel match, as in
    the paper). AUC uses at most ``max_auc_pixels`` randomly chosen pixels and is
    threshold-free.
    """
    from sklearn.metrics import roc_auc_score

    s = np.asarray(score, dtype=np.float32)
    t = np.asarray(truth).astype(bool)
    if s.shape != t.shape:
        raise ValueError(f"score {s.shape} and truth {t.shape} differ")
    ok = np.isfinite(s) if valid is None else (np.asarray(valid, bool) & np.isfinite(s))
    s2, t2, ok2 = (a.reshape((-1,) + a.shape[-2:]) for a in (s, t, ok))
    tp_p = tp_r = fp = fn = tn = 0
    for si, ti, oi in zip(s2, t2, ok2):
        pred = (si >= threshold) & oi
        tru = ti & oi
        hit_p, hit_r = _tolerant_hits(pred, tru, tolerance)
        tp_p += hit_p.sum()
        fp += (pred & ~hit_p).sum()
        tp_r += hit_r.sum()
        fn += (tru & ~hit_r).sum()
        tn += (~pred & ~tru & oi).sum()

    def prf(tp_prec, fp_, tp_rec, fn_):
        p = tp_prec / max(tp_prec + fp_, 1)
        r = tp_rec / max(tp_rec + fn_, 1)
        return p, r, 2 * p * r / max(p + r, 1e-12)

    bp, br, bf = prf(tp_p, fp, tp_r, fn)
    fp_field, fr_field, ff = prf(tn, fn, tn, fp)
    ys, yt = s[ok], t[ok]
    if len(ys) > max_auc_pixels:
        pick = np.random.default_rng(seed).choice(len(ys), max_auc_pixels, replace=False)
        ys, yt = ys[pick], yt[pick]
    auc = float(roc_auc_score(yt, ys)) if 0 < yt.sum() < len(yt) else float("nan")
    return {
        "boundary_precision": float(bp), "boundary_recall": float(br), "boundary_f1": float(bf),
        "field_precision": float(fp_field), "field_recall": float(fr_field), "field_f1": float(ff),
        "auc": auc, "threshold": float(threshold),
        "boundary_share": float(t[ok].mean()) if ok.any() else float("nan"),
    }


def select_threshold(
    score: np.ndarray,
    truth: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    thresholds: Sequence[float] | None = None,
    tolerance: int = 0,
):
    """Pick the threshold that maximises boundary F1 on validation data.

    Returns ``(best_threshold, table)``; the table has one row of :func:`edge_metrics`
    (without AUC) per candidate threshold. Choose it on validation tiles, never on test tiles.
    """
    import pandas as pd

    grid = np.linspace(0.05, 0.95, 19) if thresholds is None else np.asarray(thresholds, float)
    rows = [edge_metrics(score, truth, threshold=float(th), valid=valid, tolerance=tolerance,
                         max_auc_pixels=2) for th in grid]
    table = pd.DataFrame(rows).drop(columns="auc")
    best = float(table.loc[table["boundary_f1"].idxmax(), "threshold"])
    return best, table


# ----------------------------------------------------------------------------- Fields of The World
FTW_RELEASE_URL = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/"
FTW_BANDS = ("B04", "B03", "B02", "B08")
FTW_CLASSES = {0: "background", 1: "field", 2: "boundary"}
FTW_NORMALIZATION = 3000.0
_FTW_MIXED, _FTW_CCBY = "Mixed Open Licenses (incl. CC-BY-NC-SA)", "CC-BY-4.0"
# name -> (release path, version, licence, description); all are U-Nets on 2 x 4 Sentinel-2 bands
FTW_MODELS = {
    "FTW_PRUE_EFNET_B3": (
        "v3/prue_efnet3_checkpoint.ckpt",
        "v3",
        _FTW_MIXED,
        "3-class, EfficientNet-B3: fastest",
    ),
    "FTW_PRUE_EFNET_B5": (
        "v3/prue_efnet5_checkpoint.ckpt",
        "v3",
        _FTW_MIXED,
        "3-class, EfficientNet-B5: FTW default",
    ),
    "FTW_PRUE_EFNET_B7": (
        "v3/prue_efnet7_checkpoint.ckpt",
        "v3",
        _FTW_MIXED,
        "3-class, EfficientNet-B7: most accurate",
    ),
    "FTW_PRUE_EFNET_B3_CCBY": (
        "v3.1/prue_efnetb3_ccby_checkpoint.ckpt",
        "v3.1",
        _FTW_CCBY,
        "3-class, B3, CC-BY/CC0 training data only",
    ),
    "FTW_PRUE_EFNET_B5_CCBY": (
        "v3.1/prue_efnetb5_ccby_checkpoint.ckpt",
        "v3.1",
        _FTW_CCBY,
        "3-class, B5, CC-BY/CC0 training data only",
    ),
    "FTW_PRUE_EFNET_B7_CCBY": (
        "v3.1/prue_efnetb7_ccby_checkpoint.ckpt",
        "v3.1",
        _FTW_CCBY,
        "3-class, B7, CC-BY/CC0 training data only",
    ),
    "FTW_v2_3_Class_FULL_multiWindow": (
        "v2/3_Class_FULL_FTW_Pretrained_v2.ckpt",
        "v2",
        _FTW_MIXED,
        "legacy 3-class, two windows",
    ),
    "FTW_v1_3_Class_FULL": (
        "v1/3_Class_FULL_FTW_Pretrained.ckpt",
        "v1",
        _FTW_MIXED,
        "legacy 3-class",
    ),
    "FTW_v1_3_Class_CCBY": (
        "v1/3_Class_CCBY_FTW_Pretrained.ckpt",
        "v1",
        _FTW_CCBY,
        "legacy 3-class, CC-BY data only",
    ),
    "FTW_v1_2_Class_FULL": (
        "v1/2_Class_FULL_FTW_Pretrained.ckpt",
        "v1",
        _FTW_MIXED,
        "legacy 2-class (field / not field)",
    ),
    "FTW_v1_2_Class_CCBY": (
        "v1/2_Class_CCBY_FTW_Pretrained.ckpt",
        "v1",
        _FTW_CCBY,
        "legacy 2-class, CC-BY data only",
    ),
}
_FTW_BAND_ALIASES = {
    "B04": ("b04", "b4", "red"),
    "B03": ("b03", "b3", "green"),
    "B02": ("b02", "b2", "blue"),
    "B08": ("b08", "b8", "nir"),
}


def ftw_models():
    """The FTW U-Net checkpoints known to GeoLattice as a table (name, version, licence, URL).

    Models trained on mixed licences (standard) include CC-BY-NC-SA data; use the ``_CCBY``
    variants when the output must carry clean IP provenance (commercial use).
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {"model": n, "version": v, "license": lic, "description": d, "url": FTW_RELEASE_URL + p}
            for n, (p, v, lic, d) in FTW_MODELS.items()
        ]
    ).set_index("model")


def _ftw_cache_dir(cache_dir: str | Path | None) -> Path:
    import os

    root = (
        cache_dir or os.environ.get("GEOLATTICE_FTW_CACHE") or Path.home() / ".cache" / "geolattice" / "ftw"
    )
    return Path(root).expanduser()


def ftw_checkpoint(
    model: str | Path = "FTW_PRUE_EFNET_B5",
    *,
    cache_dir: str | Path | None = None,
    download: bool = False,
) -> Path:
    """Local path of an FTW checkpoint.

    ``model`` is either a path to a local ``.ckpt`` file (returned as is) or a registry name
    from :func:`ftw_models`, looked up in ``cache_dir`` (default ``$GEOLATTICE_FTW_CACHE`` or
    ``~/.cache/geolattice/ftw``). A missing registry checkpoint is downloaded from the FTW GitHub
    release only when ``download=True``; otherwise ``FileNotFoundError`` names the URL.
    """
    path = Path(model).expanduser()
    if path.suffix in (".ckpt", ".pt", ".pth") or path.is_file():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    name = str(model)
    if name not in FTW_MODELS:
        raise ValueError(
            f"Unknown FTW model {name!r}; choose from {list(FTW_MODELS)} or pass a .ckpt path"
        )
    release_path = FTW_MODELS[name][0]
    target = _ftw_cache_dir(cache_dir) / Path(release_path).name
    if target.is_file():
        return target
    url = FTW_RELEASE_URL + release_path
    if not download:
        raise FileNotFoundError(
            f"{target} not found. Download it from {url} (or pass download=True) and keep it in "
            f"{target.parent}, or pass the path of a local checkpoint."
        )
    import shutil
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:  # fixed https URL
        shutil.copyfileobj(response, out)
    partial.replace(target)
    return target


def load_ftw_model(
    model: str | Path = "FTW_PRUE_EFNET_B5",
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    download: bool = False,
    strict: bool = True,
):
    """Load an FTW U-Net checkpoint into a GeoLattice ``build_segmenter`` network.

    ``model`` is a local ``.ckpt`` path or an FTW registry name (see :func:`ftw_checkpoint`
    for the cache and ``download``). The architecture (``model``, ``backbone``,
    ``in_channels``, ``num_classes``) is read from the Lightning ``hyper_parameters`` stored in
    the checkpoint, the network is built with
    :func:`geolattice.rsense.classify.image_classification.build_segmenter` (``weights=None``: no
    ImageNet download), and the ``model.``-prefixed state dictionary is loaded into it. The
    checkpoint is read with ``torch.load(weights_only=True)``. The returned network carries
    ``ftw_metadata`` (name, path, backbone, classes, normalisation, licence).
    """
    torch = _torch()
    path = ftw_checkpoint(model, cache_dir=cache_dir, download=download)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    hp = dict(saved.get("hyper_parameters", {})) if isinstance(saved, dict) else {}
    backbone = hp.get("backbone")
    if not backbone:
        raise ValueError(
            f"{path} has no 'backbone' in its hyper_parameters; is it an FTW checkpoint?"
        )
    arch = str(hp.get("model", "unet")).lower()
    in_channels, num_classes = int(hp.get("in_channels", 8)), int(hp.get("num_classes", 3))
    from .classify.image_classification import build_segmenter

    network = build_segmenter(
        arch, in_channels=in_channels, num_classes=num_classes, encoder=backbone, weights=None
    )
    state = {
        k.removeprefix("model."): v for k, v in state.items()
    }  # Lightning keeps the net in .model
    network.load_state_dict(state, strict=strict)
    network.to(_resolve_device(device)).eval()
    name = str(model) if str(model) in FTW_MODELS else path.stem
    info = FTW_MODELS.get(name)
    network.ftw_metadata = {
        "name": name,
        "checkpoint": str(path),
        "architecture": arch,
        "backbone": backbone,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "classes": FTW_CLASSES if num_classes == 3 else {0: "not field", 1: "field"},
        "normalization": FTW_NORMALIZATION,
        "version": info[1] if info else None,
        "license": info[2] if info else None,
    }
    return network


def _band_index(names: Sequence[str], band: str) -> int:
    low = [str(n).lower() for n in names]
    for alias in (band.lower(), *_FTW_BAND_ALIASES[band]):
        if alias in low:
            return low.index(alias)
    raise KeyError(f"band {band} (or {_FTW_BAND_ALIASES[band]}) not found in {list(names)}")


def ftw_stack(window_a, window_b, *, bands: Sequence[str] = FTW_BANDS):
    """Stack two Sentinel-2 windows into the 8-band FTW input ``B04, B03, B02, B08`` (A, then B).

    Each window is a ``(band, y, x)`` xarray DataArray with band names (``B04`` / ``red`` ...),
    an xarray Dataset with one variable per band, or a ``(4, H, W)`` array already in FTW order.
    Values stay in Sentinel-2 L2A digital numbers (the model divides by 3000). Returns an
    xarray DataArray (keeping the georeference of window A) or a NumPy array.
    """

    def pick(w):
        if hasattr(w, "data_vars"):
            names = list(w.data_vars)
            return np.stack(
                [np.asarray(w[names[_band_index(names, b)]].values, np.float32) for b in bands]
            ), w
        if hasattr(w, "dims") and "band" in w.dims:
            names = [str(v) for v in w["band"].values]
            return np.stack(
                [np.asarray(w.isel(band=_band_index(names, b)).values, np.float32) for b in bands]
            ), w
        arr = np.asarray(w, np.float32)
        if arr.ndim != 3 or arr.shape[0] != len(bands):
            raise ValueError("array windows must be (4, H, W) in the order B04, B03, B02, B08")
        return arr, None

    a, tpl = pick(window_a)
    b, _ = pick(window_b)
    if a.shape != b.shape:
        raise ValueError(f"the two windows differ in shape: {a.shape} vs {b.shape}")
    stacked = np.concatenate([a, b])
    if tpl is None:
        return stacked
    import xarray as xr

    ref = (
        tpl[next(iter(tpl.data_vars))] if hasattr(tpl, "data_vars") else tpl.isel(band=0, drop=True)
    )
    names = [f"{b}_A" for b in bands] + [f"{b}_B" for b in bands]
    out = xr.DataArray(
        stacked,
        dims=("band", *ref.dims[-2:]),
        coords={"band": names, **{d: ref[d] for d in ref.dims[-2:]}},
        name="ftw_input",
    )
    if hasattr(ref, "rio") and ref.rio.crs is not None:
        out = out.rio.write_crs(ref.rio.crs).rio.write_transform(ref.rio.transform())
    return out


def predict_ftw(
    image,
    model,
    *,
    normalization: float | None = None,
    resize_factor: int = 2,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
    nodata_value: float = 0.0,
    device: str | None = None,
    precision: str = "32",
) -> np.ndarray:
    """FTW inference: per-class softmax probabilities ``(classes, H, W)``.

    ``image`` is the ``(8, H, W)`` stack of :func:`ftw_stack` (NumPy or xarray), in Sentinel-2
    digital numbers. As in the FTW CLI, values are divided by ``normalization`` (the model's
    ``ftw_metadata``, 3000 for FTW), NaN becomes ``nodata_value``, and each tile is up-sampled
    by ``resize_factor`` (bilinear) before the network; logits are resampled back. Tiles of
    ``tile_size`` pixels overlap by ``overlap`` and are blended with a Hann window (no seams).
    ``tile_size * resize_factor`` must be a multiple of 32 (U-Net depth).
    """
    if resize_factor < 1:
        raise ValueError("resize_factor must be >= 1")
    if (tile_size * resize_factor) % 32:
        raise ValueError("tile_size * resize_factor must be a multiple of 32")
    if overlap < 0 or overlap >= tile_size or batch_size < 1:
        raise ValueError("require 0 <= overlap < tile_size and batch_size >= 1")
    torch = _torch()
    F = torch.nn.functional
    dev = _resolve_device(device)
    meta = getattr(model, "ftw_metadata", {})
    divisor = float(normalization or meta.get("normalization", FTW_NORMALIZATION))
    x = np.asarray(getattr(image, "values", image), np.float32)
    if x.ndim != 3:
        raise ValueError("image must be (bands, H, W)")
    expected = meta.get("in_channels")
    if expected and x.shape[0] != expected:
        raise ValueError(f"the model takes {expected} bands, the image has {x.shape[0]}")
    x = np.nan_to_num(x / divisor, nan=nodata_value)
    _, height, width = x.shape
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    if pad_h or pad_w:
        x = np.pad(
            x,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect" if min(height, width) > 1 else "edge",
        )
    H, W = x.shape[1:]
    windows = [
        (r, c)
        for r in _window_starts(H, tile_size, overlap)
        for c in _window_starts(W, tile_size, overlap)
    ]
    weight = _hann_weight(tile_size)
    acc, wsum = None, np.zeros((H, W), np.float32)
    amp = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    model = model.to(dev).eval()
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            bw = windows[i : i + batch_size]
            t = torch.from_numpy(
                np.ascontiguousarray(
                    np.stack([x[:, r : r + tile_size, c : c + tile_size] for r, c in bw])
                )
            ).to(dev)
            if resize_factor > 1:
                t = F.interpolate(
                    t, scale_factor=resize_factor, mode="bilinear", align_corners=False
                )
            with torch.autocast(
                dev.type, dtype=amp, enabled=amp is not None and dev.type == "cuda"
            ):
                logits = _model_logits(model, t)
            logits = F.interpolate(
                logits.float(), size=(tile_size, tile_size), mode="bilinear", align_corners=False
            )
            probs = torch.softmax(logits, 1).cpu().numpy()
            if acc is None:
                acc = np.zeros((probs.shape[1], H, W), np.float32)
            for (r, c), p in zip(bw, probs):
                acc[:, r : r + tile_size, c : c + tile_size] += p * weight
                wsum[r : r + tile_size, c : c + tile_size] += weight
    return (acc / np.maximum(wsum, 1e-8))[:, :height, :width]


def detect_fields_ftw(
    image,
    model,
    *,
    transform=None,
    crs=None,
    field_class: int = 1,
    boundary_class: int = 2,
    field_threshold: float | None = None,
    min_area_m2: float = 500.0,
    simplify: float = 15.0,
    close_interiors: bool = True,
    **predict_kwargs: Any,
) -> FieldEdgeResult:
    """FTW field polygons from the 8-band two-window stack (see :func:`ftw_stack`).

    Pixels whose most probable class is ``field`` (and, if given, whose field probability is
    at least ``field_threshold``, like ``--softmax_threshold`` of the FTW CLI) form the field
    mask; its 4-connected components become fields, so the predicted boundary class separates
    neighbouring fields. Polygons smaller than ``min_area_m2`` are dropped, simplified with a
    tolerance of ``simplify`` CRS units and, with ``close_interiors``, have their holes removed.
    ``transform`` and ``crs`` default to the georeference of an xarray input. The result's
    ``probability`` is the boundary probability and ``class_probability`` holds all classes.
    """
    from scipy import ndimage as ndi
    from shapely.geometry import Polygon

    if transform is None and hasattr(image, "rio") and image.rio.crs is not None:
        transform, crs = image.rio.transform(), image.rio.crs
    proba = predict_ftw(image, model, **predict_kwargs)
    field = proba.argmax(0) == field_class
    if field_threshold is not None:
        field &= proba[field_class] >= field_threshold
    labels, _ = ndi.label(field, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    boundary = (
        proba[boundary_class] if proba.shape[0] > boundary_class else 1.0 - proba[field_class]
    )
    fields, boundaries = _field_geodataframes(
        labels.astype(np.int32), boundary, transform=transform, crs=crs, min_area_m2=min_area_m2
    )
    if len(fields):
        if close_interiors:
            fields["geometry"] = fields.geometry.apply(
                lambda g: Polygon(g.exterior) if g.geom_type == "Polygon" else g
            )
        if simplify:
            fields["geometry"] = fields.geometry.simplify(simplify, preserve_topology=True)
        fields = fields[~fields.geometry.is_empty].reset_index(drop=True)
        if "area_m2" in fields and crs is not None:  # measure the final (closed, simplified) shapes
            metric = fields
            if fields.crs.is_geographic:
                metric = fields.to_crs(fields.estimate_utm_crs())
            fields["area_m2"] = metric.geometry.area.to_numpy()
            fields["perimeter_m"] = metric.geometry.length.to_numpy()
        boundaries = gpd_boundary_frame(fields, crs)
    return FieldEdgeResult(
        fields, boundaries, boundary.astype(np.float32), labels.astype(np.int32), proba
    )


def to_fiboa(
    fields,
    *,
    collection: str,
    determination_datetime,
    determination_details: str | None = None,
    category: Sequence[str] | None = ("other",),
    id_prefix: str = "",
):
    """Add the fiboa core properties to a field GeoDataFrame (fiboa v0.3 / Vecorel).

    Adds ``id`` (unique within the collection), ``collection``, ``metrics:area`` (m²) and
    ``metrics:perimeter`` (m) computed from the geometry in an equal-distance UTM projection,
    ``determination:method = "auto-imagery"``, ``determination:datetime`` (the date of the
    LAST image used, e.g. window B) and optional ``determination:details`` and ``category``.
    The fiboa schema URL is stored in ``gdf.attrs["schemas"]``; write with
    ``gdf.to_parquet(...)`` and validate with the fiboa tools before publishing.
    """
    import pandas as pd

    out = fields.copy()
    metric = (
        out.to_crs(out.estimate_utm_crs()) if out.crs is not None and out.crs.is_geographic else out
    )
    out["id"] = [f"{id_prefix}{i}" for i in range(1, len(out) + 1)]
    out["collection"] = collection
    out["metrics:area"] = metric.geometry.area.to_numpy()
    out["metrics:perimeter"] = metric.geometry.length.to_numpy()
    out["determination:method"] = "auto-imagery"
    when = pd.Timestamp(determination_datetime)
    out["determination:datetime"] = (
        when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    )
    if determination_details:
        out["determination:details"] = determination_details
    if category:
        out["category"] = [list(category)] * len(out)
    out.attrs["schemas"] = {collection: ["https://fiboa.org/specification/v0.3.0/schema.yaml"]}
    first = ["id", "collection"]
    return out[first + [c for c in out.columns if c not in first]]


def gpd_boundary_frame(fields, crs):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"field_id": fields["field_id"].to_numpy()},
        geometry=fields.geometry.boundary.values,
        crs=crs,
    )


__all__ = [
    "FTW_BANDS",
    "FTW_CLASSES",
    "FTW_MODELS",
    "FieldEdgeResult",
    "available_models",
    "boundary_to_instances",
    "build_edge_model",
    "classical_edges",
    "detect_delineate_anything",
    "detect_field_edges",
    "detect_field_edges_from_raster",
    "detect_fields_ftw",
    "detect_sam2",
    "dihedral_augment",
    "edge_metrics",
    "ftw_checkpoint",
    "ftw_models",
    "ftw_stack",
    "load_delineate_anything",
    "load_edge_model",
    "load_ftw_model",
    "load_sam2",
    "predict_boundary_probability",
    "predict_ftw",
    "rasterize_boundaries",
    "select_threshold",
    "tile_pairs",
    "to_fiboa",
]
