"""Object detection models (torchvision): Faster R-CNN, RetinaNet, FCOS, Mask R-CNN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._torch import require_torch
from .layers import adapt_first_conv


def build_detector(arch: str = "faster_rcnn", *, num_classes: int = 2, in_channels: int = 3,
                   pretrained: bool = True, image_mean: Sequence[float] | None = None,
                   image_std: Sequence[float] | None = None, min_size: int = 512,
                   max_size: int = 1024, **kwargs: Any):
    """Create a torchvision detector (``num_classes`` includes background for R-CNNs).

    ``arch``: ``faster_rcnn``, ``retinanet``, ``fcos``, ``mask_rcnn``. For ``in_channels``
    other than 3 the first backbone convolution is adapted from the RGB weights and the
    normalization statistics default to zero mean / unit std (pass your band stats).
    """
    require_torch()
    import torchvision.models.detection as det

    w = "DEFAULT" if pretrained else None
    builders = {
        "faster_rcnn": (det.fasterrcnn_resnet50_fpn_v2, det.faster_rcnn.FastRCNNPredictor),
        "retinanet": (det.retinanet_resnet50_fpn_v2, None),
        "fcos": (det.fcos_resnet50_fpn, None),
        "mask_rcnn": (det.maskrcnn_resnet50_fpn_v2, None),
    }
    if arch not in builders:
        raise ValueError(f"Unknown detector {arch!r}; choose from {list(builders)}")
    fn, _ = builders[arch]
    model = fn(weights=w, min_size=min_size, max_size=max_size, **kwargs)
    # replace heads for the requested number of classes
    if arch in ("faster_rcnn", "mask_rcnn"):
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        if arch == "mask_rcnn":
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
    elif arch == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        na = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(256, na, num_classes)
    elif arch == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead

        na = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(256, na, num_classes)
    if in_channels != 3:
        body = model.backbone.body
        body.conv1 = adapt_first_conv(body.conv1, in_channels)
        model.transform.image_mean = list(image_mean or [0.0] * in_channels)
        model.transform.image_std = list(image_std or [1.0] * in_channels)
    elif image_mean is not None:
        model.transform.image_mean, model.transform.image_std = list(image_mean), list(image_std)
    return model
