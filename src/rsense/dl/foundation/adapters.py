"""Foundation-model adapters (weights are downloaded on first use)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._torch import require_torch, torch_device

MODELS = {
    "prithvi-eo-2.0-300m": {"source": "terratorch / HF ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                            "bands": ["blue", "green", "red", "nir08", "swir16", "swir22"],
                            "input": "HLS reflectance, 224 px, optional time steps"},
    "prithvi-eo-2.0-600m": {"source": "terratorch / HF ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
                            "bands": ["blue", "green", "red", "nir08", "swir16", "swir22"],
                            "input": "HLS reflectance, 224 px"},
    "satmae": {"source": "timm-style ViT checkpoint (Cong et al. 2022)",
               "bands": "Sentinel-2 10 bands (multispectral variant)", "input": "96-224 px"},
    "clay-v1.5": {"source": "claymodel package / HF made-with-clay/Clay",
                  "bands": "any sensor (wavelength-aware)", "input": "256 px + metadata"},
    "sam-vit-h": {"source": "segment-anything / segment-geospatial", "bands": "RGB",
                  "input": "any size (resized to 1024)"},
}


def list_foundation_models():
    import pandas as pd

    return pd.DataFrame(MODELS).T


class _Adapter:
    def __init__(self):
        self.model = None
        self.device = torch_device()

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self


class Prithvi(_Adapter):
    """NASA/IBM Prithvi-EO-2.0 (HLS-pretrained ViT MAE).

    Uses TerraTorch's backbone registry when available (recommended for fine-tuning);
    ``encode`` returns patch embeddings. ``segmentation_model`` builds a TerraTorch
    encoder-decoder for semantic segmentation.
    """

    def __init__(self, variant: str = "prithvi_eo_v2_300", *, pretrained: bool = True,
                 bands: Sequence[str] | None = None, num_frames: int = 1):
        super().__init__()
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except ImportError as e:
            raise ImportError("Prithvi needs `pip install terratorch`") from e
        kwargs = {"pretrained": pretrained, "num_frames": num_frames}
        if bands:
            kwargs["bands"] = [b.upper() for b in bands]
        self.variant = variant
        self.model = BACKBONE_REGISTRY.build(variant, **kwargs).to(self.device).eval()

    def encode(self, x):
        torch = require_torch()
        with torch.no_grad():
            feats = self.model(x.to(self.device))
        return feats[-1] if isinstance(feats, (list, tuple)) else feats

    @staticmethod
    def segmentation_model(num_classes: int, *, variant: str = "prithvi_eo_v2_300",
                           bands: Sequence[str] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1",
                                                   "SWIR_2"), decoder: str = "UperNetDecoder",
                           **kwargs: Any):
        from terratorch.models import EncoderDecoderFactory

        return EncoderDecoderFactory().build_model(
            task="segmentation", backbone=variant, backbone_pretrained=True,
            backbone_bands=list(bands), decoder=decoder, num_classes=num_classes, **kwargs)


class SatMAE(_Adapter):
    """SatMAE (Cong et al. 2022) ViT encoder from a checkpoint path or URL (timm required)."""

    def __init__(self, checkpoint: str, *, model_name: str = "vit_large_patch16_224",
                 in_chans: int = 10, img_size: int = 96):
        super().__init__()
        torch = require_torch()
        try:
            import timm
        except ImportError as e:
            raise ImportError("SatMAE needs `pip install timm`") from e
        self.model = timm.create_model(model_name, pretrained=False, in_chans=in_chans,
                                       img_size=img_size, num_classes=0)
        state = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu") \
            if checkpoint.startswith("http") else torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.load_report = {"missing": missing, "unexpected": unexpected}
        self.model.to(self.device).eval()

    def encode(self, x):
        torch = require_torch()
        with torch.no_grad():
            return self.model.forward_features(x.to(self.device))


class Clay(_Adapter):
    """Clay foundation model (sensor-agnostic MAE). Needs the ``claymodel`` package."""

    def __init__(self, checkpoint: str, *, metadata_path: str | None = None):
        super().__init__()
        try:
            from claymodel.module import ClayMAEModule
        except ImportError as e:
            raise ImportError("Clay needs `pip install git+https://github.com/Clay-foundation/model`"
                              ) from e
        kwargs = {"model_size": "large", "dolls": [16, 32, 64, 128, 256, 768, 1024],
                  "doll_weights": [1, 1, 1, 1, 1, 1, 1], "mask_ratio": 0.0, "shuffle": False}
        if metadata_path:
            kwargs["metadata_path"] = metadata_path
        self.module = ClayMAEModule.load_from_checkpoint(checkpoint, **kwargs)
        self.model = self.module.model.encoder.to(self.device).eval()

    def encode(self, datacube: dict):
        """``datacube``: dict with ``pixels``, ``time``, ``latlon``, ``gsd``, ``waves`` tensors
        (see Clay docs). Returns the class-token embedding per image."""
        torch = require_torch()
        with torch.no_grad():
            batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in datacube.items()}
            unmsk, *_ = self.model(batch)
        return unmsk[:, 0, :]


class SAM(_Adapter):
    """Segment Anything for georeferenced rasters (delegates to ``geolattice.rsense.uav.sam_segment``)."""

    def __init__(self, model: str = "vit_h"):
        super().__init__()
        self.model_type = model

    def segment(self, image_path: str, out_path: str = "sam_masks.tif", **kwargs: Any) -> dict:
        from ...uav.segment import sam_segment

        return sam_segment(image_path, out_path, model=self.model_type, **kwargs)
