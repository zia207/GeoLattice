"""Semantic segmentation models.

``build_segmentation_model`` uses segmentation-models-pytorch (U-Net, U-Net++, DeepLabV3+,
FPN, MAnet, PSPNet, SegFormer...) when installed, and otherwise a built-in U-Net so the
package works with plain PyTorch.
"""

from __future__ import annotations

from typing import Any

from .._torch import require_torch


def _unet_cls():
    torch = require_torch()
    nn = torch.nn

    class _Block(nn.Module):
        def __init__(self, cin, cout, dropout=0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.Dropout2d(dropout),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True))

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        """Plain U-Net (Ronneberger et al. 2015) for any number of input bands."""

        def __init__(self, in_channels: int = 4, classes: int = 2, base: int = 32, depth: int = 4,
                     dropout: float = 0.1):
            super().__init__()
            ch = [base * 2**i for i in range(depth + 1)]
            self.downs = nn.ModuleList()
            c = in_channels
            for co in ch[:-1]:
                self.downs.append(_Block(c, co))
                c = co
            self.bottom = _Block(ch[-2], ch[-1], dropout)
            self.ups = nn.ModuleList()
            self.dec = nn.ModuleList()
            for co in reversed(ch[:-1]):
                self.ups.append(nn.ConvTranspose2d(co * 2, co, 2, stride=2))
                self.dec.append(_Block(co * 2, co))
            self.head = nn.Conv2d(ch[0], classes, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for d in self.downs:
                x = d(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottom(x)
            for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != s.shape[-2:]:
                    x = nn.functional.interpolate(x, size=s.shape[-2:], mode="bilinear",
                                                  align_corners=False)
                x = dec(torch.cat([x, s], 1))
            return self.head(x)

    return UNet


def UNet(in_channels: int = 4, classes: int = 2, **kwargs: Any):
    """Built-in U-Net instance (no extra dependencies)."""
    return _unet_cls()(in_channels, classes, **kwargs)


def build_segmentation_model(arch: str = "unet", *, in_channels: int = 4, classes: int = 2,
                             encoder: str = "resnet34", weights: str | None = "imagenet",
                             **kwargs: Any):
    """Create a segmentation network.

    ``arch``: ``unet``, ``unetplusplus``, ``deeplabv3plus``, ``fpn``, ``manet``, ``pspnet``,
    ``segformer`` (via segmentation-models-pytorch), or ``unet_basic`` (built-in).
    Pretrained RGB encoders are adapted to ``in_channels`` automatically by smp.
    """
    if arch == "unet_basic":
        return UNet(in_channels, classes, **kwargs)
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        if arch == "unet":
            import warnings

            warnings.warn("segmentation-models-pytorch not installed; using the built-in U-Net "
                          "(no pretrained encoder).", UserWarning, stacklevel=2)
            return UNet(in_channels, classes)
        raise ImportError(f"arch={arch!r} needs `pip install segmentation-models-pytorch`") from None
    names = {"unet": "Unet", "unetplusplus": "UnetPlusPlus", "deeplabv3plus": "DeepLabV3Plus",
             "deeplabv3": "DeepLabV3", "fpn": "FPN", "manet": "MAnet", "pspnet": "PSPNet",
             "linknet": "Linknet", "segformer": "Segformer", "upernet": "UPerNet"}
    return getattr(smp, names[arch.lower()])(encoder_name=encoder, encoder_weights=weights,
                                             in_channels=in_channels, classes=classes, **kwargs)
