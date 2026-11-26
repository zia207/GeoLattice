"""Field-boundary segmentation networks.

FC-DenseNet variants used by the FieldEdge boundary models (layout and checkpoint names
compatible with ``Fieldedge-Detectors/models/fc_densenet.py`` by Zia Ahmed), plus the two
other networks compared by Yang et al. (2020), *Remote Sensing Applications: Society and
Environment* 20, 100413: SegNet (Badrinarayanan et al. 2017, their Table 3) and
FCN-DKConv6 (Persello et al. 2017; Musyoka 2018, their Table 5).
"""

from __future__ import annotations

from .._torch import require_torch

torch = require_torch()
nn = torch.nn


def _conv2d(in_ch: int, out_ch: int, kernel_size: int, *, bias: bool = True):
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)


class _PreactConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv = _conv2d(in_ch, out_ch, kernel_size)
        self.drop = nn.Dropout(dropout_p) if dropout_p else nn.Identity()

    def forward(self, x):
        return self.drop(self.conv(self.relu(self.bn(x))))


class _DenseBlock(nn.Module):
    def __init__(self, in_ch: int, n_layers: int, growth_rate: int, dropout_p: float):
        super().__init__()
        layers = []
        channels = in_ch
        for _ in range(n_layers):
            layers.append(_PreactConv(channels, growth_rate, dropout_p=dropout_p))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_ch = channels
        self.new_ch = growth_rate * n_layers

    def forward(self, x):
        grown = []
        for layer in self.layers:
            features = layer(x)
            grown.append(features)
            x = torch.cat([x, features], dim=1)
        return x, torch.cat(grown, dim=1)


class _TransitionDown(nn.Module):
    def __init__(self, in_ch: int, dropout_p: float):
        super().__init__()
        self.conv = _PreactConv(in_ch, in_ch, kernel_size=1, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class _TransitionUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = x[:, :, : skip.shape[-2], : skip.shape[-1]]
        return torch.cat([x, skip], dim=1)


_PRESETS = {
    "FC-DenseNet56": {"n_pool": 5, "growth_rate": 12, "layers": [4] * 11},
    "FC-DenseNet67": {"n_pool": 5, "growth_rate": 16, "layers": [5] * 11},
    "FC-DenseNet103": {
        "n_pool": 5,
        "growth_rate": 16,
        "layers": [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
    },
}


class FieldEdgeFCDenseNet(nn.Module):
    """Fully convolutional DenseNet (3 input channels by default) returning per-pixel class logits."""

    def __init__(
        self,
        preset: str = "FC-DenseNet103",
        num_classes: int = 2,
        n_filters_first_conv: int = 48,
        dropout_p: float = 0.2,
        in_channels: int = 3,
    ):
        super().__init__()
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose from {tuple(_PRESETS)}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        spec = _PRESETS[preset]
        n_pool, growth_rate, layers = spec["n_pool"], spec["growth_rate"], spec["layers"]
        self.preset_model = preset
        self.first = _conv2d(in_channels, n_filters_first_conv, 3)

        channels = n_filters_first_conv
        self.down_blocks = nn.ModuleList()
        self.down_transitions = nn.ModuleList()
        skip_channels = []
        for index in range(n_pool):
            block = _DenseBlock(channels, layers[index], growth_rate, dropout_p)
            channels = block.out_ch
            skip_channels.append(channels)
            self.down_blocks.append(block)
            self.down_transitions.append(_TransitionDown(channels, dropout_p))

        self.bottleneck = _DenseBlock(channels, layers[n_pool], growth_rate, dropout_p)
        up_channels = self.bottleneck.new_ch
        self.up_transitions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index, skip_ch in enumerate(reversed(skip_channels)):
            kept_channels = growth_rate * layers[n_pool + index]
            self.up_transitions.append(_TransitionUp(up_channels, kept_channels))
            block = _DenseBlock(
                kept_channels + skip_ch, layers[n_pool + index + 1], growth_rate, dropout_p
            )
            self.up_blocks.append(block)
            up_channels = block.new_ch
        self.logits = _conv2d(self.up_blocks[-1].out_ch, num_classes, 1)

    def forward(self, x):
        x = self.first(x)
        skips = []
        for block, transition in zip(self.down_blocks, self.down_transitions):
            x, _ = block(x)
            skips.append(x)
            x = transition(x)
        x, upsample = self.bottleneck(x)
        for skip, transition, block in zip(reversed(skips), self.up_transitions, self.up_blocks):
            x = transition(upsample, skip)
            x, upsample = block(x)
        return self.logits(x)


def build_fieldedge_fcdensenet(
    preset: str = "FC-DenseNet103", num_classes: int = 2, **kwargs
) -> FieldEdgeFCDenseNet:
    """Build an FC-DenseNet model whose state-dict keys match FieldEdge checkpoints."""
    return FieldEdgeFCDenseNet(preset=preset, num_classes=num_classes, **kwargs)


class SegNet(nn.Module):
    """SegNet: a VGG16-style encoder (13 convolutions, 5 max-pools) and a mirrored decoder that
    upsamples with the encoder's max-pooling indices. Dropout follows the three deepest encoder
    and decoder stages (the "central" Bayesian SegNet placement; Yang et al. 2020, Table 3 lists
    dropout without its position). Dropout after every stage stops the network learning thin
    classes. Input height and width must be multiples of 32."""

    _STAGES = ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3))

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout_p: float = 0.5):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        def stage(cin, cout, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.enc = nn.ModuleList()
        c = in_channels
        for cout, n in self._STAGES:
            self.enc.append(stage(c, cout, n))
            c = cout
        self.dec = nn.ModuleList()
        for i, (cout, n) in enumerate(reversed(self._STAGES)):
            nxt = list(reversed(self._STAGES))[i + 1][0] if i + 1 < len(self._STAGES) else 64
            self.dec.append(nn.Sequential(stage(cout, cout, n - 1) if n > 1 else nn.Identity(),
                                          stage(cout, nxt, 1)))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p else nn.Identity()
        self.logits = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        indices, sizes = [], []
        n = len(self.enc)
        for i, block in enumerate(self.enc):
            x = block(x)
            sizes.append(x.shape)
            x, idx = self.pool(x)
            indices.append(idx)
            if i >= n - 3:
                x = self.drop(x)
        for i, (block, idx, size) in enumerate(zip(self.dec, reversed(indices), reversed(sizes))):
            x = block(self.unpool(x, idx, output_size=size))
            if i < 3:
                x = self.drop(x)
        return self.logits(x)


class FCNDKConv6(nn.Module):
    """FCN-DKConv6: six 5x5 dilated convolutions (dilation 1-6, 16 then 32 filters), each with
    batch normalisation and leaky ReLU, then dropout and a 1x1 classifier. No pooling, so the
    output keeps the input resolution (Yang et al. 2020, Table 5)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, width: int = 32,
                 dropout_p: float = 0.5, negative_slope: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        layers, c = [], in_channels
        for d in range(1, 7):
            cout = width // 2 if d == 1 else width
            layers += [nn.Conv2d(c, cout, 5, padding=2 * d, dilation=d, bias=False),
                       nn.BatchNorm2d(cout), nn.LeakyReLU(negative_slope, inplace=True)]
            c = cout
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p else nn.Identity()
        self.logits = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        return self.logits(self.drop(self.features(x)))


__all__ = ["FCNDKConv6", "FieldEdgeFCDenseNet", "SegNet", "build_fieldedge_fcdensenet"]
