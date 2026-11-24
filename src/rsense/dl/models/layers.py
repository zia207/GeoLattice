"""Utilities for adapting RGB-pretrained networks to multispectral/SAR inputs."""

from __future__ import annotations

from typing import Any


def adapt_first_conv(conv: Any, in_channels: int, *, method: str = "mean"):
    """Return a copy of a ``nn.Conv2d`` accepting ``in_channels``, reusing pretrained RGB weights.

    ``mean``: new channels get the mean RGB filter (scaled to keep activation magnitude);
    ``repeat``: RGB filters are cycled across the new channels.
    """
    import torch
    from torch import nn

    if conv.in_channels == in_channels:
        return conv
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
    with torch.no_grad():
        w = conv.weight
        if method == "repeat":
            idx = torch.arange(in_channels) % w.shape[1]
            nw = w[:, idx] * (w.shape[1] / in_channels)
        else:
            nw = w.mean(1, keepdim=True).repeat(1, in_channels, 1, 1) * (w.shape[1] / in_channels)
            k = min(in_channels, w.shape[1])
            nw[:, :k] = w[:, :k] * (w.shape[1] / in_channels)
        new.weight.copy_(nw)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new
