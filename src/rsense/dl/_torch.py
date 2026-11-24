"""Lazy PyTorch import and device resolution from the PyRSense backend."""

from __future__ import annotations


def require_torch():
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("Deep learning modules need PyTorch: `pip install geolattice[rsense-dl]` "
                          "(see pytorch.org for the right CUDA build)") from e


def torch_device(spec: str | None = None):
    """torch.device matching the active PyRSense backend (or ``spec``)."""
    torch = require_torch()
    from ..backend import get_backend

    if spec:
        return torch.device(spec)
    b = get_backend()
    if b.name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{b.device_id}")
    if b.name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
