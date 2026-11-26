"""Numerical precision policy and mixed-precision helpers.

Array operations default to ``float32``, which halves memory versus ``float64``
with no meaningful loss for reflectance or backscatter. Deep-learning code can
use :func:`autocast` for ``float16``/``bfloat16`` mixed precision.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any

import numpy as np

from ._core import get_backend

_DTYPE: contextvars.ContextVar[str] = contextvars.ContextVar("pyrsense_dtype", default="float32")
_ALLOWED = ("float16", "bfloat16", "float32", "float64")


def set_precision(dtype: str = "float32") -> None:
    """Set the default floating dtype for array computations (``float32`` or ``float64``).

    ``float16``/``bfloat16`` are accepted for deep learning via :func:`autocast`; array
    math keeps ``float32`` in that case, because NumPy has no bfloat16 and
    float16 loses too much precision for index and filter arithmetic.
    """
    if dtype not in _ALLOWED:
        raise ValueError(f"dtype must be one of {_ALLOWED}")
    _DTYPE.set(dtype)


def get_precision() -> str:
    return _DTYPE.get()


def compute_dtype() -> np.dtype:
    """dtype used for array math under the current policy."""
    return np.dtype("float64" if _DTYPE.get() == "float64" else "float32")


def as_compute_dtype(arr: Any):
    """Cast integer or mismatched float data to the compute dtype (no copy if already right)."""
    dt = compute_dtype()
    return arr if getattr(arr, "dtype", None) == dt else arr.astype(dt)


@contextlib.contextmanager
def autocast(dtype: str | None = None, enabled: bool = True) -> Iterator[None]:
    """PyTorch mixed-precision context on the active backend.

    Defaults to ``bfloat16`` on CPU/CUDA (if supported) and ``float16`` on MPS.
    A no-op if PyTorch is not installed or ``enabled`` is False.
    """
    try:
        import torch
    except ImportError:
        yield
        return
    backend = get_backend()
    device_type = {"cuda": "cuda", "mps": "mps"}.get(backend.name, "cpu")
    if dtype is None:
        if device_type == "mps" or device_type == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = "float16"
        else:
            dtype = "bfloat16"
    with torch.autocast(device_type=device_type, dtype=getattr(torch, dtype), enabled=enabled):
        yield
