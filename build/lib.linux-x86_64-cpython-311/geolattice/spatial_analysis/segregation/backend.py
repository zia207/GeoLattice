"""CPU/GPU array dispatch for the segregation package.

A small, self-contained NumPy/CuPy switch -- deliberately not imported from
``geolattice.rsense.backend``, since that would pull the (optional, heavy) rsense
extra and its eager ``rioxarray`` import into every use of this module.
Everything here falls back to NumPy silently when CuPy is unavailable or no
GPU backend has been selected, so segregation indices run identically on
CPU-only installs.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

_BACKEND = "cpu"


def available_backends() -> list[str]:
    """``["cpu"]``, plus ``"cuda"`` if CuPy and a GPU are both importable."""
    backends = ["cpu"]
    try:
        import cupy  # noqa: F401

        backends.append("cuda")
    except ImportError:
        pass
    return backends


def set_backend(name: str) -> str:
    """Switch the module-level default backend (``"cpu"`` or ``"cuda"``). Returns the name."""
    if name not in available_backends():
        raise ValueError(f"Backend {name!r} is not available; choose one of {available_backends()}")
    global _BACKEND
    _BACKEND = name
    return _BACKEND


def get_backend() -> str:
    return _BACKEND


@contextlib.contextmanager
def device(name: str):
    """Temporarily switch the backend within a ``with`` block."""
    previous = get_backend()
    set_backend(name)
    try:
        yield name
    finally:
        set_backend(previous)


def get_xp(backend: str | None = None):
    """Return the array module (``numpy`` or ``cupy``) for ``backend`` (or the active one)."""
    backend = backend or _BACKEND
    if backend == "cuda":
        import cupy as cp

        return cp
    return np


def array_module(arr: Any):
    """Return the array module that owns ``arr`` (NumPy, or CuPy if it is a CuPy array)."""
    if type(arr).__module__.startswith("cupy"):
        import cupy as cp

        return cp
    return np


def is_gpu_array(arr: Any) -> bool:
    return array_module(arr) is not np


def to_cpu(arr: Any) -> np.ndarray:
    """Move an array to host (NumPy) memory; a no-op for arrays already there."""
    if hasattr(arr, "get") and type(arr).__module__.startswith("cupy"):
        return arr.get()
    return np.asarray(arr)


def to_device(arr: Any, backend: str | None = None):
    """Move ``arr`` to ``backend`` (or the active backend)."""
    backend = backend or _BACKEND
    if backend == "cuda":
        import cupy as cp

        return cp.asarray(arr)
    return to_cpu(arr) if is_gpu_array(arr) else np.asarray(arr)
