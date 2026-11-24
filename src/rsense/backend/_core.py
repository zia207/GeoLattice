"""Compute-backend selection and array dispatch.

PyRSense runs the same code on CPU (NumPy) or NVIDIA GPU (CuPy). Deep-learning
modules additionally use PyTorch, which can target CUDA or Apple MPS.

The active backend is held in a :class:`contextvars.ContextVar`, so it is safe
to switch backends inside threads or async tasks with :func:`device`.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

BackendName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class Backend:
    """An active compute backend."""

    name: Literal["cpu", "cuda", "mps"]
    device_id: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.name in ("cuda", "mps")

    @property
    def array_gpu(self) -> bool:
        """True if array math (not just DL) runs on GPU. MPS has no CuPy equivalent."""
        return self.name == "cuda"

    def __str__(self) -> str:
        return "cpu" if self.name == "cpu" else f"{self.name}:{self.device_id}"


_CPU = Backend("cpu")
_ACTIVE: contextvars.ContextVar[Backend] = contextvars.ContextVar("pyrsense_backend", default=_CPU)


# --------------------------------------------------------------------------- detection
def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001  (ImportError, or CUDA libs failing at import)
        return None


@lru_cache(maxsize=1)
def _cuda_device_count() -> int:
    cp = _try_import("cupy")
    if cp is None:
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # noqa: BLE001
        return 0


@lru_cache(maxsize=1)
def _mps_available() -> bool:
    torch = _try_import("torch")
    try:
        return bool(torch is not None and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def available_backends() -> list[str]:
    """Return the backends usable on this machine, e.g. ``['cpu', 'cuda']``."""
    out = ["cpu"]
    if _cuda_device_count() > 0:
        out.append("cuda")
    if _mps_available():
        out.append("mps")
    return out


def _parse(spec: str) -> tuple[str, int]:
    name, _, idx = spec.lower().partition(":")
    return name, int(idx) if idx else 0


def _resolve(spec: BackendName | str) -> Backend:
    name, idx = _parse(spec)
    if name == "auto":
        if _cuda_device_count() > 0:
            return Backend("cuda", idx)
        if _mps_available():
            return Backend("mps", idx)
        return _CPU
    if name == "cpu":
        return _CPU
    if name == "cuda":
        n = _cuda_device_count()
        if n == 0:
            warnings.warn(
                "CUDA backend requested but CuPy/CUDA is not available; falling back to CPU. "
                "Install with `pip install geolattice[rsense-gpu]`.",
                RuntimeWarning,
                stacklevel=3,
            )
            return _CPU
        if idx >= n:
            raise ValueError(f"cuda:{idx} requested but only {n} CUDA device(s) found")
        return Backend("cuda", idx)
    if name == "mps":
        if not _mps_available():
            warnings.warn("MPS backend requested but not available; falling back to CPU.",
                          RuntimeWarning, stacklevel=3)
            return _CPU
        return Backend("mps", 0)
    raise ValueError(f"Unknown backend {spec!r}; use 'auto', 'cpu', 'cuda[:N]' or 'mps'")


# --------------------------------------------------------------------------- selection
def set_backend(spec: BackendName | str = "auto") -> Backend:
    """Set the global compute backend and return it.

    ``"auto"`` picks CUDA, then MPS, then CPU. Requesting an unavailable GPU
    falls back to CPU with a ``RuntimeWarning`` rather than failing.
    """
    backend = _resolve(spec)
    _ACTIVE.set(backend)
    if backend.name == "mps":
        warnings.warn(
            "MPS accelerates PyTorch (deep learning) only; array operations run on CPU.",
            UserWarning,
            stacklevel=2,
        )
    return backend


def get_backend() -> Backend:
    """Return the currently active backend."""
    return _ACTIVE.get()


@contextlib.contextmanager
def device(spec: BackendName | str) -> Iterator[Backend]:
    """Temporarily switch backend::

        with rs.device("cuda:1"):
            ndvi = rs.optical.index(ds, "NDVI")
    """
    token = _ACTIVE.set(_resolve(spec))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


# --------------------------------------------------------------------------- dispatch
def get_xp(backend: Backend | None = None):
    """Return the array module (``numpy`` or ``cupy``) for the backend."""
    backend = backend or get_backend()
    if backend.array_gpu:
        import cupy as cp

        cp.cuda.Device(backend.device_id).use()
        return cp
    return np


def array_module(arr: Any):
    """Return the array module that owns ``arr`` (works for NumPy, CuPy and Dask chunks)."""
    if type(arr).__module__.startswith("cupy"):
        import cupy as cp

        return cp
    meta = getattr(arr, "_meta", None)  # dask array
    if meta is not None and type(meta).__module__.startswith("cupy"):
        import cupy as cp

        return cp
    return np


def is_gpu_array(arr: Any) -> bool:
    return array_module(arr) is not np


def _to_device_array(arr: Any, backend: Backend):
    if not backend.array_gpu:
        return _to_cpu_array(arr)
    import cupy as cp

    if hasattr(arr, "map_blocks") and hasattr(arr, "chunks"):  # dask
        return arr.map_blocks(cp.asarray, dtype=arr.dtype, meta=cp.empty((0,) * arr.ndim, arr.dtype))
    with cp.cuda.Device(backend.device_id):
        return cp.asarray(arr)


def _to_cpu_array(arr: Any):
    if hasattr(arr, "map_blocks") and hasattr(arr, "chunks"):  # dask
        if is_gpu_array(arr):
            return arr.map_blocks(lambda b: b.get(), dtype=arr.dtype,
                                  meta=np.empty((0,) * arr.ndim, arr.dtype))
        return arr
    if hasattr(arr, "get") and type(arr).__module__.startswith("cupy"):
        return arr.get()
    return np.asarray(arr)


def _map_xarray(obj: Any, fn):
    import xarray as xr

    if isinstance(obj, xr.DataArray):
        return obj.copy(data=fn(obj.data))
    if isinstance(obj, xr.Dataset):
        return obj.map(lambda v: v.copy(data=fn(v.data)), keep_attrs=True)
    return None


def to_device(obj: Any, spec: BackendName | str | None = None):
    """Move an array, DataArray or Dataset to the active (or given) backend.

    Coordinates stay on the host; only data variables move. Dask arrays stay
    lazy, with each chunk converted when computed.
    """
    backend = _resolve(spec) if spec else get_backend()
    mapped = _map_xarray(obj, lambda a: _to_device_array(a, backend))
    return mapped if mapped is not None else _to_device_array(obj, backend)


def to_cpu(obj: Any):
    """Move an array, DataArray or Dataset back to host (NumPy) memory."""
    mapped = _map_xarray(obj, _to_cpu_array)
    return mapped if mapped is not None else _to_cpu_array(obj)


# --------------------------------------------------------------------------- reporting
def info() -> dict[str, Any]:
    """Report hardware and library availability, useful for bug reports."""
    import platform

    report: dict[str, Any] = {
        "pyrsense_backend": str(get_backend()),
        "available_backends": available_backends(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for lib in ("numpy", "xarray", "dask", "rasterio", "rioxarray", "cupy", "cucim", "torch"):
        mod = _try_import(lib)
        report[lib] = getattr(mod, "__version__", "installed") if mod else None
    gpus = []
    if _cuda_device_count():
        import cupy as cp

        for i in range(_cuda_device_count()):
            with cp.cuda.Device(i):
                free, total = cp.cuda.runtime.memGetInfo()
                props = cp.cuda.runtime.getDeviceProperties(i)
                gpus.append({
                    "id": i,
                    "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                    "memory_total_gb": round(total / 1e9, 2),
                    "memory_free_gb": round(free / 1e9, 2),
                })
    report["gpus"] = gpus
    return report


def free_gpu_memory_bytes(backend: Backend | None = None) -> int | None:
    """Free memory on the active CUDA device, or ``None`` on CPU."""
    backend = backend or get_backend()
    if not backend.array_gpu:
        return None
    import cupy as cp

    with cp.cuda.Device(backend.device_id):
        return int(cp.cuda.runtime.memGetInfo()[0])
