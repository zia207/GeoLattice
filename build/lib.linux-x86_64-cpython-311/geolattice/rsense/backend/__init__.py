"""CPU/GPU backend selection, array dispatch, memory and precision management."""

from ._core import (
    Backend,
    array_module,
    available_backends,
    device,
    free_gpu_memory_bytes,
    get_backend,
    get_xp,
    info,
    is_gpu_array,
    set_backend,
    to_cpu,
    to_device,
)
from .dispatch import fft, ndimage, scatter_reduce
from .memory import clear_cache, memory_report, recommend_chunks
from .precision import as_compute_dtype, autocast, compute_dtype, get_precision, set_precision

__all__ = [
    "Backend",
    "array_module",
    "as_compute_dtype",
    "autocast",
    "available_backends",
    "clear_cache",
    "compute_dtype",
    "device",
    "fft",
    "free_gpu_memory_bytes",
    "get_backend",
    "get_precision",
    "get_xp",
    "info",
    "is_gpu_array",
    "memory_report",
    "ndimage",
    "recommend_chunks",
    "scatter_reduce",
    "set_backend",
    "set_precision",
    "to_cpu",
    "to_device",
]
