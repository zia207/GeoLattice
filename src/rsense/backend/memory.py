"""GPU/host memory utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import _cuda_device_count, _try_import, free_gpu_memory_bytes, get_backend


def clear_cache() -> None:
    """Release cached GPU memory held by CuPy's pool and PyTorch's allocator."""
    if _cuda_device_count():
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    torch = _try_import("torch")
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def memory_report() -> dict[str, Any]:
    """Host and active-GPU memory usage in GB."""
    report: dict[str, Any] = {"backend": str(get_backend())}
    psutil = _try_import("psutil")
    if psutil is not None:
        vm = psutil.virtual_memory()
        report.update(host_total_gb=round(vm.total / 1e9, 2), host_free_gb=round(vm.available / 1e9, 2))
    free = free_gpu_memory_bytes()
    if free is not None:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        report.update(gpu_free_gb=round(free / 1e9, 2),
                      cupy_pool_used_gb=round(pool.used_bytes() / 1e9, 3))
    return report


def recommend_chunks(shape: tuple[int, ...], dtype: Any = "float32", *, n_layers: int = 1,
                     target_mb: float | None = None, spatial_dims: int = 2) -> tuple[int, ...]:
    """Suggest Dask chunks: full extent on leading dims, square spatial chunks sized to
    ~``target_mb`` (default 128 MB on CPU, 5% of free memory on GPU)."""
    itemsize = np.dtype(dtype).itemsize
    if target_mb is None:
        free = free_gpu_memory_bytes()
        target_mb = 128 if free is None else max(free * 0.05 / 1e6, 32)
    lead = int(np.prod(shape[:-spatial_dims])) if len(shape) > spatial_dims else 1
    pixels = target_mb * 1e6 / (itemsize * max(lead, 1) * max(n_layers, 1))
    edge = int(max(256, (pixels ** 0.5) // 256 * 256))
    return tuple(shape[:-spatial_dims]) + tuple(min(edge, s) for s in shape[-spatial_dims:])
