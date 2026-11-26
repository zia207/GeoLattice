"""Dask clusters on CPU or GPU, sized from the active backend."""

from __future__ import annotations

from typing import Any, Literal


def start_cluster(kind: Literal["auto", "local", "cuda"] = "auto", *, n_workers: int | None = None,
                  threads_per_worker: int | None = None, memory_limit: str | None = "auto",
                  dashboard: bool = True, **kwargs: Any):
    """Start a Dask ``Client``.

    ``cuda`` launches one worker per GPU (``dask-cuda`` ``LocalCUDACluster``) so chunked
    CuPy arrays are processed on all GPUs; ``local`` uses processes on the CPU. ``auto``
    chooses ``cuda`` when the PyRSense backend is CUDA and dask-cuda is installed.
    """
    from dask.distributed import Client, LocalCluster

    from ..backend import get_backend

    if kind == "auto":
        kind = "local"
        if get_backend().array_gpu:
            try:
                import dask_cuda  # noqa: F401

                kind = "cuda"
            except ImportError:
                pass
    if kind == "cuda":
        from dask_cuda import LocalCUDACluster

        cluster = LocalCUDACluster(n_workers=n_workers, **kwargs)
    else:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                               memory_limit=memory_limit,
                               dashboard_address=":8787" if dashboard else None, **kwargs)
    client = Client(cluster)
    return client


def rechunk_for(obj, *, spatial: int = 2048, time: int | None = None):
    """Rechunk a Dataset/DataArray for spatial-first (``time=None`` → -1 keeps full time
    per chunk, good for pixel time series) processing."""
    chunks = {}
    for d in obj.dims:
        if d in ("x", "y", "lon", "lat"):
            chunks[d] = spatial
        elif d == "time":
            chunks[d] = time if time is not None else -1
    return obj.chunk(chunks)
