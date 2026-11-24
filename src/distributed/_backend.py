"""Capability detection and compute resolution for :mod:`geolattice.distributed`.

Every public function takes ``compute="auto" | "cpu" | "gpu"``. :func:`resolve_compute`
turns that into ``"cpu"`` or ``"gpu"``, falling back to CPU with a one-time warning
when the GPU stack is missing. Device moves reuse :mod:`geolattice.rsense.backend`, imported
lazily so ``import geolattice.distributed`` stays light.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, replace
from typing import Any, Literal

Compute = Literal["auto", "cpu", "gpu"]

_OPTIONAL = (
    "dask", "distributed", "dask_geopandas", "geopandas", "pyarrow", "rioxarray",
    "odc.stac", "cupy", "cudf", "dask_cudf", "dask_cuda", "cuspatial", "cugraph",
    "cuml", "dask_jobqueue", "sedona",
)
_warned: set[str] = set()


@dataclass(frozen=True)
class Defaults:
    """Package-wide defaults; change with :func:`set_defaults`."""

    compute: Compute = "auto"
    npartitions: int | None = None
    chunks: Any = "auto"


_DEFAULTS = Defaults()


def set_defaults(**kwargs: Any) -> Defaults:
    """Set defaults used when a function is called without ``compute``/``chunks``/``npartitions``."""
    global _DEFAULTS
    unknown = set(kwargs) - {"compute", "npartitions", "chunks"}
    if unknown:
        raise TypeError(f"Unknown default(s): {sorted(unknown)}")
    _DEFAULTS = replace(_DEFAULTS, **kwargs)
    return _DEFAULTS


def get_defaults() -> Defaults:
    return _DEFAULTS


def has(module: str) -> bool:
    """True if ``module`` is installed.

    Uses ``importlib.util.find_spec`` instead of importing, so checking for cuDF or cuML does
    not trigger their import-time side effects (see :func:`rapids`).
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def rapids(module: str) -> Any:
    """Import a RAPIDS module (``cudf``, ``cuml``, ``cugraph`` ...) without letting it
    replace CuPy's memory allocator.

    Importing cuDF or cuML calls ``cupy.cuda.set_allocator(rmm_cupy_allocator)``. Unless an
    RMM pool was configured, every CuPy allocation then becomes a ``cudaMalloc`` (several
    times slower for chunked work) and memory already cached in CuPy's own pool is stranded.
    The allocator in place before the import is restored afterwards, so a pool chosen by
    the user (for example by dask-cuda with ``rmm_pool_size``) is kept.
    """
    import cupy as cp

    before = cp.cuda.get_allocator()
    try:
        return importlib.import_module(module)
    finally:
        cp.cuda.set_allocator(before)


def gpu_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def capabilities() -> dict[str, Any]:
    """Report which optional engines are importable and how many CUDA GPUs are visible."""
    caps: dict[str, Any] = {m: has(m) for m in _OPTIONAL}
    caps["gpus"] = gpu_count()
    if caps["gpus"]:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        caps["gpu_memory_free"] = int(free)
        caps["gpu_memory_total"] = int(total)
    return caps


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=4)


def resolve_compute(compute: Compute | None = None, *, op: str | None = None,
                    gpu_available: bool | None = None) -> Literal["cpu", "gpu"]:
    """Resolve ``compute`` to ``"cpu"`` or ``"gpu"``.

    ``"auto"`` picks GPU when CuPy sees a device. Asking for ``"gpu"`` without one
    falls back to CPU with a single ``RuntimeWarning`` per operation.
    ``gpu_available`` lets an operation say it has no GPU implementation.
    """
    compute = compute or _DEFAULTS.compute
    if compute not in ("auto", "cpu", "gpu"):
        raise ValueError(f"compute must be 'auto', 'cpu' or 'gpu', got {compute!r}")
    if compute == "cpu":
        return "cpu"
    ok = gpu_count() > 0 if gpu_available is None else (gpu_available and gpu_count() > 0)
    if ok:
        return "gpu"
    if compute == "gpu":
        what = f"{op}: " if op else ""
        _warn_once(f"gpu-fallback:{op}", f"{what}GPU requested but not available for this "
                   "operation; running on CPU.")
    return "cpu"


# --------------------------------------------------------------------------- data movement
def to_gpu(obj: Any) -> Any:
    """Move a NumPy/Dask array or DataArray to CUDA (Dask stays lazy, chunk by chunk)."""
    from ..rsense.backend import to_device

    return to_device(obj, "cuda")


def to_cpu(obj: Any) -> Any:
    """Move a CuPy/Dask-CuPy array or DataArray back to host memory."""
    from ..rsense.backend import to_cpu as _to_cpu

    return _to_cpu(obj)


def is_gpu(obj: Any) -> bool:
    """True if ``obj`` (array, Dask array or DataArray) holds CuPy data."""
    from ..rsense.backend import is_gpu_array

    return is_gpu_array(getattr(obj, "data", obj))


def is_lazy(obj: Any) -> bool:
    """True for Dask collections (including Dask-backed DataArrays and dask-geopandas frames)."""
    import dask

    return dask.is_dask_collection(obj)


def compute(*objs: Any, **kwargs: Any) -> Any:
    """``dask.compute`` that returns a single object when given one."""
    import dask

    out = dask.compute(*objs, **kwargs)
    return out[0] if len(objs) == 1 else out


def persist(*objs: Any, **kwargs: Any) -> Any:
    """``dask.persist``: keep results in cluster memory, returning still-lazy handles."""
    import dask

    out = dask.persist(*objs, **kwargs)
    return out[0] if len(objs) == 1 else out
