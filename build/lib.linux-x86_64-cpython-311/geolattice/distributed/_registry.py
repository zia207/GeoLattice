"""Operation registry: maps an operation name to its CPU and GPU implementations.

Public functions look up their implementation here instead of hard-coding a library,
so an engine (for example cuSpatial) can be added or swapped without changing the API::

    @register("vector.area", "cpu")
    def _area_cpu(part): ...

    impl, used = dispatch("vector.area", compute="gpu")   # falls back to CPU if no GPU impl
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._backend import Compute, resolve_compute

_OPS: dict[str, dict[str, Callable[..., Any]]] = {}
_AUTO_CPU: set[str] = set()  # ops whose GPU version is opt-in: "auto" picks CPU


def register(op: str, compute: str, *, auto: bool = True) -> Callable[[Callable], Callable]:
    """Decorator registering ``fn`` as the ``compute`` (``"cpu"`` or ``"gpu"``) version of ``op``.

    ``auto=False`` on a GPU version means ``compute="auto"`` keeps using the CPU (the GPU
    version is slower on typical inputs); it runs only when ``compute="gpu"`` is asked for.
    """
    if compute not in ("cpu", "gpu"):
        raise ValueError("compute must be 'cpu' or 'gpu'")

    def deco(fn: Callable) -> Callable:
        _OPS.setdefault(op, {})[compute] = fn
        if compute == "gpu" and not auto:
            _AUTO_CPU.add(op)
        return fn

    return deco


def dispatch(op: str, compute: Compute | None = None) -> tuple[Callable[..., Any], str]:
    """Return ``(implementation, "cpu" | "gpu")`` for ``op``.

    The GPU version is chosen only if one is registered and a GPU is visible;
    otherwise the CPU version runs (with a one-time warning when GPU was requested).
    """
    impls = _OPS.get(op)
    if not impls:
        raise KeyError(f"No implementation registered for {op!r}")
    from ._backend import get_defaults

    if (compute or get_defaults().compute) == "auto" and op in _AUTO_CPU:
        return impls["cpu"], "cpu"
    used = resolve_compute(compute, op=op, gpu_available="gpu" in impls)
    return impls[used], used


def list_ops() -> dict[str, list[str]]:
    """Registered operations and the compute targets each supports."""
    return {op: sorted(impls) for op, impls in sorted(_OPS.items())}
