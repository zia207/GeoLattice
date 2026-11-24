"""Benchmarking, CPU-vs-GPU equivalence checks, dispatch explanations, memory and profiling.

* :func:`benchmark` times a ``geolattice.distributed`` function on each compute target,
  materialising lazy results and synchronising the GPU so timings are real.
* :func:`check_equivalence` runs a function on CPU and GPU and compares the outputs
  (arrays, DataArrays, Series, DataFrames, GeoDataFrames, result dataclasses).
* :func:`explain` says which implementation an operation would use and why.
* :func:`memory_report` gathers host, GPU and cluster-worker memory.
* :func:`profile` records a Dask performance report (HTML) or a per-task timing table.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._backend import gpu_count, is_lazy


def _materialize(obj: Any) -> Any:
    """Compute lazy results and bring GPU data to host."""
    if is_lazy(obj):
        obj = obj.compute()
    if hasattr(obj, "data") and type(getattr(obj, "data", None)).__module__.startswith("cupy"):
        obj = obj.copy(data=obj.data.get())
    if type(obj).__module__.startswith("cupy"):
        obj = obj.get()
    return obj


def _sync() -> None:
    if gpu_count():
        import cupy as cp

        cp.cuda.Device().synchronize()


def benchmark(func: Callable[..., Any], *args: Any, computes: Sequence[str] = ("cpu", "gpu"),
              repeats: int = 3, warmup: int = 1, **kwargs: Any) -> pd.DataFrame:
    """Time ``func(*args, compute=c, **kwargs)`` for each ``c`` in ``computes``.

    Each target runs ``warmup`` untimed calls (JIT and kernel compilation, caches) and
    then ``repeats`` timed ones. Targets without a GPU are skipped. Returns a table with
    best, mean and standard deviation in seconds and the speed-up relative to the first
    target.
    """
    rows = []
    for c in computes:
        if c == "gpu" and not gpu_count():
            continue
        for _ in range(warmup):
            _materialize(func(*args, compute=c, **kwargs))
            _sync()
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            _materialize(func(*args, compute=c, **kwargs))
            _sync()
            times.append(time.perf_counter() - t)
        rows.append({"compute": c, "best_s": min(times), "mean_s": float(np.mean(times)),
                     "std_s": float(np.std(times)), "repeats": repeats})
    df = pd.DataFrame(rows).set_index("compute")
    if len(df):
        df["speedup"] = df["best_s"].iloc[0] / df["best_s"]
    return df


@dataclass
class EquivalenceReport:
    """Outcome of :func:`check_equivalence`; ``details`` lists every compared part."""

    ok: bool
    max_abs: float
    max_rel: float
    details: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _compare(a: Any, b: Any, rtol: float, atol: float, path: str, out: list) -> None:
    a, b = _materialize(a), _materialize(b)
    if hasattr(a, "geometry") and hasattr(a, "columns"):  # GeoDataFrame
        if len(a) != len(b):
            out.append((path, False, np.inf, np.inf, f"length {len(a)} vs {len(b)}"))
            return
        ok = bool(a.geometry.geom_equals_exact(b.geometry, tolerance=atol).all())
        out.append((f"{path}.geometry", ok, 0.0, 0.0, "geom_equals_exact"))
        a, b = pd.DataFrame(a.drop(columns=a.geometry.name)), pd.DataFrame(b.drop(columns=b.geometry.name))
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns):
            out.append((path, False, np.inf, np.inf, "columns differ"))
            return
        for c in a.columns:
            _compare(a[c], b[c], rtol, atol, f"{path}[{c!r}]", out)
        return
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            _compare(getattr(a, f.name), getattr(b, f.name), rtol, atol, f"{path}.{f.name}", out)
        return
    if isinstance(a, dict):
        for k in a:
            _compare(a[k], b.get(k), rtol, atol, f"{path}[{k!r}]", out)
        return
    if isinstance(a, (tuple, list)) and not np.isscalar(a):
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, rtol, atol, f"{path}[{i}]", out)
        return
    if a is None or b is None:
        out.append((path, a is None and b is None, 0.0, 0.0, "None"))
        return
    x = np.asarray(getattr(a, "values", a))
    y = np.asarray(getattr(b, "values", b))
    if x.shape != y.shape:
        out.append((path, False, np.inf, np.inf, f"shape {x.shape} vs {y.shape}"))
        return
    if not (np.issubdtype(x.dtype, np.number) or x.dtype == bool):
        ok = bool(np.array_equal(x, y))
        out.append((path, ok, 0.0, 0.0, "exact"))
        return
    x, y = x.astype(float), y.astype(float)
    both_nan = np.isnan(x) & np.isnan(y)
    diff = np.where(both_nan, 0, np.abs(x - y))
    diff = np.where(np.isnan(diff), np.inf, diff)
    rel = diff / np.maximum(np.abs(y), 1e-300)
    ok = bool(np.all(diff <= atol + rtol * np.abs(np.nan_to_num(y))))
    out.append((path, ok, float(diff.max(initial=0)), float(np.where(diff > 0, rel, 0).max(initial=0)),
                "allclose"))


def check_equivalence(func: Callable[..., Any], *args: Any, rtol: float = 1e-5, atol: float = 1e-8,
                      computes: Sequence[str] = ("cpu", "gpu"), raise_on_mismatch: bool = False,
                      **kwargs: Any) -> EquivalenceReport:
    """Run ``func`` with each compute target and compare the results to the first one.

    Use it to validate a GPU kernel against the CPU path on your own data (chunk edges,
    NaNs, mixed geometries) before trusting it at scale.
    """
    if "gpu" in computes and not gpu_count():
        return EquivalenceReport(True, 0.0, 0.0, ["no GPU visible; nothing to compare"])
    results = [func(*args, compute=c, **kwargs) for c in computes]
    parts: list = []
    for c, r in zip(computes[1:], results[1:]):
        _compare(r, results[0], rtol, atol, f"{c} vs {computes[0]}", parts)
    ok = all(p[1] for p in parts)
    rep = EquivalenceReport(ok, max((p[2] for p in parts), default=0.0),
                            max((p[3] for p in parts), default=0.0),
                            [f"{'ok ' if p[1] else 'BAD'} {p[0]}: {p[4]} max_abs={p[2]:.3g}"
                             for p in parts])
    if raise_on_mismatch and not ok:
        raise AssertionError("CPU and GPU results differ:\n" + "\n".join(rep.details))
    return rep


def explain(op: str, compute: str | None = None) -> dict[str, Any]:
    """Which implementation ``op`` would use for ``compute`` and why."""
    from ._backend import get_defaults
    from ._registry import _AUTO_CPU, _OPS
    from ._session import get_session

    impls = _OPS.get(op)
    if impls is None:
        close = sorted(k for k in _OPS if op.split(".")[-1] in k)
        raise KeyError(f"Unknown operation {op!r}" + (f"; did you mean {close}?" if close else ""))
    requested = compute or get_defaults().compute
    gpus = gpu_count()
    has_gpu_impl = "gpu" in impls
    if requested == "cpu":
        used, why = "cpu", "CPU requested"
    elif requested == "auto" and op in _AUTO_CPU:
        used, why = "cpu", "GPU version is opt-in (slower on typical inputs); pass compute='gpu'"
    elif not has_gpu_impl:
        used, why = "cpu", "no GPU implementation registered"
    elif gpus == 0:
        used, why = "cpu", "no CUDA GPU visible"
    else:
        used, why = "gpu", f"GPU implementation and {gpus} GPU(s) available"
    s = get_session()
    impl = impls[used]
    return {"op": op, "requested": requested, "uses": used, "reason": why,
            "implementation": f"{impl.__module__}.{impl.__qualname__}",
            "targets": sorted(impls), "gpus": gpus,
            "session": s.kind if s is not None else "local threads"}


def memory_report() -> dict[str, Any]:
    """Host, process, GPU (CuPy pool) and cluster-worker memory, in GB."""
    from ..rsense.backend import memory_report as _rs_report
    from ._session import get_session

    rep = dict(_rs_report())
    try:
        import psutil

        rep["process_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
    except ImportError:
        pass
    if gpu_count():
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        rep.update(gpu_total_gb=round(total / 1e9, 2), gpu_free_gb=round(free / 1e9, 2))
    s = get_session()
    if s is not None:
        workers = s.client.scheduler_info().get("workers", {})
        rep["workers"] = {
            addr: {"memory_gb": round(w.get("metrics", {}).get("memory", 0) / 1e9, 3),
                   "limit_gb": round((w.get("memory_limit") or 0) / 1e9, 3)}
            for addr, w in workers.items()}
    return rep


def profile(func: Callable[..., Any], *args: Any, path: str | None = None,
            **kwargs: Any) -> tuple[Any, Any]:
    """Run ``func`` and record where the time went.

    With an active session, writes a Dask performance report (HTML, needs bokeh) to
    ``path`` (default ``geolattice-profile.html``) and returns ``(result, path)``. Without
    one, uses ``dask.diagnostics.Profiler`` and returns ``(result, table)`` with total
    seconds and call count per task type.
    """
    from ._session import get_session

    if get_session() is not None:
        from dask.distributed import performance_report

        path = path or "geolattice-profile.html"
        with performance_report(filename=path):
            res = _materialize(func(*args, **kwargs))
        return res, path
    from dask.diagnostics import Profiler

    with Profiler() as prof:
        res = _materialize(func(*args, **kwargs))
    rows = [{"task": (r.key[0] if isinstance(r.key, tuple) else str(r.key)).split("-")[0],
             "seconds": r.end_time - r.start_time} for r in prof.results]
    table = (pd.DataFrame(rows).groupby("task")["seconds"].agg(["sum", "count"])
             .sort_values("sum", ascending=False) if rows else pd.DataFrame(columns=["sum", "count"]))
    return res, table
