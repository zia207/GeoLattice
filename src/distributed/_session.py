"""Dask cluster lifecycle for :mod:`geolattice.distributed`.

:func:`start` returns a :class:`Session` wrapping a ``dask.distributed.Client``. Once a
session is active, every lazy operation in this package runs on its workers. Without a
session, Dask's local threaded scheduler is used, which still runs on all CPU cores.

Kinds
-----
``local``      worker processes on this machine (CPU).
``cuda``       one worker per GPU on this machine (``dask-cuda``).
``auto``       ``cuda`` when a GPU and ``dask-cuda`` are present, else ``local``.
``slurm`` / ``pbs``   HPC batch jobs through ``dask-jobqueue``.
``kubernetes`` Dask workers as pods through the ``dask-kubernetes`` operator
               (``image=``, ``resources=`` ... are passed to ``KubeCluster``).
``address``    connect to a running scheduler, e.g. ``start("address", address="tcp://host:8786")``.
"""

from __future__ import annotations

from typing import Any, Literal

from ._backend import gpu_count, has

Kind = Literal["auto", "local", "cuda", "slurm", "pbs", "kubernetes", "address"]

_ACTIVE: Session | None = None


class Session:
    """An active cluster and client. Use as a context manager to shut down on exit."""

    def __init__(self, client: Any, cluster: Any, kind: str):
        self.client = client
        self.cluster = cluster
        self.kind = kind

    @property
    def dashboard_link(self) -> str | None:
        return getattr(self.client, "dashboard_link", None)

    @property
    def is_gpu(self) -> bool:
        return self.kind == "cuda"

    def scale(self, n_workers: int) -> None:
        """Set a fixed number of workers (not available when connected by address)."""
        if self.cluster is None:
            raise RuntimeError("Cannot scale a cluster this session did not create")
        self.cluster.scale(n_workers)

    def adapt(self, minimum: int = 0, maximum: int | None = None, **kwargs: Any) -> None:
        """Let the cluster add and remove workers with load."""
        if self.cluster is None:
            raise RuntimeError("Cannot adapt a cluster this session did not create")
        self.cluster.adapt(minimum=minimum, maximum=maximum, **kwargs)

    def status(self) -> dict[str, Any]:
        """Worker count, threads, memory and GPU flag."""
        info = self.client.scheduler_info()
        workers = info.get("workers", {})
        return {
            "kind": self.kind,
            "workers": len(workers),
            "threads": sum(w.get("nthreads", 0) for w in workers.values()),
            "memory_limit": sum(w.get("memory_limit", 0) or 0 for w in workers.values()),
            "dashboard": self.dashboard_link,
        }

    def close(self) -> None:
        global _ACTIVE
        try:
            self.client.close()
        finally:
            if self.cluster is not None:
                self.cluster.close()
            if _ACTIVE is self:
                _ACTIVE = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Session kind={self.kind!r} dashboard={self.dashboard_link!r}>"


def start(kind: Kind = "auto", *, n_workers: int | None = None,
          threads_per_worker: int | None = None, memory_limit: str | None = "auto",
          dashboard: bool = True, address: str | None = None, **kwargs: Any) -> Session:
    """Start (or connect to) a Dask cluster and make it the active session.

    Parameters
    ----------
    kind : see module docstring.
    n_workers : workers to start; for ``cuda`` defaults to one per visible GPU, for
        ``slurm``/``pbs`` the number of jobs to request.
    threads_per_worker, memory_limit, dashboard : passed to ``LocalCluster``.
    address : scheduler address for ``kind="address"``.
    kwargs : forwarded to the cluster class (``LocalCUDACluster``, ``SLURMCluster`` ...).
    """
    global _ACTIVE
    from dask.distributed import Client, LocalCluster

    if _ACTIVE is not None:
        _ACTIVE.close()

    if kind == "auto":
        kind = "cuda" if gpu_count() > 0 and has("dask_cuda") else "local"

    cluster = None
    if kind == "address":
        if not address:
            raise ValueError("kind='address' needs address='tcp://host:port'")
        client = Client(address, **kwargs)
    elif kind == "local":
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                               memory_limit=memory_limit,
                               dashboard_address=":8787" if dashboard else None, **kwargs)
        client = Client(cluster)
    elif kind == "cuda":
        try:
            from dask_cuda import LocalCUDACluster
        except ImportError as e:
            raise ImportError("kind='cuda' needs dask-cuda: pip install geolattice[distributed-gpu]") from e
        cluster = LocalCUDACluster(n_workers=n_workers,
                                   dashboard_address=":8787" if dashboard else None, **kwargs)
        client = Client(cluster)
    elif kind in ("slurm", "pbs"):
        try:
            import dask_jobqueue
        except ImportError as e:
            raise ImportError(f"kind={kind!r} needs dask-jobqueue: pip install geolattice[distributed-hpc]") from e
        cls = dask_jobqueue.SLURMCluster if kind == "slurm" else dask_jobqueue.PBSCluster
        cluster = cls(**kwargs)
        if n_workers:
            cluster.scale(jobs=n_workers)
        client = Client(cluster)
    elif kind == "kubernetes":
        try:
            from dask_kubernetes.operator import KubeCluster
        except ImportError as e:
            raise ImportError("kind='kubernetes' needs dask-kubernetes: "
                              "pip install geolattice[distributed-hpc]") from e
        kwargs.setdefault("name", "geolattice")
        cluster = KubeCluster(n_workers=n_workers or 1, **kwargs)
        client = Client(cluster)
    else:
        raise ValueError(f"Unknown kind {kind!r}; use 'auto', 'local', 'cuda', 'slurm', 'pbs', "
                         "'kubernetes' or 'address'")

    _ACTIVE = Session(client, cluster, kind)
    return _ACTIVE


def get_session() -> Session | None:
    """The active session, or ``None`` (Dask then uses its local threaded scheduler)."""
    return _ACTIVE


def shutdown() -> None:
    """Close the active session, if any."""
    if _ACTIVE is not None:
        _ACTIVE.close()


def status() -> dict[str, Any]:
    """Status of the active session, or a note that the local scheduler is in use."""
    if _ACTIVE is None:
        import os

        return {"kind": "threads", "workers": 1, "threads": os.cpu_count(), "dashboard": None}
    return _ACTIVE.status()


def dashboard_link() -> str | None:
    return _ACTIVE.dashboard_link if _ACTIVE is not None else None
