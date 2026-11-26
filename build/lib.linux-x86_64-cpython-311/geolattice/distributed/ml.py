"""Machine learning on spatial data at scale: training, raster prediction, tiled deep
learning inference over several devices, and multi-process (DDP) training.

* :func:`fit` trains ``"rf"``, ``"xgb"`` or ``"lgbm"`` models (or any scikit-learn
  estimator) on CPU or GPU. With an active cluster session, scikit-learn fits run on
  the cluster through joblib's Dask backend, and XGBoost on Dask arrays uses
  ``xgboost.dask``.
* :func:`predict_raster` applies a fitted model to every pixel of a ``(band, y, x)``
  raster chunk by chunk (NaN pixels stay NaN); the model is sent to workers once.
* :func:`predict_tiles` runs a PyTorch model over a large image in overlapping tiles,
  spreading batches over several GPUs (one model replica per device). With
  ``blend="crop"`` each pixel comes from the centre of a tile, so when ``overlap // 2``
  covers the model's receptive-field radius the result equals whole-image inference
  everywhere except within that radius of the image border (a multi-layer network pads
  each layer's input there, which tiling of a reflect-padded image cannot reproduce).
* :func:`train_ddp` launches ``torch.distributed`` training in one process per device.
"""

from __future__ import annotations

import copy
import os
import socket
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from ._backend import Compute, gpu_count, resolve_compute
from ._registry import dispatch, register

Task = Literal["classification", "regression"]


# --------------------------------------------------------------------------- tabular models
def _make_model(name: str, task: Task, used: str, params: dict) -> tuple[Any, str]:
    """Return ``(estimator, engine)`` for a model name on the resolved device."""
    if name == "xgb":
        import xgboost as xgb

        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        dev = "cuda" if used == "gpu" else "cpu"
        return cls(device=dev, tree_method="hist", **params), f"xgboost-{dev}"
    if name == "lgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(verbose=-1, **params), "lightgbm-cpu"
    if name == "rf":
        if used == "gpu":
            try:
                from ._backend import rapids

                ens = rapids("cuml.ensemble")
                cls = ens.RandomForestClassifier if task == "classification" else ens.RandomForestRegressor
                return cls(**params), "cuml"
            except ImportError:
                pass
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        params.setdefault("n_jobs", -1)
        return cls(**params), "sklearn"
    raise ValueError("model must be 'rf', 'xgb', 'lgbm' or a scikit-learn style estimator")


def fit(model: Any, X: Any, y: Any, *, task: Task = "classification",
        compute: Compute | None = None, **params: Any) -> Any:
    """Fit ``model`` (a name or an estimator) and return the fitted estimator.

    The chosen engine is recorded on the result as ``geolattice_engine``. Labels for cuML and
    XGBoost classifiers are encoded to ``0..K-1`` and decoded again on prediction by
    :func:`predict_raster` (see ``geolattice_classes``).
    """
    from ._session import get_session

    name = model if isinstance(model, str) else None
    used = resolve_compute(compute, op="ml.fit",
                           gpu_available=name in ("xgb", "rf") if name else False)
    if name:
        est, engine = _make_model(name, task, used, dict(params))
    else:
        est, engine = model, type(model).__module__.split(".")[0]

    lazy_y = hasattr(y, "dask")
    Xa = X.to_numpy() if hasattr(X, "to_numpy") and not hasattr(X, "dask") else X
    ya = y if lazy_y else np.asarray(y)
    classes = None
    if task == "classification" and (engine == "cuml" or engine.startswith("xgboost")):
        # these engines need labels 0..K-1; keep the originals to decode predictions
        if lazy_y:
            classes = np.unique(np.asarray(y.compute()))
            ya = y.map_blocks(lambda b: np.searchsorted(classes, b), dtype=np.int64)
        else:
            classes, ya = np.unique(ya, return_inverse=True)

    session = get_session()
    if name == "xgb" and hasattr(X, "dask") and session is not None:
        import xgboost as xgb
        import xgboost.dask  # noqa: F401  (submodule is not loaded by ``import xgboost``)

        dtrain = xgb.dask.DaskDMatrix(session.client, X, ya)
        rounds = params.pop("n_estimators", 100)
        obj = "multi:softprob" if task == "classification" and len(classes) > 2 else (
            "binary:logistic" if task == "classification" else "reg:squarederror")
        p = {"tree_method": "hist", "device": "cuda" if used == "gpu" else "cpu",
             "objective": obj, **params}
        if obj == "multi:softprob":
            p["num_class"] = len(classes)
        out = xgb.dask.train(session.client, p, dtrain, num_boost_round=rounds)
        booster = out["booster"]
        booster.geolattice_engine, booster.geolattice_classes = "xgboost-dask", classes
        return booster

    if engine == "cuml":
        import cupy as cp

        est.fit(cp.asarray(np.asarray(Xa), dtype=cp.float32),
                cp.asarray(np.asarray(ya), dtype=cp.int32 if task == "classification" else cp.float32))
    elif session is not None and engine == "sklearn":
        import joblib

        with joblib.parallel_backend("dask"):
            est.fit(Xa, ya)
    else:
        est.fit(Xa, ya)
    est.geolattice_engine = engine
    est.geolattice_classes = classes
    return est


def _predict_rows(model: Any, rows: np.ndarray, proba: bool) -> np.ndarray:
    engine = getattr(model, "geolattice_engine", "")
    classes = getattr(model, "geolattice_classes", None)
    if engine == "cuml":
        import cupy as cp

        x = cp.asarray(rows, dtype=cp.float32)
        out = model.predict_proba(x) if proba else model.predict(x)
        out = cp.asnumpy(out) if hasattr(out, "get") or type(out).__module__.startswith("cupy") \
            else np.asarray(out)
    elif engine == "xgboost-cuda":
        import cupy as cp

        x = cp.asarray(rows, dtype=cp.float32)  # same device as the booster: no host copy
        out = model.predict_proba(x) if proba else model.predict(x)
        out = out.get() if hasattr(out, "get") else np.asarray(out)
    elif engine == "xgboost-dask":
        import xgboost as xgb

        out = model.predict(xgb.DMatrix(rows))
        if not proba and classes is not None:
            out = out.argmax(axis=1) if out.ndim == 2 else (out > 0.5).astype(int)
    else:
        out = model.predict_proba(rows) if proba else model.predict(rows)
    out = np.asarray(out)
    if not proba and classes is not None:
        out = classes[out.astype(int)]
    return out


def _predict_block(block: np.ndarray, model: Any, proba: bool, n_out: int) -> np.ndarray:
    b = block.get() if hasattr(block, "get") else np.asarray(block)
    nb, h, w = b.shape
    rows = b.reshape(nb, -1).T
    ok = np.all(np.isfinite(rows), axis=1)
    out = np.full((n_out, h * w), np.nan, dtype=np.float64)
    if ok.any():
        pred = _predict_rows(model, rows[ok], proba)
        out[:, ok] = pred.T if proba else pred.reshape(1, -1)
    return out.reshape(n_out, h, w) if proba else out.reshape(h, w)


register("ml.predict_raster", "cpu")(_predict_block)
register("ml.predict_raster", "gpu")(_predict_block)


def predict_raster(model: Any, stack: Any, *, proba: bool = False, chunks: Any = None,
                   compute: Compute | None = None) -> xr.DataArray:
    """Predict every pixel of ``stack`` (``(band, y, x)`` DataArray, or a Dataset whose
    variables are the bands, in the order the model was trained on).

    Returns a ``(y, x)`` DataArray of predictions, or ``(class, y, x)`` probabilities with
    ``proba=True``. Lazy input stays lazy. GPU models (cuML, XGBoost on CUDA) predict on
    the GPU inside each task.
    """
    import dask

    from .raster import _prepare

    if isinstance(stack, xr.Dataset):
        stack = stack.to_array("band")
    if not isinstance(stack, xr.DataArray) or stack.ndim != 3:
        raise ValueError("stack must be a (band, y, x) DataArray or a Dataset of bands")
    dispatch("ml.predict_raster", compute)
    da, ctx = _prepare(stack, compute="cpu", chunks=chunks, op="ml.predict_raster")
    da = da.chunk({da.dims[0]: -1})
    classes = getattr(model, "geolattice_classes", None)
    if proba:
        n_out = len(classes) if classes is not None else len(getattr(model, "classes_", []))
        if n_out == 0:
            raise ValueError("cannot tell the number of classes; fit with geolattice.distributed.ml.fit")
    else:
        n_out = 1
    model_d = dask.delayed(model, pure=True)
    data = da.data
    if proba:
        out = data.map_blocks(_predict_block, model_d, True, n_out, dtype=np.float64,
                              chunks=((n_out,),) + data.chunks[1:], meta=np.empty((0, 0, 0)))
        res = xr.DataArray(out, dims=("class",) + da.dims[1:],
                           coords={"class": classes if classes is not None else np.arange(n_out),
                                   **{d: da[d] for d in da.dims[1:] if d in da.coords}})
    else:
        out = data.map_blocks(_predict_block, model_d, False, 1, dtype=np.float64, drop_axis=0,
                              meta=np.empty((0, 0)))
        res = xr.DataArray(out, dims=da.dims[1:],
                           coords={d: da[d] for d in da.dims[1:] if d in da.coords})
    res = res.rename("probability" if proba else "prediction")
    return res if ctx.lazy else res.compute()


# --------------------------------------------------------------------------- tiled deep learning
def _devices(devices: str | Sequence[str]) -> list[str]:
    if devices == "auto":
        import torch

        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] or ["cpu"]
    return [devices] if isinstance(devices, str) else list(devices)


def _window(tile: int, margin: int, blend: str) -> np.ndarray:
    """Per-tile blending weights. ``crop`` keeps the central ``tile - 2 * margin`` square.
    ``cosine`` ignores the outer ``margin // 2`` pixels (corrupted by the tile's own edge
    padding), rises with a raised-cosine ramp over the next ``margin - margin // 2`` and
    is flat inside, so overlapping tiles are feathered and every pixel keeps weight."""
    w1 = np.zeros(tile)
    if blend == "crop":
        w1[margin:tile - margin] = 1.0
    else:
        b = margin // 2
        ramp = margin - b
        w1[margin:tile - margin] = 1.0
        if ramp:
            r = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1))
            w1[b:margin] = r
            w1[tile - margin:tile - b] = r[::-1]
    return np.outer(w1, w1)


def predict_tiles(model: Any, image: Any, *, tile: int = 256, overlap: int = 32,
                  batch_size: int = 8, devices: str | Sequence[str] = "auto",
                  blend: Literal["crop", "cosine"] = "crop", amp: bool = False) -> Any:
    """Run a PyTorch ``model`` (``(N, C, h, w) -> (N, K, h, w)``) over a ``(C, H, W)``
    image in overlapping tiles.

    The image is reflect-padded by ``overlap // 2`` and cut into ``tile`` x ``tile``
    windows with stride ``tile - overlap``. Batches go round-robin to one model replica
    per device in ``devices`` (``"auto"`` = every CUDA GPU, else CPU), each in its own
    thread. ``blend="crop"`` keeps each tile's central region, so tile seams leave no
    trace when ``overlap // 2`` covers the receptive field; ``"cosine"`` feathers the
    overlaps (seamless when ``overlap // 4`` covers it), which hides seams from models
    whose outputs drift with tile context.
    ``amp`` enables CUDA autocast (float16). Returns a ``(K, H, W)`` array, or a
    DataArray with dims ``(channel, y, x)`` for DataArray input.
    """
    import torch

    if overlap % 2 or overlap >= tile:
        raise ValueError("overlap must be even and smaller than tile")
    is_da = isinstance(image, xr.DataArray)
    arr = np.asarray(image.values if is_da else image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    c, h, w = arr.shape
    m = overlap // 2
    stride = tile - overlap
    ny = max(1, int(np.ceil((h + 2 * m - tile) / stride)) + 1)
    nx = max(1, int(np.ceil((w + 2 * m - tile) / stride)) + 1)
    ph, pw = (ny - 1) * stride + tile, (nx - 1) * stride + tile
    padded = np.pad(arr, ((0, 0), (m, ph - h - m), (m, pw - w - m)), mode="reflect")
    origins = [(i * stride, j * stride) for i in range(ny) for j in range(nx)]
    batches = [origins[k:k + batch_size] for k in range(0, len(origins), batch_size)]
    devs = _devices(devices)
    model.eval()
    # Never move the caller's model: reuse it only on the device it already lives on.
    home = next((p.device for p in model.parameters()), torch.device("cpu"))
    replicas = {d: model if torch.device(d) == home else copy.deepcopy(model).to(d) for d in devs}
    win = _window(tile, m, blend)
    acc: np.ndarray | None = None
    norm = np.zeros((ph, pw))
    lock = threading.Lock()
    errors: list[BaseException] = []

    def run(dev: str, jobs: list[list[tuple[int, int]]]) -> None:
        nonlocal acc
        try:
            net = replicas[dev]
            for batch in jobs:
                x = torch.from_numpy(np.stack([padded[:, r:r + tile, q:q + tile] for r, q in batch]))
                with torch.inference_mode(), torch.autocast(
                        "cuda", enabled=amp and dev.startswith("cuda")):
                    y = net(x.to(dev)).float().cpu().numpy()
                with lock:
                    if acc is None:
                        acc = np.zeros((y.shape[1], ph, pw))
                    for (r, q), yt in zip(batch, y):
                        acc[:, r:r + tile, q:q + tile] += yt * win
                        norm[r:r + tile, q:q + tile] += win
        except BaseException as e:  # surfaced after join
            errors.append(e)

    threads = [threading.Thread(target=run, args=(d, batches[k::len(devs)]))
               for k, d in enumerate(devs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    out = (acc / np.maximum(norm, 1e-12))[:, m:m + h, m:m + w].astype(np.float32)
    if is_da:
        ydim, xdim = image.dims[-2:]
        return xr.DataArray(out, dims=("channel", ydim, xdim),
                            coords={k: image.coords[k] for k in (ydim, xdim) if k in image.coords},
                            name="prediction")
    return out


# --------------------------------------------------------------------------- multi-process training
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_entry(rank: int, fn: Callable, world: int, backend: str, port: int, results: Any,
               args: tuple) -> None:
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        results[rank] = fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def train_ddp(fn: Callable[..., Any], *, nprocs: int | None = None, backend: str | None = None,
              args: tuple = ()) -> list[Any]:
    """Run ``fn(rank, world_size, *args)`` in ``nprocs`` processes with ``torch.distributed``
    initialised, and return each rank's return value.

    ``nprocs`` defaults to the number of CUDA GPUs (at least 1). ``backend`` defaults to
    ``nccl`` when there is a GPU per process, else ``gloo`` (CPU). ``fn`` must be a
    module-level function (it is pickled into the worker processes). Inside ``fn``, wrap
    the model with :func:`ddp_model` and shard data with
    ``torch.utils.data.DistributedSampler``.
    """
    import torch.multiprocessing as mp

    nprocs = nprocs or max(gpu_count(), 1)
    backend = backend or ("nccl" if gpu_count() >= nprocs else "gloo")
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        mp.start_processes(_ddp_entry, args=(fn, nprocs, backend, _free_port(), results, args),
                           nprocs=nprocs, join=True, start_method="spawn")
        return [results.get(r) for r in range(nprocs)]


def ddp_model(model: Any, rank: int) -> Any:
    """Move ``model`` to this rank's device and wrap it in ``DistributedDataParallel``."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if dist.get_backend() == "nccl":
        model = model.to(f"cuda:{rank}")
        return DistributedDataParallel(model, device_ids=[rank])
    return DistributedDataParallel(model.to(torch.device("cpu")))


def _ddp_selftest(rank: int, world: int) -> dict[str, Any]:
    """All-reduce and one DDP step; used by the test suite and as a smoke test."""
    import torch
    import torch.distributed as dist

    t = torch.tensor([float(rank + 1)])
    if dist.get_backend() == "nccl":
        t = t.cuda()
    dist.all_reduce(t)
    net = ddp_model(torch.nn.Linear(2, 1), rank)
    dev = next(net.parameters()).device
    x = torch.full((4, 2), float(rank + 1), device=dev)
    net(x).sum().backward()
    grad = next(net.parameters()).grad.detach().cpu().numpy().tolist()
    return {"rank": rank, "sum": float(t.item()), "grad": grad, "backend": dist.get_backend()}
