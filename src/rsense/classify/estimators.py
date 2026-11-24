"""Unified classifiers: scikit-learn on CPU, cuML/XGBoost-GPU on CUDA, with the same API."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import get_backend, to_cpu
from ..optical.bands import as_dataset

ModelName = Literal["rf", "svm", "xgb", "knn", "logistic", "kmeans", "gbm", "mlp"]


def _cpu_model(name: str, task: str, **kw):
    from sklearn import cluster, ensemble, linear_model, neighbors, neural_network, svm

    if name == "rf":
        return (ensemble.RandomForestClassifier if task == "classification"
                else ensemble.RandomForestRegressor)(n_jobs=kw.pop("n_jobs", -1), **kw)
    if name == "svm":
        return svm.SVC(probability=kw.pop("probability", True), **kw) if task == "classification" \
            else svm.SVR(**kw)
    if name == "knn":
        return (neighbors.KNeighborsClassifier if task == "classification"
                else neighbors.KNeighborsRegressor)(**kw)
    if name == "logistic":
        return linear_model.LogisticRegression(max_iter=kw.pop("max_iter", 1000), **kw)
    if name == "gbm":
        return (ensemble.HistGradientBoostingClassifier if task == "classification"
                else ensemble.HistGradientBoostingRegressor)(**kw)
    if name == "mlp":
        return (neural_network.MLPClassifier if task == "classification"
                else neural_network.MLPRegressor)(max_iter=kw.pop("max_iter", 500), **kw)
    if name == "kmeans":
        return cluster.KMeans(n_init=kw.pop("n_init", 10), **kw)
    raise ValueError(f"Unknown model {name!r}")


def _gpu_model(name: str, task: str, **kw):
    cuml = importlib.import_module("cuml")
    if name == "rf":
        return (cuml.ensemble.RandomForestClassifier if task == "classification"
                else cuml.ensemble.RandomForestRegressor)(**kw)
    if name == "svm":
        return cuml.svm.SVC(probability=kw.pop("probability", True), **kw) \
            if task == "classification" else cuml.svm.SVR(**kw)
    if name == "knn":
        return (cuml.neighbors.KNeighborsClassifier if task == "classification"
                else cuml.neighbors.KNeighborsRegressor)(**kw)
    if name == "logistic":
        return cuml.linear_model.LogisticRegression(**kw)
    if name == "kmeans":
        return cuml.cluster.KMeans(**kw)
    raise ValueError(f"No cuML GPU implementation for {name!r}")


class Classifier:
    """Pixel/object classifier or regressor with automatic CPU/GPU engine selection.

    ``model``: ``rf``, ``svm``, ``xgb``, ``knn``, ``logistic``, ``gbm``, ``mlp`` or
    ``kmeans`` (unsupervised). On the CUDA backend cuML is used when installed (falls back
    to scikit-learn otherwise); ``xgb`` uses XGBoost with ``device="cuda"``.

    Example::

        clf = rs.classify.Classifier("rf", n_estimators=500)
        X, y = rs.classify.extract_samples(ds, training_points, "class")
        clf.fit(X, y)
        landcover = clf.predict_raster(ds)
    """

    def __init__(self, model: ModelName = "rf", *, task: Literal["classification", "regression"]
                 = "classification", engine: Literal["auto", "cpu", "gpu"] = "auto", **params: Any):
        self.model_name, self.task, self.params = model, task, params
        backend = get_backend()
        use_gpu = engine == "gpu" or (engine == "auto" and backend.array_gpu)
        self.engine = "cpu"
        if model == "xgb":
            import xgboost as xgb

            dev = "cuda" if use_gpu else "cpu"
            cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
            self.model = cls(device=dev, tree_method="hist", **params)
            self.engine = f"xgboost-{dev}"
        elif use_gpu:
            try:
                self.model = _gpu_model(model, task, **params)
                self.engine = "cuml"
            except (ImportError, ValueError):
                self.model = _cpu_model(model, task, **params)
        else:
            self.model = _cpu_model(model, task, **params)
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self._label_map: dict | None = None

    # -- fitting
    def fit(self, X: Any, y: Any = None, **fit_kw: Any) -> Classifier:
        Xa, self.feature_names_ = _as_matrix(X)
        if self.model_name == "kmeans":
            self.model.fit(Xa)
            return self
        y = np.asarray(y)
        if self.task == "classification" and (self.model_name == "xgb" or self.engine == "cuml"):
            # these engines need labels 0..K-1
            self.classes_ = np.unique(y)
            self._label_map = {c: i for i, c in enumerate(self.classes_)}
            y = np.vectorize(self._label_map.get)(y)
        self.model.fit(Xa, y, **fit_kw)
        if self.classes_ is None and hasattr(self.model, "classes_"):
            self.classes_ = np.asarray(to_cpu(self.model.classes_))
        return self

    def _decode(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(to_cpu(p))
        if self._label_map is not None:
            return self.classes_[p.astype(int)]
        return p

    def predict(self, X: Any) -> np.ndarray:
        Xa, _ = _as_matrix(X, self.feature_names_)
        return self._decode(self.model.predict(Xa))

    def predict_proba(self, X: Any) -> np.ndarray:
        Xa, _ = _as_matrix(X, self.feature_names_)
        return np.asarray(to_cpu(self.model.predict_proba(Xa)))

    # -- rasters
    _string_classes = False

    def predict_raster(self, obj: xr.Dataset | xr.DataArray, *, proba: bool = False,
                       chunk_rows: int = 512) -> xr.DataArray | xr.Dataset:
        """Predict every valid pixel of a feature Dataset (variables = features).

        Processes ``chunk_rows`` rows at a time to bound memory. Integer labels are kept;
        string labels are written as class indices (``-1`` = nodata) with the names in
        ``attrs["classes"]``. With ``proba=True`` returns a Dataset of class probabilities.
        """
        ds = as_dataset(obj)
        feats = self.feature_names_ or list(ds.data_vars)
        ydim, xdim = ds[feats[0]].dims[-2:]
        H, W = ds.sizes[ydim], ds.sizes[xdim]
        n_out = len(self.classes_) if proba else 1
        self._string_classes = (self.task == "classification" and self.classes_ is not None
                                and not np.issubdtype(np.asarray(self.classes_).dtype, np.number))
        if proba or self.task == "regression":
            out = np.full((n_out, H, W), np.nan, dtype="float32")
        else:
            out = np.full((1, H, W), -1, dtype="int32")
        for r0 in range(0, H, chunk_rows):
            blk = ds[feats].isel({ydim: slice(r0, r0 + chunk_rows)})
            Xb = np.stack([np.asarray(to_cpu(blk[f]).values, dtype="float32").ravel() for f in feats], 1)
            ok = np.all(np.isfinite(Xb), axis=1)
            if not ok.any():
                continue
            h = blk.sizes[ydim]
            if proba:
                p = self.predict_proba(pd.DataFrame(Xb[ok], columns=feats))
                tmp = np.full((Xb.shape[0], n_out), np.nan, dtype="float32")
                tmp[ok] = p
                out[:, r0:r0 + h] = tmp.T.reshape(n_out, h, W)
            else:
                pr = self.predict(pd.DataFrame(Xb[ok], columns=feats))
                if self._string_classes:
                    pr = np.searchsorted(self.classes_, pr)
                tmp = np.full(Xb.shape[0], out.dtype.type(-1 if out.dtype.kind == "i" else np.nan))
                tmp[ok] = pr
                out[0, r0:r0 + h] = tmp.reshape(h, W)
        tpl = ds[feats[0]]
        if proba:
            return xr.Dataset({f"p_{c}": tpl.copy(data=out[i]) for i, c in enumerate(self.classes_)})
        res = tpl.copy(data=out[0]).rename("prediction")
        if self._string_classes:
            res = res.astype("int16").assign_attrs(classes=[str(c) for c in self.classes_],
                                                   nodata=-1)
        return res

    def feature_importance(self) -> pd.Series | None:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None or self.feature_names_ is None:
            return None
        return pd.Series(np.asarray(to_cpu(imp)), index=self.feature_names_).sort_values(ascending=False)

    def __repr__(self) -> str:
        return f"Classifier({self.model_name!r}, task={self.task!r}, engine={self.engine!r})"


def _as_matrix(X: Any, names: Sequence[str] | None = None) -> tuple[Any, list[str] | None]:
    if isinstance(X, pd.DataFrame):
        if names is not None:
            X = X[list(names)]
        return X.to_numpy(dtype="float32"), list(X.columns)
    a = np.asarray(to_cpu(X), dtype="float32")
    return a, list(names) if names is not None else None


def extract_samples(obj: xr.Dataset | xr.DataArray, samples, label_col: str | None = None, *,
                    features: Sequence[str] | None = None, max_per_polygon: int | None = None,
                    seed: int = 0) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Extract feature values at training samples.

    ``samples``: GeoDataFrame of points (nearest pixel) or polygons (all pixels inside,
    optionally subsampled to ``max_per_polygon``). Returns ``(X DataFrame with x/y columns
    dropped, y labels)``. Rows with NaN features are removed. The pixel coordinates are
    kept in ``X.attrs["coords"]`` for spatial cross-validation.
    """
    ds = as_dataset(obj)
    feats = list(features or ds.data_vars)
    s = samples.to_crs(ds.rio.crs) if samples.crs and ds.rio.crs else samples
    geom_type = s.geometry.geom_type.iloc[0]
    ydim, xdim = ds[feats[0]].dims[-2:]
    if geom_type == "Point":
        xs = xr.DataArray(s.geometry.x.values, dims="s")
        ys = xr.DataArray(s.geometry.y.values, dims="s")
        vals = {f: np.asarray(to_cpu(ds[f].sel({xdim: xs, ydim: ys}, method="nearest").values))
                for f in feats}
        df = pd.DataFrame(vals)
        df["_x"], df["_y"] = s.geometry.x.values, s.geometry.y.values
        labels = s[label_col].to_numpy() if label_col else None
    else:
        from rasterio.features import rasterize

        rng = np.random.default_rng(seed)
        ids = rasterize(((g, i + 1) for i, g in enumerate(s.geometry)),
                        out_shape=(ds.sizes[ydim], ds.sizes[xdim]), transform=ds.rio.transform(),
                        fill=0, dtype="int32")
        rows, cols = np.nonzero(ids)
        pid = ids[rows, cols] - 1
        if max_per_polygon:
            keep = np.zeros(rows.size, bool)
            for p in np.unique(pid):
                idx = np.flatnonzero(pid == p)
                keep[rng.choice(idx, min(max_per_polygon, idx.size), replace=False)] = True
            rows, cols, pid = rows[keep], cols[keep], pid[keep]
        df = pd.DataFrame({f: np.asarray(to_cpu(ds[f]).values)[rows, cols] for f in feats})
        df["_x"] = ds[xdim].values[cols]
        df["_y"] = ds[ydim].values[rows]
        df["_polygon"] = pid
        labels = s[label_col].to_numpy()[pid] if label_col else None
    ok = np.all(np.isfinite(df[feats].to_numpy(dtype=float)), axis=1)
    df = df[ok].reset_index(drop=True)
    coords = df[["_x", "_y"]].to_numpy()
    groups = df["_polygon"].to_numpy() if "_polygon" in df else None
    X = df[feats]
    X.attrs["coords"] = coords
    if groups is not None:
        X.attrs["groups"] = groups
    return X, (labels[ok] if labels is not None else None)


def cluster(obj: xr.Dataset | xr.DataArray, n_clusters: int = 8, *, sample: int = 200_000,
            seed: int = 0, **kw: Any) -> xr.DataArray:
    """Unsupervised k-means classification of a feature Dataset (fit on a random sample)."""
    ds = as_dataset(obj)
    feats = list(ds.data_vars)
    X = np.stack([np.asarray(to_cpu(ds[f]).values, dtype="float32").ravel() for f in feats], 1)
    ok = np.all(np.isfinite(X), axis=1)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    fit_idx = rng.choice(idx, min(sample, idx.size), replace=False)
    clf = Classifier("kmeans", n_clusters=n_clusters, random_state=seed, **kw)
    clf.fit(X[fit_idx])
    lab = np.full(X.shape[0], -1, dtype="int32")
    lab[ok] = np.asarray(to_cpu(clf.model.predict(X[ok])))
    return ds[feats[0]].copy(data=lab.reshape(ds[feats[0]].shape)).rename("cluster")
