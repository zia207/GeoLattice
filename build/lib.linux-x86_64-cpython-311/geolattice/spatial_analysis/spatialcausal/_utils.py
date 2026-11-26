"""Shared helpers: input coercion, learners, and weights interoperability."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.base import clone, is_classifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV

from geolattice.spatial_statistics.core.distance import coords_from_geometry
from geolattice.spatial_statistics.weights import KNN, W


# ----------------------------------------------------------------------------
# Array coercion
# ----------------------------------------------------------------------------
def as_2d(X, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_1d(y, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_binary(t, name: str = "treatment") -> np.ndarray:
    arr = as_1d(t, name)
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(f"{name} must be binary (0/1)")
    return arr.astype(int)


def as_coords(coords) -> np.ndarray:
    """Accept an (n, 2) array, a GeoSeries, or a GeoDataFrame."""
    if hasattr(coords, "geometry"):
        return coords_from_geometry(coords)
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be an (n, 2) array or a GeoDataFrame")
    return arr


def check_lengths(**arrays) -> int:
    lengths = {k: len(v) for k, v in arrays.items() if v is not None}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")
    return next(iter(lengths.values()))


# ----------------------------------------------------------------------------
# Spatial weights interoperability
# ----------------------------------------------------------------------------
def as_sparse_weights(w, n: Optional[int] = None, row_standardize: bool = False) -> sparse.csr_matrix:
    """CSR matrix from a GeoLattice ``W``, a libpysal W, a scipy sparse matrix, or an ndarray.

    The diagonal is always zeroed (a unit is never its own neighbour).
    """
    if isinstance(w, W):
        mat = w.sparse
    elif hasattr(w, "sparse") and not sparse.issparse(w):  # libpysal-like
        mat = w.sparse
    elif sparse.issparse(w):
        mat = w
    else:
        mat = sparse.csr_matrix(np.asarray(w, dtype=float))
    mat = sparse.lil_matrix(mat, dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("weights matrix must be square")
    if n is not None and mat.shape[0] != n:
        raise ValueError(f"weights are {mat.shape[0]}x{mat.shape[0]} but data has n={n}")
    mat.setdiag(0.0)
    mat = sparse.csr_matrix(mat)
    mat.eliminate_zeros()
    if row_standardize:
        rs = np.asarray(mat.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        mat = sparse.csr_matrix(sparse.diags(1.0 / rs) @ mat)
    return mat


def default_weights(coords: np.ndarray, k: int = 8) -> W:
    """Row-standardised k-nearest-neighbour weights from PySpatialStats."""
    return KNN(coords, k=min(k, len(coords) - 1), transform="row")


def spatial_lag(w_sparse: sparse.csr_matrix, X: np.ndarray) -> np.ndarray:
    return np.asarray(w_sparse @ X)


# ----------------------------------------------------------------------------
# Learners (names mirror PyCausalML: "ridge", "lasso", "rf", "gbr", ...)
# ----------------------------------------------------------------------------
_REGRESSORS = {
    "linear": lambda n, rs: LinearRegression(),
    "ridge": lambda n, rs: RidgeCV(alphas=np.logspace(-3, 3, 13)),
    "lasso": lambda n, rs: LassoCV(cv=3, random_state=rs),
    "rf": lambda n, rs: RandomForestRegressor(n_estimators=n, min_samples_leaf=5, random_state=rs, n_jobs=-1),
    "gbr": lambda n, rs: GradientBoostingRegressor(n_estimators=n, max_depth=3, random_state=rs),
    "hgb": lambda n, rs: HistGradientBoostingRegressor(max_iter=n, random_state=rs),
}
_CLASSIFIERS = {
    "logistic": lambda n, rs: LogisticRegression(C=1.0, max_iter=2000),
    "linear": lambda n, rs: LogisticRegression(C=1.0, max_iter=2000),
    "ridge": lambda n, rs: LogisticRegression(C=1.0, max_iter=2000),
    "lasso": lambda n, rs: LogisticRegression(C=1.0, penalty="l1", solver="liblinear", max_iter=2000),
    "rf": lambda n, rs: RandomForestClassifier(n_estimators=n, min_samples_leaf=5, random_state=rs, n_jobs=-1),
    "gbc": lambda n, rs: GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=rs),
    "gbr": lambda n, rs: GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=rs),
    "hgb": lambda n, rs: HistGradientBoostingClassifier(max_iter=n, random_state=rs),
}


def make_learner(spec: Any, task: str = "regression", n_estimators: int = 100, random_state: int = 0):
    """Build a fresh sklearn learner from a name, or clone a supplied estimator."""
    if hasattr(spec, "fit"):
        est = clone(spec)
        if task == "classification" and not is_classifier(est):
            raise ValueError("a classifier is required for a propensity model")
        return est
    table = _REGRESSORS if task == "regression" else _CLASSIFIERS
    if spec not in table:
        raise ValueError(f"Unknown learner '{spec}'. Choose from {sorted(table)} or pass an estimator.")
    return table[spec](n_estimators, random_state)


def clip_propensity(p: np.ndarray, trim: float) -> np.ndarray:
    return np.clip(p, trim, 1.0 - trim)
