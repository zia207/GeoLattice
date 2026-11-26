"""Spatial cross-validation: block, cluster and buffered leave-out splitters (scikit-learn API)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _coords(X, coords):
    if coords is not None:
        return np.asarray(coords, float)
    c = getattr(X, "attrs", {}).get("coords")
    if c is None:
        raise ValueError("Pass coords= (n, 2) or use X from extract_samples (has attrs['coords'])")
    return np.asarray(c, float)


class SpatialBlockCV:
    """K-fold over square spatial blocks of side ``block_size`` (map units), assigned to folds
    at random. Reduces optimistic bias from spatial autocorrelation (Roberts et al. 2017)."""

    def __init__(self, n_splits: int = 5, block_size: float = 1000.0, coords=None, seed: int = 0):
        self.n_splits, self.block_size, self.coords, self.seed = n_splits, block_size, coords, seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        c = _coords(X, self.coords)
        bx = np.floor((c[:, 0] - c[:, 0].min()) / self.block_size).astype(int)
        by = np.floor((c[:, 1] - c[:, 1].min()) / self.block_size).astype(int)
        block = bx * 100_003 + by
        ub = np.unique(block)
        rng = np.random.default_rng(self.seed)
        fold_of = dict(zip(ub, rng.permutation(len(ub)) % self.n_splits))
        folds = np.array([fold_of[b] for b in block])
        for k in range(self.n_splits):
            te = np.flatnonzero(folds == k)
            if te.size:
                yield np.flatnonzero(folds != k), te


class SpatialClusterCV:
    """K-fold over k-means clusters of the sample coordinates (spatially contiguous folds)."""

    def __init__(self, n_splits: int = 5, coords=None, seed: int = 0):
        self.n_splits, self.coords, self.seed = n_splits, coords, seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        from sklearn.cluster import KMeans

        c = _coords(X, self.coords)
        lab = KMeans(self.n_splits, n_init=10, random_state=self.seed).fit_predict(c)
        for k in range(self.n_splits):
            yield np.flatnonzero(lab != k), np.flatnonzero(lab == k)


class BufferedKFold:
    """K-fold (random or block) where training samples within ``buffer`` (map units) of any
    test sample are removed (spatial leave-out with a dead zone; Le Rest et al. 2014)."""

    def __init__(self, n_splits: int = 5, buffer: float = 500.0, coords=None, seed: int = 0,
                 block_size: float | None = None):
        self.n_splits, self.buffer, self.coords = n_splits, buffer, coords
        self.seed, self.block_size = seed, block_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        from scipy.spatial import cKDTree

        c = _coords(X, self.coords)
        if self.block_size:
            base = SpatialBlockCV(self.n_splits, self.block_size, c, self.seed).split(X)
        else:
            from sklearn.model_selection import KFold

            base = KFold(self.n_splits, shuffle=True, random_state=self.seed).split(c)
        for tr, te in base:
            tree = cKDTree(c[te])
            d, _ = tree.query(c[tr], k=1)
            yield tr[d > self.buffer], te


def cross_validate_spatial(model, X, y, cv, *, scoring: str | None = None) -> dict:
    """Run cross-validation with a spatial splitter; returns per-fold scores and the mean."""
    from sklearn.metrics import accuracy_score, r2_score

    scores = []
    for tr, te in cv.split(X, y):
        m = model.__class__(model.model_name, task=model.task, **model.params) \
            if hasattr(model, "model_name") else __import__("sklearn").base.clone(model)
        m.fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr], np.asarray(y)[tr])
        pred = m.predict(X.iloc[te] if hasattr(X, "iloc") else X[te])
        yt = np.asarray(y)[te]
        task = getattr(model, "task", "classification")
        if scoring == "r2" or (scoring is None and task == "regression"):
            scores.append(r2_score(yt, pred))
        else:
            scores.append(accuracy_score(yt, pred))
    return {"scores": np.array(scores), "mean": float(np.mean(scores)), "std": float(np.std(scores))}
