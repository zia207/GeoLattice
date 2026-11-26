"""Spatially blocked cross-fitting.

Random K-fold cross-fitting assumes training and held-out units are
independent. With spatial autocorrelation, a neighbour in the training fold
leaks information about a held-out unit, so nuisance predictions look better
than they are and the orthogonal score is no longer computed out of sample.
``SpatialBlockKFold`` holds out contiguous spatial blocks and can also drop
training units within a buffer distance of the held-out block.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from geolattice.spatial_analysis.spatialcausal._utils import as_coords


class SpatialBlockKFold:
    """K-fold splitter over spatial blocks.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    n_blocks : int or "auto", optional
        Number of spatial blocks. ``"auto"`` (default) is
        ``max(5 * n_splits, n // 15)``. Blocks are assigned to folds at
        random, so each fold is spread over the study area rather than being
        one region.

        Block size is a trade-off. Blocks should be larger than the range of
        residual dependence (to stop leakage between neighbours), but small
        relative to the spatial scale of any confounder the nuisance models
        must learn from location: a held-out block the size of the
        confounder's range forces extrapolation. In the county tutorial, 25
        blocks (~500 km) biased spatial DML by >60% while ~200 blocks
        (~100 km) did not.
    method : {"kmeans", "grid"}
        How to form blocks: k-means on coordinates, or a regular grid.
    buffer : float, default 0
        Training units closer than ``buffer`` (coordinate units) to any
        held-out unit are dropped from that fold's training set.
    random_state : int
    """

    def __init__(self, n_splits: int = 5, n_blocks="auto",
                 method: str = "kmeans", buffer: float = 0.0, random_state: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.n_blocks = "auto" if n_blocks in (None, "auto") else int(n_blocks)
        self.method = method
        self.buffer = float(buffer)
        self.random_state = int(random_state)
        self.blocks_: Optional[np.ndarray] = None

    def make_blocks(self, coords) -> np.ndarray:
        coords = as_coords(coords)
        nb = max(5 * self.n_splits, len(coords) // 15) if self.n_blocks == "auto" else self.n_blocks
        n_blocks = min(nb, len(coords))
        if self.method == "kmeans":
            km = KMeans(n_clusters=n_blocks, n_init=4, random_state=self.random_state)
            blocks = km.fit_predict(coords)
        elif self.method == "grid":
            side = int(np.ceil(np.sqrt(n_blocks)))
            mins, maxs = coords.min(0), coords.max(0)
            span = np.where(maxs > mins, maxs - mins, 1.0)
            ij = np.minimum(((coords - mins) / span * side).astype(int), side - 1)
            _, blocks = np.unique(ij[:, 0] * side + ij[:, 1], return_inverse=True)
        else:
            raise ValueError("method must be 'kmeans' or 'grid'")
        self.blocks_ = blocks
        return blocks

    def split(self, coords) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        coords = as_coords(coords)
        blocks = self.make_blocks(coords)
        uniq = np.unique(blocks)
        if len(uniq) < self.n_splits:
            raise ValueError(f"only {len(uniq)} blocks for {self.n_splits} folds")
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        tree = cKDTree(coords) if self.buffer > 0 else None
        for _, test_blocks in kf.split(uniq):
            test_mask = np.isin(blocks, uniq[test_blocks])
            test_idx = np.flatnonzero(test_mask)
            train_mask = ~test_mask
            if tree is not None:
                near = tree.query_ball_point(coords[test_idx], r=self.buffer)
                near = np.unique(np.concatenate([np.asarray(v, dtype=int) for v in near]))
                train_mask[near] = False
            yield np.flatnonzero(train_mask), test_idx

    def get_n_splits(self, *_, **__) -> int:
        return self.n_splits


def make_folds(coords, cv="spatial_block", n_splits: int = 5, random_state: int = 0, **kwargs):
    """Return a list of (train, test) index pairs.

    ``cv`` may be ``"spatial_block"`` (default), ``"random"``, a
    ``SpatialBlockKFold`` instance, or a precomputed list of index pairs.
    """
    coords = as_coords(coords)
    if isinstance(cv, (list, tuple)):
        return list(cv)
    if isinstance(cv, SpatialBlockKFold):
        return list(cv.split(coords))
    if cv == "spatial_block":
        return list(SpatialBlockKFold(n_splits=n_splits, random_state=random_state, **kwargs).split(coords))
    if cv == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(coords))
    raise ValueError("cv must be 'spatial_block', 'random', a SpatialBlockKFold, or a list of folds")


def cross_fit_predict(learner_factory, X, target, folds, method: str = "predict") -> np.ndarray:
    """Out-of-fold predictions. ``method='predict_proba'`` returns P(class 1).

    Units that land in no test fold (impossible with the built-in splitters)
    are left as NaN and flagged by callers.
    """
    out = np.full(len(target), np.nan)
    for train, test in folds:
        model = learner_factory()
        model.fit(X[train], target[train])
        if method == "predict_proba":
            out[test] = model.predict_proba(X[test])[:, 1]
        else:
            out[test] = model.predict(X[test])
    return out
