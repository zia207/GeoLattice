"""Area of Applicability (Meyer & Pebesma 2021): where can a spatial model be trusted?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu
from ..optical.bands import as_dataset


def dissimilarity_index(train_X: pd.DataFrame, new_X: np.ndarray | pd.DataFrame, *,
                        weights: Sequence[float] | None = None,
                        folds: Sequence[int] | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Dissimilarity Index of new data to training data, plus the AOA threshold.

    Predictors are standardized (training mean/std) and weighted by importance. DI is the
    distance to the nearest training sample divided by the mean pairwise training distance.
    The threshold is the upper whisker (Q75 + 1.5 IQR) of the training DI, computed with
    cross-validation ``folds`` (nearest training point outside the sample's own fold).
    Returns ``(di_new, threshold, di_train)``.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist

    T = train_X.to_numpy(float)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1
    w = np.ones(T.shape[1]) if weights is None else np.asarray(weights, float)
    Ts = (T - mu) / sd * w
    N = (np.asarray(new_X, float) - mu) / sd * w
    sample = Ts if len(Ts) <= 3000 else Ts[np.random.default_rng(0).choice(len(Ts), 3000, False)]
    dbar = pdist(sample).mean()
    tree = cKDTree(Ts)
    ok = np.all(np.isfinite(N), 1)
    di_new = np.full(len(N), np.nan)
    di_new[ok] = tree.query(N[ok], k=1)[0] / dbar
    if folds is None:
        d, _ = tree.query(Ts, k=2)
        di_train = d[:, 1] / dbar
    else:
        folds = np.asarray(folds)
        di_train = np.empty(len(Ts))
        for f in np.unique(folds):
            m = folds == f
            di_train[m] = cKDTree(Ts[~m]).query(Ts[m], k=1)[0] / dbar
    q1, q3 = np.percentile(di_train, [25, 75])
    thr = float(min(q3 + 1.5 * (q3 - q1), di_train.max()))
    return di_new, thr, di_train


def aoa(features: xr.Dataset | xr.DataArray, train_X: pd.DataFrame, *,
        weights: Sequence[float] | pd.Series | None = None, folds: Sequence[int] | None = None
        ) -> xr.Dataset:
    """Map the Dissimilarity Index and the Area of Applicability for a feature raster.

    ``weights`` can be a Series of importances (aligned to ``train_X.columns``), e.g. from
    ``Classifier.feature_importance()``.
    """
    ds = as_dataset(features)
    cols = list(train_X.columns)
    if isinstance(weights, pd.Series):
        weights = weights.reindex(cols).fillna(0).to_numpy()
    tpl = ds[cols[0]]
    X = np.stack([np.asarray(to_cpu(ds[c]).values, float).ravel() for c in cols], 1)
    di, thr, _ = dissimilarity_index(train_X, X, weights=weights, folds=folds)
    di = di.reshape(tpl.shape)
    return xr.Dataset({"DI": tpl.copy(data=di.astype("float32")),
                       "AOA": tpl.copy(data=(di <= thr) & np.isfinite(di))},
                      attrs={"threshold": thr})
