"""Conformal prediction: distribution-free uncertainty for maps (regression intervals,
classification prediction sets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu


class ConformalRegressor:
    """Split-conformal intervals (Lei et al. 2018), optionally normalized by a difficulty model.

    ``fit`` trains the base model on part of the data and calibrates on the rest; intervals
    ``pred ± q`` cover the truth with probability ≥ 1 − alpha (marginally).
    """

    def __init__(self, model, alpha: float = 0.1, calibration_size: float = 0.25, seed: int = 0):
        self.model, self.alpha, self.cal, self.seed = model, alpha, calibration_size, seed
        self.q_: float | None = None

    def fit(self, X: pd.DataFrame, y) -> ConformalRegressor:
        from sklearn.model_selection import train_test_split

        Xtr, Xc, ytr, yc = train_test_split(X, np.asarray(y, float), test_size=self.cal,
                                            random_state=self.seed)
        self.model.fit(Xtr, ytr)
        res = np.abs(yc - np.asarray(self.model.predict(Xc), float))
        n = len(res)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(res, level, method="higher"))
        return self

    def predict_interval(self, X) -> pd.DataFrame:
        p = np.asarray(self.model.predict(X), float)
        return pd.DataFrame({"pred": p, "lower": p - self.q_, "upper": p + self.q_})

    def predict_raster(self, features: xr.Dataset) -> xr.Dataset:
        pred = self.model.predict_raster(features)
        return xr.Dataset({"pred": pred, "lower": pred - self.q_, "upper": pred + self.q_},
                          attrs={"alpha": self.alpha, "q": self.q_})


class ConformalClassifier:
    """Split-conformal prediction sets (LAC score ``1 - p_true``; Sadinle et al. 2019).

    Each pixel gets the set of classes with ``p >= 1 - q``; set size is a direct,
    interpretable uncertainty map (1 = confident, >1 = ambiguous, 0 = novel/outlier).
    """

    def __init__(self, model, alpha: float = 0.1, calibration_size: float = 0.25, seed: int = 0):
        self.model, self.alpha, self.cal, self.seed = model, alpha, calibration_size, seed
        self.q_: float | None = None

    def fit(self, X: pd.DataFrame, y) -> ConformalClassifier:
        from sklearn.model_selection import train_test_split

        y = np.asarray(y)
        Xtr, Xc, ytr, yc = train_test_split(X, y, test_size=self.cal, random_state=self.seed,
                                            stratify=y)
        self.model.fit(Xtr, ytr)
        P = self.model.predict_proba(Xc)
        classes = list(np.asarray(to_cpu(self.model.classes_)))
        idx = np.array([classes.index(c) for c in yc])
        scores = 1 - P[np.arange(len(yc)), idx]
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        self.classes_ = np.asarray(classes)
        return self

    def predict_sets(self, X) -> np.ndarray:
        """Boolean (n, n_classes) membership matrix."""
        return self.model.predict_proba(X) >= 1 - self.q_

    def predict_raster(self, features: xr.Dataset) -> xr.Dataset:
        proba = self.model.predict_raster(features, proba=True)
        member = xr.Dataset({k: proba[k] >= 1 - self.q_ for k in proba.data_vars})
        size = sum(member[k].astype("int8") for k in member.data_vars)
        best = proba.to_array("cls").fillna(-1).argmax("cls")
        return xr.Dataset({"set_size": size.where(proba[list(proba.data_vars)[0]].notnull()),
                           "prediction": best}, attrs={"alpha": self.alpha, "q": self.q_,
                                                       "classes": str(list(self.classes_))})


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of intervals."""
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


