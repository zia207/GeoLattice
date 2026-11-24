"""Model explanations: SHAP values and permutation importance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def _estimator(model):
    return getattr(model, "model", model)  # unwrap pyrsense Classifier


def permutation_importance(model, X: pd.DataFrame, y, *, n_repeats: int = 10,
                           metric: Callable | None = None, seed: int = 0,
                           groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Drop in score when each feature (or feature ``groups``, e.g. all S1 bands) is shuffled.

    Works with any fitted model exposing ``predict`` on a DataFrame (including
    :class:`geolattice.rsense.classify.Classifier`). ``metric(y_true, y_pred)`` defaults to accuracy
    for classifiers and R² for regressors. Grouped permutation avoids diluting importance
    across correlated bands.
    """
    from sklearn.metrics import accuracy_score, r2_score

    task = getattr(model, "task", None) or ("classification" if hasattr(_estimator(model), "classes_")
                                             else "regression")
    metric = metric or (accuracy_score if task == "classification" else r2_score)
    y = np.asarray(y)
    base = metric(y, model.predict(X))
    rng = np.random.default_rng(seed)
    groups = groups or {c: [c] for c in X.columns}
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - metric(y, model.predict(Xp)))
        rows.append({"feature": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values(model, X: pd.DataFrame, *, background: int = 200, max_samples: int = 2000,
                seed: int = 0) -> Any:
    """SHAP explanation object (TreeExplainer for tree ensembles, else a model-agnostic
    explainer on a ``background`` sample). Plot with ``shap.plots.beeswarm(values)``."""
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap_values needs `pip install shap` (geolattice[rsense-xai])") from e
    est = _estimator(model)
    Xs = X.sample(min(max_samples, len(X)), random_state=seed)
    try:
        explainer = shap.TreeExplainer(est)
        return explainer(Xs)
    except Exception:  # noqa: BLE001  (fall back to model-agnostic explainer)
        bg = shap.sample(X, min(background, len(X)), random_state=seed)
        f = est.predict_proba if hasattr(est, "predict_proba") else est.predict
        return shap.Explainer(f, bg)(Xs)


def shap_importance(values) -> pd.Series:
    """Mean |SHAP| per feature (averaged over classes for multiclass)."""
    v = np.abs(values.values)
    if v.ndim == 3:
        v = v.mean(-1)
    return pd.Series(v.mean(0), index=values.feature_names).sort_values(ascending=False)
