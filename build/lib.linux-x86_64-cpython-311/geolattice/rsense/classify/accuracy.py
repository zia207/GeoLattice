"""Accuracy assessment, including Olofsson et al. (2014) area-adjusted estimates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def confusion_matrix(reference: Sequence, predicted: Sequence,
                     labels: Sequence | None = None) -> pd.DataFrame:
    """Confusion matrix with rows = map (predicted) classes, columns = reference classes.

    (Map-by-reference orientation, as in Olofsson et al. 2014.)
    """
    ref, pred = np.asarray(reference), np.asarray(predicted)
    labels = list(labels) if labels is not None else sorted(set(ref) | set(pred))
    cm = pd.crosstab(pd.Categorical(pred, categories=labels), pd.Categorical(ref, categories=labels),
                     dropna=False)
    cm.index.name, cm.columns.name = "map", "reference"
    return cm.astype(int)


def accuracy_report(reference: Sequence, predicted: Sequence,
                    labels: Sequence | None = None) -> dict[str, Any]:
    """Overall accuracy, Cohen's kappa, per-class user's/producer's accuracy and F1."""
    cm = confusion_matrix(reference, predicted, labels)
    m = cm.to_numpy(float)
    n = m.sum()
    diag = np.diag(m)
    oa = diag.sum() / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = diag / m.sum(1)
        pa = diag / m.sum(0)
        f1 = 2 * ua * pa / (ua + pa)
    per_class = pd.DataFrame({"users_accuracy": ua, "producers_accuracy": pa, "f1": f1,
                              "n_reference": m.sum(0).astype(int)}, index=cm.index)
    return {"overall_accuracy": float(oa), "kappa": float((oa - pe) / (1 - pe)) if pe < 1 else 1.0,
            "confusion_matrix": cm, "per_class": per_class}


def olofsson(cm: pd.DataFrame, map_area: Mapping[Any, float], *, z: float = 1.96,
             pixel_area: float | None = None) -> dict[str, Any]:
    """Area-adjusted accuracy and area estimation (Olofsson et al. 2014, stratified random sampling).

    Parameters
    ----------
    cm : confusion matrix of sample counts, rows = map class, columns = reference class
        (as returned by :func:`confusion_matrix`).
    map_area : mapped area (or pixel count) per map class.
    pixel_area : multiply pixel counts by this to obtain area units.

    Returns overall accuracy, user's and producer's accuracy with confidence intervals,
    and the error-adjusted area of each class with its confidence interval.
    """
    classes = list(cm.index)
    n = cm.loc[classes, classes].to_numpy(float)
    A = np.array([map_area[c] for c in classes], float) * (pixel_area or 1.0)
    W = A / A.sum()
    ni = n.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = W[:, None] * n / ni[:, None]
    p = np.nan_to_num(p)
    pj = p.sum(0)
    oa = np.trace(p)
    ua = np.diag(p) / p.sum(1)
    pa = np.diag(p) / pj
    with np.errstate(divide="ignore", invalid="ignore"):
        nij_ni = n / ni[:, None]
        var_oa = np.nansum(W**2 * ua * (1 - ua) / (ni - 1))
        var_ua = ua * (1 - ua) / (ni - 1)
        se_pj = np.sqrt(np.nansum((W[:, None] ** 2) * nij_ni * (1 - nij_ni) / (ni[:, None] - 1), 0))
        Nj = pj * A.sum()  # estimated area per reference class
        Ni = A
        var_pa = np.zeros(len(classes))
        for j in range(len(classes)):
            term1 = Ni[j] ** 2 * (1 - pa[j]) ** 2 * ua[j] * (1 - ua[j]) / (ni[j] - 1)
            others = [i for i in range(len(classes)) if i != j]
            term2 = pa[j] ** 2 * sum(Ni[i] ** 2 * nij_ni[i, j] * (1 - nij_ni[i, j]) / (ni[i] - 1)
                                     for i in others)
            var_pa[j] = (term1 + term2) / Nj[j] ** 2
    area = pj * A.sum()
    table = pd.DataFrame({
        "map_area": A, "adjusted_area": area, "area_ci": z * se_pj * A.sum(),
        "users_accuracy": ua, "ua_ci": z * np.sqrt(var_ua),
        "producers_accuracy": pa, "pa_ci": z * np.sqrt(var_pa)}, index=classes)
    return {"overall_accuracy": float(oa), "oa_ci": float(z * np.sqrt(var_oa)), "table": table,
            "proportions": pd.DataFrame(p, index=classes, columns=classes)}
