"""Lidar–optical fusion (biomass/height modelling) and UAV→satellite upscaling/validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu


def train_wall_to_wall(features: xr.Dataset, reference, value_col: str, *, model: str = "rf",
                       feature_names: Sequence[str] | None = None, test_size: float = 0.2,
                       seed: int = 0, **model_kw: Any) -> dict:
    """Model a sparse lidar/field variable (e.g. GEDI RH98, plot AGB) from wall-to-wall rasters
    (optical, SAR, terrain) and predict a continuous map.

    ``reference``: GeoDataFrame of points with ``value_col``. Returns the prediction map,
    holdout metrics (R², RMSE, bias) and the fitted model.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    from ..classify import Classifier, extract_samples

    X, y = extract_samples(features, reference, value_col, features=feature_names)
    y = y.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    reg = Classifier(model, task="regression", **model_kw).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    metrics = {"r2": float(r2_score(yte, pred)),
               "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
               "bias": float(np.mean(pred - yte)), "n_train": len(ytr), "n_test": len(yte)}
    reg.fit(X, y)
    return {"map": reg.predict_raster(features).rename(value_col), "metrics": metrics, "model": reg}


def upscale(fine: xr.DataArray, coarse_template: xr.DataArray, *,
            method: Literal["mean", "median", "max", "min", "fraction", "std"] = "mean",
            class_value: Any = 1) -> xr.DataArray:
    """Aggregate a fine raster (e.g. UAV) onto a coarse grid (e.g. Sentinel-2).

    ``fraction`` returns the share of fine pixels equal to ``class_value`` (sub-pixel cover).
    """
    from rasterio.enums import Resampling

    src = to_cpu(fine)
    if method == "fraction":
        src = (src == class_value).astype("float32").where(src.notnull())
        method = "mean"
    rs = {"mean": Resampling.average, "median": Resampling.med, "max": Resampling.max,
          "min": Resampling.min}
    if method == "std":
        m = src.rio.reproject_match(coarse_template, resampling=Resampling.average)
        m2 = (src**2).rio.reproject_match(coarse_template, resampling=Resampling.average)
        return np.sqrt((m2 - m**2).clip(min=0)).rename(f"{fine.name}_std")
    return src.astype("float32").rio.reproject_match(coarse_template, resampling=rs[method])


def compare(predicted: xr.DataArray, reference: xr.DataArray) -> dict[str, float]:
    """Agreement metrics between two co-registered rasters: R², RMSE, MAE, bias, n."""
    p = np.asarray(to_cpu(predicted).values, float).ravel()
    r = np.asarray(to_cpu(reference.rio.reproject_match(predicted).values
                          if reference.shape != predicted.shape else reference.values), float).ravel()
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    return {"r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean((p - r) ** 2))), "mae": float(np.mean(np.abs(p - r))),
            "bias": float(np.mean(p - r)), "n": int(ok.sum())}
