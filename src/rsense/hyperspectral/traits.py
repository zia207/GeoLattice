"""Biophysical and biochemical trait retrieval: PLSR with component selection and VIP scores,
Gaussian process regression, any scikit-learn regressor on spectra, PROSAIL look-up tables
and hybrid (radiative transfer + machine learning) retrieval, and trait maps."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .cube import from_matrix, to_matrix, wavelengths


@dataclass
class TraitModel:
    """A fitted spectra → trait model with its band grid and validation metrics."""

    model: Any
    wavelengths: np.ndarray
    target: str
    metrics: dict = field(default_factory=dict)
    preprocess: Any = None
    extra: dict = field(default_factory=dict)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return self.preprocess(X) if self.preprocess is not None else X

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(self._prep(np.asarray(X, float)))).ravel()

    def predict_map(self, cube: xr.DataArray, *, chunk: int = 200_000,
                    return_std: bool = False) -> xr.DataArray | xr.Dataset:
        """Apply to every valid pixel (the cube is resampled to the model's bands if needed)."""
        if not np.array_equal(np.round(wavelengths(cube), 3), np.round(self.wavelengths, 3)):
            from .preprocess import resample_spectral

            cube = resample_spectral(cube, self.wavelengths)
        X, valid = to_matrix(cube)
        outs, stds = [], []
        for s in range(0, len(X), chunk):
            Xi = self._prep(X[s:s + chunk])
            if return_std and "return_std" in inspect.signature(self.model.predict).parameters:
                m, sd = self.model.predict(Xi, return_std=True)
                outs.append(np.ravel(m))
                stds.append(np.ravel(sd))
            else:
                outs.append(np.ravel(self.model.predict(Xi)))
        y = np.concatenate(outs) if outs else np.zeros(0)
        da = from_matrix(y, cube, valid).isel(component=0, drop=True).rename(self.target)
        if stds:
            sd = from_matrix(np.concatenate(stds), cube, valid).isel(component=0, drop=True)
            return xr.Dataset({self.target: da, f"{self.target}_std": sd})
        return da


def _metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {"r2": float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else np.nan, "rmse": rmse,
            "bias": float(np.mean(p - y)),
            "nrmse": rmse / float(np.ptp(y)) if np.ptp(y) else np.nan,
            "rpd": float(np.std(y, ddof=1) / rmse) if rmse > 0 else np.nan, "n": int(len(y))}


def _xy(spectra: Any, y: Any = None, target: str | None = None):
    if isinstance(spectra, pd.DataFrame) and target is not None and target in spectra:
        y = spectra[target].to_numpy(float)
        spectra = spectra.drop(columns=[target])
    df = pd.DataFrame(spectra)
    num = []
    for c in df.columns:
        try:
            num.append(float(c))
        except (TypeError, ValueError):
            num.append(None)
    cols = [c for c, w in zip(df.columns, num) if w is not None]
    wl = np.array([w for w in num if w is not None], float)
    if not cols:
        cols, wl = list(df.columns), np.arange(df.shape[1], dtype=float)
    return df[cols].to_numpy(float), np.asarray(y, float), wl


def vip_scores(pls) -> np.ndarray:
    """Variable Importance in Projection for a fitted ``PLSRegression`` (VIP > 1 ≈ important)."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = w.shape[0]
    s = np.diag(t.T @ t @ q.T @ q).reshape(-1)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (wn**2 @ s) / s.sum())


def fit_plsr(spectra: Any, y: Any = None, *, target: str = "trait", max_components: int = 20,
             cv: int = 10, groups: Any = None, seed: int = 0, preprocess: Any = None
             ) -> TraitModel:
    """Partial least squares regression with the number of components chosen by
    cross-validated RMSE (the smallest model within one standard error of the minimum).

    ``spectra`` is a (samples × bands) DataFrame with wavelength columns (e.g. from
    :func:`~.cube.spectra_at`) or an array; pass ``groups`` (e.g. site ids) for grouped CV.
    ``metrics`` hold CV R², RMSE, RPD; ``extra`` has the PRESS curve and VIP scores.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

    X, yv, wl = _xy(spectra, y, target if isinstance(spectra, pd.DataFrame) else None)
    Xp = preprocess(X) if preprocess is not None else X
    splitter = GroupKFold(cv) if groups is not None else KFold(cv, shuffle=True, random_state=seed)
    kmax = int(min(max_components, Xp.shape[1], len(yv) - len(yv) // cv - 1))
    rmses, ses = [], []
    for k in range(1, kmax + 1):
        p = cross_val_predict(PLSRegression(k, scale=False), Xp, yv, cv=splitter,
                              groups=groups).ravel()
        e = (yv - p) ** 2
        rmses.append(np.sqrt(e.mean()))
        ses.append(e.std(ddof=1) / np.sqrt(len(e)) / (2 * max(rmses[-1], 1e-12)))
    rmses, ses = np.array(rmses), np.array(ses)
    kbest = int(np.argmin(rmses))
    k = int(np.flatnonzero(rmses <= rmses[kbest] + ses[kbest])[0]) + 1
    cvp = cross_val_predict(PLSRegression(k, scale=False), Xp, yv, cv=splitter,
                            groups=groups).ravel()
    model = PLSRegression(k, scale=False).fit(Xp, yv)
    return TraitModel(model, wl, target, {"n_components": k, **_metrics(yv, cvp)}, preprocess,
                      {"press": pd.DataFrame({"n_components": np.arange(1, kmax + 1),
                                              "rmse_cv": rmses}),
                       "vip": pd.Series(vip_scores(model), index=wl, name="vip"),
                       "cv_predictions": cvp})


def fit_regressor(spectra: Any, y: Any = None, *, target: str = "trait", model: str | Any = "rf",
                  cv: int = 5, groups: Any = None, seed: int = 0, preprocess: Any = None,
                  **model_kw: Any) -> TraitModel:
    """Any regressor on spectra: ``rf``, ``gbm``, ``svr``, ``gpr`` (Gaussian process with an
    RBF + white-noise kernel; ``predict_map(return_std=True)`` gives uncertainty), ``ridge``,
    or a scikit-learn estimator. Cross-validated metrics in ``metrics``."""
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, yv, wl = _xy(spectra, y, target if isinstance(spectra, pd.DataFrame) else None)
    Xp = preprocess(X) if preprocess is not None else X
    if isinstance(model, str):
        if model == "rf":
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(n_estimators=model_kw.pop("n_estimators", 300),
                                        random_state=seed, n_jobs=-1, **model_kw)
        elif model == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor

            est = HistGradientBoostingRegressor(random_state=seed, **model_kw)
        elif model == "svr":
            from sklearn.svm import SVR

            est = make_pipeline(StandardScaler(), SVR(**model_kw))
        elif model == "ridge":
            from sklearn.linear_model import RidgeCV

            est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
        elif model == "gpr":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

            kern = ConstantKernel(1.0) * RBF(np.ones(Xp.shape[1]) * 1.0, (1e-2, 1e4)) + \
                WhiteKernel(1e-2) if model_kw.pop("ard", False) else \
                ConstantKernel(1.0) * RBF(1.0, (1e-3, 1e4)) + WhiteKernel(1e-2)
            est = GaussianProcessRegressor(kern, normalize_y=True, random_state=seed, **model_kw)
        else:
            raise ValueError(f"Unknown model {model!r}")
    else:
        est = model
    splitter = GroupKFold(cv) if groups is not None else KFold(cv, shuffle=True, random_state=seed)
    cvp = cross_val_predict(est, Xp, yv, cv=splitter, groups=groups).ravel()
    est.fit(Xp, yv)
    return TraitModel(est, wl, target, _metrics(yv, cvp), preprocess, {"cv_predictions": cvp})


# ----------------------------------------------------------------------------- PROSAIL
PROSAIL_RANGES: dict[str, tuple[float, float]] = {
    "n": (1.2, 2.2), "cab": (10.0, 80.0), "car": (2.0, 20.0), "cbrown": (0.0, 0.5),
    "cw": (0.005, 0.04), "cm": (0.002, 0.015), "lai": (0.2, 7.0), "lidfa": (30.0, 70.0),
    "hspot": (0.01, 0.3), "psoil": (0.0, 1.0), "rsoil": (0.5, 1.5), "ant": (0.0, 5.0),
}


def prosail_spectrum(*, wavelengths_nm: Sequence[float] | None = None, tts: float = 30.0,
                     tto: float = 0.0, psi: float = 0.0, **params: float) -> np.ndarray:
    """One PROSAIL (PROSPECT-D + 4SAIL) canopy reflectance spectrum, 400–2500 nm, interpolated
    to ``wavelengths_nm``. Needs ``pip install prosail``. ``lidfa`` is the mean leaf angle
    (degrees, Campbell ellipsoidal distribution)."""
    try:
        import prosail
    except ImportError as e:
        raise ImportError("PROSAIL needs `pip install prosail` (geolattice[rsense-hyperspectral])") \
            from e
    p = {"n": 1.5, "cab": 40.0, "car": 8.0, "cbrown": 0.0, "cw": 0.01, "cm": 0.009, "lai": 3.0,
         "lidfa": 57.0, "hspot": 0.01, "psoil": 0.5, "rsoil": 1.0, "ant": 0.0}
    p.update(params)
    r = prosail.run_prosail(n=p["n"], cab=p["cab"], car=p["car"], cbrown=p["cbrown"], cw=p["cw"],
                            cm=p["cm"], lai=p["lai"], lidfa=p["lidfa"], hspot=p["hspot"], tts=tts,
                            tto=tto, psi=psi, ant=p["ant"], prospect_version="D", typelidf=2,
                            rsoil=p["rsoil"], psoil=p["psoil"], factor="SDR")
    wl = np.arange(400, 2501)
    return np.interp(np.asarray(wavelengths_nm, float), wl, r) if wavelengths_nm is not None else r


def prosail_lut(n: int = 2000, *, ranges: Mapping[str, tuple[float, float]] | None = None,
                fixed: Mapping[str, float] | None = None, cube: xr.DataArray | None = None,
                wavelengths_nm: Sequence[float] | None = None, fwhm: Sequence[float] | None = None,
                tts: float = 30.0, tto: float = 0.0, psi: float = 0.0, noise: float = 0.0,
                seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Latin-hypercube PROSAIL look-up table convolved to a sensor (``cube`` bands, or
    ``wavelengths_nm``/``fwhm``). ``noise`` adds Gaussian noise (relative, e.g. 0.02).
    Returns ``(parameters DataFrame, spectra (n × bands), wavelengths)``."""
    from scipy.stats import qmc

    from .preprocess import _gaussian_matrix

    rng_ = dict(PROSAIL_RANGES)
    rng_.update(ranges or {})
    for k in fixed or {}:
        rng_.pop(k, None)
    names = list(rng_)
    lo = np.array([rng_[k][0] for k in names])
    hi = np.array([rng_[k][1] for k in names])
    U = qmc.LatinHypercube(len(names), seed=seed).random(n)
    P = pd.DataFrame(lo + U * (hi - lo), columns=names)
    for k, v in (fixed or {}).items():
        P[k] = v
    full = np.stack([prosail_spectrum(tts=tts, tto=tto, psi=psi, **row) for row in
                     P.to_dict("records")])
    src = np.arange(400, 2501, dtype=float)
    if cube is not None:
        wl = wavelengths(cube)
        fw = np.asarray(cube["fwhm"].values, float) if "fwhm" in cube.coords else None
    else:
        wl = np.asarray(wavelengths_nm if wavelengths_nm is not None else src, float)
        fw = np.asarray(fwhm, float) if fwhm is not None else None
    if fw is not None:
        W = _gaussian_matrix(src, None, wl, fw)
        S = full @ W.T
    else:
        S = np.stack([np.interp(wl, src, s) for s in full])
    if noise:
        S = S * (1 + np.random.default_rng(seed + 1).normal(0, noise, S.shape))
    return P, S, wl


def hybrid_retrieval(cube: xr.DataArray, traits: Sequence[str] = ("lai", "cab"), *,
                     n: int = 2000, model: str = "gpr", noise: float = 0.02,
                     ranges: Mapping[str, tuple[float, float]] | None = None,
                     fixed: Mapping[str, float] | None = None, tts: float = 30.0,
                     tto: float = 0.0, psi: float = 0.0, n_train: int | None = 800,
                     return_std: bool = False, seed: int = 0) -> dict[str, Any]:
    """Hybrid retrieval (Verrelst et al. 2015): simulate a noisy PROSAIL LUT on the cube's
    bands, train one regressor per trait (``gpr``, ``rf``, ``gbm``, ...) on it and map the
    traits. Returns ``{"maps": Dataset, "models": {trait: TraitModel}, "lut": (params,
    spectra)}``. Validate against field data before trusting absolute values."""
    P, S, wl = prosail_lut(n, ranges=ranges, fixed=fixed, cube=cube, tts=tts, tto=tto, psi=psi,
                           noise=noise, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), min(n_train or len(P), len(P)), replace=False)
    models, maps = {}, {}
    for t in traits:
        df = pd.DataFrame(S[idx], columns=wl)
        m = fit_regressor(df, P[t].to_numpy()[idx], target=t, model=model, cv=3, seed=seed)
        models[t] = m
        out = m.predict_map(cube, return_std=return_std)
        if isinstance(out, xr.Dataset):
            maps.update({k: out[k] for k in out.data_vars})
        else:
            maps[t] = out
    return {"maps": xr.Dataset(maps), "models": models, "lut": (P, S)}
