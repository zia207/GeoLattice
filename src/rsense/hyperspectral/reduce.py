"""Dimensionality reduction and band selection: PCA, MNF (with denoising), ICA, HySime and
virtual dimensionality, and supervised/unsupervised band selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu
from .cube import from_matrix, to_matrix, wavelengths


@dataclass
class Transform:
    """A fitted linear spectral transform (``scores = (X - mean) @ components.T``)."""

    kind: str
    mean: np.ndarray
    components: np.ndarray                 # (k, bands)
    eigenvalues: np.ndarray
    inverse: np.ndarray | None = None       # (bands, k) back-projection
    extra: dict = field(default_factory=dict)

    @property
    def explained_ratio(self) -> np.ndarray:
        ev = np.clip(self.eigenvalues, 0, None)
        return ev / ev.sum() if ev.sum() > 0 else ev

    def transform(self, cube: xr.DataArray, n: int | None = None) -> xr.DataArray:
        X, valid = to_matrix(cube)
        k = n or len(self.components)
        S = (X - self.mean) @ self.components[:k].T
        return from_matrix(S, cube, valid, names=[f"{self.kind}{i + 1}" for i in range(k)])

    def inverse_transform(self, scores: xr.DataArray, cube: xr.DataArray,
                          n: int | None = None) -> xr.DataArray:
        k = n or scores.sizes["component"]
        S, valid = to_matrix(scores.isel(component=slice(0, k)).rename(component="band"))
        Inv = self.inverse if self.inverse is not None else self.components.T
        X = S @ Inv[:, :k].T + self.mean
        out = from_matrix(X, cube, valid, dim="band")
        return cube.copy(data=out.values)


def _sample(X: np.ndarray, n: int | None, seed: int = 0) -> np.ndarray:
    if n and len(X) > n:
        return X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    return X


def pca(cube: xr.DataArray, n_components: int = 10, *, sample: int | None = 200_000,
        whiten: bool = False) -> tuple[xr.DataArray, Transform]:
    """Principal components of the spectra. Returns ``(scores cube, Transform)``; see
    ``Transform.explained_ratio``."""
    X, valid = to_matrix(cube)
    Xs = _sample(X, sample)
    mu = Xs.mean(0)
    C = np.cov(Xs - mu, rowvar=False)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    comps = V.T[:n_components]
    if whiten:
        comps = comps / np.sqrt(np.maximum(w[:n_components], 1e-12))[:, None]
    T = Transform("PC", mu, comps, w, inverse=V[:, :n_components] *
                  (np.sqrt(np.maximum(w[:n_components], 1e-12)) if whiten else 1))
    return T.transform(cube), T


def _noise_cov(cube: xr.DataArray, X: np.ndarray, method: str) -> np.ndarray:
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    if method == "shift":
        d = (a[:, :, 1:] - a[:, :, :-1]).reshape(a.shape[0], -1).T
        d = d[np.all(np.isfinite(d), axis=1)]
        return np.cov(d, rowvar=False) / 2
    # spectral-neighbour regression residuals (more robust to spatial texture)
    R = np.empty_like(X)
    for b in range(X.shape[1]):
        nb = [i for i in (b - 1, b + 1) if 0 <= i < X.shape[1]]
        A = np.column_stack([X[:, nb], np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, X[:, b], rcond=None)
        R[:, b] = X[:, b] - A @ coef
    return np.diag(R.var(axis=0, ddof=1))      # band noise assumed uncorrelated


def mnf(cube: xr.DataArray, n_components: int = 20, *,
        noise: Literal["shift", "residual"] = "shift", sample: int | None = 200_000
        ) -> tuple[xr.DataArray, Transform]:
    """Minimum Noise Fraction (Green et al. 1988): noise-whitened PCA. Components are ordered
    by signal-to-noise (``Transform.eigenvalues`` are SNR + 1). Returns ``(scores, Transform)``."""
    X, valid = to_matrix(cube)
    Xs = _sample(X, sample)
    mu = Xs.mean(0)
    Cn = _noise_cov(cube, Xs, noise)
    Cn += np.eye(len(Cn)) * 1e-12 * np.trace(Cn)
    wn, Vn = np.linalg.eigh(Cn)
    Wn = Vn / np.sqrt(np.maximum(wn, 1e-20))           # noise whitening
    Cs = np.cov((Xs - mu) @ Wn, rowvar=False)
    w, V = np.linalg.eigh(Cs)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    A = (Wn @ V).T                                      # (bands, bands) forward
    Ainv = np.linalg.pinv(A)
    T = Transform("MNF", mu, A[:n_components], w, inverse=Ainv[:, :n_components],
                  extra={"noise_cov": Cn})
    return T.transform(cube), T


def mnf_denoise(cube: xr.DataArray, n_components: int = 15, **kwargs: Any) -> xr.DataArray:
    """Denoise by keeping the first ``n_components`` MNF components and inverting."""
    scores, T = mnf(cube, n_components, **kwargs)
    return T.inverse_transform(scores, cube).rename(cube.name)


def ica(cube: xr.DataArray, n_components: int = 10, *, sample: int | None = 100_000,
        seed: int = 0, **kwargs: Any) -> tuple[xr.DataArray, Any]:
    """Independent components (scikit-learn FastICA). Returns ``(sources cube, model)``."""
    from sklearn.decomposition import FastICA

    X, valid = to_matrix(cube)
    m = FastICA(n_components, random_state=seed, whiten="unit-variance", max_iter=500,
                **kwargs).fit(_sample(X, sample, seed))
    return from_matrix(m.transform(X), cube, valid,
                       names=[f"IC{i + 1}" for i in range(n_components)]), m


def hysime(cube: xr.DataArray, *, sample: int | None = 100_000) -> dict[str, Any]:
    """HySime signal-subspace identification (Bioucas-Dias & Nascimento 2008).

    Estimates the noise by spectral-neighbour regression, then keeps the eigenvectors of
    the signal correlation whose projected signal power exceeds the projected noise power.
    Returns ``{"k": subspace dimension, "eigenvectors", "noise_std"}``.
    """
    X, _ = to_matrix(cube)
    X = _sample(X, sample)
    n, B = X.shape
    Rhat = X.T @ X
    W = np.empty_like(X)
    eps = 1e-6 * np.trace(Rhat) / B
    Rinv = np.linalg.inv(Rhat + eps * np.eye(B))
    for i in range(B):
        XX = Rinv - np.outer(Rinv[:, i], Rinv[i, :]) / Rinv[i, i]
        Rai = Rhat[:, i]
        beta = XX @ Rai
        beta[i] = 0
        W[:, i] = X[:, i] - X @ beta
    Rn = np.diag(np.diag(W.T @ W / n))
    Y = X - W
    Ry = Y.T @ Y / n
    w, E = np.linalg.eigh(Ry)
    o = np.argsort(w)[::-1]
    E = E[:, o]
    py = np.einsum("ij,jk,ki->i", E.T, Ry, E)
    pn = np.einsum("ij,jk,ki->i", E.T, Rn, E)
    cost = -py + 2 * pn
    neg = np.flatnonzero(cost < 0)
    k = int(neg.size)
    return {"k": k, "eigenvectors": E[:, :max(k, 1)], "noise_std": np.sqrt(np.diag(Rn))}


def virtual_dimensionality(cube: xr.DataArray, *, alpha: float = 1e-3,
                           sample: int | None = 100_000) -> int:
    """Harsanyi–Farrand–Chang virtual dimensionality: number of correlation-matrix
    eigenvalues significantly larger than the covariance ones (Neyman–Pearson, ``alpha``)."""
    from scipy.stats import norm

    X, _ = to_matrix(cube)
    X = _sample(X, sample)
    n = len(X)
    lr = np.sort(np.linalg.eigvalsh(X.T @ X / n))[::-1]
    lk = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    var = 2 * (lr**2 + lk**2) / n
    tau = norm.ppf(1 - alpha) * np.sqrt(var)
    return int(np.sum((lr - lk) > tau))


def select_bands(cube_or_X: Any, y: Any = None, *, n: int = 10,
                 method: Literal["mutual_info", "rf", "anova", "variance", "decorrelate"]
                 = "mutual_info", min_separation_nm: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Rank and pick ``n`` informative bands.

    Supervised (needs ``y`` labels or targets for the rows): ``mutual_info``, ``rf``
    (impurity importance), ``anova`` (F-statistic). Unsupervised: ``variance`` and
    ``decorrelate`` (greedy: highest variance band, then the band least correlated with
    those already chosen). ``min_separation_nm`` avoids picking adjacent bands. Accepts a
    cube (unsupervised) or a ``(samples × bands)`` DataFrame whose columns are wavelengths.
    """
    if isinstance(cube_or_X, xr.DataArray):
        X, _ = to_matrix(cube_or_X)
        X = _sample(X, 100_000, seed)
        wl = wavelengths(cube_or_X)
    else:
        df = pd.DataFrame(cube_or_X)
        X = df.to_numpy(float)
        try:
            wl = df.columns.to_numpy(float)
        except (TypeError, ValueError):
            wl = np.arange(X.shape[1], dtype=float)
    classification = y is not None and (pd.Series(y).dtype.kind in "OUSb" or
                                        pd.Series(y).nunique() <= 20)
    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        f = mutual_info_classif if classification else mutual_info_regression
        score = f(X, np.asarray(y), random_state=seed)
    elif method == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        M = RandomForestClassifier if classification else RandomForestRegressor
        score = M(200, random_state=seed, n_jobs=-1).fit(X, np.asarray(y)).feature_importances_
    elif method == "anova":
        from sklearn.feature_selection import f_classif, f_regression

        score = (f_classif if classification else f_regression)(X, np.asarray(y))[0]
    elif method == "variance":
        score = X.var(0)
    elif method == "decorrelate":
        C = np.abs(np.corrcoef(X, rowvar=False))
        chosen = [int(np.argmax(X.var(0)))]
        while len(chosen) < min(n, X.shape[1]):
            cand = C[:, chosen].max(1)
            cand[chosen] = np.inf
            if min_separation_nm:
                for c in chosen:
                    cand[np.abs(wl - wl[c]) < min_separation_nm] = np.inf
            j = int(np.argmin(cand))
            if not np.isfinite(cand[j]):
                break
            chosen.append(j)
        return pd.DataFrame({"band": chosen, "wavelength": wl[chosen],
                             "rank": np.arange(1, len(chosen) + 1)})
    else:
        raise ValueError(f"Unknown method {method!r}")
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    chosen: list[int] = []
    for j in order:
        if all(abs(wl[j] - wl[c]) >= min_separation_nm for c in chosen):
            chosen.append(int(j))
        if len(chosen) == n:
            break
    return pd.DataFrame({"band": chosen, "wavelength": wl[chosen], "score": score[chosen],
                         "rank": np.arange(1, len(chosen) + 1)})
