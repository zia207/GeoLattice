"""Spectral matching against a library: Spectral Angle Mapper, Spectral Information Divergence,
spectral correlation, Euclidean distance and SID-SAM, plus best-match classification maps."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from .cube import from_matrix, to_matrix
from .library import as_library

Measure = Literal["sam", "sid", "sca", "euclidean", "sidsam"]


def _scores(X: np.ndarray, S: np.ndarray, measure: str) -> np.ndarray:
    """(pixels, classes) dissimilarity, lower = more similar."""
    ok = np.all(np.isfinite(S), axis=0)
    X, S = X[:, ok], S[:, ok]
    if measure in ("sam", "sidsam"):
        nx = np.linalg.norm(X, axis=1, keepdims=True)
        ns = np.linalg.norm(S, axis=1)
        cos = (X @ S.T) / np.maximum(nx * ns[None, :], 1e-12)
        sam = np.arccos(np.clip(cos, -1, 1))
        if measure == "sam":
            return sam
    if measure in ("sid", "sidsam"):
        P = np.clip(X, 1e-9, None)
        P = P / P.sum(1, keepdims=True)
        Q = np.clip(S, 1e-9, None)
        Q = Q / Q.sum(1, keepdims=True)
        lp, lq = np.log(P), np.log(Q)
        sid = ((P * lp).sum(1)[:, None] - P @ lq.T) + ((Q * lq).sum(1)[None, :] - lp @ Q.T)
        if measure == "sid":
            return sid
        return sid * np.tan(sam)                         # Du et al. 2004
    if measure == "sca":
        Xc = X - X.mean(1, keepdims=True)
        Sc = S - S.mean(1, keepdims=True)
        r = (Xc @ Sc.T) / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True)
                                     * np.linalg.norm(Sc, axis=1)[None, :], 1e-12)
        return np.arccos(np.clip((r + 1) / 2, -1, 1))    # spectral correlation angle
    if measure == "euclidean":
        return np.sqrt(np.maximum((X**2).sum(1)[:, None] - 2 * X @ S.T + (S**2).sum(1)[None, :],
                                  0))
    raise ValueError(f"Unknown measure {measure!r}")


def spectral_similarity(cube: xr.DataArray, library: Any, *, measure: Measure = "sam"
                        ) -> xr.DataArray:
    """Dissimilarity of every pixel to every library spectrum (lower = closer).
    ``sam`` and ``sca`` are angles in radians. Output dims ``(target, y, x)``."""
    lib = as_library(library, cube)
    X, valid = to_matrix(cube)
    D = _scores(X, lib.spectra, measure)
    return from_matrix(D, cube, valid, names=lib.names, dim="target").rename(measure)


def sam(cube: xr.DataArray, library: Any) -> xr.DataArray:
    """Spectral Angle Mapper angles (radians) to each library spectrum (Kruse et al. 1993)."""
    return spectral_similarity(cube, library, measure="sam")


def sid(cube: xr.DataArray, library: Any) -> xr.DataArray:
    """Spectral Information Divergence to each library spectrum (Chang 2000)."""
    return spectral_similarity(cube, library, measure="sid")


def classify_spectral(cube: xr.DataArray, library: Any, *, measure: Measure = "sam",
                      threshold: float | None = None) -> xr.Dataset:
    """Best-matching library spectrum per pixel.

    Returns ``class`` (index into the library, ``-1`` where the best score exceeds
    ``threshold``, e.g. 0.10 rad for SAM), ``score`` (best dissimilarity) and the class
    names in ``attrs["classes"]``.
    """
    lib = as_library(library, cube)
    D = spectral_similarity(cube, lib, measure=measure)
    v = np.asarray(D.values)
    finite = np.isfinite(v).all(0)
    best = np.where(finite, np.nanargmin(np.where(np.isfinite(v), v, np.inf), axis=0), -1)
    score = np.where(finite, np.nanmin(np.where(np.isfinite(v), v, np.inf), axis=0), np.nan)
    if threshold is not None:
        best = np.where(score <= threshold, best, -1)
    tpl = D.isel(target=0, drop=True)
    return xr.Dataset({"class": tpl.copy(data=best.astype("int16")),
                       "score": tpl.copy(data=score.astype("float32"))},
                      attrs={"classes": list(lib.names), "measure": measure})
