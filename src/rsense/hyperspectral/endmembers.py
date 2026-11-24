"""Endmember extraction: Pixel Purity Index, N-FINDR, Vertex Component Analysis and ATGP.

Each extractor returns a dict with ``spectra`` (k × bands), ``index`` (flat pixel index on the
cube grid), ``row``/``col`` and a :class:`~.library.SpectralLibrary` (``library``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .cube import to_matrix, wavelengths
from .library import SpectralLibrary


def _result(cube: xr.DataArray, X: np.ndarray, valid: np.ndarray, picks: np.ndarray,
            prefix: str, extra: dict | None = None) -> dict[str, Any]:
    flat = np.flatnonzero(valid)[picks]
    shape = tuple(cube.shape[1:])
    r, c = np.unravel_index(flat, shape)
    S = X[picks]
    names = [f"{prefix}{i + 1}" for i in range(len(picks))]
    out = {"spectra": S, "index": flat, "row": r, "col": c,
           "library": SpectralLibrary(wavelengths(cube), S, names)}
    out.update(extra or {})
    return out


def _reduce(X: np.ndarray, p: int) -> np.ndarray:
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return (X - mu) @ Vt[:p].T


def ppi(cube: xr.DataArray, n_endmembers: int = 5, *, n_skewers: int = 2000,
        threshold: float = 0.0, reduce_to: int | str | None = "auto",
        min_separation: float = 0.1, seed: int = 0) -> dict[str, Any]:
    """Pixel Purity Index (Boardman et al. 1995): project the data on random unit vectors
    ("skewers") and count how often each pixel is an extreme. ``reduce_to`` projects the
    data on its first principal components first (``"auto"``: ``n_endmembers − 1``, the
    simplex dimension; extra noise dimensions otherwise attract the skewers). Returns the
    ``n_endmembers`` purest pixels that are at least ``min_separation`` × the data extent
    apart in that space (so near-duplicates of one corner are skipped), and ``counts``."""
    X, valid = to_matrix(cube)
    if reduce_to == "auto":
        reduce_to = max(n_endmembers - 1, 1)
    Z = _reduce(X, int(reduce_to)) if reduce_to else X - X.mean(0)
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(Z), int)
    for s in range(0, n_skewers, 256):
        k = min(256, n_skewers - s)
        sk = rng.normal(size=(Z.shape[1], k))
        sk /= np.linalg.norm(sk, axis=0)
        P = Z @ sk
        if threshold > 0:
            lo, hi = np.quantile(P, [threshold, 1 - threshold], axis=0)
            counts += ((P <= lo) | (P >= hi)).sum(1)
        else:
            np.add.at(counts, P.argmax(0), 1)
            np.add.at(counts, P.argmin(0), 1)
    order = np.argsort(-counts, kind="stable")
    picks: list[int] = []
    min_sep = min_separation * float(np.linalg.norm(Z.max(0) - Z.min(0)))
    for j in order:
        if counts[j] == 0:
            break
        if all(np.linalg.norm(Z[j] - Z[p]) > min_sep for p in picks):
            picks.append(int(j))
        if len(picks) == n_endmembers:
            break
    cnt = np.zeros(valid.size, int)
    cnt[valid] = counts
    return _result(cube, X, valid, np.array(picks, int), "ppi",
                   {"counts": cnt.reshape(cube.shape[1:])})


def atgp(cube: xr.DataArray, n_endmembers: int = 5) -> dict[str, Any]:
    """Automatic Target Generation Process (Ren & Chang 2003): repeatedly take the pixel with
    the largest norm in the orthogonal complement of the targets found so far."""
    X, valid = to_matrix(cube)
    picks = [int(np.argmax((X**2).sum(1)))]
    for _ in range(n_endmembers - 1):
        U = X[picks].T
        P = np.eye(X.shape[1]) - U @ np.linalg.pinv(U)
        picks.append(int(np.argmax(((X @ P) ** 2).sum(1))))
    return _result(cube, X, valid, np.array(picks), "atgp")


def nfindr(cube: xr.DataArray, n_endmembers: int = 5, *, max_iter: int = 5,
           init: str = "atgp", seed: int = 0) -> dict[str, Any]:
    """N-FINDR (Winter 1999): find the pixels spanning the simplex of maximum volume in the
    (k−1)-dimensional PCA space. Each sweep tries every pixel in every vertex at once using
    Cramer's rule, so it is fully vectorised."""
    X, valid = to_matrix(cube)
    k = n_endmembers
    Z = _reduce(X, k - 1)
    A = np.vstack([np.ones(len(Z)), Z.T])            # (k, N)
    if init == "atgp":
        idx = list(atgp(cube, k)["index"])
        flat_valid = np.flatnonzero(valid)
        picks = [int(np.searchsorted(flat_valid, i)) for i in idx]
    else:
        picks = list(np.random.default_rng(seed).choice(len(Z), k, replace=False))
    vol = abs(np.linalg.det(A[:, picks]))
    for _ in range(max_iter):
        changed = False
        for i in range(k):
            M = A[:, picks]
            try:
                C = np.linalg.solve(M, A)                 # coefficients of every pixel
            except np.linalg.LinAlgError:
                C = np.linalg.lstsq(M, A, rcond=None)[0]
            cand = np.abs(C[i]) * vol                      # |det| with column i replaced
            j = int(np.argmax(cand))
            if cand[j] > vol * (1 + 1e-9) and j not in picks:
                picks[i] = j
                vol = abs(np.linalg.det(A[:, picks]))
                changed = True
        if not changed:
            break
    return _result(cube, X, valid, np.array(picks), "nfindr", {"volume": vol})


def vca(cube: xr.DataArray, n_endmembers: int = 5, *, snr: float | None = None,
        seed: int = 0) -> dict[str, Any]:
    """Vertex Component Analysis (Nascimento & Bioucas-Dias 2005)."""
    X, valid = to_matrix(cube)
    R = X.T                                              # (B, N)
    B, N = R.shape
    p = n_endmembers
    rng = np.random.default_rng(seed)
    mu = R.mean(1, keepdims=True)
    Ud = np.linalg.svd((R - mu) @ (R - mu).T / N)[0][:, :p]
    x_p = Ud.T @ (R - mu)
    if snr is None:
        P_y = np.sum(R**2) / N
        P_x = np.sum(x_p**2) / N + float(np.sum(mu**2))
        snr = 10 * np.log10(max(P_x - p / B * P_y, 1e-12) / max(P_y - P_x, 1e-12))
    snr_th = 15 + 10 * np.log10(p)
    if snr < snr_th:
        d = p - 1
        Ud = Ud[:, :d]
        x = x_p[:d]
        c = np.sqrt(np.max(np.sum(x**2, 0)))
        y = np.vstack([x, c * np.ones((1, N))])
    else:
        d = p
        Ud = np.linalg.svd(R @ R.T / N)[0][:, :d]
        x_p = Ud.T @ R
        u = x_p.mean(1, keepdims=True)
        y = x_p / np.maximum(np.sum(x_p * u, 0, keepdims=True), 1e-12)
    picks = np.zeros(p, int)
    A = np.zeros((p, p))
    A[-1, 0] = 1
    for i in range(p):
        w = rng.normal(size=(p, 1))
        f = w - A @ np.linalg.pinv(A) @ w
        f /= max(np.linalg.norm(f), 1e-12)
        v = (f.T @ y).ravel()
        picks[i] = int(np.argmax(np.abs(v)))
        A[:, i] = y[:, picks[i]]
    return _result(cube, X, valid, picks, "vca", {"snr_db": float(snr)})


def extract_endmembers(cube: xr.DataArray, n_endmembers: int = 5, *, method: str = "nfindr",
                       **kwargs: Any) -> dict[str, Any]:
    """Dispatch to ``ppi``, ``atgp``, ``nfindr`` or ``vca``."""
    return {"ppi": ppi, "atgp": atgp, "nfindr": nfindr, "vca": vca}[method](
        cube, n_endmembers, **kwargs)
