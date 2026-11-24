"""Spectral unmixing: vectorised least squares (unconstrained, sum-to-one, non-negative, fully
constrained), MESMA and sparse unmixing against a library, plus reconstruction error.

NNLS and FCLS are solved exactly for up to ``exact_max`` (8) endmembers by evaluating every
support in closed form for all pixels at once; larger problems (and sparse unmixing) use
FISTA (accelerated projected gradient onto the orthant or the probability simplex).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
import xarray as xr

from .cube import from_matrix, to_matrix
from .library import SpectralLibrary, as_library


def project_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of each column of ``V`` (k × N) onto the probability simplex."""
    k = V.shape[0]
    U = -np.sort(-V, axis=0)
    css = np.cumsum(U, axis=0) - 1
    j = np.arange(1, k + 1)[:, None]
    cond = U - css / j > 0
    rho = k - 1 - np.argmax(cond[::-1], axis=0)
    theta = css[rho, np.arange(V.shape[1])] / (rho + 1)
    return np.maximum(V - theta, 0)


def _fista(A: np.ndarray, X: np.ndarray, proj, *, l1: float = 0.0, max_iter: int = 500,
           tol: float = 1e-6, F0: np.ndarray | None = None) -> np.ndarray:
    """min ½‖A F − X‖² (+ l1‖F‖₁) s.t. proj; A (B × k), X (B × N) → F (k × N).

    Columns (pixels) that have converged (max change < ``tol``) leave the active set, so the
    cost falls as the solution settles."""
    AtA, AtX = A.T @ A, A.T @ X
    L = float(np.linalg.eigvalsh(AtA).max()) or 1.0
    k, N = A.shape[1], X.shape[1]
    F = np.full((k, N), 1.0 / k) if F0 is None else F0.copy()
    Y = F.copy()
    out = F.copy()
    act = np.arange(N)
    Fa, Ya, Xa, t = F, Y, AtX, 1.0
    for it in range(max_iter):
        Fn = proj(Ya - (AtA @ Ya - Xa + l1) / L)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Ya = Fn + ((t - 1) / tn) * (Fn - Fa)
        delta = np.max(np.abs(Fn - Fa), axis=0)
        Fa, t = Fn, tn
        if (it + 1) % 10 == 0:                 # retire converged pixels every 10 iterations
            done = delta < tol
            if done.all():
                break
            if done.mean() > 0.2:
                out[:, act[done]] = Fa[:, done]
                keep = ~done
                act, Fa, Ya, Xa = act[keep], Fa[:, keep], Ya[:, keep], Xa[:, keep]
    out[:, act] = Fa
    return out


def _inv(M: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def _exact_subsets(E: np.ndarray, X: np.ndarray, sum_to_one: bool) -> np.ndarray:
    """Exact NNLS / FCLS for small k: the optimum is the equality-constrained (or plain)
    least-squares solution on its support, so solve every support in closed form for all
    pixels at once and keep the feasible solution with the smallest residual."""
    from itertools import combinations as _comb

    k = E.shape[0]
    G = E @ E.T                                   # (k, k)
    EX = X @ E.T                                  # (N, k)
    xx = np.einsum("ij,ij->i", X, X)
    best = np.full(len(X), np.inf)
    F = np.zeros((len(X), k))
    for r in range(1, k + 1):
        for S in _comb(range(k), r):
            S = list(S)
            Gs = G[np.ix_(S, S)]
            if sum_to_one:                            # KKT system: well-posed even with shade
                K = np.zeros((r + 1, r + 1))
                K[:r, :r], K[:r, r], K[r, :r] = Gs, 1.0, 1.0
                Ki = _inv(K)
                f = np.column_stack([EX[:, S], np.ones(len(X))]) @ Ki.T
                f = f[:, :r]
            else:
                f = EX[:, S] @ _inv(Gs).T
            ok = np.all(f >= -1e-10, axis=1)
            if not ok.any():
                continue
            res = xx - 2 * np.einsum("ij,ij->i", f, EX[:, S]) + np.einsum("ij,jk,ik->i", f, Gs, f)
            better = ok & (res < best - 1e-15)
            if better.any():
                best[better] = res[better]
                F[better] = 0
                F[np.ix_(np.flatnonzero(better), S)] = np.clip(f[better], 0, None)
    return F


def solve_abundances(E: np.ndarray, X: np.ndarray, method: str = "fcls", *,
                     max_iter: int = 500, tol: float = 1e-6, exact_max: int = 8) -> np.ndarray:
    """Abundances ``(pixels, k)`` for spectra ``X`` (pixels × bands) and endmembers ``E``
    (k × bands). ``method``: ``ls``, ``scls`` (sum-to-one), ``nnls``, ``fcls``.

    NNLS/FCLS are solved exactly (all supports, vectorised) when ``k <= exact_max``, else
    with FISTA."""
    A = np.asarray(E, float).T
    Xt = np.asarray(X, float).T
    if method == "ls":
        return (np.linalg.pinv(A) @ Xt).T
    if method == "scls":
        f = np.linalg.pinv(A) @ Xt
        g = np.linalg.solve(A.T @ A, np.ones(A.shape[1]))
        return (f + np.outer(g, (1 - f.sum(0)) / g.sum())).T
    if method in ("nnls", "fcls") and A.shape[1] <= exact_max:
        return _exact_subsets(np.asarray(E, float), np.asarray(X, float), method == "fcls")
    if method == "nnls":
        return _fista(A, Xt, lambda V: np.maximum(V, 0), max_iter=max_iter, tol=tol).T
    if method == "fcls":
        return _fista(A, Xt, project_simplex, max_iter=max_iter, tol=tol).T
    raise ValueError(f"Unknown method {method!r}")


def unmix(cube: xr.DataArray, endmembers: Any, *,
          method: Literal["ls", "scls", "nnls", "fcls"] = "fcls", max_iter: int = 500,
          chunk: int = 500_000) -> xr.Dataset:
    """Linear unmixing of a cube. ``endmembers``: SpectralLibrary, dict ``{name: spectrum}``,
    an ``extract_endmembers`` result, or a (k × bands) array. Returns a Dataset with one
    abundance map per endmember plus ``rmse``."""
    if isinstance(endmembers, dict) and "library" in endmembers:
        endmembers = endmembers["library"]
    lib = as_library(endmembers, cube)
    ok = np.all(np.isfinite(lib.spectra), axis=0)
    X, valid = to_matrix(cube)
    E = lib.spectra[:, ok]
    Xo = X[:, ok]
    F = np.vstack([solve_abundances(E, Xo[s:s + chunk], method, max_iter=max_iter)
                   for s in range(0, len(Xo), chunk)]) if len(Xo) else np.zeros((0, len(E)))
    rmse = np.sqrt(np.mean((Xo - F @ E) ** 2, axis=1))
    da = from_matrix(F, cube, valid, names=lib.names, dim="endmember")
    ds = da.to_dataset("endmember")
    ds["rmse"] = from_matrix(rmse, cube, valid).isel(component=0, drop=True)
    return ds.assign_attrs(method=method)


def mesma(cube: xr.DataArray, library: SpectralLibrary | Mapping[str, Any], classes: Sequence[str]
          | None = None, *, max_classes: int = 2, shade: bool = True, max_rmse: float = 0.025,
          fraction_range: tuple[float, float] = (-0.05, 1.05), max_iter: int = 300
          ) -> xr.Dataset:
    """Multiple Endmember Spectral Mixture Analysis (Roberts et al. 1998).

    ``library`` is a SpectralLibrary with one class per spectrum (``classes`` list, or a
    ``class`` column in ``library.metadata``), or a dict ``{class: [spectra...]}`` on the
    cube's bands. Every model of 1..``max_classes`` classes (one spectrum each, plus
    photometric shade when ``shade``) is fitted with FCLS; each pixel keeps the model with the
    lowest RMSE that respects ``fraction_range`` and ``max_rmse``. Returns per-class fractions
    (shade-normalised when ``shade``), ``shade``, ``rmse`` and the chosen ``model`` index
    (``attrs["models"]`` lists them).
    """
    if isinstance(library, Mapping):
        names, spec, cls = [], [], []
        for c, specs in library.items():
            for i, s in enumerate(np.atleast_2d(np.asarray(specs, float))):
                names.append(f"{c}_{i}")
                spec.append(s)
                cls.append(c)
        lib = as_library({n: s for n, s in zip(names, spec)}, cube)
    else:
        lib = as_library(library, cube)
        cls = list(classes) if classes is not None else list(library.metadata["class"])
    ucls = list(dict.fromkeys(cls))
    X, valid = to_matrix(cube)
    ok = np.all(np.isfinite(lib.spectra), axis=0)
    Xo, S = X[:, ok], lib.spectra[:, ok]
    best = np.full(len(X), np.inf)
    best_model = np.full(len(X), -1)
    frac = np.zeros((len(X), len(ucls)))
    shade_f = np.zeros(len(X))
    models = []
    by_cls = {c: [i for i, k in enumerate(cls) if k == c] for c in ucls}
    for m in range(1, max_classes + 1):
        for combo in combinations(ucls, m):
            for members in product(*[by_cls[c] for c in combo]):
                E = S[list(members)]
                if shade:
                    E = np.vstack([E, np.zeros(E.shape[1])])
                F = solve_abundances(E, Xo, "fcls", max_iter=max_iter)
                r = np.sqrt(np.mean((Xo - F @ E) ** 2, axis=1))
                good = np.all((F >= fraction_range[0]) & (F <= fraction_range[1]), axis=1) & \
                    (r <= max_rmse) & (r < best)
                mid = len(models)
                models.append(tuple(lib.names[i] for i in members))
                best[good] = r[good]
                best_model[good] = mid
                frac[good] = 0
                for j, c in enumerate(combo):
                    frac[good, ucls.index(c)] = F[good, j]
                shade_f[good] = F[good, -1] if shade else 0
    if shade:
        tot = np.maximum(1 - shade_f, 1e-9)
        frac = np.where(best_model[:, None] >= 0, frac / tot[:, None], np.nan)
    frac[best_model < 0] = np.nan
    ds = from_matrix(frac, cube, valid, names=ucls, dim="class").to_dataset("class")
    ds["shade"] = from_matrix(np.where(best_model >= 0, shade_f, np.nan), cube, valid
                              ).isel(component=0, drop=True)
    ds["rmse"] = from_matrix(np.where(np.isfinite(best), best, np.nan), cube, valid
                             ).isel(component=0, drop=True)
    ds["model"] = from_matrix(best_model.astype(float), cube, valid, fill=-1
                              ).isel(component=0, drop=True).astype("int32")
    return ds.assign_attrs(models=[" + ".join(m) for m in models])


def sparse_unmix(cube: xr.DataArray, library: Any, *, lam: float = 1e-3, max_iter: int = 800,
                 top: int | None = None) -> xr.Dataset:
    """Sparse non-negative unmixing with a large library (SUnSAL-style, solved with FISTA):
    ``min ½‖Ef − x‖² + λ‖f‖₁, f ≥ 0``. Returns abundances for every library spectrum (or the
    ``top`` most used), ``rmse`` and ``n_active`` (non-zero members per pixel)."""
    lib = as_library(library, cube)
    ok = np.all(np.isfinite(lib.spectra), axis=0)
    X, valid = to_matrix(cube)
    A = lib.spectra[:, ok].T
    F = _fista(A, X[:, ok].T, lambda V: np.maximum(V, 0), l1=lam, max_iter=max_iter).T
    F[F < 1e-4] = 0
    rmse = np.sqrt(np.mean((X[:, ok] - F @ A.T) ** 2, axis=1))
    keep = np.arange(len(lib.names))
    if top:
        keep = np.argsort(-F.mean(0))[:top]
    ds = from_matrix(F[:, keep], cube, valid, names=[lib.names[i] for i in keep],
                     dim="endmember").to_dataset("endmember")
    ds["rmse"] = from_matrix(rmse, cube, valid).isel(component=0, drop=True)
    ds["n_active"] = from_matrix((F > 0).sum(1).astype(float), cube, valid
                                 ).isel(component=0, drop=True)
    return ds


def reconstruct(abundances: xr.Dataset, endmembers: Any, cube: xr.DataArray) -> xr.DataArray:
    """Rebuild spectra from abundance maps (for residual and error analysis)."""
    lib = as_library(endmembers, cube)
    F = np.stack([np.asarray(abundances[n].values, float) for n in lib.names])
    out = np.einsum("kb,kyx->byx", np.nan_to_num(lib.spectra), np.nan_to_num(F))
    return cube.copy(data=out.astype("float32")).rename("reconstructed")
