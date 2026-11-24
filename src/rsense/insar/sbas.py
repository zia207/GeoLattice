"""Small Baseline Subset (SBAS) InSAR: network, interferogram stack, unwrapping-error correction,
DEM-error correction and SVD inversion.

The chain follows Berardino et al. (2002) and MintPy (Yunjun et al. 2019):

* :func:`sbas_network` / :func:`network_components` -- choose interferometric pairs by temporal and
  perpendicular baseline (or nearest neighbours) and check how the network connects;
* :func:`interferogram_stack` -- from a stack co-registered with
  :func:`geolattice.rsense.sar.coregister_stack`, form, deburst, multilook, filter and unwrap every pair
  in one call, keeping coherence and the DEM-error sensitivity of each pair;
* :func:`loop_closure` / :func:`correct_unwrapping_errors` -- phase closure of interferogram triplets
  and the integer (2 pi) corrections that make it vanish with the fewest changes (an exact L1
  integer programme per distinct closure pattern, as in MintPy's phase-closure correction);
* :func:`sbas_svd` -- the original SBAS inversion: mean velocities between consecutive dates by
  SVD (minimum-norm), which bridges disconnected subsets of the network, integrated to a time
  series; per-pixel weights and missing interferograms are handled;
* :func:`dem_error_correction` -- Fattahi & Amelung (2013): fit each pixel's time series with a
  temporal model plus a term proportional to its perpendicular-baseline history, and remove the
  DEM-error part.

Phase convention as in the rest of :mod:`geolattice.rsense.insar`: interferograms are
``earlier * conj(later)``, so :func:`geolattice.rsense.insar.phase_to_los` gives motion toward the
satellite as positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# ----------------------------------------------------------------------------- network
def sbas_network(dates: Sequence[Any], bperp: Any | None = None, *, max_temporal_days: float | None = None,
                 max_perpendicular: float | None = None, n_neighbors: int | None = None,
                 connect: bool = False) -> list[tuple[str, str]]:
    """Choose small-baseline interferometric pairs ``(earlier, later)`` among ``dates``.

    A pair is kept when its time span is at most ``max_temporal_days``, its perpendicular baseline
    difference (``bperp``: one value per date, metres) at most ``max_perpendicular``, and -- with
    ``n_neighbors`` -- the later date is among the next ``n_neighbors`` acquisitions (a sequential
    network). With ``connect=True``, consecutive dates are additionally paired wherever the network
    would otherwise split into disconnected subsets (see :func:`network_components`); without it a
    split network is returned as it is, and :func:`sbas_svd` bridges the gap.
    """
    d = pd.to_datetime(list(dates))
    order = np.argsort(d.values)
    d = d[order]
    names = [str(x.date()) for x in d]
    bp = None if bperp is None else np.asarray(bperp, float)[order]
    pairs = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if n_neighbors is not None and j - i > n_neighbors:
                break
            if max_temporal_days is not None and (d[j] - d[i]).days > max_temporal_days:
                break
            if max_perpendicular is not None and bp is not None and abs(bp[j] - bp[i]) > max_perpendicular:
                continue
            pairs.append((names[i], names[j]))
    if connect:
        for i in range(len(names) - 1):
            comps = network_components(pairs, names)
            if len(comps) == 1:
                break
            if not any(names[i] in c and names[i + 1] in c for c in comps):
                pairs.append((names[i], names[i + 1]))
        pairs = sorted(set(pairs), key=lambda p: (p[0], p[1]))
    return pairs


def network_components(pairs: Sequence[tuple[Any, Any]], dates: Sequence[Any] | None = None) -> list[list[str]]:
    """Connected subsets of an interferogram network (dates linked by at least one chain of pairs),
    each sorted, largest first. Dates in ``dates`` without any pair form their own subset."""
    names = sorted({str(pd.Timestamp(x).date()) for p in pairs for x in p} |
                   ({str(pd.Timestamp(x).date()) for x in dates} if dates is not None else set()))
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(str(pd.Timestamp(a).date())), find(str(pd.Timestamp(b).date()))
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=len, reverse=True)


# ----------------------------------------------------------------------------- interferogram stack
def interferogram_stack(reference: dict, coreg: dict, pairs: Sequence[tuple[str, str]], *, reference_date: str,
                        looks: tuple[int, int] = (2, 8), alpha: float = 0.6, coherence_window: int = 3,
                        unwrap: bool = True, cost: str = "defo", min_coherence: float = 0.25,
                        verbose: bool = False) -> xr.Dataset:
    """Form, deburst, multilook, filter and unwrap every interferogram of an SBAS network.

    ``reference`` is the per-burst dict of the stack reference date and ``coreg`` the result of
    :func:`geolattice.rsense.sar.coregister_stack` for the other dates (keyed by date string);
    ``reference_date`` is the reference's key. For each pair ``(i, j)`` the differential
    interferogram ``s_i conj(s_j)`` with the DEM phase of ``delta_range_j - delta_range_i`` removed
    is debursted (:func:`geolattice.rsense.sar.deburst`), multilooked by ``looks`` (azimuth, range),
    its coherence estimated with a further ``coherence_window`` box, Goldstein-filtered (``alpha``)
    and, with ``unwrap``, unwrapped by SNAPHU (``cost``), masking coherence below
    ``min_coherence``.

    Returns a Dataset on the multilooked radar grid with ``unwrapped`` and ``coherence``
    ``(pair, y, x)`` (wrapped ``phase`` too when ``unwrap=False``), ``kz`` ``(date, y, x)`` (each
    date's range difference per metre of height relative to the stack reference: the
    DEM-error sensitivity :func:`dem_error_correction` uses), ``incidence`` and ``dem_height``
    ``(y, x)``, and the coordinates ``reference``/``secondary`` (per pair), ``date`` and
    ``azimuth_time`` (per row). Attributes give ``wavelength``, ``near_range`` and the multilooked
    ``range_spacing`` / ``azimuth_spacing`` -- what :func:`geolattice.rsense.sar.terrain_correct` needs.
    """
    from scipy import ndimage

    from ..sar.sentinel1 import SPEED_OF_LIGHT, deburst
    from ..sar.sentinel1 import multilook as ml_real
    from .dinsar import differential_interferogram
    from .interferogram import goldstein_filter, multilook_complex
    from .unwrap import unwrap as unwrap_fn

    anns, ids = list(reference["ann"]), list(reference["burst"])
    nb = len(ids)
    lam = anns[0]["wavelength"]
    stack = coreg["stack"]
    slc = {reference_date: list(reference["slc"])} | {d: v["slc"] for d, v in stack.items()}
    zeros = [np.zeros(x.shape, "float32") for x in reference["slc"]]
    drange = {reference_date: zeros} | {d: v["delta_range"] for d, v in stack.items()}
    kz = {reference_date: zeros} | {d: v.get("kz", zeros) for d, v in stack.items()}

    def db(layers):
        return deburst(list(layers), anns[0], ids, anns=anns)

    def ml(x):
        return multilook_complex(x, looks) if np.iscomplexobj(x) else ml_real(np.asarray(x, "float64"), looks)

    def box(x):
        if np.iscomplexobj(x):
            return ndimage.uniform_filter(x.real, coherence_window) + 1j * ndimage.uniform_filter(x.imag, coherence_window)
        return ndimage.uniform_filter(x, coherence_window)

    power: dict[str, np.ndarray] = {}

    def pw(d):
        if d not in power:
            power[d] = ml(db([np.abs(x) ** 2 for x in slc[d]])[0])
        return power[d]

    _, times = db(reference["slc"])
    nt = (times.size // looks[0]) * looks[0]
    t_ml = times[:nt].astype("int64").reshape(-1, looks[0]).mean(1).astype("datetime64[ns]")
    unw_l, coh_l, ph_l = [], [], []
    for n, (i, j) in enumerate(pairs):
        z = ml(db([differential_interferogram(slc[i][k], slc[j][k], drange[j][k] - drange[i][k], lam)
                   for k in range(nb)])[0])
        p1, p2 = pw(i), pw(j)
        valid = (np.abs(z) > 0) & (p1 > 0) & (p2 > 0)
        coh = np.where(valid, np.clip(np.abs(box(z)) / np.sqrt(box(p1) * box(p2) + 1e-30), 0, 1), 0).astype("float32")
        filt = goldstein_filter(np.where(valid, z / (np.abs(z) + 1e-30), 0), alpha=alpha, patch=32, overlap=8)
        coh_l.append(coh)
        if unwrap:
            u = unwrap_fn(filt, coh, method="snaphu", cost=cost, nlooks=coherence_window ** 2 * looks[0] * looks[1],
                          mask=valid & (coh > min_coherence))
            unw_l.append(np.asarray(u, "float32"))
        else:
            ph_l.append(np.where(valid, np.angle(filt), np.nan).astype("float32"))
        if verbose:
            print(f"  {n + 1}/{len(pairs)} {i} → {j}: median coherence {np.median(coh[valid]):.2f}", flush=True)
    dates = sorted(slc)
    kz_ml = np.stack([ml(db(kz[d])[0]) for d in dates]).astype("float32")
    geom = {k: ml(db(coreg["geometry"][k])[0]).astype("float32") for k in ("incidence", "dem_height")}
    H, W = coh_l[0].shape
    data = {"coherence": (("pair", "y", "x"), np.stack(coh_l)),
            "kz": (("date", "y", "x"), kz_ml[:, :H, :W]),
            "incidence": (("y", "x"), geom["incidence"][:H, :W]),
            "dem_height": (("y", "x"), geom["dem_height"][:H, :W])}
    if unwrap:
        data["unwrapped"] = (("pair", "y", "x"), np.stack(unw_l))
    else:
        data["phase"] = (("pair", "y", "x"), np.stack(ph_l))
    a0 = anns[0]
    near = a0["slant_range_time"] * SPEED_OF_LIGHT / 2 + (reference["col0"][0] + (looks[1] - 1) / 2) * a0["range_pixel_spacing"]
    ds = xr.Dataset(data, coords={"pair": np.arange(len(pairs)), "reference": ("pair", [p[0] for p in pairs]),
                                  "secondary": ("pair", [p[1] for p in pairs]), "date": dates,
                                  "azimuth_time": ("y", t_ml[:H]), "y": np.arange(H), "x": np.arange(W)})
    ds.attrs.update(wavelength=lam, near_range=float(near), range_spacing=looks[1] * a0["range_pixel_spacing"],
                    azimuth_spacing=looks[0] * a0["azimuth_pixel_spacing"], looks=list(looks),
                    reference_date=reference_date, heading=float(a0["heading"]))
    return ds


# ----------------------------------------------------------------------------- unwrapping errors
def _triplets(pairs: Sequence[tuple[Any, Any]]) -> list[tuple[int, int, int]]:
    """Index triplets ``(ij, jk, ik)`` of interferograms closing a loop of three dates."""
    key = {(str(a), str(b)): n for n, (a, b) in enumerate(pairs)}
    dates = sorted({str(x) for p in pairs for x in p})
    out = []
    for a in range(len(dates)):
        for b in range(a + 1, len(dates)):
            for c in range(b + 1, len(dates)):
                ij, jk, ik = (dates[a], dates[b]), (dates[b], dates[c]), (dates[a], dates[c])
                if ij in key and jk in key and ik in key:
                    out.append((key[ij], key[jk], key[ik]))
    return out


def loop_closure(unwrapped: Any, pairs: Sequence[tuple[Any, Any]]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Phase closure ``u_ij + u_jk - u_ik`` of every triplet of interferograms in the network,
    ``(n_triplets, ...)``. For correctly unwrapped interferograms it is close to 0; a whole multiple of
    2 pi flags an unwrapping error in one of the three. Returns ``(closure, triplets)``."""
    u = np.asarray(getattr(unwrapped, "values", unwrapped), float)
    tri = _triplets([(str(a), str(b)) for a, b in pairs])
    if not tri:
        raise ValueError("The network has no closed triplets of interferograms")
    closure = np.stack([u[i] + u[j] - u[k] for i, j, k in tri])
    return closure, tri


def correct_unwrapping_errors(unwrapped: Any, pairs: Sequence[tuple[Any, Any]], *, max_cycles: int = 5
                              ) -> tuple[Any, dict[str, Any]]:
    """Correct 2 pi unwrapping errors with phase closure (MintPy's ``phase_closure`` method).

    For every pixel, the closure of each triplet (:func:`loop_closure`) is rounded to whole cycles
    ``c``; the integer corrections ``a`` (one per interferogram) that make ``C a = -c`` with the
    smallest total ``sum |a|`` are found by mixed-integer linear programming (``scipy.optimize.milp``;
    rounded noisy closures can be inconsistent, so a closure residual is allowed at ten times the
    cost of a correction) --
    once per distinct closure pattern, since many pixels share one -- and ``2 pi a`` is added.
    Pixels whose closure is already zero, or that have a missing interferogram, are left as they
    are. Returns ``(corrected, info)`` with ``info`` holding ``corrections`` (cycles per pair and
    pixel), ``pixels_corrected`` and ``patterns``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    u = np.asarray(getattr(unwrapped, "values", unwrapped), float)
    closure, tri = loop_closure(u, pairs)
    n_pairs = u.shape[0]
    C = np.zeros((len(tri), n_pairs))
    for r, (i, j, k) in enumerate(tri):
        C[r, i] += 1
        C[r, j] += 1
        C[r, k] -= 1
    flat = closure.reshape(len(tri), -1)
    ok = np.all(np.isfinite(u.reshape(n_pairs, -1)), axis=0)
    cyc = np.zeros(flat.shape, int)
    cyc[:, ok] = np.round(flat[:, ok] / (2 * np.pi)).astype(int)
    need = ok & np.any(cyc != 0, axis=0)
    corrections = np.zeros((n_pairs, flat.shape[1]), int)
    patterns, inverse = (np.unique(cyc[:, need].T, axis=0, return_inverse=True) if need.any()
                         else (np.zeros((0, len(tri)), int), np.zeros(0, int)))
    solutions = np.zeros((len(patterns), n_pairs), int)
    n_tri = len(tri)
    # variables: a+ a- (corrections) and r+ r- (closure residual); C (a+ - a-) - (r+ - r-) = -c.
    # Residuals are allowed (rounded noisy closures need not be consistent) but cost 10x a correction.
    cost = np.r_[np.ones(2 * n_pairs), 10.0 * np.ones(2 * n_tri)]
    A = np.hstack([C, -C, -np.eye(n_tri), np.eye(n_tri)])
    for p, c in enumerate(patterns):
        res = milp(cost, constraints=LinearConstraint(A, -c, -c), integrality=np.ones(A.shape[1]),
                   bounds=Bounds(0, max_cycles))
        if res.success:
            x = np.round(res.x).astype(int)
            solutions[p] = x[:n_pairs] - x[n_pairs:2 * n_pairs]
    if need.any():
        corrections[:, need] = solutions[np.ravel(inverse)].T
    out = u + 2 * np.pi * corrections.reshape(u.shape)
    info = {"corrections": corrections.reshape(u.shape), "pixels_corrected": int(np.any(corrections != 0, axis=0).sum()),
            "patterns": int(len(patterns)), "triplets": tri}
    if isinstance(unwrapped, xr.DataArray):
        out = unwrapped.copy(data=out.astype(unwrapped.dtype))
    return out, info


# ----------------------------------------------------------------------------- inversion
def sbas_svd(ifgs: Any, pairs: Sequence[tuple[Any, Any]], *, weights: Any | None = None, chunk: int = 20000
             ) -> xr.DataArray | np.ndarray:
    """SBAS inversion of unwrapped interferograms (``(n_pairs, ...)``, phase or displacement) by
    singular value decomposition (Berardino et al. 2002).

    The unknowns are the mean velocities between consecutive dates; row ``k`` of the design matrix
    holds the time intervals spanned by interferogram ``k``. The SVD (pseudo-inverse) gives the
    minimum-norm solution, so a network split into disconnected subsets still yields a continuous
    time series instead of an arbitrary jump. Integrating the velocities gives the series at every
    date, zero at the first. Pixels with missing interferograms, and ``weights`` (e.g. coherence,
    same shape as ``ifgs``), are solved pixel by pixel in vectorised chunks; otherwise one
    pseudo-inverse serves every pixel. Returns ``(n_dates, ...)`` (a DataArray with a ``time``
    dimension when ``ifgs`` is one).
    """
    d = np.asarray(getattr(ifgs, "values", ifgs), float)
    dates = sorted({pd.Timestamp(x) for p in pairs for x in p})
    idx = {t: n for n, t in enumerate(dates)}
    dt = np.diff([(t - dates[0]).days / 365.25 for t in dates])
    B = np.zeros((len(pairs), len(dates) - 1))
    for k, (a, b) in enumerate(pairs):
        i, j = sorted((idx[pd.Timestamp(a)], idx[pd.Timestamp(b)]))
        sign = 1.0 if pd.Timestamp(a) <= pd.Timestamp(b) else -1.0
        B[k, i:j] = sign * dt[i:j]
    spatial = d.shape[1:]
    Y = d.reshape(len(pairs), -1)
    V = np.full((len(dates) - 1, Y.shape[1]), np.nan)
    complete = np.all(np.isfinite(Y), axis=0)
    if weights is None:
        V[:, complete] = np.linalg.pinv(B) @ Y[:, complete]
        rest = np.flatnonzero(~complete & np.any(np.isfinite(Y), axis=0))
        Wt = None
    else:
        Wt = np.asarray(getattr(weights, "values", weights), float).reshape(len(pairs), -1)
        rest = np.flatnonzero(np.any(np.isfinite(Y), axis=0))
    for s in range(0, rest.size, chunk):
        cols = rest[s:s + chunk]
        y = Y[:, cols].T                                       # (P, M)
        w = np.isfinite(y).astype(float) if Wt is None else np.where(np.isfinite(y), np.clip(Wt[:, cols].T, 0, None), 0.0)
        sw = np.sqrt(w)
        Bw = sw[:, :, None] * B[None]                          # (P, M, N-1)
        V[:, cols] = (np.linalg.pinv(Bw) @ (sw * np.nan_to_num(y))[:, :, None])[:, :, 0].T
    series = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(V * dt[:, None], axis=0)])
    series[:, ~np.any(np.isfinite(Y), axis=0)] = np.nan
    series = series.reshape((len(dates),) + spatial)
    if isinstance(ifgs, xr.DataArray):
        template = ifgs.isel({ifgs.dims[0]: 0}, drop=True)
        out = xr.concat([template.copy(data=s) for s in series], dim=pd.Index(dates, name="time"))
        return out.rename("timeseries").assign_attrs(n_components=len(network_components(pairs)))
    return series


def dem_error_correction(series: Any, sensitivity: Any, time_years: Any, *, seasonal: bool = False,
                         chunk: int = 200000) -> tuple[Any, np.ndarray]:
    """DEM-error correction of an SBAS time series (Fattahi & Amelung 2013).

    Each pixel's series ``(n_dates, ...)`` is fitted by least squares with a temporal model (offset
    and linear rate, plus annual sine/cosine with ``seasonal``) and a term ``sensitivity * dz``,
    where ``sensitivity`` (``(n_dates,)`` or ``(n_dates, ...)``) is how the series changes per metre
    of DEM error at each date: for LOS displacement toward the satellite, ``-kz`` (``kz`` from
    :func:`interferogram_stack`, relative to the first date); for phase, ``4 pi / lambda * kz``.
    Because the perpendicular baseline history is not smooth in time, the DEM error separates from
    the deformation. Returns ``(corrected series, dz)``, ``dz`` in metres.
    """
    s = np.asarray(getattr(series, "values", series), float)
    n = s.shape[0]
    sp = s.shape[1:]
    Y = s.reshape(n, -1)
    sens = np.asarray(getattr(sensitivity, "values", sensitivity), float)
    K = np.broadcast_to(sens[:, None], (n, Y.shape[1])) if sens.ndim == 1 else sens.reshape(n, -1)
    t = np.asarray(time_years, float)
    cols = [np.ones(n), t] + ([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)] if seasonal else [])
    T = np.stack(cols, 1)                                       # (n, m)
    dz = np.full(Y.shape[1], np.nan)
    ok = np.flatnonzero(np.all(np.isfinite(Y), axis=0) & np.all(np.isfinite(K), axis=0))
    for st in range(0, ok.size, chunk):
        c = ok[st:st + chunk]
        G = np.concatenate([np.broadcast_to(T, (c.size,) + T.shape), K[:, c].T[:, :, None]], axis=2)   # (P, n, m+1)
        GtG = np.einsum("pnk,pnl->pkl", G, G)
        Gty = np.einsum("pnk,np->pk", G, Y[:, c])
        sol = np.linalg.solve(GtG + 1e-12 * np.eye(GtG.shape[-1]), Gty[:, :, None])[:, :, 0]
        dz[c] = sol[:, -1]
    corrected = Y - K * np.nan_to_num(dz)[None, :]
    corrected[:, np.isnan(dz)] = Y[:, np.isnan(dz)]
    corrected = corrected.reshape(s.shape)
    if isinstance(series, xr.DataArray):
        corrected = series.copy(data=corrected)
    return corrected, dz.reshape(sp)
