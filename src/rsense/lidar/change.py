"""Multi-temporal lidar change: DEM of Difference, cloud-to-cloud and M3C2 distances,
canopy change."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from ..core.pointcloud import PointCloud
from .features import _tree, _xyz, eigen_features


def dem_of_difference(before: xr.DataArray, after: xr.DataArray, *, lod: float | None = None,
                      sigma_before: float = 0.1, sigma_after: float = 0.1,
                      t: float = 1.96) -> dict[str, Any]:
    """DEM of Difference (after − before) with a level of detection (Brasington et al. 2003).

    Changes smaller than ``lod`` (default ``t·√(σ₁² + σ₂²)``, the propagated vertical error)
    are treated as noise. Returns ``{"dod", "significant", "lod", "erosion_m3",
    "deposition_m3", "net_m3", "eroded_area_m2", "deposited_area_m2"}``.
    """
    b = before.rio.reproject_match(after) if before.shape != after.shape else before
    d = (after - b.values).rename("dod").assign_attrs(units="m")
    lod = float(lod if lod is not None else t * np.hypot(sigma_before, sigma_after))
    v = np.asarray(d.values, float)
    sig = np.isfinite(v) & (np.abs(v) >= lod)
    res = abs(float(after.x[1] - after.x[0])) * abs(float(after.y[1] - after.y[0]))
    ero = sig & (v < 0)
    dep = sig & (v > 0)
    return {"dod": d, "significant": d.copy(data=np.where(sig, v, np.nan)).rename("dod_sig"),
            "lod": lod, "erosion_m3": float(-v[ero].sum() * res),
            "deposition_m3": float(v[dep].sum() * res),
            "net_m3": float(v[sig].sum() * res),
            "eroded_area_m2": float(ero.sum() * res), "deposited_area_m2": float(dep.sum() * res)}


def c2c_distance(reference: PointCloud, compared: PointCloud, *, signed: bool = False,
                 k_normal: int = 12, max_distance: float | None = None) -> PointCloud:
    """Cloud-to-cloud nearest-neighbour distance from each ``compared`` point to ``reference``.

    With ``signed`` the distance takes the sign of the displacement along the reference
    normal (positive = above/outward). Adds ``c2c`` to ``compared``; distances beyond
    ``max_distance`` become NaN.
    """
    R, C = _xyz(reference), _xyz(compared)
    d, i = _tree(R).query(C, k=1, distance_upper_bound=max_distance or np.inf)
    d = d.astype("float32")
    ok = np.isfinite(d)
    if signed:
        nrm = eigen_features(reference, k_normal, features=("nx", "ny", "nz"))
        N = np.column_stack([np.asarray(nrm[k], float) for k in ("nx", "ny", "nz")])
        s = np.ones(len(C))
        s[ok] = np.sign(np.einsum("ij,ij->i", C[ok] - R[i[ok]], N[i[ok]]))
        d = d * np.where(s == 0, 1, s)
    d[~ok] = np.nan
    return compared.with_field("c2c", d)


def m3c2(reference: PointCloud, compared: PointCloud, *, core_points: PointCloud | None = None,
         normal_scale: float = 2.0, projection_scale: float = 1.0, max_depth: float = 5.0,
         registration_error: float = 0.0, min_points: int = 5,
         vertical: bool = False) -> PointCloud:
    """Multiscale Model-to-Model Cloud Comparison (Lague et al. 2013).

    For every core point (default: the reference cloud thinned to ``projection_scale``),
    a normal is fitted to reference points within ``normal_scale/2``; points of each cloud in
    a cylinder of radius ``projection_scale/2`` and half-length ``max_depth`` along that
    normal are projected on it. Output fields: ``m3c2`` (signed distance, compared −
    reference), ``lod95`` = 1.96·(√(σ₁²/n₁ + σ₂²/n₂) + ``registration_error``),
    ``significant`` and ``n1``/``n2``. ``vertical=True`` forces vertical normals (2.5-D).
    """
    R, Cc = _xyz(reference), _xyz(compared)
    cores = core_points if core_points is not None else reference.voxel_downsample(projection_scale)
    P = _xyz(cores)
    tR, tC = _tree(R), _tree(Cc)
    n = len(P)
    if vertical:
        N = np.tile([0.0, 0.0, 1.0], (n, 1))
    else:
        N = np.tile([0.0, 0.0, 1.0], (n, 1))
        for j, nb in enumerate(tR.query_ball_point(P, normal_scale / 2)):
            if len(nb) >= 3:
                Q = R[nb] - R[nb].mean(0)
                _, _, vt = np.linalg.svd(Q, full_matrices=False)
                v = vt[-1]
                N[j] = v if v[2] >= 0 else -v
    r = projection_scale / 2
    search = float(np.hypot(r, max_depth))
    out = {k: np.full(n, np.nan, "float32") for k in ("m3c2", "lod95", "n1", "n2", "std1", "std2")}

    def project(tree, X, j):
        nb = tree.query_ball_point(P[j], search)
        if not nb:
            return np.empty(0)
        V = X[nb] - P[j]
        along = V @ N[j]
        perp = np.linalg.norm(V - np.outer(along, N[j]), axis=1)
        return along[(perp <= r) & (np.abs(along) <= max_depth)]

    for j in range(n):
        a1 = project(tR, R, j)
        a2 = project(tC, Cc, j)
        out["n1"][j], out["n2"][j] = len(a1), len(a2)
        if len(a1) < min_points or len(a2) < min_points:
            continue
        s1, s2 = a1.std(ddof=1), a2.std(ddof=1)
        out["m3c2"][j] = np.mean(a2) - np.mean(a1)
        out["std1"][j], out["std2"][j] = s1, s2
        out["lod95"][j] = 1.96 * (np.sqrt(s1**2 / len(a1) + s2**2 / len(a2)) + registration_error)
    res = cores.to_cpu()
    for k, v in out.items():
        res = res.with_field(k, v)
    res = res.with_field("nx", N[:, 0].astype("float32")).with_field(
        "ny", N[:, 1].astype("float32")).with_field("nz", N[:, 2].astype("float32"))
    sig = np.isfinite(out["m3c2"]) & (np.abs(out["m3c2"]) > out["lod95"])
    return res.with_field("significant", sig)


def canopy_change(chm_before: xr.DataArray, chm_after: xr.DataArray, *,
                  loss_threshold: float = 3.0, gain_threshold: float = 1.0,
                  min_canopy: float = 2.0, min_area: float = 0.0) -> xr.Dataset:
    """Canopy loss and growth between two CHMs.

    ``loss``: cells that were canopy (> ``min_canopy``) and dropped by more than
    ``loss_threshold`` m (harvest, windthrow, fire). ``gain``: growth above
    ``gain_threshold`` m. Loss patches smaller than ``min_area`` m² are removed.
    """
    from scipy import ndimage as snd

    b = chm_before.rio.reproject_match(chm_after) if chm_before.shape != chm_after.shape \
        else chm_before
    d = np.asarray(chm_after.values, float) - np.asarray(b.values, float)
    loss = (np.asarray(b.values) > min_canopy) & (d < -loss_threshold)
    if min_area > 0:
        res = abs(float(chm_after.x[1] - chm_after.x[0])) ** 2
        lab, nl = snd.label(loss)
        if nl:
            sizes = snd.sum(loss, lab, np.arange(1, nl + 1)) * res
            loss = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
    gain = d > gain_threshold
    tpl = chm_after
    return xr.Dataset({"difference": tpl.copy(data=d.astype("float32")),
                       "loss": tpl.copy(data=loss.astype("uint8")),
                       "gain": tpl.copy(data=(gain & ~loss).astype("uint8"))})
