"""Linear spectral unmixing: estimate sub-pixel fractions of endmember materials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr

from ..backend import get_xp, to_cpu
from .bands import as_dataset


def endmembers_from_samples(obj: xr.Dataset | xr.DataArray, samples, label_col: str,
                            bands: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Mean spectrum per class at sample points (GeoDataFrame of points)."""
    ds = as_dataset(obj)
    bands = list(bands or ds.data_vars)
    pts = samples.to_crs(ds.rio.crs) if samples.crs and ds.rio.crs else samples
    xs = xr.DataArray(pts.geometry.x.values, dims="pt")
    ys = xr.DataArray(pts.geometry.y.values, dims="pt")
    vals = np.stack([to_cpu(ds[b].sel(x=xs, y=ys, method="nearest").values) for b in bands], 1)
    out = {}
    for lab in pts[label_col].unique():
        m = (pts[label_col] == lab).to_numpy()
        out[str(lab)] = np.nanmean(vals[m], axis=0)
    return out


def unmix(obj: xr.Dataset | xr.DataArray, endmembers: Mapping[str, Sequence[float]], *,
          bands: Sequence[str] | None = None,
          method: Literal["ls", "scls", "fcls"] = "fcls", delta: float = 1e3) -> xr.Dataset:
    """Linear spectral unmixing.

    Parameters
    ----------
    obj : reflectance Dataset (one variable per band) or DataArray with ``band`` dim.
    endmembers : ``{name: spectrum}``, spectra ordered like ``bands``.
    method :
        ``ls``   unconstrained least squares (fast, vectorized; GPU-capable)
        ``scls`` sum-to-one constrained (vectorized; GPU-capable)
        ``fcls`` fully constrained: non-negative and sum-to-one (Heinz & Chang 2001),
                 per-pixel NNLS with a weighted sum-to-one row (CPU)
    delta : weight of the sum-to-one row for ``fcls``.

    Returns a Dataset of fractions plus ``rmse`` (reconstruction error).
    """
    ds = as_dataset(obj)
    bands = list(bands or [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)])
    names = list(endmembers)
    E = np.asarray([endmembers[n] for n in names], dtype="float64").T  # (bands, k)
    if E.shape[0] != len(bands):
        raise ValueError(f"Endmember spectra have {E.shape[0]} values but {len(bands)} bands given")
    template = ds[bands[0]]
    X = np.stack([np.asarray(to_cpu(ds[b]).values, dtype="float64") for b in bands], 0)
    shape = X.shape[1:]
    X = X.reshape(len(bands), -1)
    valid = np.all(np.isfinite(X), axis=0)
    F = np.full((len(names), X.shape[1]), np.nan)

    if method in ("ls", "scls"):
        xp = get_xp()
        Ex, Xv = xp.asarray(E), xp.asarray(X[:, valid])
        pinv = xp.linalg.pinv(Ex)
        f = pinv @ Xv
        if method == "scls":  # project onto sum-to-one: f + (1 - 1'f) * (E'E)^-1 1 / (1'(E'E)^-1 1)
            g = xp.linalg.inv(Ex.T @ Ex) @ xp.ones((len(names), 1))
            f = f + g @ ((1 - f.sum(0, keepdims=True)) / g.sum())
        F[:, valid] = to_cpu(f)
    elif method == "fcls":
        from scipy.optimize import nnls

        Ea = np.vstack([E, delta * np.ones((1, len(names)))])
        for j in np.flatnonzero(valid):
            F[:, j] = nnls(Ea, np.append(X[:, j], delta))[0]
    else:
        raise ValueError(f"Unknown method {method!r}")

    resid = np.full(X.shape[1], np.nan)
    resid[valid] = np.sqrt(np.mean((X[:, valid] - E @ F[:, valid]) ** 2, axis=0))
    out = xr.Dataset({n: template.copy(data=F[i].reshape(shape).astype("float32"))
                      for i, n in enumerate(names)})
    out["rmse"] = template.copy(data=resid.reshape(shape).astype("float32"))
    out.attrs = {"pyrsense:unmix": method}
    return out
