"""Temporal compositing: reduce an image time series to a single cloud-free image."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import xarray as xr

from .bands import as_dataset
from .indices import index as compute_index

Method = Literal["median", "mean", "min", "max", "percentile", "max_index", "median_nearest",
                 "bap"]


def composite(obj: xr.Dataset | xr.DataArray, method: Method = "median", *, dim: str = "time",
              q: float = 50, index: str = "NDVI", sensor: str | None = None,
              bands: Mapping[str, str] | None = None,
              keep_source: bool = True, target_doy: int | None = None,
              doy_sigma: float = 30.0, cloud_distance: float = 50.0,
              weights: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> xr.Dataset:
    """Composite a time series of (already cloud-masked) images.

    Methods
    -------
    median, mean, min, max, percentile
        Per-band, per-pixel statistics. Fast, but bands may come from different
        dates, so the spectra are synthetic.
    max_index
        Best-available-pixel: for each pixel take *all bands* from the date with the
        highest ``index`` (default NDVI). Keeps spectra physically consistent.
    median_nearest
        For each pixel take all bands from the date whose spectrum is closest
        (Euclidean) to the per-band median. Robust to residual cloud and shadow,
        and spectrally consistent. A fast approximation to a medoid composite.
    bap
        Best-Available-Pixel (White et al. 2014): each pixel's dates are scored by
        ``weights`` on (1) closeness to ``target_doy`` (Gaussian, ``doy_sigma`` days),
        (2) distance to the nearest cloud/no-data pixel (saturating at ``cloud_distance``
        pixels) and (3) the ``index`` value; all bands come from the best-scoring date.

    With ``keep_source=True`` the selection methods add a ``source_time`` variable
    recording which date each pixel came from.
    """
    ds = as_dataset(obj)
    if dim not in ds.dims:
        raise ValueError(f"Dimension {dim!r} not found; dims are {tuple(ds.dims)}")

    if method == "median":
        out = ds.median(dim, skipna=True)
    elif method == "mean":
        out = ds.mean(dim, skipna=True)
    elif method == "min":
        out = ds.min(dim, skipna=True)
    elif method == "max":
        out = ds.max(dim, skipna=True)
    elif method == "percentile":
        out = ds.quantile(q / 100.0, dim=dim, skipna=True).drop_vars("quantile")
    elif method == "max_index":
        score = compute_index(ds, index, sensor=sensor, bands=bands)
        out = _select(ds, score, dim, largest=True, keep_source=keep_source)
    elif method == "median_nearest":
        med = ds.median(dim, skipna=True)
        float_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)]
        dist = sum((ds[v] - med[v]) ** 2 for v in float_vars)
        # a date counts only if every band is present
        complete = xr.concat([ds[v].notnull() for v in float_vars], "v").all("v")
        out = _select(ds, dist.where(complete), dim, largest=False, keep_source=keep_source)
    elif method == "bap":
        out = _select(ds, _bap_score(ds, dim, index, sensor, bands, target_doy, doy_sigma,
                                     cloud_distance, weights), dim, largest=True,
                      keep_source=keep_source)
    else:
        raise ValueError(f"Unknown method {method!r}")
    out.attrs = {**ds.attrs, "pyrsense:composite": method}
    return out


def _select(ds: xr.Dataset, score: xr.DataArray, dim: str, *, largest: bool,
            keep_source: bool) -> xr.Dataset:
    valid = score.notnull().any(dim)
    fill = -np.inf if largest else np.inf
    filled = score.fillna(fill)
    idx = filled.argmax(dim) if largest else filled.argmin(dim)
    out = ds.isel({dim: idx}).where(valid)
    if keep_source and dim in ds.coords:
        out["source_" + dim] = ds[dim].isel({dim: idx}).where(valid)
    return out.drop_vars(dim, errors="ignore")


def _bap_score(ds, dim, index, sensor, bands, target_doy, doy_sigma, cloud_distance, weights):
    from scipy.ndimage import distance_transform_edt

    from ..backend import to_cpu

    vi = to_cpu(compute_index(ds, index, sensor=sensor, bands=bands))
    valid = vi.notnull()
    doy = ds[dim].dt.dayofyear
    if target_doy is None:
        target_doy = int(np.median(doy.values))
    d = np.abs(doy - target_doy)
    d = np.minimum(d, 365 - d)
    s_doy = np.exp(-0.5 * (d / doy_sigma) ** 2)
    ydim, xdim = vi.dims[-2:]
    dist = xr.apply_ufunc(
        lambda m: distance_transform_edt(m), valid, input_core_dims=[[ydim, xdim]],
        output_core_dims=[[ydim, xdim]], vectorize=True)
    s_cloud = np.minimum(dist / cloud_distance, 1.0)
    vmin, vmax = vi.min(), vi.max()
    s_vi = (vi - vmin) / (vmax - vmin)
    w1, w2, w3 = weights
    return (w1 * s_doy + w2 * s_cloud + w3 * s_vi).where(valid)
