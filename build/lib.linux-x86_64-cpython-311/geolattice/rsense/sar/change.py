"""SAR change detection: log-ratio, likelihood-ratio (omnibus) tests and CFAR detection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from ..backend import ndimage, to_cpu
from ..core._utils import as_float, pix_to_map, run_array


def log_ratio(before: Any, after: Any, *, db: bool = True):
    """Log-ratio change image ``10 log10(after / before)`` (dB) or natural log if ``db=False``.

    Negative values = backscatter decrease (e.g. flooding), positive = increase.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = after / before
    out = (10 * np.log10(r)) if db else np.log(r)
    if isinstance(out, xr.DataArray):
        out = out.rename("log_ratio")
    return out


def omnibus_test(stack: xr.DataArray, *, enl: float = 4.4, dim: str = "time",
                 alpha: float = 0.01) -> xr.Dataset:
    """Likelihood-ratio test for change across k single-polarization intensity images
    (Conradsen et al. 2016, p=1 case).

    ``-2 rho ln Q`` is approximately chi-squared with ``k-1`` dof, where
    ``Q = k^(k m) prod(X_i)^m / (sum X_i)^(k m)`` and ``m`` = ENL. Returns the
    test statistic, p-value and a boolean change map at significance ``alpha``.
    """
    from scipy.stats import chi2

    k = stack.sizes[dim]
    if k < 2:
        raise ValueError("Need at least two images")
    m = float(enl)
    x = stack.where(stack > 0)
    lnq = m * (k * np.log(k) + np.log(x).sum(dim) - k * np.log(x.sum(dim)))
    rho = 1 - (1 / (6 * (k - 1))) * (k / m - 1 / (k * m))
    stat = -2 * rho * lnq
    p = xr.apply_ufunc(lambda s: chi2.sf(s, k - 1), stat, dask="parallelized",
                       output_dtypes=[float])
    return xr.Dataset({"statistic": stat, "p_value": p, "change": p < alpha},
                      attrs={"test": "omnibus (p=1)", "enl": m, "alpha": alpha})


def pairwise_change(stack: xr.DataArray, *, enl: float = 4.4, dim: str = "time",
                    alpha: float = 0.01) -> xr.Dataset:
    """Sequential two-date LR tests between consecutive images: change map per interval,
    plus first change date and count of changes."""
    tests = [omnibus_test(stack.isel({dim: [i, i + 1]}), enl=enl, dim=dim, alpha=alpha)["change"]
             for i in range(stack.sizes[dim] - 1)]
    ch = xr.concat(tests, dim=dim).assign_coords({dim: stack[dim].values[1:]})
    first = ch.argmax(dim)
    first_date = stack[dim].values[1:][to_cpu(first).values]
    any_ch = ch.any(dim)
    return xr.Dataset({"change": ch, "n_changes": ch.sum(dim),
                       "first_change": first.copy(data=np.where(any_ch, first_date,
                                                                np.datetime64("NaT")))})


def cfar(img: Any, *, guard: int = 3, background: int = 12, pfa: float = 1e-6,
         method: Literal["ca", "os"] = "ca", os_rank: float = 0.75):
    """Constant False Alarm Rate detector on linear intensity (bright targets, e.g. ships).

    ``ca``: cell-averaging with an exponential clutter model, threshold factor
    ``T = N (pfa^(-1/N) - 1)`` over a square ring (``guard`` and ``background`` half-widths).
    ``os``: ordered-statistic, using the ``os_rank`` quantile of the ring (robust in
    clutter edges; CPU percentile filter). Returns a boolean detection mask.
    """
    outer = 2 * (guard + background) + 1
    inner = 2 * guard + 1
    n_ring = outer**2 - inner**2

    def f(a, xp):
        a = as_float(a, xp)
        nd = ndimage(a)
        z = xp.nan_to_num(a, nan=0.0)
        if method == "ca":
            s_out = nd.uniform_filter(z, outer, mode="reflect") * outer**2
            s_in = nd.uniform_filter(z, inner, mode="reflect") * inner**2
            clutter = (s_out - s_in) / n_ring
            T = n_ring * (pfa ** (-1.0 / n_ring) - 1)
        else:
            fp = np.ones((outer, outer), bool)
            fp[background:background + inner, background:background + inner] = False
            clutter = nd.percentile_filter(z, os_rank * 100, footprint=xp.asarray(fp)
                                           if xp is not np else fp, mode="reflect")
            T = -np.log(pfa)  # exponential tail at the reference level (approximate)
        return (z > T * clutter) & ~xp.isnan(a)

    return run_array(img, f)


def detections_to_points(mask: xr.DataArray, intensity: xr.DataArray | None = None,
                         min_pixels: int = 2):
    """Group a detection mask into objects → GeoDataFrame of centroids with size and peak."""
    import geopandas as gpd
    from scipy import ndimage as snd

    m = np.asarray(to_cpu(mask).values, bool)
    lab, n = snd.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return gpd.GeoDataFrame({"pixels": [], "peak": []}, geometry=[], crs=mask.rio.crs)
    idx = np.arange(1, n + 1)
    sizes = snd.sum(m, lab, idx)
    cy, cx = np.array(snd.center_of_mass(m, lab, idx)).T
    peak = (snd.maximum(np.asarray(to_cpu(intensity).values), lab, idx)
            if intensity is not None else np.full(n, np.nan))
    keep = sizes >= min_pixels
    T = mask.rio.transform()
    xs, ys = pix_to_map(T, cx[keep] + 0.5, cy[keep] + 0.5)
    return gpd.GeoDataFrame({"pixels": sizes[keep].astype(int), "peak": peak[keep]},
                            geometry=gpd.points_from_xy(xs, ys), crs=mask.rio.crs)


def threshold(img: Any, method: Literal["otsu", "kmeans"] = "otsu") -> float:
    """Global threshold of a change or backscatter image."""
    a = np.asarray(to_cpu(getattr(img, "values", img)), dtype="float64")
    a = a[np.isfinite(a)]
    if method == "otsu":
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(a))
    from sklearn.cluster import KMeans

    km = KMeans(2, n_init=5, random_state=0).fit(a.reshape(-1, 1)[:: max(1, a.size // 200_000)])
    return float(np.mean(km.cluster_centers_))
