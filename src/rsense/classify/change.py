"""Optical change detection: image differencing, change vector analysis, post-classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu
from ..optical.bands import as_dataset


def image_difference(before: xr.DataArray, after: xr.DataArray, *, k: float = 2.0) -> xr.Dataset:
    """Difference image with changes flagged beyond ``k`` standard deviations of the difference."""
    d = after - before
    a = np.asarray(to_cpu(d).values, dtype="float64")
    mu, sd = np.nanmean(a), np.nanstd(a)
    return xr.Dataset({"difference": d, "increase": d > mu + k * sd, "decrease": d < mu - k * sd},
                      attrs={"mean": mu, "std": sd, "k": k})


def cva(before: xr.Dataset | xr.DataArray, after: xr.Dataset | xr.DataArray, *,
        bands: Sequence[str] | None = None, threshold: float | None = None) -> xr.Dataset:
    """Change Vector Analysis: magnitude and direction of spectral change.

    With two bands, ``direction`` is the angle (degrees, 0-360) of the change vector in
    the band-1/band-2 plane (e.g. brightness vs greenness). The change mask uses
    ``threshold`` or Otsu on the magnitude.
    """
    b, a = as_dataset(before), as_dataset(after)
    bands = list(bands or b.data_vars)
    diffs = [a[v] - b[v] for v in bands]
    mag = np.sqrt(sum(d**2 for d in diffs))
    out = xr.Dataset({"magnitude": mag})
    if len(bands) >= 2:
        out["direction"] = (np.degrees(np.arctan2(diffs[1], diffs[0])) % 360)
    if threshold is None:
        from skimage.filters import threshold_otsu

        m = np.asarray(to_cpu(mag).values)
        threshold = float(threshold_otsu(m[np.isfinite(m)]))
    out["change"] = mag > threshold
    out.attrs["threshold"] = threshold
    return out


def post_classification(map1: xr.DataArray, map2: xr.DataArray, *,
                        labels: dict | None = None, pixel_area: float = 1.0
                        ) -> tuple[xr.DataArray, pd.DataFrame]:
    """From-to change analysis of two classified maps.

    Returns a transition code raster (``from * 1000 + to``) and the transition matrix
    (rows = from, columns = to) in area units.
    """
    a = np.asarray(to_cpu(map1).values).astype("int64")
    b = np.asarray(to_cpu(map2).values).astype("int64")
    valid = (a >= 0) & (b >= 0)
    codes = np.where(valid, a * 1000 + b, -1)
    m = pd.crosstab(pd.Series(a[valid], name="from"), pd.Series(b[valid], name="to")) * pixel_area
    if labels:
        m = m.rename(index=labels, columns=labels)
    return map1.copy(data=codes).rename("transition"), m
