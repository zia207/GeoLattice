"""Contrast stretches and RGB composites (pure NumPy; no plotting dependency)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..optical.bands import as_dataset, get_band


def stretch(a: np.ndarray, method: Literal["percentile", "minmax", "std", "equalize"] = "percentile",
            *, p: tuple[float, float] = (2, 98), n_std: float = 2.0, gamma: float = 1.0) -> np.ndarray:
    """Scale an array to 0-1 for display (NaN preserved)."""
    a = np.asarray(to_cpu(a), dtype="float64")
    v = a[np.isfinite(a)]
    if v.size == 0:
        return a
    if method == "percentile":
        lo, hi = np.percentile(v, p)
    elif method == "minmax":
        lo, hi = v.min(), v.max()
    elif method == "std":
        lo, hi = v.mean() - n_std * v.std(), v.mean() + n_std * v.std()
    elif method == "equalize":
        hist, edges = np.histogram(v, bins=512)
        cdf = hist.cumsum() / hist.sum()
        out = np.interp(a, edges[:-1], cdf)
        return np.where(np.isfinite(a), out ** (1 / gamma), np.nan)
    else:
        raise ValueError(f"Unknown method {method!r}")
    out = np.clip((a - lo) / (hi - lo if hi > lo else 1), 0, 1)
    return np.where(np.isfinite(a), out ** (1 / gamma), np.nan)


COMPOSITES = {
    "true_color": ("red", "green", "blue"),
    "false_color": ("nir", "red", "green"),          # vegetation in red
    "swir": ("swir22", "nir", "red"),                 # burn scars, geology
    "agriculture": ("swir16", "nir", "blue"),
    "urban": ("swir22", "swir16", "red"),
    "water": ("nir", "swir16", "red"),
}


def rgb(obj: xr.Dataset | xr.DataArray, bands: Sequence[str] | str = "true_color", *,
        method: str = "percentile", p: tuple[float, float] = (2, 98), gamma: float = 1.0,
        per_band: bool = True, sensor: str | None = None, as_uint8: bool = False) -> xr.DataArray:
    """Build a display-ready RGB composite DataArray (y, x, rgb), each band stretched 0-1.

    ``bands`` is a preset name (see ``COMPOSITES``) or three canonical/actual band names.
    ``per_band=False`` applies one stretch to all three bands (preserves colour balance).
    """
    ds = as_dataset(obj)
    names = COMPOSITES[bands] if isinstance(bands, str) else tuple(bands)
    layers = [get_band(ds, b, sensor=sensor) for b in names]
    arrs = [np.asarray(to_cpu(lyr).values, "float64") for lyr in layers]
    if per_band:
        out = np.stack([stretch(a, method, p=p, gamma=gamma) for a in arrs], -1)
    else:
        allv = np.stack(arrs, -1)
        out = stretch(allv, method, p=p, gamma=gamma)
    tpl = layers[0]
    ydim, xdim = tpl.dims[-2:]
    if as_uint8:
        out = (np.nan_to_num(out) * 255).astype("uint8")
    da = xr.DataArray(out, dims=(ydim, xdim, "rgb"),
                      coords={ydim: tpl[ydim], xdim: tpl[xdim], "rgb": list(names)}, name="rgb")
    return da.rio.write_crs(ds.rio.crs) if ds.rio.crs else da
