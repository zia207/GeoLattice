"""Hyperspectral cube conventions and helpers.

A cube is an ``xarray.DataArray`` with dims ``("band", "y", "x")`` and band coordinates
``wavelength`` (nm) and, when known, ``fwhm`` (nm) and ``good`` (bool). Swath products that
are not yet orthorectified use ``("band", "line", "sample")`` in the same way. Every function
in :mod:`geolattice.rsense.hyperspectral` accepts and returns this layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu


def as_cube(data: Any, wavelengths: Sequence[float] | None = None, *,
            fwhm: Sequence[float] | None = None, band_dim: str | None = None,
            units: str = "nm", crs: Any = None, name: str = "reflectance") -> xr.DataArray:
    """Standardise an array/DataArray/Dataset into a cube with a ``wavelength`` coordinate.

    ``units="um"`` converts micrometres to nm. A Dataset (one variable per band) is stacked
    along ``band``. A NumPy array must be ``(bands, rows, cols)``.
    """
    if isinstance(data, xr.Dataset):
        data = data.to_array("band")
    if not isinstance(data, xr.DataArray):
        a = np.asarray(data)
        data = xr.DataArray(a, dims=("band", "y", "x")[-a.ndim:] if a.ndim <= 3 else None)
    bd = band_dim or next((d for d in data.dims if d in ("band", "wavelength", "bands")),
                          data.dims[0])
    if bd != "band":
        data = data.rename({bd: "band"})
    if data.dims[0] != "band":
        data = data.transpose("band", ...)
    if wavelengths is None:
        if "wavelength" in data.coords:
            wavelengths = data["wavelength"].values
        elif np.issubdtype(np.asarray(data["band"].values).dtype, np.number) and \
                float(np.nanmax(data["band"].values)) > 300:
            wavelengths = data["band"].values
        else:
            raise ValueError("Pass wavelengths= (band centres)")
    wl = np.asarray(wavelengths, dtype="float64")
    if units in ("um", "micrometers", "micron", "µm"):
        wl = wl * 1000.0
    if len(wl) != data.sizes["band"]:
        raise ValueError(f"{len(wl)} wavelengths for {data.sizes['band']} bands")
    out = data.assign_coords(band=np.arange(data.sizes["band"]), wavelength=("band", wl))
    if fwhm is not None:
        f = np.asarray(fwhm, dtype="float64") * (1000.0 if units in ("um", "µm") else 1.0)
        out = out.assign_coords(fwhm=("band", f))
    if crs is not None:
        out = out.rio.write_crs(crs)
    return out.rename(name) if out.name is None else out


def wavelengths(cube: xr.DataArray) -> np.ndarray:
    """Band centres in nm."""
    if "wavelength" not in cube.coords:
        raise KeyError("Cube has no 'wavelength' coordinate; use as_cube(data, wavelengths)")
    return np.asarray(cube["wavelength"].values, dtype="float64")


def band_index(cube: xr.DataArray, nm: float, *, tolerance: float | None = None) -> int:
    """Index of the band closest to ``nm``; raises if it is farther than ``tolerance``."""
    wl = wavelengths(cube)
    if "good" in cube.coords:
        wl = np.where(np.asarray(cube["good"].values, bool), wl, np.inf)
    i = int(np.argmin(np.abs(wl - nm)))
    if tolerance is not None and abs(wl[i] - nm) > tolerance:
        raise ValueError(f"No band within {tolerance} nm of {nm} nm (closest {wl[i]:.1f})")
    return i


def band_at(cube: xr.DataArray, nm: float, *, tolerance: float | None = 20.0,
            width: float | None = None) -> xr.DataArray:
    """The band nearest ``nm``, or the mean over ``nm ± width/2`` when ``width`` is given."""
    if width:
        wl = wavelengths(cube)
        m = np.abs(wl - nm) <= width / 2
        if not m.any():
            return band_at(cube, nm, tolerance=tolerance)
        return cube.isel(band=np.flatnonzero(m)).mean("band").rename(f"R{nm:g}")
    return cube.isel(band=band_index(cube, nm, tolerance=tolerance)).drop_vars(
        [c for c in ("wavelength", "fwhm", "good", "band") if c in cube.coords]).rename(f"R{nm:g}")


def sel_range(cube: xr.DataArray, start: float, stop: float) -> xr.DataArray:
    """Bands with ``start <= wavelength <= stop`` nm."""
    wl = wavelengths(cube)
    return cube.isel(band=np.flatnonzero((wl >= start) & (wl <= stop)))


def to_matrix(cube: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """``(pixels, bands)`` float64 matrix of valid pixels and the flat valid mask."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    X = a.reshape(a.shape[0], -1).T
    valid = np.all(np.isfinite(X), axis=1)
    return X[valid], valid


def from_matrix(values: np.ndarray, cube: xr.DataArray, valid: np.ndarray, *,
                names: Sequence[str] | None = None, dim: str = "component",
                fill: float = np.nan) -> xr.DataArray:
    """Inverse of :func:`to_matrix`: put ``(valid_pixels, k)`` values back on the cube grid."""
    sdims = cube.dims[1:]
    shape = tuple(cube.sizes[d] for d in sdims)
    v = np.asarray(values)
    if v.ndim == 1:
        v = v[:, None]
    out = np.full((v.shape[1], valid.size), fill, dtype="float32" if v.dtype.kind == "f"
                  else v.dtype)
    out[:, valid] = v.T
    coords = {d: cube[d] for d in sdims if d in cube.coords}
    for c in ("spatial_ref",):
        if c in cube.coords:
            coords[c] = cube[c]
    da = xr.DataArray(out.reshape(v.shape[1], *shape), dims=(dim, *sdims), coords=coords)
    if names is not None:
        da = da.assign_coords({dim: list(names)})
    if "spatial_ref" in cube.coords:
        da = da.rio.write_crs(cube.rio.crs)
    return da


def spectra_at(cube: xr.DataArray, points, *, label_col: str | None = None,
               buffer: int = 0) -> pd.DataFrame:
    """Spectra at points (GeoDataFrame or list of (x, y)); rows = points, columns = wavelengths.

    ``buffer`` averages a ``(2b+1)²`` pixel window. ``label_col`` is copied into ``label``.
    """
    from ..core._utils import map_to_pix

    if hasattr(points, "geometry"):
        g = points.to_crs(cube.rio.crs) if (points.crs is not None and cube.rio.crs is not None) \
            else points
        xs, ys = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
    else:
        xs, ys = np.asarray(points, float).T
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    try:
        cols, rows = map_to_pix(cube.rio.transform(), xs, ys)
    except Exception:  # noqa: BLE001  (no georeference: points are (col, row))
        cols, rows = xs, ys
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    out = np.full((len(xs), a.shape[0]), np.nan)
    for i, (rr, cc) in enumerate(zip(r, c)):
        win = a[:, max(rr - buffer, 0):rr + buffer + 1, max(cc - buffer, 0):cc + buffer + 1]
        if win.size:
            out[i] = np.nanmean(win.reshape(a.shape[0], -1), axis=1)
    df = pd.DataFrame(out, columns=np.round(wavelengths(cube), 2))
    if label_col is not None:
        df.insert(0, "label", np.asarray(points[label_col]))
    return df


def mean_spectrum(cube: xr.DataArray, mask: Any = None) -> pd.Series:
    """Mean spectrum (optionally over a boolean ``mask`` of the spatial shape)."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    X = a.reshape(a.shape[0], -1)
    if mask is not None:
        X = X[:, np.asarray(getattr(mask, "values", mask), bool).ravel()]
    return pd.Series(np.nanmean(X, axis=1), index=np.round(wavelengths(cube), 2),
                     name="mean")


def band_stats(cube: xr.DataArray) -> pd.DataFrame:
    """Per-band statistics: wavelength, mean, std, min, max, NaN fraction and a simple SNR
    (mean / std of horizontal pixel differences / √2)."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    wl = wavelengths(cube)
    flat = a.reshape(a.shape[0], -1)
    d = np.diff(a, axis=-1).reshape(a.shape[0], -1) / np.sqrt(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        noise = np.nanstd(d, axis=1)
        mean = np.nanmean(flat, axis=1)
        return pd.DataFrame({"wavelength": wl, "mean": mean, "std": np.nanstd(flat, axis=1),
                             "min": np.nanmin(flat, axis=1), "max": np.nanmax(flat, axis=1),
                             "nan_fraction": np.isnan(flat).mean(1),
                             "snr": np.where(noise > 0, mean / noise, np.nan)})


def apply_spectral(cube: xr.DataArray, func, *, wavelengths_out: Sequence[float] | None = None,
                   chunk: int = 200_000, name: str | None = None) -> xr.DataArray:
    """Apply ``func((pixels, bands) array) -> (pixels, bands_out)`` to every valid pixel,
    in chunks, and rebuild a cube (with ``wavelengths_out`` if the band count changes)."""
    a = np.asarray(to_cpu(cube).values, dtype="float64")
    B = a.shape[0]
    X = a.reshape(B, -1).T
    valid = np.all(np.isfinite(X), axis=1)
    idx = np.flatnonzero(valid)
    first = func(X[idx[:1]]) if idx.size else func(np.zeros((1, B)))
    Bo = np.asarray(first).shape[1]
    out = np.full((X.shape[0], Bo), np.nan, dtype="float32")
    for s in range(0, idx.size, chunk):
        j = idx[s:s + chunk]
        out[j] = func(X[j])
    res = out.T.reshape(Bo, *a.shape[1:])
    if Bo == B and wavelengths_out is None:
        return cube.copy(data=res).rename(name or cube.name)
    wl = np.asarray(wavelengths_out if wavelengths_out is not None else np.arange(Bo), float)
    sdims = cube.dims[1:]
    da = xr.DataArray(res, dims=("band", *sdims),
                      coords={**{d: cube[d] for d in sdims if d in cube.coords},
                              "band": np.arange(Bo), "wavelength": ("band", wl)},
                      name=name or cube.name)
    return da.rio.write_crs(cube.rio.crs) if cube.rio.crs is not None else da
