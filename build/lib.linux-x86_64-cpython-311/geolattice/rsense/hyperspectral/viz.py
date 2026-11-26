"""Hyperspectral plots: composites by wavelength, spectra, libraries, band statistics,
interactive spectral profile viewer and wavelength slider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from .cube import band_at, spectra_at, wavelengths

COMPOSITES = {
    "true_color": (640, 550, 460),
    "color_infrared": (860, 650, 550),
    "swir": (2200, 860, 650),
    "geology": (2200, 1650, 860),
    "vegetation_stress": (1650, 860, 650),
    "water": (650, 550, 460),
}


def rgb(cube: xr.DataArray, wavelengths_nm: Sequence[float] | str = "true_color", *,
        percentiles: tuple[float, float] = (2, 98), gamma: float = 1.0) -> xr.DataArray:
    """RGB composite (0–1, dims ``rgb, y, x``) from three wavelengths or a preset name
    (``true_color``, ``color_infrared``, ``swir``, ``geology``, ``vegetation_stress``,
    ``water``), with a percentile stretch per channel."""
    wls = COMPOSITES[wavelengths_nm] if isinstance(wavelengths_nm, str) else wavelengths_nm
    chans = []
    for w in wls:
        b = band_at(cube, w, tolerance=None).astype("float64")
        lo, hi = np.nanpercentile(b.values, percentiles)
        chans.append(((b - lo) / max(hi - lo, 1e-12)).clip(0, 1) ** (1 / gamma))
    return xr.concat(chans, "rgb").assign_coords(rgb=["r", "g", "b"]).rename("rgb")


def plot_rgb(cube: xr.DataArray, wavelengths_nm: Sequence[float] | str = "true_color", *,
             ax=None, title: str | None = None, **kw):
    import matplotlib.pyplot as plt

    im = rgb(cube, wavelengths_nm, **kw)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.moveaxis(np.nan_to_num(im.values), 0, -1))
    ax.set_title(title or (wavelengths_nm if isinstance(wavelengths_nm, str)
                           else f"R{wavelengths_nm[0]} G{wavelengths_nm[1]} B{wavelengths_nm[2]}"))
    ax.axis("off")
    return ax


def plot_spectra(data: Any, *, wavelengths_nm: Sequence[float] | None = None, labels=None,
                 ax=None, bad_ranges: Sequence[tuple[float, float]] | None = None,
                 xlabel: str = "Wavelength (nm)", ylabel: str = "Reflectance", **kw):
    """Plot spectra: a DataFrame (rows = spectra, columns = wavelengths, optional ``label``
    column), a Series, a SpectralLibrary or an array with ``wavelengths_nm``. Shaded
    ``bad_ranges`` mark atmospheric absorption windows."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from .library import SpectralLibrary

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    if isinstance(data, SpectralLibrary):
        return data.plot(ax=ax, **kw)
    if isinstance(data, pd.Series):
        data = data.to_frame().T
    if isinstance(data, pd.DataFrame):
        lab = data.pop("label") if "label" in data else None
        wl = data.columns.to_numpy(float)
        Y = data.to_numpy(float)
        labels = labels if labels is not None else (list(lab) if lab is not None
                                                    else list(data.index))
    else:
        Y = np.atleast_2d(np.asarray(data, float))
        wl = np.asarray(wavelengths_nm, float)
    for i, y in enumerate(Y):
        ax.plot(wl, y, label=None if labels is None else str(labels[i]), **kw)
    for lo, hi in bad_ranges or ():
        ax.axvspan(lo, hi, color="0.9", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels is not None and len(Y) <= 15:
        ax.legend(fontsize=8)
    return ax


def plot_band_stats(stats, *, ax=None):
    """Mean ± std and SNR per band from :func:`~.cube.band_stats`."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(stats.wavelength, stats["mean"], label="mean")
    ax.fill_between(stats.wavelength, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                    alpha=0.3, label="± std")
    ax2 = ax.twinx()
    ax2.plot(stats.wavelength, stats.snr, color="tab:red", lw=0.8, label="SNR")
    ax2.set_ylabel("SNR")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend(loc="upper left")
    return ax


def spectral_viewer(cube: xr.DataArray, points=None, *,
                    composite: Sequence[float] | str = "true_color", max_points: int = 20):
    """Plotly figure: RGB composite (left) and spectra of the clicked/selected points (right).
    ``points`` are (x, y) map coordinates or a GeoDataFrame; returns the figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    im = rgb(cube, composite)
    fig = make_subplots(1, 2, column_widths=[0.45, 0.55],
                        subplot_titles=("Composite", "Spectra"))
    fig.add_trace(go.Image(z=(np.moveaxis(np.nan_to_num(im.values), 0, -1) * 255
                              ).astype("uint8")), 1, 1)
    if points is not None:
        df = spectra_at(cube, points)
        wl = wavelengths(cube)
        for i, row in df.head(max_points).iterrows():
            fig.add_trace(go.Scatter(x=wl, y=row.to_numpy(float), mode="lines",
                                     name=f"point {i}"), 1, 2)
    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
    fig.update_yaxes(title_text="Reflectance", row=1, col=2)
    fig.update_layout(height=450, width=1000)
    return fig


def wavelength_slider(cube: xr.DataArray, *, step: int = 5, cmap: str = "viridis"):
    """Plotly figure with a slider through the bands (every ``step``-th band)."""
    import plotly.graph_objects as go

    wl = wavelengths(cube)
    idx = np.arange(0, len(wl), step)
    frames = [go.Frame(data=[go.Heatmap(z=np.asarray(cube.isel(band=int(i)).values)[::-1],
                                        colorscale=cmap)], name=f"{wl[i]:.0f}") for i in idx]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(sliders=[{"steps": [{"args": [[f.name], {"mode": "immediate"}],
                                           "label": f.name, "method": "animate"}
                                          for f in frames],
                                "currentvalue": {"prefix": "λ = ", "suffix": " nm"}}],
                      height=550, width=600)
    return fig
