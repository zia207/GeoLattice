"""Static plots (matplotlib), interactive maps (leafmap/folium) and 3-D point clouds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from ..backend import to_cpu
from .stretch import rgb as make_rgb


def _extent(da: xr.DataArray):
    ydim, xdim = da.dims[:2] if da.ndim == 3 and da.dims[-1] == "rgb" else da.dims[-2:]
    x, y = da[xdim].values, da[ydim].values
    dx = abs(x[1] - x[0]) if x.size > 1 else 1
    dy = abs(y[1] - y[0]) if y.size > 1 else 1
    return [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]


def plot_rgb(obj, bands: Sequence[str] | str = "true_color", *, ax=None, title: str | None = None,
             **rgb_kw: Any):
    """Show an RGB composite with map coordinates."""
    import matplotlib.pyplot as plt

    img = obj if isinstance(obj, xr.DataArray) and obj.dims[-1:] == ("rgb",) else \
        make_rgb(obj, bands, **rgb_kw)
    ax = ax or plt.subplots(figsize=(8, 8))[1]
    ax.imshow(np.nan_to_num(img.values), extent=_extent(img), origin="upper")
    ax.set_title(title or (bands if isinstance(bands, str) else "/".join(bands)))
    ax.set_aspect("equal")
    return ax


def plot_raster(da: xr.DataArray, *, ax=None, cmap: str = "viridis", vmin=None, vmax=None,
                title: str | None = None, colorbar: bool = True, categorical: dict | None = None,
                robust: bool = True):
    """Plot a single-band raster (continuous or categorical with ``{value: (label, color)}``)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    a = np.asarray(to_cpu(da).values, dtype="float64")
    ax = ax or plt.subplots(figsize=(8, 7))[1]
    if categorical:
        vals = sorted(categorical)
        cm = ListedColormap([categorical[v][1] for v in vals])
        norm = BoundaryNorm(np.r_[np.array(vals) - 0.5, vals[-1] + 0.5], cm.N)
        im = ax.imshow(a, cmap=cm, norm=norm, extent=_extent(da), interpolation="nearest")
        if colorbar:
            cb = plt.colorbar(im, ax=ax, ticks=vals, shrink=0.8)
            cb.ax.set_yticklabels([categorical[v][0] for v in vals])
    else:
        if robust and vmin is None and vmax is None and np.isfinite(a).any():
            vmin, vmax = np.nanpercentile(a, [2, 98])
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(da))
        if colorbar:
            plt.colorbar(im, ax=ax, shrink=0.8, label=da.attrs.get("units", ""))
    ax.set_title(title or (da.name or ""))
    return ax


def plot_timeseries(da: xr.DataArray, points: dict[str, tuple[float, float]] | None = None, *,
                    ax=None, dim: str = "time", agg: str = "mean"):
    """Plot pixel time series at named (x, y) points, or the spatial ``agg`` if none given."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(10, 4))[1]
    ydim, xdim = [d for d in da.dims if d != dim][-2:]
    if points:
        for name, (x, y) in points.items():
            s = da.sel({xdim: x, ydim: y}, method="nearest")
            ax.plot(s[dim].values, to_cpu(s).values, marker="o", ms=3, label=name)
        ax.legend()
    else:
        s = getattr(da, agg)([ydim, xdim])
        ax.plot(s[dim].values, to_cpu(s).values, marker="o", ms=3)
    ax.set_ylabel(da.name or "")
    ax.grid(alpha=0.3)
    return ax


def interactive_map(layers: dict[str, xr.DataArray] | None = None, *, vectors: dict | None = None,
                    center: tuple[float, float] | None = None, zoom: int = 10, basemap: str = "HYBRID",
                    split: tuple[str, str] | None = None, **kwargs: Any):
    """Interactive web map with raster and vector layers (leafmap). ``split=(left, right)``
    shows two raster layers with a swipe control."""
    try:
        import leafmap
    except ImportError as e:
        raise ImportError("interactive_map needs `pip install leafmap localtileserver`") from e
    import os
    import tempfile

    from ..core.raster import write

    m = leafmap.Map(center=center, zoom=zoom, **kwargs)
    m.add_basemap(basemap)
    paths = {}
    for name, da in (layers or {}).items():
        p = os.path.join(tempfile.mkdtemp(), f"{name}.tif")
        write(da.rio.reproject(4326) if da.rio.crs and da.rio.crs.to_epsg() != 4326 else da, p)
        paths[name] = p
    if split:
        m.split_map(paths[split[0]], paths[split[1]], left_label=split[0], right_label=split[1])
    else:
        for name, p in paths.items():
            m.add_raster(p, layer_name=name)
    for name, gdf in (vectors or {}).items():
        m.add_gdf(gdf.to_crs(4326), layer_name=name)
    return m


def plot_pointcloud(pc, *, color_by: str = "z", max_points: int = 200_000, backend: str = "auto",
                    point_size: float = 2.0, cmap: str = "viridis"):
    """3-D view of a PointCloud with PyVista (interactive) or matplotlib (fallback, subsampled)."""
    p = pc.to_cpu()
    n = len(p)
    idx = np.random.default_rng(0).choice(n, min(n, max_points), replace=False) if n > max_points \
        else np.arange(n)
    xyz = np.column_stack([p.x[idx], p.y[idx], p.z[idx]])
    c = np.asarray(p[color_by])[idx]
    if backend in ("auto", "pyvista"):
        try:
            import pyvista as pv

            cloud = pv.PolyData(xyz - xyz.mean(0))
            cloud[color_by] = c
            pl = pv.Plotter()
            pl.add_points(cloud, scalars=color_by, point_size=point_size, cmap=cmap,
                          render_points_as_spheres=True)
            return pl
        except ImportError:
            if backend == "pyvista":
                raise
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(*(xyz - xyz.mean(0)).T, c=c, s=point_size / 4, cmap=cmap)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=color_by)
    return ax
