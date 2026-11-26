"""Raster and gridded-data maps.

NumPy arrays, xarray DataArrays / Datasets, and (optionally) a
``geolattice.raster.RasterDataset`` are accepted. A rioxarray raster is an
xarray DataArray with a ``rio`` accessor, so the same functions draw it.

The API follows three common raster layer types used by GeoHub
(https://docs.undpgeohub.org/visualization/visualize_raster/):

* **general** — continuous / linear values with a colormap and optional
  min/max rescale
* **unique** — integer class codes with a categorical legend
* **rgb** — true-color (or false-color) composites from 3+ bands

Selection and plotting patterns also follow the Spatial Thoughts xarray
course, Python GIS static raster maps, and ggplot-style ``geom_raster``
faceting with optional vector overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from geolattice.viz._imports import VisualizationError, require
from geolattice.viz.style import CATEGORICAL, SEQUENTIAL, style_axes

RasterKind = str  # "general" | "unique" | "rgb"


def plot_raster(
    data: Any,
    *,
    kind: Optional[RasterKind] = None,
    variable: Optional[str] = None,
    band: Any = None,
    time: Any = None,
    rgb: bool = False,
    bands: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    categories: Optional[Mapping[Any, str]] = None,
    robust: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rescale: Optional[Tuple[float, float]] = None,
    interpolation: str = "nearest",
    extent: Optional[Sequence[float]] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    stretch: str = "percentile",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    vector: Any = None,
    vector_kwds: Optional[Mapping[str, Any]] = None,
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    add_colorbar: bool = True,
    axis_off: bool = False,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw a raster as a general, unique-value, or RGB layer.

    Parameters
    ----------
    data:
        A 2-D NumPy array, a bands-first 3-D array, an xarray DataArray /
        Dataset, or a ``RasterDataset``.
    kind:
        ``"general"``, ``"unique"``, or ``"rgb"``. When omitted, the kind is
        inferred (RGB when ``rgb=True`` or three+ continuous bands; unique
        when values look like a small set of integers; otherwise general).
    band / time / variable:
        Select a slice the way the Spatial Thoughts xarray plotting pattern
        does.
    bands:
        One-based band indices for an RGB / false-color composite when
        ``kind="rgb"`` (default ``(1, 2, 3)``).
    cmap:
        Matplotlib colormap. Defaults to ``viridis`` for general rasters and
        a categorical map for unique-value rasters.
    categories:
        Mapping of class code → label for unique-value legends.
    robust:
        Stretch the color scale to the 2nd–98th percentiles (general rasters).
    vmin, vmax / rescale:
        Explicit color limits. ``rescale=(lo, hi)`` is the GeoHub-style alias.
    interpolation:
        ``"nearest"`` (sharp / default) or ``"bilinear"`` / ``"bicubic"``.
    extent:
        ``(left, right, bottom, top)`` in map coordinates. Taken from the
        dataset geotransform when available.
    brightness, contrast:
        Optional image adjustments for RGB (GeoHub-style controls).
    vector:
        Optional GeoDataFrame drawn on top of the raster.
    """
    plt = require("matplotlib.pyplot")

    if rgb:
        kind = "rgb"
    if rescale is not None:
        vmin, vmax = float(rescale[0]), float(rescale[1])

    prepared = _coerce_input(data)
    if kind is None:
        kind = infer_raster_kind(prepared, band=band, rgb=rgb)

    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)

    if kind == "rgb":
        image = _rgb_array(
            prepared,
            variable=variable,
            time=time,
            bands=bands or (1, 2, 3),
            stretch=stretch,
            percentiles=percentiles,
            brightness=brightness,
            contrast=contrast,
        )
        ax.imshow(
            image,
            extent=extent or _extent_from_data(prepared),
            origin="upper",
            interpolation=interpolation,
            **plot_kwds,
        )
        add_colorbar = False
    elif kind == "unique":
        array = _prepare_array(
            prepared, variable=variable, band=band, time=time, rgb=False
        )
        np = require("numpy")
        grid = np.asarray(_values(array))
        _draw_unique(
            ax,
            grid,
            cmap=cmap or CATEGORICAL[0],
            categories=categories,
            extent=extent or _extent_from_data(prepared, array),
            interpolation=interpolation,
            add_colorbar=add_colorbar,
            **plot_kwds,
        )
    else:
        array = _prepare_array(
            prepared, variable=variable, band=band, time=time, rgb=False
        )
        if _is_xarray(array) and extent is None and _has_spatial_coords(array):
            array.plot(
                ax=ax,
                cmap=cmap or SEQUENTIAL[0],
                robust=robust,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=add_colorbar,
                **plot_kwds,
            )
        else:
            np = require("numpy")
            grid = np.asarray(_values(array), dtype=float)
            lo, hi = _limits(grid, vmin=vmin, vmax=vmax, robust=robust)
            image = ax.imshow(
                grid,
                cmap=cmap or SEQUENTIAL[0],
                vmin=lo,
                vmax=hi,
                extent=extent or _extent_from_data(prepared, array),
                origin="upper",
                interpolation=interpolation,
                **plot_kwds,
            )
            if add_colorbar:
                ax.figure.colorbar(image, ax=ax, shrink=0.75)

    if vector is not None:
        _draw_vector_overlay(ax, vector, vector_kwds or {})

    style_axes(ax, title, equal=False, axis_off=axis_off, crs_labels=True)
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_rgb(
    data: Any,
    bands: Sequence[int] = (1, 2, 3),
    *,
    variable: Optional[str] = None,
    time: Any = None,
    stretch: str = "percentile",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw a true-color RGB composite from three bands.

    Band indices are one-based, matching GDAL / rasterio convention.
    """
    return plot_raster(
        data,
        kind="rgb",
        bands=bands,
        variable=variable,
        time=time,
        stretch=stretch,
        percentiles=percentiles,
        brightness=brightness,
        contrast=contrast,
        ax=ax,
        title=title,
        figsize=figsize,
        show=show,
        **plot_kwds,
    )


def plot_false_color(
    data: Any,
    bands: Sequence[int] = (4, 3, 2),
    *,
    title: Optional[str] = "False color (NIR-R-G)",
    **plot_kwds: Any,
) -> Any:
    """Draw a false-color composite (default NIR–Red–Green)."""
    return plot_rgb(data, bands=bands, title=title, **plot_kwds)


def plot_unique_raster(
    data: Any,
    *,
    categories: Optional[Mapping[Any, str]] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw a unique-value / classified raster with a categorical legend."""
    return plot_raster(
        data,
        kind="unique",
        categories=categories,
        cmap=cmap,
        title=title,
        **plot_kwds,
    )


def plot_rasters(
    datasets: Sequence[Any],
    titles: Optional[Sequence[str]] = None,
    *,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Tuple[Any, Any]:
    """Draw several rasters in one figure, one panel each."""
    plt = require("matplotlib.pyplot")
    count = len(datasets)
    if count == 0:
        raise VisualizationError("plot_rasters needs at least one raster.")
    if titles is not None and len(titles) != count:
        raise VisualizationError("titles must have one entry per raster.")
    ncols = max(1, min(ncols, count))
    nrows = (count + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.2 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = list(axes.ravel())
    for index, data in enumerate(datasets):
        heading = None if titles is None else titles[index]
        plot_raster(
            data, ax=flat[index], title=heading, cmap=cmap, show=False, **plot_kwds
        )
    for axis in flat[count:]:
        axis.set_axis_off()
    from geolattice.viz.vector import _finish

    _finish(fig, show)
    return fig, axes


def plot_raster_bands(
    data: Any,
    bands: Optional[Sequence[int]] = None,
    *,
    cmaps: Optional[Sequence[str]] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Optional[Sequence[str]] = None,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Tuple[Any, Any]:
    """Plot selected bands side by side (Python GIS multi-band layout)."""
    plt = require("matplotlib.pyplot")
    prepared = _coerce_input(data)
    stack = _band_stack(prepared)
    count = stack.shape[0]
    indices = list(bands) if bands is not None else list(range(1, count + 1))
    if not indices:
        raise VisualizationError("plot_raster_bands needs at least one band.")
    default_cmaps = ("Reds", "Greens", "Blues", "Purples", "Oranges", "Greys")
    if cmaps is None:
        cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(len(indices))]
    if titles is None:
        titles = [f"Band {index}" for index in indices]
    arrays = [stack[index - 1] for index in indices]

    ncols = max(1, min(ncols, len(arrays)))
    nrows = (len(arrays) + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.2 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = list(axes.ravel())
    for index, array in enumerate(arrays):
        plot_raster(
            array,
            kind="general",
            ax=flat[index],
            title=titles[index],
            cmap=cmaps[index],
            show=False,
            **plot_kwds,
        )
    for axis in flat[len(arrays) :]:
        axis.set_axis_off()
    from geolattice.viz.vector import _finish

    _finish(fig, show)
    return fig, axes


def plot_raster_facet(
    data: Any,
    *,
    dim: str = "band",
    col_wrap: int = 3,
    cmap: str = "viridis",
    robust: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Facet a multiband or time-series raster (ggplot ``facet_wrap`` style)."""
    prepared = _coerce_input(data)
    if _is_xarray(prepared):
        array = prepared
        if hasattr(array, "data_vars"):
            names = list(array.data_vars)
            if len(names) != 1:
                raise VisualizationError(
                    "Pass a DataArray, or a one-variable Dataset."
                )
            array = array[names[0]]
        if dim not in array.dims:
            raise VisualizationError(f"Raster has no {dim!r} dimension to facet on.")
        plot = array.plot(
            col=dim,
            col_wrap=col_wrap,
            cmap=cmap,
            robust=robust,
            figsize=figsize,
            **plot_kwds,
        )
        if title is not None:
            plot.fig.suptitle(title, x=0.01, ha="left")
        from geolattice.viz.vector import _finish

        _finish(plot.fig, show)
        return plot

    stack = _band_stack(prepared)
    titles = [f"{dim} {index}" for index in range(1, stack.shape[0] + 1)]
    return plot_rasters(
        list(stack),
        titles=titles,
        ncols=col_wrap,
        figsize=figsize,
        cmap=cmap,
        robust=robust,
        show=show,
        **plot_kwds,
    )


def plot_raster_histogram(
    data: Any,
    *,
    bands: Optional[Sequence[int]] = None,
    bins: int = 50,
    alpha: float = 0.45,
    ax: Any = None,
    title: Optional[str] = "Raster histogram",
    figsize: Tuple[float, float] = (7, 4),
    show: Optional[bool] = None,
    **hist_kwds: Any,
) -> Any:
    """Plot value histograms for one or more bands.

    Mirrors rasterio ``show_hist`` and GeoHub histogram-driven rescale workflows.
    """
    plt = require("matplotlib.pyplot")
    np = require("numpy")
    prepared = _coerce_input(data)
    stack = _band_stack(prepared)
    indices = list(bands) if bands is not None else list(range(1, stack.shape[0] + 1))
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    for index in indices:
        values = np.asarray(stack[index - 1], dtype=float).ravel()
        values = values[np.isfinite(values)]
        ax.hist(
            values,
            bins=bins,
            alpha=alpha,
            label=f"Band {index}",
            histtype="stepfilled",
            **hist_kwds,
        )
    ax.legend(frameon=False)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, pad=8)
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_contours(
    data: Any,
    *,
    levels: Any = 8,
    filled: bool = False,
    variable: Optional[str] = None,
    cmap: str = "viridis",
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 5),
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw contour lines, or filled contours, for a 2-D grid."""
    plt = require("matplotlib.pyplot")
    prepared = _coerce_input(data)
    array = _prepare_array(prepared, variable=variable, band=None, time=None, rgb=False)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    if _is_xarray(array):
        drawer = array.plot.contourf if filled else array.plot.contour
        drawer(ax=ax, levels=levels, cmap=cmap, **plot_kwds)
    else:
        np = require("numpy")
        grid = np.asarray(array, dtype=float)
        rows, cols = grid.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        drawer = ax.contourf if filled else ax.contour
        drawn = drawer(xx, yy, grid, levels=levels, cmap=cmap, **plot_kwds)
        ax.figure.colorbar(drawn, ax=ax, shrink=0.75)
    style_axes(ax, title, equal=False, crs_labels=True)
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_gridded(
    data: Any,
    *,
    projection: str = "PlateCarree",
    coastline: bool = True,
    gridlines: bool = True,
    variable: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw a longitude/latitude grid on a Cartopy projection."""
    ccrs = require("cartopy.crs")
    plt = require("matplotlib.pyplot")
    if not hasattr(ccrs, projection):
        raise VisualizationError(f"cartopy.crs has no {projection!r} projection.")
    proj = getattr(ccrs, projection)()
    prepared = _coerce_input(data)
    array = _prepare_array(prepared, variable=variable, band=None, time=None, rgb=False)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    transform = ccrs.PlateCarree()
    if _is_xarray(array):
        array.plot(ax=ax, transform=transform, cmap=cmap, **plot_kwds)
    else:
        ax.imshow(array, transform=transform, cmap=cmap, origin="upper", **plot_kwds)
    if coastline:
        ax.coastlines()
    if gridlines:
        ax.gridlines(draw_labels=True, linewidth=0.4)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, pad=8)
    from geolattice.viz.vector import _finish

    _finish(fig, show)
    return ax


def infer_raster_kind(
    data: Any,
    *,
    band: Any = None,
    rgb: bool = False,
    unique_max_classes: int = 32,
) -> RasterKind:
    """Infer ``general``, ``unique``, or ``rgb`` from the raster values."""
    if rgb:
        return "rgb"
    prepared = _coerce_input(data)
    np = require("numpy")
    try:
        stack = _band_stack(prepared)
        if stack.ndim == 3 and stack.shape[0] >= 3 and band is None:
            sample = stack[: min(3, stack.shape[0])]
            if sample.dtype.kind == "f" or float(np.nanmax(sample)) > unique_max_classes:
                return "rgb"
    except VisualizationError:
        pass
    array = _prepare_array(prepared, variable=None, band=band, time=None, rgb=False)
    values = np.asarray(_values(array))
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return "general"
    if np.issubdtype(flat.dtype, np.integer) or np.allclose(flat, np.round(flat)):
        unique = np.unique(np.round(flat).astype(np.int64))
        if 1 < unique.size <= unique_max_classes:
            return "unique"
    return "general"


def normalize_band(
    array: Any,
    *,
    stretch: str = "minmax",
    percentiles: Tuple[float, float] = (2.0, 98.0),
) -> Any:
    """Normalize a band to ``[0, 1]`` for RGB composites."""
    np = require("numpy")
    values = np.asarray(array, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return np.zeros_like(values, dtype=float)
    if stretch == "percentile":
        lo, hi = np.nanpercentile(values[mask], percentiles)
    else:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def adjust_image(
    rgb: Any,
    *,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
) -> Any:
    """Adjust brightness ``[0, 1]``, contrast ``[-1, 1]``, saturation ``[-1, 1]``.

    These match the GeoHub raster style controls for true-color layers.
    """
    np = require("numpy")
    image = np.asarray(rgb, dtype=float)
    if image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    if contrast is not None:
        image = np.clip((image - 0.5) * (1.0 + float(contrast)) + 0.5, 0.0, 1.0)
    if brightness is not None:
        image = np.clip(image * float(np.clip(brightness, 0.0, 1.0)), 0.0, 1.0)
    if saturation is not None and image.ndim == 3 and image.shape[-1] >= 3:
        gray = image[..., :3].mean(axis=-1, keepdims=True)
        image = np.clip(gray + (image - gray) * (1.0 + float(saturation)), 0.0, 1.0)
    return image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_unique(
    ax: Any,
    grid: Any,
    *,
    cmap: str,
    categories: Optional[Mapping[Any, str]],
    extent: Optional[Sequence[float]],
    interpolation: str,
    add_colorbar: bool,
    **plot_kwds: Any,
) -> Any:
    np = require("numpy")
    mpl_colors = require("matplotlib.colors")
    plt = require("matplotlib.pyplot")
    values = np.asarray(grid)
    finite = values[np.isfinite(values)]
    codes = (
        np.unique(np.round(finite).astype(np.int64)) if finite.size else np.array([])
    )
    if codes.size == 0:
        ax.imshow(values, extent=extent, origin="upper", **plot_kwds)
        return ax
    indexed = np.full(values.shape, np.nan, dtype=float)
    for index, code in enumerate(codes):
        indexed[np.round(values) == code] = index
    colormap = plt.get_cmap(cmap, int(codes.size))
    norm = mpl_colors.BoundaryNorm(
        np.arange(-0.5, codes.size + 0.5, 1.0), colormap.N
    )
    image = ax.imshow(
        indexed,
        cmap=colormap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation=interpolation,
        **plot_kwds,
    )
    if add_colorbar:
        colorbar = ax.figure.colorbar(
            image, ax=ax, shrink=0.75, ticks=list(range(int(codes.size)))
        )
        labels = []
        for code in codes:
            key = int(code)
            if categories and key in categories:
                labels.append(str(categories[key]))
            elif categories and str(key) in categories:
                labels.append(str(categories[str(key)]))
            else:
                labels.append(str(key))
        colorbar.ax.set_yticklabels(labels)
    return ax


def _rgb_array(
    data: Any,
    *,
    variable: Optional[str],
    time: Any,
    bands: Sequence[int],
    stretch: str,
    percentiles: Tuple[float, float],
    brightness: Optional[float],
    contrast: Optional[float],
) -> Any:
    np = require("numpy")
    if len(bands) < 3:
        raise VisualizationError("RGB composites need at least three bands.")
    stack = _band_stack(_select_cube(data, variable=variable, time=time))
    planes = []
    for band in bands[:3]:
        if band < 1 or band > stack.shape[0]:
            raise VisualizationError(
                f"Band {band} is outside 1..{stack.shape[0]} for this raster."
            )
        planes.append(
            normalize_band(
                stack[band - 1], stretch=stretch, percentiles=percentiles
            )
        )
    rgb = np.dstack(planes)
    if brightness is not None or contrast is not None:
        rgb = adjust_image(rgb, brightness=brightness, contrast=contrast)
    return rgb


def _draw_vector_overlay(ax: Any, vector: Any, kwds: Mapping[str, Any]) -> None:
    try:
        import geopandas  # noqa: F401
    except ImportError as exc:
        raise VisualizationError(
            "vector= overlays need geopandas. "
            'Install with: pip install "geolattice[viz]"'
        ) from exc
    if not hasattr(vector, "plot"):
        raise VisualizationError("vector= must be a GeoDataFrame-like object.")
    style = {"facecolor": "none", "edgecolor": "#101828", "linewidth": 0.6}
    style.update(dict(kwds))
    vector.plot(ax=ax, **style)


def _coerce_input(data: Any) -> Any:
    if data is None:
        raise VisualizationError("Raster data is required.")
    if type(data).__name__ == "RasterDataset" and hasattr(data, "read"):
        return _raster_dataset_array(data)
    path_like = isinstance(data, (str, Path)) or hasattr(data, "__fspath__")
    if path_like and not hasattr(data, "read"):
        from geolattice.raster.raster_read_write import open_raster_data

        with open_raster_data(data) as dataset:
            return _raster_dataset_array(dataset)
    return data


def _raster_dataset_array(dataset: Any) -> "_RasterArray":
    np = require("numpy")
    array = np.asarray(dataset.read(), dtype=float)
    nodata = None
    bands = getattr(dataset, "bands", None)
    if bands:
        nodata = getattr(bands[0], "nodata", None)
    if nodata is None:
        nodata = getattr(dataset, "nodata", None)
    if nodata is not None and np.isfinite(nodata):
        array = np.where(array == nodata, np.nan, array)
    elif nodata is not None:
        # Sentinel values such as ±1.7e308 are not finite; mask extremes.
        array = np.where(np.isfinite(array) & (np.abs(array) < 1e30), array, np.nan)
    else:
        array = np.where(np.isfinite(array), array, np.nan)
    return _RasterArray(
        array,
        geotransform=getattr(dataset, "geotransform", None),
        crs=getattr(dataset, "crs", None),
    )


class _RasterArray:
    """Thin wrapper so geotransform survives for extent-aware plots."""

    def __init__(self, array: Any, geotransform: Any = None, crs: Any = None) -> None:
        self.array = array
        self.geotransform = geotransform
        self.crs = crs

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.array.shape)

    def __array__(self, dtype: Any = None) -> Any:
        np = require("numpy")
        return np.asarray(self.array, dtype=dtype)


def _values(data: Any) -> Any:
    if isinstance(data, _RasterArray):
        return data.array
    if hasattr(data, "values") and not isinstance(data, (list, tuple)):
        try:
            return data.values
        except Exception:
            pass
    return data


def _band_stack(data: Any) -> Any:
    np = require("numpy")
    if isinstance(data, _RasterArray):
        array = np.asarray(data.array)
    elif _is_xarray(data):
        array = _xarray_to_band_stack(data)
    else:
        array = np.asarray(_values(data))
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        if array.shape[-1] in (3, 4) and array.shape[-1] < array.shape[0]:
            return np.moveaxis(array, -1, 0)
        return array
    raise VisualizationError(f"Expected a 2-D or 3-D raster, got shape {array.shape!r}.")


def _xarray_to_band_stack(data: Any) -> Any:
    np = require("numpy")
    array = data
    if hasattr(array, "data_vars"):
        names = list(array.data_vars)
        if len(names) != 1:
            raise VisualizationError("Pass a DataArray or a one-variable Dataset.")
        array = array[names[0]]
    values = np.asarray(array.values)
    if values.ndim == 2:
        return values[None, ...]
    dims = tuple(getattr(array, "dims", ()))
    for name in ("band", "bands"):
        if name in dims:
            return np.moveaxis(values, dims.index(name), 0)
    if values.ndim == 3:
        return values
    raise VisualizationError("Could not locate a band dimension on the xarray object.")


def _select_cube(data: Any, *, variable: Optional[str], time: Any) -> Any:
    if isinstance(data, _RasterArray) or not _is_xarray(data):
        return data
    array = data
    if hasattr(array, "data_vars"):
        if variable is None:
            names = list(array.data_vars)
            if len(names) != 1:
                raise VisualizationError("Pass variable= for a multi-variable Dataset.")
            variable = names[0]
        array = array[variable]
    return _select_time(array, time)


def _extent_from_data(
    data: Any, array: Any = None
) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(data, _RasterArray) and data.geotransform is not None:
        np = require("numpy")
        gt = data.geotransform
        source = array if array is not None else data.array
        height, width = np.asarray(_values(source)).shape[-2:]
        left = float(gt[0])
        top = float(gt[3])
        right = left + width * float(gt[1]) + height * float(gt[2])
        bottom = top + width * float(gt[4]) + height * float(gt[5])
        return (left, right, bottom, top)
    target = array if array is not None else data
    if _is_xarray(target) and _has_spatial_coords(target):
        x_name = _spatial_name(target, ("x", "lon", "longitude"))
        y_name = _spatial_name(target, ("y", "lat", "latitude"))
        if x_name and y_name:
            xs = target[x_name].values
            ys = target[y_name].values
            return (
                float(xs.min()),
                float(xs.max()),
                float(ys.min()),
                float(ys.max()),
            )
    return None


def _has_spatial_coords(data: Any) -> bool:
    return (
        _spatial_name(data, ("x", "lon", "longitude")) is not None
        and _spatial_name(data, ("y", "lat", "latitude")) is not None
    )


def _spatial_name(data: Any, names: Sequence[str]) -> Optional[str]:
    dims = set(getattr(data, "dims", ()))
    coords = set(getattr(data, "coords", {}))
    for name in names:
        if name in dims or name in coords:
            return name
    return None


def _limits(
    grid: Any,
    *,
    vmin: Optional[float],
    vmax: Optional[float],
    robust: bool,
) -> Tuple[Optional[float], Optional[float]]:
    if vmin is not None or vmax is not None:
        return vmin, vmax
    if not robust:
        return None, None
    np = require("numpy")
    values = np.asarray(grid, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    lo, hi = np.nanpercentile(finite, (2.0, 98.0))
    return float(lo), float(hi)


def _prepare_array(
    data: Any,
    *,
    variable: Optional[str],
    band: Any,
    time: Any,
    rgb: bool,
) -> Any:
    np = require("numpy")
    if isinstance(data, _RasterArray):
        array = np.asarray(data.array)
        if array.ndim == 3:
            if rgb:
                if array.shape[0] >= 3:
                    return np.moveaxis(array[:3], 0, -1)
                raise VisualizationError("RGB needs at least three bands.")
            if band is None:
                return array[0]
            index = int(band)
            return array[index - 1 if index >= 1 else index]
        return array
    if isinstance(data, np.ndarray):
        if data.ndim == 3 and not rgb:
            if band is None:
                return data[0] if data.shape[0] <= 16 else data[:, :, 0]
            index = int(band)
            if data.shape[0] <= 16:
                return data[index - 1 if index >= 1 else index]
            return data[:, :, index - 1 if index >= 1 else index]
        if data.ndim != 2 and not (rgb and data.ndim == 3):
            raise VisualizationError(
                "A NumPy raster must be 2-D, or 3-D when rgb=True."
            )
        return data
    xr = _xarray()
    if isinstance(data, xr.Dataset):
        names = list(data.data_vars)
        if variable is None:
            if len(names) != 1:
                raise VisualizationError(
                    "Pass variable= for a Dataset. Available variables: "
                    + ", ".join(names)
                )
            variable = names[0]
        if variable not in data.data_vars:
            raise VisualizationError(f"The Dataset has no {variable!r} variable.")
        data = data[variable]
    elif not isinstance(data, xr.DataArray):
        raise VisualizationError(
            "Expected a NumPy array, an xarray DataArray, an xarray Dataset, "
            "or a RasterDataset."
        )
    data = _select_time(data, time)
    data = _select_band(data, band, rgb=rgb)
    data = data.squeeze(drop=True)
    if rgb:
        if "band" not in data.dims or data.sizes.get("band", 0) < 3:
            raise VisualizationError(
                "rgb=True needs a 'band' dimension with at least 3 bands."
            )
        return data.isel(band=slice(0, 3))
    if data.ndim != 2:
        raise VisualizationError(
            "The raster still has more than two dimensions. "
            "Select them with band= or time=."
        )
    return data


def _select_time(data: Any, time: Any) -> Any:
    if time is None:
        return data
    for name in ("time", "t"):
        if name in data.dims or name in data.coords:
            return data.sel({name: time})
    raise VisualizationError("This raster has no time coordinate.")


def _select_band(data: Any, band: Any, *, rgb: bool) -> Any:
    if band is None or rgb:
        return data
    if "band" not in data.dims and "band" not in getattr(data, "coords", {}):
        raise VisualizationError("This raster has no band coordinate.")
    coords = list(data.coords.get("band", range(data.sizes.get("band", 0))).values)
    if isinstance(band, int) and band not in coords:
        if band >= 1 and band - 1 < data.sizes.get("band", 0):
            return data.isel(band=band - 1)
        return data.isel(band=band)
    return data.sel(band=band)


def _is_xarray(data: Any) -> bool:
    try:
        xr = _xarray()
    except Exception:
        return False
    try:
        return isinstance(data, (xr.DataArray, xr.Dataset))
    except Exception:
        return False


def _xarray() -> Any:
    return require("xarray")
