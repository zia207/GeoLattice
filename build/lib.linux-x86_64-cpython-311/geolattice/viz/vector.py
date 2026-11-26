"""Static maps for vector layers.

These helpers follow ``GeoDataFrame.plot`` as described in the GeoPandas
mapping guide, and they keep the ``plot_layer`` / ``plot_pair`` calls used by
the vector geoprocessing tutorial.

https://geopandas.org/en/stable/docs/user_guide/mapping.html
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from geolattice.viz._data import as_geodataframe, geometry_kind, require_features
from geolattice.viz._imports import VisualizationError, require
from geolattice.viz.style import EDGE, FACE, style_axes

LayerSpec = Union[Any, Mapping[str, Any]]


def plot_layer(
    frame: Any,
    title: Optional[str] = None,
    ax: Any = None,
    facecolor: str = FACE,
    edgecolor: str = EDGE,
    linewidth: float = 0.4,
    column: Optional[str] = None,
    cmap: Optional[str] = None,
    fill: bool = True,
    *,
    figsize: Tuple[float, float] = (6, 5),
    alpha: Optional[float] = None,
    markersize: Optional[float] = None,
    categorical: bool = False,
    scheme: Optional[str] = None,
    k: int = 5,
    legend: Optional[bool] = None,
    legend_kwds: Optional[Mapping[str, Any]] = None,
    classification_kwds: Optional[Mapping[str, Any]] = None,
    missing_kwds: Optional[Mapping[str, Any]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cax: Any = None,
    zorder: Optional[int] = None,
    label: Optional[str] = None,
    axis_off: bool = False,
    crs_labels: bool = False,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw one polygon, line, or point layer.

    A ``column`` draws a choropleth. ``scheme`` names a mapclassify method
    such as ``quantiles`` or ``fisher_jenks`` (GeoPandas passes it through).
    ``missing_kwds`` styles features whose value is missing. ``fill=False``
    draws polygon outlines only.

    The return value is the Matplotlib axes. A new figure is shown when
    ``ax`` is omitted, matching the tutorial helpers.
    """
    plt = require("matplotlib.pyplot")
    data = require_features(as_geodataframe(frame))
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)

    kwargs = _layer_kwargs(
        data,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        column=column,
        cmap=cmap,
        fill=fill,
        alpha=alpha,
        markersize=markersize,
        categorical=categorical,
        scheme=scheme,
        k=k,
        legend=legend,
        legend_kwds=legend_kwds,
        classification_kwds=classification_kwds,
        missing_kwds=missing_kwds,
        vmin=vmin,
        vmax=vmax,
        cax=cax,
        zorder=zorder,
        label=label,
        plot_kwds=plot_kwds,
    )
    kwargs["ax"] = ax
    try:
        data.plot(**kwargs)
    except ImportError as exc:
        if scheme is not None:
            raise VisualizationError(
                "A classified choropleth needs mapclassify. "
                'Install it with: pip install "geolattice[viz]"'
            ) from exc
        raise
    style_axes(ax, title, axis_off=axis_off, crs_labels=crs_labels)
    if own:
        _finish(ax.figure, show)
    return ax


def plot_pair(
    left: Any,
    right: Any,
    titles: Sequence[str],
    figsize: Tuple[float, float] = (10, 5),
    fill: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Draw two layers side by side.

    The right-hand panel uses a thinner outline unless ``linewidth`` is set,
    which is how the geoprocessing tutorial compares a state layer with a
    county layer.
    """
    plt = require("matplotlib.pyplot")
    if len(titles) != 2:
        raise VisualizationError("titles must contain the left title and the right title.")
    show = kwargs.pop("show", None)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_layer(left, titles[0], ax=axes[0], fill=fill, show=False, **kwargs)
    right_kwargs = dict(kwargs)
    right_kwargs.setdefault("linewidth", 0.2)
    plot_layer(right, titles[1], ax=axes[1], fill=fill, show=False, **right_kwargs)
    _finish(fig, show)
    return fig, axes


def plot_choropleth(
    frame: Any,
    column: str,
    title: Optional[str] = None,
    *,
    cmap: str = "OrRd",
    scheme: Optional[str] = None,
    k: int = 5,
    legend: bool = True,
    legend_orientation: str = "vertical",
    legend_kwds: Optional[Mapping[str, Any]] = None,
    classification_kwds: Optional[Mapping[str, Any]] = None,
    missing_kwds: Optional[Mapping[str, Any]] = None,
    categorical: bool = False,
    ax: Any = None,
    figsize: Tuple[float, float] = (7, 6),
    edgecolor: str = "white",
    linewidth: float = 0.3,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Color each feature by an attribute.

    ``scheme`` is any mapclassify scheme GeoPandas accepts, including
    ``quantiles``, ``equal_interval``, ``fisher_jenks``, and ``natural_breaks``.
    A horizontal legend is placed on its own axes under the map, the same
    layout as the GeoPandas ``cax`` example.
    """
    if not column:
        raise VisualizationError("column is required for a choropleth.")
    plt = require("matplotlib.pyplot")
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    cax = None
    legend_options = dict(legend_kwds or {})
    if legend and legend_orientation == "horizontal":
        cax = ax.inset_axes([0.0, -0.22, 1.0, 0.06])
        legend_options.setdefault("orientation", "horizontal")
    plot_layer(
        frame,
        title,
        ax=ax,
        column=column,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        scheme=scheme,
        k=k,
        legend=legend,
        legend_kwds=legend_options or None,
        classification_kwds=classification_kwds,
        missing_kwds=missing_kwds,
        categorical=categorical,
        cax=cax,
        show=False,
        **plot_kwds,
    )
    if own:
        _finish(ax.figure, show)
    return ax


def plot_categorical(
    frame: Any,
    column: str,
    title: Optional[str] = None,
    *,
    cmap: str = "tab10",
    ax: Any = None,
    legend: bool = True,
    missing_kwds: Optional[Mapping[str, Any]] = None,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Color features by a category such as a name or a land-use code."""
    return plot_choropleth(
        frame,
        column,
        title,
        cmap=cmap,
        categorical=True,
        legend=legend,
        missing_kwds=missing_kwds,
        ax=ax,
        show=show,
        **plot_kwds,
    )


def plot_boundary(
    frame: Any,
    title: Optional[str] = None,
    ax: Any = None,
    *,
    edgecolor: str = EDGE,
    linewidth: float = 0.6,
    figsize: Tuple[float, float] = (6, 5),
    axis_off: bool = False,
    show: Optional[bool] = None,
    **plot_kwds: Any,
) -> Any:
    """Draw polygon boundaries with an empty interior.

    This uses ``GeoSeries.boundary``, which is clearer than passing
    ``facecolor="none"`` because ``None`` and ``\"none\"`` mean different
    things to Matplotlib.
    """
    plt = require("matplotlib.pyplot")
    data = require_features(as_geodataframe(frame))
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    data.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth, **plot_kwds)
    style_axes(ax, title, axis_off=axis_off)
    if own:
        _finish(ax.figure, show)
    return ax


def plot_layers(
    layers: Sequence[LayerSpec],
    *,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    legend: bool = True,
    basemap: bool = False,
    basemap_source: Any = "CartoDB.Positron",
    axis_off: bool = False,
    show: Optional[bool] = None,
) -> Any:
    """Overlay several layers on one axes.

    Each item is a GeoDataFrame or a mapping with a ``data`` key and any
    ``plot_layer`` style keyword, including ``zorder`` and ``label``. Later
    layers are reprojected to the first layer's CRS. With ``basemap=True``,
    geographic layers are drawn in Web Mercator so the tiles stay sharp, then
    contextily adds the tiles underneath.

    Lower ``zorder`` is drawn first. When ``zorder`` is omitted, the first
    layer is drawn underneath the next, and points stay above polygons.
    """
    plt = require("matplotlib.pyplot")
    if not layers:
        raise VisualizationError("plot_layers needs at least one layer.")
    specs = [_normalize_layer(item, index) for index, item in enumerate(layers)]
    prepared = _align_layers(specs, basemap=basemap)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    crs = None
    proxies = []
    for spec in prepared:
        data = spec.pop("data")
        spec.pop("name", None)
        if crs is None:
            crs = data.crs
        label = spec.get("label")
        if label:
            proxies.append(_legend_proxy(label, geometry_kind(data), spec))
        plot_layer(data, ax=ax, show=False, **spec)
    if legend and proxies:
        ax.legend(handles=proxies, loc="best", frameon=True)
    if basemap:
        from geolattice.viz.basemap import add_basemap

        add_basemap(ax, source=basemap_source, crs=crs)
    style_axes(ax, title, axis_off=axis_off or basemap)
    if own:
        _finish(ax.figure, show)
    return ax


def save_map(fig: Any, path: str, *, dpi: int = 200, bbox_inches: str = "tight") -> Any:
    """Write a Matplotlib figure to an image file and return the path."""
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches=bbox_inches)
    return destination


def plot_points(
    points: Any,
    *,
    values: Any = None,
    cmap: str = "viridis",
    s: float = 8,
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    color: str = FACE,
    show: Optional[bool] = None,
) -> Any:
    """Scatter point coordinates.

    ``points`` may be an ``(N, 2)`` or ``(N, 3)`` array. The third column is
    used as the color value when ``values`` is omitted, which is enough for a
    quick view of an elevation sample or a small point cloud.
    """
    np = require("numpy")
    plt = require("matplotlib.pyplot")
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise VisualizationError("points must have shape (N, 2) or (N, 3).")
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    color_values = values
    if color_values is None and array.shape[1] == 3:
        color_values = array[:, 2]
    if color_values is None:
        ax.scatter(array[:, 0], array[:, 1], s=s, c=color, linewidths=0)
    else:
        plotted = ax.scatter(
            array[:, 0],
            array[:, 1],
            s=s,
            c=np.asarray(color_values),
            cmap=cmap,
            linewidths=0,
        )
        ax.figure.colorbar(plotted, ax=ax, shrink=0.75)
    style_axes(ax, title)
    if own:
        _finish(ax.figure, show)
    return ax


def animate_map(
    frame: Any,
    time: str,
    *,
    column: Optional[str] = None,
    cmap: Optional[str] = None,
    interval: int = 500,
    figsize: Tuple[float, float] = (7, 5),
    title: Optional[str] = None,
    path: Optional[str] = None,
    repeat: bool = True,
) -> Any:
    """Animate one map frame for each distinct value of ``time``.

    ``path`` may end in ``.gif`` or ``.mp4``. A GIF is written with Pillow.
    The animation uses one unchanging extent so the features do not jump.
    """
    plt = require("matplotlib.pyplot")
    animation = require("matplotlib.animation")
    data = require_features(as_geodataframe(frame))
    if time not in data.columns:
        raise VisualizationError(f"The layer has no {time!r} column to animate.")
    stamps = list(dict.fromkeys(data[time].tolist()))
    if len(stamps) < 2:
        raise VisualizationError("Animation needs at least two time values.")
    bounds = data.total_bounds
    fig, ax = plt.subplots(figsize=figsize)

    def _draw(stamp: Any) -> None:
        ax.clear()
        part = data.loc[data[time] == stamp]
        heading = f"{title} — {stamp}" if title else str(stamp)
        if len(part):
            plot_layer(part, heading, ax=ax, column=column, cmap=cmap, show=False)
        else:
            style_axes(ax, heading)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    anim = animation.FuncAnimation(
        fig,
        lambda index: _draw(stamps[index]),
        frames=len(stamps),
        interval=interval,
        repeat=repeat,
    )
    if path:
        _save_animation(anim, path, interval)
    return anim


def _layer_kwargs(
    data: Any,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    column: Optional[str],
    cmap: Optional[str],
    fill: bool,
    alpha: Optional[float],
    markersize: Optional[float],
    categorical: bool,
    scheme: Optional[str],
    k: int,
    legend: Optional[bool],
    legend_kwds: Optional[Mapping[str, Any]],
    classification_kwds: Optional[Mapping[str, Any]],
    missing_kwds: Optional[Mapping[str, Any]],
    vmin: Optional[float],
    vmax: Optional[float],
    cax: Any,
    zorder: Optional[int],
    label: Optional[str],
    plot_kwds: Mapping[str, Any],
) -> dict:
    kind = geometry_kind(data)
    kwargs = {"linewidth": linewidth, **dict(plot_kwds)}
    if column is not None:
        if column not in data.columns:
            raise VisualizationError(f"The layer has no {column!r} column.")
        kwargs.update(
            column=column,
            legend=True if legend is None else legend,
            categorical=categorical,
            k=k,
        )
        if cmap is not None:
            kwargs["cmap"] = cmap
        if scheme is not None:
            kwargs["scheme"] = scheme
        if not fill:
            kwargs["facecolor"] = "none"
            kwargs["edgecolor"] = edgecolor
        elif edgecolor is not None and kind == "polygon":
            kwargs["edgecolor"] = edgecolor
    elif kind == "polygon":
        kwargs.setdefault("facecolor", facecolor if fill else "none")
        kwargs.setdefault("edgecolor", edgecolor)
        if not fill:
            kwargs["facecolor"] = "none"
    elif kind == "line":
        kwargs.setdefault("color", facecolor)
    elif kind == "point":
        kwargs.setdefault("color", facecolor)
        kwargs["markersize"] = 8 if markersize is None else markersize
        if edgecolor != EDGE:
            kwargs["edgecolor"] = edgecolor
    else:
        kwargs.setdefault("color", facecolor)
        kwargs.setdefault("edgecolor", edgecolor)
    if markersize is not None:
        kwargs["markersize"] = markersize
    if alpha is not None:
        kwargs["alpha"] = alpha
    if legend_kwds is not None:
        kwargs["legend_kwds"] = dict(legend_kwds)
    if classification_kwds is not None:
        kwargs["classification_kwds"] = dict(classification_kwds)
    if missing_kwds is not None:
        kwargs["missing_kwds"] = dict(missing_kwds)
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    if cax is not None:
        kwargs["cax"] = cax
    if zorder is not None:
        kwargs["zorder"] = zorder
    if label:
        kwargs["label"] = label
    return kwargs


def _legend_proxy(label: str, kind: str, spec: Mapping[str, Any]) -> Any:
    """Matplotlib cannot put a polygon PatchCollection in a legend, so use a proxy."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    color = spec.get("color") or spec.get("facecolor") or FACE
    edge = spec.get("edgecolor", EDGE)
    if kind == "line":
        return Line2D([0], [0], color=color, linewidth=spec.get("linewidth", 1.5), label=label)
    if kind == "point":
        return Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            linestyle="",
            label=label,
        )
    face = "none" if spec.get("fill", True) is False else color
    return Patch(facecolor=face, edgecolor=edge, label=label)


def _normalize_layer(item: LayerSpec, index: int) -> dict:
    if isinstance(item, Mapping) and "data" in item:
        spec = dict(item)
    else:
        spec = {"data": item}
    spec["data"] = require_features(as_geodataframe(spec["data"]))
    spec.setdefault("zorder", index + 1)
    return spec


def _align_layers(specs: Sequence[dict], *, basemap: bool) -> list:
    base = specs[0]["data"]
    target = base.crs
    if basemap:
        if target is None:
            raise VisualizationError("A basemap needs a layer with a CRS.")
        if _is_geographic(target):
            target = "EPSG:3857"
    aligned = []
    for spec in specs:
        data = spec["data"]
        if target is not None and data.crs is not None and not _same_crs(data.crs, target):
            warnings.warn(
                "Reprojecting a layer so it lines up with the base map.",
                UserWarning,
                stacklevel=3,
            )
            data = data.to_crs(target)
        spec = dict(spec)
        spec["data"] = data
        aligned.append(spec)
    return aligned


def _is_geographic(crs: Any) -> bool:
    try:
        return bool(crs.is_geographic)
    except Exception:
        return False


def _same_crs(left: Any, right: Any) -> bool:
    if right == "EPSG:3857":
        try:
            return left.to_epsg() == 3857
        except Exception:
            return False
    try:
        return left == right
    except Exception:
        return str(left) == str(right)


def _finish(fig: Any, show: Optional[bool]) -> None:
    fig.tight_layout()
    if show is False:
        return
    plt = require("matplotlib.pyplot")
    if show or _in_ipython():
        plt.show()


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ in {"ZMQInteractiveShell", "TerminalInteractiveShell"}


def _save_animation(anim: Any, path: str, interval: int) -> None:
    if path.lower().endswith(".gif"):
        try:
            from matplotlib.animation import PillowWriter
        except ImportError as exc:
            raise VisualizationError("Saving a GIF requires pillow.") from exc
        anim.save(path, writer=PillowWriter(fps=max(1, int(round(1000 / max(interval, 1))))))
        return
    anim.save(path)
