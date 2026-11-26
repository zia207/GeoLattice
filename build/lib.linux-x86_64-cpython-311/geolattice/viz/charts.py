"""Attribute charts that sit next to a map.

These cover the bar, pie, histogram, and scatter plots used to inspect the
columns that a choropleth is about to show.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from geolattice.viz._imports import VisualizationError, require
from geolattice.viz.style import FACE, style_axes


def plot_bar(
    data: Any,
    column: str,
    *,
    labels: Optional[str] = None,
    ax: Any = None,
    title: Optional[str] = None,
    color: str = FACE,
    horizontal: bool = False,
    figsize: Tuple[float, float] = (7, 4),
    show: Optional[bool] = None,
) -> Any:
    """Bar chart of one numeric column, summed when ``labels`` groups the rows."""
    plt = require("matplotlib.pyplot")
    frame = _table(data)
    _need(frame, column)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    if labels is None:
        series = frame[column]
        names = [str(item) for item in series.index]
        values = series.to_numpy()
    else:
        _need(frame, labels)
        grouped = frame.groupby(labels, dropna=False)[column].sum()
        names = [str(item) for item in grouped.index]
        values = grouped.to_numpy()
    if horizontal:
        ax.barh(names, values, color=color)
        ax.tick_params(axis="y", labelsize=8)
        style_axes(ax, title, equal=False, xlabel=column, ylabel="")
    else:
        ax.bar(names, values, color=color)
        ax.tick_params(axis="x", labelrotation=45)
        style_axes(ax, title, equal=False, xlabel="", ylabel=column)
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_pie(
    data: Any,
    column: str,
    *,
    labels: Optional[str] = None,
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
    show: Optional[bool] = None,
) -> Any:
    """Pie chart of one numeric column."""
    plt = require("matplotlib.pyplot")
    frame = _table(data)
    _need(frame, column)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    if labels is None:
        values = frame[column].to_numpy()
        names: Sequence[Any] = [str(item) for item in frame.index]
    else:
        _need(frame, labels)
        grouped = frame.groupby(labels, dropna=False)[column].sum()
        values = grouped.to_numpy()
        names = [str(item) for item in grouped.index]
    if (values < 0).any():
        raise VisualizationError("A pie chart needs values that are zero or positive.")
    ax.pie(values, labels=list(names), autopct="%1.0f%%")
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.set_aspect("equal")
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_histogram(
    data: Any,
    column: str,
    *,
    bins: int = 20,
    ax: Any = None,
    title: Optional[str] = None,
    color: str = FACE,
    figsize: Tuple[float, float] = (6, 4),
    show: Optional[bool] = None,
) -> Any:
    """Histogram of one numeric column."""
    plt = require("matplotlib.pyplot")
    frame = _table(data)
    _need(frame, column)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    ax.hist(frame[column].dropna().to_numpy(), bins=bins, color=color, edgecolor="white")
    style_axes(ax, title, equal=False, xlabel=column, ylabel="count")
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def plot_scatter(
    data: Any,
    x: str,
    y: str,
    *,
    column: Optional[str] = None,
    cmap: str = "viridis",
    ax: Any = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    show: Optional[bool] = None,
) -> Any:
    """Scatter two attribute columns, optionally colored by a third."""
    plt = require("matplotlib.pyplot")
    frame = _table(data)
    _need(frame, x)
    _need(frame, y)
    own = ax is None
    if own:
        _fig, ax = plt.subplots(figsize=figsize)
    if column is None:
        ax.scatter(frame[x], frame[y], c=FACE, s=18, linewidths=0)
    else:
        _need(frame, column)
        plotted = ax.scatter(
            frame[x],
            frame[y],
            c=frame[column],
            cmap=cmap,
            s=18,
            linewidths=0,
        )
        ax.figure.colorbar(plotted, ax=ax, shrink=0.75, label=column)
    style_axes(ax, title, equal=False, xlabel=x, ylabel=y)
    if own:
        from geolattice.viz.vector import _finish

        _finish(ax.figure, show)
    return ax


def _table(data: Any) -> Any:
    pd = require("pandas")
    gpd = require("geopandas")
    if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        return data
    raise VisualizationError("Charts expect a DataFrame or a GeoDataFrame.")


def _need(frame: Any, column: str) -> None:
    if column not in frame.columns:
        raise VisualizationError(f"The table has no {column!r} column.")
