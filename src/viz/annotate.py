"""Labels, a north arrow, and a scale bar for a static map."""

from __future__ import annotations

from typing import Any, Optional

from geolattice.viz._data import as_geodataframe, require_features
from geolattice.viz._imports import VisualizationError
from geolattice.viz.style import EDGE


def add_labels(
    ax: Any,
    frame: Any,
    column: str,
    *,
    fontsize: float = 8,
    color: str = EDGE,
    max_labels: int = 200,
    halo: bool = True,
) -> Any:
    """Write one attribute at each feature's representative point.

    A white halo keeps the text readable on both light and dark fills.
    """
    data = require_features(as_geodataframe(frame))
    if column not in data.columns:
        raise VisualizationError(f"The layer has no {column!r} column to label.")
    rows = data if len(data) <= max_labels else data.iloc[:max_labels]
    effects = None
    if halo:
        from matplotlib import patheffects

        effects = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for _, row in rows.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        point = geom.representative_point()
        ax.annotate(
            str(row[column]),
            (point.x, point.y),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            path_effects=effects,
        )
    return ax


def add_north_arrow(
    ax: Any,
    *,
    location: str = "upper right",
    color: str = EDGE,
) -> Any:
    """Draw a north arrow in axes coordinates so it stays put while zooming the data."""
    x, y_tip, y_tail = {
        "upper right": (0.94, 0.94, 0.82),
        "upper left": (0.08, 0.94, 0.82),
        "lower right": (0.94, 0.22, 0.10),
        "lower left": (0.08, 0.22, 0.10),
    }.get(location, (0.94, 0.94, 0.82))
    ax.annotate(
        "N",
        xy=(x, y_tip),
        xytext=(x, y_tail),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=color,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2},
    )
    return ax


def add_scale_bar(
    ax: Any,
    *,
    length: Optional[float] = None,
    location: str = "lower left",
    unit: Optional[str] = None,
    crs: Any = None,
    color: str = EDGE,
) -> Any:
    """Draw a scale bar using the axes coordinates.

    ``length`` is in the units of the axes. When it is omitted, the bar is
    about one fifth of the map width, rounded to a 1-2-5 step. Geographic
    axes are labeled in degrees unless ``unit`` is set.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span = abs(xmax - xmin)
    if span == 0:
        raise VisualizationError("The axes have a zero width, so a scale bar cannot be drawn.")
    bar = _nice_length(span / 5) if length is None else float(length)
    label_unit = unit if unit is not None else _unit_name(crs)
    x0 = xmin + 0.06 * span if "left" in location else xmax - 0.06 * span - bar
    y0 = ymin + 0.06 * abs(ymax - ymin) if "lower" in location else ymax - 0.08 * abs(ymax - ymin)
    ax.plot([x0, x0 + bar], [y0, y0], color=color, linewidth=2, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - span * 0.008, y0 + span * 0.008], color=color, linewidth=1.2)
    ax.plot(
        [x0 + bar, x0 + bar],
        [y0 - span * 0.008, y0 + span * 0.008],
        color=color,
        linewidth=1.2,
    )
    ax.text(
        x0 + bar / 2,
        y0 + span * 0.015,
        _format_length(bar, label_unit),
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
    )
    return ax


def _nice_length(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10 ** int(f"{value:.0e}".split("e")[1])
    residual = value / power
    step = 1 if residual < 1.5 else 2 if residual < 3.5 else 5 if residual < 7.5 else 10
    return step * power


def _format_length(length: float, unit: str) -> str:
    display = float(length)
    label_unit = unit
    if unit in {"m", "metre", "meter"} and display >= 1000:
        display = display / 1000.0
        label_unit = "km"
    if abs(display - round(display)) < 1e-8:
        text = str(int(round(display)))
    else:
        text = f"{display:g}"
    return f"{text} {label_unit}".strip()


def _unit_name(crs: Any) -> str:
    if crs is None:
        return ""
    try:
        if crs.is_geographic:
            return "°"
    except Exception:
        return ""
    try:
        name = crs.axis_info[0].unit_name
    except Exception:
        return ""
    if name in {"metre", "meter", "m"}:
        return "m"
    return name or ""


def cartographic_elements(
    ax: Any,
    frame: Any = None,
    *,
    north: bool = True,
    scale: bool = True,
    length: Optional[float] = None,
) -> Any:
    """Add a north arrow and a scale bar to a map that was already drawn."""
    if north:
        add_north_arrow(ax)
    if scale:
        crs = None if frame is None else getattr(as_geodataframe(frame), "crs", None)
        add_scale_bar(ax, length=length, crs=crs)
    return ax
