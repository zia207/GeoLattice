"""Map colors and axes styling.

The default face and edge colors are the ones used by the vector
geoprocessing tutorial. ``apply_map_style`` sets the same Matplotlib theme.
"""

from __future__ import annotations

from typing import Any, Optional

from geolattice.viz._imports import require

FACE = "#2A6F97"
EDGE = "#012A4A"
GRID = "#E4E7EC"

# Matplotlib colormap names used for continuous, diverging, and classed maps.
# See https://matplotlib.org/stable/tutorials/colors/colormaps.html
SEQUENTIAL = ("viridis", "OrRd", "YlGnBu", "Blues", "plasma")
DIVERGING = ("RdBu_r", "BrBG", "PiYG", "coolwarm")
CATEGORICAL = ("tab10", "tab20", "Set2", "Dark2", "rainbow")


def apply_map_style() -> None:
    """Apply the tutorial Matplotlib theme for static maps and charts."""
    plt = require("matplotlib.pyplot")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#98A2B3",
            "axes.labelcolor": "#344054",
            "axes.titlecolor": "#101828",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "font.size": 11,
        }
    )


def style_axes(
    ax: Any,
    title: Optional[str] = None,
    *,
    equal: bool = True,
    grid: bool = True,
    axis_off: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
    crs_labels: bool = False,
) -> Any:
    """Style one map axes the way the vector tutorial draws its figures.

    Parameters
    ----------
    ax:
        Matplotlib axes.
    title:
        Left-aligned title. Omitted when ``None``.
    equal:
        Keep ``x`` and ``y`` at the same scale.
    grid:
        Draw a light reference grid.
    axis_off:
        Hide the frame, ticks, and labels. GeoPandas uses the same idea via
        ``Axes.set_axis_off``.
    xlabel, ylabel:
        Axis names. Ignored when ``crs_labels`` is true so GeoPandas can keep
        the unit names stored on the CRS.
    crs_labels:
        Leave the axis labels that ``GeoDataFrame.plot`` already set.
    """
    if axis_off:
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title, loc="left", fontsize=12, pad=8)
        return ax
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, pad=8)
    try:
        ax.ticklabel_format(useOffset=False, style="plain")
    except (AttributeError, ValueError):
        pass
    if grid:
        ax.grid(True, linewidth=0.4, color=GRID)
    if not crs_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return ax
