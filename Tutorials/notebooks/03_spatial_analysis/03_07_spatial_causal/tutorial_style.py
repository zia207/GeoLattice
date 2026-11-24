"""Shared plotting style for the tutorial notebooks.

Colours follow a validated reference palette: categorical slots in fixed
order, a single-hue blue ramp for magnitudes, and a blue-red diverging map
with a neutral grey midpoint for signed effects.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TRUTH = "#0b0b0b"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a8a3", "#e34948", "#9b1c1c"])


def use():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100, "figure.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "font.size": 9.5,
    })


def county_map(gdf, column, ax=None, cmap=SEQ, center=None, title="", label="", vmin=None, vmax=None,
               categorical=False, colors=None, legend_labels=None, linewidth=0.0):
    """Choropleth of a county column with a colour bar (or a categorical legend)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6))
    vals = gdf[column] if isinstance(column, str) else column
    if categorical:
        from matplotlib.patches import Patch
        cats = sorted(np.unique(vals))
        colors = colors or SERIES
        for k, c in enumerate(cats):
            gdf[np.asarray(vals) == c].plot(ax=ax, color=colors[k], linewidth=linewidth)
        names = legend_labels or [str(c) for c in cats]
        ax.legend(handles=[Patch(color=colors[k], label=names[k]) for k in range(len(cats))],
                  loc="lower left", fontsize=8)
    else:
        norm = None
        if center is not None:
            # symmetric range about the centre keeps both arms of the colour bar comparable
            span = np.nanmax(np.abs(np.asarray(vals, float) - center))
            lo = center - span if vmin is None else vmin
            hi = center + span if vmax is None else vmax
            lo, hi = min(lo, center - 1e-9), max(hi, center + 1e-9)
            norm = TwoSlopeNorm(vcenter=center, vmin=lo, vmax=hi)
        gdf.assign(_v=np.asarray(vals)).plot(
            column="_v", ax=ax, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax,
            linewidth=linewidth, legend=True,
            legend_kwds={"shrink": 0.6, "label": label, "orientation": "vertical"})
    ax.set_axis_off()
    ax.set_title(title)
    return ax


def forest(ax, names, est, lo, hi, truth=None, title="", xlabel="estimate"):
    """Horizontal estimates with 95% intervals; truth drawn as a black tick."""
    y = np.arange(len(names))[::-1]
    est, lo, hi = map(np.asarray, (est, lo, hi))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", ms=6, capsize=3, color=BLUE, lw=1.6)
    if truth is not None:
        tr = np.broadcast_to(np.asarray(truth, float), est.shape)
        ax.scatter(tr, y, marker="|", s=260, color=TRUTH, zorder=4, label="truth")
        ax.legend(loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return ax
