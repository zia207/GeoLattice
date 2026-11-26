"""Lidar-specific plots: cross-sections along a line, vertical profiles, classification views."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.pointcloud import ASPRS_CLASSES, PointCloud

ASPRS_COLORS = {0: "#bdbdbd", 1: "#9e9e9e", 2: "#a0522d", 3: "#b8e186", 4: "#66bd63",
                5: "#1a9850", 6: "#d73027", 7: "#ff00ff", 9: "#4575b4", 10: "#525252",
                11: "#878787", 14: "#fdae61", 17: "#8073ac", 18: "#ff00ff"}


def cross_section(pc: PointCloud, start: tuple[float, float], end: tuple[float, float],
                  width: float = 2.0) -> PointCloud:
    """Points within ``width/2`` of the line ``start → end``, with a ``distance`` field (m
    along the line) and ``offset`` (signed distance from it)."""
    p = pc.to_cpu()
    a, b = np.asarray(start, float), np.asarray(end, float)
    u = b - a
    L = float(np.linalg.norm(u))
    if L == 0:
        raise ValueError("start and end are the same point")
    u /= L
    dx, dy = np.asarray(p.x, float) - a[0], np.asarray(p.y, float) - a[1]
    along = dx * u[0] + dy * u[1]
    off = -dx * u[1] + dy * u[0]
    m = (along >= 0) & (along <= L) & (np.abs(off) <= width / 2)
    sub = p.filter(np.flatnonzero(m))
    return sub.with_field("distance", along[m].astype("float32")).with_field(
        "offset", off[m].astype("float32"))


def plot_cross_section(pc: PointCloud, start=None, end=None, width: float = 2.0, *,
                       color_by: str = "classification", ax=None, s: float = 2.0,
                       cmap: str = "viridis", title: str | None = None):
    """Profile view (distance vs elevation). Pass a cloud from :func:`cross_section`, or a
    full cloud with ``start``/``end``. ``color_by="classification"`` uses ASPRS colours."""
    import matplotlib.pyplot as plt

    sec = cross_section(pc, start, end, width) if start is not None else pc
    if "distance" not in sec:
        raise ValueError("Pass start= and end=, or a cloud returned by cross_section")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    d, z = np.asarray(sec["distance"]), np.asarray(sec.z)
    if color_by == "classification" and "classification" in sec:
        c = np.asarray(sec["classification"])
        for k in np.unique(c):
            m = c == k
            ax.scatter(d[m], z[m], s=s, c=ASPRS_COLORS.get(int(k), "#000000"),
                       label=ASPRS_CLASSES.get(int(k), str(k)))
        ax.legend(markerscale=4, fontsize=8, loc="upper right", ncol=2)
    else:
        sc = ax.scatter(d, z, s=s, c=np.asarray(sec[color_by]), cmap=cmap)
        plt.colorbar(sc, ax=ax, label=color_by)
    ax.set_xlabel("Distance along section (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    return ax


def plot_height_profile(profile, *, column: str = "lad", ax=None, title: str | None = None):
    """Plot a :func:`height_profile` table (``lad``, ``fraction`` or ``gap_fraction``)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))
    mid = (profile.z_bottom + profile.z_top) / 2
    ax.plot(profile[column], mid, "-o", ms=3)
    ax.set_xlabel(column)
    ax.set_ylabel("Height above ground (m)")
    if title:
        ax.set_title(title)
    return ax


def plot_classification(pc: PointCloud, *, ax=None, s: float = 0.5, max_points: int = 300_000,
                        seed: int = 0) -> Any:
    """Plan view coloured by ASPRS class (subsampled to ``max_points``)."""
    import matplotlib.pyplot as plt

    p = pc.to_cpu()
    n = len(p)
    idx = np.random.default_rng(seed).choice(n, max_points, replace=False) if n > max_points \
        else np.arange(n)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    c = np.asarray(p["classification"])[idx]
    x, y = np.asarray(p.x)[idx], np.asarray(p.y)[idx]
    for k in np.unique(c):
        m = c == k
        ax.scatter(x[m], y[m], s=s, c=ASPRS_COLORS.get(int(k), "#000000"),
                   label=ASPRS_CLASSES.get(int(k), str(k)))
    ax.set_aspect("equal")
    ax.legend(markerscale=10, fontsize=8)
    return ax
