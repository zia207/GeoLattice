"""GeoLattice installation options and install hints for optional dependencies.

GeoLattice has six cumulative installation options; each includes everything in the one before it::

    pip install "geolattice[core]"            # 1. vector, raster, CRS, visualization
    pip install "geolattice[analysis]"        # 2. + spatial analysis
    pip install "geolattice[statistics]"      # 3. + spatial statistics and spatial causal inference
    pip install "geolattice[geostats]"        # 4. + geostatistics
    pip install "geolattice[remote-sensing]"  # 5. + remote sensing
    pip install "geolattice[all]"             # 6. everything else (web, databases, deep learning, distributed, ...)

They are built from smaller extras such as ``rsense-lidar`` or ``geostatistics-plot``.
:func:`install_hint` turns one of those extras into a message that names both the extra and
the smallest installation option that contains it, for use in ``ImportError`` messages.

    >>> install_hint("geostatistics-plot")
    'pip install "geolattice[geostatistics-plot]" (included in installation option 4, "geolattice[geostats]")'
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Optional, Set

#: The installation options in order, as (extra, short description).
INSTALL_OPTIONS = (
    ("core", "vector, raster, CRS, visualization"),
    ("analysis", "core + spatial analysis"),
    ("statistics", "core + spatial analysis + spatial statistics"),
    ("geostats", "core + spatial analysis + spatial statistics + geostatistics"),
    ("remote-sensing", "core + spatial analysis + spatial statistics + geostatistics + remote sensing"),
    ("all", "all"),
)

# Used when the installed package metadata cannot be read (for example a source checkout).
# Keep in step with [project.optional-dependencies] in pyproject.toml.
_FALLBACK: Dict[str, Set[str]] = {
    "core": {"io", "viz"},
    "analysis": {"core", "multidim", "network", "segregation", "segregation-dasymetric", "whitebox"},
    "statistics": {"analysis", "spatial-statistics", "spatial-statistics-viz", "spatialcausal", "spatialcausal-viz"},
    "geostats": {"statistics", "geostatistics", "geostatistics-plot"},
    "remote-sensing": {"geostats", "rsense", "rsense-access", "rsense-sentinel3", "rsense-sentinel5p", "rsense-sar",
                       "rsense-lidar", "rsense-hyperspectral", "rsense-uav", "rsense-classify", "rsense-viz"},
    "all": {"remote-sensing", "spatial-statistics-all", "spatialcausal-all", "geostatistics-jax",
            "geostatistics-pytorch", "geostatistics-gam", "geostatistics-optuna", "rsense-all", "web-cloud",
            "spatial-db-all", "distributed", "distributed-ml"},
    "spatial-statistics-all": {"spatial-statistics", "spatial-statistics-viz", "spatial-statistics-deep",
                               "spatial-statistics-boosting", "spatial-statistics-bayes", "spatial-statistics-gam",
                               "spatial-statistics-libpysal", "spatial-statistics-network"},
    "spatialcausal-all": {"spatialcausal", "spatialcausal-viz", "spatialcausal-causalml"},
    "rsense-all": {"rsense", "rsense-access", "rsense-sentinel3", "rsense-sentinel5p", "rsense-sar", "rsense-insar",
                   "rsense-lidar", "rsense-hyperspectral", "rsense-uav", "rsense-classify", "rsense-dl", "rsense-edge",
                   "rsense-xai", "rsense-gee", "rsense-viz", "rsense-scale", "rsense-apps"},
    "web-cloud": {"web"},
    "spatial-db-all": {"spatial-db", "spatial-db-postgis"},
    "distributed-ml": {"distributed"},
    "distributed-gpu": {"distributed"},
    "distributed-hpc": {"distributed"},
    "distributed-spark": {"distributed"},
}


@lru_cache(maxsize=1)
def _extra_includes() -> Dict[str, Set[str]]:
    """Which other extras each extra pulls in, read from the installed package metadata."""
    try:
        from importlib.metadata import PackageNotFoundError, requires

        reqs = requires("geolattice") or []
    except Exception:  # PackageNotFoundError or a broken installation
        return _FALLBACK
    graph: Dict[str, Set[str]] = {}
    for req in reqs:
        m = re.match(r'\s*geolattice\[([^\]]+)\].*extra\s*==\s*["\']([^"\']+)["\']', req)
        if m:
            graph.setdefault(m.group(2), set()).update(x.strip() for x in m.group(1).split(","))
    return graph or _FALLBACK


def _closure(extra: str) -> Set[str]:
    graph, seen, todo = _extra_includes(), set(), [extra]
    while todo:
        e = todo.pop()
        if e not in seen:
            seen.add(e)
            todo.extend(graph.get(e, ()))
    return seen


def option_for(extra: str) -> Optional[str]:
    """The smallest installation option that installs ``extra``, or None if no option does."""
    for name, _ in INSTALL_OPTIONS:
        if extra in _closure(name):
            return name
    return None


def install_hint(extra: str) -> str:
    """``pip install`` text for one extra, naming the installation option that includes it."""
    hint = f'pip install "geolattice[{extra}]"'
    option = option_for(extra)
    if option and option != extra:
        number = [n for n, _ in INSTALL_OPTIONS].index(option) + 1
        hint += f' (included in installation option {number}, "geolattice[{option}]")'
    elif option is None:
        hint += ' (not part of "geolattice[all]"; install it by name)'
    return hint


__all__ = ["INSTALL_OPTIONS", "install_hint", "option_for"]
