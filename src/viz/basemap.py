"""Web-map tiles under a Matplotlib axes.

Tile names follow the providers shipped with contextily and xyzservices, the
same catalogs GeoPandas uses for ``plot(tiles=...)``. Reproject geographic
layers to Web Mercator (EPSG:3857) before calling this if the tiles look soft:
contextily otherwise warps the image into the data CRS.
"""

from __future__ import annotations

from typing import Any, Optional

from geolattice.viz._imports import VisualizationError, require


def tile_provider(source: Any) -> Any:
    """Resolve ``\"CartoDB.Positron\"`` to a contextily provider object."""
    if not isinstance(source, str):
        return source
    contextily = require("contextily")
    node = contextily.providers
    for part in source.split("."):
        if not hasattr(node, part):
            raise VisualizationError(
                f"Unknown tile provider {source!r}. "
                "Use a contextily provider path such as 'CartoDB.Positron' "
                "or 'OpenStreetMap.Mapnik'."
            )
        node = getattr(node, part)
    return node


def add_basemap(
    ax: Any,
    *,
    source: Any = "CartoDB.Positron",
    crs: Any = None,
    zoom: Any = "auto",
    attribution: Optional[str] = None,
    alpha: float = 1.0,
) -> Any:
    """Draw XYZ tiles behind geometries already plotted on ``ax``.

    Parameters
    ----------
    ax:
        Matplotlib axes whose limits are the map extent.
    source:
        A contextily provider, or a dotted name such as ``OpenStreetMap.Mapnik``.
    crs:
        CRS of the axes. Pass the layer CRS when the axes do not store one.
        Contextily reads a GeoPandas plot axes when ``crs`` is omitted.
    zoom:
        Tile zoom, or ``\"auto\"``.
    attribution:
        Override the provider credit line. Pass an empty string to hide it.
    alpha:
        Tile opacity from 0 to 1.
    """
    contextily = require("contextily")
    kwargs = {
        "ax": ax,
        "source": tile_provider(source),
        "zoom": zoom,
        "alpha": alpha,
    }
    if crs is not None:
        kwargs["crs"] = crs
    if attribution is not None:
        kwargs["attribution"] = attribution
    try:
        contextily.add_basemap(**kwargs)
    except Exception as exc:
        raise VisualizationError(
            "Could not add a basemap. The axes need a CRS and a finite extent, "
            "and the tile service has to be reachable."
        ) from exc
    return ax
