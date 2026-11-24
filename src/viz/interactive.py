"""Interactive Leaflet maps through Folium and GeoPandas.

``interactive_map`` uses ``GeoDataFrame.explore``, which is the Folium map
GeoPandas builds. Several layers can share one map, with a layer control so
each layer can be switched on and off. Marker and GeoJSON helpers reproject
to WGS84 (EPSG:4326), because Leaflet reads longitude and latitude.

https://github.com/python-visualization/folium
https://geopandas.org/en/stable/docs/user_guide/interactive_mapping.html
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from geolattice.viz._data import as_geodataframe, geometry_kind, require_features
from geolattice.viz._imports import VisualizationError, require
from geolattice.viz.style import EDGE, FACE

LayerSpec = Union[Any, Mapping[str, Any]]


def interactive_map(
    data: Any = None,
    *,
    column: Optional[str] = None,
    cmap: Optional[str] = None,
    color: Optional[str] = None,
    categorical: bool = False,
    scheme: Optional[str] = None,
    k: int = 5,
    tiles: str = "OpenStreetMap",
    tooltip: Any = True,
    popup: Any = False,
    legend: bool = True,
    name: Optional[str] = None,
    location: Optional[Sequence[float]] = None,
    zoom_start: Optional[int] = None,
    width: str = "100%",
    height: str = "500px",
    layer_control: bool = False,
    missing_kwds: Optional[Mapping[str, Any]] = None,
    style_kwds: Optional[Mapping[str, Any]] = None,
    **explore_kwds: Any,
) -> Any:
    """Open a Folium map, optionally drawing ``data`` on it.

    ``data`` may be one GeoDataFrame or a sequence of layers. A layer in the
    sequence is either a GeoDataFrame or a mapping with a ``data`` key and
    the same keywords as this function (``column``, ``color``, ``name``,
    ``scheme``, and so on). A sequence always adds a layer control.

    The returned object is a ``folium.Map``. In a notebook, displaying that
    object draws the Leaflet map.
    """
    folium = require("folium")
    if data is None:
        return folium.Map(
            location=location,
            zoom_start=2 if zoom_start is None else zoom_start,
            tiles=tiles,
            width=width,
            height=height,
        )
    if _is_layer_list(data):
        return _map_layers(
            data,
            tiles=tiles,
            location=location,
            zoom_start=zoom_start,
            width=width,
            height=height,
        )
    frame = require_features(as_geodataframe(data))
    map_kwds = dict(explore_kwds.pop("map_kwds", {}) or {})
    kwargs = dict(
        column=column,
        color=color if color is not None else (None if column else FACE),
        categorical=categorical,
        scheme=scheme,
        k=k,
        tiles=tiles,
        tooltip=tooltip,
        popup=popup,
        legend=legend if column else False,
        width=width,
        height=height,
        missing_kwds=dict(missing_kwds or {}),
        style_kwds=_style_for(frame, style_kwds),
        map_kwds=map_kwds,
        **explore_kwds,
    )
    if cmap is not None:
        kwargs["cmap"] = cmap
    if name is not None:
        kwargs["name"] = name
    # GeoPandas copies these onto the Folium map itself. Putting them in
    # map_kwds raises ValueError.
    if location is not None:
        kwargs["location"] = list(location)
    if zoom_start is not None:
        kwargs["zoom_start"] = zoom_start
    web = frame.explore(**kwargs)
    if layer_control:
        add_layer_control(web)
    return web


def add_vector_layer(
    web_map: Any,
    data: Any,
    *,
    name: Optional[str] = None,
    column: Optional[str] = None,
    cmap: Optional[str] = None,
    color: Optional[str] = None,
    categorical: bool = False,
    scheme: Optional[str] = None,
    k: int = 5,
    tooltip: Any = True,
    popup: Any = False,
    legend: Optional[bool] = None,
    style_kwds: Optional[Mapping[str, Any]] = None,
    **explore_kwds: Any,
) -> Any:
    """Add a GeoDataFrame to an existing Folium map."""
    frame = require_features(as_geodataframe(data))
    frame.explore(
        m=web_map,
        column=column,
        cmap=cmap,
        color=color if color is not None else (None if column else FACE),
        categorical=categorical,
        scheme=scheme,
        k=k,
        tooltip=tooltip,
        popup=popup,
        legend=bool(column) if legend is None else legend,
        name=name,
        style_kwds=_style_for(frame, style_kwds),
        **explore_kwds,
    )
    return web_map


def add_geojson(
    web_map: Any,
    data: Any,
    *,
    name: Optional[str] = None,
    color: str = FACE,
    weight: float = 1.5,
    fill_opacity: float = 0.5,
    tooltip_field: Optional[str] = None,
    smooth_factor: Optional[float] = None,
) -> Any:
    """Add GeoJSON, or a GeoDataFrame converted to GeoJSON, as one layer."""
    folium = require("folium")
    payload, fields = _geojson_payload(data)
    tooltip = None
    if tooltip_field:
        tooltip = folium.GeoJsonTooltip(fields=[tooltip_field])
    elif fields:
        tooltip = folium.GeoJsonTooltip(fields=fields[:8])
    kwargs = {
        "name": name,
        "tooltip": tooltip,
        "style_function": lambda _feature: {
            "color": color,
            "weight": weight,
            "fillColor": color,
            "fillOpacity": fill_opacity,
        },
    }
    if smooth_factor is not None:
        kwargs["smooth_factor"] = smooth_factor
    folium.GeoJson(payload, **kwargs).add_to(web_map)
    return web_map


def add_markers(
    web_map: Any,
    data: Any,
    *,
    name: Optional[str] = None,
    popup: Optional[str] = None,
    tooltip: Optional[str] = None,
    radius: float = 6,
    color: str = EDGE,
    fill_color: str = FACE,
) -> Any:
    """Add one circle marker for each point.

    ``popup`` and ``tooltip`` name attribute columns. Coordinates are
    transformed to EPSG:4326 when the layer has another CRS.
    """
    folium = require("folium")
    frame = _to_wgs84(require_features(as_geodataframe(data)))
    if geometry_kind(frame) != "point":
        raise VisualizationError("add_markers expects a point layer.")
    group = folium.FeatureGroup(name=name) if name else None
    target = group if group is not None else web_map
    for _, row in frame.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            continue
        folium.CircleMarker(
            location=(geom.y, geom.x),
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            weight=1,
            popup=None if popup is None else str(row[popup]),
            tooltip=None if tooltip is None else str(row[tooltip]),
        ).add_to(target)
    if group is not None:
        group.add_to(web_map)
    return web_map


def add_choropleth(
    web_map: Any,
    data: Any,
    column: str,
    *,
    name: Optional[str] = None,
    key_on: Optional[str] = None,
    cmap: str = "YlOrRd",
    fill_opacity: float = 0.7,
    line_opacity: float = 0.2,
    legend_name: Optional[str] = None,
    scheme: Optional[str] = None,
    k: int = 5,
    **kwargs: Any,
) -> Any:
    """Add a choropleth to an existing map.

    Without ``key_on``, the layer is drawn with ``GeoDataFrame.explore``.
    With ``key_on``, ``data`` is a table and ``geo_data`` in ``kwargs`` (or a
    GeoDataFrame) is passed to ``folium.Choropleth``. ``key_on`` looks like
    ``feature.properties.NAME`` or ``feature.id``.
    """
    if key_on is None:
        return add_vector_layer(
            web_map,
            data,
            name=name or column,
            column=column,
            cmap=cmap,
            scheme=scheme,
            k=k,
            legend=True,
            **kwargs,
        )
    folium = require("folium")
    geo_data = kwargs.pop("geo_data", data)
    table = kwargs.pop("table", data)
    id_column = kwargs.pop("id_column", None)
    if id_column is None:
        raise VisualizationError(
            "key_on choropleths need id_column, the table field that matches the GeoJSON key."
        )
    folium.Choropleth(
        geo_data=_geojson_payload(geo_data)[0],
        name=name or column,
        data=table,
        columns=[id_column, column],
        key_on=key_on,
        fill_color=_folium_color(cmap),
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        legend_name=legend_name or column,
        **kwargs,
    ).add_to(web_map)
    return web_map


def add_layer_control(web_map: Any, **kwargs: Any) -> Any:
    """Add a Folium layer control if the map does not already have one."""
    folium = require("folium")
    for child in web_map._children.values():
        if child.__class__.__name__ == "LayerControl":
            return web_map
    folium.LayerControl(**kwargs).add_to(web_map)
    return web_map


def _map_layers(
    layers: Sequence[LayerSpec],
    *,
    tiles: str,
    location: Optional[Sequence[float]],
    zoom_start: Optional[int],
    width: str,
    height: str,
) -> Any:
    folium = require("folium")
    web = folium.Map(
        location=location,
        zoom_start=2 if zoom_start is None else zoom_start,
        tiles=tiles,
        width=width,
        height=height,
    )
    for index, item in enumerate(layers):
        spec = dict(item) if isinstance(item, Mapping) and "data" in item else {"data": item}
        data = spec.pop("data")
        spec.setdefault("name", f"Layer {index + 1}")
        add_vector_layer(web, data, **spec)
    add_layer_control(web)
    return web


def _is_layer_list(data: Any) -> bool:
    if isinstance(data, (str, bytes)):
        return False
    if isinstance(data, Mapping):
        return False
    try:
        require("geopandas")
        require("pandas")
    except VisualizationError:
        return False
    gpd = require("geopandas")
    pd = require("pandas")
    if isinstance(data, (gpd.GeoDataFrame, gpd.GeoSeries, pd.DataFrame)):
        return False
    if not isinstance(data, Sequence) or len(data) == 0:
        return False
    first = data[0]
    if isinstance(first, Mapping):
        return "data" in first
    return hasattr(first, "geometry") or hasattr(first, "__geo_interface__")


def _style_for(frame: Any, style_kwds: Optional[Mapping[str, Any]]) -> dict:
    style = dict(style_kwds or {})
    kind = geometry_kind(frame)
    if kind == "polygon":
        style.setdefault("weight", 0.6)
        style.setdefault("color", EDGE)
        style.setdefault("fillOpacity", 0.65)
    elif kind == "line":
        style.setdefault("weight", 2)
        style.setdefault("color", FACE)
    return style


def _to_wgs84(frame: Any) -> Any:
    crs = frame.crs
    if crs is None:
        warnings.warn(
            "The layer has no CRS. Folium will read the coordinates as longitude and latitude.",
            UserWarning,
            stacklevel=3,
        )
        return frame
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return frame
    return frame.to_crs(4326)


def _geojson_payload(data: Any) -> Tuple[Any, list]:
    if hasattr(data, "to_crs") and hasattr(data, "geometry"):
        frame = _to_wgs84(require_features(as_geodataframe(data)))
        fields = [name for name in frame.columns if name != frame.geometry.name]
        return frame.to_json(), fields
    if hasattr(data, "__geo_interface__"):
        return data.__geo_interface__, []
    return data, []


def _folium_color(cmap: str) -> str:
    """Folium's ColorBrewer ramps are a subset of Matplotlib colormap names."""
    brewer = {
        "YlGn",
        "YlGnBu",
        "GnBu",
        "BuGn",
        "PuBuGn",
        "PuBu",
        "BuPu",
        "RdPu",
        "PuRd",
        "OrRd",
        "YlOrRd",
        "YlOrBr",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "Greys",
        "PuOr",
        "BrBG",
        "PRGn",
        "PiYG",
        "RdBu",
        "RdGy",
        "RdYlBu",
        "Spectral",
        "RdYlGn",
        "Accent",
        "Dark2",
        "Paired",
        "Pastel1",
        "Pastel2",
        "Set1",
        "Set2",
        "Set3",
    }
    if cmap in brewer:
        return cmap
    return "YlOrRd"
