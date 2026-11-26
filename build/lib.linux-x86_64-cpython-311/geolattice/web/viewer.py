"""A self-contained Leaflet page for raster tile layers and GeoJSON layers."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional, Sequence

LEAFLET_VERSION = "1.9.4"


def leaflet_html(
    *,
    title: str = "GeoLattice web map",
    rasters: Sequence[Dict[str, Any]] = (),
    vectors: Sequence[Dict[str, Any]] = (),
    center: Optional[Sequence[float]] = None,
    zoom: Optional[int] = None,
    basemap: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    basemap_attribution: str = "&copy; OpenStreetMap contributors",
) -> str:
    """
    Build an HTML page with a Leaflet map.

    Args:
        title: Page title.
        rasters: Tile layers, each ``{"name", "url"}`` with an XYZ ``url`` template and optional
            ``bounds`` (west, south, east, north), ``minzoom``, ``maxzoom``, ``opacity``.
        vectors: GeoJSON layers, each ``{"name", "url"}`` where ``url`` returns a FeatureCollection
            (for example an OGC API Features ``items`` endpoint), with optional ``color``.
        center: ``(lon, lat)``; otherwise the map fits the first layer's bounds.
        zoom: Initial zoom with ``center``.
        basemap: XYZ URL of the background map.

    Leaflet is loaded from the unpkg CDN, so the page needs internet access.
    """
    config = {
        "rasters": list(rasters), "vectors": list(vectors), "center": list(center) if center else None,
        "zoom": zoom, "basemap": basemap, "attribution": basemap_attribution,
    }
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__VERSION__", LEAFLET_VERSION) \
        .replace("__CONFIG__", json.dumps(config))


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@__VERSION__/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@__VERSION__/dist/leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0}.title{position:absolute;z-index:1000;top:10px;left:50px;
background:#fff;padding:4px 10px;border-radius:4px;font:14px sans-serif;box-shadow:0 1px 4px #0004}</style>
</head>
<body>
<div id="map"></div><div class="title">__TITLE__</div>
<script>
const cfg = __CONFIG__;
const map = L.map("map");
const base = L.tileLayer(cfg.basemap, {maxZoom: 22, attribution: cfg.attribution}).addTo(map);
const overlays = {};
let fit = null;
for (const r of cfg.rasters) {
  const opts = {opacity: r.opacity ?? 0.9, maxZoom: 22, maxNativeZoom: r.maxzoom ?? 18, minZoom: 0};
  if (r.bounds) { opts.bounds = [[r.bounds[1], r.bounds[0]], [r.bounds[3], r.bounds[2]]]; fit = fit || opts.bounds; }
  overlays[r.name] = L.tileLayer(r.url, opts).addTo(map);
}
const pending = cfg.vectors.map(v => fetch(v.url).then(r => r.json()).then(fc => {
  const layer = L.geoJSON(fc, {style: {color: v.color || "#d7301f", weight: 1.5, fillOpacity: 0.1},
    pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 5, color: v.color || "#d7301f"}),
    onEachFeature: (f, l) => l.bindPopup("<pre>" + JSON.stringify(f.properties, null, 1) + "</pre>")}).addTo(map);
  overlays[v.name] = layer;
  if (!fit && layer.getBounds().isValid()) fit = layer.getBounds();
}));
Promise.allSettled(pending).then(() => {
  L.control.layers({"Basemap": base}, overlays).addTo(map);
  if (cfg.center) map.setView([cfg.center[1], cfg.center[0]], cfg.zoom ?? 10);
  else if (fit) map.fitBounds(fit); else map.setView([0, 0], 2);
});
</script>
</body>
</html>
"""

__all__ = ["leaflet_html"]
