"""
Web and cloud GIS tools for GeoLattice (``geolattice.web``).

* **Cloud access** – :func:`to_vsi_path`, :func:`cloud_env`, :func:`cloud_options`,
  :func:`remote_info`: read files on S3, Google Cloud Storage, Azure or HTTP with
  range requests.
* **Cloud-Optimized GeoTIFF** – :func:`to_cog`, :func:`validate_cog`.
* **Cloud-native vectors** – :func:`write_geoparquet`, :func:`read_geoparquet`,
  :func:`geoparquet_info`, :func:`write_flatgeobuf`, :func:`write_pmtiles`
  (also available from :mod:`geolattice.vector`).
* **Web-map tiles** – :func:`lonlat_to_tile`, :func:`tile_bounds`,
  :func:`tiles_for_bounds`, :func:`render_tile`, :func:`make_xyz_tiles`,
  :func:`tilejson`, :func:`raster_web_info`, :func:`suggest_rescale`.
* **Serving** – :func:`create_app` (XYZ tiles + OGC API – Features + a Leaflet
  viewer, on Starlette), :func:`serve` (uvicorn) and :func:`start_server`
  (background, for notebooks); :func:`leaflet_html`;
  :func:`static_server` (a static file server with range requests, CORS and
  transfer counters, for testing COG, GeoParquet, FlatGeobuf and PMTiles locally).

Install the optional dependencies with ``pip install "geolattice[web]"``.

Example::

    import geolattice.web as web

    web.to_cog("ndvi.tif", "ndvi_cog.tif")
    print(web.validate_cog("ndvi_cog.tif"))
    web.write_geoparquet("counties.parquet", counties)
    app = web.create_app(rasters={"ndvi": "ndvi_cog.tif"}, vectors={"counties": "counties.parquet"})
    web.serve(app, port=8000)
"""

from geolattice.vector.write_cloud_native_vector_data import (
    geoparquet_info,
    read_geoparquet,
    write_flatgeobuf,
    write_geoparquet,
    write_pmtiles,
)
from geolattice.web.cloud import CLOUD_OPTIMIZED_OPTIONS, cloud_env, cloud_options, remote_info, to_vsi_path
from geolattice.web.cog import COGReport, to_cog, validate_cog
from geolattice.web.server import BackgroundServer, create_app, serve, start_server
from geolattice.web.static import StaticServer, SubprocessStaticServer, static_server
from geolattice.web.tiles import (
    Tile,
    lonlat_to_tile,
    make_xyz_tiles,
    raster_web_info,
    render_tile,
    suggest_rescale,
    tile_bounds,
    tile_resolution,
    tilejson,
    tiles_for_bounds,
    zoom_for_resolution,
)
from geolattice.web.viewer import leaflet_html

__all__ = [
    "CLOUD_OPTIMIZED_OPTIONS", "to_vsi_path", "cloud_options", "cloud_env", "remote_info",
    "COGReport", "to_cog", "validate_cog",
    "write_geoparquet", "read_geoparquet", "geoparquet_info", "write_flatgeobuf", "write_pmtiles",
    "Tile", "lonlat_to_tile", "tile_bounds", "tiles_for_bounds", "tile_resolution", "zoom_for_resolution",
    "raster_web_info", "tilejson", "suggest_rescale", "render_tile", "make_xyz_tiles",
    "create_app", "serve", "start_server", "BackgroundServer", "leaflet_html",
    "StaticServer", "SubprocessStaticServer", "static_server",
]
