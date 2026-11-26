"""
Spatial databases for GeoLattice.

``geolattice.spatial_bd.geolatticedb`` gives one interface to four spatial database
backends: PostGIS, SpatiaLite, GeoPackage and DuckDB spatial. It loads vector
and raster data, runs spatial SQL back into GeoDataFrames, and adds spatial
joins, nearest neighbours, buffers, indexes, query plans, PostGIS raster
(``db.raster``), pgRouting networks (``db.network``) and vector tiles
(``db.tiles``).

The module is also reachable as ``geolattice.geolatticedb``:

>>> from geolattice import geolatticedb as pdb
>>> db = pdb.connect("duckdb")
"""

from geolattice.spatial_bd import geolatticedb
from geolattice.spatial_bd.geolatticedb import (
    BACKENDS,
    SpatialDB,
    buffer,
    connect,
    create_index,
    execute,
    explain,
    load,
    nearest,
    query,
    spatial_join,
)

__all__ = [
    "geolatticedb",
    "SpatialDB",
    "connect",
    "load",
    "query",
    "execute",
    "spatial_join",
    "nearest",
    "buffer",
    "create_index",
    "explain",
    "BACKENDS",
]
