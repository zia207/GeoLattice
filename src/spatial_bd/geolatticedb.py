"""
geolatticedb: spatial database tools for GeoLattice.

One interface for four spatial database backends:

    * ``postgis``     PostgreSQL + PostGIS (+ postgis_raster, pgRouting)
    * ``spatialite``  SQLite + mod_spatialite
    * ``geopackage``  OGC GeoPackage (SQLite file, read through SpatiaLite)
    * ``duckdb``      DuckDB + spatial extension

Quick start
-----------
>>> from geolattice import geolatticedb as pdb
>>> db = pdb.connect("postgis", host="localhost", dbname="gis",
...                  user="postgres", password="secret")
>>> db.load("counties.shp", "counties")                 # SRID detected, GiST index built
>>> gdf = db.query("SELECT * FROM counties WHERE statefp = '36'")
>>> joined = db.spatial_join("wells", "counties", predicate="within")
>>> near = db.nearest("wells", "hospitals", k=3)
>>> buf = db.buffer("hospitals", 5000, geography=True)
>>> print(db.explain("SELECT * FROM counties WHERE ST_Area(geom) > 1", analyze=True))
>>> zs = db.raster.zonal_stats("dem", "counties")        # PostGIS raster
>>> db.network.prepare("roads")
>>> path = db.network.shortest_path("roads", source=(-76.50, 42.44), target=(-76.48, 42.45))
>>> iso = db.network.isochrone("roads", source=(-76.50, 42.44), costs=[1000, 2000])
>>> db.tiles.serve(["counties"], port=8080)              # http://127.0.0.1:8080/

Every method is also available as a module-level function that takes the
connection as its first argument, e.g. ``pdb.query(db, sql)``.

Optional dependencies
---------------------
postgis    : sqlalchemy, psycopg (v3), geoalchemy2; raster2pgsql on PATH for rasters
spatialite : the mod_spatialite library (apt: libsqlite3-mod-spatialite,
             brew: libspatialite, conda: libspatialite)
geopackage : pyogrio (writing) + mod_spatialite (spatial SQL)
duckdb     : duckdb (the spatial extension is installed on first use)
raster     : rasterio (SRID detection)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

__all__ = [
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

BACKENDS = ("postgis", "spatialite", "geopackage", "duckdb")

_BACKEND_ALIASES = {
    "postgis": "postgis",
    "postgres": "postgis",
    "postgresql": "postgis",
    "pg": "postgis",
    "spatialite": "spatialite",
    "sqlite": "spatialite",
    "geopackage": "geopackage",
    "gpkg": "geopackage",
    "duckdb": "duckdb",
    "duck": "duckdb",
}

RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".asc", ".nc", ".hdf", ".dem"}

_PREDICATES = {
    "intersects": "ST_Intersects",
    "within": "ST_Within",
    "contains": "ST_Contains",
    "touches": "ST_Touches",
    "crosses": "ST_Crosses",
    "overlaps": "ST_Overlaps",
    "covers": "ST_Covers",
    "covered_by": "ST_CoveredBy",
    "equals": "ST_Equals",
    "disjoint": "ST_Disjoint",
}

_META_TABLE = "geolattice_geometry_columns"  # DuckDB only: DuckDB has no SRID catalog

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _q(name: str) -> str:
    """Quote an identifier, keeping schema qualification (``schema.table``)."""
    parts = str(name).split(".")
    out = []
    for p in parts:
        p = p.strip('"')
        out.append('"' + p.replace('"', '""') + '"')
    return ".".join(out)


def _split(name: str, default_schema: str | None = None) -> tuple[str | None, str]:
    parts = str(name).split(".")
    if len(parts) == 2:
        return parts[0].strip('"'), parts[1].strip('"')
    return default_schema, parts[0].strip('"')


def _lit(value) -> str:
    """SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _epsg(crs) -> int:
    if crs is None:
        return 0
    try:
        code = crs.to_epsg()
    except Exception:
        code = None
    if code is None:
        warnings.warn(
            f"CRS {crs!r} has no EPSG code; storing SRID 0. Pass srid= to set one.",
            stacklevel=3,
        )
        return 0
    return int(code)


def _is_raster_path(path) -> bool:
    return Path(str(path)).suffix.lower() in RASTER_EXTENSIONS


def _to_geodataframe(df: pd.DataFrame, geom_cols: Sequence[str], active: str | None, crs):
    if not geom_cols:
        return df
    for c in geom_cols:
        vals = df[c].to_numpy(dtype=object)
        mask = pd.isna(vals)
        out = np.empty(len(vals), dtype=object)
        out[:] = None
        if (~mask).any():
            out[~mask] = shapely.from_wkb(list(vals[~mask]))
        df[c] = gpd.GeoSeries(out, index=df.index)
    active = active if active in geom_cols else geom_cols[0]
    gdf = gpd.GeoDataFrame(df, geometry=active)
    if crs:
        gdf = gdf.set_crs(crs, allow_override=True)
    return gdf


def _srid_from_wkb_values(values) -> int:
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            return int(shapely.get_srid(shapely.from_wkb(v)))
        except Exception:
            return 0
    return 0


def _find_spatialite() -> str:
    env = os.environ.get("SPATIALITE_LIBRARY_PATH")
    candidates = [env] if env else []
    candidates += [
        "mod_spatialite",
        "mod_spatialite.so",
        "mod_spatialite.dylib",
        "/opt/homebrew/lib/mod_spatialite.dylib",
        "/usr/local/lib/mod_spatialite.dylib",
        "/usr/lib/x86_64-linux-gnu/mod_spatialite.so",
    ]
    return [c for c in candidates if c]


_SPATIALITE_MISSING = (
    "Could not load mod_spatialite. Install it (apt install libsqlite3-mod-spatialite, "
    "brew install libspatialite, or conda install libspatialite) or set SPATIALITE_LIBRARY_PATH."
)


def _load_spatialite(con, libs):
    """Load mod_spatialite into an open SQLite connection.

    Some CPython builds leave out ``enable_load_extension`` even when the
    linked libsqlite3 can load extensions. Those builds call the C API directly.
    """
    if hasattr(con, "enable_load_extension"):
        con.enable_load_extension(True)
        last = None
        for lib in libs:
            try:
                con.load_extension(lib)
                return
            except sqlite3.OperationalError as e:
                last = e
        raise RuntimeError(_SPATIALITE_MISSING) from last

    import ctypes
    import ctypes.util

    soname = ctypes.util.find_library("sqlite3") or "libsqlite3.so.0"
    lib = ctypes.CDLL(soname)
    try:
        enable = lib.sqlite3_enable_load_extension
        load = lib.sqlite3_load_extension
        free = lib.sqlite3_free
    except AttributeError as e:
        raise RuntimeError(_SPATIALITE_MISSING) from e
    enable.argtypes = [ctypes.c_void_p, ctypes.c_int]
    enable.restype = ctypes.c_int
    load.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)
    ]
    load.restype = ctypes.c_int
    free.argtypes = [ctypes.c_void_p]

    class _Head(ctypes.Structure):
        _fields_ = [("ob_refcnt", ctypes.c_ssize_t), ("ob_type", ctypes.c_void_p)]

    class _Conn(ctypes.Structure):
        _fields_ = [("head", _Head), ("db", ctypes.c_void_p)]

    db = _Conn.from_address(id(con)).db
    if not db or enable(db, 1) != 0:
        raise sqlite3.OperationalError("sqlite3_enable_load_extension failed")
    last = None
    for name in libs:
        err = ctypes.c_char_p()
        rc = load(db, os.fsencode(name), None, ctypes.byref(err))
        if rc == 0:
            return
        msg = err.value.decode() if err.value else f"could not load {name}"
        if err.value:
            free(err)
        last = sqlite3.OperationalError(msg)
    raise RuntimeError(_SPATIALITE_MISSING) from last


# ---------------------------------------------------------------------------
# main connection class
# ---------------------------------------------------------------------------
class SpatialDB:
    """A connection to a spatial database. Create it with :func:`connect`."""

    def __init__(self, backend: str, *, path=None, dsn=None, engine=None,
                 schema: str | None = None, echo: bool = False, **kwargs):
        backend = _BACKEND_ALIASES.get(str(backend).lower())
        if backend is None:
            raise ValueError(f"Unknown backend. Choose one of {BACKENDS}.")
        self.backend = backend
        self.path = str(path) if path is not None else None
        self.echo = echo
        self.schema = schema or ("public" if backend == "postgis" else None)
        self._con = None
        self._engine = None

        if backend == "postgis":
            self._open_postgis(dsn=dsn, engine=engine, **kwargs)
        elif backend in ("spatialite", "geopackage"):
            if self.path is None:
                raise ValueError(f"{backend} needs path='file.{'gpkg' if backend == 'geopackage' else 'sqlite'}'")
            self._open_sqlite()
        else:
            self._open_duckdb(**kwargs)

        self.raster = _Raster(self)
        self.network = _Network(self)
        self.tiles = _Tiles(self)

    # ------------------------------------------------------------- opening
    def _open_postgis(self, dsn=None, engine=None, **kw):
        try:
            import sqlalchemy as sa
        except ImportError as e:  # pragma: no cover
            raise ImportError("PostGIS backend needs: pip install sqlalchemy psycopg[binary] geoalchemy2") from e
        if engine is not None:
            self._engine = engine
        else:
            if dsn is None:
                url = sa.engine.URL.create(
                    "postgresql+psycopg",
                    username=kw.pop("user", None) or kw.pop("username", None) or os.environ.get("PGUSER"),
                    password=kw.pop("password", None) or os.environ.get("PGPASSWORD"),
                    host=kw.pop("host", None) or os.environ.get("PGHOST", "localhost"),
                    port=int(kw.pop("port", None) or os.environ.get("PGPORT", 5432)),
                    database=kw.pop("dbname", None) or kw.pop("database", None) or os.environ.get("PGDATABASE", "postgres"),
                )
            else:
                url = dsn if "://" in dsn else None
                if url is None:  # libpq keyword string "host=... dbname=..."
                    import psycopg
                    self._engine = sa.create_engine("postgresql+psycopg://", creator=lambda: psycopg.connect(dsn))
                url = url.replace("postgresql://", "postgresql+psycopg://", 1) if url else None
            if self._engine is None:
                self._engine = sa.create_engine(url, pool_pre_ping=True, **kw)
        self.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        oids = self._fetch("SELECT typname, oid FROM pg_type WHERE typname IN ('geometry','geography','raster')")
        self._pg_oids = {int(oid): name for name, oid in oids}

    def _open_sqlite(self):
        con = sqlite3.connect(self.path, check_same_thread=False)
        _load_spatialite(con, _find_spatialite())
        self._con = con
        if self.backend == "geopackage":
            if not self._fetch("SELECT name FROM sqlite_master WHERE name='gpkg_contents'"):
                self._fetch("SELECT gpkgCreateBaseTables()")
                con.execute("PRAGMA user_version = 10300")
                con.commit()
            self._fetch("SELECT EnableGpkgAmphibiousMode()")
        else:
            if not self._fetch("SELECT name FROM sqlite_master WHERE name='spatial_ref_sys'"):
                self._fetch("SELECT InitSpatialMetadata(1)")
                con.commit()

    def _open_duckdb(self, **kw):
        try:
            import duckdb
        except ImportError as e:  # pragma: no cover
            raise ImportError("DuckDB backend needs: pip install duckdb") from e
        self._con = duckdb.connect(self.path or ":memory:", **kw)
        try:
            self._con.load_extension("spatial")
        except Exception:
            self._con.install_extension("spatial")
            self._con.load_extension("spatial")
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_META_TABLE} "
            "(table_name VARCHAR, geom_col VARCHAR, srid INTEGER, PRIMARY KEY (table_name, geom_col))"
        )

    # ------------------------------------------------------- housekeeping
    def close(self):
        if self._con is not None:
            self._con.close()
            self._con = None
        if self._engine is not None:
            self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        where = self.path if self.path else str(self._engine.url.render_as_string(hide_password=True))
        return f"<SpatialDB backend={self.backend!r} at {where!r}>"

    # ------------------------------------------------------ raw execution
    def _log(self, sql):
        if self.echo:
            print(sql)

    def _fetch(self, sql: str, params=None, with_desc=False):
        self._log(sql)
        if self.backend == "postgis":
            raw = self._engine.raw_connection()
            try:
                cur = raw.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
                desc = cur.description
                raw.commit()
            except Exception:
                raw.rollback()
                raise
            finally:
                raw.close()
        else:
            cur = self._con.execute(sql, params or [])
            desc = cur.description
            rows = cur.fetchall() if desc else []
        return (rows, desc) if with_desc else rows

    def execute(self, sql: str, params=None):
        """Run SQL that returns no geometry (DDL, UPDATE, ...). Returns rows if any."""
        rows = self._fetch(sql, params)
        if self.backend in ("spatialite", "geopackage"):
            self._con.commit()
        return rows

    # ------------------------------------------------------- introspection
    def tables(self) -> list[str]:
        """List user tables."""
        if self.backend == "postgis":
            rows = self._fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s "
                "AND table_type='BASE TABLE' AND table_name <> 'spatial_ref_sys' ORDER BY 1",
                (self.schema,),
            )
        elif self.backend == "duckdb":
            rows = self._fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_name <> ? ORDER BY 1", [_META_TABLE]
            )
        elif self.backend == "geopackage":
            rows = self._fetch("SELECT table_name FROM gpkg_contents ORDER BY 1")
        else:
            rows = self._fetch(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT LIKE 'idx_%' AND name NOT LIKE 'spatial%' AND name NOT LIKE 'geometry_columns%' "
                "AND name NOT LIKE 'views_geometry%' AND name NOT LIKE 'virts_geometry%' "
                "AND name NOT IN ('ElementaryGeometries','KNN','KNN2','SpatialIndex','data_licenses','sql_statements_log') "
                "ORDER BY 1"
            )
        return [r[0] for r in rows]

    def columns(self, table: str) -> list[str]:
        """Column names of a table, in order."""
        schema, name = _split(table, self.schema)
        if self.backend == "postgis":
            rows = self._fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
                "ORDER BY ordinal_position",
                (schema, name),
            )
        elif self.backend == "duckdb":
            rows = self._fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name=? ORDER BY ordinal_position",
                [name],
            )
        else:
            rows = [(r[1],) for r in self._fetch(f"PRAGMA table_info({_q(name)})")]
        if not rows:
            raise ValueError(f"Table {table!r} not found or has no columns.")
        return [r[0] for r in rows]

    def geometry_column(self, table: str) -> str:
        """Name of the (first) geometry column registered for a table."""
        schema, name = _split(table, self.schema)
        if self.backend == "postgis":
            rows = self._fetch(
                "SELECT f_geometry_column FROM geometry_columns WHERE f_table_schema=%s AND f_table_name=%s "
                "UNION ALL SELECT f_geography_column FROM geography_columns WHERE f_table_schema=%s AND f_table_name=%s",
                (schema, name, schema, name),
            )
        elif self.backend == "duckdb":
            rows = self._fetch(f"SELECT geom_col FROM {_META_TABLE} WHERE table_name=?", [name])
            if not rows:
                rows = self._fetch(
                    "SELECT column_name FROM information_schema.columns WHERE table_name=? "
                    "AND data_type LIKE 'GEOMETRY%'",
                    [name],
                )
        elif self.backend == "geopackage":
            rows = self._fetch("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", [name])
        else:
            rows = self._fetch(
                "SELECT f_geometry_column FROM geometry_columns WHERE lower(f_table_name)=lower(?)", [name]
            )
        if not rows:
            raise ValueError(f"No geometry column registered for {table!r}.")
        return rows[0][0]

    def srid(self, table: str, geom_col: str | None = None) -> int:
        """SRID of a table's geometry column."""
        schema, name = _split(table, self.schema)
        geom_col = geom_col or self.geometry_column(table)
        if self.backend == "postgis":
            rows = self._fetch("SELECT Find_SRID(%s, %s, %s)", (schema, name, geom_col))
        elif self.backend == "duckdb":
            rows = self._fetch(f"SELECT srid FROM {_META_TABLE} WHERE table_name=? AND geom_col=?", [name, geom_col])
        elif self.backend == "geopackage":
            rows = self._fetch(
                "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name=? AND column_name=?", [name, geom_col]
            )
        else:
            rows = self._fetch(
                "SELECT srid FROM geometry_columns WHERE lower(f_table_name)=lower(?) AND lower(f_geometry_column)=lower(?)",
                [name, geom_col],
            )
        return int(rows[0][0]) if rows and rows[0][0] is not None else 0

    # ---------------------------------------------------------------- load
    def load(self, data, table: str, *, srid: int | None = None, geom_col: str = "geom",
             if_exists: str = "replace", index: bool = True, layer=None,
             tile_size: int = 256) -> str:
        """Load vector or raster data into ``table``.

        Parameters
        ----------
        data : GeoDataFrame, or path to a vector file (shp, gpkg, geojson, parquet, ...)
               or raster file (tif, vrt, ...; PostGIS only).
        table : destination table name (``schema.table`` allowed for PostGIS).
        srid : target SRID. If the data has a CRS it is reprojected to this SRID;
               if it has none, this SRID is assigned. Default: detected from the data.
        geom_col : name of the geometry column to create (vector data).
        if_exists : 'replace', 'append' or 'fail'.
        index : build a spatial index after loading.
        layer : layer name for multi-layer vector files.
        tile_size : raster tile size in pixels (PostGIS raster).

        Returns the table name.
        """
        if if_exists not in ("replace", "append", "fail"):
            raise ValueError("if_exists must be 'replace', 'append' or 'fail'")
        if isinstance(data, (str, Path)):
            if _is_raster_path(data):
                return self.raster.load(data, table, srid=srid, if_exists=if_exists,
                                        index=index, tile_size=tile_size)
            if str(data).lower().endswith((".parquet", ".geoparquet")):
                data = gpd.read_parquet(data)
            else:
                data = gpd.read_file(data, layer=layer)
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError("data must be a GeoDataFrame or a path to a vector/raster file")

        gdf = data.copy()
        if gdf.geometry.name != geom_col:
            if geom_col in gdf.columns:
                gdf = gdf.drop(columns=geom_col)
            gdf = gdf.rename_geometry(geom_col)
        if srid is not None:
            if gdf.crs is None:
                gdf = gdf.set_crs(srid)
            elif _epsg(gdf.crs) != srid:
                gdf = gdf.to_crs(srid)
        srid = _epsg(gdf.crs)

        exists = table in self.tables() or _split(table)[1] in self.tables()
        if exists and if_exists == "fail":
            raise ValueError(f"Table {table!r} already exists.")

        getattr(self, f"_load_{self.backend}")(gdf, table, srid, geom_col, if_exists, exists)
        if index:
            self.create_index(table, geom_col)
        return table

    def _load_postgis(self, gdf, table, srid, geom_col, if_exists, exists):
        schema, name = _split(table, self.schema)
        gdf.to_postgis(name, self._engine, schema=schema, if_exists=if_exists, index=False)
        if srid and self.srid(table, geom_col) != srid:  # e.g. unknown-EPSG CRS
            self.execute(f"SELECT UpdateGeometrySRID({_lit(schema)}, {_lit(name)}, {_lit(geom_col)}, {srid})")

    def _load_spatialite(self, gdf, table, srid, geom_col, if_exists, exists):
        con = self._con
        dims = "XYZ" if gdf.geometry.has_z.any() else "XY"
        stage = pd.DataFrame(gdf.drop(columns=geom_col))
        stage["__geolattice_wkb"] = shapely.to_wkb(gdf.geometry.values, output_dimension=3 if dims == "XYZ" else 2)
        stage.to_sql("__geolattice_stage", con, if_exists="replace", index=False)
        attrs = [c for c in stage.columns if c != "__geolattice_wkb"]
        if exists and if_exists == "replace":
            self._fetch("SELECT DropTable(NULL, ?, 1)", [table])
            self._fetch(f"DROP TABLE IF EXISTS {_q(table)}")
        if not exists or if_exists == "replace":
            if attrs:
                cols = ", ".join(_q(c) for c in attrs)
                self._fetch(f"CREATE TABLE {_q(table)} AS SELECT {cols} FROM __geolattice_stage WHERE 0")
            else:
                self._fetch(f"CREATE TABLE {_q(table)} (fid INTEGER PRIMARY KEY)")
            res = self._fetch("SELECT AddGeometryColumn(?, ?, ?, 'GEOMETRY', ?)", [table, geom_col, srid, dims])
            if not res or res[0][0] != 1:
                raise RuntimeError(f"AddGeometryColumn failed for {table}.{geom_col} (SRID {srid}).")
        col_list = ", ".join([_q(c) for c in attrs] + [_q(geom_col)])
        sel_list = ", ".join([_q(c) for c in attrs] + [f"GeomFromWKB(__geolattice_wkb, {srid})"])
        self._fetch(f"INSERT INTO {_q(table)} ({col_list}) SELECT {sel_list} FROM __geolattice_stage")
        self._fetch("DROP TABLE __geolattice_stage")
        con.commit()

    def _load_geopackage(self, gdf, table, srid, geom_col, if_exists, exists):
        # Written with SpatiaLite's GPKG functions (no GDAL), so the file is never
        # opened by two SQLite libraries at once.
        con = self._con
        name = _split(table)[1]
        dims_z = 1 if gdf.geometry.has_z.any() else 0
        if not self._fetch("SELECT 1 FROM gpkg_spatial_ref_sys WHERE srs_id=?", [srid]):
            self._fetch("SELECT gpkgInsertEpsgSRID(?)", [srid])
        stage = pd.DataFrame(gdf.drop(columns=geom_col))
        stage["__geolattice_wkb"] = shapely.to_wkb(gdf.geometry.values, output_dimension=3 if dims_z else 2)
        stage.to_sql("__geolattice_stage", con, if_exists="replace", index=False)
        attrs = [c for c in stage.columns if c not in ("__geolattice_wkb", "fid")]
        if exists and if_exists == "replace":
            self._drop_gpkg_table(name)
        if not exists or if_exists == "replace":
            self._fetch(f"CREATE TABLE {_q(name)} (fid INTEGER PRIMARY KEY AUTOINCREMENT)")
            if attrs:
                types = {r[1]: r[2] for r in self._fetch("PRAGMA table_info(__geolattice_stage)")}
                for c in attrs:
                    self._fetch(f"ALTER TABLE {_q(name)} ADD COLUMN {_q(c)} {types.get(c) or ''}")
            self._fetch(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id, last_change) "
                "VALUES (?, 'features', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                [name, name, srid],
            )
            self._fetch("SELECT gpkgAddGeometryColumn(?, ?, 'GEOMETRY', ?, 0, ?)", [name, geom_col, dims_z, srid])
            self._fetch("SELECT gpkgAddGeometryTriggers(?, ?)", [name, geom_col])
        col_list = ", ".join([_q(c) for c in attrs] + [_q(geom_col)])
        sel_list = ", ".join([_q(c) for c in attrs] + [f"AsGPB(GeomFromWKB(__geolattice_wkb, {srid}))"])
        self._fetch(f"INSERT INTO {_q(name)} ({col_list}) SELECT {sel_list} FROM __geolattice_stage")
        self._fetch("DROP TABLE __geolattice_stage")
        self._fetch(
            "UPDATE gpkg_contents SET (min_x, min_y, max_x, max_y) = "
            f"(SELECT MbrMinX(e), MbrMinY(e), MbrMaxX(e), MbrMaxY(e) FROM (SELECT Extent({_q(geom_col)}) AS e "
            f"FROM {_q(name)})) WHERE table_name = ?",
            [name],
        )
        con.commit()

    def _drop_gpkg_table(self, name):
        for (gc,) in self._fetch("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", [name]):
            self._fetch(f"DROP TABLE IF EXISTS {_q(f'rtree_{name}_{gc}')}")
        for t in ("gpkg_geometry_columns", "gpkg_extensions", "gpkg_contents"):
            self._fetch(f"DELETE FROM {t} WHERE table_name=?", [name])
        self._fetch(f"DROP TABLE IF EXISTS {_q(name)}")

    def _load_duckdb(self, gdf, table, srid, geom_col, if_exists, exists):
        stage = pd.DataFrame(gdf.drop(columns=geom_col))
        stage["__geolattice_wkb"] = shapely.to_wkb(gdf.geometry.values)
        self._con.register("__geolattice_stage", stage)
        try:
            sel = f"SELECT * EXCLUDE (__geolattice_wkb), ST_GeomFromWKB(__geolattice_wkb) AS {_q(geom_col)} FROM __geolattice_stage"
            if exists and if_exists == "append":
                self._con.execute(f"INSERT INTO {_q(table)} BY NAME {sel}")
            else:
                self._con.execute(f"CREATE OR REPLACE TABLE {_q(table)} AS {sel}")
        finally:
            self._con.unregister("__geolattice_stage")
        self._con.execute(f"INSERT OR REPLACE INTO {_META_TABLE} VALUES (?, ?, ?)", [table, geom_col, srid])

    # --------------------------------------------------------------- query
    def query(self, sql: str, params=None, *, geom_col: str | None = None, crs=None):
        """Run SQL and return a GeoDataFrame (or a DataFrame if no geometry is returned).

        Geometry columns are detected automatically. ``geom_col`` picks the active one
        when several are returned. ``crs`` overrides the CRS; otherwise it is taken from
        the geometries' SRID (PostGIS, SpatiaLite, GeoPackage) or from the source
        tables (DuckDB, when all registered tables share one SRID).
        Placeholders: ``%s`` for PostGIS, ``?`` for the others.
        """
        sql = sql.strip().rstrip(";")
        return getattr(self, f"_query_{'sqlite' if self.backend in ('spatialite', 'geopackage') else self.backend}")(
            sql, params, geom_col, crs
        )

    def _query_postgis(self, sql, params, geom_col, crs):
        rows, desc = self._fetch(sql, params, with_desc=True)
        names = [d.name for d in desc]
        df = pd.DataFrame.from_records(rows, columns=names)
        geom_cols = [d.name for d in desc if self._pg_oids.get(d.type_code) in ("geometry", "geography")]
        if geom_cols and crs is None:
            active = geom_col if geom_col in geom_cols else geom_cols[0]
            srid = _srid_from_wkb_values(df[active].tolist())
            if not srid and self._pg_oids.get(dict((d.name, d.type_code) for d in desc)[active]) == "geography":
                srid = 4326
            crs = srid or None
        return _to_geodataframe(df, geom_cols, geom_col, crs)

    def _query_sqlite(self, sql, params, geom_col, crs):
        probe, desc = self._fetch(f"SELECT * FROM ({sql}) AS __q LIMIT 50", params, with_desc=True)
        names = [d[0] for d in desc]
        geom_cols = []
        for i, n in enumerate(names):
            sample = next((r[i] for r in probe if isinstance(r[i], (bytes, memoryview))), None)
            if sample is not None and self._fetch("SELECT GeometryType(?)", [bytes(sample)])[0][0]:
                geom_cols.append(n)
        if not geom_cols:
            rows, desc = self._fetch(sql, params, with_desc=True)
            return pd.DataFrame.from_records(rows, columns=[d[0] for d in desc])
        sel = [f"AsBinary({_q(n)}) AS {_q(n)}" if n in geom_cols else _q(n) for n in names]
        active = geom_col if geom_col in geom_cols else geom_cols[0]
        rows = self._fetch(
            f"SELECT {', '.join(sel)}, SRID({_q(active)}) AS __geolattice_srid FROM ({sql}) AS __q", params
        )
        df = pd.DataFrame.from_records(rows, columns=names + ["__geolattice_srid"])
        if crs is None:
            s = df["__geolattice_srid"].dropna()
            crs = int(s.iloc[0]) if len(s) and int(s.iloc[0]) > 0 else None
        df = df.drop(columns="__geolattice_srid")
        return _to_geodataframe(df, geom_cols, active, crs)

    def _query_duckdb(self, sql, params, geom_col, crs):
        desc = self._con.execute(f"DESCRIBE ({sql})", params or []).fetchall()
        names = [d[0] for d in desc]
        geom_cols = [d[0] for d in desc if str(d[1]).upper().startswith("GEOMETRY")]
        if not geom_cols:
            return self._con.execute(sql, params or []).df()
        sel = [f"ST_AsWKB({_q(n)}) AS {_q(n)}" if n in geom_cols else _q(n) for n in names]
        cur = self._con.execute(f"SELECT {', '.join(sel)} FROM ({sql}) AS __q", params or [])
        df = pd.DataFrame.from_records(cur.fetchall(), columns=names)
        for c in geom_cols:
            df[c] = [bytes(v) if v is not None else None for v in df[c]]
        if crs is None:
            m = re.search(r"EPSG:(\d+)", " ".join(str(d[1]) for d in desc if d[0] in geom_cols))
            if m:
                crs = int(m.group(1))
            else:
                srids = {r[0] for r in self._fetch(f"SELECT DISTINCT srid FROM {_META_TABLE} WHERE srid > 0")}
                crs = srids.pop() if len(srids) == 1 else None
        return _to_geodataframe(df, geom_cols, geom_col, crs)

    def read_table(self, table: str, where: str | None = None, limit: int | None = None):
        """Read a whole table (optionally filtered) as a GeoDataFrame."""
        sql = f"SELECT * FROM {_q(table)}"
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        crs = None
        try:
            s = self.srid(table)
            crs = s or None
        except ValueError:
            pass
        return self.query(sql, crs=crs)

    # --------------------------------------------------- SQL-backed ops
    def _materialize(self, sql: str, out_table: str | None, geom_col="geom", crs=None, index=True):
        if out_table is None:
            return self.query(sql, crs=crs)
        if self.backend == "postgis":
            self.execute(f"DROP TABLE IF EXISTS {_q(out_table)}")
            self.execute(f"CREATE TABLE {_q(out_table)} AS {sql}")
            if index:
                self.create_index(out_table, geom_col)
        elif self.backend == "duckdb":
            self._con.execute(f"CREATE OR REPLACE TABLE {_q(out_table)} AS {sql}")
            srid = _epsg(gpd.GeoSeries([], crs=crs).crs) if crs else 0
            self._con.execute(f"INSERT OR REPLACE INTO {_META_TABLE} VALUES (?, ?, ?)", [out_table, geom_col, srid])
            if index:
                self.create_index(out_table, geom_col)
        else:
            gdf = self.query(sql, crs=crs)
            self.load(gdf, out_table, geom_col=gdf.geometry.name, index=index)
        return self.read_table(out_table)

    def _other_cols(self, table, geom_col, alias, exclude=(), prefix=""):
        cols = [c for c in self.columns(table) if c != geom_col and c not in exclude]
        return [f"{alias}.{_q(c)} AS {_q(prefix + c)}" for c in cols], cols

    def spatial_join(self, left: str, right: str, predicate: str = "intersects", how: str = "inner",
                     right_columns: Sequence[str] | None = None, distance: float | None = None,
                     out_table: str | None = None):
        """Join two tables on a spatial predicate, in the database.

        predicate : intersects, within, contains, touches, crosses, overlaps, covers,
                    covered_by, equals, disjoint, or 'dwithin' (needs ``distance``).
        how : 'inner' or 'left'.
        right_columns : columns to bring from ``right`` (default: all non-geometry).
            Name clashes with ``left`` get a ``right_`` prefix.
        Keeps the left geometry. Returns a GeoDataFrame, or writes ``out_table``.
        """
        how = how.lower()
        if how not in ("inner", "left"):
            raise ValueError("how must be 'inner' or 'left'")
        lg, rg = self.geometry_column(left), self.geometry_column(right)
        lcols = self.columns(left)
        rcols = [c for c in (right_columns or self.columns(right)) if c != rg]
        rsel = [f"r.{_q(c)} AS {_q('right_' + c if c in lcols else c)}" for c in rcols]
        if predicate == "dwithin":
            if distance is None:
                raise ValueError("predicate='dwithin' needs distance=")
            cond = (f"ST_DWithin(l.{_q(lg)}, r.{_q(rg)}, {float(distance)})" if self.backend in ("postgis", "duckdb")
                    else f"ST_Distance(l.{_q(lg)}, r.{_q(rg)}) <= {float(distance)}")
        else:
            fn = _PREDICATES.get(predicate)
            if fn is None:
                raise ValueError(f"predicate must be one of {list(_PREDICATES) + ['dwithin']}")
            cond = f"{fn}(l.{_q(lg)}, r.{_q(rg)})"
            if self.backend == "spatialite":
                cond = self._spatialite_index_filter(right, rg, f"l.{_q(lg)}", cond) if fn != "ST_Disjoint" else cond
        join = "JOIN" if how == "inner" else "LEFT JOIN"
        sql = (f"SELECT {', '.join([f'l.{_q(c)}' for c in lcols] + rsel)} "
               f"FROM {_q(left)} AS l {join} {_q(right)} AS r ON {cond}")
        return self._materialize(sql, out_table, geom_col=lg, crs=self.srid(left, lg) or None)

    def _spatialite_index_filter(self, table, geom_col, other_geom, cond):
        """Use the SpatiaLite R*Tree (if present) to pre-filter candidate rows."""
        idx = f"idx_{table}_{geom_col}"
        if self._fetch("SELECT name FROM sqlite_master WHERE name=?", [idx]):
            return (f"{cond} AND r.ROWID IN (SELECT ROWID FROM SpatialIndex WHERE f_table_name={_lit(table)} "
                    f"AND f_geometry_column={_lit(geom_col)} AND search_frame={other_geom})")
        return cond

    def nearest(self, left: str, right: str, k: int = 1, max_distance: float | None = None,
                right_columns: Sequence[str] | None = None, geography: bool = False,
                out_table: str | None = None):
        """For each row of ``left``, find the ``k`` nearest rows of ``right``.

        Adds ``distance`` (in SRID units, or meters with ``geography=True`` on PostGIS)
        and ``rank`` columns. PostGIS uses the indexed KNN ``<->`` operator; the other
        backends rank all pairs, so filter with ``max_distance`` for big tables.
        """
        k = int(k)
        lg, rg = self.geometry_column(left), self.geometry_column(right)
        lcols = self.columns(left)
        rcols = [c for c in (right_columns or self.columns(right)) if c != rg]
        rnames = {c: ("right_" + c if c in lcols or c in ("distance", "rank") else c) for c in rcols}
        lsel = ", ".join(f"l.{_q(c)}" for c in lcols)
        if self.backend == "postgis":
            dist = (f"ST_Distance(l.{_q(lg)}::geography, r.{_q(rg)}::geography)" if geography
                    else f"ST_Distance(l.{_q(lg)}, r.{_q(rg)})")
            where = ""
            if max_distance is not None:
                where = (f"WHERE ST_DWithin(l.{_q(lg)}::geography, r.{_q(rg)}::geography, {float(max_distance)})"
                         if geography else f"WHERE ST_DWithin(l.{_q(lg)}, r.{_q(rg)}, {float(max_distance)})")
            rsel = ", ".join([f"r.{_q(c)} AS {_q(n)}" for c, n in rnames.items()] + [f"{dist} AS distance"])
            sql = (f"SELECT {lsel}, n.* FROM {_q(left)} AS l CROSS JOIN LATERAL ("
                   f"SELECT {rsel}, row_number() OVER (ORDER BY l.{_q(lg)} <-> r.{_q(rg)}) AS rank FROM {_q(right)} AS r {where} "
                   f"ORDER BY l.{_q(lg)} <-> r.{_q(rg)} LIMIT {k}) AS n")
        else:
            dist = f"ST_Distance(l.{_q(lg)}, r.{_q(rg)})"
            where = f"WHERE {dist} <= {float(max_distance)}" if max_distance is not None else ""
            rsel = ", ".join(f"r.{_q(c)} AS {_q(n)}" for c, n in rnames.items())
            rsel = (rsel + ", ") if rsel else ""
            sql = (f"SELECT * FROM (SELECT {lsel}, {rsel}{dist} AS distance, "
                   f"ROW_NUMBER() OVER (PARTITION BY l.__geolattice_lid ORDER BY {dist}) AS rank "
                   f"FROM (SELECT *, ROW_NUMBER() OVER () AS __geolattice_lid FROM {_q(left)}) AS l "
                   f"CROSS JOIN {_q(right)} AS r {where}) AS p WHERE rank <= {k}")
        return self._materialize(sql, out_table, geom_col=lg, crs=self.srid(left, lg) or None)

    def buffer(self, table: str, distance: float, *, dissolve: bool = False, segments: int = 8,
               geography: bool = False, out_table: str | None = None):
        """Buffer every geometry of ``table`` by ``distance`` (SRID units).

        geography : PostGIS only, buffer in meters on the spheroid.
        dissolve : merge all buffers into one geometry.
        """
        g = self.geometry_column(table)
        d = float(distance)
        if self.backend == "postgis":
            expr = (f"ST_Buffer({_q(g)}::geography, {d}, 'quad_segs={int(segments)}')::geometry" if geography
                    else f"ST_Buffer({_q(g)}, {d}, 'quad_segs={int(segments)}')")
            union = "ST_Union"
        else:
            if geography:
                raise ValueError("geography=True is only supported on PostGIS; reproject to a metric SRID instead.")
            expr = f"ST_Buffer({_q(g)}, {d}, {int(segments)})"
            union = "ST_Union_Agg" if self.backend == "duckdb" else "ST_Union"
        if dissolve:
            sql = f"SELECT {union}({expr}) AS {_q(g)} FROM {_q(table)}"
        else:
            others = [c for c in self.columns(table) if c != g]
            sql = f"SELECT {', '.join([_q(c) for c in others] + [f'{expr} AS {_q(g)}'])} FROM {_q(table)}"
        return self._materialize(sql, out_table, geom_col=g, crs=self.srid(table, g) or None)

    # ------------------------------------------------- index & performance
    def create_index(self, table: str, geom_col: str | None = None, method: str = "gist",
                     analyze: bool = True) -> str | None:
        """Create a spatial index, or return the existing one's name if present.

        method : PostGIS 'gist' (default), 'spgist' or 'brin'. Other backends use
                 their native R-tree regardless.
        """
        geom_col = geom_col or self.geometry_column(table)
        schema, name = _split(table, self.schema)
        idx = f"{name}_{geom_col}_{method}_idx"
        if self.backend == "postgis":
            method = method.lower()
            if method not in ("gist", "spgist", "brin"):
                raise ValueError("method must be 'gist', 'spgist' or 'brin'")
            have = self._fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname=%s AND tablename=%s "
                "AND indexdef ILIKE %s AND indexdef ILIKE %s",
                (schema, name, f"%USING {method} %", f"%({geom_col})%"),
            )
            if have:
                return have[0][0]
            self.execute(f"CREATE INDEX IF NOT EXISTS {_q(idx)} ON {_q(table)} USING {method} ({_q(geom_col)})")
            if analyze:
                self.execute(f"ANALYZE {_q(table)}")
        elif self.backend == "duckdb":
            idx = f"{name}_{geom_col}_rtree_idx"
            self._con.execute(f"CREATE INDEX IF NOT EXISTS {_q(idx)} ON {_q(table)} USING RTREE ({_q(geom_col)})")
        elif self.backend == "geopackage":
            idx = f"rtree_{name}_{geom_col}"
            if not self._fetch("SELECT name FROM sqlite_master WHERE name=?", [idx]):
                self.execute("SELECT gpkgAddSpatialIndex(?, ?)", [name, geom_col])
        else:
            idx = f"idx_{name}_{geom_col}"
            if not self._fetch("SELECT name FROM sqlite_master WHERE name=?", [idx]):
                self.execute("SELECT CreateSpatialIndex(?, ?)", [name, geom_col])
            if analyze:
                self.execute("ANALYZE")
        return idx

    def explain(self, sql: str, analyze: bool = False, params=None) -> str:
        """Return the query plan as text (``analyze=True`` runs the query and adds timings)."""
        sql = sql.strip().rstrip(";")
        if self.backend == "postgis":
            rows = self._fetch(f"EXPLAIN (ANALYZE {str(analyze).upper()}, BUFFERS {str(analyze).upper()}) {sql}", params)
            return "\n".join(r[0] for r in rows)
        if self.backend == "duckdb":
            rows = self._fetch(("EXPLAIN ANALYZE " if analyze else "EXPLAIN ") + sql, params)
            return "\n".join(str(r[-1]) for r in rows)
        rows = self._fetch(f"EXPLAIN QUERY PLAN {sql}", params)
        return "\n".join(str(r[-1]) for r in rows)

    def vacuum(self, table: str | None = None):
        """VACUUM ANALYZE (PostGIS) / VACUUM + ANALYZE (others)."""
        if self.backend == "postgis":
            raw = self._engine.raw_connection()
            try:
                raw.driver_connection.autocommit = True
                raw.cursor().execute(f"VACUUM ANALYZE {_q(table) if table else ''}")
            finally:
                raw.close()
        elif self.backend == "duckdb":
            self._con.execute(f"ANALYZE {_q(table) if table else ''}")
        else:
            self._con.commit()
            self._con.execute("VACUUM")
            self._con.execute("ANALYZE")


# ---------------------------------------------------------------------------
# raster namespace (PostGIS raster)
# ---------------------------------------------------------------------------
class _Raster:
    """``db.raster``: raster storage and zonal statistics (PostGIS)."""

    STATS = ("count", "sum", "mean", "stddev", "min", "max")

    def __init__(self, db: SpatialDB):
        self._db = db

    def _require(self):
        if self._db.backend != "postgis":
            raise NotImplementedError("Raster functions need the PostGIS backend (postgis_raster).")
        self._db.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster")

    def load(self, path, table: str, *, srid: int | None = None, if_exists: str = "replace",
             index: bool = True, tile_size: int = 256, out_db: bool = False) -> str:
        """Load a raster file with ``raster2pgsql`` (tiled, constrained, indexed).

        out_db : register the file path instead of copying pixels (``-R``).
        """
        self._require()
        exe = shutil_which("raster2pgsql")
        if exe is None:
            raise RuntimeError("raster2pgsql not found on PATH (it ships with PostGIS).")
        if srid is None:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    srid = _epsg(src.crs) if src.crs else 0
            except ImportError:
                srid = 0
        mode = {"replace": "-d", "append": "-a", "fail": "-c"}[if_exists]
        cmd = [exe, mode, "-s", str(srid), "-t", f"{tile_size}x{tile_size}"]
        if if_exists != "append":
            cmd.append("-C")
        if index:
            cmd.append("-I")
        if out_db:
            cmd.append("-R")
        cmd += [str(Path(path).resolve()) if out_db else str(path), self._db_table(table)]
        sql = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        self._db.execute(sql)
        self._db.execute(f"ANALYZE {_q(table)}")
        return table

    @staticmethod
    def _db_table(table):
        schema, name = _split(table)
        return f"{schema}.{name}" if schema else name

    def info(self, table: str, raster_col: str = "rast") -> pd.DataFrame:
        """Raster metadata from the raster_columns view."""
        self._require()
        schema, name = _split(table, self._db.schema)
        return self._db.query(
            "SELECT r_raster_column, srid, scale_x, scale_y, blocksize_x, blocksize_y, num_bands, "
            "pixel_types, nodata_values, out_db FROM raster_columns WHERE r_table_schema=%s AND r_table_name=%s",
            (schema, name),
        )

    def zonal_stats(self, raster_table: str, zones_table: str, *, band: int = 1,
                    stats: Iterable[str] = STATS, raster_col: str = "rast",
                    zone_columns: Sequence[str] | None = None, out_table: str | None = None):
        """Zonal statistics of a raster band for every polygon in ``zones_table``.

        Computed in the database with ST_Clip + ST_SummaryStatsAgg; zones are
        reprojected to the raster SRID on the fly. Returns a GeoDataFrame with the
        zone attributes, the zone geometry and one column per statistic.
        """
        self._require()
        db = self._db
        stats = list(stats)
        bad = set(stats) - set(self.STATS)
        if bad:
            raise ValueError(f"Unknown stats {bad}; choose from {self.STATS}")
        zg = db.geometry_column(zones_table)
        schema, rname = _split(raster_table, db.schema)
        rs = db._fetch(
            "SELECT srid FROM raster_columns WHERE r_table_schema=%s AND r_table_name=%s AND r_raster_column=%s",
            (schema, rname, raster_col),
        )
        rsrid = int(rs[0][0]) if rs and rs[0][0] else int(
            db._fetch(f"SELECT ST_SRID({_q(raster_col)}) FROM {_q(raster_table)} LIMIT 1")[0][0])
        zcols = list(zone_columns) if zone_columns else [c for c in db.columns(zones_table) if c != zg]
        zsel = ", ".join([f"z.{_q(c)}" for c in zcols] + [f"z.{_q(zg)}"])
        ssel = ", ".join(f"(s.st).{st} AS {_q(st)}" for st in stats)
        sql = (
            f"WITH z AS (SELECT *, row_number() OVER () AS __zid, "
            f"ST_Transform({_q(zg)}, {rsrid}) AS __zgeom FROM {_q(zones_table)}), "
            f"s AS (SELECT z.__zid, ST_SummaryStatsAgg(ST_Clip(r.{_q(raster_col)}, {int(band)}, z.__zgeom, TRUE), "
            f"1, TRUE) AS st FROM z JOIN {_q(raster_table)} AS r "
            f"ON ST_Intersects(r.{_q(raster_col)}, z.__zgeom) GROUP BY z.__zid) "
            f"SELECT {zsel}, {ssel} FROM z LEFT JOIN s ON s.__zid = z.__zid ORDER BY z.__zid"
        )
        return db._materialize(sql, out_table, geom_col=zg, crs=db.srid(zones_table, zg) or None)

    def value_at(self, raster_table: str, points_table: str, *, band: int = 1, raster_col: str = "rast",
                 column: str = "value"):
        """Sample raster values at point locations (ST_Value).

        Non-point geometries are sampled at ST_PointOnSurface.
        """
        self._require()
        db = self._db
        pg = db.geometry_column(points_table)
        rsrid = int(db._fetch(f"SELECT ST_SRID({_q(raster_col)}) FROM {_q(raster_table)} LIMIT 1")[0][0])
        pcols = ", ".join(f"p.{_q(c)}" for c in db.columns(points_table))
        sql = (f"SELECT {pcols}, ST_Value(r.{_q(raster_col)}, {int(band)}, ST_Transform(ST_PointOnSurface(p.{_q(pg)}), {rsrid})) "
               f"AS {_q(column)} FROM {_q(points_table)} AS p LEFT JOIN {_q(raster_table)} AS r "
               f"ON ST_Intersects(r.{_q(raster_col)}, ST_Transform(ST_PointOnSurface(p.{_q(pg)}), {rsrid}))")
        return db.query(sql)


def shutil_which(name):
    import shutil
    return shutil.which(name)


# ---------------------------------------------------------------------------
# network namespace (pgRouting)
# ---------------------------------------------------------------------------
class _Network:
    """``db.network``: routing with pgRouting (PostGIS)."""

    def __init__(self, db: SpatialDB):
        self._db = db

    def _require(self):
        if self._db.backend != "postgis":
            raise NotImplementedError("Network functions need the PostGIS backend with pgRouting.")
        self._db.execute("CREATE EXTENSION IF NOT EXISTS pgrouting")

    def _vertices(self, edges):
        schema, name = _split(edges, self._db.schema)
        return f"{schema}.{name}_vertices" if schema else f"{name}_vertices"

    def prepare(self, edges: str, *, tolerance: float = 1e-6, cost: str | None = None,
                reverse_cost: str | None = None, geography: bool = False, oneway: bool = False) -> str:
        """Make a line table routable.

        Adds ``id``, ``source``, ``target``, ``cost`` and ``reverse_cost`` columns and
        builds ``<edges>_vertices``. Endpoints closer than ``tolerance`` are merged.

        cost / reverse_cost : existing columns or SQL expressions to use; default is
            the segment length (meters if ``geography=True``).
        oneway : if True and no reverse_cost is given, reverse travel is blocked (-1).
        Returns the vertices table name.
        """
        self._require()
        db = self._db
        g = db.geometry_column(edges)
        cols = db.columns(edges)
        e, v, tol = _q(edges), _q(self._vertices(edges)), float(tolerance)
        if "id" not in cols:
            db.execute(f"ALTER TABLE {e} ADD COLUMN id BIGSERIAL")
        for c in ("source", "target"):
            if c not in cols:
                db.execute(f"ALTER TABLE {e} ADD COLUMN {c} BIGINT")
        for c in ("cost", "reverse_cost"):
            if c not in cols:
                db.execute(f"ALTER TABLE {e} ADD COLUMN {c} DOUBLE PRECISION")
        start = f"ST_SnapToGrid(ST_StartPoint(ST_GeometryN({_q(g)}, 1)), {tol})"
        end = f"ST_SnapToGrid(ST_EndPoint(ST_GeometryN({_q(g)}, ST_NumGeometries({_q(g)}))), {tol})"
        db.execute(f"DROP TABLE IF EXISTS {v}")
        db.execute(
            f"CREATE TABLE {v} AS SELECT row_number() OVER () AS id, geom FROM ("
            f"SELECT DISTINCT ON (ST_AsBinary(p)) p AS geom FROM ("
            f"SELECT {start} AS p FROM {e} UNION ALL SELECT {end} FROM {e}) AS pts) AS d"
        )
        db.execute(f"CREATE INDEX ON {v} USING gist (geom)")
        db.execute(f"UPDATE {e} AS x SET source = v.id FROM {v} AS v WHERE v.geom ~= {start.replace(_q(g), 'x.' + _q(g))}")
        db.execute(f"UPDATE {e} AS x SET target = v.id FROM {v} AS v WHERE v.geom ~= {end.replace(_q(g), 'x.' + _q(g))}")
        length = f"ST_Length({_q(g)}::geography)" if geography else f"ST_Length({_q(g)})"
        db.execute(f"UPDATE {e} SET cost = {cost or length}")
        rc = reverse_cost or ("-1" if oneway else "cost")
        db.execute(f"UPDATE {e} SET reverse_cost = {rc}")
        db.execute(f"CREATE INDEX IF NOT EXISTS {_q(_split(edges)[1] + '_source_idx')} ON {e} (source)")
        db.execute(f"CREATE INDEX IF NOT EXISTS {_q(_split(edges)[1] + '_target_idx')} ON {e} (target)")
        db.execute(f"ANALYZE {e}")
        return self._vertices(edges)

    def nearest_vertex(self, edges: str, point, srid: int = 4326) -> int:
        """Vertex id nearest to an (x, y) point given in ``srid``."""
        self._require()
        db = self._db
        v = self._vertices(edges)
        vsrid = db._fetch(f"SELECT ST_SRID(geom) FROM {_q(v)} LIMIT 1")[0][0]
        x, y = float(point[0]), float(point[1])
        row = db._fetch(
            f"SELECT id FROM {_q(v)} ORDER BY geom <-> ST_Transform(ST_SetSRID(ST_MakePoint({x}, {y}), {int(srid)}), "
            f"{int(vsrid)}) LIMIT 1"
        )
        return int(row[0][0])

    def _vid(self, edges, node, srid):
        if isinstance(node, (tuple, list)) and len(node) == 2 and not isinstance(node[0], (tuple, list)):
            return self.nearest_vertex(edges, node, srid)
        return int(node)

    @staticmethod
    def _edges_sql(edges, directed=False):
        return f"SELECT id, source, target, cost, reverse_cost FROM {_q(edges)}"

    def shortest_path(self, edges: str, source, target, *, directed: bool = False, srid: int = 4326):
        """Shortest path with pgr_dijkstra.

        source / target : vertex ids, or (x, y) tuples in ``srid`` (snapped to the
            nearest vertex). Returns a GeoDataFrame of path edges in order with
            ``seq``, ``node``, ``edge``, ``cost``, ``agg_cost``.
        """
        self._require()
        db = self._db
        s, t = self._vid(edges, source, srid), self._vid(edges, target, srid)
        g = db.geometry_column(edges)
        sql = (f"SELECT p.seq, p.node, p.edge, p.cost, p.agg_cost, e.{_q(g)} AS geom "
               f"FROM pgr_dijkstra({_lit(self._edges_sql(edges, directed))}, {s}, {t}, {str(directed).lower()}) AS p "
               f"JOIN {_q(edges)} AS e ON p.edge = e.id ORDER BY p.seq")
        return db.query(sql)

    def isochrone(self, edges: str, source, costs: float | Sequence[float], *, directed: bool = False,
                  srid: int = 4326, hull: str = "concave", concavity: float = 0.7):
        """Service areas around ``source`` with pgr_drivingDistance.

        costs : one value or a list of cost thresholds (same units as ``cost``).
        hull : 'concave' (ST_ConcaveHull), 'convex', or 'points' (reached vertices).
        Returns one polygon per threshold (largest first), or reached vertices.
        """
        self._require()
        db = self._db
        s = self._vid(edges, source, srid)
        costs = [costs] if np.isscalar(costs) else list(costs)
        v = self._vertices(edges)
        dd = (f"SELECT d.node, d.agg_cost, v.geom FROM pgr_drivingDistance("
              f"{_lit(self._edges_sql(edges, directed))}, {s}, {float(max(costs))}, {str(directed).lower()}) AS d "
              f"JOIN {_q(v)} AS v ON v.id = d.node")
        if hull == "points":
            return db.query(dd + " ORDER BY d.agg_cost")
        if hull not in ("concave", "convex"):
            raise ValueError("hull must be 'concave', 'convex' or 'points'")
        h = (f"ST_ConcaveHull(ST_Collect(geom), {float(concavity)})" if hull == "concave"
             else "ST_ConvexHull(ST_Collect(geom))")
        parts = [f"SELECT {float(c)} AS max_cost, count(*) AS n_nodes, {h} AS geom FROM dd WHERE agg_cost <= {float(c)}"
                 for c in sorted(costs, reverse=True)]
        sql = f"WITH dd AS ({dd}) " + " UNION ALL ".join(parts)
        return db.query(sql)


# ---------------------------------------------------------------------------
# tiles namespace (MVT)
# ---------------------------------------------------------------------------
_VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>GeoLattice tiles</title>
<link href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>
<style>body,html,#map{margin:0;height:100%}</style></head><body><div id="map"></div><script>
const layers = __LAYERS__;
const map = new maplibre.Map({container:'map', center:__CENTER__, zoom:__ZOOM__,
  style:{version:8, sources:{osm:{type:'raster', tiles:['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
  tileSize:256, attribution:'&copy; OpenStreetMap contributors'}}, layers:[{id:'osm', type:'raster', source:'osm'}]}});
const colors = ['#e6550d','#3182bd','#31a354','#756bb1','#636363'];
map.on('load', () => layers.forEach((name, i) => {
  map.addSource(name, {type:'vector', tiles:[location.origin + '/' + name + '/{z}/{x}/{y}.pbf']});
  const c = colors[i % colors.length];
  map.addLayer({id:name+'-fill', type:'fill', source:name, 'source-layer':name,
    filter:['==',['geometry-type'],'Polygon'], paint:{'fill-color':c, 'fill-opacity':0.35}});
  map.addLayer({id:name+'-line', type:'line', source:name, 'source-layer':name, paint:{'line-color':c, 'line-width':1}});
  map.addLayer({id:name+'-pt', type:'circle', source:name, 'source-layer':name,
    filter:['==',['geometry-type'],'Point'], paint:{'circle-color':c, 'circle-radius':4}});
}));
</script></body></html>"""


class _Tiles:
    """``db.tiles``: Mapbox Vector Tiles (MVT) from PostGIS or DuckDB."""

    def __init__(self, db: SpatialDB):
        self._db = db
        self._servers = []

    def mvt(self, table: str, z: int, x: int, y: int, *, columns: Sequence[str] | None = None,
            extent: int = 4096, buffer: int = 64, layer: str | None = None) -> bytes:
        """Encode one tile (z/x/y, Web Mercator) of ``table`` as MVT bytes."""
        db = self._db
        if db.backend not in ("postgis", "duckdb"):
            raise NotImplementedError("Vector tiles need the PostGIS or DuckDB backend.")
        z, x, y = int(z), int(x), int(y)
        g = db.geometry_column(table)
        srid = db.srid(table, g) or 4326
        cols = list(columns) if columns is not None else [c for c in db.columns(table) if c != g]
        layer = layer or _split(table)[1]
        csel = "".join(f", t.{_q(c)}" for c in cols)
        if db.backend == "postgis":
            sql = (f"WITH b AS (SELECT ST_TileEnvelope({z}, {x}, {y}) AS env), "
                   f"m AS (SELECT ST_AsMVTGeom(ST_Transform(t.{_q(g)}, 3857), b.env, {extent}, {buffer}, true) AS geom"
                   f"{csel} FROM {_q(table)} AS t, b "
                   f"WHERE t.{_q(g)} && ST_Transform(b.env, {srid})) "
                   f"SELECT ST_AsMVT(m.*, {_lit(layer)}, {extent}, 'geom') FROM m WHERE geom IS NOT NULL")
            res = db._fetch(sql)
        else:
            to3857 = (f"t.{_q(g)}" if srid == 3857 else
                      f"ST_Transform(t.{_q(g)}, 'EPSG:{srid}', 'EPSG:3857', always_xy := true)")
            sql = (f"WITH b AS (SELECT ST_TileEnvelope({z}, {x}, {y}) AS env), "
                   f"m AS (SELECT ST_AsMVTGeom({to3857}, ST_Extent(b.env), {extent}, {buffer}, true) AS geom"
                   f"{''.join(f', t.{_q(c)} AS {_q(c)}' for c in cols)} FROM {_q(table)} AS t, b "
                   f"WHERE ST_Intersects({to3857}, b.env)) "
                   f"SELECT CASE WHEN count(*) = 0 THEN NULL ELSE ST_AsMVT({{'geom': geom{''.join(f', {_lit(c)}: {_q(c)}' for c in cols)}}}, "
                   f"{_lit(layer)}, {extent}, 'geom') END FROM m WHERE geom IS NOT NULL")
            res = db._fetch(sql)
        out = res[0][0] if res else None
        return bytes(out) if out is not None else b""

    def bounds(self, table: str):
        """(minx, miny, maxx, maxy) of a table in EPSG:4326."""
        db = self._db
        g = db.geometry_column(table)
        srid = db.srid(table, g) or 4326
        if db.backend == "postgis":
            r = db._fetch(f"SELECT ST_XMin(e), ST_YMin(e), ST_XMax(e), ST_YMax(e) FROM "
                          f"(SELECT ST_Transform(ST_SetSRID(ST_Extent({_q(g)}), {srid}), 4326) AS e FROM {_q(table)}) q")
        else:
            tr = (f"{_q(g)}" if srid == 4326 else
                  f"ST_Transform({_q(g)}, 'EPSG:{srid}', 'EPSG:4326', always_xy := true)")
            r = db._fetch(f"SELECT min(ST_XMin({tr})), min(ST_YMin({tr})), max(ST_XMax({tr})), max(ST_YMax({tr})) "
                          f"FROM {_q(table)}")
        return tuple(float(v) for v in r[0])

    def serve(self, tables: Sequence[str] | None = None, *, host: str = "127.0.0.1", port: int = 8080,
              block: bool = False, columns: dict | None = None):
        """Serve tables as vector tiles over HTTP.

        Endpoints: ``/{table}/{z}/{x}/{y}.pbf``, ``/tables.json`` (TileJSON list)
        and ``/`` (a MapLibre viewer). With ``block=False`` (default, handy in
        notebooks) the server runs in a background thread; stop it with
        ``server.shutdown()``. Returns the server.
        """
        db = self._db
        tables = list(tables or [t for t in db.tables() if _has_geom(db, t)])
        columns = columns or {}
        tiles = self
        cache_bounds = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code, body: bytes, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0].strip("/")
                try:
                    if path in ("", "index.html"):
                        b = cache_bounds.setdefault("all", tiles.bounds(tables[0]) if tables else (-180, -85, 180, 85))
                        html = (_VIEWER.replace("__LAYERS__", json.dumps(tables))
                                .replace("__CENTER__", json.dumps([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]))
                                .replace("__ZOOM__", "6"))
                        return self._send(200, html.encode(), "text/html; charset=utf-8")
                    if path == "tables.json":
                        base = f"http://{self.headers.get('Host', f'{host}:{port}')}"
                        info = [{"tilejson": "3.0.0", "name": t, "tiles": [f"{base}/{t}/{{z}}/{{x}}/{{y}}.pbf"]}
                                for t in tables]
                        return self._send(200, json.dumps(info).encode(), "application/json")
                    m = re.fullmatch(r"([^/]+)/(\d+)/(\d+)/(\d+)\.(pbf|mvt)", path)
                    if not m or m.group(1) not in tables:
                        return self._send(404, b"not found", "text/plain")
                    t, z, x, y = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
                    body = tiles.mvt(t, z, x, y, columns=columns.get(t))
                    return self._send(200, body, "application/vnd.mapbox-vector-tile")
                except Exception as exc:  # pragma: no cover
                    return self._send(500, str(exc).encode(), "text/plain")

        server = ThreadingHTTPServer((host, port), Handler)
        server.url = f"http://{host}:{server.server_address[1]}/"
        self._servers.append(server)
        print(f"Serving {tables} at {server.url}  (tiles: {server.url}<table>/{{z}}/{{x}}/{{y}}.pbf)")
        if block:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.shutdown()
        else:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def stop(self):
        """Stop all servers started by this connection."""
        for s in self._servers:
            s.shutdown()
            s.server_close()
        self._servers.clear()


def _has_geom(db, table):
    try:
        db.geometry_column(table)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# module-level API
# ---------------------------------------------------------------------------
def connect(backend: str = "postgis", **kwargs) -> SpatialDB:
    """Connect to a spatial database.

    Examples
    --------
    >>> connect("postgis", host="localhost", dbname="gis", user="me", password="...")
    >>> connect("postgis", dsn="postgresql://me:pw@localhost:5432/gis")
    >>> connect("spatialite", path="data.sqlite")
    >>> connect("geopackage", path="data.gpkg")
    >>> connect("duckdb", path="data.duckdb")      # or path=None for in-memory
    """
    return SpatialDB(backend, **kwargs)


def load(db: SpatialDB, data, table: str, **kwargs) -> str:
    """See :meth:`SpatialDB.load`."""
    return db.load(data, table, **kwargs)


def query(db: SpatialDB, sql: str, params=None, **kwargs):
    """See :meth:`SpatialDB.query`."""
    return db.query(sql, params, **kwargs)


def execute(db: SpatialDB, sql: str, params=None):
    """See :meth:`SpatialDB.execute`."""
    return db.execute(sql, params)


def spatial_join(db: SpatialDB, left: str, right: str, **kwargs):
    """See :meth:`SpatialDB.spatial_join`."""
    return db.spatial_join(left, right, **kwargs)


def nearest(db: SpatialDB, left: str, right: str, **kwargs):
    """See :meth:`SpatialDB.nearest`."""
    return db.nearest(left, right, **kwargs)


def buffer(db: SpatialDB, table: str, distance: float, **kwargs):
    """See :meth:`SpatialDB.buffer`."""
    return db.buffer(table, distance, **kwargs)


def create_index(db: SpatialDB, table: str, geom_col: str | None = None, **kwargs):
    """See :meth:`SpatialDB.create_index`."""
    return db.create_index(table, geom_col, **kwargs)


def explain(db: SpatialDB, sql: str, analyze: bool = False, **kwargs) -> str:
    """See :meth:`SpatialDB.explain`."""
    return db.explain(sql, analyze, **kwargs)
