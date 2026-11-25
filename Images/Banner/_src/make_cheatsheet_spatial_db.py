"""Render the Spatial Databases quick reference (Python API | spatial SQL, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. Python names are checked
against geolattice.geolatticedb; SQL names are checked against the SQL the 06_spatialDB notebooks run.

    python Images/Banner/_src/make_cheatsheet_spatial_db.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cheatsheet_spatial_analysis as sa  # noqa: E402

base = sa.base
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTEBOOKS = ROOT / "Tutorials" / "notebooks" / "06_spatialDB"
OUT = HERE.parent / "06_00_spatial_db_quick_reference.png"
COPY = NOTEBOOKS / "06_spatial_db_quick_reference.png"

# Python groups: tag is where the name lives (pdb = module, db = SpatialDB connection)
PYTHON = [
    ("Connect", "pdb", [
        ("connect · BACKENDS", "postgis, spatialite, geopackage or duckdb"),
        ("SpatialDB", "One connection object for every backend"),
    ]),
    ("Load, query and inspect", "db", [
        ("load", "File or GeoDataFrame to a table, SRID and index set"),
        ("query · read_table", "SQL or a table back as a GeoDataFrame"),
        ("execute", "Run DDL and DML statements"),
        ("tables · columns · geometry_column · srid", "Inspect the database"),
        ("close", "Close the connection"),
    ]),
    ("Spatial operations", "db", [
        ("spatial_join", "Join tables by a spatial predicate"),
        ("nearest", "k nearest features"),
        ("buffer", "Buffers in metres, optional geography"),
    ]),
    ("Performance", "db", [
        ("create_index · explain · vacuum", "Spatial indexes, query plans, maintenance"),
    ]),
    ("PostGIS raster", "db.raster", [
        ("load · info", "Load rasters, read metadata"),
        ("zonal_stats · value_at", "Zonal statistics, values at points"),
    ]),
    ("pgRouting", "db.network", [
        ("prepare · nearest_vertex", "Build topology, snap to the network"),
        ("shortest_path · isochrone", "Routes and drive-time areas"),
    ]),
    ("Vector tiles", "db.tiles", [
        ("mvt · bounds", "Mapbox vector tiles from tables"),
        ("serve · stop", "Local tile server"),
    ]),
    ("Raster SQL", "PostGIS", [
        ("ST_MetaData · ST_BandMetaData · ST_Value", "Raster metadata and cell values"),
        ("ST_Clip · ST_MapAlgebra · ST_Reclass", "Clip, map algebra, reclassify"),
        ("ST_SummaryStats · ST_SummaryStatsAgg · ST_Histogram", "Raster statistics"),
        ("ST_DumpAsPolygons · ST_AsGDALRaster", "Raster to polygons, export"),
    ]),
    ("Routing SQL", "pgRouting", [
        ("pgr_dijkstra · pgr_dijkstraCost", "Shortest paths and costs"),
        ("pgr_drivingDistance", "Reachable network within a cost"),
    ]),
    ("Trajectories", "PostGIS", [
        ("ST_M · ST_LocateAlong · ST_IsValidTrajectory", "Measures and trajectory checks"),
        ("ST_ClosestPointOfApproach · ST_DistanceCPA · ST_CPAWithin", "Closest approach of moving objects"),
    ]),
]

SQL = [
    ("Construct and transform", "SQL", [
        ("ST_MakePoint · ST_Point · ST_MakeEnvelope", "Points and boxes"),
        ("ST_GeomFromText · ST_GeomFromWKB", "Geometry from WKT or WKB"),
        ("ST_SetSRID · ST_Transform", "Assign or reproject a CRS"),
        ("ST_AsText · ST_X · ST_Y", "Output and coordinates"),
    ]),
    ("Measure", "SQL", [
        ("ST_Area · ST_Length · ST_Perimeter", "Size of features"),
        ("ST_Distance · ST_DistanceSpheroid", "Planar and ellipsoidal distance"),
        ("ST_Extent · ST_Envelope · ST_NPoints", "Bounds and vertex counts"),
    ]),
    ("Spatial relationships", "SQL", [
        ("ST_Intersects · ST_Within · ST_Contains · ST_Touches", "Predicates for joins and filters"),
        ("ST_DWithin", "Within a distance, index aware"),
    ]),
    ("Geometry processing", "SQL", [
        ("ST_Buffer · ST_Union · ST_Collect", "Buffer and merge"),
        ("ST_Intersection · ST_SymDifference", "Overlay"),
        ("ST_Centroid · ST_PointOnSurface · ST_ConcaveHull", "Representative points, hulls"),
        ("ST_IsValid · ST_IsValidReason · ST_MakeValid", "Find and repair invalid geometry"),
        ("ST_Simplify · ST_SimplifyPreserveTopology · ST_CoverageSimplify", "Simplify lines and polygons"),
        ("ST_SnapToGrid · ST_ReducePrecision · ST_Subdivide", "Precision and subdivision"),
    ]),
    ("Indexing and performance", "SQL", [
        ("USING GIST · USING SPGIST · USING BRIN", "Index types for CREATE INDEX"),
        ("CLUSTER · ANALYZE · EXPLAIN", "Order rows, refresh stats, read plans"),
        ("ST_GeoHash · ST_Hilbert", "Spatial sort keys"),
    ]),
    ("In-database analytics", "SQL", [
        ("ST_ClusterDBSCAN · ST_ClusterKMeans", "Clustering with window functions"),
        ("ST_HexagonGrid · h3_lat_lng_to_cell", "Hexagon and H3 grids"),
        ("OVER · PARTITION BY", "Window functions"),
    ]),
    ("DuckDB and GeoParquet", "DuckDB", [
        ("read_parquet · COPY", "Query and write GeoParquet files"),
        ("USING RTREE", "R-tree index in DuckDB"),
    ]),
]

NOTEBOOK_LIST = [
    ("06_01_01", "Spatial Database", "Concepts and backends"),
    ("06_01_02", "Environment Setup", "Docker PostGIS, SpatiaLite, DuckDB"),
    ("06_01_03", "File-based Spatial Databases", "GeoPackage and SpatiaLite"),
    ("06_02_01", "Loading and Exporting Data", "Files to tables and back"),
    ("06_02_02", "Spatial SQL Fundamentals", "Constructors, measures, transforms"),
    ("06_02_03", "Spatial Relationships and Joins", "Predicates, joins, nearest"),
    ("06_02_04", "Geometry Processing", "Overlay, validity, simplify"),
    ("06_02_05", "Indexing and Performance", "GiST, BRIN, EXPLAIN, partitions"),
    ("06_03_01", "Rasters in PostGIS", "Raster SQL, zonal statistics"),
    ("06_03_02", "Network Analysis with pgRouting", "Shortest paths, isochrones"),
    ("06_03_03", "Spatiotemporal Data", "Measures and trajectories"),
    ("06_03_04", "In-database Spatial Analytics", "Clusters, hexagons, H3"),
    ("06_03_05", "DuckDB Spatial and GeoParquet", "Columnar spatial SQL"),
    ("06_03_06", "Big Data and Cloud Spatial Engines", "Sedona, BigQuery, Snowflake"),
    ("06_03_07", "Analyzing NYC Taxi Trips", "End-to-end DuckDB project"),
]

IMPORTS = ('from geolattice import geolatticedb as pdb<br>'
           'db = pdb.connect("postgis", host="localhost", dbname="gis", user="postgres")<br>'
           'gdf = db.query("SELECT name, ST_Area(geom) AS area FROM counties")')

CSS = sa.CSS + """
.t { grid-template-columns: 92px 1fr; }
"""

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small>{d}</small></div></div>' for n, ti, d in NOTEBOOK_LIST)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Spatial Databases <span>with GeoLattice</span></h1>
<div class="sub">The <b>geolattice.geolatticedb</b> API and the spatial SQL the tutorials use, with the notebooks that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · PYTHON API &amp; EXTENSIONS", "geolatticedb, raster, routing &amp; tiles", PYTHON)}
{sa.column("PART B · SPATIAL SQL", "PostGIS, SpatiaLite &amp; DuckDB SQL", SQL)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/06_spatialDB · {len(NOTEBOOK_LIST)} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>SPATIAL DATABASES WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>pdb.query(db, sql) style calls work for the core methods · SQL runs in db.query(...)</span></div>
</div></body></html>"""


def check_names():
    """Python names must exist in geolatticedb; SQL names must appear in the notebooks' SQL."""
    from geolattice import geolatticedb as pdb

    objects = {"pdb": pdb, "db": pdb.SpatialDB}
    db_attrs = {"db.raster": "_Raster", "db.network": "_Network", "db.tiles": "_Tiles"}
    code = ""
    for nb in NOTEBOOKS.glob("*/*.ipynb"):
        cells = json.loads(nb.read_text())["cells"]
        code += "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    bad = []
    for groups in (PYTHON, SQL):
        for _, tag, rows in groups:
            for names, _ in rows:
                for name in names.split(" · "):
                    if tag in objects:
                        ok = hasattr(objects[tag], name)
                    elif tag in db_attrs:
                        ok = hasattr(getattr(pdb, db_attrs[tag]), name)
                    else:
                        ok = name.lower() in code.lower()
                    if not ok:
                        bad.append(f"{tag}: {name}")
    return bad


if __name__ == "__main__":
    missing = check_names()
    if missing:
        raise SystemExit(f"names not found: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1400,3400",
                        "--virtual-time-budget=4000", f"--screenshot={raw}", src.as_uri()],
                       check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
        # crop below the content: the yellow side bar spans exactly the page
        bar = np.asarray(im).astype(int)[:, 6]
        yellow = (bar[:, 0] > 200) & (bar[:, 1] > 170) & (bar[:, 2] < 80)
        last = int(np.flatnonzero(yellow).max()) + 1
        if last >= im.height - 2:
            print("warning: page may be taller than the window; raise --window-size height")
        im.crop((0, 0, im.width, last)).save(OUT)
    shutil.copyfile(OUT, COPY)
    print("wrote", OUT, Image.open(OUT).size, "and", COPY)
