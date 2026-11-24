"""Tests for geolatticedb.

Backends are tested when available:
  * PostGIS: set GEOLATTICE_TEST_PG_DSN (e.g. postgresql://user:pw@localhost:5432/geolattice)
  * SpatiaLite / GeoPackage: need mod_spatialite
  * DuckDB: needs duckdb (+ spatial extension)
"""
import os
import subprocess
import sys
import urllib.request

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

try:
    from geolattice import geolatticedb as pdb
except ImportError:  # running from a checkout without install
    import geolatticedb as pdb

PG_DSN = os.environ.get("GEOLATTICE_TEST_PG_DSN")


def _available(backend, tmp_path):
    try:
        if backend == "postgis":
            if not PG_DSN:
                return None
            return pdb.connect("postgis", dsn=PG_DSN)
        if backend == "spatialite":
            return pdb.connect("spatialite", path=tmp_path / "t.sqlite")
        if backend == "geopackage":
            return pdb.connect("geopackage", path=tmp_path / "t.gpkg")
        return pdb.connect("duckdb", path=None)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{backend} unavailable: {exc}")


@pytest.fixture(params=["postgis", "spatialite", "geopackage", "duckdb"])
def db(request, tmp_path):
    con = _available(request.param, tmp_path)
    if con is None:
        pytest.skip(f"{request.param} not configured")
    yield con
    if con.backend == "postgis":
        for t in ("zones", "pts", "hosp", "zones_buf", "roads", "roads_vertices", "dem", "joined"):
            con.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
    con.close()


@pytest.fixture
def zones():
    # four 1x1 degree squares, 2x2 grid (EPSG:4326)
    polys = [box(x, y, x + 1, y + 1) for x in (0, 1) for y in (0, 1)]
    return gpd.GeoDataFrame({"zone_id": [1, 2, 3, 4], "name": list("abcd")}, geometry=polys, crs=4326)


@pytest.fixture
def pts():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.05, 1.95, size=(40, 2))
    return gpd.GeoDataFrame({"pid": range(40), "v": rng.normal(size=40)},
                            geometry=[Point(x, y) for x, y in xy], crs=4326)


@pytest.fixture
def hosp():
    return gpd.GeoDataFrame({"hid": [10, 20, 30], "name": ["h1", "h2", "h3"]},
                            geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(1.5, 0.5)], crs=4326)


# ---------------------------------------------------------------- core
def test_load_and_query_roundtrip(db, zones):
    db.load(zones, "zones")
    assert "zones" in db.tables()
    assert db.geometry_column("zones") == "geom"
    assert db.srid("zones") == 4326
    out = db.query("SELECT * FROM zones ORDER BY zone_id")
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) == 4 and out.crs.to_epsg() == 4326
    assert out.geometry.area.round(6).tolist() == [1.0] * 4
    # non-spatial query returns a plain DataFrame
    df = db.query("SELECT count(*) AS n FROM zones")
    assert not isinstance(df, gpd.GeoDataFrame) and int(df.iloc[0, 0]) == 4


def test_load_reproject_and_append(db, zones):
    db.load(zones.to_crs(3857), "zones", srid=4326)
    assert db.srid("zones") == 4326
    db.load(zones, "zones", if_exists="append")
    assert len(db.read_table("zones")) == 8
    with pytest.raises(ValueError):
        db.load(zones, "zones", if_exists="fail")


def test_load_from_file(db, zones, tmp_path):
    p = tmp_path / "zones.geojson"
    zones.to_file(p, driver="GeoJSON")
    db.load(str(p), "zones")
    assert len(db.read_table("zones")) == 4


def test_spatial_join(db, zones, pts):
    db.load(zones, "zones")
    db.load(pts, "pts")
    j = db.spatial_join("pts", "zones", predicate="within")
    assert len(j) == 40
    assert set(j.columns) >= {"pid", "v", "zone_id", "name", "geom"}
    expected = gpd.sjoin(pts, zones, predicate="within").sort_values("pid")["zone_id"].tolist()
    assert j.sort_values("pid")["zone_id"].tolist() == expected
    left = db.spatial_join("zones", "pts", predicate="contains", how="left", right_columns=["pid"])
    assert len(left) == 40


def test_spatial_join_out_table(db, zones, pts):
    db.load(zones, "zones")
    db.load(pts, "pts")
    out = db.spatial_join("pts", "zones", out_table="joined")
    assert "joined" in db.tables() and len(out) == 40


def test_nearest(db, pts, hosp):
    db.load(pts, "pts")
    db.load(hosp, "hosp")
    n = db.nearest("pts", "hosp", k=2)
    assert len(n) == 80
    assert {"distance", "rank", "hid"} <= set(n.columns)
    first = n[n["rank"] == 1].sort_values("pid")
    # compare with brute force
    d = np.array([[p.distance(h) for h in hosp.geometry] for p in pts.geometry])
    assert first["hid"].tolist() == hosp["hid"].to_numpy()[d.argmin(1)].tolist()
    lim = db.nearest("pts", "hosp", k=3, max_distance=0.3)
    assert (lim["distance"] <= 0.3 + 1e-9).all()


def test_buffer(db, hosp):
    db.load(hosp.to_crs(3857), "hosp")
    b = db.buffer("hosp", 1000)
    assert len(b) == 3 and b.geom_type.unique().tolist() == ["Polygon"]
    assert abs(b.geometry.area.iloc[0] - np.pi * 1000**2) / (np.pi * 1000**2) < 0.02
    d = db.buffer("hosp", 1000, dissolve=True)
    assert len(d) == 1


def test_create_index_and_explain(db, zones):
    db.load(zones, "zones", index=False)
    idx = db.create_index("zones")
    assert idx
    db.create_index("zones")  # idempotent
    plan = db.explain("SELECT * FROM zones WHERE zone_id = 1")
    assert isinstance(plan, str) and plan
    plan2 = db.explain("SELECT count(*) FROM zones", analyze=True)
    assert plan2


def test_module_level_api(db, zones):
    pdb.load(db, zones, "zones")
    assert len(pdb.query(db, "SELECT * FROM zones")) == 4
    assert len(pdb.buffer(db, "zones", 0.1)) == 4
    assert pdb.explain(db, "SELECT * FROM zones")


# -------------------------------------------------------------- raster
def test_zonal_stats(db, zones, tmp_path):
    if db.backend != "postgis":
        with pytest.raises(NotImplementedError):
            db.raster.zonal_stats("dem", "zones")
        return
    pytest.importorskip("rasterio")
    arr = np.arange(200 * 200, dtype="float32").reshape(200, 200)
    path = tmp_path / "dem.tif"
    # written in a subprocess: rasterio and pyogrio wheels each bundle GDAL
    script = (
        "import numpy as np, rasterio, sys\n"
        "from rasterio.transform import from_origin\n"
        "a = np.arange(200*200, dtype='float32').reshape(200, 200)\n"
        "with rasterio.open(sys.argv[1], 'w', driver='GTiff', height=200, width=200, count=1,"
        " dtype='float32', crs='EPSG:4326', transform=from_origin(0, 2, 0.01, 0.01), nodata=-9999) as d:\n"
        "    d.write(a, 1)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    db.load(str(path), "dem", srid=4326)
    db.load(zones, "zones")
    zs = db.raster.zonal_stats("dem", "zones")
    assert len(zs) == 4 and {"mean", "count", "min", "max"} <= set(zs.columns)
    assert zs["count"].sum() == 200 * 200
    # zone 1 = box(0,0,1,1): bottom-left quarter (rows 100-199, cols 0-99)
    expect = arr[100:, :100].mean()
    got = zs.loc[zs.zone_id == 1, "mean"].iloc[0]
    assert abs(got - expect) < 1e-3
    vals = db.raster.value_at("dem", "zones")  # polygons sampled at a point on surface
    assert len(vals) == 4 and vals["value"].notna().all()


# ------------------------------------------------------------- network
@pytest.fixture
def roads():
    # 3x3 grid of nodes, unit spacing, plus a long diagonal shortcut
    lines = []
    for i in range(3):
        for j in range(2):
            lines.append(LineString([(j, i), (j + 1, i)]))
            lines.append(LineString([(i, j), (i, j + 1)]))
    return gpd.GeoDataFrame({"name": [f"r{k}" for k in range(len(lines))]}, geometry=lines, crs=3857)


def test_network(db, roads):
    if db.backend != "postgis":
        with pytest.raises(NotImplementedError):
            db.network.prepare("roads")
        return
    db.load(roads, "roads")
    vt = db.network.prepare("roads")
    assert vt == "public.roads_vertices"
    assert int(db.query("SELECT count(*) FROM roads_vertices").iloc[0, 0]) == 9
    path = db.network.shortest_path("roads", source=(0, 0), target=(2, 2), srid=3857)
    assert len(path) == 4 and abs(path["agg_cost"].iloc[-1] - 3.0) < 1e-9
    iso = db.network.isochrone("roads", source=(0, 0), costs=[1, 2.5], srid=3857, hull="convex")
    assert len(iso) == 2 and iso["n_nodes"].tolist() == [6, 3]
    pts = db.network.isochrone("roads", source=(0, 0), costs=2, srid=3857, hull="points")
    assert len(pts) == 6


# --------------------------------------------------------------- tiles
def test_tiles(db, zones):
    db.load(zones, "zones")
    if db.backend not in ("postgis", "duckdb"):
        with pytest.raises(NotImplementedError):
            db.tiles.mvt("zones", 0, 0, 0)
        return
    tile = db.tiles.mvt("zones", 0, 0, 0)
    assert isinstance(tile, bytes) and len(tile) > 50
    empty = db.tiles.mvt("zones", 10, 0, 0)  # far away
    assert empty == b""
    b = db.tiles.bounds("zones")
    assert np.allclose(b, (0, 0, 2, 2))
    server = db.tiles.serve(["zones"], port=0)
    try:
        base = server.url
        body = urllib.request.urlopen(base + "zones/0/0/0.pbf").read()
        assert body == tile
        assert b"maplibre" in urllib.request.urlopen(base).read()
        assert b"tilejson" in urllib.request.urlopen(base + "tables.json").read()
    finally:
        db.tiles.stop()


def test_bad_backend():
    with pytest.raises(ValueError):
        pdb.connect("oracle")
