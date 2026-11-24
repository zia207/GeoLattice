"""Tests for geolattice.web and the cloud-native vector writers."""

import http.server
import io
import json
import threading

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import geolattice
import geolattice.web as web
from geolattice.vector import open_cloud_native_vector_data, read_geoparquet, write_geoparquet


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def utm_raster(tmp_path):
    """A 600 × 500 float raster in UTM 18N near Syracuse, NY, with a no-data corner."""
    path = tmp_path / "ndvi.tif"
    yy, xx = np.mgrid[0:500, 0:600]
    data = (np.sin(xx / 40.0) * np.cos(yy / 50.0)).astype("float32")
    data[:40, :40] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=600, height=500, count=1, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(400_000, 4_775_000, 30, 30), nodata=-9999) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def rgb_raster(tmp_path):
    path = tmp_path / "rgb.tif"
    rng = np.random.default_rng(0)
    with rasterio.open(path, "w", driver="GTiff", width=300, height=300, count=3, dtype="uint16",
                       crs="EPSG:32618", transform=from_origin(400_000, 4_775_000, 30, 30)) as dst:
        dst.write(rng.integers(100, 3000, (3, 300, 300), dtype="uint16"))
    return path


@pytest.fixture
def points():
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-77, -75, 400), rng.uniform(42, 44, 400)
    return gpd.GeoDataFrame({"name": [f"p{i}" for i in range(400)], "group": rng.choice(["a", "b"], 400),
                             "value": rng.normal(size=400)},
                            geometry=[Point(x, y) for x, y in zip(lon, lat)], crs=4326)


# ---------------------------------------------------------------- tile math
def test_tile_math_matches_mercantile():
    mercantile = pytest.importorskip("mercantile")
    rng = np.random.default_rng(2)
    for lon, lat, z in zip(rng.uniform(-179, 179, 50), rng.uniform(-80, 80, 50), rng.integers(0, 18, 50)):
        t = web.lonlat_to_tile(lon, lat, int(z))
        m = mercantile.tile(lon, lat, int(z))
        assert (t.x, t.y, t.z) == (m.x, m.y, m.z)
        assert np.allclose(web.tile_bounds(*t), tuple(mercantile.bounds(m)))
        assert np.allclose(web.tile_bounds(*t, crs="EPSG:3857"), tuple(mercantile.xy_bounds(m)))
    ours = set(web.tiles_for_bounds(-76.3, 42.9, -76.0, 43.2, zooms=[8, 11]))
    theirs = {(t.x, t.y, t.z) for t in mercantile.tiles(-76.3, 42.9, -76.0, 43.2, zooms=[8, 11])}
    assert {(t.x, t.y, t.z) for t in ours} == theirs


def test_resolution_and_zoom():
    assert web.tile_resolution(0) == pytest.approx(156543.0339, rel=1e-6)
    z = web.zoom_for_resolution(30, latitude=43)
    assert web.tile_resolution(z, latitude=43) <= 30 < web.tile_resolution(z - 1, latitude=43)
    tj = web.tilejson("x", "https://h/{z}/{x}/{y}.png", bounds=(0, 0, 1, 1), minzoom=3, maxzoom=9, center=(0.5, 0.5))
    assert tj["tilejson"] == "3.0.0" and tj["center"] == [0.5, 0.5, 3]


# ---------------------------------------------------------------- cloud access
@pytest.mark.parametrize("url, expected", [
    ("s3://bucket/a/b.tif", "/vsis3/bucket/a/b.tif"),
    ("gs://bucket/b.tif", "/vsigs/bucket/b.tif"),
    ("az://container/b.tif", "/vsiaz/container/b.tif"),
    ("https://host/x/b.tif", "/vsicurl/https://host/x/b.tif"),
    ("https://host/d.zip!/dem.tif", "/vsizip//vsicurl/https://host/d.zip/dem.tif"),
    ("/vsis3/already", "/vsis3/already"),
    ("local/file.tif", "local/file.tif"),
])
def test_to_vsi_path(url, expected):
    assert web.to_vsi_path(url) == expected


def test_to_vsi_path_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        web.to_vsi_path("ftp://host/file.tif")


def test_cloud_env_sets_and_restores_gdal_options():
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.SetConfigOption("AWS_REGION", None)
    opts = web.cloud_options(anonymous=True, region="us-west-2", requester_pays=True)
    assert opts["AWS_NO_SIGN_REQUEST"] == "YES" and opts["AWS_REQUEST_PAYER"] == "requester"
    with web.cloud_env(anonymous=True, region="us-west-2") as active:
        assert gdal.GetConfigOption("AWS_REGION") == "us-west-2"
        assert active["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert gdal.GetConfigOption("AWS_REGION") is None


def test_remote_info_reports_missing_range_support(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 1000)
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        info = web.remote_info(f"{url}/f.bin")
        assert info["status"] == 200 and info["range_requests"] is False and info["size"] == 1000
        assert web.remote_info(f"{url}/missing.bin")["status"] == 404
    finally:
        server.shutdown()
        server.server_close()
    refused = web.remote_info(url + "/f.bin", timeout=2)             # server gone: connection refused
    assert refused["status"] is None and refused["error"]


# ---------------------------------------------------------------- cloud-native vectors
class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server that honours single ``Range: bytes=a-b`` requests."""

    def log_message(self, *args):
        pass

    def send_head(self):
        import os
        import re

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        handle = open(path, "rb")
        match = re.match(r"bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
        if match:
            a, b = match.groups()
            start, end = (size - int(b), size - 1) if a == "" else (int(a), min(int(b) if b else size - 1, size - 1))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            handle.seek(start)
            self._left = end - start + 1
        else:
            self.send_response(200)
            self._left = size
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(self._left))
        self.end_headers()
        return handle

    def copyfile(self, source, destination):
        destination.write(source.read(self._left))


def test_static_server_ranges_counters_and_cors(tmp_path):
    import urllib.request
    (tmp_path / "f.bin").write_bytes(bytes(range(256)) * 40)          # 10,240 bytes
    with web.static_server(tmp_path) as srv:
        info = web.remote_info(srv.url + "/f.bin")
        assert info["range_requests"] is True and info["size"] == 10240
        srv.reset()
        req = urllib.request.Request(srv.url + "/f.bin", headers={"Range": "bytes=100-199"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            assert resp.status == 206 and resp.headers["Access-Control-Allow-Origin"] == "*"
        assert body == (bytes(range(256)) * 40)[100:200]
        assert srv.stats == {"requests": 1, "bytes": 100}
        tail = urllib.request.Request(srv.url + "/f.bin", headers={"Range": "bytes=-10"})
        assert urllib.request.urlopen(tail).read() == (bytes(range(256)) * 40)[-10:]
        multi = urllib.request.Request(srv.url + "/f.bin", headers={"Range": "bytes=0-1,5-6"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(multi)
        assert err.value.code == 416
        with urllib.request.urlopen(srv.url + "/f.bin") as resp:
            assert resp.status == 200 and len(resp.read()) == 10240
    with pytest.raises(NotADirectoryError):
        web.static_server(tmp_path / "f.bin")


def test_read_geoparquet_over_http_with_bbox(points, tmp_path):
    pytest.importorskip("fsspec")
    pytest.importorskip("aiohttp")
    write_geoparquet(tmp_path / "pts.parquet", points, row_group_size=50)
    handler = lambda *a, **k: _RangeHandler(*a, directory=str(tmp_path), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/pts.parquet"
        assert web.remote_info(url)["range_requests"] is True
        query = (-76.5, 42.5, -75.5, 43.5)
        remote = read_geoparquet(url, bbox=query)
        assert len(remote) == int(points.intersects(box(*query)).sum()) > 0
        assert set(read_geoparquet(url, columns=["name"]).columns) == {"name", "geometry"}
        assert web.geoparquet_info(url)["num_rows"] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_geoparquet_round_trip_with_covering_and_bbox(points, tmp_path):
    path = write_geoparquet(tmp_path / "pts.parquet", points, row_group_size=50)
    info = web.geoparquet_info(path)
    assert info["covering"] and info["num_rows"] == 400 and info["num_row_groups"] == 8
    assert info["crs"].to_epsg() == 4326 and info["geometry_types"] == ["Point"]
    full = read_geoparquet(path)
    assert len(full) == 400 and set(full.columns) >= {"name", "group", "value", "geometry"}
    query = (-76.5, 42.5, -75.5, 43.5)
    sub = read_geoparquet(path, bbox=query)
    assert len(sub) == int(points.intersects(box(*query)).sum()) > 0
    # bbox given in UTM is transformed to the file CRS
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 32618, always_xy=True)
    utm = (*t.transform(-76.4, 42.6), *t.transform(-75.6, 43.4))
    assert 0 < len(read_geoparquet(path, bbox=utm, bbox_crs=32618)) <= len(points)


def test_geoparquet_rows_are_hilbert_sorted(points, tmp_path):
    path = write_geoparquet(tmp_path / "pts.parquet", points)
    order = read_geoparquet(path).geometry.hilbert_distance(total_bounds=points.total_bounds).to_numpy()
    assert np.all(np.diff(order) >= 0)
    unsorted = write_geoparquet(tmp_path / "raw.parquet", points, sort_spatially=False)
    assert list(read_geoparquet(unsorted)["name"]) == list(points["name"])


def test_geoparquet_accepts_feature_mappings_and_respects_overwrite(tmp_path):
    rows = [{"geometry": "POINT (1 2)", "a": 1}, {"geometry": {"type": "Point", "coordinates": [3, 4]}, "a": 2}]
    path = write_geoparquet(tmp_path / "f.parquet", rows, crs=4326)
    assert list(read_geoparquet(path)["a"]) == [1, 2]
    with pytest.raises(Exception):
        write_geoparquet(path, rows, crs=4326, overwrite=False)


def test_cloud_native_reader_prunes_geoparquet_row_groups(points, tmp_path):
    import sys

    reader = sys.modules["geolattice.vector.open_cloud_native_vector_data"]
    path = write_geoparquet(tmp_path / "pts.parquet", points, row_group_size=25)
    query = (-76.6, 42.6, -76.2, 43.0)
    with open_cloud_native_vector_data(path) as ds:
        full = ds.layer.to_dataframe()
        part = ds.layer.to_dataframe(bbox=query)
    expected = full[full.intersects(box(*query))]
    assert len(part) == len(expected) > 0
    assert sorted(part["fid"]) == sorted(expected["fid"])          # ids match an unfiltered read
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path).metadata
    info = reader._inspect_parquet(path)
    covering = reader._bbox_covering_paths(info)
    assert covering == {"xmin": "bbox.xmin", "ymin": "bbox.ymin", "xmax": "bbox.xmax", "ymax": "bbox.ymax"}
    groups, starts = reader._row_groups_for_bbox(meta, covering, query)
    assert 0 < len(groups) < meta.num_row_groups and starts[0] == 0 and len(starts) == meta.num_row_groups


def test_flatgeobuf_and_pmtiles(points, tmp_path):
    fgb = web.write_flatgeobuf(tmp_path / "pts.fgb", points)
    back = gpd.read_file(fgb, bbox=(-76.5, 42.5, -75.5, 43.5))
    assert 0 < len(back) < 400
    pm = web.write_pmtiles(tmp_path / "pts.pmtiles", points, minzoom=0, maxzoom=6, layer="sites")
    with open_cloud_native_vector_data(pm) as ds:
        assert ds.format_name == "PMTiles"
        assert [layer.name for layer in ds.layers] == ["sites"]
    with pytest.raises(ValueError):
        web.write_pmtiles(tmp_path / "bad.pmtiles", points, minzoom=8, maxzoom=2)
    # Over HTTP the archive opens from its header and directories; features are not counted up front.
    size = (tmp_path / "pts.pmtiles").stat().st_size
    with web.static_server(tmp_path) as srv, web.cloud_env():
        with open_cloud_native_vector_data(f"{srv.url}/pts.pmtiles") as ds:
            assert ds.layer.feature_count is None
            assert ds.details["header"]["max_zoom"] == 6
        assert srv.stats["bytes"] < size


# ---------------------------------------------------------------- COG
def test_to_cog_produces_a_valid_cog(utm_raster, tmp_path):
    report_plain = web.validate_cog(utm_raster)
    assert not report_plain.valid and any("striped" in e for e in report_plain.errors)
    cog = web.to_cog(utm_raster, tmp_path / "cog.tif", blocksize=256)
    report = web.validate_cog(cog)
    assert report.valid, str(report)
    assert report.details["layout"] == "COG" and report.details["tiled"]
    assert report.details["block_size"] == (256, 256) and report.details["overview_count"] >= 1
    ifd = report.details["ifd_offsets"]
    assert ifd == sorted(ifd)
    assert bool(report) and "valid" in str(report)


def test_validate_cog_flags_external_overviews(utm_raster, tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    tiled = tmp_path / "tiled.tif"
    gdal.Translate(str(tiled), str(utm_raster), creationOptions=["TILED=YES", "COMPRESS=DEFLATE"])
    ds = gdal.Open(str(tiled), gdal.GA_ReadOnly)          # read-only: overviews go to an .ovr file
    ds.BuildOverviews("AVERAGE", [2, 4])
    ds = None
    report = web.validate_cog(tiled)
    assert not report.valid and any(".ovr" in e for e in report.errors)


def test_to_cog_web_optimized(utm_raster, tmp_path):
    cog = web.to_cog(utm_raster, tmp_path / "web.tif", web_optimized=True)
    with rasterio.open(cog) as src:
        assert src.crs.to_epsg() == 3857
    assert web.validate_cog(cog).valid


# ---------------------------------------------------------------- tiles
def _png(data):
    from PIL import Image
    return Image.open(io.BytesIO(data))


def test_render_tile_inside_and_outside(utm_raster):
    info = web.raster_web_info(utm_raster)
    assert info["crs"] == "EPSG:32618" and info["maxzoom"] >= 12
    z = 11
    t = web.lonlat_to_tile(*info["center"], z)
    img = _png(web.render_tile(utm_raster, *t, colormap="RdYlGn", rescale=(-1, 1)))
    assert img.size == (256, 256) and img.mode == "RGBA"
    alpha = np.asarray(img)[..., 3]
    assert alpha.max() == 255
    assert web.render_tile(utm_raster, 0, 0, 5) is None              # far from the data
    west, north = info["bounds"][0], info["bounds"][3]
    corner = web.lonlat_to_tile(west + 1e-4, north - 1e-4, 12)        # spans the corner and real data
    corner_img = np.asarray(_png(web.render_tile(utm_raster, *corner, rescale=(-1, 1))))
    assert corner_img[..., 3].min() == 0 and corner_img[..., 3].max() == 255
    # a tile lying wholly inside the 1.2 km no-data square renders nothing
    with rasterio.open(utm_raster) as src:
        from rasterio.warp import transform as warp
        lon, lat = warp(src.crs, "EPSG:4326", [400_000 + 600], [4_775_000 - 600])
    inside_nodata = web.lonlat_to_tile(lon[0], lat[0], 17)
    assert web.render_tile(utm_raster, *inside_nodata) is None


def test_render_tile_reads_overviews_for_coarse_tiles(utm_raster, tmp_path):
    from geolattice.web.tiles import _overview_level

    cog = tmp_path / "cog.tif"
    web.to_cog(utm_raster, cog, blocksize=256)
    with rasterio.open(cog) as src:
        factors = src.overviews(1)
        assert factors                                        # the COG has internal overviews
        center = web.raster_web_info(cog)["center"]
        coarse = web.tile_bounds(*web.lonlat_to_tile(*center, 8)[:2], 8, crs="EPSG:3857")
        fine = web.tile_bounds(*web.lonlat_to_tile(*center, 14)[:2], 14, crs="EPSG:3857")
        assert _overview_level(src, coarse, 256) == len(factors) - 1
        assert _overview_level(src, fine, 256) is None
    t = web.lonlat_to_tile(*center, 9)
    full = _png(web.render_tile(utm_raster, t.x, t.y, t.z, rescale=(-1, 1)))
    fast = _png(web.render_tile(cog, t.x, t.y, t.z, rescale=(-1, 1)))
    a, b = np.asarray(full).astype(float), np.asarray(fast).astype(float)
    inside = (a[..., 3] > 0) & (b[..., 3] > 0)
    assert inside.sum() > 1000 and np.abs(a[inside, :3] - b[inside, :3]).mean() < 10   # same picture


def test_render_tile_rgb_and_errors(rgb_raster):
    info = web.raster_web_info(rgb_raster)
    t = web.lonlat_to_tile(*info["center"], 12)
    rescale = web.suggest_rescale(rgb_raster)
    assert len(rescale) == 3 and all(lo < hi for lo, hi in rescale)
    assert _png(web.render_tile(rgb_raster, *t, rescale=rescale)).mode == "RGBA"
    with pytest.raises(ValueError):
        web.render_tile(rgb_raster, *t, bands=[1, 2])
    with pytest.raises(ValueError):
        web.render_tile(rgb_raster, *t, rescale=[(0, 1), (0, 1)])


def test_make_xyz_tiles_writes_pyramid_and_viewer(utm_raster, tmp_path):
    out = web.make_xyz_tiles(utm_raster, tmp_path / "tiles", zooms=[9, 10, 11], colormap="viridis")
    files = sorted((tmp_path / "tiles").rglob("*.png"))
    assert len(files) == out["tiles"] > 0
    expected = len(list(web.tiles_for_bounds(*web.raster_web_info(utm_raster)["bounds"], zooms=[9, 10, 11])))
    assert out["tiles"] + out["empty"] == expected
    assert (tmp_path / "tiles" / "index.html").exists()
    assert json.loads((tmp_path / "tiles" / "tilejson.json").read_text())["minzoom"] == 9


def test_leaflet_html_escapes_title():
    page = web.leaflet_html(title="<b>x</b>", rasters=[{"name": "r", "url": "{z}/{x}/{y}.png"}])
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "leaflet" in page and '"r"' in page


# ---------------------------------------------------------------- server
@pytest.fixture
def client(utm_raster, points, tmp_path):
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    parquet = write_geoparquet(tmp_path / "pts.parquet", points)
    app = web.create_app(rasters={"ndvi": {"path": utm_raster, "colormap": "RdYlGn", "rescale": (-1, 1)}},
                         vectors={"points": parquet, "frame": points.to_crs(32618)}, default_limit=50)
    return TestClient(app)


def test_server_ogc_api_features(client, points):
    landing = client.get("/").json()
    assert {link["rel"] for link in landing["links"]} >= {"conformance", "data", "tiles"}
    assert "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core" in client.get("/conformance").json()["conformsTo"]
    cols = client.get("/collections").json()["collections"]
    assert [c["id"] for c in cols] == ["points", "frame"]
    assert client.get("/collections/nope").status_code == 404

    page = client.get("/collections/points/items")
    assert page.headers["content-type"].startswith("application/geo+json")
    body = page.json()
    assert body["numberMatched"] == 400 and body["numberReturned"] == 50
    assert any(link["rel"] == "next" for link in body["links"])
    last = client.get("/collections/points/items?offset=380&limit=50").json()
    assert last["numberReturned"] == 20 and not any(link["rel"] == "next" for link in last["links"])

    query = (-76.5, 42.5, -75.5, 43.5)
    hits = client.get("/collections/points/items", params={"bbox": ",".join(map(str, query)), "limit": 1000}).json()
    assert hits["numberMatched"] == int(points.intersects(box(*query)).sum())
    group = client.get("/collections/points/items?group=a&limit=1000").json()
    assert group["numberMatched"] == int((points["group"] == "a").sum())
    assert client.get("/collections/points/items?bbox=1,2,3").status_code == 400
    assert client.get("/collections/points/items?nosuch=1").status_code == 400
    assert client.get("/collections/points/items?limit=0").status_code == 400

    # the projected layer is served in longitude/latitude
    feature = client.get("/collections/frame/items?limit=1").json()["features"][0]
    lon, lat = feature["geometry"]["coordinates"]
    assert -77 <= lon <= -75 and 42 <= lat <= 44
    one = client.get(f"/collections/frame/items/{feature['id']}")
    assert one.status_code == 200 and one.json()["id"] == feature["id"]
    assert client.get("/collections/frame/items/999999").status_code == 404


def test_server_tiles_and_viewer(client, utm_raster):
    layers = client.get("/tiles").json()["tiles"]
    assert layers[0]["name"] == "ndvi"
    tj = client.get("/tiles/ndvi/tilejson.json").json()
    assert tj["tiles"][0].endswith("/tiles/ndvi/{z}/{x}/{y}.png")
    t = web.lonlat_to_tile(*web.raster_web_info(utm_raster)["center"], 11)
    tile = client.get(f"/tiles/ndvi/{t.z}/{t.x}/{t.y}.png")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    assert _png(tile.content).size == (256, 256)
    assert client.get("/tiles/ndvi/3/0/0.png").status_code == 204
    assert client.get("/tiles/ndvi/3/9/0.png").status_code == 400
    assert client.get("/tiles/nope/3/0/0.png").status_code == 404
    page = client.get("/viewer")
    assert page.status_code == 200 and "L.tileLayer" in page.text and "/collections/points/items" in page.text


def test_server_openapi_and_service_desc(client):
    landing = client.get("/").json()
    desc = [link for link in landing["links"] if link["rel"] == "service-desc"]
    assert desc and desc[0]["href"].endswith("/api")
    doc = client.get("/api")
    assert doc.status_code == 200 and doc.headers["content-type"].startswith("application/vnd.oai.openapi+json")
    paths = doc.json()["paths"]
    assert {"/", "/api", "/conformance", "/collections", "/collections/points/items",
            "/collections/frame/items/{featureId}", "/tiles/ndvi/{z}/{x}/{y}.png"} <= set(paths)
    names = {p["name"] for p in paths["/collections/points/items"]["get"]["parameters"]}
    assert names == {"bbox", "datetime", "limit", "offset"}


def test_server_datetime_filter(points):
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    import pandas as pd
    from starlette.testclient import TestClient

    timed = points.assign(observed=pd.date_range("2024-01-01", periods=len(points), freq="D"))
    client = TestClient(web.create_app(vectors={"timed": timed, "plain": points}))
    extent = client.get("/collections/timed").json()["extent"]
    assert extent["temporal"]["interval"] == [["2024-01-01T00:00:00Z", "2025-02-03T00:00:00Z"]]
    one = client.get("/collections/timed/items?datetime=2024-01-10T00:00:00Z").json()
    assert one["numberMatched"] == 1 and one["features"][0]["properties"]["name"] == "p9"
    assert one["features"][0]["properties"]["observed"] == "2024-01-10T00:00:00"     # timestamps encode as ISO 8601
    span = client.get("/collections/timed/items?datetime=2024-01-01T00:00:00Z/2024-01-31T23:59:59Z").json()
    assert span["numberMatched"] == 31
    assert client.get("/collections/timed/items?datetime=../2024-01-05T00:00:00Z").json()["numberMatched"] == 5
    assert client.get("/collections/timed/items?datetime=2025-02-01T00:00:00Z/..").json()["numberMatched"] == 3
    # collections without a datetime column accept the parameter and ignore it
    assert client.get("/collections/plain/items?datetime=2024-01-10T00:00:00Z").json()["numberMatched"] == 400
    for bad in ["yesterday", "../..", "2024-02-01T00:00:00Z/2024-01-01T00:00:00Z"]:
        assert client.get("/collections/timed/items", params={"datetime": bad}).status_code == 400


def test_server_feature_encoding_matches_geopandas(points):
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    frame = points.assign(geometry=points.buffer(0.01), value=points["value"].where(points.index % 7 != 0))
    client = TestClient(web.create_app(vectors={"f": frame}))
    served = client.get("/collections/f/items?limit=1000").json()["features"]
    expected = json.loads(frame.to_crs(4326).to_json(drop_id=False, na="null"))["features"]
    assert [f["id"] for f in served] == [f["id"] for f in expected]
    assert [f["properties"] for f in served] == [f["properties"] for f in expected]      # NaN -> null
    assert all(np.allclose(np.array(a["geometry"]["coordinates"]), np.array(b["geometry"]["coordinates"]), atol=1e-12)
               for a, b in zip(served, expected))
    one = client.get(f"/collections/f/items/{expected[7]['id']}").json()
    assert one["properties"]["value"] is None and one["links"][0]["rel"] == "self"


def test_server_checks_raster_config_at_start(rgb_raster):
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    with pytest.raises(ValueError, match="rescale has 2 pairs for 3 bands"):
        web.create_app(rasters={"rgb": {"path": rgb_raster, "rescale": [(0, 1), (0, 2)]}})
    with pytest.raises(ValueError, match="do not fit"):
        web.create_app(rasters={"rgb": {"path": rgb_raster, "bands": [4], "rescale": (0, 1)}})
    # one pair given as a list applies to all three bands
    client = TestClient(web.create_app(rasters={"rgb": {"path": rgb_raster, "rescale": [(100, 3000)]}}))
    t = web.lonlat_to_tile(*web.raster_web_info(rgb_raster)["center"], 12)
    assert client.get(f"/tiles/rgb/{t.z}/{t.x}/{t.y}.png").status_code == 200


def test_server_feature_endpoints_run_off_the_event_loop(points):
    pytest.importorskip("starlette")
    import inspect

    app = web.create_app(vectors={"p": points})
    handlers = {route.path: route.endpoint for route in app.routes}
    assert not inspect.iscoroutinefunction(handlers["/collections/{cid}/items"])
    assert not inspect.iscoroutinefunction(handlers["/collections/{cid}/items/{fid}"])


def test_start_server_runs_in_background(points):
    pytest.importorskip("uvicorn")
    import urllib.request

    with web.start_server(web.create_app(vectors={"p": points})) as srv:
        assert srv.port > 0
        body = json.loads(urllib.request.urlopen(srv.url + "/collections/p/items?limit=3").read())
        assert body["numberReturned"] == 3
    with pytest.raises(OSError):
        urllib.request.urlopen(srv.url + "/", timeout=2)


def test_subprocess_static_server_serves_rasterio(utm_raster):
    import urllib.request

    with web.static_server(utm_raster.parent, subprocess=True) as srv:
        url = f"{srv.url}/{utm_raster.name}"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206 and len(resp.read()) == 100
        assert srv.stats == {"requests": 1, "bytes": 100}
        srv.reset()
        with rasterio.open(web.to_vsi_path(url)) as src:            # would deadlock in-process
            assert src.read(1, window=((0, 10), (0, 10))).shape == (10, 10)
        assert srv.stats["requests"] >= 1


def test_web_is_a_lazy_top_level_subpackage():
    assert geolattice.web is web and "web" in dir(geolattice)
    missing = [name for name in web.__all__ if not hasattr(web, name)]
    assert missing == []
