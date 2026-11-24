"""
Tests for cloud-native vector, lakehouse, and GeoParquet profile readers.
"""

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from shapely.geometry import Point
from shapely.wkb import dumps as dump_wkb

from geolattice.vector.open_cloud_native_vector_data import (
    CloudNativeOpenError,
    logical_geometry_kind,
    open_cloud_native_vector,
    open_cloud_native_vector_data,
    supported_cloud_native_formats,
    tileid_to_zxy,
    zxy_to_tileid,
)

REQUIRED_FORMATS = {
    "geoparquet",
    "geoarrow",
    "flatgeobuf",
    "pmtiles",
    "geopackage",
    "copc",
    "iceberg",
    "delta",
    "parquet-geometry",
    "fiboa",
    "stac-geoparquet",
    "overture",
    "geojsonseq",
    "h3-s2",
    "tiledb",
    "lance",
}


def _pyarrow():
    return pytest.importorskip("pyarrow")


def _wkb(x_coord, y_coord):
    return dump_wkb(Point(x_coord, y_coord))


def _write_geoparquet(path, columns, metadata=None, geo=None):
    pa = _pyarrow()
    import pyarrow.parquet as pq

    arrays = []
    fields = []
    for name, values in columns.items():
        if name == "geometry" or isinstance(values[0], (bytes, bytearray)):
            arrays.append(pa.array(values, type=pa.binary()))
            fields.append(pa.field(name, pa.binary()))
        elif isinstance(values[0], (dict, list)):
            arrays.append(pa.array(values))
            fields.append(pa.field(name, arrays[-1].type))
        else:
            arrays.append(pa.array(values))
            fields.append(pa.field(name, arrays[-1].type))
    file_meta = {}
    if geo is not None:
        file_meta[b"geo"] = json.dumps(geo).encode("utf-8")
    if metadata:
        for key, value in metadata.items():
            file_meta[key.encode("utf-8") if isinstance(key, str) else key] = (
                value.encode("utf-8") if isinstance(value, str) else value
            )
    schema = pa.schema(fields, metadata=file_meta or None)
    table = pa.table(dict(zip(columns, arrays)), schema=schema)
    pq.write_table(table, path)


def _point_geo(xs, ys):
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "crs": {"id": {"authority": "EPSG", "code": 4326}},
            }
        },
    }


def _ogr():
    pytest.importorskip("osgeo")
    from osgeo import gdal, ogr, osr

    gdal._UseExceptions()
    ogr._UseExceptions()
    osr._UseExceptions()
    return gdal, ogr, osr


def _write_ogr_points(driver_name, path, layer_name="layer"):
    _, ogr, osr = _ogr()
    driver = ogr.GetDriverByName(driver_name)
    if driver is None or driver.GetMetadataItem("DCAP_CREATE") != "YES":
        pytest.skip(f"{driver_name} cannot create datasets in this GDAL build")
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    try:
        datasource = driver.CreateDataSource(str(path))
    except RuntimeError as exc:
        pytest.skip(f"Cannot create {driver_name} dataset: {exc}")
    try:
        layer = datasource.CreateLayer(layer_name, spatial_ref, ogr.wkbPoint)
    except RuntimeError as exc:
        datasource = None
        pytest.skip(f"{driver_name} cannot create a point layer: {exc}")
    layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
    for label, x_coord, y_coord in (("alpha", 1.0, 2.0), ("beta", 3.0, 4.0)):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("name", label)
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint_2D(x_coord, y_coord)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
    layer = None
    datasource = None


def _encode_varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_directory(entries):
    """Independent PMTiles directory encoder used to test the reader."""
    buffer = bytearray(_encode_varint(len(entries)))
    last = 0
    for tile_id, _offset, _length, _run in entries:
        buffer.extend(_encode_varint(tile_id - last))
        last = tile_id
    for _tile_id, _offset, _length, run in entries:
        buffer.extend(_encode_varint(run))
    for _tile_id, _offset, length, _run in entries:
        buffer.extend(_encode_varint(length))
    next_byte = 0
    for index, (_tile_id, offset, length, _run) in enumerate(entries):
        if index > 0 and offset == next_byte:
            buffer.extend(_encode_varint(0))
        else:
            buffer.extend(_encode_varint(offset + 1))
        next_byte = offset + length
    return bytes(buffer)


def _write_pmtiles(path, tiles, metadata=None, tile_type=0):
    """Write an uncompressed PMTiles v3 archive. tiles is {tile_id: bytes}."""
    ordered = sorted(tiles)
    payload = b"".join(tiles[tile_id] for tile_id in ordered)
    entries = []
    cursor = 0
    for tile_id in ordered:
        blob = tiles[tile_id]
        entries.append((tile_id, cursor, len(blob), 1))
        cursor += len(blob)
    directory = _encode_directory(entries)
    meta = json.dumps(metadata or {"name": "demo"}).encode("utf-8")
    header = bytearray(127)
    header[0:7] = b"PMTiles"
    header[7] = 3
    root_offset = 127
    meta_offset = root_offset + len(directory)
    tile_offset = meta_offset + len(meta)
    struct.pack_into("<QQ", header, 8, root_offset, len(directory))
    struct.pack_into("<QQ", header, 24, meta_offset, len(meta))
    struct.pack_into("<QQ", header, 40, tile_offset, 0)
    struct.pack_into("<QQ", header, 56, tile_offset, len(payload))
    struct.pack_into("<QQQ", header, 72, len(ordered), len(ordered), len(ordered))
    header[96] = 1  # clustered
    header[97] = 1  # internal compression: none
    header[98] = 1  # tile compression: none
    header[99] = tile_type
    header[100] = 0
    header[101] = 2
    struct.pack_into("<ii", header, 102, int(-180 * 1e7), int(-85 * 1e7))
    struct.pack_into("<ii", header, 110, int(180 * 1e7), int(85 * 1e7))
    header[118] = 1
    struct.pack_into("<ii", header, 119, 0, 0)
    path.write_bytes(bytes(header) + directory + meta + payload)


def _write_copc(path, points):
    wkt = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
    ).encode("ascii") + b"\x00"
    record_length = 30
    scale = 0.001
    point_blob = bytearray()
    for x_coord, y_coord, z_coord in points:
        point_blob.extend(
            struct.pack(
                "<3iHBBBBhHd",
                int(round(x_coord / scale)),
                int(round(y_coord / scale)),
                int(round(z_coord / scale)),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
            )
        )
    info_payload = struct.pack(
        "<5dQQ2d11d",
        0.0,
        0.0,
        0.0,
        50.0,
        1.0,
        0,
        32,
        0.0,
        0.0,
        *([0.0] * 11),
    )
    vlr_copc = bytearray(54 + len(info_payload))
    vlr_copc[2:6] = b"copc"
    struct.pack_into("<HH", vlr_copc, 18, 1, len(info_payload))
    vlr_copc[54:] = info_payload
    vlr_wkt = bytearray(54 + len(wkt))
    vlr_wkt[2:18] = b"LASF_Projection"
    struct.pack_into("<HH", vlr_wkt, 18, 2112, len(wkt))
    vlr_wkt[54:] = wkt
    offset_points = 375 + len(vlr_copc) + len(vlr_wkt)
    hier_offset = offset_points + len(point_blob)
    # Patch the hierarchy location inside the COPC info VLR.
    struct.pack_into("<Q", vlr_copc, 54 + 40, hier_offset)
    struct.pack_into("<Q", vlr_copc, 54 + 48, 32)
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    zs = [item[2] for item in points]
    header = bytearray(375)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 4
    struct.pack_into("<H", header, 94, 375)
    struct.pack_into("<I", header, 96, offset_points)
    struct.pack_into("<I", header, 100, 2)
    header[104] = 6
    struct.pack_into("<H", header, 105, record_length)
    struct.pack_into("<3d", header, 131, scale, scale, scale)
    struct.pack_into("<6d", header, 179, max(xs), min(xs), max(ys), min(ys), max(zs), min(zs))
    struct.pack_into("<Q", header, 247, len(points))
    hierarchy = struct.pack(
        "<4iQii", 0, 0, 0, 0, offset_points, len(point_blob), len(points)
    )
    path.write_bytes(bytes(header) + bytes(vlr_copc) + bytes(vlr_wkt) + bytes(point_blob) + hierarchy)


class TestCatalog:
    """The public catalog covers the requested layouts."""

    def test_formats_cover_cloud_native_and_lakehouse(self):
        found = {item["format_id"] for item in supported_cloud_native_formats()}
        assert REQUIRED_FORMATS <= found

    def test_families_and_access(self):
        by_id = {item["format_id"]: item for item in supported_cloud_native_formats()}
        assert by_id["flatgeobuf"]["spatial_access"] == "http-range"
        assert by_id["pmtiles"]["spatial_access"] == "http-range"
        assert by_id["copc"]["spatial_access"] == "http-range"
        assert by_id["geoparquet"]["family"] == "cloud-native"
        assert by_id["iceberg"]["family"] == "table"
        assert by_id["fiboa"]["family"] == "specification"
        assert by_id["geojsonseq"]["family"] == "other"

    def test_alias(self):
        assert open_cloud_native_vector is not open_cloud_native_vector_data

    def test_package_export(self):
        from geolattice.vector import open_cloud_native_vector_data as exported

        assert exported is open_cloud_native_vector_data


class TestHilbert:
    """PMTiles tile ids follow the spec's Hilbert table."""

    @pytest.mark.parametrize(
        ("z", "x", "y", "tile_id"),
        [(0, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 2), (1, 1, 1, 3), (1, 1, 0, 4), (2, 0, 0, 5)],
    )
    def test_spec_table(self, z, x, y, tile_id):
        assert zxy_to_tileid(z, x, y) == tile_id
        assert tileid_to_zxy(tile_id) == (z, x, y)


class TestLogicalGeometry:
    """Native Parquet GEOMETRY and GEOGRAPHY logical types."""

    def test_geometry_and_geography_labels(self):
        class GeometryType:
            type = "GEOMETRY"

            def to_json(self):
                return '{"Type": "Geometry"}'

        class GeographyType:
            type = "GEOGRAPHY"

            def to_json(self):
                return '{"Type": "Geography"}'

        assert logical_geometry_kind(GeometryType()) == "parquet-geometry"
        assert logical_geometry_kind(GeographyType()) == "parquet-geography"
        assert logical_geometry_kind(None) is None


class TestGeoParquet:
    """GeoParquet, its profiles, and cell-id columns."""

    def test_points_bbox_columns_and_where(self, tmp_path):
        path = tmp_path / "cities.parquet"
        _write_geoparquet(
            path,
            {
                "name": ["alpha", "beta"],
                "geometry": [_wkb(1, 2), _wkb(3, 4)],
            },
            geo=_point_geo([1, 3], [2, 4]),
        )
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "geoparquet"
            assert dataset.schema_name is None
            assert dataset.feature_count == 2
            assert dataset.bounds.minx == pytest.approx(1)
            assert dataset.bounds.maxy == pytest.approx(4)
            assert dataset.crs.to_epsg() == 4326
            assert dataset.geometry_encoding == "WKB"
            features = dataset.read()
            assert features[0].geometry.x == pytest.approx(1)
            assert features[0].properties["name"] == "alpha"
            assert "geometry" not in features[0].properties
            inside = dataset.read(bbox=(0, 0, 1.5, 2.5))
            assert [feature.properties["name"] for feature in inside] == ["alpha"]
            selected = dataset.read(where="name = 'beta'", columns=["name"])
            assert len(selected) == 1
            assert list(selected[0].properties) == ["name"]
            frame = dataset.to_dataframe()
            assert list(frame["name"]) == ["alpha", "beta"]
            assert frame.geometry.iloc[0].y == pytest.approx(2)

    def test_directory_of_parquet_files(self, tmp_path):
        folder = tmp_path / "dataset"
        folder.mkdir()
        for index, name in enumerate(("a.parquet", "b.parquet")):
            _write_geoparquet(
                folder / name,
                {"name": [name], "geometry": [_wkb(index, index + 1)]},
                geo=_point_geo([index], [index + 1]),
            )
        with open_cloud_native_vector_data(folder) as dataset:
            assert dataset.feature_count == 2
            assert sorted(feature.properties["name"] for feature in dataset) == ["a.parquet", "b.parquet"]

    def test_fiboa_profile(self, tmp_path):
        path = tmp_path / "fields.parquet"
        _write_geoparquet(
            path,
            {
                "id": ["field-1"],
                "area": [10.5],
                "determination_datetime": ["2024-01-01T00:00:00Z"],
                "geometry": [_wkb(10, 20)],
            },
            metadata={"fiboa": "1.0.0"},
            geo=_point_geo([10], [20]),
        )
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_name == "GeoParquet"
            assert dataset.schema_name == "fiboa"
            assert dataset.read()[0].properties["id"] == "field-1"

    def test_stac_geoparquet_profile(self, tmp_path):
        path = tmp_path / "items.parquet"
        _write_geoparquet(
            path,
            {
                "id": ["scene-1"],
                "datetime": ["2024-06-01T00:00:00Z"],
                "assets": [{"visual": {"href": "https://example.com/a.tif"}}],
                "geometry": [_wkb(-10, 35)],
            },
            geo=_point_geo([-10], [35]),
        )
        with open_cloud_native_vector_data(path, format="stac-geoparquet") as dataset:
            assert dataset.schema_name == "stac-geoparquet"
            assert dataset.read()[0].properties["assets"]["visual"]["href"].endswith(".tif")

    def test_overture_profile(self, tmp_path):
        path = tmp_path / "buildings.parquet"
        _write_geoparquet(
            path,
            {
                "id": ["bld-1"],
                "sources": [[{"dataset": "osm"}]],
                "names": [{"primary": "Library"}],
                "geometry": [_wkb(12, 48)],
            },
            metadata={"dataset": "Overture Maps buildings"},
            geo=_point_geo([12], [48]),
        )
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.schema_name == "overture"
            assert dataset.read()[0].properties["names"]["primary"] == "Library"

    def test_wkb_without_geo_metadata(self, tmp_path):
        path = tmp_path / "plain.parquet"
        _write_geoparquet(path, {"name": ["alpha"], "geometry": [_wkb(5, 6)]})
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.read()[0].geometry.x == pytest.approx(5)
            assert any("GeoParquet metadata" in warning for warning in dataset.warnings)

    def test_h3_cell_column(self, tmp_path):
        path = tmp_path / "hex.parquet"
        _write_geoparquet(path, {"h3_index": ["8928308280fffff"], "count": [4]})
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "h3-s2"
            assert dataset.spatial_index == "h3"
            feature = dataset.read()[0]
            assert feature.properties["h3_index"] == "8928308280fffff"
            try:
                import h3
            except ImportError:
                # without h3 the cell cannot be turned into a polygon, and the reader says so
                assert feature.geometry is None
                assert any("h3" in warning for warning in dataset.warnings)
            else:
                # with h3 the cell becomes its hexagon, which contains the cell centre
                from shapely.geometry import Point

                lat, lon = h3.cell_to_latlng("8928308280fffff")
                assert feature.geometry.geom_type == "Polygon"
                assert len(feature.geometry.exterior.coords) == 7
                assert feature.geometry.contains(Point(lon, lat))
                assert not any("Install the h3 package" in w for w in dataset.warnings)


class TestGeoArrow:
    """GeoArrow extension types on Arrow IPC / Feather."""

    def test_wkb_extension(self, tmp_path):
        pa = _pyarrow()
        import pyarrow.feather as feather

        path = tmp_path / "sites.feather"
        schema = pa.schema(
            [
                pa.field("name", pa.string()),
                pa.field(
                    "geometry",
                    pa.binary(),
                    metadata={b"ARROW:extension:name": b"geoarrow.wkb"},
                ),
            ]
        )
        table = pa.table({"name": ["alpha"], "geometry": [_wkb(8, 9)]}, schema=schema)
        feather.write_feather(table, path)
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "geoarrow"
            assert dataset.geometry_encoding == "geoarrow.wkb"
            assert dataset.read()[0].geometry.y == pytest.approx(9)


class TestGdalCloudFiles:
    """FlatGeobuf and GeoPackage go through GDAL, including a spatial filter."""

    def test_flatgeobuf_bbox(self, tmp_path):
        path = tmp_path / "sites.fgb"
        _write_ogr_points("FlatGeobuf", path)
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "flatgeobuf"
            assert dataset.spatial_index == "hilbert-rtree"
            assert dataset.spatial_access == "http-range"
            assert dataset.feature_count == 2
            inside = dataset.read(bbox=(0, 0, 1.5, 2.5))
            assert len(inside) == 1
            assert inside[0].properties["name"] == "alpha"

    def test_geopackage_layer(self, tmp_path):
        path = tmp_path / "sites.gpkg"
        _write_ogr_points("GPKG", path, layer_name="sites")
        with open_cloud_native_vector_data(path, layer="sites") as dataset:
            assert dataset.format_id == "geopackage"
            assert dataset.layer.name == "sites"
            assert [feature.geometry.x for feature in dataset] == pytest.approx([1, 3])


class TestPMTiles:
    """PMTiles header, range-request tile reads, and vector-tile features."""

    def test_uncompressed_archive_and_tile_lookup(self, tmp_path):
        path = tmp_path / "demo.pmtiles"
        _write_pmtiles(
            path,
            {0: b"ROOT", 4: b"TILE-4"},
            metadata={"name": "demo", "bounds": "-180,-85,180,85"},
        )
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "pmtiles"
            assert dataset.details["header"]["version"] == 3
            assert dataset.details["metadata"]["name"] == "demo"
            assert dataset.read_tile(0, 0, 0) == b"ROOT"
            z, x, y = tileid_to_zxy(4)
            assert dataset.read_tile(z, x, y) == b"TILE-4"
            assert dataset.read_tile(2, 0, 0) is None
            footprints = dataset.read()
            assert {feature.properties["tile_id"] for feature in footprints} == {0, 4}
            assert footprints[0].geometry.geom_type == "Polygon"

    def test_http_range_requests(self, tmp_path):
        path = tmp_path / "remote.pmtiles"
        payload = b"RANGE-TILE"
        _write_pmtiles(path, {0: payload}, metadata={"name": "remote"})
        body = path.read_bytes()
        ranges = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                ranges.append(header)
                if header is None or not header.startswith("bytes="):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                start_text, end_text = header[len("bytes=") :].split("-", 1)
                start = int(start_text)
                end = int(end_text)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                self.end_headers()
                self.wfile.write(chunk)

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/remote.pmtiles"
            with open_cloud_native_vector_data(url) as dataset:
                assert dataset.read_tile(0, 0, 0) == payload
                assert dataset.details["metadata"]["name"] == "remote"
        finally:
            server.shutdown()
            thread.join(timeout=5)
        assert ranges
        assert all(item is not None and item.startswith("bytes=") for item in ranges)

    def test_gdal_vector_tiles(self, tmp_path):
        _, ogr, osr = _ogr()
        driver = ogr.GetDriverByName("PMTiles")
        if driver is None or driver.GetMetadataItem("DCAP_CREATE") != "YES":
            pytest.skip("PMTiles driver cannot create datasets in this GDAL build")
        path = tmp_path / "places.pmtiles"
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)
        spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        datasource = driver.CreateDataSource(str(path))
        layer = datasource.CreateLayer(
            "places", spatial_ref, ogr.wkbPoint, options=["MINZOOM=0", "MAXZOOM=2"]
        )
        layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("name", "alpha")
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint_2D(10, 20)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        layer = None
        datasource = None
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "pmtiles"
            assert dataset.reader == "pmtiles+gdal"
            assert dataset.details["metadata"]["vector_layers"][0]["id"] == "places"
            assert dataset.feature_count >= 1
            assert dataset.read()[0].properties["name"] == "alpha"
            tile = dataset.read_tile(0, 0, 0)
            assert tile
            assert not tile.startswith(b"\x1f\x8b")


class TestCOPC:
    """Cloud Optimized Point Cloud header, octree, and uncompressed points."""

    def test_points_hierarchy_and_bbox(self, tmp_path):
        path = tmp_path / "cloud.copc.laz"
        _write_copc(path, [(1.0, 2.0, 3.0), (4.0, 6.0, 8.0)])
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "copc"
            assert dataset.spatial_index == "octree"
            assert dataset.feature_count == 2
            assert dataset.copc["spacing"] == pytest.approx(1.0)
            assert len(dataset.hierarchy) == 1
            assert dataset.hierarchy[0]["point_count"] == 2
            points = dataset.read()
            assert points[0].geometry.x == pytest.approx(1.0)
            assert points[0].geometry.y == pytest.approx(2.0)
            assert points[0].properties["z"] == pytest.approx(3.0)
            assert dataset.crs is not None
            assert dataset.crs.to_epsg() == 4326
            selected = dataset.read(bbox=(0, 0, 2, 3), columns=["x"])
            assert len(selected) == 1
            assert list(selected[0].properties) == ["x"]


class TestLakehouse:
    """Delta Lake WKB and Iceberg geometry fields."""

    def test_delta_log_keeps_live_files(self, tmp_path):
        _pyarrow()
        root = tmp_path / "delta_table"
        root.mkdir()
        _write_geoparquet(root / "part-000.parquet", {"name": ["old"], "geometry": [_wkb(1, 1)]})
        _write_geoparquet(root / "part-001.parquet", {"name": ["new"], "geometry": [_wkb(9, 8)]})
        schema = {
            "type": "struct",
            "fields": [
                {"name": "name", "type": "string", "nullable": True, "metadata": {}},
                {"name": "geometry", "type": "binary", "nullable": True, "metadata": {}},
            ],
        }
        log = root / "_delta_log"
        log.mkdir()

        def commit(number, actions):
            text = "".join(json.dumps(action) + "\n" for action in actions)
            (log / f"{number:020d}.json").write_text(text, encoding="utf-8")

        commit(
            0,
            [
                {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
                {
                    "metaData": {
                        "id": "table",
                        "format": {"provider": "parquet", "options": {}},
                        "schemaString": json.dumps(schema),
                        "partitionColumns": [],
                        "configuration": {},
                        "createdTime": 1,
                    }
                },
                {
                    "add": {
                        "path": "part-000.parquet",
                        "partitionValues": {},
                        "size": 1,
                        "modificationTime": 1,
                        "dataChange": True,
                    }
                },
            ],
        )
        commit(
            1,
            [
                {
                    "remove": {
                        "path": "part-000.parquet",
                        "deletionTimestamp": 2,
                        "dataChange": True,
                    }
                },
                {
                    "add": {
                        "path": "part-001.parquet",
                        "partitionValues": {},
                        "size": 1,
                        "modificationTime": 2,
                        "dataChange": True,
                    }
                },
            ],
        )
        with open_cloud_native_vector_data(root) as dataset:
            assert dataset.format_id == "delta"
            assert dataset.details["delta_version"] == 1
            features = dataset.read()
            assert [feature.properties["name"] for feature in features] == ["new"]
            assert features[0].geometry.x == pytest.approx(9)
            assert any("no native geometry" in warning.lower() for warning in dataset.warnings)

    def test_iceberg_geometry_type(self, tmp_path):
        _pyarrow()
        root = tmp_path / "iceberg_table"
        data = root / "data"
        meta = root / "metadata"
        data.mkdir(parents=True)
        meta.mkdir()
        _write_geoparquet(data / "00000.parquet", {"name": ["alpha"], "geometry": [_wkb(7, 11)]})
        metadata = {
            "format-version": 3,
            "table-uuid": "00000000-0000-0000-0000-000000000001",
            "location": str(root),
            "current-schema-id": 0,
            "schemas": [
                {
                    "schema-id": 0,
                    "type": "struct",
                    "fields": [
                        {"id": 1, "name": "name", "required": False, "type": "string"},
                        {"id": 2, "name": "geometry", "required": False, "type": "geometry"},
                    ],
                }
            ],
            "current-snapshot-id": None,
            "snapshots": [],
        }
        (meta / "v3.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with open_cloud_native_vector_data(root) as dataset:
            assert dataset.format_id == "iceberg"
            assert dataset.details["iceberg_format_version"] == 3
            assert dataset.geometry_encoding == "iceberg-geometry"
            assert dataset.details["geometry_fields"] == [["geometry", "geometry"]] or dataset.details[
                "geometry_fields"
            ] == [("geometry", "geometry")]
            feature = dataset.read()[0]
            assert feature.properties["name"] == "alpha"
            assert feature.geometry.y == pytest.approx(11)


class TestGeoJSONSeq:
    """Newline-delimited GeoJSON, including an RFC 8142 record separator."""

    def test_stream_filter(self, tmp_path):
        path = tmp_path / "seq.geojsonl"
        path.write_text(
            "\x1e"
            '{"type":"Feature","properties":{"name":"alpha"},'
            '"geometry":{"type":"Point","coordinates":[1,2]}}\n'
            '{"type":"Feature","properties":{"name":"beta"},'
            '"geometry":{"type":"Point","coordinates":[30,40]}}\n',
            encoding="utf-8",
        )
        with open_cloud_native_vector_data(path) as dataset:
            assert dataset.format_id == "geojsonseq"
            assert dataset.feature_count == 2
            assert dataset.crs.to_epsg() == 4326
            selected = dataset.read(where="name = 'beta'", bbox=(0, 0, 5, 5))
            assert selected == []
            alpha = dataset.read(bbox=(0, 0, 5, 5), limit=1)
            assert alpha[0].properties["name"] == "alpha"
            assert alpha[0].geometry.x == pytest.approx(1)


class TestOptionalLibraries:
    """TileDB and Lance need their own packages."""

    def test_lance_without_package(self, tmp_path):
        pytest.importorskip("pyarrow")
        try:
            import lance  # noqa: F401
        except ImportError:
            path = tmp_path / "embeddings.lance"
            path.mkdir()
            with pytest.raises(CloudNativeOpenError, match="lance"):
                open_cloud_native_vector_data(path)
        else:
            pytest.skip("lance is installed; the missing-package path is not under test")

    def test_tiledb_without_package(self):
        try:
            import tiledb  # noqa: F401
        except ImportError:
            with pytest.raises(CloudNativeOpenError, match="tiledb"):
                open_cloud_native_vector_data("memory.tiledb", format="tiledb")
        else:
            pytest.skip("tiledb is installed; the missing-package path is not under test")


class TestErrors:
    """Unknown layouts and missing paths."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_cloud_native_vector_data(tmp_path / "missing.parquet")

    def test_unknown_format(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(CloudNativeOpenError, match="cloud-native"):
            open_cloud_native_vector_data(path)

    def test_unknown_format_name(self):
        with pytest.raises(CloudNativeOpenError, match="Unknown"):
            open_cloud_native_vector_data("unused.parquet", format="shapefile")

    def test_closed_dataset_still_reports_schema(self, tmp_path):
        path = tmp_path / "cities.parquet"
        _write_geoparquet(
            path,
            {"name": ["alpha"], "geometry": [_wkb(1, 2)]},
            geo=_point_geo([1], [2]),
        )
        dataset = open_cloud_native_vector_data(path)
        dataset.close()
        assert dataset.format_name == "GeoParquet"
        assert dataset.layer.field_names == ("name",)
