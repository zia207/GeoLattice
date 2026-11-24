"""
Open the GDAL sample files in Data/00_shared/data_gdal with open_vectors_data.

One case covers each file-based format the module advertises. Files that this
GDAL build cannot read (Parquet, Arrow, Personal Geodatabase, DGN v8, newer
DWG) are asserted as that specific failure.
"""

from pathlib import Path

import pytest

from geolattice.vector.read_vector_data import (
    DriverUnavailableError,
    VectorOpenError,
    driver_available,
    open_vector,
    open_vectors_data,
)

DATA = Path(__file__).resolve().parents[1] / "Data" / "00_shared" / "data_gdal"

# rel, open kwargs, driver, format name, layer, feature count, geometry expected
SAMPLES = [
    ("poly.shp", {}, "ESRI Shapefile", "ESRI Shapefile", "poly", 10, True),
    ("gpkg/poly_golden.gpkg", {}, "GPKG", "GeoPackage", "poly", 10, True),
    ("geojson/linestring.geojson", {}, "GeoJSON", "GeoJSON", "linestring", 1, True),
    (
        "geojson/featurecollection_point.json",
        {},
        "GeoJSON",
        "GeoJSON",
        "featurecollection_point",
        1,
        True,
    ),
    ("geojsonseq/test.geojsonl", {}, "GeoJSONSeq", "GeoJSON Sequence", "test", 2, True),
    ("geojsonseq/test.geojsons", {}, "GeoJSONSeq", "GeoJSON Sequence", "test", 2, True),
    ("kml/geometries.kml", {}, "LIBKML", "KML / KMZ", "APTBoundary", 25, True),
    ("gml/expected_gml_gml32.gml", {}, "GML", "GML", "firstlayer", 2, True),
    ("gpx/test.gpx", {"layer": "waypoints"}, "GPX", "GPX", "waypoints", 2, True),
    ("wkt.csv", {}, "CSV", "Delimited text (CSV/TXT)", "wkt", None, True),
    ("mitab/all_geoms.tab", {}, "MapInfo File", "MapInfo TAB / MIF", "all_geoms", 16, True),
    ("mitab/small.mif", {}, "MapInfo File", "MapInfo TAB / MIF", "small", 2, True),
    ("mitab/small.mid", {}, "MapInfo File", "MapInfo TAB / MIF", "small", 2, True),
    ("dxf/circle.dxf", {}, "DXF", "AutoCAD DXF", "entities", 2, True),
    ("cad/line_r2000.dwg", {}, "CAD", "AutoCAD DWG", None, 1, True),
    ("dgn/smalltest.dgn", {}, "DGN", "MicroStation DGN", "elements", 4, True),
    ("openfilegdb/polygon_golden.gdb", {}, "OpenFileGDB", "ESRI File Geodatabase", None, 1, True),
    ("sqlite/poly_spatialite.sqlite", {}, "SQLite", "SpatiaLite / SQLite", "poly", 10, True),
    ("sqlite/first_geometry_null.db", {}, "SQLite", "SpatiaLite / SQLite", "test", 2, True),
    ("testfgb/poly.fgb", {}, "FlatGeobuf", "FlatGeobuf", "poly", 10, True),
    ("osm/test.osm", {"layer": "points"}, "OSM", "OpenStreetMap", "points", None, True),
    ("osm/two_points.pbf", {"layer": "points"}, "OSM", "OpenStreetMap", "points", None, True),
    ("xlsx/test.xlsx", {}, "XLSX", "Microsoft Excel", "Feuille1", 26, False),
    ("xls/test972000xp.xls", {}, "XLS", "Microsoft Excel (legacy)", "Feuille1", 3, False),
    ("ods/test.ods", {}, "ODS", "LibreOffice ODS", "Feuille1", 26, False),
]


def setup_module():
    if not DATA.is_dir():
        pytest.skip(f"GDAL sample data is not present at {DATA}")


@pytest.mark.parametrize(
    ("relative", "kwargs", "driver", "format_name", "layer_name", "count", "expect_geometry"),
    SAMPLES,
    ids=[item[0] for item in SAMPLES],
)
def test_sample_opens(relative, kwargs, driver, format_name, layer_name, count, expect_geometry):
    path = DATA / relative
    assert path.exists(), relative
    with open_vectors_data(path, **kwargs) as dataset:
        layer = dataset.layer
        assert dataset.driver == driver
        assert dataset.format_name == format_name
        assert dataset.layer_count >= 1
        if layer_name is not None:
            assert layer.name == layer_name
        assert layer.feature_count == count
        features = layer.read(limit=30)
        assert features
        has_geometry = any(feature.geometry is not None for feature in features)
        assert has_geometry is expect_geometry
        frame = layer.to_dataframe(limit=5)
        assert "geometry" in frame.columns
        assert len(frame) == min(5, len(features))


def test_shapefile_directory_is_multilayer():
    with open_vectors_data(DATA / "shp") as dataset:
        assert dataset.driver == "ESRI Shapefile"
        assert dataset.layer_count > 1
        assert dataset.layer.feature_count >= 0


def test_shapefile_crs_and_alias():
    with open_vector(DATA / "poly.shp") as dataset:
        assert str(dataset.layer.crs) == "EPSG:27700"
        assert dataset.layer.field_names == ("AREA", "EAS_ID", "PRFEDEA")
        frame = dataset.to_dataframe()
        assert len(frame) == 10


def test_gpx_layer_names():
    with open_vectors_data(DATA / "gpx" / "test.gpx") as dataset:
        assert dataset.layer_names == [
            "waypoints",
            "routes",
            "tracks",
            "route_points",
            "track_points",
        ]


def test_mid_opens_sibling_mif():
    with open_vectors_data(DATA / "mitab" / "small.mid") as dataset:
        assert Path(dataset.path).suffix.lower() == ".mif"
        assert Path(dataset.path).name == "small.mif"


def test_gml_does_not_write_gfs_sidecar():
    path = DATA / "gml" / "expected_gml_gml32.gml"
    sidecar = path.with_suffix(".gfs")
    assert not sidecar.exists()
    with open_vectors_data(path) as dataset:
        assert dataset.read(limit=1)
    assert not sidecar.exists()


def test_csv_wkt_polygons():
    with open_vectors_data(DATA / "wkt.csv") as dataset:
        features = dataset.read()
        assert len(features) == 3
        assert all(feature.geometry.geom_type == "Polygon" for feature in features)
        assert dataset.layer.feature_count is None


def test_newer_dwg_is_reported():
    path = DATA / "cad" / "AC1018_signature.dwg"
    with pytest.raises(VectorOpenError, match="CAD|DWG|version"):
        open_vectors_data(path)


def test_personal_geodatabase_sample():
    path = DATA / "pgeo" / "geometry_types.mdb"
    if not driver_available("PGeo"):
        with pytest.raises(DriverUnavailableError):
            open_vectors_data(path)
        return
    try:
        with open_vectors_data(path) as dataset:
            assert dataset.driver == "PGeo"
            assert dataset.layer_count >= 1
    except VectorOpenError as exc:
        message = str(exc)
        assert "ODBC" in message or "Access" in message or "Personal Geodatabase" in message


def test_dgn_v8_sample():
    path = DATA / "dgnv8" / "test_dgnv8.dgn"
    if driver_available("DGNv8"):
        with open_vectors_data(path) as dataset:
            assert dataset.driver in ("DGNv8", "DGN")
            assert dataset.layer_count >= 1
        return
    with pytest.raises(VectorOpenError, match="DGN|DGNv8|not recognized"):
        open_vectors_data(path)


@pytest.mark.parametrize(
    ("relative", "driver_name"),
    [
        ("parquet/example.parquet", "Parquet"),
        ("arrow/test.feather", "Arrow"),
        ("arrow/from_paleolimbot_geoarrow/multipoint-default.feather", "Arrow"),
    ],
)
def test_arrow_family_samples(relative, driver_name):
    path = DATA / relative
    if not driver_available(driver_name):
        with pytest.raises(DriverUnavailableError, match=driver_name):
            open_vectors_data(path)
        return
    with open_vectors_data(path) as dataset:
        assert dataset.driver == driver_name
        assert dataset.layer_count >= 1
        dataset.read(limit=1)
