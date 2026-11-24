"""Round-trip vector files through write_vector_data and read_vector_data."""

from shapely.geometry import Point

from geolattice.vector import (
    open_vectors_data,
    read_vector_data,
    write_csv,
    write_geopackage,
    write_geojson,
    write_shapefile,
    write_vector_data,
)
from geolattice.vector.read_vector_data import VectorFeature


FEATURES = [
    {"geometry": Point(-73.9, 40.7), "name": "nyc", "pop": 8},
    {"geometry": "POINT (-118.2 34.0)", "name": "la", "pop": 4},
]


def _names(path):
    with read_vector_data(path) as dataset:
        rows = dataset.layer.read()
    return sorted(feature.properties["name"] for feature in rows), rows


def test_read_vector_data_matches_open_vectors_data(tmp_path):
    path = tmp_path / "sites.geojson"
    write_geojson(path, FEATURES, crs=4326)
    with read_vector_data(path) as dataset:
        written = dataset.layer.feature_count
    with open_vectors_data(path) as dataset:
        opened = dataset.layer.feature_count
    assert written == opened == 2


def test_round_trip_geojson_shapefile_geopackage_and_csv(tmp_path):
    geojson = tmp_path / "sites.geojson"
    write_geojson(geojson, FEATURES, layer="sites", crs="EPSG:4326")
    names, rows = _names(geojson)
    assert names == ["la", "nyc"]
    assert rows[0].geometry.geom_type == "Point"
    assert abs(rows[0].geometry.x) in (73.9, 118.2)

    shapefile = tmp_path / "sites.shp"
    write_shapefile(shapefile, rows, crs=4326)
    shape_names, _ = _names(shapefile)
    assert shape_names == ["la", "nyc"]

    package = tmp_path / "sites.gpkg"
    write_geopackage(package, {"sites": FEATURES, "empty": []}, crs=4326)
    with read_vector_data(package, layer="sites") as dataset:
        assert dataset.layer_names == ["sites", "empty"]
        assert dataset.layer.feature_count == 2

    table = tmp_path / "sites.csv"
    write_csv(table, FEATURES, crs=4326)
    csv_names, csv_rows = _names(table)
    assert csv_names == ["la", "nyc"]
    assert csv_rows[0].geometry.geom_type == "Point"


def test_write_vector_feature_and_xy_columns(tmp_path):
    feature = VectorFeature(1, Point(1, 2), {"name": "a"})
    path = tmp_path / "one.geojson"
    write_vector_data(path, feature, crs=4326)
    with read_vector_data(path) as dataset:
        saved = dataset.layer.read()
    assert saved[0].properties["name"] == "a"
    assert saved[0].geometry.x == 1

    csv_path = tmp_path / "xy.csv"
    write_csv(csv_path, [{"lon": -10, "lat": 20, "name": "b"}], crs=4326)
    with read_vector_data(csv_path) as dataset:
        saved = dataset.layer.read()
    assert saved[0].geometry.x == -10
    assert saved[0].geometry.y == 20
