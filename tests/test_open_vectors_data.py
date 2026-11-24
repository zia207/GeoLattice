"""
Tests for GDAL/OGR vector opening.
"""

import shutil
import zipfile

import pytest

from geolattice.vector.read_vector_data import (
    DriverUnavailableError,
    VectorOpenError,
    driver_available,
    open_vector,
    open_vectors_data,
    supported_formats,
)


REQUIRED_EXTENSIONS = {
    ".shp",
    ".gpkg",
    ".geojson",
    ".json",
    ".kml",
    ".kmz",
    ".gml",
    ".gpx",
    ".csv",
    ".txt",
    ".tab",
    ".mif",
    ".mid",
    ".dxf",
    ".dwg",
    ".dgn",
    ".gdb",
    ".mdb",
    ".sqlite",
    ".fgb",
    ".parquet",
    ".arrow",
    ".feather",
    ".osm",
    ".pbf",
    ".xlsx",
    ".xls",
    ".ods",
}


def _ogr():
    pytest.importorskip("osgeo")
    from osgeo import gdal, ogr, osr

    gdal._UseExceptions()
    ogr._UseExceptions()
    osr._UseExceptions()
    return gdal, ogr, osr


def _write_points(driver_name, path, dataset_options=None, layer_name="layer", epsg=4326):
    """Write one point layer with a name field. Skip when the driver cannot create it."""
    _, ogr, osr = _ogr()
    driver = ogr.GetDriverByName(driver_name)
    if driver is None or driver.GetMetadataItem("DCAP_CREATE") != "YES":
        pytest.skip(f"{driver_name} cannot create datasets in this GDAL build")
    if path.exists():
        try:
            driver.DeleteDataSource(str(path))
        except RuntimeError:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    try:
        datasource = driver.CreateDataSource(str(path), options=list(dataset_options or []))
    except RuntimeError as exc:
        pytest.skip(f"Cannot create {driver_name} dataset: {exc}")
    if datasource is None:
        pytest.skip(f"{driver_name} did not create {path.name}")
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)
    spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    try:
        layer = datasource.CreateLayer(layer_name, spatial_ref, ogr.wkbPoint)
    except RuntimeError as exc:
        datasource = None
        pytest.skip(f"{driver_name} cannot create a point layer: {exc}")
    name_field = ogr.FieldDefn("name", ogr.OFTString)
    try:
        layer.CreateField(name_field)
    except RuntimeError:
        name_field = None
    for label, x_coord, y_coord in (("alpha", 1.0, 2.0), ("beta", 3.0, 4.0)):
        feature = ogr.Feature(layer.GetLayerDefn())
        if name_field is not None:
            feature.SetField("name", label)
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint_2D(x_coord, y_coord)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
    layer = None
    datasource = None


def _property(feature, name):
    folded = {key.lower(): value for key, value in feature.properties.items()}
    return folded[name.lower()]


class TestFormatCatalog:
    """Format registry covers the requested file types."""

    def test_extensions_cover_requested_formats(self):
        found = {extension for item in supported_formats() for extension in item["extensions"]}
        assert REQUIRED_EXTENSIONS <= found

    def test_driver_availability_matches_catalog(self):
        for item in supported_formats():
            expected = any(driver_available(name) for name in item["drivers"])
            assert item["available"] is expected

    def test_open_vector_alias(self):
        assert open_vector is not open_vectors_data
        assert open_vector.__name__ == "open_vector"


class TestGeoJSON:
    """GeoJSON and GeoJSON Sequence."""

    def test_feature_collection(self, tmp_path):
        path = tmp_path / "cities.geojson"
        path.write_text(
            """
            {"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"name":"alpha","code":null},
               "geometry":{"type":"Point","coordinates":[1,2]}},
              {"type":"Feature","properties":{"name":"beta","code":7},
               "geometry":{"type":"Point","coordinates":[3,4]}}
            ]}
            """.strip(),
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            assert dataset.driver == "GeoJSON"
            assert dataset.format_name == "GeoJSON"
            assert dataset.layer.feature_count == 2
            assert dataset.layer.bounds.minx == pytest.approx(1)
            assert dataset.layer.bounds.miny == pytest.approx(2)
            assert dataset.layer.bounds.maxx == pytest.approx(3)
            assert dataset.layer.bounds.maxy == pytest.approx(4)
            features = dataset.read()
            assert _property(features[0], "name") == "alpha"
            assert features[0].properties["code"] is None
            assert features[0].geometry.x == pytest.approx(1)
            assert features[0].geometry.y == pytest.approx(2)
            assert features[0].wkt.startswith("POINT")
            selected = dataset.read(where="name = 'beta'", columns=["name"])
            assert len(selected) == 1
            assert list(selected[0].properties) == ["name"]
            inside = dataset.read(bbox=(0, 0, 1.5, 2.5))
            assert len(inside) == 1
            assert _property(inside[0], "name") == "alpha"
            assert len(dataset.read(limit=1)) == 1
            frame = dataset.to_dataframe()
            assert list(frame["name"]) == ["alpha", "beta"]
            assert frame.geometry.iloc[0].x == pytest.approx(1)
            layer_name = dataset.layer.name
            dataset.close()
            assert dataset.layer.name == layer_name
            with pytest.raises(VectorOpenError):
                dataset.read()

    def test_json_feature_collection_uses_geojson_driver(self, tmp_path):
        path = tmp_path / "cities.json"
        path.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            '"properties":{"name":"alpha"},"geometry":{"type":"Point","coordinates":[8,9]}}]}',
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            assert dataset.driver == "GeoJSON"
            assert dataset.read()[0].geometry.x == pytest.approx(8)

    def test_geojson_sequence(self, tmp_path):
        path = tmp_path / "seq.geojsonl"
        path.write_text(
            '{"type":"Feature","properties":{"name":"alpha"},'
            '"geometry":{"type":"Point","coordinates":[1,2]}}\n'
            '{"type":"Feature","properties":{"name":"beta"},'
            '"geometry":{"type":"Point","coordinates":[3,4]}}\n',
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            assert dataset.driver == "GeoJSONSeq"
            assert dataset.layer.feature_count == 2
            assert [feature.geometry.x for feature in dataset] == pytest.approx([1, 3])

    def test_json_sequence_prefers_geojsonseq(self, tmp_path):
        path = tmp_path / "seq.json"
        path.write_text(
            '{"type":"Feature","properties":{"name":"alpha"},'
            '"geometry":{"type":"Point","coordinates":[5,6]}}\n'
            '{"type":"Feature","properties":{"name":"beta"},'
            '"geometry":{"type":"Point","coordinates":[7,8]}}\n',
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            assert dataset.driver == "GeoJSONSeq"
            assert dataset.layer.feature_count == 2

    def test_empty_collection(self, tmp_path):
        path = tmp_path / "empty.geojson"
        path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        with open_vectors_data(path) as dataset:
            assert dataset.layer.feature_count == 0
            assert dataset.layer.bounds is None
            frame = dataset.to_dataframe()
            assert len(frame) == 0
            assert "geometry" in frame.columns

    def test_invalid_queries(self, tmp_path):
        path = tmp_path / "one.geojson"
        path.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            '"properties":{"name":"alpha"},"geometry":{"type":"Point","coordinates":[1,2]}}]}',
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            with pytest.raises(ValueError):
                dataset.read(bbox=(2, 2, 1, 1))
            with pytest.raises(ValueError):
                dataset.read(limit=-1)
            with pytest.raises(TypeError):
                dataset.read(limit=True)
            with pytest.raises(VectorOpenError):
                dataset.read(columns=["missing"])
            with pytest.raises(VectorOpenError):
                dataset.get_layer("missing")
            with pytest.raises(VectorOpenError):
                dataset.read(where="name = ")


class TestShapefileAndPackages:
    """Shapefile sidecars, GeoPackage layers, and directory datasets."""

    def test_shapefile_roundtrip_and_sidecars(self, tmp_path):
        path = tmp_path / "sites.shp"
        _write_points("ESRI Shapefile", path)
        assert path.with_suffix(".shx").exists()
        assert path.with_suffix(".dbf").exists()
        assert path.with_suffix(".prj").exists()
        with open_vectors_data(path, encoding="UTF-8") as dataset:
            assert dataset.driver == "ESRI Shapefile"
            assert dataset.layer.feature_count == 2
            assert dataset.crs is not None
            assert dataset.crs.to_epsg() == 4326
            assert _property(dataset.read()[0], "name") == "alpha"
            assert dataset.warnings == ()

        path.with_suffix(".prj").unlink()
        with open_vectors_data(path) as dataset:
            assert any(".prj" in warning for warning in dataset.warnings)
            assert dataset.layer.feature_count == 2

        path.with_suffix(".shx").unlink()
        with pytest.raises(VectorOpenError) as excinfo:
            open_vectors_data(path)
        assert ".shx" in str(excinfo.value)
        assert ".dbf" in str(excinfo.value)

    def test_gpkg_layer_selection(self, tmp_path):
        _, ogr, osr = _ogr()
        path = tmp_path / "network.gpkg"
        driver = ogr.GetDriverByName("GPKG")
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)
        datasource = driver.CreateDataSource(str(path))
        for layer_name, x_coord in (("roads", 1.0), ("rail", 9.0)):
            layer = datasource.CreateLayer(layer_name, spatial_ref, ogr.wkbPoint)
            layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("name", layer_name)
            geometry = ogr.Geometry(ogr.wkbPoint)
            geometry.AddPoint_2D(x_coord, 2.0)
            feature.SetGeometry(geometry)
            layer.CreateFeature(feature)
        datasource = None

        with open_vectors_data(path, layer="rail") as dataset:
            assert dataset.format_name == "GeoPackage"
            assert set(dataset.layer_names) == {"roads", "rail"}
            assert dataset.layer.name == "rail"
            assert dataset["roads"].read()[0].geometry.x == pytest.approx(1)
            assert dataset.layer.read()[0].geometry.x == pytest.approx(9)
            tables = dataset.to_dataframes()
            assert set(tables) == {"roads", "rail"}

    def test_shapefile_directory(self, tmp_path):
        folder = tmp_path / "shapes"
        folder.mkdir()
        _write_points("ESRI Shapefile", folder / "cities.shp", layer_name="cities")
        _write_points("ESRI Shapefile", folder / "rivers.shp", layer_name="rivers")
        with open_vectors_data(folder) as dataset:
            assert dataset.driver == "ESRI Shapefile"
            assert dataset.layer_count == 2


class TestDelimitedAndSpreadsheets:
    """CSV/TXT and spreadsheet geometry from columns or WKT."""

    def test_csv_xy_and_crs(self, tmp_path):
        path = tmp_path / "sites.csv"
        path.write_text("name,lon,lat\nalpha,10.5,20.25\nbeta,11,21\n", encoding="utf-8")
        with open_vectors_data(path, crs=4326) as dataset:
            assert dataset.driver == "CSV"
            assert dataset.crs.to_epsg() == 4326
            assert dataset.layer.geometry_source == "layer"
            first = dataset.read()[0]
            assert first.geometry.x == pytest.approx(10.5)
            assert first.geometry.y == pytest.approx(20.25)
            assert _property(first, "name") == "alpha"
            assert dataset.layer.bounds.minx == pytest.approx(10.5)
            assert dataset.layer.bounds.maxy == pytest.approx(21)

    def test_csv_xyz(self, tmp_path):
        path = tmp_path / "sites.csv"
        path.write_text("name,x,y,z\nalpha,1,2,3\n", encoding="utf-8")
        with open_vectors_data(path) as dataset:
            coords = list(dataset.read()[0].geometry.coords)[0]
            assert coords == pytest.approx((1, 2, 3))

    def test_csv_wkt_and_ewkt(self, tmp_path):
        path = tmp_path / "lines.csv"
        path.write_text(
            'name,wkt\nalpha,"LINESTRING (0 0, 1 1)"\nbeta,"SRID=4326;POINT (5 6)"\n',
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            features = dataset.read()
            assert features[0].geometry.geom_type == "LineString"
            assert list(features[0].geometry.coords) == [(0, 0), (1, 1)]
            assert features[1].geometry.x == pytest.approx(5)
            assert features[1].geometry.y == pytest.approx(6)

    def test_tab_separated_text(self, tmp_path):
        path = tmp_path / "sites.txt"
        path.write_text("name\teasting\tnorthing\nalpha\t100\t200\n", encoding="utf-8")
        with open_vectors_data(path) as dataset:
            feature = dataset.read()[0]
            assert feature.geometry.x == pytest.approx(100)
            assert feature.geometry.y == pytest.approx(200)

    def test_spreadsheet_xy(self, tmp_path):
        _, ogr, _ = _ogr()
        for driver_name, suffix in (("XLSX", ".xlsx"), ("ODS", ".ods")):
            path = tmp_path / f"sheet{suffix}"
            driver = ogr.GetDriverByName(driver_name)
            if driver is None:
                pytest.skip(f"{driver_name} driver is not available")
            datasource = driver.CreateDataSource(str(path))
            try:
                layer = datasource.CreateLayer("sites", geom_type=ogr.wkbNone)
            except RuntimeError as exc:
                datasource = None
                pytest.skip(f"{driver_name} cannot create a table: {exc}")
            layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
            layer.CreateField(ogr.FieldDefn("lon", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("lat", ogr.OFTReal))
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("name", "alpha")
            feature.SetField("lon", 12.0)
            feature.SetField("lat", 34.0)
            layer.CreateFeature(feature)
            datasource = None

            with open_vectors_data(path) as dataset:
                assert dataset.layer.geometry_source == "attributes"
                opened = dataset.read()[0]
                assert opened.geometry.x == pytest.approx(12)
                assert opened.geometry.y == pytest.approx(34)
                assert _property(opened, "name") == "alpha"


class TestOtherFileFormats:
    """Formats opened directly by their GDAL/OGR drivers."""

    @pytest.mark.parametrize(
        ("driver_name", "suffix", "options"),
        [
            ("GML", ".gml", None),
            ("FlatGeobuf", ".fgb", None),
            ("SQLite", ".sqlite", None),
            ("OpenFileGDB", ".gdb", None),
            ("MapInfo File", ".tab", None),
            ("MapInfo File", ".mif", ["FORMAT=MIF"]),
            ("GeoJSON", ".geojson", None),
            ("DXF", ".dxf", None),
            ("DGN", ".dgn", None),
            ("KML", ".kml", None),
        ],
    )
    def test_driver_roundtrip(self, tmp_path, driver_name, suffix, options):
        path = tmp_path / f"data{suffix}"
        _write_points(driver_name, path, dataset_options=options)
        with open_vectors_data(path) as dataset:
            assert dataset.layer.feature_count == 2
            found = set()
            for feature in dataset.read():
                geometry = feature.geometry
                if geometry.geom_type == "Point":
                    found.add((round(geometry.x, 5), round(geometry.y, 5)))
                else:
                    for x_coord, y_coord, *_rest in geometry.coords:
                        found.add((round(x_coord, 5), round(y_coord, 5)))
            assert (1.0, 2.0) in found
            assert (3.0, 4.0) in found

    def test_gml_read_does_not_write_gfs(self, tmp_path):
        path = tmp_path / "places.gml"
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml"
                xmlns:ogr="http://ogr.maptools.org/">
              <gml:featureMember>
                <ogr:point fid="p.0">
                  <ogr:geometryProperty>
                    <gml:Point><gml:coordinates>1,2</gml:coordinates></gml:Point>
                  </ogr:geometryProperty>
                  <ogr:name>alpha</ogr:name>
                </ogr:point>
              </gml:featureMember>
            </gml:FeatureCollection>
            """,
            encoding="utf-8",
        )
        with open_vectors_data(path) as dataset:
            feature = dataset.read()[0]
            assert feature.geometry.x == pytest.approx(1)
            assert feature.geometry.y == pytest.approx(2)
        assert not path.with_suffix(".gfs").exists()

    def test_mapinfo_mid_opens_mif(self, tmp_path):
        path = tmp_path / "parcels.mif"
        _write_points("MapInfo File", path, dataset_options=["FORMAT=MIF"])
        mid = path.with_suffix(".mid")
        assert mid.exists()
        with open_vectors_data(mid) as dataset:
            assert dataset.path.endswith(".mif")
            assert dataset.read()[0].geometry.x == pytest.approx(1)

    def test_kml_and_kmz(self, tmp_path):
        kml = """<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Placemark>
            <name>Cafe</name>
            <Point><coordinates>10.0,20.0</coordinates></Point>
          </Placemark>
        </kml>
        """
        kml_path = tmp_path / "place.kml"
        kml_path.write_text(kml, encoding="utf-8")
        kmz_path = tmp_path / "place.kmz"
        with zipfile.ZipFile(kmz_path, "w") as archive:
            archive.writestr("doc.kml", kml)
        for path in (kml_path, kmz_path):
            with open_vectors_data(path) as dataset:
                feature = dataset.read()[0]
                folded = {key.lower(): value for key, value in feature.properties.items()}
                assert folded.get("name") == "Cafe"
                assert feature.geometry.x == pytest.approx(10)
                assert feature.geometry.y == pytest.approx(20)

    def test_gpx_waypoints(self, tmp_path):
        path = tmp_path / "route.gpx"
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <gpx version="1.1" creator="geolattice">
              <wpt lat="20.0" lon="10.0"><name>alpha</name></wpt>
            </gpx>
            """,
            encoding="utf-8",
        )
        with open_vectors_data(path, layer="waypoints") as dataset:
            assert "waypoints" in dataset.layer_names
            feature = dataset.read()[0]
            assert feature.geometry.x == pytest.approx(10)
            assert feature.geometry.y == pytest.approx(20)
            assert _property(feature, "name") == "alpha"

    def test_osm_nodes(self, tmp_path):
        path = tmp_path / "town.osm"
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
            <osm version="0.6">
              <node id="1" lat="20.0" lon="10.0">
                <tag k="name" v="alpha"/>
              </node>
            </osm>
            """,
            encoding="utf-8",
        )
        with open_vectors_data(path, layer="points") as dataset:
            assert dataset.driver == "OSM"
            assert "points" in dataset.layer_names
            feature = dataset.read()[0]
            assert feature.geometry.x == pytest.approx(10)
            assert feature.geometry.y == pytest.approx(20)


class TestOpenErrors:
    """Missing files, unknown types, and drivers this GDAL build does not have."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_vectors_data(tmp_path / "missing.geojson")

    def test_invalid_geojson(self, tmp_path):
        path = tmp_path / "bad.geojson"
        path.write_text("this is not json", encoding="utf-8")
        with pytest.raises(VectorOpenError):
            open_vectors_data(path)

    def test_unknown_extension(self, tmp_path):
        path = tmp_path / "notes.zzz"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(VectorOpenError):
            open_vectors_data(path)

    def test_dwg_uses_cad_driver(self, tmp_path):
        path = tmp_path / "drawing.dwg"
        path.write_bytes(b"not a dwg")
        with pytest.raises(VectorOpenError) as excinfo:
            open_vectors_data(path)
        message = str(excinfo.value)
        assert "CAD" in message or "DWG" in message

    def test_personal_geodatabase_reports_odbc_requirement(self, tmp_path):
        path = tmp_path / "parcels.mdb"
        path.write_bytes(b"not a database")
        with pytest.raises(VectorOpenError) as excinfo:
            open_vectors_data(path)
        message = str(excinfo.value)
        assert "Personal Geodatabase" in message or "Access" in message or "ODBC" in message

    def test_parquet_driver_availability(self, tmp_path):
        path = tmp_path / "table.parquet"
        path.write_bytes(b"not parquet")
        if driver_available("Parquet"):
            with pytest.raises(VectorOpenError):
                open_vectors_data(path)
        else:
            with pytest.raises(DriverUnavailableError) as excinfo:
                open_vectors_data(path)
            assert "Parquet" in str(excinfo.value)
            assert "Arrow" in str(excinfo.value)

    def test_arrow_driver_availability(self, tmp_path):
        path = tmp_path / "table.feather"
        path.write_bytes(b"not feather")
        if driver_available("Arrow"):
            with pytest.raises(VectorOpenError):
                open_vectors_data(path)
        else:
            with pytest.raises(DriverUnavailableError):
                open_vectors_data(path)

    def test_missing_mapinfo_mif(self, tmp_path):
        path = tmp_path / "only.mid"
        path.write_text("1", encoding="utf-8")
        with pytest.raises(VectorOpenError) as excinfo:
            open_vectors_data(path)
        assert ".mif" in str(excinfo.value)

    def test_bad_open_option(self):
        with pytest.raises(ValueError):
            open_vectors_data("ignored.geojson", open_options=["NOT_AN_OPTION"])
