"""
Open file-based vector datasets with GDAL/OGR.

The reader targets the formats used by desktop GIS and GPS tools:

- ESRI Shapefile (.shp, with .shx, .dbf, and optional .prj / .cpg)
- GeoPackage (.gpkg)
- GeoJSON and GeoJSON Sequence (.geojson, .json, .geojsonl, .geojsons)
- KML / KMZ
- GML
- GPX
- Delimited text (.csv, .txt) with X/Y columns or WKT
- MapInfo TAB and MIF/MID
- AutoCAD DXF, and DWG through GDAL's CAD driver
- MicroStation DGN
- ESRI File Geodatabase (.gdb) and Personal Geodatabase (.mdb)
- SpatiaLite / SQLite
- FlatGeobuf (.fgb)
- (Geo)Parquet and Arrow/Feather when GDAL is built with Apache Arrow
- OpenStreetMap XML and PBF (.osm, .pbf)
- Excel (.xlsx, .xls) and LibreOffice (.ods)

Example:
    from geolattice.vector import open_vectors_data

    with open_vectors_data("parcels.shp", encoding="UTF-8") as dataset:
        print(dataset.format_name, dataset.layer.feature_count, dataset.layer.crs)
        table = dataset.to_dataframe()

    sites = open_vectors_data("sites.csv", crs=4326)
    tracks = open_vectors_data("route.gpx", layer="tracks")
"""

import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from geolattice.core.bounds import Bounds
from geolattice.crs import CRS

try:
    from shapely import from_wkb as _from_wkb
except ImportError:  # shapely 1.8
    from shapely.wkb import loads as _from_wkb
from shapely.geometry import Point
from shapely.wkt import loads as _load_wkt


# Column names accepted when a driver has no native geometry (CSV, spreadsheets).
X_POSSIBLE_NAMES: Tuple[str, ...] = (
    "longitude",
    "lon",
    "long",
    "x",
    "easting",
    "east",
    "x_coord",
    "xcoord",
)
Y_POSSIBLE_NAMES: Tuple[str, ...] = (
    "latitude",
    "lat",
    "y",
    "northing",
    "north",
    "y_coord",
    "ycoord",
)
Z_POSSIBLE_NAMES: Tuple[str, ...] = ("z", "elevation", "altitude", "height")
GEOM_POSSIBLE_NAMES: Tuple[str, ...] = (
    "wkt",
    "geom",
    "geometry",
    "the_geom",
    "wkb_geometry",
)

_WKT_PREFIXES: Tuple[str, ...] = (
    "POINT",
    "LINESTRING",
    "POLYGON",
    "MULTI",
    "GEOMETRYCOLLECTION",
    "CIRCULARSTRING",
    "COMPOUNDCURVE",
    "CURVEPOLYGON",
    "POLYHEDRALSURFACE",
    "TIN",
    "TRIANGLE",
)

_DRIVER_HINTS: Dict[str, str] = {
    "Parquet": "(Geo)Parquet needs GDAL 3.5 or newer built with Apache Arrow.",
    "Arrow": "(Geo)Arrow and Feather need GDAL 3.5 or newer built with Apache Arrow.",
    "FileGDB": (
        "The ESRI FileGDB SDK driver is optional. OpenFileGDB reads .gdb "
        "directories without that SDK."
    ),
    "PGeo": (
        "ESRI Personal Geodatabase (.mdb) is read through the PGeo ODBC driver. "
        "On Windows, install the Microsoft Access ODBC driver. On other platforms, "
        "GDAL can use MDB Tools when that ODBC driver is installed."
    ),
    "CAD": "DWG is read with GDAL's CAD driver (libopencad).",
    "DWG": "DWG is read with GDAL's CAD driver when a separate DWG driver is installed.",
    "DGNv8": "MicroStation v8 design files need the DGNv8 driver.",
}

_GDAL: Optional[Tuple[Any, Any, Any]] = None
_DRIVER_AVAILABLE: Dict[str, bool] = {}


class VectorOpenError(Exception):
    """Raised when a vector dataset cannot be opened or read."""


class DriverUnavailableError(VectorOpenError):
    """Raised when this GDAL build does not include the driver for a format."""


class VectorFormat:
    """
    One file-based vector format and the GDAL/OGR drivers that read it.

    Drivers are listed in the order they are tried.
    """

    def __init__(
        self,
        name: str,
        extensions: Sequence[str],
        drivers: Sequence[str],
        description: str,
        notes: str = "",
    ) -> None:
        self.name = name
        self.extensions = tuple(extensions)
        self.drivers = tuple(drivers)
        self.description = description
        self.notes = notes

    def __repr__(self) -> str:
        return f"VectorFormat({self.name!r}, extensions={self.extensions!r})"


VECTOR_FORMATS: Tuple[VectorFormat, ...] = (
    VectorFormat(
        name="ESRI Shapefile",
        extensions=(".shp",),
        drivers=("ESRI Shapefile",),
        description="ESRI Shapefile directory of .shp, .shx, and .dbf files.",
        notes=(
            "Requires .shx and .dbf beside the .shp. .prj defines the CRS and "
            ".cpg defines the DBF encoding. A directory of shapefiles opens as "
            "one layer per .shp file."
        ),
    ),
    VectorFormat(
        name="GeoPackage",
        extensions=(".gpkg",),
        drivers=("GPKG",),
        description="GeoPackage, the default vector format in QGIS.",
        notes="SQLite container. A file can hold several named layers.",
    ),
    VectorFormat(
        name="GeoJSON",
        extensions=(".geojson", ".json"),
        drivers=("GeoJSON",),
        description="GeoJSON FeatureCollection or Feature document.",
        notes="A .json file that contains one feature per line is opened as GeoJSON Sequence.",
    ),
    VectorFormat(
        name="GeoJSON Sequence",
        extensions=(".geojsonl", ".geojsons", ".json", ".jsonl"),
        drivers=("GeoJSONSeq",),
        description="Newline-delimited GeoJSON (GeoJSONSeq, GeoJSON Text Sequences).",
        notes="Each non-empty line is one GeoJSON feature.",
    ),
    VectorFormat(
        name="KML / KMZ",
        extensions=(".kml", ".kmz"),
        drivers=("LIBKML", "KML"),
        description="Keyhole Markup Language and compressed KMZ archives.",
        notes=(
            "LIBKML is used when this GDAL build includes it, which is what reads "
            "KMZ directly. The KML driver is the fallback. A KMZ that neither driver "
            "opens is unpacked and the embedded .kml is read."
        ),
    ),
    VectorFormat(
        name="GML",
        extensions=(".gml",),
        drivers=("GML",),
        description="OGC Geography Markup Language.",
        notes="Opening leaves WRITE_GFS=NO so a .gfs schema cache is not written beside the file.",
    ),
    VectorFormat(
        name="GPX",
        extensions=(".gpx",),
        drivers=("GPX",),
        description="GPS Exchange Format.",
        notes="Layers are waypoints, routes, tracks, route_points, and track_points.",
    ),
    VectorFormat(
        name="Delimited text (CSV/TXT)",
        extensions=(".csv", ".txt"),
        drivers=("CSV",),
        description="Comma, tab, semicolon, or pipe separated text with X/Y columns or WKT.",
        notes=(
            "The CSV driver reads .csv and .txt. Point geometry is built from X/Y "
            "columns, or from a WKT column. Pass x_possible_names, y_possible_names, "
            "or geom_possible_names to override the default column list."
        ),
    ),
    VectorFormat(
        name="MapInfo TAB / MIF",
        extensions=(".tab", ".mif", ".mid"),
        drivers=("MapInfo File",),
        description="MapInfo TAB and MIF/MID interchange files.",
        notes=(
            "A TAB dataset needs .tab, .map, .id, and .dat (or .dbf). "
            "A .mid path opens the sibling .mif geometry file."
        ),
    ),
    VectorFormat(
        name="AutoCAD DXF",
        extensions=(".dxf",),
        drivers=("DXF",),
        description="AutoCAD DXF drawing exchange.",
        notes="Entities are exposed as OGR features.",
    ),
    VectorFormat(
        name="AutoCAD DWG",
        extensions=(".dwg",),
        drivers=("CAD", "DWG"),
        description="AutoCAD DWG through GDAL's CAD driver.",
        notes=(
            "The CAD driver (libopencad) is tried first. A DWG driver is used when it is installed."
        ),
    ),
    VectorFormat(
        name="MicroStation DGN",
        extensions=(".dgn",),
        drivers=("DGN", "DGNv8"),
        description="MicroStation design files.",
        notes="DGN v7 uses the DGN driver. DGN v8 uses DGNv8 when that driver is installed.",
    ),
    VectorFormat(
        name="ESRI File Geodatabase",
        extensions=(".gdb",),
        drivers=("OpenFileGDB", "FileGDB"),
        description="ESRI File Geodatabase directory (.gdb).",
        notes=(
            "OpenFileGDB reads .gdb directories without the ESRI FileGDB SDK. "
            "The FileGDB driver is used when it is installed."
        ),
    ),
    VectorFormat(
        name="ESRI Personal Geodatabase",
        extensions=(".mdb",),
        drivers=("PGeo",),
        description="ESRI Personal Geodatabase (.mdb) via ODBC.",
        notes=(
            "On Windows this uses the Microsoft Access ODBC driver. Elsewhere, "
            "GDAL can use MDB Tools when that ODBC driver is installed."
        ),
    ),
    VectorFormat(
        name="SpatiaLite / SQLite",
        extensions=(".sqlite", ".sqlite3", ".db", ".spatialite"),
        drivers=("SQLite",),
        description="SQLite databases, including SpatiaLite.",
        notes="The SQLite driver reads SpatiaLite geometry when GDAL is built with SpatiaLite.",
    ),
    VectorFormat(
        name="FlatGeobuf",
        extensions=(".fgb",),
        drivers=("FlatGeobuf",),
        description="FlatGeobuf single-file binary vectors.",
        notes="Cloud-friendly format with a spatial index in the file.",
    ),
    VectorFormat(
        name="(Geo)Parquet",
        extensions=(".parquet", ".geoparquet"),
        drivers=("Parquet",),
        description="GeoParquet and Parquet datasets.",
        notes="Requires a GDAL build with Apache Arrow (GDAL 3.5 or later).",
    ),
    VectorFormat(
        name="(Geo)Arrow / Feather",
        extensions=(".arrow", ".arrows", ".feather"),
        drivers=("Arrow",),
        description="Apache Arrow IPC and Feather.",
        notes="Requires a GDAL build with Apache Arrow (GDAL 3.5 or later).",
    ),
    VectorFormat(
        name="OpenStreetMap",
        extensions=(".osm", ".pbf"),
        drivers=("OSM",),
        description="OpenStreetMap XML (.osm) and PBF (.pbf).",
        notes="Layers are points, lines, multilinestrings, multipolygons, and other_relations.",
    ),
    VectorFormat(
        name="Microsoft Excel",
        extensions=(".xlsx",),
        drivers=("XLSX",),
        description="Excel Office Open XML workbook (.xlsx).",
        notes="Geometry is read from X/Y columns or a WKT column when the sheet has no geometry.",
    ),
    VectorFormat(
        name="Microsoft Excel (legacy)",
        extensions=(".xls",),
        drivers=("XLS",),
        description="Legacy Excel BIFF workbook (.xls).",
        notes=(
            "The XLS driver is read-only. Geometry is read from X/Y columns or a WKT column "
            "when the sheet has no geometry."
        ),
    ),
    VectorFormat(
        name="LibreOffice ODS",
        extensions=(".ods",),
        drivers=("ODS",),
        description="LibreOffice / OpenDocument spreadsheet.",
        notes="Geometry is read from X/Y columns or a WKT column when the sheet has no geometry.",
    ),
)


class VectorField:
    """One attribute field on a vector layer."""

    def __init__(self, name: str, type_name: str, width: int = 0, precision: int = 0) -> None:
        self.name = name
        self.type_name = type_name
        self.width = width
        self.precision = precision

    def __repr__(self) -> str:
        return f"VectorField({self.name!r}, type_name={self.type_name!r})"


class VectorFeature:
    """
    One vector feature.

    ``geometry`` is a Shapely geometry, or None when the feature has no coordinates.
    ``properties`` holds the attribute values. ``wkt`` is derived from the geometry.
    """

    def __init__(
        self,
        fid: int,
        geometry: Any,
        properties: Dict[str, Any],
        wkt: Optional[str] = None,
    ) -> None:
        self.fid = fid
        self.geometry = geometry
        self.properties = properties
        self._wkt = wkt

    @property
    def wkt(self) -> Optional[str]:
        """Well-known text for the geometry."""
        if self._wkt is None and self.geometry is not None:
            self._wkt = self.geometry.wkt
        return self._wkt

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain dictionary with fid, geometry, properties, and wkt.

        Returns:
            Dictionary copy of the feature
        """
        return {
            "fid": self.fid,
            "geometry": self.geometry,
            "properties": dict(self.properties),
            "wkt": self.wkt,
        }

    def __repr__(self) -> str:
        geom_type = None if self.geometry is None else self.geometry.geom_type
        return (
            f"VectorFeature(fid={self.fid!r}, geometry={geom_type!r}, "
            f"properties={self.properties!r})"
        )


class VectorLayer:
    """One OGR layer inside an opened dataset."""

    def __init__(self, owner: "VectorDataset", index: int, ogr_layer: Any) -> None:
        self._owner = owner
        self.index = index
        self.name = ogr_layer.GetName()
        self.geometry_column = ogr_layer.GetGeometryColumn() or None
        self.fid_column = ogr_layer.GetFIDColumn() or None
        self._fields = _read_fields(ogr_layer)
        self.geometry_type_code = int(ogr_layer.GetGeomType())
        self.geometry_type = _geometry_type_name(ogr_layer)
        self.crs, self.crs_wkt = _crs_from_layer(ogr_layer)
        if self.crs is None and owner._fallback_crs is not None:
            self.crs = owner._fallback_crs
        scan_later = _defer_layer_scan(owner.driver, self.geometry_type_code, getattr(owner, "path", None))
        if scan_later:
            self._feature_count = None
            self._bounds = None
        else:
            self._feature_count = _feature_count(ogr_layer, 1)
            self._bounds = _bounds_from_layer(ogr_layer, 1, self._feature_count)
        self.geometry_source = _detect_geometry_source(ogr_layer, self)

    @property
    def fields(self) -> Tuple[VectorField, ...]:
        """Attribute fields in file order."""
        return self._fields

    @property
    def feature_count(self) -> Optional[int]:
        """
        Number of features.

        None when the driver cannot report a count without a full scan
        (typical for OpenStreetMap).
        """
        return self._feature_count

    @property
    def bounds(self) -> Optional[Bounds]:
        """Layer extent, or None when the driver has no extent."""
        return self._bounds

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Attribute names in file order."""
        return tuple(field.name for field in self._fields)

    def compute_bounds(self) -> Optional[Bounds]:
        """
        Compute the layer extent, scanning features when the header has none.

        Returns:
            Bounds, or None when the layer has no features
        """
        bounds = _bounds_from_layer(self._ogr_layer(), 1, self._feature_count)
        self._bounds = bounds
        return bounds

    def features(
        self,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        """
        Iterate features.

        Args:
            where: OGR SQL attribute filter, such as ``"name = 'a'"``
            bbox: ``(minx, miny, maxx, maxy)`` spatial filter, or a Bounds
            columns: Attribute names to keep. Geometry is still returned.
            limit: Stop after this many features. None reads the whole layer.

        Yields:
            VectorFeature objects in file order
        """
        selected = self._validate_columns(columns)
        limit_value = _validate_limit(limit)
        bbox_value = _validate_bbox(bbox)
        where_value = where.strip() if isinstance(where, str) else where
        if where_value == "":
            where_value = None
        ogr_layer = self._ogr_layer()
        _push_filters(ogr_layer, where_value, bbox_value)
        try:
            ogr_layer.ResetReading()
            produced = 0
            while True:
                if limit_value is not None and produced >= limit_value:
                    break
                try:
                    ogr_feature = _next_feature(ogr_layer)
                except RuntimeError as exc:
                    raise VectorOpenError(
                        f"Could not read the next feature from layer {self.name!r}: {exc}"
                    ) from exc
                if ogr_feature is None:
                    break
                try:
                    feature = _feature_from_ogr(ogr_feature, self._owner, selected)
                finally:
                    ogr_feature = None
                produced += 1
                yield feature
        finally:
            _clear_filters(ogr_layer)

    def read(
        self,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[VectorFeature]:
        """
        Read features into a list.

        Args:
            where: OGR SQL attribute filter
            bbox: Spatial filter
            columns: Attribute names to keep
            limit: Maximum number of features

        Returns:
            List of features
        """
        return list(self.features(where=where, bbox=bbox, columns=columns, limit=limit))

    def to_dataframe(
        self,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """
        Read the layer into a pandas DataFrame.

        When GeoPandas is installed the result is a GeoDataFrame. The geometry
        column contains Shapely geometries. A source field also named
        ``geometry`` is replaced by that column; the original text remains on
        each feature's ``properties`` when you iterate ``features()`` before
        building the table from those features yourself.

        Args:
            where: OGR SQL attribute filter
            bbox: Spatial filter
            columns: Attribute names to keep
            limit: Maximum number of features

        Returns:
            pandas DataFrame, or a GeoPandas GeoDataFrame when GeoPandas is installed
        """
        import pandas as pd

        selected = self._validate_columns(columns)
        field_names = list(self.field_names if selected is None else selected)
        rows: List[Dict[str, Any]] = []
        for feature in self.features(where=where, bbox=bbox, columns=selected, limit=limit):
            row = {"fid": feature.fid}
            row.update(feature.properties)
            row["geometry"] = feature.geometry
            rows.append(row)
        ordered = ["fid", *field_names, "geometry"]
        frame = pd.DataFrame(rows, columns=ordered)
        return _as_geodataframe(frame, self)

    def _validate_columns(self, columns: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        selected = _validate_columns(columns)
        if selected is None:
            return None
        available = set(self.field_names)
        missing = [name for name in selected if name not in available]
        if missing:
            raise VectorOpenError(
                f"Column(s) {missing} are not in layer {self.name!r}. "
                f"Available columns: {list(self.field_names)}"
            )
        return selected

    def _ogr_layer(self) -> Any:
        dataset = self._owner._dataset
        if dataset is None:
            raise VectorOpenError(f"Dataset is closed: {self._owner.path}")
        layer = dataset.GetLayer(self.index)
        if layer is None:
            raise VectorOpenError(
                f"Layer {self.name!r} is no longer available in {self._owner.path}"
            )
        return layer

    def __iter__(self) -> Iterator[VectorFeature]:
        return self.features()

    def __repr__(self) -> str:
        return (
            f"VectorLayer({self.name!r}, features={self._feature_count!r}, "
            f"geometry_type={self.geometry_type!r})"
        )


class VectorDataset:
    """
    An open GDAL/OGR vector dataset.

    Use it as a context manager so file handles are released:

        with open_vectors_data("cities.gpkg") as dataset:
            for feature in dataset:
                ...

    Layer schema, CRS, and feature counts stay available after ``close()``.
    Reading features requires the dataset to still be open.
    """

    def __init__(
        self,
        dataset: Any,
        path: str,
        driver: str,
        format_name: str,
        warnings: Sequence[str],
        fallback_crs: Optional[CRS],
        x_names: Sequence[str],
        y_names: Sequence[str],
        z_names: Sequence[str],
        geom_names: Sequence[str],
        layer: Union[str, int, None],
        tempdir: Optional[tempfile.TemporaryDirectory],
    ) -> None:
        self.path = path
        self.driver = driver
        self.format_name = format_name
        self.warnings = tuple(warnings)
        self._dataset = dataset
        self._fallback_crs = fallback_crs
        self._x_names = tuple(x_names)
        self._y_names = tuple(y_names)
        self._z_names = tuple(z_names)
        self._geom_names = tuple(geom_names)
        self._tempdir = tempdir
        self._layers: Tuple[VectorLayer, ...] = ()
        self._active: Optional[VectorLayer] = None
        try:
            self._layers = tuple(
                VectorLayer(self, index, dataset.GetLayer(index))
                for index in range(dataset.GetLayerCount())
            )
            if not self._layers:
                raise VectorOpenError(f"Dataset has no vector layers: {path}")
            self._active = self.get_layer(layer) if layer is not None else self._layers[0]
        except Exception:
            self.close()
            raise

    @property
    def layers(self) -> Tuple[VectorLayer, ...]:
        """Every layer in GDAL order."""
        return self._layers

    @property
    def layer_names(self) -> List[str]:
        """Layer names in GDAL order."""
        return [layer.name for layer in self._layers]

    @property
    def layer_count(self) -> int:
        """Number of layers."""
        return len(self._layers)

    @property
    def layer(self) -> VectorLayer:
        """
        Layer selected when the dataset was opened.

        This is the first layer when ``layer`` was omitted.
        """
        if self._active is None:
            raise VectorOpenError(f"Dataset is closed: {self.path}")
        return self._active

    @property
    def crs(self) -> Optional[CRS]:
        """CRS of the active layer."""
        return self.layer.crs

    @property
    def bounds(self) -> Optional[Bounds]:
        """Extent of the active layer."""
        return self.layer.bounds

    @property
    def feature_count(self) -> Optional[int]:
        """Feature count of the active layer."""
        return self.layer.feature_count

    def get_layer(self, layer: Union[str, int]) -> VectorLayer:
        """
        Return a layer by name or index.

        Args:
            layer: Layer name or zero-based index

        Returns:
            VectorLayer

        Raises:
            VectorOpenError: If the layer is not in the dataset
        """
        if isinstance(layer, bool) or not isinstance(layer, (str, int)):
            raise TypeError("layer must be a layer name or an integer index")
        if isinstance(layer, int):
            if layer < 0 or layer >= len(self._layers):
                raise VectorOpenError(
                    f"Layer index {layer} is outside 0..{len(self._layers) - 1} for {self.path}"
                )
            return self._layers[layer]
        exact = [item for item in self._layers if item.name == layer]
        if not exact:
            folded = [item for item in self._layers if item.name.lower() == layer.lower()]
            if len(folded) == 1:
                exact = folded
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise VectorOpenError(
                f"Layer name {layer!r} matches more than one layer in {self.path}: "
                f"{[item.name for item in exact]}"
            )
        raise VectorOpenError(
            f"Layer {layer!r} was not found in {self.path}. Available layers: {self.layer_names}"
        )

    def features(
        self,
        layer: Union[str, int, None] = None,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        """
        Iterate features from a layer.

        Args:
            layer: Layer name or index. Defaults to the active layer.
            where: OGR SQL attribute filter
            bbox: Spatial filter
            columns: Attribute names to keep
            limit: Maximum number of features

        Yields:
            VectorFeature objects
        """
        target = self.layer if layer is None else self.get_layer(layer)
        return target.features(where=where, bbox=bbox, columns=columns, limit=limit)

    def read(
        self,
        layer: Union[str, int, None] = None,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[VectorFeature]:
        """
        Read features from a layer into a list.

        Args:
            layer: Layer name or index. Defaults to the active layer.
            where: OGR SQL attribute filter
            bbox: Spatial filter
            columns: Attribute names to keep
            limit: Maximum number of features

        Returns:
            List of features
        """
        return list(
            self.features(layer=layer, where=where, bbox=bbox, columns=columns, limit=limit)
        )

    def to_dataframe(
        self,
        layer: Union[str, int, None] = None,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """
        Read a layer into a DataFrame.

        Args:
            layer: Layer name or index. Defaults to the active layer.
            where: OGR SQL attribute filter
            bbox: Spatial filter
            columns: Attribute names to keep
            limit: Maximum number of features

        Returns:
            pandas DataFrame, or a GeoPandas GeoDataFrame when GeoPandas is installed
        """
        target = self.layer if layer is None else self.get_layer(layer)
        return target.to_dataframe(where=where, bbox=bbox, columns=columns, limit=limit)

    def to_dataframes(self) -> Dict[str, Any]:
        """
        Read every layer into a DataFrame.

        Returns:
            Mapping of layer name to DataFrame
        """
        return {layer.name: layer.to_dataframe() for layer in self._layers}

    def close(self) -> None:
        """Release the GDAL dataset and any temporary KMZ extract."""
        self._dataset = None
        tempdir = self._tempdir
        self._tempdir = None
        if tempdir is not None:
            tempdir.cleanup()

    def __enter__(self) -> "VectorDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[VectorFeature]:
        return self.features()

    def __getitem__(self, layer: Union[str, int]) -> VectorLayer:
        return self.get_layer(layer)

    def __repr__(self) -> str:
        return (
            f"VectorDataset({self.path!r}, driver={self.driver!r}, layers={self.layer_names!r})"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return


def open_vectors_data(
    path: Union[str, Path],
    layer: Union[str, int, None] = None,
    driver: Optional[str] = None,
    encoding: Optional[str] = None,
    crs: Union[str, int, CRS, None] = None,
    x_possible_names: Optional[Sequence[str]] = None,
    y_possible_names: Optional[Sequence[str]] = None,
    z_possible_names: Optional[Sequence[str]] = None,
    geom_possible_names: Optional[Sequence[str]] = None,
    open_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
) -> VectorDataset:
    """
    Open a file-based vector dataset with GDAL/OGR.

    Args:
        path: Dataset path. ``/vsi...`` and URL paths are passed to GDAL unchanged.
            A MapInfo ``.mid`` path opens the sibling ``.mif``. A directory ending
            in ``.gdb`` opens as a File Geodatabase. Any other directory that
            contains shapefiles opens as a multi-layer shapefile dataset.
        layer: Layer name or zero-based index. Defaults to the first layer.
        driver: GDAL/OGR driver short name. When omitted, the driver is chosen
            from the file extension.
        encoding: DBF encoding for ESRI Shapefile, overriding a ``.cpg`` file.
        crs: CRS used when the file itself has no coordinate system (common for
            CSV, TXT, and spreadsheets). An EPSG integer, ``"EPSG:4326"``, a WKT
            string, or a :class:`~geolattice.crs.CRS` are accepted.
            A CRS already stored in the file is kept.
        x_possible_names: X or longitude column names for delimited text and
            spreadsheets. Defaults to :data:`X_POSSIBLE_NAMES`.
        y_possible_names: Y or latitude column names. Defaults to
            :data:`Y_POSSIBLE_NAMES`.
        z_possible_names: Optional Z column names. Defaults to
            :data:`Z_POSSIBLE_NAMES`.
        geom_possible_names: WKT column names. Defaults to
            :data:`GEOM_POSSIBLE_NAMES`.
        open_options: Extra GDAL open options, either ``{"KEY": "VALUE"}`` or
            ``["KEY=VALUE", ...]``. These override the options this function
            would otherwise set for that driver.

    Returns:
        Open vector dataset. Close it with :meth:`VectorDataset.close` or a
        ``with`` block.

    Raises:
        FileNotFoundError: If a local path does not exist
        DriverUnavailableError: If this GDAL build lacks the required driver
        VectorOpenError: If the file is incomplete or GDAL cannot read it
    """
    user_options = _normalize_open_options(open_options)
    x_names = _resolve_names(x_possible_names, X_POSSIBLE_NAMES)
    y_names = _resolve_names(y_possible_names, Y_POSSIBLE_NAMES)
    z_names = _resolve_names(z_possible_names, Z_POSSIBLE_NAMES)
    geom_names = _resolve_names(geom_possible_names, GEOM_POSSIBLE_NAMES)
    fallback_crs = _coerce_crs(crs)

    requested = os.fspath(path)
    location, local = _resolve_input(requested)
    if local is not None and local.suffix.lower() == ".mid":
        sibling = _sibling(local, ".mif")
        if sibling is None:
            raise VectorOpenError(
                f"MapInfo MID file {local.name!r} has no sibling .mif geometry file. "
                "Open the .mif file; .mid holds attributes only."
            )
        local = sibling
        location = str(local)

    warnings = _sidecar_warnings(local) if local is not None else []
    formats = _candidate_formats(local, location)
    drivers = _driver_list(formats, driver)

    tempdir: Optional[tempfile.TemporaryDirectory] = None
    try:
        dataset, driver_name = _open_with_drivers(
            location, drivers, formats, user_options, encoding, x_names, y_names, z_names
        )
    except VectorOpenError:
        if local is None or local.suffix.lower() != ".kmz":
            raise
        dataset, driver_name, tempdir = _open_kmz_fallback(
            local, user_options, encoding, x_names, y_names, z_names
        )

    format_name = _format_name(driver_name, formats)
    return VectorDataset(
        dataset=dataset,
        path=location,
        driver=driver_name,
        format_name=format_name,
        warnings=warnings,
        fallback_crs=fallback_crs,
        x_names=x_names,
        y_names=y_names,
        z_names=z_names,
        geom_names=geom_names,
        layer=layer,
        tempdir=tempdir,
    )


def open_vector(
    path: Union[str, Path],
    layer: Union[str, int, None] = None,
    **kwargs: Any,
) -> VectorDataset:
    """
    Open a vector dataset. Alias of :func:`open_vectors_data`.

    Args:
        path: Dataset path
        layer: Layer name or index
        **kwargs: Passed through to :func:`open_vectors_data`

    Returns:
        Open vector dataset
    """
    return open_vectors_data(path, layer=layer, **kwargs)


def read_vector_data(
    path: Union[str, Path],
    layer: Union[str, int, None] = None,
    **kwargs: Any,
) -> VectorDataset:
    """
    Read a file-based vector dataset. Alias of :func:`open_vectors_data`.

    Args:
        path: Dataset path
        layer: Layer name or index
        **kwargs: Passed through to :func:`open_vectors_data`

    Returns:
        Open vector dataset
    """
    return open_vectors_data(path, layer=layer, **kwargs)


def supported_formats() -> List[Dict[str, Any]]:
    """
    List the file-based formats this module opens, and which drivers exist.

    Returns:
        One dictionary per format, with name, extensions, drivers,
        available_drivers, available, description, and notes.
    """
    records: List[Dict[str, Any]] = []
    for fmt in VECTOR_FORMATS:
        available = [name for name in fmt.drivers if driver_available(name)]
        records.append(
            {
                "name": fmt.name,
                "extensions": list(fmt.extensions),
                "drivers": list(fmt.drivers),
                "available_drivers": available,
                "available": bool(available),
                "description": fmt.description,
                "notes": fmt.notes,
            }
        )
    return records


def driver_available(name: str) -> bool:
    """
    Return whether this GDAL build can open datasets with ``name``.

    Args:
        name: OGR driver short name, such as ``"GPKG"`` or ``"Parquet"``

    Returns:
        True when ``gdal.GetDriverByName(name)`` finds the driver
    """
    cached = _DRIVER_AVAILABLE.get(name)
    if cached is not None:
        return cached
    gdal, _, _ = _gdal_modules()
    found = gdal.GetDriverByName(name) is not None
    _DRIVER_AVAILABLE[name] = found
    return found


def _gdal_modules() -> Tuple[Any, Any, Any]:
    """Import GDAL/OGR and enable exceptions without loading the NumPy array bridge."""
    global _GDAL
    if _GDAL is not None:
        return _GDAL
    try:
        from osgeo import gdal, ogr, osr
    except ImportError as exc:
        raise VectorOpenError(
            "GDAL/OGR Python bindings are required to open vector data. "
            "Install the gdal package that matches the GDAL library on this machine."
        ) from exc
    # gdal.UseExceptions() also imports osgeo.gdal_array, which fails when the
    # system GDAL bindings were compiled against NumPy 1.x. Vector I/O only
    # needs the OGR modules, and the private hooks still silence the GDAL 4 warning.
    gdal._UseExceptions()
    ogr._UseExceptions()
    osr._UseExceptions()
    _GDAL = (gdal, ogr, osr)
    return _GDAL


def _gdal_version() -> str:
    gdal, _, _ = _gdal_modules()
    return str(gdal.VersionInfo("RELEASE_NAME"))


def _resolve_input(text: str) -> Tuple[str, Optional[Path]]:
    if _is_virtual_path(text):
        return text, None
    local = Path(text).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Vector dataset does not exist: {local}")
    return str(local), local


def _is_virtual_path(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("/vsi") or "://" in text


def _sibling(path: Path, suffix: str) -> Optional[Path]:
    candidate = path.with_suffix(suffix)
    if candidate.exists():
        return candidate
    parent = path.parent
    if not parent.is_dir():
        return None
    wanted = (path.stem + suffix).lower()
    for child in parent.iterdir():
        if child.name.lower() == wanted:
            return child
    return None


def _sidecar_warnings(path: Path) -> List[str]:
    if path.is_dir():
        return []
    suffix = path.suffix.lower()
    warnings: List[str] = []
    if suffix == ".shp":
        missing = [ext for ext in (".shx", ".dbf") if _sibling(path, ext) is None]
        if missing:
            raise VectorOpenError(
                f"ESRI Shapefile {path.name!r} is missing required sidecar file(s): "
                f"{', '.join(missing)}. A shapefile needs .shp, .shx, and .dbf. "
                ".prj (CRS) and .cpg (encoding) are optional."
            )
        if _sibling(path, ".prj") is None:
            warnings.append(
                f"No .prj found beside {path.name!r}. The shapefile has no CRS definition."
            )
    elif suffix == ".tab":
        missing_parts: List[str] = []
        if _sibling(path, ".map") is None:
            missing_parts.append(".map")
        if _sibling(path, ".dat") is None and _sibling(path, ".dbf") is None:
            missing_parts.append(".dat/.dbf")
        if _sibling(path, ".id") is None:
            missing_parts.append(".id")
        if missing_parts:
            raise VectorOpenError(
                f"MapInfo TAB dataset {path.name!r} is missing {', '.join(missing_parts)}. "
                "A TAB dataset needs .tab, .map, .id, and .dat (or .dbf)."
            )
    elif suffix == ".mif" and _sibling(path, ".mid") is None:
        warnings.append(
            f"No .mid attribute file beside {path.name!r}. "
            "Geometry will be read without attributes."
        )
    return warnings


def _candidate_formats(local: Optional[Path], location: str) -> List[VectorFormat]:
    if local is not None and local.is_dir():
        if local.suffix.lower() == ".gdb":
            return [fmt for fmt in VECTOR_FORMATS if fmt.name == "ESRI File Geodatabase"]
        if _directory_has_suffix(local, ".shp"):
            return [fmt for fmt in VECTOR_FORMATS if fmt.name == "ESRI Shapefile"]
        return []
    suffix = Path(location).suffix.lower()
    formats = [fmt for fmt in VECTOR_FORMATS if suffix in fmt.extensions]
    if (
        suffix in {".json", ".geojson"}
        and local is not None
        and local.is_file()
        and _looks_like_geojsonseq(local)
    ):
        sequence = [fmt for fmt in formats if "GeoJSONSeq" in fmt.drivers]
        others = [fmt for fmt in formats if "GeoJSONSeq" not in fmt.drivers]
        return sequence + others
    return formats


def _directory_has_suffix(directory: Path, suffix: str) -> bool:
    try:
        children = list(directory.iterdir())
    except OSError:
        return False
    return any(child.is_file() and child.suffix.lower() == suffix for child in children)


def _looks_like_geojsonseq(path: Path) -> bool:
    """True when two lines are JSON documents and the first is not a FeatureCollection."""
    lines: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                if len(lines) >= 2:
                    break
    except OSError:
        return False
    if len(lines) < 2:
        return False
    first, second = lines[0], lines[1]
    if "FeatureCollection" in first:
        return False
    return first.startswith("{") and second.startswith("{")


def _driver_list(formats: Sequence[VectorFormat], explicit: Optional[str]) -> List[str]:
    if explicit:
        return [explicit]
    ordered: List[str] = []
    for fmt in formats:
        for name in fmt.drivers:
            if name not in ordered:
                ordered.append(name)
    return ordered


def _format_name(driver_name: str, formats: Sequence[VectorFormat]) -> str:
    for fmt in formats:
        if driver_name in fmt.drivers:
            return fmt.name
    return driver_name


def _resolve_names(names: Optional[Sequence[str]], defaults: Sequence[str]) -> Tuple[str, ...]:
    if names is None:
        return tuple(defaults)
    return tuple(str(name).strip() for name in names if str(name).strip())


def _normalize_open_options(
    options: Optional[Union[Sequence[str], Dict[str, Any]]]
) -> Dict[str, str]:
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(key): str(value) for key, value in options.items()}
    normalized: Dict[str, str] = {}
    for item in options:
        text = str(item)
        if "=" not in text:
            raise ValueError(f"open option must be KEY=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        normalized[key] = value
    return normalized


def _options_for_driver(
    driver_name: str,
    user_options: Dict[str, str],
    encoding: Optional[str],
    x_names: Sequence[str],
    y_names: Sequence[str],
    z_names: Sequence[str],
) -> List[str]:
    options: Dict[str, str] = {}
    if driver_name == "CSV":
        options.update(
            {
                "SEPARATOR": "AUTO",
                "HEADERS": "YES",
                "AUTODETECT_TYPE": "YES",
                "KEEP_GEOM_COLUMNS": "YES",
                # The CSV driver's default geometry column name is WKT. Parsing that
                # column inside GDAL aborts on EWKT (SRID=...;...), so name a column
                # that is not present and parse WKT in this module instead.
                "GEOM_POSSIBLE_NAMES": "__geolattice_no_geom__",
            }
        )
        if x_names:
            options["X_POSSIBLE_NAMES"] = ",".join(x_names)
        if y_names:
            options["Y_POSSIBLE_NAMES"] = ",".join(y_names)
        if z_names:
            options["Z_POSSIBLE_NAMES"] = ",".join(z_names)
    if driver_name == "GML":
        options["WRITE_GFS"] = "NO"
    if driver_name == "ESRI Shapefile" and encoding:
        options["ENCODING"] = encoding
    options.update(user_options)
    return [f"{key}={value}" for key, value in options.items()]


def _open_with_drivers(
    location: str,
    drivers: Sequence[str],
    formats: Sequence[VectorFormat],
    user_options: Dict[str, str],
    encoding: Optional[str],
    x_names: Sequence[str],
    y_names: Sequence[str],
    z_names: Sequence[str],
) -> Tuple[Any, str]:
    if drivers:
        available = [name for name in drivers if driver_available(name)]
        if not available:
            raise DriverUnavailableError(
                _unavailable_message(location, drivers, formats)
            )
    else:
        available = []

    errors: List[str] = []
    attempts = available or [""]
    for name in attempts:
        option_list = _options_for_driver(
            name, user_options, encoding, x_names, y_names, z_names
        )
        try:
            dataset = _open_ex(location, name or None, option_list)
        except RuntimeError as exc:
            message = str(exc).strip()
            if "No such file or directory" in message:
                raise FileNotFoundError(message) from exc
            errors.append(f"{name or 'GDAL'}: {message}")
            continue
        if dataset is None or dataset.GetLayerCount() == 0:
            errors.append(f"{name or 'GDAL'}: dataset has no vector layers")
            continue
        actual = dataset.GetDriver().ShortName
        return dataset, actual

    raise VectorOpenError(_open_failure_message(location, available, formats, errors))


def _open_ex(location: str, driver_name: Optional[str], option_list: Sequence[str]) -> Any:
    gdal, _, _ = _gdal_modules()
    flags = gdal.OF_VECTOR | gdal.OF_READONLY | getattr(gdal, "OF_VERBOSE_ERROR", 0)
    # None is not a valid SWIG value for the char** arguments, so omit them.
    kwargs: Dict[str, Any] = {"nOpenFlags": flags}
    if driver_name:
        kwargs["allowed_drivers"] = [driver_name]
    if option_list:
        kwargs["open_options"] = [str(item) for item in option_list]
    return gdal.OpenEx(_driver_location(location, driver_name), **kwargs)


def _driver_location(location: str, driver_name: Optional[str]) -> str:
    """Point the CSV driver at .txt and other non-.csv paths."""
    if driver_name != "CSV" or location.upper().startswith("CSV:"):
        return location
    suffix = Path(location.split("?", 1)[0]).suffix.lower()
    if suffix == ".csv":
        return location
    return "CSV:" + location


def _unavailable_message(
    location: str, drivers: Sequence[str], formats: Sequence[VectorFormat]
) -> str:
    hints = _hints_for(drivers)
    return (
        f"Cannot open {location!r} as {_format_phrase(formats)}. "
        f"GDAL {_gdal_version()} does not include driver(s): {', '.join(drivers)}. {hints}"
    ).strip()


def _open_failure_message(
    location: str,
    drivers: Sequence[str],
    formats: Sequence[VectorFormat],
    errors: Sequence[str],
) -> str:
    attempted = ", ".join(drivers) if drivers else "GDAL's format detection"
    detail = " ".join(errors).strip()
    hints = _hints_for(drivers)
    message = (
        f"Could not open {location!r} as {_format_phrase(formats)} "
        f"with GDAL/OGR driver(s) {attempted}."
    )
    if detail:
        message = f"{message} {detail}"
    if hints:
        message = f"{message} {hints}"
    return message


def _format_phrase(formats: Sequence[VectorFormat]) -> str:
    if not formats:
        return "a vector dataset"
    names = [fmt.name for fmt in formats]
    if len(names) == 1:
        return names[0]
    return " or ".join(names)


def _hints_for(drivers: Sequence[str]) -> str:
    hints: List[str] = []
    for name in drivers:
        hint = _DRIVER_HINTS.get(name)
        if hint and hint not in hints:
            hints.append(hint)
    return " ".join(hints)


def _open_kmz_fallback(
    path: Path,
    user_options: Dict[str, str],
    encoding: Optional[str],
    x_names: Sequence[str],
    y_names: Sequence[str],
    z_names: Sequence[str],
) -> Tuple[Any, str, tempfile.TemporaryDirectory]:
    if not zipfile.is_zipfile(path):
        raise VectorOpenError(f"KMZ file is not a zip archive: {path}")
    tempdir = tempfile.TemporaryDirectory(prefix="geolattice-kmz-")
    try:
        with zipfile.ZipFile(path) as archive:
            _safe_extract_kmz(archive, Path(tempdir.name))
            kml_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".kml") and not name.endswith("/")
            ]
        if not kml_names:
            raise VectorOpenError(f"KMZ archive {path.name!r} does not contain a .kml file.")
        kml_names.sort(key=lambda name: (Path(name).name.lower() != "doc.kml", len(name)))
        kml_path = str(Path(tempdir.name) / kml_names[0])
        formats = [fmt for fmt in VECTOR_FORMATS if fmt.name == "KML / KMZ"]
        dataset, driver_name = _open_with_drivers(
            kml_path,
            ["LIBKML", "KML"],
            formats,
            user_options,
            encoding,
            x_names,
            y_names,
            z_names,
        )
        return dataset, driver_name, tempdir
    except Exception:
        tempdir.cleanup()
        raise


def _safe_extract_kmz(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise VectorOpenError(
                f"KMZ member {member.filename!r} escapes the archive and was not extracted."
            )
    archive.extractall(destination)


def _coerce_crs(value: Union[str, int, CRS, None]) -> Optional[CRS]:
    if value is None:
        return None
    if isinstance(value, CRS):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return CRS.from_epsg(value)
    return CRS(str(value))


def _read_fields(ogr_layer: Any) -> Tuple[VectorField, ...]:
    definition = ogr_layer.GetLayerDefn()
    fields: List[VectorField] = []
    for index in range(definition.GetFieldCount()):
        field = definition.GetFieldDefn(index)
        fields.append(
            VectorField(
                name=field.GetName(),
                type_name=field.GetTypeName(),
                width=int(field.GetWidth() or 0),
                precision=int(field.GetPrecision() or 0),
            )
        )
    return tuple(fields)


def _geometry_type_name(ogr_layer: Any) -> Optional[str]:
    _, ogr, _ = _gdal_modules()
    code = ogr_layer.GetGeomType()
    if code == ogr.wkbNone:
        return None
    return str(ogr.GeometryTypeToName(code))


def _next_feature(ogr_layer: Any) -> Any:
    """Read the next feature, keeping a bad geometry from aborting the layer.

    The CSV driver raises ``Unsupported WKB type`` when a row's WKT is EWKT
    (``SRID=...;POINT ...``). With exceptions disabled it still returns the
    feature and leaves the geometry empty, so this module can parse the text.
    """
    gdal, ogr, _ = _gdal_modules()
    ogr._DontUseExceptions()
    gdal._DontUseExceptions()
    try:
        return ogr_layer.GetNextFeature()
    finally:
        gdal.ErrorReset()
        gdal._UseExceptions()
        ogr._UseExceptions()


def _defer_layer_scan(driver_name: str, geometry_type_code: int, path: Optional[str] = None) -> bool:
    """Skip count and extent scans that would touch every feature up front.

    OpenStreetMap reports counts only by reading the whole file. The CSV driver
    can also corrupt its reader if WKT geometry is scanned before iteration.
    Vector tiles (PMTiles, MVT) count features by decoding every tile at the
    feature zoom; over the network (``/vsicurl/``, ``/vsis3/``, ...) that means
    downloading most of the archive, so the scan is deferred for remote paths.
    """
    if driver_name == "OSM":
        return True
    if driver_name in ("PMTiles", "MVT") and str(path or "").startswith("/vsi"):
        return True
    if driver_name != "CSV":
        return False
    _, ogr, _ = _gdal_modules()
    return geometry_type_code in (ogr.wkbUnknown, ogr.wkbNone)


def _feature_count(ogr_layer: Any, force: int) -> Optional[int]:
    try:
        count = int(ogr_layer.GetFeatureCount(force))
    except RuntimeError:
        return None
    if count < 0:
        return None
    return count


def _bounds_from_layer(
    ogr_layer: Any, force: int, feature_count: Optional[int]
) -> Optional[Bounds]:
    """Read an OGR extent. GetExtent returns (minx, maxx, miny, maxy)."""
    if feature_count == 0:
        return None
    try:
        min_x, max_x, min_y, max_y = ogr_layer.GetExtent(force)
    except RuntimeError:
        return None
    # force=0 returns (0, 0, 0, 0) when the driver has not cached an extent.
    if force == 0 and min_x == max_x == min_y == max_y == 0.0:
        return None
    if min_x > max_x or min_y > max_y:
        return None
    return Bounds(min_x, min_y, max_x, max_y)


def _crs_from_layer(ogr_layer: Any) -> Tuple[Optional[CRS], Optional[str]]:
    try:
        spatial_ref = ogr_layer.GetSpatialRef()
    except RuntimeError:
        return None, None
    if spatial_ref is None:
        return None, None
    _, _, osr = _gdal_modules()
    try:
        cloned = spatial_ref.Clone()
    except RuntimeError:
        return None, None
    traditional = getattr(osr, "OAMS_TRADITIONAL_GIS_ORDER", None)
    if traditional is not None and hasattr(cloned, "SetAxisMappingStrategy"):
        try:
            cloned.SetAxisMappingStrategy(traditional)
        except RuntimeError:
            pass
    try:
        wkt = cloned.ExportToWkt() or None
    except RuntimeError:
        # The CAD driver can attach a spatial reference that OSR cannot export.
        wkt = None
    authority = None
    code = None
    try:
        authority = cloned.GetAuthorityName(None)
        code = cloned.GetAuthorityCode(None)
    except RuntimeError:
        authority = None
        code = None
    if not code:
        try:
            cloned.AutoIdentifyEPSG()
            authority = cloned.GetAuthorityName(None)
            code = cloned.GetAuthorityCode(None)
        except RuntimeError:
            authority = None
            code = None
    if authority and code and str(authority).upper() == "EPSG":
        try:
            return CRS(int(code)), wkt
        except (TypeError, ValueError):
            pass
    if wkt:
        try:
            return CRS.from_wkt(wkt), wkt
        except (RuntimeError, ValueError):
            return None, wkt
    return None, None


def _detect_geometry_source(ogr_layer: Any, layer: VectorLayer) -> str:
    _, ogr, _ = _gdal_modules()
    code = ogr_layer.GetGeomType()
    if code not in (ogr.wkbNone, ogr.wkbUnknown):
        return "layer"
    ogr_layer.ResetReading()
    feature = _next_feature(ogr_layer)
    has_geometry = False
    if feature is not None:
        geometry = feature.GetGeometryRef()
        has_geometry = geometry is not None and not geometry.IsEmpty()
        feature = None
    ogr_layer.ResetReading()
    if has_geometry:
        return "layer"
    lookup = {name.lower(): name for name in layer.field_names}
    has_wkt = _find_field(layer._owner._geom_names, lookup) is not None
    has_xy = (
        _find_field(layer._owner._x_names, lookup) is not None
        and _find_field(layer._owner._y_names, lookup) is not None
    )
    if has_wkt or has_xy:
        return "attributes"
    return "none"


def _feature_from_ogr(
    ogr_feature: Any, owner: VectorDataset, columns: Optional[Sequence[str]]
) -> VectorFeature:
    # Geometry columns have to be visible to X/Y and WKT detection even when the
    # caller asked for a subset of attribute columns.
    properties = _properties_from_ogr(ogr_feature, None)
    ogr_geometry = ogr_feature.GetGeometryRef()
    geometry = _shapely_from_ogr(ogr_geometry)
    wkt: Optional[str] = None
    if geometry is None and ogr_geometry is not None and not ogr_geometry.IsEmpty():
        wkt = _ogr_wkt(ogr_geometry)
    if geometry is None and wkt is None:
        geometry = _geometry_from_properties(
            properties, owner._x_names, owner._y_names, owner._z_names, owner._geom_names
        )
    if columns is not None:
        properties = {name: properties[name] for name in columns}
    return VectorFeature(
        fid=int(ogr_feature.GetFID()), geometry=geometry, properties=properties, wkt=wkt
    )


def _properties_from_ogr(ogr_feature: Any, columns: Optional[Sequence[str]]) -> Dict[str, Any]:
    _, ogr, _ = _gdal_modules()
    definition = ogr_feature.GetDefnRef()
    selected = None if columns is None else set(columns)
    properties: Dict[str, Any] = {}
    date_types = {ogr.OFTDate, ogr.OFTTime, ogr.OFTDateTime}
    boolean_subtype = getattr(ogr, "OFSTBoolean", None)
    for index in range(definition.GetFieldCount()):
        field = definition.GetFieldDefn(index)
        name = field.GetName()
        if selected is not None and name not in selected:
            continue
        if ogr_feature.IsFieldNull(index) or not ogr_feature.IsFieldSet(index):
            properties[name] = None
            continue
        if boolean_subtype is not None and field.GetSubType() == boolean_subtype:
            properties[name] = bool(ogr_feature.GetFieldAsInteger(index))
            continue
        if field.GetType() == ogr.OFTBinary:
            properties[name] = bytes(ogr_feature.GetFieldAsBinary(index))
            continue
        if field.GetType() in date_types and hasattr(ogr_feature, "GetFieldAsISO8601DateTime"):
            properties[name] = ogr_feature.GetFieldAsISO8601DateTime(index)
            continue
        value = ogr_feature.GetField(index)
        if isinstance(value, tuple):
            value = list(value)
        properties[name] = value
    return properties


def _shapely_from_ogr(geometry: Any) -> Any:
    if geometry is None or geometry.IsEmpty():
        return None
    raw = _ogr_export(geometry, ("ExportToIsoWkb", "ExportToWkb"))
    if raw is not None:
        try:
            return _from_wkb(bytes(raw))
        except Exception:
            pass
    text = _ogr_export(geometry, ("ExportToIsoWkt", "ExportToWkt"))
    if text:
        try:
            return _load_wkt(text)
        except Exception:
            return None
    return None


def _ogr_wkt(geometry: Any) -> Optional[str]:
    text = _ogr_export(geometry, ("ExportToIsoWkt", "ExportToWkt"))
    return text if isinstance(text, str) and text else None


def _ogr_export(geometry: Any, names: Sequence[str]) -> Any:
    """Call the first OGR export that this geometry supports."""
    for name in names:
        method = getattr(geometry, name, None)
        if method is None:
            continue
        try:
            return method()
        except RuntimeError:
            continue
    return None


def _geometry_from_properties(
    properties: Dict[str, Any],
    x_names: Sequence[str],
    y_names: Sequence[str],
    z_names: Sequence[str],
    geom_names: Sequence[str],
) -> Any:
    lookup = {key.lower(): key for key in properties}
    geom_field = _find_field(geom_names, lookup)
    if geom_field is not None:
        raw_value = properties.get(geom_field)
        if raw_value is not None and str(raw_value).strip():
            text = _strip_ewkt(str(raw_value).strip())
            if _looks_like_wkt(text):
                try:
                    return _load_wkt(text)
                except Exception as exc:
                    raise VectorOpenError(
                        f"Could not parse WKT from column {geom_field!r}: {text[:80]}"
                    ) from exc
    x_field = _find_field(x_names, lookup)
    y_field = _find_field(y_names, lookup)
    if x_field is None or y_field is None:
        return None
    x_value = _as_float(properties.get(x_field))
    y_value = _as_float(properties.get(y_field))
    if x_value is None or y_value is None:
        return None
    z_field = _find_field(z_names, lookup)
    if z_field is not None:
        z_value = _as_float(properties.get(z_field))
        if z_value is not None:
            return Point(x_value, y_value, z_value)
    return Point(x_value, y_value)


def _find_field(candidates: Sequence[str], lookup: Dict[str, str]) -> Optional[str]:
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _strip_ewkt(value: str) -> str:
    if value.upper().startswith("SRID="):
        _, separator, remainder = value.partition(";")
        if separator:
            return remainder.strip()
    return value


def _looks_like_wkt(value: str) -> bool:
    token = value.lstrip().upper()
    return token.startswith(_WKT_PREFIXES)


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _push_filters(
    ogr_layer: Any,
    where: Optional[str],
    bbox: Optional[Tuple[float, float, float, float]],
) -> None:
    try:
        ogr_layer.SetAttributeFilter(where)
        if bbox is None:
            ogr_layer.SetSpatialFilter(None)
        else:
            min_x, min_y, max_x, max_y = bbox
            ogr_layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
    except RuntimeError as exc:
        _clear_filters(ogr_layer)
        raise VectorOpenError(f"Could not apply the feature filter: {exc}") from exc


def _clear_filters(ogr_layer: Any) -> None:
    try:
        ogr_layer.SetAttributeFilter(None)
        ogr_layer.SetSpatialFilter(None)
    except RuntimeError:
        return


def _validate_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be zero or a positive integer")
    return limit


def _validate_bbox(
    bbox: Union[Sequence[float], Bounds, None]
) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    if isinstance(bbox, Bounds):
        values: Tuple[float, ...] = bbox.to_tuple()
    else:
        values = tuple(float(item) for item in bbox)
    if len(values) != 4:
        raise ValueError("bbox must be (minx, miny, maxx, maxy)")
    min_x, min_y, max_x, max_y = values
    if min_x > max_x or min_y > max_y:
        raise ValueError(
            f"Invalid bbox: minx ({min_x}) > maxx ({max_x}) or miny ({min_y}) > maxy ({max_y})"
        )
    return min_x, min_y, max_x, max_y


def _validate_columns(columns: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if columns is None:
        return None
    names = tuple(str(name) for name in columns)
    if len(names) != len(set(names)):
        raise ValueError("columns contains duplicate names")
    return names


def _as_geodataframe(frame: Any, layer: VectorLayer) -> Any:
    try:
        import geopandas as gpd
    except ImportError:
        return frame
    crs = layer.crs_wkt if layer.crs_wkt else (None if layer.crs is None else str(layer.crs))
    try:
        return gpd.GeoDataFrame(frame, geometry="geometry", crs=crs)
    except Exception:
        return frame


__all__ = [
    "X_POSSIBLE_NAMES",
    "Y_POSSIBLE_NAMES",
    "Z_POSSIBLE_NAMES",
    "GEOM_POSSIBLE_NAMES",
    "VECTOR_FORMATS",
    "VectorFormat",
    "VectorField",
    "VectorFeature",
    "VectorLayer",
    "VectorDataset",
    "VectorOpenError",
    "DriverUnavailableError",
    "open_vectors_data",
    "open_vector",
    "read_vector_data",
    "supported_formats",
    "driver_available",
]
