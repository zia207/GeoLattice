"""
Open cloud-native vector data, lakehouse tables, and related spatial layouts.

Cloud-native files are laid out so a client can fetch a footer, an index, or a
bounding box with HTTP range requests instead of downloading the whole object:

- GeoParquet stores analysis-ready features in columnar Parquet. Selective
  column reads are the fast path. Profiles built on GeoParquet (fiboa field
  boundaries, stac-geoparquet scene footprints, and the Overture Maps schema)
  are recognized from the file metadata and columns.
- GeoArrow is the in-memory Arrow layout for passing geometry between tools.
  On disk it is an Arrow IPC / Feather file, often with a ``geoarrow.*``
  extension type.
- FlatGeobuf (``.fgb``) is a row-oriented FlatBuffers file with an optional
  packed Hilbert R-tree at the front. GDAL issues range requests for the
  features that intersect a bounding box.
- PMTiles (``.pmtiles``) is one archive of map tiles. The header, directory,
  and tile bytes are read with range requests. Vector tiles can also be
  decoded to features. The archive is for map display, not columnar analysis.
- GeoPackage (``.gpkg``) is the OGC SQLite container and the practical
  replacement for a shapefile. It is not cloud-native, though GDAL can read it
  through ``/vsicurl/``.
- COPC (``.copc.laz``) is a LAZ point cloud organized as an octree so a client
  can range-request the voxels inside a cube.

Lakehouse tables:

- Apache Iceberg v3 added native ``geometry`` and ``geography`` types. This
  reader records those types and reads the Parquet data files under the table.
- Delta Lake still has no geometry type. Geometry is read from WKB columns.
- Parquet's own ``GEOMETRY`` and ``GEOGRAPHY`` logical types, added in 2025,
  are detected when the Parquet library reports them.

Other layouts:

- GeoJSON Sequence / newline-delimited GeoJSON is streamed a feature at a time.
- H3 or S2 tables store a cell id instead of, or beside, a geometry column.
- TileDB arrays and Lance datasets are opened when those libraries are installed.
  Lance has no geo spec; WKB columns are decoded.

Example:
    from geolattice.vector import open_cloud_native_vector_data

    with open_cloud_native_vector_data("fields.parquet") as dataset:
        print(dataset.format_name, dataset.schema_name, dataset.feature_count)
        frame = dataset.to_dataframe(bbox=(-80, 25, -79, 26), columns=["id"])

    with open_cloud_native_vector_data("https://example.com/map.pmtiles") as archive:
        tile = archive.read_tile(z=12, x=1171, y=1566)
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from geolattice.core.bounds import Bounds
from geolattice.crs import CRS
from geolattice.vector.read_vector_data import (
    DriverUnavailableError,
    VectorFeature,
    VectorField,
    VectorOpenError,
)

try:
    from shapely import from_wkb as _from_wkb
except ImportError:  # shapely 1.8
    from shapely.wkb import loads as _from_wkb
from shapely.geometry import Point, Polygon, box, shape
from shapely.wkt import loads as _load_wkt


class CloudNativeOpenError(VectorOpenError):
    """Raised when a cloud-native vector dataset cannot be opened or read."""


class CloudNativeFormat:
    """One cloud-native, lakehouse, or related vector layout."""

    def __init__(
        self,
        format_id: str,
        name: str,
        family: str,
        extensions: Sequence[str],
        description: str,
        spatial_access: str,
        notes: str = "",
    ) -> None:
        self.format_id = format_id
        self.name = name
        self.family = family
        self.extensions = tuple(extensions)
        self.description = description
        self.spatial_access = spatial_access
        self.notes = notes

    def __repr__(self) -> str:
        return f"CloudNativeFormat({self.format_id!r}, family={self.family!r})"


CLOUD_NATIVE_FORMATS: Tuple[CloudNativeFormat, ...] = (
    CloudNativeFormat(
        format_id="geoparquet",
        name="GeoParquet",
        family="cloud-native",
        extensions=(".parquet", ".geoparquet"),
        description="Columnar Parquet for storing and sharing large spatial datasets.",
        spatial_access="columnar",
        notes=(
            "GeoParquet metadata lives in the Parquet footer under the geo key. "
            "Readers fetch selected columns. A directory of Parquet files is read "
            "as one dataset. fiboa, stac-geoparquet, and Overture are GeoParquet "
            "profiles reported as schema_name."
        ),
    ),
    CloudNativeFormat(
        format_id="geoarrow",
        name="GeoArrow",
        family="cloud-native",
        extensions=(".arrow", ".arrows", ".feather", ".ipc"),
        description="Arrow IPC layout for passing geometry between tools in memory and on disk.",
        spatial_access="columnar",
        notes=(
            "Extension types such as geoarrow.wkb, geoarrow.wkt, and geoarrow.point "
            "are decoded to Shapely geometries."
        ),
    ),
    CloudNativeFormat(
        format_id="flatgeobuf",
        name="FlatGeobuf",
        family="cloud-native",
        extensions=(".fgb",),
        description=(
            "Row-based FlatBuffers file with an optional packed Hilbert R-tree. "
            "A client can range-request the features inside a bounding box."
        ),
        spatial_access="http-range",
        notes=(
            "Better than GeoParquet when a spatial lookup should return whole features. "
            "GeoParquet is the better choice for analytics on a few columns. "
            "HTTP and s3 URLs are opened through GDAL /vsicurl/ and /vsis3/."
        ),
    ),
    CloudNativeFormat(
        format_id="pmtiles",
        name="PMTiles",
        family="cloud-native",
        extensions=(".pmtiles",),
        description=(
            "Single-file archive of vector or raster map tiles, readable with HTTP "
            "range requests from S3 or any static host."
        ),
        spatial_access="http-range",
        notes=(
            "Replaces MBTiles and a tile server for web display. Tiles are for "
            "rendering. Vector (MVT) archives can also be decoded to features. "
            "Use read_tile(z, x, y) to fetch one tile's bytes."
        ),
    ),
    CloudNativeFormat(
        format_id="geopackage",
        name="GeoPackage",
        family="cloud-native",
        extensions=(".gpkg",),
        description=(
            "OGC GeoPackage: a SQLite database with vector, raster, and tile support."
        ),
        spatial_access="sqlite",
        notes=(
            "Not cloud-native. GDAL can still read a remote file. It is the practical "
            "single-file replacement for a shapefile on disk or for exchange."
        ),
    ),
    CloudNativeFormat(
        format_id="copc",
        name="COPC",
        family="cloud-native",
        extensions=(".copc.laz", ".copc.las"),
        description=(
            "Cloud Optimized Point Cloud: a LAZ file organized as an octree for "
            "range-request access."
        ),
        spatial_access="http-range",
        notes=(
            "The point-cloud counterpart of FlatGeobuf and PMTiles. The LAS header, "
            "COPC info VLR, and hierarchy pages are read with range requests. "
            "Uncompressed chunks are decoded directly. LAZ chunks need laspy with "
            "a LAZ backend, or the hierarchy is still available on the dataset."
        ),
    ),
    CloudNativeFormat(
        format_id="iceberg",
        name="Apache Iceberg",
        family="table",
        extensions=(),
        description=(
            "Lakehouse table. Iceberg v3 adds native geometry and geography types "
            "on top of Parquet, with versioning, partitioning, and schema evolution."
        ),
        spatial_access="lakehouse",
        notes=(
            "A table directory contains metadata/*.metadata.json and Parquet data "
            "files. geometry and geography fields are recorded from the Iceberg schema. "
            "pyiceberg is used when it is installed; otherwise Parquet files under "
            "data/ are read and WKB columns are decoded."
        ),
    ),
    CloudNativeFormat(
        format_id="delta",
        name="Delta Lake",
        family="table",
        extensions=(),
        description=(
            "Lakehouse table. Delta Lake has no native geometry type; geometry is "
            "stored as WKB and decoded on read."
        ),
        spatial_access="lakehouse",
        notes=(
            "The _delta_log JSON commits are replayed to find live Parquet files. "
            "The deltalake package is used when it is installed."
        ),
    ),
    CloudNativeFormat(
        format_id="parquet-geometry",
        name="Parquet GEOMETRY/GEOGRAPHY",
        family="table",
        extensions=(".parquet",),
        description=(
            "Native Parquet GEOMETRY and GEOGRAPHY logical types added to the "
            "Parquet format in 2025."
        ),
        spatial_access="columnar",
        notes=(
            "These annotations sit on the column and gradually overlap GeoParquet "
            "footer metadata. A file that has both is reported as GeoParquet."
        ),
    ),
    CloudNativeFormat(
        format_id="fiboa",
        name="fiboa",
        family="specification",
        extensions=(".parquet", ".geoparquet"),
        description="GeoParquet schema for agricultural field-boundary data.",
        spatial_access="columnar",
        notes=(
            "Detected when footer metadata mentions fiboa, or the columns include "
            "a geometry and determination_datetime. Reported as schema_name='fiboa' "
            "on a GeoParquet dataset. Several national field-boundary datasets use it."
        ),
    ),
    CloudNativeFormat(
        format_id="stac-geoparquet",
        name="stac-geoparquet",
        family="specification",
        extensions=(".parquet", ".geoparquet"),
        description="STAC items (scene metadata and footprints) stored as GeoParquet.",
        spatial_access="columnar",
        notes=(
            "Detected from a stac_version column or from assets plus datetime plus "
            "a geometry. Useful for bulk queries over satellite scene footprints."
        ),
    ),
    CloudNativeFormat(
        format_id="overture",
        name="Overture Maps",
        family="specification",
        extensions=(".parquet", ".geoparquet"),
        description=(
            "Overture Maps schema for global buildings, places, roads, and boundaries, "
            "published as GeoParquet."
        ),
        spatial_access="columnar",
        notes=(
            "Detected from overture metadata, a theme and type pair, or the "
            "sources plus names columns used on Overture features."
        ),
    ),
    CloudNativeFormat(
        format_id="geojsonseq",
        name="GeoJSON Sequence",
        family="other",
        extensions=(".geojsonl", ".geojsons", ".ndjson", ".jsonl"),
        description="Newline-delimited GeoJSON. Streamable and append-friendly text.",
        spatial_access="stream",
        notes=(
            "Simple and large compared with the binary formats. Record separators "
            "from GeoJSON Text Sequences (RFC 8142) are skipped. Features are read "
            "one line at a time."
        ),
    ),
    CloudNativeFormat(
        format_id="h3-s2",
        name="H3 / S2 cell index",
        family="other",
        extensions=(".parquet", ".arrow", ".feather"),
        description=(
            "A cell id column (H3 or S2) stored instead of, or beside, a geometry "
            "in ordinary Parquet or Arrow."
        ),
        spatial_access="cell-index",
        notes=(
            "Common for aggregating large point tables. When the h3 package is "
            "installed, an H3 id is turned into a polygon. Otherwise the cell id "
            "stays on the feature and geometry is empty."
        ),
    ),
    CloudNativeFormat(
        format_id="tiledb",
        name="TileDB",
        family="other",
        extensions=(),
        description="Array database for point clouds and sparse spatial data, including cloud storage.",
        spatial_access="array",
        notes="Requires the tiledb package. Dimensions named x/y or X/Y become point geometry.",
    ),
    CloudNativeFormat(
        format_id="lance",
        name="Lance",
        family="other",
        extensions=(".lance",),
        description=(
            "Columnar format aimed at machine-learning workloads with fast random access."
        ),
        spatial_access="columnar",
        notes=(
            "Lance has no geo-specific spec. WKB columns are decoded. Requires the lance package."
        ),
    ),
)

_FORMAT_BY_ID: Dict[str, CloudNativeFormat] = {item.format_id: item for item in CLOUD_NATIVE_FORMATS}

_FORMAT_ALIASES: Dict[str, str] = {
    "parquet": "geoparquet",
    "geoparquet": "geoparquet",
    "arrow": "geoarrow",
    "geoarrow": "geoarrow",
    "feather": "geoarrow",
    "ipc": "geoarrow",
    "fgb": "flatgeobuf",
    "flatgeobuf": "flatgeobuf",
    "pmtiles": "pmtiles",
    "gpkg": "geopackage",
    "geopackage": "geopackage",
    "copc": "copc",
    "iceberg": "iceberg",
    "delta": "delta",
    "deltalake": "delta",
    "delta-lake": "delta",
    "parquet-geometry": "parquet-geometry",
    "parquet-geography": "parquet-geometry",
    "geometry": "parquet-geometry",
    "fiboa": "fiboa",
    "stac": "stac-geoparquet",
    "stac-geoparquet": "stac-geoparquet",
    "tac-geoparquet": "stac-geoparquet",
    "overture": "overture",
    "overture-maps": "overture",
    "geojsonseq": "geojsonseq",
    "geojsonl": "geojsonseq",
    "ndjson": "geojsonseq",
    "h3": "h3-s2",
    "s2": "h3-s2",
    "h3-s2": "h3-s2",
    "tiledb": "tiledb",
    "lance": "lance",
}

_PROFILE_FORMATS = {"fiboa", "stac-geoparquet", "overture"}
_H3_COLUMNS = ("h3", "h3_index", "h3_id", "h3index", "h3_cell", "hex_id")
_S2_COLUMNS = ("s2", "s2_cell", "s2_id", "s2_token", "s2cell", "s2_index")
_GEOM_COLUMNS = ("geometry", "geom", "wkb", "wkb_geometry", "the_geom", "wkt")
_PMTILES_MAGIC = b"PMTiles"
_PMTILES_HEADER_SIZE = 127
_COPC_INFO_SIZE = 160
_LAS_VLR_HEADER = 54

_WHERE_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_WHERE_TERM = re.compile(r"^([A-Za-z_][\w:.]*)\s*==?\s*(.+)$")


class CloudNativeLayer:
    """One layer of features from a cloud-native dataset."""

    def __init__(
        self,
        name: str,
        fields: Sequence[VectorField],
        geometry_column: Optional[str],
        geometry_encoding: Optional[str],
        geometry_type: Optional[str],
        crs: Optional[CRS],
        crs_wkt: Optional[str],
        feature_count: Optional[int],
        bounds: Optional[Bounds],
        spatial_index: Optional[str],
        reader: Callable[..., Iterator[VectorFeature]],
        delegate: Any = None,
    ) -> None:
        self.name = name
        self._fields = tuple(fields)
        self.geometry_column = geometry_column
        self.geometry_encoding = geometry_encoding
        self.geometry_type = geometry_type
        self.crs = crs
        self.crs_wkt = crs_wkt
        self._feature_count = feature_count
        self._bounds = bounds
        self.spatial_index = spatial_index
        self._reader = reader
        self._delegate = delegate

    @property
    def fields(self) -> Tuple[VectorField, ...]:
        """Attribute fields in file order."""
        return self._fields

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Attribute names in file order."""
        return tuple(field.name for field in self._fields)

    @property
    def feature_count(self) -> Optional[int]:
        """Number of features, or None when the format cannot report one without a scan."""
        if self._delegate is not None:
            return self._delegate.feature_count
        return self._feature_count

    @property
    def bounds(self) -> Optional[Bounds]:
        """Layer extent from the file footer or index, when the format stores one."""
        if self._bounds is not None:
            return self._bounds
        if self._delegate is not None:
            return self._delegate.bounds
        return None

    def compute_bounds(self) -> Optional[Bounds]:
        """
        Extent of the features, scanning geometries when the footer has no bbox.

        Returns:
            Bounds, or None when the layer has no coordinates
        """
        if self._bounds is not None:
            return self._bounds
        if self._delegate is not None:
            self._bounds = self._delegate.compute_bounds()
            return self._bounds
        min_x = min_y = max_x = max_y = None
        for feature in self.features():
            geom = feature.geometry
            if geom is None or geom.is_empty:
                continue
            gx, gy, gx2, gy2 = geom.bounds
            min_x = gx if min_x is None else min(min_x, gx)
            min_y = gy if min_y is None else min(min_y, gy)
            max_x = gx2 if max_x is None else max(max_x, gx2)
            max_y = gy2 if max_y is None else max(max_y, gy2)
        if min_x is None:
            return None
        self._bounds = Bounds(min_x, min_y, max_x, max_y)
        return self._bounds

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
            where: Attribute filter. GDAL layers take an OGR SQL expression.
                Columnar layers take ``column = value`` terms joined by AND.
            bbox: ``(minx, miny, maxx, maxy)`` or a Bounds
            columns: Attribute names to keep. Geometry is still returned.
            limit: Stop after this many features

        Yields:
            VectorFeature objects
        """
        return self._reader(where=where, bbox=bbox, columns=columns, limit=limit)

    def read(
        self,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[VectorFeature]:
        """Read features into a list."""
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

        When GeoPandas is installed the result is a GeoDataFrame.
        """
        if self._delegate is not None and columns is None and bbox is None and where is None:
            return self._delegate.to_dataframe(limit=limit)
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
            raise CloudNativeOpenError(
                f"Column(s) {missing} are not in layer {self.name!r}. "
                f"Available columns: {list(self.field_names)}"
            )
        return selected

    def __iter__(self) -> Iterator[VectorFeature]:
        return self.features()

    def __repr__(self) -> str:
        return (
            f"CloudNativeLayer({self.name!r}, features={self.feature_count!r}, "
            f"geometry_type={self.geometry_type!r})"
        )


class CloudNativeDataset:
    """
    An open cloud-native vector dataset.

    Use it as a context manager. ``read_tile`` is available for PMTiles.
    ``hierarchy`` and ``copc`` are available for COPC. Schema profiles such as
    fiboa are reported on ``schema_name`` while ``format_name`` stays GeoParquet.
    """

    def __init__(
        self,
        path: str,
        format_id: str,
        reader: str,
        layers: Sequence[CloudNativeLayer],
        active: Union[str, int, None],
        schema_name: Optional[str],
        spatial_access: str,
        warnings: Sequence[str],
        details: Dict[str, Any],
        closer: Optional[Callable[[], None]] = None,
    ) -> None:
        fmt = _FORMAT_BY_ID.get(format_id)
        self.path = path
        self.format_id = format_id
        self.format_name = fmt.name if fmt is not None else format_id
        self.family = fmt.family if fmt is not None else ""
        self.reader = reader
        self.schema_name = schema_name
        self.spatial_access = spatial_access
        self.warnings = tuple(warnings)
        self.details = details
        self._layers = tuple(layers)
        self._closer = closer
        self._closed = False
        if not self._layers:
            self.close()
            raise CloudNativeOpenError(f"Dataset has no vector layers: {path}")
        self._active = _select_layer(self._layers, active, path) if active is not None else self._layers[0]

    @property
    def layers(self) -> Tuple[CloudNativeLayer, ...]:
        """Every layer in file order."""
        return self._layers

    @property
    def layer_names(self) -> List[str]:
        """Layer names in file order."""
        return [layer.name for layer in self._layers]

    @property
    def layer_count(self) -> int:
        """Number of layers."""
        return len(self._layers)

    @property
    def layer(self) -> CloudNativeLayer:
        """Layer selected when the dataset was opened."""
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

    @property
    def geometry_encoding(self) -> Optional[str]:
        """How geometry is stored, such as WKB, MVT, or parquet-geometry."""
        return self.layer.geometry_encoding

    @property
    def spatial_index(self) -> Optional[str]:
        """Index used for spatial lookup, when the format has one."""
        return self.layer.spatial_index

    @property
    def hierarchy(self) -> List[Dict[str, Any]]:
        """COPC octree nodes. Empty for every other format."""
        nodes = self.details.get("hierarchy")
        return list(nodes) if isinstance(nodes, list) else []

    @property
    def copc(self) -> Optional[Dict[str, Any]]:
        """COPC info VLR fields, or None when this is not a COPC dataset."""
        info = self.details.get("copc")
        return dict(info) if isinstance(info, dict) else None

    def get_layer(self, layer: Union[str, int]) -> CloudNativeLayer:
        """Return a layer by name or index."""
        return _select_layer(self._layers, layer, self.path)

    def features(
        self,
        layer: Union[str, int, None] = None,
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        """Iterate features from a layer."""
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
        """Read features from a layer into a list."""
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
        """Read a layer into a DataFrame or a GeoDataFrame."""
        target = self.layer if layer is None else self.get_layer(layer)
        return target.to_dataframe(where=where, bbox=bbox, columns=columns, limit=limit)

    def read_tile(self, z: int, x: int, y: int, *, decode: bool = True) -> Optional[bytes]:
        """
        Fetch one PMTiles tile.

        Args:
            z: Zoom
            x: Column
            y: Row, with 0 at the north edge (XYZ)
            decode: Decompress the tile when the archive uses gzip, brotli, or zstd

        Returns:
            Tile bytes, or None when that tile is not in the archive

        Raises:
            CloudNativeOpenError: If this dataset is not PMTiles
        """
        reader = self.details.get("read_tile")
        if not callable(reader):
            raise CloudNativeOpenError(
                f"{self.path} is {self.format_name}, which has no tile archive. "
                "read_tile is for PMTiles."
            )
        return reader(z, x, y, decode)

    def close(self) -> None:
        """Release file handles and any GDAL dataset."""
        if self._closed:
            return
        self._closed = True
        closer = self._closer
        self._closer = None
        if closer is not None:
            closer()

    def __enter__(self) -> "CloudNativeDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[VectorFeature]:
        return self.features()

    def __getitem__(self, layer: Union[str, int]) -> CloudNativeLayer:
        return self.get_layer(layer)

    def __repr__(self) -> str:
        schema = "" if not self.schema_name else f", schema={self.schema_name!r}"
        return (
            f"CloudNativeDataset({self.path!r}, format={self.format_name!r}{schema}, "
            f"layers={self.layer_names!r})"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return


def open_cloud_native_vector_data(
    path: Union[str, Path],
    layer: Union[str, int, None] = None,
    format: Optional[str] = None,
    crs: Union[str, int, CRS, None] = None,
    open_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
) -> CloudNativeDataset:
    """
    Open cloud-native vector data, a lakehouse table, or a related layout.

    Args:
        path: Local path, ``s3://`` URI, or HTTP(S) URL. PMTiles, FlatGeobuf,
            COPC, and GeoParquet footers are read with range requests when the
            storage supports them.
        layer: Layer name or zero-based index. Defaults to the first layer.
        format: Format id such as ``"geoparquet"``, ``"pmtiles"``, ``"fiboa"``,
            ``"iceberg"``, or ``"h3"``. When omitted, the format is detected
            from the path and the file footer.
        crs: CRS used when the dataset itself has none. An EPSG integer,
            ``"EPSG:4326"``, a WKT string, or a CRS object.
        open_options: Extra GDAL open options for FlatGeobuf, GeoPackage, and
            PMTiles feature decoding, either ``{"ZOOM_LEVEL": "12"}`` or
            ``["ZOOM_LEVEL=12"]``.

    Returns:
        Open dataset. Close it with :meth:`CloudNativeDataset.close` or a
        ``with`` block.

    Raises:
        FileNotFoundError: If a local path does not exist
        CloudNativeOpenError: If the file is not one of these layouts or a
            required library is missing
        DriverUnavailableError: If a GDAL-backed format is missing its driver
    """
    requested = _resolve_format_argument(format)
    location = os.fspath(path)
    fallback = _coerce_crs(crs)
    options = _normalize_open_options(open_options)
    source_format = requested or _detect_format(location)
    if source_format is None:
        raise CloudNativeOpenError(
            f"Could not tell which cloud-native layout {location!r} is. "
            f"Pass format= with one of: {', '.join(sorted(_FORMAT_BY_ID))}."
        )
    physical = _physical_format(source_format)
    opener = _OPENERS.get(physical)
    if opener is None:
        raise CloudNativeOpenError(f"No reader is registered for {physical!r}.")
    dataset = opener(location, layer=layer, open_options=options, requested=source_format)
    _apply_fallback_crs(dataset, fallback)
    if requested in _PROFILE_FORMATS and dataset.schema_name != requested:
        dataset.warnings = dataset.warnings + (
            f"Opened as {dataset.format_name}, but it was not detected as {requested}.",
        )
    return dataset


def open_cloud_native_vector(
    path: Union[str, Path],
    layer: Union[str, int, None] = None,
    **kwargs: Any,
) -> CloudNativeDataset:
    """
    Open a cloud-native vector dataset.

    Alias of :func:`open_cloud_native_vector_data`.
    """
    return open_cloud_native_vector_data(path, layer=layer, **kwargs)


def supported_cloud_native_formats() -> List[Dict[str, Any]]:
    """
    List the layouts this module opens.

    Returns:
        One dictionary per layout, with format id, family, extensions,
        spatial access, whether the reader can run in this environment,
        description, and notes.
    """
    records: List[Dict[str, Any]] = []
    for fmt in CLOUD_NATIVE_FORMATS:
        records.append(
            {
                "format_id": fmt.format_id,
                "name": fmt.name,
                "family": fmt.family,
                "extensions": list(fmt.extensions),
                "spatial_access": fmt.spatial_access,
                "available": _reader_available(fmt.format_id),
                "description": fmt.description,
                "notes": fmt.notes,
            }
        )
    return records


def logical_geometry_kind(logical_type: Any) -> Optional[str]:
    """
    Return ``parquet-geometry`` or ``parquet-geography`` for a Parquet logical type.

    Args:
        logical_type: A PyArrow ``ParquetLogicalType``, or any object with
            ``type`` / ``to_json``. None when the column is not a native
            geometry type.

    Returns:
        Encoding name, or None
    """
    return _geometry_kind_from_logical(logical_type)


def _resolve_format_argument(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace(" ", "-")
    resolved = _FORMAT_ALIASES.get(key)
    if resolved is None:
        known = ", ".join(sorted(_FORMAT_ALIASES))
        raise CloudNativeOpenError(f"Unknown cloud-native format {value!r}. Known names: {known}.")
    return resolved


def _physical_format(format_id: str) -> str:
    if format_id in _PROFILE_FORMATS:
        return "geoparquet"
    if format_id == "h3-s2":
        return "geoparquet"
    if format_id == "parquet-geometry":
        return "geoparquet"
    return format_id


def _detect_format(location: str) -> Optional[str]:
    if _is_remote(location):
        suffix = _remote_suffix(location)
        return _format_from_suffix(suffix, location)
    local = Path(location).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Vector dataset does not exist: {local}")
    name = local.name.lower()
    if name.endswith(".copc.laz") or name.endswith(".copc.las"):
        return "copc"
    if local.is_dir():
        return _detect_directory(local)
    return _format_from_suffix(local.suffix.lower(), name)


def _detect_directory(path: Path) -> Optional[str]:
    if (path / "_delta_log").is_dir():
        return "delta"
    metadata = path / "metadata"
    if metadata.is_dir() and any(metadata.glob("*.metadata.json")):
        return "iceberg"
    if path.name.lower().endswith(".lance") or (path / "_versions").is_dir():
        return "lance"
    if (path / "__tiledb_group.tdb").exists() or (path / "__schema").exists():
        return "tiledb"
    try:
        children = list(path.iterdir())
    except OSError:
        return None
    if any(child.is_file() and child.suffix.lower() == ".parquet" for child in children):
        return "geoparquet"
    return None


def _format_from_suffix(suffix: str, name: str) -> Optional[str]:
    lowered = name.lower()
    if lowered.endswith(".copc.laz") or lowered.endswith(".copc.las"):
        return "copc"
    mapping = {
        ".parquet": "geoparquet",
        ".geoparquet": "geoparquet",
        ".arrow": "geoarrow",
        ".arrows": "geoarrow",
        ".feather": "geoarrow",
        ".ipc": "geoarrow",
        ".fgb": "flatgeobuf",
        ".pmtiles": "pmtiles",
        ".gpkg": "geopackage",
        ".geojsonl": "geojsonseq",
        ".geojsons": "geojsonseq",
        ".ndjson": "geojsonseq",
        ".jsonl": "geojsonseq",
        ".lance": "lance",
    }
    return mapping.get(suffix)


def _remote_suffix(location: str) -> str:
    path = urlparse(location).path
    name = path.lower()
    if name.endswith(".copc.laz"):
        return ".copc.laz"
    if name.endswith(".copc.las"):
        return ".copc.las"
    return Path(path).suffix.lower()


def _is_remote(location: str) -> bool:
    lowered = location.lower()
    return lowered.startswith("/vsi") or "://" in location


def _reader_available(format_id: str) -> bool:
    physical = _physical_format(format_id)
    if physical in {"pmtiles", "geojsonseq", "copc"}:
        return True
    if physical in {"geoparquet", "delta", "iceberg"}:
        return _module_available("pyarrow") or (
            physical == "geoparquet" and _gdal_driver_available("Parquet")
        )
    if physical == "geoarrow":
        return _module_available("pyarrow") or _gdal_driver_available("Arrow")
    if physical == "flatgeobuf":
        return _gdal_driver_available("FlatGeobuf")
    if physical == "geopackage":
        return _gdal_driver_available("GPKG")
    if physical == "lance":
        return _module_available("lance")
    if physical == "tiledb":
        return _module_available("tiledb")
    return False


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _gdal_driver_available(name: str) -> bool:
    try:
        from geolattice.vector.read_vector_data import driver_available
    except ImportError:
        return False
    try:
        return driver_available(name)
    except VectorOpenError:
        return False


def _open_geoparquet(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options
    if not _module_available("pyarrow"):
        if _gdal_driver_available("Parquet"):
            return _open_gdal(
                location,
                layer,
                {},
                requested,
                format_id="geoparquet",
                spatial_index="geoparquet",
                encoding="WKB",
            )
        raise CloudNativeOpenError(
            "Reading GeoParquet requires pyarrow, or a GDAL build with the Parquet driver. "
            "Install pyarrow to read GeoParquet, Iceberg, Delta, fiboa, stac-geoparquet, "
            "and Overture files."
        )
    paths = _parquet_paths(location)
    if not paths and _is_remote(location):
        paths = [location]
    if not paths:
        raise CloudNativeOpenError(f"No Parquet files found at {location}")
    info = _inspect_parquet(paths[0])
    if len(paths) > 1:
        info.feature_count = sum(_parquet_num_rows(item) for item in paths)
    if requested == "h3-s2" and not info.cell_index:
        raise CloudNativeOpenError(
            f"{location} has no H3 or S2 cell-id column. "
            f"Looked for {list(_H3_COLUMNS + _S2_COLUMNS)}."
        )
    if requested == "parquet-geometry" and info.native_geometry is None and info.geo_metadata is None:
        info.warnings.append(
            "No Parquet GEOMETRY/GEOGRAPHY logical type or GeoParquet metadata was found."
        )
    format_id = _parquet_format_id(info, requested)
    schema_name = info.schema_name
    if requested in _PROFILE_FORMATS:
        schema_name = info.schema_name

    def read_rows(
        columns: Optional[Sequence[str]], bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> List[Dict[str, Any]]:
        return _read_parquet_rows(paths, columns, info, bbox=bbox)

    read_rows.accepts_bbox = True  # type: ignore[attr-defined]

    return _dataset_from_rows(
        location=location,
        format_id=format_id,
        reader_name="pyarrow",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=schema_name,
        spatial_access=_FORMAT_BY_ID[format_id].spatial_access,
        read_rows=read_rows,
    )


def _open_geoarrow(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    if not _module_available("pyarrow"):
        if _gdal_driver_available("Arrow"):
            return _open_gdal(
                location,
                layer,
                {},
                "geoarrow",
                format_id="geoarrow",
                spatial_index=None,
                encoding="geoarrow",
            )
        raise CloudNativeOpenError(
            "Reading GeoArrow / Feather requires pyarrow, or a GDAL build with the Arrow driver."
        )
    table = _read_arrow_table(location)
    info = _inspect_arrow_table(table)
    rows = table.to_pylist()

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        del columns
        return rows

    return _dataset_from_rows(
        location=location,
        format_id="geoarrow",
        reader_name="pyarrow",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=info.schema_name,
        spatial_access="columnar",
        read_rows=read_rows,
    )


def _open_flatgeobuf(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del requested
    return _open_gdal(
        location,
        layer,
        open_options,
        "flatgeobuf",
        format_id="flatgeobuf",
        spatial_index="hilbert-rtree",
        encoding="flatgeobuf",
    )


def _open_geopackage(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del requested
    return _open_gdal(
        location,
        layer,
        open_options,
        "geopackage",
        format_id="geopackage",
        spatial_index="rtree",
        encoding="gpkg",
    )


def _open_gdal(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
    format_id: str,
    spatial_index: Optional[str],
    encoding: Optional[str],
) -> CloudNativeDataset:
    del requested
    from geolattice.vector.read_vector_data import open_vectors_data

    gdal_path = _gdal_cloud_path(location)
    option_list = [f"{key}={value}" for key, value in open_options.items()] or None
    inner = open_vectors_data(gdal_path, layer=None, open_options=option_list)
    layers: List[CloudNativeLayer] = []
    for inner_layer in inner.layers:
        layers.append(_layer_from_gdal(inner_layer, spatial_index, encoding))
    return CloudNativeDataset(
        path=location,
        format_id=format_id,
        reader="gdal",
        layers=layers,
        active=layer,
        schema_name=None,
        spatial_access=_FORMAT_BY_ID[format_id].spatial_access,
        warnings=list(inner.warnings),
        details={"gdal_driver": inner.driver, "gdal_path": gdal_path},
        closer=inner.close,
    )


def _layer_from_gdal(
    inner: Any, spatial_index: Optional[str], encoding: Optional[str]
) -> CloudNativeLayer:
    def reader(
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        return inner.features(where=where, bbox=bbox, columns=columns, limit=limit)

    return CloudNativeLayer(
        name=inner.name,
        fields=inner.fields,
        geometry_column=inner.geometry_column,
        geometry_encoding=encoding,
        geometry_type=inner.geometry_type,
        crs=inner.crs,
        crs_wkt=inner.crs_wkt,
        feature_count=inner.feature_count,
        bounds=inner.bounds,
        spatial_index=spatial_index,
        reader=reader,
        delegate=inner,
    )


def _open_pmtiles(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del requested
    source = _byte_source(location)
    try:
        archive = _PMTilesArchive(source)
    except Exception:
        source.close()
        raise
    details: Dict[str, Any] = {
        "header": archive.header,
        "metadata": archive.metadata,
        "entries": [entry.as_dict() for entry in archive.root_entries],
        "read_tile": archive.read_tile,
        "tile_type": archive.header["tile_type_name"],
        "compression": archive.header["tile_compression_name"],
    }
    closer = source.close
    vector_tiles = archive.header["tile_type"] in {1, 6}
    if vector_tiles and _gdal_driver_available("PMTiles"):
        options = dict(open_options)
        options.setdefault("ZOOM_LEVEL", _pmtiles_feature_zoom(archive))
        try:
            dataset = _open_gdal(
                location,
                layer,
                options,
                "pmtiles",
                format_id="pmtiles",
                spatial_index="hilbert",
                encoding="mvt",
            )
        except (VectorOpenError, DriverUnavailableError) as exc:
            dataset = None
            details["feature_decode_error"] = str(exc)
        else:
            dataset.details.update(details)
            dataset.reader = "pmtiles+gdal"
            previous = dataset._closer

            def _close_both() -> None:
                if previous is not None:
                    previous()
                closer()

            dataset._closer = _close_both
            return dataset
    layer_obj = _pmtiles_tile_layer(archive)
    return CloudNativeDataset(
        path=location,
        format_id="pmtiles",
        reader="pmtiles",
        layers=[layer_obj],
        active=layer,
        schema_name=None,
        spatial_access="http-range",
        warnings=[
            "This PMTiles archive is being read as tiles. "
            "Features are tile footprints, not the geometries drawn inside each tile."
        ]
        if not vector_tiles
        else [],
        details=details,
        closer=closer,
    )


def _pmtiles_tile_layer(archive: "_PMTilesArchive") -> CloudNativeLayer:
    fields = [
        VectorField("z", "Integer"),
        VectorField("x", "Integer"),
        VectorField("y", "Integer"),
        VectorField("tile_id", "Integer64"),
        VectorField("run_length", "Integer"),
        VectorField("length", "Integer"),
    ]
    header = archive.header
    bounds = _pmtiles_bounds(header, archive.metadata)

    def reader(
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        selected = _checked_columns(columns, [field.name for field in fields], "tiles")
        query = _validate_bbox(bbox)
        predicates = _parse_where(where)
        produced = 0
        limit_value = _validate_limit(limit)
        for entry in archive.entries():
            z, x, y = tileid_to_zxy(entry.tile_id)
            polygon = _tile_polygon(z, x, y)
            if query is not None and not _geom_hits(polygon, query):
                continue
            properties: Dict[str, Any] = {
                "z": z,
                "x": x,
                "y": y,
                "tile_id": entry.tile_id,
                "run_length": entry.run_length,
                "length": entry.length,
            }
            if not _matches(properties, predicates):
                continue
            if selected is not None:
                properties = {name: properties[name] for name in selected}
            yield VectorFeature(fid=entry.tile_id, geometry=polygon, properties=properties)
            produced += 1
            if limit_value is not None and produced >= limit_value:
                break

    return CloudNativeLayer(
        name="tiles",
        fields=fields,
        geometry_column=None,
        geometry_encoding="tile",
        geometry_type="Polygon",
        crs=CRS(4326),
        crs_wkt=None,
        feature_count=header.get("tile_entries") or None,
        bounds=bounds,
        spatial_index="hilbert",
        reader=reader,
    )


def _open_copc(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    source = _byte_source(location)
    try:
        copc = _read_copc(source)
    except Exception:
        source.close()
        raise
    info = copc["info"]
    fields = [
        VectorField("x", "Real"),
        VectorField("y", "Real"),
        VectorField("z", "Real"),
    ]
    warnings: List[str] = []
    if copc["compressed_voxels"] and not _module_available("laspy"):
        warnings.append(
            "Some COPC voxels are LAZ-compressed. Header and hierarchy were read. "
            "Install laspy and a LAZ backend (lazrs) to decode those points."
        )

    def reader(
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        selected = _checked_columns(columns, ["x", "y", "z"], "points")
        query = _validate_bbox(bbox)
        predicates = _parse_where(where)
        limit_value = _validate_limit(limit)
        produced = 0
        fid = 0
        if copc["compressed_voxels"]:
            point_iter: List[Tuple[float, float, float]] = _laspy_points(location)
        else:
            point_iter = []
            for node in copc["hierarchy"]:
                if node["point_count"] < 0:
                    continue
                if query is not None and not _bounds_hit(_voxel_bounds(info, node), query):
                    continue
                point_iter.extend(_decode_raw_voxel(source, copc, node))
        for x_coord, y_coord, z_coord in point_iter:
            properties: Dict[str, Any] = {"x": x_coord, "y": y_coord, "z": z_coord}
            if not _matches(properties, predicates):
                fid += 1
                continue
            geom = Point(x_coord, y_coord, z_coord)
            if query is not None and not _geom_hits(geom, query):
                fid += 1
                continue
            if selected is not None:
                properties = {name: properties[name] for name in selected}
            yield VectorFeature(fid=fid, geometry=geom, properties=properties)
            fid += 1
            produced += 1
            if limit_value is not None and produced >= limit_value:
                return

    count = copc["point_count"]
    layer_obj = CloudNativeLayer(
        name="points",
        fields=fields,
        geometry_column=None,
        geometry_encoding="copc",
        geometry_type="Point",
        crs=copc["crs"],
        crs_wkt=copc["crs_wkt"],
        feature_count=count,
        bounds=copc["bounds"],
        spatial_index="octree",
        reader=reader,
    )
    return CloudNativeDataset(
        path=location,
        format_id="copc",
        reader="copc",
        layers=[layer_obj],
        active=layer,
        schema_name=None,
        spatial_access="http-range",
        warnings=warnings,
        details={"copc": info, "hierarchy": copc["hierarchy"], "point_format": copc["point_format"]},
        closer=source.close,
    )


def _open_geojsonseq(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    fields = [VectorField("properties", "JSON")]
    # Field names are discovered from the first feature when possible, but the
    # layer stays schemaless: each feature keeps the properties on that line.
    sample_fields, count = _geojsonseq_header(location)

    def reader(
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        known = list(sample_fields)
        if columns is not None and known:
            _checked_columns(columns, known, _layer_name(location))
        query = _validate_bbox(bbox)
        predicates = _parse_where(where)
        limit_value = _validate_limit(limit)
        produced = 0
        for fid, properties, geom in _iter_geojsonseq(location):
            if not _matches(properties, predicates):
                continue
            if query is not None and not _geom_hits(geom, query):
                continue
            if columns is not None:
                missing = [name for name in columns if name not in properties and name not in known]
                if missing and not known:
                    properties = {name: properties.get(name) for name in columns}
                else:
                    properties = {name: properties.get(name) for name in columns}
            yield VectorFeature(fid=fid, geometry=geom, properties=properties)
            produced += 1
            if limit_value is not None and produced >= limit_value:
                break

    layer_obj = CloudNativeLayer(
        name=_layer_name(location),
        fields=tuple(VectorField(name, "JSON") for name in sample_fields) or tuple(fields[:0]),
        geometry_column="geometry",
        geometry_encoding="geojson",
        geometry_type=None,
        crs=CRS(4326),
        crs_wkt=None,
        feature_count=count,
        bounds=None,
        spatial_index=None,
        reader=reader,
    )
    return CloudNativeDataset(
        path=location,
        format_id="geojsonseq",
        reader="geojsonseq",
        layers=[layer_obj],
        active=layer,
        schema_name=None,
        spatial_access="stream",
        warnings=[],
        details={},
        closer=None,
    )


def _open_delta(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    if _module_available("deltalake"):
        return _open_delta_library(location, layer)
    if not _module_available("pyarrow"):
        raise CloudNativeOpenError(
            "Reading Delta Lake requires pyarrow to decode the Parquet data files, "
            "or the deltalake package."
        )
    root = _local_dir(location)
    log = root / "_delta_log"
    if not log.is_dir():
        raise CloudNativeOpenError(f"No _delta_log directory in Delta table {root}")
    files, schema_fields, version = _replay_delta_log(log)
    if not files:
        raise CloudNativeOpenError(f"Delta table {root} has no live Parquet files.")
    paths = [str(root / name) for name in files]
    missing = [item for item in paths if not Path(item).is_file()]
    if missing:
        raise CloudNativeOpenError(
            f"Delta log references Parquet files that are not in {root}: {missing}"
        )
    info = _inspect_parquet(paths[0])
    if info.geometry_column is None:
        binary = _binary_field(schema_fields)
        if binary is not None:
            info.geometry_column = binary
            info.geometry_encoding = "WKB"
            info.warnings.append(
                "Delta Lake has no native geometry type. "
                f"Column {binary!r} is being read as WKB."
            )
    info.details["delta_version"] = version
    info.details["data_files"] = files
    info.warnings.append(
        "Delta Lake has no native geometry type. Geometry is decoded from WKB."
    )

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        return _read_parquet_rows(paths, columns, info)

    return _dataset_from_rows(
        location=location,
        format_id="delta",
        reader_name="delta-log",
        layer_name=root.name,
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="lakehouse",
        read_rows=read_rows,
    )


def _open_delta_library(location: str, layer: Union[str, int, None]) -> CloudNativeDataset:
    from deltalake import DeltaTable

    table = DeltaTable(location)
    arrow = table.to_pyarrow_table()
    info = _inspect_arrow_table(arrow)
    info.details["delta_version"] = table.version()
    info.warnings.append(
        "Delta Lake has no native geometry type. Geometry is decoded from WKB."
    )
    rows = arrow.to_pylist()

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        del columns
        return rows

    return _dataset_from_rows(
        location=location,
        format_id="delta",
        reader_name="deltalake",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="lakehouse",
        read_rows=read_rows,
    )


def _open_iceberg(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    root = _local_dir(location)
    metadata_path = _latest_iceberg_metadata(root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = _iceberg_current_schema(metadata)
    geometry_fields = _iceberg_geometry_fields(schema)
    if _module_available("pyiceberg"):
        try:
            return _open_iceberg_library(location, layer, metadata_path, metadata, geometry_fields)
        except Exception as exc:
            # A partial pyiceberg install should not hide the Parquet fallback.
            if not _module_available("pyarrow"):
                raise CloudNativeOpenError(f"Could not read Iceberg table {root}: {exc}") from exc
    if not _module_available("pyarrow"):
        raise CloudNativeOpenError(
            "Reading Iceberg data files requires pyarrow, or the pyiceberg package."
        )
    paths = _iceberg_data_files(root)
    if not paths:
        raise CloudNativeOpenError(
            f"Iceberg table {root} has metadata but no Parquet files under data/. "
            "Install pyiceberg to follow manifest lists."
        )
    info = _inspect_parquet(paths[0])
    info.details["iceberg_format_version"] = metadata.get("format-version")
    info.details["iceberg_schema"] = schema
    info.details["geometry_fields"] = geometry_fields
    for name, type_name in geometry_fields:
        info.geometry_column = name
        info.geometry_encoding = f"iceberg-{type_name}"
        info.warnings.append(
            f"Iceberg field {name!r} is native type {type_name}. "
            "Binary values are decoded as WKB."
        )
    if not geometry_fields and info.geometry_encoding == "WKB":
        info.warnings.append(
            "This Iceberg table stores geometry as WKB. "
            "A native geometry type is an Iceberg v3 field."
        )

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        return _read_parquet_rows(paths, columns, info)

    return _dataset_from_rows(
        location=location,
        format_id="iceberg",
        reader_name="iceberg-metadata",
        layer_name=root.name,
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="lakehouse",
        read_rows=read_rows,
    )


def _open_iceberg_library(
    location: str,
    layer: Union[str, int, None],
    metadata_path: Path,
    metadata: Dict[str, Any],
    geometry_fields: Sequence[Tuple[str, str]],
) -> CloudNativeDataset:
    from pyiceberg.table import StaticTable

    table = StaticTable.from_metadata(str(metadata_path))
    arrow = table.scan().to_arrow()
    info = _inspect_arrow_table(arrow)
    info.details["iceberg_format_version"] = metadata.get("format-version")
    info.details["geometry_fields"] = list(geometry_fields)
    rows = arrow.to_pylist()

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        del columns
        return rows

    return _dataset_from_rows(
        location=location,
        format_id="iceberg",
        reader_name="pyiceberg",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="lakehouse",
        read_rows=read_rows,
    )


def _open_lance(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    if not _module_available("lance"):
        raise CloudNativeOpenError(
            "Reading Lance requires the lance package. Lance has no geo spec; "
            "once installed, WKB columns are decoded."
        )
    import lance

    dataset = lance.dataset(location)
    arrow = dataset.to_table()
    info = _inspect_arrow_table(arrow)
    if info.geometry_column is not None and info.geo_metadata is None:
        info.warnings.append("Lance has no geo-specific spec. WKB columns are decoded.")
    rows = arrow.to_pylist()

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        del columns
        return rows

    return _dataset_from_rows(
        location=location,
        format_id="lance",
        reader_name="lance",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="columnar",
        read_rows=read_rows,
    )


def _open_tiledb(
    location: str,
    layer: Union[str, int, None],
    open_options: Dict[str, str],
    requested: str,
) -> CloudNativeDataset:
    del open_options, requested
    if not _module_available("tiledb"):
        raise CloudNativeOpenError(
            "Reading TileDB requires the tiledb package. "
            "TileDB stores point clouds and sparse spatial arrays, including on cloud storage."
        )
    import tiledb

    with tiledb.open(location) as array:
        schema = array.schema
        dim_names = [schema.domain.dim(i).name for i in range(schema.domain.ndim)]
        attr_names = [schema.attr(i).name for i in range(schema.nattr)]
        data = array.df[:]
    frame_columns = list(getattr(data, "columns", []))
    x_name = _first_present(("x", "X", "lon", "longitude"), dim_names + frame_columns)
    y_name = _first_present(("y", "Y", "lat", "latitude"), dim_names + frame_columns)
    info = _TableInfo()
    info.fields = [VectorField(name, "TileDB") for name in list(dict.fromkeys(dim_names + attr_names))]
    info.geometry_column = None
    info.geometry_encoding = "xy" if x_name and y_name else None
    info.geometry_type = "Point" if x_name and y_name else None
    info.spatial_index = "tiledb"
    info.details = {"dimensions": dim_names, "attributes": attr_names}
    info.x_column = x_name
    info.y_column = y_name
    records = data.to_dict(orient="records")
    info.feature_count = len(records)

    def read_rows(columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        del columns
        return records

    return _dataset_from_rows(
        location=location,
        format_id="tiledb",
        reader_name="tiledb",
        layer_name=_layer_name(location),
        layer=layer,
        info=info,
        schema_name=None,
        spatial_access="array",
        read_rows=read_rows,
    )


_OPENERS = {
    "geoparquet": _open_geoparquet,
    "geoarrow": _open_geoarrow,
    "flatgeobuf": _open_flatgeobuf,
    "geopackage": _open_geopackage,
    "pmtiles": _open_pmtiles,
    "copc": _open_copc,
    "geojsonseq": _open_geojsonseq,
    "delta": _open_delta,
    "iceberg": _open_iceberg,
    "lance": _open_lance,
    "tiledb": _open_tiledb,
}


class _TableInfo:
    """Schema and geometry description shared by columnar readers."""

    def __init__(self) -> None:
        self.fields: List[VectorField] = []
        self.feature_count: Optional[int] = None
        self.bounds: Optional[Bounds] = None
        self.crs: Optional[CRS] = None
        self.crs_wkt: Optional[str] = None
        self.geometry_column: Optional[str] = None
        self.geometry_encoding: Optional[str] = None
        self.geometry_type: Optional[str] = None
        self.spatial_index: Optional[str] = None
        self.schema_name: Optional[str] = None
        self.cell_index: Optional[str] = None
        self.cell_column: Optional[str] = None
        self.native_geometry: Optional[str] = None
        self.geo_metadata: Optional[Dict[str, Any]] = None
        self.x_column: Optional[str] = None
        self.y_column: Optional[str] = None
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}


def _dataset_from_rows(
    location: str,
    format_id: str,
    reader_name: str,
    layer_name: str,
    layer: Union[str, int, None],
    info: _TableInfo,
    schema_name: Optional[str],
    spatial_access: str,
    read_rows: Callable[[Optional[Sequence[str]]], List[Dict[str, Any]]],
) -> CloudNativeDataset:
    fields = [field for field in info.fields if field.name != info.geometry_column]
    if info.cell_index and info.spatial_index is None:
        info.spatial_index = info.cell_index
    if info.geometry_column is None and info.cell_index:
        info.geometry_encoding = info.cell_index

    def reader(
        where: Optional[str] = None,
        bbox: Union[Sequence[float], Bounds, None] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[VectorFeature]:
        names = [field.name for field in fields]
        selected = _checked_columns(columns, names, layer_name)
        query = _validate_bbox(bbox)
        predicates = _parse_where(where)
        limit_value = _validate_limit(limit)
        wanted = _columns_to_load(selected, info)
        produced = 0
        if query is not None and getattr(read_rows, "accepts_bbox", False):
            rows = read_rows(wanted, bbox=query)          # prunes row groups by bounding box
        else:
            rows = read_rows(wanted)
        for fid, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            fid = row.pop(_FID_KEY, fid)
            if not _matches(row, predicates):
                continue
            geom = _geometry_from_row(row, info)
            if query is not None and not _geom_hits(geom, query):
                continue
            properties = dict(row)
            if info.geometry_column and info.geometry_column in properties:
                properties.pop(info.geometry_column, None)
            if selected is not None:
                properties = {name: properties.get(name) for name in selected}
            yield VectorFeature(fid=fid, geometry=geom, properties=properties)
            produced += 1
            if limit_value is not None and produced >= limit_value:
                break

    layer_obj = CloudNativeLayer(
        name=layer_name,
        fields=fields,
        geometry_column=info.geometry_column,
        geometry_encoding=info.geometry_encoding,
        geometry_type=info.geometry_type,
        crs=info.crs,
        crs_wkt=info.crs_wkt,
        feature_count=info.feature_count,
        bounds=info.bounds,
        spatial_index=info.spatial_index,
        reader=reader,
    )
    return CloudNativeDataset(
        path=location,
        format_id=format_id,
        reader=reader_name,
        layers=[layer_obj],
        active=layer,
        schema_name=schema_name,
        spatial_access=spatial_access,
        warnings=list(info.warnings),
        details=dict(info.details),
        closer=None,
    )


def _inspect_parquet(path: str) -> _TableInfo:
    import pyarrow.parquet as pq

    parquet = _parquet_file(path)
    info = _inspect_arrow_schema(parquet.schema_arrow, getattr(parquet.metadata, "metadata", None))
    info.feature_count = int(parquet.metadata.num_rows)
    native = _native_geometry_columns(parquet.schema)
    if native:
        column, kind = native[0]
        info.native_geometry = kind
        info.details["parquet_logical_types"] = {name: encoding for name, encoding in native}
        if info.geometry_column is None:
            info.geometry_column = column
            info.geometry_encoding = kind
            info.spatial_index = "parquet-geometry"
        else:
            info.details["native_type"] = kind
    if info.geo_metadata is None and info.native_geometry:
        info.warnings.append(
            "Parquet GEOMETRY/GEOGRAPHY logical type is present without GeoParquet metadata."
        )
    return info


def _inspect_arrow_table(table: Any) -> _TableInfo:
    info = _inspect_arrow_schema(table.schema, table.schema.metadata)
    info.feature_count = int(table.num_rows)
    return info


def _inspect_arrow_schema(schema: Any, metadata: Optional[Dict[Any, Any]]) -> _TableInfo:
    info = _TableInfo()
    decoded = _decode_metadata(metadata)
    info.details["metadata_keys"] = sorted(decoded)
    geo = _parse_geo_metadata(decoded.get("geo"))
    info.geo_metadata = geo
    names = [schema.field(i).name for i in range(len(schema))]
    for index, name in enumerate(names):
        field = schema.field(index)
        info.fields.append(VectorField(name, str(field.type)))
        extension = _extension_name(field)
        if extension and extension.startswith("geoarrow.") and info.geometry_column is None:
            info.geometry_column = name
            info.geometry_encoding = extension
            info.geometry_type = _geometry_type_from_extension(extension)
    if geo is not None:
        primary = geo.get("primary_column") or "geometry"
        columns = geo.get("columns") if isinstance(geo.get("columns"), dict) else {}
        column_meta = columns.get(primary) if isinstance(columns.get(primary), dict) else {}
        if primary in names:
            info.geometry_column = primary
        encoding = column_meta.get("encoding")
        if encoding and info.geometry_encoding is None:
            info.geometry_encoding = str(encoding)
        types = column_meta.get("geometry_types") or column_meta.get("geometry_type")
        info.geometry_type = _geometry_type_label(types)
        info.bounds = _bounds_from_bbox(column_meta.get("bbox"))
        info.crs, info.crs_wkt = _crs_from_geo(column_meta.get("crs"), geo_present=True)
        info.spatial_index = "geoparquet"
        info.details["geo"] = geo
    elif info.geometry_column is None:
        for candidate in _GEOM_COLUMNS:
            if candidate in names:
                info.geometry_column = candidate
                field_type = str(schema.field(candidate).type).lower()
                if "string" in field_type or "utf8" in field_type:
                    info.geometry_encoding = "WKT"
                else:
                    info.geometry_encoding = "WKB"
                info.warnings.append(
                    f"No GeoParquet metadata. Column {candidate!r} is being decoded "
                    f"as {info.geometry_encoding}."
                )
                break
    info.cell_column, info.cell_index = _cell_column(names)
    if info.cell_index:
        info.details["cell_column"] = info.cell_column
        if info.geometry_column is None:
            info.spatial_index = info.cell_index
        if info.cell_index == "h3" and not _module_available("h3"):
            info.warnings.append(
                "An H3 cell-id column is present. Install the h3 package to build cell polygons."
            )
        if info.cell_index == "s2" and not _module_available("s2sphere"):
            info.warnings.append(
                "An S2 cell-id column is present. Install s2sphere to build cell polygons."
            )
    info.schema_name = _detect_profile(decoded, names, geo)
    if info.schema_name:
        info.details["schema_name"] = info.schema_name
    if info.crs is None and info.geometry_encoding and str(info.geometry_encoding).startswith("geoarrow"):
        info.crs = CRS(4326)
    return info


def _parquet_format_id(info: _TableInfo, requested: str) -> str:
    cell_only = (
        info.cell_index is not None
        and info.geometry_column is None
        and info.geo_metadata is None
        and info.native_geometry is None
    )
    if cell_only and requested in {None, "geoparquet", "h3-s2"}:
        return "h3-s2"
    if info.geo_metadata is not None or info.schema_name is not None:
        return "geoparquet"
    if info.native_geometry and info.geo_metadata is None:
        return "parquet-geometry"
    return "geoparquet"


def _detect_profile(
    metadata: Dict[str, str], names: Sequence[str], geo: Optional[Dict[str, Any]]
) -> Optional[str]:
    blob = "\n".join(f"{key}={value}" for key, value in metadata.items()).lower()
    if geo is not None:
        blob += "\n" + json.dumps(geo).lower()
    folded = {name.lower() for name in names}
    if "fiboa" in blob or "determination_datetime" in folded or "determination:datetime" in folded:
        return "fiboa"
    if "stac_version" in folded or "stac-geoparquet" in blob or (
        "assets" in folded and "datetime" in folded and (_geom_name(names) is not None)
    ):
        return "stac-geoparquet"
    if "overture" in blob or ({"theme", "type"} <= folded) or (
        "sources" in folded and "names" in folded and _geom_name(names) is not None
    ):
        return "overture"
    return None


def _geom_name(names: Sequence[str]) -> Optional[str]:
    folded = {name.lower(): name for name in names}
    for candidate in _GEOM_COLUMNS:
        if candidate in folded:
            return folded[candidate]
    return None


def _cell_column(names: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    folded = {name.lower(): name for name in names}
    for candidate in _H3_COLUMNS:
        if candidate in folded:
            return folded[candidate], "h3"
    for candidate in _S2_COLUMNS:
        if candidate in folded:
            return folded[candidate], "s2"
    return None, None


def _native_geometry_columns(parquet_schema: Any) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    try:
        count = len(parquet_schema)
    except TypeError:
        return found
    for index in range(count):
        try:
            column = parquet_schema.column(index)
        except Exception:
            continue
        kind = _geometry_kind_from_logical(getattr(column, "logical_type", None))
        if kind:
            found.append((column.name, kind))
    return found


def _geometry_kind_from_logical(logical_type: Any) -> Optional[str]:
    if logical_type is None:
        return None
    parts = [str(getattr(logical_type, "type", "") or "")]
    to_json = getattr(logical_type, "to_json", None)
    if callable(to_json):
        try:
            parts.append(str(to_json()))
        except Exception:
            pass
    text = " ".join(parts).lower()
    if "geography" in text:
        return "parquet-geography"
    if "geometry" in text:
        return "parquet-geometry"
    return None


def _decode_metadata(metadata: Optional[Dict[Any, Any]]) -> Dict[str, str]:
    if not metadata:
        return {}
    decoded: Dict[str, str] = {}
    for key, value in metadata.items():
        name = key.decode("utf-8", "replace") if isinstance(key, (bytes, bytearray)) else str(key)
        if name.startswith("ARROW:"):
            continue
        if isinstance(value, (bytes, bytearray)):
            text = value.decode("utf-8", "replace")
        else:
            text = str(value)
        decoded[name] = text
    return decoded


def _parse_geo_metadata(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extension_name(field: Any) -> Optional[str]:
    metadata = getattr(field, "metadata", None) or {}
    raw = metadata.get(b"ARROW:extension:name")
    if raw is None:
        raw = metadata.get("ARROW:extension:name")
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _geometry_type_from_extension(extension: str) -> Optional[str]:
    tail = extension.split(".")[-1].lower()
    mapping = {
        "point": "Point",
        "linestring": "LineString",
        "polygon": "Polygon",
        "multipoint": "MultiPoint",
        "multilinestring": "MultiLineString",
        "multipolygon": "MultiPolygon",
    }
    return mapping.get(tail)


def _geometry_type_label(types: Any) -> Optional[str]:
    if isinstance(types, str):
        return types
    if isinstance(types, list) and len(types) == 1:
        return str(types[0])
    if isinstance(types, list) and types:
        return "Mixed"
    return None


def _bounds_from_bbox(value: Any) -> Optional[Bounds]:
    coords = _bbox_tuple(value)
    if coords is None:
        return None
    min_x, min_y, max_x, max_y = coords
    if min_x > max_x or min_y > max_y:
        return None
    return Bounds(min_x, min_y, max_x, max_y)


def _bbox_tuple(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(value, dict):
        keys = {key.lower(): key for key in value}
        if {"xmin", "ymin", "xmax", "ymax"} <= set(keys):
            return (
                float(value[keys["xmin"]]),
                float(value[keys["ymin"]]),
                float(value[keys["xmax"]]),
                float(value[keys["ymax"]]),
            )
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            min_x, min_y, max_x, max_y = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        return min_x, min_y, max_x, max_y
    return None


def _crs_from_geo(crs_value: Any, geo_present: bool) -> Tuple[Optional[CRS], Optional[str]]:
    if crs_value is None:
        if geo_present:
            # GeoParquet 1.1: a missing CRS means OGC:CRS84 (longitude, latitude).
            return CRS(4326), None
        return None, None
    if isinstance(crs_value, str):
        text = crs_value.strip()
        if text.upper() in {"OGC:CRS84", "CRS84", "EPSG:4326"}:
            return CRS(4326), None
        try:
            crs = CRS(text)
        except (TypeError, ValueError):
            return None, text
        return crs, crs.wkt
    if isinstance(crs_value, dict):
        ident = crs_value.get("id") if isinstance(crs_value.get("id"), dict) else {}
        authority = str(ident.get("authority", "")).upper()
        code = ident.get("code")
        if authority in {"OGC", "EPSG"} and str(code).upper() in {"4326", "CRS84"}:
            return CRS(4326), None
        if authority == "EPSG" and code is not None:
            try:
                return CRS(int(code)), None
            except (TypeError, ValueError):
                pass
        try:
            import pyproj

            projected = pyproj.CRS.from_json_dict(crs_value)
            wkt = projected.to_wkt()
            return CRS.from_wkt(wkt), wkt
        except Exception:
            return None, None
    return None, None


def _geometry_from_row(row: Dict[str, Any], info: _TableInfo) -> Any:
    if info.x_column and info.y_column:
        x_value = _as_float(row.get(info.x_column))
        y_value = _as_float(row.get(info.y_column))
        if x_value is not None and y_value is not None:
            return Point(x_value, y_value)
    column = info.geometry_column
    if column is not None and column in row:
        geom = _decode_geometry_value(row.get(column), info.geometry_encoding)
        if geom is not None:
            return geom
    if info.cell_index == "h3" and info.cell_column:
        return _h3_polygon(row.get(info.cell_column))
    if info.cell_index == "s2" and info.cell_column:
        return _s2_polygon(row.get(info.cell_column))
    return None


def _decode_geometry_value(value: Any, encoding: Optional[str]) -> Any:
    if value is None:
        return None
    label = (encoding or "").lower()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            return None
        try:
            return _from_wkb(raw)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if label == "wkb" or _looks_like_hex_wkb(text):
            try:
                return _from_wkb(bytes.fromhex(text))
            except (ValueError, Exception):
                pass
        try:
            return _load_wkt(text)
        except Exception:
            return None
    if isinstance(value, dict):
        if "type" in value and "coordinates" in value:
            try:
                return shape(value)
            except Exception:
                return None
        if "x" in value and "y" in value:
            x_value = _as_float(value.get("x"))
            y_value = _as_float(value.get("y"))
            if x_value is not None and y_value is not None:
                z_value = _as_float(value.get("z"))
                if z_value is None:
                    return Point(x_value, y_value)
                return Point(x_value, y_value, z_value)
    if isinstance(value, (list, tuple)) and len(value) >= 2 and label.startswith("geoarrow.point"):
        x_value = _as_float(value[0])
        y_value = _as_float(value[1])
        if x_value is None or y_value is None:
            return None
        if len(value) >= 3:
            z_value = _as_float(value[2])
            if z_value is not None:
                return Point(x_value, y_value, z_value)
        return Point(x_value, y_value)
    return None


def _looks_like_hex_wkb(value: str) -> bool:
    if len(value) < 10 or len(value) % 2:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _h3_polygon(cell: Any) -> Any:
    if cell is None:
        return None
    try:
        import h3
    except ImportError:
        return None
    token = str(cell)
    boundary = None
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(token)
    elif hasattr(h3, "h3_to_geo_boundary"):
        boundary = h3.h3_to_geo_boundary(token)
    if not boundary:
        return None
    ring = [(lon, lat) for lat, lon in boundary]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return Polygon(ring)


def _s2_polygon(cell: Any) -> Any:
    if cell is None:
        return None
    try:
        import s2sphere
    except ImportError:
        return None
    token = str(cell)
    cell_id = s2sphere.CellId.from_token(token)
    cell_obj = s2sphere.Cell(cell_id)
    ring = []
    for index in range(4):
        point = cell_obj.get_vertex(index)
        lat_lng = s2sphere.LatLng.from_point(point)
        ring.append((lat_lng.lng().degrees, lat_lng.lat().degrees))
    ring.append(ring[0])
    return Polygon(ring)


def _parquet_paths(location: str) -> List[str]:
    if _is_remote(location):
        return [location]
    path = Path(location).expanduser()
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return sorted(str(child) for child in path.iterdir() if child.suffix.lower() == ".parquet")
    return []


def _remote_arrow_filesystem(location: str) -> Tuple[Any, str]:
    """PyArrow filesystem and path for a remote location.

    PyArrow resolves ``s3://``, ``gs://`` and ``hdfs://`` itself but has no HTTP
    filesystem, so ``http(s)://`` URLs go through fsspec (which reads with range
    requests and needs ``aiohttp``).
    """
    import pyarrow.fs as pafs

    if location.lower().startswith(("http://", "https://")):
        try:
            import fsspec
        except ImportError as exc:
            raise CloudNativeOpenError(
                "Reading Parquet or Arrow over HTTP needs fsspec and aiohttp: pip install fsspec aiohttp"
            ) from exc
        # No read-ahead cache: PyArrow asks for exact column-chunk ranges, and fsspec's default
        # 5 MB read-ahead would multiply the bytes transferred many times over.
        http = fsspec.filesystem("http", cache_type="none", skip_instance_cache=True)
        return pafs.PyFileSystem(pafs.FSSpecHandler(http)), location
    return pafs.FileSystem.from_uri(location)


def _parquet_file(path: str) -> Any:
    import pyarrow.parquet as pq

    if not _is_remote(path):
        return pq.ParquetFile(path)
    filesystem, inner = _remote_arrow_filesystem(path)
    return pq.ParquetFile(inner, filesystem=filesystem, pre_buffer=True)


def _arrow_input(location: str) -> Any:
    if not _is_remote(location):
        return location
    filesystem, inner = _remote_arrow_filesystem(location)
    return filesystem.open_input_file(inner)


def _parquet_num_rows(path: str) -> int:
    return int(_parquet_file(path).metadata.num_rows)


def _read_arrow_table(location: str) -> Any:
    import pyarrow as pa

    source = _arrow_input(location)
    try:
        with pa.ipc.open_file(source) as reader:
            return reader.read_all()
    except Exception as ipc_error:
        try:
            import pyarrow.feather as feather

            return feather.read_table(location if not _is_remote(location) else source)
        except Exception as feather_error:
            raise CloudNativeOpenError(
                f"Could not read Arrow IPC file {location}: {ipc_error}"
            ) from feather_error


_FID_KEY = "__geolattice_fid__"


def _read_parquet_rows(
    paths: Sequence[str],
    columns: Optional[Sequence[str]],
    info: _TableInfo,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """Read rows; with ``bbox``, skip row groups whose covering-bbox statistics miss it.

    Rows read with a ``bbox`` carry their position in the full table under
    ``_FID_KEY`` so feature ids match an unfiltered read.
    """
    import pyarrow as pa

    covering = _bbox_covering_paths(info) if bbox is not None else None
    tables = []
    offset = 0
    for path in paths:
        parquet = _parquet_file(path)
        available = list(parquet.schema_arrow.names)
        if columns is None:
            wanted = available
        else:
            wanted = [name for name in columns if name in set(available)]
        if not wanted:
            wanted = available
        meta = parquet.metadata
        if covering is None:
            table = parquet.read(columns=wanted)
            if bbox is not None:
                table = table.append_column(_FID_KEY, pa.array(range(offset, offset + table.num_rows), pa.int64()))
            tables.append(table)
        else:
            groups, starts = _row_groups_for_bbox(meta, covering, bbox)
            for group in groups:
                table = parquet.read_row_group(group, columns=wanted)
                first = offset + starts[group]
                tables.append(table.append_column(
                    _FID_KEY, pa.array(range(first, first + table.num_rows), pa.int64())))
        offset += meta.num_rows
    if not tables:
        return []
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    return table.to_pylist()


def _bbox_covering_paths(info: _TableInfo) -> Optional[Dict[str, str]]:
    """Dotted Parquet column paths of the GeoParquet 1.1 ``bbox`` covering, if declared."""
    geo = info.geo_metadata or {}
    column = (geo.get("columns") or {}).get(geo.get("primary_column") or info.geometry_column or "", {})
    spec = (column.get("covering") or {}).get("bbox")
    if not isinstance(spec, dict) or not all(k in spec for k in ("xmin", "ymin", "xmax", "ymax")):
        return None
    return {key: ".".join(spec[key]) for key in ("xmin", "ymin", "xmax", "ymax")}


def _row_groups_for_bbox(
    meta: Any, covering: Dict[str, str], bbox: Tuple[float, float, float, float]
) -> Tuple[List[int], List[int]]:
    """Row groups that may intersect ``bbox``, and the first row index of every group."""
    min_x, min_y, max_x, max_y = bbox
    starts, keep, row = [], [], 0
    for group in range(meta.num_row_groups):
        rg = meta.row_group(group)
        starts.append(row)
        row += rg.num_rows
        stats: Dict[str, Any] = {}
        for index in range(rg.num_columns):
            chunk = rg.column(index)
            for key, dotted in covering.items():
                if chunk.path_in_schema == dotted and chunk.statistics is not None and chunk.statistics.has_min_max:
                    stats[key] = chunk.statistics
        if len(stats) < 4:
            keep.append(group)                        # no statistics: cannot rule the group out
            continue
        disjoint = (stats["xmin"].min > max_x or stats["xmax"].max < min_x
                    or stats["ymin"].min > max_y or stats["ymax"].max < min_y)
        if not disjoint:
            keep.append(group)
    return keep, starts


def _columns_to_load(selected: Optional[Sequence[str]], info: _TableInfo) -> Optional[List[str]]:
    if selected is None:
        return None
    names = list(selected)
    for extra in (info.geometry_column, info.cell_column, info.x_column, info.y_column):
        if extra and extra not in names:
            names.append(extra)
    return names


def _gdal_cloud_path(location: str) -> str:
    if location.startswith("/vsi"):
        return location
    if location.startswith("https://") or location.startswith("http://"):
        return "/vsicurl/" + location
    if location.startswith("s3://"):
        return "/vsis3/" + location[len("s3://") :]
    return location


class _ByteSource:
    """Random-access bytes for a local file or an HTTP range request."""

    def read(self, offset: int, size: int) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def size(self) -> Optional[int]:
        return None


class _FileSource(_ByteSource):
    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self._handle.seek(0, os.SEEK_END)
        self._size = self._handle.tell()

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0:
            raise CloudNativeOpenError(f"Invalid byte range offset={offset} size={size}")
        self._handle.seek(offset)
        data = self._handle.read(size)
        if len(data) != size:
            raise CloudNativeOpenError(
                f"Needed {size} bytes at offset {offset}, and the file ended after {len(data)}."
            )
        return data

    def close(self) -> None:
        self._handle.close()

    @property
    def size(self) -> Optional[int]:
        return self._size


class _HttpSource(_ByteSource):
    def __init__(self, url: str, timeout: float = 60) -> None:
        self.url = url
        self.timeout = timeout

    def read(self, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        end = offset + size - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={offset}-{end}",
                "Accept-Encoding": "identity",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                payload = response.read()
        except Exception as exc:
            raise CloudNativeOpenError(
                f"HTTP range request failed for {self.url} bytes {offset}-{end}: {exc}"
            ) from exc
        if status == 200 and offset != 0:
            payload = payload[offset : offset + size]
        if len(payload) < size:
            raise CloudNativeOpenError(
                f"Range bytes {offset}-{end} on {self.url} returned {len(payload)} bytes."
            )
        return payload[:size]


def _byte_source(location: str) -> _ByteSource:
    if location.startswith("/vsicurl/"):
        return _HttpSource(location[len("/vsicurl/") :])
    parsed = urlparse(location)
    if parsed.scheme in {"http", "https"}:
        return _HttpSource(location)
    if parsed.scheme == "file":
        return _FileSource(Path(parsed.path))
    if parsed.scheme:
        raise CloudNativeOpenError(
            f"Range reads for {parsed.scheme}:// are not built in. "
            "Use an https URL, a local path, or a GDAL /vsicurl/ path."
        )
    path = Path(location).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Vector dataset does not exist: {path}")
    return _FileSource(path)


def _decompress(data: bytes, compression: int, what: str) -> bytes:
    if compression in {0, 1}:
        return data
    if compression == 2:
        try:
            return gzip.decompress(data)
        except OSError as exc:
            raise CloudNativeOpenError(f"Could not gzip-decompress {what}: {exc}") from exc
    if compression == 3:
        try:
            import brotli
        except ImportError as exc:
            raise CloudNativeOpenError(
                f"{what} uses brotli. Install the brotli package to read it."
            ) from exc
        return brotli.decompress(data)
    if compression == 4:
        try:
            import zstandard
        except ImportError as exc:
            raise CloudNativeOpenError(
                f"{what} uses zstd. Install the zstandard package to read it."
            ) from exc
        return zstandard.ZstdDecompressor().decompress(data)
    raise CloudNativeOpenError(f"Unknown compression code {compression} on {what}.")


class _TileEntry:
    def __init__(self, tile_id: int, offset: int, length: int, run_length: int) -> None:
        self.tile_id = tile_id
        self.offset = offset
        self.length = length
        self.run_length = run_length

    def as_dict(self) -> Dict[str, int]:
        z, x, y = tileid_to_zxy(self.tile_id) if self.run_length else (0, 0, 0)
        return {
            "tile_id": self.tile_id,
            "offset": self.offset,
            "length": self.length,
            "run_length": self.run_length,
            "z": z,
            "x": x,
            "y": y,
        }


class _PMTilesArchive:
    def __init__(self, source: _ByteSource) -> None:
        raw = source.read(0, _PMTILES_HEADER_SIZE)
        self.header = _parse_pmtiles_header(raw)
        self._source = source
        root = source.read(self.header["root_offset"], self.header["root_length"])
        root = _decompress(root, self.header["internal_compression"], "PMTiles root directory")
        self.root_entries = _decode_directory(root)
        self.metadata = _read_pmtiles_metadata(source, self.header)

    def entries(self) -> Iterator[_TileEntry]:
        pending = list(self.root_entries)
        seen = 0
        while pending:
            entry = pending.pop(0)
            seen += 1
            if seen > 2_000_000:
                raise CloudNativeOpenError("PMTiles directory walk exceeded 2000000 entries.")
            if entry.run_length == 0:
                blob = self._source.read(
                    self.header["leaf_offset"] + entry.offset, entry.length
                )
                blob = _decompress(blob, self.header["internal_compression"], "PMTiles leaf directory")
                pending = _decode_directory(blob) + pending
                continue
            yield entry

    def read_tile(self, z: int, x: int, y: int, decode: bool = True) -> Optional[bytes]:
        _validate_tile_coord(z, x, y)
        tile_id = zxy_to_tileid(z, x, y)
        for entry in self.entries():
            if entry.run_length <= 0:
                continue
            if entry.tile_id <= tile_id < entry.tile_id + entry.run_length:
                blob = self._source.read(self.header["tile_data_offset"] + entry.offset, entry.length)
                if not decode:
                    return blob
                return _decompress(blob, self.header["tile_compression"], f"PMTiles tile {z}/{x}/{y}")
        return None


def _parse_pmtiles_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < _PMTILES_HEADER_SIZE or raw[:7] != _PMTILES_MAGIC:
        raise CloudNativeOpenError("Not a PMTiles archive (missing the PMTiles magic number).")
    version = raw[7]
    if version != 3:
        raise CloudNativeOpenError(f"Only PMTiles version 3 is supported (found version {version}).")
    fields = struct.unpack_from("<7Q", raw, 8)
    (
        root_offset,
        root_length,
        metadata_offset,
        metadata_length,
        leaf_offset,
        leaf_length,
        tile_data_offset,
    ) = fields[:7]
    tile_data_length, addressed, tile_entries, tile_contents = struct.unpack_from("<4Q", raw, 64)
    clustered, internal, tile_compression, tile_type, min_zoom, max_zoom = struct.unpack_from(
        "<6B", raw, 96
    )
    min_lon, min_lat = _position(raw, 102)
    max_lon, max_lat = _position(raw, 110)
    center_zoom = raw[118]
    center_lon, center_lat = _position(raw, 119)
    return {
        "version": version,
        "root_offset": root_offset,
        "root_length": root_length,
        "metadata_offset": metadata_offset,
        "metadata_length": metadata_length,
        "leaf_offset": leaf_offset,
        "leaf_length": leaf_length,
        "tile_data_offset": tile_data_offset,
        "tile_data_length": tile_data_length,
        "addressed_tiles": addressed,
        "tile_entries": tile_entries,
        "tile_contents": tile_contents,
        "clustered": clustered,
        "internal_compression": internal,
        "tile_compression": tile_compression,
        "tile_type": tile_type,
        "tile_type_name": _TILE_TYPES.get(tile_type, "unknown"),
        "internal_compression_name": _COMPRESSION.get(internal, "unknown"),
        "tile_compression_name": _COMPRESSION.get(tile_compression, "unknown"),
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "center_zoom": center_zoom,
        "center_lon": center_lon,
        "center_lat": center_lat,
    }


_COMPRESSION = {0: "unknown", 1: "none", 2: "gzip", 3: "brotli", 4: "zstd"}
_TILE_TYPES = {
    0: "unknown",
    1: "mvt",
    2: "png",
    3: "jpeg",
    4: "webp",
    5: "avif",
    6: "mlt",
}


def _position(raw: bytes, offset: int) -> Tuple[float, float]:
    lon, lat = struct.unpack_from("<ii", raw, offset)
    return lon / 10_000_000, lat / 10_000_000


def _read_pmtiles_metadata(source: _ByteSource, header: Dict[str, Any]) -> Dict[str, Any]:
    if header["metadata_length"] <= 0:
        return {}
    blob = source.read(header["metadata_offset"], header["metadata_length"])
    blob = _decompress(blob, header["internal_compression"], "PMTiles metadata")
    try:
        value = json.loads(blob.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CloudNativeOpenError(f"PMTiles metadata is not a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise CloudNativeOpenError("PMTiles metadata must be a JSON object.")
    return value


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(buf):
            raise CloudNativeOpenError("Truncated PMTiles directory.")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise CloudNativeOpenError("PMTiles varint is longer than 64 bits.")


def _decode_directory(buf: bytes) -> List[_TileEntry]:
    count, pos = _read_varint(buf, 0)
    if count <= 0 or count > 2_000_000:
        raise CloudNativeOpenError(f"PMTiles directory entry count {count} is not usable.")
    tile_ids: List[int] = []
    last = 0
    for _ in range(count):
        delta, pos = _read_varint(buf, pos)
        last += delta
        tile_ids.append(last)
    run_lengths: List[int] = []
    for _ in range(count):
        value, pos = _read_varint(buf, pos)
        run_lengths.append(value)
    lengths: List[int] = []
    for _ in range(count):
        value, pos = _read_varint(buf, pos)
        if value <= 0:
            raise CloudNativeOpenError("PMTiles entry length must be greater than 0.")
        lengths.append(value)
    entries: List[_TileEntry] = []
    for index in range(count):
        value, pos = _read_varint(buf, pos)
        if value == 0 and index > 0:
            offset = entries[index - 1].offset + entries[index - 1].length
        else:
            offset = value - 1
        entries.append(_TileEntry(tile_ids[index], offset, lengths[index], run_lengths[index]))
    return entries


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y


def zxy_to_tileid(z: int, x: int, y: int) -> int:
    """Hilbert tile id used by PMTiles, accumulated from zoom 0."""
    _validate_tile_coord(z, x, y)
    if z == 0:
        return 0
    return (4**z - 1) // 3 + _xy_to_hilbert(z, x, y)


def tileid_to_zxy(tile_id: int) -> Tuple[int, int, int]:
    """Inverse of :func:`zxy_to_tileid`."""
    if tile_id < 0:
        raise CloudNativeOpenError(f"Tile id {tile_id} is negative.")
    z = 0
    while tile_id >= (4 ** (z + 1) - 1) // 3:
        z += 1
        if z > 26:
            raise CloudNativeOpenError(f"Tile id {tile_id} is past zoom 26.")
    if z == 0:
        return 0, 0, 0
    x, y = _hilbert_to_xy(z, tile_id - (4**z - 1) // 3)
    return z, x, y


def _xy_to_hilbert(z: int, x: int, y: int) -> int:
    n = 1 << z
    d = 0
    s = n >> 1
    while s:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s >>= 1
    return d


def _hilbert_to_xy(z: int, d: int) -> Tuple[int, int]:
    x = y = 0
    s = 1
    n = 1 << z
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def _validate_tile_coord(z: int, x: int, y: int) -> None:
    if isinstance(z, bool) or isinstance(x, bool) or isinstance(y, bool):
        raise TypeError("z, x, and y must be integers")
    if not all(isinstance(value, int) for value in (z, x, y)):
        raise TypeError("z, x, and y must be integers")
    if z < 0 or z > 26:
        raise ValueError("zoom must be between 0 and 26")
    span = 1 << z
    if x < 0 or y < 0 or x >= span or y >= span:
        raise ValueError(f"tile {z}/{x}/{y} is outside 0..{span - 1}")


def _tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    span = float(1 << z)
    min_lon = x / span * 360.0 - 180.0
    max_lon = (x + 1) / span * 360.0 - 180.0
    max_lat = _tile_lat(y, span)
    min_lat = _tile_lat(y + 1, span)
    return min_lon, min_lat, max_lon, max_lat


def _tile_lat(y: float, span: float) -> float:
    merc = math.pi * (1 - 2 * y / span)
    return math.degrees(math.atan(math.sinh(merc)))


def _tile_polygon(z: int, x: int, y: int) -> Polygon:
    min_lon, min_lat, max_lon, max_lat = _tile_bounds(z, x, y)
    return Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )


def _pmtiles_feature_zoom(archive: "_PMTilesArchive") -> str:
    """Highest zoom that has a tile, which is the one GDAL can decode features from."""
    zooms = []
    for entry in archive.root_entries:
        if entry.run_length <= 0:
            continue
        zooms.append(tileid_to_zxy(entry.tile_id)[0])
        zooms.append(tileid_to_zxy(entry.tile_id + entry.run_length - 1)[0])
    if zooms:
        return str(max(zooms))
    layers = archive.metadata.get("vector_layers")
    if isinstance(layers, list):
        layer_zooms = [
            int(layer["maxzoom"])
            for layer in layers
            if isinstance(layer, dict) and str(layer.get("maxzoom", "")).isdigit()
        ]
        if layer_zooms:
            return str(max(layer_zooms))
    return str(archive.header["max_zoom"])


def _pmtiles_bounds(header: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Bounds]:
    raw = metadata.get("bounds")
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 4:
            try:
                min_lon, min_lat, max_lon, max_lat = (float(part) for part in parts)
            except ValueError:
                min_lon = None
            else:
                if min_lon <= max_lon and min_lat <= max_lat:
                    return Bounds(min_lon, min_lat, max_lon, max_lat)
    try:
        return Bounds(header["min_lon"], header["min_lat"], header["max_lon"], header["max_lat"])
    except (KeyError, ValueError):
        return None


def _read_copc(source: _ByteSource) -> Dict[str, Any]:
    header = source.read(0, 375)
    if header[:4] != b"LASF":
        raise CloudNativeOpenError("COPC files are LAS/LAZ and must start with LASF.")
    version = (header[24], header[25])
    if version < (1, 4):
        raise CloudNativeOpenError(f"COPC requires LAS 1.4 (found {version[0]}.{version[1]}).")
    header_size = struct.unpack_from("<H", header, 94)[0]
    offset_to_points = struct.unpack_from("<I", header, 96)[0]
    num_vlrs = struct.unpack_from("<I", header, 100)[0]
    point_format = header[104]
    record_length = struct.unpack_from("<H", header, 105)[0]
    scale = struct.unpack_from("<3d", header, 131)
    offset = struct.unpack_from("<3d", header, 155)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", header, 179)
    point_count = struct.unpack_from("<Q", header, 247)[0]
    cursor = header_size
    info = None
    wkt = None
    for _ in range(num_vlrs):
        vlr = source.read(cursor, _LAS_VLR_HEADER)
        user_id = vlr[2:18].split(b"\x00", 1)[0].decode("ascii", "replace")
        record_id, record_length_vlr = struct.unpack_from("<HH", vlr, 18)
        payload = source.read(cursor + _LAS_VLR_HEADER, record_length_vlr)
        if user_id == "copc" and record_id == 1:
            info = _parse_copc_info(payload)
        if user_id == "LASF_Projection" and record_id == 2112:
            wkt = payload.split(b"\x00", 1)[0].decode("utf-8", "replace")
        cursor += _LAS_VLR_HEADER + record_length_vlr
    if info is None:
        raise CloudNativeOpenError(
            "This LAS/LAZ file has no COPC info VLR (user id 'copc', record id 1)."
        )
    hierarchy = _read_copc_hierarchy(source, info)
    compressed = [
        node
        for node in hierarchy
        if node["point_count"] >= 0 and node["byte_size"] != node["point_count"] * record_length
    ]
    crs, crs_wkt = _crs_from_wkt_text(wkt)
    try:
        bounds = Bounds(min_x, min_y, max_x, max_y)
    except ValueError:
        bounds = _cube_bounds(info)
    return {
        "info": info,
        "hierarchy": hierarchy,
        "point_format": point_format,
        "record_length": record_length,
        "scale": scale,
        "offset": offset,
        "point_count": point_count,
        "bounds": bounds,
        "crs": crs,
        "crs_wkt": crs_wkt,
        "offset_to_points": offset_to_points,
        "compressed_voxels": compressed,
        "_laspy_cache": None,
    }


def _parse_copc_info(payload: bytes) -> Dict[str, Any]:
    if len(payload) < 72:
        raise CloudNativeOpenError(f"COPC info VLR is {len(payload)} bytes; at least 72 are required.")
    center_x, center_y, center_z, halfsize, spacing, root_offset, root_size, gps_min, gps_max = (
        struct.unpack_from("<5dQQ2d", payload)
    )
    return {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "halfsize": halfsize,
        "spacing": spacing,
        "root_hier_offset": root_offset,
        "root_hier_size": root_size,
        "gpstime_minimum": gps_min,
        "gpstime_maximum": gps_max,
    }


def _read_copc_hierarchy(source: _ByteSource, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = [(info["root_hier_offset"], info["root_hier_size"])]
    nodes: List[Dict[str, Any]] = []
    seen = set()
    while pages:
        offset, size = pages.pop(0)
        if size <= 0 or (offset, size) in seen:
            continue
        seen.add((offset, size))
        if size % 32 != 0:
            raise CloudNativeOpenError(f"COPC hierarchy page size {size} is not a multiple of 32.")
        blob = source.read(offset, size)
        for index in range(0, size, 32):
            level, key_x, key_y, key_z, entry_offset, byte_size, point_count = struct.unpack_from(
                "<4iQii", blob, index
            )
            node = {
                "level": level,
                "x": key_x,
                "y": key_y,
                "z": key_z,
                "offset": entry_offset,
                "byte_size": byte_size,
                "point_count": point_count,
            }
            nodes.append(node)
            if point_count < 0 and byte_size > 0:
                pages.append((entry_offset, byte_size))
        if len(seen) > 100_000:
            raise CloudNativeOpenError("COPC hierarchy has more than 100000 pages.")
    return nodes


def _voxel_bounds(info: Dict[str, Any], node: Dict[str, Any]) -> Bounds:
    level = int(node["level"])
    width = (2 * float(info["halfsize"])) / float(1 << level) if level >= 0 else 0.0
    min_x = float(info["center_x"]) - float(info["halfsize"]) + node["x"] * width
    min_y = float(info["center_y"]) - float(info["halfsize"]) + node["y"] * width
    min_z = float(info["center_z"]) - float(info["halfsize"]) + node["z"] * width
    return Bounds(min_x, min_y, min_x + width, min_y + width)


def _cube_bounds(info: Dict[str, Any]) -> Bounds:
    half = float(info["halfsize"])
    return Bounds(
        float(info["center_x"]) - half,
        float(info["center_y"]) - half,
        float(info["center_x"]) + half,
        float(info["center_y"]) + half,
    )


def _decode_raw_voxel(
    source: _ByteSource, copc: Dict[str, Any], node: Dict[str, Any]
) -> List[Tuple[float, float, float]]:
    count = int(node["point_count"])
    if count <= 0:
        return []
    record_length = int(copc["record_length"])
    expected = count * record_length
    if int(node["byte_size"]) != expected:
        raise CloudNativeOpenError(
            "COPC voxel is compressed, so it cannot be decoded as raw LAS points."
        )
    blob = source.read(int(node["offset"]), expected)
    return _decode_raw_points(blob, count, record_length, copc["scale"], copc["offset"])


def _decode_raw_points(
    blob: bytes,
    count: int,
    record_length: int,
    scale: Sequence[float],
    offset: Sequence[float],
) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    for index in range(count):
        start = index * record_length
        ix, iy, iz = struct.unpack_from("<3i", blob, start)
        points.append(
            (
                ix * scale[0] + offset[0],
                iy * scale[1] + offset[1],
                iz * scale[2] + offset[2],
            )
        )
    return points


def _laspy_points(location: str) -> List[Tuple[float, float, float]]:
    if _is_remote(location):
        raise CloudNativeOpenError(
            "LAZ-compressed COPC voxels on an HTTP URL need a range-capable LAZ decoder. "
            "The header and octree were read. Download the file, or install laspy and open "
            "the local path, to decode points."
        )
    if not _module_available("laspy"):
        raise CloudNativeOpenError(
            "These COPC points are LAZ-compressed. Install laspy and lazrs to decode them. "
            "The COPC header and octree hierarchy are available on the dataset."
        )
    import laspy

    cloud = laspy.read(location)
    return list(zip((float(v) for v in cloud.x), (float(v) for v in cloud.y), (float(v) for v in cloud.z)))


def _crs_from_wkt_text(wkt: Optional[str]) -> Tuple[Optional[CRS], Optional[str]]:
    if not wkt or not wkt.strip():
        return None, None
    try:
        crs = CRS.from_wkt(wkt)
    except (RuntimeError, ValueError, TypeError):
        return None, wkt
    return crs, wkt


def _local_dir(location: str) -> Path:
    if _is_remote(location):
        raise CloudNativeOpenError(
            f"{location} is remote. Iceberg and Delta tables are opened from a local "
            "table directory (metadata/ or _delta_log) unless pyiceberg or deltalake is installed."
        )
    path = Path(location).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Vector dataset does not exist: {path}")
    if not path.is_dir():
        raise CloudNativeOpenError(f"Expected a table directory, got a file: {path}")
    return path


def _replay_delta_log(log: Path) -> Tuple[List[str], List[Tuple[str, str]], int]:
    commits = sorted(path for path in log.glob("*.json") if path.stem.isdigit())
    if not commits:
        raise CloudNativeOpenError(f"Delta log {log} has no JSON commits.")
    live: Dict[str, None] = {}
    schema_fields: List[Tuple[str, str]] = []
    version = 0
    for commit in commits:
        version = int(commit.stem)
        for line in commit.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            action = json.loads(line)
            if "metaData" in action:
                schema_fields = _delta_schema_fields(action["metaData"].get("schemaString", ""))
            add = action.get("add")
            if isinstance(add, dict) and add.get("path"):
                live[str(add["path"])] = None
            remove = action.get("remove")
            if isinstance(remove, dict) and remove.get("path"):
                live.pop(str(remove["path"]), None)
    return list(live), schema_fields, version


def _delta_schema_fields(schema_string: str) -> List[Tuple[str, str]]:
    if not schema_string:
        return []
    try:
        schema = json.loads(schema_string)
    except json.JSONDecodeError:
        return []
    fields = schema.get("fields") if isinstance(schema, dict) else None
    if not isinstance(fields, list):
        return []
    parsed: List[Tuple[str, str]] = []
    for field in fields:
        if isinstance(field, dict) and "name" in field:
            parsed.append((str(field["name"]), _type_name(field.get("type"))))
    return parsed


def _binary_field(fields: Sequence[Tuple[str, str]]) -> Optional[str]:
    for name, type_name in fields:
        if name.lower() in _GEOM_COLUMNS and type_name.lower() in {"binary", "bytes"}:
            return name
    for name, type_name in fields:
        if type_name.lower() in {"binary", "bytes"}:
            return name
    return None


def _type_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("type", value))
    return str(value)


def _latest_iceberg_metadata(root: Path) -> Path:
    metadata = root / "metadata"
    files = sorted(metadata.glob("*.metadata.json"))
    if not files:
        raise CloudNativeOpenError(f"No Iceberg metadata JSON in {metadata}")

    def sort_key(path: Path) -> Tuple[int, str]:
        stem = path.name.split(".", 1)[0]
        # v3.metadata.json or 00001-uuid.metadata.json
        digits = "".join(char for char in stem if char.isdigit())
        return (int(digits) if digits else 0, path.name)

    return max(files, key=sort_key)


def _iceberg_current_schema(metadata: Dict[str, Any]) -> Dict[str, Any]:
    schemas = metadata.get("schemas") or []
    current = metadata.get("current-schema-id")
    for schema in schemas:
        if isinstance(schema, dict) and schema.get("schema-id") == current:
            return schema
    if schemas and isinstance(schemas[-1], dict):
        return schemas[-1]
    return {}


def _iceberg_geometry_fields(schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for field in schema.get("fields") or []:
        if not isinstance(field, dict):
            continue
        type_name = _type_name(field.get("type")).lower()
        if type_name in {"geometry", "geography"}:
            found.append((str(field["name"]), type_name))
    return found


def _iceberg_data_files(root: Path) -> List[str]:
    data = root / "data"
    if not data.is_dir():
        return []
    return sorted(str(path) for path in data.rglob("*.parquet"))


def _geojsonseq_header(location: str) -> Tuple[List[str], Optional[int]]:
    if _is_remote(location):
        return [], None
    names: List[str] = []
    count = 0
    for _fid, properties, _geom in _iter_geojsonseq(location):
        count += 1
        if not names:
            names = list(properties)
    return names, count


def _iter_geojsonseq(location: str) -> Iterator[Tuple[int, Dict[str, Any], Any]]:
    if _is_remote(location):
        request = urllib.request.Request(location, headers={"Accept-Encoding": "identity"})
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except Exception as exc:
            raise CloudNativeOpenError(f"Could not stream {location}: {exc}") from exc
        handle = response
        closer: Optional[Callable[[], None]] = response.close
    else:
        path = Path(location).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Vector dataset does not exist: {path}")
        handle = path.open("r", encoding="utf-8", errors="replace")
        closer = handle.close
    fid = 0
    try:
        for line in handle:
            text = line.replace("\x1e", "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CloudNativeOpenError(
                    f"Line {fid + 1} of {location} is not a GeoJSON feature: {exc}"
                ) from exc
            if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                features = payload.get("features") or []
            else:
                features = [payload]
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                properties = feature.get("properties") or {}
                if not isinstance(properties, dict):
                    properties = {"properties": properties}
                geometry = feature.get("geometry")
                geom = None
                if isinstance(geometry, dict):
                    try:
                        geom = shape(geometry)
                    except Exception as exc:
                        raise CloudNativeOpenError(
                            f"Could not read geometry on feature {fid} of {location}: {exc}"
                        ) from exc
                yield fid, properties, geom
                fid += 1
    finally:
        if closer is not None:
            closer()


def _apply_fallback_crs(dataset: CloudNativeDataset, fallback: Optional[CRS]) -> None:
    if fallback is None:
        return
    for layer in dataset.layers:
        if layer.crs is None:
            layer.crs = fallback


def _coerce_crs(value: Union[str, int, CRS, None]) -> Optional[CRS]:
    if value is None:
        return None
    if isinstance(value, CRS):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return CRS.from_epsg(value)
    return CRS(str(value))


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


def _select_layer(layers: Sequence[CloudNativeLayer], layer: Union[str, int], path: str) -> CloudNativeLayer:
    if isinstance(layer, bool) or not isinstance(layer, (str, int)):
        raise TypeError("layer must be a layer name or an integer index")
    if isinstance(layer, int):
        if layer < 0 or layer >= len(layers):
            raise CloudNativeOpenError(
                f"Layer index {layer} is outside 0..{len(layers) - 1} for {path}"
            )
        return layers[layer]
    exact = [item for item in layers if item.name == layer]
    if not exact:
        folded = [item for item in layers if item.name.lower() == layer.lower()]
        if len(folded) == 1:
            exact = folded
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise CloudNativeOpenError(
            f"Layer name {layer!r} matches more than one layer in {path}: "
            f"{[item.name for item in exact]}"
        )
    raise CloudNativeOpenError(
        f"Layer {layer!r} was not found in {path}. "
        f"Available layers: {[item.name for item in layers]}"
    )


def _layer_name(location: str) -> str:
    if _is_remote(location):
        name = Path(urlparse(location).path).name
    else:
        name = Path(location).name
    if name.endswith(".copc.laz") or name.endswith(".copc.las"):
        return name.split(".")[0]
    stem = Path(name).stem
    return stem or name or "features"


def _checked_columns(
    columns: Optional[Sequence[str]], available: Sequence[str], layer_name: str
) -> Optional[Tuple[str, ...]]:
    selected = _validate_columns(columns)
    if selected is None:
        return None
    known = set(available)
    missing = [name for name in selected if name not in known]
    if missing and known:
        raise CloudNativeOpenError(
            f"Column(s) {missing} are not in layer {layer_name!r}. Available columns: {list(available)}"
        )
    return selected


def _parse_where(where: Optional[str]) -> List[Tuple[str, Any]]:
    if where is None:
        return []
    text = where.strip()
    if not text:
        return []
    predicates: List[Tuple[str, Any]] = []
    for part in _WHERE_AND.split(text):
        match = _WHERE_TERM.fullmatch(part.strip())
        if match is None:
            raise CloudNativeOpenError(
                f"Unsupported where clause {where!r}. "
                "Use column = value, combined with AND."
            )
        predicates.append((match.group(1), _literal(match.group(2).strip())))
    return predicates


def _literal(raw: str) -> Any:
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    folded = raw.lower()
    if folded == "null":
        return None
    if folded == "true":
        return True
    if folded == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _matches(properties: Dict[str, Any], predicates: Sequence[Tuple[str, Any]]) -> bool:
    for name, expected in predicates:
        if properties.get(name) != expected:
            return False
    return True


def _geom_hits(geom: Any, bbox: Tuple[float, float, float, float]) -> bool:
    if geom is None or geom.is_empty:
        return False
    min_x, min_y, max_x, max_y = bbox
    try:
        return bool(geom.intersects(box(min_x, min_y, max_x, max_y)))
    except Exception:
        gx1, gy1, gx2, gy2 = geom.bounds
        return not (gx2 < min_x or gx1 > max_x or gy2 < min_y or gy1 > max_y)


def _bounds_hit(node: Bounds, bbox: Tuple[float, float, float, float]) -> bool:
    min_x, min_y, max_x, max_y = bbox
    return not (node.maxx < min_x or node.minx > max_x or node.maxy < min_y or node.miny > max_y)


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


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_present(candidates: Sequence[str], names: Sequence[str]) -> Optional[str]:
    folded = {name.lower(): name for name in names}
    for candidate in candidates:
        found = folded.get(candidate.lower())
        if found is not None:
            return found
    return None


def _as_geodataframe(frame: Any, layer: CloudNativeLayer) -> Any:
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
    "CLOUD_NATIVE_FORMATS",
    "CloudNativeFormat",
    "CloudNativeDataset",
    "CloudNativeLayer",
    "CloudNativeOpenError",
    "open_cloud_native_vector_data",
    "open_cloud_native_vector",
    "supported_cloud_native_formats",
    "logical_geometry_kind",
    "zxy_to_tileid",
    "tileid_to_zxy",
]
