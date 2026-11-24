"""
Write cloud-native vector layouts: GeoParquet, FlatGeobuf and PMTiles.

These are the formats :mod:`geolattice.vector.open_cloud_native_vector_data` reads
with HTTP range requests. They complete the file writers in
:mod:`geolattice.vector.write_vector_data`, which cannot write GeoParquet when the
GDAL build has no Parquet driver.

- GeoParquet is written with GeoPandas and PyArrow, as GeoParquet 1.1 with a
  ``bbox`` covering column and rows sorted along a Hilbert curve, so readers
  can skip row groups that fall outside a bounding box.
- FlatGeobuf is written with a packed Hilbert R-tree, so GDAL can fetch only
  the features inside a window.
- PMTiles is written with GDAL's PMTiles driver (GDAL 3.8 or later), as one
  archive of Mapbox vector tiles for web maps.

Example:
    from geolattice.vector import read_geoparquet, write_geoparquet

    write_geoparquet("counties.parquet", counties)
    syracuse = read_geoparquet("counties.parquet", bbox=(-76.3, 42.9, -76.0, 43.2))
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from geolattice.vector.read_vector_data import DriverUnavailableError
from geolattice.vector.write_vector_data import VectorWriteError

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]


def write_geoparquet(
    path: PathLike,
    data: Any,
    *,
    crs: Any = None,
    compression: str = "zstd",
    covering_bbox: bool = True,
    sort_spatially: bool = True,
    row_group_size: Optional[int] = 65_536,
    geometry_encoding: str = "WKB",
    schema_version: str = "1.1.0",
    overwrite: bool = True,
) -> str:
    """
    Write features to a GeoParquet file.

    Args:
        path: Output ``.parquet`` path.
        data: GeoDataFrame, GeoSeries, DataFrame with a ``geometry`` column, a
            vector file path, or an iterable of mappings with a ``geometry``
            key (shapely geometry, WKT, or GeoJSON geometry mapping).
        crs: CRS to assign when ``data`` has none.
        compression: Parquet compression codec (``"zstd"``, ``"snappy"``,
            ``"gzip"`` or ``"none"``).
        covering_bbox: Add the GeoParquet 1.1 ``bbox`` covering column, which
            lets readers filter row groups by bounding box.
        sort_spatially: Order rows along a Hilbert curve so nearby features
            share row groups. Without it the covering column filters poorly.
        row_group_size: Rows per row group. Smaller groups filter more finely
            and cost more metadata.
        geometry_encoding: ``"WKB"`` or ``"geoarrow"``.
        schema_version: GeoParquet metadata version.
        overwrite: Replace an existing file.

    Returns:
        The output path.

    Raises:
        VectorWriteError: If the file exists and ``overwrite`` is false, or the
            data cannot be written.
    """
    destination = Path(path)
    _check_overwrite(destination, overwrite)
    frame = _as_geodataframe(data, crs)
    if sort_spatially and len(frame) > 1:
        frame = _hilbert_sorted(frame)
    kwargs: Dict[str, Any] = {}
    if row_group_size:
        kwargs["row_group_size"] = int(row_group_size)
    try:
        frame.to_parquet(
            destination,
            index=False,
            compression=None if compression == "none" else compression,
            geometry_encoding=geometry_encoding,
            write_covering_bbox=covering_bbox,
            schema_version=schema_version,
            **kwargs,
        )
    except Exception as exc:  # pyarrow raises several types
        raise VectorWriteError(f"Could not write GeoParquet {destination}: {exc}") from exc
    return str(destination)


def read_geoparquet(
    path: PathLike,
    *,
    bbox: Optional[Sequence[float]] = None,
    bbox_crs: Any = None,
    columns: Optional[Sequence[str]] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Read a GeoParquet file, optionally only the features inside a bounding box.

    With a ``bbox`` covering column (GeoParquet 1.1) the filter is pushed down
    to row groups, so only matching groups are read, also over HTTP or S3.
    Without one the whole file is read and filtered afterwards.

    Args:
        path: Local path, ``s3://`` / ``gs://`` URI (PyArrow) or ``http(s)://`` URL (fsspec,
            which needs ``aiohttp``). Remote footers and row groups are read with range requests.
        bbox: ``(minx, miny, maxx, maxy)`` filter.
        bbox_crs: CRS of ``bbox`` when it differs from the file's CRS.
        columns: Attribute columns to read (the geometry column is always read).
        storage_options: Options for fsspec (for example ``{"anon": True}``).

    Returns:
        A GeoDataFrame.
    """
    gpd = _geopandas()
    location = os.fspath(path)
    query = None
    if bbox is not None:
        query = tuple(float(v) for v in bbox)
        if bbox_crs is not None:
            info = geoparquet_info(location, storage_options=storage_options)
            if info.get("crs") is not None:
                query = _transform_bbox(query, bbox_crs, info["crs"])
    kwargs: Dict[str, Any] = {}
    if location.lower().startswith(("http://", "https://")):
        # PyArrow has no HTTP filesystem and GeoPandas does not route http(s) URLs to fsspec,
        # so pass fsspec's HTTP filesystem explicitly; it reads the footer and row groups
        # with range requests.
        fsspec = _require("fsspec", "fsspec aiohttp")
        kwargs["filesystem"] = fsspec.filesystem("http", **dict(storage_options or {}))
    elif storage_options:
        kwargs["storage_options"] = dict(storage_options)
    if columns is not None:
        geom = geoparquet_info(location, storage_options=storage_options).get("primary_column", "geometry")
        kwargs["columns"] = list(dict.fromkeys([*columns, geom]))
    try:
        frame = gpd.read_parquet(location, bbox=query, **kwargs)
    except ValueError:
        # No covering column: read everything, then filter.
        frame = gpd.read_parquet(location, **kwargs)
        if query is not None:
            frame = frame.cx[query[0]:query[2], query[1]:query[3]]
    if query is not None:
        # Row-group filtering keeps whole groups; trim to the exact box.
        from shapely.geometry import box

        frame = frame[frame.intersects(box(*query))]
    return frame


def geoparquet_info(path: PathLike, *, storage_options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Summarize a GeoParquet file from its footer, without reading the data.

    Returns a dictionary with ``version``, ``primary_column``, ``encoding``,
    ``crs`` (a pyproj CRS or ``None``), ``bbox``, ``geometry_types``,
    ``covering`` (``True`` when a bbox covering column is declared),
    ``num_rows``, ``num_row_groups`` and ``columns``.
    """
    pq = _require("pyarrow.parquet", "pyarrow")
    location = os.fspath(path)
    if _is_url(location):
        fsspec = _require("fsspec", "fsspec")
        with fsspec.open(location, "rb", **dict(storage_options or {})) as handle:
            meta = pq.ParquetFile(handle).metadata
            schema = pq.ParquetFile(handle).schema_arrow
    else:
        parquet = pq.ParquetFile(location)
        meta, schema = parquet.metadata, parquet.schema_arrow
    raw = (schema.metadata or {}).get(b"geo")
    if raw is None:
        raise VectorWriteError(f"{location} is Parquet but has no GeoParquet 'geo' metadata.")
    geo = json.loads(raw)
    primary = geo.get("primary_column", "geometry")
    column = geo.get("columns", {}).get(primary, {})
    crs = None
    if "crs" not in column:
        crs = _pyproj_crs("OGC:CRS84")
    elif column["crs"] is not None:
        crs = _pyproj_crs(column["crs"])
    return {
        "version": geo.get("version"),
        "primary_column": primary,
        "encoding": column.get("encoding"),
        "crs": crs,
        "bbox": column.get("bbox"),
        "geometry_types": column.get("geometry_types", []),
        "covering": "covering" in column,
        "num_rows": meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "columns": schema.names,
    }


def write_flatgeobuf(
    path: PathLike,
    data: Any,
    *,
    crs: Any = None,
    spatial_index: bool = True,
    layer: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Write features to FlatGeobuf, with a packed Hilbert R-tree by default.

    The spatial index lets GDAL, QGIS and web clients read only the features
    inside a window with HTTP range requests.
    """
    destination = Path(path)
    _check_overwrite(destination, overwrite)
    frame = _as_geodataframe(data, crs)
    try:
        frame.to_file(destination, driver="FlatGeobuf", layer=layer or destination.stem,
                      SPATIAL_INDEX="YES" if spatial_index else "NO")
    except Exception as exc:
        raise VectorWriteError(f"Could not write FlatGeobuf {destination}: {exc}") from exc
    return str(destination)


def write_pmtiles(
    path: PathLike,
    data: Any,
    *,
    crs: Any = None,
    layer: Optional[str] = None,
    minzoom: int = 0,
    maxzoom: int = 14,
    simplification: Optional[float] = None,
    max_size: Optional[int] = None,
    overwrite: bool = True,
) -> str:
    """
    Write features to a PMTiles archive of Mapbox vector tiles.

    PMTiles serves map tiles from one static file with HTTP range requests, so
    a web map needs no tile server. The archive is for display; use GeoParquet
    or FlatGeobuf for analysis.

    Args:
        path: Output ``.pmtiles`` path.
        data: Features, as accepted by :func:`write_geoparquet`. They must have a CRS.
        crs: CRS to assign when ``data`` has none.
        layer: Vector-tile layer name; defaults to the file stem.
        minzoom, maxzoom: Zoom range to generate.
        simplification: Geometry simplification tolerance in tile pixels.
        max_size: Maximum compressed tile size in bytes.
        overwrite: Replace an existing file.

    Raises:
        DriverUnavailableError: If GDAL has no PMTiles driver (needs GDAL 3.8+).
        VectorWriteError: If GDAL cannot write the archive.
    """
    gdal = _gdal()
    if gdal.GetDriverByName("PMTiles") is None:
        raise DriverUnavailableError("GDAL has no PMTiles driver; GDAL 3.8 or later is required.")
    if not 0 <= minzoom <= maxzoom <= 24:
        raise ValueError("Zoom levels must satisfy 0 <= minzoom <= maxzoom <= 24.")
    destination = Path(path)
    _check_overwrite(destination, overwrite)
    frame = _as_geodataframe(data, crs)
    if frame.crs is None:
        raise VectorWriteError("PMTiles needs a CRS; pass crs= or set one on the data.")
    name = layer or destination.stem
    dataset_options = [f"MINZOOM={minzoom}", f"MAXZOOM={maxzoom}", f"NAME={name}"]
    if simplification is not None:
        dataset_options.append(f"SIMPLIFICATION={simplification}")
    if max_size is not None:
        dataset_options.append(f"MAX_SIZE={int(max_size)}")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / f"{name}.fgb"
        frame.to_file(staged, driver="FlatGeobuf", layer=name)
        try:
            result = gdal.VectorTranslate(
                str(destination), str(staged), format="PMTiles",
                datasetCreationOptions=dataset_options, layerCreationOptions=[f"NAME={name}"],
            )
        except Exception as exc:
            raise VectorWriteError(f"Could not write PMTiles {destination}: {exc}") from exc
        if result is None:
            raise VectorWriteError(f"GDAL could not write PMTiles {destination}.")
        result = None
    return str(destination)


# ----------------------------------------------------------------------------- helpers
def _as_geodataframe(data: Any, crs: Any = None) -> Any:
    gpd = _geopandas()
    pd = _require("pandas", "pandas")
    if isinstance(data, gpd.GeoDataFrame):
        frame = data
    elif isinstance(data, gpd.GeoSeries):
        frame = gpd.GeoDataFrame(geometry=data, crs=data.crs)
    elif isinstance(data, pd.DataFrame):
        if "geometry" not in data.columns:
            raise VectorWriteError("The table has no 'geometry' column.")
        frame = gpd.GeoDataFrame(data, geometry="geometry")
    elif isinstance(data, (str, Path)):
        location = os.fspath(data)
        frame = read_geoparquet(location) if location.lower().endswith((".parquet", ".geoparquet")) \
            else gpd.read_file(location)
    elif isinstance(data, Iterable):
        rows = [dict(row) for row in data]
        if not rows or any("geometry" not in row for row in rows):
            raise VectorWriteError("Each feature mapping needs a 'geometry' key.")
        for row in rows:
            row["geometry"] = _to_shapely(row["geometry"])
        frame = gpd.GeoDataFrame(rows, geometry="geometry")
    else:
        raise VectorWriteError(
            "Expected a GeoDataFrame, GeoSeries, DataFrame with geometry, a vector path, "
            "or an iterable of feature mappings."
        )
    if frame.crs is None and crs is not None:
        frame = frame.set_crs(crs)
    return frame


def _to_shapely(geom: Any) -> Any:
    from shapely import wkt
    from shapely.geometry import shape
    from shapely.geometry.base import BaseGeometry

    if geom is None or isinstance(geom, BaseGeometry):
        return geom
    if isinstance(geom, str):
        return wkt.loads(geom)
    if isinstance(geom, Mapping):
        return shape(geom)
    raise VectorWriteError(f"Unsupported geometry value: {type(geom).__name__}")


def _hilbert_sorted(frame: Any) -> Any:
    valid = frame.geometry.notna() & ~frame.geometry.is_empty
    if valid.sum() < 2:
        return frame
    distance = frame.geometry[valid].hilbert_distance()
    order = list(distance.sort_values(kind="stable").index) + list(frame.index[~valid])
    return frame.loc[order]


def _check_overwrite(destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise VectorWriteError(f"{destination} exists; pass overwrite=True to replace it.")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)


def _transform_bbox(bbox: BBox, src_crs: Any, dst_crs: Any) -> BBox:
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tuple(transformer.transform_bounds(*bbox, densify_pts=21))  # type: ignore[return-value]


def _pyproj_crs(value: Any) -> Any:
    from pyproj import CRS as ProjCRS

    return ProjCRS.from_user_input(value)


def _is_url(location: str) -> bool:
    return "://" in location and not location.startswith("file://")


def _geopandas() -> Any:
    return _require("geopandas", "geopandas")


def _gdal() -> Any:
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def _require(module: str, package: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module} is required here: pip install {package}") from exc


__all__ = [
    "write_geoparquet",
    "read_geoparquet",
    "geoparquet_info",
    "write_flatgeobuf",
    "write_pmtiles",
]
