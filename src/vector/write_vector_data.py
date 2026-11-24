"""
Write file-based vector datasets that :mod:`geolattice.vector.read_vector_data` can open.

The writer uses the same GDAL/OGR drivers as the reader. A file written here
can be opened with :func:`geolattice.vector.read_vector_data.read_vector_data`
or :func:`~geolattice.vector.read_vector_data.open_vectors_data`.

Example:
    from geolattice.vector import read_vector_data, write_vector_data

    write_vector_data(
        "sites.gpkg",
        [{"geometry": "POINT (-73.9 40.7)", "name": "nyc"}],
        layer="sites",
        crs=4326,
    )
    with read_vector_data("sites.gpkg") as dataset:
        print(dataset.layer.feature_count)
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from geolattice.crs import CRS
from geolattice.vector.read_vector_data import (
    VECTOR_FORMATS,
    DriverUnavailableError,
    VectorDataset,
    VectorFeature,
    VectorLayer,
    VectorOpenError,
    _gdal_modules,
    driver_available,
)

try:
    from shapely.geometry import Point, shape as _shape
except ImportError:  # shapely 1.8
    from shapely.geometry import Point
    from shapely.geometry import shape as _shape
from shapely.wkt import loads as _load_wkt

_X_NAMES = ("longitude", "lon", "long", "x", "easting", "east", "x_coord", "xcoord")
_Y_NAMES = ("latitude", "lat", "y", "northing", "north", "y_coord", "ycoord")
_GEOM_NAMES = ("wkt", "geom", "geometry", "the_geom", "wkb_geometry")
_GPX_LAYERS = ("waypoints", "routes", "tracks", "route_points", "track_points")
_SHAPE_SIDECARS = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".sbn",
    ".sbx",
    ".shp.xml",
    ".qix",
)


class VectorWriteError(VectorOpenError):
    """Raised when vector features cannot be written."""


def write_vector_data(
    path: Union[str, Path],
    data: Any,
    layer: str = "layer",
    driver: Optional[str] = None,
    crs: Any = None,
    encoding: Optional[str] = None,
    overwrite: bool = False,
    layer_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    dataset_options: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
) -> str:
    """
    Write vector features to a file the reader can open.

    ``data`` may be a :class:`VectorDataset`, a :class:`VectorLayer`, a
    GeoDataFrame or DataFrame, a GeoJSON Feature or FeatureCollection, a
    sequence of :class:`VectorFeature` or feature mappings, or a mapping of
    layer name to any of those. Geometry may be a Shapely geometry, WKT,
    GeoJSON, or longitude/latitude columns.

    Args:
        path: Output path. The extension selects the GDAL driver.
        data: Features or a mapping of layer name to features.
        layer: Layer name when ``data`` is a single layer.
        driver: GDAL/OGR driver short name. When omitted, the driver is chosen
            from the file extension.
        crs: CRS stored with the layer. An EPSG code, authority string, WKT,
            :class:`~geolattice.crs.CRS`, or a pyproj CRS are accepted.
            A CRS already attached to ``data`` is used when this is omitted.
        encoding: DBF encoding for ESRI Shapefile.
        overwrite: Replace an existing dataset at ``path``.
        layer_options: Extra GDAL layer creation options.
        dataset_options: Extra GDAL dataset creation options.

    Returns:
        Path of the written dataset.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is false.
        DriverUnavailableError: If this GDAL build cannot create the format.
        VectorWriteError: If the features or the driver reject the write.
    """
    destination = Path(path)
    if destination.suffix.lower() == ".kmz":
        return _write_kmz(
            destination,
            data,
            layer=layer,
            crs=crs,
            encoding=encoding,
            overwrite=overwrite,
            layer_options=layer_options,
            dataset_options=dataset_options,
        )

    driver_name = _choose_driver(destination, driver)
    _prepare_destination(destination, driver_name, overwrite)
    layers = _collect_layers(data, layer, crs)
    _write_ogr(
        destination,
        layers,
        driver_name,
        encoding=encoding,
        layer_options=_option_map(layer_options),
        dataset_options=_option_map(dataset_options),
    )
    return str(destination)


def write_shapefile(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write an ESRI Shapefile that :func:`read_vector_data` can open."""
    kwargs["driver"] = "ESRI Shapefile"
    return write_vector_data(path, data, **kwargs)


def write_geopackage(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write a GeoPackage that :func:`read_vector_data` can open."""
    kwargs["driver"] = "GPKG"
    return write_vector_data(path, data, **kwargs)


def write_geojson(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write GeoJSON that :func:`read_vector_data` can open."""
    kwargs["driver"] = "GeoJSON"
    return write_vector_data(path, data, **kwargs)


def write_csv(path: Union[str, Path], data: Any, **kwargs: Any) -> str:
    """Write delimited text with a WKT column that :func:`read_vector_data` can open."""
    kwargs["driver"] = "CSV"
    return write_vector_data(path, data, **kwargs)


def supported_write_formats() -> List[Dict[str, Any]]:
    """
    List file-based formats and whether this GDAL build can create them.

    Returns:
        One dictionary per reader format, with extensions, drivers, and
        ``writable`` true when a create-capable driver is installed.
        ``.kmz`` is included with KML because the writer zips a ``doc.kml``.
    """
    _, ogr, _ = _gdal_modules()
    records: List[Dict[str, Any]] = []
    for fmt in VECTOR_FORMATS:
        creatable = [
            name
            for name in fmt.drivers
            if _driver_can_create(ogr.GetDriverByName(name) if driver_available(name) else None)
        ]
        extensions = list(fmt.extensions)
        if fmt.name == "KML / KMZ" and ".kmz" not in extensions:
            extensions.append(".kmz")
        records.append(
            {
                "name": fmt.name,
                "extensions": extensions,
                "drivers": list(fmt.drivers),
                "creatable_drivers": creatable,
                "writable": bool(creatable) or (fmt.name == "KML / KMZ" and driver_available("KML")),
                "description": fmt.description,
            }
        )
    return records


def _write_kmz(
    destination: Path,
    data: Any,
    layer: str,
    crs: Any,
    encoding: Optional[str],
    overwrite: bool,
    layer_options: Optional[Union[Sequence[str], Dict[str, Any]]],
    dataset_options: Optional[Union[Sequence[str], Dict[str, Any]]],
) -> str:
    _prepare_destination(destination, "KML", overwrite)
    with tempfile.TemporaryDirectory(prefix="geolattice-kmz-") as folder:
        kml_path = Path(folder) / "doc.kml"
        write_vector_data(
            kml_path,
            data,
            layer=layer,
            driver="KML",
            crs=crs,
            encoding=encoding,
            overwrite=True,
            layer_options=layer_options,
            dataset_options=dataset_options,
        )
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(kml_path, "doc.kml")
    return str(destination)


def _write_ogr(
    destination: Path,
    layers: Sequence[Tuple[str, List[Tuple[Any, Dict[str, Any]]], Any]],
    driver_name: str,
    encoding: Optional[str],
    layer_options: Dict[str, str],
    dataset_options: Dict[str, str],
) -> None:
    _, ogr, osr = _gdal_modules()
    driver = ogr.GetDriverByName(driver_name)
    if not _driver_can_create(driver):
        raise DriverUnavailableError(
            f"GDAL driver {driver_name!r} cannot create datasets in this build."
        )
    options = _dataset_options(driver_name, destination.suffix.lower(), dataset_options)
    try:
        dataset = driver.CreateDataSource(str(destination), options=options)
    except Exception as exc:
        raise VectorWriteError(
            f"Could not create {driver_name} dataset {destination}: {exc}"
        ) from exc
    if dataset is None:
        raise VectorWriteError(f"Could not create {driver_name} dataset {destination}")
    try:
        for name, records, layer_crs in layers:
            _create_layer(
                ogr,
                osr,
                dataset,
                driver_name,
                name,
                records,
                layer_crs,
                encoding,
                layer_options,
            )
    finally:
        dataset.FlushCache()
        dataset = None


def _create_layer(
    ogr: Any,
    osr: Any,
    dataset: Any,
    driver_name: str,
    name: str,
    records: Sequence[Tuple[Any, Dict[str, Any]]],
    layer_crs: Any,
    encoding: Optional[str],
    user_options: Dict[str, str],
) -> None:
    geom_code = _layer_geom_type(ogr, records)
    layer_name = _gpx_layer_name(driver_name, name, geom_code, ogr)
    spatial_ref = _spatial_reference(osr, layer_crs)
    options = _layer_creation_options(driver_name, encoding, user_options)
    try:
        ogr_layer = dataset.CreateLayer(layer_name, spatial_ref, geom_code, options)
    except Exception as exc:
        raise VectorWriteError(
            f"Driver {driver_name!r} could not create layer {layer_name!r}: {exc}"
        ) from exc
    if ogr_layer is None:
        raise VectorWriteError(
            f"Driver {driver_name!r} could not create layer {layer_name!r}."
        )
    fields = _field_plan(ogr, records, driver_name)
    for spec in fields:
        definition = ogr.FieldDefn(spec["ogr_name"], spec["type"])
        if spec["width"]:
            definition.SetWidth(spec["width"])
        subtype = spec["subtype"]
        if subtype is not None and hasattr(definition, "SetSubType"):
            definition.SetSubType(subtype)
        ogr_layer.CreateField(definition)
    definition = ogr_layer.GetLayerDefn()
    for geometry, properties in records:
        feature = ogr.Feature(definition)
        ogr_geometry = _ogr_geometry(ogr, geometry)
        if ogr_geometry is not None:
            feature.SetGeometry(ogr_geometry)
        for spec in fields:
            value = _field_value(properties.get(spec["name"]), spec, driver_name)
            if value is None:
                continue
            feature.SetField(spec["ogr_name"], value)
        ogr_layer.CreateFeature(feature)
        feature = None


def _collect_layers(
    data: Any, layer_name: str, crs: Any
) -> List[Tuple[str, List[Tuple[Any, Dict[str, Any]]], Any]]:
    if isinstance(data, VectorDataset):
        layers = [
            (
                item.name if layer_name == "layer" or len(data.layers) != 1 else layer_name,
                _records_from_iterable(item.features()),
                _first_crs(crs, item.crs),
            )
            for item in data.layers
        ]
        if len(layers) == 1 and layer_name != "layer":
            name, records, layer_crs = layers[0]
            return [(layer_name, records, layer_crs)]
        return [(name, records, layer_crs) for name, records, layer_crs in layers]
    if isinstance(data, VectorLayer):
        name = data.name if layer_name == "layer" else layer_name
        return [(name, _records_from_iterable(data.features()), _first_crs(crs, data.crs))]
    if _is_dataframe(data):
        return [(layer_name, _records_from_frame(data), _first_crs(crs, getattr(data, "crs", None)))]
    if isinstance(data, dict) and _is_feature_collection(data):
        return [(layer_name, _records_from_iterable(data.get("features") or []), crs)]
    if isinstance(data, dict) and _is_layer_mapping(data):
        layers = []
        for name, value in data.items():
            collected = _collect_layers(value, "layer", crs)
            if len(collected) != 1:
                raise VectorWriteError(
                    f"Layer {name!r} contains more than one layer. Write that dataset on its own."
                )
            _, records, layer_crs = collected[0]
            layers.append((str(name), records, layer_crs))
        return layers
    if isinstance(data, VectorFeature) or _is_feature_mapping(data):
        return [(layer_name, _records_from_iterable([data]), crs)]
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return [(layer_name, _records_from_iterable(data), crs)]
    raise TypeError(
        "data must be a VectorDataset, VectorLayer, DataFrame, feature, "
        f"feature sequence, or layer mapping, got {type(data).__name__}"
    )


def _records_from_iterable(items: Iterable[Any]) -> List[Tuple[Any, Dict[str, Any]]]:
    records: List[Tuple[Any, Dict[str, Any]]] = []
    for item in items:
        geometry, properties = _record_from_item(item)
        records.append((geometry, properties))
    return records


def _records_from_frame(frame: Any) -> List[Tuple[Any, Dict[str, Any]]]:
    geometry_column = _frame_geometry_column(frame)
    records: List[Tuple[Any, Dict[str, Any]]] = []
    for _, row in frame.iterrows():
        properties = {}
        for key, value in row.to_dict().items():
            if _missing(value):
                properties[str(key)] = None
            else:
                properties[str(key)] = value
        raw_geometry = properties.pop(geometry_column, None) if geometry_column else None
        geometry, properties = _split_geometry(raw_geometry, properties)
        records.append((geometry, properties))
    return records


def _record_from_item(item: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(item, VectorFeature):
        return _as_shapely(item.geometry), _clean_properties(item.properties)
    if isinstance(item, dict):
        if str(item.get("type", "")).lower() == "feature":
            properties = _clean_properties(item.get("properties") or {})
            return _as_shapely(item.get("geometry")), properties
        copied = dict(item)
        raw = _pop_named_geometry(copied)
        return _split_geometry(raw, _clean_properties(copied))
    if _is_geometry(item):
        return item, {}
    if isinstance(item, str):
        return _as_shapely(item), {}
    raise TypeError(f"Cannot write feature of type {type(item).__name__}")


def _pop_named_geometry(properties: Dict[str, Any]) -> Any:
    folded = {key.lower(): key for key in properties}
    for name in _GEOM_NAMES:
        original = folded.get(name)
        if original is not None:
            return properties.pop(original)
    return None


def _split_geometry(raw: Any, properties: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    geometry = _as_shapely(raw) if raw is not None else None
    if geometry is None:
        geometry = _point_from_properties(properties)
    return geometry, properties


def _point_from_properties(properties: Dict[str, Any]) -> Any:
    folded = {key.lower(): key for key in properties}
    x_name = next((folded[name] for name in _X_NAMES if name in folded), None)
    y_name = next((folded[name] for name in _Y_NAMES if name in folded), None)
    if x_name is None or y_name is None:
        return None
    try:
        x_value = float(properties[x_name])
        y_value = float(properties[y_name])
    except (TypeError, ValueError):
        return None
    if math.isnan(x_value) or math.isnan(y_value):
        return None
    return Point(x_value, y_value)


def _as_shapely(value: Any) -> Any:
    if value is None or _missing(value):
        return None
    if _is_geometry(value):
        return None if value.is_empty else value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.upper().startswith("SRID="):
            _, separator, remainder = text.partition(";")
            text = remainder.strip() if separator else text
        return _load_wkt(text)
    if isinstance(value, dict):
        return _shape(value)
    raise VectorWriteError(f"Unsupported geometry value: {type(value).__name__}")


def _clean_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in properties.items():
        if _missing(value):
            cleaned[str(key)] = None
        elif _is_geometry(value):
            continue
        else:
            cleaned[str(key)] = value
    return cleaned


def _field_plan(ogr: Any, records: Sequence[Tuple[Any, Dict[str, Any]]], driver_name: str) -> List[Dict[str, Any]]:
    names: List[str] = []
    seen = set()
    values: Dict[str, List[Any]] = {}
    for _, properties in records:
        for name, value in properties.items():
            if name not in seen:
                seen.add(name)
                names.append(name)
                values[name] = []
            if not _missing(value):
                values[name].append(value)
    used_ogr_names: set = set()
    plan: List[Dict[str, Any]] = []
    shapefile = driver_name == "ESRI Shapefile"
    for name in names:
        ogr_type, subtype = _ogr_field_type(ogr, values[name])
        width = 0
        if shapefile and ogr_type == ogr.OFTString:
            longest = max((len(_as_text(item)) for item in values[name]), default=1)
            width = min(254, max(longest, 1))
        plan.append(
            {
                "name": name,
                "ogr_name": _ogr_field_name(name, used_ogr_names, shapefile),
                "type": ogr_type,
                "subtype": subtype,
                "width": width,
            }
        )
    return plan


def _ogr_field_type(ogr: Any, values: Sequence[Any]) -> Tuple[int, Optional[int]]:
    if not values:
        return ogr.OFTString, None
    if all(isinstance(value, bool) for value in values):
        return ogr.OFTInteger, getattr(ogr, "OFSTBoolean", None)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        if any(value < -2147483648 or value > 2147483647 for value in values):
            integer64 = getattr(ogr, "OFTInteger64", ogr.OFTString)
            return integer64, None
        return ogr.OFTInteger, None
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return ogr.OFTReal, None
    return ogr.OFTString, None


def _ogr_field_name(name: str, used: set, shapefile: bool) -> str:
    if not shapefile:
        if name in used:
            raise VectorWriteError(f"Duplicate field name {name!r}")
        used.add(name)
        return name
    cleaned = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    cleaned = cleaned[:10] or "field"
    if cleaned[0].isdigit():
        cleaned = ("f_" + cleaned)[:10]
    candidate = cleaned
    index = 1
    while candidate.upper() in used:
        suffix = str(index)
        candidate = cleaned[: 10 - len(suffix)] + suffix
        index += 1
    used.add(candidate.upper())
    return candidate


def _field_value(value: Any, spec: Dict[str, Any], driver_name: str) -> Any:
    if _missing(value):
        return None
    ogr_type_name = spec["type"]
    _, ogr, _ = _gdal_modules()
    if ogr_type_name == ogr.OFTInteger and isinstance(value, bool):
        return int(value)
    if ogr_type_name in (ogr.OFTInteger, getattr(ogr, "OFTInteger64", -1)):
        return int(value)
    if ogr_type_name == ogr.OFTReal:
        return float(value)
    text = _as_text(value)
    if driver_name == "ESRI Shapefile" and spec["width"]:
        return text[: spec["width"]]
    return text


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _layer_geom_type(ogr: Any, records: Sequence[Tuple[Any, Dict[str, Any]]]) -> int:
    kinds = set()
    has_z = False
    for geometry, _ in records:
        if geometry is None or geometry.is_empty:
            continue
        kinds.add(geometry.geom_type)
        has_z = has_z or bool(getattr(geometry, "has_z", False))
    if len(kinds) != 1:
        return ogr.wkbUnknown
    kind = kinds.pop()
    table = {
        "Point": ("wkbPoint25D", "wkbPoint"),
        "LineString": ("wkbLineString25D", "wkbLineString"),
        "Polygon": ("wkbPolygon25D", "wkbPolygon"),
        "MultiPoint": ("wkbMultiPoint25D", "wkbMultiPoint"),
        "MultiLineString": ("wkbMultiLineString25D", "wkbMultiLineString"),
        "MultiPolygon": ("wkbMultiPolygon25D", "wkbMultiPolygon"),
        "GeometryCollection": ("wkbGeometryCollection25D", "wkbGeometryCollection"),
    }
    z_name, flat_name = table.get(kind, (None, None))
    if z_name is None:
        return ogr.wkbUnknown
    if has_z and hasattr(ogr, z_name):
        return getattr(ogr, z_name)
    return getattr(ogr, flat_name)


def _ogr_geometry(ogr: Any, geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return None
    created = ogr.CreateGeometryFromWkb(bytes(geometry.wkb))
    if created is None:
        raise VectorWriteError(f"Could not convert {geometry.geom_type} geometry to OGR")
    return created


def _spatial_reference(osr: Any, crs_value: Any) -> Any:
    if crs_value is None:
        return None
    if isinstance(crs_value, CRS):
        wkt = crs_value.wkt
    elif hasattr(crs_value, "to_wkt"):
        wkt = crs_value.to_wkt()
    else:
        wkt = CRS(crs_value).wkt
    if not wkt:
        raise VectorWriteError(f"CRS {crs_value!r} has no WKT definition to write")
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(wkt)
    if hasattr(spatial_ref, "SetAxisMappingStrategy"):
        spatial_ref.SetAxisMappingStrategy(getattr(osr, "OAMS_TRADITIONAL_GIS_ORDER", 0))
    return spatial_ref


def _choose_driver(path: Path, explicit: Optional[str]) -> str:
    _, ogr, _ = _gdal_modules()
    if explicit:
        driver = ogr.GetDriverByName(explicit)
        if not _driver_can_create(driver):
            raise DriverUnavailableError(
                f"GDAL driver {explicit!r} is not available or cannot create datasets."
            )
        return explicit
    suffix = path.suffix.lower()
    candidates: List[str] = []
    for fmt in VECTOR_FORMATS:
        if suffix in fmt.extensions:
            candidates.extend(fmt.drivers)
    if not candidates:
        known = ", ".join(
            extension for fmt in VECTOR_FORMATS for extension in fmt.extensions
        )
        raise VectorWriteError(
            f"No vector writer is registered for {path.name!r}. Known extensions: {known}"
        )
    for name in candidates:
        if _driver_can_create(ogr.GetDriverByName(name)):
            return name
    raise DriverUnavailableError(
        f"None of the GDAL drivers for {path.name!r} can create a dataset: {', '.join(candidates)}"
    )


def _driver_can_create(driver: Any) -> bool:
    return driver is not None and driver.GetMetadataItem("DCAP_CREATE") == "YES"


def _prepare_destination(path: Path, driver_name: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if not exists and driver_name == "ESRI Shapefile":
        exists = any(path.with_suffix(suffix).exists() for suffix in _SHAPE_SIDECARS)
    if not exists:
        return
    if not overwrite:
        raise FileExistsError(f"Dataset already exists: {path}. Pass overwrite=True to replace it.")
    _, ogr, _ = _gdal_modules()
    driver = ogr.GetDriverByName(driver_name if path.suffix.lower() != ".kmz" else "LIBKML")
    try:
        if driver is not None and path.exists():
            driver.DeleteDataSource(str(path))
    except Exception:
        pass
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    if driver_name == "ESRI Shapefile":
        for suffix in _SHAPE_SIDECARS:
            sidecar = path.with_suffix(suffix)
            if sidecar.exists() and sidecar != path:
                sidecar.unlink()


def _dataset_options(driver_name: str, suffix: str, user: Dict[str, str]) -> List[str]:
    options: Dict[str, str] = {}
    if driver_name == "SQLite":
        options["SPATIALITE"] = "YES"
    if driver_name == "MapInfo File":
        options["FORMAT"] = "MIF" if suffix == ".mif" else "TAB"
    options.update(user)
    return [f"{key}={value}" for key, value in options.items()]


def _layer_creation_options(
    driver_name: str, encoding: Optional[str], user: Dict[str, str]
) -> List[str]:
    options: Dict[str, str] = {}
    if driver_name == "GPKG":
        options["SPATIAL_INDEX"] = "YES"
    elif driver_name == "FlatGeobuf":
        options["SPATIAL_INDEX"] = "YES"
    elif driver_name == "CSV":
        options["GEOMETRY"] = "AS_WKT"
        options["CREATE_CSVT"] = "YES"
        options["SEPARATOR"] = "COMMA"
    elif driver_name == "ESRI Shapefile" and encoding:
        options["ENCODING"] = encoding
    elif driver_name == "GML":
        options["FORMAT"] = "GML3"
    options.update(user)
    return [f"{key}={value}" for key, value in options.items()]


def _gpx_layer_name(driver_name: str, name: str, geom_code: int, ogr: Any) -> str:
    if driver_name != "GPX" or name in _GPX_LAYERS:
        return name
    point_codes = {
        ogr.wkbPoint,
        ogr.wkbMultiPoint,
        getattr(ogr, "wkbPoint25D", -1),
        getattr(ogr, "wkbMultiPoint25D", -1),
    }
    if geom_code in point_codes:
        return "waypoints"
    return "tracks"


def _option_map(options: Optional[Union[Sequence[str], Dict[str, Any]]]) -> Dict[str, str]:
    if options is None:
        return {}
    if isinstance(options, dict):
        return {str(key): str(value) for key, value in options.items()}
    mapped: Dict[str, str] = {}
    for item in options:
        text = str(item)
        if "=" not in text:
            raise ValueError(f"creation option must be KEY=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        mapped[key] = value
    return mapped


def _first_crs(explicit: Any, attached: Any) -> Any:
    return explicit if explicit is not None else attached


def _is_dataframe(value: Any) -> bool:
    return hasattr(value, "iterrows") and hasattr(value, "columns")


def _is_feature_collection(value: Dict[str, Any]) -> bool:
    return str(value.get("type", "")).lower() == "featurecollection"


def _is_feature_mapping(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if str(value.get("type", "")).lower() == "feature":
        return True
    return any(name in {key.lower() for key in value} for name in _GEOM_NAMES)


def _is_layer_mapping(value: Dict[str, Any]) -> bool:
    if _is_feature_mapping(value) or _is_feature_collection(value):
        return False
    return True


def _frame_geometry_column(frame: Any) -> Optional[str]:
    geometry = getattr(frame, "geometry", None)
    name = getattr(geometry, "name", None)
    if isinstance(name, str) and name in list(frame.columns):
        return name
    folded = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in _GEOM_NAMES:
        if candidate in folded:
            return folded[candidate]
    return None


def _is_geometry(value: Any) -> bool:
    return hasattr(value, "geom_type") and hasattr(value, "wkb") and hasattr(value, "is_empty")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return isinstance(value, float) and math.isnan(value)


__all__ = [
    "VectorWriteError",
    "write_vector_data",
    "write_shapefile",
    "write_geopackage",
    "write_geojson",
    "write_csv",
    "supported_write_formats",
]
