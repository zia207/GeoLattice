"""
Multidimensional raster data management and analysis.

Open-source counterparts to ArcGIS Multidimension / Image Analyst tools,
built on ``xarray`` (NetCDF, GRIB, HDF, Zarr, OPeNDAP) with NumPy / SciPy
for aggregation, anomalies, trends, change detection, and PCA.

Typical workflow
----------------
Make Multidimensional Raster Layer → Subset → Build Multidimensional
Transpose → Aggregate → Anomaly or Trend (or CCDC / LandTrendr) →
Zonal Statistics as Table → Space Time Cube.

Recommended on-disk formats for analysis results are NetCDF (``.nc``) or
Zarr (``.zarr``). Esri Cloud Raster Format (``.crf``) is ArcGIS-oriented;
``copy_multidimensional_raster`` converts between supported open formats.

Requires optional extras::

    pip install "geolattice[multidim]"
"""

from __future__ import annotations

import importlib
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from geolattice.raster.basic_raster_operation import (
    RasterOperationError,
    _prepare_destination,
)

PathLike = Union[str, Path]
ArrayLike = Any  # xarray.DataArray | xarray.Dataset | MultidimensionalRaster
DimensionSpec = Union[str, Mapping[str, Any], Sequence[Any]]

_SPATIAL_DIM_CANDIDATES = (
    ("y", "x"),
    ("lat", "lon"),
    ("latitude", "longitude"),
    ("northing", "easting"),
    ("row", "col"),
)
_TIME_DIM_CANDIDATES = ("time", "StdTime", "t", "date", "datetime")


class MultidimensionalRasterError(RasterOperationError):
    """Raised when a multidimensional raster operation cannot be completed."""


# ---------------------------------------------------------------------------
# Optional dependency helpers
# ---------------------------------------------------------------------------


def _require_xarray() -> Any:
    try:
        return importlib.import_module("xarray")
    except ImportError as exc:
        raise MultidimensionalRasterError(
            "xarray is required for multidimensional raster tools. "
            'Install with: pip install "geolattice[multidim]"'
        ) from exc


def _require_scipy_stats() -> Any:
    try:
        from scipy import stats
    except ImportError as exc:
        raise MultidimensionalRasterError(
            "scipy is required for trend and change-detection tools."
        ) from exc
    return stats


# ---------------------------------------------------------------------------
# Core object (arcpy.Raster(..., is_multidimensional=True) style)
# ---------------------------------------------------------------------------


class MultidimensionalRaster:
    """
    In-memory multidimensional raster backed by an ``xarray.Dataset``.

    Mirrors the ArcPy multidimensional ``Raster`` object: expose variables,
    dimension names / values, and support slice-wise map algebra via the
    underlying DataArrays.
    """

    def __init__(self, data: Any) -> None:
        xr = _require_xarray()
        if isinstance(data, MultidimensionalRaster):
            self._ds = data._ds
        elif isinstance(data, xr.Dataset):
            self._ds = data
        elif isinstance(data, xr.DataArray):
            name = data.name or "variable"
            self._ds = data.to_dataset(name=name)
        else:
            raise MultidimensionalRasterError(
                "MultidimensionalRaster expects an xarray Dataset/DataArray "
                f"or MultidimensionalRaster; got {type(data)!r}."
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: PathLike,
        *,
        engine: Optional[str] = None,
        chunks: Optional[Union[str, Mapping[str, Any]]] = None,
        decode_times: bool = True,
        **open_kwargs: Any,
    ) -> "MultidimensionalRaster":
        """Open NetCDF / Zarr / GRIB / HDF / OPeNDAP into a multidimensional raster."""
        return make_multidimensional_raster_layer(
            path,
            engine=engine,
            chunks=chunks,
            decode_times=decode_times,
            **open_kwargs,
        )

    # -- properties ---------------------------------------------------------

    @property
    def dataset(self) -> Any:
        """Underlying ``xarray.Dataset``."""
        return self._ds

    @property
    def variables(self) -> List[str]:
        """Data variable names (excluding coordinates)."""
        return list(self._ds.data_vars)

    def get_dimension_names(self, variable: Optional[str] = None) -> List[str]:
        """Dimension names for the dataset or a single variable."""
        if variable is None:
            return list(self._ds.dims)
        return list(self._ds[variable].dims)

    def get_dimension_values(
        self,
        dimension: str,
        variable: Optional[str] = None,
    ) -> np.ndarray:
        """Coordinate values along ``dimension``."""
        if variable is not None and dimension in self._ds[variable].coords:
            return np.asarray(self._ds[variable].coords[dimension].values)
        if dimension not in self._ds.coords and dimension not in self._ds.dims:
            raise MultidimensionalRasterError(
                f"Dimension {dimension!r} not found."
            )
        return np.asarray(self._ds[dimension].values)

    def get_dimension_attributes(
        self,
        dimension: str,
    ) -> Dict[str, Any]:
        """Attributes on a dimension coordinate."""
        if dimension not in self._ds.coords:
            return {}
        return dict(self._ds[dimension].attrs)

    def __getitem__(self, key: Any) -> Any:
        return self._ds[key]

    def __repr__(self) -> str:
        vars_ = ", ".join(self.variables) or "(none)"
        dims = ", ".join(f"{k}: {v}" for k, v in self._ds.sizes.items())
        return f"MultidimensionalRaster(variables=[{vars_}], dims={{{dims}}})"

    def sel(self, **indexers: Any) -> "MultidimensionalRaster":
        return MultidimensionalRaster(self._ds.sel(**indexers))

    def isel(self, **indexers: Any) -> "MultidimensionalRaster":
        return MultidimensionalRaster(self._ds.isel(**indexers))

    def to_netcdf(self, path: PathLike, **kwargs: Any) -> str:
        return raster_to_netcdf(self, path, **kwargs)

    def to_zarr(self, path: PathLike, **kwargs: Any) -> str:
        return _write_dataset(self._ds, path, engine="zarr", **kwargs)


# ---------------------------------------------------------------------------
# 1. Data management
# ---------------------------------------------------------------------------


def make_multidimensional_raster_layer(
    source: PathLike,
    *,
    variables: Optional[Sequence[str]] = None,
    dimension_ranges: Optional[Mapping[str, DimensionSpec]] = None,
    engine: Optional[str] = None,
    chunks: Optional[Union[str, Mapping[str, Any]]] = None,
    decode_times: bool = True,
    group: Optional[str] = None,
    **open_kwargs: Any,
) -> MultidimensionalRaster:
    """
    Create a multidimensional raster layer from NetCDF, GRIB, HDF, Zarr, or
    OPeNDAP (ArcGIS *Make Multidimensional Raster Layer*).

    ``dimension_ranges`` maps dimension names to a slice, list of values, or
    ``{"min": ..., "max": ...}`` / ``{"start": ..., "end": ...}`` dict.
    """
    xr = _require_xarray()
    path = os.fspath(source)
    engine = engine or _infer_engine(path)

    open_kwargs = dict(open_kwargs)
    if group is not None:
        open_kwargs["group"] = group
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    try:
        if engine == "zarr":
            ds = xr.open_zarr(path, **open_kwargs)
        elif engine == "opendap":
            ds = xr.open_dataset(path, decode_times=decode_times, **open_kwargs)
        else:
            open_args = dict(open_kwargs)
            if engine is not None:
                open_args["engine"] = engine
            ds = xr.open_dataset(path, decode_times=decode_times, **open_args)
    except Exception as exc:
        raise MultidimensionalRasterError(
            f"Could not open multidimensional source {path!r} "
            f"(engine={engine!r}): {exc}"
        ) from exc

    if variables is not None:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise MultidimensionalRasterError(
                f"Variables not found: {missing}. Available: {list(ds.data_vars)}"
            )
        ds = ds[list(variables)]

    mdr = MultidimensionalRaster(ds)
    if dimension_ranges:
        mdr = subset_multidimensional_raster(
            mdr,
            dimension_definitions=dimension_ranges,
        )
    return mdr


def make_netcdf_raster_layer(
    source: PathLike,
    variable: str,
    *,
    x_variable: Optional[str] = None,
    y_variable: Optional[str] = None,
    band_dimension: Optional[str] = None,
    dimension_values: Optional[Mapping[str, Any]] = None,
    **open_kwargs: Any,
) -> MultidimensionalRaster:
    """
    Read a NetCDF variable as a raster layer
    (ArcGIS *Make NetCDF Raster Layer*).
    """
    mdr = make_multidimensional_raster_layer(
        source,
        variables=[variable],
        engine=open_kwargs.pop("engine", None),
        **open_kwargs,
    )
    ds = mdr.dataset
    da = ds[variable]

    rename: Dict[str, str] = {}
    if x_variable and x_variable in da.dims:
        rename[x_variable] = "x"
    if y_variable and y_variable in da.dims:
        rename[y_variable] = "y"
    if rename:
        da = da.rename(rename)
        ds = da.to_dataset(name=variable)

    if band_dimension and band_dimension in ds.dims and dimension_values:
        if band_dimension in dimension_values:
            ds = ds.sel({band_dimension: dimension_values[band_dimension]})
    elif dimension_values:
        ds = ds.sel(dimension_values)

    return MultidimensionalRaster(ds)


def make_netcdf_feature_layer(
    source: PathLike,
    *,
    x_variable: str,
    y_variable: str,
    variables: Optional[Sequence[str]] = None,
    row_dimension: Optional[str] = None,
    **open_kwargs: Any,
) -> pd.DataFrame:
    """
    Read NetCDF coordinates / variables as a point feature table
    (ArcGIS *Make NetCDF Feature Layer*).

    Returns a pandas DataFrame with columns ``x``, ``y``, and selected
    variables. Convert to a GeoDataFrame externally when geometries are needed.
    """
    mdr = make_multidimensional_raster_layer(source, variables=variables, **open_kwargs)
    ds = mdr.dataset
    if x_variable not in ds and x_variable not in ds.coords:
        raise MultidimensionalRasterError(f"x_variable {x_variable!r} not found.")
    if y_variable not in ds and y_variable not in ds.coords:
        raise MultidimensionalRasterError(f"y_variable {y_variable!r} not found.")

    data_vars = list(variables) if variables is not None else list(ds.data_vars)
    frame = ds[data_vars].to_dataframe().reset_index()
    if x_variable != "x" and x_variable in frame.columns:
        frame = frame.rename(columns={x_variable: "x"})
    if y_variable != "y" and y_variable in frame.columns:
        frame = frame.rename(columns={y_variable: "y"})
    if row_dimension and row_dimension in frame.columns:
        frame = frame.sort_values(row_dimension)
    return frame


def make_netcdf_table_view(
    source: PathLike,
    *,
    variables: Optional[Sequence[str]] = None,
    row_dimension: Optional[str] = None,
    **open_kwargs: Any,
) -> pd.DataFrame:
    """
    Read NetCDF variables as a tabular view
    (ArcGIS *Make NetCDF Table View*).
    """
    mdr = make_multidimensional_raster_layer(source, variables=variables, **open_kwargs)
    ds = mdr.dataset
    data_vars = list(variables) if variables is not None else list(ds.data_vars)
    frame = ds[data_vars].to_dataframe().reset_index()
    if row_dimension and row_dimension in frame.columns:
        frame = frame.sort_values(row_dimension)
    return frame


def make_opendap_raster_layer(
    url: str,
    *,
    variables: Optional[Sequence[str]] = None,
    dimension_ranges: Optional[Mapping[str, DimensionSpec]] = None,
    chunks: Optional[Union[str, Mapping[str, Any]]] = None,
    **open_kwargs: Any,
) -> MultidimensionalRaster:
    """
    Stream a multidimensional raster from an OPeNDAP / THREDDS server
    (ArcGIS *Make OPeNDAP Raster Layer*).
    """
    return make_multidimensional_raster_layer(
        url,
        variables=variables,
        dimension_ranges=dimension_ranges,
        engine="opendap",
        chunks=chunks if chunks is not None else "auto",
        **open_kwargs,
    )


def subset_multidimensional_raster(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variables: Optional[Sequence[str]] = None,
    dimension_definitions: Optional[Mapping[str, DimensionSpec]] = None,
    extent: Optional[Sequence[float]] = None,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Subset by variable, dimension values / ranges, and/or spatial extent
    (ArcGIS *Subset Multidimensional Raster*).

    ``extent`` is ``(xmin, ymin, xmax, ymax)`` in the spatial coordinate
    system of the dataset.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset

    if variables is not None:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise MultidimensionalRasterError(
                f"Variables not found: {missing}. Available: {list(ds.data_vars)}"
            )
        ds = ds[list(variables)]

    if dimension_definitions:
        ds = _apply_dimension_definitions(ds, dimension_definitions)

    if extent is not None:
        ds = _clip_extent(ds, extent)

    result = MultidimensionalRaster(ds)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def merge_multidimensional_rasters(
    sources: Sequence[Union[PathLike, MultidimensionalRaster, Any]],
    *,
    dimension: str = "time",
    join: str = "outer",
    combine: str = "by_coords",
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Combine multidimensional rasters along a dimension
    (ArcGIS *Merge Multidimensional Rasters*).

    ``combine`` is ``"by_coords"`` (``xarray.combine_by_coords``) or
    ``"nested"`` (``xarray.concat`` along ``dimension``).
    """
    xr = _require_xarray()
    if not sources:
        raise MultidimensionalRasterError("merge requires at least one source.")

    datasets = [_as_mdr(src).dataset for src in sources]
    combine_key = str(combine).strip().lower()
    try:
        if combine_key in {"nested", "concat"}:
            merged = xr.concat(datasets, dim=dimension, join=join)
        else:
            merged = xr.combine_by_coords(datasets, join=join, combine_attrs="override")
    except Exception as exc:
        raise MultidimensionalRasterError(f"Merge failed: {exc}") from exc

    result = MultidimensionalRaster(merged)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def build_multidimensional_info(
    sources: Sequence[PathLike],
    *,
    variable_name: str = "value",
    dimension_name: str = "time",
    dimension_values: Optional[Sequence[Any]] = None,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Build a multidimensional dataset from a mosaic / file series
    (ArcGIS *Build Multidimensional Info*).

    Each input file is treated as one slice along ``dimension_name``. When
    ``dimension_values`` is omitted, values are ``0 .. N-1`` (or parsed from
    filenames when they look like ISO dates).
    """
    xr = _require_xarray()
    from geolattice.raster.raster_read_write import open_raster_data

    if not sources:
        raise MultidimensionalRasterError("build_multidimensional_info requires sources.")

    arrays: List[Any] = []
    spatial_coords: Dict[str, Any] = {}
    for path in sources:
        with open_raster_data(path) as dataset:
            data = np.asarray(dataset.read(), dtype=np.float64)
            if data.ndim == 3:
                data = data[0]
            gt = dataset.geotransform
            height, width = data.shape
            xs = gt[0] + (np.arange(width) + 0.5) * gt[1]
            ys = gt[3] + (np.arange(height) + 0.5) * gt[5]
            spatial_coords = {"y": ys, "x": xs}
            nodata = dataset.get_band(1).nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            arrays.append(data)

    stack = np.stack(arrays, axis=0)
    if dimension_values is None:
        dimension_values = _infer_dimension_values_from_paths(sources)
    if len(dimension_values) != stack.shape[0]:
        raise MultidimensionalRasterError(
            "dimension_values length must match the number of source files."
        )

    da = xr.DataArray(
        stack,
        dims=(dimension_name, "y", "x"),
        coords={
            dimension_name: list(dimension_values),
            "y": spatial_coords["y"],
            "x": spatial_coords["x"],
        },
        name=variable_name,
    )
    result = MultidimensionalRaster(da)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def build_multidimensional_transpose(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    destination: Optional[PathLike] = None,
    chunks: Optional[Mapping[str, int]] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Restructure storage for fast per-pixel time-series reads
    (ArcGIS *Build Multidimensional Transpose*).

    Reorders dimensions to ``(y, x, <time_or_dimension>)`` and optionally
    rechunks so the non-spatial dimension is contiguous per pixel. Writing
    to Zarr with appropriate chunks is recommended for large series.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    y_dim, x_dim = _spatial_dims(ds[var])

    da = ds[var].transpose(y_dim, x_dim, dim, missing_dims="ignore")
    if chunks is None:
        # Favour full time series per spatial chunk.
        chunks = {y_dim: min(256, ds.sizes[y_dim]), x_dim: min(256, ds.sizes[x_dim]), dim: -1}
    try:
        da = da.chunk(chunks)
    except Exception:
        # Chunking requires dask; keep transpose without chunks.
        pass

    out = da.to_dataset(name=var)
    # Preserve other variables unchanged when present.
    for name in ds.data_vars:
        if name != var:
            out[name] = ds[name]

    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    engine = "zarr" if str(destination).lower().endswith(".zarr") else None
    return _write_mdr(result, destination, overwrite=overwrite, engine=engine)


def raster_to_netcdf(
    source: Union[PathLike, MultidimensionalRaster, Any],
    destination: PathLike,
    *,
    variable_name: Optional[str] = None,
    overwrite: bool = False,
    **to_netcdf_kwargs: Any,
) -> str:
    """Export a multidimensional raster to NetCDF (ArcGIS *Raster to NetCDF*)."""
    mdr = _as_mdr(source)
    ds = mdr.dataset
    if variable_name is not None and len(ds.data_vars) == 1:
        only = list(ds.data_vars)[0]
        if only != variable_name:
            ds = ds.rename({only: variable_name})
    engine = _available_netcdf_engine() or "netcdf4"
    return _write_dataset(ds, destination, engine=engine, overwrite=overwrite, **to_netcdf_kwargs)


def feature_to_netcdf(
    frame: Union[pd.DataFrame, Any],
    destination: PathLike,
    *,
    x_field: str = "x",
    y_field: str = "y",
    variables: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> str:
    """Export point features / a DataFrame to NetCDF (ArcGIS *Feature to NetCDF*)."""
    xr = _require_xarray()
    if hasattr(frame, "to_dataframe"):
        # GeoDataFrame-like
        try:
            df = frame.drop(columns="geometry", errors="ignore").copy()
        except Exception:
            df = pd.DataFrame(frame)
    else:
        df = pd.DataFrame(frame)

    for field in (x_field, y_field):
        if field not in df.columns:
            raise MultidimensionalRasterError(f"Field {field!r} not found in table.")

    value_cols = (
        list(variables)
        if variables is not None
        else [c for c in df.columns if c not in {x_field, y_field}]
    )
    ds = df.set_index([y_field, x_field])[value_cols].to_xarray()
    if y_field != "y" or x_field != "x":
        rename = {}
        if y_field in ds.dims:
            rename[y_field] = "y"
        if x_field in ds.dims:
            rename[x_field] = "x"
        if rename:
            ds = ds.rename(rename)
    engine = _available_netcdf_engine() or "netcdf4"
    return _write_dataset(ds, destination, engine=engine, overwrite=overwrite)


def table_to_netcdf(
    frame: Union[pd.DataFrame, Mapping[str, Sequence[Any]]],
    destination: PathLike,
    *,
    row_dimension: str = "record",
    overwrite: bool = False,
) -> str:
    """Export a table to NetCDF (ArcGIS *Table to NetCDF*)."""
    df = pd.DataFrame(frame)
    df.index.name = row_dimension
    ds = df.to_xarray()
    engine = _available_netcdf_engine() or "netcdf4"
    return _write_dataset(ds, destination, engine=engine, overwrite=overwrite)


def copy_multidimensional_raster(
    source: Union[PathLike, MultidimensionalRaster, Any],
    destination: PathLike,
    *,
    format: Optional[str] = None,
    overwrite: bool = False,
    **write_kwargs: Any,
) -> str:
    """
    Copy / convert a multidimensional raster to another format.

    Open equivalents of ArcGIS CRF for analysis outputs: NetCDF (``.nc``),
    Zarr (``.zarr``), or a multi-band GeoTIFF (``.tif``) when the data are
    a single variable with one non-spatial dimension.
    """
    mdr = _as_mdr(source)
    dest = Path(destination)
    fmt = (format or _format_from_destination(dest)).lower()

    if fmt in {"crf", "cloud_raster_format"}:
        warnings.warn(
            "Esri Cloud Raster Format (.crf) is not written natively. "
            "Writing Zarr instead; use ArcGIS Copy Raster for true CRF.",
            UserWarning,
            stacklevel=2,
        )
        if dest.suffix.lower() == ".crf":
            dest = dest.with_suffix(".zarr")
        fmt = "zarr"

    if fmt in {"nc", "netcdf", "netcdf4"}:
        nc_engine = _available_netcdf_engine() or "netcdf4"
        return _write_dataset(
            mdr.dataset, dest, engine=nc_engine, overwrite=overwrite, **write_kwargs
        )
    if fmt in {"zarr"}:
        return _write_dataset(mdr.dataset, dest, engine="zarr", overwrite=overwrite, **write_kwargs)
    if fmt in {"tif", "tiff", "gtiff", "geotiff"}:
        return _write_multiband_geotiff(mdr, dest, overwrite=overwrite)
    raise MultidimensionalRasterError(
        f"Unsupported output format {fmt!r}. Use netcdf, zarr, or geotiff."
    )


# ---------------------------------------------------------------------------
# 2. Multidimensional analysis
# ---------------------------------------------------------------------------


def aggregate_multidimensional_raster(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    aggregation_method: str = "MEAN",
    interval_keyword: Optional[str] = None,
    interval_value: Optional[int] = None,
    interval_unit: Optional[str] = None,
    aggregation_definition: str = "ALL",
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Aggregate along a dimension (ArcGIS *Aggregate Multidimensional Raster*).

    ``aggregation_method``: MEAN, SUM, MIN, MAX, MEDIAN, STD, COUNT.
    ``interval_keyword``: HOURLY, DAILY, WEEKLY, MONTHLY, YEARLY / ANNUALLY,
    or seasonal aliases (DJF, MAM, JJA, SON).
    ``aggregation_definition``: ALL, INTERVAL_KEYWORD, INTERVAL_VALUE.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    method = _normalize_agg_method(aggregation_method)
    definition = str(aggregation_definition).strip().upper()

    if definition in {"INTERVAL_KEYWORD", "INTERVAL"} and interval_keyword:
        key = str(interval_keyword).strip().upper()
        seasonal = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
        if key in seasonal:
            months = seasonal[key]
            da = da.sel({dim: da[dim].dt.month.isin(months)})
            aggregated = getattr(da, method)(dim=dim, skipna=True)
            aggregated = aggregated.expand_dims({dim: [key]})
        else:
            freq = _interval_keyword_to_freq(key)
            aggregated = _resample_reduce(da, dim, freq, method)
    elif definition == "INTERVAL_VALUE":
        if interval_value is None or interval_unit is None:
            raise MultidimensionalRasterError(
                "INTERVAL_VALUE requires interval_value and interval_unit."
            )
        freq = f"{int(interval_value)}{_unit_to_offset(interval_unit)}"
        aggregated = _resample_reduce(da, dim, freq, method)
    else:
        aggregated = getattr(da, method)(dim=dim, skipna=True)

    if isinstance(aggregated, type(da)) and dim not in aggregated.dims:
        # Scalar along dimension — keep as 2-D spatial field under same variable.
        out = aggregated.to_dataset(name=var)
    else:
        out = aggregated.to_dataset(name=var)

    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def generate_multidimensional_anomaly(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    anomaly_type: str = "DIFFERENCE_FROM_MEAN",
    calculation_interval: str = "ALL",
    reference: Optional[Union[PathLike, MultidimensionalRaster, Any]] = None,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Compute anomalies relative to a mean, median, or external baseline
    (ArcGIS *Generate Multidimensional Anomaly*).

    ``anomaly_type``: DIFFERENCE_FROM_MEAN, PERCENT_DIFFERENCE_FROM_MEAN,
    PERCENT_OF_MEAN, Z_SCORE, DIFFERENCE_FROM_MEDIAN,
    PERCENT_DIFFERENCE_FROM_MEDIAN, PERCENT_OF_MEDIAN,
    DIFFERENCE_FROM_REFERENCE, PERCENT_DIFFERENCE_FROM_REFERENCE.

    ``calculation_interval``: ALL, MONTHLY / BY_MONTH, YEARLY / BY_YEAR,
    HOURLY, DAILY, WEEKLY.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var].astype(np.float64)
    atype = str(anomaly_type).strip().upper()
    interval = str(calculation_interval).strip().upper()

    if "REFERENCE" in atype:
        if reference is None:
            raise MultidimensionalRasterError(
                "Reference baseline is required for REFERENCE anomaly types."
            )
        ref_mdr = _as_mdr(reference)
        baseline = ref_mdr.dataset[var] if var in ref_mdr.dataset else ref_mdr.dataset[
            _default_variable(ref_mdr.dataset)
        ]
        base = _broadcast_baseline(da, baseline, dim=dim, interval="ALL")
    else:
        base = _broadcast_baseline(
            da,
            _interval_baseline(da, dim, interval, use_median="MEDIAN" in atype),
            dim=dim,
            interval=interval,
        )

    if atype in {"DIFFERENCE_FROM_MEAN", "DIFFERENCE_FROM_MEDIAN", "DIFFERENCE_FROM_REFERENCE"}:
        anom = da - base
    elif atype in {
        "PERCENT_DIFFERENCE_FROM_MEAN",
        "PERCENT_DIFFERENCE_FROM_MEDIAN",
        "PERCENT_DIFFERENCE_FROM_REFERENCE",
    }:
        anom = 100.0 * (da - base) / base
    elif atype in {"PERCENT_OF_MEAN", "PERCENT_OF_MEDIAN"}:
        anom = 100.0 * da / base
    elif atype == "Z_SCORE":
        if interval in {"MONTHLY", "BY_MONTH"}:
            mean = _broadcast_baseline(
                da, da.groupby(f"{dim}.month").mean(dim), dim=dim, interval=interval
            )
            std = _broadcast_baseline(
                da, da.groupby(f"{dim}.month").std(dim), dim=dim, interval=interval
            )
            anom = (da - mean) / std
        else:
            mean = da.mean(dim=dim, skipna=True)
            std = da.std(dim=dim, skipna=True)
            anom = (da - mean) / std
    else:
        raise MultidimensionalRasterError(f"Unknown anomaly_type {anomaly_type!r}.")

    result = MultidimensionalRaster(anom.to_dataset(name=var))
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def find_argument_statistics(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    statistics_type: str = "ARGUMENT_OF_MAXIMUM",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    multiple_occurrence: str = "FIRST_OCCURRENCE",
    ignore_nodata: bool = True,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Find the dimension value where a statistic occurs
    (ArcGIS *Find Argument Statistics*).

    ``statistics_type``: ARGUMENT_OF_MAXIMUM, ARGUMENT_OF_MINIMUM,
    DURATION, ARGUMENT_OF_MIN_VALUE / ARGUMENT_OF_MAX_VALUE (threshold).
    """
    xr = _require_xarray()
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    stype = str(statistics_type).strip().upper()
    multi = str(multiple_occurrence).strip().upper()

    values = da.values
    coords = np.asarray(da[dim].values)
    axis = da.get_axis_num(dim)

    if stype == "ARGUMENT_OF_MAXIMUM":
        idx = _arg_extreme(values, axis=axis, mode="max", multiple=multi, ignore_nodata=ignore_nodata)
        out_vals = _index_to_coord(coords, idx)
        name = f"{var}_arg_max"
    elif stype == "ARGUMENT_OF_MINIMUM":
        idx = _arg_extreme(values, axis=axis, mode="min", multiple=multi, ignore_nodata=ignore_nodata)
        out_vals = _index_to_coord(coords, idx)
        name = f"{var}_arg_min"
    elif stype in {"DURATION", "ARGUMENT_OF_MIN_VALUE", "ARGUMENT_OF_MAX_VALUE"}:
        if min_value is None and max_value is None:
            raise MultidimensionalRasterError(
                "DURATION / threshold argument statistics require min_value and/or max_value."
            )
        mask = np.ones(values.shape, dtype=bool)
        if min_value is not None:
            mask &= values >= float(min_value)
        if max_value is not None:
            mask &= values <= float(max_value)
        if ignore_nodata:
            mask &= np.isfinite(values)
        if stype == "DURATION":
            out_vals = np.sum(mask, axis=axis).astype(np.float64)
            name = f"{var}_duration"
        else:
            # First / last index where condition holds.
            if multi == "LAST_OCCURRENCE":
                idx = _last_true(mask, axis=axis)
            else:
                idx = _first_true(mask, axis=axis)
            out_vals = _index_to_coord(coords, idx)
            name = f"{var}_arg_threshold"
    else:
        raise MultidimensionalRasterError(f"Unknown statistics_type {statistics_type!r}.")

    out_dims = [d for d in da.dims if d != dim]
    out_coords = {d: da[d] for d in out_dims}
    out_da = xr.DataArray(out_vals, dims=out_dims, coords=out_coords, name=name)
    # Store numeric ordinal when coords are datetime for easier GIS use.
    if np.issubdtype(np.asarray(coords).dtype, np.datetime64) and stype != "DURATION":
        ordinal = _datetime_to_ordinal(out_vals)
        out_ds = xr.Dataset({name: out_da, f"{name}_ordinal": (out_dims, ordinal)})
    else:
        out_ds = out_da.to_dataset(name=name)

    result = MultidimensionalRaster(out_ds)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def generate_trend_raster(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    trend_type: str = "LINEAR",
    frequency: int = 1,
    cycle_length: Optional[float] = None,
    polynomial_order: int = 2,
    ignore_nodata: bool = True,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Per-pixel trend along a dimension (ArcGIS *Generate Trend Raster*).

    ``trend_type``: LINEAR, HARMONIC, POLYNOMIAL, MANN_KENDALL,
    SEASONAL_KENDALL.

    Returns a dataset of coefficient / statistic bands suitable for
    ``predict_using_trend_raster``.
    """
    xr = _require_xarray()
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    ttype = str(trend_type).strip().upper().replace("-", "_")

    t = _dimension_as_float(da[dim].values)
    data = np.moveaxis(np.asarray(da.values, dtype=np.float64), da.get_axis_num(dim), -1)
    spatial_shape = data.shape[:-1]
    n_time = data.shape[-1]
    flat = data.reshape(-1, n_time)

    if ttype == "LINEAR":
        coeff = _fit_polynomial_pixels(flat, t, order=1, ignore_nodata=ignore_nodata)
        names = ["intercept", "slope", "r2", "p_value"]
    elif ttype == "POLYNOMIAL":
        order = max(1, int(polynomial_order))
        coeff = _fit_polynomial_pixels(flat, t, order=order, ignore_nodata=ignore_nodata)
        names = [f"coeff_{i}" for i in range(order + 1)] + ["r2", "p_value"]
    elif ttype == "HARMONIC":
        freq = max(1, int(frequency))
        length = float(cycle_length) if cycle_length is not None else _infer_cycle_length(t)
        coeff = _fit_harmonic_pixels(flat, t, frequency=freq, cycle_length=length, ignore_nodata=ignore_nodata)
        names = ["intercept", "slope"]
        for k in range(1, freq + 1):
            names.extend([f"cos_{k}", f"sin_{k}"])
        names.extend(["r2", "rmse"])
    elif ttype == "MANN_KENDALL":
        coeff = _mann_kendall_pixels(flat, ignore_nodata=ignore_nodata)
        names = ["tau", "p_value", "slope", "intercept"]
    elif ttype == "SEASONAL_KENDALL":
        periods = max(1, int(frequency) if frequency else 12)
        coeff = _seasonal_kendall_pixels(flat, periods=periods, ignore_nodata=ignore_nodata)
        names = ["tau", "p_value", "slope", "intercept"]
    else:
        raise MultidimensionalRasterError(f"Unknown trend_type {trend_type!r}.")

    y_dim, x_dim = _spatial_dims(da)
    out_coords = {y_dim: da[y_dim], x_dim: da[x_dim]}
    data_vars = {}
    for i, name in enumerate(names):
        band = coeff[:, i].reshape(spatial_shape)
        data_vars[name] = xr.DataArray(band, dims=(y_dim, x_dim), coords=out_coords)

    out = xr.Dataset(data_vars)
    out.attrs.update(
        {
            "trend_type": ttype,
            "variable": var,
            "dimension": dim,
            "t_min": float(np.nanmin(t)),
            "t_max": float(np.nanmax(t)),
            "frequency": int(frequency),
            "polynomial_order": int(polynomial_order),
            "cycle_length": float(cycle_length) if cycle_length is not None else _infer_cycle_length(t),
        }
    )
    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def predict_using_trend_raster(
    trend_raster: Union[PathLike, MultidimensionalRaster, Any],
    *,
    dimension_values: Sequence[Any],
    dimension_name: str = "time",
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Project values at new dimension values from a trend raster
    (ArcGIS *Predict Using Trend Raster*).
    """
    xr = _require_xarray()
    trend = _as_mdr(trend_raster).dataset
    ttype = str(trend.attrs.get("trend_type", "LINEAR")).upper()
    t_vals = _dimension_as_float(np.asarray(dimension_values))

    y_dim, x_dim = _spatial_dims_from_dataset(trend)

    if ttype == "LINEAR":
        intercept = np.asarray(trend["intercept"].values, dtype=np.float64)
        slope = np.asarray(trend["slope"].values, dtype=np.float64)
        predicted = intercept[..., None] + slope[..., None] * t_vals.reshape((1,) * intercept.ndim + (-1,))
    elif ttype == "POLYNOMIAL":
        order = int(trend.attrs.get("polynomial_order", 2))
        predicted = np.zeros(trend["coeff_0"].shape + (len(t_vals),), dtype=np.float64)
        for i in range(order + 1):
            predicted += np.asarray(trend[f"coeff_{i}"].values)[..., None] * (t_vals**i)
    elif ttype == "HARMONIC":
        freq = int(trend.attrs.get("frequency", 1))
        cycle = float(trend.attrs.get("cycle_length", 1.0))
        intercept = np.asarray(trend["intercept"].values, dtype=np.float64)
        slope = np.asarray(trend["slope"].values, dtype=np.float64)
        predicted = intercept[..., None] + slope[..., None] * t_vals.reshape((1,) * intercept.ndim + (-1,))
        omega = 2.0 * math.pi / cycle
        for k in range(1, freq + 1):
            cos_c = np.asarray(trend[f"cos_{k}"].values)[..., None]
            sin_c = np.asarray(trend[f"sin_{k}"].values)[..., None]
            predicted += cos_c * np.cos(omega * k * t_vals) + sin_c * np.sin(omega * k * t_vals)
    elif ttype in {"MANN_KENDALL", "SEASONAL_KENDALL"}:
        intercept = np.asarray(trend["intercept"].values, dtype=np.float64)
        slope = np.asarray(trend["slope"].values, dtype=np.float64)
        predicted = intercept[..., None] + slope[..., None] * t_vals.reshape((1,) * intercept.ndim + (-1,))
    else:
        raise MultidimensionalRasterError(f"Cannot predict from trend_type {ttype!r}.")

    out = xr.DataArray(
        predicted,
        dims=(y_dim, x_dim, dimension_name),
        coords={
            y_dim: trend[y_dim],
            x_dim: trend[x_dim],
            dimension_name: list(dimension_values),
        },
        name=trend.attrs.get("variable", "predicted"),
    )
    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


# ---------------------------------------------------------------------------
# 3. Change detection and advanced tools
# ---------------------------------------------------------------------------


def analyze_changes_using_ccdc(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    min_observations: int = 6,
    chi_square_probability: float = 0.99,
    max_breaks: int = 3,
    harmonic_frequency: int = 1,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Continuous Change Detection and Classification style analysis
    (ArcGIS *Analyze Changes Using CCDC*).

    Fits successive harmonic models and records breakpoints when residuals
    exceed a chi-square threshold. Returns change count, first / last break
    ordinals, and magnitude.
    """
    xr = _require_xarray()
    stats = _require_scipy_stats()
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]

    t = _dimension_as_float(da[dim].values)
    data = np.moveaxis(np.asarray(da.values, dtype=np.float64), da.get_axis_num(dim), -1)
    spatial_shape = data.shape[:-1]
    flat = data.reshape(-1, data.shape[-1])
    threshold = float(stats.chi2.ppf(chi_square_probability, df=1))

    n_pix = flat.shape[0]
    change_count = np.zeros(n_pix, dtype=np.float64)
    first_break = np.full(n_pix, np.nan)
    last_break = np.full(n_pix, np.nan)
    magnitude = np.full(n_pix, np.nan)

    cycle = _infer_cycle_length(t)
    for i in range(n_pix):
        series = flat[i]
        breaks, mag = _ccdc_breaks(
            t,
            series,
            min_observations=min_observations,
            threshold=threshold,
            max_breaks=max_breaks,
            frequency=harmonic_frequency,
            cycle_length=cycle,
        )
        change_count[i] = len(breaks)
        if breaks:
            first_break[i] = breaks[0]
            last_break[i] = breaks[-1]
            magnitude[i] = mag

    y_dim, x_dim = _spatial_dims(da)
    coords = {y_dim: da[y_dim], x_dim: da[x_dim]}
    out = xr.Dataset(
        {
            "change_count": ((y_dim, x_dim), change_count.reshape(spatial_shape)),
            "first_break": ((y_dim, x_dim), first_break.reshape(spatial_shape)),
            "last_break": ((y_dim, x_dim), last_break.reshape(spatial_shape)),
            "magnitude": ((y_dim, x_dim), magnitude.reshape(spatial_shape)),
        },
        coords=coords,
        attrs={"method": "CCDC", "variable": var, "dimension": dim},
    )
    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def analyze_changes_using_landtrendr(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    min_observations: int = 6,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    LandTrendr-style temporal segmentation for disturbance and recovery
    (ArcGIS *Analyze Changes Using LandTrendr*).

    Returns year / ordinal of greatest disturbance, magnitude, duration, and
    recovery rate.
    """
    xr = _require_xarray()
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]

    t = _dimension_as_float(da[dim].values)
    data = np.moveaxis(np.asarray(da.values, dtype=np.float64), da.get_axis_num(dim), -1)
    spatial_shape = data.shape[:-1]
    flat = data.reshape(-1, data.shape[-1])

    n_pix = flat.shape[0]
    dist_time = np.full(n_pix, np.nan)
    dist_mag = np.full(n_pix, np.nan)
    dist_dur = np.full(n_pix, np.nan)
    recovery = np.full(n_pix, np.nan)

    for i in range(n_pix):
        result = _landtrendr_pixel(
            t,
            flat[i],
            max_segments=max_segments,
            spike_threshold=spike_threshold,
            recovery_threshold=recovery_threshold,
            min_observations=min_observations,
        )
        dist_time[i] = result["disturbance_time"]
        dist_mag[i] = result["magnitude"]
        dist_dur[i] = result["duration"]
        recovery[i] = result["recovery_rate"]

    y_dim, x_dim = _spatial_dims(da)
    coords = {y_dim: da[y_dim], x_dim: da[x_dim]}
    out = xr.Dataset(
        {
            "disturbance_time": ((y_dim, x_dim), dist_time.reshape(spatial_shape)),
            "magnitude": ((y_dim, x_dim), dist_mag.reshape(spatial_shape)),
            "duration": ((y_dim, x_dim), dist_dur.reshape(spatial_shape)),
            "recovery_rate": ((y_dim, x_dim), recovery.reshape(spatial_shape)),
        },
        coords=coords,
        attrs={"method": "LandTrendr", "variable": var, "dimension": dim},
    )
    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def detect_change_using_change_analysis_raster(
    change_analysis_raster: Union[PathLike, MultidimensionalRaster, Any],
    *,
    change_type: str = "TIME_OF_LATEST_CHANGE",
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Extract change date, magnitude, or count from CCDC / LandTrendr output
    (ArcGIS *Detect Change Using Change Analysis Raster*).

    ``change_type``: TIME_OF_LATEST_CHANGE, TIME_OF_EARLIEST_CHANGE,
    NUM_OF_CHANGES, CHANGE_MAGNITUDE.
    """
    mdr = _as_mdr(change_analysis_raster)
    ds = mdr.dataset
    ctype = str(change_type).strip().upper()

    mapping = {
        "TIME_OF_LATEST_CHANGE": ("last_break", "disturbance_time"),
        "TIME_OF_EARLIEST_CHANGE": ("first_break", "disturbance_time"),
        "NUM_OF_CHANGES": ("change_count",),
        "CHANGE_MAGNITUDE": ("magnitude",),
    }
    if ctype not in mapping:
        raise MultidimensionalRasterError(f"Unknown change_type {change_type!r}.")

    selected = None
    for name in mapping[ctype]:
        if name in ds:
            selected = ds[name]
            break
    if selected is None:
        raise MultidimensionalRasterError(
            f"Change analysis raster has no band for {change_type!r}. "
            f"Available: {list(ds.data_vars)}"
        )

    out_name = ctype.lower()
    result = MultidimensionalRaster(selected.to_dataset(name=out_name))
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


def multidimensional_principal_components(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    n_components: int = 3,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    PCA / EOF-style analysis across a dimension
    (ArcGIS *Multidimensional Principal Components*).

    Returns the first ``n_components`` spatial loading rasters and a
    DataFrame-friendly ``explained_variance_ratio`` attribute.
    """
    xr = _require_xarray()
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]

    data = np.moveaxis(np.asarray(da.values, dtype=np.float64), da.get_axis_num(dim), 0)
    n_time = data.shape[0]
    flat = data.reshape(n_time, -1)
    # Columns = pixels; drop all-NaN columns for SVD.
    col_mean = np.nanmean(flat, axis=0)
    centered = flat - col_mean
    centered = np.where(np.isfinite(centered), centered, 0.0)

    # Economy SVD on time × pixel matrix.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n_comp = max(1, min(int(n_components), u.shape[1]))
    loadings = vt[:n_comp, :].reshape((n_comp,) + data.shape[1:])
    explained = (s**2) / np.sum(s**2)
    scores = u[:, :n_comp] * s[:n_comp]

    y_dim, x_dim = _spatial_dims(da)
    data_vars = {}
    for i in range(n_comp):
        data_vars[f"pc_{i + 1}"] = xr.DataArray(
            loadings[i],
            dims=(y_dim, x_dim),
            coords={y_dim: da[y_dim], x_dim: da[x_dim]},
        )
    # Attach PC scores along the analysis dimension.
    data_vars["pc_scores"] = xr.DataArray(
        scores,
        dims=(dim, "component"),
        coords={dim: da[dim], "component": np.arange(1, n_comp + 1)},
    )
    out = xr.Dataset(
        data_vars,
        attrs={
            "explained_variance_ratio": explained[:n_comp].tolist(),
            "variable": var,
            "dimension": dim,
        },
    )
    result = MultidimensionalRaster(out)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


# ---------------------------------------------------------------------------
# 4. Space Time Pattern Mining bridge
# ---------------------------------------------------------------------------


def create_space_time_cube_from_multidimensional_raster(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    destination: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Union[MultidimensionalRaster, str]:
    """
    Convert a gridded series into a space-time cube structure
    (ArcGIS *Create Space Time Cube From Multidimensional Raster Layer*).

    The cube is an ``xarray.Dataset`` with dims ``(time, y, x)`` (or the
    detected spatial / temporal names), ready for hot-spot style summaries,
    clustering, or forecasting helpers. When ``destination`` ends with
    ``.nc`` / ``.zarr`` the cube is written to disk.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    y_dim, x_dim = _spatial_dims(da)
    cube = da.transpose(dim, y_dim, x_dim).to_dataset(name=var)
    cube.attrs["space_time_cube"] = True
    cube.attrs["time_dimension"] = dim
    cube.attrs["spatial_dimensions"] = [y_dim, x_dim]

    result = MultidimensionalRaster(cube)
    if destination is None:
        return result
    return _write_mdr(result, destination, overwrite=overwrite)


# ---------------------------------------------------------------------------
# 5. Raster-function equivalents and visualization helpers
# ---------------------------------------------------------------------------


def aggregate_function(
    source: Union[PathLike, MultidimensionalRaster, Any],
    **kwargs: Any,
) -> MultidimensionalRaster:
    """On-the-fly aggregate (raster-function style; no write unless destination set)."""
    result = aggregate_multidimensional_raster(source, **kwargs)
    return result if isinstance(result, MultidimensionalRaster) else MultidimensionalRaster.open(result)


def anomaly_function(
    source: Union[PathLike, MultidimensionalRaster, Any],
    **kwargs: Any,
) -> MultidimensionalRaster:
    """On-the-fly anomaly (raster-function style)."""
    result = generate_multidimensional_anomaly(source, **kwargs)
    return result if isinstance(result, MultidimensionalRaster) else MultidimensionalRaster.open(result)


def trend_function(
    source: Union[PathLike, MultidimensionalRaster, Any],
    **kwargs: Any,
) -> MultidimensionalRaster:
    """On-the-fly trend (raster-function style)."""
    result = generate_trend_raster(source, **kwargs)
    return result if isinstance(result, MultidimensionalRaster) else MultidimensionalRaster.open(result)


def multidimensional_filter(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variables: Optional[Sequence[str]] = None,
    dimension_definitions: Optional[Mapping[str, DimensionSpec]] = None,
    extent: Optional[Sequence[float]] = None,
) -> MultidimensionalRaster:
    """On-the-fly multidimensional filter (subset without writing)."""
    result = subset_multidimensional_raster(
        source,
        variables=variables,
        dimension_definitions=dimension_definitions,
        extent=extent,
    )
    assert isinstance(result, MultidimensionalRaster)
    return result


def temporal_profile(
    source: Union[PathLike, MultidimensionalRaster, Any],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    geometry: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Extract a pixel or zone time series for a Temporal Profile chart.

    Provide either map coordinates ``(x, y)``, array indices ``(row, col)``,
    or a shapely-like ``geometry`` with ``.centroid``.
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    y_dim, x_dim = _spatial_dims(da)

    if geometry is not None:
        centroid = getattr(geometry, "centroid", geometry)
        x = float(centroid.x)
        y = float(centroid.y)

    if x is not None and y is not None:
        series = da.sel({x_dim: x, y_dim: y}, method="nearest")
    elif row is not None and col is not None:
        series = da.isel({y_dim: int(row), x_dim: int(col)})
    else:
        raise MultidimensionalRasterError(
            "temporal_profile requires (x, y), (row, col), or geometry."
        )

    frame = series.to_dataframe(name=var).reset_index()
    return frame[[dim, var]] if dim in frame.columns else frame


def zonal_statistics_as_table(
    source: Union[PathLike, MultidimensionalRaster, Any],
    zones: Any,
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    zone_field: Optional[str] = None,
    stats: Sequence[str] = ("mean", "min", "max", "std", "count"),
) -> pd.DataFrame:
    """
    Per-slice zonal statistics for a multidimensional raster
    (ArcGIS *Zonal Statistics as Table* with multidimensional input).

    ``zones`` may be a 2-D numpy array of zone IDs aligned to the raster grid,
    or a path to a single-band zone raster.
    """
    from geolattice.raster.raster_read_write import open_raster_data

    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    y_dim, x_dim = _spatial_dims(da)

    if isinstance(zones, (str, Path)):
        with open_raster_data(zones) as zone_ds:
            zone_arr = np.asarray(zone_ds.read(bands=1))
    else:
        zone_arr = np.asarray(zones)

    if zone_arr.shape != (da.sizes[y_dim], da.sizes[x_dim]):
        raise MultidimensionalRasterError(
            f"Zone array shape {zone_arr.shape} does not match raster "
            f"{(da.sizes[y_dim], da.sizes[x_dim])}."
        )

    rows: List[Dict[str, Any]] = []
    zone_ids = np.unique(zone_arr)
    zone_ids = zone_ids[zone_ids != 0]
    for i, dim_val in enumerate(np.asarray(da[dim].values)):
        slice_ = np.asarray(da.isel({dim: i}).values, dtype=np.float64)
        for zid in zone_ids:
            mask = zone_arr == zid
            vals = slice_[mask]
            vals = vals[np.isfinite(vals)]
            record: Dict[str, Any] = {
                zone_field or "zone": zid,
                dim: dim_val,
                "variable": var,
            }
            for stat in stats:
                key = str(stat).lower()
                if key == "mean":
                    record["mean"] = float(np.mean(vals)) if vals.size else np.nan
                elif key == "min":
                    record["min"] = float(np.min(vals)) if vals.size else np.nan
                elif key == "max":
                    record["max"] = float(np.max(vals)) if vals.size else np.nan
                elif key in {"std", "stddev"}:
                    record["std"] = float(np.std(vals)) if vals.size else np.nan
                elif key == "sum":
                    record["sum"] = float(np.sum(vals)) if vals.size else np.nan
                elif key == "count":
                    record["count"] = int(vals.size)
                else:
                    raise MultidimensionalRasterError(f"Unknown stat {stat!r}.")
            rows.append(record)
    return pd.DataFrame(rows)


def sample_multidimensional(
    source: Union[PathLike, MultidimensionalRaster, Any],
    points: Union[pd.DataFrame, Sequence[Tuple[float, float]]],
    *,
    variable: Optional[str] = None,
    dimension: Optional[str] = None,
    x_field: str = "x",
    y_field: str = "y",
) -> pd.DataFrame:
    """
    Sample a multidimensional raster at point locations for every slice
    (ArcGIS *Sample* with multidimensional input).
    """
    mdr = _as_mdr(source)
    ds = mdr.dataset
    var = variable or _default_variable(ds)
    dim = dimension or _default_time_dimension(ds[var])
    da = ds[var]
    y_dim, x_dim = _spatial_dims(da)

    if isinstance(points, pd.DataFrame):
        xs = points[x_field].to_numpy()
        ys = points[y_field].to_numpy()
        ids = points.index.to_numpy()
    else:
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        ids = np.arange(len(points))

    rows: List[Dict[str, Any]] = []
    for pid, x, y in zip(ids, xs, ys):
        series = da.sel({x_dim: x, y_dim: y}, method="nearest")
        for dim_val, value in zip(np.asarray(series[dim].values), np.asarray(series.values)):
            rows.append(
                {
                    "point_id": pid,
                    "x": float(x),
                    "y": float(y),
                    dim: dim_val,
                    var: float(value) if np.isfinite(value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_mdr(source: Union[PathLike, MultidimensionalRaster, Any]) -> MultidimensionalRaster:
    if isinstance(source, MultidimensionalRaster):
        return source
    xr = _require_xarray()
    if isinstance(source, (xr.Dataset, xr.DataArray)):
        return MultidimensionalRaster(source)
    return make_multidimensional_raster_layer(source)


def _infer_engine(path: str) -> Optional[str]:
    lower = path.lower()
    if lower.startswith(("http://", "https://", "dods://", "opendap://")):
        return "opendap"
    if lower.endswith(".zarr") or lower.rstrip("/").endswith(".zarr"):
        return "zarr"
    if lower.endswith((".grib", ".grib2", ".grb", ".grb2")):
        return "cfgrib"
    if lower.endswith((".h5", ".hdf5", ".hdf", ".he5")):
        return "h5netcdf"
    # NetCDF: leave engine unset so xarray auto-detects netcdf4 / h5netcdf / scipy.
    return None


def _available_netcdf_engine() -> Optional[str]:
    """Prefer netcdf4, then scipy (classic), then h5netcdf for writes."""
    xr = _require_xarray()
    try:
        engines = set(xr.backends.list_engines())
    except Exception:
        engines = set()
    for candidate in ("netcdf4", "scipy", "h5netcdf"):
        if candidate in engines:
            return candidate
    return None


def _default_variable(ds: Any) -> str:
    vars_ = list(ds.data_vars)
    if not vars_:
        raise MultidimensionalRasterError("Dataset has no data variables.")
    return vars_[0]


def _default_time_dimension(da: Any) -> str:
    for name in _TIME_DIM_CANDIDATES:
        if name in da.dims:
            return name
    non_spatial = [d for d in da.dims if d not in sum(_SPATIAL_DIM_CANDIDATES, ())]
    if not non_spatial:
        raise MultidimensionalRasterError(
            "Could not detect a time / analysis dimension."
        )
    return non_spatial[0]


def _spatial_dims(da: Any) -> Tuple[str, str]:
    dims = set(da.dims)
    for y_name, x_name in _SPATIAL_DIM_CANDIDATES:
        if y_name in dims and x_name in dims:
            return y_name, x_name
    # Fall back to last two dims.
    if len(da.dims) < 2:
        raise MultidimensionalRasterError("Raster needs at least two spatial dimensions.")
    return da.dims[-2], da.dims[-1]


def _spatial_dims_from_dataset(ds: Any) -> Tuple[str, str]:
    return _spatial_dims(ds[list(ds.data_vars)[0]])


def _apply_dimension_definitions(ds: Any, definitions: Mapping[str, DimensionSpec]) -> Any:
    indexers: Dict[str, Any] = {}
    for dim, spec in definitions.items():
        if dim not in ds.dims and dim not in ds.coords:
            raise MultidimensionalRasterError(f"Dimension {dim!r} not found.")
        if isinstance(spec, Mapping):
            start = spec.get("min", spec.get("start"))
            end = spec.get("max", spec.get("end"))
            if start is None and end is None:
                raise MultidimensionalRasterError(
                    f"Dimension range for {dim!r} needs min/max or start/end."
                )
            indexers[dim] = slice(start, end)
        elif isinstance(spec, slice):
            indexers[dim] = spec
        else:
            indexers[dim] = spec
    return ds.sel(indexers)


def _clip_extent(ds: Any, extent: Sequence[float]) -> Any:
    if len(extent) != 4:
        raise MultidimensionalRasterError("extent must be (xmin, ymin, xmax, ymax).")
    xmin, ymin, xmax, ymax = map(float, extent)
    # Use first data variable to detect spatial dims.
    var = _default_variable(ds)
    y_dim, x_dim = _spatial_dims(ds[var])
    x_asc = bool(ds[x_dim][0] < ds[x_dim][-1])
    y_asc = bool(ds[y_dim][0] < ds[y_dim][-1])
    x_slice = slice(xmin, xmax) if x_asc else slice(xmax, xmin)
    y_slice = slice(ymin, ymax) if y_asc else slice(ymax, ymin)
    return ds.sel({x_dim: x_slice, y_dim: y_slice})


def _write_mdr(
    mdr: MultidimensionalRaster,
    destination: PathLike,
    *,
    overwrite: bool = False,
    engine: Optional[str] = None,
) -> str:
    dest = Path(destination)
    if engine is None:
        if str(dest).lower().endswith(".zarr"):
            engine = "zarr"
        else:
            engine = _available_netcdf_engine() or "netcdf4"
    return _write_dataset(mdr.dataset, dest, engine=engine, overwrite=overwrite)


def _write_dataset(
    ds: Any,
    destination: PathLike,
    *,
    engine: str = "netcdf4",
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    dest = Path(destination)
    if engine == "zarr":
        if dest.exists():
            if not overwrite:
                raise MultidimensionalRasterError(f"Destination exists: {dest}")
            if dest.is_dir():
                import shutil

                shutil.rmtree(dest)
            else:
                dest.unlink()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(os.fspath(dest), **kwargs)
    else:
        _prepare_destination(dest, overwrite=overwrite)
        # Load dask arrays before netCDF write when present.
        try:
            ds = ds.load()
        except Exception:
            pass
        write_engine = engine
        if write_engine in {"netcdf", "nc"}:
            write_engine = _available_netcdf_engine() or "netcdf4"
        ds.to_netcdf(os.fspath(dest), engine=write_engine, **kwargs)
    return str(dest)


def _write_multiband_geotiff(
    mdr: MultidimensionalRaster,
    destination: PathLike,
    *,
    overwrite: bool = False,
) -> str:
    from geolattice.raster.raster_read_write import write_raster_data

    ds = mdr.dataset
    var = _default_variable(ds)
    da = ds[var]
    dim = _default_time_dimension(da)
    y_dim, x_dim = _spatial_dims(da)
    data = np.asarray(da.transpose(dim, y_dim, x_dim).values)
    # Approximate geotransform from coordinates.
    xs = np.asarray(da[x_dim].values, dtype=np.float64)
    ys = np.asarray(da[y_dim].values, dtype=np.float64)
    x_res = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    y_res = float(ys[1] - ys[0]) if len(ys) > 1 else -1.0
    gt = (float(xs[0] - x_res / 2.0), x_res, 0.0, float(ys[0] - y_res / 2.0), 0.0, y_res)
    return write_raster_data(
        destination,
        data,
        geotransform=gt,
        nodata=np.nan,
        overwrite=overwrite,
    )


def _format_from_destination(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "netcdf"
    if suffix == ".zarr" or path.name.endswith(".zarr"):
        return "zarr"
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix == ".crf":
        return "crf"
    return "netcdf"


def _infer_dimension_values_from_paths(sources: Sequence[PathLike]) -> List[Any]:
    values: List[Any] = []
    for path in sources:
        stem = Path(path).stem
        parsed = None
        for token in stem.replace("-", "_").split("_"):
            try:
                parsed = pd.to_datetime(token)
                break
            except Exception:
                continue
        values.append(parsed if parsed is not None else len(values))
    return values


def _normalize_agg_method(method: str) -> str:
    key = str(method).strip().lower()
    aliases = {
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "median": "median",
        "std": "std",
        "stddev": "std",
        "stdev": "std",
        "count": "count",
    }
    if key not in aliases:
        raise MultidimensionalRasterError(f"Unknown aggregation_method {method!r}.")
    return aliases[key]


def _interval_keyword_to_freq(keyword: str) -> str:
    mapping = {
        "HOURLY": "1H",
        "DAILY": "1D",
        "WEEKLY": "1W",
        "MONTHLY": "1MS",
        "YEARLY": "1YS",
        "ANNUALLY": "1YS",
        "QUARTERLY": "1QS",
    }
    if keyword not in mapping:
        raise MultidimensionalRasterError(
            f"Unknown interval_keyword {keyword!r}. "
            f"Known: {', '.join(sorted(mapping))}"
        )
    return mapping[keyword]


def _unit_to_offset(unit: str) -> str:
    key = str(unit).strip().lower()
    mapping = {
        "hour": "H",
        "hours": "H",
        "day": "D",
        "days": "D",
        "week": "W",
        "weeks": "W",
        "month": "MS",
        "months": "MS",
        "year": "YS",
        "years": "YS",
    }
    if key not in mapping:
        raise MultidimensionalRasterError(f"Unknown interval_unit {unit!r}.")
    return mapping[key]


def _resample_reduce(da: Any, dim: str, freq: str, method: str) -> Any:
    try:
        resampler = da.resample({dim: freq})
    except Exception as exc:
        raise MultidimensionalRasterError(
            f"Resample along {dim!r} with frequency {freq!r} failed: {exc}. "
            "Ensure the dimension is a datetime coordinate."
        ) from exc
    return getattr(resampler, method)(skipna=True)


def _interval_baseline(da: Any, dim: str, interval: str, *, use_median: bool) -> Any:
    reducer = "median" if use_median else "mean"
    if interval in {"ALL", "NONE", ""}:
        return getattr(da, reducer)(dim=dim, skipna=True)
    if interval in {"MONTHLY", "BY_MONTH"}:
        return getattr(da.groupby(f"{dim}.month"), reducer)(skipna=True)
    if interval in {"YEARLY", "BY_YEAR", "ANNUALLY"}:
        return getattr(da.groupby(f"{dim}.year"), reducer)(skipna=True)
    if interval == "DAILY":
        return getattr(da.groupby(f"{dim}.dayofyear"), reducer)(skipna=True)
    if interval == "HOURLY":
        return getattr(da.groupby(f"{dim}.hour"), reducer)(skipna=True)
    if interval == "WEEKLY":
        return getattr(da.groupby(f"{dim}.week"), reducer)(skipna=True)
    raise MultidimensionalRasterError(f"Unknown calculation_interval {interval!r}.")


def _broadcast_baseline(
    da: Any,
    baseline: Any,
    *,
    dim: str,
    interval: str,
) -> Any:
    """Map a climatology / summary baseline onto the full data time axis."""
    interval = str(interval).strip().upper()
    if interval in {"ALL", "NONE", ""}:
        return baseline
    key_map = {
        "MONTHLY": "month",
        "BY_MONTH": "month",
        "YEARLY": "year",
        "BY_YEAR": "year",
        "ANNUALLY": "year",
        "DAILY": "dayofyear",
        "HOURLY": "hour",
        "WEEKLY": "week",
    }
    key = key_map.get(interval)
    if key is None:
        return baseline
    if key not in baseline.dims and key not in baseline.coords:
        return baseline
    selectors = getattr(da[dim].dt, key)
    aligned = baseline.sel({key: selectors})
    # Drop the climatology key so dims match the original cube.
    if key in aligned.coords:
        aligned = aligned.drop_vars(key, errors="ignore")
    if key in getattr(aligned, "dims", ()):
        aligned = aligned.reset_coords(drop=True)
    # Ensure the time dimension name is restored when sel produced a renamed index.
    if dim not in aligned.dims and aligned.ndim == da.ndim:
        # sel on a scalar calendar field can leave a positional dim; rename if needed.
        for d in aligned.dims:
            if d not in da.dims:
                aligned = aligned.rename({d: dim})
                break
    return aligned


def _arg_extreme(
    values: np.ndarray,
    *,
    axis: int,
    mode: str,
    multiple: str,
    ignore_nodata: bool,
) -> np.ndarray:
    work = np.array(values, dtype=np.float64, copy=True)
    if ignore_nodata:
        if mode == "max":
            work[~np.isfinite(work)] = -np.inf
        else:
            work[~np.isfinite(work)] = np.inf
    if multiple == "LAST_OCCURRENCE":
        work = np.flip(work, axis=axis)
        idx = np.argmax(work, axis=axis) if mode == "max" else np.argmin(work, axis=axis)
        idx = work.shape[axis] - 1 - idx
    else:
        idx = np.argmax(work, axis=axis) if mode == "max" else np.argmin(work, axis=axis)
    return idx


def _index_to_coord(coords: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat_idx = np.asarray(idx, dtype=np.int64)
    # Invalid indices (all-nan slices) → NaT / NaN
    out = np.empty(flat_idx.shape, dtype=object)
    valid = (flat_idx >= 0) & (flat_idx < len(coords))
    out[valid] = coords[flat_idx[valid]]
    out[~valid] = np.datetime64("NaT") if np.issubdtype(coords.dtype, np.datetime64) else np.nan
    return out


def _first_true(mask: np.ndarray, axis: int) -> np.ndarray:
    any_true = mask.any(axis=axis)
    idx = np.argmax(mask, axis=axis)
    idx = np.where(any_true, idx, -1)
    return idx


def _last_true(mask: np.ndarray, axis: int) -> np.ndarray:
    flipped = np.flip(mask, axis=axis)
    any_true = flipped.any(axis=axis)
    idx = np.argmax(flipped, axis=axis)
    idx = mask.shape[axis] - 1 - idx
    idx = np.where(any_true, idx, -1)
    return idx


def _datetime_to_ordinal(values: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(values), np.nan, dtype=np.float64)
    flat_in = np.asarray(values, dtype=object).ravel()
    flat_out = out.ravel()
    for i, val in enumerate(flat_in):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        try:
            ts = pd.Timestamp(val)
            if pd.isna(ts):
                continue
            flat_out[i] = float(ts.toordinal())
        except Exception:
            continue
    return out


def _dimension_as_float(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        # Days since first timestamp.
        origin = arr[0]
        return (arr - origin) / np.timedelta64(1, "D")
    return arr.astype(np.float64)


def _infer_cycle_length(t: np.ndarray) -> float:
    if len(t) < 2:
        return 365.25
    span = float(np.nanmax(t) - np.nanmin(t))
    # Prefer annual cycle when series spans > ~180 days.
    if span > 180:
        return 365.25
    return max(span, 1.0)


def _fit_polynomial_pixels(
    flat: np.ndarray,
    t: np.ndarray,
    *,
    order: int,
    ignore_nodata: bool,
) -> np.ndarray:
    stats = _require_scipy_stats()
    n_pix = flat.shape[0]
    n_coeff = order + 1
    out = np.full((n_pix, n_coeff + 2), np.nan, dtype=np.float64)
    for i in range(n_pix):
        y = flat[i]
        mask = np.isfinite(y) & np.isfinite(t) if ignore_nodata else np.ones_like(y, dtype=bool)
        if np.count_nonzero(mask) < order + 2:
            continue
        tt = t[mask]
        yy = y[mask]
        try:
            coeffs = np.polyfit(tt, yy, deg=order)
            # polyfit returns highest degree first; store low→high.
            coeffs = coeffs[::-1]
            y_hat = np.zeros_like(yy)
            for p, c in enumerate(coeffs):
                y_hat += c * tt**p
            ss_res = np.sum((yy - y_hat) ** 2)
            ss_tot = np.sum((yy - np.mean(yy)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if order == 1:
                slope, intercept, r_value, p_value, _ = stats.linregress(tt, yy)
                out[i, 0] = intercept
                out[i, 1] = slope
                out[i, 2] = r_value**2
                out[i, 3] = p_value
            else:
                out[i, :n_coeff] = coeffs
                out[i, n_coeff] = r2
                # Approximate p-value from linear slope term when available.
                if n_coeff >= 2:
                    _, _, _, p_value, _ = stats.linregress(tt, yy)
                    out[i, n_coeff + 1] = p_value
        except Exception:
            continue
    return out


def _fit_harmonic_pixels(
    flat: np.ndarray,
    t: np.ndarray,
    *,
    frequency: int,
    cycle_length: float,
    ignore_nodata: bool,
) -> np.ndarray:
    n_pix = flat.shape[0]
    n_coeff = 2 + 2 * frequency + 2  # intercept, slope, cos/sin..., r2, rmse
    out = np.full((n_pix, n_coeff), np.nan, dtype=np.float64)
    omega = 2.0 * math.pi / float(cycle_length)
    for i in range(n_pix):
        y = flat[i]
        mask = np.isfinite(y) & np.isfinite(t) if ignore_nodata else np.ones_like(y, dtype=bool)
        if np.count_nonzero(mask) < n_coeff:
            continue
        tt = t[mask]
        yy = y[mask]
        cols = [np.ones_like(tt), tt]
        for k in range(1, frequency + 1):
            cols.append(np.cos(omega * k * tt))
            cols.append(np.sin(omega * k * tt))
        design = np.column_stack(cols)
        try:
            beta, _, _, _ = np.linalg.lstsq(design, yy, rcond=None)
            y_hat = design @ beta
            ss_res = np.sum((yy - y_hat) ** 2)
            ss_tot = np.sum((yy - np.mean(yy)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = math.sqrt(ss_res / max(len(yy) - len(beta), 1))
            out[i, : len(beta)] = beta
            out[i, -2] = r2
            out[i, -1] = rmse
        except Exception:
            continue
    return out


def _mann_kendall_pixels(flat: np.ndarray, *, ignore_nodata: bool) -> np.ndarray:
    n_pix = flat.shape[0]
    out = np.full((n_pix, 4), np.nan, dtype=np.float64)
    t_idx = np.arange(flat.shape[1], dtype=np.float64)
    for i in range(n_pix):
        y = flat[i]
        mask = np.isfinite(y) if ignore_nodata else np.ones_like(y, dtype=bool)
        if np.count_nonzero(mask) < 4:
            continue
        yy = y[mask]
        tt = t_idx[mask]
        tau, p_value = _mann_kendall_stats(yy)
        slope, intercept = _theil_sen(tt, yy)
        out[i] = (tau, p_value, slope, intercept)
    return out


def _seasonal_kendall_pixels(
    flat: np.ndarray,
    *,
    periods: int,
    ignore_nodata: bool,
) -> np.ndarray:
    """Seasonal Mann-Kendall by pooling season-wise Mann-Kendall scores."""
    n_pix = flat.shape[0]
    out = np.full((n_pix, 4), np.nan, dtype=np.float64)
    n_time = flat.shape[1]
    t_idx = np.arange(n_time, dtype=np.float64)
    for i in range(n_pix):
        y = flat[i]
        mask = np.isfinite(y) if ignore_nodata else np.ones_like(y, dtype=bool)
        if np.count_nonzero(mask) < periods * 2:
            continue
        s_total = 0.0
        var_total = 0.0
        for season in range(periods):
            idx = np.where(mask & (np.arange(n_time) % periods == season))[0]
            if idx.size < 3:
                continue
            s, var_s = _mann_kendall_s_var(y[idx])
            s_total += s
            var_total += var_s
        if var_total <= 0:
            continue
        z = s_total / math.sqrt(var_total)
        # Two-sided normal p-value without scipy dependency on erf path.
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
        tau = s_total  # keep score; not standardized like classic tau
        slope, intercept = _theil_sen(t_idx[mask], y[mask])
        out[i] = (tau, p_value, slope, intercept)
    return out


def _mann_kendall_stats(y: np.ndarray) -> Tuple[float, float]:
    s, var_s = _mann_kendall_s_var(y)
    n = len(y)
    if n < 3 or var_s <= 0:
        return np.nan, np.nan
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    tau = s / (0.5 * n * (n - 1))
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return float(tau), float(p_value)


def _mann_kendall_s_var(y: np.ndarray) -> Tuple[float, float]:
    n = len(y)
    s = 0.0
    for i in range(n - 1):
        diffs = y[i + 1 :] - y[i]
        s += float(np.sum(diffs > 0) - np.sum(diffs < 0))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    return s, var_s


def _theil_sen(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    slopes: List[float] = []
    n = len(t)
    for i in range(n - 1):
        dt = t[i + 1 :] - t[i]
        dy = y[i + 1 :] - y[i]
        valid = dt != 0
        if np.any(valid):
            slopes.extend((dy[valid] / dt[valid]).tolist())
    if not slopes:
        return np.nan, np.nan
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * t))
    return slope, intercept


def _ccdc_breaks(
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_observations: int,
    threshold: float,
    max_breaks: int,
    frequency: int,
    cycle_length: float,
) -> Tuple[List[float], float]:
    mask = np.isfinite(y) & np.isfinite(t)
    tt = t[mask]
    yy = y[mask]
    if len(yy) < min_observations * 2:
        return [], np.nan

    breaks: List[float] = []
    start = 0
    last_mag = np.nan
    while start + min_observations < len(yy) and len(breaks) < max_breaks:
        end = start + min_observations
        fitted = False
        while end < len(yy):
            beta, rmse = _fit_harmonic_segment(
                tt[start:end],
                yy[start:end],
                frequency=frequency,
                cycle_length=cycle_length,
            )
            if beta is None:
                end += 1
                continue
            fitted = True
            # Peek next observation residual.
            if end >= len(yy):
                break
            pred = _eval_harmonic(tt[end], beta, frequency=frequency, cycle_length=cycle_length)
            resid = yy[end] - pred
            # Chi-square style single residual test.
            if rmse > 0 and (resid**2) / (rmse**2) > threshold:
                breaks.append(float(tt[end]))
                last_mag = float(resid)
                start = end
                break
            end += 1
        if not fitted or end >= len(yy):
            break
        if not breaks or start >= len(yy) - min_observations:
            break
    return breaks, last_mag


def _fit_harmonic_segment(
    t: np.ndarray,
    y: np.ndarray,
    *,
    frequency: int,
    cycle_length: float,
) -> Tuple[Optional[np.ndarray], float]:
    if len(y) < 2 + 2 * frequency:
        return None, np.nan
    omega = 2.0 * math.pi / float(cycle_length)
    cols = [np.ones_like(t), t]
    for k in range(1, frequency + 1):
        cols.append(np.cos(omega * k * t))
        cols.append(np.sin(omega * k * t))
    design = np.column_stack(cols)
    try:
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        rmse = float(np.sqrt(np.mean(resid**2)))
        return beta, rmse
    except Exception:
        return None, np.nan


def _eval_harmonic(
    t: float,
    beta: np.ndarray,
    *,
    frequency: int,
    cycle_length: float,
) -> float:
    omega = 2.0 * math.pi / float(cycle_length)
    value = float(beta[0] + beta[1] * t)
    idx = 2
    for k in range(1, frequency + 1):
        value += float(beta[idx] * math.cos(omega * k * t) + beta[idx + 1] * math.sin(omega * k * t))
        idx += 2
    return value


def _landtrendr_pixel(
    t: np.ndarray,
    y: np.ndarray,
    *,
    max_segments: int,
    spike_threshold: float,
    recovery_threshold: float,
    min_observations: int,
) -> Dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(t)
    tt = t[mask]
    yy = y[mask]
    empty = {
        "disturbance_time": np.nan,
        "magnitude": np.nan,
        "duration": np.nan,
        "recovery_rate": np.nan,
    }
    if len(yy) < min_observations:
        return empty

    # Spike filter: replace outliers relative to local median.
    if len(yy) >= 5:
        med = pd.Series(yy).rolling(3, center=True, min_periods=1).median().to_numpy()
        resid = yy - med
        scale = np.nanmedian(np.abs(resid)) + 1e-9
        spike = np.abs(resid) > spike_threshold * 6.0 * scale
        yy = yy.copy()
        yy[spike] = med[spike]

    # Vertex detection via largest residual stepwise segmentation (simple LandTrendr-like).
    vertices = [0, len(yy) - 1]
    for _ in range(max(0, max_segments - 1)):
        best_idx = None
        best_gain = 0.0
        for a, b in zip(vertices[:-1], vertices[1:]):
            if b - a < 3:
                continue
            idx, gain = _best_split(tt[a : b + 1], yy[a : b + 1])
            if idx is not None and gain > best_gain:
                best_gain = gain
                best_idx = a + idx
        if best_idx is None:
            break
        vertices.append(best_idx)
        vertices = sorted(set(vertices))

    # Greatest disturbance = largest drop between consecutive vertices.
    best_mag = 0.0
    best_i = None
    for i in range(len(vertices) - 1):
        a, b = vertices[i], vertices[i + 1]
        mag = float(yy[a] - yy[b])  # positive = decline
        if mag > best_mag:
            best_mag = mag
            best_i = i
    if best_i is None or best_mag <= 0:
        return empty

    a, b = vertices[best_i], vertices[best_i + 1]
    duration = float(tt[b] - tt[a])
    recovery_rate = np.nan
    if best_i + 2 < len(vertices):
        c = vertices[best_i + 2]
        recovery = float(yy[c] - yy[b])
        dt = float(tt[c] - tt[b])
        if dt > 0:
            recovery_rate = recovery / dt
            if recovery_rate < recovery_threshold * (-best_mag / max(duration, 1e-9)):
                # Weak recovery flag kept as numeric rate.
                pass

    return {
        "disturbance_time": float(tt[b]),
        "magnitude": float(best_mag),
        "duration": duration,
        "recovery_rate": float(recovery_rate) if recovery_rate == recovery_rate else np.nan,
    }


def _best_split(t: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], float]:
    if len(y) < 4:
        return None, 0.0
    # Fit single segment SSE.
    full_sse = _segment_sse(t, y)
    best_idx = None
    best_gain = 0.0
    for i in range(1, len(y) - 1):
        sse = _segment_sse(t[: i + 1], y[: i + 1]) + _segment_sse(t[i:], y[i:])
        gain = full_sse - sse
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    return best_idx, best_gain


def _segment_sse(t: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    design = np.column_stack([np.ones_like(t), t])
    try:
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        return float(np.sum(resid**2))
    except Exception:
        return float(np.sum((y - np.mean(y)) ** 2))
