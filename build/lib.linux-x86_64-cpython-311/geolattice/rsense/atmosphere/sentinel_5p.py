"""Sentinel-5P: TROPOMI atmospheric composition products.

Sentinel-5P does not fit the lazy-STAC-raster loader the optical sensor modules share, and it
is a different kind of product from Sentinel-3's multi-file ``.SEN3`` swaths too: each TROPOMI
scene is one self-contained NetCDF-4 file with a nested group structure (``PRODUCT``,
``PRODUCT/SUPPORT_DATA/DETAILED_RESULTS``, ``PRODUCT/SUPPORT_DATA/INPUT_DATA``,
``PRODUCT/SUPPORT_DATA/GEOLOCATIONS``, ``METADATA``), holding a retrieved trace-gas column
rather than reflectance. This module gives one reader per product and a small set of shared
helpers that read everything they can from the file's own metadata rather than a hardcoded
constant:

- quality filtering reads the recommended ``qa_value`` threshold straight out of the variable's
  own ``comment`` attribute (ESA changes this threshold between algorithm versions; a constant
  in this module would silently go stale);
- unit conversion to molecules/cm² reads the file's own
  ``multiplication_factor_to_convert_to_molecules_percm2`` attribute, never Avogadro's number
  typed in by hand.

Input is a single Level-2 product NetCDF file (what you get from the Copernicus Data Space
Ecosystem, EUMETSAT Data Store, or Planetary Computer's ``sentinel-5p-l2-netcdf`` collection).
Reading it needs the optional ``netCDF4`` or ``h5netcdf`` package, installed with
``pip install geolattice[rsense-sentinel5p]``.

Product short names, matching the ``s5p:product_name`` STAC property: ``no2``, ``so2``, ``o3``,
``co``, ``ch4``, ``hcho`` (trace gases); ``aer_ai`` (absorbing aerosol index); ``aer_lh``
(aerosol mid-layer height); ``cloud`` (cloud fraction, pressure, height, optical thickness).

The standard Level-2-to-3 workflow this module supports end to end: search and download,
read the ``PRODUCT`` group, filter by ``qa_value``, convert units, grid the swath onto a
regular map (:func:`bin_to_grid` for simple pixel-centre averaging, :func:`oversample_to_grid`
for pixel-corner area-weighted gridding on finer grids), aggregate over time
(:func:`temporal_composite`), then validate. Product-specific extras: averaging-kernel model
comparison for NO2 (:func:`apply_averaging_kernel`), CH4 point-source emission estimation
(:func:`ch4_enhancement`, :func:`integrated_mass_enhancement`, :func:`estimate_emission_rate`),
dry-air mixing ratio for CO (:func:`co_mixing_ratio`), and the SO2 volcanic layer-height
variants (``read_so2(..., layer_height=...)``).

Two things this module deliberately does not do: wrap Google Earth Engine's pre-gridded
``COPERNICUS/S5P/OFFL/L3_*`` collections (a different, already-gridded data source, not a
NetCDF file this module reads) or HARP's command-line gridding (an external tool with its own
Python bindings, ``import harp``) -- both are valid alternative paths to the same Level-3
products. It also does not read Level-1B radiances: that path is for algorithm development
(building your own DOAS or SWIR retrieval), well outside reading an official product.

A few things worth knowing before you start: Sentinel-5P observes at roughly 13:30 local solar
time, so align any ground or model data you compare against to that time, not a daily mean.
Never average SO2 or HCHO after clipping away negative values -- they are retrieval noise
distributed around zero, and clipping biases the mean high. Do not mix processor versions in a
time series (:func:`check_processing_consistency` flags this, along with the 6 August 2019
along-track pixel size change from 7 km to 5.5 km).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..optical.sensors.sentinel3 import swath_to_grid  # generic swath-geometry resampler

__all__ = [
    "S5P_PIXEL_SIZE_CHANGE",
    "S5P_PRODUCTS",
    "S5P_QA_RECOMMENDED",
    "apply_averaging_kernel",
    "bin_to_grid",
    "check_processing_consistency",
    "ch4_enhancement",
    "co_mixing_ratio",
    "dobson_units",
    "estimate_emission_rate",
    "integrated_mass_enhancement",
    "molecules_per_cm2",
    "oversample_to_grid",
    "qa_mask",
    "read_aerosol_index",
    "read_aerosol_layer_height",
    "read_ch4",
    "read_cloud",
    "read_co",
    "read_geolocation_bounds",
    "read_hcho",
    "read_metadata",
    "read_no2",
    "read_o3",
    "read_so2",
    "swath_to_grid",
    "temporal_composite",
    "tm5_pressure_levels",
]

# Product short name -> (content, default PRODUCT-group main variable). The default is a
# starting point, not a guarantee across every processing baseline -- pass `variables=` to any
# reader to override it if ESA renames a variable in a future algorithm version.
S5P_PRODUCTS = {
    "no2": ("Tropospheric NO2 vertical column", "nitrogendioxide_tropospheric_column"),
    "so2": ("Total SO2 vertical column", "sulfurdioxide_total_vertical_column"),
    "o3": ("Total ozone vertical column", "ozone_total_vertical_column"),
    "co": ("Total CO column", "carbonmonoxide_total_column"),
    "ch4": ("Bias-corrected CH4 dry-air mixing ratio", "methane_mixing_ratio_bias_corrected"),
    "hcho": ("Tropospheric HCHO vertical column", "formaldehyde_tropospheric_vertical_column"),
    "aer_ai": ("UV absorbing aerosol index (354/388 nm)", "aerosol_index_354_388"),
    "aer_lh": ("Aerosol mid-layer height", "aerosol_mid_height"),
}

# Literature-recommended qa_value cutoffs (stricter, in places, than the generic "< 0.5"
# comment most product files carry on the qa_value variable itself -- see qa_mask). Pass
# `min_qa=S5P_QA_RECOMMENDED["no2"]` to use these instead of the per-file dynamic default.
S5P_QA_RECOMMENDED = {
    "no2": 0.75, "so2": 0.5, "o3": 0.5, "co": 0.5, "ch4": 0.5, "hcho": 0.5,
    "aer_ai": 0.8, "aer_lh": 0.5, "cloud": 0.5,
}

# Along-track pixel size dropped from 7 km to 5.5 km on this date (a ground-processor change,
# not a satellite manoeuvre); see check_processing_consistency.
S5P_PIXEL_SIZE_CHANGE = "2019-08-06"

_QA_FALLBACK = 0.5  # used only if qa_value has no parseable "ignore ... < X" comment


def read_metadata(path: str | PathLike) -> dict[str, Any]:
    """Global attributes of a Sentinel-5P product: time coverage, orbit, processing mode
    (``NRTI``/``OFFL``/``RPRO``), processor version. Cheap -- does not read any pixel data."""
    with xr.open_dataset(path) as ds:
        keys = ("time_coverage_start", "time_coverage_end", "orbit", "processing_mode",
                "processor_version", "product_version", "platform_shortname", "sensor")
        return {k: ds.attrs[k] for k in keys if k in ds.attrs}


def _recommended_qa(qa: xr.DataArray) -> float:
    """Parse the recommended ``qa_value`` threshold from the variable's own ``comment``
    attribute (e.g. ``"Recommend to ignore data with qa_value < 0.5"``), falling back to
    :data:`_QA_FALLBACK` if the comment is missing or not in the expected form."""
    comment = str(qa.attrs.get("comment", ""))
    match = re.search(r"qa_value\s*<\s*([\d.]+)", comment)
    return float(match.group(1)) if match else _QA_FALLBACK


def qa_mask(qa_value: xr.DataArray, *, min_qa: float | None = None) -> xr.DataArray:
    """Boolean validity mask from ``qa_value`` (0 = no data, 1 = full quality).

    ``min_qa=None`` (the default) reads ESA's own recommended threshold from the variable's
    ``comment`` attribute rather than assuming one -- the right cutoff differs by product
    (NO2 and HCHO typically recommend a stricter threshold than CO) and has changed between
    processing baselines.
    """
    threshold = _recommended_qa(qa_value) if min_qa is None else min_qa
    return (qa_value >= threshold).rename("valid")


def molecules_per_cm2(da: xr.DataArray) -> xr.DataArray:
    """Convert a mol/m² column density to molecules/cm², the unit most air-quality literature
    reports in, using the file's own ``multiplication_factor_to_convert_to_molecules_percm2``
    attribute (never a hardcoded Avogadro's-number calculation, since the factor already
    folds in the mol/m² -> molecules/cm² unit change)."""
    factor = da.attrs.get("multiplication_factor_to_convert_to_molecules_percm2")
    if factor is None:
        raise ValueError(f"{da.name!r} has no multiplication_factor_to_convert_to_molecules_percm2 "
                         "attribute; it may not be a column-density variable.")
    out = (da * float(factor)).astype("float32")
    return out.rename(da.name).assign_attrs({**da.attrs, "units": "molecules cm-2"})


def dobson_units(total_column_mol_m2: xr.DataArray) -> xr.DataArray:
    """Convert a total-column density in mol/m² (ozone's native unit) to Dobson Units.

    1 DU = 2.6867e16 molecules/cm² = 2.6867e20 molecules/m²; dividing by Avogadro's number
    (6.02214e23 /mol) gives 1 DU = 4.4615e-4 mol/m², so DU = mol_m2 / 4.4615e-4.
    """
    du = (total_column_mol_m2 / 4.4615e-4).astype("float32")
    return du.rename(total_column_mol_m2.name).assign_attrs(
        {**total_column_mol_m2.attrs, "units": "DU"})


def _read_product(path: str | PathLike, default_vars: Sequence[str],
                  variables: Sequence[str] | None = None) -> xr.Dataset:
    """Open the ``PRODUCT`` group, keep the requested variables (plus their ``_precision``
    companions when present), ``qa_value``, and ``latitude``/``longitude`` as coordinates,
    and squeeze the orbit file's singleton ``time`` dimension."""
    path = Path(path)
    wanted = list(variables) if variables is not None else list(default_vars)
    with xr.open_dataset(path, group="PRODUCT") as ds:
        keep = []
        for v in wanted:
            if v not in ds.data_vars:
                raise KeyError(f"{v!r} not found in {path.name}'s PRODUCT group. "
                               f"Available: {sorted(ds.data_vars)}")
            keep.append(v)
            if f"{v}_precision" in ds.data_vars:
                keep.append(f"{v}_precision")
        if "qa_value" in ds.data_vars:
            keep.append("qa_value")
        for geo in ("latitude", "longitude"):
            if geo in ds.data_vars or geo in ds.coords:
                keep.append(geo)
        out = ds[keep].load()
    if "time" in out.dims and out.sizes["time"] == 1:
        out = out.squeeze("time", drop=True)
    coords = [c for c in ("latitude", "longitude") if c in out.data_vars]
    if coords:
        out = out.set_coords(coords)
    return out


# --------------------------------------------------------------------------- trace gases
def read_no2(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P NO2: tropospheric vertical column (default), mol/m², plus
    ``qa_value``. Pass ``variables=["nitrogendioxide_total_column"]`` etc. for other columns
    held in the ``PRODUCT`` group's ``SUPPORT_DATA/DETAILED_RESULTS`` subgroup -- read that
    subgroup directly with ``xr.open_dataset(path, group="PRODUCT/SUPPORT_DATA/DETAILED_RESULTS")``
    for the stratospheric column, air mass factors and fit diagnostics."""
    return _read_product(path, [S5P_PRODUCTS["no2"][1]], variables)


def read_so2(path: str | PathLike, variables: Sequence[str] | None = None, *,
            layer_height: str = "default") -> xr.Dataset:
    """Read Sentinel-5P SO2: total vertical column, mol/m², plus ``qa_value``.

    ``layer_height`` picks the a priori plume-height assumption: ``"default"`` (the standard
    boundary-layer column, appropriate for anthropogenic sources) or ``"1km"``/``"7km"``/
    ``"15km"`` (the volcanic-plume variants, held in
    ``PRODUCT/SUPPORT_DATA/DETAILED_RESULTS`` as ``sulfurdioxide_total_vertical_column_1km``
    etc. -- confirmed against a real downloaded product). Match the variant to the plume: a
    weak boundary-layer source read with the 15 km assumption (or vice versa) is biased.
    Keep negative values when averaging; they are retrieval noise, not missing data, and
    clipping them biases the mean high.
    """
    if layer_height == "default":
        return _read_product(path, [S5P_PRODUCTS["so2"][1]], variables)
    if layer_height not in ("1km", "7km", "15km"):
        raise ValueError(f"layer_height must be 'default', '1km', '7km' or '15km', got {layer_height!r}")
    var = f"sulfurdioxide_total_vertical_column_{layer_height}"
    wanted = list(variables) if variables is not None else [var]
    path = Path(path)
    with xr.open_dataset(path, group="PRODUCT/SUPPORT_DATA/DETAILED_RESULTS") as detail, \
         xr.open_dataset(path, group="PRODUCT") as product:
        keep = []
        for v in wanted:
            if v not in detail.data_vars:
                raise KeyError(f"{v!r} not found in {path.name}'s DETAILED_RESULTS group. "
                               f"Available: {sorted(n for n in detail.data_vars if 'sulfurdioxide' in n)}")
            keep.append(v)
            if f"{v}_precision" in detail.data_vars:
                keep.append(f"{v}_precision")
        out = detail[keep].load()
        extra = {}
        if "qa_value" in product.data_vars:
            extra["qa_value"] = product["qa_value"]
        for geo in ("latitude", "longitude"):
            if geo in product.data_vars or geo in product.coords:
                extra[geo] = product[geo]
        out = out.assign(**extra) if extra else out
    if "time" in out.dims and out.sizes["time"] == 1:
        out = out.squeeze("time", drop=True)
    coords = [c for c in ("latitude", "longitude") if c in out.data_vars]
    if coords:
        out = out.set_coords(coords)
    return out


def read_o3(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P O3: total vertical column (default), mol/m², plus ``qa_value``.
    Convert to Dobson Units with :func:`dobson_units`."""
    return _read_product(path, [S5P_PRODUCTS["o3"][1]], variables)


def read_co(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P CO: total column (default), mol/m², plus ``qa_value``. CO's SWIR
    retrieval is far less cloud-sensitive than the UV/VIS gases, so its recommended
    ``qa_value`` threshold (read dynamically by :func:`qa_mask`) is typically more lenient."""
    return _read_product(path, [S5P_PRODUCTS["co"][1]], variables)


def read_ch4(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P CH4: bias-corrected column-averaged dry-air mixing ratio (default,
    ppb), plus ``qa_value``. Use the bias-corrected variable, not the raw
    ``methane_mixing_ratio``, for emission estimates -- the correction removes a systematic
    retrieval bias that is large enough to matter for point-source quantification."""
    return _read_product(path, [S5P_PRODUCTS["ch4"][1]], variables)


def read_hcho(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P HCHO (formaldehyde): tropospheric vertical column (default), mol/m²,
    plus ``qa_value``."""
    return _read_product(path, [S5P_PRODUCTS["hcho"][1]], variables)


# --------------------------------------------------------------------------- aerosol and cloud
def read_aerosol_index(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P UV Aerosol Index (354/388 nm, default), plus ``qa_value``. A
    positive index indicates UV-absorbing aerosol (smoke, dust, volcanic ash); a negative
    index indicates non-absorbing aerosol or clean/cloudy scenes. Index values can be
    meaningful even where ``qa_value`` is middling, since the underlying reflectance-residual
    calculation does not need the same cloud-free conditions a full retrieval does -- check
    both before deciding how strict a cutoff your use case needs."""
    return _read_product(path, [S5P_PRODUCTS["aer_ai"][1]], variables)


def read_cloud(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P CLOUD: fraction, top pressure, top height and optical thickness
    (default), plus ``qa_value``. This is the dedicated OCRA/ROCINN cloud product, distinct
    from the lighter-weight FRESCO cloud fields embedded as an input inside the trace-gas
    products."""
    default = ["cloud_fraction", "cloud_top_pressure", "cloud_top_height",
              "cloud_optical_thickness"]
    return _read_product(path, default, variables)


def read_aerosol_layer_height(path: str | PathLike, variables: Sequence[str] | None = None) -> xr.Dataset:
    """Read Sentinel-5P Aerosol Layer Height: mid-layer height in metres (default), plus
    ``qa_value``. Valid only over dark surfaces (ocean, dark vegetation) under a thick
    absorbing aerosol layer -- the retrieval needs UV absorption contrast against the
    surface, so it is sparse and mainly useful for smoke and dust plume height, not a
    routine gap-free product like the trace gases."""
    return _read_product(path, [S5P_PRODUCTS["aer_lh"][1]], variables)


def read_geolocation_bounds(path: str | PathLike) -> xr.Dataset:
    """Read each ground pixel's four corner coordinates from
    ``PRODUCT/SUPPORT_DATA/GEOLOCATIONS`` (``latitude_bounds``, ``longitude_bounds``,
    shape ``(scanline, ground_pixel, corner=4)``). Needed for :func:`oversample_to_grid`;
    not read by the per-product readers above, since most uses only need the pixel centre
    (``latitude``/``longitude``, already included in their output)."""
    path = Path(path)
    with xr.open_dataset(path, group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS") as ds:
        out = ds[["latitude_bounds", "longitude_bounds"]].load()
    if "time" in out.dims and out.sizes["time"] == 1:
        out = out.squeeze("time", drop=True)
    return out


# --------------------------------------------------------------------------- Level 2 -> Level 3 gridding
def bin_to_grid(da: Any, lat: Any, lon: Any, lat_edges: Any, lon_edges: Any, *,
                min_count: int = 1) -> xr.Dataset:
    """Grid swath data onto a regular lat/lon grid by averaging the pixel centres that fall
    in each cell (binning, not interpolation) -- fast, and the right choice for most cells at
    least about 0.05 degrees across. For finer grids, or a composite built from several
    overlapping orbits, pixel-corner area-weighted gridding (:func:`oversample_to_grid`) is
    more accurate.

    ``lat_edges``/``lon_edges`` are grid cell boundaries, N+1 edges for N cells (e.g.
    ``np.arange(30.0, 45.01, 0.05)``). Pass data already filtered with
    ``.where(qa_mask(...))`` -- NaN pixels are dropped before binning. Cells with fewer than
    ``min_count`` valid pixels come back as NaN, not a mean of one or two noisy retrievals.

    Returns a Dataset with ``mean`` (float32) and ``count`` (int32, pixels per cell).
    """
    values = np.asarray(da, dtype="float64").ravel()
    lat_v = np.asarray(lat, dtype="float64").ravel()
    lon_v = np.asarray(lon, dtype="float64").ravel()
    ok = np.isfinite(values) & np.isfinite(lat_v) & np.isfinite(lon_v)
    values, lat_v, lon_v = values[ok], lat_v[ok], lon_v[ok]

    lat_edges = np.asarray(lat_edges, dtype="float64")
    lon_edges = np.asarray(lon_edges, dtype="float64")
    sums, _, _ = np.histogram2d(lat_v, lon_v, bins=[lat_edges, lon_edges], weights=values)
    counts, _, _ = np.histogram2d(lat_v, lon_v, bins=[lat_edges, lon_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts >= min_count, sums / counts, np.nan)

    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    out = xr.Dataset(
        {"mean": (("y", "x"), mean.astype("float32")), "count": (("y", "x"), counts.astype("int32"))},
        coords={"y": lat_c, "x": lon_c},
    )
    return out.rio.write_crs(4326)


def oversample_to_grid(da: Any, lat_bounds: Any, lon_bounds: Any, lat_edges: Any,
                       lon_edges: Any) -> xr.Dataset:
    """Area-weighted oversampling: grid swath data by exact polygon overlap between each
    pixel's four corners (from :func:`read_geolocation_bounds`) and each destination grid
    cell, weighting by the fraction of the pixel's area inside that cell. More accurate than
    :func:`bin_to_grid` for grids finer than the pixel footprint, or for a composite built
    from many overlapping orbits, at a real computational cost -- this is a reference
    implementation (shapely polygon intersection per pixel/cell pair found through a spatial
    index), not the optimised C routine HARP's own ``bin_spatial`` uses. Expect it to be
    noticeably slower than binning on a full-swath, fine grid; subset to your AOI first.

    Returns a Dataset with ``mean`` (float32) and ``weight`` (float32, summed area fraction
    per cell -- comparable to, but not identical to, :func:`bin_to_grid`'s pixel ``count``).
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    values = np.asarray(da, dtype="float64").ravel()
    lat_b = np.asarray(lat_bounds, dtype="float64").reshape(-1, 4)
    lon_b = np.asarray(lon_bounds, dtype="float64").reshape(-1, 4)
    ok = (np.isfinite(values) & np.all(np.isfinite(lat_b), axis=1) & np.all(np.isfinite(lon_b), axis=1))
    values, lat_b, lon_b = values[ok], lat_b[ok], lon_b[ok]

    lat_edges = np.asarray(lat_edges, dtype="float64")
    lon_edges = np.asarray(lon_edges, dtype="float64")
    ny, nx = lat_edges.size - 1, lon_edges.size - 1
    cells, cell_ij = [], []
    for i in range(ny):
        for j in range(nx):
            cells.append(Polygon([(lon_edges[j], lat_edges[i]), (lon_edges[j + 1], lat_edges[i]),
                                  (lon_edges[j + 1], lat_edges[i + 1]), (lon_edges[j], lat_edges[i + 1])]))
            cell_ij.append((i, j))
    tree = STRtree(cells)

    sums = np.zeros((ny, nx))
    weights = np.zeros((ny, nx))
    for v, lat4, lon4 in zip(values, lat_b, lon_b):
        pix = Polygon(zip(lon4, lat4))
        if not pix.is_valid:
            pix = pix.buffer(0)
        pix_area = pix.area
        if pix_area <= 0:
            continue
        for idx in tree.query(pix):
            inter_area = pix.intersection(cells[idx]).area
            if inter_area <= 0:
                continue
            i, j = cell_ij[idx]
            frac = inter_area / pix_area
            sums[i, j] += v * frac
            weights[i, j] += frac

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weights > 0, sums / weights, np.nan)
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    out = xr.Dataset(
        {"mean": (("y", "x"), mean.astype("float32")), "weight": (("y", "x"), weights.astype("float32"))},
        coords={"y": lat_c, "x": lon_c},
    )
    return out.rio.write_crs(4326)


def temporal_composite(swaths: Sequence[tuple], lat_edges: Any, lon_edges: Any, *,
                       min_count: float = 3, method: str = "bin") -> xr.Dataset:
    """Accumulate several swaths into one time-composite grid -- the standard way to build a
    monthly, seasonal or multi-orbit mean, since a single daily TROPOMI map is noisy and full
    of cloud gaps (monthly is the usual minimum for NO2 and HCHO; CH4 and SO2 often need
    longer). Keep negative values in ``swaths`` for SO2 and HCHO -- they are retrieval noise
    around zero, and dropping them biases the composite high.

    ``method="bin"``: each entry of ``swaths`` is ``(values, lat, lon)``, gridded per-swath
    with :func:`bin_to_grid`. ``method="oversample"``: each entry is
    ``(values, lat_bounds, lon_bounds)``, gridded per-swath with :func:`oversample_to_grid`.
    Cells with fewer than ``min_count`` total observations across every swath are NaN.
    """
    if method not in ("bin", "oversample"):
        raise ValueError(f"method must be 'bin' or 'oversample', got {method!r}")
    lat_edges = np.asarray(lat_edges, dtype="float64")
    lon_edges = np.asarray(lon_edges, dtype="float64")
    ny, nx = lat_edges.size - 1, lon_edges.size - 1
    sums = np.zeros((ny, nx))
    counts = np.zeros((ny, nx))
    for entry in swaths:
        values, a, b = entry
        if method == "bin":
            part = bin_to_grid(values, a, b, lat_edges, lon_edges, min_count=1)
            part_mean, part_count = part["mean"].values, part["count"].values.astype("float64")
        elif method == "oversample":
            part = oversample_to_grid(values, a, b, lat_edges, lon_edges)
            part_mean, part_count = part["mean"].values, part["weight"].values
        else:
            raise ValueError(f"method must be 'bin' or 'oversample', got {method!r}")
        ok = np.isfinite(part_mean)
        sums[ok] += part_mean[ok] * part_count[ok]
        counts[ok] += part_count[ok]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts >= min_count, sums / counts, np.nan)
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    key = "count" if method == "bin" else "weight"
    out = xr.Dataset(
        {"mean": (("y", "x"), mean.astype("float32")), key: (("y", "x"), counts.astype("float32"))},
        coords={"y": lat_c, "x": lon_c},
    )
    return out.rio.write_crs(4326)


# --------------------------------------------------------------------------- model comparison (NO2)
_G = 9.80665              # m/s^2, standard gravity
_M_DRY_AIR = 28.9644e-3   # kg/mol, dry air molar mass
_M_CH4 = 16.04e-3         # kg/mol, methane molar mass


def tm5_pressure_levels(tm5_constant_a: Any, tm5_constant_b: Any, surface_pressure: Any) -> tuple:
    """TM5 hybrid sigma-pressure layer interfaces: ``p = a + b * p_surface``, giving the
    lower and upper boundary pressure of each of NO2's 34 TM5 layers. ``tm5_constant_a``/
    ``tm5_constant_b`` come from the NO2 ``PRODUCT`` group, shape ``(layer, 2)`` (the lower
    and upper interface coefficient for that layer); ``surface_pressure`` is per pixel, Pa.

    Returns ``(p_lower, p_upper)``, each the shape of ``surface_pressure`` plus a trailing
    ``layer`` dimension. Sanity-check against your own file: ``p_lower`` at the bottom layer
    should be close to ``surface_pressure``, and ``p_lower > p_upper`` everywhere.
    """
    a = np.asarray(tm5_constant_a, dtype="float64")  # (layer, 2)
    b = np.asarray(tm5_constant_b, dtype="float64")
    ps = np.asarray(surface_pressure, dtype="float64")[..., None]
    p_lower = a[:, 0] + b[:, 0] * ps
    p_upper = a[:, 1] + b[:, 1] * ps
    return p_lower, p_upper


def apply_averaging_kernel(averaging_kernel: Any, model_vmr: Any, tm5_constant_a: Any,
                           tm5_constant_b: Any, surface_pressure: Any) -> np.ndarray:
    """AK-smoothed model NO2 column, for a fair comparison between a chemical transport
    model's own profile and TROPOMI's reported tropospheric column, which has its own
    vertical sensitivity baked in through the averaging kernel -- comparing it to a raw
    model column is not apples to apples.

    ``model_vmr`` is the model's NO2 volume mixing ratio already interpolated onto the same
    34 TM5 layers as ``averaging_kernel`` (interpolating a model's own vertical grid onto
    TM5's pressure levels, from :func:`tm5_pressure_levels`, is model-specific and left to
    you). Each layer's VMR is converted to a partial column using the hydrostatic relation --
    moles of dry air per m² in a layer of pressure thickness ``dp`` is ``dp / (g * M_air)`` --
    then the averaging-kernel-weighted sum is taken. This is the standard recipe used to
    compare a profile against an AK-smoothed satellite column; confirm against the current
    ATBD if your use case needs an exact match to ESA's own validation numbers.

    Returns the AK-smoothed column in mol/m², directly comparable to
    ``nitrogendioxide_tropospheric_column``.
    """
    ak = np.asarray(averaging_kernel, dtype="float64")
    vmr = np.asarray(model_vmr, dtype="float64")
    p_lower, p_upper = tm5_pressure_levels(tm5_constant_a, tm5_constant_b, surface_pressure)
    dp = p_lower - p_upper                            # layer pressure thickness, Pa (positive)
    partial_column = vmr * dp / (_G * _M_DRY_AIR)      # mol/m^2 per layer
    return np.sum(ak * partial_column, axis=-1)


# --------------------------------------------------------------------------- CH4 point-source emissions
def ch4_enhancement(xch4: Any, *, background: float | None = None) -> np.ndarray:
    """CH4 enhancement above background: ``xch4 - background``. ``background=None`` (the
    default) uses the scene's own median as a quick, scene-relative background -- good enough
    to spot a plume, but use a proper regional or upwind reference (a CAMS/GOSAT climatology,
    or the upwind edge of your own scene) for an emission-rate estimate you intend to
    publish."""
    xch4 = np.asarray(xch4, dtype="float64")
    bg = float(np.nanmedian(xch4)) if background is None else float(background)
    return xch4 - bg


def integrated_mass_enhancement(enhancement_ppb: Any, surface_pressure: Any,
                                pixel_area_m2: Any) -> float:
    """Integrated Mass Enhancement (IME; Varon et al. 2018): total excess CH4 mass in a
    plume, in kg. Converts each pixel's ppb enhancement (from :func:`ch4_enhancement`) to a
    mass-per-area enhancement using the dry-air column implied by ``surface_pressure`` (the
    same hydrostatic relation :func:`co_mixing_ratio` uses in reverse), then sums
    ``mass_per_area * pixel_area`` over the given pixels -- pass only the plume pixels (e.g.
    ``enhancement_ppb[plume_mask]``, with matching ``surface_pressure``/``pixel_area_m2``),
    not the whole scene.
    """
    enh = np.asarray(enhancement_ppb, dtype="float64")
    ps = np.asarray(surface_pressure, dtype="float64")
    dry_air_column = ps / (_G * _M_DRY_AIR)            # mol/m^2
    ch4_column_enhancement = enh * 1e-9 * dry_air_column  # mol/m^2
    mass_per_area = ch4_column_enhancement * _M_CH4    # kg/m^2
    return float(np.nansum(mass_per_area * np.asarray(pixel_area_m2, dtype="float64")))


def estimate_emission_rate(ime_kg: float, wind_speed_m_s: float, plume_length_m: float) -> float:
    """Point-source emission rate from the Integrated Mass Enhancement method (Varon et al.
    2018): ``Q = IME * U_eff / L``, in kg/s. ``wind_speed_m_s`` is the effective wind speed
    at plume height (commonly from ERA5 or GEOS-FP reanalysis, not the satellite itself);
    ``plume_length_m`` is the along-wind plume length, typically measured from the source to
    where the enhancement falls back to background."""
    return float(ime_kg) * float(wind_speed_m_s) / float(plume_length_m)


# --------------------------------------------------------------------------- CO
def co_mixing_ratio(co_column_mol_m2: Any, surface_pressure_pa: Any) -> np.ndarray:
    """Column-averaged CO dry-air mixing ratio (XCO, ppb): the retrieved CO column divided
    by the dry-air column implied by surface pressure (``surface_pressure / (g * M_dry_air)``,
    mol/m²) -- the same hydrostatic relation :func:`apply_averaging_kernel` and
    :func:`integrated_mass_enhancement` use. This ignores water vapour's small effect on the
    dry-air column; fine for most uses, but an exact correction needs a humidity profile this
    module does not read.
    """
    co = np.asarray(co_column_mol_m2, dtype="float64")
    ps = np.asarray(surface_pressure_pa, dtype="float64")
    dry_air_column = ps / (_G * _M_DRY_AIR)
    return (co / dry_air_column) * 1e9


# --------------------------------------------------------------------------- consistency checks
def check_processing_consistency(paths: Sequence[str | PathLike]) -> pd.DataFrame:
    """Flag processing differences across a list of product files that would bias a time
    series if ignored: a processor-version change (NO2 v2.x, for example, retrieves
    noticeably higher columns over polluted areas than v1.x -- do not mix them) and crossing
    the 6 August 2019 ground-processor change that dropped the along-track pixel size from
    7 km to 5.5 km. For trend analysis, use RPRO data plus a consistent OFFL version, not a
    time series that silently straddles either boundary.

    Returns a DataFrame, one row per file (cheap -- only global attributes are read, not
    pixel data), with a ``flag`` column naming any issue found relative to the first file.
    """
    rows = [{"path": str(p), **read_metadata(p)} for p in paths]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time_coverage_start"] = pd.to_datetime(df.get("time_coverage_start"))
    pixel_change = pd.Timestamp(S5P_PIXEL_SIZE_CHANGE, tz="UTC")
    df["pixel_size_era"] = np.where(df["time_coverage_start"] < pixel_change,
                                    "7km (pre-2019-08-06)", "5.5km (post-2019-08-06)")
    base_version = df["processor_version"].iloc[0] if "processor_version" in df.columns else None
    base_era = df["pixel_size_era"].iloc[0]
    flags = []
    for _, row in df.iterrows():
        issues = []
        if base_version is not None and row.get("processor_version") != base_version:
            issues.append(f"processor_version {row.get('processor_version')!r} != first file's {base_version!r}")
        if row["pixel_size_era"] != base_era:
            issues.append(f"pixel size era {row['pixel_size_era']!r} != first file's {base_era!r}")
        flags.append("; ".join(issues))
    df["flag"] = flags
    return df
