"""Sentinel-2 canopy biophysical variables via the BioPAR openEO service.

Estimates leaf area index (LAI), the fraction of absorbed photosynthetically
active radiation (FAPAR), the fraction of vegetation cover (FCOVER), canopy
chlorophyll content (CCC) and canopy water content (CWC) from Sentinel-2 L2A.
Processing runs on the Copernicus Data Space Ecosystem; this module only builds
the batch jobs and downloads the GeoTIFF results.

The retrieval is the INRA-EMMAH neural network in the SNAP Sentinel-2 Toolbox
Level-2 biophysical processor (Weiss and Baret 2016): green, red and NIR
top-of-canopy reflectance plus the sun-view geometry, after a dilated scene-
classification cloud mask. VITO publishes that workflow as the BioPAR
user-defined process:

https://algorithm-catalogue.apex.esa.int/apps/biopar
https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-community-examples/python/BioPAR/biopar_service.html

Needs ``pip install geolattice[rsense-biopar]`` (the openEO client) and a free CDSE account
(https://dataspace.copernicus.eu).
``connect`` signs in with interactive OpenID Connect unless you pass a connection
that is already authenticated.

Example::

    from geolattice.rsense.optical.sentinel2_biophysical import estimate

    paths = estimate(
        {"west": 5.22, "south": 51.16, "east": 5.41, "north": 51.24},
        ("2024-05-01", "2024-05-31"),
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

CDSE_OPENEO = "openeo.dataspace.copernicus.eu"
BIOPAR_PROCESS = "biopar"
BIOPAR_NAMESPACE = (
    "https://raw.githubusercontent.com/ESA-APEx/apex_algorithms/refs/heads/main/"
    "algorithm_catalog/vito/biopar/openeo_udp/biopar.json"
)
REFERENCE = (
    "Weiss, M. and Baret, F. (2016). S2ToolBox Level 2 products: LAI, FAPAR, "
    "FCOVER, version 1.1. http://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf"
)

# Names the BioPAR process accepts. CWY is a common slip for canopy water content.
_ALIASES = {"CWY": "CWC", "CW": "CWC", "FVC": "FCOVER"}


@dataclass(frozen=True)
class BiophysicalVariable:
    """One BioPAR product and the units written by the service."""

    name: str
    long_name: str
    units: str
    description: str


VARIABLES: dict[str, BiophysicalVariable] = {
    "LAI": BiophysicalVariable(
        "LAI", "Leaf area index", "m2 m-2",
        "Half the total green canopy area per unit horizontal ground area."),
    "FAPAR": BiophysicalVariable(
        "FAPAR", "Fraction of absorbed photosynthetically active radiation", "1",
        "Fraction of incoming PAR absorbed by green leaves."),
    "FCOVER": BiophysicalVariable(
        "FCOVER", "Fraction of vegetation cover", "1",
        "Fraction of ground covered by green vegetation."),
    "CCC": BiophysicalVariable(
        "CCC", "Canopy chlorophyll content", "ug cm-2",
        "Total chlorophyll per unit ground area."),
    "CWC": BiophysicalVariable(
        "CWC", "Canopy water content", "g cm-2",
        "Water mass per unit ground area."),
}
DEFAULT_VARIABLES: tuple[str, ...] = ("LAI", "FAPAR", "FCOVER", "CCC", "CWC")


def normalize_variable(name: str) -> str:
    """Return the BioPAR process name for ``name``.

    ``CWY`` and ``CW`` are accepted as canopy water content and sent as ``CWC``,
    which is the name the service enumerates. ``FVC`` is sent as ``FCOVER``.
    """
    key = str(name).strip().upper()
    key = _ALIASES.get(key, key)
    if key not in VARIABLES:
        known = ", ".join(DEFAULT_VARIABLES)
        raise ValueError(f"Unknown biophysical variable {name!r}; choose from {known}")
    return key


def as_spatial_extent(area: Any) -> dict[str, Any]:
    """Bounding box for the BioPAR ``spatial_extent`` parameter.

    A mapping with ``west``, ``south``, ``east`` and ``north`` is passed through,
    including an optional ``crs``. A four-number sequence is ``(west, south, east,
    north)`` in longitude/latitude. Any other area of interest (GeoJSON, a vector
    file, a GeoDataFrame or a shapely geometry) is reduced to the bounding box of
    its EPSG:4326 geometry.
    """
    if isinstance(area, Mapping) and {"west", "south", "east", "north"} <= set(area):
        extent: dict[str, Any] = {k: float(area[k]) for k in ("west", "south", "east", "north")}
        if area.get("crs") is not None:
            extent["crs"] = area["crs"]
        _check_geographic(extent)
        return extent
    if (isinstance(area, Sequence) and not isinstance(area, (str, bytes)) and len(area) == 4):
        try:
            west, south, east, north = (float(v) for v in area)
        except (TypeError, ValueError):
            pass
        else:
            extent = {"west": west, "south": south, "east": east, "north": north}
            _check_geographic(extent)
            return extent
    from shapely.geometry import shape

    from ..access.stac import aoi_to_geojson

    west, south, east, north = shape(aoi_to_geojson(area)).bounds
    return {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}


def _check_geographic(extent: Mapping[str, Any]) -> None:
    if "crs" in extent:
        return
    if max(abs(extent["west"]), abs(extent["east"]), abs(extent["south"]), abs(extent["north"])) > 180:
        raise ValueError(
            "Bounding box looks projected. Pass longitude/latitude, or set crs "
            "(for example 32631) on the extent mapping.")
    if extent["west"] >= extent["east"] or extent["south"] >= extent["north"]:
        raise ValueError("spatial extent must have west < east and south < north")


def as_temporal_extent(dates: Any) -> list[str]:
    """``(start, end)`` or ``'start/end'`` as the two-item interval the service expects."""
    if isinstance(dates, str) and "/" in dates:
        start, end = dates.split("/", 1)
        return [start, end]
    if isinstance(dates, Sequence) and not isinstance(dates, (str, bytes)) and len(dates) == 2:
        return [str(dates[0]), str(dates[1])]
    raise TypeError("temporal extent must be (start, end) or 'start/end'")


def connect(url: str = CDSE_OPENEO, *, authenticate: bool = True):
    """Connect to the Copernicus Data Space openEO backend.

    With ``authenticate=True`` (the default) this opens a browser or device-code
    prompt via ``authenticate_oidc``. Pass ``authenticate=False`` to attach your
    own authentication before calling :func:`estimate`.
    """
    try:
        import openeo
    except ImportError as e:
        raise ImportError(
            "Sentinel-2 BioPAR needs the openEO client: pip install geolattice[rsense-biopar]") from e
    connection = openeo.connect(url)
    if authenticate:
        connection = connection.authenticate_oidc()
    return connection


def biopar_cube(connection: Any, variable: str, spatial_extent: Any, temporal_extent: Any, *,
                namespace: str = BIOPAR_NAMESPACE):
    """openEO data cube for one BioPAR variable. Does not start a job."""
    return connection.datacube_from_process(
        BIOPAR_PROCESS,
        namespace=namespace,
        temporal_extent=as_temporal_extent(temporal_extent),
        spatial_extent=as_spatial_extent(spatial_extent),
        biopar_type=normalize_variable(variable),
    )


def estimate(spatial_extent: Any, temporal_extent: Any,
             variables: Sequence[str] | None = None, *,
             out_dir: str | PathLike = "biopar_results", connection: Any = None,
             out_format: str = "GTiff", namespace: str = BIOPAR_NAMESPACE,
             url: str = CDSE_OPENEO) -> dict[str, list[Path]]:
    """Estimate Sentinel-2 biophysical variables and download one GeoTIFF per date.

    One CDSE batch job is submitted for each variable, in order. Results are
    written to ``out_dir/<variable>/``. The return value maps each variable name
    to the files downloaded for it.

    ``variables`` defaults to LAI, FAPAR, FCOVER, CCC and CWC. ``CWY`` is read
    as CWC. ``connection`` may be an authenticated openEO connection; otherwise
    :func:`connect` signs in against ``url``.
    """
    if connection is None:
        connection = connect(url)
    requested = DEFAULT_VARIABLES if variables is None else variables
    if isinstance(requested, str):
        requested = (requested,)
    names = [normalize_variable(v) for v in requested]
    root = Path(out_dir)
    downloaded: dict[str, list[Path]] = {}
    for name in names:
        cube = biopar_cube(connection, name, spatial_extent, temporal_extent, namespace=namespace)
        job = cube.create_job(out_format=out_format, title=f"Sentinel2_BioPAR_{name}")
        job.start_and_wait()
        target = root / name
        paths = job.get_results().download_files(target=str(target))
        downloaded[name] = [Path(p) for p in paths]
    return downloaded


def read(paths: str | PathLike | Sequence[str | PathLike], *,
         variable: str | None = None):
    """Stack downloaded BioPAR GeoTIFFs into one ``xarray.DataArray`` along ``time``.

    ``paths`` may be a list of files or a directory (only ``*.tif`` / ``*.tiff``).
    Dates are taken from names such as ``openEO_2024-05-07Z.tif``. When ``variable``
    is set, the array is renamed and given that variable's units.
    """
    import re

    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    files = _tiff_paths(paths)
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files in {paths}")
    arrays = []
    times = []
    for path in files:
        da = xr.open_dataarray(path, engine="rasterio", mask_and_scale=True)
        if "band" in da.dims and da.sizes.get("band") == 1:
            da = da.squeeze("band", drop=True)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        times.append(match.group(1) if match else path.stem)
        arrays.append(da)
    stacked = xr.concat(arrays, dim="time")
    stacked = stacked.assign_coords(time=("time", times))
    if variable is not None:
        key = normalize_variable(variable)
        info = VARIABLES[key]
        stacked = stacked.rename(key).assign_attrs(
            long_name=info.long_name, units=info.units, reference=REFERENCE)
    return stacked.sortby("time")


def _tiff_paths(paths: str | PathLike | Sequence[str | PathLike]) -> list[Path]:
    if isinstance(paths, (str, PathLike)):
        folder = Path(paths)
        if folder.is_dir():
            found = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
            return found
        return [folder]
    return [Path(p) for p in paths]


__all__ = [
    "BIOPAR_NAMESPACE",
    "BIOPAR_PROCESS",
    "CDSE_OPENEO",
    "DEFAULT_VARIABLES",
    "REFERENCE",
    "VARIABLES",
    "BiophysicalVariable",
    "as_spatial_extent",
    "as_temporal_extent",
    "biopar_cube",
    "connect",
    "estimate",
    "normalize_variable",
    "read",
]
