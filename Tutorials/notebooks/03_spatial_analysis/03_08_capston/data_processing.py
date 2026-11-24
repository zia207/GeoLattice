"""
Shared data preparation and helper functions for the Module 03.08 capstones.

The five capstone notebooks in this folder import from this file so that the
notebooks stay focused on analysis:

    03_08_01_flood_susceptibility_mapping.ipynb     Sylhet haor basin, Bangladesh
    03_08_02_watershed_characterization.ipynb       Onondaga Lake watershed, NY
    03_08_03_agricultural_suitability.ipynb         Rajshahi / Barind Tract
    03_08_04_landslide_susceptibility.ipynb         Western Great Plains, USA
    03_08_05_habitat_suitability.ipynb              Huanglong, Shaanxi, China

Contents
--------
1. Repository paths                 find_repo_root, capstone_dir
2. Raster I/O                       read_band, write_band
3. Study-grid preparation           prepare_onondaga_dem,
                                    AnalysisGrid, warp_to_grid, block_fraction,
                                    rasterize_geometries
4. Drainage-network helpers         d8_receivers, strahler_order,
                                    flow_distance_to_targets, subbasin_outlets
5. External data register           EXTERNAL_DATA, check_external_data

Raster reading and writing go through ``geolattice.raster`` so every output is a
GeoTIFF that the other GeoLattice tools can open directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

NODATA = -9999.0

# D8 neighbour order and pointer codes used by geolattice.spatial_analysis
# (E, SE, S, SW, W, NW, N, NE  ->  1, 2, 4, 8, 16, 32, 64, 128).
D8_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
D8_CODES = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# 1. Repository paths
# ---------------------------------------------------------------------------


def find_repo_root(start: Optional[PathLike] = None) -> Path:
    """Walk up from ``start`` until the GeoLattice repository root is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        has_data = (candidate / "Data").is_dir()
        has_pkg = (candidate / "geolattice").is_dir() or (candidate / "src" / "spatial_analysis").is_dir()
        if has_data and has_pkg:
            return candidate
    raise FileNotFoundError("Could not find the GeoLattice repository root.")


def capstone_dir(root: PathLike, case: str) -> Path:
    """Return (and create) ``Data/03_spatial_analysis/capstone_spatial_analysis/<case>``."""
    out = Path(root) / "Data" / "03_spatial_analysis" / "capstone_spatial_analysis" / case
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# 2. Raster I/O
# ---------------------------------------------------------------------------


def read_band(path: PathLike, band: int = 1) -> Tuple[np.ndarray, Tuple[float, ...], Any]:
    """Read one band as float64 with nodata set to NaN.

    Returns ``(array, geotransform, crs)``.
    """
    from geolattice.raster import open_raster_data

    with open_raster_data(path) as ds:
        arr = np.asarray(ds.read(bands=band), dtype=np.float64)
        nodata = ds.get_band(band).nodata
        gt = tuple(ds.geotransform)
        crs = ds.crs
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, gt, crs


def write_band(
    path: PathLike,
    array: np.ndarray,
    like: PathLike,
    *,
    nodata: float = NODATA,
    dtype: str = "float32",
) -> str:
    """Write a 2-D array as a GeoTIFF on the grid of the raster ``like``."""
    from geolattice.raster import write_raster_data

    _, gt, crs = read_band(like)
    out = np.where(np.isfinite(array), array, nodata).astype(dtype)
    return write_raster_data(path, out, crs=crs, geotransform=gt, nodata=nodata, overwrite=True)


def extent_of(path: PathLike) -> Tuple[float, float, float, float]:
    """Matplotlib ``imshow`` extent ``(xmin, xmax, ymin, ymax)`` of a raster."""
    arr, gt, _ = read_band(path)
    rows, cols = arr.shape
    return (gt[0], gt[0] + gt[1] * cols, gt[3] + gt[5] * rows, gt[3])


def xy_to_rowcol(x: float, y: float, gt: Sequence[float]) -> Tuple[int, int]:
    """Map coordinates to the (row, col) of the cell that contains them."""
    return int((y - gt[3]) / gt[5]), int((x - gt[0]) / gt[1])


def rowcol_to_xy(row: int, col: int, gt: Sequence[float]) -> Tuple[float, float]:
    """Cell-centre map coordinates of (row, col)."""
    return gt[0] + (col + 0.5) * gt[1], gt[3] + (row + 0.5) * gt[5]


# ---------------------------------------------------------------------------
# 3. Study-grid preparation
# ---------------------------------------------------------------------------

ONONDAGA_CRS = "EPSG:26918"  # NAD83 / UTM zone 18N, metres


def prepare_onondaga_dem(root: PathLike, *, resolution: float = 100.0, overwrite: bool = False) -> Path:
    """Project the Onondaga County DEM to UTM 18N on a square metric grid.

    The source DEM (``Data/00_shared/raster_data/USA/Onondaga_DEM.tif``) is in geographic
    coordinates with cells of roughly 92 m x 126 m. Slope, flow length and
    area all need metric, square cells, so the capstones work on a 100 m
    UTM grid. The result is cached in
    ``Data/03_spatial_analysis/capstone_spatial_analysis/onondaga/dem_utm.tif``.
    """
    from geolattice.raster import reproject_raster

    root = Path(root)
    src = root / "Data" / "00_shared" / "raster_data" / "USA" / "Onondaga_DEM.tif"
    if not src.exists():
        raise FileNotFoundError(f"Missing DEM: {src}")
    dst = capstone_dir(root, "onondaga") / "dem_utm.tif"
    if dst.exists() and not overwrite:
        return dst
    reproject_raster(src, dst, ONONDAGA_CRS, resample="bilinear", resolution=float(resolution), overwrite=True)
    return dst



@dataclass(frozen=True)
class AnalysisGrid:
    """A north-up raster grid: bounds ``(xmin, ymin, xmax, ymax)``, cell size and CRS."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float
    crs: str

    @property
    def shape(self) -> Tuple[int, int]:
        return int(round((self.ymax - self.ymin) / self.res)), int(round((self.xmax - self.xmin) / self.res))

    @property
    def geotransform(self) -> Tuple[float, ...]:
        return (self.xmin, self.res, 0.0, self.ymax, 0.0, -self.res)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Matplotlib ``imshow`` extent."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @classmethod
    def around(cls, bounds: Sequence[float], res: float, crs: str) -> "AnalysisGrid":
        """Smallest grid with cells of ``res`` that are aligned to multiples of ``res`` and cover ``bounds``."""
        x0, y0, x1, y1 = bounds
        return cls(math.floor(x0 / res) * res, math.floor(y0 / res) * res,
                   math.ceil(x1 / res) * res, math.ceil(y1 / res) * res, float(res), crs)

    def write(self, path: PathLike, array: np.ndarray, *, nodata: float = NODATA, dtype: str = "float32") -> str:
        """Write a 2-D array on this grid as a GeoTIFF (NaN becomes ``nodata``)."""
        from geolattice.raster import write_raster_data

        out = np.where(np.isfinite(array), array, nodata).astype(dtype)
        return write_raster_data(path, out, crs=self.crs, geotransform=self.geotransform, nodata=nodata,
                                 overwrite=True)


def warp_to_grid(
    source: Any,
    grid: AnalysisGrid,
    *,
    resample: str = "near",
    src_srs: Optional[str] = None,
    res: Optional[float] = None,
) -> np.ndarray:
    """Reproject and resample a raster onto ``grid`` and return a float array with NaN for nodata.

    ``source`` is a path or an open GDAL dataset. ``resample`` is a GDAL name:
    ``near`` for classes, ``bilinear`` for continuous layers, ``average`` when
    going to coarser cells. ``src_srs`` overrides a missing or broken source CRS.
    ``res`` warps to a finer cell size over the same bounds (used by
    :func:`block_fraction`).
    """
    from osgeo import gdal

    gdal.UseExceptions()
    cell = float(res or grid.res)
    ds = gdal.Warp("", str(source) if isinstance(source, (str, Path)) else source, format="MEM",
                   dstSRS=grid.crs, srcSRS=src_srs, outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
                   xRes=cell, yRes=cell, resampleAlg=resample, dstNodata=NODATA, outputType=gdal.GDT_Float32)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    arr[arr == NODATA] = np.nan
    return arr


def block_fraction(mask: np.ndarray, like: PathLike, grid: AnalysisGrid, *, sub: float = 50.0) -> np.ndarray:
    """Share of a fine binary ``mask`` (on the grid of raster ``like``) inside each cell of ``grid``.

    The mask is first resampled by nearest neighbour to ``sub`` metre cells,
    which divide ``grid.res`` exactly, and then averaged in blocks. NaN in the
    mask is ignored; cells with no valid fine pixels are NaN.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    k = int(round(grid.res / sub))
    if abs(k * sub - grid.res) > 1e-6:
        raise ValueError("sub must divide the grid cell size exactly")
    src = gdal.Open(str(like))
    mem = gdal.GetDriverByName("MEM").Create("", mask.shape[1], mask.shape[0], 1, gdal.GDT_Float32)
    mem.SetGeoTransform(src.GetGeoTransform())
    mem.SetProjection(src.GetProjection())
    band = mem.GetRasterBand(1)
    band.WriteArray(np.where(np.isfinite(mask), mask, NODATA).astype(np.float32))
    band.SetNoDataValue(NODATA)
    fine = warp_to_grid(mem, grid, resample="near", res=sub)
    rows, cols = grid.shape
    blocks = fine.reshape(rows, k, cols, k)
    valid = np.isfinite(blocks).sum(axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.where(valid > 0, total / np.maximum(valid, 1), np.nan)


def rasterize_geometries(geoms: Iterable[Any], grid: AnalysisGrid, *, all_touched: bool = True) -> np.ndarray:
    """Burn shapely geometries (already in ``grid.crs``) into a 0/1 uint8 array on ``grid``."""
    from rasterio import features, transform

    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=np.uint8)
    tr = transform.from_origin(grid.xmin, grid.ymax, grid.res, grid.res)
    return features.rasterize(shapes, out_shape=grid.shape, transform=tr, all_touched=all_touched, dtype="uint8")


# ---------------------------------------------------------------------------
# 4. Drainage-network helpers
# ---------------------------------------------------------------------------


def d8_receivers(pointer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flat index of the downstream cell (-1 if none) and step length in cells.

    ``pointer`` holds the GeoLattice / Whitebox-style D8 codes (NaN or 0 = no flow).
    """
    rows, cols = pointer.shape
    recv = np.full(rows * cols, -1, dtype=np.int64)
    step = np.zeros(rows * cols, dtype=np.float64)
    rr, cc = np.indices((rows, cols))
    flat = np.arange(rows * cols).reshape(rows, cols)
    for (dr, dc), code in zip(D8_OFFSETS, D8_CODES):
        m = pointer == code
        r2, c2 = rr + dr, cc + dc
        ok = m & (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
        recv[flat[ok]] = flat[r2[ok], c2[ok]]
        step[flat[ok]] = math.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
    return recv, step


def strahler_order(streams: PathLike, pointer: PathLike, dem: PathLike, destination: PathLike) -> str:
    """Strahler stream order for every stream cell.

    Head cells get order 1. Where two or more streams of the same highest
    order meet, the order increases by one; otherwise the highest order
    continues downstream. ``dem`` must be the conditioned DEM that produced
    ``pointer`` so cells can be visited from upstream to downstream.
    """
    st, _, _ = read_band(streams)
    ptr, _, _ = read_band(pointer)
    z, _, _ = read_band(dem)
    is_stream = (np.isfinite(st) & (st > 0)).ravel()
    recv, _ = d8_receivers(ptr)

    n = is_stream.size
    order = np.zeros(n, dtype=np.int16)
    max_in = np.zeros(n, dtype=np.int16)  # highest order arriving from upstream
    n_max = np.zeros(n, dtype=np.int16)  # how many tributaries carry that order

    idx = np.where(is_stream)[0]
    idx = idx[np.argsort(-z.ravel()[idx], kind="stable")]  # upstream first
    for i in idx:
        if max_in[i] == 0:
            order[i] = 1
        elif n_max[i] >= 2:
            order[i] = max_in[i] + 1
        else:
            order[i] = max_in[i]
        j = recv[i]
        if j >= 0 and is_stream[j]:
            if order[i] > max_in[j]:
                max_in[j] = order[i]
                n_max[j] = 1
            elif order[i] == max_in[j]:
                n_max[j] += 1

    out = order.reshape(st.shape).astype(np.float64)
    out[out == 0] = np.nan
    return write_band(destination, out, streams, nodata=0.0)


def flow_distance_to_targets(
    pointer: PathLike,
    dem: PathLike,
    targets: np.ndarray,
    destination: Optional[PathLike] = None,
) -> np.ndarray:
    """Along-flow distance (map units) from every cell to the first target cell.

    ``targets`` is a boolean array on the pointer grid (for example the outlet
    cell of each sub-basin). Cells that never reach a target are NaN. The
    maximum of this surface inside a basin is the basin's longest flow path.
    """
    ptr, gt, _ = read_band(pointer)
    z, _, _ = read_band(dem)
    cell = 0.5 * (abs(gt[1]) + abs(gt[5]))
    recv, step = d8_receivers(ptr)

    dist = np.full(ptr.size, np.nan)
    tflat = np.asarray(targets, dtype=bool).ravel()
    dist[tflat] = 0.0
    valid = np.isfinite(z).ravel()
    idx = np.where(valid)[0]
    idx = idx[np.argsort(z.ravel()[idx], kind="stable")]  # downstream first
    for i in idx:
        if tflat[i]:
            continue
        j = recv[i]
        if j >= 0 and np.isfinite(dist[j]):
            dist[i] = dist[j] + step[i] * cell
    out = dist.reshape(ptr.shape)
    if destination is not None:
        write_band(destination, out, pointer)
    return out


def subbasin_outlets(
    flow_accumulation: PathLike,
    pointer: PathLike,
    *,
    threshold_cells: float,
    mask: Optional[np.ndarray] = None,
) -> list:
    """Outlet points that split a watershed into sub-basins at major confluences.

    The "major network" is every cell whose flow accumulation is at least
    ``threshold_cells`` (inside ``mask`` when given). Wherever two or more
    major branches join, the last cell of each branch becomes a sub-basin
    outlet. Passing the returned ``(x, y)`` list, together with the watershed
    outlet, to ``geolattice.spatial_analysis.watershed`` yields one sub-basin per
    major tributary plus the inter-basin areas along the main stem.
    """
    fa, gt, _ = read_band(flow_accumulation)
    ptr, _, _ = read_band(pointer)
    major = np.isfinite(fa) & (fa >= float(threshold_cells))
    if mask is not None:
        major &= np.asarray(mask, dtype=bool)
    recv, _ = d8_receivers(ptr)
    major_flat = major.ravel()
    donors = np.zeros(major_flat.size, dtype=np.int32)
    idx = np.where(major_flat)[0]
    down = recv[idx]
    ok = (down >= 0) & major_flat[np.clip(down, 0, None)]
    np.add.at(donors, down[ok], 1)
    cols = fa.shape[1]
    outlets = []
    for i, j in zip(idx[ok], down[ok]):
        if donors[j] >= 2:  # i is the last cell of a branch entering a confluence
            outlets.append(rowcol_to_xy(int(i // cols), int(i % cols), gt))
    return outlets


# ---------------------------------------------------------------------------
# 5. External data register
# ---------------------------------------------------------------------------

# Layers that are not shipped in the repository. Each entry gives the file the
# notebooks look for (relative to Data/03_spatial_analysis/capstone_spatial_analysis/) and where the
# data comes from. Files marked "download_data.py" are fetched by that script;
# the others need a manual export. Notebooks call check_external_data() and skip
# the optional sections that depend on a missing file.
EXTERNAL_DATA: Dict[str, Dict[str, str]] = {
    # Onondaga County, NY
    "onondaga_worldcover": {"file": "onondaga/worldcover_2021.tif", "source": "ESA WorldCover 2021, 10 m (download_data.py)"},
    "onondaga_fema": {"file": "onondaga/fema_nfhl_flood_zones.gpkg", "source": "FEMA NFHL flood hazard zones (download_data.py)"},
    "onondaga_roads": {"file": "onondaga/osm_roads.gpkg", "source": "OpenStreetMap roads (download_data.py)"},
    "onondaga_nlcd": {"file": "onondaga/nlcd_landcover.tif", "source": "NLCD land cover, 30 m (USGS / MRLC)"},
    "onondaga_hsg": {"file": "onondaga/soil_hydrologic_group.tif", "source": "gSSURGO hydrologic soil group, coded A=1..D=4 (manual)"},
    # Chittagong Hill Tracts
    "cht_dem": {"file": "cht/dem_glo30.tif", "source": "Copernicus GLO-30 DEM (download_data.py)"},
    "cht_worldcover": {"file": "cht/worldcover_2021.tif", "source": "ESA WorldCover 2021, 10 m (download_data.py)"},
    "cht_chirps": {"file": "cht/chirps_annual_2011_2020.tif", "source": "CHIRPS v2.0 annual rainfall (download_data.py)"},
    "cht_landslides": {"file": "cht/landslides_coolr.gpkg", "source": "NASA COOLR landslide points (download_data.py)"},
    "cht_gbif": {"file": "cht/gbif_elephas_maximus.csv", "source": "GBIF occurrences (download_data.py)"},
    "cht_roads": {"file": "cht/osm_roads.gpkg", "source": "OpenStreetMap roads (download_data.py)"},
    "cht_waterways": {"file": "cht/osm_waterways.gpkg", "source": "OpenStreetMap waterways (download_data.py)"},
    "cht_protected": {"file": "cht/protected_areas.gpkg", "source": "WDPA protected areas (manual)"},
    "cht_lithology": {"file": "cht/lithology.gpkg", "source": "Lithology and faults (manual)"},
    # Rajshahi / Barind Tract
    "rajshahi_worldcover": {"file": "rajshahi/worldcover_2021.tif", "source": "ESA WorldCover 2021, 10 m (download_data.py)"},
    "rajshahi_tavg": {"file": "rajshahi/worldclim_tavg.tif", "source": "WorldClim 2.1 mean temperature (download_data.py)"},
    "rajshahi_tmin": {"file": "rajshahi/worldclim_tmin.tif", "source": "WorldClim 2.1 minimum temperature (download_data.py)"},
    "rajshahi_prec": {"file": "rajshahi/worldclim_prec.tif", "source": "WorldClim 2.1 precipitation (download_data.py)"},
    "rajshahi_chirps": {"file": "rajshahi/chirps_annual_2011_2020.tif", "source": "CHIRPS v2.0 annual rainfall (download_data.py)"},
}


def check_external_data(root: PathLike, keys: Optional[Iterable[str]] = None) -> Dict[str, Optional[Path]]:
    """Return ``{key: path or None}`` for the external layers a notebook needs."""
    base = Path(root) / "Data" / "03_spatial_analysis" / "capstone_spatial_analysis"
    found: Dict[str, Optional[Path]] = {}
    for key in keys or EXTERNAL_DATA:
        path = base / EXTERNAL_DATA[key]["file"]
        found[key] = path if path.exists() else None
    return found
