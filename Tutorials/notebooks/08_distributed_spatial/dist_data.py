"""Data helpers for the distributed and GPU spatial computing chapter (08).

Every notebook in the chapter reads its inputs through these functions, so it runs on its own.
Most data are already in the repository's ``Data/`` folder (Great Plains rasters and points, the
2025 10 m Cropland Data Layer, NYC taxi trips and zones). Two datasets are downloaded on first use
and cached under ``Data/08_distributed_spatial/``:

* Microsoft Building Footprints for Kansas, from the Planetary Computer ``ms-buildings``
  collection (GeoParquet partitioned by level 9 quadkey, read in parallel);
* the OpenStreetMap drivable road network of Wichita, Kansas, plus its hospitals and schools,
  from OSMnx (public Overpass servers are tried in turn, since they are often busy).
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

CHAPTER = Path(__file__).resolve().parent
ROOT = CHAPTER.parents[2]
DATA = ROOT / "Data"
CACHE = DATA / "08_distributed_spatial"
SCRATCH = CHAPTER / "outputs"

GP_RASTERS = DATA / "03_spatial_analysis" / "Great_plain" / "Raster"
GP_POINTS = DATA / "00_shared" / "data_geostatistics" / "Data_GP" / "GP_POINT_SOC.shp"
GP_COUNTIES = DATA / "00_shared" / "data_geostatistics" / "Data_GP" / "GP_COUNTY.shp"
GP_STATES = DATA / "03_spatial_analysis" / "Great_plain" / "GP_STATE.shp"
CDL = (DATA / "08_remote_sensing" / "08_11_Capstones" / "NISAR_crop_classification" /
       "2025_10m_cdls" / "2025_10m_cdls.tif")
TAXI = DATA / "06_spatialDB" / "yellow_tripdata_2023.parquet"
TAXI_ZONES = DATA / "06_spatialDB" / "taxi_zones" / "taxi_zones.shp"
BUILDINGS = CACHE / "buildings_kansas"
ROADS = CACHE / "wichita_roads.parquet"
FACILITIES = CACHE / "wichita_facilities.parquet"

# The 15 Great Plains covariates (file stem -> short description), all 90 m, EPSG:5070
COVARIATES = {
    "elev": "elevation (m)", "slope": "slope (degrees)", "aspect": "aspect (degrees)",
    "h_curvature": "horizontal curvature", "v_curvature": "vertical curvature",
    "tpi_2000": "topographic position index", "topo_2000": "topographic class",
    "slopeleng": "slope length", "ndvi": "NDVI", "nlcd": "NLCD land cover",
    "preci": "mean annual precipitation (mm)", "temp": "mean annual temperature (C)",
    "frg": "forest fragmentation class", "silt_clay": "silt plus clay (%)",
    "kfactor": "soil erodibility K",
}
# Continuous covariates used as predictors of soil organic carbon
PREDICTORS = ["elev", "slope", "aspect", "h_curvature", "v_curvature", "tpi_2000", "slopeleng",
              "ndvi", "preci", "temp", "silt_clay", "kfactor"]

# Cropland Data Layer codes grouped for county summaries (USDA CDL legend)
CDL_GROUPS = {
    "Corn": [1, 12, 13, 225, 226, 228, 237, 241],
    "Sorghum": [4, 236],
    "Soybeans": [5, 26, 239, 240, 254],
    "Wheat": [22, 23, 24, 230, 234, 236, 238],
    "Other crops": [2, 3, 6, 10, 11, 14, 21, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                    39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 204, 205, 206, 207, 208,
                    209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224,
                    227, 229, 231, 232, 233, 235, 242, 243, 244, 245, 246, 247, 248, 249, 250],
    "Fallow": [61],
    "Grass and shrubs": [62, 152, 176],
    "Forest": [63, 141, 142, 143],
    "Developed": [82, 121, 122, 123, 124],
    "Water and wetland": [83, 87, 92, 111, 190, 195],
    "Barren and other": [64, 65, 81, 88, 112, 131],
}
CROP_GROUPS = ["Corn", "Sorghum", "Soybeans", "Wheat", "Other crops"]
GROUP_COLORS = {"No data": "#ffffff", "Corn": "#ffd300", "Sorghum": "#ff9e0a", "Soybeans": "#267000",
                "Wheat": "#a57000", "Other crops": "#e2007c", "Fallow": "#c7d79e",
                "Grass and shrubs": "#e8ffbf", "Forest": "#38814e", "Developed": "#9c9c9c",
                "Water and wetland": "#4970a3", "Barren and other": "#ccbfa3"}


# --------------------------------------------------------------------------- utilities
@contextmanager
def timer(label: str, results: dict | None = None):
    """Print (and optionally store) the wall-clock seconds of a ``with`` block."""
    t = time.perf_counter()
    yield
    s = time.perf_counter() - t
    if results is not None:
        results[label] = s
    print(f"{label}: {s:.2f} s")


def gb(nbytes: float) -> str:
    """Bytes as a readable GB string."""
    return f"{nbytes / 1e9:.2f} GB"


def rss_gb() -> float:
    """Resident memory of this Python process in GB."""
    import psutil

    return psutil.Process().memory_info().rss / 1e9


class PeakMemory:
    """Context manager that samples this process's resident memory in a background thread.

    ``with PeakMemory() as pm: ...`` then ``pm.peak_gb`` (highest RSS) and ``pm.start_gb``.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_gb = self.start_gb = 0.0

    def __enter__(self):
        import threading

        self.start_gb = self.peak_gb = rss_gb()
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                self.peak_gb = max(self.peak_gb, rss_gb())
                self._stop.wait(self.interval)

        self._t = threading.Thread(target=sample, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak_gb = max(self.peak_gb, rss_gb())


def quicklook(da, step: int = 20):
    """Every ``step``-th pixel of a (y, x) raster, computed, for plotting."""
    return da[..., ::step, ::step].compute() if hasattr(da.data, "dask") else da[..., ::step, ::step]


def extent(da) -> list[float]:
    """Matplotlib ``imshow`` extent of a raster."""
    x, y = da.x.values, da.y.values
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    return [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]


# --------------------------------------------------------------------------- Great Plains
_GRID: tuple | None = None


def gp_grid():
    """``(x, y)`` pixel-centre coordinates of the reference grid (the elevation raster)."""
    global _GRID
    if _GRID is None:
        import rioxarray

        r = rioxarray.open_rasterio(GP_RASTERS / "elev.tif")  # header only
        _GRID = (r.x.values, r.y.values)
    return _GRID


def gp_raster(name: str = "elev", align: bool = True, **kwargs):
    """One Great Plains covariate as a lazy DataArray (``pdist.read_raster``).

    The 15 files are not on exactly the same grid: origins differ by up to 38 m (less than half
    a 90 m pixel) and some have one row or column more. With ``align=True`` every layer is
    snapped to the elevation grid by nearest pixel (pixels that fall outside the file become
    NaN), so layers can be combined pixel by pixel.
    """
    import geolattice.distributed as pdist

    chunks = kwargs.setdefault("chunks", {"y": 2048, "x": 2048})
    da = pdist.read_raster(GP_RASTERS / f"{name}.tif", **kwargs).rename(name)
    if not align:
        return da
    x, y = gp_grid()
    if da.sizes["x"] == len(x) and da.sizes["y"] == len(y) and np.allclose(da.x, x) and np.allclose(da.y, y):
        return da.assign_coords(x=x, y=y)  # same grid up to rounding: use identical labels
    fx = (x - float(da.x[0])) / 90.0  # fractional source column of each target column
    fy = (float(da.y[0]) - y) / 90.0  # fractional source row of each target row
    ix, iy = np.rint(fx).astype(int), np.rint(fy).astype(int)  # nearest source pixel
    if max(np.abs(fx - ix).max(), np.abs(fy - iy).max()) > 0.5:
        raise ValueError(f"{name} is not within half a pixel of the reference grid")
    if not (np.all(np.diff(ix) == 1) and np.all(np.diff(iy) == 1)):
        raise ValueError(f"{name} does not have the reference resolution")
    out = da
    for dim, idx in (("x", ix), ("y", iy)):  # a whole-pixel shift: slice, then pad with NaN
        n_src, n_ref = da.sizes[dim], len(idx)
        lo, hi = max(idx[0], 0), min(idx[0] + n_ref, n_src)  # source pixels that fall on the grid
        out = out.isel({dim: slice(lo, hi)})
        before, after = lo - idx[0], idx[0] + n_ref - hi  # target pixels beyond the file
        if before or after:
            out = out.pad({dim: (before, after)})
    out = out.assign_coords(x=x, y=y)
    if isinstance(chunks, dict):
        out = out.chunk(chunks)
    return out.rename(name)


def gp_stack(names=None, chunks=None):
    """The Great Plains covariates as one lazy ``(band, y, x)`` DataArray."""
    import xarray as xr

    names = list(names or PREDICTORS)
    chunks = chunks or {"y": 2048, "x": 2048}
    layers = [gp_raster(n, chunks=chunks).astype("float32") for n in names]
    return xr.concat(layers, dim="band").assign_coords(band=names).rename("covariates")


def gp_points():
    import geopandas as gpd

    return gpd.read_file(GP_POINTS)


def gp_counties():
    import geopandas as gpd

    g = gpd.read_file(GP_COUNTIES)
    g["FIPS"] = g["FIPS"].astype(int).astype(str).str.zfill(5)
    return g


def gp_states():
    import geopandas as gpd

    return gpd.read_file(GP_STATES)[["STATE", "geometry"]]


def cdl(**kwargs):
    """The 2025 10 m CONUS Cropland Data Layer as a lazy uint8 DataArray (0 = no data)."""
    import geolattice.distributed as pdist

    kwargs.setdefault("chunks", {"y": 8192, "x": 8192})
    return pdist.read_raster(CDL, masked=False, **kwargs).rename("cdl")


def cdl_groups_plot(ax, codes, extent=None, legend=True):
    """Draw a CDL code array as land cover groups with the USDA-like group colours."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    grp = cdl_lookup()[np.asarray(codes, dtype=np.uint8)]
    names = list(GROUP_COLORS)
    ax.imshow(grp, cmap=ListedColormap(list(GROUP_COLORS.values())), vmin=-0.5,
              vmax=len(names) - 0.5, interpolation="nearest", extent=extent)
    if legend:
        present = np.unique(grp)
        ax.legend(handles=[Patch(color=GROUP_COLORS[names[i]], label=names[i]) for i in present if i > 0],
                  loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)


def cdl_lookup() -> np.ndarray:
    """256-entry table mapping each CDL code to its group number (0 = no data or unmapped)."""
    lut = np.zeros(256, dtype=np.uint8)
    for i, codes in enumerate(CDL_GROUPS.values(), start=1):
        lut[codes] = i
    return lut


# --------------------------------------------------------------------------- NYC taxi
def taxi_zones(crs: str | int | None = None):
    import geopandas as gpd

    z = gpd.read_file(TAXI_ZONES)[["LocationID", "zone", "borough", "geometry"]]
    z = z.dissolve(by="LocationID", aggfunc="first").reset_index()  # one row per zone id
    return z.to_crs(crs) if crs else z


def taxi_trips(columns=None, lazy: bool = True):
    """The 38.3 million 2023 yellow taxi trips (Dask DataFrame when ``lazy``)."""
    if lazy:
        import dask.dataframe as dd

        return dd.read_parquet(TAXI, columns=columns, split_row_groups=True)
    return pd.read_parquet(TAXI, columns=columns)


# --------------------------------------------------------------------------- Kansas buildings
KANSAS_BBOX = (-102.06, 36.99, -94.58, 40.01)


def _ms_buildings_files():
    """Signed file system and the US ``ms-buildings`` quadkey partitions touching Kansas."""
    import adlfs
    import mercantile
    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=pc.sign_inplace)
    item = cat.get_collection("ms-buildings").get_item("United-States_2022-07-06")
    asset = item.assets["data"]
    fs = adlfs.AzureBlobFileSystem(**asset.extra_fields["table:storage_options"])
    root = asset.href.replace("abfs://", "")
    keys = {mercantile.quadkey(t) for t in mercantile.tiles(*KANSAS_BBOX, zooms=9)}
    parts = [p for p in fs.ls(root) if p.rsplit("=", 1)[-1] in keys]
    files = [f for p in parts for f in fs.ls(p) if f.endswith(".parquet")]
    return fs, files


def _read_part(fs, path):
    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely

    with fs.open(path) as f:
        t = pq.read_table(f, columns=["geometry"])
    return gpd.GeoDataFrame(geometry=shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)),
                            crs=4326)


def kansas_buildings(download: bool = True, verbose: bool = True):
    """Microsoft Building Footprints inside Kansas (EPSG:4326), as a GeoDataFrame.

    The first call reads the Kansas quadkey partitions of ``ms-buildings`` from Planetary
    Computer in parallel (one Dask task per file), keeps buildings whose centroid falls in
    Kansas and writes a Hilbert-partitioned GeoParquet dataset to the cache.
    """
    import geopandas as gpd

    import geolattice.distributed as pdist

    if BUILDINGS.exists():
        return pdist.read_vector(BUILDINGS)
    if not download:
        raise FileNotFoundError(BUILDINGS)
    import dask

    fs, files = _ms_buildings_files()
    if verbose:
        print(f"reading {len(files)} ms-buildings files from Planetary Computer ...")
    parts = dask.compute(*[dask.delayed(_read_part)(fs, f) for f in files])
    allb = pd.concat(parts, ignore_index=True)
    allb = gpd.GeoDataFrame(allb, geometry="geometry", crs=4326)
    ks = gp_states().query("STATE == 'Kansas'").to_crs(4326)
    cen = allb.geometry.representative_point()
    inside = cen.within(ks.geometry.iloc[0])
    out = allb[inside.values].reset_index(drop=True)
    pdist.write_vector(out, BUILDINGS, npartitions=24)
    return pdist.read_vector(BUILDINGS)


# --------------------------------------------------------------------------- Wichita roads
WICHITA = "Wichita, Kansas, USA"
OVERPASS = ["https://overpass-api.de/api", "https://maps.mail.ru/osm/tools/overpass/api",
            "https://overpass.kumi.systems/api"]


def _overpass(fn):
    """Run an OSMnx download, trying each public Overpass server in turn (they are often busy)."""
    import osmnx as ox

    last = None
    for url in OVERPASS:
        ox.settings.overpass_url = url
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fn()
        except Exception as e:  # busy or timed out: try the next server
            last = e
    raise RuntimeError(f"every Overpass server failed; try again later ({last})")


def wichita_roads():
    """Drivable OSM road network of Wichita as undirected line segments (EPSG:5070)."""
    import geopandas as gpd

    if ROADS.exists():
        return gpd.read_parquet(ROADS)
    import osmnx as ox

    g = _overpass(lambda: ox.graph_from_place(WICHITA, network_type="drive", simplify=True))
    g = ox.convert.to_undirected(g)
    edges = ox.graph_to_gdfs(g, nodes=False).reset_index()
    keep = ["u", "v", "highway", "name", "maxspeed", "length", "geometry"]
    edges = edges[[c for c in keep if c in edges.columns]].copy()
    for c in ("highway", "name", "maxspeed"):
        if c in edges:
            edges[c] = edges[c].astype(str)
    edges = edges.to_crs(5070)
    CACHE.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(ROADS)
    return edges


def wichita_facilities():
    """Hospitals and schools of Wichita from OSM, as points (EPSG:5070)."""
    import geopandas as gpd

    if FACILITIES.exists():
        return gpd.read_parquet(FACILITIES)
    import osmnx as ox

    tags = {"amenity": ["hospital", "school"]}
    f = _overpass(lambda: ox.features_from_place(WICHITA, tags))
    f = f.reset_index()[["amenity", "name", "geometry"]]
    f["name"] = f["name"].astype(str)
    f = f.to_crs(5070)
    f["geometry"] = f.geometry.representative_point()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.to_parquet(FACILITIES)
    return f
