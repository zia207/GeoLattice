"""Data helpers for the surface energy balance chapter (08_12).

Every notebook in the chapter reads its inputs through these functions, so it runs on its own:
the first call downloads a scene, its elevation, crop map and weather and caches them under
``Data/08_remote_sensing/08_12_surface_energy_balance/``; later calls read the cache. Notebook
01 builds the same stack step by step and explains each part; notebooks 03 and 04 explain the
radiation and aerodynamic inputs that :func:`energy_inputs` collects.

Sources (all free, no account): Landsat Collection 2 Level-2, Sentinel-2 L2A, Sentinel-3 SLSTR
LST, MODIS and the Copernicus DEM from Microsoft Planetary Computer; the USDA Cropland Data Layer
from CropScape; hourly and daily ERA5 weather from the Open-Meteo archive; daily gridMET from the
Climatology Lab THREDDS server.
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xarray as xr

import geolattice.rsense.surface_energy_balance as seb

warnings.filterwarnings("ignore", category=RuntimeWarning)

CHAPTER = Path(__file__).resolve().parent
DATA = CHAPTER.parents[3] / "Data" / "08_remote_sensing" / "08_12_surface_energy_balance"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Westside San Joaquin Valley, California (Five Points), lon/lat
WESTSIDE = {"name": "westside", "bbox": (-120.25, 36.30, -119.95, 36.55), "lat": 36.425,
            "lon": -120.10, "utc_offset": -7}
# Boro rice belt near Mymensingh, Bangladesh
MYMENSINGH = {"name": "mymensingh", "bbox": (90.20, 24.30, 90.50, 24.60), "lat": 24.45,
              "lon": 90.35, "utc_offset": 6}

MAIN_SCENE = "LC08_L2SP_042035_20240705_02_T1"
# Landsat 8/9 path 42 row 35 scenes of 2024 with less than 13 percent cloud over the scene
SEASON_2024 = [
    "LC08_L2SP_042035_20240111_02_T1", "LC08_L2SP_042035_20240212_02_T1",
    "LC08_L2SP_042035_20240228_02_T1", "LC09_L2SP_042035_20240408_02_T1",
    "LC08_L2SP_042035_20240416_02_T1", "LC08_L2SP_042035_20240502_02_T1",
    "LC09_L2SP_042035_20240510_02_T1", "LC08_L2SP_042035_20240518_02_T1",
    "LC09_L2SP_042035_20240526_02_T1", "LC09_L2SP_042035_20240611_02_T1",
    "LC08_L2SP_042035_20240619_02_T1", "LC09_L2SP_042035_20240627_02_T1",
    "LC08_L2SP_042035_20240705_02_T1", "LC08_L2SP_042035_20240721_02_T1",
    "LC09_L2SP_042035_20240729_02_T1", "LC08_L2SP_042035_20240806_02_T1",
    "LC09_L2SP_042035_20240814_02_T1", "LC08_L2SP_042035_20240822_02_T1",
    "LC09_L2SP_042035_20240830_02_T1", "LC08_L2SP_042035_20240907_02_T1",
    "LC09_L2SP_042035_20240915_02_T1", "LC08_L2SP_042035_20240923_02_T1",
    "LC09_L2SP_042035_20241001_02_T1", "LC08_L2SP_042035_20241009_02_T1",
    "LC09_L2SP_042035_20241017_02_T1", "LC08_L2SP_042035_20241110_02_T1",
    "LC09_L2SP_042035_20241204_02_T1",
]
# Landsat 8/9 path 137 row 43 scenes of the 2023-24 Boro season
BORO_2024 = [
    "LC08_L2SP_137043_20231220_02_T1", "LC09_L2SP_137043_20231228_02_T1",
    "LC09_L2SP_137043_20240129_02_T1", "LC09_L2SP_137043_20240301_02_T1",
    "LC08_L2SP_137043_20240309_02_T1", "LC09_L2SP_137043_20240317_02_T1",
    "LC09_L2SP_137043_20240402_02_T1", "LC08_L2SP_137043_20240426_02_T1",
]

REFLECTANCE = {"blue": "blue", "green": "green", "red": "red", "nir": "nir08",
               "swir16": "swir16", "swir22": "swir22"}

# USDA CDL codes common on the Westside (names from the CDL legend)
CDL_NAMES = {
    1: "Corn", 2: "Cotton", 21: "Barley", 24: "Winter wheat", 28: "Oats", 36: "Alfalfa",
    37: "Other hay", 47: "Misc vegetables and fruits", 49: "Onions", 51: "Chickpeas", 54: "Tomatoes", 57: "Herbs",
    61: "Fallow/idle cropland", 66: "Cherries", 67: "Peaches", 69: "Grapes", 72: "Citrus",
    75: "Almonds", 76: "Walnuts", 111: "Open water", 121: "Developed, open space",
    122: "Developed, low intensity", 123: "Developed, medium intensity",
    124: "Developed, high intensity", 131: "Barren", 152: "Shrubland",
    176: "Grassland/pasture", 204: "Pistachios", 208: "Garlic", 209: "Cantaloupes",
    211: "Olives", 212: "Oranges", 217: "Pomegranates", 225: "Double crop winter wheat/corn",
    227: "Lettuce",
}


# ----------------------------------------------------------------------------- utilities
def _catalog():
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(STAC, modifier=pc.sign_inplace)


def _site_dir(site: dict) -> Path:
    d = DATA / site["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bbox_in(crs, bbox):
    from pyproj import Transformer

    t = Transformer.from_crs(4326, crs, always_xy=True)
    xs, ys = t.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    return min(xs), min(ys), max(xs), max(ys)


def site_for(scene_id: str) -> dict:
    """The study site of a Landsat scene id (path 42 row 35 or path 137 row 43)."""
    return MYMENSINGH if "_137043_" in scene_id else WESTSIDE


def cdl_name(code: int) -> str:
    return CDL_NAMES.get(int(code), f"CDL {int(code)}")


# ----------------------------------------------------------------------------- Landsat
def landsat_item(scene_id: str):
    """STAC item of a Landsat Collection 2 Level-2 scene on Planetary Computer."""
    return _catalog().get_collection("landsat-c2-l2").get_item(scene_id)


def read_landsat(scene_id: str, site: dict | None = None) -> xr.Dataset:
    """Landsat C2 L2 bands clipped to the site, scaled to physical units, with a clear mask.

    Reflectance bands (0 to 1), ``lst`` (``ST_B10``, K), ``emis`` (``ST_EMIS``), thermal
    radiance ``trad``, upwelled ``urad`` and downwelled ``drad`` radiance (W m-2 sr-1 um-1),
    ``atran`` (transmissivity), and ``clear`` (no fill, cloud, cloud shadow, dilated cloud,
    cirrus or snow in ``QA_PIXEL``). Attributes hold the overpass time and sun angles.
    """
    site = site or site_for(scene_id)
    item = landsat_item(scene_id)
    epsg = item.properties.get("proj:epsg") or int(item.properties["proj:code"].split(":")[1])
    x0, y0, x1, y1 = _bbox_in(epsg, site["bbox"])

    def band(name):
        da = rioxarray.open_rasterio(item.assets[name].href).squeeze("band", drop=True)
        return da.sel(x=slice(x0, x1), y=slice(y1, y0)).load()

    out = {}
    for k, a in REFLECTANCE.items():
        dn = band(a)
        out[k] = (dn.where(dn > 0) * 2.75e-5 - 0.2).astype("float32")
    st = band("lwir11")
    out["lst"] = (st.where(st > 0) * 0.00341802 + 149.0).astype("float32")
    for k, scale in {"emis": 1e-4, "trad": 1e-3, "urad": 1e-3, "drad": 1e-3, "atran": 1e-4}.items():
        dn = band(k)
        out[k] = (dn.where(dn != -9999) * scale).astype("float32")
    qa = band("qa_pixel").astype("int64")
    bad = (qa & 1) | ((qa >> 1) & 1) | ((qa >> 2) & 1) | ((qa >> 3) & 1) | ((qa >> 4) & 1) \
        | ((qa >> 5) & 1)
    out["clear"] = ((bad == 0) & out["lst"].notnull()).astype("uint8")
    out["water"] = ((qa >> 7) & 1).astype("uint8")
    ds = xr.Dataset(out).rio.write_crs(epsg)
    geo = seb.scene_geometry(item.properties)
    ds.attrs.update(scene_id=scene_id, datetime=str(geo["datetime"]), doy=geo["doy"],
                    sun_elevation=geo["sun_elevation"], sun_azimuth=geo["sun_azimuth"],
                    dr=geo["dr"], platform=item.properties["platform"],
                    cloud_cover=float(item.properties["eo:cloud_cover"]))
    return ds


# ----------------------------------------------------------------------------- DEM and CDL
def dem_on(grid: xr.DataArray, site: dict) -> xr.DataArray:
    """Copernicus GLO-30 elevation (m) reprojected bilinearly onto ``grid``; cached."""
    path = _site_dir(site) / "dem_cop30.tif"
    if not path.exists():
        from rioxarray.merge import merge_arrays

        items = list(_catalog().search(collections=["cop-dem-glo-30"], bbox=site["bbox"]).items())
        tiles = [rioxarray.open_rasterio(i.assets["data"].href).squeeze("band", drop=True)
                 for i in items]
        b = site["bbox"]
        tiles = [t.rio.clip_box(b[0] - 0.02, b[1] - 0.02, b[2] + 0.02, b[3] + 0.02) for t in tiles]
        merge_arrays(tiles).rio.to_raster(path)
    dem = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    from rasterio.enums import Resampling

    return dem.rio.reproject_match(grid, resampling=Resampling.bilinear).astype("float32") \
        .rename("dem").assign_attrs(units="m")


def cdl_on(grid: xr.DataArray, site: dict = WESTSIDE, year: int = 2024) -> xr.DataArray:
    """USDA Cropland Data Layer for ``year`` from CropScape, nearest neighbour onto ``grid``."""
    path = _site_dir(site) / f"cdl_{year}.tif"
    if not path.exists():
        x0, y0, x1, y1 = _bbox_in(5070, site["bbox"])
        x0, y0, x1, y1 = x0 - 5000, y0 - 5000, x1 + 5000, y1 + 5000  # cover the UTM grid corners
        r = requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                         params={"year": year, "bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}"},
                         timeout=120)
        r.raise_for_status()
        url = r.text.split("<returnURL>")[1].split("</returnURL>")[0]
        path.write_bytes(requests.get(url, timeout=300).content)
    cdl = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    from rasterio.enums import Resampling

    return cdl.rio.reproject_match(grid, resampling=Resampling.nearest).astype("int16") \
        .rename("cdl")


def worldcover_on(grid: xr.DataArray, site: dict = MYMENSINGH) -> xr.DataArray:
    """ESA WorldCover 2021 (10 m) classes, mode resampled onto ``grid``; cached."""
    path = _site_dir(site) / "worldcover_2021.tif"
    if not path.exists():
        from rioxarray.merge import merge_arrays

        items = list(_catalog().search(collections=["esa-worldcover"], bbox=site["bbox"],
                                       datetime="2021").items())
        b = site["bbox"]
        tiles = [rioxarray.open_rasterio(i.assets["map"].href).squeeze("band", drop=True)
                 .rio.clip_box(*b) for i in items]
        merge_arrays(tiles).rio.to_raster(path)
    wc = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    from rasterio.enums import Resampling

    return wc.rio.reproject_match(grid, resampling=Resampling.mode).astype("int16") \
        .rename("worldcover")


# ----------------------------------------------------------------------------- weather
def weather(site: dict, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Hourly and daily ERA5 weather at the site centre from Open-Meteo (UTC), cached as CSV.

    Hourly: temperature, humidity, dew point, wind at 10 m (m s-1), shortwave (W m-2),
    surface pressure. Daily: max and min temperature, shortwave sum (MJ m-2), mean wind,
    precipitation and Open-Meteo's FAO-56 ETo.
    """
    d = _site_dir(site)
    ph, pd_ = d / f"era5_hourly_{start}_{end}.csv", d / f"era5_daily_{start}_{end}.csv"
    if not ph.exists():
        w = seb.weather_open_meteo(site["lat"], site["lon"], start, end,
                                   daily=["temperature_2m_max", "temperature_2m_min",
                                          "shortwave_radiation_sum", "wind_speed_10m_mean",
                                          "dew_point_2m_mean", "precipitation_sum",
                                          "et0_fao_evapotranspiration"])
        w["hourly"].to_csv(ph)
        w["daily"].to_csv(pd_)
    h = pd.read_csv(ph, index_col=0, parse_dates=True)
    dly = pd.read_csv(pd_, index_col=0, parse_dates=True)
    return {"hourly": h, "daily": dly}


def gridmet_point(site: dict, start: str, end: str,
                  variables=("etr", "pet", "tmmx", "tmmn", "srad", "vs", "sph")) -> pd.DataFrame:
    """Daily gridMET (4 km) at the site centre from the Climatology Lab THREDDS server, cached.

    ``etr`` alfalfa and ``pet`` grass reference ET (mm), ``tmmx``/``tmmn`` (K), ``srad``
    (W m-2), ``vs`` wind at 10 m (m s-1), ``sph`` specific humidity (kg kg-1).
    """
    path = _site_dir(site) / f"gridmet_{start}_{end}.csv"
    if not path.exists():
        long = {"etr": "daily_mean_reference_evapotranspiration_alfalfa",
                "pet": "daily_mean_reference_evapotranspiration_grass",
                "tmmx": "daily_maximum_temperature", "tmmn": "daily_minimum_temperature",
                "srad": "daily_mean_shortwave_radiation_at_surface",
                "vs": "daily_mean_wind_speed", "sph": "daily_mean_specific_humidity"}
        scale = {"etr": 0.1, "pet": 0.1, "tmmx": 0.1, "tmmn": 0.1, "srad": 0.1, "vs": 0.1,
                 "sph": 1e-5}
        offset = {"tmmx": 220.0, "tmmn": 210.0}
        cols = {}
        for v in variables:
            url = (f"http://thredds.northwestknowledge.net:8080/thredds/ncss/"
                   f"agg_met_{v}_1979_CurrentYear_CONUS.nc")
            r = requests.get(url, params={"var": long[v], "latitude": site["lat"],
                                          "longitude": site["lon"],
                                          "time_start": f"{start}T00:00:00Z",
                                          "time_end": f"{end}T00:00:00Z", "accept": "csv"},
                             timeout=300)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            raw = pd.to_numeric(df.iloc[:, -1], errors="coerce")
            # the subset service returns the packed integers: apply scale_factor and add_offset
            cols[v] = (raw * scale[v] + offset.get(v, 0.0)).values
            idx = pd.to_datetime(df.iloc[:, 0]).dt.tz_localize(None)
        out = pd.DataFrame(cols, index=idx)
        out.index.name = "time"
        out.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ----------------------------------------------------------------------------- scene stacks
def scene_path(scene_id: str) -> Path:
    return _site_dir(site_for(scene_id)) / "scenes" / f"{scene_id}.nc"


def load_scene(scene_id: str = MAIN_SCENE, *, rebuild: bool = False) -> xr.Dataset:
    """Analysis-ready stack of one Landsat scene: reflectance, LST, thermal radiance terms,
    clear and water masks, elevation, land cover (CDL or WorldCover) and the weather at the
    overpass (attributes ``t_air_c``, ``ea_kpa``, ``u10``, ``rs_wm2``, ``pressure_kpa``) plus the
    daily weather of that day. Built once and cached as NetCDF; notebook 01 explains each step.
    """
    path = scene_path(scene_id)
    if path.exists() and not rebuild:
        with xr.open_dataset(path) as f:
            ds = f.load()
        return ds.rio.write_crs(ds.attrs["crs"]) if "crs" in ds.attrs else ds
    site = site_for(scene_id)
    ds = read_landsat(scene_id, site)
    ds["dem"] = dem_on(ds["lst"], site)
    if site is WESTSIDE:
        ds["cdl"] = cdl_on(ds["lst"], site, int(scene_id[17:21]))
    else:
        ds["worldcover"] = worldcover_on(ds["lst"], site)
    when = pd.Timestamp(ds.attrs["datetime"])
    day = when.strftime("%Y-%m-%d")
    w = weather(site, day, day)
    wx = seb.interpolate_weather(w["hourly"], when)
    dly = w["daily"].iloc[0]
    ds.attrs.update(
        t_air_c=float(wx["temperature_2m"]), rh=float(wx["relative_humidity_2m"]),
        t_dew_c=float(wx["dew_point_2m"]), u10=float(wx["wind_speed_10m"]),
        rs_wm2=float(wx["shortwave_radiation"]), pressure_kpa=float(wx["surface_pressure"]) / 10,
        ea_kpa=float(seb.actual_vapour_pressure(t_dew_c=float(wx["dew_point_2m"]))),
        tmax_c=float(dly["temperature_2m_max"]), tmin_c=float(dly["temperature_2m_min"]),
        rs_daily_mj=float(dly["shortwave_radiation_sum"]),
        u10_daily=float(dly["wind_speed_10m_mean"]), t_dew_daily_c=float(dly["dew_point_2m_mean"]),
        eto_openmeteo=float(dly["et0_fao_evapotranspiration"]),
        lat=site["lat"], lon=site["lon"], site=site["name"], crs=ds.rio.crs.to_string())
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(ds, path)
    return ds


def _save(ds: xr.Dataset, path: Path):
    """Write a stack without the encodings inherited from the source GeoTIFFs: a ``_FillValue``
    of 0 copied from ``QA_PIXEL`` would turn every 0 of the clear and water masks into NaN."""
    for v in ds.variables:
        ds[v].encoding = {}
        for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
            ds[v].attrs.pop(k, None)  # values are already scaled; a kept scale_factor would apply twice
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)


def build_scene(scene_id: str) -> str:
    """Build (or find) the cached stack of a scene and return its path; importable, so it can
    run in a process pool on every platform."""
    load_scene(scene_id)
    return str(scene_path(scene_id))


# ----------------------------------------------------------------------------- other sensors
S2_SCENE = "S2B_MSIL2A_20240704T183919_R070_T10SGF_20240706T003648"
S3_LST_SCENE = "S3B_SL_2_LST_20240704T182446_20240704T182746_0180_095_027_2340"
WESTSIDE_REGION = {"name": "westside_region", "bbox": (-120.60, 36.00, -119.60, 36.90),
                   "lat": 36.425, "lon": -120.10, "utc_offset": -7}


def sentinel2_20m(item_id: str = S2_SCENE, site: dict = WESTSIDE, block: int = 50) -> xr.Dataset:
    """Sentinel-2 L2A reflectance (0 to 1) at 20 m over the site, with a clear mask from the scene
    classification (SCL), cropped to whole ``block`` x ``block`` pixel blocks (50 x 20 m = 1 km) so
    a 1 km grid nests exactly. 10 m bands are averaged to 20 m. Cached as NetCDF."""
    path = _site_dir(site) / f"{item_id[:60]}_20m.nc"
    if path.exists():
        with xr.open_dataset(path) as f:
            ds = f.load()
        return ds.rio.write_crs(ds.attrs["crs"])
    item = _catalog().get_collection("sentinel-2-l2a").get_item(item_id)
    epsg = int(str(item.properties.get("proj:epsg") or item.properties["proj:code"]).split(":")[-1])
    x0, y0, x1, y1 = _bbox_in(epsg, site["bbox"])
    scl = rioxarray.open_rasterio(item.assets["SCL"].href).squeeze("band", drop=True)
    scl = scl.sel(x=slice(x0, x1), y=slice(y1, y0))
    ny, nx = (scl.sizes["y"] // block) * block, (scl.sizes["x"] // block) * block
    scl = scl.isel(y=slice(0, ny), x=slice(0, nx)).load()
    baseline = float(item.properties.get("s2:processing_baseline", "0"))
    offset = 1000 if baseline >= 4.0 else 0  # BOA_ADD_OFFSET from processing baseline 04.00
    bands = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B8A", "swir16": "B11",
             "swir22": "B12"}
    out = {}
    for name, b in bands.items():
        da = rioxarray.open_rasterio(item.assets[b].href).squeeze("band", drop=True)
        if abs(da.rio.resolution()[0]) < 15:  # 10 m band: average to 20 m
            da = da.sel(x=slice(x0 - 20, x1 + 20), y=slice(y1 + 20, y0 - 20)).load()
            da = da.where(da > 0).astype("float32").coarsen(x=2, y=2, boundary="trim").mean()
            da = da.interp(x=scl.x, y=scl.y, method="nearest")
        else:
            da = da.sel(x=slice(x0, x1), y=slice(y1, y0)).isel(y=slice(0, ny), x=slice(0, nx)).load()
            da = da.where(da > 0).astype("float32")
        out[name] = ((da - offset) / 10000.0).clip(0, 1).astype("float32")
    ds = xr.Dataset(out)
    ds["clear"] = scl.isin([4, 5, 6, 7]).astype("uint8")  # vegetation, bare, water, unclassified
    ds["water"] = (scl == 6).astype("uint8")
    ds = ds.rio.write_crs(epsg)
    ds.attrs.update(scene_id=item_id, datetime=str(pd.Timestamp(item.properties["datetime"])
                                                   .tz_convert(None)),
                    processing_baseline=baseline, crs=f"EPSG:{epsg}")
    _save(ds, path)
    return ds


def sentinel3_lst_folder(item_id: str = S3_LST_SCENE, site: dict = WESTSIDE) -> Path:
    """Download the SLSTR LST, flags and geolocation NetCDF files of one granule (about 10 MB)
    into a folder that ``geolattice.rsense.optical.sensors.sentinel3.read_slstr_lst`` reads."""
    folder = _site_dir(site) / item_id
    folder.mkdir(exist_ok=True)
    item = _catalog().get_collection("sentinel-3-slstr-lst-l2-netcdf").get_item(item_id)
    for key, name in {"lst-in": "LST_in.nc", "slstr-flags-in": "flags_in.nc",
                      "slstr-geodetic-in": "geodetic_in.nc", "slstr-time-in": "time_in.nc"}.items():
        f = folder / name
        if not f.exists():
            f.write_bytes(requests.get(item.assets[key].href, timeout=300).content)
    (folder / "datetime.txt").write_text(str(pd.Timestamp(item.properties["datetime"])
                                             .tz_convert(None)))
    return folder


def utm_grid(bbox, epsg: int, res: float) -> xr.DataArray:
    """An empty regular grid (NaN) covering a lon/lat box in a UTM zone, aligned to ``res``."""
    x0, y0, x1, y1 = _bbox_in(epsg, bbox)
    x0, y0 = np.floor(x0 / res) * res, np.floor(y0 / res) * res
    x1, y1 = np.ceil(x1 / res) * res, np.ceil(y1 / res) * res
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y1 - res / 2, y0, -res)
    da = xr.DataArray(np.full((ys.size, xs.size), np.nan, "float32"), coords={"y": ys, "x": xs},
                      dims=("y", "x"))
    return da.rio.write_crs(epsg).rio.write_transform()


def modis_1km(date: str = "2024-07-05", site: dict = WESTSIDE_REGION, epsg: int = 32611
              ) -> xr.Dataset:
    """MODIS inputs for one day on a 1 km UTM grid: Terra MOD11A1 daytime LST (K) and view time
    (local solar hours), MCD43A4 nadir BRDF-adjusted reflectance (averaged from 500 m), MCD15A2H
    LAI (8-day), and Copernicus elevation. Cached as NetCDF."""
    path = _site_dir(site) / f"modis_1km_{date}.nc"
    if path.exists():
        with xr.open_dataset(path) as f:
            ds = f.load()
        return ds.rio.write_crs(ds.attrs["crs"])
    from rasterio.enums import Resampling

    grid = utm_grid(site["bbox"], epsg, 1000.0)
    cat = _catalog()
    day = pd.Timestamp(date)

    def first(col, dt):
        items = list(cat.search(collections=[col], bbox=site["bbox"], datetime=dt).items())
        terra = [i for i in items if i.id.startswith(("MOD", "MCD"))]
        return (terra or items)[0]

    def get(item, asset, scale, valid, resampling):
        da = rioxarray.open_rasterio(item.assets[asset].href).squeeze("band", drop=True)
        da = da.where((da >= valid[0]) & (da <= valid[1])) * scale
        return da.rio.reproject_match(grid, resampling=resampling).astype("float32")

    lst_item = first("modis-11A1-061", date)
    out = {"lst": get(lst_item, "LST_Day_1km", 0.02, (7500, 65535), Resampling.nearest),
           "view_time": get(lst_item, "Day_view_time", 0.1, (0, 240), Resampling.nearest),
           "qc": rioxarray.open_rasterio(lst_item.assets["QC_Day"].href).squeeze("band", drop=True)
           .rio.reproject_match(grid, resampling=Resampling.nearest).astype("float32")}
    nbar = first("modis-43A4-061", date)
    names = {1: "red", 2: "nir", 3: "blue", 4: "green", 5: "swir12", 6: "swir16", 7: "swir22"}
    for b, n in names.items():
        out[n] = get(nbar, f"Nadir_Reflectance_Band{b}", 1e-4, (0, 32766), Resampling.average)
    lai_items = list(cat.search(collections=["modis-15A2H-061"], bbox=site["bbox"],
                                datetime=f"{(day - pd.Timedelta(days=8)).date()}/{day.date()}")
                     .items())
    lai_item = [i for i in lai_items if i.id.startswith("MCD")][0]
    out["lai"] = get(lai_item, "Lai_500m", 0.1, (0, 100), Resampling.average)
    ds = xr.Dataset(out)
    ds["dem"] = dem_on(grid, site)
    ds = ds.rio.write_crs(epsg)
    ds.attrs.update(date=date, lst_item=lst_item.id, nbar_item=nbar.id, lai_item=lai_item.id,
                    crs=f"EPSG:{epsg}")
    _save(ds, path)
    return ds


# ----------------------------------------------------------------------------- shared chain
def reference_et(ds: xr.Dataset) -> dict[str, float]:
    """Hourly alfalfa reference ET at the overpass (mm/h) and daily alfalfa and grass reference
    ET (mm/d) from the ERA5 weather in a scene stack's attributes (ASCE standardized)."""
    a = ds.attrs
    z = float(np.nanmedian(ds["dem"].values))
    u2 = float(seb.wind_2m(a["u10"], 10.0))
    u2d = float(seb.wind_2m(a["u10_daily"], 10.0))
    ea_d = float(seb.actual_vapour_pressure(t_dew_c=a["t_dew_daily_c"]))
    when = pd.Timestamp(a["datetime"])
    etr_h = float(seb.asce_pm_hourly(a["t_air_c"], a["ea_kpa"], a["rs_wm2"] * 0.0036, u2, z,
                                     a["lat"], a["lon"], when, reference="tall"))
    etr_d = float(seb.asce_pm_daily(a["tmax_c"], a["tmin_c"], ea_d, a["rs_daily_mj"], u2d, z,
                                    a["lat"], a["doy"], reference="tall"))
    eto_d = float(seb.asce_pm_daily(a["tmax_c"], a["tmin_c"], ea_d, a["rs_daily_mj"], u2d, z,
                                    a["lat"], a["doy"], reference="short"))
    return {"etr_inst": etr_h, "etr_daily": etr_d, "eto_daily": eto_d, "u2": u2, "z": z}


def daily_reference(site: dict, start: str, end: str, z: float | None = None) -> pd.DataFrame:
    """Daily ASCE standardized alfalfa (``etr``) and grass (``eto``) reference ET (mm/d) from
    ERA5 daily weather at the site centre (notebook 02 explains every term)."""
    d = weather(site, start, end)["daily"]
    z = 65.0 if z is None and site is WESTSIDE else (10.0 if z is None else z)
    ea = seb.actual_vapour_pressure(t_dew_c=d["dew_point_2m_mean"].values)
    u2 = seb.wind_2m(d["wind_speed_10m_mean"].values, 10.0)
    args = (d["temperature_2m_max"].values, d["temperature_2m_min"].values, ea,
            d["shortwave_radiation_sum"].values, u2, z, site["lat"], d.index.dayofyear.values)
    return pd.DataFrame({"etr": seb.asce_pm_daily(*args, reference="tall"),
                         "eto": seb.asce_pm_daily(*args, reference="short"),
                         "eto_openmeteo": d["et0_fao_evapotranspiration"].values,
                         "precip": d["precipitation_sum"].values}, index=d.index)


def energy_inputs(ds: xr.Dataset, *, savi_l: float = 0.1) -> dict:
    """The radiation and aerodynamic inputs every one-source model needs (notebooks 03 and 04).

    Returns NDVI, SAVI, LAI, albedo, emissivity, pressure, clear-sky transmissivity, incoming
    shortwave (flat ground), anchors, incoming longwave, net radiation, soil heat flux (METRIC),
    roughness, wind at 200 m and the reference ET terms, all on the scene grid with clouds masked.
    """
    m = ds["clear"].astype(bool)
    sr = ds[list(REFLECTANCE)].where(m)
    lst = ds["lst"].where(m)
    a = ds.attrs
    ref = reference_et(ds)
    ndvi = seb.ndvi(sr.red, sr.nir)
    savi = seb.savi(sr.red, sr.nir, L=savi_l)
    lai = seb.leaf_area_index(savi)
    alb = seb.albedo(sr, sensor="landsat8")
    e0 = seb.surface_emissivity(lai, ndvi)
    pressure = seb.pressure_from_elevation(ds["dem"])
    cos_t = float(np.cos(np.radians(90 - a["sun_elevation"])))
    tau = seb.broadband_transmissivity(cos_t, pressure=pressure, ea=a["ea_kpa"])
    rs_in = seb.incoming_shortwave(cos_t, tau, dr=a["dr"])
    anchors = seb.select_anchors(ndvi, lst, albedo_=alb, mask=m & (ds["water"] == 0))
    tau_c = float(np.asarray(tau)[anchors["cold"]["iy"], anchors["cold"]["ix"]])
    rl_in = seb.incoming_longwave(anchors["cold"]["lst"], seb.air_emissivity(tau_sw=tau_c))
    rn = seb.net_radiation(rs_in, alb, rl_in, seb.outgoing_longwave(lst, e0), e0)
    g = seb.soil_heat_flux(rn, lst, lai=lai, method="metric", ndvi_=ndvi)
    zom = seb.momentum_roughness(lai=lai)
    u200 = float(seb.wind_at_blending_height(ref["u2"], z=2.0))
    return {"mask": m, "sr": sr, "lst": lst, "ndvi": ndvi, "savi": savi, "lai": lai,
            "albedo": alb, "emissivity": e0, "pressure": pressure, "tau": tau, "rs_in": rs_in,
            "anchors": anchors, "rl_in": rl_in, "rn": rn, "g": g, "zom": zom, "u200": u200,
            "cos_theta": cos_t, **ref}


def run_metric_scene(ds: xr.Dataset, inputs: dict | None = None, **kwargs) -> xr.Dataset:
    """METRIC for one scene stack with the chapter defaults (notebook 06)."""
    p = inputs or energy_inputs(ds)
    out = seb.run_metric(p["lst"], p["rn"], p["g"], p["zom"], p["u200"], p["pressure"],
                         p["etr_inst"], anchors=p["anchors"], etr_daily=p["etr_daily"], **kwargs)
    out.attrs.update(etr_inst=p["etr_inst"], etr_daily=p["etr_daily"],
                     scene_id=ds.attrs["scene_id"], datetime=ds.attrs["datetime"])
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
