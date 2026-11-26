"""Data helpers for the satellite soil moisture chapter (09_13).

Every notebook reads its inputs through these functions, so it runs on its own: the first call
subsets or downloads the data and caches it under ``Data/08_remote_sensing/08_13_soil_moisture/``;
later calls read the cache. The subsetting itself is done by ``geolattice.rsense.soil_moisture``
(``sm.open_product``): only the study area is transferred, never the global files.

Sources: SMAP L3, L3 Enhanced, L4 and SPL2SMAP_S, SMOS with the SMAP algorithm (NSIDC-0800) and the
SMAPVEX16 field data from NSIDC, and NISAR soil moisture from ASF (free NASA Earthdata login in
``~/.netrc``); ESA CCI SM from CEDA; ERA5-Land from the Open-Meteo archive; Sentinel-1 RTC,
Sentinel-2 L2A, MODIS, the Copernicus DEM, ESA WorldCover and the USDA Cropland Data Layer from
Microsoft Planetary Computer; SoilGrids 2.0 from ISRIC; USCRN from NOAA NCEI; SCAN from the NRCS
AWDB service; gridMET from the Climatology Lab THREDDS server; USGS daily streamflow from the
NWIS web service. None of them needs a key apart from Earthdata.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xarray as xr

import geolattice.rsense.soil_moisture as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)

CHAPTER = Path(__file__).resolve().parent
DATA = CHAPTER.parents[3] / "Data" / "08_remote_sensing" / "08_13_soil_moisture"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ----------------------------------------------------------------------------- study areas
# Central Iowa field site: Ames and the South Fork of the Iowa River (SMAPVEX16), lon/lat
IOWA = {"name": "iowa", "bbox": (-93.9, 41.8, -93.3, 42.2), "lat": 42.0, "lon": -93.6, "utc_offset": -6}
# coarse products: Iowa and its neighbours, and the Midwest box that adds Nebraska
IOWA_REGION = (-97.0, 40.0, -90.0, 44.0)
MIDWEST = (-101.0, 40.0, -90.0, 44.0)
# irrigated Platte valley, Nebraska, with rainfed fields north and south of it
NEBRASKA = (-100.5, 40.4, -97.5, 41.6)
BANGLADESH = (88.0, 20.5, 92.7, 26.7)

# in situ stations inside the Iowa region (coordinates from the station metadata)
STATIONS = pd.DataFrame([
    {"station": "USCRN Des Moines 17 E", "network": "USCRN", "id": "IA_Des_Moines_17_E", "lon": -93.2861, "lat": 41.5564},
    {"station": "SCAN Ames", "network": "SCAN", "id": "2031:IA:SCAN", "lon": -93.72977, "lat": 42.0136},
]).set_index("station")

SMAP_PERIOD = ("2015-03-31", "2025-12-31")


def _dir(*parts) -> Path:
    d = DATA.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _catalog():
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(STAC, modifier=pc.sign_inplace)


# ----------------------------------------------------------------------------- coarse products
def smap_l3e(region=MIDWEST, period=SMAP_PERIOD, name: str = "midwest") -> xr.Dataset:
    """SMAP L3 Enhanced (9 km) AM and PM cube with flags, VWC, tau, Tb and surface temperature."""
    path = _dir("smap") / f"spl3smp_e_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    return sm.open_product("SMAP_L3_E", region, period, cache=str(path))


def smap_l3(region=IOWA_REGION, period=("2024-01-01", "2024-12-31"), name: str = "iowa") -> xr.Dataset:
    """SMAP L3 radiometer (36 km) cube."""
    path = _dir("smap") / f"spl3smp_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    return sm.open_product("SMAP_L3", region, period, cache=str(path))


def smap_l4(region=MIDWEST, period=SMAP_PERIOD, name: str = "midwest", hours=("103000",)) -> xr.Dataset:
    """SMAP L4 surface and root-zone soil moisture, one 3-hourly field per day (10:30 UTC)."""
    tag = "_".join(hours)
    path = _dir("smap") / f"spl4smgp_{name}_{period[0][:4]}_{period[1][:4]}_{tag}.nc"
    return sm.open_product("SMAP_L4", region, period, cache=str(path), hours=tuple(hours))


def smos(region=IOWA_REGION, period=("2022-01-01", "2024-12-31"), name: str = "iowa") -> xr.Dataset:
    """SMOS brightness temperatures retrieved with the SMAP algorithm (NSIDC-0800, 9 km)."""
    path = _dir("smos") / f"nsidc0800_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    return sm.open_product("SMOS_SMAP_L3", region, period, cache=str(path), threads=12)


def cci(product: str = "COMBINED", region=IOWA_REGION, period=("2015-01-01", "2024-12-31"),
        name: str = "iowa") -> xr.Dataset:
    """ESA CCI SM v09.2 daily cube (0.25 degree) for one product."""
    path = _dir("cci") / f"cci_{product.lower()}_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    return sm.open_product("ESA_CCI", region, period, product=product, cache=str(path), threads=12)


def era5_land(region=IOWA_REGION, period=("2015-01-01", "2025-12-31"), name: str = "iowa",
              resolution: float = 0.5, layers=(1,)) -> xr.Dataset:
    """ERA5-Land daily soil water on a grid of points (Open-Meteo archive)."""
    tag = "".join(str(i) for i in layers)
    path = _dir("era5_land") / f"era5land_{name}_{period[0][:4]}_{period[1][:4]}_{resolution}deg_l{tag}.nc"
    if not path.exists():
        ds = sm.read_era5_land(region, period, layers=layers, resolution=resolution)
        sm.to_cache(ds, str(path))
    return sm.open_cached(str(path))


def era5_land_points(points: pd.DataFrame = STATIONS, period=("2015-01-01", "2025-12-31"),
                     name: str = "stations") -> xr.Dataset:
    """ERA5-Land daily soil water in three layers at points (``lon``, ``lat`` columns)."""
    path = _dir("era5_land") / f"era5land_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    if not path.exists():
        ds = sm.read_era5_land(None, period, points=list(zip(points.lon, points.lat)), layers=(1, 2, 3))
        ds = ds.assign_coords(point=list(points.index))
        sm.to_cache(ds, str(path))
    return xr.open_dataset(path).load()


# ----------------------------------------------------------------------------- in situ
def uscrn(station: str = "IA_Des_Moines_17_E", years=range(2015, 2026)) -> pd.DataFrame:
    """Hourly USCRN soil moisture (5 to 100 cm, m3 m-3), soil and air temperature and precipitation (``sm.read_uscrn``)."""
    path = _dir("in_situ") / f"uscrn_{station}_{min(years)}_{max(years)}.csv"
    if not path.exists():
        sm.read_uscrn(station, years).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def scan(triplet: str = "2031:IA:SCAN", start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Daily SCAN soil moisture (5, 10, 20, 51, 102 cm, m3 m-3), soil temperature and precipitation (``sm.read_scan``)."""
    path = _dir("in_situ") / f"scan_{triplet.replace(':', '_')}_{start[:4]}_{end[:4]}.csv"
    if not path.exists():
        sm.read_scan(triplet, start, end).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def usgs_daily_flow(site: str = "05470000", start: str = "2015-01-01", end: str = "2025-12-31") -> pd.Series:
    """Daily mean discharge (m3 s-1) of a USGS gauge from the NWIS daily values service."""
    path = _dir("in_situ") / f"usgs_{site}_{start[:4]}_{end[:4]}.csv"
    if not path.exists():
        r = requests.get("https://waterservices.usgs.gov/nwis/dv/", params={
            "format": "json", "sites": site, "parameterCd": "00060", "startDT": start, "endDT": end}, timeout=300)
        r.raise_for_status()
        ts = r.json()["value"]["timeSeries"][0]
        vals = pd.Series({pd.Timestamp(v["dateTime"][:10]): float(v["value"]) for v in ts["values"][0]["value"]})
        vals = vals.where(vals >= 0) * 0.0283168  # cfs to m3 s-1
        vals.rename("q_m3s").rename_axis("date").to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)["q_m3s"]


def gridmet(region=IOWA_REGION, start: str = "2015-01-01", end: str = "2025-12-31",
            variables=("pr", "pet"), name: str = "iowa") -> xr.Dataset:
    """Daily gridMET grids (4 km): precipitation ``pr`` (mm) and grass reference ET ``pet`` (mm)."""
    path = _dir("gridmet") / f"gridmet_{name}_{start[:4]}_{end[:4]}_{'_'.join(variables)}.nc"
    if not path.exists():
        long = {"pr": "precipitation_amount", "pet": "daily_mean_reference_evapotranspiration_grass",
                "tmmx": "daily_maximum_temperature", "tmmn": "daily_minimum_temperature"}
        w, s, e, n = region
        parts = {}
        for v in variables:
            url = f"http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{v}_1979_CurrentYear_CONUS.nc"
            r = requests.get(url, params={"var": long[v], "north": n, "south": s, "east": e, "west": w,
                                          "time_start": f"{start}T00:00:00Z", "time_end": f"{end}T00:00:00Z",
                                          "accept": "netcdf"}, timeout=1800)
            r.raise_for_status()
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".nc") as tmpf:  # NetCDF-3 (the server's netcdf4 option fails)
                tmpf.write(r.content)
                tmpf.flush()
                da = xr.open_dataset(tmpf.name, engine="netcdf4")[long[v]].load()
            parts[v] = da.rename({"day": "time"}) if "day" in da.dims else da
        ds = xr.Dataset(parts)
        ds = ds.rename({d: {"latitude": "lat", "longitude": "lon"}.get(d, d) for d in ds.dims})
        sm.to_cache(ds, str(path))
    return xr.open_dataset(path).load()


def gridmet_point(lon: float, lat: float, start: str = "2015-01-01", end: str = "2025-12-31",
                  variables=("pr", "pet")) -> pd.DataFrame:
    """Daily gridMET at one point (mm), cached."""
    path = _dir("gridmet") / f"gridmet_{lon:.3f}_{lat:.3f}_{start[:4]}_{end[:4]}_{'_'.join(variables)}.csv"
    if not path.exists():
        long = {"pr": "precipitation_amount", "pet": "daily_mean_reference_evapotranspiration_grass"}
        scale = {"pr": 0.1, "pet": 0.1}
        cols = {}
        for v in variables:
            url = f"http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{v}_1979_CurrentYear_CONUS.nc"
            r = requests.get(url, params={"var": long[v], "latitude": lat, "longitude": lon,
                                          "time_start": f"{start}T00:00:00Z", "time_end": f"{end}T00:00:00Z",
                                          "accept": "csv"}, timeout=600)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # the point subset service returns the packed integers: apply the scale factor
            cols[v] = pd.to_numeric(df.iloc[:, -1], errors="coerce").values * scale[v]
            idx = pd.to_datetime(df.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
        out = pd.DataFrame(cols, index=idx)
        out.index.name = "date"
        out.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ----------------------------------------------------------------------------- soils
def soilgrids_profile(station: str) -> pd.DataFrame:
    """SoilGrids 2.0 sand, clay, organic carbon and bulk density at six depths below a station."""
    s = STATIONS.loc[station]
    path = _dir("soilgrids") / f"profile_{s['id'].replace(':', '_')}.csv"
    if not path.exists():
        prof = sm.load_soilgrids((s.lon, s.lat), properties=("sand", "silt", "clay", "soc", "bdod"),
                                 depths=sm.SOILGRIDS_DEPTHS)
        prof.to_dataframe().to_csv(path)
    return pd.read_csv(path, index_col=0)


def soilgrids_grid(bbox=IOWA["bbox"], name: str = "iowa", depths=("0-5cm",),
                   properties=("sand", "clay", "soc", "bdod"), crs="EPSG:4326", overview_level=None) -> xr.Dataset:
    """SoilGrids 2.0 layers over an AOI (about 250 m, or coarser with ``overview_level``), cached."""
    path = _dir("soilgrids") / f"grid_{name}_{'_'.join(depths)}.nc"
    if not path.exists():
        ds = sm.load_soilgrids(bbox, properties=properties, depths=depths, crs=crs, overview_level=overview_level)
        sm.to_cache(ds, str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- MODIS (Planetary Computer)
MODIS = {
    "ndvi": ("modis-13A1-061", ["500m_16_days_NDVI"], 1e-4, 500),
    "lst": ("modis-11A1-061", ["LST_Day_1km", "LST_Night_1km"], 0.02, 1000),
    "lai": ("modis-15A2H-061", ["Lai_500m"], 0.1, 500),
    "nbar": ("modis-43A4-061", ["Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band7"], 1e-4, 500),
}


def modis(kind: str, bbox=IOWA["bbox"], start: str = "2024-01-01", end: str = "2024-12-31", name: str = "iowa",
          crs: str = "EPSG:6933", resolution: float | None = None) -> xr.Dataset:
    """MODIS NDVI (16-day, 500 m), daily day and night LST (1 km, K) or LAI on an equal-area grid, cached."""
    from geolattice.rsense.access import stac_load, stac_search

    coll, bands, scale, res = MODIS[kind]
    path = _dir("modis") / f"{kind}_{name}_{start}_{end}.nc"
    if not path.exists():
        items = list(stac_search(coll, aoi=bbox, dates=f"{start}/{end}"))
        tiles = {}  # MODIS composites have no single datetime, so tiles are loaded apart and combined
        for it in items:
            tiles.setdefault(it.id.split(".")[2], []).append(it)
        ds = None
        for tile_items in tiles.values():
            part = stac_load(tile_items, bands=bands, aoi=bbox, crs=crs, resolution=resolution or res, groupby="solar_day",
                             chunks={"x": 2048, "y": 2048}).load()
            for b_ in bands:  # fill values are no data, so the other tile can fill them
                part[b_] = part[b_].where(part[b_] != part[b_].attrs.get("nodata", -32768))
            ds = part if ds is None else ds.combine_first(part)
        out = xr.Dataset()
        for b in bands:
            v = ds[b].astype("float32")
            valid = {"lst": v != 0, "ndvi": (v > -3000) & (v < 10001), "lai": v < 249, "nbar": (v >= 0) & (v < 32767)}[kind]
            out[{"500m_16_days_NDVI": "ndvi", "LST_Day_1km": "lst_day", "LST_Night_1km": "lst_night", "Lai_500m": "lai",
                 "Nadir_Reflectance_Band1": "red", "Nadir_Reflectance_Band2": "nir",
                 "Nadir_Reflectance_Band7": "swir22"}[b]] = v.where(valid) * scale
        out = out.rio.write_crs(crs)
        sm.to_cache(out, str(path))
    return sm.open_cached(str(path))


def ease2_clip(ds, bbox):
    """Cells of an EASE-Grid 2.0 cube (EPSG:6933) whose centres fall inside a lon/lat bbox."""
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 6933, always_xy=True)
    x0, y0 = t.transform(bbox[0], bbox[1])
    x1, y1 = t.transform(bbox[2], bbox[3])
    return ds.sel(x=slice(x0, x1), y=slice(y1, y0))


# ----------------------------------------------------------------------------- Sentinel-1 (Planetary Computer RTC)
# SAR field site: both stations inside, central Iowa (lon/lat)
IOWA_SAR = (-93.9, 41.5, -93.2, 42.2)
SAR_PERIOD = ("2019-01-01", "2021-12-31")  # Sentinel-1A and 1B years with regular acquisitions


def s1_cube(bbox=IOWA_SAR, period=SAR_PERIOD, name: str = "iowa", resolution: float = 50,
            relative_orbit: int | None = None) -> xr.Dataset:
    """Sentinel-1 RTC VV and VH (dB) for one relative orbit, multitemporal speckle filter, cached."""
    path = _dir("sentinel1") / f"s1_{name}_{period[0][:4]}_{period[1][:4]}_{int(resolution)}m.nc"
    if not path.exists():
        ds = sm.s1_soil_moisture_cube(bbox, period, resolution=resolution, relative_orbit=relative_orbit)
        sm.to_cache(ds.astype("float32"), str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- boundaries
def boundaries(iso: str = "BGD", level: str = "ADM0"):
    """Administrative boundaries from geoBoundaries (simplified), as a GeoDataFrame, cached."""
    import geopandas as gpd

    path = _dir("boundaries") / f"geoboundaries_{iso}_{level}.geojson"
    if not path.exists():
        meta = requests.get(f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/", timeout=120).json()
        r = requests.get(meta["simplifiedGeometryGeoJSON"], timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return gpd.read_file(path)


def us_counties(state_fips: str = "19"):
    """US counties of one state (TIGER cartographic boundaries 2023, 1:500k), cached."""
    import geopandas as gpd

    path = _dir("boundaries") / "cb_2023_us_county_500k.zip"
    if not path.exists():
        r = requests.get("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip", timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    g = gpd.read_file(f"zip://{path}")
    return g[g.STATEFP == state_fips].reset_index(drop=True)


# ----------------------------------------------------------------------------- static layers (Planetary Computer)
def _pc_static(collection: str, asset: str, bbox, crs, resolution, query=None, dates=None, resampling="nearest"):
    from geolattice.rsense.access import stac_load, stac_search

    items = list(stac_search(collection, aoi=bbox, dates=dates, query=query))
    ds = stac_load(items, bands=[asset], aoi=bbox, crs=crs, resolution=resolution, groupby=None,
                   resampling=resampling, chunks={"x": 2048, "y": 2048})
    da = ds[asset].max("time") if "time" in ds.dims else ds[asset]
    return da.load().rio.write_crs(crs)


def static_layers(bbox=IOWA_SAR, name: str = "iowa_sar", crs: str = "EPSG:32615", resolution: float = 50,
                  cdl_years=(2019, 2020, 2021)) -> xr.Dataset:
    """Copernicus DEM (m), ESA WorldCover 2021 classes and USDA CDL crop codes on one grid, cached."""
    path = _dir("static") / f"static_{name}_{int(resolution)}m.nc"
    if not path.exists():
        out = xr.Dataset()
        out["dem"] = _pc_static("cop-dem-glo-30", "data", bbox, crs, resolution, resampling="average").astype("float32")
        out["worldcover"] = _pc_static("esa-worldcover", "map", bbox, crs, resolution, dates="2021-01-01/2021-12-31")
        for y in cdl_years:
            out[f"cdl_{y}"] = _pc_static("usda-cdl", "cropland", bbox, crs, resolution, dates=f"{y}-01-01/{y}-12-31",
                                         query={"usda_cdl:type": {"eq": "cropland"}})
        sm.to_cache(out.rio.write_crs(crs), str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- SMAPVEX16 Iowa (NSIDC)
SMAPVEX16 = (-93.57, 42.39, -93.19, 42.64)  # campaign domain, South Fork of the Iowa River
_SITE_FIX = {"JPL Corn": "JPL-Corn", "LTAR Corn": "LTAR-Corn", "LTAR Soy": "LTAR-Soy", "NA04": "NASA04",
             "SF01 Corn": "SF01-Corn", "SF01 Soy": "SF01-Soy"}


def smapvex16_thetaprobe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """SMAPVEX16 Iowa ThetaProbe samples (site, date, crop, 0-5 cm soil moisture) and site coordinates."""
    d = _dir("smapvex16")
    if not (d / "SMAPVEX16_IA_GVSM_v0_ThetaProbe.csv").exists():
        import earthaccess

        earthaccess.login(strategy="netrc")
        earthaccess.download(earthaccess.search_data(short_name="SV16I_TPSM", count=10), str(d), show_progress=False)
    tp = pd.read_csv(d / "SMAPVEX16_IA_GVSM_v0_ThetaProbe.csv")
    tp["Site"] = tp.Site.replace(_SITE_FIX)
    tp["sm"] = tp[["A", "B", "C", "D"]].mean(axis=1)  # four readings per point
    tp["date"] = pd.to_datetime(tp.Date)
    gps = pd.concat([pd.read_csv(d / f"SMAPVEX16_IA_GVSM_GPS_IOP{i}.csv") for i in (1, 2)]).dropna()
    sites = gps.groupby("Site")[["Latitude", "Longitude"]].mean().rename(columns={"Latitude": "lat", "Longitude": "lon"})
    daily = tp.groupby(["Site", "Crop", "date"]).sm.agg(["mean", "std", "count"]).reset_index()
    daily = daily[daily.Site.isin(sites.index)].rename(columns={"Site": "site", "Crop": "crop", "mean": "sm"})
    return daily, sites


def _s1_incidence(item, lon, lat) -> np.ndarray:
    """Ellipsoid incidence angle (degrees) at points from a Sentinel-1 GRD annotation file."""
    import tempfile

    import planetary_computer as pc
    from scipy.interpolate import griddata

    from geolattice.rsense.sar.sentinel1 import read_geolocation_grid

    href = pc.sign(item.assets["schema-product-vv"].href)
    with tempfile.NamedTemporaryFile(suffix=".xml") as f:
        f.write(requests.get(href, timeout=300).content)
        f.flush()
        _, _, g = read_geolocation_grid(f.name)
    pts = np.column_stack([g["longitude"].ravel(), g["latitude"].ravel()])
    return griddata(pts, g["incidenceAngle"].ravel(), np.column_stack([lon, lat]), method="linear")


def smapvex16_samples(max_days: int = 1) -> pd.DataFrame:
    """Matched samples for Water Cloud Model calibration: site, date, crop, ThetaProbe soil moisture,
    Sentinel-1 VV and VH (dB, 3 x 3 pixels of 20 m around the site centre), incidence angle and
    Sentinel-2 NDVI interpolated to the radar date; cached."""
    path = _dir("smapvex16") / "wcm_samples.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["s1_time", "date"])
    from pyproj import Transformer

    from geolattice.rsense.access import stac_load, stac_search

    tp, sites = smapvex16_thetaprobe()
    crs = "EPSG:32615"
    t = Transformer.from_crs(4326, crs, always_xy=True)
    sx, sy = t.transform(sites.lon.values, sites.lat.values)
    # Sentinel-1 RTC on every pass of the campaign summer
    items = list(stac_search("sentinel-1-rtc", aoi=SMAPVEX16, dates="2016-05-20/2016-08-20"))
    grd = {(pd.Timestamp(i.properties["datetime"]).tz_localize(None).floor("D"), i.properties["sat:relative_orbit"]): i
           for i in stac_search("sentinel-1-grd", aoi=SMAPVEX16, dates="2016-05-20/2016-08-20")}  # annotation source per pass
    rows = []
    for it in items:
        day = pd.Timestamp(it.properties["datetime"]).tz_localize(None).floor("D")
        near = tp[(tp.date - day).abs() <= pd.Timedelta(days=max_days)]
        if near.empty:
            continue
        ds = stac_load([it], bands=["vv", "vh"], aoi=SMAPVEX16, crs=crs, resolution=20, groupby=None,
                       chunks={"x": 2048, "y": 2048}).isel(time=0).load()
        g = grd.get((day, it.properties["sat:relative_orbit"]))
        inc = _s1_incidence(g, sites.lon.values, sites.lat.values) if g is not None else np.full(len(sites), np.nan)
        for k, site in enumerate(sites.index):
            ix, iy = int(np.abs(ds.x - sx[k]).argmin()), int(np.abs(ds.y - sy[k]).argmin())
            win = ds.isel(x=slice(ix - 1, ix + 2), y=slice(iy - 1, iy + 2))
            vv, vh = float(win.vv.where(win.vv > 0).mean()), float(win.vh.where(win.vh > 0).mean())
            obs = near[near.site == site]
            if obs.empty or not np.isfinite(vv):
                continue
            rows.append({"site": site, "crop": obs.crop.iloc[0], "s1_time": day, "date": obs.date.iloc[0],
                         "sm": obs.sm.mean(), "sm_std": obs["std"].mean(), "vv_db": 10 * np.log10(vv),
                         "vh_db": 10 * np.log10(vh), "incidence": inc[k], "orbit": it.properties["sat:relative_orbit"],
                         "lon": sites.lon.iloc[k], "lat": sites.lat.iloc[k]})
    df = pd.DataFrame(rows)
    # Sentinel-2 NDVI at the sites, clear pixels only, interpolated to the radar dates
    s2 = list(stac_search("sentinel-2-l2a", aoi=SMAPVEX16, dates="2016-05-01/2016-09-15", query={"eo:cloud_cover": {"lt": 60}}))
    b = stac_load(s2, bands=["B04", "B08", "SCL"], aoi=SMAPVEX16, crs=crs, resolution=20, groupby="solar_day",
                  chunks={"x": 2048, "y": 2048}).load()
    clear = b.SCL.isin([4, 5])
    ndvi = ((b.B08 - b.B04) / (b.B08 + b.B04)).where(clear & (b.B08 > 0))
    nd = {}
    for k, site in enumerate(sites.index):
        ix, iy = int(np.abs(b.x - sx[k]).argmin()), int(np.abs(b.y - sy[k]).argmin())
        s = ndvi.isel(x=slice(ix - 2, ix + 3), y=slice(iy - 2, iy + 3)).mean(("y", "x")).to_series().dropna()
        nd[site] = s
    df["ndvi"] = [np.interp(pd.Timestamp(r.s1_time).value, nd[r.site].index.values.astype("datetime64[ns]").astype("int64"),
                            nd[r.site].values) if len(nd[r.site]) > 1 else np.nan for r in df.itertuples()]
    df.to_csv(path, index=False)
    return df


def smapvex16_scene(day: str = "2016-08-07", orbit: int = 165) -> xr.Dataset:
    """One Sentinel-1 pass over the SMAPVEX16 domain with the nearest clear Sentinel-2 NDVI and the
    2016 CDL crop map on a 20 m UTM grid, cached."""
    path = _dir("smapvex16") / f"scene_{day}_{orbit}.nc"
    if not path.exists():
        from geolattice.rsense.access import stac_load, stac_search

        crs = "EPSG:32615"
        t0 = pd.Timestamp(day)
        s1i = [i for i in stac_search("sentinel-1-rtc", aoi=SMAPVEX16, dates=f"{t0:%Y-%m-%d}/{t0:%Y-%m-%d}")
               if i.properties["sat:relative_orbit"] == orbit]
        s1 = stac_load(s1i, bands=["vv", "vh"], aoi=SMAPVEX16, crs=crs, resolution=20, chunks={"x": 2048, "y": 2048}).isel(time=0).load()
        s2i = list(stac_search("sentinel-2-l2a", aoi=SMAPVEX16, dates=f"{t0 - pd.Timedelta(days=20):%Y-%m-%d}/{t0 + pd.Timedelta(days=20):%Y-%m-%d}",
                               query={"eo:cloud_cover": {"lt": 30}}))
        b = stac_load(s2i, bands=["B04", "B08", "SCL"], aoi=SMAPVEX16, crs=crs, resolution=20, groupby="solar_day",
                      chunks={"x": 2048, "y": 2048}).load()
        b = b.isel(time=np.argsort(np.abs(b.time.values - np.datetime64(t0))))  # nearest date first
        nd = ((b.B08 - b.B04) / (b.B08 + b.B04)).where(b.SCL.isin([4, 5]) & (b.B08 > 0))
        ndvi = nd.isel(time=0)
        for k in range(1, nd.sizes["time"]):  # fill clouds from the next nearest dates
            ndvi = ndvi.fillna(nd.isel(time=k))
        cdl = _pc_static("usda-cdl", "cropland", SMAPVEX16, crs, 20, dates="2016-01-01/2016-12-31",
                         query={"usda_cdl:type": {"eq": "cropland"}})
        layers = {"vv_db": 10 * np.log10(s1.vv.where(s1.vv > 0)), "vh_db": 10 * np.log10(s1.vh.where(s1.vh > 0)),
                  "ndvi": ndvi.drop_vars("time", errors="ignore"), "cdl": cdl.reindex_like(s1.vv, method="nearest")}
        ds = xr.Dataset({k: v.drop_vars(["spatial_ref", "time"], errors="ignore") for k, v in layers.items()})
        ds.attrs.update(s1_date=day, orbit=orbit, s2_date=str(pd.Timestamp(b.time.values[0]).date()))
        for v in ("vv_db", "vh_db", "ndvi"):
            ds[v] = ds[v].astype("float32")
        sm.to_cache(ds.rio.write_crs(crs), str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- SMAP/Sentinel-1 active-passive (SPL2SMAP_S)
def spl2smap_s(bbox=IOWA_SAR, period=SAR_PERIOD, months=range(4, 11), name: str = "iowa_sar") -> xr.Dataset:
    """SPL2SMAP_S 1 km and 3 km soil moisture placed on EASE-Grid 2.0 cubes over the AOI with the
    EASE row and column indices stored in each granule (April to October), cached. Granules are
    downloaded, read and deleted."""
    path = _dir("spl2smap_s") / f"spl2smap_s_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    if not path.exists():
        import h5py

        granules = [g for g in sm.search_product("SPL2SMAP_S", bbox, period) if sm._granule_date(g).month in months]
        tmp = _dir("spl2smap_s", "granules")
        out = {}
        for res in ("1km", "3km"):
            grid = sm.ease2_grid(f"EASE2_M0{res[0]}km", aoi=bbox)
            out[res] = {"grid": grid, "r0": int(grid.row.min()), "c0": int(grid.col.min()), "frames": []}
        for i in range(0, len(granules), 20):
            files = sm.download_product(granules[i:i + 20], str(tmp), threads=8)
            for f in files:
                with h5py.File(f, "r") as h:
                    t = pd.Timestamp(sm._granule_date({"umm": {"GranuleUR": Path(f).name}}))
                    for res, o in out.items():
                        g = h[f"Soil_Moisture_Retrieval_Data_{res}"]
                        rows, cols = g[f"EASE_row_index_{res}"][:].astype(int), g[f"EASE_column_index_{res}"][:].astype(int)
                        val = g[f"soil_moisture_{res}"][:].astype("float32")
                        qc = g[f"retrieval_qual_flag_{res}"][:]
                        arr = np.full((o["grid"].sizes["y"], o["grid"].sizes["x"]), np.nan, "float32")
                        rr, cc = rows - o["r0"], cols - o["c0"]
                        ok = (rr >= 0) & (rr < arr.shape[0]) & (cc >= 0) & (cc < arr.shape[1]) & (val > -9000) & ((qc & 1) == 0)
                        arr[rr[ok], cc[ok]] = val[ok]
                        if np.isfinite(arr).any():
                            o["frames"].append((t, arr))
                Path(f).unlink()
        ds = xr.Dataset()
        for res, o in out.items():
            fr = sorted(o["frames"], key=lambda x: x[0])
            da = xr.DataArray(np.stack([a for _, a in fr]), dims=("time", f"y_{res}", f"x_{res}"),
                              coords={"time": [t for t, _ in fr], f"y_{res}": o["grid"].y.values, f"x_{res}": o["grid"].x.values})
            ds[f"sm_{res}"] = da.groupby("time").mean()  # granules of the same pass
        ds.attrs.update(product="SPL2SMAP_S", version="003", crs="EPSG:6933", quality="retrieval_qual_flag bit 0 clear")
        sm.to_cache(ds, str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- NISAR (ASF)
def nisar(bbox=IOWA_SAR, period=("2025-10-01", "2026-11-25"), name: str = "iowa_sar",
          collection: str = "NISAR_SM") -> xr.Dataset:
    """NISAR L3 soil moisture (provisional collection) over the AOI, one layer per granule, read by HTTP
    range requests from ASF (``sm.read_nisar_sm``), cached. Granules that do not cover the AOI are dropped."""
    path = _dir("nisar") / f"nisar_sm_{name}_{period[0][:7]}_{period[1][:7]}.nc"
    if not path.exists():
        from concurrent.futures import ThreadPoolExecutor

        import earthaccess

        granules = sm.search_product(collection, bbox, period)
        earthaccess.login(strategy="netrc")

        def one(g):
            f = earthaccess.open([g], show_progress=False)[0]
            d = sm.read_nisar_sm(f, aoi=bbox)
            return d if d.data_vars and int(d.sm.notnull().sum()) > 0 else None

        with ThreadPoolExecutor(6) as pool:
            parts = [p for p in pool.map(one, granules) if p is not None]
        ds = xr.concat(parts, "time", join="outer").sortby("time")
        sm.to_cache(ds.rio.write_crs("EPSG:6933"), str(path))
    return sm.open_cached(str(path))


# ----------------------------------------------------------------------------- downscaling stack (notebook 15)
DS_PERIOD = ("2019-04-01", "2021-10-31")
DS_MONTHS = range(4, 11)


def downscaling_grids(bbox=IOWA_SAR):
    """The 9 km EASE cells covering the AOI, their nested 1 km cells and the lon/lat box of the 9 km window."""
    from pyproj import Transformer

    g9 = sm.ease2_grid("EASE2_M09km", aoi=bbox)
    c9 = xr.DataArray(np.zeros((g9.sizes["y"], g9.sizes["x"]), "float32"), coords={"y": g9.y, "x": g9.x},
                      dims=("y", "x")).rio.write_crs("EPSG:6933")
    f1 = sm.nested_upsample(c9, 9)
    h = 4504.0
    xs = [float(g9.x.min()) - h, float(g9.x.max()) + h]
    ys = [float(g9.y.min()) - h, float(g9.y.max()) + h]
    lon, lat = Transformer.from_crs(6933, 4326, always_xy=True).transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    window = (min(lon) - 0.01, min(lat) - 0.01, max(lon) + 0.01, max(lat) + 0.01)
    return c9, f1, window


def downscaling_stack(bbox=IOWA_SAR, period=DS_PERIOD, months=DS_MONTHS, name: str = "iowa_sar") -> xr.Dataset:
    """Daily inputs of the downscaling comparison on nested 9 km and 1 km EASE-Grid 2.0 cells, cached.

    ``smap_9km`` (SMAP L3 Enhanced, all successful AM-first retrievals) on the coarse grid; on the 1 km
    grid ``lst_day``, ``lst_night`` (MODIS MOD11A1, K), ``ndvi`` (MOD13A1, interpolated daily), ``red``,
    ``nir``, ``swir22`` (MCD43A4 NBAR), ``rain_3d`` and ``rain_10d`` (gridMET, mm before the day) and the
    static ``sand``, ``clay``, ``soc`` (SoilGrids 0 to 5 cm), ``elevation``, ``slope`` (Copernicus DEM),
    ``corn``, ``soy`` (CDL fractions of the year) for April to October.
    """
    path = _dir("downscaling") / f"stack_{name}_{period[0][:4]}_{period[1][:4]}.nc"
    if path.exists():
        return sm.open_cached(str(path))
    c9, f1, win = downscaling_grids(bbox)
    days = pd.date_range(*period, freq="D")
    days = days[days.month.isin(list(months))]
    raw = smap_l3e().sel(x=c9.x, y=c9.y, method="nearest").assign_coords(x=c9.x, y=c9.y)
    coarse = sm.daily_cube(sm.mask_quality(raw.sel(time=slice(*period)), level="all")).sm.reindex(time=days)
    out = xr.Dataset({"smap_9km": coarse.rename({"y": "y9", "x": "x9"})})
    fine = {}
    for kind in ("lst", "ndvi", "nbar"):
        parts = []
        for yr in sorted(set(days.year)):
            m = modis(kind, win, f"{yr}-03-15" if kind == "ndvi" else f"{yr}-04-01", f"{yr}-10-31", name=f"ds_{name}_{yr}")
            parts.append(m)
        mm = xr.concat(parts, "time").sortby("time")
        mm = mm.groupby(mm.time.dt.floor("D").rename("time")).mean()  # one image per day
        for v in mm.data_vars:
            r = sm.regrid(mm[v].rio.write_crs("EPSG:6933"), f1, method="average")
            if kind == "ndvi":  # 16-day composites to daily
                r = r.reindex(time=r.time.to_index().union(days))
                r = sm._interp_gaps(r, 40).reindex(time=days)
            fine[v] = r.reindex(time=days)
    pr = gridmet(win, period[0], period[1], variables=("pr",), name=f"ds_{name}").pr
    pr = pr.rename({"lat": "y", "lon": "x"}).rio.write_crs("EPSG:4326") if "lat" in pr.dims else pr.rio.write_crs("EPSG:4326")
    pr = pr.sortby("time").reindex(time=pd.date_range(period[0], period[1], freq="D"))
    pr = _spatial_only(pr).astype("float32")
    pr.attrs, pr.encoding = {}, {}  # drop the file's grid_mapping reference
    p1 = sm.regrid(pr.rio.write_crs("EPSG:4326"), f1, method="bilinear")
    fine["rain_3d"] = p1.rolling(time=3, min_periods=1).sum().shift(time=1).reindex(time=days)
    fine["rain_10d"] = p1.rolling(time=10, min_periods=1).sum().shift(time=1).reindex(time=days)
    soil = soilgrids_grid(win, f"ds_{name}").isel(depth=0)
    for v in ("sand", "clay", "soc"):
        fine[v] = sm.regrid(soil[v].rio.write_crs("EPSG:4326"), f1, method="average")
    st = static_layers(win, name=f"ds_{name}", crs="EPSG:6933", resolution=250)
    fine["elevation"] = sm.regrid(st.dem, f1, method="average")
    gy, gx = np.gradient(st.dem.values.astype("float64"), 250.0)
    slope = st.dem.copy(data=np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32"))
    fine["slope"] = sm.regrid(slope, f1, method="average")
    crops = {}
    for name_, code in (("corn", 1), ("soy", 5)):
        fr = [sm.regrid((st[f"cdl_{yr}"] == code).astype("float32").rio.write_crs("EPSG:6933"), f1, method="average")
              .expand_dims(year=[yr]) for yr in (2019, 2020, 2021)]
        crops[name_] = xr.concat(fr, "year").sel(year=xr.DataArray(days.year, coords={"time": days}, dims="time")).drop_vars("year")
    fine.update(crops)
    for k, v in fine.items():
        out[k] = _spatial_only(v).astype("float32")
    out = out.assign_coords(y9=c9.y.values, x9=c9.x.values)
    sm.to_cache(out.rio.write_crs("EPSG:6933"), str(path))
    return sm.open_cached(str(path))


def _spatial_only(da):
    keep = set(da.dims) | {"spatial_ref"}
    return da.drop_vars([c for c in da.coords if c not in keep])
