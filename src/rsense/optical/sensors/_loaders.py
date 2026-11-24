"""One-call loaders: search, load, scale and cloud-mask a sensor's imagery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

# canonical band -> STAC asset key, per (catalog, sensor)
ASSET_KEYS: dict[tuple[str, str], dict[str, str]] = {
    ("planetary-computer", "sentinel2"): {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05",
        "rededge2": "B06", "rededge3": "B07", "nir": "B08", "nir08": "B8A",
        "swir16": "B11", "swir22": "B12", "qa": "SCL"},
    ("earth-search", "sentinel2"): {
        "coastal": "coastal", "blue": "blue", "green": "green", "red": "red",
        "rededge1": "rededge1", "rededge2": "rededge2", "rededge3": "rededge3", "nir": "nir",
        "nir08": "nir08", "swir16": "swir16", "swir22": "swir22", "qa": "scl"},
    ("planetary-computer", "landsat"): {
        "coastal": "coastal", "blue": "blue", "green": "green", "red": "red", "nir": "nir08",
        "swir16": "swir16", "swir22": "swir22", "thermal": "lwir11", "qa": "qa_pixel"},
    ("planetary-computer", "modis"): {
        "red": "sur_refl_b01", "nir": "sur_refl_b02", "blue": "sur_refl_b03",
        "green": "sur_refl_b04", "swir12": "sur_refl_b05", "swir16": "sur_refl_b06",
        "swir22": "sur_refl_b07", "qa": "sur_refl_state_500m"},
    ("nasa-lpcloud", "hls_l30"): {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "nir": "B05",
        "swir16": "B06", "swir22": "B07", "thermal": "B10", "qa": "Fmask"},
    ("nasa-lpcloud", "hls_s30"): {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05",
        "rededge2": "B06", "rededge3": "B07", "nir": "B08", "nir08": "B8A",
        "swir16": "B11", "swir22": "B12", "qa": "Fmask"},
}

COLLECTIONS = {
    ("planetary-computer", "sentinel2"): "sentinel-2-l2a",
    ("earth-search", "sentinel2"): "sentinel-2-l2a",
    ("planetary-computer", "landsat"): "landsat-c2-l2",
    ("planetary-computer", "modis"): "modis-09A1-061",
    ("nasa-lpcloud", "hls_l30"): "HLSL30_2.0",
    ("nasa-lpcloud", "hls_s30"): "HLSS30_2.0",
}

DEFAULT_BANDS = {
    "sentinel2": ("blue", "green", "red", "rededge1", "nir", "swir16", "swir22"),
    "landsat": ("blue", "green", "red", "nir", "swir16", "swir22"),
    "modis": ("blue", "green", "red", "nir", "swir16", "swir22"),
    "hls_l30": ("blue", "green", "red", "nir", "swir16", "swir22"),
    "hls_s30": ("blue", "green", "red", "nir", "swir16", "swir22"),
}
DEFAULT_RES = {"sentinel2": 10, "landsat": 30, "modis": 500, "hls_l30": 30, "hls_s30": 30}


def _load(sensor: str, aoi: Any, dates: Any, *, bands: Sequence[str] | None, catalog: str,
          max_cloud: float | None, resolution: float | None, crs: Any, scale: bool,
          mask: bool, mask_kwargs: dict | None, query: dict | None, chunks: Any,
          rename: bool, **kwargs: Any) -> xr.Dataset:
    from ...access import stac_load, stac_search
    from ..qa import mask_clouds
    from ..scaling import scale as scale_fn

    key = (catalog, sensor)
    if key not in ASSET_KEYS:
        raise ValueError(f"No loader configured for {sensor} on {catalog!r}; "
                         f"available: {sorted(ASSET_KEYS)}")
    assets = ASSET_KEYS[key]
    wanted = list(bands or DEFAULT_BANDS[sensor])
    if mask and "qa" not in wanted:
        wanted.append("qa")
    unknown = [b for b in wanted if b not in assets]
    if unknown:
        raise ValueError(f"Bands {unknown} unavailable for {sensor}; choose from {sorted(assets)}")
    items = stac_search(COLLECTIONS[key], aoi=aoi, dates=dates,
                        max_cloud=None if sensor == "modis" else max_cloud,
                        catalog=catalog, query=query, **kwargs)
    ds = stac_load(items, bands=[assets[b] for b in wanted], aoi=aoi,
                   resolution=resolution or DEFAULT_RES[sensor], crs=crs, chunks=chunks,
                   sensor=sensor)
    if scale:
        ds = scale_fn(ds, sensor=sensor)
    if mask:
        ds = mask_clouds(ds, sensor=sensor, drop_qa=True, **(mask_kwargs or {}))
    if rename:  # expose canonical names so every downstream call is sensor-agnostic
        inv = {v: k for k, v in assets.items() if v in ds.data_vars}
        ds = ds.rename(inv)
    ds.attrs["pyrsense:sensor"] = sensor
    return ds


def load_sentinel2(aoi: Any, dates: Any, *, bands: Sequence[str] | None = None,
                   catalog: str = "planetary-computer", max_cloud: float | None = 30,
                   resolution: float | None = None, crs: Any = None, scale: bool = True,
                   mask: bool = True, mask_kwargs: dict | None = None,
                   query: dict | None = None, chunks: Any = None, rename: bool = True,
                   **kwargs: Any) -> xr.Dataset:
    """Search, load, scale to reflectance and cloud-mask Sentinel-2 L2A.

    Returns a lazy Dataset with canonical band names (``blue``, ``red``, ``nir``...).
    """
    return _load("sentinel2", aoi, dates, bands=bands, catalog=catalog, max_cloud=max_cloud,
                 resolution=resolution, crs=crs, scale=scale, mask=mask, mask_kwargs=mask_kwargs,
                 query=query, chunks=chunks, rename=rename, **kwargs)


def load_landsat(aoi: Any, dates: Any, *, bands: Sequence[str] | None = None,
                 platforms: Sequence[str] | None = ("landsat-8", "landsat-9"),
                 catalog: str = "planetary-computer", max_cloud: float | None = 30,
                 resolution: float | None = None, crs: Any = None, scale: bool = True,
                 mask: bool = True, mask_kwargs: dict | None = None,
                 query: dict | None = None, chunks: Any = None, rename: bool = True,
                 **kwargs: Any) -> xr.Dataset:
    """Search, load, scale and cloud-mask Landsat Collection 2 Level-2.

    ``platforms`` filters missions (e.g. ``("landsat-5", "landsat-7")``); ``None`` keeps all.
    """
    q = dict(query or {})
    if platforms:
        q["platform"] = {"in": list(platforms)}
    return _load("landsat", aoi, dates, bands=bands, catalog=catalog, max_cloud=max_cloud,
                 resolution=resolution, crs=crs, scale=scale, mask=mask, mask_kwargs=mask_kwargs,
                 query=q, chunks=chunks, rename=rename, **kwargs)


def load_modis(aoi: Any, dates: Any, *, bands: Sequence[str] | None = None,
               catalog: str = "planetary-computer", resolution: float | None = None,
               crs: Any = None, scale: bool = True, mask: bool = True,
               mask_kwargs: dict | None = None, chunks: Any = None, rename: bool = True,
               **kwargs: Any) -> xr.Dataset:
    """Search, load, scale and cloud-mask MODIS 8-day surface reflectance (MOD09A1/MYD09A1)."""
    return _load("modis", aoi, dates, bands=bands, catalog=catalog, max_cloud=None,
                 resolution=resolution, crs=crs, scale=scale, mask=mask, mask_kwargs=mask_kwargs,
                 query=None, chunks=chunks, rename=rename, **kwargs)


def _earthdata_gdal_auth() -> None:
    """Log in to Earthdata and configure GDAL to send the session cookies (needed for HLS)."""
    import os

    from ...access.stac import _require

    earthaccess = _require("earthaccess")
    earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
    cookie = os.path.expanduser("~/.pyrsense_cookies.txt")
    os.environ.setdefault("GDAL_HTTP_COOKIEFILE", cookie)
    os.environ.setdefault("GDAL_HTTP_COOKIEJAR", cookie)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "TIF")


def load_hls(aoi: Any, dates: Any, *, product: str = "both", bands: Sequence[str] | None = None,
             max_cloud: float | None = 30, resolution: float | None = None, crs: Any = None,
             scale: bool = True, mask: bool = True, mask_kwargs: dict | None = None,
             chunks: Any = None, **kwargs: Any) -> xr.Dataset:
    """Load Harmonized Landsat Sentinel-2 (HLS v2.0) from NASA LP DAAC.

    ``product``: ``"L30"``, ``"S30"`` or ``"both"`` (merged on canonical band names and
    sorted by time, giving a dense 30 m series). Needs an Earthdata Login (``~/.netrc`` or
    ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD``).
    """
    _earthdata_gdal_auth()
    which = {"L30": ["hls_l30"], "S30": ["hls_s30"], "BOTH": ["hls_l30", "hls_s30"]}[product.upper()]
    parts = [_load(s, aoi, dates, bands=bands, catalog="nasa-lpcloud", max_cloud=max_cloud,
                   resolution=resolution, crs=crs, scale=scale, mask=mask,
                   mask_kwargs=mask_kwargs, query=None, chunks=chunks, rename=True, **kwargs)
             for s in which]
    if len(parts) == 1:
        return parts[0]
    common = sorted(set(parts[0].data_vars) & set(parts[1].data_vars))
    ds = xr.concat([p[common] for p in parts], dim="time").sortby("time")
    ds.attrs["pyrsense:sensor"] = "hls"
    return ds
