"""NASA Earthdata access (MODIS, HLS, GEDI, ICESat-2...) via ``earthaccess``."""

from __future__ import annotations

from os import PathLike
from typing import Any

from .stac import _date_range, _require, aoi_to_geojson


def earthdata_search(short_name: str, *, aoi: Any = None, dates: Any = None,
                     version: str | None = None, count: int = -1, **kwargs: Any):
    """Search NASA CMR for granules, e.g. ``earthdata_search("MOD13Q1", aoi=..., dates=...)``.

    Requires an Earthdata Login; ``earthaccess.login()`` is called automatically and reads
    credentials from ``~/.netrc`` or the ``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` vars.
    """
    earthaccess = _require("earthaccess")
    earthaccess.login(strategy="environment" if _env_creds() else "netrc")
    params: dict[str, Any] = {"short_name": short_name, "count": count, **kwargs}
    if version:
        params["version"] = version
    if aoi is not None:
        from shapely.geometry import shape

        params["bounding_box"] = tuple(shape(aoi_to_geojson(aoi)).bounds)
    if dates is not None:
        rng = _date_range(dates)
        params["temporal"] = tuple(rng.split("/")) if "/" in rng else (rng, rng)
    return earthaccess.search_data(**params)


def earthdata_download(granules, out_dir: str | PathLike = "data", threads: int = 8) -> list[str]:
    """Download granules found by :func:`earthdata_search`; returns local file paths."""
    earthaccess = _require("earthaccess")
    return [str(p) for p in earthaccess.download(granules, local_path=str(out_dir), threads=threads)]


def _env_creds() -> bool:
    import os

    return bool(os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
