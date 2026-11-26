"""Alaska Satellite Facility (ASF): Sentinel-1 SLC/GRD/burst search and download.

Requires ``asf_search`` (``pip install geolattice[rsense-sar]``) and an Earthdata Login for downloads.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

from .stac import _date_range, _require, aoi_to_geojson


def asf_search(*, aoi: Any = None, dates: Any = None, product: str = "SLC",
               beam_mode: str = "IW", flight_direction: str | None = None,
               relative_orbit: int | None = None, max_results: int | None = None,
               **kwargs: Any):
    """Search Sentinel-1 products. ``product``: ``SLC``, ``GRD_HD``, ``BURST``, ``OCN``."""
    asf = _require("asf_search", "sar")
    from shapely.geometry import shape

    params: dict[str, Any] = {"platform": asf.PLATFORM.SENTINEL1, "processingLevel": product,
                              "beamMode": beam_mode, **kwargs}
    if aoi is not None:
        params["intersectsWith"] = shape(aoi_to_geojson(aoi)).wkt
    if dates is not None:
        start, _, end = _date_range(dates).partition("/")
        params["start"], params["end"] = start, end or start
    if flight_direction:
        params["flightDirection"] = flight_direction.upper()
    if relative_orbit is not None:
        params["relativeOrbit"] = relative_orbit
    if max_results:
        params["maxResults"] = max_results
    return asf.search(**params)


def _earthdata_netrc() -> tuple[str, str] | None:
    """Earthdata Login credentials from ``~/.netrc`` (machine ``urs.earthdata.nasa.gov``), if any."""
    import netrc

    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        return None
    return (auth[0], auth[2]) if auth else None


def asf_download(results, out_dir: str | PathLike = "data", *, username: str | None = None,
                 password: str | None = None, processes: int = 4, all_files: bool = True) -> None:
    """Download search results using Earthdata credentials (arguments, or ``~/.netrc`` with
    ``machine urs.earthdata.nasa.gov``).

    ``all_files=True`` also fetches a product's additional files -- for ``BURST`` results the
    metadata XML (annotation, calibration and noise) that goes with each burst GeoTIFF.
    """
    asf = _require("asf_search", "sar")
    session = asf.ASFSession()
    creds = (username, password) if username and password else _earthdata_netrc()
    if creds:
        session.auth_with_creds(*creds)
    kwargs = {"fileType": asf.FileDownloadType.ALL_FILES} if all_files else {}
    for product in results:
        product.download(path=str(out_dir), session=session, **kwargs)


def baseline_stack(reference_scene: str):
    """Temporal/perpendicular baseline stack for InSAR pair selection."""
    asf = _require("asf_search", "sar")
    return asf.stack_from_id(reference_scene)
