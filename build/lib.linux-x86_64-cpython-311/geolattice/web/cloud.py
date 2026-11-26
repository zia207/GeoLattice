"""
Read geospatial files straight from cloud object storage and HTTP servers.

GDAL reads remote files through virtual file systems (``/vsis3/``,
``/vsigs/``, ``/vsiaz/``, ``/vsicurl/``), fetching only the byte ranges it
needs. :func:`to_vsi_path` converts the familiar URL forms to those paths,
:func:`cloud_env` sets the GDAL options that make range reads fast (or
anonymous), and :func:`remote_info` checks whether a server supports range
requests at all.

Example:
    from geolattice.web import cloud_env, to_vsi_path
    import geolattice.rsense as rs

    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/18/T/UN/2025/8/S2C_18TUN_20250802_0_L2A/B04.tif"
    with cloud_env(anonymous=True, region="us-west-2"):
        red = rs.read(to_vsi_path(url), chunks={"x": 1024, "y": 1024})
"""

from __future__ import annotations

import contextlib
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Optional

_SCHEMES = {
    "s3": "/vsis3/",
    "gs": "/vsigs/",
    "gcs": "/vsigs/",
    "az": "/vsiaz/",
    "azure": "/vsiaz/",
    "abfs": "/vsiaz/",
    "abfss": "/vsiaz/",
    "http": "/vsicurl/",
    "https": "/vsicurl/",
}
_ARCHIVES = {".zip": "/vsizip/", ".tar": "/vsitar/", ".tgz": "/vsitar/", ".gz": "/vsigzip/"}

#: GDAL options that make range reads of COGs and other cloud-native files fast.
CLOUD_OPTIMIZED_OPTIONS: Dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",   # do not list the "directory" of a URL
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",    # combine adjacent tile requests
    "GDAL_HTTP_MULTIPLEX": "YES",                   # HTTP/2 multiplexing when available
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",         # read headers in one request
}


def to_vsi_path(url: str) -> str:
    """
    Convert a URL or URI to the GDAL virtual-file-system path that reads it.

    ``s3://bucket/key`` becomes ``/vsis3/bucket/key``, ``gs://`` becomes
    ``/vsigs/``, ``az://``/``abfs://`` become ``/vsiaz/``, and ``http(s)://``
    becomes ``/vsicurl/https://...``. A ``!`` after an archive name selects a
    file inside it: ``https://host/data.zip!/dem.tif`` becomes
    ``/vsizip//vsicurl/https://host/data.zip/dem.tif``. Local paths and paths
    that already start with ``/vsi`` are returned unchanged.
    """
    text = str(url)
    if text.startswith("/vsi"):
        return text
    inner = None
    if "!" in text:
        text, inner = text.split("!", 1)
    scheme, sep, rest = text.partition("://")
    if not sep:
        base = text
    elif scheme.lower() in ("http", "https"):
        base = _SCHEMES["http"] + text
    elif scheme.lower() in _SCHEMES:
        base = _SCHEMES[scheme.lower()] + rest
    elif scheme.lower() == "file":
        base = rest
    else:
        raise ValueError(f"Unsupported URL scheme {scheme!r} in {url!r}")
    if inner is None:
        return base
    prefix = next((v for k, v in _ARCHIVES.items() if base.lower().endswith(k)), "/vsizip/")
    return f"{prefix}{base}/{inner.lstrip('/')}"


def cloud_options(
    *,
    anonymous: Optional[bool] = None,
    region: Optional[str] = None,
    requester_pays: bool = False,
    endpoint: Optional[str] = None,
    optimized: bool = True,
    **extra: Any,
) -> Dict[str, str]:
    """
    Build the GDAL configuration options for remote reads.

    Args:
        anonymous: ``True`` for public buckets (no credentials are sent).
        region: AWS region, e.g. ``"us-west-2"``.
        requester_pays: Accept charges on requester-pays buckets (USGS Landsat on S3).
        endpoint: S3-compatible endpoint host (MinIO, Ceph, Wasabi, ...).
        optimized: Include :data:`CLOUD_OPTIMIZED_OPTIONS`.
        **extra: Any other GDAL options, e.g. ``AWS_PROFILE="work"``.
    """
    options: Dict[str, str] = dict(CLOUD_OPTIMIZED_OPTIONS) if optimized else {}
    if anonymous:
        options["AWS_NO_SIGN_REQUEST"] = "YES"
        options["GS_NO_SIGN_REQUEST"] = "YES"
        options["AZURE_NO_SIGN_REQUEST"] = "YES"
    if region:
        options["AWS_REGION"] = region
    if requester_pays:
        options["AWS_REQUEST_PAYER"] = "requester"
    if endpoint:
        options["AWS_S3_ENDPOINT"] = endpoint
        options.setdefault("AWS_VIRTUAL_HOSTING", "FALSE")
    options.update({k: str(v) for k, v in extra.items()})
    return options


@contextlib.contextmanager
def cloud_env(**kwargs: Any) -> Iterator[Dict[str, str]]:
    """
    Context manager that applies :func:`cloud_options` to GDAL while it is active.

    The options are set both in the ``osgeo.gdal`` library and in rasterio's
    GDAL (a rasterio wheel can bundle its own copy), and restored on exit.
    Accepts the same keywords as :func:`cloud_options`; yields the options.
    """
    options = cloud_options(**kwargs)
    previous: Dict[str, Optional[str]] = {}
    gdal = _osgeo_gdal()
    if gdal is not None:
        for key, value in options.items():
            previous[key] = gdal.GetConfigOption(key)
            gdal.SetConfigOption(key, value)
    try:
        try:
            import rasterio
        except ImportError:
            yield options
        else:
            with rasterio.Env(**options):
                yield options
    finally:
        if gdal is not None:
            for key, value in previous.items():
                gdal.SetConfigOption(key, value)


def remote_info(url: str, *, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Ask a server about a remote file with a one-byte range request.

    Returns ``url``, ``status``, ``size`` (bytes, when reported),
    ``content_type``, ``range_requests`` (``True`` when the server answered
    206 Partial Content, which cloud-native reads need), ``etag`` and
    ``error``. Network failures (DNS, refused or reset connections, time-outs)
    are reported with ``status=None`` and a message in ``error`` instead of
    being raised.
    Only ``http(s)`` URLs are supported; sign S3/GCS URLs or use their HTTPS form.
    """
    if not str(url).lower().startswith(("http://", "https://")):
        raise ValueError("remote_info needs an http(s) URL.")
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "geolattice"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            headers = response.headers
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "size": None, "content_type": None,
                "range_requests": False, "etag": None, "error": f"HTTP {exc.code} {exc.reason}"}
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return {"url": url, "status": None, "size": None, "content_type": None,
                "range_requests": False, "etag": None, "error": str(reason)}
    size = None
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        size = int(total) if total.isdigit() else None
    elif headers.get("Content-Length") and status == 200:
        size = int(headers["Content-Length"])
    return {
        "url": url,
        "status": status,
        "size": size,
        "content_type": headers.get("Content-Type"),
        "range_requests": status == 206,
        "etag": headers.get("ETag"),
        "error": None,
    }


def _osgeo_gdal() -> Any:
    try:
        from osgeo import gdal
    except ImportError:
        return None
    return gdal


__all__ = ["CLOUD_OPTIMIZED_OPTIONS", "to_vsi_path", "cloud_options", "cloud_env", "remote_info"]
