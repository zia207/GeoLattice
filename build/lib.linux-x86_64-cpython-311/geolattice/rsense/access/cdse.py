"""Copernicus Data Space Ecosystem (CDSE): Sentinel-1/2/3/5P search and download.

Search uses the public STAC API (no login). Downloads need a free CDSE account;
credentials are read from ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` if not passed.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Any

from .stac import stac_search

CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                  "protocol/openid-connect/token")
CDSE_ODATA = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
CDSE_CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

COLLECTIONS = {
    "sentinel1-grd": "sentinel-1-grd", "sentinel1-slc": "sentinel-1-slc",
    "sentinel2-l2a": "sentinel-2-l2a", "sentinel2-l1c": "sentinel-2-l1c",
}


def cdse_search(collection: str, *, aoi: Any = None, dates: Any = None,
                max_cloud: float | None = None, **kwargs: Any):
    """Search CDSE STAC. ``collection`` may be a CDSE id or a short alias from ``COLLECTIONS``."""
    return stac_search(COLLECTIONS.get(collection, collection), aoi=aoi, dates=dates,
                       max_cloud=max_cloud, catalog=CDSE_STAC, **kwargs)


def cdse_token(username: str | None = None, password: str | None = None) -> str:
    """Obtain an access token (valid ~10 minutes)."""
    import requests

    username = username or os.environ.get("CDSE_USERNAME")
    password = password or os.environ.get("CDSE_PASSWORD")
    if not (username and password):
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD or pass credentials")
    r = requests.post(CDSE_TOKEN_URL, data={"client_id": "cdse-public", "grant_type": "password",
                                            "username": username, "password": password},
                      timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def _product_uuid(name_or_id: str) -> str:
    import re

    import requests

    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", name_or_id):
        return name_or_id
    name = name_or_id if name_or_id.endswith(".SAFE") else f"{name_or_id}.SAFE"
    r = requests.get(CDSE_CATALOGUE, params={"$filter": f"Name eq '{name}'"}, timeout=60)
    r.raise_for_status()
    values = r.json().get("value", [])
    if not values:
        raise LookupError(f"No CDSE product named {name}")
    return values[0]["Id"]


def cdse_download(item_or_id: Any, out_dir: str | PathLike = "data", *, token: str | None = None,
                  chunk_mb: int = 8) -> Path:
    """Download a full product (zipped SAFE) by STAC item, product name or CDSE UUID."""
    import requests

    name = getattr(item_or_id, "id", item_or_id)
    uuid = _product_uuid(str(name))
    token = token or cdse_token()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{str(name).removesuffix('.SAFE')}.zip"
    with requests.get(f"{CDSE_ODATA}({uuid})/$value", headers={"Authorization": f"Bearer {token}"},
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            f.writelines(r.iter_content(chunk_mb * 2**20))
    return path
