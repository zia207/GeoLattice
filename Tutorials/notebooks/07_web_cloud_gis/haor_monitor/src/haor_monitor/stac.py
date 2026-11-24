"""Finding and reading Sentinel-2 scenes from a STAC API (Planetary Computer by default)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from .config import Settings
from .process import reflectance


def search(settings: Settings, start: date, end: date) -> list:
    """All scenes over the AOI between two dates (inclusive), filtered by scene cloud cover."""
    import planetary_computer
    import pystac_client

    modifier = planetary_computer.sign_inplace if "planetarycomputer" in settings.stac_url else None
    client = pystac_client.Client.open(settings.stac_url, modifier=modifier)
    found = client.search(
        collections=[settings.collection],
        bbox=list(settings.aoi),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": settings.max_cloud}},
    )
    return sorted(found.items(), key=lambda item: item.datetime)


def group_by_day(items: list) -> dict[date, list]:
    """Scenes of the same overpass (neighbouring tiles) are mosaicked into one daily observation."""
    days: dict[date, list] = defaultdict(list)
    for item in items:
        days[item.datetime.date()].append(item)
    return dict(sorted(days.items()))


def load_day(items: list, settings: Settings) -> dict[str, np.ndarray]:
    """Green, red, NIR reflectance and SCL for one day, mosaicked on the AOI grid."""
    import odc.stac

    ds = odc.stac.load(
        items,
        bands=["B03", "B04", "B08", "SCL"],
        bbox=list(settings.aoi),
        crs=settings.crs,
        resolution=settings.resolution,
        groupby="solar_day",
        dtype="uint16",
    ).isel(time=0)
    baseline = items[0].properties.get("s2:processing_baseline", "04.00")
    return {
        "green": reflectance(ds.B03.values, baseline),
        "red": reflectance(ds.B04.values, baseline),
        "nir": reflectance(ds.B08.values, baseline),
        "scl": ds.SCL.values,
        "baseline": str(baseline),
        "scenes": [item.id for item in items],
    }
