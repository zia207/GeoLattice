"""Publish your outputs as a static STAC catalog (pystac), with projection metadata."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any


def export_stac(files: Sequence[str | PathLike], catalog_dir: str | PathLike, *,
                collection_id: str = "pyrsense-outputs", description: str = "PyRSense products",
                datetimes: Sequence[datetime] | None = None, license: str = "CC-BY-4.0",
                properties: dict[str, Any] | None = None, asset_key: str = "data") -> Path:
    """Create a self-contained STAC catalog with one item per raster file.

    Bounding boxes/geometry come from each file (reprojected to WGS84); the projection
    extension records EPSG, shape and transform. Returns the catalog.json path.
    """
    import pystac
    import rasterio
    from pystac.extensions.projection import ProjectionExtension
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, mapping

    out = Path(catalog_dir)
    out.mkdir(parents=True, exist_ok=True)
    items, bboxes = [], []
    for i, f in enumerate(files):
        f = Path(f)
        with rasterio.open(f) as src:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            epsg = src.crs.to_epsg() if src.crs else None
            shape, transform = [src.height, src.width], list(src.transform)[:6]
        dt = datetimes[i] if datetimes else datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
        item = pystac.Item(id=f.stem, geometry=mapping(box(*b)), bbox=list(b), datetime=dt,
                           properties=dict(properties or {}))
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        proj.epsg, proj.shape, proj.transform = epsg, shape, transform
        item.add_asset(asset_key, pystac.Asset(href=str(f.resolve()),
                                               media_type=pystac.MediaType.COG, roles=["data"]))
        items.append(item)
        bboxes.append(b)
    minx = min(b[0] for b in bboxes)
    miny = min(b[1] for b in bboxes)
    maxx = max(b[2] for b in bboxes)
    maxy = max(b[3] for b in bboxes)
    times = [it.datetime for it in items]
    coll = pystac.Collection(
        id=collection_id, description=description, license=license,
        extent=pystac.Extent(pystac.SpatialExtent([[minx, miny, maxx, maxy]]),
                             pystac.TemporalExtent([[min(times), max(times)]])))
    coll.add_items(items)
    cat = pystac.Catalog(id=f"{collection_id}-catalog", description=description)
    cat.add_child(coll)
    cat.normalize_hrefs(str(out))
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return out / "catalog.json"
