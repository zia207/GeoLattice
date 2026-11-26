"""
XYZ web-map tiles: tile math, on-the-fly rendering and static tile pyramids.

Web maps (Leaflet, MapLibre, OpenLayers, Google Maps) show the world as
256 × 256 pixel images in the Web Mercator projection (EPSG:3857). Zoom level
``z`` has ``2**z × 2**z`` tiles; tile ``(x, y)`` counts from the top-left. The
functions here convert between coordinates and tiles, render any raster that
rasterio can open into a PNG tile, and write a folder of ``{z}/{x}/{y}.png``
tiles that any static web host can serve.

Example:
    from geolattice.web import make_xyz_tiles, render_tile, tiles_for_bounds

    png = render_tile("ndvi_cog.tif", x=301, y=376, z=10, colormap="RdYlGn", rescale=(-0.2, 0.9))
    make_xyz_tiles("ndvi_cog.tif", "tiles/ndvi", zooms=range(8, 13), colormap="RdYlGn")
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from geolattice.web.cloud import to_vsi_path

EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS              # half the Web Mercator world width, metres
MAX_LATITUDE = 85.0511287798066

Rescale = Union[Tuple[float, float], Sequence[Tuple[float, float]]]


class Tile(NamedTuple):
    """An XYZ tile index."""

    x: int
    y: int
    z: int


# ----------------------------------------------------------------------------- tile math
def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tile:
    """Tile containing a longitude/latitude point at ``zoom``."""
    lat = max(min(lat, MAX_LATITUDE), -MAX_LATITUDE)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return Tile(min(max(x, 0), n - 1), min(max(y, 0), n - 1), zoom)


def tile_bounds(x: int, y: int, z: int, *, crs: str = "EPSG:4326") -> Tuple[float, float, float, float]:
    """
    Bounds ``(west, south, east, north)`` of a tile.

    ``crs`` is ``"EPSG:4326"`` (degrees) or ``"EPSG:3857"`` (Web Mercator metres).
    """
    n = 2 ** z
    if crs.upper() in ("EPSG:3857", "EPSG:900913"):
        size = 2 * ORIGIN_SHIFT / n
        west = -ORIGIN_SHIFT + x * size
        north = ORIGIN_SHIFT - y * size
        return (west, north - size, west + size, north)
    if crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        raise ValueError("crs must be 'EPSG:4326' or 'EPSG:3857'")

    def lat(row: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    return (x / n * 360.0 - 180.0, lat(y + 1), (x + 1) / n * 360.0 - 180.0, lat(y))


def tiles_for_bounds(west: float, south: float, east: float, north: float,
                     zooms: Union[int, Iterable[int]]) -> Iterator[Tile]:
    """All tiles intersecting a longitude/latitude box at the given zoom level(s)."""
    for z in ([zooms] if isinstance(zooms, int) else zooms):
        top_left = lonlat_to_tile(west, north, z)
        bottom_right = lonlat_to_tile(np.nextafter(east, -np.inf), np.nextafter(south, np.inf), z)
        for x in range(top_left.x, bottom_right.x + 1):
            for y in range(top_left.y, bottom_right.y + 1):
                yield Tile(x, y, z)


def tile_resolution(zoom: int, *, latitude: float = 0.0, tile_size: int = 256) -> float:
    """Ground size of one pixel (metres) at ``zoom`` and ``latitude``."""
    return 2 * ORIGIN_SHIFT / (tile_size * 2 ** zoom) * math.cos(math.radians(latitude))


def zoom_for_resolution(resolution_m: float, *, latitude: float = 0.0, tile_size: int = 256) -> int:
    """Smallest zoom whose pixels are at least as fine as ``resolution_m`` metres."""
    z = math.log2(2 * ORIGIN_SHIFT * math.cos(math.radians(latitude)) / (tile_size * resolution_m))
    return max(0, int(math.ceil(z - 1e-9)))


def raster_web_info(source: Union[str, Path, Any]) -> Dict[str, Any]:
    """
    Longitude/latitude bounds, centre, pixel size and suggested zoom range of a raster.

    The maximum zoom is where map pixels match the raster's pixels; the minimum
    is six levels coarser (but not below 0).
    """
    rasterio = _rasterio()
    from rasterio.warp import transform_bounds

    with _open(source) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        res = abs(src.transform.a)
        if src.crs.is_geographic:
            res *= 111_320 * math.cos(math.radians((bounds[1] + bounds[3]) / 2))
        lat = (bounds[1] + bounds[3]) / 2
        maxzoom = min(zoom_for_resolution(res, latitude=lat), 24)
        return {"bounds": tuple(round(b, 7) for b in bounds), "center": ((bounds[0] + bounds[2]) / 2, lat),
                "resolution_m": res, "minzoom": max(maxzoom - 6, 0), "maxzoom": maxzoom,
                "count": src.count, "dtype": src.dtypes[0], "crs": src.crs.to_string()}


def tilejson(name: str, tiles_url: str, *, bounds: Sequence[float], minzoom: int, maxzoom: int,
             center: Optional[Sequence[float]] = None, attribution: Optional[str] = None,
             description: Optional[str] = None) -> Dict[str, Any]:
    """A TileJSON 3.0 document describing an XYZ tile layer."""
    doc: Dict[str, Any] = {"tilejson": "3.0.0", "name": name, "tiles": [tiles_url], "scheme": "xyz",
                           "minzoom": int(minzoom), "maxzoom": int(maxzoom), "bounds": [float(b) for b in bounds]}
    if center is not None:
        doc["center"] = [float(center[0]), float(center[1]), int(minzoom)]
    if attribution:
        doc["attribution"] = attribution
    if description:
        doc["description"] = description
    return doc


# ----------------------------------------------------------------------------- rendering
def suggest_rescale(source: Union[str, Path, Any], bands: Optional[Sequence[int]] = None, *,
                    percentiles: Tuple[float, float] = (2.0, 98.0), max_size: int = 1024) -> List[Tuple[float, float]]:
    """
    Per-band display range from percentiles of a decimated read of the raster.

    Use the result as ``rescale`` so every tile of a layer is stretched the
    same way (a per-tile stretch makes tile edges visible).
    """
    with _open(source) as src:
        idx = list(bands or ([1, 2, 3] if src.count >= 3 else [1]))
        scale = max(src.width, src.height) / max_size
        shape = (len(idx), max(1, int(src.height / max(scale, 1))), max(1, int(src.width / max(scale, 1))))
        data = src.read(idx, out_shape=shape, masked=True).astype("float64")
    ranges = []
    for band in data:
        values = band.compressed() if np.ma.isMaskedArray(band) else band.ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(values, percentiles)
        ranges.append((float(lo), float(hi if hi > lo else lo + 1)))
    return ranges


def render_tile(
    source: Union[str, Path, Any],
    x: int,
    y: int,
    z: int,
    *,
    bands: Optional[Sequence[int]] = None,
    rescale: Optional[Rescale] = None,
    colormap: Optional[str] = None,
    nodata: Optional[float] = None,
    tile_size: int = 256,
    resampling: str = "bilinear",
    image_format: str = "PNG",
) -> Optional[bytes]:
    """
    Render one XYZ tile of a raster as an image.

    The raster is warped to Web Mercator for the tile's footprint only, so a
    COG with overviews is read efficiently, also from cloud storage.

    Args:
        source: Raster path, ``/vsi`` path or URL, or an open rasterio dataset.
        x, y, z: Tile index.
        bands: Band numbers (1-based): one band (with ``colormap``) or three (RGB).
            Defaults to 1–3 for multi-band rasters and 1 otherwise.
        rescale: ``(min, max)`` for all bands or one pair per band; values are
            stretched linearly to 0–255. Defaults to 2–98 % percentiles of the
            tile, which differ between tiles; pass a fixed range for maps.
        colormap: Matplotlib colormap name for single-band rasters.
        nodata: No-data value when the file has none.
        tile_size: Tile width and height in pixels.
        resampling: ``nearest``, ``bilinear``, ``cubic``, ``average``, ...
        image_format: ``PNG`` (with transparency) or ``WEBP``/``JPEG``.

    Returns:
        Encoded image bytes, or ``None`` when the tile lies outside the raster.
    """
    rasterio = _rasterio()
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    west, south, east, north = tile_bounds(x, y, z, crs="EPSG:3857")
    with contextlib.ExitStack() as stack:
        src = stack.enter_context(_open(source))
        sb = transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21)
        if west >= sb[2] or east <= sb[0] or south >= sb[3] or north <= sb[1]:
            return None
        idx = list(bands or ([1, 2, 3] if src.count >= 3 else [1]))
        if len(idx) not in (1, 3):
            raise ValueError("Pass one band (with an optional colormap) or three bands (RGB).")
        level = _overview_level(src, (west, south, east, north), tile_size)
        if level is not None:                    # read a coarse tile from an overview, not full resolution
            overview = rasterio.open(src.name, overview_level=level)
            src = stack.enter_context(overview)
        vrt_options: Dict[str, Any] = {
            "crs": "EPSG:3857", "transform": from_bounds(west, south, east, north, tile_size, tile_size),
            "width": tile_size, "height": tile_size, "resampling": Resampling[resampling],
        }
        if nodata is not None:
            vrt_options["src_nodata"] = nodata
            vrt_options["nodata"] = nodata
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read(idx, masked=True).astype("float64")
    valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.filled(np.nan)).all(axis=0)
    if not valid.any():
        return None
    values = data.filled(np.nan)
    pairs = _rescale_pairs(rescale, values, valid)
    scaled = np.stack([np.clip((b - lo) / (hi - lo), 0, 1) for b, (lo, hi) in zip(values, pairs)])
    scaled = np.nan_to_num(scaled)
    if len(idx) == 1:
        if colormap:
            rgba = _colormap(colormap)(scaled[0])
            rgb = (rgba[..., :3] * 255).astype("uint8")
        else:
            rgb = np.repeat((scaled[0] * 255).astype("uint8")[..., None], 3, axis=-1)
    else:
        rgb = (np.moveaxis(scaled, 0, -1) * 255).astype("uint8")
    alpha = (valid * 255).astype("uint8")[..., None]
    return _encode(np.concatenate([rgb, alpha], axis=-1), image_format)


def make_xyz_tiles(
    source: Union[str, Path],
    out_dir: Union[str, Path],
    zooms: Union[int, Iterable[int], None] = None,
    *,
    bands: Optional[Sequence[int]] = None,
    rescale: Optional[Rescale] = None,
    colormap: Optional[str] = None,
    nodata: Optional[float] = None,
    resampling: str = "bilinear",
    image_format: str = "PNG",
    viewer: bool = True,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write a static ``{z}/{x}/{y}.png`` tile pyramid of a raster.

    The folder can be copied to any static host (GitHub Pages, S3, a web
    server) and opened in Leaflet or MapLibre. A fixed display range is computed
    once (:func:`suggest_rescale`) unless ``rescale`` is given, so tiles match.

    Args:
        source: Raster path or URL.
        out_dir: Output folder.
        zooms: Zoom level(s); defaults to the range from :func:`raster_web_info`,
            capped at six levels.
        viewer: Also write ``index.html`` (a Leaflet map) and ``tilejson.json``.
        name: Layer name; defaults to the file stem.
        Other arguments are passed to :func:`render_tile`.

    Returns:
        ``{"tiles": n_written, "empty": n_skipped, "zooms": [...], "out_dir": ..., "tilejson": {...}}``
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = raster_web_info(source)
    zoom_list = (list(range(info["minzoom"], info["maxzoom"] + 1)) if zooms is None
                 else [zooms] if isinstance(zooms, int) else list(zooms))
    if rescale is None:
        rescale = suggest_rescale(source, bands)
    ext = image_format.lower().replace("jpeg", "jpg")
    written = empty = 0
    with _open(source) as src:
        for tile in tiles_for_bounds(*info["bounds"], zooms=zoom_list):
            img = render_tile(src, *tile, bands=bands, rescale=rescale, colormap=colormap, nodata=nodata,
                              resampling=resampling, image_format=image_format)
            if img is None:
                empty += 1
                continue
            path = out / str(tile.z) / str(tile.x) / f"{tile.y}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img)
            written += 1
    layer = name or Path(os.fspath(source)).stem
    tj = tilejson(layer, f"{{z}}/{{x}}/{{y}}.{ext}", bounds=info["bounds"], minzoom=min(zoom_list),
                  maxzoom=max(zoom_list), center=info["center"])
    if viewer:
        from geolattice.web.viewer import leaflet_html

        (out / "tilejson.json").write_text(json.dumps(tj, indent=2))
        (out / "index.html").write_text(leaflet_html(
            title=layer, rasters=[{"name": layer, "url": f"{{z}}/{{x}}/{{y}}.{ext}", "bounds": info["bounds"],
                                   "minzoom": min(zoom_list), "maxzoom": max(zoom_list)}]))
    return {"tiles": written, "empty": empty, "zooms": zoom_list, "out_dir": str(out), "tilejson": tj}


# ----------------------------------------------------------------------------- helpers
class _open:
    """Open a path/URL with rasterio, or pass an open dataset through unchanged."""

    def __init__(self, source: Any):
        self.source = source
        self.dataset = None

    def __enter__(self) -> Any:
        if hasattr(self.source, "read") and hasattr(self.source, "crs"):
            return self.source
        rasterio = _rasterio()
        self.dataset = rasterio.open(to_vsi_path(os.fspath(self.source)))
        return self.dataset

    def __exit__(self, *exc: Any) -> None:
        if self.dataset is not None:
            self.dataset.close()


def _overview_level(src: Any, bounds_3857: Tuple[float, float, float, float], tile_size: int) -> Optional[int]:
    """
    Index of the coarsest overview that is still at least as fine as the tile, or ``None``.

    ``WarpedVRT`` warps from full resolution, so a zoomed-out tile of a large COG would
    otherwise read every pixel underneath it (seconds per tile over the network).
    """
    from rasterio.warp import transform_bounds

    try:
        factors = src.overviews(1)
    except Exception:  # noqa: BLE001 - datasets without overview support
        return None
    if not factors or not getattr(src, "name", None):
        return None
    left, bottom, right, top = transform_bounds("EPSG:3857", src.crs, *bounds_3857, densify_pts=5)
    target = min((right - left) / tile_size, (top - bottom) / tile_size)
    native = min(abs(src.res[0]), abs(src.res[1]))
    usable = [i for i, f in enumerate(factors) if native * f <= target]
    return usable[-1] if usable else None


def _rescale_pairs(rescale: Optional[Rescale], values: np.ndarray, valid: np.ndarray) -> List[Tuple[float, float]]:
    n = values.shape[0]
    if rescale is None:
        pairs = []
        for band in values:
            v = band[valid]
            lo, hi = np.percentile(v, (2, 98)) if v.size else (0.0, 1.0)
            pairs.append((float(lo), float(hi if hi > lo else lo + 1)))
        return pairs
    if len(rescale) == 2 and np.isscalar(rescale[0]):
        return [(float(rescale[0]), float(rescale[1]))] * n  # type: ignore[index]
    pairs = [(float(a), float(b)) for a, b in rescale]  # type: ignore[misc]
    if len(pairs) == 1:                     # one pair, given as a list, applies to every band
        return pairs * n
    if len(pairs) != n:
        raise ValueError(f"rescale has {len(pairs)} pairs for {n} bands")
    return pairs


def _colormap(name: str) -> Any:
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise ImportError("colormap= needs matplotlib: pip install matplotlib") from exc
    return colormaps[name]


def _encode(rgba: np.ndarray, image_format: str) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Rendering tiles needs Pillow: pip install pillow") from exc
    fmt = image_format.upper()
    image = Image.fromarray(rgba, mode="RGBA")
    if fmt in ("JPEG", "JPG"):
        image, fmt = image.convert("RGB"), "JPEG"
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Tile rendering needs rasterio: pip install rasterio") from exc
    return rasterio


__all__ = [
    "Tile", "lonlat_to_tile", "tile_bounds", "tiles_for_bounds", "tile_resolution", "zoom_for_resolution",
    "raster_web_info", "tilejson", "suggest_rescale", "render_tile", "make_xyz_tiles",
]
