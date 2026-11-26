"""Reading UAV frames (PNG/JPG/TIFF), RGB + NIR pairs, video frames, quality checks and
catalogues of multi-date flight folders.

UAV frames and orthomosaics exported as PNG/JPG carry no georeferencing. :func:`read_frame`
puts them on a regular grid anyway: in metres when the ground sampling distance (``gsd``)
is known, otherwise in pixels. Every ``rs.uav`` function then works on them the same way
as on GeoTIFFs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_GEO_EXTS = {".tif", ".tiff"}


def grid_coords(
    rows: int, cols: int, res: float = 1.0, origin: tuple[float, float] = (0.0, 0.0)
) -> dict[str, np.ndarray]:
    """North-up pixel-centre coordinates for a ``rows x cols`` image with pixel size ``res``."""
    x0, y0 = origin
    return {"y": y0 - res * (np.arange(rows) + 0.5), "x": x0 + res * (np.arange(cols) + 0.5)}


def as_image(
    arr: np.ndarray,
    *,
    bands: Sequence[str] | None = None,
    gsd: float | None = None,
    name: str | None = None,
    attrs: dict | None = None,
) -> xr.DataArray:
    """Wrap a ``(rows, cols)`` or ``(rows, cols, channels)`` array as a ``(band, y, x)`` DataArray.

    With ``gsd`` (metres per pixel) the coordinates are metres, otherwise pixels. The pixel
    size is kept in ``attrs["gsd_m"]`` (or ``None``) and ``attrs["units_xy"]``.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3:
        a = np.moveaxis(a, -1, 0)
    else:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {a.shape}")
    n = a.shape[0]
    if bands is None:
        bands = {
            1: ["gray"],
            3: ["red", "green", "blue"],
            4: ["red", "green", "blue", "alpha"],
        }.get(n, [f"b{i + 1}" for i in range(n)])
    if len(bands) != n:
        raise ValueError(f"{len(bands)} band names for {n} channels")
    coords = grid_coords(a.shape[1], a.shape[2], gsd or 1.0)
    da = xr.DataArray(a, dims=("band", "y", "x"), coords={"band": list(bands), **coords}, name=name)
    da.attrs.update(attrs or {})
    da.attrs.update(gsd_m=gsd, units_xy="m" if gsd else "px")
    return da


def _to_float(a: np.ndarray, scale: bool) -> np.ndarray:
    if not scale:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return (a.astype(np.float32) / np.iinfo(a.dtype).max).astype(np.float32)
    return a.astype(np.float32)


def read_frame(
    path: str | PathLike,
    *,
    gsd: float | None = None,
    bands: Sequence[str] | None = None,
    scale: bool = True,
    drop_alpha: bool = True,
    max_dim: int | None = None,
) -> xr.DataArray:
    """Read a UAV frame or exported orthomosaic as a ``(band, y, x)`` DataArray.

    PNG/JPG/BMP are read with Pillow (EXIF orientation applied); GeoTIFFs keep their CRS
    and transform via :func:`geolattice.rsense.read`. ``scale`` turns integer DNs into 0-1
    floats. ``max_dim`` downsizes large images on read (``gsd`` is adjusted to match).
    """
    p = Path(path)
    if p.suffix.lower() in _GEO_EXTS:
        from ..core.raster import read

        da = read(p, band_names=bands)
        if da.rio.crs is not None:
            if scale:
                da = da.copy(data=_to_float(np.asarray(da.values), True))
            da.attrs.update(gsd_m=abs(float(da.rio.resolution()[0])), units_xy="crs")
            return da
        arr = np.moveaxis(np.asarray(da.values), 0, -1)
    else:
        from PIL import Image, ImageOps

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            if max_dim and max(im.size) > max_dim:
                f = max_dim / max(im.size)
                im = im.resize((round(im.width * f), round(im.height * f)), Image.BILINEAR)
                gsd = gsd / f if gsd else None
            if im.mode in ("P", "CMYK", "YCbCr", "LA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            arr = np.asarray(im)
    if drop_alpha and arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return as_image(
        _to_float(arr, scale), bands=bands, gsd=gsd, name=p.stem, attrs={"source": str(p)}
    )


def to_rgb_array(img: Any, *, uint8: bool = False) -> np.ndarray:
    """``(band, y, x)`` DataArray (or array) -> ``(rows, cols, 3)`` RGB array for OpenCV/torch."""
    if isinstance(img, xr.DataArray):
        if "band" in img.dims:
            names = [str(b) for b in img.band.values]
            sel = [n for n in ("red", "green", "blue") if n in names]
            img = img.sel(band=sel) if len(sel) == 3 else img.isel(band=slice(0, 3))
            a = np.moveaxis(np.asarray(img.values), 0, -1)
        else:
            a = np.asarray(img.values)
    else:
        a = np.asarray(img)
    if uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(np.nan_to_num(a) * (255 if np.nanmax(a) <= 1.0 + 1e-6 else 1), 0, 255)
        a = a.astype(np.uint8)
    return a


def load_pair(
    rgb_path: str | PathLike,
    nir_path: str | PathLike,
    *,
    gsd: float | None = None,
    register: bool | str = False,
    nir_channel: int | str = "mean",
    **register_kw: Any,
) -> xr.DataArray:
    """Load an RGB frame and its NIR frame into one ``(band=[red, green, blue, nir], y, x)`` array.

    NIR images saved as 3-channel PNGs are reduced with ``nir_channel`` (an index or
    ``"mean"``). The NIR band is resized to the RGB grid, and with ``register`` (``True``,
    ``"orb"``, ``"sift"`` or ``"ecc"``) aligned to the green band first, since the two
    cameras of a dual-sensor rig never see exactly the same footprint.
    """
    rgb = read_frame(rgb_path, gsd=gsd)
    nir_raw = read_frame(nir_path)
    nv = np.asarray(nir_raw.values)
    nir = nv.mean(0) if nir_channel == "mean" else nv[int(nir_channel)]
    H, W = rgb.sizes["y"], rgb.sizes["x"]
    if nir.shape != (H, W):
        from skimage.transform import resize

        nir = resize(nir, (H, W), order=1, preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
    info = {}
    if register:
        from .register import coregister

        method = register if isinstance(register, str) else "orb"
        nir, info = coregister(
            nir, np.asarray(rgb.sel(band="green").values), method=method, **register_kw
        )
    out = xr.concat(
        [
            rgb,
            rgb.isel(band=[0]).copy(data=nir[None].astype(np.float32)).assign_coords(band=["nir"]),
        ],
        dim="band",
    )
    out.attrs.update(
        rgb.attrs,
        nir_source=str(nir_path),
        **{f"register_{k}": v for k, v in info.items() if np.isscalar(v)},
    )
    out.name = rgb.name
    return out


# --------------------------------------------------------------------- quality control
def _gray(img: Any) -> np.ndarray:
    a = (
        to_rgb_array(img)
        if (isinstance(img, xr.DataArray) or np.ndim(img) == 3)
        else np.asarray(img)
    )
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    if a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return a


def blur_score(img: Any) -> float:
    """Variance of the Laplacian (higher = sharper). Motion-blurred frames score low."""
    from scipy.ndimage import laplace

    return float(laplace(_gray(img)).var() * 1e4)


def exposure_stats(img: Any, *, dark: float = 0.03, bright: float = 0.97) -> dict[str, float]:
    """Mean brightness and the fractions of near-black and near-saturated pixels (0-1 scale)."""
    g = _gray(img)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "dark_frac": float((g <= dark).mean()),
        "saturated_frac": float((g >= bright).mean()),
    }


def frame_quality(paths: Sequence[str | PathLike], *, max_dim: int = 1024) -> pd.DataFrame:
    """Blur and exposure statistics for many frames (read at ``max_dim`` for speed)."""
    rows = []
    for p in paths:
        img = read_frame(p, max_dim=max_dim)
        rows.append({"path": str(p), "blur": blur_score(img), **exposure_stats(img)})
    return pd.DataFrame(rows)


def extract_frames(
    video: str | PathLike,
    out_dir: str | PathLike,
    *,
    step: int = 6,
    start_s: float = 0.0,
    end_s: float | None = None,
    min_blur: float | None = None,
    ext: str = "png",
) -> pd.DataFrame:
    """Save every ``step``-th frame of a flight video (the LUMS workflow keeps every 6th frame of
    24 fps 4K video). Frames with :func:`blur_score` below ``min_blur`` are skipped.

    Returns a table of ``frame``, ``time_s``, ``blur`` and ``path``. Needs OpenCV.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("extract_frames needs `pip install opencv-python`") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first = int(round(start_s * fps))
    last = int(round(end_s * fps)) if end_s is not None else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, i = [], first
    while True:
        ok, frame = cap.read()
        if not ok or (last is not None and i > last):
            break
        if (i - first) % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            b = blur_score(rgb)
            if min_blur is None or b >= min_blur:
                path = out / f"frame{i}.{ext}"
                cv2.imwrite(str(path), frame)
                rows.append({"frame": i, "time_s": i / fps, "blur": b, "path": str(path)})
        i += 1
    cap.release()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- catalogues
_BAND_DIR = re.compile(r"^(rgb|nir|red|green|blue|rededge|re|thermal|ms)(?:[-_ ]?\d+)?$", re.I)
_ISO = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
_COMPACT = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def parse_date(text: str) -> date | None:
    """Find a date in a folder name: ``2025-08-22``, ``30072018`` (DDMMYYYY) or ``20250822``."""
    m = _ISO.search(text)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = _COMPACT.search(text)
    if m:
        s = m[1]
        for fmt in ("%d%m%Y", "%Y%m%d"):
            try:
                d = datetime.strptime(s, fmt).date()
                if 1990 <= d.year <= 2100:
                    return d
            except ValueError:
                continue
    return None


def _crop_name(session: str) -> str:
    stem = _COMPACT.sub("", _ISO.sub("", session))
    return re.sub(r"[-_ ]+$", "", stem).strip().lower() or ""


def build_catalog(root: str | PathLike, *, exts: Sequence[str] | None = None) -> pd.DataFrame:
    """Index every image under ``root`` into a tidy table, one row per file.

    Works for layouts such as ``<dataset>/<crop_date>/<RGB|NIR>/<frame>.png`` (the Canola,
    Drybean, Wheat and Lentil-Wheat sets) and ``<dataset>/<YYYY-MM-DD>/<frame>.png`` (LUMS).
    Columns: ``dataset, crop, session, date, band, rig, frame, path``. Band folders like
    ``RGB-1`` map to band ``RGB`` and rig ``1`` (sessions flown with two camera rigs, as in
    the Drybean set, keep their frames apart); files outside a band folder are labelled
    ``RGB`` with an empty rig.
    """
    root = Path(root)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or IMAGE_EXTS)}
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root).parts
        parents = list(rel[:-1])
        band, rig = "RGB", ""
        if parents and _BAND_DIR.match(parents[-1]):
            folder = parents.pop()
            band = re.sub(r"[-_ ]?\d+$", "", folder).upper()
            m = re.search(r"(\d+)$", folder)
            rig = m.group(1) if m else ""
        session = parents[-1] if parents else ""
        dataset = parents[0] if parents else root.name
        d = parse_date(session)
        crop = _crop_name(session) if d else ""
        if not crop:
            crop = re.sub(r"[-_ ]?dataset$", "", dataset, flags=re.I).strip().lower()
        rows.append(
            {
                "dataset": dataset,
                "crop": crop,
                "session": session,
                "date": pd.Timestamp(d) if d else pd.NaT,
                "band": band,
                "rig": rig,
                "frame": p.stem,
                "path": str(p),
            }
        )
    cols = ["dataset", "crop", "session", "date", "band", "rig", "frame", "path"]
    return pd.DataFrame(rows, columns=cols)


def pair_catalog(catalog: pd.DataFrame, bands: Sequence[str] = ("RGB", "NIR")) -> pd.DataFrame:
    """Turn a :func:`build_catalog` table into one row per frame with a path column per band.

    Only frames present in every requested band are kept (e.g. matched RGB/NIR pairs).
    Frames are matched within a camera rig (``RGB-1`` with ``NIR-1``) when the catalogue
    has a ``rig`` column.
    """
    keys = ["dataset", "crop", "session", "date"] + (["rig"] if "rig" in catalog else []) + ["frame"]
    sub = catalog[catalog["band"].isin(bands)]
    wide = sub.pivot_table(
        index=keys, columns="band", values="path", aggfunc="first", dropna=False
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={b: f"{b.lower()}_path" for b in bands})
    need = [f"{b.lower()}_path" for b in bands]
    for c in need:
        if c not in wide:
            wide[c] = np.nan
    return wide.dropna(subset=need).sort_values(keys).reset_index(drop=True)
