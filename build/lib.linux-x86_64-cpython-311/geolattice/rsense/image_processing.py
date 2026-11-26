"""Standalone image processing for remote-sensing rasters.

General-purpose image processing and low-level computer-vision operations for satellite,
aerial and UAV imagery, following the OpenCV and scikit-image workflows (read/write,
geometric transforms, radiometric operations, filtering, edge detection, morphology,
thresholding, features). Everything is implemented on top of NumPy, SciPy and scikit-image,
so it needs only ``pip install geolattice[rsense]``; OpenCV is not required.

Inputs are either a ``xarray.DataArray`` with dims ``(band, y, x)`` (or ``(y, x)``) or a
NumPy array of shape ``(band, y, x)`` / ``(y, x)``. Output has the same type. For a
DataArray the CRS, attributes and band coordinate are kept, and for operations that change
the grid (resize, rescale, crop, flip, rotate, shift...) the ``x``/``y`` coordinates are
rebuilt so the result stays georeferenced. NaN pixels are ignored by per-pixel filters and
stay NaN in same-shape outputs.

Sections
--------
1. **I/O and info** -- :func:`read_image`, :func:`write_image`, :func:`image_info`
2. **Color spaces** -- :func:`to_gray`, :func:`rgb_to_hsv`, :func:`hsv_to_rgb`,
   :func:`rgb_to_lab`, :func:`to_uint8`
3. **Geometric transforms** -- :func:`resize`, :func:`rescale`, :func:`downscale_local_mean`,
   :func:`rotate`, :func:`shift`, :func:`flip`, :func:`crop`, :func:`affine_transform`,
   :func:`perspective_transform`
4. **Radiometric operations** -- :func:`stretch`, :func:`adjust_gamma`,
   :func:`adjust_brightness_contrast`, :func:`histogram`, :func:`equalize_hist`,
   :func:`equalize_adapthist`, :func:`match_histograms`
5. **Thresholding** -- :func:`threshold`
6. **Filtering** -- :func:`convolve`, :func:`mean_blur`, :func:`gaussian_blur`,
   :func:`median_blur`, :func:`bilateral_denoise`, :func:`tv_denoise`, :func:`sharpen`,
   :func:`unsharp_mask`, :func:`add_noise`, :data:`KERNELS`
7. **Edge detection** -- :func:`edges` (Sobel, Roberts, Prewitt, Scharr, Laplacian, Canny),
   :func:`sobel_h`, :func:`sobel_v`
8. **Morphology** -- :func:`morphology`, :func:`remove_small_objects`
9. **Features** -- :func:`harris_corners`, :func:`hog_features`, :func:`to_vector`
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

try:  # xarray is a core rsense dependency, but keep numpy-only use possible
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

__all__ = [
    "KERNELS",
    "add_noise",
    "adjust_brightness_contrast",
    "adjust_gamma",
    "affine_transform",
    "bilateral_denoise",
    "convolve",
    "crop",
    "downscale_local_mean",
    "edges",
    "equalize_adapthist",
    "equalize_hist",
    "flip",
    "gaussian_blur",
    "harris_corners",
    "histogram",
    "hog_features",
    "hsv_to_rgb",
    "image_info",
    "match_histograms",
    "mean_blur",
    "median_blur",
    "morphology",
    "perspective_transform",
    "read_image",
    "remove_small_objects",
    "rescale",
    "resize",
    "rgb_to_hsv",
    "rgb_to_lab",
    "rotate",
    "sharpen",
    "shift",
    "sobel_h",
    "sobel_v",
    "stretch",
    "threshold",
    "to_gray",
    "to_uint8",
    "to_vector",
    "tv_denoise",
    "unsharp_mask",
    "write_image",
]

Image = "xr.DataArray | np.ndarray"

# Classic 3x3 convolution kernels (OpenCV ``filter2D`` examples).
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "box": np.full((3, 3), 1 / 9.0),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "sharpen_strong": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}


# --------------------------------------------------------------------------- helpers


def _is_da(x) -> bool:
    return xr is not None and isinstance(x, xr.DataArray)


def _split(img):
    """Return ``(float32 array (B, H, W), meta)``; ``meta`` rebuilds the input type."""
    if _is_da(img):
        if img.ndim == 2:
            arr, squeeze = img.values[None], True
        elif img.ndim == 3:
            arr, squeeze = img.values, False
        else:
            raise ValueError(f"expected (y, x) or (band, y, x); got dims {img.dims}")
        return arr, {"da": img, "squeeze": squeeze}
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr[None], {"da": None, "squeeze": True}
    if arr.ndim == 3:
        return arr, {"da": None, "squeeze": False}
    raise ValueError(f"expected (y, x) or (band, y, x); got shape {arr.shape}")


def _grid_coords(da, new_h, new_w, y0=0.0, x0=0.0, sy=1.0, sx=1.0):
    """New y/x coordinates for a grid resampled from ``da``.

    ``sy``/``sx`` are the new pixel size in units of the old pixel size and ``y0``/``x0`` the
    offset of the new grid's first pixel in old pixels.
    """
    ydim, xdim = da.dims[-2], da.dims[-1]
    out = {}
    for dim, n, off, s in ((ydim, new_h, y0, sy), (xdim, new_w, x0, sx)):
        c = np.asarray(da[dim].values, dtype=float) if dim in da.coords else None
        if c is None or c.size < 2:
            continue
        d = c[1] - c[0]
        start = c[0] - d / 2 + off * d
        out[dim] = start + d * s * (np.arange(n) + 0.5)
    return out


def _merge(arr, meta, coords=None, name=None):
    """Wrap ``arr`` (B, H, W) back into the input's type."""
    if meta["squeeze"]:
        arr = arr[0]
    da = meta["da"]
    if da is None:
        return arr
    dims = da.dims[-arr.ndim:]
    keep = {}
    if arr.ndim == 3 and da.ndim == 3 and da.dims[0] in da.coords and arr.shape[0] == da.shape[0]:
        keep[da.dims[0]] = da[da.dims[0]].values
    if coords is None:
        if arr.shape[-2:] == da.shape[-2:]:
            coords = {d: da[d].values for d in da.dims[-2:] if d in da.coords}
        else:
            coords = {}
    keep.update(coords)
    out = xr.DataArray(arr, dims=dims, coords=keep, attrs=dict(da.attrs), name=name or da.name)
    crs = getattr(da, "rio", None)
    try:
        if crs is not None and crs.crs is not None:
            out = out.rio.write_crs(crs.crs)
    except Exception:  # pragma: no cover - rioxarray absent or no CRS
        pass
    return out


def _per_band(img, func: Callable[[np.ndarray], np.ndarray], dtype=np.float32,
              keep_nan=True):
    """Apply a 2-D ``func`` to every band, filling NaNs with the band median first."""
    arr, meta = _split(img)
    out = []
    for band in arr:
        b = band.astype(np.float64)
        nan = ~np.isfinite(b)
        if nan.any():
            fill = np.nanmedian(b) if (~nan).any() else 0.0
            b = np.where(nan, fill, b)
        r = np.asarray(func(b))
        if keep_nan and nan.any() and r.shape == b.shape:
            r = r.astype(np.float64)
            r[nan] = np.nan
        out.append(r)
    res = np.stack(out)
    if dtype is not None and not (np.issubdtype(np.dtype(dtype), np.integer) and np.isnan(res).any()):
        res = res.astype(dtype)
    return _merge(res, meta)


def _unit(arr):
    """Rescale to [0, 1] by dtype range (integer) or by data range (float outside [0, 1])."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float64) - info.min) / (info.max - info.min)
    a = arr.astype(np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if lo >= 0 and hi <= 1:
        return a
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _rgb(img):
    arr, meta = _split(img)
    if arr.shape[0] < 3:
        raise ValueError("this operation needs at least 3 bands (R, G, B)")
    return arr[:3], meta


# --------------------------------------------------------------------------- 1. I/O


def read_image(path, bands: Sequence[int] | None = None, masked: bool = True,
               as_dataarray: bool = True, window: tuple[int, int, int, int] | None = None):
    """Read an image or GeoTIFF as a ``(band, y, x)`` array.

    GeoTIFFs keep their CRS and transform (``rioxarray``). Plain images (PNG/JPEG) are read
    through GDAL via rasterio and have no georeferencing. ``bands`` is a 1-based list.
    ``window=(row0, row1, col0, col1)`` reads only that pixel window (useful for rasters too
    large to load). With ``as_dataarray=False`` a NumPy array is returned.
    """
    import rioxarray  # noqa: F401

    da = rioxarray.open_rasterio(str(path), masked=masked)
    if bands is not None:
        da = da.sel(band=list(bands))
    if window is not None:
        r0, r1, c0, c1 = window
        da = da.isel(y=slice(r0, r1), x=slice(c0, c1))
    da = da.load()
    da.name = da.name or "image"
    return da if as_dataarray else da.values


def write_image(img, path, dtype: str | None = None, crs=None, transform=None,
                nodata=None) -> str:
    """Write an image to GeoTIFF (or any GDAL-writable format chosen by the extension).

    For a DataArray the CRS and transform are taken from the array; for NumPy input pass
    ``crs`` / ``transform`` (a rasterio ``Affine``) or get an un-referenced file.
    """
    import rasterio
    from rasterio.transform import Affine

    arr, meta = _split(img)
    da = meta["da"]
    if da is not None:
        crs = crs or (da.rio.crs if da.rio.crs is not None else None)
        transform = transform or da.rio.transform()
    arr = arr if dtype is None else arr.astype(dtype)
    path = str(path)
    driver = "PNG" if path.lower().endswith(".png") else "GTiff"
    if driver == "PNG":
        arr = to_uint8(arr)
    with rasterio.open(path, "w", driver=driver, height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, crs=crs,
                       transform=transform or Affine.identity(), nodata=nodata) as dst:
        dst.write(arr)
    return path


def image_info(img) -> dict:
    """Shape, dtype, per-band min/max/mean and NaN count (OpenCV ``img.shape``/``dtype``)."""
    arr, meta = _split(img)
    f = arr.astype(np.float64)
    return {
        "bands": arr.shape[0],
        "height": arr.shape[1],
        "width": arr.shape[2],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": [float(np.nanmin(b)) for b in f],
        "max": [float(np.nanmax(b)) for b in f],
        "mean": [float(np.nanmean(b)) for b in f],
        "nan_count": int(np.isnan(f).sum()),
        "georeferenced": meta["da"] is not None and getattr(meta["da"].rio, "crs", None) is not None,
    }


# --------------------------------------------------------------------------- 2. color


def to_uint8(img, percentiles: tuple[float, float] | None = None):
    """Scale to 0-255 ``uint8``: dtype range for ints, min-max (or percentiles) otherwise."""
    arr, meta = _split(img)
    a = arr.astype(np.float64)
    if percentiles is not None:
        lo, hi = np.nanpercentile(a, percentiles)
        u = np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(a)
    elif arr.dtype == np.uint8:
        return img
    else:
        u = _unit(arr)
    return _merge(np.nan_to_num(np.round(u * 255)).astype(np.uint8), meta)


def to_gray(img, weights: Sequence[float] = (0.2125, 0.7154, 0.0721)):
    """RGB to gray (``skimage.color.rgb2gray`` / ``cv2.COLOR_BGR2GRAY``); first 3 bands are R, G, B."""
    arr, meta = _rgb(img)
    w = np.asarray(weights, dtype=np.float64)
    g = np.tensordot(w, arr.astype(np.float64), axes=1)[None].astype(np.float32)
    meta = dict(meta, squeeze=True)
    return _merge(g, meta)


def rgb_to_hsv(img):
    """RGB to HSV; bands are returned as ``(H, S, V)`` all in [0, 1]."""
    from skimage.color import rgb2hsv

    arr, meta = _rgb(img)
    u = np.stack([_unit(arr[i]) for i in range(3)], axis=-1)
    out = np.moveaxis(rgb2hsv(u), -1, 0).astype(np.float32)
    return _merge(out, dict(meta, squeeze=False))


def hsv_to_rgb(img):
    """HSV (all in [0, 1], bands H, S, V) back to RGB in [0, 1]."""
    from skimage.color import hsv2rgb

    arr, meta = _rgb(img)
    out = np.moveaxis(hsv2rgb(np.moveaxis(arr.astype(np.float64), 0, -1)), -1, 0)
    return _merge(out.astype(np.float32), dict(meta, squeeze=False))


def rgb_to_lab(img):
    """RGB to CIE L*a*b* (D65); bands are ``(L, a, b)``."""
    from skimage.color import rgb2lab

    arr, meta = _rgb(img)
    u = np.stack([_unit(arr[i]) for i in range(3)], axis=-1)
    out = np.moveaxis(rgb2lab(u), -1, 0).astype(np.float32)
    return _merge(out, dict(meta, squeeze=False))


# --------------------------------------------------------------------------- 3. geometry


def resize(img, shape: tuple[int, int] | None = None, scale: float | tuple[float, float] | None = None,
           order: int = 1, anti_aliasing: bool | None = None, keep_aspect: bool = False):
    """Resize to ``shape=(height, width)`` or by ``scale`` (``skimage.transform.resize``).

    ``order``: 0 nearest, 1 bilinear, 3 bicubic. With ``keep_aspect`` and a ``shape`` where
    one side is ``None`` the other side is derived from the aspect ratio. Value range and
    dtype are preserved (floats stay floats, integers are rounded back).
    """
    from skimage.transform import resize as _resize

    arr, meta = _split(img)
    _, h, w = arr.shape
    if scale is not None:
        sy, sx = (scale, scale) if np.isscalar(scale) else scale
        new = (max(1, int(round(h * sy))), max(1, int(round(w * sx))))
    else:
        if shape is None:
            raise ValueError("give shape or scale")
        nh, nw = shape
        if nh is None and nw is None:
            raise ValueError("shape needs at least one side")
        if nh is None:
            nh = int(round(h * nw / w))
        if nw is None:
            nw = int(round(w * nh / h))
        if keep_aspect and None not in shape:
            raise ValueError("keep_aspect needs one side of shape to be None")
        new = (int(nh), int(nw))
    if anti_aliasing is None:
        anti_aliasing = new[0] < h or new[1] < w
    out = np.stack([
        _resize(b.astype(np.float64), new, order=order, mode="edge", preserve_range=True,
                anti_aliasing=anti_aliasing) for b in arr])
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.clip(np.round(out), info.min, info.max)
    out = out.astype(arr.dtype if arr.dtype.kind in "iuf" else np.float32)
    coords = None
    if meta["da"] is not None:
        coords = _grid_coords(meta["da"], new[0], new[1], sy=h / new[0], sx=w / new[1])
    return _merge(out, meta, coords=coords)


def rescale(img, factor: float | tuple[float, float], order: int = 1,
            anti_aliasing: bool | None = None):
    """Scale both sides by ``factor`` (``skimage.transform.rescale`` / ``cv2.resize(fx, fy)``)."""
    return resize(img, scale=factor, order=order, anti_aliasing=anti_aliasing)


def downscale_local_mean(img, factors: tuple[int, int] = (2, 2)):
    """Block-mean downscale by integer ``factors=(fy, fx)`` (edges are padded with 0, as skimage)."""
    from skimage.transform import downscale_local_mean as _dlm

    arr, meta = _split(img)
    out = np.stack([_dlm(b.astype(np.float64), tuple(factors)) for b in arr]).astype(np.float32)
    coords = None
    if meta["da"] is not None:
        coords = _grid_coords(meta["da"], out.shape[1], out.shape[2],
                              sy=factors[0], sx=factors[1])
    return _merge(out, meta, coords=coords)


def rotate(img, angle: float, resize_output: bool = False, order: int = 1,
           cval: float = np.nan, center: tuple[float, float] | None = None):
    """Rotate counter-clockwise by ``angle`` degrees (``skimage.transform.rotate``).

    ``cval`` fills the corners (NaN by default so they are visible as no-data); NaN requires
    a float image, so integer inputs are returned as float32. ``center`` is ``(col, row)``.
    """
    from skimage.transform import rotate as _rot

    arr, meta = _split(img)
    out = np.stack([_rot(b.astype(np.float64), angle, resize=resize_output, order=order,
                         cval=cval, center=center, preserve_range=True) for b in arr])
    out = out.astype(np.float32)
    coords = None
    if meta["da"] is not None and out.shape[1:] != arr.shape[1:]:
        coords = _grid_coords(meta["da"], out.shape[1], out.shape[2])
    return _merge(out, meta, coords=coords)


def shift(img, dx: float = 0, dy: float = 0, mode: str = "constant", cval: float = np.nan,
          order: int = 1):
    """Translate by ``dx`` columns right and ``dy`` rows down (``cv2.warpAffine`` translation).

    ``mode``: ``constant`` (fill with ``cval``), ``edge``, ``wrap`` or ``reflect``.
    """
    from scipy.ndimage import shift as _shift

    arr, meta = _split(img)
    m = {"edge": "nearest", "constant": "constant", "wrap": "wrap", "reflect": "reflect"}[mode]
    out = np.stack([_shift(b.astype(np.float64), (dy, dx), order=order, mode=m, cval=cval)
                    for b in arr]).astype(np.float32)
    return _merge(out, meta)


def flip(img, axis: str = "horizontal"):
    """Flip ``horizontal`` (left-right), ``vertical`` (up-down) or ``both``."""
    arr, meta = _split(img)
    if axis not in ("horizontal", "vertical", "both"):
        raise ValueError("axis must be 'horizontal', 'vertical' or 'both'")
    out = arr
    if axis in ("horizontal", "both"):
        out = out[:, :, ::-1]
    if axis in ("vertical", "both"):
        out = out[:, ::-1, :]
    out = np.ascontiguousarray(out)
    coords = None
    da = meta["da"]
    if da is not None:
        coords = {d: da[d].values for d in da.dims[-2:] if d in da.coords}
        ydim, xdim = da.dims[-2], da.dims[-1]
        if axis in ("horizontal", "both") and xdim in coords:
            coords[xdim] = coords[xdim][::-1]
        if axis in ("vertical", "both") and ydim in coords:
            coords[ydim] = coords[ydim][::-1]
    return _merge(out, meta, coords=coords)


def crop(img, window: tuple[int, int, int, int] | None = None, bounds=None):
    """Crop to ``window=(row0, row1, col0, col1)`` pixels, or to map ``bounds=(xmin, ymin, xmax, ymax)``.

    Equivalent to NumPy slicing ``img[row0:row1, col0:col1]``; coordinates are kept.
    """
    arr, meta = _split(img)
    da = meta["da"]
    if bounds is not None:
        if da is None:
            raise ValueError("bounds needs a georeferenced DataArray")
        xmin, ymin, xmax, ymax = bounds
        sub = da.rio.clip_box(xmin, ymin, xmax, ymax)
        return sub
    if window is None:
        raise ValueError("give window or bounds")
    r0, r1, c0, c1 = window
    out = arr[:, r0:r1, c0:c1]
    if out.size == 0:
        raise ValueError("empty crop window")
    coords = None
    if da is not None:
        coords = {d: da[d].values[sl] for d, sl in
                  ((da.dims[-2], slice(r0, r1)), (da.dims[-1], slice(c0, c1)))
                  if d in da.coords}
    return _merge(out.copy(), meta, coords=coords)


def affine_transform(img, matrix, order: int = 1, cval: float = np.nan):
    """Warp with a 3x3 (or 2x3) forward matrix in pixel ``(col, row)`` coordinates.

    ``matrix`` maps input to output pixels, like OpenCV's ``cv2.warpAffine`` matrix (given
    here as a ``(2, 3)`` or ``(3, 3)`` array). Output keeps the input shape.
    """
    from skimage.transform import AffineTransform, warp

    m = np.asarray(matrix, dtype=float)
    if m.shape == (2, 3):
        m = np.vstack([m, [0, 0, 1]])
    if m.shape != (3, 3):
        raise ValueError("matrix must be (2, 3) or (3, 3)")
    tf = AffineTransform(matrix=m)
    arr, meta = _split(img)
    out = np.stack([warp(b.astype(np.float64), tf.inverse, order=order, cval=cval,
                         preserve_range=True) for b in arr]).astype(np.float32)
    return _merge(out, meta)


def perspective_transform(img, src_points, dst_points, output_shape: tuple[int, int] | None = None,
                          order: int = 1, cval: float = np.nan):
    """Perspective (homography) warp from four ``src_points`` to ``dst_points``, ``(col, row)`` each.

    Equivalent to ``cv2.getPerspectiveTransform`` + ``cv2.warpPerspective``. Output has
    ``output_shape=(height, width)`` (default: input shape) and is not georeferenced.
    """
    from skimage.transform import ProjectiveTransform, warp

    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src_points and dst_points must each be four (col, row) points")
    try:
        tf = ProjectiveTransform.from_estimate(src, dst)
        ok = bool(tf)
    except AttributeError:  # scikit-image < 0.26
        tf = ProjectiveTransform()
        ok = tf.estimate(src, dst)
    if not ok:
        raise ValueError("could not estimate a homography from the given points")
    arr, meta = _split(img)
    shp = output_shape or arr.shape[1:]
    out = np.stack([warp(b.astype(np.float64), tf.inverse, output_shape=shp, order=order,
                         cval=cval, preserve_range=True) for b in arr]).astype(np.float32)
    if meta["da"] is not None:
        dims = ("y", "x") if meta["squeeze"] else ("band", "y", "x")
        return xr.DataArray(out[0] if meta["squeeze"] else out, dims=dims,
                            attrs=dict(meta["da"].attrs))
    return out[0] if meta["squeeze"] else out


# --------------------------------------------------------------------------- 4. radiometric


def stretch(img, percentiles: tuple[float, float] = (2, 98), out_range: tuple[float, float] = (0, 1),
            per_band: bool = True):
    """Linear contrast stretch between percentiles (``skimage.exposure.rescale_intensity``)."""
    arr, meta = _split(img)
    a = arr.astype(np.float64)
    out = np.empty_like(a)
    groups = [a[i:i + 1] for i in range(a.shape[0])] if per_band else [a]
    pos = 0
    for g in groups:
        lo, hi = np.nanpercentile(g, percentiles)
        r = np.clip((g - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(g)
        r = np.where(np.isfinite(g), r, np.nan)
        out[pos:pos + g.shape[0]] = r * (out_range[1] - out_range[0]) + out_range[0]
        pos += g.shape[0]
    return _merge(out.astype(np.float32), meta)


def adjust_gamma(img, gamma: float = 1.0, gain: float = 1.0):
    """Gamma correction ``gain * v ** gamma`` on values scaled to [0, 1] (``exposure.adjust_gamma``).

    ``gamma < 1`` brightens, ``gamma > 1`` darkens.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    arr, meta = _split(img)
    out = np.stack([gain * np.power(_unit(b), gamma) for b in arr]).astype(np.float32)
    return _merge(out, meta)


def adjust_brightness_contrast(img, alpha: float = 1.0, beta: float = 0.0, clip: bool = True):
    """``alpha * v + beta`` on values scaled to [0, 1] (``cv2.convertScaleAbs`` style).

    ``alpha`` is the contrast gain, ``beta`` the brightness offset (in units of the [0, 1] range).
    """
    arr, meta = _split(img)
    out = np.stack([alpha * _unit(b) + beta for b in arr])
    if clip:
        out = np.clip(out, 0, 1)
    return _merge(out.astype(np.float32), meta)


def histogram(img, bins: int = 256, value_range: tuple[float, float] | None = None):
    """Per-band histogram. Returns ``{"counts": (B, bins), "edges": (bins + 1,)}`` (NaN ignored)."""
    arr, _ = _split(img)
    a = arr.astype(np.float64)
    if value_range is None:
        value_range = (float(np.nanmin(a)), float(np.nanmax(a)))
        if value_range[0] == value_range[1]:
            value_range = (value_range[0], value_range[0] + 1)
    counts = np.stack([np.histogram(b[np.isfinite(b)], bins=bins, range=value_range)[0] for b in a])
    return {"counts": counts, "edges": np.linspace(value_range[0], value_range[1], bins + 1)}


def equalize_hist(img, bins: int = 256):
    """Global histogram equalization per band, result in [0, 1] (``cv2.equalizeHist``)."""
    from skimage.exposure import equalize_hist as _eq

    return _per_band(img, lambda b: _eq(b, nbins=bins))


def equalize_adapthist(img, kernel_size: int | None = None, clip_limit: float = 0.01,
                       bins: int = 256):
    """Contrast-limited adaptive histogram equalization (CLAHE), result in [0, 1]."""
    from skimage.exposure import equalize_adapthist as _clahe

    return _per_band(img, lambda b: _clahe(_unit(b), kernel_size=kernel_size,
                                           clip_limit=clip_limit, nbins=bins))


def match_histograms(img, reference):
    """Match each band's histogram to ``reference`` (same band count or a single band).

    Useful for normalising images from different dates or sensors before comparing them.
    """
    from skimage.exposure import match_histograms as _match

    arr, meta = _split(img)
    ref, _ = _split(reference)
    if ref.shape[0] not in (1, arr.shape[0]):
        raise ValueError("reference must have one band or the same number of bands as img")
    out = np.stack([_match_one(b, ref[0 if ref.shape[0] == 1 else i], _match)
                    for i, b in enumerate(arr)]).astype(np.float32)
    return _merge(out, meta)


def _match_one(band, ref, _match):
    b = band.astype(np.float64)
    r = ref.astype(np.float64)
    r = r[np.isfinite(r)]
    nan = ~np.isfinite(b)
    res = _match(np.where(nan, np.nanmedian(b), b), r.reshape(1, -1) if r.ndim == 1 else r)
    res[nan] = np.nan
    return res


# --------------------------------------------------------------------------- 5. threshold


def threshold(img, method: str | float = "otsu", block_size: int = 35, offset: float = 0.0,
              return_value: bool = False):
    """Binary threshold per band. ``method``: a number, or ``otsu``, ``li``, ``triangle``,
    ``yen``, ``isodata``, ``mean``, ``local`` (adaptive over ``block_size`` pixels).

    Returns a ``uint8`` 0/1 image (NaN pixels are 0), plus the threshold(s) if
    ``return_value`` (``None`` for ``local``).
    """
    from skimage import filters

    funcs = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
             "triangle": filters.threshold_triangle, "yen": filters.threshold_yen,
             "isodata": filters.threshold_isodata, "mean": filters.threshold_mean}
    arr, meta = _split(img)
    outs, vals = [], []
    for b in arr.astype(np.float64):
        ok = np.isfinite(b)
        if isinstance(method, (int, float, np.number)):
            t = float(method)
            m = b > t
        elif method == "local":
            filled = np.where(ok, b, np.nanmedian(b))
            t = None
            m = filled > filters.threshold_local(filled, block_size=block_size, offset=offset)
        elif method in funcs:
            t = float(funcs[method](b[ok]))
            m = b > t
        else:
            raise ValueError(f"unknown threshold method {method!r}")
        outs.append((m & ok).astype(np.uint8))
        vals.append(t)
    res = _merge(np.stack(outs), meta)
    if return_value:
        return res, (vals[0] if meta["squeeze"] else vals)
    return res


# --------------------------------------------------------------------------- 6. filters


def convolve(img, kernel: np.ndarray | str = "sharpen", mode: str = "reflect"):
    """2-D convolution (correlation, as ``cv2.filter2D``) with a kernel array or a name in :data:`KERNELS`."""
    from scipy.ndimage import correlate

    k = KERNELS[kernel] if isinstance(kernel, str) else np.asarray(kernel, dtype=float)
    if k.ndim != 2:
        raise ValueError("kernel must be 2-D")
    return _per_band(img, lambda b: correlate(b, k, mode=mode))


def mean_blur(img, size: int = 5):
    """Box (mean) blur over ``size`` x ``size`` pixels (``cv2.blur``)."""
    from scipy.ndimage import uniform_filter

    if size < 1:
        raise ValueError("size must be >= 1")
    return _per_band(img, lambda b: uniform_filter(b, size=size, mode="reflect"))


def gaussian_blur(img, sigma: float = 1.0, truncate: float = 4.0):
    """Gaussian blur with standard deviation ``sigma`` pixels (``cv2.GaussianBlur``)."""
    from scipy.ndimage import gaussian_filter

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return _per_band(img, lambda b: gaussian_filter(b, sigma=sigma, truncate=truncate, mode="reflect"))


def median_blur(img, size: int = 3):
    """Median filter, robust to salt-and-pepper noise and SAR speckle spikes (``cv2.medianBlur``)."""
    from scipy.ndimage import median_filter

    if size < 1:
        raise ValueError("size must be >= 1")
    return _per_band(img, lambda b: median_filter(b, size=size, mode="reflect"))


def bilateral_denoise(img, sigma_color: float = 0.1, sigma_spatial: float = 3.0,
                      win_size: int | None = None):
    """Edge-preserving bilateral denoising on values scaled to [0, 1]; result is in that scale."""
    from skimage.restoration import denoise_bilateral

    return _per_band(img, lambda b: denoise_bilateral(_unit(b), win_size=win_size,
                                                      sigma_color=sigma_color,
                                                      sigma_spatial=sigma_spatial))


def tv_denoise(img, weight: float = 0.1):
    """Total-variation (Chambolle) denoising on values scaled to [0, 1]."""
    from skimage.restoration import denoise_tv_chambolle

    return _per_band(img, lambda b: denoise_tv_chambolle(_unit(b), weight=weight))


def sharpen(img, strength: float = 1.0):
    """Sharpen with the 3x3 Laplacian-based kernel, blended by ``strength`` (0 = unchanged)."""
    k = KERNELS["identity"] + strength * (KERNELS["sharpen"] - KERNELS["identity"])
    return convolve(img, k)


def unsharp_mask(img, radius: float = 2.0, amount: float = 1.0):
    """Unsharp masking: ``v + amount * (v - gaussian(v, radius))`` per band."""
    from scipy.ndimage import gaussian_filter

    return _per_band(img, lambda b: b + amount * (b - gaussian_filter(b, radius)))


def add_noise(img, mode: str = "gaussian", amount: float = 0.05, seed: int | None = None):
    """Add synthetic noise to values scaled to [0, 1] (``skimage.util.random_noise``).

    ``mode``: ``gaussian`` (``amount`` is the standard deviation), ``salt``, ``pepper``,
    ``s&p`` (``amount`` is the pixel fraction), ``speckle`` (multiplicative, as in SAR).
    """
    from skimage.util import random_noise

    kw = {"gaussian": {"var": amount ** 2}, "speckle": {"var": amount ** 2},
          "salt": {"amount": amount}, "pepper": {"amount": amount}, "s&p": {"amount": amount}}
    if mode not in kw:
        raise ValueError(f"unknown noise mode {mode!r}")
    arr, meta = _split(img)
    rng = np.random.default_rng(seed)
    out = np.stack([random_noise(_unit(b), mode=mode, rng=rng, **kw[mode]) for b in arr])
    return _merge(out.astype(np.float32), meta)


# --------------------------------------------------------------------------- 7. edges


def edges(img, method: str = "sobel", sigma: float = 1.0, low: float | None = None,
          high: float | None = None):
    """Edge detection per band.

    ``method``: ``sobel``, ``roberts``, ``prewitt``, ``scharr`` and ``laplacian`` return the
    float edge magnitude; ``canny`` returns a ``uint8`` 0/1 edge map (``sigma`` smoothing,
    hysteresis ``low`` / ``high`` thresholds).
    """
    from skimage import feature, filters
    from scipy.ndimage import laplace

    if method == "canny":
        return _per_band(img, lambda b: feature.canny(
            _unit(b), sigma=sigma, low_threshold=low, high_threshold=high).astype(np.uint8),
            dtype=np.uint8, keep_nan=False)
    fn = {"sobel": filters.sobel, "roberts": filters.roberts, "prewitt": filters.prewitt,
          "scharr": filters.scharr, "laplacian": lambda b: np.abs(laplace(b))}.get(method)
    if fn is None:
        raise ValueError(f"unknown edge method {method!r}")
    return _per_band(img, fn)


def sobel_h(img):
    """Horizontal-edge Sobel response (gradient along rows), signed."""
    from skimage.filters import sobel_h as _f

    return _per_band(img, _f)


def sobel_v(img):
    """Vertical-edge Sobel response (gradient along columns), signed."""
    from skimage.filters import sobel_v as _f

    return _per_band(img, _f)


# --------------------------------------------------------------------------- 8. morphology


def morphology(img, op: str = "open", size: int = 3, shape: str = "square"):
    """Morphological operation on a binary or gray image.

    ``op``: ``erode``, ``dilate``, ``open``, ``close``, ``gradient``, ``tophat``, ``blackhat``.
    ``shape``: ``square``, ``disk`` or ``cross`` structuring element of ``size`` pixels.
    """
    from scipy import ndimage as ndi

    if size < 1:
        raise ValueError("size must be >= 1")
    if shape == "square":
        fp = np.ones((size, size), bool)
    elif shape == "disk":
        r = size / 2.0
        yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
        fp = (yy ** 2 + xx ** 2) <= r ** 2 + 1e-9
    elif shape == "cross":
        fp = np.zeros((size, size), bool)
        fp[size // 2, :] = True
        fp[:, size // 2] = True
    else:
        raise ValueError(f"unknown structuring element {shape!r}")
    ops = {
        "erode": lambda b: ndi.grey_erosion(b, footprint=fp),
        "dilate": lambda b: ndi.grey_dilation(b, footprint=fp),
        "open": lambda b: ndi.grey_opening(b, footprint=fp),
        "close": lambda b: ndi.grey_closing(b, footprint=fp),
        "gradient": lambda b: ndi.grey_dilation(b, footprint=fp) - ndi.grey_erosion(b, footprint=fp),
        "tophat": lambda b: b - ndi.grey_opening(b, footprint=fp),
        "blackhat": lambda b: ndi.grey_closing(b, footprint=fp) - b,
    }
    if op not in ops:
        raise ValueError(f"unknown morphological op {op!r}")
    arr, _ = _split(img)
    dt = arr.dtype if arr.dtype.kind in "iuf" else np.float32
    return _per_band(img, ops[op], dtype=dt)


def remove_small_objects(mask, min_size: int = 20, connectivity: int = 1):
    """Remove connected ``True`` regions smaller than ``min_size`` pixels from a binary mask."""
    from skimage.morphology import remove_small_objects as _rso

    arr, meta = _split(mask)
    out = np.stack([_rso(b.astype(bool), max_size=min_size - 1, connectivity=connectivity)
                    if _rso_new() else _rso(b.astype(bool), min_size=min_size,
                                            connectivity=connectivity)
                    for b in arr]).astype(np.uint8)
    return _merge(out, meta)


def _rso_new() -> bool:
    import inspect

    from skimage.morphology import remove_small_objects as _rso

    return "max_size" in inspect.signature(_rso).parameters


# --------------------------------------------------------------------------- 9. features


def harris_corners(img, sigma: float = 1.0, k: float = 0.05, min_distance: int = 5,
                   threshold_rel: float = 0.02, band: int = 0) -> np.ndarray:
    """Harris corner detector on one band. Returns an ``(n, 2)`` array of ``(row, col)`` peaks."""
    from skimage.feature import corner_harris, corner_peaks

    arr, _ = _split(img)
    b = arr[band].astype(np.float64)
    b = np.where(np.isfinite(b), b, np.nanmedian(b))
    resp = corner_harris(_unit(b), k=k, sigma=sigma)
    return corner_peaks(resp, min_distance=min_distance, threshold_rel=threshold_rel)


def hog_features(img, orientations: int = 9, pixels_per_cell: tuple[int, int] = (8, 8),
                 cells_per_block: tuple[int, int] = (2, 2), band: int = 0,
                 visualize: bool = False):
    """Histogram of Oriented Gradients descriptor of one band (flat vector; ``visualize`` adds the image)."""
    from skimage.feature import hog

    arr, _ = _split(img)
    b = arr[band].astype(np.float64)
    b = _unit(np.where(np.isfinite(b), b, np.nanmedian(b)))
    return hog(b, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, visualize=visualize)


def to_vector(img, valid_only: bool = False) -> np.ndarray:
    """Flatten to a ``(n_pixels, n_bands)`` feature matrix (``np.reshape`` of the image).

    With ``valid_only`` rows containing NaN are dropped.
    """
    arr, _ = _split(img)
    m = arr.reshape(arr.shape[0], -1).T
    if valid_only:
        m = m[np.isfinite(m.astype(np.float64)).all(axis=1)]
    return m
