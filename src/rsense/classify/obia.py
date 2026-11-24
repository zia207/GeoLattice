"""Object-based image analysis: segmentation, per-segment features, segment classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import to_cpu
from ..optical.bands import as_dataset


def segment(obj: xr.Dataset | xr.DataArray, *, method: Literal["slic", "felzenszwalb", "watershed"]
            = "slic", bands: Sequence[str] | None = None, n_segments: int = 2000,
            compactness: float = 1.0, scale: float = 100.0, sigma: float = 0.5,
            min_size: int = 20) -> xr.DataArray:
    """Segment an image into objects (superpixels). Bands are min-max scaled first.

    ``slic`` (Achanta et al. 2012) gives compact, similarly sized segments; ``felzenszwalb``
    adapts to image content (``scale`` controls size); ``watershed`` floods the gradient
    magnitude from regular markers.
    """
    ds = as_dataset(obj)
    bands = list(bands or ds.data_vars)
    img = np.stack([np.asarray(to_cpu(ds[b]).values, dtype="float64") for b in bands], -1)
    valid = np.all(np.isfinite(img), -1)
    lo, hi = np.nanpercentile(img, 2, (0, 1)), np.nanpercentile(img, 98, (0, 1))
    img = np.clip((img - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
    img = np.where(valid[..., None], img, 0)
    if method == "slic":
        from skimage.segmentation import slic

        lab = slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma,
                   start_label=1, mask=valid if not valid.all() else None, channel_axis=-1,
                   convert2lab=False)
    elif method == "felzenszwalb":
        from skimage.segmentation import felzenszwalb

        lab = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size, channel_axis=-1) + 1
        lab[~valid] = 0
    elif method == "watershed":
        from skimage.filters import sobel
        from skimage.segmentation import watershed

        grad = sobel(img.mean(-1))
        step = max(2, int(np.sqrt(img.shape[0] * img.shape[1] / n_segments)))
        markers = np.zeros(grad.shape, int)
        rr, cc = np.mgrid[step // 2:grad.shape[0]:step, step // 2:grad.shape[1]:step]
        markers[rr, cc] = np.arange(1, rr.size + 1).reshape(rr.shape)
        lab = watershed(grad, markers, mask=valid, compactness=compactness)
    else:
        raise ValueError(f"Unknown method {method!r}")
    tpl = ds[bands[0]]
    return tpl.copy(data=lab.astype("int32")).rename("segment")


def segment_features(obj: xr.Dataset | xr.DataArray, segments: xr.DataArray, *,
                     bands: Sequence[str] | None = None,
                     stats: Sequence[str] = ("mean", "std"), shape: bool = True) -> pd.DataFrame:
    """Per-segment statistics (mean, std, min, max, median) for each band, plus shape
    descriptors (area in pixels, perimeter, compactness). Index = segment id."""
    from scipy import ndimage as snd

    ds = as_dataset(obj)
    bands = list(bands or ds.data_vars)
    lab = np.asarray(to_cpu(segments).values)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    fn = {"mean": snd.mean, "std": snd.standard_deviation, "min": snd.minimum,
          "max": snd.maximum, "median": snd.median}
    cols = {}
    for b in bands:
        v = np.asarray(to_cpu(ds[b]).values, dtype="float64")
        v = np.where(np.isfinite(v), v, 0)
        for s in stats:
            cols[f"{b}_{s}"] = fn[s](v, lab, ids)
    df = pd.DataFrame(cols, index=pd.Index(ids, name="segment"))
    if shape:
        area = snd.sum(np.ones_like(lab), lab, ids)
        edge = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
        perim = snd.sum(edge.astype(float), lab, ids)
        df["area_px"] = area
        df["perimeter_px"] = perim
        df["compactness"] = 4 * np.pi * area / np.maximum(perim, 1) ** 2
    return df


def segments_to_raster(values: pd.Series, segments: xr.DataArray, fill=np.nan) -> xr.DataArray:
    """Map per-segment values (indexed by segment id) back onto the segment raster."""
    lab = np.asarray(to_cpu(segments).values)
    lut_size = int(max(lab.max(), values.index.max())) + 1
    is_num = np.issubdtype(np.asarray(values.values).dtype, np.number)
    lut = np.full(lut_size, fill, dtype="float64" if is_num else object)
    lut[values.index.to_numpy().astype(int)] = values.to_numpy()
    out = lut[lab]
    out[lab == 0] = fill
    return segments.copy(data=out.astype("float32") if is_num else out).rename(values.name)


def segment_labels(segments: xr.DataArray, samples, label_col: str) -> pd.Series:
    """Majority label per segment from training points/polygons (for training an OBIA model)."""
    from rasterio.features import rasterize

    s = samples.to_crs(segments.rio.crs) if samples.crs else samples
    codes, uniques = pd.factorize(s[label_col])
    r = rasterize(((g, c + 1) for g, c in zip(s.geometry, codes)), out_shape=segments.shape,
                  transform=segments.rio.transform(), fill=0, dtype="int32", all_touched=True)
    lab = np.asarray(to_cpu(segments).values)
    m = (r > 0) & (lab > 0)
    df = pd.DataFrame({"seg": lab[m], "cls": r[m] - 1})
    maj = df.groupby("seg")["cls"].agg(lambda v: np.bincount(v).argmax())
    return pd.Series(uniques[maj.to_numpy()], index=maj.index, name=label_col)
