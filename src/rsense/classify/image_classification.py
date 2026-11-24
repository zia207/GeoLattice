"""Remote-sensing image classification and related vision tasks.

Module ``geolattice.rsense.classify.image_classification`` (also importable as
``geolattice.rsense.image_classification``).

One module that runs the full workflow from georeferenced rasters to evaluated, georeferenced
outputs. The sections follow the order in which a project is usually built:

1. **Data and task definition**: :func:`load_stack` (keeps CRS, transform, nodata, band
   names, dates and sensor in :class:`RasterMeta`), label tables and rasters, leakage-safe
   :func:`spatial_block_split` / :func:`temporal_split` / :func:`group_split`, and a
   :class:`Normalizer` fitted on training data only.
2. **Learning paradigm and spatial unit** (independent choices, see :class:`RunConfig`):
   :func:`unsupervised_classify` (k-means, mini-batch k-means, GMM), supervised
   :func:`train_pixel_classifier` (RF, SVM, XGBoost, ...) with spatial cross-validation, and
   :func:`object_based_classify` (segments as the spatial unit).
3. **PyTorch foundations**: :class:`ChipDataset`, :func:`extract_chips`,
   :func:`chips_from_folder`, :func:`build_cnn` (LeNet / VGG-style / ResNet / EfficientNet /
   ConvNeXt / ViT / Swin through torchvision or timm) with N-band stem adaptation,
   :func:`receptive_field`, overhead-imagery augmentation and :func:`cutmix`, losses
   (cross-entropy, weighted CE, focal, Dice, Lovasz, GCE, SCE) and :func:`train_model`
   (optimizer, schedule, AMP, gradient clipping, early stopping, checkpoints, seeds,
   TensorBoard / W&B / MLflow logging).
4. **Transfer learning and transformers**: :func:`set_trainable` (linear probe, partial,
   full fine-tuning), :func:`load_rs_backbone` (TorchGeo SSL4EO / Satlas weights),
   :func:`compare_transfer_modes`, :func:`vit_attention_rollout`.
5. **Semantic segmentation**: :func:`build_segmenter` (FCN, U-Net, DeepLabV3(+), SegFormer),
   :func:`sliding_window_predict` with seam blending, :func:`write_class_map`.
6. **Object detection**: :func:`build_object_detector`, :func:`box_iou`, :func:`nms`,
   :func:`mean_average_precision`, oriented boxes, scale-aware :func:`detection_tiles` and
   :func:`merge_detections`, :func:`count_objects`.
7. **Instance segmentation and boundaries**: :func:`clean_mask`,
   :func:`instances_from_semantic`, :func:`boundary_weight_map`, :func:`polygonize`
   (simplify / regularize), :func:`instances_from_detector`, :func:`export_vectors`.
8. **Change detection**: differencing, :func:`image_ratio`, change vector analysis,
   :func:`pca_change`, :func:`check_coregistration`, Siamese / early-fusion U-Nets and a
   BIT-style transformer (:func:`build_change_model`), :func:`change_metrics`.
9. **Spatio-temporal and multi-sensor learning**: TempCNN, LSTM, ConvLSTM, L-TAE, PSE-TAE
   and a U-TAE-style network (:func:`build_temporal_model`), early / middle / late
   optical-SAR fusion, SAR-assisted gap filling with a reconstruction flag, pansharpening
   and its quality metrics.
10. **Self-supervised pretraining**: SimCLR / MoCo / SeCo-style contrastive pretraining,
    masked-image modeling, :func:`extract_embeddings`, :func:`linear_probe` and geographic
    overlap checks.
11. **Geospatial foundation models**: :func:`foundation_model_info`,
    :func:`load_foundation_encoder`, TerraTorch fine-tuning configs, embedding similarity
    search and few-shot prototypes.
12. **Promptable segmentation and vision-language models**: SAM / SAM 2 and text-prompted
    segmentation through segment-geospatial, RemoteCLIP / GeoRSCLIP zero-shot
    classification. Outputs are flagged as unverified model output.
13. **Few or imperfect labels**: active learning, pseudo-labeling and self-training,
    FixMatch, weak labels from coarse products, noise-robust losses, CORAL and histogram
    domain adaptation, TENT test-time adaptation, and split-isolation checks.
14. **Evaluation, mapping and reproducibility**: :func:`classification_report` with
    bootstrap intervals, :func:`segmentation_metrics`, :func:`error_map`,
    :func:`compare_with_baselines`, :func:`save_run` / :func:`load_run`.

The command line (``python -m geolattice.rsense.classify.image_classification``) exposes the orthogonal
task, learning-mode and spatial-unit choices and runs the classical pixel/object workflows
end to end. Deep-learning parts need ``pip install geolattice[rsense-dl]``; scikit-learn models
need ``geolattice[rsense-classify]``. Heavy dependencies are imported only when used.

Example::

    import geolattice.rsense.classify.image_classification as ic

    img = ic.load_stack("Sentinel_2_UB/Images", sensor="Sentinel-2")
    pts = ic.labels_from_table("train_data.csv", label="Class_ID", crs=img.rio.crs)
    X, y, coords = ic.sample_pixels(img, pts, "Class_ID")
    split = ic.spatial_block_split(coords, block_size=500, test_size=0.3)
    clf, cv = ic.train_pixel_classifier(X.iloc[split.train], y[split.train], model="rf",
                                        coords=coords[split.train], block_size=500)
    landcover = ic.predict_map(clf, img)
    report = ic.classification_report(y[split.test], clf.predict(X.iloc[split.test]))
    ic.write_class_map(landcover, "landcover.tif", class_names={1: "water", 2: "building"})
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

TASKS = (
    "classification",
    "semantic-segmentation",
    "detection",
    "instance-segmentation",
    "change-detection",
    "time-series",
)
LEARNING_MODES = (
    "supervised",
    "unsupervised",
    "semi-supervised",
    "self-supervised",
)
SPATIAL_UNITS = ("pixel", "object")

UNVERIFIED_NOTE = ("Model output, not an authoritative geospatial fact: verify against imagery "
                   "or reference data before use.")


# =============================================================================================
# 0. Run configuration
# =============================================================================================


@dataclass(frozen=True)
class RunConfig:
    """Run selection: the orthogonal task, learning-mode and spatial-unit choices plus settings.

    ``task``, ``learning_mode`` and ``spatial_unit`` are independent. For example an
    ``object``-based ``semi-supervised`` ``classification`` is a valid combination.
    """

    task: str = "classification"
    learning_mode: str = "supervised"
    spatial_unit: str = "pixel"
    bands: tuple[str, ...] = ("red", "green", "blue")
    seed: int = 42
    model: str = "rf"
    n_classes: int = 6
    block_size: float | None = None
    test_size: float = 0.3
    val_size: float = 0.0
    image: str | None = None
    image2: str | None = None
    labels: str | None = None
    label_col: str = "class"
    x_col: str = "x"
    y_col: str = "y"
    out: str | None = None
    n_segments: int = 2000
    confidence: float = 0.9
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.learning_mode not in LEARNING_MODES:
            raise ValueError(f"learning_mode must be one of {LEARNING_MODES}")
        if self.spatial_unit not in SPATIAL_UNITS:
            raise ValueError(f"spatial_unit must be one of {SPATIAL_UNITS}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bands"] = list(self.bands)
        d["extra"] = dict(self.extra)
        return d


# =============================================================================================
# 1. Data and task definition
# =============================================================================================


@dataclass
class RasterMeta:
    """Geospatial metadata carried alongside arrays so outputs never lose their georeference."""

    crs: str | None
    transform: tuple[float, ...] | None
    nodata: float | None
    band_names: list[str]
    shape: tuple[int, int]
    dtype: str
    acquisition_dates: list[str] = field(default_factory=list)
    sensor: str | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dataarray(cls, da: xr.DataArray, **overrides: Any) -> RasterMeta:
        crs = da.rio.crs
        tr = da.rio.transform()
        bands = [str(b) for b in da["band"].values] if "band" in da.dims else [str(da.name)]
        meta = cls(crs=crs.to_wkt() if crs else None, transform=tuple(tr)[:6],
                   nodata=da.attrs.get("nodata_original", da.rio.nodata),
                   band_names=bands, shape=(int(da.shape[-2]), int(da.shape[-1])),
                   dtype=str(da.attrs.get("dtype_original", da.dtype)),
                   acquisition_dates=list(da.attrs.get("acquisition_dates", [])),
                   sensor=da.attrs.get("sensor"), sources=list(da.attrs.get("sources", [])))
        for k, v in overrides.items():
            setattr(meta, k, v)
        return meta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def _band_name_from_path(p: Path) -> str:
    stem = p.stem
    m = re.search(r"(B\d+A?)", stem, re.IGNORECASE)
    return m.group(1).upper() if m else stem.strip("_")


def load_stack(sources: str | PathLike | Sequence[str | PathLike], *,
               band_names: Sequence[str] | None = None, pattern: str = "*.tif",
               exclude: Sequence[str] = ("multi",), nodata: float | None = None,
               sensor: str | None = None, acquisition_dates: Sequence[str] | None = None,
               crs: Any = None, align: bool = False, dtype: str = "float32") -> xr.DataArray:
    """Read a georeferenced image as a ``(band, y, x)`` DataArray with metadata kept.

    ``sources`` is a multi-band file, a directory of single-band files (matched by
    ``pattern``, natural-sorted so ``B2 < B3 < B11``; files whose stem contains an
    ``exclude`` token are skipped), or an explicit list of files. Nodata becomes NaN and the
    original nodata value and dtype are kept in ``attrs``. CRS, transform, band names,
    acquisition dates and sensor are stored so they can be written back later.

    Files on different grids raise an error. Pass ``align=True`` to resample them onto the
    first file's grid; this is recorded in ``attrs["aligned"]`` and is never done silently.
    ``crs`` is assigned only to files that have no CRS (e.g. bare GeoTIFFs exported without
    a ``.prj``); a different CRS stored in the file is never overwritten.
    """
    import rioxarray

    if isinstance(sources, (str, PathLike)) and Path(sources).is_dir():
        files = sorted((f for f in Path(sources).glob(pattern)
                        if not any(t.lower() in f.stem.lower() for t in exclude)), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern!r} in {sources}")
    elif isinstance(sources, (str, PathLike)):
        files = [Path(sources)]
    else:
        files = [Path(s) for s in sources]
    arrays, names, aligned = [], [], False
    nodata_orig, dtype_orig = None, None
    ref = None
    for f in files:
        da = rioxarray.open_rasterio(f, masked=False)
        if da.rio.crs is None and crs is not None:
            da = da.rio.write_crs(crs)
        nd = nodata if nodata is not None else da.rio.nodata
        if nodata_orig is None and nd is not None:
            nodata_orig = float(nd)
        dtype_orig = dtype_orig or str(da.dtype)
        if ref is None:
            ref = da.isel(band=0)
        elif (da.shape[-2:] != ref.shape or da.rio.crs != ref.rio.crs
              or not np.allclose(tuple(da.rio.transform()), tuple(ref.rio.transform()))):
            if not align:
                raise ValueError(f"{f.name} is not on the grid of {files[0].name}; resample it "
                                 "first or pass align=True (recorded in attrs).")
            da = da.rio.reproject_match(ref)
            aligned = True
        a = da.astype(dtype)
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            a = a.where(da != nd)
        desc = list(da.attrs.get("long_name", ())) if isinstance(da.attrs.get("long_name"),
                                                                 (list, tuple)) else []
        n = da.sizes["band"]
        if len(files) == 1:
            names += desc if len(desc) == n and all(desc) else [f"b{i + 1}" for i in range(n)]
        else:
            names += [_band_name_from_path(f)] if n == 1 else [f"{_band_name_from_path(f)}_{i + 1}"
                                                               for i in range(n)]
        arrays.append(a)
    out = xr.concat(arrays, dim="band") if len(arrays) > 1 else arrays[0]
    if band_names is not None:
        if len(band_names) != out.sizes["band"]:
            raise ValueError(f"{len(band_names)} band names for {out.sizes['band']} bands")
        names = list(band_names)
    out = out.assign_coords(band=list(names))
    if ref.rio.crs is not None:
        out = out.rio.write_crs(ref.rio.crs)
    else:
        warnings.warn("image has no CRS; pass crs= to load_stack", UserWarning, stacklevel=2)
    out = out.rio.write_transform(ref.rio.transform())
    out = out.rio.write_nodata(np.nan, encoded=False)
    out.attrs = {"nodata_original": nodata_orig, "dtype_original": dtype_orig,
                 "sensor": sensor, "sources": [str(f) for f in files],
                 "acquisition_dates": list(acquisition_dates or []), "aligned": aligned}
    out.name = "image"
    return out


def as_band_dataset(image: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """``(band, y, x)`` DataArray → Dataset with one variable per band (feature table form)."""
    if isinstance(image, xr.Dataset):
        return image
    if "band" not in image.dims:
        return image.to_dataset(name=image.name or "band_1")
    ds = xr.Dataset({str(b): image.sel(band=b, drop=True) for b in image["band"].values})
    return ds.rio.write_crs(image.rio.crs) if image.rio.crs else ds


def as_band_array(image: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Dataset of bands → ``(band, y, x)`` DataArray (inverse of :func:`as_band_dataset`)."""
    if isinstance(image, xr.DataArray):
        return image if "band" in image.dims else image.expand_dims(band=[image.name or "b1"])
    da = image.to_array("band")
    return da.rio.write_crs(image.rio.crs) if image.rio.crs else da


def labels_from_table(table: str | PathLike | pd.DataFrame, *, x: str = "x", y: str = "y",
                      label: str | None = None, crs: Any = None):
    """Point labels from a CSV/DataFrame with coordinate columns → GeoDataFrame."""
    import geopandas as gpd

    df = pd.read_csv(table) if not isinstance(table, pd.DataFrame) else table.copy()
    if label is not None and label not in df:
        raise KeyError(f"label column {label!r} not in {list(df.columns)}")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)


def rasterize_labels(gdf, template: xr.DataArray, column: str, *, fill: int = -1,
                     all_touched: bool = False, codes: Mapping[Any, int] | None = None
                     ) -> tuple[xr.DataArray, dict[str, int]]:
    """Burn vector labels onto the image grid (class per pixel) → ``(label raster, codes)``.

    String classes are encoded (``codes`` maps names to integers, created if not given).
    Unlabeled pixels get ``fill``. The template's CRS is used; labels are reprojected to it.
    """
    from rasterio.features import rasterize

    g = gdf.to_crs(template.rio.crs) if gdf.crs and template.rio.crs else gdf
    vals = g[column]
    numeric = pd.api.types.is_numeric_dtype(vals.dtype)   # also safe for pandas string/categorical dtypes
    if codes is None:
        if numeric:
            codes = {str(v): int(v) for v in sorted(vals.unique())}
        else:
            codes = {str(v): i for i, v in enumerate(sorted(vals.astype(str).unique()))}
    num = vals if numeric else vals.astype(str).map(codes)
    tpl = template.isel(band=0, drop=True) if "band" in template.dims else template
    ydim, xdim = tpl.dims[-2:]
    arr = rasterize(((geom, int(c)) for geom, c in zip(g.geometry, num)),
                    out_shape=(tpl.sizes[ydim], tpl.sizes[xdim]), transform=tpl.rio.transform(),
                    fill=fill, dtype="int32", all_touched=all_touched)
    da = xr.DataArray(arr, dims=(ydim, xdim), coords={ydim: tpl[ydim], xdim: tpl[xdim]},
                      name="labels", attrs={"nodata": fill, "codes": json.dumps(codes)})
    return da.rio.write_crs(tpl.rio.crs), dict(codes)


def sample_pixels(image: xr.DataArray | xr.Dataset, samples, label_col: str | None = None, *,
                  max_per_polygon: int | None = None, seed: int = 0
                  ) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Spectral vectors at labeled points/polygons → ``(X, y, coords)``.

    ``samples`` is a GeoDataFrame (points: nearest pixel; polygons: pixels inside) or a
    label raster on the image grid (``-1``/NaN = unlabeled). ``coords`` (map x, y) is what
    the spatial splitters use. Rows with any NaN feature are dropped.
    """
    ds = as_band_dataset(image)
    if isinstance(samples, xr.DataArray):
        lab = np.asarray(samples.values)
        nod = samples.attrs.get("nodata", -1)
        ok = np.isfinite(lab) & (lab != nod) if lab.dtype.kind == "f" else lab != nod
        rows, cols = np.nonzero(ok)
        feats = list(ds.data_vars)
        X = pd.DataFrame({f: np.asarray(ds[f].values)[rows, cols] for f in feats})
        ydim, xdim = ds[feats[0]].dims[-2:]
        coords = np.c_[ds[xdim].values[cols], ds[ydim].values[rows]]
        y = lab[rows, cols]
        good = np.all(np.isfinite(X.to_numpy(float)), 1)
        return X[good].reset_index(drop=True), y[good], coords[good]
    from .estimators import extract_samples

    X, y = extract_samples(ds, samples, label_col, max_per_polygon=max_per_polygon, seed=seed)
    coords = X.attrs.get("coords")
    X = X.copy()
    X.attrs = {}
    return X, y, np.asarray(coords)


@dataclass
class Split:
    """Index arrays for train / validation / test plus the grouping used to make them."""

    train: np.ndarray
    test: np.ndarray
    val: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    groups: np.ndarray | None = None
    kind: str = "spatial-block"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "train": self.train.tolist(), "val": self.val.tolist(),
                "test": self.test.tolist()}

    def check(self) -> Split:
        """Raise if any index is in more than one split."""
        a, b, c = set(self.train.tolist()), set(self.val.tolist()), set(self.test.tolist())
        if a & b or a & c or b & c:
            raise ValueError("train/val/test splits overlap")
        return self


def spatial_blocks(coords: np.ndarray, block_size: float, origin: Sequence[float] | None = None
                   ) -> np.ndarray:
    """Integer block id of each ``(x, y)`` on a square grid of ``block_size`` map units."""
    c = np.asarray(coords, float)
    o = np.asarray(origin if origin is not None else c.min(0))
    ij = np.floor((c - o) / block_size).astype("int64")
    _, ids = np.unique(ij, axis=0, return_inverse=True)
    return ids.ravel()


def _split_groups(groups: np.ndarray, test_size: float, val_size: float, seed: int,
                  stratify: np.ndarray | None, kind: str) -> Split:
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    n_test = max(1, int(round(test_size * ug.size)))
    n_val = int(round(val_size * ug.size))
    if n_test + n_val >= ug.size:
        raise ValueError(f"only {ug.size} groups; use a smaller block/group size")
    if stratify is not None:
        # try a few shuffles and keep the one whose test set covers the most classes
        best, best_cov = ug.copy(), -1
        classes = np.unique(stratify)
        for _ in range(50):
            rng.shuffle(ug)
            cov = np.isin(classes, stratify[np.isin(groups, ug[:n_test])]).sum() + \
                np.isin(classes, stratify[np.isin(groups, ug[n_test + n_val:])]).sum()
            if cov > best_cov:
                best, best_cov = ug.copy(), cov
        ug = best
    test_g, val_g = ug[:n_test], ug[n_test:n_test + n_val]
    idx = np.arange(groups.size)
    test = idx[np.isin(groups, test_g)]
    val = idx[np.isin(groups, val_g)]
    train = idx[~np.isin(groups, np.r_[test_g, val_g])]
    return Split(train=train, test=test, val=val, groups=groups, kind=kind).check()


def spatial_block_split(coords: np.ndarray, *, block_size: float, test_size: float = 0.3,
                        val_size: float = 0.0, seed: int = 0, stratify: np.ndarray | None = None
                        ) -> Split:
    """Leakage-safe split by geography: whole spatial blocks go to train, val or test.

    Neighboring pixels are strongly autocorrelated, so a random pixel split overstates
    accuracy. ``block_size`` (map units) should exceed the autocorrelation range of the
    features. ``stratify`` (labels) picks the block shuffle that best covers all classes.
    """
    groups = spatial_blocks(coords, block_size)
    return _split_groups(groups, test_size, val_size, seed, stratify, "spatial-block")


def group_split(groups: Sequence, *, test_size: float = 0.3, val_size: float = 0.0,
                seed: int = 0, stratify: np.ndarray | None = None) -> Split:
    """Split by an explicit grouping such as scene id, tile id, field id or region."""
    return _split_groups(np.asarray(groups), test_size, val_size, seed, stratify, "group")


def temporal_split(dates: Sequence, *, train_end: str, val_end: str | None = None) -> Split:
    """Split by acquisition date.

    Train ``<= train_end``, val ``(train_end, val_end]``, test after ``val_end``.
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    te = np.datetime64(pd.Timestamp(train_end))
    ve = np.datetime64(pd.Timestamp(val_end)) if val_end else te
    idx = np.arange(d.size)
    return Split(train=idx[d <= te], val=idx[(d > te) & (d <= ve)], test=idx[d > ve],
                 kind="temporal").check()


def spatial_kfold(coords: np.ndarray, *, block_size: float, n_splits: int = 5, seed: int = 0
                  ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Spatial block k-fold: yields ``(train_idx, test_idx)`` with whole blocks held out."""
    groups = spatial_blocks(coords, block_size)
    ug = np.unique(groups)
    if ug.size < n_splits:
        raise ValueError(f"{ug.size} blocks < n_splits={n_splits}; reduce block_size")
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    folds = np.array_split(ug, n_splits)
    idx = np.arange(groups.size)
    for f in folds:
        m = np.isin(groups, f)
        yield idx[~m], idx[m]


@dataclass
class Normalizer:
    """Per-band normalization fitted on **training data only** and saved with the run.

    ``method``: ``zscore`` (mean/std), ``minmax`` or ``percentile`` (``low``/``high``
    percentiles, clipped). ``fit`` accepts ``(n, bands)`` tables, ``(band, y, x)`` images or
    ``(N, band, h, w)`` chip stacks.
    """

    method: Literal["zscore", "minmax", "percentile"] = "zscore"
    low: float = 2.0
    high: float = 98.0
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    band_names: list[str] | None = None

    @staticmethod
    def _as_table(a: Any, band_axis: int | None) -> np.ndarray:
        if isinstance(a, pd.DataFrame):
            return a.to_numpy("float64")
        if isinstance(a, (xr.DataArray, xr.Dataset)):
            a = as_band_array(a).values
        a = np.asarray(a, "float64")
        if a.ndim == 2 and band_axis is None:
            return a
        ax = band_axis if band_axis is not None else (0 if a.ndim == 3 else 1)
        return np.moveaxis(a, ax, -1).reshape(-1, a.shape[ax])

    def fit(self, data: Any, band_axis: int | None = None) -> Normalizer:
        t = self._as_table(data, band_axis)
        if isinstance(data, pd.DataFrame):
            self.band_names = list(data.columns)
        if self.method == "zscore":
            self.center, self.scale = np.nanmean(t, 0), np.nanstd(t, 0)
        elif self.method == "minmax":
            lo, hi = np.nanmin(t, 0), np.nanmax(t, 0)
            self.center, self.scale = lo, hi - lo
        elif self.method == "percentile":
            lo, hi = np.nanpercentile(t, self.low, 0), np.nanpercentile(t, self.high, 0)
            self.center, self.scale = lo, hi - lo
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.scale = np.where(self.scale > 0, self.scale, 1.0)
        return self

    def transform(self, data: Any, band_axis: int | None = None) -> Any:
        if self.center is None:
            raise RuntimeError("Normalizer.fit must be called on training data first")
        if isinstance(data, pd.DataFrame):
            out = (data - self.center) / self.scale
            return out.clip(0, 1) if self.method == "percentile" else out
        is_da = isinstance(data, xr.DataArray)
        a = np.asarray(data.values if is_da else data, "float32")
        if a.ndim == 2 and band_axis is None:
            shape = (1, -1)
        else:
            ax = band_axis if band_axis is not None else (0 if a.ndim == 3 else 1)
            shape = [1] * a.ndim
            shape[ax] = -1
        c = self.center.reshape(shape).astype("float32")
        s = self.scale.reshape(shape).astype("float32")
        out = (a - c) / s
        if self.method == "percentile":
            out = np.clip(out, 0, 1)
        return data.copy(data=out) if is_da else out

    def fit_transform(self, data: Any, band_axis: int | None = None) -> Any:
        return self.fit(data, band_axis).transform(data, band_axis)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "low": self.low, "high": self.high,
                "center": None if self.center is None else self.center.tolist(),
                "scale": None if self.scale is None else self.scale.tolist(),
                "band_names": self.band_names}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Normalizer:
        n = cls(method=d["method"], low=d.get("low", 2.0), high=d.get("high", 98.0),
                band_names=d.get("band_names"))
        n.center = None if d.get("center") is None else np.asarray(d["center"])
        n.scale = None if d.get("scale") is None else np.asarray(d["scale"])
        return n


# =============================================================================================
# 2. Learning paradigm and spatial unit: classical baselines
# =============================================================================================


def _image_matrix(image: xr.DataArray | xr.Dataset) -> tuple[np.ndarray, np.ndarray, xr.DataArray,
                                                             list[str]]:
    ds = as_band_dataset(image)
    feats = list(ds.data_vars)
    X = np.stack([np.asarray(ds[f].values, "float32").ravel() for f in feats], 1)
    ok = np.all(np.isfinite(X), 1)
    return X, ok, ds[feats[0]], feats


def unsupervised_classify(image: xr.DataArray | xr.Dataset, n_classes: int = 6, *,
                          method: Literal["kmeans", "minibatch_kmeans", "gmm"] = "kmeans",
                          sample: int = 200_000, normalize: bool = True, seed: int = 0,
                          **kw: Any) -> tuple[xr.DataArray, Any]:
    """Unsupervised pixel classification (k-means, mini-batch k-means or Gaussian mixture).

    Fits on a random sample of valid pixels (z-scored when ``normalize``) and labels every
    valid pixel; nodata pixels get ``-1``. Returns ``(cluster map, fitted model)``. Clusters
    are spectral classes: name them by inspecting their mean spectra (``model.cluster_means_``).
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture

    X, ok, tpl, feats = _image_matrix(image)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    fit_idx = rng.choice(idx, min(sample, idx.size), replace=False)
    norm = Normalizer().fit(X[fit_idx]) if normalize else None
    Xn = norm.transform(X) if norm else X
    if method == "kmeans":
        model = KMeans(n_clusters=n_classes, n_init=kw.pop("n_init", 10), random_state=seed, **kw)
    elif method == "minibatch_kmeans":
        model = MiniBatchKMeans(n_clusters=n_classes, random_state=seed, n_init=3, **kw)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_classes, random_state=seed,
                                covariance_type=kw.pop("covariance_type", "full"), **kw)
    else:
        raise ValueError(f"unknown method {method!r}")
    model.fit(Xn[fit_idx])
    lab = np.full(X.shape[0], -1, "int32")
    lab[ok] = model.predict(Xn[ok])
    model.cluster_means_ = pd.DataFrame(
        [X[ok][lab[ok] == k].mean(0) if np.any(lab[ok] == k) else np.full(len(feats), np.nan)
         for k in range(n_classes)], columns=feats)
    model.normalizer_ = norm
    out = tpl.copy(data=lab.reshape(tpl.shape)).rename("cluster")
    out.attrs = {"nodata": -1, "method": method}
    return out, model


def spatial_cross_validate(model_factory: Callable[[], Any], X: pd.DataFrame | np.ndarray,
                           y: np.ndarray, coords: np.ndarray, *, block_size: float,
                           n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Spatial block k-fold CV of any estimator with ``fit``/``predict`` → per-fold scores."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    rows = []
    Xa = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
    for k, (tr, te) in enumerate(spatial_kfold(coords, block_size=block_size, n_splits=n_splits,
                                               seed=seed)):
        m = model_factory()
        Xtr = Xa.iloc[tr] if isinstance(Xa, pd.DataFrame) else Xa[tr]
        Xte = Xa.iloc[te] if isinstance(Xa, pd.DataFrame) else Xa[te]
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                     "accuracy": accuracy_score(y[te], p),
                     "macro_f1": f1_score(y[te], p, average="macro", zero_division=0),
                     "kappa": cohen_kappa_score(y[te], p)})
    return pd.DataFrame(rows)


def train_pixel_classifier(X: pd.DataFrame | np.ndarray, y: np.ndarray, *, model: str = "rf",
                           coords: np.ndarray | None = None, block_size: float | None = None,
                           n_splits: int = 5, seed: int = 0, **params: Any
                           ) -> tuple[Any, pd.DataFrame | None]:
    """Fit a supervised pixel (or object) classifier, with spatial CV when ``coords`` are given.

    ``model``: ``rf``, ``svm``, ``xgb``, ``knn``, ``logistic``, ``gbm`` or ``mlp`` (see
    :class:`geolattice.rsense.classify.Classifier`, which picks cuML / XGBoost-GPU on CUDA).
    Returns ``(fitted classifier, CV table or None)``.
    """
    from .estimators import Classifier

    if model in ("rf", "gbm", "mlp", "logistic") and "random_state" not in params:
        params["random_state"] = seed

    def factory():
        return Classifier(model, **params)

    cv = None
    if coords is not None and block_size:
        cv = spatial_cross_validate(factory, X, np.asarray(y), coords, block_size=block_size,
                                    n_splits=n_splits, seed=seed)
    clf = factory().fit(X, np.asarray(y))
    return clf, cv


def predict_map(model: Any, image: xr.DataArray | xr.Dataset, *, proba: bool = False,
                normalizer: Normalizer | None = None, chunk_rows: int = 512
                ) -> xr.DataArray | xr.Dataset:
    """Apply a fitted pixel classifier to every valid pixel; nodata stays ``-1`` / NaN."""
    ds = as_band_dataset(image)
    if normalizer is not None:
        arr = normalizer.transform(as_band_array(ds))
        ds = as_band_dataset(arr)
    if hasattr(model, "predict_raster"):
        return model.predict_raster(ds, proba=proba, chunk_rows=chunk_rows)
    X, ok, tpl, feats = _image_matrix(ds)
    if proba:
        p = np.full((X.shape[0], len(model.classes_)), np.nan, "float32")
        p[ok] = model.predict_proba(pd.DataFrame(X[ok], columns=feats))
        return xr.Dataset({f"p_{c}": tpl.copy(data=p[:, i].reshape(tpl.shape))
                           for i, c in enumerate(model.classes_)})
    lab = np.full(X.shape[0], -1, "int32")
    lab[ok] = model.predict(pd.DataFrame(X[ok], columns=feats))
    return tpl.copy(data=lab.reshape(tpl.shape)).rename("prediction")


def object_based_classify(image: xr.DataArray | xr.Dataset, samples, label_col: str, *,
                          model: str = "rf", segments: xr.DataArray | None = None,
                          method: str = "slic", n_segments: int = 2000, compactness: float = 1.0,
                          stats: Sequence[str] = ("mean", "std"), seed: int = 0,
                          **params: Any) -> dict[str, Any]:
    """Object-based classification: segment, describe segments, label, classify, map back.

    Segments (``slic``, ``felzenszwalb`` or ``watershed``) are the spatial unit. Each gets
    per-band statistics plus shape descriptors; training segments take the majority label
    of ``samples``. Returns ``segments``, ``features``, ``train_labels``, the fitted
    ``model``, per-segment ``predictions`` and the ``map`` (``-1`` = no segment).
    """
    from .estimators import Classifier
    from .obia import segment, segment_features, segment_labels

    ds = as_band_dataset(image)
    seg = segments if segments is not None else segment(ds, method=method, n_segments=n_segments,
                                                        compactness=compactness)
    feats = segment_features(ds, seg, stats=stats)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    labels = segment_labels(seg, samples, label_col)
    labels = labels[labels.index.isin(feats.index)]
    if model in ("rf", "gbm", "mlp", "logistic") and "random_state" not in params:
        params["random_state"] = seed
    clf = Classifier(model, **params).fit(feats.loc[labels.index], labels.to_numpy())
    pred = pd.Series(clf.predict(feats), index=feats.index, name="class")
    classes = list(np.unique(np.asarray(labels)))
    numeric = np.issubdtype(np.asarray(classes).dtype, np.number)
    code = pred if numeric else pred.map({c: i for i, c in enumerate(classes)})
    lab = np.asarray(seg.values)
    lut = np.full(int(max(lab.max(), feats.index.max())) + 1, -1, "int64")
    lut[code.index.to_numpy()] = code.to_numpy().astype("int64")
    m = lut[lab]
    m[lab == 0] = -1
    cmap = seg.copy(data=m.astype("int32")).rename("class")
    cmap.attrs = {"nodata": -1}
    if not numeric:
        cmap.attrs["classes"] = [str(c) for c in classes]
    return {"segments": seg, "features": feats, "train_labels": labels, "model": clf,
            "predictions": pred, "map": cmap}


# =============================================================================================
# 3. PyTorch foundations
# =============================================================================================


def _torch():
    from ..dl._torch import require_torch

    return require_torch()


def _device(spec: str | None = None):
    from ..dl._torch import torch_device

    return torch_device(spec)


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices) for reproducible runs."""
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class ChipDataset:
    """Map-style dataset of image chips for chip classification or dense prediction.

    ``images``: ``(N, C, H, W)`` array. ``targets``: ``(N,)`` class ids (classification),
    ``(N, H, W)`` masks (segmentation / change) or None (self-supervised). ``normalizer``
    is a fitted :class:`Normalizer` (applied per band); ``transform`` gets and returns
    ``(image_tensor, target_tensor_or_None)``. Items are ``{"image", "label"|"mask"}`` dicts,
    which :func:`train_model` understands. Works with ``torch.utils.data.DataLoader``.
    """

    def __init__(self, images: np.ndarray, targets: np.ndarray | None = None, *,
                 normalizer: Normalizer | None = None, transform: Callable | None = None,
                 ignore_index: int = -100):
        self.images = np.asarray(images, "float32")
        if normalizer is not None:
            self.images = normalizer.transform(self.images, band_axis=1)
        self.images = np.nan_to_num(self.images)
        self.targets = None if targets is None else np.asarray(targets).astype("int64")
        self.transform, self.ignore_index = transform, ignore_index
        self.dense = self.targets is not None and self.targets.ndim == 3

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        torch = _torch()
        x = torch.from_numpy(self.images[i].copy())
        t = None if self.targets is None else torch.as_tensor(self.targets[i])
        if self.transform is not None:
            x, t = self.transform(x, t)
        out = {"image": x}
        if t is not None:
            out["mask" if self.dense else "label"] = t
        return out


def as_loader(dataset: Any, batch_size: int = 16, *, shuffle: bool = True, num_workers: int = 0,
              seed: int = 0, drop_last: bool = False, **kw: Any):
    """``torch.utils.data.DataLoader`` with a seeded generator (reproducible shuffling)."""
    torch = _torch()
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, generator=g, drop_last=drop_last,
                                       **kw)


def extract_chips(image: xr.DataArray | xr.Dataset, samples, label_col: str | None = None, *,
                  size: int = 32, max_nan_fraction: float = 0.0
                  ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Cut ``size``×``size`` chips centered on labeled points → ``(chips, labels, coords)``.

    ``samples`` is a point GeoDataFrame (reprojected to the image CRS) or an ``(n, 2)``
    array of map coordinates. Chips touching the image edge or with more than
    ``max_nan_fraction`` NaNs are dropped. ``chips`` has shape ``(N, C, size, size)``.
    """
    img = as_band_array(image)
    T = img.rio.transform()
    if hasattr(samples, "geometry"):
        s = samples.to_crs(img.rio.crs) if samples.crs and img.rio.crs else samples
        xy = np.c_[s.geometry.x.values, s.geometry.y.values]
        labels = s[label_col].to_numpy() if label_col else None
    else:
        xy, labels = np.asarray(samples, float), None
    inv = ~T
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    arr = np.asarray(img.values, "float32")
    H, W = arr.shape[-2:]
    h = size // 2
    chips, keep = [], []
    for k, (r, c) in enumerate(zip(rows, cols)):
        r0, c0 = r - h, c - h
        if r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
            continue
        chip = arr[:, r0:r0 + size, c0:c0 + size]
        if np.mean(~np.isfinite(chip)) > max_nan_fraction:
            continue
        chips.append(chip)
        keep.append(k)
    keep = np.asarray(keep, int)
    out = np.stack(chips) if chips else np.zeros((0, arr.shape[0], size, size), "float32")
    return out, (labels[keep] if labels is not None else None), xy[keep]


def rgb_mask_to_index(mask: np.ndarray, colors: Mapping[int, Sequence[int]] | np.ndarray,
                      unknown: int = -100) -> np.ndarray:
    """Color-coded mask ``(H, W, 3)`` → class index ``(H, W)`` (DeepGlobe/LandCover style).

    Colors are matched after thresholding at 128, which absorbs JPEG/resampling noise.
    """
    m = (np.asarray(mask)[..., :3] >= 128).astype("int64")
    code = m[..., 0] * 4 + m[..., 1] * 2 + m[..., 2]
    lut = np.full(8, unknown, "int64")
    items = colors.items() if isinstance(colors, Mapping) else enumerate(colors)
    for k, rgb in items:
        r, g, b = (np.asarray(rgb) >= 128).astype(int)
        lut[r * 4 + g * 2 + b] = k
    return lut[code]


def read_class_dict(path: str | PathLike) -> tuple[list[str], dict[int, tuple[int, int, int]]]:
    """``name,r,g,b`` CSV → ``(class names, {index: (r, g, b)})``."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    names = df[cols.get("name", df.columns[0])].astype(str).tolist()
    colors = {i: (int(r), int(g), int(b)) for i, (r, g, b) in
              enumerate(df[[cols["r"], cols["g"], cols["b"]]].to_numpy())}
    return names, colors


def chips_from_folder(folder: str | PathLike, *, image_suffix: str = "_sat.jpg",
                      mask_suffix: str = "_mask.png",
                      colors: Mapping[int, Sequence[int]] | None = None, size: int = 256,
                      stride: int | None = None, downsample: int = 1,
                      max_images: int | None = None, ignore_index: int = -100,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Tile paired image / mask files (e.g. DeepGlobe ``*_sat.jpg`` + ``*_mask.png``).

    Returns ``(chips (N, C, size, size) float32 in [0, 1], masks (N, size, size) or None,
    source file stems)``. ``colors`` maps class index to the mask RGB color (use
    :func:`read_class_dict`); single-band masks are used as class ids directly.
    ``downsample`` block-averages images before tiling to cut memory.
    """
    from PIL import Image

    folder = Path(folder)
    imgs = sorted(folder.glob(f"*{image_suffix}"))
    if max_images:
        rng = np.random.default_rng(seed)
        imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), min(max_images, len(imgs)),
                                                   replace=False))]
    stride = stride or size
    X, Y, src = [], [], []
    for p in imgs:
        im = np.asarray(Image.open(p).convert("RGB"), "float32") / 255.0
        mp = p.with_name(p.name.replace(image_suffix, mask_suffix))
        mk = None
        if mp.exists():
            raw = np.asarray(Image.open(mp))
            mk = rgb_mask_to_index(raw, colors, ignore_index) if (raw.ndim == 3 and colors) \
                else raw.astype("int64")
        if downsample > 1:
            f = downsample
            Hh, Ww = im.shape[0] // f * f, im.shape[1] // f * f
            im = im[:Hh, :Ww].reshape(Hh // f, f, Ww // f, f, 3).mean((1, 3))
            if mk is not None:
                mk = mk[:Hh:f, :Ww:f]
        H, W = im.shape[:2]
        for r in range(0, H - size + 1, stride):
            for c in range(0, W - size + 1, stride):
                X.append(np.moveaxis(im[r:r + size, c:c + size], -1, 0))
                if mk is not None:
                    Y.append(mk[r:r + size, c:c + size])
                src.append(p.stem)
    Xa = np.stack(X).astype("float32") if X else np.zeros((0, 3, size, size), "float32")
    Ya = np.stack(Y) if Y and len(Y) == len(X) else None
    return Xa, Ya, src


# ---- models ---------------------------------------------------------------------------------

_TORCHVISION_CLS = ("resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_b0",
                    "efficientnet_b1", "efficientnet_b2", "efficientnet_v2_s", "convnext_tiny",
                    "convnext_small", "convnext_base", "vgg11_bn", "vgg13_bn", "vgg16_bn",
                    "vit_b_16", "vit_b_32", "swin_t", "swin_s", "swin_v2_t", "mobilenet_v3_large",
                    "regnet_y_400mf", "densenet121")


@lru_cache(maxsize=None)
def _nets() -> SimpleNamespace:
    """Built-in networks, defined once on first use so PyTorch stays optional."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    def conv_bn(cin, cout, k=3, s=1, p=1):
        return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    class LeNet(nn.Module):
        """LeNet-5 style CNN (two conv/pool stages, three FC layers), size-agnostic."""

        def __init__(self, in_channels=3, num_classes=10, dropout=0.0):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(),
                                          nn.MaxPool2d(2), nn.Conv2d(6, 16, 5), nn.ReLU(),
                                          nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(5))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(400, 120), nn.ReLU(),
                                            nn.Dropout(dropout), nn.Linear(120, 84), nn.ReLU(),
                                            nn.Linear(84, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    class VGGSmall(nn.Module):
        """VGG-style stacks of 3x3 conv + BN + ReLU with max pooling and dropout."""

        def __init__(self, in_channels=3, num_classes=10, widths=(32, 64, 128), dropout=0.3):
            super().__init__()
            layers, c = [], in_channels
            for w in widths:
                layers += [conv_bn(c, w), conv_bn(w, w), nn.MaxPool2d(2)]
                c = w
            self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                            nn.Linear(c, num_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    class SimpleCNN(VGGSmall):
        def __init__(self, in_channels=3, num_classes=10, dropout=0.2):
            super().__init__(in_channels, num_classes, widths=(16, 32), dropout=dropout)

    class DictOut(nn.Module):
        """Unwrap torchvision segmentation outputs (``{"out": logits}``) to a tensor."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out["out"] if isinstance(out, dict) else out

    # ---- losses
    def _flat(logits, target, ignore_index):
        C = logits.shape[1]
        lg = logits.movedim(1, -1).reshape(-1, C)
        t = target.reshape(-1)
        keep = t != ignore_index
        return lg[keep], t[keep]

    class FocalLoss(nn.Module):
        """Focal loss (Lin et al. 2017): down-weights easy examples; ``alpha`` = class weights."""

        def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
            super().__init__()
            self.gamma, self.ignore_index = gamma, ignore_index
            self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha).float())

        def forward(self, logits, target):
            lg, t = _flat(logits, target, self.ignore_index)
            if t.numel() == 0:
                return logits.sum() * 0
            logp = F.log_softmax(lg.float(), 1).gather(1, t[:, None]).squeeze(1)
            loss = -((1 - logp.exp()) ** self.gamma) * logp
            if self.alpha is not None:
                loss = loss * self.alpha.to(loss.device)[t]
            return loss.mean()

    class DiceLoss(nn.Module):
        """Soft multi-class Dice loss (1 - mean Dice over classes present in the batch)."""

        def __init__(self, smooth=1.0, ignore_index=-100):
            super().__init__()
            self.smooth, self.ignore_index = smooth, ignore_index

        def forward(self, logits, target):
            lg, t = _flat(logits, target, self.ignore_index)
            if t.numel() == 0:
                return logits.sum() * 0
            p = F.softmax(lg.float(), 1)
            oh = F.one_hot(t, lg.shape[1]).float()
            inter = (p * oh).sum(0)
            denom = p.sum(0) + oh.sum(0)
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            present = oh.sum(0) > 0
            return 1 - dice[present].mean()

    def _lovasz_grad(gt_sorted):
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jac = 1.0 - inter / union
        if gt_sorted.numel() > 1:
            jac[1:] = jac[1:] - jac[:-1]
        return jac

    class LovaszSoftmaxLoss(nn.Module):
        """Lovasz-Softmax loss (Berman et al. 2018), a convex surrogate of the Jaccard index."""

        def __init__(self, ignore_index=-100, classes="present"):
            super().__init__()
            self.ignore_index, self.classes = ignore_index, classes

        def forward(self, logits, target):
            lg, t = _flat(logits, target, self.ignore_index)
            if t.numel() == 0:
                return logits.sum() * 0
            p = F.softmax(lg.float(), 1)
            losses = []
            for c in range(p.shape[1]):
                fg = (t == c).float()
                if self.classes == "present" and fg.sum() == 0:
                    continue
                err = (fg - p[:, c]).abs()
                err_s, perm = torch.sort(err, descending=True)
                losses.append(torch.dot(err_s, _lovasz_grad(fg[perm])))
            return torch.stack(losses).mean() if losses else logits.sum() * 0

    class GeneralizedCrossEntropy(nn.Module):
        """Noise-robust GCE loss (Zhang & Sabuncu 2018): ``(1 - p_y^q) / q``."""

        def __init__(self, q=0.7, ignore_index=-100):
            super().__init__()
            self.q, self.ignore_index = q, ignore_index

        def forward(self, logits, target):
            lg, t = _flat(logits, target, self.ignore_index)
            p = F.softmax(lg.float(), 1).gather(1, t[:, None]).squeeze(1).clamp_min(1e-7)
            return ((1 - p ** self.q) / self.q).mean()

    class SymmetricCrossEntropy(nn.Module):
        """Noise-robust SCE loss (Wang et al. 2019): ``alpha * CE + beta * reverse CE``."""

        def __init__(self, alpha=0.1, beta=1.0, ignore_index=-100):
            super().__init__()
            self.a, self.b, self.ignore_index = alpha, beta, ignore_index

        def forward(self, logits, target):
            lg, t = _flat(logits, target, self.ignore_index)
            ce = F.cross_entropy(lg.float(), t)
            p = F.softmax(lg.float(), 1).clamp(1e-7, 1.0)
            oh = F.one_hot(t, lg.shape[1]).float().clamp(1e-4, 1.0)
            rce = -(p * oh.log()).sum(1).mean()
            return self.a * ce + self.b * rce

    class ComboLoss(nn.Module):
        """Weighted sum of losses, e.g. CE + Dice for imbalanced segmentation."""

        def __init__(self, losses, weights):
            super().__init__()
            self.losses = nn.ModuleList(losses)
            self.weights = list(weights)

        def forward(self, logits, target):
            return sum(w * L(logits, target) for w, L in zip(self.weights, self.losses))

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if isinstance(v, type) or callable(v)})


def _replace_module(model, name: str, new) -> None:
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def adapt_input_channels(model, in_channels: int, *,
                         method: Literal["mean", "repeat", "random"] = "mean"):
    """Make the first convolution accept ``in_channels`` bands (multispectral / SAR stems).

    ``mean`` fills extra channels with the averaged RGB filter, ``repeat`` cycles the RGB
    filters, both rescaled to keep activation magnitude; ``random`` re-initializes the stem
    (Kaiming) so new per-band filters are learned. Returns the model (modified in place).
    """
    torch = _torch()
    nn = torch.nn
    from ..dl.models.layers import adapt_first_conv

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if m.in_channels == in_channels:
                return model
            if method == "random":
                new = nn.Conv2d(in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                                m.dilation, m.groups, m.bias is not None, m.padding_mode)
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            else:
                new = adapt_first_conv(m, in_channels, method=method)
            _replace_module(model, name, new)
            return model
    raise ValueError("model has no Conv2d layer to adapt")


def _replace_head(model, num_classes: int):
    nn = _torch().nn
    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = (name, m)
    if last is None:
        raise ValueError("no Linear classification head found")
    name, m = last
    _replace_module(model, name, nn.Identity() if num_classes == 0
                    else nn.Linear(m.in_features, num_classes))
    return model


def build_cnn(arch: str = "resnet18", *, in_channels: int = 3, num_classes: int = 10,
              pretrained: bool = False, first_conv: Literal["mean", "repeat", "random"] = "mean",
              dropout: float | None = None, **kwargs: Any):
    """Chip/scene classifier for any number of bands.

    ``arch``:

    * built-in (plain PyTorch): ``lenet``, ``vgg_small``, ``simple_cnn``;
    * torchvision: ``resnet18/34/50/101``, ``efficientnet_b0..b2``, ``efficientnet_v2_s``,
      ``convnext_tiny/small/base``, ``vgg11_bn/13_bn/16_bn``, ``vit_b_16/32``, ``swin_t/s``,
      ``swin_v2_t``, ``mobilenet_v3_large``, ``densenet121`` (ViT needs 224 px input);
    * timm: ``timm:<name>`` or any other timm model name (``convnext_atto``,
      ``vit_small_patch16_224``, ``swin_tiny_patch4_window7_224``, ...).

    With ``pretrained=True`` ImageNet weights are loaded and the stem is adapted to
    ``in_channels`` with ``first_conv`` (see :func:`adapt_input_channels`); compare against
    ``pretrained=False`` (training the stem from scratch). ``num_classes=0`` returns a
    feature extractor (head removed).
    """
    torch = _torch()
    N = _nets()
    a = arch.lower()
    builtin = {"lenet": N.LeNet, "vgg_small": N.VGGSmall, "simple_cnn": N.SimpleCNN}
    if a in builtin:
        kw = {} if dropout is None else {"dropout": dropout}
        model = builtin[a](in_channels, max(num_classes, 1), **kw)
        if num_classes == 0:
            model.classifier[-1] = torch.nn.Identity()
        return model
    if a in _TORCHVISION_CLS and not a.startswith("timm:"):
        import torchvision

        model = torchvision.models.get_model(a, weights="DEFAULT" if pretrained else None,
                                             **kwargs)
        if in_channels != 3:
            adapt_input_channels(model, in_channels, method=first_conv)
        _replace_head(model, num_classes)
        if dropout is not None:
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = dropout
        return model
    try:
        import timm
    except ImportError as e:
        raise ImportError(f"arch={arch!r} needs `pip install timm` (or use a torchvision / "
                          "built-in architecture)") from e
    name = arch.split(":", 1)[1] if a.startswith("timm:") else arch
    kw = dict(kwargs)
    if dropout is not None:
        kw["drop_rate"] = dropout
    model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels,
                              num_classes=num_classes, **kw)
    if first_conv == "random" and pretrained and in_channels != 3:
        for m in model.modules():  # timm adapted the stem; re-initialize it instead
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                break
    return model


def receptive_field(model, *, input_size: int | None = None) -> dict[str, float]:
    """Theoretical receptive field of the conv/pool layers in registration order.

    Exact for sequential CNNs (LeNet, VGG); for branched networks (ResNet) it follows the
    main path and is an upper-bound style estimate. Returns ``rf`` (pixels), ``jump``
    (output stride) and, with ``input_size``, the fraction of the input seen per output unit.
    """
    nn = _torch().nn
    rf, jump = 1.0, 1.0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            d = getattr(m, "dilation", 1)
            d = d if isinstance(d, int) else d[0]
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.stride == (1, 1):
                continue
            rf += (k - 1) * d * jump
            jump *= s
    out = {"rf": rf, "jump": jump}
    if input_size:
        out["coverage"] = min(1.0, rf / input_size)
    return out


def count_parameters(model, trainable_only: bool = True) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only))


# ---- augmentation ---------------------------------------------------------------------------


class OverheadAugment:
    """Augmentation for nadir imagery, usable as :class:`ChipDataset` ``transform``.

    Overhead images have no intrinsic "up", so all 8 dihedral transforms (90° rotations ×
    flips) are valid. Radiometric jitter (``brightness``, ``contrast`` as fractions,
    ``noise`` as std in normalized units) should only be used where physically sensible,
    e.g. not on calibrated indices. ``crop`` takes a random crop and resizes back.
    Masks receive the same geometric transform (nearest neighbor).
    """

    def __init__(self, *, dihedral: bool = True, brightness: float = 0.0, contrast: float = 0.0,
                 noise: float = 0.0, crop: float | None = None, p: float = 1.0, seed: int = 0):
        self.dihedral, self.brightness, self.contrast = dihedral, brightness, contrast
        self.noise, self.crop, self.p = noise, crop, p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, t=None):
        torch = _torch()
        if self.rng.random() > self.p:
            return x, t
        dense = t is not None and t.ndim >= 2
        if self.dihedral:
            k = int(self.rng.integers(0, 4))
            x = torch.rot90(x, k, dims=(-2, -1))
            if dense:
                t = torch.rot90(t, k, dims=(-2, -1))
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
                if dense:
                    t = torch.flip(t, dims=(-1,))
        if self.crop:
            H, W = x.shape[-2:]
            ch, cw = max(2, int(H * self.crop)), max(2, int(W * self.crop))
            r = int(self.rng.integers(0, H - ch + 1))
            c = int(self.rng.integers(0, W - cw + 1))
            F = torch.nn.functional
            x = F.interpolate(x[None, :, r:r + ch, c:c + cw], size=(H, W), mode="bilinear",
                              align_corners=False)[0]
            if dense:
                t = F.interpolate(t[None, None, r:r + ch, c:c + cw].float(), size=(H, W),
                                  mode="nearest")[0, 0].long()
        if self.brightness:
            x = x + float(self.rng.uniform(-self.brightness, self.brightness)) * x.abs().mean()
        if self.contrast:
            m = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - m) * float(1 + self.rng.uniform(-self.contrast, self.contrast)) + m
        if self.noise:
            x = x + torch.randn_like(x) * self.noise
        return x, t


def cutmix(images, targets, *, alpha: float = 1.0, rng: np.random.Generator | None = None):
    """CutMix (Yun et al. 2019) on a batch.

    Classification targets ``(N,)`` → ``(mixed, y_a, y_b, lam)`` for the loss
    ``lam * L(y_a) + (1 - lam) * L(y_b)``. Dense targets ``(N, H, W)`` are cut and pasted with
    the image → ``(mixed, mixed_mask, None, 1.0)``.
    """
    torch = _torch()
    rng = rng or np.random.default_rng()
    N, _, H, W = images.shape
    lam = float(rng.beta(alpha, alpha))
    perm = torch.as_tensor(rng.permutation(N), device=images.device)
    ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    mixed = images.clone()
    mixed[:, :, y0:y1, x0:x1] = images[perm, :, y0:y1, x0:x1]
    if targets.ndim >= 3:
        mt = targets.clone()
        mt[:, y0:y1, x0:x1] = targets[perm, y0:y1, x0:x1]
        return mixed, mt, None, 1.0
    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return mixed, targets, targets[perm], lam


# ---- losses ---------------------------------------------------------------------------------


def class_weights(y: Sequence, *, num_classes: int | None = None, ignore_index: int = -100,
                  method: Literal["inverse", "sqrt_inverse", "effective"] = "inverse",
                  beta: float = 0.999) -> np.ndarray:
    """Class weights from label frequencies (normalized to mean 1) for imbalanced data.

    ``effective`` uses the effective number of samples (Cui et al. 2019).
    """
    a = np.asarray(y).ravel()
    a = a[a != ignore_index]
    k = num_classes or int(a.max()) + 1
    n = np.bincount(a.astype(int), minlength=k).astype(float)
    n = np.maximum(n, 1)
    if method == "inverse":
        w = 1 / n
    elif method == "sqrt_inverse":
        w = 1 / np.sqrt(n)
    elif method == "effective":
        w = (1 - beta) / (1 - beta ** n)
    else:
        raise ValueError(method)
    return (w / w.mean()).astype("float32")


def get_loss(name: str = "ce", *, class_weights: Sequence[float] | None = None,
             ignore_index: int = -100, gamma: float = 2.0, label_smoothing: float = 0.0,
             **kw: Any):
    """Loss by name.

    ``ce`` (cross-entropy), ``weighted_ce`` (needs ``class_weights``), ``focal``, ``dice``,
    ``lovasz``, ``ce+dice``, ``gce`` / ``sce`` (robust to noisy labels). All accept
    ``(N, C)`` logits with ``(N,)`` targets or ``(N, C, H, W)`` with ``(N, H, W)`` masks.
    """
    torch = _torch()
    N = _nets()
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
    n = name.lower()
    if n in ("ce", "cross_entropy"):
        return torch.nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
    if n in ("weighted_ce", "wce"):
        if w is None:
            raise ValueError("weighted_ce needs class_weights (see class_weights())")
        return torch.nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index,
                                         label_smoothing=label_smoothing)
    if n == "focal":
        return N.FocalLoss(gamma=gamma, alpha=w, ignore_index=ignore_index)
    if n == "dice":
        return N.DiceLoss(ignore_index=ignore_index, **kw)
    if n == "lovasz":
        return N.LovaszSoftmaxLoss(ignore_index=ignore_index)
    if n in ("ce+dice", "ce_dice"):
        return N.ComboLoss([torch.nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index),
                            N.DiceLoss(ignore_index=ignore_index)], kw.get("weights", (1.0, 1.0)))
    if n == "gce":
        return N.GeneralizedCrossEntropy(q=kw.get("q", 0.7), ignore_index=ignore_index)
    if n == "sce":
        return N.SymmetricCrossEntropy(kw.get("alpha", 0.1), kw.get("beta", 1.0), ignore_index)
    raise ValueError(f"unknown loss {name!r}")


# ---- training -------------------------------------------------------------------------------


def _unpack(batch, dev):
    if isinstance(batch, Mapping):
        x = batch["image"]
        y = batch.get("label", batch.get("mask"))
    else:
        x, y = batch[0], (batch[1] if len(batch) > 1 else None)
    if isinstance(x, (list, tuple)):
        x = tuple(t.to(dev).float() for t in x)
    else:
        x = x.to(dev).float()
    return x, (None if y is None else y.to(dev).long())


def _forward(model, x):
    return model(*x) if isinstance(x, tuple) else model(x)


class _Logger:
    def __init__(self, kind: Any, config: Mapping | None, log_dir: str | None):
        self.kind, self.impl = kind, None
        if callable(kind):
            self.impl = kind
        elif kind == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.impl = SummaryWriter(log_dir=log_dir)
        elif kind == "wandb":
            import wandb

            self.impl = wandb.init(config=dict(config or {}), dir=log_dir, reinit=True)
        elif kind == "mlflow":
            import mlflow

            mlflow.start_run()
            if config:
                mlflow.log_params({k: str(v) for k, v in config.items()})
            self.impl = mlflow

    def log(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if self.impl is None:
            return
        if callable(self.kind):
            self.kind(epoch, dict(metrics))
        elif self.kind == "tensorboard":
            for k, v in metrics.items():
                self.impl.add_scalar(k, v, epoch)
        elif self.kind == "wandb":
            self.impl.log(dict(metrics), step=epoch)
        elif self.kind == "mlflow":
            self.impl.log_metrics(dict(metrics), step=epoch)

    def close(self) -> None:
        if self.kind == "tensorboard" and self.impl is not None:
            self.impl.close()
        elif self.kind == "wandb" and self.impl is not None:
            self.impl.finish()
        elif self.kind == "mlflow" and self.impl is not None:
            self.impl.end_run()


def _confusion_np(pred: np.ndarray, target: np.ndarray, n: int, ignore: int) -> np.ndarray:
    m = target != ignore
    idx = target[m].astype("int64") * n + pred[m].astype("int64")
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def train_model(model, train_loader, val_loader=None, *, task: Literal["classification",
                "segmentation"] = "classification", epochs: int = 20,
                loss: str | Any = "ce", optimizer: Literal["adamw", "adam", "sgd"] = "adamw",
                lr: float = 1e-3, weight_decay: float = 1e-4,
                scheduler: Literal["cosine", "onecycle", "step", "plateau", "none"] = "cosine",
                precision: Literal["32", "16", "bf16"] = "32", grad_clip: float | None = 1.0,
                early_stopping: int | None = None,
                monitor: Literal["val_loss", "val_score"] = "val_score",
                checkpoint: str | PathLike | None = None, seed: int | None = 42,
                num_classes: int | None = None, ignore_index: int = -100,
                cutmix_prob: float = 0.0, logger: Any = None, log_dir: str | None = None,
                device: str | None = None, verbose: bool = True,
                config: Mapping | None = None) -> dict[str, list]:
    """Explicit PyTorch training/validation loop for chip classification or segmentation.

    Features: AdamW/Adam/SGD; cosine, one-cycle, step or plateau schedules; mixed precision
    (``"16"`` with gradient scaling on CUDA, ``"bf16"``); gradient clipping; CutMix
    (``cutmix_prob``); early stopping on ``monitor`` with ``early_stopping`` patience;
    best-weights restore and ``checkpoint`` saving; seeding; logging to TensorBoard,
    Weights & Biases, MLflow (``logger="tensorboard"|"wandb"|"mlflow"``) or any callable
    ``logger(epoch, metrics)``. ``val_score`` is accuracy (classification) or mean IoU
    (segmentation). Returns the per-epoch history.
    """
    torch = _torch()
    if seed is not None:
        set_seed(seed)
    dev = _device(device)
    model.to(dev)
    loss_fn = get_loss(loss, ignore_index=ignore_index) if isinstance(loss, str) else loss
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError(optimizer)
    steps = max(1, len(train_loader))
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    elif scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, max(1, epochs // 3), 0.1)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2)
    else:
        sched = None
    amp = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler(enabled=precision == "16" and dev.type == "cuda")
    log = _Logger(logger, config, log_dir)
    rng = np.random.default_rng(seed)
    hist: dict[str, list] = {"train_loss": [], "val_loss": [], "val_score": [], "lr": []}
    best, best_state, bad = None, None, 0
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            x, y = _unpack(batch, dev)
            yb, lam = None, 1.0
            if cutmix_prob and not isinstance(x, tuple) and rng.random() < cutmix_prob:
                x, y, yb, lam = cutmix(x, y, rng=rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                out = _forward(model, x)
                L = loss_fn(out, y) if yb is None else lam * loss_fn(out, y) + \
                    (1 - lam) * loss_fn(out, yb)
            scaler.scale(L).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            if scheduler == "onecycle":
                sched.step()
            bs = (x[0] if isinstance(x, tuple) else x).shape[0]
            tot += float(L.detach()) * bs
            n += bs
        hist["train_loss"].append(tot / max(n, 1))
        hist["lr"].append(float(opt.param_groups[0]["lr"]))
        metrics = {"train_loss": hist["train_loss"][-1]}
        if val_loader is not None:
            model.eval()
            vl, vn, conf = 0.0, 0, None
            with torch.no_grad():
                for batch in val_loader:
                    x, y = _unpack(batch, dev)
                    with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                        out = _forward(model, x)
                    out = out.float()
                    bs = (x[0] if isinstance(x, tuple) else x).shape[0]
                    vl += float(loss_fn(out, y)) * bs
                    vn += bs
                    k = num_classes or out.shape[1]
                    c = _confusion_np(out.argmax(1).cpu().numpy(), y.cpu().numpy(), k, ignore_index)
                    conf = c if conf is None else conf + c
            tp = np.diag(conf)
            if task == "segmentation":
                with np.errstate(divide="ignore", invalid="ignore"):
                    score = float(np.nanmean(tp / (conf.sum(0) + conf.sum(1) - tp)))
            else:
                score = float(tp.sum() / max(conf.sum(), 1))
            hist["val_loss"].append(vl / max(vn, 1))
            hist["val_score"].append(score)
            metrics.update(val_loss=hist["val_loss"][-1], val_score=score)
            cur = -hist["val_loss"][-1] if monitor == "val_loss" else score
            if best is None or cur > best:
                best, bad = cur, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint:
                    torch.save({"state_dict": best_state, "epoch": ep, "metrics": metrics},
                               checkpoint)
            else:
                bad += 1
        if sched is not None and scheduler != "onecycle":
            if scheduler == "plateau":
                sched.step(hist["val_loss"][-1] if hist["val_loss"] else hist["train_loss"][-1])
            else:
                sched.step()
        log.log(ep, metrics)
        if verbose:
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"epoch {ep + 1}/{epochs} {msg}")
        if early_stopping and bad >= early_stopping:
            if verbose:
                print(f"early stopping at epoch {ep + 1}")
            break
    log.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint:
        torch.save({"state_dict": model.state_dict(), "epoch": epochs - 1}, checkpoint)
    return hist


def load_checkpoint(model, path: str | PathLike, *, strict: bool = True, map_location="cpu"):
    """Load weights saved by :func:`train_model` (or a raw ``state_dict``) into ``model``."""
    torch = _torch()
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, Mapping) else ck
    missing = model.load_state_dict(sd, strict=strict)
    return model, missing


def predict_proba_batches(model, data: Any, *, batch_size: int = 64, device: str | None = None,
                          tta: bool = False) -> np.ndarray:
    """Softmax probabilities for chips (array ``(N, C, H, W)``, ChipDataset or DataLoader).

    ``tta=True`` averages over the 8 dihedral transforms (test-time augmentation).
    Segmentation models return ``(N, K, H, W)``.
    """
    torch = _torch()
    dev = _device(device)
    model.to(dev).eval()
    if isinstance(data, np.ndarray):
        data = ChipDataset(data)
    loader = data if hasattr(data, "batch_size") else as_loader(data, batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for batch in loader:
            x, _ = _unpack(batch, dev)
            if not tta:
                p = torch.softmax(_forward(model, x).float(), 1)
            else:
                p = 0
                for k in range(4):
                    for flip in (False, True):
                        xt = torch.rot90(x, k, (-2, -1))
                        xt = torch.flip(xt, (-1,)) if flip else xt
                        o = torch.softmax(_forward(model, xt).float(), 1)
                        if o.ndim == 4:  # undo for dense outputs
                            o = torch.flip(o, (-1,)) if flip else o
                            o = torch.rot90(o, -k, (-2, -1))
                        p = p + o / 8
            outs.append(p.cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0,))


def predict_chip_map(model, image: xr.DataArray | xr.Dataset, *, size: int = 32,
                     stride: int | None = None, normalizer: Normalizer | None = None,
                     batch_size: int = 256, device: str | None = None) -> xr.DataArray:
    """Apply a chip classifier on a sliding grid → georeferenced class map.

    Each output pixel takes the class of the chip centered closest to it (stride
    ``stride``; default ``size // 4``). Chips with NaNs predict nodata (``-1``).
    """
    img = as_band_array(image)
    arr = np.asarray(img.values, "float32")
    if normalizer is not None:
        arr = normalizer.transform(arr, band_axis=0)
    C, H, W = arr.shape
    stride = stride or max(1, size // 4)
    from ..dl.datasets import grid_windows

    wins = grid_windows(H, W, size, stride)
    chips = np.stack([arr[:, r:r + size, c:c + size] for r, c in wins])
    bad = ~np.all(np.isfinite(chips), axis=(1, 2, 3))
    p = predict_proba_batches(model, np.nan_to_num(chips), batch_size=batch_size, device=device)
    lab = p.argmax(1)
    lab[bad] = -1
    centers = np.array([(r + size / 2, c + size / 2) for r, c in wins])
    rr, cc = np.mgrid[0:H, 0:W]
    from scipy.spatial import cKDTree

    _, nn_idx = cKDTree(centers).query(np.c_[rr.ravel() + 0.5, cc.ravel() + 0.5])
    out = lab[nn_idx].reshape(H, W).astype("int32")
    valid = np.all(np.isfinite(np.asarray(img.values)), 0)
    out[~valid] = -1
    tpl = img.isel(band=0, drop=True)
    res = tpl.copy(data=out).rename("class")
    res.attrs = {"nodata": -1}
    return res


# =============================================================================================
# 4. Transfer learning and vision transformers
# =============================================================================================

#: Remote-sensing pretrained backbones available through TorchGeo (``torchgeo.models``).
#: Values are ``(timm architecture, TorchGeo weight enum)``; any other enum string can be
#: passed directly to :func:`load_rs_backbone` as ``"torchgeo:<Enum.MEMBER>"``.
RS_WEIGHTS: dict[str, tuple[str, str]] = {
    "ssl4eo_s2_resnet18_moco": ("resnet18", "ResNet18_Weights.SENTINEL2_ALL_MOCO"),
    "ssl4eo_s2_resnet50_moco": ("resnet50", "ResNet50_Weights.SENTINEL2_ALL_MOCO"),
    "ssl4eo_s2_resnet50_dino": ("resnet50", "ResNet50_Weights.SENTINEL2_ALL_DINO"),
    "ssl4eo_s2_rgb_resnet50_moco": ("resnet50", "ResNet50_Weights.SENTINEL2_RGB_MOCO"),
    "ssl4eo_s1_resnet50_moco": ("resnet50", "ResNet50_Weights.SENTINEL1_ALL_MOCO"),
    "ssl4eo_s2_vits16_dino": ("vit_small_patch16_224", "ViTSmall16_Weights.SENTINEL2_ALL_DINO"),
    "ssl4eo_l8_resnet50_moco": ("resnet50", "ResNet50_Weights.LANDSAT_OLI_SR_MOCO"),
    "satlas_s2_swinb_rgb": ("swin_v2_b", "Swin_V2_B_Weights.SENTINEL2_SI_RGB_SATLAS"),
    "satlas_s2_swinb_ms": ("swin_v2_b", "Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS"),
    "satlas_s2_swint_rgb": ("swin_v2_t", "Swin_V2_T_Weights.SENTINEL2_SI_RGB_SATLAS"),
    "satlas_s2_swint_ms": ("swin_v2_t", "Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS"),
    "satlas_naip_swinb": ("swin_v2_b", "Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS"),
    "seco_resnet50": ("resnet50", "ResNet50_Weights.SENTINEL2_RGB_SECO"),
}


def load_rs_backbone(name: str, *, num_classes: int = 0, weights_path: str | None = None):
    """Backbone with remote-sensing pretrained weights (TorchGeo SSL4EO, Satlas, SeCo, ...).

    ``name`` is a key of :data:`RS_WEIGHTS` or ``"torchgeo:<Enum.MEMBER>"``. Band count,
    order and normalization expected by each checkpoint are in ``model.rs_weights_meta``;
    check them (and the license) before use. ``num_classes > 0`` replaces the head.
    ``weights_path`` loads any compatible state dict instead, e.g. a BigEarthNet checkpoint.
    """
    try:
        from torchgeo.models import get_model, get_weight
    except ImportError as e:
        raise ImportError("load_rs_backbone needs `pip install torchgeo`") from e
    if name.startswith("torchgeo:"):
        enum = name.split(":", 1)[1]
        arch = None
    else:
        if name not in RS_WEIGHTS:
            raise KeyError(f"unknown backbone {name!r}; choose from {sorted(RS_WEIGHTS)}")
        arch, enum = RS_WEIGHTS[name]
    w = get_weight(enum)
    arch = arch or w.meta.get("model", "resnet50")
    model = get_model(arch, weights=None if weights_path else w)
    if weights_path:
        load_checkpoint(model, weights_path, strict=False)
    model.rs_weights_meta = dict(getattr(w, "meta", {}))
    if num_classes:
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes)
        else:
            _replace_head(model, num_classes)
    return model


def _head_names(model) -> list[str]:
    nn = _torch().nn
    for attr in ("fc", "head", "heads", "classifier"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
            return [attr]
    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = name
    return [last] if last else []


def set_trainable(model, mode: Literal["linear_probe", "partial", "full"] = "full", *,
                  n_blocks: int = 1) -> dict[str, int]:
    """Freeze a pretrained encoder for transfer learning.

    ``linear_probe``: only the classification head trains (the encoder is frozen; keep
    BatchNorm statistics frozen too by calling ``model.eval()`` on the encoder).
    ``partial``: the head plus the last ``n_blocks`` top-level encoder stages train.
    ``full``: everything trains. Returns trainable / total parameter counts.
    """
    heads = _head_names(model)
    for p in model.parameters():
        p.requires_grad = mode == "full"
    if mode != "full":
        for h in heads:
            mod = model.get_submodule(h)
            for p in mod.parameters():
                p.requires_grad = True
    if mode == "partial":
        stages = [(n, m) for n, m in model.named_children()
                  if n not in heads and any(True for _ in m.parameters())]
        if hasattr(model, "blocks"):  # ViT style
            stages = [(f"blocks.{i}", b) for i, b in enumerate(model.blocks)]
        elif hasattr(model, "layers") and not isinstance(model.layers, _torch().nn.Linear):
            stages = [(f"layers.{i}", b) for i, b in enumerate(model.layers)]
        elif isinstance(getattr(model, "features", None), _torch().nn.Sequential):
            # torchvision Swin / ConvNeXt / EfficientNet: stages live in ``features``
            stages = [(f"features.{n}", b) for n, b in model.features.named_children()
                      if any(True for _ in b.parameters())]
        for _, m in stages[-n_blocks:]:
            for p in m.parameters():
                p.requires_grad = True
        for nm in ("norm", "fc_norm"):  # final normalization layer in front of the head
            mod = getattr(model, nm, None)
            if isinstance(mod, _torch().nn.Module) and nm not in heads:
                for p in mod.parameters():
                    p.requires_grad = True
    elif mode not in ("linear_probe", "full"):
        raise ValueError(mode)
    return {"trainable": count_parameters(model), "total": count_parameters(model, False)}


def compare_transfer_modes(build_fn: Callable[[], Any], train_loader, val_loader, *,
                           modes: Sequence[str] = ("linear_probe", "partial", "full"),
                           epochs: int = 5, lr: Mapping[str, float] | float = 1e-3,
                           region: str | None = None, **train_kw: Any) -> pd.DataFrame:
    """Train the same pretrained model under several transfer modes and tabulate results.

    ``build_fn`` returns a fresh pretrained model each call. The table records the mode,
    trainable parameters, number of training chips (data volume), the ``region`` tag (for
    geographic-transfer experiments) and the best validation score / loss.
    """
    rows = []
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None
    for mode in modes:
        model = build_fn()
        counts = set_trainable(model, mode)
        lr_m = lr[mode] if isinstance(lr, Mapping) else lr
        h = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr_m, verbose=False,
                        **train_kw)
        rows.append({"mode": mode, **counts, "n_train": n_train, "region": region,
                     "best_val_score": max(h["val_score"]) if h["val_score"] else np.nan,
                     "best_val_loss": min(h["val_loss"]) if h["val_loss"] else np.nan})
    return pd.DataFrame(rows)


def vit_attention_rollout(model, x, *, discard_ratio: float = 0.0, head_fusion: str = "mean"
                          ) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) for timm ViT/DeiT models → ``(N, gh, gw)``.

    Attention maps show where the class token attends, not a causal explanation of the
    prediction; read them cautiously and compare with occlusion / gradient methods.
    """
    torch = _torch()
    if not hasattr(model, "blocks"):
        raise TypeError("attention rollout supports timm ViT models (with .blocks)")
    attns, hooks = [], []
    for blk in model.blocks:
        a = blk.attn
        if hasattr(a, "fused_attn"):
            a.fused_attn = False
        hooks.append(a.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o.detach())))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    result = None
    for A in attns:
        A = A.max(1).values if head_fusion == "max" else A.mean(1)
        if discard_ratio:
            flat = A.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            idx = flat.topk(k, largest=False).indices
            flat.scatter_(1, idx, 0)
        eye = torch.eye(A.shape[-1], device=A.device)
        A = (A + eye) / 2
        A = A / A.sum(-1, keepdim=True)
        result = A if result is None else A @ result
    mask = result[:, 0, n_prefix:]
    g = int(round(math.sqrt(mask.shape[-1])))
    mask = mask.reshape(-1, g, g)
    mask = mask / mask.flatten(1).max(1).values[:, None, None]
    return mask.cpu().numpy()


# =============================================================================================
# 5. Semantic segmentation
# =============================================================================================


def build_segmenter(arch: str = "unet", *, in_channels: int = 3, num_classes: int = 2,
                    encoder: str = "resnet34", weights: str | None = None,
                    first_conv: Literal["mean", "repeat", "random"] = "mean", **kwargs: Any):
    """Semantic segmentation network returning ``(N, K, H, W)`` logits.

    * ``fcn_resnet50``, ``fcn_resnet101``, ``deeplabv3_resnet50``, ``deeplabv3_mobilenet``,
      ``lraspp_mobilenet``: torchvision (``weights="imagenet"`` loads an ImageNet backbone);
    * ``unet``, ``unetplusplus``, ``deeplabv3plus``, ``fpn``, ``manet``, ``pspnet``,
      ``segformer``, ``upernet``: segmentation-models-pytorch (falls back to the built-in
      U-Net for ``unet`` when smp is not installed);
    * ``unet_basic``: built-in U-Net, plain PyTorch.
    """
    N = _nets()
    a = arch.lower()
    tv = {"fcn_resnet50": "fcn_resnet50", "fcn_resnet101": "fcn_resnet101",
          "deeplabv3_resnet50": "deeplabv3_resnet50", "deeplabv3_resnet101": "deeplabv3_resnet101",
          "deeplabv3_mobilenet": "deeplabv3_mobilenet_v3_large",
          "lraspp_mobilenet": "lraspp_mobilenet_v3_large"}
    if a in tv:
        import torchvision.models.segmentation as tvs

        wb = "DEFAULT" if weights in ("imagenet", "DEFAULT") else None
        model = getattr(tvs, tv[a])(weights=None, weights_backbone=wb, num_classes=num_classes,
                                    **({"aux_loss": False} if not a.startswith("lraspp") else {}),
                                    **kwargs)
        if in_channels != 3:
            adapt_input_channels(model, in_channels, method=first_conv)
        return N.DictOut(model)
    from ..dl.models import build_segmentation_model

    return build_segmentation_model(a, in_channels=in_channels, classes=num_classes,
                                    encoder=encoder, weights=weights, **kwargs)


def sliding_window_predict(model, image: xr.DataArray | xr.Dataset, *, tile: int = 256,
                           overlap: int = 64, normalizer: Normalizer | None = None,
                           blend: Literal["hann", "triangle", "none"] = "hann",
                           return_proba: bool = False, batch_size: int = 8,
                           device: str | None = None,
                           precision: Literal["32", "16", "bf16"] = "32"):
    """Large-image inference with overlapping windows and blended probabilities (no seams).

    Wraps :func:`geolattice.rsense.dl.predict_segmentation`; the result keeps the image grid
    and CRS so it can go straight to :func:`write_class_map`.
    """
    from ..dl.infer import predict_segmentation

    img = as_band_array(image)
    norm = None
    if normalizer is not None:
        norm = (np.asarray(normalizer.center, "float32"), np.asarray(normalizer.scale, "float32"))
    out = predict_segmentation(model, img, tile=tile, overlap=overlap, batch_size=batch_size,
                               normalize=norm, blend=blend, return_proba=return_proba,
                               device=device, precision=precision)
    return out.rio.write_transform(img.rio.transform())


def write_class_map(class_map: xr.DataArray, path: str | PathLike, *,
                    class_names: Mapping[int, str] | Sequence[str] | None = None,
                    colormap: Mapping[int, Sequence[int]] | None = None,
                    nodata: int | None = None, metadata: Mapping[str, Any] | None = None,
                    compress: str = "deflate") -> Path:
    """Write a class raster as a georeferenced GeoTIFF with class metadata.

    ``-1`` (nodata in this module) is remapped to 255 for uint8 output. Class names are
    stored as a GDAL raster attribute-style tag (``class_names`` JSON) and in the band
    description; a ``colormap`` (``{class: (r, g, b)}``) is embedded for uint8 maps.
    ``metadata`` adds extra tags (e.g. model, date, run id).
    """
    import rasterio

    path = Path(path)
    a = np.asarray(class_map.values)
    if a.dtype.kind == "f":
        a = np.where(np.isfinite(a), a, -1)
    a = a.astype("int64")
    small = a.min() >= -1 and a.max() < 255
    if small:
        nd = 255 if nodata is None else nodata
        data = np.where(a < 0, nd, a).astype("uint8")
    else:
        nd = -1 if nodata is None else nodata
        data = np.where(a < 0, nd, a).astype("int32")
    if class_names is not None and not isinstance(class_names, Mapping):
        class_names = dict(enumerate(class_names))
    crs = class_map.rio.crs
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs=crs, transform=class_map.rio.transform(),
                       nodata=nd, compress=compress) as dst:
        dst.write(data, 1)
        tags = {"class_names": json.dumps({str(k): v for k, v in (class_names or {}).items()})}
        tags.update({k: (v if isinstance(v, str) else json.dumps(v, default=str))
                     for k, v in (metadata or {}).items()})
        dst.update_tags(**tags)
        dst.set_band_description(1, "class")
        if colormap and data.dtype == np.uint8:
            dst.write_colormap(1, {int(k): tuple(int(c) for c in v[:3]) + (255,)
                                   for k, v in colormap.items()})
    return path


def segmentation_metrics(pred: Any, target: Any, *, num_classes: int | None = None,
                         ignore_index: int = -100, class_names: Sequence[str] | None = None
                         ) -> dict[str, Any]:
    """Per-class IoU, Dice/F1, precision, recall, support and mean IoU for masks."""
    p = np.asarray(getattr(pred, "values", pred)).ravel()
    t = np.asarray(getattr(target, "values", target)).ravel()
    keep = (t != ignore_index) & (t >= 0) & (p >= 0)
    p, t = p[keep].astype(int), t[keep].astype(int)
    k = num_classes or int(max(p.max(initial=0), t.max(initial=0))) + 1
    cm = _confusion_np(p, t, k, -10**9)  # rows = reference, cols = prediction
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        prec, rec = tp / (tp + fp), tp / (tp + fn)
    names = list(class_names) if class_names else list(range(k))
    per = pd.DataFrame({"iou": iou, "dice": dice, "precision": prec, "recall": rec,
                        "support": cm.sum(1)}, index=names[:k])
    return {"miou": float(np.nanmean(iou)), "mean_dice": float(np.nanmean(dice)),
            "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)), "per_class": per,
            "confusion_matrix": pd.DataFrame(cm, index=names[:k], columns=names[:k])}


# =============================================================================================
# 6. Object detection
# =============================================================================================


def build_object_detector(arch: str = "faster_rcnn", *, num_classes: int = 2,
                          in_channels: int = 3, pretrained: bool = True, **kwargs: Any):
    """Detector: two-stage ``faster_rcnn`` / ``mask_rcnn``, one-stage ``retinanet`` / ``fcos``
    (torchvision, ``num_classes`` includes background) or ``yolo`` (Ultralytics; pass a
    weights name such as ``yolov8n.pt`` or ``yolov8n-obb.pt`` for oriented boxes)."""
    if arch.lower().startswith("yolo"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("YOLO needs `pip install ultralytics`") from e
        return YOLO(kwargs.pop("weights", "yolov8n.pt"))
    from ..dl.models import build_detector

    if not pretrained:
        kwargs.setdefault("weights_backbone", None)  # no ImageNet download from scratch
    return build_detector(arch, num_classes=num_classes, in_channels=in_channels,
                          pretrained=pretrained, **kwargs)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of axis-aligned boxes ``(n, 4)`` × ``(m, 4)`` in ``x0, y0, x1, y1``."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5,
        labels: np.ndarray | None = None) -> np.ndarray:
    """Greedy non-maximum suppression (class-wise when ``labels`` given) → kept indices."""
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    if labels is not None:
        keep = [np.flatnonzero(labels == c)[nms(boxes[labels == c], scores[labels == c],
                                                iou_threshold)] for c in np.unique(labels)]
        k = np.concatenate(keep) if keep else np.array([], int)
        return k[np.argsort(-scores[k])]
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, int)


def _average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision(predictions: Sequence[Mapping], ground_truth: Sequence[Mapping], *,
                           iou_thresholds: Sequence[float] = (0.5,),
                           iou_fn: Callable | None = None) -> dict[str, Any]:
    """Detection mAP (all-point interpolated AP, VOC/COCO style).

    ``predictions[i]`` = ``{"boxes", "scores", "labels"}`` and ``ground_truth[i]`` =
    ``{"boxes", "labels"}`` for image ``i``. ``iou_thresholds=np.arange(0.5, 0.96, 0.05)``
    gives COCO mAP@[.5:.95]. ``iou_fn`` can replace :func:`box_iou`, e.g. for oriented boxes.
    Returns ``map``, per-threshold ``map@t`` and a per-class AP table.
    """
    iou_fn = iou_fn or box_iou
    classes = sorted({int(c) for g in ground_truth for c in np.asarray(g["labels"]).ravel()} |
                     {int(c) for p in predictions for c in np.asarray(p["labels"]).ravel()})
    table = {}
    for t in iou_thresholds:
        aps = {}
        for c in classes:
            dets, npos, gts = [], 0, {}
            for i, (p, g) in enumerate(zip(predictions, ground_truth)):
                gm = np.asarray(g["labels"]) == c
                gts[i] = (np.asarray(g["boxes"])[gm], np.zeros(gm.sum(), bool))
                npos += int(gm.sum())
                pm = np.asarray(p["labels"]) == c
                for b, s in zip(np.asarray(p["boxes"])[pm], np.asarray(p["scores"])[pm]):
                    dets.append((float(s), i, b))
            if npos == 0:
                continue
            dets.sort(key=lambda d: -d[0])
            tp = np.zeros(len(dets))
            for k, (_, i, b) in enumerate(dets):
                gb, used = gts[i]
                if len(gb):
                    ious = iou_fn(np.asarray(b)[None], gb)[0]
                    j = int(np.argmax(ious))
                    if ious[j] >= t and not used[j]:
                        tp[k], used[j] = 1, True
            ctp = np.cumsum(tp)
            rec = ctp / npos
            prec = ctp / np.arange(1, len(dets) + 1)
            aps[c] = _average_precision(rec, prec) if len(dets) else 0.0
        table[t] = aps
    per = pd.DataFrame(table)
    per.columns = [f"AP@{t:.2f}" for t in iou_thresholds]
    out = {"map": float(np.nanmean(per.to_numpy())) if per.size else 0.0, "per_class": per}
    for t, col in zip(iou_thresholds, per.columns):
        out[f"map@{t:.2f}"] = float(per[col].mean()) if per.size else 0.0
    return out


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle: float):
    """Oriented box (center, size, rotation in degrees counter-clockwise) → shapely Polygon."""
    from shapely import affinity
    from shapely.geometry import box

    return affinity.rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), angle,
                           origin=(cx, cy))


def polygon_iou(a: Sequence, b: Sequence) -> np.ndarray:
    """Pairwise IoU of shapely polygons (e.g. oriented boxes) → ``(len(a), len(b))``."""
    out = np.zeros((len(a), len(b)))
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            inter = pa.intersection(pb).area
            u = pa.area + pb.area - inter
            out[i, j] = inter / u if u > 0 else 0.0
    return out


def detection_tiles(height: int, width: int, *, tile: int = 640, overlap: float = 0.2,
                    scales: Sequence[float] = (1.0,)) -> list[tuple[int, int, int]]:
    """Scale-aware tiling for small objects → ``(row, col, window_size)`` triplets.

    For each scale ``s`` the window is ``tile / s`` pixels (``s > 1`` zooms in on small
    objects once the window is resized to ``tile``); consecutive windows overlap by
    ``overlap`` so objects on tile edges are seen whole at least once.
    """
    from ..dl.datasets import grid_windows

    out = []
    for s in scales:
        size = max(8, int(round(tile / s)))
        stride = max(1, int(size * (1 - overlap)))
        for r, c in grid_windows(max(height, size), max(width, size), size, stride):
            out.append((r, c, size))
    return out


def merge_detections(gdf, *, iou_threshold: float = 0.5, by: str | None = "label",
                     score_col: str = "score"):
    """Remove duplicate detections from overlapping tiles (polygon NMS in map units).

    Works for axis-aligned and oriented boxes. Keeps the highest-scoring polygon of each
    overlapping group (per ``by`` class when given).
    """
    from shapely import STRtree

    if len(gdf) == 0:
        return gdf
    g = gdf.sort_values(score_col, ascending=False).reset_index(drop=True)
    geoms = list(g.geometry)
    tree = STRtree(geoms)
    removed = np.zeros(len(g), bool)
    for i, gi in enumerate(geoms):
        if removed[i]:
            continue
        for j in tree.query(gi):
            if j <= i or removed[j]:
                continue
            if by and g[by].iloc[j] != g[by].iloc[i]:
                continue
            inter = gi.intersection(geoms[j]).area
            u = gi.area + geoms[j].area - inter
            if u > 0 and inter / u > iou_threshold:
                removed[j] = True
    return g[~removed].reset_index(drop=True)


def detect_objects(model, image: xr.DataArray | xr.Dataset, *, tile: int = 640,
                   overlap: int = 128, score_threshold: float = 0.3, iou_threshold: float = 0.5,
                   class_names: Mapping[int, str] | None = None,
                   normalizer: Normalizer | None = None,
                   device: str | None = None):
    """Sliced detection over a large image → GeoDataFrame of boxes in map coordinates."""
    from ..dl.infer import predict_detection

    norm = None if normalizer is None else (normalizer.center, normalizer.scale)
    return predict_detection(model, as_band_array(image), tile=tile, overlap=overlap,
                             score_threshold=score_threshold, iou_threshold=iou_threshold,
                             normalize=norm, class_names=dict(class_names or {}), device=device)


def count_objects(detections, zones, *, zone_id: str | None = None, label: str | None = None):
    """Count detections (e.g. trees, livestock) per zone polygon → zones with a ``count`` column."""
    import geopandas as gpd

    d = detections if label is None else detections[detections["label"] == label]
    pts = gpd.GeoDataFrame(d.drop(columns="geometry"), geometry=d.geometry.centroid, crs=d.crs)
    z = zones.to_crs(d.crs).reset_index(drop=True)
    z["_zid"] = np.arange(len(z))
    j = gpd.sjoin(pts, z[["_zid", "geometry"]], predicate="within")
    counts = j.groupby("_zid").size()
    z["count"] = z["_zid"].map(counts).fillna(0).astype(int)
    return z.drop(columns="_zid")


# =============================================================================================
# 7. Instance segmentation and boundary delineation
# =============================================================================================


def clean_mask(mask: Any, *, min_size: int = 16, hole_size: int = 16, opening: int = 0) -> Any:
    """Remove small objects and fill small holes in a binary mask (optionally open first)."""
    from skimage import morphology

    is_da = isinstance(mask, xr.DataArray)
    m = np.asarray(mask.values if is_da else mask).astype(bool)
    if opening:
        m = morphology.binary_opening(m, morphology.disk(opening))
    if min_size:
        m = morphology.remove_small_objects(m, min_size=min_size)
    if hole_size:
        m = morphology.remove_small_holes(m, area_threshold=hole_size)
    return mask.copy(data=m.astype("uint8")) if is_da else m


def instances_from_semantic(class_map: Any, *, classes: Sequence[int] | None = None,
                            method: Literal["connected", "watershed"] = "connected",
                            boundary: Any = None, min_distance: int = 5, min_size: int = 8
                            ) -> tuple[Any, pd.DataFrame]:
    """Split a semantic mask into instances → ``(instance id raster, id → class table)``.

    ``connected`` labels connected components per class; ``watershed`` splits touching
    objects (e.g. adjacent fields or tree crowns) from distance-transform peaks, or from a
    predicted ``boundary`` probability raster when given.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    is_da = isinstance(class_map, xr.DataArray)
    a = np.asarray(class_map.values if is_da else class_map)
    classes = classes if classes is not None else [c for c in np.unique(a) if c > 0]
    inst = np.zeros(a.shape, "int32")
    rows, nxt = [], 1
    for c in classes:
        m = a == c
        if method == "connected":
            lab, n = ndi.label(m)
        else:
            if boundary is not None:
                elev = np.asarray(getattr(boundary, "values", boundary), float)
            else:
                elev = -ndi.distance_transform_edt(m)
            dist = ndi.distance_transform_edt(m)
            pk = peak_local_max(dist, min_distance=min_distance, labels=m.astype(int))
            markers = np.zeros(a.shape, "int32")
            markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
            lab = watershed(elev, markers, mask=m)
            n = int(lab.max())
        for k in range(1, n + 1):
            sel = lab == k
            if sel.sum() < min_size:
                continue
            inst[sel] = nxt
            rows.append({"instance": nxt, "class": int(c), "area_px": int(sel.sum())})
            nxt += 1
    table = pd.DataFrame(rows, columns=["instance", "class", "area_px"])
    out = class_map.copy(data=inst).rename("instance") if is_da else inst
    return out, table


def boundary_weight_map(labels: np.ndarray, *, w0: float = 10.0, sigma: float = 5.0,
                        class_balance: bool = True) -> np.ndarray:
    """U-Net style pixel weights emphasizing gaps between touching objects.

    ``w(x) = w_c(x) + w0 * exp(-(d1 + d2)^2 / (2 sigma^2))`` with ``d1``/``d2`` the distances
    to the nearest and second-nearest instance (Ronneberger et al. 2015). Use with
    ``torch.nn.CrossEntropyLoss(reduction="none")`` multiplied by the map (boundary-aware loss).
    """
    from scipy import ndimage as ndi

    lab = np.asarray(labels)
    ids = [i for i in np.unique(lab) if i > 0]
    w = np.ones(lab.shape, "float32")
    if class_balance:
        fg = lab > 0
        frac = fg.mean() if fg.any() else 0.5
        w = np.where(fg, 0.5 / max(frac, 1e-6), 0.5 / max(1 - frac, 1e-6)).astype("float32")
        w /= w.mean()
    if len(ids) >= 2:
        d = np.stack([ndi.distance_transform_edt(lab != i) for i in ids])
        d.sort(0)
        w = w + w0 * np.exp(-((d[0] + d[1]) ** 2) / (2 * sigma ** 2)) * (lab == 0)
    return w.astype("float32")


def mask_boundaries(labels: np.ndarray, *, width: int = 1) -> np.ndarray:
    """Binary boundary raster of a label/instance map (target for boundary-detection heads)."""
    from skimage.segmentation import find_boundaries

    b = find_boundaries(np.asarray(labels), mode="inner")
    if width > 1:
        from scipy import ndimage as ndi

        b = ndi.binary_dilation(b, iterations=width - 1)
    return b.astype("uint8")


def polygonize(label_map: xr.DataArray, *, class_map: xr.DataArray | None = None,
               confidence: xr.DataArray | None = None, class_names: Mapping[int, str] | None = None,
               min_area: float = 0.0, simplify: float | None = None,
               regularize: Literal["none", "rectangle", "convex", "auto"] = "none",
               ignore: Sequence[int] = (0, -1)):
    """Vectorize a label/instance raster → GeoDataFrame in the raster CRS (never reprojected).

    Attributes: ``id``, ``class`` (majority of ``class_map``, or the label value itself),
    ``class_name``, mean ``confidence``, ``area``, ``perimeter`` and ``compactness``
    (Polsby-Popper). Geometry is cleaned by dropping polygons below ``min_area`` (map units²),
    ``simplify`` (Douglas-Peucker tolerance, topology-preserving) and ``regularize``
    (``rectangle`` = minimum rotated rectangle, ``convex`` = convex hull, ``auto`` =
    rectangle only for already rectangular shapes, useful for buildings).
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    lab = np.asarray(label_map.values)
    lab = np.where(np.isfinite(lab), lab, -1).astype("int32") if lab.dtype.kind == "f" \
        else lab.astype("int32")
    valid = ~np.isin(lab, list(ignore))
    T = label_map.rio.transform()
    parts: dict[int, list] = {}
    for geom, v in shapes(lab, mask=valid, transform=T):
        parts.setdefault(int(v), []).append(shape(geom))
    cls_a = None if class_map is None else np.asarray(class_map.values)
    conf_a = None if confidence is None else np.asarray(confidence.values, float)
    rows, geoms = [], []
    for v, gl in parts.items():
        g = unary_union(gl) if len(gl) > 1 else gl[0]
        if g.area < min_area:
            continue
        sel = lab == v
        cls = v
        if cls_a is not None:
            vals = cls_a[sel]
            vals = vals[vals >= 0]
            cls = int(np.bincount(vals.astype(int)).argmax()) if vals.size else -1
        if regularize == "rectangle":
            g = g.minimum_rotated_rectangle
        elif regularize == "convex":
            g = g.convex_hull
        elif regularize == "auto":
            r = g.minimum_rotated_rectangle
            if r.area > 0 and g.area / r.area > 0.85:
                g = r
        if simplify:
            g = g.simplify(simplify, preserve_topology=True)
        rows.append({"id": v, "class": cls,
                     "class_name": (class_names or {}).get(cls, str(cls)),
                     "confidence": float(np.nanmean(conf_a[sel])) if conf_a is not None else np.nan,
                     "area": g.area, "perimeter": g.length,
                     "compactness": 4 * math.pi * g.area / g.length ** 2 if g.length else 0.0})
        geoms.append(g)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=label_map.rio.crs)


def instances_from_detector(outputs: Mapping[str, Any], image: xr.DataArray, *,
                            score_threshold: float = 0.5, mask_threshold: float = 0.5,
                            class_names: Mapping[int, str] | None = None,
                            simplify: float | None = None, row_offset: int = 0,
                            col_offset: int = 0):
    """Mask R-CNN output dict (``masks``, ``labels``, ``scores``) → georeferenced polygons."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    T = as_band_array(image).rio.transform()
    from rasterio.transform import Affine

    T = T * Affine.translation(col_offset, row_offset)
    masks = np.asarray(getattr(outputs["masks"], "cpu", lambda: outputs["masks"])())
    labels = np.asarray(getattr(outputs["labels"], "cpu", lambda: outputs["labels"])())
    scores = np.asarray(getattr(outputs["scores"], "cpu", lambda: outputs["scores"])())
    rows, geoms = [], []
    for m, lb, sc in zip(masks, labels, scores):
        if sc < score_threshold:
            continue
        b = (np.squeeze(m) >= mask_threshold).astype("uint8")
        polys = [shape(g) for g, v in shapes(b, mask=b.astype(bool), transform=T) if v == 1]
        if not polys:
            continue
        g = unary_union(polys)
        if simplify:
            g = g.simplify(simplify, preserve_topology=True)
        rows.append({"label": int(lb), "class_name": (class_names or {}).get(int(lb), str(lb)),
                     "confidence": float(sc), "area": g.area})
        geoms.append(g)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=image.rio.crs)


def export_vectors(gdf, path: str | PathLike, *, layer: str | None = None, to_crs: Any = None
                   ) -> Path:
    """Write vectors to GeoPackage (``.gpkg``), GeoJSON (``.geojson``/``.json``), Shapefile or
    GeoParquet (``.parquet``). The CRS is kept; reprojection happens only with ``to_crs``."""
    path = Path(path)
    g = gdf.to_crs(to_crs) if to_crs is not None else gdf
    if g.crs is None:
        warnings.warn("exporting vectors without a CRS", UserWarning, stacklevel=2)
    ext = path.suffix.lower()
    if ext == ".parquet":
        g.to_parquet(path)
    elif ext == ".gpkg":
        g.to_file(path, layer=layer or path.stem, driver="GPKG")
    elif ext in (".geojson", ".json"):
        g.to_file(path, driver="GeoJSON")
    elif ext == ".shp":
        g.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"unsupported vector format {ext!r}")
    return path


# =============================================================================================
# 8. Change detection
# =============================================================================================


def image_ratio(before: xr.DataArray, after: xr.DataArray, *, log: bool = True,
                k: float = 2.0, eps: float = 1e-6) -> xr.Dataset:
    """Ratio (or log-ratio, preferred for SAR) change image with ±``k``·σ change flags."""
    r = (after + eps) / (before + eps)
    if log:
        r = np.log(r)
    a = np.asarray(r.values, float)
    mu, sd = np.nanmean(a), np.nanstd(a)
    base = 0.0 if log else 1.0
    return xr.Dataset({"ratio": r, "increase": r > mu + k * sd, "decrease": r < mu - k * sd},
                      attrs={"mean": mu, "std": sd, "k": k, "log": log, "no_change": base})


def image_difference(before, after, *, k: float = 2.0) -> xr.Dataset:
    """Difference image with ±``k``·σ change flags (see :mod:`geolattice.rsense.classify`)."""
    from .change import image_difference as _d

    return _d(before, after, k=k)


def change_vector_analysis(before, after, *, bands: Sequence[str] | None = None,
                           threshold: float | None = None) -> xr.Dataset:
    """Change vector analysis: magnitude, direction and an Otsu (or fixed) change mask."""
    from .change import cva

    b, a = as_band_dataset(before), as_band_dataset(after)
    if threshold is None:  # Otsu, robust to constant magnitudes (no structure to separate)
        names = list(bands or b.data_vars)
        mag = np.sqrt(sum((np.asarray(a[v].values, float) - np.asarray(b[v].values, float)) ** 2
                          for v in names))
        m = mag[np.isfinite(mag)]
        if m.size == 0 or np.ptp(m) <= 1e-5 * float(np.abs(m).max()):
            threshold = float(m.max()) if m.size else 0.0
        else:
            from skimage.filters import threshold_otsu

            threshold = float(threshold_otsu(m))
    return cva(b, a, bands=bands, threshold=threshold)


def pca_change(before: xr.DataArray | xr.Dataset, after: xr.DataArray | xr.Dataset, *,
               method: Literal["stacked", "difference"] = "stacked",
               n_components: int | None = None,
               threshold: float | None = None) -> xr.Dataset:
    """PCA change detection.

    ``stacked``: PCA of the standardized two-date band stack; the first components capture
    stable variance, later ones capture change (inspect ``loadings``). ``difference``: PCA of
    the difference image; ``change`` thresholds |PC1| with Otsu (or ``threshold``).
    """
    from sklearn.decomposition import PCA

    b, a = as_band_array(before), as_band_array(after)
    B, A = np.asarray(b.values, float), np.asarray(a.values, float)
    C, H, W = B.shape
    stack = np.concatenate([B, A]) if method == "stacked" else A - B
    X = stack.reshape(stack.shape[0], -1).T
    ok = np.all(np.isfinite(X), 1)
    mu, sd = X[ok].mean(0), X[ok].std(0)
    sd[sd == 0] = 1
    pca = PCA(n_components=n_components or X.shape[1]).fit((X[ok] - mu) / sd)
    pcs = np.full((X.shape[0], pca.n_components_), np.nan)
    pcs[ok] = pca.transform((X[ok] - mu) / sd)
    tpl = b.isel(band=0, drop=True)
    out = xr.Dataset({f"pc{i + 1}": tpl.copy(data=pcs[:, i].reshape(H, W))
                      for i in range(pca.n_components_)})
    out.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    names = ([f"t1_{n}" for n in b["band"].values] + [f"t2_{n}" for n in a["band"].values]
             if method == "stacked" else [f"d_{n}" for n in b["band"].values])
    out.attrs["loadings"] = json.dumps(pd.DataFrame(pca.components_, columns=names).round(4)
                                       .to_dict(orient="list"))
    if method == "difference":
        from skimage.filters import threshold_otsu

        mag = np.abs(pcs[:, 0])
        thr = threshold if threshold is not None else float(threshold_otsu(mag[ok]))
        out["change"] = tpl.copy(data=(mag > thr).reshape(H, W))
        out.attrs["threshold"] = thr
    return out


def check_coregistration(reference: xr.DataArray, moving: xr.DataArray, *, band: Any = None,
                         upsample_factor: int = 10, tolerance: float = 0.5) -> dict[str, float]:
    """Estimate the sub-pixel shift between two dates (phase cross-correlation).

    Misregistration creates false change along edges; a shift above ``tolerance`` pixels
    raises a warning. Also checks that CRS and grid match.
    """
    from skimage.registration import phase_cross_correlation

    r, m = as_band_array(reference), as_band_array(moving)
    if r.rio.crs != m.rio.crs or r.shape[-2:] != m.shape[-2:]:
        raise ValueError("images are not on the same grid/CRS; resample one onto the other first")
    sel = (lambda d: d.sel(band=band)) if band is not None else (lambda d: d.isel(band=0))
    ra, ma = (np.nan_to_num(np.asarray(sel(d).values, float)) for d in (r, m))
    shift, err, _ = phase_cross_correlation(ra, ma, upsample_factor=upsample_factor)
    out = {"dy": float(shift[0]), "dx": float(shift[1]), "error": float(err),
           "magnitude": float(np.hypot(*shift))}
    if out["magnitude"] > tolerance:
        warnings.warn(f"images are misregistered by {out['magnitude']:.2f} px; co-register before "
                      "change detection", UserWarning, stacklevel=2)
    return out


@lru_cache(maxsize=None)
def _change_nets() -> SimpleNamespace:
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Encoder(nn.Module):
        def __init__(self, cin, widths):
            super().__init__()
            self.blocks = nn.ModuleList()
            c = cin
            for w in widths:
                self.blocks.append(block(c, w))
                c = w
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            feats = []
            for i, b in enumerate(self.blocks):
                x = b(x if i == 0 else self.pool(x))
                feats.append(x)
            return feats

    class Decoder(nn.Module):
        def __init__(self, widths, skip_mult, num_classes):
            super().__init__()
            self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
            rev = list(reversed(widths))
            c = rev[0] * skip_mult
            for w in rev[1:]:
                self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
                self.decs.append(block(w + w * skip_mult, w))
                c = w
            self.head = nn.Conv2d(c, num_classes, 1)

        def forward(self, feats):
            x = feats[-1]
            for up, dec, s in zip(self.ups, self.decs, reversed(feats[:-1])):
                x = up(x)
                if x.shape[-2:] != s.shape[-2:]:
                    x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
                x = dec(torch.cat([x, s], 1))
            return self.head(x)

    class ChangeUNet(nn.Module):
        """Change U-Net: ``early_fusion`` (FC-EF), ``siamese_diff`` (FC-Siam-diff) or
        ``siamese_concat`` (FC-Siam-conc) after Daudt et al. (2018). ``forward(t1, t2)``."""

        def __init__(self, in_channels=3, num_classes=2, widths=(16, 32, 64, 128),
                     mode="siamese_diff"):
            super().__init__()
            self.mode = mode
            cin = in_channels * 2 if mode == "early_fusion" else in_channels
            self.encoder = Encoder(cin, widths)
            self.decoder = Decoder(widths, 2 if mode == "siamese_concat" else 1, num_classes)

        def forward(self, t1, t2):
            if self.mode == "early_fusion":
                return self.decoder(self.encoder(torch.cat([t1, t2], 1)))
            f1, f2 = self.encoder(t1), self.encoder(t2)
            if self.mode == "siamese_diff":
                feats = [torch.abs(a - b) for a, b in zip(f1, f2)]
            else:
                feats = [torch.cat([a, b], 1) for a, b in zip(f1, f2)]
            return self.decoder(feats)

    class BITLite(nn.Module):
        """Bitemporal image transformer (after Chen et al. 2021, BIT), compact version.

        A shared CNN extracts features of both dates; spatial attention compresses each into
        ``n_tokens`` semantic tokens; a transformer encoder models the joint token context;
        a transformer decoder projects tokens back to pixels; the absolute feature difference
        feeds a small prediction head. ChangeFormer-style hierarchical transformers can be
        plugged in through ``build_change_model(custom=...)``.
        """

        def __init__(self, in_channels=3, num_classes=2, dim=32, n_tokens=4, depth=1, heads=4):
            super().__init__()
            self.backbone = nn.Sequential(block(in_channels, dim), nn.MaxPool2d(2), block(dim, dim))
            self.token_attn = nn.Conv2d(dim, n_tokens, 1)
            self.pos = nn.Parameter(torch.randn(1, 2 * n_tokens, dim) * 0.02)
            enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, depth)
            dec = nn.TransformerDecoderLayer(dim, heads, dim * 2, batch_first=True)
            self.decoder = nn.TransformerDecoder(dec, depth)
            self.head = nn.Sequential(block(dim, dim), nn.Conv2d(dim, num_classes, 1))
            self.n_tokens = n_tokens

        def _tokens(self, f):
            a = self.token_attn(f).flatten(2).softmax(-1)  # B, L, HW
            return torch.einsum("blp,bcp->blc", a, f.flatten(2))

        def _refine(self, f, tokens):
            B, C, H, W = f.shape
            q = f.flatten(2).transpose(1, 2)
            return self.decoder(q, tokens).transpose(1, 2).reshape(B, C, H, W)

        def forward(self, t1, t2):
            H, W = t1.shape[-2:]
            f1, f2 = self.backbone(t1), self.backbone(t2)
            tok = torch.cat([self._tokens(f1), self._tokens(f2)], 1) + self.pos
            tok = self.encoder(tok)
            t1k, t2k = tok[:, :self.n_tokens], tok[:, self.n_tokens:]
            d = torch.abs(self._refine(f1, t1k) - self._refine(f2, t2k))
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            return self.head(d)

    return SimpleNamespace(ChangeUNet=ChangeUNet, BITLite=BITLite)


def build_change_model(arch: Literal["siamese_diff", "siamese_concat", "early_fusion", "bit"]
                       = "siamese_diff", *, in_channels: int = 3, num_classes: int = 2,
                       custom: Any = None, **kwargs: Any):
    """Bitemporal change model with ``forward(t1, t2)`` → ``(N, K, H, W)`` logits.

    Siamese and early-fusion U-Nets (Daudt et al. 2018) or a compact BIT transformer. Train
    with :func:`train_model` using batches ``{"image": (t1, t2), "mask": change}``
    (``task="segmentation"``). ``custom`` returns any user model (e.g. ChangeFormer).
    """
    if custom is not None:
        return custom
    N = _change_nets()
    if arch == "bit":
        return N.BITLite(in_channels, num_classes, **kwargs)
    if arch not in ("siamese_diff", "siamese_concat", "early_fusion"):
        raise ValueError(f"unknown change model {arch!r}")
    return N.ChangeUNet(in_channels, num_classes, mode=arch, **kwargs)


class PairChipDataset(ChipDataset):
    """Bitemporal chips ``(N, C, H, W)`` × 2 with change masks, for change models."""

    def __init__(self, before: np.ndarray, after: np.ndarray, masks: np.ndarray | None = None, *,
                 normalizer: Normalizer | None = None, transform: Callable | None = None):
        super().__init__(before, masks, normalizer=normalizer)
        self.after = np.nan_to_num(np.asarray(after, "float32") if normalizer is None
                                   else normalizer.transform(np.asarray(after, "float32"), 1))
        self.pair_transform = transform

    def __getitem__(self, i: int) -> dict:
        torch = _torch()
        a = torch.from_numpy(self.images[i].copy())
        b = torch.from_numpy(self.after[i].copy())
        t = None if self.targets is None else torch.as_tensor(self.targets[i])
        if self.pair_transform is not None:
            x, t = self.pair_transform(torch.cat([a, b]), t)  # same geometric transform for both
            a, b = x[:a.shape[0]], x[a.shape[0]:]
        out = {"image": (a, b)}
        if t is not None:
            out["mask"] = t
        return out


def change_metrics(predicted: Any, reference: Any) -> dict[str, float]:
    """Binary change accuracy: precision, recall, F1, IoU, overall accuracy and kappa."""
    p = np.asarray(getattr(predicted, "values", predicted)).astype(bool).ravel()
    r = np.asarray(getattr(reference, "values", reference)).ravel()
    keep = np.isfinite(r.astype(float))
    p, r = p[keep], r[keep].astype(bool)
    tp, fp = float(np.sum(p & r)), float(np.sum(p & ~r))
    fn, tn = float(np.sum(~p & r)), float(np.sum(~p & ~r))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    oa = (tp + tn) / n if n else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else 0.0
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "overall_accuracy": oa,
            "kappa": (oa - pe) / (1 - pe) if pe < 1 else 1.0}


# =============================================================================================
# 9. Spatio-temporal and multi-sensor learning
# =============================================================================================


def pixel_time_series(cube: xr.DataArray | xr.Dataset, samples, label_col: str | None = None, *,
                      time_dim: str = "time") -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Satellite image time series at points → ``(X (N, T, C), labels, dates)``.

    ``cube`` is a ``(time, band, y, x)`` DataArray or a Dataset of ``(time, y, x)`` bands.
    Points are reprojected to the cube CRS and sampled at the nearest pixel.
    """
    da = cube.to_array("band") if isinstance(cube, xr.Dataset) else cube
    da = da.transpose(time_dim, "band", ...)
    s = samples.to_crs(da.rio.crs) if samples.crs and da.rio.crs else samples
    ydim, xdim = da.dims[-2:]
    xs = xr.DataArray(s.geometry.x.values, dims="s")
    ys = xr.DataArray(s.geometry.y.values, dims="s")
    v = np.asarray(da.sel({xdim: xs, ydim: ys}, method="nearest").values, "float32")  # T, C, N
    X = np.moveaxis(v, -1, 0)
    lab = s[label_col].to_numpy() if label_col else None
    return X, lab, np.asarray(da[time_dim].values)


@lru_cache(maxsize=None)
def _temporal_nets() -> SimpleNamespace:
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    def positional(T, d, positions=None, device=None):
        pos = (torch.arange(T, device=device).float() if positions is None
               else positions.float())
        i = torch.arange(d // 2, device=device).float()
        ang = pos[..., None] / (1000 ** (2 * i / d))
        pe = torch.zeros(*ang.shape[:-1], d, device=device)
        pe[..., 0::2] = torch.sin(ang)
        pe[..., 1::2] = torch.cos(ang)
        return pe

    class TempCNN(nn.Module):
        """TempCNN (Pelletier et al. 2019): 1-D convolutions over time. Input ``(N, T, C)``."""

        def __init__(self, in_channels, num_classes, n_timesteps, hidden=64, kernel=5, dropout=0.2):
            super().__init__()
            layers, c = [], in_channels
            for _ in range(3):
                layers += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2),
                           nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
                c = hidden
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hidden * n_timesteps, 4 * hidden),
                                    nn.BatchNorm1d(4 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(4 * hidden, num_classes))

        def forward(self, x):
            return self.fc(self.conv(x.transpose(1, 2)))

    class LSTMClassifier(nn.Module):
        """(Bi)LSTM over the time axis; the final hidden state feeds the classifier."""

        def __init__(self, in_channels, num_classes, hidden=64, layers=2, bidirectional=True,
                     dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(in_channels, hidden, layers, batch_first=True,
                                bidirectional=bidirectional, dropout=dropout if layers > 1 else 0)
            self.fc = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    class LTAE(nn.Module):
        """Lightweight temporal attention encoder (after Garnot & Landrieu 2020).

        Learned master queries attend over time steps (with sinusoidal day-of-year or index
        encoding). Input ``(N, T, C)``; optional ``positions`` ``(N, T)`` (e.g. day of year).
        """

        def __init__(self, in_channels, num_classes, d_model=64, n_heads=4, dropout=0.1,
                     return_attention=False):
            super().__init__()
            self.inp = nn.Linear(in_channels, d_model)
            self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(d_model, num_classes))
            self.d, self.return_attention = d_model, return_attention

        def encode(self, x, positions=None):
            h = self.inp(x) + positional(x.shape[1], self.d, positions, x.device)
            q = self.query.expand(x.shape[0], -1, -1)
            z, w = self.attn(q, h, h)
            return self.norm(z[:, 0]), w[:, 0]

        def forward(self, x, positions=None):
            z, w = self.encode(x, positions)
            out = self.mlp(z)
            return (out, w) if self.return_attention else out

    class PSETAE(nn.Module):
        """Pixel-set encoder + temporal attention (PSE-TAE, Garnot et al. 2020).

        Input ``(N, T, C, S)``: ``S`` pixels sampled inside each parcel at every date. A shared
        MLP embeds pixels, mean/std pooling summarizes the set, the L-TAE models time.
        """

        def __init__(self, in_channels, num_classes, d_pixel=32, d_model=64, n_heads=4):
            super().__init__()
            self.pixel = nn.Sequential(nn.Linear(in_channels, d_pixel), nn.ReLU(),
                                       nn.Linear(d_pixel, d_pixel))
            self.tae = LTAE(2 * d_pixel, num_classes, d_model, n_heads)

        def forward(self, x, positions=None):
            e = self.pixel(x.transpose(2, 3))  # N, T, S, d
            z = torch.cat([e.mean(2), e.std(2, unbiased=False)], -1)
            return self.tae(z, positions)

    class ConvLSTMCell(nn.Module):
        def __init__(self, cin, hidden, k=3):
            super().__init__()
            self.conv = nn.Conv2d(cin + hidden, 4 * hidden, k, padding=k // 2)
            self.hidden = hidden

        def forward(self, x, state):
            h, c = state
            i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class ConvLSTMSegmenter(nn.Module):
        """ConvLSTM (Shi et al. 2015) over an image time series ``(N, T, C, H, W)`` → class map."""

        def __init__(self, in_channels, num_classes, hidden=32, layers=1):
            super().__init__()
            self.cells = nn.ModuleList([ConvLSTMCell(in_channels if i == 0 else hidden, hidden)
                                        for i in range(layers)])
            self.head = nn.Conv2d(hidden, num_classes, 1)

        def forward(self, x):
            N, T, _, H, W = x.shape
            seq = x
            for cell in self.cells:
                h = torch.zeros(N, cell.hidden, H, W, device=x.device)
                c = torch.zeros_like(h)
                outs = []
                for t in range(T):
                    h, c = cell(seq[:, t], (h, c))
                    outs.append(h)
                seq = torch.stack(outs, 1)
            return self.head(seq[:, -1])

    def cblock(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(4, cout),
                             nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                             nn.GroupNorm(4, cout), nn.ReLU(inplace=True))

    class UTAE(nn.Module):
        """U-TAE style network (after Garnot & Landrieu 2021) for ``(N, T, C, H, W)`` series.

        A shared 2-D encoder runs on every date; temporal attention at the coarsest level
        produces per-pixel weights over dates, which are upsampled to aggregate the skip
        features at every resolution; a U-Net decoder outputs ``(N, K, H, W)`` logits.
        """

        def __init__(self, in_channels, num_classes, widths=(32, 64, 128), n_heads=4):
            super().__init__()
            self.enc = nn.ModuleList()
            c = in_channels
            for i, w in enumerate(widths):
                self.enc.append(cblock(c, w, 1 if i == 0 else 2))
                c = w
            self.inp = nn.Linear(widths[-1], widths[-1])
            self.query = nn.Parameter(torch.randn(1, 1, widths[-1]) * 0.02)
            self.attn = nn.MultiheadAttention(widths[-1], n_heads, batch_first=True)
            self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
            rev = list(reversed(widths))
            for a, b in zip(rev[:-1], rev[1:]):
                self.ups.append(nn.ConvTranspose2d(a, b, 2, 2))
                self.decs.append(cblock(2 * b, b))
            self.head = nn.Conv2d(widths[0], num_classes, 1)
            self.d = widths[-1]

        def forward(self, x, positions=None):
            N, T, C, H, W = x.shape
            feats = []
            h = x.reshape(N * T, C, H, W)
            for e in self.enc:
                h = e(h)
                feats.append(h.reshape(N, T, *h.shape[1:]))
            top = feats[-1]
            _, _, D, h0, w0 = top.shape
            seq = top.permute(0, 3, 4, 1, 2).reshape(N * h0 * w0, T, D)
            seq = self.inp(seq) + positional(T, D, None if positions is None else
                                             positions.repeat_interleave(h0 * w0, 0), x.device)
            q = self.query.expand(seq.shape[0], -1, -1)
            _, wts = self.attn(q, seq, seq)  # (N*h0*w0, 1, T)
            wts = wts[:, 0].reshape(N, h0, w0, T).permute(0, 3, 1, 2)  # N, T, h0, w0
            agg = []
            for f in feats:
                wt = F.interpolate(wts, size=f.shape[-2:], mode="bilinear", align_corners=False)
                wt = wt / wt.sum(1, keepdim=True).clamp_min(1e-6)
                agg.append((f * wt[:, :, None]).sum(1))
            y = agg[-1]
            for up, dec, s in zip(self.ups, self.decs, reversed(agg[:-1])):
                y = up(y)
                if y.shape[-2:] != s.shape[-2:]:
                    y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
                y = dec(torch.cat([y, s], 1))
            return self.head(y)

    class MiddleFusionNet(nn.Module):
        """Two-branch chip classifier fusing optical and SAR features mid-network.

        ``forward(optical, sar)``; each branch is a small CNN, their pooled features are
        concatenated (middle fusion) before the classifier.
        """

        def __init__(self, optical_channels, sar_channels, num_classes, width=32):
            super().__init__()

            def branch(c):
                return nn.Sequential(cblock(c, width), cblock(width, 2 * width, 2),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())

            self.opt, self.sar = branch(optical_channels), branch(sar_channels)
            self.head = nn.Sequential(nn.Linear(4 * width, 2 * width), nn.ReLU(),
                                      nn.Linear(2 * width, num_classes))

        def forward(self, optical, sar):
            return self.head(torch.cat([self.opt(optical), self.sar(sar)], 1))

    return SimpleNamespace(TempCNN=TempCNN, LSTMClassifier=LSTMClassifier, LTAE=LTAE,
                           PSETAE=PSETAE, ConvLSTMSegmenter=ConvLSTMSegmenter, UTAE=UTAE,
                           MiddleFusionNet=MiddleFusionNet)


def build_temporal_model(arch: Literal["tempcnn", "lstm", "ltae", "psetae", "convlstm", "utae"]
                         = "tempcnn", *, in_channels: int, num_classes: int,
                         n_timesteps: int | None = None, **kwargs: Any):
    """Satellite image time-series model.

    Pixel/parcel classifiers take ``(N, T, C)`` (``tempcnn`` needs ``n_timesteps``,
    ``lstm``, ``ltae``) or ``(N, T, C, S)`` pixel sets (``psetae``); dense models take
    ``(N, T, C, H, W)`` and return maps (``convlstm``, ``utae``; PASTIS-style crop mapping).
    """
    N = _temporal_nets()
    a = arch.lower()
    if a == "tempcnn":
        if not n_timesteps:
            raise ValueError("tempcnn needs n_timesteps")
        return N.TempCNN(in_channels, num_classes, n_timesteps, **kwargs)
    table = {"lstm": N.LSTMClassifier, "ltae": N.LTAE, "psetae": N.PSETAE,
             "convlstm": N.ConvLSTMSegmenter, "utae": N.UTAE}
    if a not in table:
        raise ValueError(f"unknown temporal model {arch!r}")
    return table[a](in_channels, num_classes, **kwargs)


def build_fusion_model(optical_channels: int, sar_channels: int, num_classes: int, **kw: Any):
    """Middle-fusion optical + SAR chip classifier (``forward(optical, sar)``)."""
    return _temporal_nets().MiddleFusionNet(optical_channels, sar_channels, num_classes, **kw)


def fuse_early(optical: xr.DataArray | xr.Dataset, sar: xr.DataArray | xr.Dataset, *,
               align: bool = False, prefixes: tuple[str, str] = ("opt_", "sar_")) -> xr.DataArray:
    """Early fusion: stack optical and SAR bands into one ``(band, y, x)`` input.

    Inputs must share the grid unless ``align=True`` (SAR resampled to the optical grid,
    recorded in ``attrs["aligned"]``).
    """
    o, s = as_band_array(optical), as_band_array(sar)
    if (o.shape[-2:] != s.shape[-2:] or o.rio.crs != s.rio.crs or
            not np.allclose(tuple(o.rio.transform()), tuple(s.rio.transform()))):
        if not align:
            raise ValueError("optical and SAR grids differ; pass align=True to resample SAR")
        s = s.rio.reproject_match(o.isel(band=0))
        s = s.assign_coords({d: o[d] for d in o.dims[-2:]})
    o = o.assign_coords(band=[f"{prefixes[0]}{b}" for b in o["band"].values])
    s = s.assign_coords(band=[f"{prefixes[1]}{b}" for b in s["band"].values])
    out = xr.concat([o, s], "band").rio.write_crs(o.rio.crs)
    out.attrs = {"fusion": "early", "aligned": bool(align)}
    return out


def fuse_late(probabilities: Sequence[np.ndarray | xr.Dataset], *,
              weights: Sequence[float] | None = None) -> np.ndarray | xr.Dataset:
    """Late fusion: weighted average of per-sensor class probabilities."""
    w = np.asarray(weights if weights is not None else [1.0] * len(probabilities), float)
    w = w / w.sum()
    if isinstance(probabilities[0], xr.Dataset):
        out = sum(wi * p for wi, p in zip(w, probabilities))
        out.attrs["fusion"] = "late"
        return out
    return sum(wi * np.asarray(p) for wi, p in zip(w, probabilities))


def sar_assisted_gap_fill(optical: xr.Dataset | xr.DataArray, sar: xr.Dataset | xr.DataArray, *,
                          model: str = "rf", seed: int = 0, **kw: Any) -> tuple[xr.Dataset, dict]:
    """Fill cloudy (NaN) optical pixels from SAR, flagging every reconstructed value.

    The output has a boolean ``reconstructed`` variable; reconstructed reflectances are
    predictions, not observations, so exclude them from validation and report them.
    Returns ``(filled Dataset, per-band holdout R²)``.
    """
    from ..fusion.optical_sar import sar_gap_fill

    o, s = as_band_dataset(optical), as_band_dataset(sar)
    before = np.any(np.stack([~np.isfinite(np.asarray(o[v].values)) for v in o.data_vars]), 0)
    filled, scores = sar_gap_fill(o, s, model=model, seed=seed, **kw)
    after = np.any(np.stack([~np.isfinite(np.asarray(filled[v].values))
                             for v in filled.data_vars]), 0)
    tpl = o[list(o.data_vars)[0]]
    filled["reconstructed"] = tpl.copy(data=before & ~after).astype(bool)
    filled.attrs["note"] = "values where reconstructed=True are SAR-based predictions"
    return filled, scores


def pansharpen(multispectral: xr.DataArray | np.ndarray, panchromatic: xr.DataArray | np.ndarray,
               *, method: Literal["highpass", "intensity", "brovey"] = "highpass", **kw: Any):
    """Pansharpen (high-pass, intensity substitution, Brovey) at the pan resolution.

    Inputs must already be on the pan grid (upsample MS first, e.g. with
    :func:`geolattice.rsense.super_resolution.upsample`). Validate with
    :func:`pansharpening_quality` at reduced resolution (Wald protocol).
    """
    from ..super_resolution import pansharpen as _ps

    ms = np.asarray(getattr(multispectral, "values", multispectral), "float32")
    pan = np.asarray(getattr(panchromatic, "values", panchromatic), "float32")
    out = _ps(ms, pan, method=method, **kw)
    if isinstance(panchromatic, xr.DataArray):
        bands = (multispectral["band"].values if isinstance(multispectral, xr.DataArray)
                 and "band" in multispectral.dims else np.arange(out.shape[0]))
        tpl = panchromatic.squeeze()
        da = xr.DataArray(out, dims=("band", *tpl.dims[-2:]),
                          coords={"band": bands, **{d: tpl[d] for d in tpl.dims[-2:]}})
        return da.rio.write_crs(panchromatic.rio.crs)
    return out


def pansharpening_quality(fused: Any, reference: Any, *, ratio: float = 4.0) -> dict[str, float]:
    """Spectral quality vs. a reference: RMSE, ERGAS, SAM (degrees) and mean correlation.

    Generated detail can be hallucinated (especially from learned super-resolution), so
    compare against true high-resolution references, not the inputs.
    """
    f = np.asarray(getattr(fused, "values", fused), float)
    r = np.asarray(getattr(reference, "values", reference), float)
    m = np.all(np.isfinite(f), 0) & np.all(np.isfinite(r), 0)
    F_, R_ = f[:, m], r[:, m]
    rmse_b = np.sqrt(((F_ - R_) ** 2).mean(1))
    ergas = 100 / ratio * np.sqrt(np.mean((rmse_b / np.maximum(np.abs(R_.mean(1)), 1e-9)) ** 2))
    dot = (F_ * R_).sum(0)
    nrm = np.linalg.norm(F_, axis=0) * np.linalg.norm(R_, axis=0)
    sam = np.degrees(np.arccos(np.clip(dot / np.maximum(nrm, 1e-12), -1, 1))).mean()
    cc = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(F_, R_)])
    return {"rmse": float(np.sqrt(((F_ - R_) ** 2).mean())), "ergas": float(ergas),
            "sam_deg": float(sam), "cc": float(cc)}


# =============================================================================================
# 10. Self-supervised pretraining
# =============================================================================================


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    """SimCLR NT-Xent contrastive loss for two views ``(N, D)`` of the same batch."""
    torch = _torch()
    F = torch.nn.functional
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def make_encoder(arch: str = "resnet18", *, in_channels: int = 3, pretrained: bool = False,
                 **kw: Any):
    """Backbone without its head (``forward`` → ``(N, D)`` features) for SSL / embeddings."""
    model = build_cnn(arch, in_channels=in_channels, num_classes=0, pretrained=pretrained, **kw)
    return model


def _feat_dim(encoder, in_channels: int, size: int = 64) -> int:
    torch = _torch()
    dev = next(encoder.parameters()).device
    was = encoder.training
    encoder.eval()
    with torch.no_grad():
        d = encoder(torch.zeros(2, in_channels, size, size, device=dev)).flatten(1).shape[1]
    encoder.train(was)
    return int(d)


def pretrain_contrastive(encoder, loader, *, in_channels: int,
                         method: Literal["simclr", "moco"] = "simclr", epochs: int = 10,
                         lr: float = 1e-3, temperature: float = 0.2, proj_dim: int = 64,
                         queue_size: int = 4096, momentum: float = 0.99,
                         augment: Callable | None = None, device: str | None = None,
                         seed: int = 0, verbose: bool = True) -> tuple[Any, dict]:
    """Contrastive pretraining of ``encoder`` on unlabeled chips (SimCLR or MoCo v2-style).

    Each batch is either images (two random :class:`OverheadAugment` views are made) or an
    explicit pair ``(view1, view2)``, which gives SeCo-style seasonal contrast when the two
    views are the same location on different dates. Returns ``(encoder, history)``.
    """
    import copy

    torch = _torch()
    nn = torch.nn
    set_seed(seed)
    dev = _device(device)
    encoder.to(dev)
    d = _feat_dim(encoder, in_channels)
    proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, proj_dim)).to(dev)
    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    aug = augment or OverheadAugment(brightness=0.1, contrast=0.2, noise=0.02, crop=0.8, seed=seed)
    if method == "moco":
        k_enc, k_proj = copy.deepcopy(encoder), copy.deepcopy(proj)
        for p in list(k_enc.parameters()) + list(k_proj.parameters()):
            p.requires_grad = False
        queue = nn.functional.normalize(torch.randn(queue_size, proj_dim, device=dev), dim=1)
        qptr = 0
    hist = {"loss": []}
    for ep in range(epochs):
        encoder.train()
        tot, n = 0.0, 0
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2 and hasattr(batch[0], "shape") \
                    and batch[0].ndim == 4:
                v1, v2 = batch[0].to(dev).float(), batch[1].to(dev).float()
            else:
                x = batch["image"] if isinstance(batch, Mapping) else batch
                x = x.float()
                v1 = torch.stack([aug(xi, None)[0] for xi in x]).to(dev)
                v2 = torch.stack([aug(xi, None)[0] for xi in x]).to(dev)
            z1 = proj(encoder(v1).flatten(1))
            if method == "simclr":
                loss = nt_xent_loss(z1, proj(encoder(v2).flatten(1)), temperature)
            else:
                q = nn.functional.normalize(z1, dim=1)
                with torch.no_grad():
                    for pq, pk in zip(list(encoder.parameters()) + list(proj.parameters()),
                                      list(k_enc.parameters()) + list(k_proj.parameters())):
                        pk.data.mul_(momentum).add_(pq.data, alpha=1 - momentum)
                    k = nn.functional.normalize(k_proj(k_enc(v2).flatten(1)), dim=1)
                logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.clone().t()], 1)
                logits = logits / temperature
                loss = nn.functional.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long,
                                                                       device=dev))
                with torch.no_grad():
                    bs = k.shape[0]
                    idx = (torch.arange(bs, device=dev) + qptr) % queue_size
                    queue[idx] = k
                    qptr = (qptr + bs) % queue_size
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * v1.shape[0]
            n += v1.shape[0]
        hist["loss"].append(tot / max(n, 1))
        if verbose:
            print(f"ssl epoch {ep + 1}/{epochs} loss={hist['loss'][-1]:.4f}")
    return encoder, hist


@lru_cache(maxsize=None)
def _mae_nets() -> SimpleNamespace:
    torch = _torch()
    nn = torch.nn

    class MaskedAutoencoder(nn.Module):
        """Convolutional masked-image-modeling model (MAE / SatMAE-style objective).

        Random ``patch``×``patch`` patches are masked (``mask_ratio``); the network
        reconstructs the image and the loss is computed on masked patches only. ``encoder``
        plus global pooling gives features for linear probing. Use Scale-MAE / SatMAE
        checkpoints through :func:`load_foundation_encoder` for ViT-based variants.
        """

        def __init__(self, in_channels, width=32, patch=8, mask_ratio=0.6):
            super().__init__()

            def cb(a, b, s=1):
                return nn.Sequential(nn.Conv2d(a, b, 3, s, 1), nn.BatchNorm2d(b), nn.ReLU(True))

            self.encoder = nn.Sequential(cb(in_channels, width), cb(width, 2 * width, 2),
                                         cb(2 * width, 4 * width, 2))
            self.decoder = nn.Sequential(nn.ConvTranspose2d(4 * width, 2 * width, 2, 2),
                                         nn.ReLU(True), nn.ConvTranspose2d(2 * width, width, 2, 2),
                                         nn.ReLU(True), nn.Conv2d(width, in_channels, 3, 1, 1))
            self.patch, self.mask_ratio = patch, mask_ratio
            self.pool = nn.AdaptiveAvgPool2d(1)

        def random_mask(self, x):
            N, _, H, W = x.shape
            gh, gw = max(1, H // self.patch), max(1, W // self.patch)
            m = (torch.rand(N, 1, gh, gw, device=x.device) < self.mask_ratio).float()
            return nn.functional.interpolate(m, size=(H, W), mode="nearest")

        def forward(self, x):
            m = self.random_mask(x)
            rec = self.decoder(self.encoder(x * (1 - m)))
            rec = rec[..., :x.shape[-2], :x.shape[-1]]
            loss = (((rec - x) ** 2) * m).sum() / (m.sum() * x.shape[1]).clamp_min(1)
            return loss, rec, m

        def features(self, x):
            return self.pool(self.encoder(x)).flatten(1)

    return SimpleNamespace(MaskedAutoencoder=MaskedAutoencoder)


def build_mae(in_channels: int, **kw: Any):
    """Convolutional masked autoencoder for masked-image-modeling pretraining."""
    return _mae_nets().MaskedAutoencoder(in_channels, **kw)


def pretrain_masked(model, loader, *, epochs: int = 10, lr: float = 1e-3,
                    device: str | None = None, seed: int = 0, verbose: bool = True
                    ) -> tuple[Any, dict]:
    """Masked-image-modeling pretraining of a :func:`build_mae` model on unlabeled chips."""
    torch = _torch()
    set_seed(seed)
    dev = _device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = {"loss": []}
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in loader:
            x = (batch["image"] if isinstance(batch, Mapping) else batch).to(dev).float()
            loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        hist["loss"].append(tot / max(n, 1))
        if verbose:
            print(f"mae epoch {ep + 1}/{epochs} loss={hist['loss'][-1]:.4f}")
    return model, hist


def extract_embeddings(encoder, data: Any, *, batch_size: int = 128, device: str | None = None
                       ) -> np.ndarray:
    """Encoder features ``(N, D)`` for chips (array, dataset or loader).

    Uses ``encoder.features`` when present (masked autoencoders), else ``encoder(x)``
    flattened; spatial outputs are global-average pooled.
    """
    torch = _torch()
    dev = _device(device)
    encoder.to(dev).eval()
    if isinstance(data, np.ndarray):
        data = ChipDataset(data)
    loader = data if hasattr(data, "batch_size") else as_loader(data, batch_size, shuffle=False)
    fn = getattr(encoder, "features", None)
    fn = fn if callable(fn) else encoder
    out = []
    with torch.no_grad():
        for batch in loader:
            x, _ = _unpack(batch, dev)
            f = fn(x)
            if isinstance(f, (list, tuple)):
                f = f[-1]
            if f.ndim == 4:
                f = f.mean((-2, -1))
            elif f.ndim == 3:  # tokens
                f = f.mean(1)
            out.append(f.flatten(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def linear_probe(train_embeddings: np.ndarray, y_train: np.ndarray, test_embeddings: np.ndarray,
                 y_test: np.ndarray | None = None, *, C: float = 1.0, max_iter: int = 2000
                 ) -> dict[str, Any]:
    """Logistic-regression probe on frozen embeddings (standard SSL evaluation)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    clf.fit(train_embeddings, y_train)
    pred = clf.predict(test_embeddings)
    out = {"model": clf, "predictions": pred}
    if y_test is not None:
        out["report"] = classification_report(y_test, pred)
    return out


def geographic_holdout(coords: np.ndarray, test_bounds: Sequence[float]) -> Split:
    """Hold out one region (``xmin, ymin, xmax, ymax``) as test.

    Measures transfer to unseen areas.
    """
    c = np.asarray(coords, float)
    x0, y0, x1, y1 = test_bounds
    m = (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
    idx = np.arange(len(c))
    return Split(train=idx[~m], test=idx[m], kind="geographic-holdout").check()


def check_geographic_overlap(train_coords: np.ndarray, test_coords: np.ndarray, *,
                             min_distance: float, raise_error: bool = False) -> dict[str, float]:
    """Distance from each test sample to the nearest training/pretraining sample.

    Reports the share of test samples closer than ``min_distance`` (map units). Use it to
    keep pretraining chips away from evaluation regions when measuring generalization.
    """
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(train_coords, float)).query(np.asarray(test_coords, float))
    frac = float(np.mean(d < min_distance)) if d.size else 0.0
    out = {"min_distance": float(d.min()) if d.size else np.inf,
           "median_distance": float(np.median(d)) if d.size else np.inf,
           "fraction_within": frac}
    if frac > 0 and raise_error:
        raise ValueError(f"{frac:.1%} of test samples lie within {min_distance} of training data")
    return out


# =============================================================================================
# 11. Geospatial foundation models
# =============================================================================================

#: Survey of geospatial foundation models. Licenses and preprocessing change between
#: releases: always confirm them on the model card before use.
FOUNDATION_MODELS: dict[str, dict[str, Any]] = {
    "prithvi-eo-2.0": {
        "provider": "NASA / IBM", "architecture": "ViT MAE (3-D patch embedding)",
        "inputs": "HLS: blue, green, red, nir08, swir16, swir22", "temporal": True,
        "hub": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M (also -600M, -TL variants)",
        "loader": "terratorch (BACKBONE_REGISTRY 'prithvi_eo_v2_300')",
        "license": "check model card", "preprocessing": "HLS reflectance, dataset mean/std"},
    "clay-v1.5": {
        "provider": "Clay Foundation", "architecture": "ViT MAE, wavelength-aware patch embed",
        "inputs": "any sensor (band wavelengths supplied)", "temporal": "via date metadata",
        "hub": "made-with-clay/Clay", "loader": "claymodel / terratorch",
        "license": "check model card", "preprocessing": "per-sensor normalization from metadata"},
    "satlaspretrain": {
        "provider": "Allen Institute for AI", "architecture": "Swin-v2 + FPN",
        "inputs": "Sentinel-2 RGB or 9-band, NAIP, Landsat, Sentinel-1", "temporal": "multi-image",
        "hub": "allenai/satlas-pretrain", "loader": "torchgeo Swin_V2_B_Weights / satlaspretrain",
        "license": "check model card", "preprocessing": "reflectance / 3000 clipped to [0, 1]"},
    "dofa": {
        "provider": "TUM / Zhu lab", "architecture": "ViT with dynamic wavelength-conditioned stem",
        "inputs": "any band set (wavelengths in µm)", "temporal": False,
        "hub": "earthflow/DOFA", "loader": "torchgeo dofa_base_patch16_224",
        "license": "check model card", "preprocessing": "per-band standardization"},
    "skysense": {
        "provider": "Ant Group / Wuhan University",
        "architecture": "multimodal factorized spatiotemporal encoder",
        "inputs": "high-res optical, Sentinel-2 time series, Sentinel-1", "temporal": True,
        "hub": "weights released on request", "loader": "authors' code",
        "license": "check model card", "preprocessing": "per-modality"},
    "ssl4eo-s12": {
        "provider": "TUM", "architecture": "ResNet / ViT (MoCo, DINO, MAE, data2vec)",
        "inputs": "Sentinel-2 13 bands, Sentinel-1 VV/VH", "temporal": "seasonal views",
        "hub": "torchgeo weights", "loader": "torchgeo / load_rs_backbone",
        "license": "check model card", "preprocessing": "reflectance / 10000"},
    "satmae": {
        "provider": "Stanford", "architecture": "ViT MAE with temporal/spectral encoding",
        "inputs": "fMoW RGB / Sentinel-2 10 bands", "temporal": True,
        "hub": "authors' checkpoints", "loader": "geolattice.rsense.dl.foundation.SatMAE",
        "license": "check model card", "preprocessing": "dataset mean/std"},
    "scale-mae": {
        "provider": "UC Berkeley", "architecture": "ViT MAE with GSD-aware positional encoding",
        "inputs": "RGB", "temporal": False, "hub": "authors' checkpoints",
        "loader": "timm-compatible ViT state dict", "license": "check model card",
        "preprocessing": "ImageNet mean/std"},
}


def foundation_model_info(name: str | None = None) -> pd.DataFrame | dict[str, Any]:
    """Table of geospatial foundation models (bands, temporal support, loader, license note)."""
    if name is not None:
        return FOUNDATION_MODELS[name]
    return pd.DataFrame(FOUNDATION_MODELS).T


def load_foundation_encoder(name: str, **kwargs: Any):
    """Load a foundation-model encoder.

    ``prithvi*`` / ``clay*`` / ``satmae`` use :mod:`geolattice.rsense.dl.foundation` adapters;
    ``dofa`` and ``satlas*`` / ``ssl4eo*`` use TorchGeo; other TerraTorch backbones
    are loaded by registry name (``terratorch:<name>``).
    """
    n = name.lower()
    from ..dl import foundation as fm

    if n.startswith("prithvi"):
        return fm.Prithvi(**kwargs)
    if n.startswith("clay"):
        return fm.Clay(**kwargs)
    if n.startswith("satmae"):
        return fm.SatMAE(**kwargs)
    if n.startswith("dofa"):
        try:
            from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224
        except ImportError as e:
            raise ImportError("DOFA needs `pip install torchgeo`") from e
        return dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE, **kwargs)
    if n in RS_WEIGHTS or n.startswith("torchgeo:"):
        return load_rs_backbone(n, **kwargs)
    if n.startswith("terratorch:"):
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except ImportError as e:
            raise ImportError("needs `pip install terratorch`") from e
        return BACKBONE_REGISTRY.build(n.split(":", 1)[1], **kwargs)
    raise KeyError(f"unknown foundation model {name!r}; see foundation_model_info()")


def terratorch_config(task: Literal["segmentation", "classification", "regression"], *,
                      backbone: str = "prithvi_eo_v2_300", num_classes: int = 2,
                      bands: Sequence[str] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1",
                                              "SWIR_2"),
                      num_frames: int = 1, decoder: str = "UperNetDecoder", lr: float = 1e-4,
                      max_epochs: int = 50, freeze_backbone: bool = False) -> dict[str, Any]:
    """TerraTorch Lightning-CLI config (as a dict; dump with ``yaml.safe_dump``) for fine-tuning
    a foundation model on flood, burn-scar or crop tasks (``terratorch fit -c config.yaml``)."""
    task_cls = {"segmentation": "terratorch.tasks.SemanticSegmentationTask",
                "classification": "terratorch.tasks.ClassificationTask",
                "regression": "terratorch.tasks.PixelwiseRegressionTask"}[task]
    model_args = {"backbone": backbone, "backbone_pretrained": True, "backbone_bands": list(bands),
                  "backbone_num_frames": num_frames, "decoder": decoder}
    if task != "regression":
        model_args["num_classes"] = num_classes
    return {"trainer": {"max_epochs": max_epochs, "precision": "bf16-mixed", "accelerator": "auto"},
            "model": {"class_path": task_cls,
                      "init_args": {"model_factory": "EncoderDecoderFactory",
                                    "model_args": model_args, "freeze_backbone": freeze_backbone,
                                    "loss": "ce" if task != "regression" else "rmse"}},
            "optimizer": {"class_path": "torch.optim.AdamW", "init_args": {"lr": lr}}}


def embedding_search(index: np.ndarray, query: np.ndarray, *, k: int = 10,
                     metric: Literal["cosine", "euclidean"] = "cosine"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbors of ``query`` embeddings in ``index`` → ``(indices, scores)`` (k each).

    Cosine scores are similarities (higher = closer); euclidean scores are distances.
    """
    X = np.asarray(index, "float32")
    Q = np.atleast_2d(np.asarray(query, "float32"))
    k = min(k, len(X))
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        s = Qn @ Xn.T
        idx = np.argsort(-s, 1)[:, :k]
        return idx, np.take_along_axis(s, idx, 1)
    d = ((Q[:, None] - X[None]) ** 2).sum(-1) ** 0.5
    idx = np.argsort(d, 1)[:, :k]
    return idx, np.take_along_axis(d, idx, 1)


def few_shot_prototypes(support: np.ndarray, support_labels: np.ndarray, query: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Prototype (nearest class-mean, cosine) few-shot classification of embeddings."""
    S = np.asarray(support, "float32")
    classes = np.unique(support_labels)
    P = np.stack([S[np.asarray(support_labels) == c].mean(0) for c in classes])
    idx, sc = embedding_search(P, query, k=len(classes))
    return classes[idx[:, 0]], sc[:, 0]


# =============================================================================================
# 12. Promptable segmentation and vision-language models
# =============================================================================================


def sam_segment(image_path: str | PathLike, output_path: str | PathLike, *,
                points: Sequence[Sequence[float]] | None = None,
                point_labels: Sequence[int] | None = None,
                boxes: Sequence[Sequence[float]] | None = None, point_crs: str | None = None,
                model: Literal["sam", "sam2"] = "sam", model_type: str | None = None,
                automatic: bool = False, vectorize: str | PathLike | None = None,
                **kw: Any) -> dict[str, Any]:
    """Segment Anything (SAM / SAM 2) on a GeoTIFF via segment-geospatial.

    Prompts are points (with ``point_labels``, 1 = foreground) or boxes, in pixel or map
    coordinates (``point_crs``); ``automatic=True`` segments everything. The mask GeoTIFF
    keeps the input georeference; ``vectorize`` also writes polygons. Validate extracted
    boundaries against the imagery: SAM has no notion of land-cover semantics.
    """
    try:
        import samgeo
    except ImportError as e:
        raise ImportError("sam_segment needs `pip install segment-geospatial`") from e
    if model == "sam2":
        from samgeo import SamGeo2

        sam = SamGeo2(model_id=model_type or "sam2-hiera-small", automatic=automatic, **kw)
    else:
        sam = samgeo.SamGeo(model_type=model_type or "vit_h", automatic=automatic, **kw)
    if automatic:
        sam.generate(str(image_path), str(output_path))
    else:
        sam.set_image(str(image_path))
        if boxes is not None:
            sam.predict(boxes=[list(b) for b in boxes], point_crs=point_crs,
                        output=str(output_path))
        else:
            sam.predict(point_coords=[list(p) for p in points or []], point_labels=point_labels,
                        point_crs=point_crs, output=str(output_path))
    out = {"mask": Path(output_path), "verified": False, "note": UNVERIFIED_NOTE}
    if vectorize:
        samgeo.common.raster_to_vector(str(output_path), str(vectorize))
        out["vector"] = Path(vectorize)
    return out


def text_prompt_segment(image_path: str | PathLike, text: str, output_path: str | PathLike, *,
                        box_threshold: float = 0.24, text_threshold: float = 0.24,
                        vectorize: str | PathLike | None = None) -> dict[str, Any]:
    """Grounding DINO + SAM text-prompted segmentation (e.g. ``"building"``, ``"field"``)."""
    try:
        from samgeo.text_sam import LangSAM
    except ImportError as e:
        raise ImportError("text_prompt_segment needs `pip install segment-geospatial "
                          "groundingdino-py`") from e
    sam = LangSAM()
    sam.predict(str(image_path), text, box_threshold=box_threshold,
                text_threshold=text_threshold, output=str(output_path))
    out = {"mask": Path(output_path), "prompt": text, "verified": False, "note": UNVERIFIED_NOTE}
    if vectorize:
        sam.raster_to_vector(str(output_path), str(vectorize))
        out["vector"] = Path(vectorize)
    return out


REMOTE_CLIP_CHECKPOINTS = {
    "remoteclip-vit-b-32": ("ViT-B-32", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
    "remoteclip-vit-l-14": ("ViT-L-14", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"),
    "georsclip-vit-b-32": ("ViT-B-32", "Zilun/GeoRSCLIP",
                           "ckpt/RS5M_ViT-B-32.pt"),
}


def load_rs_clip(name: str = "remoteclip-vit-b-32", *, device: str | None = None):
    """Load RemoteCLIP / GeoRSCLIP (open_clip + Hugging Face Hub).

    Returns ``(model, preprocess, tokenizer)``.
    """
    try:
        import open_clip
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("needs `pip install open_clip_torch huggingface_hub`") from e
    torch = _torch()
    arch, repo, fname = REMOTE_CLIP_CHECKPOINTS[name]
    model, _, preprocess = open_clip.create_model_and_transforms(arch)
    ck = torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("state_dict", ck) if isinstance(ck, Mapping) else ck,
                          strict=False)
    model.to(_device(device)).eval()
    return model, preprocess, open_clip.get_tokenizer(arch)


def zero_shot_classify(images: Sequence[Any], class_names: Sequence[str], *,
                       model: str | tuple = "remoteclip-vit-b-32",
                       template: str = "a satellite image of {}.", device: str | None = None
                       ) -> pd.DataFrame:
    """Zero-shot scene classification with RemoteCLIP / GeoRSCLIP.

    ``images`` are PIL images or ``(H, W, 3)`` uint8 arrays. Returns per-class
    probabilities, the top class and ``verified=False``: treat results as model output that
    needs checking, not as authoritative labels.
    """
    torch = _torch()
    from PIL import Image

    m, pre, tok = load_rs_clip(model, device=device) if isinstance(model, str) else model
    dev = next(m.parameters()).device
    ims = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im, "uint8"))
           for im in images]
    with torch.no_grad():
        x = torch.stack([pre(im) for im in ims]).to(dev)
        fi = m.encode_image(x)
        ft = m.encode_text(tok([template.format(c) for c in class_names]).to(dev))
        fi = fi / fi.norm(dim=-1, keepdim=True)
        ft = ft / ft.norm(dim=-1, keepdim=True)
        p = (100 * fi @ ft.T).softmax(-1).cpu().numpy()
    df = pd.DataFrame(p, columns=list(class_names))
    df["prediction"] = [class_names[i] for i in p.argmax(1)]
    df["verified"] = False
    return df


# =============================================================================================
# 13. Few or imperfect labels
# =============================================================================================


def uncertainty_scores(proba: np.ndarray, strategy: Literal["entropy", "margin", "least_confidence"]
                       = "entropy") -> np.ndarray:
    """Per-sample uncertainty from class probabilities (higher = more informative)."""
    p = np.clip(np.asarray(proba, float), 1e-12, 1)
    if strategy == "entropy":
        return -(p * np.log(p)).sum(1)
    s = np.sort(p, 1)
    if strategy == "margin":
        return 1 - (s[:, -1] - s[:, -2]) if p.shape[1] > 1 else 1 - s[:, -1]
    if strategy == "least_confidence":
        return 1 - s[:, -1]
    raise ValueError(strategy)


def select_for_labeling(proba: np.ndarray, n: int, *, strategy: str = "entropy",
                        embeddings: np.ndarray | None = None, diversity_pool: int = 5,
                        seed: int = 0) -> np.ndarray:
    """Pick ``n`` unlabeled samples for review: the most uncertain, kept diverse.

    With ``embeddings`` (or features) the top ``n * diversity_pool`` uncertain samples are
    clustered into ``n`` groups (k-means) and the most uncertain of each cluster is chosen.
    """
    u = uncertainty_scores(proba, strategy)
    order = np.argsort(-u)
    if embeddings is None or n >= len(order):
        return order[:n]
    from sklearn.cluster import KMeans

    pool = order[:min(len(order), n * diversity_pool)]
    km = KMeans(n_clusters=n, n_init=3, random_state=seed).fit(np.asarray(embeddings)[pool])
    picks = []
    for c in range(n):
        members = pool[km.labels_ == c]
        if members.size:
            picks.append(members[np.argmax(u[members])])
    if len(picks) < n:  # empty clusters (duplicate spectra): top up with the next most uncertain
        rest = [i for i in order if i not in set(picks)]
        picks += rest[:n - len(picks)]
    return np.asarray(picks, int)


def active_learning_loop(model_factory: Callable[[], Any], X_labeled, y_labeled, X_pool, *,
                         oracle: Callable[[np.ndarray], np.ndarray], rounds: int = 5,
                         batch: int = 20, strategy: str = "entropy", X_val=None, y_val=None,
                         diversity: bool = True, seed: int = 0) -> dict[str, Any]:
    """Pool-based active learning: train, score the pool, query the oracle, retrain.

    ``oracle(indices)`` returns labels for pool rows (an analyst, or reference data in
    simulations). Returns the final model, the labeled indices and a history table with
    validation accuracy per round.
    """
    from sklearn.metrics import accuracy_score

    Xl = np.asarray(X_labeled, "float32")
    yl = np.asarray(y_labeled)
    Xp = np.asarray(X_pool, "float32")
    remaining = np.arange(len(Xp))
    picked: list[int] = []
    hist = []
    model = None
    for r in range(rounds + 1):
        model = model_factory()
        model.fit(Xl, yl)
        acc = accuracy_score(y_val, model.predict(np.asarray(X_val, "float32"))) \
            if X_val is not None else np.nan
        hist.append({"round": r, "n_labeled": len(yl), "val_accuracy": acc})
        if r == rounds or remaining.size == 0:
            break
        p = model.predict_proba(Xp[remaining])
        sel = select_for_labeling(p, min(batch, remaining.size), strategy=strategy,
                                  embeddings=Xp[remaining] if diversity else None, seed=seed)
        chosen = remaining[sel]
        new_y = np.asarray(oracle(chosen))
        Xl = np.concatenate([Xl, Xp[chosen]])
        yl = np.concatenate([yl, new_y])
        picked += chosen.tolist()
        remaining = np.setdiff1d(remaining, chosen)
    return {"model": model, "queried": np.asarray(picked, int), "history": pd.DataFrame(hist)}


def pseudo_label(proba: np.ndarray, threshold: float = 0.9, classes: Sequence | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Confident pseudo-labels → ``(indices above threshold, their labels)``."""
    p = np.asarray(proba)
    conf = p.max(1)
    idx = np.flatnonzero(conf >= threshold)
    lab = p[idx].argmax(1)
    if classes is not None:
        lab = np.asarray(classes)[lab]
    return idx, lab


def self_training(model_factory: Callable[[], Any], X_labeled, y_labeled, X_unlabeled, *,
                  threshold: float = 0.9, rounds: int = 3, max_per_round: int | None = None,
                  X_val=None, y_val=None) -> dict[str, Any]:
    """Semi-supervised self-training with confidence-thresholded pseudo-labels.

    A labeled validation set (never pseudo-labeled) tracks whether pseudo-labels help.
    """
    from sklearn.metrics import accuracy_score

    Xl, yl = np.asarray(X_labeled, "float32"), np.asarray(y_labeled)
    Xu = np.asarray(X_unlabeled, "float32")
    used = np.zeros(len(Xu), bool)
    hist = []
    model = model_factory().fit(Xl, yl)
    for r in range(rounds):
        acc = accuracy_score(y_val, model.predict(np.asarray(X_val, "float32"))) \
            if X_val is not None else np.nan
        hist.append({"round": r, "n_train": len(yl), "val_accuracy": acc})
        cand = np.flatnonzero(~used)
        if cand.size == 0:
            break
        p = model.predict_proba(Xu[cand])
        idx, lab = pseudo_label(p, threshold, model.classes_)
        if max_per_round and idx.size > max_per_round:
            idx, lab = idx[np.argsort(-p[idx].max(1))[:max_per_round]], lab[:max_per_round]
            lab = np.asarray(model.classes_)[p[idx].argmax(1)]
        if idx.size == 0:
            break
        used[cand[idx]] = True
        Xl, yl = np.concatenate([Xl, Xu[cand[idx]]]), np.concatenate([yl, lab])
        model = model_factory().fit(Xl, yl)
    acc = accuracy_score(y_val, model.predict(np.asarray(X_val, "float32"))) \
        if X_val is not None else np.nan
    hist.append({"round": len(hist), "n_train": len(yl), "val_accuracy": acc})
    return {"model": model, "pseudo_labeled": np.flatnonzero(used), "history": pd.DataFrame(hist)}


def fixmatch_loss(model, x_labeled, y_labeled, x_weak, x_strong, *, threshold: float = 0.95,
                  lambda_u: float = 1.0, loss_fn: Any = None) -> tuple[Any, dict[str, float]]:
    """FixMatch (Sohn et al. 2020) loss: supervised CE plus consistency on confident
    weak-view pseudo-labels applied to strong views. Works for chips and dense masks."""
    torch = _torch()
    F = torch.nn.functional
    sup = (loss_fn or F.cross_entropy)(model(x_labeled), y_labeled)
    with torch.no_grad():
        pw = torch.softmax(model(x_weak), 1)
        conf, pl = pw.max(1)
        mask = (conf >= threshold).float()
    lu = F.cross_entropy(model(x_strong), pl, reduction="none")
    unsup = (lu * mask).sum() / mask.numel()
    return sup + lambda_u * unsup, {"supervised": float(sup.detach()),
                                    "unsupervised": float(unsup.detach()),
                                    "mask_rate": float(mask.mean())}


def train_fixmatch(model, labeled_loader, unlabeled_loader, val_loader=None, *, epochs: int = 10,
                   lr: float = 1e-3, threshold: float = 0.95, lambda_u: float = 1.0,
                   weak: Callable | None = None, strong: Callable | None = None,
                   device: str | None = None, seed: int = 0, verbose: bool = True) -> dict:
    """Semi-supervised training loop with :func:`fixmatch_loss`; validation uses labels only."""
    torch = _torch()
    set_seed(seed)
    dev = _device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weak = weak or OverheadAugment(seed=seed)
    strong = strong or OverheadAugment(brightness=0.3, contrast=0.4, noise=0.1, crop=0.7,
                                       seed=seed + 1)
    hist = {"loss": [], "mask_rate": [], "val_score": []}
    for ep in range(epochs):
        model.train()
        tot, rate, n = 0.0, 0.0, 0
        for lb, ub in zip(labeled_loader, unlabeled_loader):
            x, y = _unpack(lb, dev)
            xu = (ub["image"] if isinstance(ub, Mapping) else ub[0] if isinstance(ub, (list, tuple))
                  else ub).float()
            xw = torch.stack([weak(t, None)[0] for t in xu]).to(dev)
            xs = torch.stack([strong(t, None)[0] for t in xu]).to(dev)
            loss, info = fixmatch_loss(model, x, y, xw, xs, threshold=threshold, lambda_u=lambda_u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            rate += info["mask_rate"]
            n += 1
        hist["loss"].append(tot / max(n, 1))
        hist["mask_rate"].append(rate / max(n, 1))
        if val_loader is not None:
            p = predict_proba_batches(model, val_loader.dataset, device=device)
            yv = np.concatenate([np.asarray(_unpack(b, "cpu")[1]) for b in val_loader])
            hist["val_score"].append(float((p.argmax(1) == yv).mean()))
        if verbose:
            print(f"fixmatch epoch {ep + 1}/{epochs} loss={hist['loss'][-1]:.4f} "
                  f"mask_rate={hist['mask_rate'][-1]:.2f}")
    return hist


def weak_labels_from_product(product: xr.DataArray, mapping: Mapping[int, int], *,
                             erode: int = 0, ignore_index: int = -100) -> xr.DataArray:
    """Weak labels from a coarse map product (e.g. ESA WorldCover, CDL): remap product codes
    to task classes, unmapped codes → ``ignore_index``; ``erode`` pixels drop uncertain
    class edges where coarse products are least reliable."""
    a = np.asarray(product.values)
    out = np.full(a.shape, ignore_index, "int64")
    for src, dst in mapping.items():
        out[a == src] = dst
    if erode:
        from scipy import ndimage as ndi

        keep = np.zeros(a.shape, bool)
        for dst in set(mapping.values()):
            keep |= ndi.binary_erosion(out == dst, iterations=erode)
        out[~keep] = ignore_index
    res = product.copy(data=out).rename("weak_labels")
    res.attrs = {"nodata": ignore_index, "source": "weak supervision from coarse product"}
    return res


def coral(source: np.ndarray, target: np.ndarray, *, eps: float = 1.0) -> np.ndarray:
    """CORAL domain adaptation (Sun et al. 2016): re-color source features so their mean and
    covariance match the target domain (other region, season or sensor)."""
    from scipy import linalg

    Xs, Xt = np.asarray(source, float), np.asarray(target, float)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    Ws = np.real(linalg.fractional_matrix_power(Cs, -0.5))
    Wt = np.real(linalg.fractional_matrix_power(Ct, 0.5))
    return ((Xs - Xs.mean(0)) @ Ws @ Wt + Xt.mean(0)).astype("float32")


def histogram_match(image: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Per-band histogram matching of an image to a reference (radiometric domain alignment)."""
    from skimage.exposure import match_histograms

    a, r = as_band_array(image), as_band_array(reference)
    A, R = np.asarray(a.values, float), np.asarray(r.values, float)
    out = np.full_like(A, np.nan)
    for i in range(A.shape[0]):
        ma, mr = np.isfinite(A[i]), np.isfinite(R[i])
        out[i][ma] = match_histograms(A[i][ma], R[i][mr])
    return a.copy(data=out.astype("float32"))


def tent_adapt(model, loader, *, steps: int = 1, lr: float = 1e-4, device: str | None = None):
    """TENT test-time adaptation (Wang et al. 2021): minimize prediction entropy on unlabeled
    target batches, updating only BatchNorm affine parameters (statistics re-estimated).

    Never adapt on the final test labels' split if that split is also used for reporting;
    adapt on a separate unlabeled target sample (see :func:`check_split_isolation`).
    """
    torch = _torch()
    nn = torch.nn
    dev = _device(device)
    model.to(dev).train()
    params = []
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad = False
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats, m.running_mean, m.running_var = False, None, None
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad = True
                    params.append(p)
    if not params:
        raise ValueError("TENT needs BatchNorm layers")
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for _ in range(steps):
        for batch in loader:
            x, _ = _unpack(batch, dev)
            p = torch.softmax(_forward(model, x), 1)
            ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            ent.backward()
            opt.step()
            hist.append(float(ent.detach()))
    model.eval()
    return model, hist


def check_split_isolation(**splits: Sequence) -> bool:
    """Raise if any two named id sets overlap (e.g. ``train=..., adapt=..., test=...``).

    Keeps final test data out of adaptation, pseudo-labeling and model selection.
    """
    names = list(splits)
    sets = {k: set(np.asarray(v).ravel().tolist()) for k, v in splits.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                raise ValueError(f"splits {a!r} and {b!r} share {len(inter)} ids")
    return True


# =============================================================================================
# 14. Evaluation, mapping and reproducibility
# =============================================================================================


def classification_report(y_true: Sequence, y_pred: Sequence, *, labels: Sequence | None = None,
                          class_names: Mapping[Any, str] | None = None, n_boot: int = 0,
                          seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Per-class precision / recall / F1 with support, confusion matrix, overall accuracy,
    Cohen's kappa and macro F1; ``n_boot > 0`` adds bootstrap confidence intervals."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(yt.tolist()) | set(yp.tolist()))

    def stats(t, p):
        cm = pd.crosstab(pd.Categorical(t, labels), pd.Categorical(p, labels), dropna=False)
        m = cm.to_numpy(float)
        tp = np.diag(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec, rec = tp / m.sum(0), tp / m.sum(1)
            f1 = 2 * prec * rec / (prec + rec)
        n = m.sum()
        oa = tp.sum() / n if n else np.nan
        pe = (m.sum(0) * m.sum(1)).sum() / n ** 2 if n else np.nan
        kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
        return cm, prec, rec, f1, oa, kappa

    cm, prec, rec, f1, oa, kappa = stats(yt, yp)
    names = [(class_names or {}).get(c, c) for c in labels]
    per = pd.DataFrame({"precision": prec, "recall": rec, "f1": f1,
                        "support": cm.to_numpy().sum(1)}, index=names)
    cm.index, cm.columns = names, names
    cm.index.name, cm.columns.name = "reference", "predicted"
    out = {"overall_accuracy": float(oa), "kappa": float(kappa),
           "macro_f1": float(np.nanmean(f1)), "per_class": per, "confusion_matrix": cm,
           "n": int(len(yt))}
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = {"oa": [], "kappa": [], "macro_f1": [], "f1": []}
        for _ in range(n_boot):
            i = rng.integers(0, len(yt), len(yt))
            _, _, _, f, o, k = stats(yt[i], yp[i])
            boots["oa"].append(o)
            boots["kappa"].append(k)
            boots["macro_f1"].append(np.nanmean(f))
            boots["f1"].append(f)
        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        out["ci"] = {k: (float(np.nanpercentile(boots[k], lo)),
                         float(np.nanpercentile(boots[k], hi)))
                     for k in ("oa", "kappa", "macro_f1")}
        F1 = np.asarray(boots["f1"])
        per["f1_ci_low"] = np.nanpercentile(F1, lo, 0)
        per["f1_ci_high"] = np.nanpercentile(F1, hi, 0)
    return out


def error_map(predicted: xr.DataArray, reference: xr.DataArray, *,
              nodata: int = -1) -> xr.DataArray:
    """Map of agreement: 1 = correct, 0 = error, NaN = no reference or no prediction.

    Inspect it on a map: clustered errors point to label problems, domain shift or
    misregistration rather than random noise.
    """
    p = np.asarray(predicted.values).astype(float)
    r = np.asarray(reference.values).astype(float)
    valid = np.isfinite(p) & np.isfinite(r) & (p != nodata) & (r != nodata) & (r != -100)
    out = np.where(valid, (p == r).astype(float), np.nan)
    return predicted.copy(data=out.astype("float32")).rename("correct")


def compare_with_baselines(y_train: Sequence, y_test: Sequence,
                           predictions: Mapping[str, Sequence], *, seed: int = 0) -> pd.DataFrame:
    """Compare models with majority-class and stratified-random baselines on the same test set."""
    yt = np.asarray(y_test)
    ytr = np.asarray(y_train)
    vals, cnt = np.unique(ytr, return_counts=True)
    rng = np.random.default_rng(seed)
    preds = {"majority_baseline": np.full(len(yt), vals[np.argmax(cnt)]),
             "stratified_random_baseline": rng.choice(vals, len(yt), p=cnt / cnt.sum()),
             **{k: np.asarray(v) for k, v in predictions.items()}}
    rows = []
    for k, p in preds.items():
        r = classification_report(yt, p)
        rows.append({"model": k, "overall_accuracy": r["overall_accuracy"], "kappa": r["kappa"],
                     "macro_f1": r["macro_f1"]})
    return pd.DataFrame(rows).set_index("model")


def _package_versions() -> dict[str, str]:
    import importlib
    import platform

    out = {"python": platform.python_version()}
    for m in ("geolattice", "numpy", "pandas", "xarray", "rioxarray", "rasterio", "geopandas",
              "sklearn", "skimage", "torch", "torchvision", "timm", "torchgeo",
              "segmentation_models_pytorch", "xgboost"):
        try:
            out[m] = str(getattr(importlib.import_module(m), "__version__", "?"))
        except Exception:  # noqa: BLE001 - optional packages
            continue
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_run(path: str | PathLike, *, config: RunConfig | Mapping | None = None,
             normalizer: Normalizer | None = None, class_names: Mapping | Sequence | None = None,
             raster_meta: RasterMeta | Mapping | None = None, split: Split | None = None,
             metrics: Mapping | None = None, checkpoint: str | None = None,
             extra: Mapping | None = None) -> Path:
    """Write everything needed to reproduce and interpret a run to one JSON file.

    Includes the configuration, package versions, checkpoint path, normalization values,
    class names, CRS / transform / band names, split identifiers and metrics.
    """
    rec = {"config": config.to_dict() if isinstance(config, RunConfig) else dict(config or {}),
           "versions": _package_versions(), "checkpoint": checkpoint,
           "normalizer": normalizer.to_dict() if normalizer else None,
           "class_names": dict(class_names) if isinstance(class_names, Mapping)
           else (list(class_names) if class_names is not None else None),
           "raster_meta": raster_meta.to_dict() if isinstance(raster_meta, RasterMeta)
           else (dict(raster_meta) if raster_meta else None),
           "split": split.to_dict() if split is not None else None,
           "metrics": {k: v for k, v in (metrics or {}).items()},
           "extra": dict(extra or {})}
    path = Path(path)
    path.write_text(json.dumps(rec, indent=2, default=_jsonable))
    return path


def load_run(path: str | PathLike) -> dict[str, Any]:
    """Read a run record; ``normalizer`` is restored as a :class:`Normalizer`."""
    rec = json.loads(Path(path).read_text())
    if rec.get("normalizer"):
        rec["normalizer"] = Normalizer.from_dict(rec["normalizer"])
    return rec


# =============================================================================================
# Workflow driver and command line
# =============================================================================================


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse workflow choices and data paths from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m geolattice.rsense.classify.image_classification",
        description="Geospatial image learning workflows (classification, segmentation, "
                    "detection, change detection, time series).")
    parser.add_argument("--task", choices=TASKS, default="classification")
    parser.add_argument("--learning-mode", choices=LEARNING_MODES, default="supervised")
    parser.add_argument("--spatial-unit", choices=SPATIAL_UNITS, default="pixel",
                        help="Pixel-level analysis or object/segment-level analysis.")
    parser.add_argument("--bands", nargs="+", default=None,
                        help="Band names in the order of the image stack (default: from files).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", help="Multi-band GeoTIFF or folder of single-band GeoTIFFs.")
    parser.add_argument("--image2", help="Second-date image (change detection).")
    parser.add_argument("--labels", help="Training labels: CSV with x/y columns, or vector file.")
    parser.add_argument("--label-col", default="class")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--model", default="rf", help="rf, svm, xgb, knn, logistic, gbm, mlp, "
                        "or kmeans / minibatch_kmeans / gmm for unsupervised runs.")
    parser.add_argument("--n-classes", type=int, default=6, help="Clusters (unsupervised).")
    parser.add_argument("--block-size", type=float, default=None,
                        help="Spatial block size in map units for leakage-safe splits/CV.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-segments", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.9,
                        help="Pseudo-label threshold (semi-supervised).")
    parser.add_argument("--out", help="Output folder for the map, metrics and run record.")
    a = parser.parse_args(argv)
    return RunConfig(task=a.task, learning_mode=a.learning_mode, spatial_unit=a.spatial_unit,
                     bands=tuple(a.bands) if a.bands else (), seed=a.seed, model=a.model,
                     n_classes=a.n_classes, block_size=a.block_size, test_size=a.test_size,
                     image=a.image, image2=a.image2, labels=a.labels, label_col=a.label_col,
                     x_col=a.x_col, y_col=a.y_col, out=a.out, n_segments=a.n_segments,
                     confidence=a.confidence)


def _read_labels(config: RunConfig, crs: Any):
    p = Path(config.labels)
    if p.suffix.lower() in (".csv", ".txt"):
        return labels_from_table(p, x=config.x_col, y=config.y_col, label=config.label_col, crs=crs)
    import geopandas as gpd

    return gpd.read_file(p)


def run_workflow(config: RunConfig, *, verbose: bool = True) -> dict[str, Any]:
    """Run a classical end-to-end workflow selected by ``config`` and write its outputs.

    * ``classification`` × ``unsupervised`` × ``pixel``: clustering map;
    * ``classification`` × ``supervised`` × ``pixel`` / ``object``: spatial (block) split,
      spatial CV, model fit, held-out evaluation, map;
    * ``classification`` × ``semi-supervised`` × ``pixel``: self-training with pseudo-labels;
    * ``change-detection`` (``--image2``): co-registration check, CVA + Otsu change map.

    Deep-learning tasks need task-specific chips and training settings; use the Python API
    (:func:`build_cnn`, :func:`build_segmenter`, :func:`build_object_detector`,
    :func:`build_temporal_model`, :func:`train_model`) for those.
    """
    if config.image is None:
        raise ValueError("config.image is required")
    set_seed(config.seed)
    img = load_stack(config.image, band_names=list(config.bands) or None)
    meta = RasterMeta.from_dataarray(img)
    out_dir = Path(config.out) if config.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"meta": meta}
    if config.task == "change-detection":
        if not config.image2:
            raise ValueError("change detection needs --image2")
        img2 = load_stack(config.image2, band_names=list(img["band"].values))
        result["coregistration"] = check_coregistration(img, img2)
        cv = change_vector_analysis(img, img2)
        result["change"] = cv
        if out_dir:
            write_class_map(cv["change"].astype("int32"), out_dir / "change.tif",
                            class_names={0: "no change", 1: "change"})
            save_run(out_dir / "run.json", config=config, raster_meta=meta,
                     metrics={"coregistration": result["coregistration"],
                              "threshold": cv.attrs["threshold"]})
        return result
    if config.task != "classification":
        raise NotImplementedError(
            f"task={config.task!r} needs deep-learning training data; use the Python API "
            "(build_segmenter / build_object_detector / build_temporal_model + train_model).")
    if config.learning_mode == "unsupervised":
        method = config.model if config.model in ("kmeans", "minibatch_kmeans", "gmm") else "kmeans"
        cmap, model = unsupervised_classify(img, config.n_classes, method=method, seed=config.seed)
        result.update(map=cmap, model=model)
        if out_dir:
            write_class_map(cmap, out_dir / "clusters.tif")
            model.cluster_means_.to_csv(out_dir / "cluster_means.csv")
            save_run(out_dir / "run.json", config=config, raster_meta=meta)
        return result
    if config.labels is None:
        raise ValueError(f"{config.learning_mode} classification needs --labels")
    gdf = _read_labels(config, img.rio.crs)
    if config.learning_mode == "self-supervised":
        raise NotImplementedError("self-supervised pretraining needs chips; use "
                                  "pretrain_contrastive / pretrain_masked from the Python API")
    if config.spatial_unit == "object":
        res = object_based_classify(img, gdf, config.label_col, model=config.model,
                                    n_segments=config.n_segments, seed=config.seed)
        result.update(res)
        if out_dir:
            write_class_map(res["map"], out_dir / "object_classes.tif")
            save_run(out_dir / "run.json", config=config, raster_meta=meta,
                     extra={"n_segments": int(len(res["features"]))})
        return result
    X, y, coords = sample_pixels(img, gdf, config.label_col)
    bs = config.block_size or float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) / 10 or 1)
    split = spatial_block_split(coords, block_size=bs, test_size=config.test_size,
                                seed=config.seed, stratify=y)
    Xtr, ytr = X.iloc[split.train], y[split.train]
    if config.learning_mode == "semi-supervised":
        from .estimators import Classifier

        Xi, ok, _, _ = _image_matrix(img)
        rng = np.random.default_rng(config.seed)
        pool = Xi[rng.choice(np.flatnonzero(ok), min(20_000, int(ok.sum())), replace=False)]
        st = self_training(lambda: Classifier(config.model), Xtr.to_numpy("float32"), ytr, pool,
                           threshold=config.confidence,
                           X_val=X.iloc[split.test].to_numpy("float32"), y_val=y[split.test])
        clf, cv = st["model"], st["history"]
    else:
        clf, cv = train_pixel_classifier(Xtr, ytr, model=config.model, coords=coords[split.train],
                                         block_size=bs, seed=config.seed)
    Xte = X.iloc[split.test]
    pred_te = clf.predict(Xte.to_numpy("float32") if config.learning_mode == "semi-supervised"
                          else Xte)
    report = classification_report(y[split.test], pred_te, n_boot=200, seed=config.seed)
    cmap = predict_map(clf, img)
    result.update(model=clf, cv=cv, split=split, report=report, map=cmap)
    if verbose:
        print(f"held-out (spatial blocks of {bs:g}) OA={report['overall_accuracy']:.3f} "
              f"kappa={report['kappa']:.3f} macro-F1={report['macro_f1']:.3f}")
    if out_dir:
        write_class_map(cmap, out_dir / "classification.tif",
                        metadata={"model": config.model, "learning_mode": config.learning_mode})
        report["per_class"].to_csv(out_dir / "per_class.csv")
        report["confusion_matrix"].to_csv(out_dir / "confusion_matrix.csv")
        if cv is not None:
            cv.to_csv(out_dir / "cv.csv", index=False)
        save_run(out_dir / "run.json", config=config, raster_meta=meta, split=split,
                 metrics={k: report[k] for k in ("overall_accuracy", "kappa", "macro_f1", "n")}
                 | {"ci": report.get("ci")})
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    """Command-line entry point."""
    config = parse_args(argv)
    print("Selected workflow:")
    print(f"  task:          {config.task}")
    print(f"  learning mode: {config.learning_mode}")
    print(f"  spatial unit:  {config.spatial_unit}")
    print(f"  bands:         {', '.join(config.bands) if config.bands else '(from files)'}")
    print(f"  seed:          {config.seed}")
    if config.image is None:
        print("No --image given; nothing to run. See --help.")
        return
    res = run_workflow(config)
    if config.out:
        print(f"Outputs written to {config.out}")
    return res


__all__ = [
    "TASKS", "LEARNING_MODES", "SPATIAL_UNITS", "RunConfig", "RasterMeta", "Split", "Normalizer",
    "load_stack", "as_band_dataset", "as_band_array", "labels_from_table", "rasterize_labels",
    "sample_pixels", "spatial_blocks", "spatial_block_split", "group_split", "temporal_split",
    "spatial_kfold", "unsupervised_classify", "spatial_cross_validate", "train_pixel_classifier",
    "predict_map", "object_based_classify", "set_seed", "ChipDataset", "PairChipDataset",
    "as_loader", "extract_chips", "rgb_mask_to_index", "read_class_dict", "chips_from_folder",
    "adapt_input_channels", "build_cnn", "receptive_field", "count_parameters", "OverheadAugment",
    "cutmix", "class_weights", "get_loss", "train_model", "load_checkpoint",
    "predict_proba_batches", "predict_chip_map", "RS_WEIGHTS", "load_rs_backbone",
    "set_trainable", "compare_transfer_modes", "vit_attention_rollout", "build_segmenter",
    "sliding_window_predict", "write_class_map", "segmentation_metrics", "build_object_detector",
    "box_iou", "nms", "mean_average_precision", "obb_to_polygon", "polygon_iou",
    "detection_tiles", "merge_detections", "detect_objects", "count_objects", "clean_mask",
    "instances_from_semantic", "boundary_weight_map", "mask_boundaries", "polygonize",
    "instances_from_detector", "export_vectors", "image_ratio", "image_difference",
    "change_vector_analysis", "pca_change", "check_coregistration", "build_change_model",
    "change_metrics", "pixel_time_series", "build_temporal_model", "build_fusion_model",
    "fuse_early", "fuse_late", "sar_assisted_gap_fill", "pansharpen", "pansharpening_quality",
    "nt_xent_loss", "make_encoder", "pretrain_contrastive", "build_mae", "pretrain_masked",
    "extract_embeddings", "linear_probe", "geographic_holdout", "check_geographic_overlap",
    "FOUNDATION_MODELS", "foundation_model_info", "load_foundation_encoder", "terratorch_config",
    "embedding_search", "few_shot_prototypes", "sam_segment", "text_prompt_segment",
    "load_rs_clip", "zero_shot_classify", "uncertainty_scores", "select_for_labeling",
    "active_learning_loop", "pseudo_label", "self_training", "fixmatch_loss", "train_fixmatch",
    "weak_labels_from_product", "coral", "histogram_match", "tent_adapt", "check_split_isolation",
    "classification_report", "error_map", "compare_with_baselines", "save_run", "load_run",
    "parse_args", "run_workflow", "main",
]


if __name__ == "__main__":
    main()
