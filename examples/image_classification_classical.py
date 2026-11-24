"""Classical workflows of geolattice.rsense.classify.image_classification on the tutorial data.

Usage (from the repository root)::

    python examples/image_classification_classical.py [DATA_DIR] [OUT_DIR]

DATA_DIR defaults to Tutorials/notebooks/07_remote_sensing/07_08_image_classification.ipynb/data
and OUT_DIR to ./image_classification_outputs.
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import geolattice.rsense.classify.image_classification as ic

D = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "Tutorials/notebooks/07_remote_sensing/07_08_image_classification.ipynb/data"
)
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "image_classification_outputs")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# --- Sentinel-2 UB North Campus (10 bands, no CRS in files -> UTM 17N) ----------------------
img = ic.load_stack(D / "Sentinel_2_UB/Images", sensor="Sentinel-2", crs="EPSG:32617")
meta = ic.RasterMeta.from_dataarray(img)
print("S2:", img.shape, meta.band_names, meta.nodata, meta.dtype)
train = ic.labels_from_table(
    D / "Sentinel_2_UB/Label_Point_Data/train_data.csv", label="Class_ID", crs=img.rio.crs
)
test = ic.labels_from_table(
    D / "Sentinel_2_UB/Label_Point_Data/test_data.csv", label="Class_ID", crs=img.rio.crs
)
names = dict(train.groupby("Class_ID")["Landuse"].first())
print("classes:", names, "train", len(train), "test", len(test))

X, y, coords = ic.sample_pixels(img, train, "Class_ID")
split = ic.spatial_block_split(coords, block_size=300, test_size=0.3, seed=0, stratify=y)
print("spatial split:", len(split.train), len(split.test), "blocks", len(np.unique(split.groups)))
norm = ic.Normalizer().fit(X.iloc[split.train])
clf, cv = ic.train_pixel_classifier(
    X.iloc[split.train],
    y[split.train],
    model="rf",
    coords=coords[split.train],
    block_size=300,
    n_estimators=200,
)
print("spatial CV:\n", cv.round(3))
rep = ic.classification_report(
    y[split.test], clf.predict(X.iloc[split.test]), class_names=names, n_boot=200
)
print(
    "block-held-out OA %.3f kappa %.3f macroF1 %.3f CI %s"
    % (rep["overall_accuracy"], rep["kappa"], rep["macro_f1"], rep["ci"]["oa"])
)
# independent test set provided with the tutorial
Xt, yt, _ = ic.sample_pixels(img, test, "Class_ID")
rep2 = ic.classification_report(yt, clf.predict(Xt), class_names=names)
print("tutorial test_data OA %.3f kappa %.3f" % (rep2["overall_accuracy"], rep2["kappa"]))
print(rep2["per_class"].round(3))
print(ic.compare_with_baselines(y[split.train], yt, {"rf": clf.predict(Xt)}).round(3))
lc = ic.predict_map(clf, img)
assert lc.shape == img.shape[1:] and lc.rio.crs == img.rio.crs
ic.write_class_map(
    lc,
    OUT / "s2_rf.tif",
    class_names=names,
    colormap={1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 160, 0), 5: (160, 160, 160)},
)
em = ic.error_map(lc, ic.rasterize_labels(test, img, "Class_ID")[0])
print("error map accuracy at test pixels: %.3f" % float(np.nanmean(em.values)))
ic.save_run(
    OUT / "s2_run.json",
    config=ic.RunConfig(bands=tuple(meta.band_names)),
    normalizer=norm,
    class_names=names,
    raster_meta=meta,
    split=split,
    metrics={"oa": rep["overall_accuracy"], "kappa": rep["kappa"]},
)
rec = ic.load_run(OUT / "s2_run.json")
assert rec["raster_meta"]["crs"] and rec["normalizer"].center is not None

# svm and xgb-free alternatives
for m in ("svm", "gbm", "knn"):
    c2, _ = ic.train_pixel_classifier(norm.transform(X.iloc[split.train]), y[split.train], model=m)
    r = ic.classification_report(yt, c2.predict(norm.transform(Xt)))
    print(f"{m}: OA={r['overall_accuracy']:.3f}")

# unsupervised
for method in ("kmeans", "minibatch_kmeans", "gmm"):
    cm, model = ic.unsupervised_classify(img, 5, method=method, seed=0)
    print(method, "clusters:", np.unique(cm.values).tolist())

# object-based
obj = ic.object_based_classify(img, train, "Class_ID", n_segments=1500, compactness=0.1)
r = ic.classification_report(
    yt,
    ic.sample_pixels(obj["map"].astype("float32").rename("c"), test, "Class_ID")[0]["c"].to_numpy(),
)
print("OBIA segments:", len(obj["features"]), "OA vs test points %.3f" % r["overall_accuracy"])

# semi-supervised self-training (pixel)
from geolattice.rsense.classify import Classifier  # noqa: E402

Xi, ok, _, _ = ic._image_matrix(img)
pool = Xi[np.flatnonzero(ok)[::7]]
small = np.random.default_rng(0).choice(split.train, 300, replace=False)
st = ic.self_training(
    lambda: Classifier("rf", n_estimators=100, random_state=0),
    X.iloc[small].to_numpy("float32"),
    y[small],
    pool,
    threshold=0.9,
    X_val=Xt.to_numpy("float32"),
    y_val=yt,
)
print("self-training history:\n", st["history"].round(3))

# active learning simulation, oracle = reference labels
lab0 = small[:100]
pool_idx = np.setdiff1d(split.train, lab0)
al = ic.active_learning_loop(
    lambda: Classifier("rf", n_estimators=100, random_state=0),
    X.iloc[lab0],
    y[lab0],
    X.iloc[pool_idx],
    oracle=lambda i: y[pool_idx][i],
    rounds=3,
    batch=50,
    X_val=Xt,
    y_val=yt,
)
print("active learning:\n", al["history"].round(3))

# --- Landsat Bangladesh change: PCA / ratio / CVA on two synthetic "dates" ----------------
ls = ic.load_stack(D / "Landsat_BD/Images/multi.tif")
print("Landsat:", ls.shape, ls.rio.crs)
after = ls.copy(data=ls.values * 1.0)
after.values[:, 100:200, 100:200] *= 1.6  # simulated change
cv_ = ic.change_vector_analysis(ls.isel(band=slice(1, 7)), after.isel(band=slice(1, 7)))
pc = ic.pca_change(ls.isel(band=slice(1, 7)), after.isel(band=slice(1, 7)), method="difference")
ref = np.zeros(ls.shape[1:], bool)
ref[100:200, 100:200] = True
print(
    "CVA change metrics", {k: round(v, 3) for k, v in ic.change_metrics(cv_["change"], ref).items()}
)
print(
    "PCA-diff change metrics",
    {k: round(v, 3) for k, v in ic.change_metrics(pc["change"], ref).items()},
)
print("coreg", ic.check_coregistration(ls, after))
rat = ic.image_ratio(ls.isel(band=4), after.isel(band=4))
print("ratio increase px:", int(rat["increase"].sum()))

# --- WorldView-2 field boundaries: polygons, polygonize, export --------------------------
wv = ic.load_stack(D / "WorldView2_BD/Images/multi.tif")
fields = gpd.read_file(D / "WorldView2_BD/Labeled_data/Training_Field_Boundary.shp")
print("WV2:", wv.shape, wv.rio.crs, "fields", len(fields), fields.columns.tolist()[:6])
fields = fields.assign(fid=np.arange(1, len(fields) + 1), one=1)
lab, codes = ic.rasterize_labels(fields, wv, "one", fill=0)
crop, crop_codes = ic.rasterize_labels(fields, wv, "crop_name", fill=-1)
print("crops:", crop_codes)
inst_true, _ = ic.rasterize_labels(fields, wv, "fid", fill=0)
# boundary-aware split of the merged field mask with a boundary raster (watershed)
mask = ic.clean_mask(lab, min_size=50, hole_size=50)
inst, tab = ic.instances_from_semantic(mask, method="watershed", min_distance=15, min_size=50)
print(
    "connected components:",
    int(ic.instances_from_semantic(mask)[1].shape[0]),
    "watershed instances:",
    len(tab),
    "true fields:",
    len(fields),
)
w = ic.boundary_weight_map(np.asarray(inst_true.values)[:400, :400])
b = ic.mask_boundaries(np.asarray(inst_true.values))
print("boundary weights range %.2f-%.2f, boundary px %d" % (w.min(), w.max(), b.sum()))
polys = ic.polygonize(
    inst_true,
    class_map=crop,
    class_names={v: k for k, v in crop_codes.items()},
    min_area=50,
    simplify=1.5,
    regularize="auto",
)
print(polys[["class_name", "area"]].groupby("class_name").size().to_dict())
print(
    "polygons:",
    len(polys),
    polys.crs.to_epsg(),
    polys[["area", "compactness"]].describe().loc[["mean"]].round(2).to_dict(),
)
ic.export_vectors(polys, OUT / "wv2_fields.gpkg")
ic.export_vectors(polys.head(20), OUT / "wv2_fields.geojson")
back = gpd.read_file(OUT / "wv2_fields.gpkg")
assert back.crs == polys.crs and len(back) == len(polys)
segm = ic.segmentation_metrics(lab, lab)
assert abs(segm["miou"] - 1) < 1e-9
print(f"classical workflows done in {time.time() - t0:.1f}s")
