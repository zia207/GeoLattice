"""Deep-learning workflows of geolattice.rsense.classify.image_classification on the tutorial data (CPU-sized).

Usage (from the repository root)::

    python examples/image_classification_deep.py [DATA_DIR] [OUT_DIR]

DATA_DIR defaults to Tutorials/notebooks/07_remote_sensing/07_08_image_classification.ipynb/data
and OUT_DIR to ./image_classification_outputs.
"""

import sys
import time
from pathlib import Path

import numpy as np
import geolattice.rsense.classify.image_classification as ic
import torch

D = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "Tutorials/notebooks/07_remote_sensing/07_08_image_classification.ipynb/data"
)
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "image_classification_outputs")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
t0 = time.time()

# --- DeepGlobe land cover (Landcover_data): semantic segmentation -----------------------------
names, colors = ic.read_class_dict(D / "Landcover_data/class_dict.csv")
X, M, src = ic.chips_from_folder(
    D / "Landcover_data/train", colors=colors, size=64, stride=64, downsample=8
)
print("DeepGlobe chips", X.shape, M.shape, "classes present", np.unique(M).tolist(), names)
img_ids = np.array(src)
split = ic.group_split(img_ids, test_size=0.34, seed=0)  # whole images held out
print(
    "train chips",
    len(split.train),
    "test chips",
    len(split.test),
    "test images",
    sorted(set(img_ids[split.test])),
)
norm = ic.Normalizer().fit(X[split.train], band_axis=1)
M = np.where(M == 6, -100, M)  # 'unknown' is ignored
w = ic.class_weights(M[split.train], num_classes=6, method="sqrt_inverse")
tr = ic.ChipDataset(X[split.train], M[split.train], normalizer=norm, transform=ic.OverheadAugment())
te = ic.ChipDataset(X[split.test], M[split.test], normalizer=norm)
seg = ic.build_segmenter("unet_basic", in_channels=3, num_classes=6, base=16, depth=3)
h = ic.train_model(
    seg,
    ic.as_loader(tr, 8),
    ic.as_loader(te, 8, shuffle=False),
    task="segmentation",
    epochs=12,
    lr=3e-3,
    loss=ic.get_loss("ce+dice", class_weights=w),
    early_stopping=5,
    checkpoint=OUT / "unet.pt",
    verbose=False,
)
print("U-Net val mIoU by epoch", np.round(h["val_score"], 3).tolist())
p = ic.predict_proba_batches(seg, te.images)
sm = ic.segmentation_metrics(p.argmax(1), M[split.test], num_classes=6, class_names=names[:6])
print("held-out images: pixel acc %.3f mIoU %.3f" % (sm["pixel_accuracy"], sm["miou"]))
print(sm["per_class"].round(3))

# --- Sentinel-2 UB: chip CNN from labeled points, chip map -----------------------------------
s2 = ic.load_stack(D / "Sentinel_2_UB/Images", crs="EPSG:32617")
tr_pts = ic.labels_from_table(
    D / "Sentinel_2_UB/Label_Point_Data/train_data.csv", label="Class_ID", crs=s2.rio.crs
)
te_pts = ic.labels_from_table(
    D / "Sentinel_2_UB/Label_Point_Data/test_data.csv", label="Class_ID", crs=s2.rio.crs
)
rng = np.random.default_rng(0)
tr_pts = tr_pts.iloc[rng.choice(len(tr_pts), 3000, replace=False)]
cx, cy, _ = ic.extract_chips(s2, tr_pts, "Class_ID", size=9)
tx, ty, _ = ic.extract_chips(s2, te_pts, "Class_ID", size=9)
cy, ty = cy - 1, ty - 1
n2 = ic.Normalizer().fit(cx, band_axis=1)
cnn = ic.build_cnn("vgg_small", in_channels=10, num_classes=5, dropout=0.2)
print("receptive field", ic.receptive_field(cnn, input_size=9))
h2 = ic.train_model(
    cnn,
    ic.as_loader(ic.ChipDataset(cx, cy, normalizer=n2, transform=ic.OverheadAugment()), 64),
    ic.as_loader(ic.ChipDataset(tx, ty, normalizer=n2), 256, shuffle=False),
    epochs=8,
    lr=3e-3,
    loss="focal",
    verbose=False,
)
print("S2 chip CNN val acc", np.round(h2["val_score"], 3).tolist())
rep = ic.classification_report(
    ty, ic.predict_proba_batches(cnn, n2.transform(tx, 1), tta=True).argmax(1)
)
print("S2 chip CNN + TTA OA %.3f kappa %.3f" % (rep["overall_accuracy"], rep["kappa"]))
cmap = ic.predict_chip_map(cnn, s2, size=9, stride=3, normalizer=n2)
ic.write_class_map(cmap, OUT / "s2_cnn.tif")

# ResNet18 transfer: ImageNet weights cannot be downloaded here; compare modes from scratch
res = ic.compare_transfer_modes(
    lambda: ic.build_cnn("resnet18", in_channels=10, num_classes=5),
    ic.as_loader(ic.ChipDataset(cx, cy, normalizer=n2), 64),
    ic.as_loader(ic.ChipDataset(tx, ty, normalizer=n2), 256, shuffle=False),
    epochs=2,
    modes=("linear_probe", "full"),
    region="UB",
)
print(res)

# --- Sentinel-1 VV/VH + Sentinel-2: early fusion (S1 on a different grid -> align) ------------
s1 = ic.load_stack(
    [D / "Sentinel_01_UB/Bands/VV.tif", D / "Sentinel_01_UB/Bands/VH.tif"], band_names=["VV", "VH"]
)
try:
    ic.fuse_early(s2, s1)
except ValueError as e:
    print("grid mismatch caught:", str(e)[:60])
fused = ic.fuse_early(s2, s1, align=True)
print("fused", fused.shape, list(fused.band.values)[-2:], fused.attrs)
Xf, yf, cf = ic.sample_pixels(
    fused,
    ic.labels_from_table(
        D / "Sentinel_2_UB/Label_Point_Data/train_data.csv", label="Class_ID", crs=s2.rio.crs
    ),
    "Class_ID",
)
Xft, yft, _ = ic.sample_pixels(fused, te_pts, "Class_ID")
clf_f, _ = ic.train_pixel_classifier(Xf, yf, model="rf", n_estimators=150)
clf_o, _ = ic.train_pixel_classifier(Xf.iloc[:, :10], yf, model="rf", n_estimators=150)
print(
    "RF optical OA %.3f | optical+SAR OA %.3f"
    % (
        ic.classification_report(yft, clf_o.predict(Xft.iloc[:, :10]))["overall_accuracy"],
        ic.classification_report(yft, clf_f.predict(Xft))["overall_accuracy"],
    )
)
pl = ic.fuse_late([clf_o.predict_proba(Xft.iloc[:, :10]), clf_f.predict_proba(Xft)])
print(
    "late fusion OA %.3f"
    % ic.classification_report(yft, clf_f.classes_[pl.argmax(1)])["overall_accuracy"]
)

# --- Self-supervised: MAE + SimCLR on unlabeled S2 chips, then linear probe ----------------
grid = np.stack(np.meshgrid(s2.x.values[5:-5:3], s2.y.values[5:-5:3]), -1).reshape(-1, 2)
ux, _, ucoords = ic.extract_chips(s2, grid, size=9)
ov = ic.check_geographic_overlap(
    ucoords, ic.sample_pixels(s2, te_pts, "Class_ID")[2], min_distance=20
)
print("unlabeled chips", ux.shape, "overlap with test points", ov)
loader = ic.as_loader(ic.ChipDataset(ux, normalizer=n2), 64)
mae, hm = ic.pretrain_masked(ic.build_mae(10, width=16, patch=3), loader, epochs=5, verbose=False)
enc, hs = ic.pretrain_contrastive(
    ic.make_encoder("simple_cnn", in_channels=10), loader, in_channels=10, epochs=3, verbose=False
)
for nm, model in (("MAE", mae), ("SimCLR", enc)):
    etr = ic.extract_embeddings(model, n2.transform(cx, 1))
    ete = ic.extract_embeddings(model, n2.transform(tx, 1))
    lp = ic.linear_probe(etr, cy, ete, ty)
    print(
        f"{nm} loss {np.round((hm if nm == 'MAE' else hs)['loss'], 3).tolist()} "
        f"linear-probe OA {lp['report']['overall_accuracy']:.3f}"
    )

# --- Change model on real Landsat chips with a simulated change ------------------------------
ls = ic.load_stack(D / "Landsat_BD/Images/multi.tif").isel(band=slice(1, 7)).fillna(0)
a = np.asarray(ls.values)[:, :256, :256]
chips_b = np.stack(
    [a[:, r : r + 32, c : c + 32] for r in range(0, 256, 32) for c in range(0, 256, 32)]
)
chips_a = chips_b.copy()
masks = np.zeros((len(chips_b), 32, 32), int)
rng = np.random.default_rng(1)
for i in range(len(chips_b)):
    r, c = rng.integers(0, 20, 2)
    chips_a[i, :, r : r + 12, c : c + 12] *= 1.8
    masks[i, r : r + 12, c : c + 12] = 1
nb = ic.Normalizer().fit(chips_b, band_axis=1)
pds = ic.PairChipDataset(chips_b, chips_a, masks, normalizer=nb, transform=ic.OverheadAugment())
for arch in ("siamese_diff", "bit"):
    cm = ic.build_change_model(arch, in_channels=6, num_classes=2)
    hc = ic.train_model(
        cm,
        ic.as_loader(pds, 8),
        ic.as_loader(pds, 8, shuffle=False),
        task="segmentation",
        epochs=6,
        lr=3e-3,
        verbose=False,
    )
    print(f"{arch} change mIoU {max(hc['val_score']):.3f}")

print(f"deep-learning workflows done in {time.time() - t0:.1f}s")
