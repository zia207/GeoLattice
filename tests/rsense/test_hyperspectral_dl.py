"""Hyperspectral deep learning (1-D/3-D CNN, HybridSN, spectral transformer) — needs PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geolattice.rsense import dl  # noqa: E402
from geolattice.rsense import hyperspectral as hs  # noqa: E402

WL = np.arange(400.0, 2501.0, 20.0)


def _labelled(seed=0, H=48, W=48):
    """Four homogeneous class blocks (library spectra with brightness variation + noise)."""
    rng = np.random.default_rng(seed)
    lib = hs.synthetic_library(WL)
    S = lib.subset(["green_vegetation", "soil", "water", "clay_kaolinite"]).spectra
    lab = np.zeros((H, W), int)
    lab[: H // 2, : W // 2], lab[: H // 2, W // 2:] = 1, 2
    lab[H // 2:, : W // 2], lab[H // 2:, W // 2:] = 3, 4
    cube = S[lab - 1].transpose(2, 0, 1) * rng.uniform(0.85, 1.15, (1, H, W))
    cube += rng.normal(0, 0.01, cube.shape)
    labels = lab.copy()
    labels[rng.random((H, W)) > 0.4] = 0          # only 40% labelled
    return hs.as_cube(cube.astype("float32"), WL), labels, lab


def test_block_split_and_dataset():
    cube, labels, _ = _labelled()
    tr, te = dl.block_split(labels, block=8, test_fraction=0.3)
    assert not (tr & te).any() and (tr | te).sum() == (labels > 0).sum()
    by = np.arange(48) // 8
    blk = by[:, None] * 6 + by[None, :]
    assert not set(blk[tr]) & set(blk[te])        # no shared blocks
    ds = dl.HSIPatchDataset(cube, labels, patch=7, mask=tr, augment=True)
    s = ds[0]
    assert s["image"].shape == (1, WL.size, 7, 7) and ds.num_classes == 4
    assert list(ds.classes) == [1, 2, 3, 4] and int(s["label"]) in range(4)
    flat = dl.HSIPatchDataset(cube, labels, patch=1, flat=True)
    assert flat[0]["image"].shape == (WL.size,)


@pytest.mark.parametrize("name", ["cnn1d", "cnn3d", "hybridsn", "transformer"])
def test_models_forward(name):
    B, p = WL.size, 7
    x5 = torch.randn(2, 1, B, p, p)
    if name == "cnn1d":
        m, x = dl.SpectralCNN1D(B, 4, width=8), torch.randn(2, B)
    elif name == "cnn3d":
        m, x = dl.CNN3D(B, 4, width=4), x5
    elif name == "hybridsn":
        m, x = dl.HybridSN(20, 4, patch=p, width=4), torch.randn(2, 1, 20, p, p)
    else:
        m, x = dl.SpectralTransformer(B, 4, group=5, dim=32, depth=2, patch=p), x5
    out = m(x)
    assert out.shape == (2, 4)
    out.sum().backward()
    if name == "transformer":
        flat = dl.SpectralTransformer(B, 3, dim=16, depth=1)
        assert flat(torch.randn(3, B)).shape == (3, 3)
    with pytest.raises(ValueError):
        dl.HybridSN(10, 4)


@pytest.mark.parametrize("arch", ["cnn1d", "cnn3d"])
def test_train_and_predict_map(arch):
    torch.manual_seed(0)
    cube, labels, full = _labelled()
    tr, te = dl.block_split(labels, block=8, test_fraction=0.3, seed=1)
    flat = arch == "cnn1d"
    p = 1 if flat else 5
    dtr = dl.HSIPatchDataset(cube, labels, patch=p, mask=tr, flat=flat)
    dte = dl.HSIPatchDataset(cube, labels, patch=p, mask=te, flat=flat, stats=dtr.stats,
                             classes=dtr.classes)
    model = dl.SpectralCNN1D(WL.size, 4, width=8) if flat else dl.CNN3D(WL.size, 4, width=4)
    L = torch.utils.data.DataLoader
    hist = dl.train_hsi_classifier(model, L(dtr, batch_size=64, shuffle=True),
                                   L(dte, batch_size=128), epochs=6, lr=3e-3)
    assert hist["train_loss"][-1] < hist["train_loss"][0] and max(hist["val_accuracy"]) > 0.9
    yt, yp = dl.hyperspectral.evaluate(model, L(dte, batch_size=128))
    assert (yt == yp).mean() > 0.9
    pred, prob = dl.predict_hsi_map(model, cube, dtr, proba=True)
    assert pred.shape == (48, 48) and set(np.unique(pred.values)) <= {1, 2, 3, 4}
    assert (pred.values == full).mean() > 0.9 and float(prob.min()) > 0
