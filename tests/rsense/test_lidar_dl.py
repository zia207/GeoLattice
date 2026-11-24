"""Point cloud deep learning (PointNet / PointNet++) — runs only where PyTorch is installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geolattice.rsense import dl  # noqa: E402
from geolattice.rsense.core import PointCloud  # noqa: E402


def _labelled_scene(seed=0, n=12_000):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 40, n), rng.uniform(0, 40, n)
    z = 100 + 0.02 * x
    lab = np.full(n, 2, "uint8")
    bld = (x > 5) & (x < 15) & (y > 5) & (y < 15)
    z[bld] += 6
    lab[bld] = 6
    tree = np.hypot(x - 28, y - 28) < 6
    z[tree] += rng.uniform(3, 10, tree.sum())
    lab[tree] = 5
    h = (z - (100 + 0.02 * x)).astype("float32")
    return PointCloud.from_arrays(x + 4e5, y + 4.7e6, z, crs=32618, classification=lab,
                                  height=h, intensity=rng.integers(0, 300, n).astype("uint16"))


def test_point_block_dataset():
    pc = _labelled_scene()
    ds = dl.PointBlockDataset(pc, block_size=20, num_points=512, features=("height", "intensity"))
    assert len(ds) == 4 and ds.in_channels == 5 and ds.num_classes == 3
    assert list(ds.classes) == [2, 5, 6]
    s = ds[0]
    assert s["points"].shape == (5, 512) and s["labels"].shape == (512,)
    assert float(s["points"][0].abs().max()) <= 0.5 + 1e-6
    assert set(s["labels"].unique().tolist()) <= {0, 1, 2}
    ign = dl.PointBlockDataset(pc, block_size=20, num_points=256, ignore=(5,))
    assert ign.num_classes == 2 and (ign[3]["labels"] == -100).any()


@pytest.mark.parametrize("arch", ["pointnet", "pointnet2"])
def test_models_forward(arch):
    x = torch.randn(2, 5, 512)
    if arch == "pointnet":
        m = dl.PointNetSeg(5, 3, width=16)
    else:
        m = dl.PointNet2Seg(5, 3, npoints=(128, 32, 8), radii=(0.1, 0.2, 0.4), nsample=16,
                            width=8)
    out = m(x)
    assert out.shape == (2, 3, 512)
    out.sum().backward()
    xyz_only = dl.PointNet2Seg(3, 4, npoints=(64, 16, 4), width=8)
    assert xyz_only(torch.randn(1, 3, 256)).shape == (1, 4, 256)


def test_train_and_predict_pointnet2():
    torch.manual_seed(0)
    pc = _labelled_scene()
    ds = dl.PointBlockDataset(pc, block_size=20, stride=10, num_points=512,
                              features=("height",), samples_per_block=4, augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    val = torch.utils.data.DataLoader(dl.PointBlockDataset(pc, block_size=20, num_points=512,
                                                           features=("height",), stats=ds.stats,
                                                           classes=ds.classes), batch_size=4)
    model = dl.PointNet2Seg(ds.in_channels, ds.num_classes, npoints=(128, 32, 8), nsample=16,
                            width=16)
    hist = dl.train_point_segmentation(model, loader, val, epochs=12, lr=3e-3, log_every=0)
    assert len(hist["train_loss"]) == 12 and hist["train_loss"][-1] < hist["train_loss"][0]
    assert max(hist["val_accuracy"]) > 0.8
    out = dl.predict_point_segmentation(model, pc, ds, batch_size=8)
    pred = np.asarray(out["pred"])
    assert set(np.unique(pred)) <= {2, 5, 6}
    assert (pred == np.asarray(pc["classification"])).mean() > 0.8
    assert float(np.min(out["pred_prob"])) >= 0 and float(np.max(out["pred_prob"])) <= 1
