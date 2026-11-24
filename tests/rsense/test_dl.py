"""Deep-learning tests run only where PyTorch is installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geolattice.rsense import dl


def test_tile_dataset_and_unet(reflectance):
    img = reflectance.isel(time=0)[["red", "green", "blue", "nir"]]
    labels = (img.nir > 0.35).astype("int64").rename("labels")
    ds = dl.RasterTileDataset(img, labels, size=16, stride=8)
    s = ds[0]
    assert s["image"].shape == (4, 16, 16) and s["mask"].shape == (16, 16)
    model = dl.UNet(4, 2, base=8, depth=2)
    hist = dl.train_segmentation(model, dl.make_loader(ds, 4), dl.make_loader(ds, 4, shuffle=False),
                                 epochs=2, log_every=0)
    assert len(hist["train_loss"]) == 2
    pred = dl.predict_segmentation(model, img, tile=16, overlap=4, normalize=(ds.mean, ds.std))
    assert pred.shape == labels.shape and set(np.unique(pred.values)) <= {0, 1}


def test_adapt_first_conv():
    conv = torch.nn.Conv2d(3, 8, 3)
    new = dl.adapt_first_conv(conv, 6)
    assert new.weight.shape == (8, 6, 3, 3)
