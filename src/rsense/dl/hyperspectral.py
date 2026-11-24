"""Deep learning for hyperspectral images: patch datasets with leakage-safe block splits,
1-D spectral CNN, 3-D CNN, HybridSN and a spectral transformer (SpectralFormer-style group
tokens), training and full-image prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._torch import require_torch, torch_device


def block_split(labels: Any, *, block: int = 16, test_fraction: float = 0.3, seed: int = 0,
                ignore: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split labelled pixels into train/test masks by whole spatial blocks of ``block`` pixels,
    so neighbouring (overlapping) patches never fall on both sides."""
    lab = np.asarray(getattr(labels, "values", labels))
    H, W = lab.shape
    by, bx = np.arange(H) // block, np.arange(W) // block
    bid = by[:, None] * (bx.max() + 1) + bx[None, :]
    ids = np.unique(bid[lab != ignore])
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, max(1, int(round(len(ids) * test_fraction))), replace=False)
    test = np.isin(bid, test_ids) & (lab != ignore)
    return (lab != ignore) & ~test, test


class HSIPatchDataset:
    """Patches ``(1, bands, p, p)`` (or spectra ``(bands,)`` with ``patch=1, flat=True``)
    centred on labelled pixels. ``labels`` is a (y, x) array with class codes; ``ignore``
    (default 0) marks unlabelled pixels. Codes are remapped to 0..C-1 (:attr:`classes`).
    Spectra are standardised per band with statistics from ``stats`` or the whole cube."""

    def __init__(self, cube, labels, *, patch: int = 9, mask: Any = None, ignore: int = 0,
                 classes: Any = None, stats: tuple | None = None, flat: bool = False,
                 augment: bool = False, seed: int = 0):
        a = np.asarray(getattr(cube, "values", cube), dtype="float32")
        if stats is None:
            mu = np.nanmean(a, axis=(1, 2))
            sd = np.nanstd(a, axis=(1, 2))
            stats = (mu, np.where(sd > 0, sd, 1))
        self.stats = stats
        a = (a - stats[0][:, None, None]) / stats[1][:, None, None]
        self.r = patch // 2
        self.a = np.pad(np.nan_to_num(a), ((0, 0), (self.r, self.r), (self.r, self.r)),
                        mode="reflect")
        lab = np.asarray(getattr(labels, "values", labels))
        sel = lab != ignore
        if mask is not None:
            sel &= np.asarray(getattr(mask, "values", mask), bool)
        self.rows, self.cols = np.nonzero(sel)
        self.classes = np.array(sorted(np.unique(lab[sel]))) if classes is None \
            else np.asarray(classes)
        lut = {int(c): i for i, c in enumerate(self.classes)}
        self.y = np.array([lut.get(int(v), -100) for v in lab[self.rows, self.cols]])
        self.patch, self.flat, self.augment = patch, flat, augment
        self.rng = np.random.default_rng(seed)

    @property
    def bands(self) -> int:
        return self.a.shape[0]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.rows)

    def _patch(self, r: int, c: int) -> np.ndarray:
        return self.a[:, r:r + self.patch, c:c + self.patch]

    def __getitem__(self, i: int) -> dict[str, Any]:
        torch = require_torch()
        p = self._patch(self.rows[i], self.cols[i])
        if self.augment and not self.flat:
            p = np.rot90(p, self.rng.integers(4), axes=(1, 2))
            if self.rng.random() < 0.5:
                p = p[:, :, ::-1]
        x = p[:, self.r, self.r] if self.flat else p[None]
        return {"image": torch.from_numpy(np.ascontiguousarray(x, dtype="float32")),
                "label": torch.tensor(int(self.y[i]))}


# ----------------------------------------------------------------------------- models
def SpectralCNN1D(bands: int, num_classes: int, *, width: int = 32, dropout: float = 0.3):
    """1-D CNN on single-pixel spectra (input ``(B, bands)``)."""
    torch = require_torch()
    nn = torch.nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.f = nn.Sequential(
                nn.Conv1d(1, w, 7, padding=3), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(w, 2 * w, 5, padding=2), nn.BatchNorm1d(2 * w), nn.ReLU(),
                nn.MaxPool1d(2), nn.Conv1d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm1d(4 * w),
                nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                      nn.Linear(4 * w, num_classes))

        def forward(self, x):
            if x.dim() == 2:
                x = x[:, None]
            return self.head(self.f(x))

    return _Net()


def CNN3D(bands: int, num_classes: int, *, width: int = 8, dropout: float = 0.4):
    """Compact 3-D CNN on ``(B, 1, bands, p, p)`` patches (joint spectral–spatial kernels)."""
    torch = require_torch()
    nn = torch.nn
    w = width
    return nn.Sequential(
        nn.Conv3d(1, w, (7, 3, 3), padding=(3, 1, 1)), nn.BatchNorm3d(w), nn.ReLU(),
        nn.MaxPool3d((2, 1, 1)),
        nn.Conv3d(w, 2 * w, (5, 3, 3), padding=(2, 1, 1)), nn.BatchNorm3d(2 * w), nn.ReLU(),
        nn.MaxPool3d((2, 2, 2), ceil_mode=True),
        nn.Conv3d(2 * w, 4 * w, (3, 3, 3), padding=1), nn.BatchNorm3d(4 * w), nn.ReLU(),
        nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, num_classes))


def HybridSN(bands: int, num_classes: int, *, patch: int = 9, width: int = 8,
             dropout: float = 0.4):
    """HybridSN (Roy et al. 2020): three 3-D convolutions followed by a 2-D convolution.
    Works best on 15–30 PCA/MNF components rather than all bands."""
    torch = require_torch()
    nn = torch.nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.c3 = nn.Sequential(
                nn.Conv3d(1, w, (7, 3, 3), padding=(0, 1, 1)), nn.ReLU(),
                nn.Conv3d(w, 2 * w, (5, 3, 3), padding=(0, 1, 1)), nn.ReLU(),
                nn.Conv3d(2 * w, 4 * w, (3, 3, 3), padding=(0, 1, 1)), nn.ReLU())
            d = bands - 6 - 4 - 2
            if d < 1:
                raise ValueError("HybridSN needs at least 13 bands/components")
            self.c2 = nn.Sequential(nn.Conv2d(4 * w * d, 8 * w, 3, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w, 32 * w), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32 * w, num_classes))

        def forward(self, x):
            y = self.c3(x)
            b, c, d, h, w_ = y.shape
            return self.head(self.c2(y.reshape(b, c * d, h, w_)))

    return _Net()


def SpectralTransformer(bands: int, num_classes: int, *, group: int = 5, dim: int = 64,
                        depth: int = 4, heads: int = 4, patch: int = 1, dropout: float = 0.1):
    """Transformer over groups of neighbouring bands (SpectralFormer-style group-wise
    embedding, Hong et al. 2022). Input: spectra ``(B, bands)`` or patches
    ``(B, 1, bands, p, p)`` (the patch is flattened into each band group)."""
    torch = require_torch()
    nn = torch.nn
    ntok = int(np.ceil(bands / group))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(group * patch * patch, dim)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, ntok + 1, dim) * 0.02)
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, depth)
            self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        def forward(self, x):
            if x.dim() == 2:
                x = x[:, :, None, None]
            elif x.dim() == 5:
                x = x[:, 0]
            b = x.shape[0]
            pad = ntok * group - x.shape[1]
            if pad:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
            tok = x.reshape(b, ntok, group * x.shape[2] * x.shape[3])
            z = torch.cat([self.cls.expand(b, -1, -1), self.embed(tok)], 1) + self.pos
            return self.head(self.enc(z)[:, 0])

    return _Net()


# ----------------------------------------------------------------------------- training
def train_hsi_classifier(model, train_loader, val_loader=None, *, epochs: int = 50,
                         lr: float = 1e-3, weight_decay: float = 1e-4, class_weights: Any = None,
                         device: str | None = None, log_every: int = 0) -> dict:
    """Cross-entropy training; keeps the weights with the best validation accuracy."""
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    w = torch.as_tensor(class_weights, dtype=torch.float32, device=dev) \
        if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    hist: dict[str, list] = {"train_loss": [], "val_accuracy": [], "val_kappa": []}
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in train_loader:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        hist["train_loss"].append(tot / max(n, 1))
        if val_loader is not None:
            yt, yp = evaluate(model, val_loader, device=device)
            acc = float((yt == yp).mean())
            hist["val_accuracy"].append(acc)
            hist["val_kappa"].append(_kappa(yt, yp))
            if acc > best:
                best = acc
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_every and (ep + 1) % log_every == 0:
            print(f"epoch {ep + 1}: loss {hist['train_loss'][-1]:.4f}"
                  + (f" val acc {hist['val_accuracy'][-1]:.3f}" if hist["val_accuracy"] else ""))
    if state is not None:
        model.load_state_dict(state)
    return hist


def _kappa(y, p) -> float:
    labels = np.unique(np.r_[y, p])
    po = float((y == p).mean())
    pe = sum(float((y == c).mean() * (p == c).mean()) for c in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def evaluate(model, loader, *, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(true, predicted)`` class indices over a loader."""
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev).eval()
    yt, yp = [], []
    with torch.no_grad():
        for b in loader:
            yp.append(model(b["image"].to(dev)).argmax(1).cpu().numpy())
            yt.append(b["label"].numpy())
    return np.concatenate(yt), np.concatenate(yp)


def predict_hsi_map(model, cube, dataset: HSIPatchDataset, *, batch_size: int = 512,
                    mask: Any = None, device: str | None = None, proba: bool = False):
    """Classify every valid pixel of ``cube`` with the patch size, normalisation and class codes
    of ``dataset``. Returns a DataArray of original class codes (0 where masked/NaN), and with
    ``proba=True`` also the max softmax probability."""
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev).eval()
    src = HSIPatchDataset(cube, np.ones(cube.shape[1:], int), patch=dataset.patch, ignore=0,
                          classes=[1], stats=dataset.stats, flat=dataset.flat)
    a = np.asarray(cube.values)
    ok = np.isfinite(a).all(0)
    if mask is not None:
        ok &= np.asarray(getattr(mask, "values", mask), bool)
    rr, cc = np.nonzero(ok)
    pred = np.zeros(a.shape[1:], dtype="int32")
    prob = np.full(a.shape[1:], np.nan, dtype="float32")
    with torch.no_grad():
        for s in range(0, len(rr), batch_size):
            r, c = rr[s:s + batch_size], cc[s:s + batch_size]
            P = np.stack([src._patch(i, j) for i, j in zip(r, c)])
            x = P[:, :, src.r, src.r] if src.flat else P[:, None]
            out = torch.softmax(model(torch.from_numpy(np.ascontiguousarray(x, "float32"))
                                      .to(dev)), 1).cpu().numpy()
            pred[r, c] = dataset.classes[out.argmax(1)]
            prob[r, c] = out.max(1)
    tpl = cube.isel(band=0, drop=True)
    tpl = tpl.drop_vars([v for v in ("wavelength", "fwhm", "good") if v in tpl.coords])
    da = tpl.copy(data=pred).rename("class")
    if proba:
        return da, tpl.copy(data=prob).rename("probability")
    return da
