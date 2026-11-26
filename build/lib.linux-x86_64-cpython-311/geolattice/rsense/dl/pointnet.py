"""Deep learning on point clouds: block datasets, PointNet / PointNet++ (SSG) semantic
segmentation, training and whole-cloud inference with voting.

Pure PyTorch (no compiled CUDA ops), so it runs on CPU, CUDA and MPS.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._torch import require_torch, torch_device

DEFAULT_FEATURES = ("height", "intensity", "return_number", "number_of_returns")


# ----------------------------------------------------------------------------- dataset
def _block_index(x: np.ndarray, y: np.ndarray, block: float, stride: float):
    x0, y0 = x.min(), y.min()
    nx = max(1, int(np.ceil((x.max() - x0 - block) / stride)) + 1)
    ny = max(1, int(np.ceil((y.max() - y0 - block) / stride)) + 1)
    out = []
    for i in range(nx):
        for j in range(ny):
            bx, by = x0 + i * stride, y0 + j * stride
            m = np.flatnonzero((x >= bx) & (x < bx + block) & (y >= by) & (y < by + block))
            if m.size:
                out.append((bx, by, m))
    return out


def _point_features(pc, idx: np.ndarray, bx: float, by: float, block: float,
                    features: Sequence[str], stats: dict) -> np.ndarray:
    x = (np.asarray(pc.x)[idx] - bx) / block - 0.5
    y = (np.asarray(pc.y)[idx] - by) / block - 0.5
    z = np.asarray(pc.z)[idx]
    z = (z - z.min()) / max(block, 1e-6)
    cols = [x, y, z]
    for f in features:
        v = np.asarray(pc[f], float)[idx]
        mu, sd = stats[f]
        cols.append((v - mu) / sd)
    return np.stack(cols).astype("float32")


class PointBlockDataset:
    """Fixed-size point samples from square blocks of one or more PointClouds.

    Each item is ``{"points": (3 + F, N) float tensor, "labels": (N,) long tensor}`` where the
    first three channels are block-normalised x, y, z and the next ``features`` (those
    present) are standardised. ``label`` (e.g. ``"classification"``) is remapped to
    0..C-1; :attr:`classes` keeps the original codes. ``ignore`` codes map to -100.
    """

    def __init__(self, clouds, label: str | None = "classification", *, block_size: float = 20.0,
                 stride: float | None = None, num_points: int = 4096,
                 features: Sequence[str] = DEFAULT_FEATURES, classes: Sequence[int] | None = None,
                 ignore: Sequence[int] = (), samples_per_block: int = 1, seed: int = 0,
                 augment: bool = False, stats: dict | None = None):
        clouds = clouds if isinstance(clouds, (list, tuple)) else [clouds]
        self.clouds = [c.to_cpu() for c in clouds]
        self.label, self.block, self.n = label, float(block_size), int(num_points)
        self.stride = float(stride or block_size)
        self.features = [f for f in features if all(f in c for c in self.clouds)]
        self.augment, self.rng = augment, np.random.default_rng(seed)
        self.stats = stats or {f: (float(np.mean(np.concatenate([np.asarray(c[f], float)
                                                                  for c in self.clouds]))),
                                   float(np.std(np.concatenate([np.asarray(c[f], float)
                                                                for c in self.clouds])) or 1.0))
                               for f in self.features}
        if label is not None:
            allc = np.unique(np.concatenate([np.asarray(c[label]) for c in self.clouds]))
            self.classes = np.array(sorted(classes) if classes is not None
                                    else [c for c in allc if c not in set(ignore)])
            lut = {int(c): i for i, c in enumerate(self.classes)}
            self._maps = [np.array([lut.get(int(v), -100) for v in np.asarray(c[label])])
                          for c in self.clouds]
        else:
            self.classes, self._maps = None, None
        self.items = []
        for ci, c in enumerate(self.clouds):
            for bx, by, idx in _block_index(np.asarray(c.x), np.asarray(c.y), self.block,
                                            self.stride):
                if idx.size >= max(16, self.n // 16):
                    self.items += [(ci, bx, by, idx)] * samples_per_block

    @property
    def in_channels(self) -> int:
        return 3 + len(self.features)

    @property
    def num_classes(self) -> int:
        return 0 if self.classes is None else len(self.classes)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        torch = require_torch()
        ci, bx, by, idx = self.items[i]
        pick = self.rng.choice(idx, self.n, replace=idx.size < self.n)
        feats = _point_features(self.clouds[ci], pick, bx, by, self.block, self.features,
                                self.stats)
        if self.augment:
            a = self.rng.uniform(0, 2 * np.pi)
            c, s = np.cos(a), np.sin(a)
            x, y = feats[0].copy(), feats[1].copy()
            feats[0], feats[1] = c * x - s * y, s * x + c * y
            feats[:3] *= self.rng.uniform(0.9, 1.1)
            feats[:3] += self.rng.normal(0, 0.002, feats[:3].shape).astype("float32")
        out = {"points": torch.from_numpy(feats), "index": torch.from_numpy(pick)}
        if self._maps is not None:
            out["labels"] = torch.from_numpy(self._maps[ci][pick].astype("int64"))
        return out


# ----------------------------------------------------------------------------- models
def _ops():
    torch = require_torch()

    def fps(xyz, m):
        """Farthest point sampling. xyz (B, N, 3) → indices (B, m)."""
        B, N, _ = xyz.shape
        idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
        dist = torch.full((B, N), 1e10, device=xyz.device)
        far = torch.randint(0, N, (B,), device=xyz.device)
        ar = torch.arange(B, device=xyz.device)
        for i in range(m):
            idx[:, i] = far
            c = xyz[ar, far].unsqueeze(1)
            dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
            far = dist.argmax(-1)
        return idx

    def gather(x, idx):
        """x (B, N, C), idx (B, ...) → (B, ..., C)."""
        B = x.shape[0]
        flat = idx.reshape(B, -1)
        out = torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return out.reshape(*idx.shape, x.shape[-1])

    def ball_query(radius, k, xyz, new_xyz):
        d = torch.cdist(new_xyz, xyz)                    # (B, S, N)
        dk, ik = d.topk(k, dim=-1, largest=False)
        first = ik[..., :1].expand_as(ik)
        return torch.where(dk > radius, first, ik)

    return torch, fps, gather, ball_query


def _mlp(dims: Sequence[int], conv: str = "2d"):
    torch = require_torch()
    nn = torch.nn
    Conv = nn.Conv2d if conv == "2d" else nn.Conv1d
    BN = nn.BatchNorm2d if conv == "2d" else nn.BatchNorm1d
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [Conv(a, b, 1), BN(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def PointNetSeg(in_channels: int, num_classes: int, *, width: int = 64, dropout: float = 0.3):
    """Vanilla PointNet segmentation network (Qi et al. 2017a), without T-nets."""
    torch = require_torch()
    nn = torch.nn

    class _PointNetSeg(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.local = _mlp([in_channels, w, w], "1d")
            self.glob = _mlp([w, w * 2, w * 16], "1d")
            self.head = nn.Sequential(_mlp([w + w * 16, w * 8, w * 4, w * 2], "1d"),
                                      nn.Dropout(dropout), nn.Conv1d(w * 2, num_classes, 1))

        def forward(self, x):                            # x (B, C, N)
            loc = self.local(x)
            g = self.glob(loc).max(-1, keepdim=True)[0].expand(-1, -1, x.shape[-1])
            return self.head(torch.cat([loc, g], 1))

    return _PointNetSeg()


def PointNet2Seg(in_channels: int, num_classes: int, *, npoints: Sequence[int] = (1024, 256, 64),
                 radii: Sequence[float] = (0.1, 0.2, 0.4), nsample: int = 32, width: int = 32,
                 dropout: float = 0.5):
    """PointNet++ single-scale-grouping segmentation network (Qi et al. 2017b).

    Three set-abstraction levels (``npoints`` centroids, ball ``radii`` in block-normalised
    units) and three feature-propagation levels. Input (B, C, N) with xyz in the first three
    channels; output logits (B, classes, N).
    """
    torch, fps, gather, ball_query = _ops()
    nn = torch.nn
    extra = in_channels - 3
    w = width

    class SA(nn.Module):
        def __init__(self, m, r, cin, dims):
            super().__init__()
            self.m, self.r = m, r
            self.mlp = _mlp([cin + 3, *dims])

        def forward(self, xyz, feats):                   # xyz (B,N,3), feats (B,N,C) or None
            m = min(self.m, xyz.shape[1])
            cidx = fps(xyz, m)
            new_xyz = gather(xyz, cidx)
            gidx = ball_query(self.r, min(nsample, xyz.shape[1]), xyz, new_xyz)
            g_xyz = gather(xyz, gidx) - new_xyz.unsqueeze(2)       # (B, m, k, 3)
            g = torch.cat([g_xyz, gather(feats, gidx)], -1) if feats is not None else g_xyz
            g = self.mlp(g.permute(0, 3, 1, 2))                    # (B, C', m, k)
            return new_xyz, g.max(-1)[0].transpose(1, 2)           # (B, m, C')

    class FP(nn.Module):
        def __init__(self, cin, dims):
            super().__init__()
            self.mlp = _mlp([cin, *dims], "1d")

        def forward(self, xyz1, xyz2, f1, f2):
            d = torch.cdist(xyz1, xyz2)
            k = min(3, xyz2.shape[1])
            dk, ik = d.topk(k, dim=-1, largest=False)
            wgt = 1.0 / (dk + 1e-8)
            wgt = wgt / wgt.sum(-1, keepdim=True)
            interp = (gather(f2, ik) * wgt.unsqueeze(-1)).sum(2)   # (B, N1, C2)
            x = torch.cat([f1, interp], -1) if f1 is not None else interp
            return self.mlp(x.transpose(1, 2)).transpose(1, 2)

    class _PointNet2Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa1 = SA(npoints[0], radii[0], extra, [w, w, w * 2])
            self.sa2 = SA(npoints[1], radii[1], w * 2, [w * 2, w * 2, w * 4])
            self.sa3 = SA(npoints[2], radii[2], w * 4, [w * 4, w * 4, w * 8])
            self.fp3 = FP(w * 8 + w * 4, [w * 8, w * 8])
            self.fp2 = FP(w * 8 + w * 2, [w * 8, w * 4])
            self.fp1 = FP(w * 4 + extra, [w * 4, w * 4, w * 4])
            self.head = nn.Sequential(nn.Conv1d(w * 4, w * 4, 1), nn.BatchNorm1d(w * 4),
                                      nn.ReLU(inplace=True), nn.Dropout(dropout),
                                      nn.Conv1d(w * 4, num_classes, 1))

        def forward(self, x):
            xyz = x[:, :3].transpose(1, 2).contiguous()
            f0 = x[:, 3:].transpose(1, 2).contiguous() if extra > 0 else None
            l1x, l1f = self.sa1(xyz, f0)
            l2x, l2f = self.sa2(l1x, l1f)
            l3x, l3f = self.sa3(l2x, l2f)
            l2f = self.fp3(l2x, l3x, l2f, l3f)
            l1f = self.fp2(l1x, l2x, l1f, l2f)
            l0f = self.fp1(xyz, l1x, f0, l1f)
            return self.head(l0f.transpose(1, 2))

    return _PointNet2Seg()


# ----------------------------------------------------------------------------- training
def train_point_segmentation(model, train_loader, val_loader=None, *, epochs: int = 50,
                             lr: float = 1e-3, weight_decay: float = 1e-4,
                             num_classes: int | None = None, class_weights: Any = None,
                             device: str | None = None, log_every: int = 1) -> dict:
    """Train a point segmentation model with cross-entropy (labels -100 are ignored).

    Keeps the best weights by validation mIoU. Returns a history of losses and metrics.
    """
    from .train import _confusion, _seg_metrics

    torch = require_torch()
    dev = torch_device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    w = torch.as_tensor(class_weights, dtype=torch.float32, device=dev) \
        if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    hist: dict[str, list] = {"train_loss": [], "val_loss": [], "val_miou": [], "val_accuracy": []}
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in train_loader:
            x, y = b["points"].to(dev).float(), b["labels"].to(dev).long()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        sched.step()
        hist["train_loss"].append(tot / max(n, 1))
        if val_loader is not None:
            model.eval()
            vl, vn, conf = 0.0, 0, None
            with torch.no_grad():
                for b in val_loader:
                    x, y = b["points"].to(dev).float(), b["labels"].to(dev).long()
                    out = model(x)
                    vl += float(loss_fn(out, y)) * x.shape[0]
                    vn += x.shape[0]
                    k = num_classes or out.shape[1]
                    c = _confusion(out.argmax(1).flatten(), y.flatten(), k, -100)
                    conf = c if conf is None else conf + c
            m = _seg_metrics(conf)
            hist["val_loss"].append(vl / max(vn, 1))
            hist["val_miou"].append(m["miou"])
            hist["val_accuracy"].append(m["accuracy"])
            if m["miou"] > best:
                best = m["miou"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_every and (ep + 1) % log_every == 0:
            msg = f"epoch {ep + 1}/{epochs} loss {hist['train_loss'][-1]:.4f}"
            if hist["val_miou"]:
                msg += f" val mIoU {hist['val_miou'][-1]:.3f}"
            print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def predict_point_segmentation(model, pc, dataset: PointBlockDataset, *, field: str = "pred",
                               votes: int = 1, batch_size: int = 8, device: str | None = None,
                               seed: int = 0):
    """Label every point of ``pc`` by sweeping blocks (same size/features/statistics as the
    training ``dataset``) and averaging softmax scores over overlapping blocks and ``votes``
    random passes. Adds ``field`` (original class codes) and ``{field}_prob``."""
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev).eval()
    p = pc.to_cpu()
    n = len(p)
    C = dataset.num_classes or None
    rng = np.random.default_rng(seed)
    acc = None
    blocks = _block_index(np.asarray(p.x), np.asarray(p.y), dataset.block,
                          dataset.stride if dataset.stride < dataset.block else dataset.block / 2)
    with torch.no_grad():
        batch, bidx = [], []

        def flush():
            nonlocal acc
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch)).to(dev)
            prob = torch.softmax(model(x), 1).cpu().numpy()      # (B, C, N)
            if acc is None:
                acc = np.zeros((n, prob.shape[1]), "float32")
            for pr, ix in zip(prob, bidx):
                np.add.at(acc, ix, pr.T)
            batch.clear()
            bidx.clear()

        for _ in range(votes):
            for bx, by, idx in blocks:
                perm = rng.permutation(idx)
                for s in range(0, len(perm), dataset.n):
                    ix = perm[s:s + dataset.n]
                    if len(ix) < dataset.n:
                        ix = np.concatenate([ix, rng.choice(idx, dataset.n - len(ix))])
                    batch.append(_point_features(p, ix, bx, by, dataset.block, dataset.features,
                                                 dataset.stats))
                    bidx.append(ix)
                    if len(batch) == batch_size:
                        flush()
        flush()
    if acc is None:
        acc = np.zeros((n, C or 1), "float32")
    s = acc.sum(1, keepdims=True)
    prob = np.where(s > 0, acc / np.maximum(s, 1e-12), 0)
    lab = prob.argmax(1)
    codes = dataset.classes[lab] if dataset.classes is not None else lab
    return pc.with_field(field, codes.astype("uint8") if codes.max() < 256 else codes).with_field(
        f"{field}_prob", prob.max(1).astype("float32"))
