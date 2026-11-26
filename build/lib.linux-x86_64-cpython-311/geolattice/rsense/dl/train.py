"""Training loops: Lightning (multi-GPU, AMP) when available, or a compact PyTorch loop."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._torch import require_torch, torch_device


def _seg_metrics(conf: np.ndarray) -> dict[str, float]:
    tp = np.diag(conf)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)
    return {"miou": float(np.nanmean(iou)), "accuracy": float(tp.sum() / max(conf.sum(), 1))}


def _confusion(pred, target, n, ignore):
    torch = require_torch()
    m = target != ignore
    idx = target[m] * n + pred[m]
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy()


def train_segmentation(model, train_loader, val_loader=None, *, epochs: int = 20, lr: float = 1e-3,
                       weight_decay: float = 1e-4, num_classes: int | None = None,
                       ignore_index: int = -100, class_weights: Any = None,
                       precision: Literal["32", "16", "bf16"] = "32", device: str | None = None,
                       scheduler: Literal["cosine", "none"] = "cosine",
                       log_every: int = 1, checkpoint: str | None = None) -> dict:
    """Train a segmentation model with cross-entropy (plain PyTorch, single device).

    Mixed precision via ``precision="bf16"``/``"16"``. Returns the history of loss/mIoU and
    keeps the best weights (by validation mIoU) in the model at the end.
    """
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if scheduler == "cosine" else None
    w = torch.as_tensor(class_weights, dtype=torch.float32, device=dev) if class_weights is not None \
        else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
    amp_dtype = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler(enabled=precision == "16" and dev.type == "cuda")
    hist: dict[str, list] = {"train_loss": [], "val_loss": [], "val_miou": []}
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            x, y = batch["image"].to(dev).float(), batch["mask"].to(dev).long()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        if sched:
            sched.step()
        hist["train_loss"].append(tot / max(n, 1))
        if val_loader is not None:
            vl, vn, conf = 0.0, 0, None
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch["image"].to(dev).float(), batch["mask"].to(dev).long()
                    with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        out = model(x)
                        vl += float(loss_fn(out, y)) * x.shape[0]
                    vn += x.shape[0]
                    k = num_classes or out.shape[1]
                    c = _confusion(out.argmax(1), y, k, ignore_index)
                    conf = c if conf is None else conf + c
            m = _seg_metrics(conf)
            hist["val_loss"].append(vl / max(vn, 1))
            hist["val_miou"].append(m["miou"])
            if m["miou"] > best:
                best = m["miou"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_every and (ep + 1) % log_every == 0:
            msg = f"epoch {ep + 1}/{epochs} train_loss={hist['train_loss'][-1]:.4f}"
            if val_loader is not None:
                msg += f" val_loss={hist['val_loss'][-1]:.4f} val_mIoU={hist['val_miou'][-1]:.3f}"
            print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint:
            torch.save(best_state, checkpoint)
    return hist


def train_detection(model, train_loader, val_loader=None, *, epochs: int = 20, lr: float = 1e-4,
                    device: str | None = None, precision: Literal["32", "16", "bf16"] = "32") -> dict:
    """Train a torchvision detector. Batches must be ``(images: list[Tensor], targets: list[dict])``
    with ``boxes`` (xyxy pixels) and ``labels`` per target (see torchvision detection docs)."""
    torch = require_torch()
    dev = torch_device(device)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    amp_dtype = {"16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    hist = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for images, targets in train_loader:
            images = [i.to(dev).float() for i in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        hist["train_loss"].append(tot / max(n, 1))
        if val_loader is not None:
            vl, vn = 0.0, 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [i.to(dev).float() for i in images]
                    targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
                    vl += float(sum(model(images, targets).values()))  # train mode returns losses
                    vn += 1
            hist["val_loss"].append(vl / max(vn, 1))
        print(f"epoch {ep + 1}/{epochs} loss={hist['train_loss'][-1]:.4f}")
    return hist


def lightning_task(model, *, task: Literal["segmentation"] = "segmentation", num_classes: int,
                   lr: float = 1e-3, ignore_index: int = -100):
    """Wrap a model in a LightningModule for multi-GPU / multi-node training::

        task = rs.dl.lightning_task(model, num_classes=6)
        L.Trainer(devices="auto", precision="bf16-mixed", max_epochs=50).fit(task, train, val)
    """
    torch = require_torch()
    try:
        import lightning as L
    except ImportError as e:
        raise ImportError("lightning_task needs `pip install lightning`") from e

    class SegTask(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        def forward(self, x):
            return self.model(x)

        def _step(self, batch, stage):
            out = self(batch["image"].float())
            loss = self.loss(out, batch["mask"].long())
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            if stage == "val":
                conf = _confusion(out.argmax(1), batch["mask"].long(), num_classes, ignore_index)
                self.log("val_miou", _seg_metrics(conf)["miou"], prog_bar=True, sync_dist=True)
            return loss

        def training_step(self, batch, _):
            return self._step(batch, "train")

        def validation_step(self, batch, _):
            return self._step(batch, "val")

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
            return {"optimizer": opt,
                    "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, self.trainer.max_epochs or 10)}

    if task != "segmentation":
        raise NotImplementedError("Only segmentation tasks are wrapped; use train_detection")
    return SegTask()
