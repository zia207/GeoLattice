"""Check that every trained checkpoint loads and gives the same output on CPU and GPU.

Run from the ``09_09_image_classification.ipynb`` folder::

    python -m models.check_weights            # all checkpoints in models/weights

For each checkpoint: load on the CPU, run a fixed input, then (when CUDA or MPS is present)
load on the GPU and compare the softmax outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from models import load_model, pick_device

HERE = Path(__file__).resolve().parent


def checkpoints(folder: Path = HERE / "weights") -> list[Path]:
    files = sorted(p for p in folder.glob("*.pt"))
    files += sorted(p for p in folder.iterdir() if p.is_dir() and (p / "index.json").exists())
    return files


def check(path: Path, size: int = 128) -> dict:
    torch.manual_seed(0)
    x = torch.rand(2, 3, size, size)
    cpu = load_model(path, device="cpu")
    with torch.inference_mode():
        p_cpu = cpu(x).softmax(1)
    assert p_cpu.shape == (2, cpu.num_classes, size, size)
    out = {"model": cpu.name, "cpu": "ok", "gpu": "not available"}
    gpu_dev = pick_device()
    if gpu_dev.type != "cpu":
        gpu = load_model(path, device=gpu_dev)
        with torch.inference_mode():
            p_gpu = gpu(x.to(gpu_dev)).softmax(1).cpu()
        diff = float((p_gpu - p_cpu).abs().max())
        out["gpu"] = f"{gpu_dev.type} ok (max |p_gpu - p_cpu| = {diff:.2e})"
        if diff > 1e-3:
            raise AssertionError(f"{path.name}: CPU and GPU outputs differ by {diff}")
    meta = cpu.metadata.get("test_metrics", {})
    out["test_boundary_iou"] = meta.get("boundary_iou")
    return out


def main() -> int:
    files = checkpoints()
    if not files:
        print("no checkpoints found in models/weights")
        return 1
    for f in files:
        r = check(f)
        iou = r["test_boundary_iou"]
        print(f"{r['model']:24} CPU {r['cpu']:3}  GPU {r['gpu']}  "
              f"test boundary IoU {iou:.3f}" if iou is not None else r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
