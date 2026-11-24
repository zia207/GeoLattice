# Trained crop-boundary weights (WorldView-2, Bangladesh)

PyTorch weights for six of the 28 networks in `models/`, trained on the tutorial data in
`Data/08_remote_sensing/08_10_object_detection/WorldView2_BD`. They load on a CUDA GPU, on Apple Silicon (MPS) or on the CPU: the
tensors are stored on the CPU and `load_model` moves the network to the best available device
(or the one you name).

```python
from models import load_model, load_metadata

model = load_model("models/weights/FC-DenseNet56.pt")                 # GPU if present, else CPU
model = load_model("models/weights/DeepLabV3_plus-Res50", device="cpu")  # shard folder
meta = load_metadata("models/weights/GCN-Res50")                      # classes, stretch, metrics
```

Command line, from the `07_08_image_classification.ipynb` folder:

```bash
python -m models.check_weights                                      # load all on CPU and GPU, compare
python -m models.predict models/weights/FC-DenseNet56.pt --out boundary.tif
python -m models.train --models all --epochs 60 --batch-size 16      # retrain (best on a GPU)
```

## Task and inputs

- **Classes:** 0 = field interior, 1 = field boundary (from `boundary_raster.tif`; nodata = unlabeled, ignored).
- **Input:** NCHW float32, 3 bands in the order red (WV-2 band 5), green (band 3), blue (band 2), resampled bilinearly to the 1 m label grid.
- **Normalization:** `(x - low) / (high - low)` clipped to [0, 1], with the per-band 2nd / 98th percentiles of the training tiles stored in each checkpoint (`metadata["input"]["stretch_low"]`, `["stretch_high"]`). `models.predict` applies it for you.
- **Output:** logits `(N, 2, H, W)`; take `softmax(1)[:, 1]` for the boundary probability.

## Training

- 113 tiles of 128 × 128 px (stride 96) with at least 50% labeled pixels, split by 512 × 512 px spatial blocks into 84 train, 8 validation and 21 test tiles, so test tiles come from areas the models never saw.
- Random 90° rotations and flips; class-weighted cross-entropy (inverse frequency) + Dice loss; AdamW (lr 5e-4), cosine schedule, gradient clipping; best epoch chosen on the validation boundary F1 (2 px tolerance), early stopping after 12 epochs with patience 8.
- Trained from scratch on CPU (2 threads), at most 40 epochs or 15 minutes per model. These are baseline weights. A GPU run with more epochs (`--epochs 60`, larger batches, mixed precision is automatic on CUDA) should do better, especially for the ResNet-based models.

## Test results (21 held-out tiles)

Boundaries are 1 px wide, so pixel-exact scores are low by nature; the field-boundary literature
usually reports F1 with a small distance tolerance, given here as **F1 @ 2 px**.

| Model | Epochs | Boundary F1 @ 2 px | Precision @ 2 px | Recall @ 2 px | Strict boundary F1 | Strict boundary IoU | Field IoU | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FC-DenseNet56** | 19 | **0.780** | 0.757 | 0.803 | 0.336 | 0.202 | 0.838 | 5.7 MB (1 file) |
| FC-DenseNet103 | 13 | 0.768 | 0.712 | 0.834 | 0.325 | 0.194 | 0.817 | 38 MB (3 shards) |
| MobileUNet-Skip | 40 | 0.743 | 0.632 | 0.902 | 0.293 | 0.171 | 0.777 | 36 MB (3 shards) |
| GCN-Res50 | 24 | 0.721 | 0.607 | 0.886 | 0.269 | 0.155 | 0.733 | 96 MB (6 shards) |
| DeepLabV3_plus-Res50 | 30 | 0.675 | 0.567 | 0.835 | 0.223 | 0.126 | 0.675 | 116 MB (8 shards) |
| ICNet-Res50 | 21 | 0.510 | 0.397 | 0.715 | 0.150 | 0.081 | 0.498 | 110 MB (7 shards) |

FC-DenseNet56 is the recommended starting point. Full metrics and per-epoch history are in
`metrics.json` and inside each checkpoint (`load_metadata(path)["history"]`).

## Checkpoint format

- Single file `<model>.pt`: `{"name", "num_classes", "metadata", "state_dict"}`.
- Shard folder `<model>/`: `index.json` (name, classes, metadata, shard list and tensor-to-shard map) and `shard-00000.pt`, `shard-00001.pt`, ... (each under 19 MB). `save_model(..., max_shard_mb=19)` writes this form for large models; pass `max_shard_mb=None` for one file.

The other 22 architectures (larger ResNet-101/152 variants, RefineNet, PSPNet, FRRN, AdapNet,
Encoder-Decoder, MobileUNet, FC-DenseNet67) have no trained weights here; train them with
`python -m models.train --models <name>` on a GPU.
