"""Deep learning: tile datasets, segmentation/detection models, foundation models, training, inference,
point cloud segmentation (PointNet / PointNet++, :mod:`.pointnet`) and hyperspectral
classification (1-D/3-D CNN, HybridSN, spectral transformer, :mod:`.hyperspectral`).

Requires PyTorch (``pip install geolattice[rsense-dl]``); imports are lazy so the rest of
PyRSense works without it.
"""

from . import foundation, hyperspectral, models, pointnet
from .datasets import (
                       RasterTileDataset,
                       band_stats,
                       from_torchgeo,
                       grid_windows,
                       labels_from_vectors,
                       make_loader,
                       random_flip_rotate,
)
from .hyperspectral import (
                       CNN3D,
                       HSIPatchDataset,
                       HybridSN,
                       SpectralCNN1D,
                       SpectralTransformer,
                       block_split,
                       predict_hsi_map,
                       train_hsi_classifier,
)
from .infer import export_onnx, predict_detection, predict_segmentation
from .pointnet import (
                       PointBlockDataset,
                       PointNet2Seg,
                       PointNetSeg,
                       predict_point_segmentation,
                       train_point_segmentation,
)
from .models import UNet, adapt_first_conv, build_detector, build_segmentation_model
from .train import lightning_task, train_detection, train_segmentation

__all__ = [
                       "CNN3D",
                       "HSIPatchDataset",
                       "HybridSN",
                       "SpectralCNN1D",
                       "SpectralTransformer",
                       "block_split",
                       "hyperspectral",
                       "predict_hsi_map",
                       "train_hsi_classifier",
                       "PointBlockDataset",
                       "PointNet2Seg",
                       "PointNetSeg",
                       "pointnet",
                       "predict_point_segmentation",
                       "train_point_segmentation",
                       "RasterTileDataset",
                       "UNet",
                       "adapt_first_conv",
                       "band_stats",
                       "build_detector",
                       "build_segmentation_model",
                       "export_onnx",
                       "foundation",
                       "from_torchgeo",
                       "grid_windows",
                       "labels_from_vectors",
                       "lightning_task",
                       "make_loader",
                       "models",
                       "predict_detection",
                       "predict_segmentation",
                       "random_flip_rotate",
                       "train_detection",
                       "train_segmentation",
]
