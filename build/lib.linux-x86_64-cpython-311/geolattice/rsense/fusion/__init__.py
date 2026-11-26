"""Data fusion: optical–SAR, STARFM spatiotemporal fusion, lidar–optical modelling, UAV→satellite upscaling."""

from .lidar_optical import compare, train_wall_to_wall, upscale
from .optical_sar import sar_gap_fill, stack_features
from .starfm import starfm

__all__ = ["compare", "sar_gap_fill", "stack_features", "starfm", "train_wall_to_wall", "upscale"]
