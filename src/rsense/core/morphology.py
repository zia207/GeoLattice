"""Small, version-independent morphology helpers on boolean masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as snd

_EIGHT = np.ones((3, 3), bool)


def remove_small(mask: np.ndarray, min_pixels: int, connectivity: int = 8) -> np.ndarray:
    """Drop connected True regions with fewer than ``min_pixels`` pixels."""
    st = _EIGHT if connectivity == 8 else None
    lab, n = snd.label(mask, structure=st)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Fill False regions (holes) smaller than ``min_pixels`` that don't touch the border."""
    inv = ~mask.astype(bool)
    lab, n = snd.label(inv)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = sizes < min_pixels
    small[0] = False
    small[border] = False
    return mask.astype(bool) | small[lab]
