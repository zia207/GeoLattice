"""Backend-aware access to SciPy-style submodules (ndimage, fft, linalg) and scatter ops."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import array_module, get_xp


def ndimage(arr: Any | None = None):
    """``scipy.ndimage`` on CPU, ``cupyx.scipy.ndimage`` on CUDA (chosen from ``arr`` if given)."""
    xp = array_module(arr) if arr is not None else get_xp()
    if xp is np:
        import scipy.ndimage as nd

        return nd
    import cupyx.scipy.ndimage as nd

    return nd


def fft(arr: Any | None = None):
    xp = array_module(arr) if arr is not None else get_xp()
    return xp.fft


def scatter_reduce(target, index, values, op: str):
    """In-place unbuffered reduction ``target[index] = op(target[index], values)``.

    ``op`` is ``"add"``, ``"min"`` or ``"max"``. Works for NumPy and CuPy.
    """
    xp = array_module(target)
    if xp is np:
        {"add": np.add, "min": np.minimum, "max": np.maximum}[op].at(target, index, values)
        return target
    import cupyx

    {"add": cupyx.scatter_add, "min": cupyx.scatter_min, "max": cupyx.scatter_max}[op](
        target, index, values)
    return target
