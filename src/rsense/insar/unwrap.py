"""Phase unwrapping: SNAPHU (via ``snaphu-py``) or a native quality-guided fallback."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Literal

import numpy as np

from ..backend import to_cpu


@contextmanager
def _quiet_subprocess_output():
    """Silence a child process's console log. SNAPHU writes progress to the process-level stdout
    and stderr (file descriptors 1 and 2), which ``contextlib.redirect_stdout`` cannot catch and
    which Jupyter forwards into the notebook."""
    import os
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    saved = {}
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        for fd in (1, 2):
            try:
                saved[fd] = os.dup(fd)
                os.dup2(null, fd)
            except OSError:
                pass
        yield
    finally:
        for fd, dup in saved.items():
            os.dup2(dup, fd)
            os.close(dup)
        os.close(null)


def unwrap(phase_or_ifg: Any, coherence: Any | None = None, *,
           method: Literal["snaphu", "native"] = "snaphu", nlooks: float = 5.0,
           cost: Literal["smooth", "defo"] = "smooth", init: Literal["mcf", "mst"] = "mcf",
           mask: Any | None = None, ntiles: tuple[int, int] = (1, 1)) -> Any:
    """Unwrap interferometric phase.

    ``snaphu`` (statistical-cost network flow; Chen & Zebker 2001) is the standard for
    InSAR and needs ``pip install snaphu``. ``native`` uses scikit-image's reliability-
    sorted unwrapping (Herráez et al. 2002): dependency-free, fine for good coherence.
    Accepts wrapped phase (real) or a complex interferogram. Returns unwrapped phase (rad).
    """
    data = np.asarray(to_cpu(getattr(phase_or_ifg, "data", phase_or_ifg)))
    ifg = data if np.iscomplexobj(data) else np.exp(1j * data)
    bad = ~np.isfinite(ifg)
    if mask is not None:
        bad |= ~np.asarray(to_cpu(getattr(mask, "data", mask)), bool)
    ifg = np.where(bad, 0, ifg)
    if method == "snaphu":
        try:
            import snaphu
        except ImportError as e:
            raise ImportError("method='snaphu' needs `pip install snaphu` (or use method='native')"
                              ) from e
        corr = (np.asarray(to_cpu(getattr(coherence, "data", coherence)), dtype="float32")
                if coherence is not None else np.abs(ifg).astype("float32") * 0 + 0.5)
        corr = np.nan_to_num(corr)
        with _quiet_subprocess_output():
            unw, _ = snaphu.unwrap(ifg.astype("complex64"), corr, nlooks=nlooks, cost=cost, init=init,
                                   mask=~bad, ntiles=ntiles)
    elif method == "native":
        from skimage.restoration import unwrap_phase

        unw = unwrap_phase(np.ma.masked_array(np.angle(ifg), mask=bad)).filled(np.nan)
    else:
        raise ValueError(f"Unknown method {method!r}")
    unw = np.where(bad, np.nan, unw).astype("float32")
    if hasattr(phase_or_ifg, "dims"):
        return phase_or_ifg.copy(data=unw).rename("unwrapped_phase")
    return unw
