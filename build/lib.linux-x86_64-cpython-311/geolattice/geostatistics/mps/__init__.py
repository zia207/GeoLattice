"""
Multiple-point statistics simulation.

Modules: ``direct_sampling`` (`DirectSampling`, `dssim`, pattern and
connectivity diagnostics, synthetic channel training images) and
``snesim`` (`SNESIM`, `snesim`, `tau_model`).

``geolattice.geostatistics.mps`` used to be the Direct Sampling module; its names
are re-exported here so ``from geolattice.geostatistics.mps import DirectSampling``
keeps working.
"""

from .direct_sampling import *  # noqa: F401,F403
from .direct_sampling import __all__ as _ds_all
from .snesim import SNESIM, snesim, tau_model  # noqa: F401

__all__ = list(_ds_all) + ["SNESIM", "snesim", "tau_model"]
