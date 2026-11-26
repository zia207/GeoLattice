"""Deprecated location of the spatial causal inference package.

``geolattice.spatialcausal`` moved to :mod:`geolattice.spatial_analysis.spatialcausal` in GeoLattice 0.3.43.
This shim keeps old imports working (``import geolattice.spatialcausal as sc``,
``from geolattice.spatialcausal.quasi import GeographicRDD``): every name and submodule resolves
to the same objects as the new location, so classes and results stay interchangeable.
It will be removed in a future release.
"""

import importlib
import pkgutil
import sys
import warnings

warnings.warn(
    "geolattice.spatialcausal moved to geolattice.spatial_analysis.spatialcausal; "
    "update imports (the old path will be removed in a future release)",
    DeprecationWarning,
    stacklevel=2,
)

_new = importlib.import_module("geolattice.spatial_analysis.spatialcausal")


def _alias(package, old_prefix):
    """Register every submodule of ``package`` under ``old_prefix`` in sys.modules."""
    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        module = importlib.import_module(info.name)
        sys.modules[old_prefix + info.name[len(package.__name__):]] = module


_alias(_new, __name__)
sys.modules[__name__] = _new  # ``geolattice.spatialcausal`` *is* the new package
