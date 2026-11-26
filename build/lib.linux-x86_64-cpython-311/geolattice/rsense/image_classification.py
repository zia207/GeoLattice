"""Alias of :mod:`geolattice.rsense.classify.image_classification`, kept so existing imports work.

The module moved into the ``classify`` subpackage in GeoLattice 0.3.30. Both
``import geolattice.rsense.image_classification`` and
``python -m geolattice.rsense.image_classification`` still work and refer to the same module.
"""

import sys

from .classify import image_classification as _ic

if __name__ == "__main__":
    _ic.main()
else:
    sys.modules[__name__] = _ic
