"""Alias of :mod:`geolattice.spatial_bd.geolatticedb`.

``from geolattice import geolatticedb`` and ``import geolattice.geolatticedb`` both return the
same module object as ``geolattice.spatial_bd.geolatticedb``.
"""

import sys as _sys

from geolattice.spatial_bd import geolatticedb as _geolatticedb

_sys.modules[__name__] = _geolatticedb
