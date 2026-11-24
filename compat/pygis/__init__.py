"""Deprecated alias of :mod:`geolattice` (PyGIS was renamed to GeoLattice).

``import pygis`` and ``from pygis.x import y`` keep working: submodules resolve
to the ``geolattice`` module objects, top-level names forward to
``geolattice``, and a ``FutureWarning`` is emitted. ``pygis.pygisdb`` maps to
``geolattice.geolatticedb``. Switch imports to ``geolattice``; this alias will
be removed in a future release.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

warnings.warn(
    "The 'pygis' package has been renamed to 'geolattice'. "
    "Use 'import geolattice' instead; the 'pygis' alias will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

_RENAMED_PARTS = {"pygisdb": "geolatticedb"}


def _target_name(fullname):
    parts = fullname.split(".")
    parts[0] = "geolattice"
    return ".".join(_RENAMED_PARTS.get(part, part) for part in parts)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The import system reads the module back from sys.modules, so the
        # alias resolves to the real geolattice module object.
        sys.modules[module.__name__] = importlib.import_module(self._target)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("pygis."):
            return None
        target_name = _target_name(fullname)
        if importlib.util.find_spec(target_name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

_geolattice = importlib.import_module("geolattice")


def __getattr__(name):
    if name in _RENAMED_PARTS:
        return importlib.import_module(f"{__name__}.{name}")
    try:
        return getattr(_geolattice, name)
    except AttributeError:
        pass
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(dir(_geolattice)) | set(_RENAMED_PARTS))
