"""Build GeoLattice, including spatial_statistics and the suitability C extension."""

from setuptools import Extension, find_namespace_packages, find_packages, setup

import numpy


def _geolattice_packages():
    """Map ``src/`` contents onto the ``geolattice`` import namespace."""
    discovered = set(find_packages("src")) | set(find_namespace_packages("src"))
    return ["geolattice", *[f"geolattice.{name}" for name in sorted(discovered)]]


extensions = [
    Extension(
        "geolattice.spatial_analysis._suitability",
        sources=["src/spatial_analysis/_suitability.c"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


setup(
    # ``pygis`` is a deprecated alias package that forwards to ``geolattice``.
    package_dir={"geolattice": "src", "pygis": "compat/pygis"},
    packages=[*_geolattice_packages(), "pygis"],
    ext_modules=extensions,
)
