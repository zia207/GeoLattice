"""Atmospheric composition: Sentinel-5P/TROPOMI trace-gas, aerosol and cloud products.

``sentinel_5p`` is the only submodule so far. Its products are NetCDF-4 swath files, not
georeferenced imagery, so it exposes readers and a quality mask directly rather than a
``load``/``qa_mask`` pair tied to a lazy STAC loader -- see its module docstring.
"""

from . import sentinel_5p

__all__ = ["sentinel_5p"]
