"""
geolattice.spatial_statistics.pointpattern
=========================
Point pattern analysis: intensity estimation, distance-based tests, Ripley's
K/L/pair-correlation functions, and CSR Monte Carlo envelopes.

Quick start
-----------
>>> from . import PointPattern, Window, ripley_l, envelope
>>> win = Window.from_bounds(0, 0, 1, 1)
>>> pp = PointPattern(coords, window=win)
>>> env = envelope(pp, "L", nsim=99, seed=1)
>>> env.test("mad")                       # global CSR p-value

Coordinates must be in projected units (see ``geolattice.spatial_statistics.core.ensure_projected``).
"""

from .window import Window
from .pattern import PointPattern
from .intensity import (
    quadrat_test, QuadratResult, kde, KDEResult, bandwidth_scott, bandwidth_cv,
    intensity_at_points, relative_risk,
)
from .functions import (
    FunctionResult, nearest_neighbor_distances, clark_evans,
    g_function, f_function, j_function,
    ripley_k, ripley_l, pair_correlation, cross_k, k_inhom,
)
from .simulate import (
    simulate_csr, simulate_thomas, simulate_matern_inhibition,
    envelope, EnvelopeResult,
)

__all__ = [
    "Window", "PointPattern",
    "quadrat_test", "QuadratResult", "kde", "KDEResult", "bandwidth_scott", "bandwidth_cv",
    "intensity_at_points", "relative_risk",
    "FunctionResult", "nearest_neighbor_distances", "clark_evans",
    "g_function", "f_function", "j_function",
    "ripley_k", "ripley_l", "pair_correlation", "cross_k", "k_inhom",
    "simulate_csr", "simulate_thomas", "simulate_matern_inhibition",
    "envelope", "EnvelopeResult",
]
