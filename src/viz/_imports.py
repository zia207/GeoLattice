"""Lazy imports for optional visualization libraries."""

from __future__ import annotations

import importlib
from typing import Any


class VisualizationError(ValueError):
    """Raised when a spatial dataset cannot be drawn."""


def require(module: str) -> Any:
    """Import ``module`` or raise a message that names the viz extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise VisualizationError(
            f"{module} is required for this plot. "
            'Install the visualization extras with: pip install "geolattice[viz]"'
        ) from exc
