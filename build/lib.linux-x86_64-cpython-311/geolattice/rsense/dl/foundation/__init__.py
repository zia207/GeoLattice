"""Adapters for geospatial foundation models: Prithvi, SatMAE, Clay and SAM.

Each adapter loads public weights on demand and exposes ``encode(x)`` (embeddings) and,
where meaningful, a task head builder for fine-tuning. Heavy dependencies are optional.
"""

from .adapters import SAM, Clay, Prithvi, SatMAE, list_foundation_models

__all__ = ["SAM", "Clay", "Prithvi", "SatMAE", "list_foundation_models"]
