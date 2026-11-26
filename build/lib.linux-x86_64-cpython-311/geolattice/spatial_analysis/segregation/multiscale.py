"""Multiscale ("spatial segregation profile") measures (Reardon et al. 2008; Reardon &
O'Sullivan 2004): how a unit's local group composition, pooled from everything within
radius ``r``, converges toward the region's overall composition as ``r`` grows. PySAL's
own implementation of this idea builds a dense ``n x n`` distance matrix and an
``n^2``-row table of pairwise terms, which runs out of memory well before ``n`` reaches a
few thousand units. This version never materializes either: it queries a
``scipy.spatial.cKDTree`` for each unit's neighbors within the *largest* radius only,
sorts just that unit's own (typically small) neighbor list by distance, and reads off the
cumulative pooled composition at every smaller radius with one ``searchsorted`` call --
memory scales with how many neighbors are actually within range, not with ``n^2``.

Three outputs, all designed to be mapped directly instead of eyeballing a profile curve
per unit:

- :func:`segregation_scale` -- the distance at which a unit's divergence from the
  regional composition first drops below a threshold (small = segregation is a very
  local, short-range phenomenon around that unit; large / ``inf`` = it persists across
  most of the study area).
- :func:`profile_auc` -- area under the divergence-vs-radius curve out to the largest
  radius queried (a single number summarizing "how segregated, integrated over scale").
- :func:`distortion_coefficient` -- the fraction of a unit's radius-zero (own-unit-only)
  divergence that is still left at the largest radius queried: close to 0 means
  segregation at that unit is purely a short-range, local pattern that neighborhood
  pooling erases quickly; close to 1 means it persists essentially unchanged across the
  whole measured range (often a sign the study area itself is too small to see it decay).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .data import extract_counts, global_proportions


def _centroids(gdf) -> np.ndarray:
    geom = gdf.geometry
    cent = geom.centroid if geom.geom_type.isin(["Polygon", "MultiPolygon"]).any() else geom
    return np.c_[cent.x.to_numpy(), cent.y.to_numpy()]


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) along the last axis, row-wise; 0 contribution wherever p_k == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log(p / q), 0.0)
    # p_k > 0 but q_k == 0 is a genuine +inf divergence (impossible under evenness); keep it.
    return np.sum(term, axis=-1)


@dataclass
class MultiscaleProfile:
    """Per-unit divergence at each queried radius, plus the compact summaries.

    ``profile`` is ``(n_units, n_radii)``; ``radii`` the grid it was evaluated at.
    ``segregation_scale``, ``auc``, and ``distortion`` are each ``(n_units,)``, ready to
    assign straight onto a GeoDataFrame column for mapping.
    """

    radii: np.ndarray
    profile: np.ndarray
    segregation_scale: np.ndarray
    auc: np.ndarray
    distortion: np.ndarray


def compute_profile(gdf, groups: Sequence[str], *, radii: Sequence[float],
                     divergence: str = "kl", scale_threshold: float = 0.05,
                     chunk_size: int = 500, max_neighbors: int = 2000) -> MultiscaleProfile:
    """Compute the multiscale profile for every unit at each radius in ``radii`` (ascending).

    ``divergence`` is ``"kl"`` (Kullback-Leibler, the Reardon profile's own measure) or
    ``"l1"`` (half the total-variation / dissimilarity-style distance, bounded in [0, 1]
    and well-defined even where a group's pooled composition hits exactly 0). Queries run
    in batches of ``chunk_size`` units at a time, each unit capped at its ``max_neighbors``
    nearest candidates within the largest radius -- raise it only if a unit's true neighbor
    count at the largest radius exceeds that cap (a warning-free silent truncation would
    otherwise flatten its profile early; this raises instead).
    """
    xy = _centroids(gdf)
    counts = extract_counts(gdf, groups)
    P = global_proportions(counts)
    radii = np.asarray(sorted(radii), dtype="float64")
    n = len(gdf)
    tree = cKDTree(xy)

    profile = np.empty((n, len(radii)))
    k_used = min(max_neighbors, n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_xy = xy[start:end]
        dist, idx = tree.query(chunk_xy, k=k_used, workers=-1)
        if dist.ndim == 1:  # k == 1 edge case (a single-unit input)
            dist, idx = dist[:, None], idx[:, None]
        # only a real truncation risk when fewer than every unit was queried as a candidate
        if k_used < n and np.any(dist[:, -1] < radii[-1]):
            raise ValueError(
                f"max_neighbors={max_neighbors} is not enough to cover radius {radii[-1]:g} "
                "for at least one unit; raise max_neighbors or shrink the largest radius")
        cum_counts = np.cumsum(counts[idx], axis=1)  # (chunk, k, n_groups), running pooled totals
        for r_pos, r in enumerate(radii):
            cutoffs = np.sum(dist <= r, axis=1)
            cutoffs = np.clip(cutoffs, 1, cum_counts.shape[1])
            pooled = cum_counts[np.arange(end - start), cutoffs - 1, :]
            totals = pooled.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.where(totals[:, None] > 0, pooled / totals[:, None], 0.0)
            if divergence == "kl":
                profile[start:end, r_pos] = _kl_divergence(p, P[None, :])
            elif divergence == "l1":
                profile[start:end, r_pos] = 0.5 * np.sum(np.abs(p - P[None, :]), axis=1)
            else:
                raise ValueError("divergence must be 'kl' or 'l1'")

    auc = np.trapezoid(profile, radii, axis=1) if hasattr(np, "trapezoid") else np.trapz(profile, radii, axis=1)
    below = profile <= scale_threshold
    scale = np.where(below.any(axis=1), radii[np.argmax(below, axis=1)], np.inf)
    # argmax on an all-False row returns 0, which the np.where above already routes to inf instead
    first_col = profile[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        distortion = np.where(first_col > 0, profile[:, -1] / first_col, 0.0)

    return MultiscaleProfile(radii=radii, profile=profile, segregation_scale=scale,
                             auc=auc, distortion=distortion)
