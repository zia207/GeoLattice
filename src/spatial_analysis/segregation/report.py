"""One consistent entry point: ``compute(gdf, index="theil_h", groups=[...])`` picks the
right function out of :mod:`indices`/:mod:`local`/:mod:`multiscale`, optionally attaches
a bootstrap confidence interval, and returns a small, uniform result object with
``.summary()``/``.plot()``/``.map()`` instead of every caller remembering a different
function name and return shape for each index.

:func:`export_for_causal` is the hook into PyCausalML / ``geolattice.spatial_analysis.spatialcausal``: it turns
a local segregation surface into a tidy, geometry-free column keyed by unit id, ready to
merge in as an exposure or confounder variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import indices as _indices
from .inference import spatial_block_bootstrap

_REGION_INDEX_FNS: dict[str, Callable] = {
    "dissimilarity": _indices.dissimilarity,
    "gini": _indices.gini,
    "theil_h": _indices.theil_h,
    "simpson_diversity": _indices.simpson_diversity,
    "multigroup_dissimilarity": _indices.multigroup_dissimilarity,
}

_PARAMETRIC_INDEX_FNS: dict[str, Callable] = {
    "isolation": _indices.isolation,
    "exposure": _indices.exposure,
    "correlation_ratio": _indices.correlation_ratio,
}


@dataclass
class SegregationResult:
    """Uniform result of :func:`compute`. ``value`` is the point estimate; ``ci_lower``/
    ``ci_upper`` are set only when ``ci=True`` was passed. ``local_values`` holds a
    per-unit array when ``index`` names a local measure, ``None`` for a region-wide one."""

    index: str
    groups: list[str]
    value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float | None = None
    local_values: np.ndarray | None = None
    gdf: Any = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.value is not None:
            ci = f" (95% CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}])" if self.ci_lower is not None else ""
            return f"{self.index}({', '.join(self.groups)}) = {self.value:.4f}{ci}"
        return (f"{self.index}({', '.join(self.groups)}): local values for {len(self.local_values)} "
                f"units, mean={np.nanmean(self.local_values):.4f}")

    def to_frame(self) -> pd.DataFrame:
        if self.local_values is not None:
            return pd.DataFrame({self.index: self.local_values})
        return pd.DataFrame([{"index": self.index, "groups": ", ".join(self.groups),
                             "value": self.value, "ci_lower": self.ci_lower, "ci_upper": self.ci_upper}])

    def map(self, **kwargs):
        from . import viz

        if self.local_values is None or self.gdf is None:
            raise ValueError("map() needs a local index computed with the source gdf retained")
        return viz.plot_local_map(self.gdf, self.local_values, title=self.summary(), **kwargs)


def compute(gdf, index: str, groups: Sequence[str], *, group: str | None = None,
            group_a: str | None = None, group_b: str | None = None,
            neighborhood: Any | None = None, ci: bool = False, n_boot: int = 499,
            n_blocks: int = 10, seed: int | None = None) -> SegregationResult:
    """Compute one named index and return it in a uniform :class:`SegregationResult`.

    ``index`` is one of the region-wide names in :mod:`indices`
    (``"dissimilarity"``, ``"gini"``, ``"theil_h"``, ``"simpson_diversity"``,
    ``"multigroup_dissimilarity"``, ``"isolation"``, ``"exposure"``,
    ``"correlation_ratio"`` -- the last three need ``group``/``group_a``/``group_b``), or
    a local one from :mod:`local` (``"local_quotient"``, ``"local_entropy"``,
    ``"local_simpson_diversity"``, ``"local_exposure_deviation"``,
    ``"local_dissimilarity_contribution"``), in which case ``neighborhood`` (a
    :class:`~geolattice.spatial_analysis.segregation.neighborhoods.Neighborhood`) optionally
    pools composition across neighbors first. ``ci=True`` adds a
    :func:`~geolattice.spatial_analysis.segregation.inference.spatial_block_bootstrap`
    confidence interval (region-wide indices only).
    """
    groups = list(groups)
    if index in _REGION_INDEX_FNS:
        fn = _REGION_INDEX_FNS[index]
        value = float(fn(gdf, groups))
        result = SegregationResult(index=index, groups=groups, value=value, gdf=gdf)
        if ci:
            boot = spatial_block_bootstrap(gdf, groups, fn, n_boot=n_boot, n_blocks=n_blocks, seed=seed)
            result.ci_lower, result.ci_upper, result.ci_level = boot["ci_lower"], boot["ci_upper"], boot["ci_level"]
        return result

    if index in _PARAMETRIC_INDEX_FNS:
        fn = _PARAMETRIC_INDEX_FNS[index]
        kwargs = {}
        if index == "exposure":
            kwargs = {"group_a": group_a or groups[0], "group_b": group_b or groups[1]}
        else:
            kwargs = {"group": group or groups[0]}
        value = float(fn(gdf, groups, **kwargs))
        result = SegregationResult(index=index, groups=groups, value=value, gdf=gdf)
        if ci:
            boot = spatial_block_bootstrap(gdf, groups, fn, n_boot=n_boot, n_blocks=n_blocks, seed=seed, **kwargs)
            result.ci_lower, result.ci_upper, result.ci_level = boot["ci_lower"], boot["ci_upper"], boot["ci_level"]
        return result

    from . import local as _local

    local_fns = {
        "local_quotient": _local.local_quotient,
        "local_entropy": _local.local_entropy,
        "local_simpson_diversity": _local.local_simpson_diversity,
        "local_exposure_deviation": _local.local_exposure_deviation,
        "local_dissimilarity_contribution": _local.local_dissimilarity_contribution,
    }
    if index not in local_fns:
        known = sorted(set(_REGION_INDEX_FNS) | set(_PARAMETRIC_INDEX_FNS) | set(local_fns))
        raise ValueError(f"Unknown index {index!r}; choose one of {known}")
    fn = local_fns[index]
    kwargs = {"neighborhood": neighborhood}
    if index in ("local_quotient", "local_exposure_deviation"):
        kwargs["group"] = group or groups[0]
    values = fn(gdf, groups, **kwargs)
    return SegregationResult(index=index, groups=groups, local_values=np.asarray(values), gdf=gdf)


def export_for_causal(gdf, local_values: np.ndarray, *, id_col: str | None = None,
                       name: str = "segregation_exposure") -> pd.DataFrame:
    """A unit-id-keyed, geometry-free table of ``local_values`` -- the shape PyCausalML's
    and ``geolattice.spatial_analysis.spatialcausal``'s panel/cross-sectional estimators expect for an exposure
    or confounder column. Merge it onto an outcome dataset by ``id_col`` (or by the
    original row order, if ``id_col`` is not given).
    """
    ids = gdf[id_col].to_numpy() if id_col else np.arange(len(gdf))
    id_name = id_col or "unit_id"
    return pd.DataFrame({id_name: ids, name: np.asarray(local_values)})
