"""``SpatialCausalData`` — one container for treatment, outcome, covariates and space."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from geolattice.spatial_analysis.spatialcausal._utils import (
    as_1d,
    as_2d,
    as_coords,
    as_sparse_weights,
    check_lengths,
    default_weights,
)


class SpatialCausalData:
    """Spatially indexed causal data.

    Parameters
    ----------
    X : array-like (n, p)
        Pre-treatment covariates.
    treatment : array-like (n,)
        Treatment (binary or continuous).
    y : array-like (n,)
        Outcome.
    coords : array-like (n, 2) or GeoDataFrame
        Point coordinates (projected units recommended), or polygons whose
        representative points are used.
    w : GeoLattice ``W``, scipy sparse, or ndarray, optional
        Spatial weights. Defaults to row-standardised 8-NN weights built
        with ``geolattice.spatial_statistics.weights.KNN``.
    feature_names : sequence of str, optional
    truth : dict, optional
        True effects, filled in by the simulators for benchmarking.
    """

    def __init__(self, X, treatment, y, coords, w=None,
                 feature_names: Optional[Sequence[str]] = None,
                 time=None, gdf=None, truth: Optional[dict] = None):
        self.X = as_2d(X, "X")
        self.treatment = as_1d(treatment, "treatment")
        self.y = as_1d(y, "y")
        self.coords = as_coords(coords)
        self.time = None if time is None else np.asarray(time)
        self.n = check_lengths(X=self.X, treatment=self.treatment, y=self.y,
                               coords=self.coords, time=self.time)
        self.feature_names = list(feature_names) if feature_names is not None else [
            f"x{j}" for j in range(self.X.shape[1])]
        self._w = w
        self.gdf = gdf
        self.truth = truth or {}

    # ------------------------------------------------------------------
    @classmethod
    def from_geodataframe(cls, gdf, covariates: Sequence[str], treatment: str,
                          outcome: str, w=None, time: Optional[str] = None):
        """Build from a GeoDataFrame, the PySpatialStats convention."""
        return cls(
            X=gdf[list(covariates)].to_numpy(dtype=float),
            treatment=gdf[treatment].to_numpy(dtype=float),
            y=gdf[outcome].to_numpy(dtype=float),
            coords=gdf,
            w=w,
            feature_names=list(covariates),
            time=None if time is None else gdf[time].to_numpy(),
            gdf=gdf,
        )

    # ------------------------------------------------------------------
    @property
    def w(self):
        if self._w is None:
            self._w = default_weights(self.coords)
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    def w_sparse(self, row_standardize: bool = True):
        return as_sparse_weights(self.w, self.n, row_standardize=row_standardize)

    @property
    def is_binary(self) -> bool:
        return bool(np.all(np.isin(self.treatment, (0.0, 1.0))))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["treatment"] = self.treatment
        df["y"] = self.y
        df["coord_x"], df["coord_y"] = self.coords[:, 0], self.coords[:, 1]
        return df

    def __repr__(self) -> str:
        kind = "binary" if self.is_binary else "continuous"
        return (f"SpatialCausalData(n={self.n}, p={self.X.shape[1]}, "
                f"treatment={kind}, treated_share={self.treatment.mean():.3f})")
