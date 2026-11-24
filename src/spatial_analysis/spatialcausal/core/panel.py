"""``SpatialPanel`` — long-format panel data with one location per unit."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from geolattice.spatial_analysis.spatialcausal._utils import as_1d, as_coords


class SpatialPanel:
    """Unit × time panel with unit locations.

    Parameters
    ----------
    unit, time, y : array-like (n_obs,)
        Long-format identifiers and outcome.
    unit_ids : array-like (n_units,)
        Unit identifiers matching ``unit``.
    unit_coords : array-like (n_units, 2) or GeoDataFrame
        Location of each unit in ``unit_ids`` (projected units recommended).
    treated : array-like (n_units,), optional
        1 for treated units (unit level).
    post : array-like (n_obs,), optional
        1 for observations in the post-treatment period. Alternatively give
        ``treat_time`` (a scalar) and ``post`` is ``time >= treat_time``.
    X : array-like (n_obs, p), optional
        Time-varying covariates (used by the DiD estimator).
    """

    def __init__(self, unit, time, y, unit_ids, unit_coords, treated=None, post=None,
                 treat_time=None, X=None, truth: Optional[dict] = None):
        self.unit = np.asarray(unit)
        self.time = np.asarray(time)
        self.y = as_1d(y, "y")
        self.units = np.asarray(unit_ids)
        if len(np.unique(self.units)) != len(self.units):
            raise ValueError("unit_ids must be unique")
        self.unit_coords = as_coords(unit_coords)
        if len(self.unit_coords) != len(self.units):
            raise ValueError("unit_coords must have one row per unit id")
        if not (len(self.unit) == len(self.time) == len(self.y)):
            raise ValueError("unit, time and y must have the same length")
        lookup = {u: i for i, u in enumerate(self.units)}
        try:
            self.unit_index = np.array([lookup[u] for u in self.unit])
        except KeyError as exc:
            raise ValueError(f"unit {exc.args[0]!r} has no coordinates") from None
        self.times = np.unique(self.time)
        self.time_index = np.searchsorted(self.times, self.time)
        self.treated = (np.zeros(len(self.units), int) if treated is None
                        else np.asarray(treated, float).astype(int))
        if post is not None:
            self.post = np.asarray(post, float).astype(int)
        elif treat_time is not None:
            self.post = (self.time >= treat_time).astype(int)
        else:
            self.post = None
        self.treat_time = treat_time
        self.X = None if X is None else np.asarray(X, float)
        self.truth = truth or {}

    # ------------------------------------------------------------------
    @classmethod
    def from_long(cls, df: pd.DataFrame, unit: str, time: str, outcome: str,
                  coords: Sequence[str] = ("x", "y"), geometry: Optional[str] = None,
                  treated: Optional[str] = None, post: Optional[str] = None,
                  treat_time=None, covariates: Optional[Sequence[str]] = None):
        """Build from a long DataFrame.

        Coordinates come from columns ``coords`` (constant within unit) or a
        ``geometry`` column; ``treated`` is a unit-level 0/1 column.
        """
        first = df.groupby(unit, sort=True).first()
        if geometry is not None:
            import geopandas as gpd
            uc = gpd.GeoSeries(first[geometry].values)
        else:
            uc = first[list(coords)].to_numpy(float)
        return cls(df[unit].to_numpy(), df[time].to_numpy(), df[outcome].to_numpy(float),
                   first.index.to_numpy(), uc,
                   treated=None if treated is None else first[treated].to_numpy(),
                   post=None if post is None else df[post].to_numpy(),
                   treat_time=treat_time,
                   X=None if covariates is None else df[list(covariates)].to_numpy(float))

    # ------------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def is_balanced(self) -> bool:
        return self.n_obs == self.n_units * len(self.times) and not np.any(
            pd.Series(self.unit_index * len(self.times) + self.time_index).duplicated())

    def wide(self) -> np.ndarray:
        """Outcome as an (n_units, n_times) matrix (NaN where missing)."""
        Y = np.full((self.n_units, len(self.times)), np.nan)
        Y[self.unit_index, self.time_index] = self.y
        return Y

    def adoption_times(self) -> np.ndarray:
        """First post period for each treated unit (NaN for never-treated)."""
        if self.post is None:
            raise ValueError("panel has no post / treat_time information")
        out = np.full(self.n_units, np.nan)
        on = (self.post == 1) & (self.treated[self.unit_index] == 1)
        if on.any():
            s = pd.Series(self.time[on]).groupby(self.unit_index[on]).min()
            out[s.index.to_numpy()] = s.to_numpy()
        return out

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"unit": self.unit, "time": self.time, "y": self.y,
                           "treated": self.treated[self.unit_index]})
        if self.post is not None:
            df["post"] = self.post
        df["coord_x"] = self.unit_coords[self.unit_index, 0]
        df["coord_y"] = self.unit_coords[self.unit_index, 1]
        return df

    def __repr__(self) -> str:
        return (f"SpatialPanel(units={self.n_units}, periods={len(self.times)}, obs={self.n_obs}, "
                f"treated_units={int(self.treated.sum())}, balanced={self.is_balanced})")
