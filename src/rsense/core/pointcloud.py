"""PointCloud: columnar point storage on CPU (NumPy) or GPU (CuPy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backend import array_module, get_xp, to_cpu

ASPRS_CLASSES = {
    0: "never_classified", 1: "unclassified", 2: "ground", 3: "low_vegetation",
    4: "medium_vegetation", 5: "high_vegetation", 6: "building", 7: "low_noise",
    9: "water", 10: "rail", 11: "road", 17: "bridge_deck", 18: "high_noise",
}


@dataclass
class PointCloud:
    """Points as aligned 1-D arrays: ``x``, ``y``, ``z`` plus optional attributes.

    Common attributes follow LAS names: ``intensity``, ``return_number``,
    ``number_of_returns``, ``classification``, ``scan_angle``, ``red/green/blue``.
    """

    x: Any
    y: Any
    z: Any
    attrs: dict[str, Any] = field(default_factory=dict)
    crs: Any = None

    def __post_init__(self):
        n = len(self.x)
        for k, v in {"y": self.y, "z": self.z, **self.attrs}.items():
            if len(v) != n:
                raise ValueError(f"Field {k!r} has length {len(v)}, expected {n}")

    # -- basics
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, name: str):
        if name in ("x", "y", "z"):
            return getattr(self, name)
        return self.attrs[name]

    def __contains__(self, name: str) -> bool:
        return name in ("x", "y", "z") or name in self.attrs

    @property
    def fields(self) -> list[str]:
        return ["x", "y", "z", *self.attrs]

    @property
    def xp(self):
        return array_module(self.x)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        xp = self.xp
        return tuple(float(f(a)) for a in (self.x, self.y, self.z) for f in (xp.min, xp.max))[0:6]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xp = self.xp
        return (float(xp.min(self.x)), float(xp.min(self.y)),
                float(xp.max(self.x)), float(xp.max(self.y)))

    def copy(self) -> PointCloud:
        return PointCloud(self.x.copy(), self.y.copy(), self.z.copy(),
                          {k: v.copy() for k, v in self.attrs.items()}, self.crs)

    # -- selection
    def filter(self, mask: Any) -> PointCloud:
        """Subset by a boolean mask (or integer index array)."""
        return PointCloud(self.x[mask], self.y[mask], self.z[mask],
                          {k: v[mask] for k, v in self.attrs.items()}, self.crs)

    def by_class(self, *classes: int) -> PointCloud:
        xp = self.xp
        if "classification" not in self.attrs:
            raise KeyError("Point cloud has no 'classification' field")
        return self.filter(xp.isin(self.attrs["classification"], xp.asarray(classes)))

    def ground(self) -> PointCloud:
        return self.by_class(2)

    def first_returns(self) -> PointCloud:
        if "return_number" not in self.attrs:
            return self
        return self.filter(self.attrs["return_number"] == 1)

    def crop(self, minx: float, miny: float, maxx: float, maxy: float) -> PointCloud:
        m = (self.x >= minx) & (self.x <= maxx) & (self.y >= miny) & (self.y <= maxy)
        return self.filter(m)

    def drop_noise(self) -> PointCloud:
        if "classification" not in self.attrs:
            return self
        c = self.attrs["classification"]
        return self.filter((c != 7) & (c != 18))

    def voxel_downsample(self, size: float) -> PointCloud:
        """Keep one point (the first) per ``size``-metre voxel."""
        xp = self.xp
        keys = xp.stack([xp.floor((a - xp.min(a)) / size).astype(xp.int64)
                         for a in (self.x, self.y, self.z)], axis=1)
        _, idx = np.unique(to_cpu(keys), axis=0, return_index=True)
        return self.filter(xp.asarray(np.sort(idx)))

    def with_field(self, name: str, values: Any) -> PointCloud:
        pc = self.copy()
        if len(values) != len(pc):
            raise ValueError("length mismatch")
        pc.attrs[name] = values
        return pc

    # -- devices and conversion
    def to_device(self, spec: str | None = None) -> PointCloud:
        from ..backend import to_device

        return PointCloud(to_device(self.x, spec), to_device(self.y, spec), to_device(self.z, spec),
                          {k: to_device(v, spec) for k, v in self.attrs.items()}, self.crs)

    def to_cpu(self) -> PointCloud:
        return PointCloud(to_cpu(self.x), to_cpu(self.y), to_cpu(self.z),
                          {k: to_cpu(v) for k, v in self.attrs.items()}, self.crs)

    def to_dataframe(self):
        import pandas as pd

        pc = self.to_cpu()
        return pd.DataFrame({f: pc[f] for f in pc.fields})

    def to_geodataframe(self):
        import geopandas as gpd

        df = self.to_dataframe()
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y, df.z), crs=self.crs)

    @classmethod
    def from_dataframe(cls, df, crs: Any = None) -> PointCloud:
        xp = get_xp()
        cols = {c: xp.asarray(df[c].to_numpy()) for c in df.columns}
        return cls(cols.pop("x"), cols.pop("y"), cols.pop("z"), cols, crs)

    @classmethod
    def from_arrays(cls, x, y, z, crs: Any = None, **attrs: Any) -> PointCloud:
        return cls(np.asarray(x), np.asarray(y), np.asarray(z),
                   {k: np.asarray(v) for k, v in attrs.items()}, crs)

    def __repr__(self) -> str:
        b = self.bbox if len(self) else (np.nan,) * 4
        where = "GPU" if self.xp is not np else "CPU"
        return (f"<PointCloud {len(self):,} pts [{where}] fields={self.fields} "
                f"bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}) crs={self.crs}>")
