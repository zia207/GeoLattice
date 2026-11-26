"""Result containers built on ``geolattice.spatial_statistics.core.SpatialResult``."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from geolattice.spatial_statistics.core.result import SpatialResult


class EffectEstimate:
    """One causal contrast with its standard error and confidence interval."""

    def __init__(self, name: str, estimate: float, se: float, level: float = 0.95):
        self.name = name
        self.estimate = float(estimate)
        self.se = float(se)
        self.level = float(level)

    @property
    def ci(self):
        z = stats.norm.ppf(0.5 + self.level / 2)
        return (self.estimate - z * self.se, self.estimate + z * self.se)

    @property
    def z(self) -> float:
        return self.estimate / self.se if self.se > 0 else np.nan

    @property
    def p_value(self) -> float:
        return float(2 * stats.norm.sf(abs(self.z))) if self.se > 0 else np.nan

    def as_dict(self) -> Dict[str, float]:
        lo, hi = self.ci
        return {"effect": self.name, "estimate": self.estimate, "se": self.se,
                "ci_lower": lo, "ci_upper": hi, "p_value": self.p_value}

    def __repr__(self) -> str:
        lo, hi = self.ci
        return f"{self.name}: {self.estimate:.4f} (SE {self.se:.4f}, {int(self.level*100)}% CI [{lo:.4f}, {hi:.4f}])"


class SpatialCausalResult(SpatialResult):
    """A table of causal contrasts plus metadata about how SEs were computed."""

    def __init__(self, name: str, effects: Dict[str, EffectEstimate],
                 se_method: str = "iid", n: Optional[int] = None, **meta):
        super().__init__(name=name, se_method=se_method, n=n, **meta)
        self.effects = effects

    def __getitem__(self, key: str) -> EffectEstimate:
        return self.effects[key]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.as_dict() for e in self.effects.values()]).set_index("effect")

    def summary(self) -> str:
        head = f"{self.name}  (n={self._stats.get('n')}, SE: {self._stats.get('se_method')})"
        lines = [head, "=" * len(head)]
        df = self.to_frame()
        lines.append(df.to_string(float_format=lambda v: f"{v:.4f}"))
        extra = {k: v for k, v in self._stats.items() if k not in ("n", "se_method")}
        for k, v in extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
