"""Settings, read from environment variables so the same code runs locally, in CI and in a container."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    aoi: tuple[float, float, float, float] = (91.02, 25.08, 91.14, 25.17)  # Tanguar Haor, Sunamganj
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "sentinel-2-l2a"
    max_cloud: float = 40.0  # scene-level cloud cover filter, %
    resolution: int = 30  # metres; the haor is about 12 x 10 km
    crs: str = "EPSG:32646"
    min_clear: float = 0.2  # observations with less clear area are kept but flagged
    data_dir: Path = field(default_factory=lambda: Path("monitor_data"))

    @classmethod
    def from_env(cls) -> Settings:
        """Override defaults with HAOR_* environment variables (e.g. HAOR_DATA_DIR, HAOR_MAX_CLOUD)."""
        kwargs: dict = {}
        if "HAOR_AOI" in os.environ:
            kwargs["aoi"] = tuple(float(v) for v in os.environ["HAOR_AOI"].split(","))
        for name, cast in [
            ("stac_url", str),
            ("collection", str),
            ("max_cloud", float),
            ("resolution", int),
            ("crs", str),
            ("min_clear", float),
        ]:
            if f"HAOR_{name.upper()}" in os.environ:
                kwargs[name] = cast(os.environ[f"HAOR_{name.upper()}"])
        if "HAOR_DATA_DIR" in os.environ:
            kwargs["data_dir"] = Path(os.environ["HAOR_DATA_DIR"])
        return cls(**kwargs)
