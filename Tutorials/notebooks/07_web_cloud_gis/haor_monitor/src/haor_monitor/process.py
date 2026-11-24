"""Pure functions that turn one day's Sentinel-2 bands into statistics. No input/output: easy to test."""

from __future__ import annotations

import math

import numpy as np

CLEAR_CLASSES = (4, 5, 6)  # SCL: vegetation, bare soil, water
EXPECTED = {"clear_fraction": (0.0, 1.0), "water_fraction": (0.0, 1.0), "ndvi_land_median": (-1.0, 1.0)}


def reflectance(dn: np.ndarray, processing_baseline: str | float) -> np.ndarray:
    """Surface reflectance from Sentinel-2 L2A DN as distributed by ESA / Planetary Computer.

    From processing baseline 04.00 (January 2022) DN carry an offset of 1000:
    reflectance = (DN - 1000) / 10000. Earlier baselines have no offset. DN 0 means no data.
    """
    offset = 1000.0 if float(processing_baseline) >= 4.0 else 0.0
    out = (dn.astype("float32") - offset) / 10_000.0
    out[dn == 0] = np.nan
    return out


def scene_stats(green: np.ndarray, red: np.ndarray, nir: np.ndarray, scl: np.ndarray) -> dict:
    """Clear fraction, open-water fraction (NDWI > 0) and median land NDVI of one observation.

    Bands are reflectances on the same grid; ``scl`` is the scene classification layer.
    """
    clear = np.isin(scl, CLEAR_CLASSES) & np.isfinite(green) & np.isfinite(red) & np.isfinite(nir)
    n = int(scl.size)
    if not clear.any():
        return {"pixels": n, "clear_fraction": 0.0, "water_fraction": math.nan, "ndvi_land_median": math.nan}
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (green - nir) / (green + nir)
        ndvi = (nir - red) / (nir + red)
    water = clear & (ndwi > 0)
    land = clear & ~water
    return {
        "pixels": n,
        "clear_fraction": float(clear.mean()),
        "water_fraction": float(water.sum() / clear.sum()),
        "ndvi_land_median": float(np.nanmedian(ndvi[land])) if land.any() else math.nan,
    }


def quality_flags(stats: dict, min_clear: float = 0.2) -> list[str]:
    """Data-quality checks; an empty list means the observation is usable."""
    flags = []
    for key, (lo, hi) in EXPECTED.items():
        value = stats.get(key)
        if value is None:
            flags.append(f"missing:{key}")
        elif not math.isnan(value) and not lo <= value <= hi:
            flags.append(f"out_of_range:{key}")
    if stats.get("clear_fraction", 0.0) < min_clear:
        flags.append("low_clear_fraction")
    return flags
