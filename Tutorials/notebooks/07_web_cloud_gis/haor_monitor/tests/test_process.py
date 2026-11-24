"""Unit tests of the pure processing functions, on small synthetic arrays."""

import math

import numpy as np
import pytest

from haor_monitor.process import quality_flags, reflectance, scene_stats


def test_reflectance_offset_depends_on_processing_baseline():
    dn = np.array([1000, 1500, 0], dtype="uint16")
    new = reflectance(dn, "05.10")
    old = reflectance(dn, "03.01")
    assert new[:2] == pytest.approx([0.0, 0.05]) and math.isnan(new[2])
    assert old[:2] == pytest.approx([0.1, 0.15]) and math.isnan(old[2])


def make_scene(water_rows=2, cloud_rows=1, size=10):
    """A 10 x 10 scene: top rows water, bottom rows cloud, the rest green vegetation."""
    green = np.full((size, size), 0.06, "float32")
    red = np.full((size, size), 0.04, "float32")
    nir = np.full((size, size), 0.40, "float32")
    scl = np.full((size, size), 4, "uint8")
    green[:water_rows], nir[:water_rows], scl[:water_rows] = 0.08, 0.02, 6
    scl[size - cloud_rows :] = 9
    return green, red, nir, scl


def test_scene_stats_water_and_vegetation():
    stats = scene_stats(*make_scene(water_rows=2, cloud_rows=1))
    assert stats["clear_fraction"] == pytest.approx(0.9)
    assert stats["water_fraction"] == pytest.approx(2 / 9)
    assert stats["ndvi_land_median"] == pytest.approx((0.40 - 0.04) / (0.40 + 0.04), rel=1e-5)
    assert quality_flags(stats) == []


def test_fully_cloudy_scene_is_flagged_not_crashing():
    green, red, nir, scl = make_scene()
    scl[:] = 9
    stats = scene_stats(green, red, nir, scl)
    assert stats["clear_fraction"] == 0 and math.isnan(stats["water_fraction"])
    assert quality_flags(stats) == ["low_clear_fraction"]


def test_all_water_scene_has_no_land_ndvi():
    stats = scene_stats(*make_scene(water_rows=10, cloud_rows=0))
    assert stats["water_fraction"] == 1.0 and math.isnan(stats["ndvi_land_median"])


def test_quality_flags_catch_impossible_values():
    flags = quality_flags({"clear_fraction": 0.8, "water_fraction": 1.4, "ndvi_land_median": 0.5})
    assert flags == ["out_of_range:water_fraction"]
