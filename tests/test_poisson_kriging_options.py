"""Tests for AreaPoissonKriging(exclude_self) and empirical_st_variogram(zero_distance_bin)."""
import numpy as np
import pandas as pd

from geolattice.geostatistics import Variogram, PointSupport, AreaPoissonKriging
from geolattice.geostatistics.spacetime.st_variogram_models import empirical_st_variogram


def _toy_areas(n=6, m=20, seed=0):
    rs = np.random.default_rng(seed)
    centres = rs.uniform(0, 100, (n, 2))
    pts, ids = [], []
    for i, c in enumerate(centres):
        pts.append(c + rs.normal(0, 3, (m, 2))); ids += [i] * m
    pts = np.vstack(pts)
    pop = rs.integers(1000, 50000, n).astype(float)
    w = np.repeat(pop / m, m)
    rate = rs.uniform(50, 150, n)
    vg = Variogram(centres, rate, model="exponential").set_params(50.0, 400.0, 30.0)
    apk = AreaPoissonKriging(vg, rate_base=1e5).fit(PointSupport(pts, w, np.array(ids)), pd.Series(rate))
    return apk, rate


def test_exclude_self_default_is_leave_one_out():
    apk, _ = _toy_areas()
    a = apk.predict_area(0, number_of_neighbors=4)
    b = apk.predict_area(0, number_of_neighbors=4, exclude_self=True)
    assert a == b


def test_exclude_self_false_uses_own_rate_and_is_coherent():
    apk, rate = _toy_areas()
    f = apk.predict_area(0, number_of_neighbors=5, exclude_self=False)
    loo = apk.predict_area(0, number_of_neighbors=5)
    # the filtered estimate lies closer to the area's own rate than the leave-one-out one
    assert abs(f["zhat"] - rate[0]) < abs(loo["zhat"] - rate[0])
    pp = apk.predict_points(0, number_of_neighbors=5, exclude_self=False)
    assert np.isclose(np.average(pp["zhat"], weights=pp["population"]), f["zhat"])


def test_zero_distance_bin_separates_same_area_pairs():
    rs = np.random.default_rng(1)
    coords = rs.uniform(0, 100, (15, 2))
    area_effect = rs.normal(0, 3, 15)[:, None]
    Z = area_effect + rs.normal(0, 0.3, (15, 8))          # strong persistence, little temporal noise
    t = np.arange(8.0)
    e0 = empirical_st_variogram(coords, t, Z, n_space_bins=5)
    e1 = empirical_st_variogram(coords, t, Z, n_space_bins=5, zero_distance_bin=True)
    assert not np.any(e0["dist"] == 0)
    z = (e1["dist"] == 0) & (e1["timelag"] == 1)
    assert z.sum() == 1
    first = (e1["dist"] > 0) & (e1["timelag"] == 1)
    assert e1["gamma"][z][0] < 0.2 * e1["gamma"][first].min()
    # cells at lag 0 are unchanged
    assert np.allclose(e0["gamma"][e0["timelag"] == 0], e1["gamma"][e1["timelag"] == 0])
