"""Tests for NaN-aware empirical_st_variogram and its max_time_lag option."""
import numpy as np

from geolattice.geostatistics.spacetime.st_variogram_models import empirical_st_variogram


def _panel(seed=0, n=15, nt=12, frac_missing=0.3):
    rs = np.random.default_rng(seed)
    xy = rs.uniform(0, 50, (n, 2))
    z = rs.normal(size=(n, nt)).cumsum(1)
    zm = z.copy()
    zm[rs.random(z.shape) < frac_missing] = np.nan
    return xy, np.arange(nt, dtype=float), z, zm


def _brute(xy, z, lag, lo, hi, zero_bin):
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    s = n = 0.0
    for i in range(len(xy)):
        for j in range(len(xy)):
            if lag == 0 and j <= i and not (zero_bin and i == j):
                continue
            if lag == 0 and i == j and not zero_bin:
                continue
            if not (lo <= d[i, j] <= hi) if (zero_bin and lo == 0 and hi == 0) else not (lo < d[i, j] <= hi):
                continue
            a, b = z[i, lag:], z[j, : z.shape[1] - lag]
            k = np.isfinite(a) & np.isfinite(b)
            s += ((a[k] - b[k]) ** 2).sum(); n += k.sum()
    return s / (2 * n) if n else np.nan, n


def test_complete_panel_unchanged_by_nan_support():
    xy, t, z, _ = _panel()
    full = empirical_st_variogram(xy, t, z, n_space_bins=5, max_space_lag=40)
    short = empirical_st_variogram(xy, t, z, n_space_bins=5, max_space_lag=40, max_time_lag=3)
    for k in ("gamma", "np", "dist"):
        a, b = np.asarray(full[k]), np.asarray(short[k])
        assert np.allclose(a[: len(b)], b, equal_nan=True)
    assert set(np.asarray(short["timelag"])) <= {0, 1, 2, 3}


def test_gaps_use_pairwise_complete_pairs():
    xy, t, _, zm = _panel(seed=3)
    out = empirical_st_variogram(xy, t, zm, n_space_bins=4, max_space_lag=40, zero_distance_bin=True, max_time_lag=2)
    g, npair = np.asarray(out["gamma"]), np.asarray(out["np"])
    assert np.all(np.isfinite(g[npair > 0]))
    lag1_zero = (np.asarray(out["timelag"]) == 1) & (np.asarray(out["dist"]) == 0)
    gb, nb = _brute(xy, zm, 1, 0, 0, True)
    assert np.isclose(npair[lag1_zero][0], nb)
    assert np.isclose(g[lag1_zero][0], gb)
