"""SBAS: network, interferogram stack, loop-closure correction, SVD inversion, DEM-error correction."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import insar, sar
from tests.rsense.test_insar_topo import LON0, _ann, _orbit

DATES = [str(d.date()) for d in pd.date_range("2025-01-01", periods=10, freq="12D")]


def test_network_thresholds_and_components():
    bperp = np.array([0, 20, 150, 30, 10, -20, 40, 0, 10, 5], float)
    net = insar.sbas_network(DATES, bperp, max_temporal_days=36, max_perpendicular=100)
    assert all((pd.Timestamp(b) - pd.Timestamp(a)).days <= 36 for a, b in net)
    assert all(abs(bperp[DATES.index(b)] - bperp[DATES.index(a)]) <= 100 for a, b in net)
    assert DATES[2] not in {d for p in net for d in p}          # the 150 m outlier is excluded
    seq = insar.sbas_network(DATES, n_neighbors=2)
    assert len(seq) == 9 + 8
    split = [(a, b) for a, b in seq if not (a <= DATES[4] < b)]  # remove every pair spanning the gap after date 4
    comps = insar.network_components(split, DATES)
    assert len(comps) == 2 and sorted(comps) == [DATES[:5], DATES[5:]]
    joined = insar.sbas_network(DATES, n_neighbors=1, max_temporal_days=12, connect=True)
    assert len(insar.network_components(joined, DATES)) == 1


def _series(n_px=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.array([(pd.Timestamp(d) - pd.Timestamp(DATES[0])).days / 365.25 for d in DATES])
    rate = rng.uniform(-30, 5, n_px)                             # rad/yr
    phi = t[:, None] * rate[None, :] + rng.normal(0, 0.2, (len(DATES), n_px))
    phi -= phi[0]
    return t, phi


def test_svd_inversion_connected_weighted_and_disconnected():
    t, phi = _series()
    pairs = insar.sbas_network(DATES, n_neighbors=3)
    idx = {d: i for i, d in enumerate(DATES)}
    ifg = np.stack([phi[idx[b]] - phi[idx[a]] for a, b in pairs])
    ts = insar.sbas_svd(ifg, pairs)
    np.testing.assert_allclose(ts, phi, atol=1e-9)
    w = np.full(ifg.shape, 0.7)
    ifg_nan = ifg.copy()
    ifg_nan[0, :5] = np.nan                                      # a missing interferogram at a few pixels
    np.testing.assert_allclose(insar.sbas_svd(ifg_nan, pairs, weights=w), phi, atol=1e-9)
    # a network split into two subsets: SVD still gives a series, exact within each subset
    split = [(a, b) for a, b in pairs if not (a <= DATES[4] < b)]
    ifg_s = np.stack([phi[idx[b]] - phi[idx[a]] for a, b in split])
    ts_s = insar.sbas_svd(ifg_s, split)
    assert np.all(np.isfinite(ts_s))
    np.testing.assert_allclose(ts_s[:5] - ts_s[0], phi[:5] - phi[0], atol=1e-9)
    np.testing.assert_allclose(ts_s[5:] - ts_s[5], phi[5:] - phi[5], atol=1e-9)
    with pytest.raises(ValueError):                              # the plain inversion refuses the split network
        insar.sbas_inversion(ifg_s, split)
    da = insar.sbas_svd(xr.DataArray(ifg[:, :, None], dims=("pair", "y", "x")), pairs)
    assert da.dims == ("time", "y", "x") and da.sizes["time"] == len(DATES)


def test_loop_closure_corrects_injected_cycle_errors():
    t, phi = _series(n_px=200, seed=1)
    pairs = insar.sbas_network(DATES, n_neighbors=3)
    idx = {d: i for i, d in enumerate(DATES)}
    truth = np.stack([phi[idx[b]] - phi[idx[a]] for a, b in pairs])
    bad = truth.copy()
    bad[4, :60] += 2 * np.pi                                     # one interferogram off by a cycle
    bad[10, 100:130] -= 4 * np.pi                                # another off by two cycles
    closure, tri = insar.loop_closure(bad, pairs)
    assert len(tri) > 5 and np.abs(np.round(closure / (2 * np.pi))).max() >= 2
    fixed, info = insar.correct_unwrapping_errors(bad, pairs)
    np.testing.assert_allclose(fixed, truth, atol=1e-9)
    assert info["pixels_corrected"] == 90 and info["patterns"] == 2
    clean, info0 = insar.correct_unwrapping_errors(truth, pairs)
    assert info0["pixels_corrected"] == 0 and np.array_equal(clean, truth)


def test_dem_error_correction_recovers_height_error():
    rng = np.random.default_rng(3)
    n, P = len(DATES), 400
    t = np.array([(pd.Timestamp(d) - pd.Timestamp(DATES[0])).days / 365.25 for d in DATES])
    kz = rng.uniform(-3e-4, 3e-4, (n, 1)) * np.ones((1, P))      # metres of range per metre of height
    kz -= kz[0]
    v = rng.uniform(-0.3, 0.05, P)                               # m/yr
    dz = rng.uniform(-30, 30, P)                                 # m
    los = v[None] * t[:, None] - kz * dz[None] + rng.normal(0, 2e-4, (n, P))
    corrected, est = insar.dem_error_correction(los, -kz, t)
    assert np.median(np.abs(est - dz)) < 1.0
    resid = corrected - v[None] * t[:, None]
    assert np.std(resid - resid.mean(0)) < 5e-4
    c2, e2 = insar.dem_error_correction(los, -kz[:, 0], t, seasonal=True)   # one sensitivity per date
    assert np.median(np.abs(e2 - dz)) < 2.0


def test_interferogram_stack_on_coregistered_bursts():
    ref_orb = _orbit()
    tc, rc = sar.geo2rdr([0.0], [LON0 + 2.7], [0.0], ref_orb)
    near = float(rc[0]) - 60 * 2.33 / 2

    def member(orbit, seed):
        ann = _ann(orbit, near)
        ann["bursts"][0]["azimuth_time"] = tc[0] - np.timedelta64(150, "ms")
        ann["bursts"][1]["azimuth_time"] = tc[0] + np.timedelta64(10, "ms")
        ann.update(heading=0.0, range_pixel_spacing=2.33, azimuth_pixel_spacing=14.0)
        rng = np.random.default_rng(seed)
        slcs = [(rng.normal(size=(100, 60)) + 1j * rng.normal(size=(100, 60))).astype("complex64") for _ in range(2)]
        return {"slc": slcs, "ann": [ann, ann], "burst": [0, 1], "col0": [0, 0]}

    ref = member(ref_orb, 1)
    lat = np.arange(0.012, -0.012, -0.0002)
    lon = np.arange(LON0 + 2.696, LON0 + 2.704, 0.0002)
    dem = xr.DataArray(np.zeros((lat.size, lon.size)), coords={"y": lat, "x": lon}, dims=("y", "x")).rio.write_crs(4326)
    co = sar.coregister_stack(ref, {"2025-01-13": dict(ref), "2025-01-25": dict(ref)}, dem, esd=False)
    pairs = [("2025-01-01", "2025-01-13"), ("2025-01-13", "2025-01-25"), ("2025-01-01", "2025-01-25")]
    ds = insar.interferogram_stack(ref, co, pairs, reference_date="2025-01-01", looks=(2, 4), unwrap=False)
    assert set(ds.data_vars) >= {"phase", "coherence", "kz", "incidence", "dem_height"}
    assert ds.sizes["pair"] == 3 and ds.sizes["date"] == 3 and list(ds.reference.values) == [p[0] for p in pairs]
    core = ds.coherence.isel(y=slice(10, -10), x=slice(2, -2))
    assert float(core.median()) > 0.95                            # identical images: coherent, zero phase
    assert float(np.abs(ds.phase.isel(y=slice(10, -10), x=slice(2, -2))).max()) < 0.05
    assert ds.attrs["range_spacing"] == pytest.approx(4 * ref["ann"][0]["range_pixel_spacing"])
