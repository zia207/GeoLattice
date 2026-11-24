"""PS-InSAR: candidate selection, phase-stability estimation, sparse unwrapping, APS, geocoding."""

import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import insar, sar


def test_amplitude_dispersion_separates_point_targets_from_speckle():
    rng = np.random.default_rng(0)
    n, shape = 30, (40, 40)
    speckle = np.abs(rng.normal(size=(n,) + shape) + 1j * rng.normal(size=(n,) + shape))  # Rayleigh
    amp = speckle.copy()
    amp[:, 10, 10] = 20 + rng.normal(0, 1, n)          # a stable bright scatterer
    gains = np.linspace(0.7, 1.4, n)[:, None, None]    # uncalibrated brightness changes between dates
    da, mean = insar.amplitude_dispersion(amp * gains)
    assert da[10, 10] < 0.1
    assert np.median(da) == pytest.approx(np.sqrt(4 / np.pi - 1), rel=0.1)   # Rayleigh speckle: 0.52
    raw, _ = insar.amplitude_dispersion(amp * gains, normalize=False)
    assert raw[10, 10] > 0.15                          # without normalisation the gains inflate D_A
    cand = insar.ps_candidates(da, 0.25, mean_amplitude=mean, peak_window=5)
    assert cand[10, 10] and cand.sum() < 5


def test_sparse_lowpass_recovers_smooth_field():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 5000, 3000), rng.uniform(0, 5000, 3000)
    field = 2 * np.sin(x / 1500) + 0.0004 * y
    noisy = field + rng.normal(0, 0.5, x.size)
    lp = insar.sparse_lowpass(noisy[None], x, y, cell=50, sigma=250)[0]
    inner = (x > 600) & (x < 4400) & (y > 600) & (y < 4400)
    assert np.std((lp - field)[inner]) < 0.15
    ph = insar.sparse_lowpass(np.exp(1j * field)[None], x, y, cell=50, sigma=250)[0]
    assert np.abs(np.angle(ph * np.exp(-1j * field))[inner]).max() < 0.2
    loo = insar.sparse_lowpass(noisy[None], x, y, cell=50, sigma=250, exclude_self=True)[0]
    assert np.std((loo - field)[inner]) < 0.2


def test_ps_periodogram_estimates_dem_error_and_rate():
    rng = np.random.default_rng(2)
    n, P = 24, 300
    kz = rng.uniform(-0.08, 0.08, (n, 1)) * np.ones((1, P))       # rad per metre (B_perp spread)
    t = np.sort(rng.uniform(0, 1, n))
    dh_true = rng.uniform(-20, 20, P)
    rate_true = rng.choice([-6.0, 0.0, 4.0], P)                       # rad/yr
    ph = np.angle(np.exp(1j * (kz * dh_true + rate_true * t[:, None] + 0.7 + rng.normal(0, 0.3, (n, P)))))
    out = insar.ps_periodogram(ph, kz, dem_error=np.arange(-30, 30.25, 0.25), time_years=t,
                               velocity_phase=np.arange(-8, 8.1, 1.0))
    assert np.median(np.abs(out["dem_error"] - dh_true)) < 1.5
    assert np.mean(out["phase_rate"] == rate_true) > 0.9
    assert np.median(out["coherence"]) > 0.9 and out["residual"].shape == (n, P)
    rnd = insar.ps_periodogram(rng.uniform(-np.pi, np.pi, (n, P)), kz)
    assert np.median(rnd["coherence"]) < 0.45


def test_coherence_threshold_reflects_number_of_images():
    rng = np.random.default_rng(3)
    few = insar.coherence_threshold(rng.uniform(-0.04, 0.04, 10), n_sim=4000)
    many = insar.coherence_threshold(rng.uniform(-0.04, 0.04, 40), n_sim=4000)
    assert 0.3 < many < few < 0.95
    with pytest.raises(ValueError, match="NaN"):
        insar.coherence_threshold(np.array([0.01, np.nan, 0.02]), n_sim=100)


@pytest.mark.parametrize("method", ["snaphu", "native"])
def test_unwrap_points_recovers_multicycle_phase(method):
    if method == "snaphu":
        pytest.importorskip("snaphu")
    rng = np.random.default_rng(4)
    x, y = rng.uniform(0, 4000, 4000), rng.uniform(0, 4000, 4000)
    truth = np.stack([0.006 * x + 0.003 * y, -0.004 * x + 8 * np.exp(-((x - 2000) ** 2 + (y - 2000) ** 2) / 8e5)])
    unw = insar.unwrap_points(np.angle(np.exp(1j * truth)), x, y, cell=100, method=method)
    for k in range(2):
        d = unw[k] - truth[k]
        ok = np.isfinite(d)
        assert ok.mean() > 0.95 and np.abs(d[ok] - np.median(d[ok])).max() < 1e-4   # float32 grids


def test_atmospheric_filter_and_velocity():
    rng = np.random.default_rng(5)
    P, n = 3000, 15
    x, y = rng.uniform(0, 8000, P), rng.uniform(0, 8000, P)
    t = np.linspace(0, 1, n)
    vel = -3 + 0.0005 * x                                   # rad/yr, varies in space
    kx, ky = rng.uniform(0.0005, 0.0015, n), rng.uniform(0.0005, 0.0015, n)
    atm = np.stack([np.sin(kx[k] * x + 2 * k) * np.cos(ky[k] * y) for k in range(n)])
    atm -= atm.mean(0)                                      # zero temporal mean per point
    unw = vel[None] * t[:, None] + atm
    aps = insar.atmospheric_filter(unw, x, y, t, spatial_sigma=300, cell=50)
    inner = (x > 800) & (x < 7200) & (y > 800) & (y < 7200)
    assert np.corrcoef(aps[:, inner].ravel(), atm[:, inner].ravel())[0, 1] > 0.9
    v = insar.point_velocity(unw - aps, t)
    assert np.median(np.abs(v["velocity"][inner] - vel[inner])) < 0.2
    assert np.median(insar.point_velocity(unw, t)["rmse"]) > np.median(v["rmse"])


def test_geocode_points_and_height_shift():
    x = np.arange(500000.0, 501000.0, 10.0)
    y = np.arange(2150000.0, 2149000.0, -10.0)
    X, Y = np.meshgrid(x, y)
    tc = xr.Dataset({"radar_row": (("y", "x"), (2150000.0 - Y) / 14.0),
                     "radar_col": (("y", "x"), (X - 500000.0) / 4.0)}, coords={"x": x, "y": y})
    gx, gy = sar.geocode_points(np.array([10.0, 30.5]), np.array([20.0, 100.25]), tc)
    np.testing.assert_allclose(gx, [500080.0, 500401.0], atol=1e-6)
    np.testing.assert_allclose(gy, [2149860.0, 2149573.0], atol=1e-6)
    # 20 m above the DEM, 35 deg incidence, descending (heading 192 deg) looks west-north-west
    hx, hy = sar.geocode_points(np.array([10.0]), np.array([20.0]), tc, height_error=np.array([20.0]),
                                incidence_deg=35.0, heading_deg=192.0)
    shift = 20 / np.tan(np.radians(35))
    assert np.hypot(hx[0] - gx[0], hy[0] - gy[0]) == pytest.approx(shift)
    assert hx[0] < gx[0]                                   # moved away from the radar, i.e. westward


def test_sequential_unwrapping_keeps_a_stable_outcrop():
    pytest.importorskip("snaphu")
    rng = np.random.default_rng(6)
    x, y = rng.uniform(0, 3000, 6000), rng.uniform(0, 3000, 6000)
    t = np.linspace(0, 1, 24)
    hill = np.hypot(x - 1500, y - 1500) < 180                 # a stable outcrop in sinking ground
    rate = np.where(hill, 0.0, -40.0)                          # rad/yr: 40 rad = 6.4 cycles in a year
    truth = t[:, None] * rate[None, :]
    wrapped = np.angle(np.exp(1j * truth))
    seq = insar.unwrap_points(wrapped, x, y, cell=100, sequential=True)
    err = seq - truth
    assert np.nanmax(np.abs(err[:, hill])) < 1e-3 and np.nanmax(np.abs(err[:, ~hill])) < 1e-3
    star = insar.unwrap_points(wrapped, x, y, cell=100)        # each date unwrapped on its own
    off = star - truth
    off -= np.nanmedian(off[:, ~hill], axis=1, keepdims=True)
    assert np.nanmax(np.abs(off[:, hill])) > 1.0               # the outcrop takes its neighbours' cycles


def test_amplitude_dispersion_needs_enough_valid_dates():
    amp = np.ones((10, 3, 3))
    amp[1:, 0, 0] = 0                                          # only one valid date
    da, _ = insar.amplitude_dispersion(amp)
    assert np.isnan(da[0, 0]) and da[1, 1] == 0
    out = insar.ps_periodogram(np.zeros((4, 2)), np.array([[0.1, np.nan]] * 4))
    assert np.isnan(out["coherence"][1]) and out["coherence"][0] == pytest.approx(1.0)
