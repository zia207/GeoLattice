import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import timeseries as ts


def _cube(values, t):
    v = np.asarray(values, float)
    return xr.DataArray(np.broadcast_to(v[:, None, None], (len(t), 3, 4)).copy(),
                        dims=("time", "y", "x"), coords={"time": t}, name="ndvi")


def _season(t):
    doy = t.dayofyear.values
    return 0.2 + 0.6 * np.exp(-((doy - 200) / 40.0) ** 2)


def test_smoothers_fill_and_denoise():
    t = pd.date_range("2026-01-01", "2026-12-31", freq="5D")
    truth = _season(t)
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 0.03, truth.size)
    noisy[10:14] = np.nan
    da = _cube(noisy, t)
    for out in (ts.whittaker(da, lmbda=20), ts.savgol(da, window=9), ts.interpolate(da)):
        assert out.notnull().all()
    w = ts.whittaker(da, lmbda=20).values[:, 0, 0]
    assert np.sqrt(np.mean((w - truth) ** 2)) < 0.025


def test_harmonic_fit():
    t = pd.date_range("2026-01-01", periods=73, freq="5D")
    days = (t - t[0]).days.values
    y = 0.5 + 0.2 * np.cos(2 * np.pi * days / 365.25) + 0.01 * days / 365.25
    out = ts.harmonic_fit(_cube(y, t), n_harmonics=1)
    assert float(out.amplitude[0, 0]) == pytest.approx(0.2, rel=1e-3)


def test_phenology():
    t = pd.date_range("2026-01-01", "2026-12-31", freq="5D")
    ph = ts.phenology(_cube(_season(t), t), threshold=0.5)
    # half-maximum of a Gaussian: 200 ± 40*sqrt(ln 2)
    half = 40 * np.sqrt(np.log(2))
    assert float(ph.sos[0, 0]) == pytest.approx(200 - half, abs=2)
    assert float(ph.eos[0, 0]) == pytest.approx(200 + half, abs=2)
    assert float(ph.amplitude[0, 0]) == pytest.approx(0.6, abs=0.01)


def test_bfast_monitor_detects_drop():
    t = pd.date_range("2020-01-01", "2025-12-31", freq="16D")
    days = (t - t[0]).days.values
    rng = np.random.default_rng(1)
    y = 0.6 + 0.15 * np.sin(2 * np.pi * days / 365.25) + rng.normal(0, 0.02, len(t))
    stable = _cube(y, t)
    y2 = y.copy()
    y2[t >= "2024-06-01"] -= 0.3
    da = xr.concat([stable.isel(x=[0, 1]), _cube(y2, t).isel(x=[0, 1])], "x")
    res = ts.bfast_monitor(da, "2024-01-01")
    assert not bool(res.detected[0, 0]) and bool(res.detected[0, 2])
    bp = pd.Timestamp(res.breakpoint.values[0, 2])
    assert pd.Timestamp("2024-06-01") <= bp <= pd.Timestamp("2024-10-01")
    assert float(res.magnitude[0, 2]) < -0.2


def test_ccdc_detects_break():
    t = pd.date_range("2018-01-01", "2023-12-31", freq="16D")
    days = (t - t[0]).days.values
    rng = np.random.default_rng(2)
    base = 0.3 + 0.05 * np.sin(2 * np.pi * days / 365.25)
    red = base + rng.normal(0, 0.01, len(t))
    nir = 0.4 + 0.1 * np.sin(2 * np.pi * days / 365.25) + rng.normal(0, 0.01, len(t))
    nir_b = nir.copy()
    nir_b[t >= "2021-03-01"] -= 0.2
    ds = xr.Dataset({"red": xr.concat([_cube(red, t), _cube(red, t)], "x"),
                     "nir": xr.concat([_cube(nir, t), _cube(nir_b, t)], "x")})
    out = ts.ccdc(ds)
    assert int(out.n_breaks[0, 0]) == 0
    assert int(out.n_breaks[0, 5]) >= 1
    fb = pd.Timestamp(out.first_break.values[0, 5])
    assert pd.Timestamp("2021-02-01") <= fb <= pd.Timestamp("2021-05-01")


def test_anomalies_and_vci():
    t = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    y = 0.5 + 0.2 * np.sin(2 * np.pi * np.asarray(t.month) / 12)
    y = y.astype(float)
    y[-1] -= 0.1
    da = _cube(y, t)
    an = ts.anomaly(da)
    assert float(an.isel(time=-1)[0, 0]) < -0.05
    z = ts.anomaly(da, kind="standardized")
    assert float(z.isel(time=-1)[0, 0]) < -1
    v = ts.vci(da)
    assert float(v.isel(time=-1)[0, 0]) == pytest.approx(0)
    assert float(v.max()) == pytest.approx(100)
