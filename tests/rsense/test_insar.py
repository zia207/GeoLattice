import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import insar


@pytest.fixture
def slc_pair():
    rng = np.random.default_rng(0)
    H, W = 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    phase = 0.15 * xx + 0.05 * yy                           # smooth ramp (needs unwrapping)
    s1 = rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))
    noise = 0.05 * (rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W)))
    s2 = s1 * np.exp(-1j * phase) + noise
    return s1, s2, phase


def test_interferogram_phase_and_coherence(backend, slc_pair):
    s1, s2, phase = slc_pair
    out = insar.interferogram(s1, s2, window=(5, 5))
    err = np.angle(np.exp(1j * (np.asarray(out["phase"]) - phase)))
    assert np.abs(err[5:-5, 5:-5]).mean() < 0.15
    assert np.asarray(out["coherence"])[5:-5, 5:-5].mean() > 0.9
    rng = np.random.default_rng(9)
    s3 = rng.normal(size=s1.shape) + 1j * rng.normal(size=s1.shape)
    assert np.asarray(insar.interferogram(s1, s3)["coherence"]).mean() < 0.4


def test_unwrap_native_and_ramp(slc_pair):
    s1, s2, phase = slc_pair
    ifg = insar.interferogram(s1, s2, window=(3, 3))["interferogram"]
    unw = insar.unwrap(ifg, method="native")
    d = unw - phase
    assert np.nanstd(d) < 0.15                           # equal up to a constant
    flat = insar.remove_ramp(unw)
    assert np.nanstd(flat) < 0.2


def test_goldstein_and_multilook(slc_pair):
    s1, s2, _ = slc_pair
    ifg = insar.interferogram(s1, s2, window=(1, 1))["interferogram"]
    f = insar.goldstein_filter(ifg, alpha=0.5, patch=32, overlap=8)
    assert f.shape == ifg.shape and np.isfinite(f).all()
    ml = insar.multilook_complex(ifg, (4, 2))
    assert ml.shape == (16, 32)


def test_los_conversions_and_decomposition():
    d = insar.phase_to_los(np.array([-np.pi]), wavelength=0.056)
    assert d[0] == pytest.approx(0.056 / 4)
    assert insar.los_to_vertical(np.array([0.01]), 60.0)[0] == pytest.approx(0.02)
    # forward-model LOS from known (east, up), then invert
    east, up = 0.01, -0.03

    def los(inc, head):
        inc_r, look = np.radians(inc), np.radians(head) + np.pi / 2
        return -np.sin(inc_r) * np.sin(look) * east + np.cos(inc_r) * up

    la, ld = los(39.0, -10.0), los(41.0, -170.0)
    res = insar.decompose_asc_desc(la, ld, 39.0, 41.0)
    assert res["east"] == pytest.approx(east, abs=1e-9) and res["up"] == pytest.approx(up, abs=1e-9)


def test_sbas_and_velocity():
    dates = pd.date_range("2026-01-01", periods=6, freq="12D")
    truth = np.array([0, -2, -4, -6, -8, -10.0])          # mm, linear subsidence
    pairs = [(dates[i], dates[j]) for i in range(6) for j in range(i + 1, min(i + 3, 6))]
    ifgs = np.array([truth[dates.get_loc(b)] - truth[dates.get_loc(a)] for a, b in pairs])
    ifgs = np.broadcast_to(ifgs[:, None, None], (len(pairs), 3, 4)).copy()
    da = xr.DataArray(ifgs, dims=("pair", "y", "x"))
    ts = insar.sbas_inversion(da, pairs)
    np.testing.assert_allclose(ts.values[:, 0, 0], truth, atol=1e-9)
    v = insar.velocity(ts.assign_attrs(units="mm"))
    assert float(v.velocity[0, 0]) == pytest.approx(-10 / (60 / 365.25), rel=1e-6)
    with pytest.raises(ValueError, match="disconnected"):
        insar.sbas_inversion(da[:2], [(dates[0], dates[1]), (dates[3], dates[4])])


def test_read_gacos(tmp_path):
    data = np.arange(12, dtype="<f4").reshape(3, 4)
    p = tmp_path / "20260101.ztd"
    data.tofile(p)
    (tmp_path / "20260101.ztd.rsc").write_text(
        "WIDTH 4\nFILE_LENGTH 3\nX_FIRST 90.0\nY_FIRST 24.0\nX_STEP 0.01\nY_STEP -0.01\n")
    da = insar.read_gacos(p)
    assert da.shape == (3, 4) and float(da[1, 2]) == 6.0 and da.rio.crs.to_epsg() == 4326


def test_wrappers_write_configs(tmp_path):
    xml = insar.isce.write_topsapp_xml(["a.zip"], ["b.zip"], tmp_path, bbox=(23, 24, 90, 91))
    assert "region of interest" in xml.read_text()
    cfg = insar.mintpy.write_config(tmp_path, reference_lalo=(23.5, 90.4))
    assert "mintpy.reference.lalo = 23.5,90.4" in cfg.read_text()
