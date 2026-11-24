import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import uav


def _grid(a, res=0.05):
    H, W = a.shape
    y = 4_700_000 - res * (np.arange(H) + 0.5)
    x = 400_000 + res * (np.arange(W) + 0.5)
    return xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)


def test_flight_plan():
    p = uav.flight_plan(100)
    assert p["gsd_cm"] == pytest.approx(13.2 * 100 / (8.8 * 5472) * 100, rel=1e-6)
    assert p["line_spacing_m"] == pytest.approx(p["footprint_w_m"] * 0.3)


def test_panel_and_empirical_line():
    ds = xr.Dataset({"red": ("x", np.array([1000.0, 2000.0])), "nir": ("x", np.array([500.0, 4000.0]))})
    cal = uav.panel_calibration(ds, {"red": 2000, "nir": 4000}, {"red": 0.5, "nir": 0.5})
    assert float(cal.red[0]) == pytest.approx(0.25) and float(cal.nir[1]) == pytest.approx(0.5)
    out, fits = uav.empirical_line(ds, {"red": [1000, 3000]}, {"red": [0.1, 0.5]})
    assert fits["red"][2] == pytest.approx(1.0) and float(out.red[1]) == pytest.approx(0.3)


def test_count_plants():
    yy, xx = np.mgrid[0:200, 0:200]
    centers = [(r, c) for r in range(20, 200, 40) for c in range(20, 200, 40)]
    a = sum(np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / 30.0) for r, c in centers)
    pts = uav.count_plants(_grid(a), min_distance_m=0.5, threshold=0.5)
    assert len(pts) == len(centers)


def test_chm_cover_cwsi():
    dsm = _grid(np.full((10, 10), 105.0))
    dtm = _grid(np.full((10, 10), 100.0))
    assert float(uav.chm_from_surfaces(dsm, dtm).mean()) == 5.0
    vi = _grid(np.r_[np.full(50, 0.1), np.full(50, 0.8)].reshape(10, 10))
    mask, cover = uav.canopy_cover(vi)
    assert cover == pytest.approx(0.5)
    tc = _grid(np.linspace(25, 35, 100).reshape(10, 10))
    c = uav.cwsi(tc, method="reference", t_wet=25, t_dry=35)
    assert float(c.min()) == 0 and float(c.max()) == 1
    s = uav.cwsi(tc)
    assert 0 <= float(s.mean()) <= 1
    e = uav.cwsi(tc, method="empirical", air_temp=30, vpd=2.0)
    assert e.attrs["t_wet"] == pytest.approx(30 + 3.38 - 3.25 * 2)  # Idso wheat baseline
