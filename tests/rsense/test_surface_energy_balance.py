import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geolattice.rsense.surface_energy_balance as seb


# ----------------------------------------------------------------------------- worked examples
def test_pressure_fao56_example2():
    assert seb.pressure_from_elevation(1800) == pytest.approx(81.8, abs=0.1)


def test_vapour_pressure_curve():
    assert seb.saturation_vapour_pressure(25.0) == pytest.approx(3.168, abs=0.002)
    assert seb.slope_vapour_pressure_curve(25.0) == pytest.approx(0.189, abs=0.001)


def test_ra_daily_fao56_example8():
    # 20 S, 3 September: Ra = 32.2 MJ m-2 d-1
    assert seb.extraterrestrial_radiation_daily(-20.0, 246) == pytest.approx(32.2, abs=0.1)


def test_eto_daily_fao56_example18():
    # Brussels, 6 July: ETo = 3.9 mm/day
    eto = seb.fao56_eto(21.5, 12.3, 1.409, 22.07, 2.078, 100, 50.8, 187)
    assert eto == pytest.approx(3.9, abs=0.05)


def test_eto_hourly_fao56_example19():
    # N'Diaye, Senegal, 1 October 14:00 to 15:00 local (midpoint 15:30 UTC): ETo = 0.63 mm/h
    ea = seb.actual_vapour_pressure(t_c=38.0, rh=52.0)
    ra = seb.extraterrestrial_radiation_hourly(16.217, -16.25, "2026-10-01 15:30")
    assert ra == pytest.approx(3.543, abs=0.03)
    eto = seb.asce_pm_hourly(38.0, ea, 2.450, 3.3, 8, 16.217, -16.25, "2026-10-01 15:30",
                             cd=0.34, g_ratio=0.1)
    assert eto == pytest.approx(0.63, abs=0.02)


def test_tall_reference_exceeds_short():
    short = seb.asce_pm_daily(32.0, 15.0, 1.2, 28.0, 3.0, 100, 36.5, 190)
    tall = seb.asce_pm_daily(32.0, 15.0, 1.2, 28.0, 3.0, 100, 36.5, 190, reference="tall")
    assert tall > short * 1.15


def test_wind_2m():
    assert seb.wind_2m(10 / 3.6, 10) == pytest.approx(2.078, abs=0.005)


# ----------------------------------------------------------------------------- geometry
def test_solar_position_equinox_noon():
    pos = seb.solar_position(0.0, 0.0, "2026-03-20 12:07")
    assert pos["zenith"] < 2.0


def test_cos_incidence_flat_matches_zenith():
    when = "2026-07-05 18:40"
    pos = seb.solar_position(36.4, -120.1, when)
    c = seb.cos_incidence(36.4, when, -120.1)
    assert c == pytest.approx(np.cos(np.radians(pos["zenith"])), abs=1e-9)


def test_slope_aspect_east_facing():
    y = 1000 - 30 * np.arange(10)
    x = 30 * np.arange(12)
    dem = xr.DataArray(np.tile(-0.1 * x, (10, 1)), coords={"y": y, "x": x}, dims=("y", "x"))
    slope, aspect = seb.slope_aspect(dem)
    assert float(slope.mean()) == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-4)
    assert float(aspect.mean()) == pytest.approx(90.0, abs=1e-4)


def test_scene_geometry_mtl():
    meta = {"SUN_ELEVATION": 64.2, "SUN_AZIMUTH": 120.5, "EARTH_SUN_DISTANCE": 1.0167,
            "DATE_ACQUIRED": "2024-07-05", "SCENE_CENTER_TIME": "18:40:12.3456789Z"}
    g = seb.scene_geometry(meta)
    assert g["doy"] == 187 and g["sun_zenith"] == pytest.approx(25.8)
    assert g["dr"] == pytest.approx(1 / 1.0167 ** 2)


# ----------------------------------------------------------------------------- radiation pieces
def test_albedo_aliases_and_range():
    sr = xr.Dataset({k: xr.DataArray(np.full((3, 3), v)) for k, v in
                     {"SR_B2": 0.05, "SR_B4": 0.06, "SR_B5": 0.4, "SR_B6": 0.2,
                      "SR_B7": 0.1}.items()})
    a = seb.albedo(sr, sensor="landsat8")
    expected = 0.356 * 0.05 + 0.130 * 0.06 + 0.373 * 0.4 + 0.085 * 0.2 + 0.072 * 0.1 - 0.0018
    np.testing.assert_allclose(a.values, expected)


def test_lai_and_emissivity():
    s = np.array([0.0, 0.3, 0.69])
    lai = seb.leaf_area_index(s)
    assert lai[0] == 0 and lai[-1] == 6.0 and 0 < lai[1] < 6
    e = seb.surface_emissivity(lai, np.array([-0.1, 0.4, 0.8]))
    assert e[0] == 0.985 and e[-1] == 0.98


def test_net_radiation_balance():
    rs, alb, rl_in, eps0, ts = 850.0, 0.2, 330.0, 0.97, 305.0
    rl_out = seb.outgoing_longwave(ts, eps0)
    rn = seb.net_radiation(rs, alb, rl_in, rl_out, eps0)
    assert rn == pytest.approx((1 - alb) * rs + eps0 * rl_in - rl_out)


def test_transmissivity_reasonable():
    tau = seb.broadband_transmissivity(0.9, pressure=100.0, ea=1.5)
    assert 0.7 < tau < 0.8


def test_soil_heat_flux_methods():
    rn = np.array([600.0, 600.0])
    g1 = seb.soil_heat_flux(rn, np.array([300.0, 320.0]), np.array([0.2, 0.25]),
                            np.array([0.8, 0.15]))
    assert g1[0] < g1[1]
    g2 = seb.soil_heat_flux(rn, np.array([300.0, 320.0]), lai=np.array([4.0, 0.2]),
                            method="metric")
    assert g2[0] == pytest.approx(600 * (0.05 + 0.18 * np.exp(-0.521 * 4)))


def test_stability_corrections():
    pm, ph = seb.stability_corrections(2.0, np.inf)
    assert pm == 0 and ph == 0
    pm, ph = seb.stability_corrections(2.0, -10.0)
    assert pm > 0 and ph > pm
    pm, ph = seb.stability_corrections(2.0, 50.0)
    assert pm == pytest.approx(-0.2)


# ----------------------------------------------------------------------------- synthetic scene
@pytest.fixture
def scene():
    """A 30 x 40 scene: irrigated crops in the west grading to dry fallow in the east."""
    rng = np.random.default_rng(1)
    ny, nx = 30, 40
    frac = np.tile(np.linspace(1, 0, nx), (ny, 1))
    ndvi = 0.1 + 0.75 * frac + rng.normal(0, 0.01, (ny, nx))
    lst = 300.0 + 22 * (1 - frac) + rng.normal(0, 0.3, (ny, nx))
    alb = 0.22 - 0.03 * frac
    y = 4_040_000 - 30 * (np.arange(ny) + 0.5)
    x = 740_000 + 30 * (np.arange(nx) + 0.5)

    def da(v):
        return xr.DataArray(v, coords={"y": y, "x": x}, dims=("y", "x")).rio.write_crs(32610)

    ndvi, lst, alb = da(ndvi), da(lst), da(alb)
    lai = seb.leaf_area_index(np.clip(ndvi * 0.9, -1, 1))
    e0 = seb.surface_emissivity(lai, ndvi)
    tau = 0.75
    rs_in = seb.incoming_shortwave(0.9, tau)
    rl_in = seb.incoming_longwave(300.0, seb.air_emissivity(tau_sw=tau))
    rn = seb.net_radiation(rs_in, alb, rl_in, seb.outgoing_longwave(lst, e0), e0)
    g = seb.soil_heat_flux(rn, lst, alb, ndvi)
    zom = seb.momentum_roughness(lai=lai)
    return {"ndvi": ndvi, "lst": lst, "albedo": alb, "lai": lai, "rn": rn, "g": g, "zom": zom,
            "u200": seb.wind_at_blending_height(3.0), "pressure": 100.5}


def test_select_anchors(scene):
    a = seb.select_anchors(scene["ndvi"], scene["lst"], albedo_=None)
    assert a["cold"]["ndvi"] > 0.75 and a["hot"]["ndvi"] < 0.3
    assert a["hot"]["lst"] - a["cold"]["lst"] > 15
    rep = seb.anchor_report(a, rn=scene["rn"])
    assert list(rep.index) == ["cold", "hot"] and "rn" in rep


def test_candidate_anchors_albedo_screen_falls_back(scene):
    dark = scene["albedo"] * 0 + 0.12  # every pixel darker than the 0.18 to 0.25 screen
    with pytest.warns(RuntimeWarning, match="albedo"):
        cand = seb.candidate_anchors(scene["ndvi"], scene["lst"], albedo_=dark)
    assert bool(cand["cold"].any()) and bool(cand["hot"].any())


def test_candidate_anchors_hot_range_falls_back(scene):
    low = scene["ndvi"] - 0.2  # every bare pixel below the 0.10 to 0.28 range (but above 0)
    with pytest.warns(RuntimeWarning, match="NDVI range"):
        cand = seb.candidate_anchors(low.clip(min=0.01), scene["lst"])
    assert bool(cand["hot"].any())


def test_sebal_anchor_constraints(scene):
    s = scene
    a = seb.select_anchors(s["ndvi"], s["lst"])
    out = seb.run_sebal(s["lst"], s["rn"], s["g"], s["zom"], s["u200"], s["pressure"],
                        anchors=a, clip_ef=False)
    assert isinstance(out, xr.Dataset) and out.rio.crs is not None
    (hy, hx), (cy, cx) = (a["hot"]["iy"], a["hot"]["ix"]), (a["cold"]["iy"], a["cold"]["ix"])
    assert float(out.le[hy, hx]) == pytest.approx(0.0, abs=1.0)
    assert float(out.h[cy, cx]) == pytest.approx(0.0, abs=1.0)
    assert out.attrs["converged"] == 1 and out.attrs["b"] > 0
    assert float(out.ef[:, :5].mean()) > float(out.ef[:, -5:].mean())


def test_metric_anchor_constraints(scene):
    s = scene
    a = seb.select_anchors(s["ndvi"], s["lst"])
    out = seb.run_metric(s["lst"], s["rn"], s["g"], s["zom"], s["u200"], s["pressure"],
                         etr_inst=0.8, anchors=a, etr_daily=9.0, clip_etrf=None)
    cy, cx = a["cold"]["iy"], a["cold"]["ix"]
    hy, hx = a["hot"]["iy"], a["hot"]["ix"]
    assert float(out.etrf[cy, cx]) == pytest.approx(1.05, abs=0.01)
    assert float(out.etrf[hy, hx]) == pytest.approx(0.0, abs=0.01)
    np.testing.assert_allclose(out.et24, out.etrf * 9.0)


def test_monte_carlo_anchors(scene):
    s = scene
    cand = seb.candidate_anchors(s["ndvi"], s["lst"])

    def run(anchors):
        return seb.run_metric(s["lst"], s["rn"], s["g"], s["zom"], s["u200"], s["pressure"],
                              etr_inst=0.8, anchors=anchors, etr_daily=9.0).et24

    stack, used = seb.monte_carlo_anchors(run, cand, n=5)
    assert stack.sizes["draw"] == used["ok"].sum() == 5


def test_ssebop(scene):
    s = scene
    tc, c = seb.cold_reference_temperature(s["lst"], s["ndvi"], 305.0)
    assert 0.95 < c < 1.0
    dt = seb.dt_from_climate(250.0, 305.0, 100.5)
    out = seb.run_ssebop(s["lst"], tc, dt, eto=8.0)
    assert float(out.etf[:, :3].mean()) > 0.8 and float(out.etf[:, -3:].mean()) < 0.3
    assert float(out.eta.max()) <= 1.05 * 8.0 + 1e-6


def test_ssebi_and_triangle():
    # albedo and cover vary independently of soil moisture m, as in a real scene
    rng = np.random.default_rng(3)
    shape = (60, 60)
    m = rng.uniform(0, 1, shape)
    alb = rng.uniform(0.10, 0.30, shape)
    fc = rng.uniform(0, 1, shape)
    rn, g = np.full(shape, 600.0), np.full(shape, 60.0)
    ts1 = 298 + (1 - m) * (15 + 60 * (alb - 0.10))
    o1 = seb.ssebi(alb, ts1, rn, g)
    ts2 = 298 + (1 - m) * 25 * (1 - 0.6 * fc)
    o2 = seb.triangle_method(fc, ts2, rn, g, 300.0, 100.5)
    for o in (o1, o2):
        assert o["ef"].min() >= 0 and o["ef"].max() <= 1
    assert np.corrcoef(o1["ef"].ravel(), m.ravel())[0, 1] > 0.95
    # the triangle gives full cover a higher phi at the dry edge: phi = phi_max (fc + (1 - fc) m)
    phi_true = 1.26 * (fc + (1 - fc) * m)
    assert np.corrcoef(o2["phi"].ravel(), phi_true.ravel())[0, 1] > 0.95
    np.testing.assert_allclose(o1["le"], o1["ef"] * 540.0)


def test_pt_jpl(scene):
    s = scene
    sv = seb.savi(xr.zeros_like(s["ndvi"]) + 0.05, xr.zeros_like(s["ndvi"]) + 0.05
                  + 0.5 * s["ndvi"])
    out = seb.priestley_taylor_jpl(s["rn"], s["ndvi"], sv, 30.0, 1.5, 100.5, g=s["g"])
    np.testing.assert_allclose(out.le, out.le_c + out.le_s + out.le_i)
    assert float(out.le_c[:, :5].mean()) > float(out.le_c[:, -5:].mean())


def test_tseb_energy_conservation(scene):
    s = scene
    out = seb.tseb_pt(s["lst"], s["lai"], 300.0, 3.0, 1.5, 100.5, s["rn"], 25.0, hc=0.8)
    np.testing.assert_allclose(out.rn_s + out.rn_c, s["rn"], rtol=1e-6)
    np.testing.assert_allclose(out.le + out.h_s + out.h_c + out.g, s["rn"], atol=1e-6)
    assert float(out.le_s.min()) >= 0
    t_et = float(out.t_et[:, :5].mean())
    assert 0.5 < t_et <= 1


def test_tseb_tall_canopy_with_2m_forcing_has_no_gaps():
    # 4 m orchard forced with 2 m air temperature: displacement height is above 2 m
    out = seb.tseb_pt(np.array([320.0]), np.array([1.5]), 313.0, 2.2, 0.7, 100.6,
                      np.array([560.0]), 23.5, hc=4.0, z_u=10.0, z_t=2.0, clumping=0.7)
    for v in ("le", "le_c", "le_s", "t_soil", "t_canopy"):
        assert np.isfinite(out[v]).all(), v
    assert out["le_c"][0] > 0


def test_tseb_bare_hot_soil_is_dry():
    # hot bare fallow: one-source branch, nearly all available energy goes to H
    out = seb.tseb_pt(np.array([335.0]), np.array([0.0]), 313.0, 2.2, 0.7, 100.6,
                      np.array([480.0]), 23.5)
    assert out["le_c"][0] == 0 and out["le"][0] < 0.25 * (480 - out["g"][0])


# ----------------------------------------------------------------------------- sharpening
def test_aggregate_lst_emissive():
    a = np.array([[300.0, 310.0], [300.0, 310.0]])
    out = seb.aggregate_lst(a, 2)
    assert out[0, 0] == pytest.approx(((300 ** 4 + 310 ** 4) / 2) ** 0.25)
    assert out[0, 0] > 305


def test_thermal_sharpen_recovers_fine_pattern():
    # smooth fields with variation both within and between coarse pixels
    yy, xx = np.mgrid[0:60, 0:60]
    nir = 0.3 + 0.15 * np.sin(xx / 6.0) * np.cos(yy / 8.0)
    red = 0.1 + 0.05 * np.cos(xx / 9.0 + yy / 11.0)
    fine = 320 - 40 * (nir - red) / (nir + red)
    coarse = seb.aggregate_lst(fine, 6, emissive=False)
    out = seb.thermal_sharpen(coarse, np.stack([red, nir]), factor=6, model="forest",
                              smooth_residuals=False)
    np.testing.assert_allclose(seb.aggregate_lst(out, 6, emissive=False), coarse, atol=1e-6)
    assert np.corrcoef(out.ravel(), fine.ravel())[0, 1] > 0.9


# ----------------------------------------------------------------------------- upscaling
def test_interpolate_and_seasonal():
    t = pd.to_datetime(["2024-07-01", "2024-07-11"])
    stack = xr.DataArray(np.array([[[0.2]], [[0.8]]]), coords={"time": t, "y": [0], "x": [0]},
                         dims=("time", "y", "x"), name="etrf")
    daily = seb.interpolate_etrf(stack)
    assert daily.sizes["time"] == 11
    assert float(daily.sel(time=pd.Timestamp("2024-07-06")).squeeze()) == pytest.approx(0.5)
    etr = pd.Series(10.0, index=pd.date_range("2024-07-01", "2024-07-11"))
    total = seb.seasonal_et(daily, etr)
    assert float(total.squeeze()) == pytest.approx(10 * 0.5 * 11)


def test_daily_conversions():
    assert seb.instantaneous_to_daily(0.8, etr24=10.0) == pytest.approx(8.0)
    assert seb.instantaneous_to_daily(0.5, method="ef", rn24=150.0) == pytest.approx(
        86400 * 0.5 * 150 / 2.45e6)
    assert seb.instantaneous_et(680.6) == pytest.approx(1.0, abs=0.001)


def test_zonal_et():
    et = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]),
                      coords={"y": [45.0, 15.0], "x": [15.0, 45.0]}, dims=("y", "x"))
    zones = np.array([[1, 1], [2, 2]])
    out = seb.zonal_et(et, zones, names={1: "alfalfa", 2: "fallow"}, pixel_area=900.0)
    assert out.loc[1, "mean_mm"] == 1.5 and out.loc[2, "name"] == "fallow"
    assert out.loc[2, "volume_m3"] == pytest.approx(3.5 / 1000 * 2 * 900)


# ----------------------------------------------------------------------------- validation
def test_compare_and_closure():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    m = seb.compare_to_tower(obs + 0.5, obs)
    assert m["bias"] == pytest.approx(0.5) and m["r2"] == pytest.approx(1.0)
    c = seb.energy_closure(np.array([500.0]), np.array([50.0]), np.array([100.0]),
                           np.array([260.0]))
    assert c["closure_ratio"] == pytest.approx(0.8)


def test_sensitivity():
    df = seb.sensitivity(lambda a, b: a * b, {"a": 2.0, "b": 3.0}, {"a": [0.1, -0.1]})
    assert df.loc[1, "change_pct"] == pytest.approx(10.0)
