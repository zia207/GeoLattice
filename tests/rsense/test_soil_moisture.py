import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geolattice.rsense as rs
import geolattice.rsense.soil_moisture as sm


# ----------------------------------------------------------------------------- registration
def test_registered_as_rsense_submodule():
    assert rs.soil_moisture is sm
    assert "soil_moisture" in dir(rs)
    assert "soil_moisture" in rs.__all__


def test_outlined_functions_raise_not_implemented():
    with pytest.raises(NotImplementedError, match="module 11"):
        sm.change_detection(xr.DataArray(np.zeros(3), dims="time"))
    with pytest.raises(NotImplementedError, match="module 1"):
        sm.load_soilgrids("aoi.geojson")


def test_public_names_are_exported():
    import inspect

    public = {n for n, o in vars(sm).items()
              if inspect.isfunction(o) and not n.startswith("_") and o.__module__ == sm.__name__}
    assert public <= set(sm.__all__)


# ----------------------------------------------------------------------------- soil water
def test_gravimetric_and_porosity():
    assert sm.volumetric_from_gravimetric(0.2, 1.3) == pytest.approx(0.26)
    assert sm.porosity_from_bulk_density(1.33) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "sand, clay, wp, fc, sat, rho",
    [  # Saxton and Rawls (2006) Table 3, 2.5 % organic matter
        (0.88, 0.05, 0.05, 0.10, 0.46, 1.43),  # sand
        (0.40, 0.20, 0.14, 0.28, 0.46, 1.43),  # loam
        (0.25, 0.50, 0.30, 0.42, 0.50, 1.33),  # clay
    ],
)
def test_saxton_rawls_table(sand, clay, wp, fc, sat, rho):
    out = sm.saxton_rawls(sand, clay)
    assert out["wilting_point"] == pytest.approx(wp, abs=0.006)
    assert out["field_capacity"] == pytest.approx(fc, abs=0.006)
    assert out["saturation"] == pytest.approx(sat, abs=0.006)
    assert out["bulk_density"] == pytest.approx(rho, abs=0.01)
    assert out["plant_available_water"] == pytest.approx(out["field_capacity"] - out["wilting_point"])


def test_saxton_rawls_units_and_compaction():
    frac = sm.saxton_rawls(0.40, 0.20)
    pct = sm.saxton_rawls(40.0, 20.0)
    gkg = sm.saxton_rawls(400.0, 200.0)  # SoilGrids g kg-1
    assert pct["saturation"] == pytest.approx(frac["saturation"])
    assert gkg["saturation"] == pytest.approx(frac["saturation"])
    dense = sm.saxton_rawls(0.40, 0.20, density_factor=1.2)
    assert dense["saturation"] < frac["saturation"]
    assert dense["bulk_density"] == pytest.approx(1.2 * frac["bulk_density"])
    assert dense["ksat"] < frac["ksat"]
    assert sm.saxton_rawls(0.88, 0.05)["ksat"] > 50 * sm.saxton_rawls(0.25, 0.50)["ksat"]


def test_saxton_rawls_xarray_returns_dataset():
    sand = xr.DataArray([[0.4, 0.88], [0.25, 0.2]], dims=("y", "x"))
    clay = xr.DataArray([[0.2, 0.05], [0.5, 0.15]], dims=("y", "x"))
    ds = sm.saxton_rawls(sand, clay)
    assert isinstance(ds, xr.Dataset)
    assert ds.field_capacity.dims == ("y", "x")
    assert ds.ksat.attrs["units"] == "mm h-1"
    assert float(ds.saturation[0, 0]) == pytest.approx(sm.porosity_from_texture(0.4, 0.2))


def test_saturation_conversions():
    assert sm.degree_of_saturation(0.23, 0.46) == pytest.approx(0.5)
    assert sm.saturation_to_volumetric(0.5, 0.46) == pytest.approx(0.23)
    assert sm.saturation_to_volumetric(50.0, 0.46) == pytest.approx(0.23)  # ASCAT %
    assert sm.saturation_to_volumetric(np.array([0.0, 1.0]), 0.46, residual=0.05) == pytest.approx([0.05, 0.46])
    status = sm.soil_water_status(np.array([0.05, 0.2, 0.4]), 0.1, 0.3)
    assert status == pytest.approx([0.0, 0.5, 1.0])


# ----------------------------------------------------------------------------- dielectric models
def test_free_water_debye():
    eps = sm.free_water_dielectric(1e-3, 20.0)
    assert eps.real == pytest.approx(sm.EPS_WATER_STATIC, abs=0.05)
    eps_l = sm.free_water_dielectric(1.41, 20.0)
    assert eps_l.real == pytest.approx(79.6, abs=0.3)
    assert -eps_l.imag == pytest.approx(6.1, abs=0.2)
    f = np.linspace(1.0, 40.0, 391)
    peak = f[np.argmax(-np.imag(sm.free_water_dielectric(f, 20.0)))]
    assert peak == pytest.approx(17.1, abs=0.3)  # relaxation frequency near 17 GHz at 20 C
    salty = sm.free_water_dielectric(1.41, 20.0, conductivity=1.0)
    assert -salty.imag > -eps_l.imag


@pytest.mark.parametrize("model", ["dobson", "mironov"])
def test_soil_permittivity_increases_with_moisture(model):
    theta = np.linspace(0.0, 0.45, 10)
    if model == "dobson":
        eps = sm.dobson_dielectric(theta, 0.4, 0.2)
    else:
        eps = sm.mironov_dielectric(theta, 0.2)
    assert np.all(np.diff(eps.real) > 0)
    assert 2.0 < eps.real[0] < 3.0  # dry soil
    assert np.all(eps.imag <= 0)  # eps = eps' - j eps''


def test_dobson_and_mironov_agree_at_l_band():
    for theta in (0.1, 0.2, 0.3, 0.4):
        d = sm.dobson_dielectric(theta, 0.4, 0.2)
        m = sm.mironov_dielectric(theta, 0.2)
        assert m.real == pytest.approx(d.real, rel=0.25)


def test_peplinski_correction_below_1p4_ghz():
    hi = sm.dobson_dielectric(0.3, 0.4, 0.2, freq_ghz=1.41)
    lo = sm.dobson_dielectric(0.3, 0.4, 0.2, freq_ghz=1.2)
    assert lo.real == pytest.approx(1.15 * hi.real - 0.68, rel=0.02)


def test_topp_and_inverse():
    assert sm.topp(20.0) == pytest.approx(0.3454, abs=1e-4)
    theta = np.array([0.0, 0.1, 0.25, 0.4, 0.5])
    assert sm.topp(sm.topp_inverse(theta)) == pytest.approx(theta, abs=1e-9)
    # Topp and Mironov give similar permittivity in the mid range
    assert sm.topp_inverse(0.3) == pytest.approx(sm.mironov_dielectric(0.3, 0.2).real, rel=0.1)


def test_penetration_depth_by_band_and_moisture():
    depth = {f: sm.penetration_depth(sm.mironov_dielectric(0.25, 0.2, freq_ghz=f), f)
             for f in (1.41, 5.405, 10.65)}
    assert depth[1.41] > depth[5.405] > depth[10.65]
    dry = sm.penetration_depth(sm.mironov_dielectric(0.05, 0.2), 1.41)
    assert dry > depth[1.41]


# ----------------------------------------------------------------------------- reflectivity and tau-omega
def test_fresnel_nadir_and_brewster():
    eps = 4.0 + 0j
    rh = sm.fresnel_reflectivity(eps, 0.0, pol="h")
    rv = sm.fresnel_reflectivity(eps, 0.0, pol="v")
    assert rh == pytest.approx(rv)
    assert rh == pytest.approx(((1 - 2) / (1 + 2)) ** 2)
    brewster = np.rad2deg(np.arctan(2.0))
    assert sm.fresnel_reflectivity(eps, brewster, pol="v") == pytest.approx(0.0, abs=1e-12)
    assert sm.fresnel_reflectivity(eps, 40.0, pol="h") > sm.fresnel_reflectivity(eps, 40.0, pol="v")
    with pytest.raises(ValueError):
        sm.fresnel_reflectivity(eps, 40.0, pol="x")


def test_rough_reflectivity():
    assert sm.rough_reflectivity(0.3, 40.0, h=0.0) == pytest.approx(0.3)
    expected = 0.3 * np.exp(-0.2 * np.cos(np.deg2rad(40.0)) ** 2)
    assert sm.rough_reflectivity(0.3, 40.0, h=0.2) == pytest.approx(expected)
    assert sm.rough_reflectivity(0.3, 40.0, h=0.0, q=0.5, r_other=0.1) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        sm.rough_reflectivity(0.3, 40.0, q=0.2)


def test_vegetation_water_content_and_vod():
    assert sm.vod_from_vwc(2.0, b=0.11) == pytest.approx(0.22)
    assert sm.vwc_from_ndvi(0.5) == pytest.approx(1.9134 * 0.25 - 0.3215 * 0.5)
    assert sm.vwc_from_ndvi(0.05) == 0.0
    with_stems = sm.vwc_from_ndvi(0.5, stem_factor=3.0, ndvi_max=0.8)
    assert with_stems == pytest.approx(sm.vwc_from_ndvi(0.5) + 3.0 * 0.7 / 0.9)


def test_tau_omega_limits():
    eps = sm.mironov_dielectric(0.2, 0.2)
    bare = sm.tau_omega(300.0, eps, 40.0, tau=0.0, omega=0.0, h=0.0, pol="h")
    assert bare == pytest.approx(300.0 * (1 - sm.fresnel_reflectivity(eps, 40.0, pol="h")))
    opaque = sm.tau_omega(300.0, eps, 40.0, tau=50.0, omega=0.05, pol="h")
    assert opaque == pytest.approx(300.0 * 0.95, rel=1e-6)
    assert sm.tau_omega(295.0, eps, 40.0, pol="v") > sm.tau_omega(295.0, eps, 40.0, pol="h")
    wet = sm.tau_omega(295.0, sm.mironov_dielectric(0.4, 0.2), 40.0)
    assert wet < sm.tau_omega(295.0, eps, 40.0)
    q_mix = sm.tau_omega(295.0, eps, 40.0, q=0.1, pol="h")
    assert q_mix > sm.tau_omega(295.0, eps, 40.0, pol="h")  # h borrows the lower v reflectivity


@pytest.mark.parametrize("pol", ["v", "h"])
def test_invert_tau_omega_round_trip(pol):
    theta = np.array([0.05, 0.15, 0.3, 0.45])
    tau = np.array([0.0, 0.2, 0.4, 0.6])
    tb = sm.tau_omega(295.0, sm.mironov_dielectric(theta, 0.2), 40.0, tau=tau, pol=pol)
    out = sm.invert_tau_omega(tb, 295.0, 40.0, clay=0.2, tau=tau, pol=pol)
    assert out == pytest.approx(theta, abs=1e-6)


def test_invert_tau_omega_xarray_dobson_and_out_of_range():
    theta = xr.DataArray([[0.1, 0.2], [0.3, 0.35]], dims=("y", "x"))
    eps = sm.dobson_dielectric(theta, 0.4, 0.2, temperature_c=295.0 - 273.15)
    tb = sm.tau_omega(295.0, eps, 40.0, tau=0.2)
    out = sm.invert_tau_omega(tb, 295.0, 40.0, clay=0.2, sand=0.4, tau=0.2, dielectric="dobson",
                              pol="h")
    assert isinstance(out, xr.DataArray) and out.dims == ("y", "x")
    assert out.attrs["units"] == "m3 m-3"
    assert out.values == pytest.approx(theta.values, abs=1e-6)
    assert np.isnan(sm.invert_tau_omega(150.0, 295.0, 40.0, clay=0.2))  # open water
    with pytest.raises(ValueError, match="sand"):
        sm.invert_tau_omega(250.0, 295.0, 40.0, clay=0.2, dielectric="dobson")


def test_tb_sensitivity_drops_with_vod():
    table = sm.tb_sensitivity()
    assert isinstance(table, pd.DataFrame)
    assert set(table.columns) == {"tau", "theta", "tb", "dtb_dtheta"}
    sens = table.groupby("tau").dtb_dtheta.mean().abs()
    assert sens.is_monotonic_decreasing
    assert sens.iloc[0] > 5 * sens.iloc[-1]
    assert (table.dtb_dtheta < 0).all()
