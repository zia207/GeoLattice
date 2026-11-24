"""DInSAR helpers: topographic phase removal, stack co-registration, reference points, LOS projection."""

import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import insar, sar
from tests.rsense.test_insar_topo import LON0, T0, _ann, _orbit

LAM = 0.05546576


def test_differential_interferogram_removes_dem_phase():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:40, 0:50].astype(float)
    delta_range = 0.3 * xx + 0.01 * yy          # flat-Earth + topography, metres of range difference
    defo = 0.002 * (xx - 25)                     # the signal, metres of range change
    m = (rng.normal(size=xx.shape) + 1j * rng.normal(size=xx.shape)).astype("complex64")
    # secondary seen from a slightly different position and after the ground moved
    s = m * np.exp(-1j * 4 * np.pi / LAM * (delta_range + defo))
    s[0, 0] = 0
    ifg = insar.differential_interferogram(m, s, delta_range, LAM)
    assert ifg[0, 0] == 0
    err = np.angle(ifg[1:, 1:] * np.exp(-1j * 4 * np.pi / LAM * defo[1:, 1:]))
    assert np.abs(err).max() < 1e-3
    np.testing.assert_allclose(insar.topographic_phase(delta_range, LAM), 4 * np.pi / LAM * delta_range)
    # range increase between dates = motion away from the satellite = negative LOS displacement
    los = insar.phase_to_los(np.angle(ifg[5, 30]))
    assert los == pytest.approx(-defo[5, 30], abs=1e-6)


def test_select_and_apply_reference_point():
    coh = np.full((60, 60), 0.3)
    coh[5:15, 5:15] = 0.95       # best patch, but moving
    coh[40:52, 40:52] = 0.9      # second best, stable
    vel = np.zeros_like(coh)
    vel[0:20, 0:20] = -0.2
    r0, c0 = insar.select_reference_point(coh)
    assert 5 <= r0 < 15 and 5 <= c0 < 15     # most coherent patch
    r, c = insar.select_reference_point(coh, velocity=vel, max_velocity=0.01)
    assert 40 <= r < 52 and 40 <= c < 52
    mask = np.zeros_like(coh, bool)
    mask[30:, 30:] = True
    r2, c2 = insar.select_reference_point(coh, mask=mask)
    assert r2 >= 33 and c2 >= 33
    with pytest.raises(ValueError):
        insar.select_reference_point(coh, mask=np.zeros_like(coh, bool))
    stack = np.stack([np.full((60, 60), 5.0), np.full((60, 60), -2.0)]) + np.arange(60)[None, None, :] * 0.1
    ref = insar.reference_to_point(stack, (r, c))
    assert ref[0, r, c] == pytest.approx(0) and ref[1, r, c] == pytest.approx(0)
    da = insar.reference_to_point(xr.DataArray(stack, dims=("time", "y", "x")), (r, c))
    assert isinstance(da, xr.DataArray) and float(da[1, r, c]) == pytest.approx(0)


def test_los_unit_vector_and_enu_projection():
    orbit = _orbit()
    lat, lon = np.array([0.0, 0.01]), np.array([LON0 + 2.7, LON0 + 2.7])
    v = sar.los_unit_vector(lat, lon, 0.0, orbit)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0)
    # northbound, right-looking: the radar looks east, so ground-to-satellite points west and up
    assert (v[:, 0] < 0).all() and (v[:, 2] > 0).all() and np.abs(v[:, 1]).max() < 0.01
    b = insar.perpendicular_baseline(orbit, _orbit(150.0), lat[:1], lon[:1], 0.0)
    assert np.degrees(np.arccos(v[0, 2])) == pytest.approx(b["incidence"][0], abs=0.05)
    # 10 cm of subsidence seen in the line of sight
    los = insar.enu_to_los(0.0, 0.0, -0.10, v[0])
    assert los == pytest.approx(-0.10 * v[0, 2])


def test_coregister_stack_identity_and_offset():
    ref_orb = _orbit()
    tc, rc = sar.geo2rdr([0.0], [LON0 + 2.7], [0.0], ref_orb)
    near = float(rc[0]) - 60 * 2.33 / 2

    def member(orbit, rng_seed):
        ann = _ann(orbit, near)
        ann["bursts"][0]["azimuth_time"] = tc[0] - np.timedelta64(150, "ms")
        ann["bursts"][1]["azimuth_time"] = tc[0] + np.timedelta64(10, "ms")
        rng = np.random.default_rng(rng_seed)
        slcs = [(rng.normal(size=(100, 60)) + 1j * rng.normal(size=(100, 60))).astype("complex64") for _ in range(2)]
        return {"slc": slcs, "ann": [ann, ann], "burst": [0, 1], "col0": [0, 0]}

    ref = member(ref_orb, 1)
    same = {k: v for k, v in ref.items()}
    lat = np.arange(0.012, -0.012, -0.0002)
    lon = np.arange(LON0 + 2.696, LON0 + 2.704, 0.0002)
    dem = xr.DataArray(np.zeros((lat.size, lon.size)), coords={"y": lat, "x": lon}, dims=("y", "x")).rio.write_crs(4326)
    out = sar.coregister_stack(ref, {"same": same, "higher": dict(member(_orbit(150.0), 1), slc=ref["slc"])}, dem, esd=True)
    st = out["stack"]["same"]
    core = (slice(20, 80), slice(10, 50))
    for k in range(2):
        np.testing.assert_allclose(st["slc"][k][core], ref["slc"][k][core], rtol=1e-4, atol=1e-4)
        assert np.nanmax(np.abs(st["delta_range"][k][core])) < 1e-3
    assert abs(st["esd"]["shift_lines"]) < 1e-3
    hi = out["stack"]["higher"]
    # a radially offset orbit: the range difference is positive and the DEM-flat surface equals the ellipsoid
    assert np.nanmedian(hi["delta_range"][0][core]) != 0
    np.testing.assert_allclose(hi["delta_range"][0][core], hi["flat_delta_range"][0][core], atol=1e-3)
    assert set(out["geometry"]) == {"dem_height", "dheight_ddelta_range", "slant_range", "incidence"}
    assert len(out["geometry"]["incidence"]) == 2
