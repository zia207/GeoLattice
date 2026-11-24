"""Sentinel-1 SLC/TOPS geometry and topographic-InSAR functions on a synthetic orbit."""

import numpy as np
import pytest
import xarray as xr

from geolattice.rsense import insar, sar
from geolattice.rsense.sar import sentinel1 as s1

LON0 = 30.0
T0 = np.datetime64("2025-06-14T01:50:00", "ns")


def _orbit(radius_offset=0.0, span=200, step=10):
    """Circular polar orbit (no Earth rotation) crossing the equator northbound at LON0 at T0."""
    r, v = 7_071_000.0 + radius_offset, 7_500.0
    w = v / 7_071_000.0
    t = np.arange(-span, span + step, step, dtype=float)
    e1 = np.array([np.cos(np.radians(LON0)), np.sin(np.radians(LON0)), 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    pos = r * (np.cos(w * t)[:, None] * e1 + np.sin(w * t)[:, None] * e3)
    vel = r * w * (-np.sin(w * t)[:, None] * e1 + np.cos(w * t)[:, None] * e3)
    return {"time": T0 + (t * 1e9).astype("int64").astype("timedelta64[ns]"), "position": pos, "velocity": vel}


def _ann(orbit, near_range, lines=100, samples=60, burst_gap=80):
    dt = 0.002
    bursts = []
    for k in range(2):
        fvs = np.full(lines, 2)
        fvs[:5] = fvs[-5:] = -1
        bursts.append({"azimuth_time": T0 + np.timedelta64(int(k * burst_gap * dt * 1e9), "ns"),
                       "first_valid_sample": fvs, "last_valid_sample": np.where(fvs >= 0, samples - 3, -1),
                       "first_valid_line": 5, "last_valid_line": lines - 6})
    tau0 = 2 * near_range / s1.SPEED_OF_LIGHT
    return {"radar_frequency": 5.405e9, "wavelength": s1.SPEED_OF_LIGHT / 5.405e9,
            "range_sampling_rate": 64_345_238.0, "slant_range_time": tau0, "azimuth_time_interval": dt,
            "azimuth_steering_rate": np.radians(1.59), "lines_per_burst": lines, "samples_per_burst": samples,
            "bursts": bursts, "fm_rate": [(T0, tau0, np.array([-2300.0, 4.5e5, 0.0]))],
            "doppler": [(T0, tau0, np.array([10.0, 0.0, 0.0]))], "orbit": orbit}


def test_geo2rdr_is_zero_doppler():
    orbit = _orbit()
    lat, lon, h = np.array([0.05, -0.02]), np.array([LON0 + 2.7, LON0 + 3.0]), np.array([500.0, 0.0])
    t, r = sar.geo2rdr(lat, lon, h, orbit)
    orb = s1._Orbit(orbit)
    sec = orb.seconds(t)
    P = sar.llh_to_ecef(lat, lon, h)
    d = P - orb.pos(sec)
    assert np.abs(np.einsum("ij,ij->i", d, orb.vel(sec))).max() / 7500 < 1e-3  # metres along track
    np.testing.assert_allclose(r, np.linalg.norm(d, axis=1), rtol=1e-12)
    assert t[0] > T0 > t[1]  # north of the equator is seen later on a northbound pass


def test_perpendicular_baseline_and_height_of_ambiguity():
    ref, sec = _orbit(), _orbit(radius_offset=200.0)
    lat, lon = np.array([0.0]), np.array([LON0 + 2.7])
    b = insar.perpendicular_baseline(ref, sec, lat, lon, 0.0)
    P = sar.llh_to_ecef(lat, lon, 0.0)[0]
    orb = s1._Orbit(ref)
    S = orb.pos(orb.seconds(T0))
    look = np.arccos(np.dot(-S / np.linalg.norm(S), (P - S) / np.linalg.norm(P - S)))
    assert b["bperp"][0] == pytest.approx(200 * np.sin(look), rel=0.01)  # secondary higher: positive
    lam = 0.0555
    ha = insar.height_of_ambiguity(lam, b["slant_range"], b["incidence"], b["bperp"])[0]
    assert ha == pytest.approx(lam * b["slant_range"][0] * np.sin(np.radians(b["incidence"][0])) / (2 * b["bperp"][0]))
    # phase -> height round trip: 750 m on a pair whose range difference grows 2 mm per 10 m of height
    slope = 2e-4
    phase = insar.flat_earth_phase(slope * 750.0, lam)
    h, offset = insar.phase_to_height(np.array([phase]), lam, np.array([1 / slope]))
    assert h[0] == pytest.approx(750.0, rel=1e-5) and offset == 0.0
    h, offset = insar.phase_to_height(np.array([phase]), lam, np.array([1 / slope]), reference=np.array([800.0]))
    assert h[0] == pytest.approx(800.0) and offset == pytest.approx(50.0)
    # separately unwrapped components each get their own tie
    ph = np.r_[np.full(200, phase), np.full(200, phase + 2 * np.pi)]
    comp = np.r_[np.full(200, 1), np.full(200, 2)]
    h, off = insar.phase_to_height(ph, lam, np.full(400, 1 / slope), reference=np.full(400, 800.0), components=comp)
    np.testing.assert_allclose(h, 800.0) and set(off) == {1, 2}
    assert off[2] - off[1] == pytest.approx(-lam / 2 / slope)  # one cycle = h_a


def test_deburst_cuts_overlaps_in_the_middle():
    ann = _ann(_orbit(), 850_000.0)
    a = np.ones((100, 4))
    b = 2 * np.ones((100, 4))
    img, times = sar.deburst([a, b], ann, [0, 1])
    assert img.shape == (170, 4) and (img[:85] == 1).all() and (img[85:] == 2).all()
    assert np.all(np.diff(times.astype("int64")) > 0)


def test_tops_deramp_and_doppler_centroid():
    ann = _ann(_orbit(), 850_000.0)
    lines = np.arange(100, dtype=float)[:, None]
    f = sar.tops_doppler_centroid(ann, 0, lines, np.array([[30.0]]))[:, 0]
    assert f[-1] - f[0] > 200  # the TOPS beam sweeps the Doppler forward through the burst
    phi = sar.tops_deramp_phase(ann, 0, lines, np.array([[30.0]]))[:, 0]
    # the deramping phase's frequency is minus the Doppler centroid
    inst = np.gradient(phi) / (2 * np.pi * ann["azimuth_time_interval"])
    np.testing.assert_allclose(inst[1:-1], -f[1:-1], rtol=1e-3, atol=0.5)
    # a signal at the TOPS Doppler is brought to baseband by exp(+1j phi)
    sig = np.exp(-1j * phi)
    _, measured = sar.estimate_doppler_centroid(np.tile(sig[:, None] * np.exp(1j * phi[:, None]), (1, 3)),
                                                ann["azimuth_time_interval"], block=20)
    assert np.abs(measured).max() < 1e-6


def test_esd_recovers_an_azimuth_shift():
    ann = _ann(_orbit(), 850_000.0)
    rng = np.random.default_rng(1)
    delta = 1e-4  # seconds: about 0.05 line
    cols = np.arange(60, dtype=float)[None, :]
    lines = np.arange(100, dtype=float)[:, None]
    ref, sec = [], []
    for k in range(2):
        m = (rng.normal(size=(100, 60)) + 1j * rng.normal(size=(100, 60))).astype("complex64")
        f = sar.tops_doppler_centroid(ann, k, lines, cols)
        ref.append(m)
        sec.append(m * np.exp(2j * np.pi * f * delta))  # secondary sampled delta seconds late
    out = sar.esd_azimuth_shift(ref, sec, ann, [0, 1])
    assert out["shift_seconds_mean"] == pytest.approx(delta, rel=1e-3)
    assert out["shift_lines"] == pytest.approx(delta / ann["azimuth_time_interval"], rel=1e-3)


def test_terrain_correct_places_radar_pixels_on_the_dem():
    orbit = _orbit()
    lat = np.linspace(0.02, -0.02, 40)
    lon = np.linspace(LON0 + 2.68, LON0 + 2.72, 30)
    dem = xr.DataArray(np.full((40, 30), 300.0), coords={"y": lat, "x": lon}, dims=("y", "x")).rio.write_crs(4326)
    lo, la = np.meshgrid(lon, lat)
    t, r = sar.geo2rdr(la, lo, dem.values, orbit)
    # a radar grid covering the DEM, with slant range as the layer
    times = np.arange(t.min() - np.timedelta64(50, "ms"), t.max() + np.timedelta64(50, "ms"), np.timedelta64(2, "ms"))
    near, dr = float(r.min()) - 100, 5.0
    rr = near + dr * np.arange(int((r.max() - near) / dr) + 20)
    layer = np.tile(rr, (times.size, 1))
    out = sar.terrain_correct(layer, times, near, dr, orbit, dem)
    np.testing.assert_allclose(out.values, r, atol=0.01)

    # rows stored in decreasing-time order (seen in some real RCM Stripmap products) must give
    # the identical result: np.interp needs ascending xp internally, so this is not a no-op check
    out_desc = sar.terrain_correct(layer[::-1], times[::-1], near, dr, orbit, dem)
    np.testing.assert_allclose(out_desc.values, r, atol=0.01)


def test_back_geocode_flat_earth_and_height_factor():
    ref, sec = _orbit(), _orbit(radius_offset=150.0)
    lat0, lon0 = 0.0, LON0 + 2.7
    tc, rc = sar.geo2rdr([lat0], [lon0], [0.0], ref)
    ann_r = _ann(ref, float(rc[0]) - 60 * 2.33 / 2)
    ann_r["bursts"][0]["azimuth_time"] = tc[0] - np.timedelta64(100, "ms")
    ann_s = _ann(sec, float(rc[0]) - 60 * 2.33 / 2)
    ann_s["bursts"][0]["azimuth_time"] = tc[0] - np.timedelta64(100, "ms")
    lat = np.arange(0.006, -0.006, -0.0002)
    lon = np.arange(lon0 - 0.004, lon0 + 0.004, 0.0002)
    dem = xr.DataArray(np.full((lat.size, lon.size), 0.0), coords={"y": lat, "x": lon}, dims=("y", "x")).rio.write_crs(4326)
    g = sar.back_geocode(ann_r, ann_s, 0, 0, dem, cols=(0, 60), step=(10, 10))
    core = (slice(20, 80), slice(10, 50))
    # a flat DEM at the ellipsoid: the surface range difference equals the flat-Earth one
    np.testing.assert_allclose(g["delta_range"][core], g["flat_delta_range"][core], atol=1e-3)
    # identical timing, radially offset orbit: the secondary positions stay close to the reference
    assert np.abs(g["sec_line"][core] - np.arange(100)[20:80, None]).max() < 0.1
    b = insar.perpendicular_baseline(ref, sec, [lat0], [lon0], 0.0)
    ha = insar.height_of_ambiguity(ann_r["wavelength"], b["slant_range"], b["incidence"], b["bperp"])[0]
    expected = ha / (ann_r["wavelength"] / 2)  # one fringe = lambda/2 of range difference = h_a of height
    assert np.nanmedian(np.abs(g["dheight_ddelta_range"][core])) == pytest.approx(abs(expected), rel=0.05)


def test_layover_and_shadow():
    yy, xx = np.mgrid[0:20, 0:20] * 10.0
    east_up = xx * np.tan(np.radians(50))  # rises towards the east
    look_east = sar.layover_shadow_mask(east_up, 10.0, heading_deg=-10, incidence_deg=35)  # ascending
    assert (look_east[2:-2, 2:-2] == 1).all()  # faces the east-looking radar, steeper than 35 deg
    west_up = (190 - xx) * np.tan(np.radians(60))
    assert (sar.layover_shadow_mask(west_up, 10.0, -10, 35)[2:-2, 2:-2] == 2).all()
    gentle = xx * np.tan(np.radians(10))
    assert (sar.layover_shadow_mask(gentle, 10.0, -10, 35) == 0).all()


def test_height_error_tools():
    h = np.array([[100.0, 110.0], [np.nan, 130.0]])
    ref = np.array([[101.0, 108.0], [120.0, 133.0]])
    st = insar.dem_error_stats(h, ref)
    assert st["n"] == 3 and st["bias"] == pytest.approx((-1 + 2 - 3) / 3)
    assert st["rmse"] == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
    assert insar.height_std(0.9, 16, 60.0) < insar.height_std(0.5, 16, 60.0)
    mean, sig = insar.combine_heights([np.array([10.0, np.nan]), np.array([20.0, 5.0])],
                                      [np.array([1.0, 1.0]), np.array([2.0, 1.0])])
    assert mean[0] == pytest.approx((10 / 1 + 20 / 4) / (1 + 1 / 4)) and mean[1] == 5.0
    assert sig[0] == pytest.approx(1 / np.sqrt(1.25))


def test_fill_dem_voids_delta_surface():
    yy, xx = np.mgrid[0:30, 0:30].astype(float)
    ref = 5 * xx + 2 * yy
    dem = ref + 7.0  # constant offset from the reference
    dem[10:15, 10:15] = np.nan
    da = xr.DataArray(dem, dims=("y", "x"))
    out = insar.fill_dem_voids(da, xr.DataArray(ref, dims=("y", "x")))
    np.testing.assert_allclose(out.values[10:15, 10:15], (ref + 7.0)[10:15, 10:15], atol=1e-3)


def test_read_slc_annotation_and_burst(tmp_path):
    import rasterio

    lines = 6
    fvs = " ".join(["-1"] + ["1"] * (lines - 2) + ["-1"])
    lvs = " ".join(["-1"] + ["3"] * (lines - 2) + ["-1"])
    osv = "".join(f"<orbit><time>2025-06-14T01:50:{s:02d}.000000</time><position><x>{7e6 + s}</x><y>0</y><z>{s * 7500}</z>"
                  f"</position><velocity><x>0</x><y>0</y><z>7500</z></velocity></orbit>" for s in range(0, 50, 10))
    xml = f"""<product><adsHeader><missionId>S1A</missionId><polarisation>VV</polarisation><swath>IW3</swath></adsHeader>
<generalAnnotation><productInformation><pass>Ascending</pass><radarFrequency>5.405e9</radarFrequency>
<rangeSamplingRate>6.4345e7</rangeSamplingRate><azimuthSteeringRate>1.59</azimuthSteeringRate>
<platformHeading>-12.5</platformHeading></productInformation><orbitList count="5">{osv}</orbitList>
<azimuthFmRateList><azimuthFmRate><azimuthTime>2025-06-14T01:50:20</azimuthTime><t0>0.006</t0>
<azimuthFmRatePolynomial>-2300 450000 -8e7</azimuthFmRatePolynomial></azimuthFmRate></azimuthFmRateList></generalAnnotation>
<imageAnnotation><imageInformation><slantRangeTime>0.006</slantRangeTime><azimuthTimeInterval>0.002</azimuthTimeInterval>
<rangePixelSpacing>2.33</rangePixelSpacing><azimuthPixelSpacing>13.9</azimuthPixelSpacing>
<incidenceAngleMidSwath>43.5</incidenceAngleMidSwath></imageInformation></imageAnnotation>
<dopplerCentroid><dcEstimateList><dcEstimate><azimuthTime>2025-06-14T01:50:20</azimuthTime><t0>0.006</t0>
<dataDcPolynomial>-2 3500 -3e6</dataDcPolynomial></dcEstimate></dcEstimateList></dopplerCentroid>
<swathTiming><linesPerBurst>{lines}</linesPerBurst><samplesPerBurst>5</samplesPerBurst><burstList count="1"><burst>
<azimuthTime>2025-06-14T01:50:20.000000</azimuthTime><firstValidSample>{fvs}</firstValidSample>
<lastValidSample>{lvs}</lastValidSample></burst></burstList></swathTiming>
<geolocationGrid><geolocationGridPointList>""" + "".join(
        f"<geolocationGridPoint><line>{li}</line><pixel>{p}</pixel><latitude>{36 + li / 100}</latitude><longitude>{-117 + p / 10}</longitude>"
        f"<height>100</height><incidenceAngle>40</incidenceAngle><elevationAngle>35</elevationAngle></geolocationGridPoint>"
        for li in (0, 5) for p in (0, 4)) + "</geolocationGridPointList></geolocationGrid></product>"
    path = tmp_path / "ann.xml"
    path.write_text(xml)
    ann = sar.read_slc_annotation(path)
    assert ann["mission"] == "S1A" and ann["pass"] == "Ascending" and ann["heading"] == pytest.approx(347.5)
    assert ann["azimuth_steering_rate"] == pytest.approx(np.radians(1.59))
    assert ann["bursts"][0]["first_valid_line"] == 1 and ann["bursts"][0]["last_valid_line"] == lines - 2
    assert len(ann["orbit"]["time"]) == 5 and ann["fm_rate"][0][2][0] == -2300
    # an ASF burst-extractor XML wraps one annotation per sub-swath
    body = xml.split("<product>", 1)[1].rsplit("</product>", 1)[0]
    wrapped = ("<burst><metadata>" + "".join(f"<product><content>{body.replace('IW3', sw)}</content></product>"
                                             for sw in ("IW1", "IW2", "IW3")) + "</metadata></burst>")
    (tmp_path / "burst.xml").write_text(wrapped)
    with pytest.raises(ValueError, match="swath="):
        sar.read_slc_annotation(tmp_path / "burst.xml")
    assert sar.read_slc_annotation(tmp_path / "burst.xml", swath="IW2")["swath"] == "IW2"
    tif = tmp_path / "burst.tiff"
    data = (np.arange(30).reshape(lines, 5) + 1j).astype("complex64")
    with rasterio.open(tif, "w", driver="GTiff", width=5, height=lines, count=1, dtype="complex64") as dst:
        dst.write(data, 1)
    burst = sar.read_slc_burst(tif, ann, 0)
    assert burst[0].sum() == 0 and burst[-1].sum() == 0
    assert (burst[1:-1, 1:4] == data[1:-1, 1:4]).all() and (burst[1:-1, [0, 4]] == 0).all()
    burst2 = sar.read_slc_burst(tif, ann, 0, cols=(1, 3))
    assert burst2.shape == (lines, 2) and (burst2[2] == data[2, 1:3]).all()


def test_download_orbit_picks_covering_file(monkeypatch, tmp_path):
    import urllib.request

    keys = ["AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20250704T070555_V20250613T225942_20250615T005942.EOF",
            "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20250703T070541_V20250612T225942_20250614T005942.EOF"]

    class Resp:
        def __init__(self, url):
            self.body = "".join(f"<Key>{k}</Key>" for k in keys) if "202507" in url else ""

        def read(self):
            return self.body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    got = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=60: Resp(url))
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dst: got.append(url) or open(dst, "w").close())
    path = sar.download_orbit("SENTINEL-1A", "2025-06-14T01:50:10", tmp_path)
    assert path.name == keys[0].split("/")[1] and got and got[0].endswith(keys[0])


def test_unwrap_snaphu_topo():
    pytest.importorskip("snaphu")
    yy, xx = np.mgrid[0:120, 0:160]
    true = 0.08 * xx + 0.05 * yy
    ifg = np.exp(1j * true).astype("complex64")
    unw, cc = insar.unwrap_snaphu_topo(ifg, np.full(true.shape, 0.8, "float32"), wavelength=0.0555, bperp=250.0,
                                       near_range=880e3, range_spacing=18.6, azimuth_spacing=28.0, altitude=693e3,
                                       nlooks_range=8, nlooks_azimuth=2)
    d = unw - true
    assert np.abs(d - np.median(d)).max() < 1e-3 and (cc > 0).all()


def test_resample_slc_leaves_input_untouched():
    ann = _ann(_orbit(), 850_000.0)
    rng = np.random.default_rng(3)
    z = (rng.normal(size=(100, 60)) + 1j * rng.normal(size=(100, 60))).astype("complex64")
    before = z.copy()
    lines, samples = np.mgrid[0:100, 0:60].astype(float)
    out = sar.resample_slc(z, lines, samples, ann=ann, burst=0)
    np.testing.assert_array_equal(z, before)
    # resampling at the original positions returns the original samples (deramp and reramp cancel)
    np.testing.assert_allclose(out[5:-5, 5:-5], z[5:-5, 5:-5], rtol=1e-4, atol=1e-4)


def test_detrend_heights_removes_ramp_and_elevation_term():
    yy, xx = np.mgrid[0:50, 0:80].astype(float)
    ref = 1000 + 20 * np.sin(xx / 7) * np.cos(yy / 5) + 15 * xx
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, ref.shape)
    h = ref + 30 + 0.4 * xx - 0.2 * yy + 0.03 * ref + noise  # offset, ramp, 3 % elevation term
    out, cf = insar.detrend_heights(h, ref)
    assert cf["elevation"] == pytest.approx(0.03, abs=2e-3)
    assert np.std(out - ref) == pytest.approx(1.0, rel=0.1)
