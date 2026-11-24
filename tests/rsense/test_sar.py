import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import sar


def _speckled(mean_img, looks=4.4, seed=0):
    rng = np.random.default_rng(seed)
    return (mean_img * rng.gamma(looks, 1 / looks, mean_img.shape)).astype("float32")


@pytest.fixture
def edge_scene():
    img = np.full((80, 80), 0.05)
    img[:, 40:] = 0.5
    return img, _speckled(img)


@pytest.mark.parametrize("name", ["boxcar", "median", "lee", "gamma_map", "refined_lee"])
def test_filters_reduce_speckle(backend, edge_scene, name):
    truth, noisy = edge_scene
    out = np.asarray(getattr(sar, name)(noisy))
    homog = (slice(10, 70), slice(5, 30))
    assert out[homog].std() < 0.6 * noisy[homog].std()
    assert out[homog].mean() == pytest.approx(truth[homog].mean(), rel=0.15)


def test_refined_lee_preserves_edges(edge_scene):
    truth, noisy = edge_scene
    rl = sar.refined_lee(noisy)
    box = sar.boxcar(noisy, 7)
    col = slice(37, 43)
    assert np.abs(rl[:, col] - truth[:, col]).mean() < np.abs(box[:, col] - truth[:, col]).mean()


def test_nan_handling():
    a = _speckled(np.full((30, 30), 0.1))
    a[5, 5] = np.nan
    out = sar.lee(a)
    assert np.isnan(out[5, 5]) and np.isfinite(out[6, 6])


def test_multitemporal():
    rng = np.random.default_rng(1)
    stack = np.stack([_speckled(np.full((40, 40), 0.1), seed=i) for i in range(6)])
    out = sar.multitemporal(stack)
    assert out.shape == stack.shape and out[0].std() < stack[0].std() * 0.6
    _ = rng


def test_enl_estimate():
    a = _speckled(np.full((200, 200), 0.1), looks=4.4)
    assert sar.enl_estimate(a) == pytest.approx(4.4, rel=0.1)


def test_db_roundtrip():
    x = np.array([0.01, 0.1, 1.0])
    np.testing.assert_allclose(sar.from_db(sar.to_db(x)), x)
    assert sar.to_db(np.array([0.1]))[0] == pytest.approx(-10)


def test_calibrate_and_lut(tmp_path):
    xml = tmp_path / "cal.xml"
    xml.write_text("""<calibration><calibrationVectorList>
    <calibrationVector><line>0</line><pixel>0 10</pixel><sigmaNought>100 200</sigmaNought></calibrationVector>
    <calibrationVector><line>10</line><pixel>0 10</pixel><sigmaNought>100 200</sigmaNought></calibrationVector>
    </calibrationVectorList></calibration>""")
    lines, pix, vals = sar.read_calibration_lut(xml)
    lut = sar.interpolate_lut(lines, pix, vals, (11, 11))
    assert lut[0, 0] == pytest.approx(100) and lut[5, 5] == pytest.approx(150)
    dn = np.full((11, 11), 1000, "uint16")
    s0 = sar.calibrate(dn, lut)
    assert s0[0, 0] == pytest.approx(1000**2 / 100**2)


def test_omnibus_detects_change():
    t = pd.date_range("2026-01-01", periods=4, freq="12D")
    base = np.full((4, 30, 30), 0.1)
    base[2:, :, 15:] = 1.0          # 10x change in the right half from date 3
    stack = xr.DataArray(np.stack([_speckled(b, seed=i) for i, b in enumerate(base)]),
                         dims=("time", "y", "x"), coords={"time": t})
    res = sar.omnibus_test(stack, enl=4.4, alpha=0.01)
    assert float(res.change[:, 20:].mean()) > 0.9 and float(res.change[:, :10].mean()) < 0.1
    pw = sar.pairwise_change(stack, enl=4.4)
    assert float(pw.change.isel(time=1)[:, 20:].mean()) > 0.5
    assert float(pw.change.isel(time=1)[:, :10].mean()) < 0.1


def test_cfar_ship_detection():
    rng = np.random.default_rng(3)
    sea = rng.exponential(0.01, (120, 120)).astype("float32")
    for r, c in [(30, 30), (80, 90)]:
        sea[r:r + 3, c:c + 3] = 1.0
    y = 4_700_000 - 10 * (np.arange(120) + 0.5)
    x = 400_000 + 10 * (np.arange(120) + 0.5)
    da = xr.DataArray(sea, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    ships = sar.ship_detection(da, pfa=1e-7)
    assert len(ships) == 2
    os_det = sar.cfar(da, method="os", guard=2, background=6, pfa=1e-7)
    assert bool(os_det.values[31, 31])


def test_flood_map_and_change():
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 0.1)
    img[:, :60] = 0.005
    noisy = _speckled(img)
    y = 4_700_000 - 10 * (np.arange(200) + 0.5)
    x = 400_000 + 10 * (np.arange(200) + 0.5)
    da = xr.DataArray(sar.lee(noisy), dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(32618)
    for method in ("otsu", "split_otsu"):
        res = sar.flood_map(da, method=method, tile=50)
        w = res.water.values
        assert w[:, 5:55].mean() > 0.95 and w[:, 70:].mean() < 0.05
    dry = da.copy(data=sar.lee(_speckled(np.full((200, 200), 0.1), seed=5)))
    ch = sar.flood_map(da, method="change", reference=dry, tile=50)
    assert ch.water.values[:, 5:55].mean() > 0.9
    cm = sar.change_map(dry, da)
    assert float(cm.decrease[:, 5:55].mean()) > 0.9
    _ = rng


def test_rice_map():
    t = pd.date_range("2026-06-01", periods=10, freq="12D")
    db = np.full((10, 4, 4), -15.0)
    db[:, :2, :] = np.array([-22, -23, -18, -14, -12, -11, -12, -13, -14, -15])[:, None, None]
    vh = xr.DataArray(10 ** (db / 10), dims=("time", "y", "x"), coords={"time": t})
    out = sar.rice_map(vh)
    assert out.rice.values[:2].all() and not out.rice.values[2:].any()
    assert pd.Timestamp(out.transplanting_date.values[0, 0]) == t[1]


def test_oil_spill():
    rng = np.random.default_rng(2)
    img = np.full((150, 150), 0.05)
    img[60:90, 60:90] = 0.005
    da = xr.DataArray(_speckled(img), dims=("y", "x"))
    res = sar.oil_spill_detection(sar.boxcar(da, 5), min_pixels=50)
    assert res.dark_spot.values[70:80, 70:80].mean() > 0.9
    assert res.dark_spot.values[:40, :40].mean() < 0.02
    _ = rng


def test_utm_from_gcps():
    class _GCP:
        def __init__(self, x, y):
            self.x, self.y = x, y

    # Bangladesh coast (northern hemisphere, ~91 E -> UTM zone 46N)
    bd = [_GCP(90.0, 22.0), _GCP(92.0, 22.5), _GCP(91.0, 21.0)]
    assert sar.sentinel1._utm_from_gcps(bd) == "EPSG:32646"
    # Southern hemisphere: ~-70 W -> UTM zone 19S
    sh = [_GCP(-70.5, -33.0), _GCP(-69.5, -34.0)]
    assert sar.sentinel1._utm_from_gcps(sh) == "EPSG:32719"


def test_remove_border_noise():
    dn = np.zeros((5, 1000), dtype="float32")
    dn[:, 50:-50] = 1000.0
    out = sar.remove_border_noise(dn, threshold=0.0, margin=100)
    assert (out[:, :50] == 0).all() and (out[:, -50:] == 0).all()
    assert (out[:, 50:-50] == 1000.0).all()
    # interior pixels beyond `margin` of either edge are never touched, even if they are zero
    dn2 = np.full((5, 1000), 1000.0, dtype="float32")
    dn2[:, 400:600] = 0.0
    out2 = sar.remove_border_noise(dn2, threshold=0.0, margin=100)
    assert (out2[:, 400:600] == 0.0).all() and (out2[:, :100] == 1000.0).all()


def test_remove_border_noise_respects_scene_edges_for_windows():
    crop = np.full((2, 200), 10.0, dtype="float32")
    crop[:, :5] = 0.5
    crop[:, -5:] = 0.5

    interior = sar.remove_border_noise(
        crop, threshold=1.0, margin=20, col_offset=100, scene_width=1000
    )
    np.testing.assert_array_equal(interior, crop)

    left_edge = sar.remove_border_noise(
        crop, threshold=1.0, margin=20, col_offset=0, scene_width=1000
    )
    assert (left_edge[:, :5] == 0).all()
    np.testing.assert_array_equal(left_edge[:, -5:], crop[:, -5:])

    right_edge = sar.remove_border_noise(
        crop, threshold=1.0, margin=20, col_offset=800, scene_width=1000
    )
    np.testing.assert_array_equal(right_edge[:, :5], crop[:, :5])
    assert (right_edge[:, -5:] == 0).all()


def test_orbit_file(tmp_path):
    eof = tmp_path / "orbit.EOF"
    eof.write_text("""<Earth_Explorer_File><Data_Block type="xml"><List_of_OSVs count="3">
    <OSV><UTC>UTC=2020-03-29T23:55:34.000000</UTC>
    <X unit="m">1000000.0</X><Y unit="m">2000000.0</Y><Z unit="m">6600000.0</Z>
    <VX unit="m/s">100.0</VX><VY unit="m/s">200.0</VY><VZ unit="m/s">-7000.0</VZ></OSV>
    <OSV><UTC>UTC=2020-03-29T23:55:44.000000</UTC>
    <X unit="m">1001000.0</X><Y unit="m">2002000.0</Y><Z unit="m">6530000.0</Z>
    <VX unit="m/s">101.0</VX><VY unit="m/s">201.0</VY><VZ unit="m/s">-7001.0</VZ></OSV>
    <OSV><UTC>UTC=2020-03-29T23:55:54.000000</UTC>
    <X unit="m">1002200.0</X><Y unit="m">2004100.0</Y><Z unit="m">6460000.0</Z>
    <VX unit="m/s">102.0</VX><VY unit="m/s">202.0</VY><VZ unit="m/s">-7002.0</VZ></OSV>
    </List_of_OSVs></Data_Block></Earth_Explorer_File>""")
    orbit = sar.read_orbit_file(eof)
    assert orbit["position"].shape == (3, 3) and orbit["time"].size == 3
    pos, vel = sar.interpolate_orbit(orbit, np.array(["2020-03-29T23:55:44"], dtype="datetime64[ns]"))
    np.testing.assert_allclose(pos[0], [1001000.0, 2002000.0, 6530000.0], rtol=1e-6)
    np.testing.assert_allclose(vel[0], [101.0, 201.0, -7001.0], rtol=1e-6)


def test_geolocation_grid_and_heading(tmp_path):
    xml = tmp_path / "ann.xml"
    pts = []
    for i, line in enumerate((0, 100)):
        for j, pixel in enumerate((0, 50, 100)):
            pts.append(f"""<geolocationGridPoint><line>{line}</line><pixel>{pixel}</pixel>
            <latitude>{25.0 + i}</latitude><longitude>{90.0 + j}</longitude><height>10.0</height>
            <incidenceAngle>{30.0 + j}</incidenceAngle><elevationAngle>27.0</elevationAngle>
            </geolocationGridPoint>""")
    xml.write_text(f"""<product><geolocationGrid><geolocationGridPointList count="6">
    {"".join(pts)}
    </geolocationGridPointList></geolocationGrid>
    <generalAnnotation><productInformation><platformHeading>-168.0</platformHeading>
    <pass>DESCENDING</pass></productInformation></generalAnnotation></product>""")

    lines, pixels, grids = sar.read_geolocation_grid(xml)
    assert lines.tolist() == [0, 100] and pixels.tolist() == [0, 50, 100]
    np.testing.assert_allclose(grids["incidenceAngle"][0], [30.0, 31.0, 32.0])

    inc = sar.incidence_angle_grid(xml, (101, 101))
    assert inc.shape == (101, 101)
    assert inc[0, 0] == pytest.approx(30.0) and inc[0, 100] == pytest.approx(32.0, rel=0.05)

    heading, pas = sar.read_platform_heading(xml)
    assert heading == pytest.approx(192.0) and pas == "DESCENDING"


def test_local_incidence_angle_and_terrain_flatten():
    dem = np.zeros((50, 50))   # perfectly flat ground
    lia = sar.local_incidence_angle(dem, resolution=10.0, heading_deg=192.0, incidence_deg=35.0)
    np.testing.assert_allclose(lia, 35.0, atol=1e-6)   # flat ground: local == ellipsoid incidence

    beta0 = np.full((50, 50), 0.2)
    gamma0 = sar.terrain_flatten(beta0, lia)
    np.testing.assert_allclose(gamma0, 0.2 / np.cos(np.radians(35.0)))

    # a slope facing the sensor (steeper local incidence than the flat case) should brighten less
    # for the same beta0 -- i.e. local incidence angle should differ from the flat-ground case
    sloped = np.tile(np.linspace(0, 200, 50), (50, 1))
    lia_slope = sar.local_incidence_angle(sloped, resolution=10.0, heading_deg=192.0, incidence_deg=35.0)
    assert not np.allclose(lia_slope, 35.0)


def test_polsar():
    rng = np.random.default_rng(0)
    vv = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
    pure = sar.h_alpha_dual(*sar.covariance_c2(vv, 0.5 * vv, 5).values())
    assert np.nanmax(pure["entropy"]) < 1e-3                       # one scattering mechanism
    vh = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
    rnd = sar.h_alpha_dual(*sar.covariance_c2(vv, vh, 7).values())
    assert np.nanmean(rnd["entropy"]) > 0.8                        # independent channels
    f = sar.dual_pol_features(xr.DataArray([0.1]), xr.DataArray([0.02]))
    assert float(f.rvi[0]) == pytest.approx(4 * 0.02 / 0.12)
    zones = sar.h_alpha_classify(xr.DataArray([0.2, 0.95]), xr.DataArray([10.0, 60.0]))
    assert zones.values.tolist() == [9, 1]


def test_stokes_and_model_decompositions():
    rng = np.random.default_rng(1)
    vv = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
    pure = sar.covariance_c2(vv, 0.5 * vv, 5)
    vh = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
    rnd = sar.covariance_c2(vv, vh, 7)
    span_pure = pure["c11"] + pure["c22"]
    span_rnd = rnd["c11"] + rnd["c22"]

    st_pure = sar.stokes_parameters(**pure)
    st_rnd = sar.stokes_parameters(**rnd)
    assert np.nanmin(st_pure["degree_of_polarization"]) > 0.999         # one scattering mechanism
    assert np.nanmean(st_rnd["degree_of_polarization"]) < 0.3           # independent channels

    for decomp in (sar.m_delta_decomposition, sar.m_chi_decomposition, sar.dual_pol_model_decomposition):
        for c2, span in ((pure, span_pure), (rnd, span_rnd)):
            out = decomp(**c2)
            total = out["ps"] + out["pd"] + out["pv"]
            assert np.allclose(total, span, equal_nan=True)             # power conserved
            assert np.all(out["ps"][~np.isnan(out["ps"])] >= -1e-9)
            assert np.all(out["pd"][~np.isnan(out["pd"])] >= -1e-9)
            assert np.all(out["pv"][~np.isnan(out["pv"])] >= -1e-9)
        # mostly depolarized (independent channels): most power lands in pv
        assert np.nanmean(decomp(**rnd)["pv"] / span_rnd) > 0.6

    # xarray path: a DataArray input must come back as a Dataset with aligned dims
    c11, c22 = xr.DataArray([1.0, 2.0]), xr.DataArray([1.0, 0.5])
    c12_re, c12_im = xr.DataArray([0.5, 0.1]), xr.DataArray([0.0, 0.2])
    for fn in (sar.stokes_parameters, sar.m_delta_decomposition, sar.m_chi_decomposition,
               sar.dual_pol_model_decomposition):
        ds = fn(c11, c22, c12_re, c12_im)
        assert isinstance(ds, xr.Dataset)
        assert set(ds.dims) == {"dim_0"}                                 # no stray extra dimension
        assert ds.sizes["dim_0"] == 2


def test_quad_pol():
    # coherency_t3 from real-ish complex channels: Pauli power sums to the Pauli span exactly
    rng = np.random.default_rng(3)
    n = 30
    hh = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    hv = 0.4 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    vh = hv + 0.05 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))   # near-reciprocal
    vv = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    t3 = sar.coherency_t3(hh, hv, vh, vv, size=5)
    assert set(t3) == {"t11", "t22", "t33", "t12_re", "t12_im", "t13_re", "t13_im", "t23_re", "t23_im"}
    span = t3["t11"] + t3["t22"] + t3["t33"]
    assert np.all(np.isfinite(span)) and np.all(span > 0)

    ha = sar.h_alpha_quad(**t3)
    assert set(ha) == {"entropy", "alpha", "anisotropy", "lambda1", "lambda2", "lambda3"}
    assert np.all(ha["entropy"] >= -1e-9) and np.all(ha["entropy"] <= 1 + 1e-9)
    assert np.all(ha["alpha"] >= -1e-6) and np.all(ha["alpha"] <= 90 + 1e-6)
    assert np.all(ha["lambda1"] >= ha["lambda2"] - 1e-9) and np.all(ha["lambda2"] >= ha["lambda3"] - 1e-9)
    assert np.allclose(ha["lambda1"] + ha["lambda2"] + ha["lambda3"], span)          # eigenvalues sum to trace

    # known pure-mechanism T3 matrices (off-diagonal zero): exact, hand-computable entropy/alpha
    z = np.zeros(1)

    def diag_t3(a, b, c):
        return {"t11": np.array([float(a)]), "t22": np.array([float(b)]), "t33": np.array([float(c)]),
               "t12_re": z, "t12_im": z, "t13_re": z, "t13_im": z, "t23_re": z, "t23_im": z}

    surf = sar.h_alpha_quad(**diag_t3(1, 0, 0))
    assert surf["entropy"][0] == pytest.approx(0, abs=1e-9) and surf["alpha"][0] == pytest.approx(0, abs=1e-6)
    dihedral = sar.h_alpha_quad(**diag_t3(0, 1, 0))
    assert dihedral["entropy"][0] == pytest.approx(0, abs=1e-9) and dihedral["alpha"][0] == pytest.approx(90, abs=1e-6)
    # fully depolarized (three equal eigenvalues): the textbook H=1, alpha=60 deg limit point
    iso = sar.h_alpha_quad(**diag_t3(1, 1, 1))
    assert iso["entropy"][0] == pytest.approx(1, abs=1e-9)
    assert iso["alpha"][0] == pytest.approx(60, abs=1e-6)
    # a non-degenerate case worked out by hand: p = (4, 2, 1) / 7
    mix = sar.h_alpha_quad(**diag_t3(4, 2, 1))
    p = np.array([4, 2, 1]) / 7
    assert mix["entropy"][0] == pytest.approx(-np.sum(p * np.log(p) / np.log(3)), rel=1e-9)
    assert mix["alpha"][0] == pytest.approx((p[1] + p[2]) * 90, rel=1e-9)   # eigenvectors 2,3 both orthogonal to axis 1
    assert mix["anisotropy"][0] == pytest.approx(1 / 3, rel=1e-9)          # (2-1)/(2+1)

    # xarray path: Dataset output, aligned on one dimension
    t11, t22, t33 = xr.DataArray([1.0, 0.0]), xr.DataArray([0.0, 1.0]), xr.DataArray([0.0, 0.0])
    zda = xr.DataArray([0.0, 0.0])
    ds = sar.h_alpha_quad(t11, t22, t33, zda, zda, zda, zda, zda, zda)
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) == {"dim_0"} and ds.sizes["dim_0"] == 2

    # NaN input (e.g. a box-average edge pixel) must stay NaN, not silently become a valid 0
    t11n = t11.copy(); t11n[0] = np.nan
    ds_nan = sar.h_alpha_quad(t11n, t22, t33, zda, zda, zda, zda, zda, zda)
    assert np.isnan(ds_nan["entropy"].values[0]) and np.isfinite(ds_nan["entropy"].values[1])


def test_radarsat():
    rng = np.random.default_rng(2)
    n = 40
    rr = rng.gamma(4, 50, size=(n, n))
    rl = rng.gamma(4, 30, size=(n, n))
    # RRRL* correlated with sqrt(RR*RL) (partially polarized, random phase)
    mag = 0.6 * np.sqrt(rr * rl)
    phase = rng.uniform(-np.pi, np.pi, size=(n, n))
    rrrl_re, rrrl_im = mag * np.cos(phase), mag * np.sin(phase)

    st = sar.rr_rl_to_stokes(rr, rl, rrrl_re, rrrl_im)
    assert np.allclose(st["s0"], rr + rl)
    assert np.all(st["degree_of_polarization"] >= 0) and np.all(st["degree_of_polarization"] <= 1)

    rhrv = sar.rrrl_to_rhrv(rr, rl, rrrl_re, rrrl_im)
    assert set(rhrv) == {"c11", "c22", "c12_re", "c12_im"}
    assert np.allclose(rhrv["c11"] + rhrv["c22"], rr + rl)               # power conserved by the basis change

    # the converted C2 is basis-general: polsar's dual-pol H/alpha applies to it directly
    ha = sar.h_alpha_dual(**rhrv)
    assert np.all(ha["entropy"] >= 0) and np.all(ha["entropy"] <= 1)
    assert np.all(ha["alpha"] >= 0) and np.all(ha["alpha"] <= 90)

    mc = sar.compact_pol_mchi(st["s0"], st["s1"], st["s2"], st["s3"])
    assert np.allclose(mc["ps"] + mc["pd"] + mc["pv"], st["s0"])         # power conserved
    assert np.all(mc["ps"] >= -1e-9) and np.all(mc["pd"] >= -1e-9) and np.all(mc["pv"] >= -1e-9)

    # a single scattering mechanism (RRRL* fully correlated with RR/RL) is fully polarized
    full_re, full_im = np.sqrt(rr * rl) * np.cos(phase), np.sqrt(rr * rl) * np.sin(phase)
    st_full = sar.rr_rl_to_stokes(rr, rl, full_re, full_im)
    assert np.allclose(st_full["degree_of_polarization"], 1, atol=1e-6)

    # split_rrrl: band-dimensioned DataArray and plain (2, y, x) array give the same split
    stacked = np.stack([rrrl_re, rrrl_im])
    re_a, im_a = sar.split_rrrl(xr.DataArray(stacked, dims=("band", "y", "x")))
    re_b, im_b = sar.split_rrrl(stacked)
    assert np.allclose(re_a.values, rrrl_re) and np.allclose(im_a.values, rrrl_im)
    assert np.allclose(re_b, rrrl_re) and np.allclose(im_b, rrrl_im)

    assert sar.gamma0_to_sigma0(2.0, 0.5) == pytest.approx(1.0)

    # xarray path: Dataset outputs stay aligned on one dimension (no stray dimension coordinate)
    rr_da, rl_da = xr.DataArray([100.0, 50.0]), xr.DataArray([80.0, 40.0])
    re_da, im_da = xr.DataArray([10.0, 5.0]), xr.DataArray([5.0, -2.0])
    for fn in (sar.rr_rl_to_stokes, sar.rrrl_to_rhrv):
        ds = fn(rr_da, rl_da, re_da, im_da)
        assert isinstance(ds, xr.Dataset)
        assert set(ds.dims) == {"dim_0"}
    st_da = sar.rr_rl_to_stokes(rr_da, rl_da, re_da, im_da)
    mc_da = sar.compact_pol_mchi(st_da["s0"], st_da["s1"], st_da["s2"], st_da["s3"])
    assert isinstance(mc_da, xr.Dataset)
    assert set(mc_da.dims) == {"dim_0"}


def test_simulate_compact_pol():
    # canonical single-bounce (surface) Pauli scatterer: Shh = Svv, no cross-pol
    n = 5
    one = np.ones((n, n), dtype="complex128")
    zero = np.zeros((n, n), dtype="complex128")
    cp = sar.simulate_compact_pol(one, zero, zero, one, size=1)
    assert set(cp) == {"c11", "c22", "c12_re", "c12_im"}
    st = sar.rhrv_to_stokes(**cp)
    assert np.allclose(st["degree_of_polarization"], 1, atol=1e-9)      # deterministic scatterer
    mc = sar.compact_pol_mchi(st["s0"], st["s1"], st["s2"], st["s3"])
    assert np.allclose(mc["ps"], st["s0"], atol=1e-9)                   # all power -> surface
    assert np.allclose(mc["pd"], 0, atol=1e-9) and np.allclose(mc["pv"], 0, atol=1e-9)

    # canonical double-bounce Pauli scatterer: Shh = -Svv, no cross-pol
    cp2 = sar.simulate_compact_pol(one, zero, zero, -one, size=1)
    st2 = sar.rhrv_to_stokes(**cp2)
    mc2 = sar.compact_pol_mchi(st2["s0"], st2["s1"], st2["s2"], st2["s3"])
    assert np.allclose(mc2["pd"], st2["s0"], atol=1e-9)                 # all power -> double-bounce
    assert np.allclose(mc2["ps"], 0, atol=1e-9) and np.allclose(mc2["pv"], 0, atol=1e-9)

    # random quad-pol data: power-conservation and value-range checks still hold downstream
    rng = np.random.default_rng(3)
    shape = (30, 30)

    def crand(scale):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * scale

    s_hh, s_vv = crand(1.0), crand(1.0)
    s_x = crand(0.3)
    cp3 = sar.simulate_compact_pol(s_hh, s_x, s_x, s_vv, size=7)       # reciprocal: Shv = Svh
    st3 = sar.rhrv_to_stokes(**cp3)
    mc3 = sar.compact_pol_mchi(st3["s0"], st3["s1"], st3["s2"], st3["s3"])
    assert np.allclose(mc3["ps"] + mc3["pd"] + mc3["pv"], st3["s0"], atol=1e-9)
    assert np.all(mc3["ps"] >= -1e-9) and np.all(mc3["pd"] >= -1e-9) and np.all(mc3["pv"] >= -1e-9)

    ha3 = sar.h_alpha_dual(**cp3)                                      # basis-independent on the simulated C2
    assert np.all(ha3["entropy"] >= 0) and np.all(ha3["entropy"] <= 1)
    assert np.all(ha3["alpha"] >= 0) and np.all(ha3["alpha"] <= 90)

    # xarray path
    s_hh_da = xr.DataArray(s_hh, dims=("y", "x"))
    s_x_da = xr.DataArray(s_x, dims=("y", "x"))
    s_vv_da = xr.DataArray(s_vv, dims=("y", "x"))
    cp_da = sar.simulate_compact_pol(s_hh_da, s_x_da, s_x_da, s_vv_da, size=7)
    assert isinstance(cp_da, xr.Dataset)
    st_da2 = sar.rhrv_to_stokes(cp_da["c11"], cp_da["c22"], cp_da["c12_re"], cp_da["c12_im"])
    assert isinstance(st_da2, xr.Dataset)


def _rcm_lut_xml(path, pixel_first, step, gains, offset=0.0):
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<lut xmlns="rcmGsProductSchema">\n'
        f"\t<pixelFirstLutValue>{pixel_first}</pixelFirstLutValue>\n"
        f"\t<stepSize>{step}</stepSize>\n"
        f"\t<numberOfValues>{len(gains)}</numberOfValues>\n"
        f"\t<offset>{offset}</offset>\n"
        f"\t<gains>{' '.join(str(g) for g in gains)}</gains>\n"
        "</lut>\n"
    )


def _rcm_noise_xml(path, segments):
    """``segments``: list of (beam, cal_type, pixel_first, step, values_db)."""
    body = "".join(
        "\t<perBeamReferenceNoiseLevel>\n"
        f"\t\t<beam>{beam}</beam>\n"
        f"\t\t<sarCalibrationType>{cal_type}</sarCalibrationType>\n"
        f"\t\t<pixelFirstNoiseValue>{pixel_first}</pixelFirstNoiseValue>\n"
        f"\t\t<stepSize>{step}</stepSize>\n"
        f"\t\t<numberOfValues>{len(values)}</numberOfValues>\n"
        f'\t\t<noiseLevelValues units="dB">{" ".join(str(v) for v in values)}</noiseLevelValues>\n'
        "\t</perBeamReferenceNoiseLevel>\n"
        for beam, cal_type, pixel_first, step, values in segments
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<noiseLevels xmlns="rcmGsProductSchema">\n' + body + "</noiseLevels>\n"
    )


def test_radarsat_grd_calibration(tmp_path):
    # A synthetic lutSigma XML (real RCM gains are ~1e7; a flat 1e7 isolates the interpolation
    # and calibration formula from any range-dependent gain falloff).
    lut_xml = tmp_path / "lutSigma_VV.xml"
    _rcm_lut_xml(lut_xml, pixel_first=0, step=10, gains=[1.0e7] * 11)
    lut = sar.radarsat.read_rcm_lut(lut_xml)
    assert lut == {"pixel_first": 0, "step": 10, "n": 11, "offset": 0.0,
                   "gains": pytest.approx([1.0e7] * 11)}
    width = 120
    gain = sar.radarsat.interpolate_rcm_lut(lut, width)
    assert gain.shape == (width,)
    assert np.allclose(gain, 1.0e7)                                   # flat LUT -> flat gain everywhere

    # Two overlapping ScanSAR-style beam segments (like SC-7/SC-8 in a real noiseLevels file):
    # the later (higher pixel_first) beam wins in the overlap, by construction.
    noise_xml = tmp_path / "noiseLevels_VV.xml"
    _rcm_noise_xml(noise_xml, [
        ("SC-7", "Sigma Nought", 0, 10, [-35.0] * 7),
        ("SC-7", "Beta Nought", 0, 10, [-10.0] * 7),                  # wrong cal_type: must be filtered out
        ("SC-8", "Sigma Nought", 50, 10, [-30.0] * 7),
    ])
    segs = sar.radarsat.read_rcm_noise_levels(noise_xml, cal_type="Sigma Nought")
    assert [s["beam"] for s in segs] == ["SC-7", "SC-8"]
    noise_db = sar.radarsat.interpolate_rcm_noise(segs, width)
    assert noise_db.shape == (width,)
    assert np.allclose(noise_db[:50], -35.0)                          # SC-7 only, before the overlap
    assert np.allclose(noise_db[60:120], -30.0)                       # SC-8, past SC-7's own range

    # DN^2 / gain, in the plausible sigma0 range for a real product (gain ~1e7, DN ~1e3 -> ~0.1)
    dn = np.full((5, width), 1000, dtype="uint16")
    sigma0 = sar.radarsat.calibrate_rcm(dn, gain, offset=lut["offset"])
    assert sigma0.dtype == np.float32
    assert np.allclose(sigma0, 1000**2 / 1.0e7)

    # subtracting a noise floor well above the signal must hit the clip, not go negative
    noisy = sar.radarsat.calibrate_rcm(dn, gain, offset=lut["offset"], noise_db=np.zeros(width))
    assert np.all(noisy == pytest.approx(1e-6))

    # DN == 0 is flagged invalid (no-data), not calibrated to a bogus near-zero value
    dn0 = dn.copy()
    dn0[0, :] = 0
    out0 = sar.radarsat.calibrate_rcm(dn0, gain, offset=lut["offset"])
    assert np.all(np.isnan(out0[0]))
    assert np.all(np.isfinite(out0[1:]))


def _rcm_stripmap_product_xml(path, *, t_first, t_last, n_lines=50, n_samples=20, prf=2742.73176,
                              near_range=728324.67, pixel_spacing=6.971918, line_spacing=2.5272,
                              state_vectors=3):
    """A minimal RCM Level-1 Stripmap SLC ``product.xml`` (the real tags ``read_rcm_annotation``
    reads, with everything else the schema has omitted)."""
    svs = "".join(
        f"<stateVector><timeStamp>{t_first}</timeStamp>"
        f"<xPosition units=\"m\">{1.0e6 + 1000.0 * i}</xPosition>"
        f"<yPosition units=\"m\">{-4.9e6 + 2000.0 * i}</yPosition>"
        f"<zPosition units=\"m\">{4.8e6 + 1500.0 * i}</zPosition>"
        f"<xVelocity units=\"m/s\">-2780.0</xVelocity>"
        f"<yVelocity units=\"m/s\">4660.0</yVelocity>"
        f"<zVelocity units=\"m/s\">5380.0</zVelocity></stateVector>"
        for i in range(state_vectors)
    )
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<product xmlns="rcmGsProductSchema">
  <sourceAttributes>
    <satellite>RCM-1</satellite>
    <radarParameters>
      <radarCenterFrequency units="Hz">5404999999.449389</radarCenterFrequency>
      <pulseRepetitionFrequency units="Hz">{prf}</pulseRepetitionFrequency>
    </radarParameters>
    <orbitAndAttitude><orbitInformation>{svs}</orbitInformation></orbitAndAttitude>
  </sourceAttributes>
  <imageGenerationParameters>
    <sarProcessingInformation>
      <numLines>{n_lines}</numLines>
      <samplesPerLine>{n_samples}</samplesPerLine>
      <zeroDopplerTimeFirstLine>{t_first}</zeroDopplerTimeFirstLine>
      <zeroDopplerTimeLastLine>{t_last}</zeroDopplerTimeLastLine>
    </sarProcessingInformation>
  </imageGenerationParameters>
  <imageAttributes>
    <productType>SLC</productType>
    <rasterAttributes>
      <sampledPixelSpacing units="m">{pixel_spacing}</sampledPixelSpacing>
      <sampledLineSpacing units="m">{line_spacing}</sampledLineSpacing>
    </rasterAttributes>
  </imageAttributes>
  <sceneAttributes>
    <imageAttributes>
      <slantRangeNearEdge units="m">{near_range}</slantRangeNearEdge>
      <incAngNearRng units="deg">36.9</incAngNearRng>
      <incAngFarRng units="deg">39.0</incAngFarRng>
    </imageAttributes>
  </sceneAttributes>
  <productAttributes>
    <passDirection>Ascending</passDirection>
    <beamModeMnemonic>16M11</beamModeMnemonic>
    <polarizations>VV</polarizations>
  </productAttributes>
</product>
""")


def test_read_rcm_annotation(tmp_path):
    xml = tmp_path / "product.xml"
    n_lines, prf = 50, 2742.73176
    # t_first > t_last (rows stored in decreasing-time order), spaced by exactly 1/PRF per line,
    # the way a real product's own zeroDopplerTimeFirstLine/LastLine are internally consistent
    t0 = np.datetime64("2023-09-13T22:44:09.547599", "ns")
    t_last_dt = t0 - np.timedelta64(round((n_lines - 1) / prf * 1e9), "ns")
    t_first, t_last = f"{t0}Z", f"{t_last_dt}Z"
    _rcm_stripmap_product_xml(xml, t_first=t_first, t_last=t_last, n_lines=n_lines, n_samples=20, prf=prf)

    ann = sar.radarsat.read_rcm_annotation(xml)
    assert ann["mission"] == "RCM-1" and ann["beam"] == "16M11" and ann["pass"] == "Ascending"
    assert ann["polarisation"] == "VV"
    assert ann["lines_per_burst"] == 50 and ann["samples_per_burst"] == 20
    assert ann["wavelength"] == pytest.approx(0.055465765, rel=1e-6)   # c / 5.405 GHz

    # a single "burst" spanning the whole image, in the shape sentinel1.back_geocode/resample_slc expect
    assert len(ann["bursts"]) == 1
    b = ann["bursts"][0]
    assert b["first_valid_line"] == 0 and b["last_valid_line"] == 49
    assert b["azimuth_time"] == np.datetime64(t_first.rstrip("Z"), "ns")
    assert b["first_valid_sample"].shape == (50,) and np.all(b["first_valid_sample"] == 0)
    assert np.all(b["last_valid_sample"] == 19)

    # row time decreases here (t_first > t_last): azimuth_time_interval must come out negative,
    # with the same magnitude as 1/PRF
    assert ann["azimuth_time_interval"] < 0
    assert abs(ann["azimuth_time_interval"]) == pytest.approx(1 / 2742.73176, rel=1e-6)

    # slant_range_time / range_sampling_rate round-trip back to the stated near range and pixel spacing
    assert ann["slant_range_time"] * sar.SPEED_OF_LIGHT / 2 == pytest.approx(728324.67, rel=1e-6)
    assert sar.SPEED_OF_LIGHT / (2 * ann["range_sampling_rate"]) == pytest.approx(6.971918, rel=1e-6)

    assert ann["orbit"]["time"].shape == (3,)
    assert ann["orbit"]["position"].shape == (3, 3)
    assert ann["orbit"]["velocity"].shape == (3, 3)

    # an ascending-time product (t_first < t_last) must give a positive interval the same way
    xml2 = tmp_path / "product_ascending.xml"
    _rcm_stripmap_product_xml(xml2, t_first="2023-09-13T22:44:06.0Z", t_last="2023-09-13T22:44:09.0Z", n_lines=50)
    ann2 = sar.radarsat.read_rcm_annotation(xml2)
    assert ann2["azimuth_time_interval"] > 0


# ----------------------------------------------------------------------------- multitemporal stacks
def _gcp_scene(tmp_path, name, shape=(120, 150), lon0=92.0, lat0=21.3, step=0.0002, cal=None):
    """A synthetic GRD-like GeoTIFF with GCPs (north-up, `step` degrees per pixel) and a bright square."""
    import rasterio
    from rasterio.control import GroundControlPoint

    h, w = shape
    rng = np.random.default_rng(len(name))
    dn = rng.gamma(4.4, 300 / 4.4, size=shape).astype("float64")
    yy, xx = np.mgrid[0:h, 0:w]
    lon, lat = lon0 + xx * step, lat0 - yy * step
    dn[(lon > 92.010) & (lon < 92.016) & (lat < 21.294) & (lat > 21.288)] *= 4
    dn = np.clip(dn, 1, 65535).astype("uint16")
    gcps = [GroundControlPoint(row=r, col=c, x=lon0 + c * step, y=lat0 - r * step, z=0.0)
            for r in (0, h // 2, h) for c in (0, w // 2, w)]
    path = tmp_path / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint16",
                       gcps=gcps, crs="EPSG:4326") as dst:
        dst.write(dn, 1)
    return path, dn


def _cal_xml(tmp_path, width, a=(500.0, 700.0)):
    xml = tmp_path / "cal.xml"
    xml.write_text(f"""<calibration><calibrationVectorList>
    <calibrationVector><line>0</line><pixel>0 {width}</pixel><sigmaNought>{a[0]} {a[1]}</sigmaNought></calibrationVector>
    <calibrationVector><line>1000</line><pixel>0 {width}</pixel><sigmaNought>{a[0]} {a[1]}</sigmaNought></calibrationVector>
    </calibrationVectorList></calibration>""")
    return xml


def test_read_grd_subset_matches_full_scene_calibration(tmp_path):
    path, dn = _gcp_scene(tmp_path, "scene")
    xml = _cal_xml(tmp_path, dn.shape[1])
    bounds = (92.008, 21.284, 92.020, 21.296)
    sub = sar.read_grd_subset(path, xml, bounds=bounds, margin=2)
    r0, c0, h, w = sub["window"]
    assert 0 < h < dn.shape[0] and 0 < w < dn.shape[1]
    full = sar.calibrate(dn, sar.interpolate_lut(*sar.read_calibration_lut(xml), dn.shape))
    np.testing.assert_allclose(sub["data"], full[r0:r0 + h, c0:c0 + w], rtol=1e-5)  # LUT sampled at true offset
    assert min(g.row for g in sub["gcps"]) == -r0 and min(g.col for g in sub["gcps"]) == -c0


def test_create_stack_coregisters_offset_windows(tmp_path):
    path, dn = _gcp_scene(tmp_path, "scene")
    xml = _cal_xml(tmp_path, dn.shape[1])
    bounds = (92.008, 21.284, 92.020, 21.296)
    a = sar.read_grd_subset(path, xml, bounds=bounds, margin=3)
    b = sar.read_grd_subset(path, xml, bounds=bounds, margin=9)       # different window, same ground
    assert a["window"] != b["window"]
    import datetime as dt

    utc = dt.timezone.utc
    stack = sar.create_stack([a, b], [dt.datetime(2017, 1, 3, 23, 48, tzinfo=utc), "2017-01-27"],
                             bounds=bounds, crs="EPSG:32646", resolution=20)
    assert stack.time.dtype.kind == "M" and str(stack.time.values[0])[:16] == "2017-01-03T23:48"  # tz-naive UTC
    assert stack.dims == ("time", "y", "x") and stack.sizes["time"] == 2 and str(stack.rio.crs) == "EPSG:32646"
    assert stack.attrs["units"] == "linear"
    both = np.isfinite(stack[0]) & np.isfinite(stack[1])
    assert both.mean() > 0.9
    np.testing.assert_allclose(stack[0].values[both], stack[1].values[both], rtol=1e-4)
    transform, w, h = sar.common_grid(bounds, "EPSG:32646", 20)
    assert (stack.sizes["x"], stack.sizes["y"]) == (w, h) and transform.c % 20 == 0
    with pytest.raises(ValueError):
        sar.create_stack([a], ["2017-01-03", "2017-01-27"], bounds=bounds)


def test_multitemporal_speckle_filter_xarray_and_dataset():
    import xarray as xr

    stack = np.stack([_speckled(np.full((40, 40), 0.1), seed=i) for i in range(8)])
    da = xr.DataArray(stack, dims=("time", "y", "x"),
                      coords={"time": np.arange(8), "y": np.arange(40), "x": np.arange(40)}, attrs={"units": "linear"})
    out = sar.multitemporal_speckle_filter(da, size=5)
    assert out.dims == da.dims and (out.time == da.time).all() and "Quegan" in out.attrs["speckle_filter"]
    assert float(out[0].std()) < float(da[0].std()) * 0.6
    assert abs(float(out.mean()) - float(da.mean())) / float(da.mean()) < 0.05   # radiometry preserved
    ds = sar.multitemporal_speckle_filter(xr.Dataset({"vv": da, "vh": da * 0.2}), size=5)
    assert set(ds.data_vars) == {"vv", "vh"} and float(ds.vh[0].std()) < float(da[0].std()) * 0.2
    np.testing.assert_allclose(sar.multitemporal_speckle_filter(stack, size=5), sar.multitemporal(stack, size=5))


def test_lut_readers_accept_urls(tmp_path):
    from geolattice.web import static_server

    _cal_xml(tmp_path, 100)
    with static_server(tmp_path) as srv:
        lines, pixels, values = sar.read_calibration_lut(f"{srv.url}/cal.xml")
    assert lines.tolist() == [0, 1000] and values[0].tolist() == [500.0, 700.0]


def test_estimate_and_apply_offset():
    import xarray as xr

    rng = np.random.default_rng(3)
    ref = np.zeros((120, 120))
    ref[30:70, 40:90] = 1.0
    ref += rng.normal(0, 0.05, ref.shape)
    moving = np.roll(np.roll(ref, -4, axis=0), 11, axis=1) + rng.normal(0, 0.05, ref.shape)
    dy, dx, corr = sar.estimate_offset(moving, ref, max_shift=15, smooth=3)
    assert (dy, dx) == (4, -11) and corr > 0.9
    da = xr.DataArray(np.stack([moving, moving]), dims=("time", "y", "x"))
    fixed = sar.apply_offset(da, dy, dx)
    assert fixed.attrs["offset_pixels"] == (4, -11) and np.isnan(fixed.values[0, :4]).all()
    inner = (slice(20, 100), slice(20, 100))
    assert np.corrcoef(fixed.values[0][inner].ravel(), ref[inner].ravel())[0, 1] > 0.9
    with pytest.raises(ValueError):
        sar.estimate_offset(moving[:50], ref)


def test_sar_indices():
    # hand-computable scalar cases
    vv, vh = np.array([1.0]), np.array([1.0])
    assert sar.cross_pol_ratio(vv, vh) == pytest.approx(1.0)
    assert sar.radar_vegetation_index(vv, vh) == pytest.approx(2.0)
    assert sar.rvi4s1(vv, vh) == pytest.approx(np.sqrt(2))
    assert sar.radar_forest_degradation_index(vv, vh) == pytest.approx(0.0)
    assert sar.sentinel_normalized_index(vv, vh) == pytest.approx(0.0)
    assert sar.polarimetric_rvi(vv, vh) == pytest.approx(0.5)
    assert sar.dpsvi(vv, vh) == pytest.approx(2.0)

    vv2, vh2 = np.array([3.0]), np.array([1.0])
    assert sar.cross_pol_ratio(vv2, vh2) == pytest.approx(1 / 3)
    assert sar.radar_vegetation_index(vv2, vh2) == pytest.approx(1.0)
    assert sar.rvi4s1(vv2, vh2) == pytest.approx(np.sqrt(3 / 4))
    assert sar.radar_forest_degradation_index(vv2, vh2) == pytest.approx(0.5)
    assert sar.sentinel_normalized_index(vv2, vh2) == pytest.approx(-0.5)
    assert sar.polarimetric_rvi(vv2, vh2) == pytest.approx(0.25)
    assert sar.dpsvi(vv2, vh2) == pytest.approx(4 / 3)

    # algebraic identities, on random positive data
    rng = np.random.default_rng(7)
    vv_r, vh_r = rng.gamma(4, 1, 500), rng.gamma(4, 1, 500)
    assert np.allclose(sar.sentinel_normalized_index(vv_r, vh_r),
                        -sar.radar_forest_degradation_index(vv_r, vh_r))
    assert np.allclose(sar.polarimetric_rvi(vv_r, vh_r), vh_r**2 / (vh_r + vv_r))

    # zero/negative denominator -> NaN, not inf or a raised error
    zero = np.array([0.0])
    assert np.isnan(sar.cross_pol_ratio(zero, zero))
    assert np.isnan(sar.radar_vegetation_index(zero, zero))
    assert np.isnan(sar.dpsvi(zero, np.array([1.0])))

    # temporal helpers
    x_t, x_t1 = np.array([4.0, 10.0]), np.array([2.0, 5.0])
    assert np.allclose(sar.temporal_change(x_t, x_t1), [2.0, 5.0])
    assert np.allclose(sar.temporal_ratio(x_t, x_t1), [2.0, 2.0])
    assert np.isnan(sar.temporal_ratio(np.array([1.0]), zero))

    # dual_pol_indices: complete key set, matching the individual function calls
    out = sar.dual_pol_indices(vv_r, vh_r)
    expected_keys = {"cross_pol_ratio", "rvi", "rvi4s1", "forest_degradation_index",
                      "normalized_index", "polarimetric_rvi", "dpsvi", "vv_minus_vh",
                      "vv_plus_vh", "vh_vv_ratio", "vv_vh_ratio"}
    assert set(out) == expected_keys
    assert np.allclose(out["cross_pol_ratio"], sar.cross_pol_ratio(vv_r, vh_r))
    assert np.allclose(out["rvi"], sar.radar_vegetation_index(vv_r, vh_r))
    assert np.allclose(out["vv_minus_vh"], vv_r - vh_r)
    assert np.allclose(out["vv_plus_vh"], vv_r + vh_r)

    # xarray path stays a DataArray
    vv_da = xr.DataArray(vv_r, dims=("x",))
    vh_da = xr.DataArray(vh_r, dims=("x",))
    cr_da = sar.cross_pol_ratio(vv_da, vh_da)
    assert isinstance(cr_da, xr.DataArray)
    assert np.allclose(cr_da.values, sar.cross_pol_ratio(vv_r, vh_r))


def _grid(arr, res=30.0, crs=32646):
    rows, cols = arr.shape
    y = 2_758_000 - res * (np.arange(rows) + 0.5)
    x = 315_000 + res * (np.arange(cols) + 0.5)
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(crs)


def test_hand_is_height_above_the_channel():
    rows, cols = 40, 31
    r, c = np.indices((rows, cols))
    # Valley floor at column 15, draining toward larger row index (south on a north-up grid).
    z = (np.abs(c - 15) * 4.0 + (rows - 1 - r) * 0.2).astype("float32")
    h = sar.hand(_grid(z), channel_cells=20)
    assert h.name == "hand"
    assert np.nanmedian(h.values[:, 15]) < 1.0
    assert np.nanmedian(h.values[:, 16]) == pytest.approx(4.0, abs=1.0)
    assert np.nanmedian(h.values[:, 5]) > 30
    z[0, 0] = np.nan
    h_nan = sar.hand(_grid(z), channel_cells=20)
    assert np.isnan(h_nan.values[0, 0])


def test_flood_map_db_ratio_flags_a_drop():
    """UN-SPIDER test: after/before on dB values, threshold 1.25, flags darker after pixels."""
    land = 10 ** (-11.0 / 10)
    water = 10 ** (-22.0 / 10)
    before = np.full((40, 60), land, "float32")
    after = before.copy()
    after[:, :20] = water
    dry = xr.DataArray(before, dims=("y", "x"))
    wet = xr.DataArray(after, dims=("y", "x"))
    res = sar.flood_map(wet, method="ratio", reference=dry, min_pixels=1)
    assert res.attrs["method"] == "ratio"
    assert res.attrs["threshold_db"] == pytest.approx(1.25)
    assert res.water.values[:, :20].all()
    assert not res.water.values[:, 25:].any()
    brighter = before.copy()
    brighter[:, :20] = 10 ** (-5.0 / 10)
    up = sar.flood_map(xr.DataArray(brighter, dims=("y", "x")), method="ratio", reference=dry,
                       min_pixels=1)
    assert not up.water.values.any()
    # The same threshold on arrays that are already in dB.
    res_db = sar.flood_map(sar.to_db(wet), method="ratio", reference=sar.to_db(dry),
                           db_input=True, min_pixels=1)
    assert res_db.water.values[:, :20].mean() == 1


def test_flood_map_slope_shadow_and_polygons():
    img = np.full((80, 80), 0.1, "float32")
    img[20:60, 10:40] = 0.004
    da = _grid(img, res=10.0, crs=32618)
    slope = xr.zeros_like(da)
    slope.values[20:60, 10:20] = 20.0
    res = sar.flood_map(da, method="fixed", fixed_db=-15, slope=slope, slope_max=5, min_pixels=1)
    assert not res.water.values[30, 15]
    assert res.water.values[30, 30]
    shadow = np.zeros((80, 80), "uint8")
    shadow[20:60, 30:40] = 2
    res2 = sar.flood_map(da, method="fixed", fixed_db=-15, shadow=shadow, min_pixels=1)
    assert not res2.water.values[30, 35]
    assert res2.water.values[30, 15]
    polys = sar.mask_to_polygons(res2.water, min_pixels=1)
    assert len(polys) == 1
    assert int(polys.pixels.iloc[0]) == 40 * 20
    assert polys.area_m2.iloc[0] == pytest.approx(40 * 20 * 100, rel=0.02)
    assert polys.crs == da.rio.crs
    empty = sar.mask_to_polygons(da.copy(data=np.zeros((80, 80), bool)))
    assert empty.empty
    assert {"pixels", "area_m2", "area_km2"} <= set(empty.columns)


# ----------------------------------------------------------------------------- RCM GRD product reader
_RCM_BOX = (-68.0, 48.0, -67.0, 49.0)      # lon/lat footprint of the synthetic scene


def _rcm_grd_product(root, *, lines=60, samples=80, pols=("HH", "HV"), tif_gcps=True,
                     gain=1.0e7, noise_db=-30.0, inc_near=20.0, inc_far=40.0, dn=1000):
    """A minimal unzipped RCM Level-1 GRD product folder: imagery/*_<pol>.tif plus
    metadata/product.xml (tie-point grid) and metadata/calibration/*.xml, on a lon/lat
    footprint ``_RCM_BOX`` with row 0 at the north edge and column 0 at the west (near range)."""
    import rasterio
    from rasterio.control import GroundControlPoint

    west, south, east, north = _RCM_BOX
    (root / "imagery").mkdir(parents=True)
    cal = root / "metadata" / "calibration"
    cal.mkdir(parents=True)
    tie = []
    for r in np.linspace(0, lines - 1, 5):
        for c in np.linspace(0, samples - 1, 5):
            lon = west + (east - west) * c / (samples - 1)
            lat = north - (north - south) * r / (lines - 1)
            tie.append((r, c, lon, lat))
    gcps = [GroundControlPoint(row=r, col=c, x=lon, y=lat, z=0.0) for r, c, lon, lat in tie]
    for pol in pols:
        arr = np.full((lines, samples), dn, dtype="uint16")
        with rasterio.open(root / "imagery" / f"RCM1_TEST_{pol}.tif", "w", driver="GTiff", height=lines,
                           width=samples, count=1, dtype="uint16") as dst:
            dst.write(arr, 1)
            if tif_gcps:
                dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))
        _rcm_lut_xml(cal / f"lutSigma_{pol}.xml", pixel_first=0, step=10,
                     gains=[gain] * (samples // 10 + 1))
        _rcm_noise_xml(cal / f"noiseLevels_{pol}.xml",
                       [("SC-1", "Sigma Nought", 0, 10, [noise_db] * (samples // 10 + 1))])
    n_inc = samples // 10 + 1
    angles = np.linspace(inc_near, inc_near + (inc_far - inc_near) * 10 * (n_inc - 1) / (samples - 1), n_inc)
    (cal / "incidenceAngles.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<incidenceAngles xmlns="rcmGsProductSchema">\n'
        "\t<pixelFirstAnglesValue>0</pixelFirstAnglesValue>\n\t<stepSize>10</stepSize>\n"
        f"\t<numberOfValues>{n_inc}</numberOfValues>\n"
        f"\t<angles>{' '.join(f'{a:.6f}' for a in angles)}</angles>\n</incidenceAngles>\n")
    tps = "".join(
        "<imageTiePoint><imageCoordinate>"
        f"<line>{r}</line><pixel>{c}</pixel></imageCoordinate><geodeticCoordinate>"
        f"<latitude>{lat}</latitude><longitude>{lon}</longitude><height>0</height>"
        "</geodeticCoordinate></imageTiePoint>"
        for r, c, lon, lat in tie)
    (root / "metadata" / "product.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<product xmlns="rcmGsProductSchema">
  <sourceAttributes><satellite>RCM-1</satellite><beamModeMnemonic>SCLNA</beamModeMnemonic>
    <rawDataStartTime>2026-01-28T10:47:01.000000Z</rawDataStartTime>
    <orbitAndAttitude><orbitInformation><passDirection>Descending</passDirection></orbitInformation></orbitAndAttitude>
  </sourceAttributes>
  <imageGenerationParameters><generalProcessingInformation><productType>GRD</productType>
  </generalProcessingInformation></imageGenerationParameters>
  <sceneAttributes><imageAttributes><incAngNearRng>{inc_near}</incAngNearRng>
    <incAngFarRng>{inc_far}</incAngFarRng></imageAttributes></sceneAttributes>
  <imageReferenceAttributes><rasterAttributes><numLines>{lines}</numLines>
    <samplesPerLine>{samples}</samplesPerLine><sampledPixelSpacing>50.0</sampledPixelSpacing>
    <sampledLineSpacing>50.0</sampledLineSpacing><lineTimeOrdering>Increasing</lineTimeOrdering>
    <pixelTimeOrdering>Increasing</pixelTimeOrdering></rasterAttributes>
    <geographicInformation><geolocationGrid>{tps}</geolocationGrid></geographicInformation>
  </imageReferenceAttributes>
  <polarizations>{' '.join(pols)}</polarizations>
</product>
""")
    return root


def test_find_rcm_files_and_metadata(tmp_path):
    prod = _rcm_grd_product(tmp_path / "RCM1_TEST_GRD")
    f = sar.radarsat.find_rcm_files(prod, "hv")
    assert f["image"].name == "RCM1_TEST_HV.tif"
    assert f["lut_sigma"].name == "lutSigma_HV.xml" and f["noise"].name == "noiseLevels_HV.xml"
    assert f["lut_beta"] is None and f["incidence"].name == "incidenceAngles.xml"
    with pytest.raises(FileNotFoundError):
        sar.radarsat.find_rcm_files(prod, "VV")

    meta = sar.radarsat.read_rcm_metadata(f["product_xml"])
    assert meta["satellite"] == "RCM-1" and meta["beam_mode"] == "SCLNA"
    assert meta["product_type"] == "GRD" and meta["pass"] == "Descending"
    assert meta["acquisition_start"] == np.datetime64("2026-01-28T10:47:01", "ns")
    assert (meta["lines"], meta["samples"]) == (60, 80)
    assert meta["incidence_near"] == 20.0 and meta["incidence_far"] == 40.0
    assert meta["polarizations"] == "HH HV"

    gcps = sar.radarsat.read_rcm_tie_points(f["product_xml"])
    assert len(gcps) == 25
    assert gcps[0].row == 0 and gcps[0].col == 0 and gcps[0].x == pytest.approx(-68.0)


def test_rcm_incidence_angle(tmp_path):
    prod = _rcm_grd_product(tmp_path / "p")
    f = sar.radarsat.find_rcm_files(prod)
    inc = sar.radarsat.read_rcm_incidence(f["incidence"])
    assert inc["pixel_first"] == 0 and inc["step"] == 10 and inc["n"] == 9
    full = sar.radarsat.rcm_incidence_angle(f["incidence"], 80)
    assert full[0] == pytest.approx(20.0) and full[79] == pytest.approx(40.0, abs=1e-6)
    win = sar.radarsat.rcm_incidence_angle(inc, 10, col0=30)
    assert np.allclose(win, full[30:40])
    meta = sar.radarsat.read_rcm_metadata(f["product_xml"])
    lin = sar.radarsat.rcm_incidence_angle(meta, 80)
    assert np.allclose(lin, full, atol=1e-6)
    meta_rev = dict(meta, pixel_time_ordering="Decreasing")      # far range at column 0
    assert sar.radarsat.rcm_incidence_angle(meta_rev, 80)[0] == pytest.approx(40.0)


def test_interpolate_rcm_lut_noise_col0():
    lut = {"pixel_first": 0, "step": 10, "n": 3, "offset": 0.0, "gains": np.array([1.0, 2.0, 3.0])}
    assert np.allclose(sar.radarsat.interpolate_rcm_lut(lut, 5, col0=5), [1.5, 1.6, 1.7, 1.8, 1.9])
    segs = [{"beam": "a", "pixel_first": 0, "step": 10, "n": 2, "values_db": np.array([-30.0, -30.0])},
            {"beam": "b", "pixel_first": 20, "step": 10, "n": 2, "values_db": np.array([-20.0, -20.0])}]
    out = sar.radarsat.interpolate_rcm_noise(segs, 10, col0=10)        # 10..19: seam gap between beams
    assert out[0] == pytest.approx(-30.0) and np.all(np.diff(out) > 0) and out[-1] < -20.0
    assert np.isnan(sar.radarsat.interpolate_rcm_noise([], 4)).all()


@pytest.mark.parametrize("tif_gcps", [True, False])
def test_read_rcm_grd_subset(tmp_path, tif_gcps):
    prod = _rcm_grd_product(tmp_path / "p", tif_gcps=tif_gcps, noise_db=-30.0)
    bounds = (-67.75, 48.25, -67.25, 48.75)                     # middle half of the scene
    out = sar.radarsat.read_rcm_grd_subset(prod, "HV", bounds=bounds, margin=2)
    row0, col0, h, w = out["window"]
    assert 0 < row0 and 0 < col0 and row0 + h < 60 and col0 + w < 80   # only a window was read
    expect = 1000**2 / 1.0e7 - 10 ** (-30.0 / 10)                       # sigma0 minus NESZ
    assert out["data"].dtype == np.float32 and np.allclose(out["data"], expect, rtol=1e-5)
    assert out["incidence"].shape == out["data"].shape
    assert out["incidence"][0, 0] == pytest.approx(20.0 + 20.0 * col0 / 79, abs=1e-4)
    assert out["pol"] == "HV" and out["meta"]["beam_mode"] == "SCLNA"
    assert min(g.row for g in out["gcps"]) <= 0 <= max(g.row for g in out["gcps"])   # shifted to window

    raw = sar.radarsat.read_rcm_grd_subset(prod, "HV", bounds=bounds, margin=2, noise=False, incidence=False)
    assert np.allclose(raw["data"], 1000**2 / 1.0e7) and raw["incidence"] is None
    with pytest.raises(FileNotFoundError):
        sar.radarsat.read_rcm_grd_subset(prod, "HV", bounds=bounds, kind="beta")
    with pytest.raises(ValueError):
        sar.radarsat.read_rcm_grd_subset(prod, "HV", bounds=(10.0, 10.0, 11.0, 11.0))


def test_read_rcm_grd_subset_stacks(tmp_path):
    # Two dates through create_stack land on one map grid, with incidence geocoded alongside.
    a = _rcm_grd_product(tmp_path / "a", dn=1000)
    b = _rcm_grd_product(tmp_path / "b", dn=2000)
    bounds = (-67.75, 48.25, -67.25, 48.75)
    sa = sar.radarsat.read_rcm_grd_subset(a, "HH", bounds=bounds)
    sb = sar.radarsat.read_rcm_grd_subset(b, "HH", bounds=bounds)
    stack = sar.sentinel1.create_stack([sa, sb], ["2026-01-28", "2026-02-03"], bounds=bounds,
                                       crs="EPSG:32619", resolution=2000.0)
    assert stack.sizes["time"] == 2
    ratio = (stack.isel(time=1) / stack.isel(time=0)).values
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size > 0 and np.allclose(ratio, (2000**2 / 1e7 - 1e-3) / (1000**2 / 1e7 - 1e-3), rtol=1e-3)
    inc = sar.sentinel1.create_stack([{"data": sa["incidence"], "gcps": sa["gcps"]}], ["2026-01-28"],
                                     bounds=bounds, crs="EPSG:32619", resolution=2000.0, name="incidence")
    vals = inc.values[np.isfinite(inc.values)]
    assert vals.size > 0 and 20.0 <= vals.min() and vals.max() <= 40.0


def test_incidence_normalization():
    rng = np.random.default_rng(0)
    theta = np.broadcast_to(np.linspace(20, 45, 200), (100, 200))
    s = -15.0 - 0.22 * (theta - 35.0) + rng.normal(0, 0.3, theta.shape)
    fit = sar.radarsat.fit_incidence_slope(s, theta)
    assert fit["slope"] == pytest.approx(-0.22, abs=0.01) and fit["r2"] > 0.9 and fit["n"] == s.size
    norm = sar.radarsat.normalize_incidence(s, theta, fit["slope"], ref_deg=35.0)
    assert abs(np.polyfit(theta.ravel(), norm.ravel(), 1)[0]) < 0.01      # trend removed
    assert norm.mean() == pytest.approx(-15.0, abs=0.05)
    da = xr.DataArray(s, dims=("y", "x"))
    nda = sar.radarsat.normalize_incidence(da, theta, -0.22)
    assert isinstance(nda, xr.DataArray) and nda.attrs["incidence_reference_deg"] == 35.0
    mask = np.zeros(s.shape, bool)
    mask[:2, :2] = True
    with pytest.raises(ValueError):
        sar.radarsat.fit_incidence_slope(s, theta, mask)
    with pytest.raises(ValueError):
        sar.radarsat.fit_incidence_slope(s, np.full(s.shape, 30.0))


def test_rcm_real_product_layouts(tmp_path):
    # Layouts seen in real RCM ScanSAR GRD products (pixelTimeOrdering "Decreasing"): range curves
    # stored from the last column backwards (negative stepSize), one <angles> element per value,
    # the calibration type spelled "Gamma", and a whole-swath referenceNoiseLevel.
    lut_xml = tmp_path / "lutSigma_HH.xml"
    _rcm_lut_xml(lut_xml, pixel_first=99, step=-11, gains=list(np.linspace(3.0e8, 2.0e8, 10)))
    lut = sar.radarsat.read_rcm_lut(lut_xml)
    gain = sar.radarsat.interpolate_rcm_lut(lut, 100)
    assert gain[99] == pytest.approx(3.0e8) and gain[0] == pytest.approx(2.0e8)
    assert np.all(np.diff(gain) >= 0)                                   # monotone, not garbled
    assert np.allclose(sar.radarsat.interpolate_rcm_lut(lut, 10, col0=50), gain[50:60])

    inc_xml = tmp_path / "incidenceAngles.xml"
    angles = np.linspace(19.0, 46.0, 10)                                # near range at the LAST column
    inc_xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<incidenceAngles xmlns="rcmGsProductSchema">\n'
        "<pixelFirstAnglesValue>99</pixelFirstAnglesValue><stepSize>-11</stepSize>"
        "<numberOfValues>10</numberOfValues>"
        + "".join(f'<angles units="deg">{a}</angles>' for a in angles) + "</incidenceAngles>")
    inc = sar.radarsat.read_rcm_incidence(inc_xml)
    assert inc["angles"].size == 10
    theta = sar.radarsat.rcm_incidence_angle(inc_xml, 100)
    assert theta[99] == pytest.approx(19.0) and theta[0] == pytest.approx(46.0)

    noise_xml = tmp_path / "noiseLevels_HH.xml"
    seg = lambda tag, cal, beam, first, step, vals: (
        f"<{tag}>" + (f"<beam>{beam}</beam>" if beam else "") + f"<sarCalibrationType>{cal}</sarCalibrationType>"
        f"<pixelFirstNoiseValue>{first}</pixelFirstNoiseValue><stepSize>{step}</stepSize>"
        f"<numberOfValues>{len(vals)}</numberOfValues><noiseLevelValues units=\"dB\">"
        + " ".join(map(str, vals)) + f"</noiseLevelValues></{tag}>")
    noise_xml.write_text(
        '<noiseLevels xmlns="rcmGsProductSchema">'
        + seg("referenceNoiseLevel", "Sigma Nought", None, 99, -11, [-30.0] * 10)
        + seg("perBeamReferenceNoiseLevel", "Sigma Nought", "SC-1", 99, -2, list(np.linspace(-34, -36, 26)))
        + seg("perBeamReferenceNoiseLevel", "Sigma Nought", "SC-2", 49, -2, list(np.linspace(-33, -35, 25)))
        + seg("perBeamReferenceNoiseLevel", "Gamma", "SC-1", 99, -2, [-20.0] * 26)
        + "</noiseLevels>")
    segs = sar.radarsat.read_rcm_noise_levels(noise_xml, "Sigma Nought")
    assert [s["beam"] for s in segs] == ["SC-1", "SC-2"]
    nesz = sar.radarsat.interpolate_rcm_noise(segs, 100)
    assert nesz[99] == pytest.approx(-34.0) and nesz[0] == pytest.approx(-35.0)
    assert nesz[49] == pytest.approx(-36.0)                             # shared column: higher-column beam (SC-1) wins
    assert nesz[48] == pytest.approx(-33.0 - 2.0 / 48, abs=1e-6)        # inside SC-2 only
    assert -36.0 <= nesz.min() and nesz.max() <= -33.0                  # values from inside the segments
    assert len(sar.radarsat.read_rcm_noise_levels(noise_xml, "Gamma Nought")) == 1   # alias of "Gamma"

    ref_only = tmp_path / "noiseLevels_HV.xml"
    ref_only.write_text('<noiseLevels xmlns="rcmGsProductSchema">'
                        + seg("referenceNoiseLevel", "Sigma Nought", None, 99, -11, [-30.0] * 10) + "</noiseLevels>")
    segs = sar.radarsat.read_rcm_noise_levels(ref_only)
    assert len(segs) == 1 and segs[0]["beam"] == "all"
    assert np.allclose(sar.radarsat.interpolate_rcm_noise(segs, 100), -30.0)
