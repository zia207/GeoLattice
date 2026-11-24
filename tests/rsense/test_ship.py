import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geolattice.rsense import sar
from geolattice.rsense.sar import ship


def _speckled(mean_img, looks=4.4, seed=0):
    rng = np.random.default_rng(seed)
    return (mean_img * rng.gamma(looks, 1 / looks, mean_img.shape)).astype("float64")


# ============================================================================= land mask
def test_land_mask_for_window_buffers_land():
    from rasterio.control import GroundControlPoint as GCP

    shape = (100, 100)
    gcps = [
        GCP(row=0, col=0, x=10.0, y=50.0),
        GCP(row=0, col=100, x=10.1, y=50.0),
        GCP(row=100, col=0, x=10.0, y=49.9),
        GCP(row=100, col=100, x=10.1, y=49.9),
    ]
    from shapely.geometry import box

    # Land covers the western (low-column) half of the window in lon/lat.
    land_geoms = [box(10.0, 49.9, 10.05, 50.0)]
    mask = ship.land_mask_for_window(gcps, shape, land_geoms, buffer_m=0.0, pixel_spacing=10.0)
    assert mask.dtype == bool and mask.shape == shape
    assert mask[:, :45].mean() > 0.9
    assert mask[:, 55:].mean() < 0.1


def test_land_mask_for_window_no_geoms_is_all_false():
    shape = (20, 20)
    mask = ship.land_mask_for_window([], shape, [])
    assert mask.shape == shape and not mask.any()


def test_land_mask_for_window_buffer_grows_mask():
    from rasterio.control import GroundControlPoint as GCP
    from shapely.geometry import box

    shape = (100, 100)
    gcps = [
        GCP(row=0, col=0, x=10.0, y=50.0),
        GCP(row=0, col=100, x=10.1, y=50.0),
        GCP(row=100, col=0, x=10.0, y=49.9),
        GCP(row=100, col=100, x=10.1, y=49.9),
    ]
    land_geoms = [box(10.0, 49.9, 10.02, 50.0)]
    small = ship.land_mask_for_window(gcps, shape, land_geoms, buffer_m=0.0, pixel_spacing=10.0)
    big = ship.land_mask_for_window(gcps, shape, land_geoms, buffer_m=300.0, pixel_spacing=10.0)
    assert big.sum() > small.sum()
    assert (big | small).sum() == big.sum()  # dilation only grows the mask


# ============================================================================= clutter statistics
def test_estimate_enl_recovers_looks():
    img = _speckled(np.full((200, 200), 0.1), looks=4.4, seed=1)
    valid = np.ones_like(img, bool)
    assert ship.estimate_enl(img, valid) == pytest.approx(4.4, rel=0.15)


def test_estimate_enl_no_homogeneous_blocks_raises():
    img = np.zeros((10, 10))
    valid = np.zeros((10, 10), bool)
    with pytest.raises(ValueError):
        ship.estimate_enl(img, valid, block=64)


def test_ring_stats_matches_brute_force():
    rng = np.random.default_rng(2)
    img = rng.normal(5.0, 1.0, (40, 40))
    valid = np.ones_like(img, bool)
    valid[0, 0] = False
    mean, std, n = ship.ring_stats(img, valid, outer=9, guard=3)

    r, c = 20, 20
    half_o, half_g = 9 // 2, 3 // 2
    ring_mask = np.zeros_like(valid)
    ring_mask[r - half_o:r + half_o + 1, c - half_o:c + half_o + 1] = True
    ring_mask[r - half_g:r + half_g + 1, c - half_g:c + half_g + 1] = False
    ring_mask &= valid
    expected_mean = img[ring_mask].mean()
    expected_n = ring_mask.sum()
    assert mean[r, c] == pytest.approx(expected_mean, rel=1e-6)
    assert n[r, c] == pytest.approx(expected_n, abs=1e-6)
    assert std[r, c] > 0


def test_ring_stats_guard_ge_outer_raises():
    img = np.ones((20, 20))
    valid = np.ones((20, 20), bool)
    with pytest.raises(ValueError):
        ship.ring_stats(img, valid, outer=9, guard=9)


# ============================================================================= CFAR
@pytest.fixture
def cfar_scene():
    """A uniform clutter field with one bright target blob, away from the edges."""
    shape = (220, 220)
    img = _speckled(np.full(shape, 1.0), looks=4.4, seed=3)
    img[108:112, 108:112] = _speckled(np.full((4, 4), 60.0), looks=4.4, seed=4)
    valid = np.ones(shape, bool)
    return img, valid


@pytest.mark.parametrize("method", ["gamma", "lognormal"])
def test_cfar_detects_target_not_clutter(cfar_scene, method):
    img, valid = cfar_scene
    det, bg = ship.cfar(img, valid, method=method, pfa=1e-6, enl=4.4, outer=41, guard=21)
    assert det.dtype == bool
    assert det[108:112, 108:112].any()
    far = det.copy()
    far[90:130, 90:130] = False
    assert far.sum() < 5  # near-zero false alarms elsewhere at this pfa
    assert np.isfinite(bg[far.shape[0] // 2, far.shape[1] // 2])


def test_cfar_unknown_method_raises(cfar_scene):
    img, valid = cfar_scene
    with pytest.raises(ValueError):
        ship.cfar(img, valid, method="bogus")


def test_cfar_respects_min_ring_fraction(cfar_scene):
    img, valid = cfar_scene
    valid = valid.copy()
    valid[:110, :] = False  # kill the ring coverage of the lower half
    det, _ = ship.cfar(img, valid, method="gamma", pfa=1e-6, enl=4.4, outer=41, guard=21,
                       min_ring_fraction=0.5)
    assert not det[:60, :].any()


def test_cfar_tiled_matches_single_pass(cfar_scene):
    img, valid = cfar_scene
    kwargs = dict(method="gamma", pfa=1e-6, enl=4.4, outer=41, guard=21)
    det_full, bg_full = ship.cfar(img, valid, **kwargs)
    det_tiled, bg_tiled = ship.cfar_tiled(img, valid, tile=64, **kwargs)
    np.testing.assert_array_equal(det_full, det_tiled)
    np.testing.assert_allclose(bg_full, bg_tiled, equal_nan=True)


# ============================================================================= targets
def test_extract_targets_basic_geometry_and_contrast():
    det = np.zeros((60, 60), bool)
    det[10:13, 10:13] = True  # 3x3 blob -> length/width in pixels
    det[40, 40] = True  # single-pixel blob, dropped by min_px=2
    vh = np.full((60, 60), 10 ** (-20 / 10))
    vh[10:13, 10:13] = 10 ** (0 / 10)
    background_db = np.full((60, 60), -20.0)

    df = ship.extract_targets(det, vh, gcps=None, background_db=background_db,
                              pixel_spacing=10.0, dilate=0, min_px=2, max_px=100)
    assert len(df) == 1
    row = df.iloc[0]
    assert row.row == pytest.approx(11.0, abs=0.5)
    assert row.col == pytest.approx(11.0, abs=0.5)
    assert row.n_px == 9
    assert row.length_m > 0 and row.width_m > 0
    assert row.contrast_db == pytest.approx(row.vh_peak_db - (-20.0), abs=1e-5)


def test_extract_targets_empty_detection_returns_empty():
    det = np.zeros((20, 20), bool)
    vh = np.ones((20, 20))
    df = ship.extract_targets(det, vh)
    assert df.empty


def test_extract_targets_min_contrast_filters():
    det = np.zeros((40, 40), bool)
    det[5:8, 5:8] = True
    det[25:28, 25:28] = True
    vh = np.full((40, 40), 1.0)
    vh[5:8, 5:8] = 100.0   # strong target
    vh[25:28, 25:28] = 1.5  # weak target
    background_db = np.full((40, 40), float(ship.to_db(np.array(1.0))))
    df = ship.extract_targets(det, vh, background_db=background_db, min_contrast_db=10.0,
                              dilate=0, min_px=2)
    assert len(df) == 1
    assert df.iloc[0].row == pytest.approx(6.0, abs=0.5)


def test_extract_targets_can_filter_by_mean_contrast():
    det = np.zeros((40, 40), bool)
    det[10:13, 10:13] = True
    vh = np.ones((40, 40))
    vh[10, 10] = 10.0
    background_db = np.zeros((40, 40))

    candidates = ship.extract_targets(
        det, vh, background_db=background_db, dilate=0, min_px=2
    )
    assert len(candidates) == 1
    assert candidates.iloc[0].contrast_db == pytest.approx(10.0)
    assert candidates.iloc[0].mean_contrast_db < 5.0

    filtered = ship.extract_targets(
        det, vh, background_db=background_db, dilate=0, min_px=2,
        min_mean_contrast_db=5.0,
    )
    assert filtered.empty


def test_extract_targets_with_vv_ratio():
    det = np.zeros((30, 30), bool)
    det[10:13, 10:13] = True
    vh = np.full((30, 30), 1.0)
    vh[10:13, 10:13] = 10.0
    vv = np.full((30, 30), 1.0)
    vv[10:13, 10:13] = 5.0
    df = ship.extract_targets(det, vh, vv, dilate=0, min_px=2)
    assert "vv_peak_db" in df and "vh_vv_ratio_db" in df
    assert len(df) == 1


def test_extract_targets_gcps_adds_lonlat():
    from rasterio.control import GroundControlPoint as GCP

    det = np.zeros((100, 100), bool)
    det[50, 50] = det[50, 51] = det[51, 50] = True
    vh = np.full((100, 100), 1.0)
    vh[50:52, 50:52] = 10.0
    lon0, lat0, dlon, dlat = 10.0, 50.0, 0.001, 0.0008
    W = H = 100
    gcps = [
        GCP(row=0, col=0, x=lon0, y=lat0),
        GCP(row=0, col=W, x=lon0 + W * dlon, y=lat0),
        GCP(row=H, col=0, x=lon0, y=lat0 - H * dlat),
        GCP(row=H, col=W, x=lon0 + W * dlon, y=lat0 - H * dlat),
    ]
    df = ship.extract_targets(det, vh, gcps=gcps, dilate=0, min_px=2)
    assert "lon" in df and "lat" in df
    assert lon0 <= df.iloc[0].lon <= lon0 + W * dlon
    assert lat0 - H * dlat <= df.iloc[0].lat <= lat0


# ============================================================================= azimuth ambiguities
def test_read_prf_sorted_unique():
    xml = """<product>
        <downlinkInformation><prf>486.49</prf></downlinkInformation>
        <downlinkInformation><prf>486.49</prf></downlinkInformation>
        <downlinkInformation><prf>486.56</prf></downlinkInformation>
    </product>"""
    root = ET.fromstring(xml)
    assert ship.read_prf(root) == [486.49, 486.56]


def test_flag_azimuth_ambiguities_flags_ghost():
    altitude, velocity, azimuth_spacing = 693e3, 7590.0, 10.0
    wavelength = sar.SPEED_OF_LIGHT / 5.405e9
    prf = 486.49
    incidence_deg = np.full((300, 100), 35.0)
    R = altitude / np.cos(np.radians(35.0))
    dx = wavelength * R * prf / (2 * velocity) / azimuth_spacing

    main_row, main_col = 150.0, 50.0
    targets = pd.DataFrame({
        "row": [main_row, main_row + dx],
        "col": [main_col, main_col + 2.0],
        "vh_peak_db": [20.0, 5.0],
    })
    flag = ship.flag_azimuth_ambiguities(targets, [prf], incidence_deg,
                                         wavelength=wavelength, altitude=altitude,
                                         velocity=velocity, azimuth_spacing=azimuth_spacing,
                                         min_ratio_db=10.0)
    assert flag.iloc[1] and not flag.iloc[0]


def test_flag_azimuth_ambiguities_needs_two_targets_and_prfs():
    incidence_deg = np.full((10, 10), 35.0)
    one = pd.DataFrame({"row": [5.0], "col": [5.0], "vh_peak_db": [20.0]})
    assert not ship.flag_azimuth_ambiguities(one, [486.0], incidence_deg).any()

    two = pd.DataFrame({"row": [5.0, 6.0], "col": [5.0, 6.0], "vh_peak_db": [20.0, 19.0]})
    assert not ship.flag_azimuth_ambiguities(two, [], incidence_deg).any()


# ============================================================================= GRD pipeline (mocked I/O)
def test_detect_ships_grd_pipeline(monkeypatch, tmp_path):
    shape = (220, 220)
    rng = np.random.default_rng(5)
    vv = _speckled(np.full(shape, 0.02), looks=4.4, seed=10)
    vh = _speckled(np.full(shape, 0.01), looks=4.4, seed=11)
    vh[108:112, 108:112] = _speckled(np.full((4, 4), 2.0), looks=4.4, seed=12)
    vv[108:112, 108:112] = _speckled(np.full((4, 4), 1.0), looks=4.4, seed=13)

    gcps = []  # no geolocation needed for this test
    window = (0, 0, shape[0], shape[1])

    def fake_read_safe(safe_dir):
        return {"measurement": {"vv": "vv.tif", "vh": "vh.tif"},
                "calibration": {"vv": "cal_vv.xml", "vh": "cal_vh.xml"},
                "noise": {"vv": "noise_vv.xml", "vh": "noise_vh.xml"}}

    def fake_read_grd_subset(measurement, calibration, noise, *, bounds, border_noise=False):
        data = vv if "vv" in str(measurement) else vh
        return {"data": data, "gcps": gcps, "window": window}

    monkeypatch.setattr("geolattice.rsense.core.io.read_safe", fake_read_safe)
    monkeypatch.setattr(ship, "read_grd_subset", fake_read_grd_subset)

    out = ship.detect_ships_grd(tmp_path, (10.0, 50.0, 10.1, 50.1), [], pol="vh",
                                enl=4.4, pfa=1e-6, outer=41, guard=21,
                                min_contrast_db=3.0, remove_ambiguities=False)
    assert set(out) == {"targets", "sigma0", "detections", "sea", "gcps", "window", "enl"}
    assert out["enl"] == 4.4
    assert out["detections"].shape == shape
    assert len(out["targets"]) >= 1
    row = out["targets"].iloc[(out["targets"].row - 110).abs().idxmin()]
    assert abs(row.row - 110) < 5 and abs(row.col - 110) < 5


# ============================================================================= SLC sub-looks
def test_read_processing_params():
    xml = """<product>
        <imageAnnotation>
          <processingInformation>
            <swathProcParamsList>
              <swathProcParams>
                <rangeProcessing>
                  <processingBandwidth>56500000.0</processingBandwidth>
                  <windowCoefficient>0.75</windowCoefficient>
                  <windowType>Hamming</windowType>
                </rangeProcessing>
                <azimuthProcessing>
                  <processingBandwidth>327.0</processingBandwidth>
                  <windowCoefficient>0.75</windowCoefficient>
                  <windowType>Hamming</windowType>
                </azimuthProcessing>
              </swathProcParams>
            </swathProcParamsList>
          </processingInformation>
        </imageAnnotation>
    </product>"""
    root = ET.fromstring(xml)
    params = ship.read_processing_params(root)
    assert params["range_bandwidth"] == pytest.approx(56500000.0)
    assert params["azimuth_bandwidth"] == pytest.approx(327.0)
    assert params["range_window_coefficient"] == pytest.approx(0.75)
    assert params["azimuth_window_type"] == "Hamming"


def test_read_processing_params_missing_section_raises():
    root = ET.fromstring("<product><imageAnnotation/></product>")
    with pytest.raises(ValueError):
        ship.read_processing_params(root)


def test_split_spectrum_reconstructs_energy():
    rng = np.random.default_rng(6)
    n = 256
    slc = (rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n))).astype("complex64")
    lo, hi = ship.split_spectrum(slc, axis=1, bandwidth_fraction=0.8, alpha=0.75)
    assert lo.shape == slc.shape and hi.shape == slc.shape
    # Energy of each sub-look should be well below the full-band signal (half-band + gap removed)
    assert np.abs(lo).sum() < np.abs(slc).sum()
    assert np.abs(hi).sum() < np.abs(slc).sum()


def test_sublook_coherence_identical_sublooks_are_coherent():
    rng = np.random.default_rng(7)
    s = (rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))).astype("complex64")
    coh, ihp = ship.sublook_coherence(s, s, window=(3, 9))
    interior = (slice(5, 35), slice(10, 30))
    assert np.nanmean(coh[interior]) > 0.95
    assert np.all(ihp[interior] >= 0)


def test_sublook_coherence_uncorrelated_is_low():
    rng = np.random.default_rng(8)
    s1 = (rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))).astype("complex64")
    s2 = (rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))).astype("complex64")
    coh, _ = ship.sublook_coherence(s1, s2, window=(3, 9))
    interior = (slice(10, 50), slice(10, 50))
    assert np.nanmean(coh[interior]) < 0.5


def test_sublook_coherence_respects_valid_mask():
    s = np.ones((20, 20), "complex64")
    valid = np.zeros((20, 20), bool)
    valid[5:15, 5:15] = True
    coh, ihp = ship.sublook_coherence(s, s, valid=valid)
    assert np.isnan(coh[0, 0]) and np.isnan(ihp[0, 0])
    assert np.isfinite(coh[8, 8])


def test_azimuth_sublooks_uses_deramp_phase(monkeypatch):
    rng = np.random.default_rng(9)
    burst = (rng.normal(size=(32, 16)) + 1j * rng.normal(size=(32, 16))).astype("complex64")
    ann = {"azimuth_time_interval": 0.002055}
    params = {"azimuth_bandwidth": 327.0, "azimuth_window_coefficient": 0.75}

    monkeypatch.setattr("geolattice.rsense.sar.sentinel1.tops_deramp_phase",
                        lambda ann, burst_index, lines, cols: np.zeros((lines.shape[0], cols.shape[1])))
    lo, hi = ship.azimuth_sublooks(burst, ann, burst_index=0, params=params)
    assert lo.shape == burst.shape and hi.shape == burst.shape


def test_range_sublooks_matches_split_spectrum():
    rng = np.random.default_rng(10)
    burst = (rng.normal(size=(16, 64)) + 1j * rng.normal(size=(16, 64))).astype("complex64")
    params = {"range_bandwidth": 56500000.0, "range_window_coefficient": 0.75}
    range_sampling_rate = 64_345_238.0
    lo, hi = ship.range_sublooks(burst, params, range_sampling_rate)
    exp_lo, exp_hi = ship.split_spectrum(burst, 1, params["range_bandwidth"] / range_sampling_rate,
                                         alpha=params["range_window_coefficient"])
    np.testing.assert_allclose(lo, exp_lo)
    np.testing.assert_allclose(hi, exp_hi)


def test_coherence_at_points_max_in_search_window(monkeypatch):
    H, W = 40, 40
    coh = np.zeros((H, W), "float32")
    coh[20, 20] = 0.8

    def fake_geo2rdr(lat, lon, height, orbit, **kwargs):
        lat = np.atleast_1d(lat)
        return np.zeros(lat.shape), np.zeros(lat.shape)

    def fake_burst_coordinates(ann, burst, t, r):
        return np.full(t.shape, 20.0), np.full(t.shape, 20.0)

    monkeypatch.setattr(ship, "geo2rdr", fake_geo2rdr)
    monkeypatch.setattr(ship, "burst_coordinates", fake_burst_coordinates)

    ann = {"orbit": {}, "bursts": {0: {"first_valid_line": 0, "last_valid_line": H - 1}}}
    out = ship.coherence_at_points([10.0], [20.0], ann, bursts=[0], cols=(0, W),
                                   coherence=[coh], search=(2, 2))
    assert out[0] == pytest.approx(0.8)


def test_coherence_at_points_outside_burst_is_nan(monkeypatch):
    H, W = 40, 40
    coh = np.zeros((H, W), "float32")

    def fake_geo2rdr(lat, lon, height, orbit, **kwargs):
        lat = np.atleast_1d(lat)
        return np.zeros(lat.shape), np.zeros(lat.shape)

    def fake_burst_coordinates(ann, burst, t, r):
        return np.full(t.shape, -100.0), np.full(t.shape, -100.0)

    monkeypatch.setattr(ship, "geo2rdr", fake_geo2rdr)
    monkeypatch.setattr(ship, "burst_coordinates", fake_burst_coordinates)

    ann = {"orbit": {}, "bursts": {0: {"first_valid_line": 0, "last_valid_line": H - 1}}}
    out = ship.coherence_at_points([10.0], [20.0], ann, bursts=[0], cols=(0, W),
                                   coherence=[coh], search=(2, 2))
    assert np.isnan(out[0])


# ============================================================================= AIS
_DMA_HEADER = ("# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,"
              "SOG,COG,Heading,IMO,Callsign,Name,Ship type,Cargo type,Width,Length,"
              "Type of position fixing device,Draught,Destination,ETA,Data source type")


def _dma_row(ts, mmsi, lat, lon, sog=10.0, cog=90.0, heading=90.0, length=100.0,
            ship_type="Cargo", mobile="Class A"):
    return (f"{ts},{mobile},{mmsi},{lat},{lon},Under way using engine,{sog},{cog},{heading},"
           f"0,,TESTSHIP,{ship_type},,20,{length},GPS,5,,,AIS")


@pytest.fixture
def ais_csv(tmp_path):
    rows = [
        _dma_row("27/10/2026 10:00:00", 200000001, 50.05, 10.05),
        _dma_row("27/10/2026 10:30:00", 200000002, 50.02, 10.02),  # outside time window
        _dma_row("27/10/2026 10:01:00", 200000003, 60.0, 60.0),    # outside bounds
        _dma_row("27/10/2026 10:01:00", 200000004, 50.03, 10.03, mobile="Class B"),
        _dma_row("27/10/2026 10:01:00", 999999999, 50.03, 10.03),  # invalid MMSI range
    ]
    path = tmp_path / "ais.csv"
    path.write_text(_DMA_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_read_ais_dma_filters(ais_csv):
    t0 = pd.Timestamp("2026-10-27 10:00:30")
    df = ship.read_ais_dma(ais_csv, bounds=(10.0, 50.0, 10.1, 50.1), t0=t0, window_min=15.0)
    assert set(df.mmsi) == {200000001, 200000004}
    assert "mobile" not in df.columns
    assert {"time", "mmsi", "lat", "lon", "sog", "cog", "heading", "length", "ship_type",
           "name"} <= set(df.columns)


# ============================================================================= AIS interpolation
def test_ais_at_time_interpolates_between_reports():
    ais = pd.DataFrame({
        "mmsi": [1, 1],
        "time": pd.to_datetime(["2026-10-27 10:00:00", "2026-10-27 10:02:00"]),
        "lat": [50.0, 50.02],
        "lon": [10.0, 10.02],
        "sog": [10.0, 10.0],
        "cog": [45.0, 45.0],
        "length": [100.0, 100.0],
        "ship_type": ["Cargo", "Cargo"],
    })
    out = ship.ais_at_time(ais, "2026-10-27 10:01:00", max_gap_s=180.0)
    assert len(out) == 1
    assert out.iloc[0].lat == pytest.approx(50.01, abs=1e-6)
    assert out.iloc[0].lon == pytest.approx(10.01, abs=1e-6)


def test_ais_at_time_dead_reckons_single_report():
    ais = pd.DataFrame({
        "mmsi": [2],
        "time": pd.to_datetime(["2026-10-27 10:00:00"]),
        "lat": [50.0],
        "lon": [10.0],
        "sog": [10.0],  # knots
        "cog": [0.0],   # due north
        "length": [50.0],
        "ship_type": ["Fishing"],
    })
    out = ship.ais_at_time(ais, "2026-10-27 10:01:00", max_gap_s=180.0)
    assert len(out) == 1
    assert out.iloc[0].lat > 50.0  # moved north
    assert out.iloc[0].lon == pytest.approx(10.0, abs=1e-6)


def test_ais_at_time_excludes_reports_outside_max_gap():
    ais = pd.DataFrame({
        "mmsi": [3],
        "time": pd.to_datetime(["2026-10-27 09:00:00"]),
        "lat": [50.0],
        "lon": [10.0],
        "sog": [0.0],
        "cog": [0.0],
        "length": [50.0],
        "ship_type": ["Fishing"],
    })
    out = ship.ais_at_time(ais, "2026-10-27 10:01:00", max_gap_s=180.0)
    assert out.empty


# ============================================================================= azimuth displacement
def test_azimuth_displacement_m_zero_for_along_track_motion():
    # Vessel moving along the flight direction (cog == heading) has no radial (line-of-sight)
    # velocity component, so no azimuth displacement.
    ais = pd.DataFrame({"sog": [10.0], "cog": [0.0]})
    out = ship.azimuth_displacement_m(ais, heading_deg=0.0, incidence_deg=35.0)
    assert out[0] == pytest.approx(0.0, abs=1e-6)


def test_azimuth_displacement_m_max_for_cross_track_motion():
    # Vessel moving along the look direction (cog == heading + 90) has the largest radial
    # velocity component, so the largest azimuth displacement.
    ais = pd.DataFrame({"sog": [20.0], "cog": [90.0]})
    out = ship.azimuth_displacement_m(ais, heading_deg=0.0, incidence_deg=35.0)
    assert abs(out[0]) > 0


# ============================================================================= evaluation
def test_haversine_m_one_degree_latitude():
    d = ship.haversine_m([0.0], [0.0], [1.0], [0.0])
    assert d[0, 0] == pytest.approx(111_195.0, rel=0.01)


def test_haversine_m_zero_distance():
    d = ship.haversine_m([10.0], [20.0], [10.0], [20.0])
    assert d[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_match_points_assigns_nearest_within_threshold():
    pred = pd.DataFrame({"lat": [50.0, 50.1], "lon": [10.0, 10.1]})
    truth = pd.DataFrame({"lat": [50.0001, 50.2], "lon": [10.0001, 10.3]})
    pairs, D = ship.match_points(pred, truth, max_dist_m=200.0)
    assert pairs == [(0, 0)]
    assert D.shape == (2, 2)


def test_match_points_empty_inputs():
    pred = pd.DataFrame({"lat": [], "lon": []})
    truth = pd.DataFrame({"lat": [50.0], "lon": [10.0]})
    pairs, D = ship.match_points(pred, truth)
    assert pairs == [] and D.shape == (0, 1)


def test_match_points_shift_m_enables_match_for_moving_target():
    # Truth is 150 m east of pred; without a shift this exceeds max_dist_m.
    pred = pd.DataFrame({"lat": [50.0], "lon": [10.0]})
    dlon = 150.0 / (6_371_000.0 * np.cos(np.radians(50.0))) * 180 / np.pi
    truth = pd.DataFrame({"lat": [50.0], "lon": [10.0 + dlon]})
    pairs, _ = ship.match_points(pred, truth, max_dist_m=50.0)
    assert pairs == []
    pairs, _ = ship.match_points(pred, truth, max_dist_m=50.0, shift_m=np.array([150.0]),
                                 heading_deg=90.0)
    assert pairs == [(0, 0)]


def test_detection_scores_perfect_match():
    scores = ship.detection_scores(5, 5, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    assert scores["tp"] == 5 and scores["fp"] == 0 and scores["fn"] == 0
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(1.0)


def test_detection_scores_no_matches():
    scores = ship.detection_scores(3, 4, [])
    assert scores["tp"] == 0 and scores["fp"] == 3 and scores["fn"] == 4
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == pytest.approx(0.0)


# ============================================================================= public API wiring
def test_ship_functions_reexported_from_sar_package():
    for name in ("detect_ships_grd", "read_ais_dma", "ais_at_time", "match_points",
                "detection_scores", "extract_targets", "land_mask_for_window"):
        assert getattr(sar, name) is getattr(ship, name)
    assert sar.ship_cfar is ship.cfar
    assert sar.ship is ship
