import numpy as np
import pytest
import xarray as xr

h5py = pytest.importorskip("h5py")

from geolattice.rsense import sar


def _write_group(handle, path, values, attrs=None):
    dataset = handle.create_dataset(path, data=values)
    for key, value in (attrs or {}).items():
        dataset.attrs[key] = value
    return dataset


@pytest.fixture
def nisar_file(tmp_path):
    path = tmp_path / "nisar_test.h5"
    with h5py.File(path, "w") as handle:
        ident = handle.create_group("/science/LSAR/identification")
        ident.attrs["missionId"] = "NISAR"
        ident.attrs["productType"] = "RSLC"
        ident.create_dataset("trackNumber", data=42)
        _write_group(handle, "/science/LSAR/RSLC/swaths/frequencyA/HH",
                     np.array([[3 + 4j, 0 + 1j, 2 + 0j, 2 + 0j],
                               [0 + 2j, 1 + 0j, 1 + 1j, 1 + 0j],
                               [1 + 0j, 1 + 0j, 2 + 0j, 2 + 0j],
                               [1 + 0j, 1 + 0j, 2 + 0j, 2 + 0j]], dtype="complex64"))
        _write_group(handle, "/science/LSAR/RSLC/swaths/frequencyA/slantRange",
                     np.array([800_000.0, 800_010.0, 800_020.0, 800_030.0]))
        _write_group(handle, "/science/LSAR/RSLC/swaths/frequencyA/zeroDopplerTime",
                     np.array([0.0, 1.0, 2.0, 3.0]))
        geo = handle.create_group("/science/LSAR/RSLC/metadata/geolocationGrid")
        geo.create_dataset("coordinateX", data=np.full((2, 4, 4), 500_000.0))
        geo.create_dataset("coordinateY", data=np.full((2, 4, 4), 4_700_000.0))

        _write_group(handle, "/science/LSAR/GCOV/grids/frequencyA/HHHH",
                     np.array([[4.0, 4.0], [4.0, 4.0]], dtype="float32"))
        _write_group(handle, "/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
                     np.array([500_000.0, 500_030.0]))
        _write_group(handle, "/science/LSAR/GCOV/grids/frequencyA/yCoordinates",
                     np.array([4_700_000.0, 4_699_970.0]))
        _write_group(handle, "/science/LSAR/GCOV/grids/frequencyA/projection", np.int32(32618))
        _write_group(
            handle,
            "/science/LSAR/GCOV/grids/frequencyA/VHVH",
            np.array([[1.0, -9999.0], [2.0, 3.0]], dtype="float32"),
            {"_FillValue": -9999.0, "units": "linear gamma0"},
        )
        nested_phase = _write_group(
            handle,
            "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase",
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
        )
        nested_x = _write_group(
            handle,
            "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/xCoordinates",
            np.array([500_000.0, 500_080.0]),
        )
        nested_y = _write_group(
            handle,
            "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/yCoordinates",
            np.array([4_700_000.0, 4_699_920.0]),
        )
        nested_x.make_scale("xCoordinates")
        nested_y.make_scale("yCoordinates")
        nested_phase.dims[0].attach_scale(nested_y)
        nested_phase.dims[1].attach_scale(nested_x)
        _write_group(
            handle,
            "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/projection",
            np.int32(32618),
        )
    return path


def test_inspect_hdf5_and_read_identification(nisar_file):
    structure = sar.inspect_hdf5(nisar_file, "/science/LSAR/RSLC")
    assert structure["/science/LSAR/RSLC/swaths/frequencyA/HH"]["dtype"] == "complex64"
    metadata = sar.read_identification(nisar_file, "RSLC")
    assert metadata["product"] == "RSLC"
    assert metadata["missionId"] == "NISAR"
    assert metadata["trackNumber"] == 42


def test_read_nisar_radar_and_l2(nisar_file):
    radar = sar.read_radar(nisar_file, "RSLC", "HH")
    assert radar.dims == ("azimuth", "range")
    assert radar.shape == (4, 4)
    assert radar.slantRange.values[0] == 800_000
    assert radar.zeroDopplerTime.values[-1] == 3

    geolocation = sar.read_nisar(
        nisar_file, "RSLC", "metadata/geolocationGrid/coordinateX"
    )
    assert geolocation.dims == ("band", "azimuth", "range")
    assert geolocation.shape == (2, 4, 4)

    layer = sar.read_l2(nisar_file, "GCOV", "VHVH")
    assert layer.dims == ("y", "x")
    assert layer.attrs["crs"] == "EPSG:32618"
    assert np.isnan(layer.values[0, 1])
    assert layer.attrs["units"] == "linear gamma0"
    nested_layer = sar.read_l2(
        nisar_file, "GUNW", "unwrappedInterferogram/HH/unwrappedPhase"
    )
    assert nested_layer.dims == ("y", "x")
    assert nested_layer.attrs["crs"] == "EPSG:32618"
    assert nested_layer.x.values.tolist() == [500_000.0, 500_080.0]
    assert nested_layer.y.values.tolist() == [4_700_000.0, 4_699_920.0]
    with pytest.raises(ValueError, match="Level-1"):
        sar.read_l2(nisar_file, "RSLC", "HH")


def test_read_rslc_backscatter(nisar_file):
    incidence = np.full((4, 4), 30.0)
    result = sar.read_rslc_backscatter(
        nisar_file, "HH", incidence_angle=incidence, looks=(2, 2)
    )
    expected_beta = np.mean(np.abs(np.array([[3 + 4j, 0 + 1j], [0 + 2j, 1 + 0j]])) ** 2)
    assert result.beta0.shape == (2, 2)
    assert result.beta0.values[0, 0] == pytest.approx(expected_beta)
    assert result.sigma0.values[0, 0] == pytest.approx(expected_beta * np.sin(np.deg2rad(30)))
    assert result.beta0_db.attrs["units"] == "dB"


def test_process_rifg_masks_low_coherence_and_multilooks():
    ifg = np.ones((4, 4), dtype="complex64") * np.exp(0.5j)
    coherence = np.full((4, 4), 0.8)
    coherence[0, 0] = 0.1
    result = sar.process_rifg(ifg, coherence, min_coherence=0.3, looks=(2, 2))
    assert result.interferogram.shape == (2, 2)
    assert result.coherence.values[0, 0] == pytest.approx(0.8)
    assert result.wrapped_phase.values[0, 0] == pytest.approx(0.5)
    assert result.valid.values[1, 1]


def test_gslc_interferogram():
    y = np.arange(12.0)
    x = np.arange(12.0)
    ref = xr.DataArray(np.ones((12, 12), dtype="complex64"),
                       dims=("y", "x"), coords={"y": y, "x": x})
    sec = ref * np.exp(-0.2j)
    result = sar.gslc_interferogram(ref, sec, window=(3, 3), min_coherence=0.9)
    assert np.nanmean(result.coherence.values) == pytest.approx(1.0)
    assert np.nanmean(result.wrapped_phase.values) == pytest.approx(0.2, abs=1e-5)
    with pytest.raises(ValueError, match="grids must match"):
        sar.gslc_interferogram(ref, sec.assign_coords(x=x + 1))


def test_process_unwrapped_phase_reference_iono_and_vertical():
    phase = xr.DataArray(np.full((4, 4), 2.0), dims=("y", "x"))
    phase.values[0, 0] = 20.0
    coherence = np.full((4, 4), 0.8)
    coherence[0, 0] = 0.1
    component = np.ones((4, 4), dtype="uint8")
    component[0, 0] = 0
    reference = np.zeros((4, 4), dtype=bool)
    reference[1:, 1:] = True
    result = sar.process_unwrapped_phase(
        phase, wavelength=0.24, coherence=coherence, connected_components=component,
        reference_mask=reference, ionosphere_phase=np.full((4, 4), 0.5),
        incidence_angle=np.full((4, 4), 60.0), vertical_only=True,
    )
    assert result.unwrapped_phase.values[2, 2] == pytest.approx(0)
    assert result.los_displacement.values[2, 2] == pytest.approx(0)
    assert np.isnan(result.los_displacement.values[0, 0])
    assert "vertical_displacement" in result


def test_gcov_indices():
    coords = {"y": [1.0, 0.0], "x": [0.0, 1.0]}
    hh = xr.DataArray(np.full((2, 2), 4.0), dims=("y", "x"), coords=coords)
    hv = xr.DataArray(np.full((2, 2), 1.0), dims=("y", "x"), coords=coords)
    vv = xr.DataArray(np.full((2, 2), 3.0), dims=("y", "x"), coords=coords)
    result = sar.gcov_indices(hh, hv, vv)
    assert np.allclose(result.hv_hh_ratio, 0.25)
    assert np.allclose(result.dual_pol_rvi, 0.8)
    assert np.allclose(result.quad_pol_rvi, 8 / 9)


def test_gcov_indices_dual_pol():
    coords = {"y": [1.0, 0.0], "x": [0.0, 1.0]}
    hh = xr.DataArray(np.full((2, 2), 4.0), dims=("y", "x"), coords=coords)
    hv = xr.DataArray(np.full((2, 2), 1.0), dims=("y", "x"), coords=coords)
    result = sar.gcov_indices(hh, hv)
    assert set(result.data_vars) == {"hv_hh_ratio", "dual_pol_rvi"}
    assert result.attrs["polarization_mode"] == "dual"
    assert np.allclose(result.dual_pol_rvi, 0.8)


def test_process_offsets_quality_and_velocity():
    azimuth = np.full((9, 9), 2.0)
    slant = np.full((9, 9), 1.0)
    azimuth[4, 4] = 100.0
    snr = np.full((9, 9), 10.0)
    snr[0, 0] = 1.0
    stable = np.zeros((9, 9), dtype=bool)
    stable[:3, :3] = True
    original = azimuth.copy()
    result = sar.process_offsets(
        azimuth, slant, snr=snr, incidence_angle=np.full((9, 9), 30.0),
        heading=0.0, stable_mask=stable, interval_days=10,
    )
    assert np.array_equal(azimuth, original)
    assert not result.valid.values[0, 0]
    assert np.isnan(result.along_track_offset_m.values[0, 0])
    assert "east_offset_m" in result and "horizontal_velocity_m_per_year" in result


def test_geocode_l1():
    y, x = np.mgrid[0:5, 0:5]
    values = (x + y).astype("float32")
    result = sar.geocode_l1(
        values,
        500_000 + x * 30,
        4_700_000 - y * 30,
        epsg=32618,
        resolution=30,
    )
    assert result.ndim == 2
    assert np.isfinite(result.values).sum() > 0
    assert result.attrs.get("crs") == "EPSG:32618" or result.rio.crs.to_epsg() == 32618
