import numpy as np
import pytest
import xarray as xr

pytest.importorskip("h5netcdf")  # geolattice[rsense-sentinel5p]

from geolattice.rsense.atmosphere import sentinel_5p as s5p  # noqa: E402


def _qa_da(values, comment="Recommend to ignore data with qa_value < 0.5"):
    da = xr.DataArray(np.asarray(values, dtype="float32"), dims=("scanline", "ground_pixel"))
    da.attrs["comment"] = comment
    da.name = "qa_value"
    return da


def test_qa_mask_reads_threshold_from_comment():
    qa = _qa_da([[0.2, 0.6], [0.9, 0.4]], comment="Recommend to ignore data with qa_value < 0.75")
    valid = s5p.qa_mask(qa)
    assert valid.values.tolist() == [[False, False], [True, False]]


def test_qa_mask_explicit_threshold_overrides_comment():
    qa = _qa_da([[0.2, 0.6]], comment="Recommend to ignore data with qa_value < 0.75")
    valid = s5p.qa_mask(qa, min_qa=0.5)
    assert valid.values.tolist() == [[False, True]]


def test_qa_mask_falls_back_without_a_comment():
    qa = xr.DataArray(np.array([[0.3, 0.6]], dtype="float32"), dims=("scanline", "ground_pixel"))
    valid = s5p.qa_mask(qa)
    assert valid.values.tolist() == [[False, True]]  # default fallback is 0.5


def test_molecules_per_cm2_uses_the_files_own_factor():
    da = xr.DataArray(np.array([1e-5, 2e-5], dtype="float32"), dims="x", name="nitrogendioxide_tropospheric_column")
    da.attrs["multiplication_factor_to_convert_to_molecules_percm2"] = 6.02214e19
    out = s5p.molecules_per_cm2(da)
    np.testing.assert_allclose(out.values, [1e-5 * 6.02214e19, 2e-5 * 6.02214e19], rtol=1e-5)
    assert out.attrs["units"] == "molecules cm-2"


def test_molecules_per_cm2_requires_the_attribute():
    da = xr.DataArray(np.array([1.0]), dims="x", name="x")
    with pytest.raises(ValueError, match="multiplication_factor"):
        s5p.molecules_per_cm2(da)


def test_dobson_units_matches_known_conversion():
    # 1 DU = 4.4615e-4 mol/m^2 by definition; round-trip should return to 1.0 DU.
    one_du_in_mol_m2 = xr.DataArray(np.array([4.4615e-4], dtype="float32"), dims="x", name="ozone_total_vertical_column")
    du = s5p.dobson_units(one_du_in_mol_m2)
    np.testing.assert_allclose(du.values, [1.0], rtol=1e-3)
    assert du.attrs["units"] == "DU"


def _write_product_group(tmp_path, data_vars, filename="s5p_test.nc"):
    path = tmp_path / filename
    ds = xr.Dataset(data_vars)
    ds.to_netcdf(path, group="PRODUCT", engine="h5netcdf")
    return path


def test_read_no2_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, cols = 5, 6
    no2 = xr.DataArray(rng.uniform(0, 1e-4, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    no2.attrs["multiplication_factor_to_convert_to_molecules_percm2"] = 6.02214e19
    qa = xr.DataArray(rng.uniform(0, 1, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    qa.attrs["comment"] = "Recommend to ignore data with qa_value < 0.75"
    lon = xr.DataArray(rng.uniform(10, 11, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    lat = xr.DataArray(rng.uniform(45, 46, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    path = _write_product_group(tmp_path, {
        "nitrogendioxide_tropospheric_column": no2,
        "qa_value": qa,
        "longitude": lon,
        "latitude": lat,
    })

    ds = s5p.read_no2(path)
    assert "time" not in ds.dims  # singleton orbit dim squeezed away
    assert ds["nitrogendioxide_tropospheric_column"].shape == (rows, cols)
    # latitude/longitude must come back as coordinates, not merely be present somewhere --
    # every spatial use (mapping, swath_to_grid, zonal stats) needs them, and a regression
    # once dropped them from every reader's output entirely.
    assert "longitude" in ds.coords and "latitude" in ds.coords
    assert ds.longitude.shape == (rows, cols)
    valid = s5p.qa_mask(ds["qa_value"])
    assert valid.shape == (rows, cols)
    molec = s5p.molecules_per_cm2(ds["nitrogendioxide_tropospheric_column"])
    assert molec.attrs["units"] == "molecules cm-2"
    gridded = s5p.swath_to_grid(molec.where(valid), ds.longitude, ds.latitude, resolution=0.1)
    assert gridded.rio.crs is not None


def test_read_product_keeps_precision_companion_when_present(tmp_path):
    rng = np.random.default_rng(1)
    rows, cols = 4, 4
    main = xr.DataArray(rng.uniform(0, 1, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    prec = xr.DataArray(rng.uniform(0, 0.1, (1, rows, cols)).astype("float32"), dims=("time", "scanline", "ground_pixel"))
    path = _write_product_group(tmp_path, {
        "ozone_total_vertical_column": main,
        "ozone_total_vertical_column_precision": prec,
    })
    ds = s5p.read_o3(path)
    assert "ozone_total_vertical_column_precision" in ds.data_vars


def test_read_product_raises_for_unknown_variable(tmp_path):
    rows, cols = 3, 3
    main = xr.DataArray(np.zeros((1, rows, cols), dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    path = _write_product_group(tmp_path, {"sulfurdioxide_total_vertical_column": main})
    with pytest.raises(KeyError, match="not found"):
        s5p.read_so2(path, variables=["not_a_real_variable"])


def test_read_cloud_does_not_require_qa_value_to_be_present(tmp_path):
    # Real CLOUD products do carry a qa_value (confirmed against a real download); this just
    # checks the reader degrades gracefully for a fixture file that omits it, since qa_value
    # is included only `if "qa_value" in ds.data_vars`, never assumed.
    rows, cols = 3, 3
    cf = xr.DataArray(np.full((1, rows, cols), 0.5, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    ctp = xr.DataArray(np.full((1, rows, cols), 500.0, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    cth = xr.DataArray(np.full((1, rows, cols), 5000.0, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    cot = xr.DataArray(np.full((1, rows, cols), 10.0, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    path = _write_product_group(tmp_path, {
        "cloud_fraction": cf, "cloud_top_pressure": ctp,
        "cloud_top_height": cth, "cloud_optical_thickness": cot,
    })
    ds = s5p.read_cloud(path)
    assert "qa_value" not in ds.data_vars  # this product has none in the fixture, and none is required
    assert set(ds.data_vars) == {"cloud_fraction", "cloud_top_pressure", "cloud_top_height", "cloud_optical_thickness"}


def test_read_metadata_reads_global_attrs(tmp_path):
    path = tmp_path / "meta_test.nc"
    ds = xr.Dataset(attrs={
        "time_coverage_start": "2023-07-05T22:01:01Z",
        "time_coverage_end": "2023-07-05T22:06:14Z",
        "orbit": 29673,
        "processing_mode": "OFFL",
        "processor_version": "2.5.0",
        "irrelevant_attr": "ignored",
    })
    ds.to_netcdf(path, engine="h5netcdf")
    meta = s5p.read_metadata(path)
    assert meta["processing_mode"] == "OFFL"
    assert meta["orbit"] == 29673
    assert "irrelevant_attr" not in meta


def test_s5p_products_lists_expected_gases():
    assert {"no2", "so2", "o3", "co", "ch4", "hcho", "aer_ai"} <= set(s5p.S5P_PRODUCTS)


def test_swath_to_grid_is_reexported_from_sentinel3():
    from geolattice.rsense.optical.sensors.sentinel3 import swath_to_grid as s3_swath_to_grid
    assert s5p.swath_to_grid is s3_swath_to_grid


def test_s5p_qa_recommended_covers_all_products():
    assert set(s5p.S5P_QA_RECOMMENDED) >= set(s5p.S5P_PRODUCTS) | {"cloud"}
    assert s5p.S5P_QA_RECOMMENDED["no2"] == 0.75
    assert s5p.S5P_QA_RECOMMENDED["aer_ai"] == 0.8


def test_read_aerosol_layer_height_default_variable():
    assert s5p.S5P_PRODUCTS["aer_lh"][1] == "aerosol_mid_height"


def test_read_so2_rejects_bad_layer_height():
    with pytest.raises(ValueError, match="layer_height"):
        s5p.read_so2("unused.nc", layer_height="3km")


def test_read_so2_layer_height_end_to_end(tmp_path):
    rows, cols = 4, 5
    main = xr.DataArray(np.full((1, rows, cols), 1e-4, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    qa = _qa_da(np.full((rows, cols), 0.8, dtype="float32"))
    lon = xr.DataArray(np.full((1, rows, cols), 10.0, dtype="float32"), dims=("time", "scanline", "ground_pixel"))
    lat = xr.DataArray(np.full((1, rows, cols), 45.0, dtype="float32"), dims=("time", "scanline", "ground_pixel"))

    path = tmp_path / "s5p_so2_layer.nc"
    xr.Dataset({"sulfurdioxide_total_vertical_column_7km": main}).to_netcdf(
        path, group="PRODUCT/SUPPORT_DATA/DETAILED_RESULTS", engine="h5netcdf")
    xr.Dataset({"qa_value": qa.expand_dims(time=[0]), "longitude": lon, "latitude": lat}).to_netcdf(
        path, group="PRODUCT", engine="h5netcdf", mode="a")

    ds = s5p.read_so2(path, layer_height="7km")
    assert "sulfurdioxide_total_vertical_column_7km" in ds.data_vars
    assert "qa_value" in ds.data_vars
    assert "longitude" in ds.coords and "latitude" in ds.coords
    assert "time" not in ds.dims


# --------------------------------------------------------------------------- gridding
def test_bin_to_grid_averages_pixels_per_cell():
    # four pixels: two in one cell (values 1, 3 -> mean 2), two in another (values 10, 20 -> mean 15)
    values = np.array([1.0, 3.0, 10.0, 20.0])
    lat = np.array([0.1, 0.1, 0.1, 0.1])
    lon = np.array([0.1, 0.1, 1.1, 1.1])
    lat_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0, 2.0])
    out = s5p.bin_to_grid(values, lat, lon, lat_edges, lon_edges, min_count=1)
    np.testing.assert_allclose(out["mean"].values, [[2.0, 15.0]])
    np.testing.assert_array_equal(out["count"].values, [[2, 2]])
    assert out.rio.crs is not None


def test_bin_to_grid_min_count_drops_sparse_cells():
    values = np.array([1.0, 3.0, 10.0])
    lat = np.array([0.1, 0.1, 0.1])
    lon = np.array([0.1, 0.1, 1.1])
    lat_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0, 2.0])
    out = s5p.bin_to_grid(values, lat, lon, lat_edges, lon_edges, min_count=2)
    assert np.isnan(out["mean"].values[0, 1])  # only 1 pixel in that cell
    assert out["mean"].values[0, 0] == pytest.approx(2.0)


def test_bin_to_grid_ignores_nan_pixels():
    values = np.array([1.0, np.nan, 3.0])
    lat = np.array([0.1, 0.1, 0.1])
    lon = np.array([0.1, 0.1, 0.1])
    out = s5p.bin_to_grid(values, lat, lon, np.array([0.0, 1.0]), np.array([0.0, 1.0]), min_count=1)
    assert out["count"].values[0, 0] == 2
    assert out["mean"].values[0, 0] == pytest.approx(2.0)


def test_oversample_to_grid_matches_bin_to_grid_for_point_pixels():
    # degenerate "corner" pixels (all 4 corners equal, i.e. a point) should behave like binning
    values = np.array([4.0, 8.0])
    lat = np.array([0.5, 0.5])
    lon = np.array([0.5, 1.5])
    lat_bounds = np.column_stack([lat - 0.01, lat - 0.01, lat + 0.01, lat + 0.01])
    lon_bounds = np.column_stack([lon - 0.01, lon + 0.01, lon + 0.01, lon - 0.01])
    lat_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0, 2.0])
    out = s5p.oversample_to_grid(values, lat_bounds, lon_bounds, lat_edges, lon_edges)
    np.testing.assert_allclose(out["mean"].values, [[4.0, 8.0]], atol=1e-6)
    assert out.rio.crs is not None


def test_temporal_composite_bin_method_accumulates_across_swaths():
    lat_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0])
    swath1 = (np.array([1.0, 3.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5]))  # mean 2, count 2
    swath2 = (np.array([10.0]), np.array([0.5]), np.array([0.5]))               # mean 10, count 1
    out = s5p.temporal_composite([swath1, swath2], lat_edges, lon_edges, min_count=1, method="bin")
    expected = (1.0 + 3.0 + 10.0) / 3
    assert out["mean"].values[0, 0] == pytest.approx(expected)
    assert out["count"].values[0, 0] == pytest.approx(3.0)


def test_temporal_composite_respects_min_count():
    lat_edges = np.array([0.0, 1.0])
    lon_edges = np.array([0.0, 1.0])
    swath = (np.array([1.0]), np.array([0.5]), np.array([0.5]))
    out = s5p.temporal_composite([swath], lat_edges, lon_edges, min_count=3, method="bin")
    assert np.isnan(out["mean"].values[0, 0])


def test_temporal_composite_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        s5p.temporal_composite([], np.array([0.0, 1.0]), np.array([0.0, 1.0]), method="nope")


# --------------------------------------------------------------------------- averaging kernel
def test_tm5_pressure_levels_bottom_layer_matches_surface_pressure():
    tm5_a = np.array([[0.0, 100.0], [100.0, 200.0]])
    tm5_b = np.array([[1.0, 0.9], [0.9, 0.8]])
    ps = np.array([100000.0])
    p_lower, p_upper = s5p.tm5_pressure_levels(tm5_a, tm5_b, ps)
    np.testing.assert_allclose(p_lower[..., 0], ps)  # layer 0's lower edge is the surface
    assert np.all(p_lower >= p_upper)


def test_apply_averaging_kernel_with_unit_ak_matches_simple_hydrostatic_column():
    # AK = 1 everywhere should reduce to the plain hydrostatic column of a uniform VMR profile
    n_layers = 5
    tm5_a = np.zeros((n_layers, 2))
    tm5_b = np.linspace(1.0, 0.0, n_layers + 1)
    tm5_b = np.column_stack([tm5_b[:-1], tm5_b[1:]])
    surface_pressure = np.array([101325.0])
    vmr = np.full(n_layers, 2e-9)
    ak = np.ones(n_layers)

    smoothed = s5p.apply_averaging_kernel(ak, vmr, tm5_a, tm5_b, surface_pressure)
    dry_air_column = surface_pressure / (9.80665 * 28.9644e-3)
    expected = 2e-9 * dry_air_column  # uniform VMR * total dry-air column
    np.testing.assert_allclose(smoothed, expected, rtol=1e-6)


def test_apply_averaging_kernel_zero_ak_gives_zero_column():
    n_layers = 4
    tm5_a = np.zeros((n_layers, 2))
    tm5_b = np.column_stack([np.linspace(1, 0.25, n_layers), np.linspace(0.75, 0.0, n_layers)])
    smoothed = s5p.apply_averaging_kernel(np.zeros(n_layers), np.full(n_layers, 1e-9),
                                          tm5_a, tm5_b, np.array([100000.0]))
    np.testing.assert_allclose(smoothed, [0.0])


# --------------------------------------------------------------------------- CH4 / CO
def test_ch4_enhancement_default_background_is_scene_median():
    xch4 = np.array([1900.0, 1900.0, 1950.0, 2000.0, 1910.0])
    enh = s5p.ch4_enhancement(xch4)
    np.testing.assert_allclose(enh, xch4 - 1910.0)


def test_ch4_enhancement_explicit_background():
    enh = s5p.ch4_enhancement(np.array([1900.0, 2000.0]), background=1850.0)
    np.testing.assert_allclose(enh, [50.0, 150.0])


def test_integrated_mass_enhancement_and_emission_rate_are_physically_consistent():
    # one plume pixel, 50 ppb enhancement, known pixel area -> check against manual formula
    enh_ppb = np.array([50.0])
    ps = np.array([101325.0])
    area = np.array([5500.0 * 5500.0])
    ime = s5p.integrated_mass_enhancement(enh_ppb, ps, area)

    dry_air_col = 101325.0 / (9.80665 * 28.9644e-3)
    expected_mass_per_area = 50e-9 * dry_air_col * 16.04e-3
    expected_ime = expected_mass_per_area * area[0]
    assert ime == pytest.approx(expected_ime, rel=1e-6)

    q = s5p.estimate_emission_rate(ime, wind_speed_m_s=5.0, plume_length_m=1000.0)
    assert q == pytest.approx(ime * 5.0 / 1000.0)


def test_co_mixing_ratio_matches_manual_dry_air_column_formula():
    co_column = np.array([0.035])
    ps = np.array([101325.0])
    xco = s5p.co_mixing_ratio(co_column, ps)
    dry_air_col = 101325.0 / (9.80665 * 28.9644e-3)
    expected = (0.035 / dry_air_col) * 1e9
    np.testing.assert_allclose(xco, expected)


# --------------------------------------------------------------------------- consistency checks
def test_check_processing_consistency_flags_version_and_era_change(tmp_path):
    def _write(name, start, version):
        path = tmp_path / name
        xr.Dataset(attrs={
            "time_coverage_start": start, "processor_version": version,
        }).to_netcdf(path, engine="h5netcdf")
        return path

    before = _write("a.nc", "2019-01-01T00:00:00Z", "1.4.0")
    after_same_version = _write("b.nc", "2019-01-02T00:00:00Z", "1.4.0")
    after_pixel_change = _write("c.nc", "2020-01-01T00:00:00Z", "1.4.0")
    new_version = _write("d.nc", "2019-01-03T00:00:00Z", "2.0.0")

    df = s5p.check_processing_consistency([before, after_same_version, after_pixel_change, new_version])
    assert df.loc[0, "flag"] == ""
    assert df.loc[1, "flag"] == ""
    assert "pixel size era" in df.loc[2, "flag"]
    assert "processor_version" in df.loc[3, "flag"]


def test_check_processing_consistency_empty_list_returns_empty_frame():
    df = s5p.check_processing_consistency([])
    assert df.empty
