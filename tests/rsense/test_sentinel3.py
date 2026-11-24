import numpy as np
import pytest
import xarray as xr

pytest.importorskip("h5netcdf")  # geolattice[rsense-sentinel3]

from geolattice.rsense.optical import INDICES, guess_sensor  # noqa: E402
from geolattice.rsense.optical.sensors import sentinel3 as s3  # noqa: E402


def _flag_da(raw, masks, meanings, dims="n"):
    da = xr.DataArray(np.asarray(raw, dtype="int32"), dims=dims)
    da.attrs["flag_masks"] = np.asarray(masks, dtype="int32")
    da.attrs["flag_meanings"] = meanings
    return da


def test_decode_flags_reads_cf_attributes():
    flag = _flag_da([0, 1, 2, 3], [1, 2], "cloud_in confidence_in")
    decoded = s3.decode_flags(flag)
    assert set(decoded) == {"cloud_in", "confidence_in"}
    assert decoded["cloud_in"].values.tolist() == [False, True, False, True]
    assert decoded["confidence_in"].values.tolist() == [False, False, True, True]


def test_decode_flags_requires_cf_attrs():
    with pytest.raises(ValueError, match="flag_masks"):
        s3.decode_flags(xr.DataArray(np.array([1, 2]), dims="n"))


def test_flag_mask_exact_and_contains():
    flag = _flag_da([0, 1, 2, 3], [1, 2], "LAND CLOUD")
    assert s3.flag_mask(flag, ["LAND"]).values.tolist() == [True, False, True, False]
    assert s3.flag_mask(flag, ["land"], match="contains").values.tolist() == [True, False, True, False]
    # SLSTR_L1_DEFAULT_EXCLUDE = ("cloud", "summary_cloud"); "cloud" matches "CLOUD" (bit 1, mask=2)
    assert s3.slstr_flag_mask(flag).values.tolist() == [True, True, False, False]


def test_olci_water_mask_excludes_default_flags():
    wqsf = _flag_da([0, 1, 1 << 7, 1 << 1],
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                    "LAND CLOUD CLOUD_AMBIGUOUS CLOUD_MARGIN INVALID SNOW_ICE SUSPECT "
                    "HISOLZEN SATURATED HIGHGLINT WHITECAPS")
    valid = s3.olci_water_mask(wqsf)
    assert valid.values.tolist() == [True, False, False, False]


def test_olci_toa_reflectance_matches_formula():
    radiance = xr.DataArray(np.array([[50.0, 100.0]]), dims=("y", "x"), name="oa08")
    sza = xr.DataArray(np.array([[30.0, 60.0]]), dims=("y", "x"))
    rho = s3.olci_toa_reflectance(radiance, 1500.0, sza)
    expected = np.pi * radiance.values / (1500.0 * np.cos(np.radians(sza.values)))
    np.testing.assert_allclose(rho.values, expected, rtol=1e-6)
    assert rho.attrs["units"] == "reflectance"


def test_log_to_linear_back_transforms():
    log_chl = xr.DataArray(np.array([-1.0, 0.0, 1.0]), dims="x")
    np.testing.assert_allclose(s3.log_to_linear(log_chl).values, [0.1, 1.0, 10.0])
    # the back-transform must not be stored under the "transform" attribute name: rioxarray
    # reserves that key for a GDAL-style affine geotransform and crashes write_crs() on it.
    assert "transform" not in s3.log_to_linear(log_chl).attrs
    assert s3.log_to_linear(log_chl).attrs["back_transform"] == "10**x"


def test_log_to_linear_then_swath_to_grid_does_not_crash():
    # Regression test: olci_water_mask -> log_to_linear -> swath_to_grid is the documented
    # chlorophyll workflow; log_to_linear's extra attribute must not break write_crs() in
    # swath_to_grid's final step.
    rng = np.random.default_rng(1)
    n = 200
    lon = xr.DataArray(rng.uniform(10.0, 11.0, n), dims="points")
    lat = xr.DataArray(rng.uniform(45.0, 46.0, n), dims="points")
    log_chl = xr.DataArray(rng.uniform(-1.0, 1.0, n), dims="points", name="chl_nn")
    linear = s3.log_to_linear(log_chl)
    gridded = s3.swath_to_grid(linear, lon, lat, resolution=0.05, crs=4326)
    assert gridded.rio.crs is not None


def test_kelvin_to_celsius():
    k = xr.DataArray(np.array([273.15, 300.0]), dims="x")
    np.testing.assert_allclose(s3.kelvin_to_celsius(k).values, [0.0, 26.85])


def test_sst_bias_correct_and_quality_mask():
    sst = xr.Dataset({
        "sea_surface_temperature": ("n", np.array([290.0, 291.0, 292.0])),
        "sses_bias": ("n", np.array([0.2, -0.1, 0.0])),
        "quality_level": ("n", np.array([5, 3, 4])),
    })
    corrected = s3.sst_bias_correct(sst)
    np.testing.assert_allclose(corrected.values, [289.8, 291.1, 292.0])
    assert s3.sst_quality_mask(sst, min_level=4).values.tolist() == [True, False, True]


def test_sral_marine_mask_editing_criteria():
    ds = xr.Dataset({
        "ssha_01_ku": ("n", np.array([0.1, 0.1, 3.0, 0.1])),
        "surf_type_01": ("n", np.array([0, 1, 0, 0])),
        "rain_flag_01_ku": ("n", np.array([0, 0, 0, 1])),
    })
    valid = s3.sral_marine_mask(ds)
    assert valid.values.tolist() == [True, False, False, False]


def test_frp_to_points_and_aggregate():
    gpd = pytest.importorskip("geopandas")
    frp = xr.Dataset({
        "latitude": ("n", np.array([10.0, 11.0, 12.0])),
        "longitude": ("n", np.array([20.0, 21.0, 22.0])),
        "FRP_MWIR": ("n", np.array([5.0, 50.0, 500.0])),
        "FRP_MWIR_confidence": ("n", np.array([10, 60, 90])),
    })
    points = s3.frp_to_points(frp, min_confidence=50)
    assert isinstance(points, gpd.GeoDataFrame)
    assert len(points) == 2
    agg = s3.aggregate_frp(points, resolution=1.0)
    assert float(agg.sum()) == pytest.approx(550.0)


def test_swath_to_grid_dataarray_and_dataset():
    rng = np.random.default_rng(0)
    n = 500
    lon = xr.DataArray(rng.uniform(10.0, 11.0, n), dims="points")
    lat = xr.DataArray(rng.uniform(45.0, 46.0, n), dims="points")
    val = xr.DataArray((lon.values - 10.0) * 100, dims="points")
    gridded = s3.swath_to_grid(val, lon, lat, resolution=0.05, crs=4326)
    assert gridded.rio.crs is not None
    west = float(gridded.isel(x=slice(0, 3)).mean())
    east = float(gridded.isel(x=slice(-3, None)).mean())
    assert east > west  # the west-to-east gradient survives regridding

    ds = xr.Dataset({"a": val, "b": val * 2})
    out = s3.swath_to_grid(ds, lon, lat, resolution=0.05, crs=4326)
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"a", "b"}


def test_mci_and_otci_registered_with_correct_formulas():
    assert "MCI" in INDICES and "OTCI" in INDICES
    ds = xr.Dataset({
        "oa10": (("y", "x"), np.full((2, 2), 0.10)),
        "oa11": (("y", "x"), np.full((2, 2), 0.14)),
        "oa12": (("y", "x"), np.full((2, 2), 0.12)),
    })
    mci = ds.rs.index("MCI")
    otci = ds.rs.index("OTCI")
    expected_mci = 0.14 - 0.10 - (0.12 - 0.10) * (708.75 - 681.25) / (753.75 - 681.25)
    expected_otci = (0.12 - 0.14) / (0.14 - 0.10)
    assert float(mci.mean()) == pytest.approx(expected_mci)
    assert float(otci.mean()) == pytest.approx(expected_otci)


def test_guess_sensor_detects_olci_bands():
    ds = xr.Dataset({"oa08": (("y", "x"), np.zeros((2, 2))), "oa17": (("y", "x"), np.zeros((2, 2)))})
    assert guess_sensor(ds) == "sentinel3_olci"


def test_read_manifest(tmp_path):
    manifest = """<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:s3="http://www.esa.int/safe/sentinel/sentinel-3/1.0">
 <metadataSection>
  <metadataObject>
   <metadataWrap>
    <xmlData>
     <s3:generalProductInformation>
      <s3:processingBaseline>OL_L2M__003.01.00</s3:processingBaseline>
      <s3:productName>S3A_OL_2_WFR____TEST</s3:productName>
     </s3:generalProductInformation>
     <s3:sentinel3ProductInformation>
      <s3:productType>OL_2_WFR___</s3:productType>
     </s3:sentinel3ProductInformation>
    </xmlData>
   </metadataWrap>
  </metadataObject>
 </metadataSection>
</xfdu:XFDU>
"""
    (tmp_path / "xfdumanifest.xml").write_text(manifest)
    meta = s3.read_manifest(tmp_path)
    assert meta["processingBaseline"] == "OL_L2M__003.01.00"
    assert meta["productType"] == "OL_2_WFR___"


def test_read_olci_l1_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    rows, cols = 12, 16
    tie_rows, tie_cols = 12, 4

    for b in ["Oa08", "Oa10"]:
        xr.Dataset({f"{b}_radiance": (("rows", "columns"),
                                      rng.uniform(20, 100, (rows, cols)).astype("float32"))}
                  ).to_netcdf(tmp_path / f"{b}_radiance.nc", engine="h5netcdf")
    xr.Dataset({"solar_flux": (("bands", "detectors"),
                               rng.uniform(1400, 1900, (21, 4)).astype("float32"))}
              ).to_netcdf(tmp_path / "instrument_data.nc", engine="h5netcdf")
    xr.Dataset({"SZA": (("tie_rows", "tie_columns"),
                        rng.uniform(20, 40, (tie_rows, tie_cols)).astype("float32"))}
              ).to_netcdf(tmp_path / "tie_geometries.nc", engine="h5netcdf")
    qf = _flag_da(rng.integers(0, 2**4, (rows, cols)).astype("int32"), [1, 2, 4, 8],
                 "invalid saturated bright dubious", dims=("rows", "columns"))
    xr.Dataset({"quality_flags": qf}).to_netcdf(tmp_path / "qualityFlags.nc", engine="h5netcdf")

    ds = s3.read_olci_l1(tmp_path, bands=["oa08", "oa10"])
    assert ds["oa08"].shape == (rows, cols)
    assert ds["sza"].shape == (rows, cols)

    refl = s3.olci_toa_reflectance(ds["oa08"], 1850.0, ds["sza"])
    assert refl.shape == (rows, cols)
    assert bool((refl > 0).all())
    assert s3.olci_l1_mask(ds["quality_flags"]).shape == (rows, cols)


def test_read_olci_water_end_to_end(tmp_path):
    # Real ESA chl_nn.nc holds a variable literally named "CHL_NN" (plus "CHL_NN_err"), not
    # "chl_nn" -- confirmed against a real Planetary Computer OLCI WFR download. read_olci_water
    # must rename it to the documented lowercase canonical name.
    rng = np.random.default_rng(2)
    rows, cols = 10, 10
    xr.Dataset({
        "CHL_NN": (("rows", "columns"), rng.uniform(-1, 1, (rows, cols)).astype("float32")),
        "CHL_NN_err": (("rows", "columns"), rng.uniform(0, 1, (rows, cols)).astype("float32")),
    }).to_netcdf(tmp_path / "chl_nn.nc", engine="h5netcdf")
    wqsf = _flag_da(rng.integers(0, 2**8, (rows, cols)).astype("int32"), [1, 2, 4, 8],
                    "LAND CLOUD INVALID SUSPECT", dims=("rows", "columns"))
    xr.Dataset({"WQSF": wqsf}).to_netcdf(tmp_path / "wqsf.nc", engine="h5netcdf")
    xr.Dataset({
        "longitude": (("rows", "columns"), np.linspace(10, 11, cols)[None, :].repeat(rows, 0)),
        "latitude": (("rows", "columns"), np.linspace(45, 46, rows)[:, None].repeat(cols, 1)),
    }).to_netcdf(tmp_path / "geo_coordinates.nc", engine="h5netcdf")

    ds = s3.read_olci_water(tmp_path, variables=["chl_nn"])
    assert "chl_nn" in ds.data_vars and "chl_nn_err" in ds.data_vars and "wqsf" in ds.data_vars
    assert "CHL_NN" not in ds.data_vars
    mask = s3.olci_water_mask(ds["wqsf"])
    assert mask.shape == (rows, cols)


def test_read_olci_land_end_to_end(tmp_path):
    # Real ESA rc_gifapar.nc holds two variables, "RC681" and "RC865", not one "RC_GIFAPAR" --
    # confirmed against a real Planetary Computer OLCI LFR download.
    rng = np.random.default_rng(4)
    rows, cols = 8, 8
    xr.Dataset({"OTCI": (("rows", "columns"), rng.uniform(0, 1, (rows, cols)).astype("float32"))}
              ).to_netcdf(tmp_path / "otci.nc", engine="h5netcdf")
    xr.Dataset({
        "RC681": (("rows", "columns"), rng.uniform(0, 1, (rows, cols)).astype("float32")),
        "RC865": (("rows", "columns"), rng.uniform(0, 1, (rows, cols)).astype("float32")),
    }).to_netcdf(tmp_path / "rc_gifapar.nc", engine="h5netcdf")
    lqsf = _flag_da(rng.integers(0, 2**4, (rows, cols)).astype("int32"), [1, 2, 4, 8],
                    "CLOUD LAND WATER INVALID", dims=("rows", "columns"))
    xr.Dataset({"LQSF": lqsf}).to_netcdf(tmp_path / "lqsf.nc", engine="h5netcdf")
    xr.Dataset({
        "longitude": (("rows", "columns"), np.linspace(10, 11, cols)[None, :].repeat(rows, 0)),
        "latitude": (("rows", "columns"), np.linspace(45, 46, rows)[:, None].repeat(cols, 1)),
    }).to_netcdf(tmp_path / "geo_coordinates.nc", engine="h5netcdf")

    ds = s3.read_olci_land(tmp_path, variables=["otci", "rc_gifapar"])
    assert {"otci", "rc681", "rc865", "lqsf"} <= set(ds.data_vars)
    assert "RC681" not in ds.data_vars and "OTCI" not in ds.data_vars
    mask = s3.olci_land_mask(ds["lqsf"])
    assert mask.shape == (rows, cols)


def test_read_synergy_vgt_uses_core_read_netcdf(tmp_path):
    rng = np.random.default_rng(3)
    y, x = np.linspace(46, 45, 8), np.linspace(10, 11, 10)
    xr.Dataset({"NDVI": (("lat", "lon"), rng.uniform(0, 1, (8, 10)).astype("float32"))},
              coords={"lat": y, "lon": x}).to_netcdf(tmp_path / "vgt.nc", engine="h5netcdf")
    out = s3.read_synergy_vgt(tmp_path / "vgt.nc", "NDVI")
    assert out.rio.crs is not None
    assert out.shape == (8, 10)
