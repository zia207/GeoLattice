import numpy as np
import pytest

import geolattice.rsense as rs
from geolattice.rsense.core import PointCloud, RasterCube, mosaic, utm_epsg


def test_utm_epsg():
    assert utm_epsg(-76.5, 42.4) == 32618       # Ithaca
    assert utm_epsg(90.4, 23.8) == 32646        # Dhaka
    assert utm_epsg(151.2, -33.9) == 32756      # Sydney
    assert utm_epsg(5.3, 60.4) == 32632         # Bergen exception


def test_estimate_utm(single_band):
    assert rs.core.estimate_utm_crs(single_band) == 32618


def test_rastercube(reflectance, tmp_path):
    cube = RasterCube(reflectance, sensor="sentinel2")
    assert cube.shape == (4, 40, 50) and "red" in cube.bands
    assert len(cube.sel_period("2026-06-05", "2026-06-25").times) == 2
    monthly = cube.resample("1MS", "median")
    assert len(monthly.times) == 2
    assert float(cube.valid_fraction().min()) == 1.0
    cube.to_zarr(tmp_path / "c.zarr")
    back = RasterCube.from_zarr(tmp_path / "c.zarr")
    np.testing.assert_allclose(back["red"].values, reflectance["red"].values)
    assert back.crs.to_epsg() == 32618
    assert "RasterCube" in repr(cube)


def test_rastercube_from_files(reflectance, tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"d{i}.tif"
        rs.write(reflectance.isel(time=i)[["red", "nir"]].to_array("band"), p)
        paths.append(p)
    cube = RasterCube.from_files(paths, ["2026-01-01", "2026-02-01"], ["red", "nir"])
    assert cube.shape[0] == 2
    np.testing.assert_allclose(cube["nir"].isel(time=1), reflectance.nir.isel(time=1), rtol=1e-6)


def test_pointcloud_basics():
    rng = np.random.default_rng(0)
    pc = PointCloud.from_arrays(rng.uniform(0, 10, 1000), rng.uniform(0, 10, 1000),
                                rng.uniform(0, 5, 1000), crs=32618,
                                classification=rng.choice([1, 2, 7], 1000),
                                return_number=rng.choice([1, 2], 1000))
    assert len(pc) == 1000 and "classification" in pc
    assert len(pc.ground()) == int((pc["classification"] == 2).sum())
    assert len(pc.drop_noise()) == int((pc["classification"] != 7).sum())
    assert len(pc.crop(0, 0, 5, 5)) < 1000
    assert len(pc.voxel_downsample(2.0)) <= 5 * 5 * 3
    assert pc.to_dataframe().shape == (1000, 5)
    assert pc.to_geodataframe().crs.to_epsg() == 32618
    with pytest.raises(ValueError):
        PointCloud(np.zeros(3), np.zeros(2), np.zeros(3))


def test_mosaic(single_band):
    left = single_band.isel(x=slice(0, 50))
    right = single_band.isel(x=slice(30, 80)) + 1
    m = mosaic([left, right], "first")
    assert m.sizes["x"] == 80
    np.testing.assert_allclose(m.isel(x=slice(0, 50)).squeeze(), left, rtol=1e-6)
    mm = mosaic([left, right], "mean")
    overlap = mm.squeeze().isel(x=slice(30, 50)).values
    np.testing.assert_allclose(overlap, single_band.isel(x=slice(30, 50)).values + 0.5, rtol=1e-5, atol=1e-6)


def test_netcdf_roundtrip(tmp_path, single_band):
    p = tmp_path / "x.nc"
    single_band.to_dataset(name="v").to_netcdf(p, engine="scipy")
    back = rs.core.read_netcdf(p, "v", crs=32618)
    assert back.rio.crs.to_epsg() == 32618


def test_read_safe_s1(tmp_path):
    safe = tmp_path / "S1A_IW_GRDH_1SDV_20260101T000000_20260101T000025_000000_000000_0000.SAFE"
    (safe / "measurement").mkdir(parents=True)
    (safe / "annotation" / "calibration").mkdir(parents=True)
    for pol in ("vv", "vh"):
        (safe / "measurement" / f"s1a-iw-grd-{pol}-20260101t000000-x-001.tiff").touch()
        (safe / "annotation" / "calibration" / f"calibration-s1a-iw-grd-{pol}-2026-x-001.xml").touch()
        (safe / "annotation" / "calibration" / f"noise-s1a-iw-grd-{pol}-2026-x-001.xml").touch()
    idx = rs.core.read_safe(safe)
    assert set(idx["measurement"]) == {"vv", "vh"} and set(idx["calibration"]) == {"vv", "vh"}
