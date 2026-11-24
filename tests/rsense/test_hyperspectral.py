"""Tests for geolattice.rsense.hyperspectral on synthetic cubes with known endmembers,
abundances, noise, stripes, targets and gas plumes."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import hyperspectral as hs

WL = np.arange(400.0, 2501.0, 10.0)
X0, Y0 = 400_000.0, 4_700_000.0


def _scene(seed=0, H=40, W=50, noise=0.003, stripes=0.0, smooth=True):
    """Mixture of 4 library endmembers (spatially smooth abundances, like real scenes) with
    pure-pixel blocks in the corners."""
    from scipy import ndimage as snd

    rng = np.random.default_rng(seed)
    lib = hs.synthetic_library(WL)
    names = ["green_vegetation", "soil", "water", "clay_kaolinite"]
    E = lib.subset(names).spectra
    if smooth:
        Z = np.stack([snd.gaussian_filter(rng.normal(size=(H, W)), 3) for _ in range(4)], -1)
        Z = np.exp(Z / Z.std() * 1.2)
        A = Z / Z.sum(-1, keepdims=True)
    else:
        A = rng.dirichlet(np.ones(4) * 0.6, size=(H, W))
    for k, (r, c) in enumerate([(0, 0), (0, W - 4), (H - 4, 0), (H - 4, W - 4)]):
        A[r:r + 4, c:c + 4] = np.eye(4)[k]
    cube = np.einsum("yxk,kb->byx", A, E)
    cube += rng.normal(0, noise, cube.shape)
    if stripes:
        cube += rng.normal(0, stripes, (1, 1, W))
    da = xr.DataArray(cube.astype("float32"), dims=("band", "y", "x"),
                      coords={"y": Y0 - 10 * (np.arange(H) + 0.5),
                              "x": X0 + 10 * (np.arange(W) + 0.5)})
    c = hs.as_cube(da, WL, fwhm=np.full(WL.size, 10.0), crs=32618)
    return c, A, E, names


@pytest.fixture(scope="module")
def scene():
    return _scene()


# ----------------------------------------------------------------------------- cube helpers
def test_cube_helpers(scene):
    c, A, E, _ = scene
    assert c.dims == ("band", "y", "x") and c["wavelength"].values[0] == 400
    um = hs.as_cube(np.zeros((3, 2, 2)), [0.45, 0.55, 0.65], units="um")
    assert um["wavelength"].values.tolist() == [450, 550, 650]
    assert float(hs.band_at(c, 663)[0, 0]) == pytest.approx(float(c.isel(band=26)[0, 0]))
    assert hs.band_at(c, 663, width=40).shape == (40, 50)
    with pytest.raises(ValueError):
        hs.band_at(hs.sel_range(c, 400, 800), 2200)
    assert hs.sel_range(c, 1000, 1100).sizes["band"] == 11
    pts = [(X0 + 15, Y0 - 15), (X0 + 495, Y0 - 395)]
    sp = hs.spectra_at(c, pts)
    assert sp.shape == (2, WL.size)
    np.testing.assert_allclose(sp.iloc[0].to_numpy(), E[0], atol=0.02)
    st = hs.band_stats(c)
    assert {"snr", "nan_fraction"} <= set(st.columns) and len(st) == WL.size
    assert hs.mean_spectrum(c).shape == (WL.size,)


# ----------------------------------------------------------------------------- I/O
@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
def test_envi_roundtrip(scene, tmp_path, interleave):
    c = scene[0].assign_coords(good=("band", hs.bad_bands(scene[0]) == 0))
    p = hs.write_envi(c, tmp_path / f"cube_{interleave}.img", interleave=interleave)
    h = hs.read_envi_header(p + ".hdr" if not p.endswith(".hdr") else p)
    assert h["bands"] == WL.size and h["interleave"] == interleave
    back = hs.read_envi(p)
    np.testing.assert_allclose(back.values, c.values, atol=1e-6)
    np.testing.assert_allclose(back["wavelength"], WL)
    np.testing.assert_allclose(back.x, c.x)
    np.testing.assert_allclose(back.y, c.y)
    assert back.rio.crs.to_epsg() == 32618
    assert (back["good"].values == c["good"].values).all()
    sub = hs.read_envi(p, bands=[0, 10, 20], window=(slice(5, 15), slice(2, 12)))
    assert sub.shape == (3, 10, 10)
    np.testing.assert_allclose(sub.values, c.values[[0, 10, 20], 5:15, 2:12], atol=1e-6)
    np.testing.assert_allclose(sub.x, c.x[2:12])
    assert hs.read_envi(p, mask_bad=True).sizes["band"] == int(c["good"].sum())


def test_envi_header_variants(tmp_path):
    a = (np.arange(2 * 3 * 4).reshape(2, 3, 4) * 100).astype(">i2")
    a.tofile(tmp_path / "x.dat")
    (tmp_path / "x.hdr").write_text(
        "ENVI\nsamples = 4\nlines = 3\nbands = 2\nheader offset = 0\ndata type = 2\n"
        "interleave = bsq\nbyte order = 1\nreflectance scale factor = 10000\n"
        "wavelength units = Micrometers\nwavelength = {0.55,\n 0.86}\nfwhm = {0.01, 0.02}\n"
        "data ignore value = 0\n"
        "map info = {UTM, 1.000, 1.000, 500000.0, 4700000.0, 30.0, 30.0, 18, North, WGS-84}\n")
    c = hs.read_envi(tmp_path / "x.dat")
    assert c["wavelength"].values.tolist() == [550, 860] and c["fwhm"].values[1] == 20
    assert np.isnan(c.values[0, 0, 0]) and c.values[1, 2, 3] == pytest.approx(0.23)
    assert c.rio.crs.to_epsg() == 32618 and float(c.x[0]) == 500015.0
    assert hs.read_hyperspectral(tmp_path / "x.hdr").shape == (2, 3, 4)


def test_emit_reader_and_glt(tmp_path):
    import h5py

    rng = np.random.default_rng(0)
    dt, ct, nb = 6, 5, 4
    refl = rng.uniform(0.05, 0.5, (dt, ct, nb)).astype("f4")
    refl[0, 0] = -9999
    gx = np.zeros((7, 8), "i4")
    gy = np.zeros((7, 8), "i4")
    for r in range(dt):
        for col in range(ct):
            gy[r, col + 1], gx[r, col + 1] = r + 1, col + 1
    p = tmp_path / "EMIT_L2A_RFL_001.nc"
    with h5py.File(p, "w") as f:
        f["reflectance"] = refl
        g = f.create_group("sensor_band_parameters")
        g["wavelengths"] = np.array([450.0, 650, 860, 1400])
        g["fwhm"] = np.full(nb, 8.0)
        g["good_wavelengths"] = np.array([1, 1, 1, 0], "u1")
        loc = f.create_group("location")
        loc["glt_x"], loc["glt_y"] = gx, gy
        f.attrs["geotransform"] = np.array([-76.5, 0.0005, 0, 42.5, 0, -0.0005])
        f.attrs["spatial_ref"] = "EPSG:4326"
    c = hs.read_emit(p)
    assert c.shape == (4, 7, 8) and c.rio.crs.to_epsg() == 4326
    assert c["good"].values.tolist() == [True, True, True, False]
    assert np.isnan(c.values[:, 0, 0]).all() and np.isnan(c.values[:, 0, 1]).all()
    np.testing.assert_allclose(c.values[:, 2, 3], refl[2, 2])
    assert float(c.x[0]) == pytest.approx(-76.5 + 0.00025)
    sw = hs.read_emit(p, ortho=False, mask_bad=True)
    assert sw.dims == ("band", "line", "sample") and sw.shape == (3, 6, 5)
    o = hs.glt_orthorectify(sw, gx, gy, [-76.5, 0.0005, 0, 42.5, 0, -0.0005])
    np.testing.assert_allclose(o.values[:, 2, 3], refl[2, 2, :3])
    assert hs.read_hyperspectral(p).shape == (4, 7, 8)


def test_prisma_reader(tmp_path):
    import h5py

    rng = np.random.default_rng(1)
    p = tmp_path / "PRS_L2D_STD_20240101.he5"
    with h5py.File(p, "w") as f:
        d = f.create_group("HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields")
        g = f.create_group("HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields")
        d["VNIR_Cube"] = rng.integers(1, 65535, (5, 3, 4)).astype("u2")    # rows, bands, cols
        d["SWIR_Cube"] = rng.integers(1, 65535, (5, 2, 4)).astype("u2")
        g["Latitude"] = np.zeros((5, 4))
        g["Longitude"] = np.zeros((5, 4))
        f.attrs["List_Cw_Vnir"] = np.array([700.0, 600.0, 0.0])        # descending + unused band
        f.attrs["List_Cw_Swir"] = np.array([1600.0, 2200.0])
        f.attrs["List_Fwhm_Vnir"] = np.array([10.0, 10.0, 0])
        f.attrs["List_Fwhm_Swir"] = np.array([12.0, 12.0])
        f.attrs["L2ScaleVnirMin"], f.attrs["L2ScaleVnirMax"] = 0.0, 1.0
        f.attrs["L2ScaleSwirMin"], f.attrs["L2ScaleSwirMax"] = 0.0, 0.5
        f.attrs["Epsg_Code"] = 32633
        f.attrs["Product_ULcorner_easting"] = 300000.0
        f.attrs["Product_ULcorner_northing"] = 5000000.0
        vn = d["VNIR_Cube"][:]
    c = hs.read_prisma(p)
    wl = c["wavelength"].values
    assert np.all(np.diff(wl[np.isfinite(wl)]) > 0) and c.rio.crs.to_epsg() == 32633
    assert c["good"].values.sum() == 4
    np.testing.assert_allclose(c.isel(band=0).values, vn[:, 1, :] / 65535.0, rtol=1e-5)
    assert float(c.x[0]) == 300015.0
    assert hs.read_prisma(p, mask_bad=True).sizes["band"] == 4


def test_enmap_and_geotiff_readers(tmp_path, scene):
    import rasterio
    from rasterio.transform import from_origin

    c = scene[0].isel(band=[5, 25, 45, 120])
    d = tmp_path / "ENMAP01-____L2A-DT0000001_20240101T000000Z"
    d.mkdir()
    dn = (c.values * 10000).astype("int16")
    T = from_origin(X0, Y0, 10, 10)
    prof = dict(driver="GTiff", width=50, height=40, count=4, dtype="int16", crs=32618,
                transform=T, nodata=-32768)
    with rasterio.open(d / "ENMAP01-L2A-SPECTRAL_IMAGE.TIF", "w", **prof) as dst:
        dst.write(dn)
    bands = "".join(f'<bandID number="{i + 1}"><wavelengthCenterOfBand>{w}</wavelengthCenterOfBand>'
                    f"<FWHMOfBand>8.0</FWHMOfBand><GainOfBand>0.0001</GainOfBand>"
                    f"<OffsetOfBand>0</OffsetOfBand></bandID>"
                    for i, w in enumerate(c["wavelength"].values))
    (d / "ENMAP01-METADATA.XML").write_text(
        f"<level_X><specific><bandCharacterisation>{bands}</bandCharacterisation></specific>"
        "</level_X>")
    e = hs.read_enmap(d)
    np.testing.assert_allclose(e["wavelength"], c["wavelength"])
    np.testing.assert_allclose(e.values, c.values, atol=1e-4)
    assert hs.read_hyperspectral(d).shape == (4, 40, 50)
    t = tmp_path / "cube.tif"
    with rasterio.open(t, "w", **{**prof, "dtype": "float32", "nodata": None}) as dst:
        dst.write(c.values.astype("float32"))
        for i, w in enumerate(c["wavelength"].values, 1):
            dst.set_band_description(i, f"Band {i} ({w:.1f} nm)")
    g = hs.read_geotiff_cube(t)
    np.testing.assert_allclose(g["wavelength"], c["wavelength"])
    out = hs.subset_file(hs.write_envi(c, tmp_path / "big.img"), tmp_path / "s" / "small.img",
                         window=(slice(0, 8), slice(0, 8)), bands=[0, 1])
    assert hs.read_envi(out).shape == (2, 8, 8)


def _tanager_file(path, *, grid=True, nb=5, H=6, W=8, seed=2):
    """Minimal Tanager HDF-EOS5 file (ortho grid or basic swath)."""
    import h5py

    rng = np.random.default_rng(seed)
    a = rng.uniform(0.02, 0.6, (nb, H, W)).astype("f4")
    a[:, 0, 0] = -9999.0
    kind = "GRIDS" if grid else "SWATHS"
    var = "surface_reflectance" if grid else "toa_radiance"
    with h5py.File(path, "w") as f:
        F = f.create_group(f"HDFEOS/{kind}/HYP/Data Fields")
        d = F.create_dataset(var, data=a)
        d.attrs["wavelengths"] = np.linspace(450.0, 2400.0, nb)
        d.attrs["fwhm"] = np.full(nb, 5.5)
        d.attrs["_FillValue"] = -9999.0
        d.attrs["Unit"] = "Unitless" if grid else "W/(m^2 sr um)"
        if grid:
            d.attrs["good_wavelengths"] = np.array([1, 1, 0, 1, 1][:nb], "u1")
            u = F.create_dataset("surface_reflectance_uncertainty", data=a * 0.01)
            u.attrs["_FillValue"] = -9999.0
        nod = np.zeros((H, W), "u1")
        nod[-1, -1] = 1
        cloud = np.zeros((H, W), "u1")
        cloud[2, 3] = 1
        F["nodata_pixels"], F["beta_cloud_mask"] = nod, cloud
        F["beta_cirrus_mask"] = np.zeros((H, W), "u1")
        F["column_water_vapour"] = np.full((H, W), 2.5, "f4")
        F["column_water_vapour"].attrs["_FillValue"] = np.float32(-9999)
        if grid:
            f["HDFEOS INFORMATION/StructMetadata.0"] = np.bytes_(
                f"GROUP=GridStructure\n\tGROUP=GRID_1\n\t\tGridName=\"HYP\"\n\t\tXDim={W}\n"
                f"\t\tYDim={H}\n\t\tUpperLeftPointMtrs=(419490.00,7556490.00)\n"
                f"\t\tLowerRightMtrs=({419490 + 30 * W:.2f},{7556490 - 30 * H:.2f})\n"
                "\t\tProjection=HE5_GCTP_UTM\n\t\tZoneCode=-20\n")
        else:
            G = f.create_group("HDFEOS/SWATHS/HYP/Geolocation Fields")
            lat, lon = np.meshgrid(-22.0 - 0.0003 * np.arange(H), -63.7 + 0.0003 * np.arange(W),
                                   indexing="ij")
            G["Latitude"], G["Longitude"] = lat, lon
    return a


def test_tanager_reader(tmp_path):
    p = tmp_path / "20250224_145149_32_4001_ortho_sr_hdf5.h5"
    a = _tanager_file(p)
    c = hs.read_tanager(p)
    assert c.dims == ("band", "y", "x") and c.shape == (5, 6, 8)
    assert c.rio.crs.to_epsg() == 32720 and c.attrs["sensor"] == "Tanager"
    assert float(c.x[0]) == 419505.0 and float(c.y[0]) == 7556475.0
    assert np.isnan(c.values[:, 0, 0]).all() and np.isnan(c.values[:, -1, -1]).all()
    np.testing.assert_allclose(c.values[:, 3, 4], a[:, 3, 4])
    assert c["good"].values.tolist() == [True, True, False, True, True]
    assert np.isfinite(c.values[:, 2, 3]).all()
    assert np.isnan(hs.read_tanager(p, mask_clouds=True).values[:, 2, 3]).all()
    s = hs.read_tanager(p, wavelength_range=(400, 1500), window=(slice(1, 4), slice(2, 6)))
    assert s.shape == (3, 3, 4) and float(s.x[0]) == 419490 + 30 * 2.5
    np.testing.assert_allclose(s.values, a[:3, 1:4, 2:6])
    assert hs.read_tanager(p, bands=[0, 3], mask_bad=True).sizes["band"] == 2
    assert hs.read_tanager(p, mask_bad=True).sizes["band"] == 4
    unc = hs.read_tanager(p, variable="surface_reflectance_uncertainty")  # no band attrs of its own
    np.testing.assert_allclose(unc["wavelength"], c["wavelength"])
    np.testing.assert_allclose(unc.values[:, 3, 4], a[:, 3, 4] * 0.01, rtol=1e-6)
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(32720, 4326, always_xy=True).transform(
        [419490 + 30 * 2.2, 419490 + 30 * 5.8], [7556490 - 30 * 4.8, 7556490 - 30 * 1.2])
    b = hs.read_tanager(p, bbox=(lon[0], lat[0], lon[1], lat[1]))
    assert 0 < b.sizes["y"] < 6 and 0 < b.sizes["x"] < 8
    anc = hs.read_tanager_ancillary(p, window=(slice(0, 3), slice(None)))
    assert anc["column_water_vapour"].shape == (3, 8) and anc.rio.crs.to_epsg() == 32720
    assert hs.read_hyperspectral(p).shape == (5, 6, 8)


def test_tanager_swath_and_gridding(tmp_path):
    import rasterio

    p = tmp_path / "20250224_145149_32_4001_basic_radiance_hdf5.h5"
    a = _tanager_file(p, grid=False)
    s = hs.read_hyperspectral(p)
    assert s.dims == ("band", "line", "sample") and s.attrs["units"] == "W/(m^2 sr um)"
    assert s["good"].values.tolist() == [True, True, False, False, True]  # 1425, 1912 nm
    assert float(s.lat[1, 0]) == pytest.approx(-22.0003)
    g = hs.swath_to_grid(s, 30.0)
    assert g.dims == ("band", "y", "x") and g.rio.crs.to_epsg() == 32720
    assert np.isfinite(g.values[0]).mean() > 0.5
    v = g.values[1][np.isfinite(g.values[1])]
    assert set(np.round(v, 5)) <= set(np.round(a[1].ravel(), 5))
    geo = tmp_path / "geolocation_array.tif"
    with rasterio.open(geo, "w", driver="GTiff", width=8, height=6, count=2, dtype="float64",
                       nodata=-9999.0) as dst:
        dst.write(np.stack([np.full((6, 8), 10.0), np.full((6, 8), 50.0)]))
    s2 = hs.read_tanager(p, geolocation=geo, window=(slice(0, 2), slice(0, 3)))
    assert s2.shape == (5, 2, 3) and float(s2.lon[0, 0]) == 10.0 and float(s2.lat[0, 0]) == 50.0


def test_ohs_reader(tmp_path):
    import rasterio
    from rasterio.rpc import RPC

    H = W = 40
    z = [0.0] * 20
    rpc = RPC(height_off=500.0, height_scale=100.0, lat_off=39.0, lat_scale=0.002,
              long_off=-5.3, long_scale=0.0025, line_off=20.0, line_scale=20.0,
              samp_off=20.0, samp_scale=20.0, line_num_coeff=[0.0, 0.0, -1.0] + z[3:],
              line_den_coeff=[1.0] + z[1:], samp_num_coeff=[0.0, 1.0] + z[2:],
              samp_den_coeff=[1.0] + z[1:])
    scene = tmp_path / "HAZ2_20180715215912_0002_L1_MSS_CCD1"
    scene.mkdir()
    rng = np.random.default_rng(3)
    data = {}
    for b in (1, 2, 3):
        data[b] = rng.integers(1, 4000, (H, W)).astype("u2")
        data[b][0, 0] = 0
        with rasterio.open(scene / f"HAZ2_20180715215912_0002_L1_MSS_B{b}_CCD1.tif", "w",
                           driver="GTiff", width=W, height=H, count=1, dtype="uint16",
                           rpcs=rpc) as dst:
            dst.write(data[b], 1)
    (scene / "HAZ2_20180715215912_0002_meta.xml").write_text(
        "<MetaData><ProductInfo><SatelliteID>ovs-2</SatelliteID><ImageRowGSD>10</ImageRowGSD>"
        "<PixelInvalidVal>0</PixelInvalidVal></ProductInfo></MetaData>")
    c = hs.read_ohs(scene)
    assert c.dims == ("band", "line", "sample") and c.shape == (3, H, W)
    np.testing.assert_allclose(c["wavelength"], [466.0, 480.0, 500.0])
    assert np.isnan(c.values[:, 0, 0]).all() and c.attrs["SatelliteID"] == "ovs-2"
    np.testing.assert_allclose(c.values[1, 5, 7], data[2][5, 7])
    w = hs.read_ohs(scene / "HAZ2_20180715215912_0002_L1_MSS_B2_CCD1.tif", bands=[3, 1],
                    window=(slice(10, 20), slice(5, 25)), scale=0.01)
    assert w.shape == (2, 10, 20) and w["wavelength"].values.tolist() == [500.0, 466.0]
    np.testing.assert_allclose(w.values[0], data[3][10:20, 5:25] * 0.01, rtol=1e-6)
    assert w.attrs["rpcs"].line_off == 10.0 and float(w.line[0]) == 10
    o = hs.read_ohs(scene, ortho=True, crs="EPSG:4326", resolution=0.0001)
    assert o.dims == ("band", "y", "x") and o.rio.crs.to_epsg() == 4326
    assert float(o.x.min()) == pytest.approx(-5.3025, abs=3e-4)
    assert float(o.y.max()) == pytest.approx(39.002, abs=3e-4)
    assert np.isfinite(o.values[0]).mean() > 0.8
    u = hs.read_hyperspectral(scene, ortho=True, resolution=10.0)
    assert u.rio.crs.to_epsg() == 32630
    with pytest.raises(KeyError):
        hs.read_ohs(scene, bands=[7])


# ----------------------------------------------------------------------------- preprocessing
def test_bad_bands_noise_destripe(scene):
    c = scene[0]
    bad = hs.bad_bands(c)
    wl = c["wavelength"].values
    assert bad[(wl > 1350) & (wl < 1450)].all() and not bad[(wl > 500) & (wl < 1300)].any()
    assert hs.drop_bad_bands(c).sizes["band"] == (~bad).sum()
    n = hs.estimate_noise(c, method="residual")
    assert np.median(n) == pytest.approx(0.003, rel=0.5)
    striped = _scene(stripes=0.02)[0]
    col_bias = lambda x: float(np.std(np.nanmean(x.values, axis=1), axis=1).mean())  # noqa: E731
    assert col_bias(hs.destripe(striped)) < 0.6 * col_bias(striped)
    assert hs.destripe(striped, method="median").shape == striped.shape


def test_quality_mask(scene):
    q = hs.quality_mask(scene[0])
    assert set(q.data_vars) == {"cloud", "water", "shadow", "saturated", "valid"}
    assert int(q.water[38, 2]) == 1 and int(q.water[2, 2]) == 0
    assert int(q.cloud[2, 2]) == 0 and int(q.cloud[38, 2]) == 0 and int(q.valid[2, 2]) == 1


def test_empirical_line_and_flat_field(scene):
    c = scene[0]
    gain, off = 3000.0, 150.0
    dn = c * gain + off
    dark = (slice(36, 40), slice(0, 4))       # water corner
    bright = (slice(0, 4), slice(46, 50))     # soil corner
    E = scene[2]
    cal, table = hs.empirical_line(dn, {"water": (dark, E[2]), "soil": (bright, E[1])})
    assert float(np.abs(cal - c).mean()) < 1e-3 and float(np.abs(cal - c).max()) < 0.03
    assert table.r2.min() > 0.999 and table.gain.iloc[50] == pytest.approx(1 / gain, rel=1e-2)
    field_wl = np.arange(350, 2600, 1.0)
    cal2, _ = hs.empirical_line(dn, {"w": (dark, np.interp(field_wl, WL, E[2])),
                                     "s": (bright, np.interp(field_wl, WL, E[1]))},
                                field_wavelengths=field_wl)
    assert float(np.abs(cal2 - c).mean()) < 1.5e-3
    ff = hs.flat_field(c, bright)
    assert float(ff.isel(band=30)[1:3, 47:49].mean()) == pytest.approx(1, abs=0.02)
    assert hs.iar_reflectance(c).shape == c.shape


def test_spectral_filters(scene):
    c = scene[0]
    noisy = _scene(noise=0.02)[0]
    sm = hs.smooth(noisy, window=9)
    truth = c.values
    assert np.abs(sm.values - truth).mean() < np.abs(noisy.values - truth).mean()
    for m in ("moving", "gaussian"):
        assert hs.smooth(noisy, method=m).shape == c.shape
    lin = hs.as_cube(np.broadcast_to((0.001 * WL)[:, None, None], (WL.size, 2, 2)).copy(), WL)
    np.testing.assert_allclose(hs.derivative(lin).values, 0.001, atol=1e-6)
    np.testing.assert_allclose(hs.derivative(lin, 2).values, 0, atol=1e-8)
    s = hs.snv(c)
    np.testing.assert_allclose(s.mean("band"), 0, atol=1e-5)
    np.testing.assert_allclose(s.std("band"), 1, atol=1e-3)
    v = hs.normalize(c)
    np.testing.assert_allclose(np.sqrt((v**2).sum("band")), 1, atol=1e-4)
    for m in ("brightness", "max", "area"):
        assert np.isfinite(hs.normalize(c, m).values).all()


def test_continuum_removal_and_band_depth(scene):
    c = scene[0]
    cr, hull = hs.continuum_removal(c, start=2000, stop=2400, return_hull=True)
    v = cr.values[:, ~(hs.sel_range(c, 2000, 2400) <= 0).any("band").values]
    assert np.nanmax(v) <= 1 + 1e-6 and np.nanmin(v) > 0
    assert (hull.values >= hs.sel_range(c, 2000, 2400).values - 1e-6).all()
    clay = cr.isel(y=slice(36, 40), x=slice(46, 50)).mean(("y", "x"))
    wl = clay["wavelength"].values
    assert wl[int(np.argmin(clay.values))] == pytest.approx(2205, abs=10)
    full = hs.continuum_removal(c)
    assert np.allclose(full.values[[0, -1]], 1)
    bd = hs.band_depth(c, 2205, 2120, 2280)
    assert set(bd.data_vars) == {"depth", "position", "width", "area", "asymmetry"}
    assert float(bd.position[38, 48]) == pytest.approx(2205, abs=15)
    assert float(bd.depth[38, 48]) > 0.15 > float(bd.depth[2, 2])
    fixed = hs.band_depth(c, 2200, 2120, 2280, feature=False)
    assert float(fixed.position[0, 0]) == 2200


def test_resampling(scene):
    c = scene[0]
    r = hs.resample_spectral(c, [500, 600, 700, 3000], 20)
    assert r.sizes["band"] == 4 and np.isnan(r.isel(band=3)).all()
    np.testing.assert_allclose(r.isel(band=1), c.sel(band=20), atol=0.01)
    li = hs.resample_spectral(c, [505, 615])
    np.testing.assert_allclose(li.isel(band=0), (c.isel(band=10) + c.isel(band=11)) / 2, atol=1e-5)
    s2 = hs.resample_to_sensor(c, "sentinel2")
    assert {"red", "nir08", "swir22"} <= set(s2.data_vars) and s2.attrs["sensor"] == "sentinel2"
    import geolattice.rsense as rs

    ndvi = rs.optical.index(s2, "NDVI")
    assert float(ndvi[2, 2]) > 0.6 > float(ndvi[2, 48])
    l8 = hs.resample_to_sensor(c, "landsat89", bands=["red", "nir08"])
    assert list(l8.data_vars) == ["red", "nir08"]


# ----------------------------------------------------------------------------- reduction
def test_pca_mnf_ica(scene):
    c = scene[0]
    sc, T = hs.pca(c, 6)
    assert sc.dims[0] == "component" and T.explained_ratio[:3].sum() > 0.95
    back = T.inverse_transform(sc, c)
    assert float(np.abs(back - c).mean()) < 0.005
    noisy = _scene(noise=0.02)[0]
    m, Tm = hs.mnf(noisy, 10)
    assert Tm.eigenvalues[0] > Tm.eigenvalues[5] > 0
    den = hs.mnf_denoise(noisy, 4)
    truth = c.values
    assert np.abs(den.values - truth).mean() < 0.5 * np.abs(noisy.values - truth).mean()
    m2, _ = hs.mnf(noisy, 5, noise="residual")
    assert m2.sizes["component"] == 5
    ic, model = hs.ica(c, 4)
    assert ic.sizes["component"] == 4


def test_subspace_estimators(scene):
    noisy = _scene(noise=0.01)[0]
    h = hs.hysime(noisy)
    assert 3 <= h["k"] <= 8
    vd = hs.virtual_dimensionality(noisy)
    assert 2 <= vd <= 10


def test_select_bands(scene):
    c, A, E, _ = scene
    df = hs.spectra_at(c, [(X0 + 5 + 10 * j, Y0 - 5 - 10 * i) for i in range(0, 40, 3)
                           for j in range(0, 50, 3)])
    y = (A[::3, ::3, 0] > 0.5).ravel()
    for method in ("mutual_info", "rf", "anova"):
        sel = hs.select_bands(df, y, n=5, method=method, min_separation_nm=30)
        assert len(sel) == 5 and np.min(np.diff(np.sort(sel.wavelength))) >= 30
    dec = hs.select_bands(c, n=6, method="decorrelate", min_separation_nm=50)
    assert len(dec) == 6
    assert len(hs.select_bands(c, n=4, method="variance")) == 4


# ----------------------------------------------------------------------------- indices
def test_narrowband_indices(scene):
    c = scene[0]
    av = hs.available_indices(c)
    assert "PRI" in av and "NDVI" in av and len(av) == len(hs.list_indices())
    ndvi = hs.index(c, "NDVI")
    assert float(ndvi[2, 2]) > 0.7 and float(ndvi[2, 48]) < 0.4
    ds = hs.index(c, ["NDRE", "WBI", "ClayAlOH", "CAI"])
    assert set(ds.data_vars) == {"NDRE", "WBI", "ClayAlOH", "CAI"}
    assert float(ds.ClayAlOH[38, 48]) > float(ds.ClayAlOH[2, 48])
    assert ndvi.attrs["reference"].startswith("Rouse")
    rep = hs.red_edge_position(c)
    assert float(rep[2, 2]) == pytest.approx(715, abs=12)
    assert float(hs.red_edge_position(c, method="linear")[2, 2]) == pytest.approx(715, abs=15)
    hs.register_index("TEST", (800, 670), lambda a, b: a - b, "test")
    assert float(hs.index(c, "TEST")[2, 2]) > 0.2
    narrow = hs.sel_range(c, 400, 1000)
    assert "CAI" not in hs.available_indices(narrow)


# ----------------------------------------------------------------------------- libraries
def test_libraries(tmp_path, scene):
    lib = hs.synthetic_library()
    assert len(lib) == 6 and "6 spectra" in repr(lib)
    sub = lib.subset("veg")
    assert sub.names == ["green_vegetation", "dry_vegetation"]
    r = lib.resample(scene[0])
    assert r.spectra.shape == (6, WL.size)
    r2 = lib.resample([500, 600], fwhm=10)
    assert r2.spectra.shape == (6, 2)
    m = hs.merge_libraries([lib.subset(["soil"]), lib.subset(["water"])])
    assert m.names == ["soil", "water"]
    lib.to_frame().T.rename_axis("wl").reset_index().to_csv(tmp_path / "lib.csv", index=False)
    from geolattice.rsense.hyperspectral.library import read_csv

    back = read_csv(tmp_path / "lib.csv")
    np.testing.assert_allclose(back.spectra, lib.spectra)
    # USGS splib07-style files
    d = tmp_path / "usgs"
    d.mkdir()
    um = np.arange(0.35, 2.501, 0.01)
    (d / "splib07a_Wavelengths_ASD.txt").write_text("Wavelengths\n" + "\n".join(map(str, um)))
    for name, s in [("Kaolinite_CM9", 0.5 + 0 * um), ("Calcite_WS272", 0.6 + 0.01 * um)]:
        vals = s.copy()
        vals[3] = -1.23e34
        (d / f"splib07a_{name}_ASDFRa_AREF.txt").write_text(
            f"splib07a Record=1: {name}\n" + "\n".join(f"{v:.6g}" for v in vals))
    u = hs.read_usgs(d)
    assert len(u) == 2 and u.wavelengths[0] == pytest.approx(350) and np.isnan(u.spectra[0, 3])
    # ECOSTRESS-style file
    e = tmp_path / "eco.txt"
    e.write_text("Name: Oak leaf\nClass: vegetation\nX Units: Wavelength (micrometers)\n"
                 "Y Units: Reflectance (percent)\n\n" +
                 "\n".join(f"{w:.3f}\t{40.0}" for w in np.arange(0.4, 2.5, 0.01)))
    ec = hs.read_ecostress(e)
    assert ec.names == ["Oak leaf"] and np.nanmax(ec.spectra) == pytest.approx(0.4)
    assert ec.metadata["class"].iloc[0] == "vegetation"
    import geopandas as gpd

    pts = gpd.GeoDataFrame({"cls": ["veg", "veg", "soil"]},
                           geometry=gpd.points_from_xy([X0 + 15, X0 + 25, X0 + 485],
                                                       [Y0 - 15, Y0 - 25, Y0 - 15]), crs=32618)
    fi = hs.from_image(scene[0], pts, "cls")
    assert fi.names == ["soil", "veg"]


# ----------------------------------------------------------------------------- matching
def test_spectral_matching(scene):
    c, A, E, names = scene
    lib = hs.synthetic_library().subset(names)
    s = hs.sam(c, lib)
    assert s.dims[0] == "target" and list(s.target.values) == lib.names
    out = hs.classify_spectral(c, lib)
    pure = {(1, 1): 0, (1, 48): 1, (38, 1): 2, (38, 48): 3}
    for (r, cc), k in pure.items():
        assert int(out["class"][r, cc]) == lib.names.index(names[k])
    t = hs.classify_spectral(c, lib, threshold=0.02)
    assert (t["class"].values == -1).mean() > 0.3
    for m in ("sid", "sca", "euclidean", "sidsam"):
        o = hs.classify_spectral(c, lib, measure=m)
        assert int(o["class"][1, 1]) == lib.names.index("green_vegetation")
    assert hs.sid(c, lib).name == "sid"


# ----------------------------------------------------------------------------- endmembers, unmixing
def _rmse_to_truth(found, E):
    """For each true endmember, RMSE to the closest extracted spectrum (SAM is unreliable
    for the near-black water spectrum)."""
    return np.sqrt(((E[:, None, :] - found[None]) ** 2).mean(-1)).min(1)


@pytest.mark.parametrize("method", ["nfindr", "vca", "atgp", "ppi"])
def test_endmember_extraction(scene, method):
    c, A, E, _ = scene
    res = hs.extract_endmembers(c, 4, method=method)
    assert res["spectra"].shape == (4, WL.size) and len(res["row"]) == 4
    tol = 0.01 if method in ("nfindr", "vca", "atgp") else 0.05
    assert _rmse_to_truth(res["spectra"], E).max() < tol
    if method == "ppi":
        assert res["counts"].shape == (40, 50)


def test_unmixing_methods(scene):
    c, A, E, names = scene
    lib = hs.synthetic_library().subset(names)
    lib = hs.SpectralLibrary(lib.wavelengths, lib.spectra, names)
    f = hs.unmix(c, lib)
    est = np.stack([f[n].values for n in names], -1)
    assert np.abs(est - A).mean() < 0.01
    np.testing.assert_allclose(est.sum(-1), 1, atol=1e-4)
    assert est.min() >= 0 and float(f.rmse.mean()) < 0.005
    for m in ("ls", "scls", "nnls"):
        g = hs.unmix(c, lib, method=m)
        e2 = np.stack([g[n].values for n in names], -1)
        assert np.abs(e2 - A).mean() < 0.02
    em = hs.extract_endmembers(c, 4, method="nfindr")
    assert set(hs.unmix(c, em).data_vars) >= {"nfindr1", "rmse"}
    rec = hs.reconstruct(f, lib, c)
    assert float(np.abs(rec - c).mean()) < 0.005
    from geolattice.rsense.hyperspectral.unmixing import project_simplex

    P = project_simplex(np.array([[0.5, 2.0, -1.0], [0.8, -1.0, 0.2], [0.1, 0.0, 0.3]]))
    np.testing.assert_allclose(P.sum(0), 1) and np.testing.assert_array_less(-1e-12, P)


def test_mesma_and_sparse(scene):
    c, A, E, names = scene
    lib = hs.synthetic_library()
    # two-class mixtures of vegetation and soil, with brightness variation
    rng = np.random.default_rng(3)
    f = rng.uniform(0, 1, (20, 20))
    bright = rng.uniform(0.8, 1.0, (20, 20))
    veg, soil = lib.subset(["green_vegetation"]).spectra[0], lib.subset(["soil"]).spectra[0]
    data = (f[None] * veg[:, None, None] + (1 - f[None]) * soil[:, None, None]) * bright[None]
    cube = hs.as_cube(data, lib.wavelengths)
    mlib = {"veg": [veg, veg * 0.9 + 0.01], "soil": [soil], "water": [lib.to_dict()["water"]]}
    out = hs.mesma(cube, mlib, max_classes=2, max_rmse=0.01)
    assert np.nanmean(np.abs(out["veg"].values - f)) < 0.05
    assert np.nanmean(out["water"].values) < 0.02
    assert float(np.nanmean(out.shade)) == pytest.approx(1 - bright.mean(), abs=0.05)
    assert (out.model.values >= 0).all() and len(out.attrs["models"]) > 3
    sp = hs.sparse_unmix(c, hs.synthetic_library(WL), lam=1e-4, top=4)
    assert sp.sizes["y"] == 40 and float(sp.rmse.mean()) < 0.02
    assert set(sp.data_vars) == set(names) | {"rmse", "n_active"}      # top-4 = true members
    full = hs.sparse_unmix(c, hs.synthetic_library(WL), lam=1e-4)
    assert float(full["dry_vegetation"].mean()) < 0.03
    assert float(full["carbonate_calcite"].mean()) < 0.03


# ----------------------------------------------------------------------------- detection
def test_target_detection(scene):
    c, A, E, names = scene
    rng = np.random.default_rng(5)
    target = hs.synthetic_library(WL).to_dict()["carbonate_calcite"]
    truth = np.zeros((40, 50), bool)
    truth[rng.integers(5, 35, 20), rng.integers(5, 45, 20)] = True
    clean = np.einsum("yxk,kb->byx", A, E)
    v = c.values.copy()                     # keep the sensor noise of every pixel
    v[:, truth] += 0.4 * (target[:, None] - clean[:, truth])
    ct = c.copy(data=v)
    for fn in (hs.matched_filter, hs.ace, hs.cem):
        s = fn(ct, target)
        roc = hs.detection_roc(s, truth)
        assert roc.attrs["auc"] > 0.97, fn.__name__
    mf = hs.matched_filter(ct, target)
    assert float(mf.values[truth].mean()) == pytest.approx(0.4, abs=0.15)
    o = hs.osp(ct, target, E)
    assert hs.detection_roc(o, truth).attrs["auc"] > 0.97
    r = hs.rx(ct)
    assert hs.detection_roc(r, truth).attrs["auc"] > 0.8
    rl = hs.rx(ct, mode="local", window=11, guard=3, n_components=6)
    assert rl.shape == (40, 50) and hs.detection_roc(rl, truth).attrs["auc"] > 0.75
    bg = ~truth
    assert np.isfinite(hs.matched_filter(ct, target, background=bg).values).all()


def test_gas_matched_filter_and_ime():
    rng = np.random.default_rng(0)
    wl = np.arange(2100.0, 2501.0, 5.0)
    k = -1e-5 * np.exp(-0.5 * ((wl - 2300) / 15) ** 2) \
        - 8e-6 * np.exp(-0.5 * ((wl - 2360) / 10) ** 2)
    H, W = 200, 40
    base = 5 + 3 * np.exp(-((wl - 2200) / 300) ** 2)
    alb = rng.uniform(0.7, 1.3, (H, W))
    rad = base[:, None, None] * alb[None] * (1 + rng.normal(0, 0.003, (wl.size, H, W)))
    yy, xx = np.mgrid[0:H, 0:W]
    plume = 1500 * np.exp(-(((yy - 100) / 5) ** 2 + ((xx - 20) / 4) ** 2))
    rad = rad * np.exp(k[:, None, None] * plume[None])
    da = xr.DataArray(rad, dims=("band", "y", "x"),
                      coords={"y": Y0 - 60 * (np.arange(H) + 0.5),
                              "x": X0 + 60 * (np.arange(W) + 0.5)})
    cube = hs.as_cube(da, wl, crs=32618)
    enh = hs.gas_matched_filter(cube, k, columnwise=False)
    assert enh.attrs["units"] == "ppm m"
    assert float(enh[100, 20]) == pytest.approx(1500, rel=0.25)
    assert abs(float(np.median(enh.values[plume < 1]))) < 100
    col = hs.gas_matched_filter(cube, k, columnwise=True, min_pixels=20)
    assert float(col[100, 20]) > 600
    labels, table = hs.plume_mask(enh, min_pixels=5)
    assert len(table) >= 1 and table.attrs["threshold"] > 0
    big = int(table.sort_values("n_pixels").plume.iloc[-1])
    ime = hs.plume_ime(enh, labels, big)
    expected = plume[labels.values == big].sum() * 1e-6 * 0.6785 * 3600
    assert ime["ime_kg"] == pytest.approx(expected, rel=0.2)
    assert ime["n_pixels"] > 40
    assert ime["pixel_area_m2"] == pytest.approx(3600)
    q = hs.emission_rate(ime["ime_kg"], 3.0, ime["length_m"])
    assert q > 0 and hs.emission_rate(10, 2, 100, per="s") == pytest.approx(0.2)


# ----------------------------------------------------------------------------- traits
def test_plsr_and_regressors(scene):
    c, A, E, names = scene
    rng = np.random.default_rng(0)
    idx = rng.choice(40 * 50, 300, replace=False)
    r, cc = np.unravel_index(idx, (40, 50))
    spectra = pd.DataFrame(c.values[:, r, cc].T, columns=WL)
    y = 3 * A[r, cc, 0] + 1.0 * A[r, cc, 3] + rng.normal(0, 0.02, len(idx))
    m = hs.fit_plsr(spectra, y, target="trait", max_components=10)
    assert m.metrics["r2"] > 0.95 and 2 <= m.metrics["n_components"] <= 10
    assert len(m.extra["vip"]) == WL.size and m.extra["press"].rmse_cv.min() < 0.1
    mp = m.predict_map(c)
    truth = 3 * A[..., 0] + A[..., 3]
    assert np.corrcoef(mp.values.ravel(), truth.ravel())[0, 1] > 0.97
    df = spectra.assign(trait=y)
    m2 = hs.fit_plsr(df, target="trait", max_components=6)
    assert m2.metrics["r2"] > 0.9
    rf = hs.fit_regressor(spectra, y, model="rf", n_estimators=50)
    assert rf.metrics["r2"] > 0.8
    gpr = hs.fit_regressor(spectra.iloc[:120, ::4], y[:120], model="gpr")
    out = gpr.predict_map(c.isel(band=slice(None, None, 4)), return_std=True)
    assert set(out.data_vars) == {"trait", "trait_std"} and float(out.trait_std.mean()) > 0
    for mod in ("ridge", "svr", "gbm"):
        assert hs.fit_regressor(spectra, y, model=mod).metrics["r2"] > 0.5
    sub = m.predict_map(hs.resample_spectral(c, WL[::2]))
    assert sub.shape == (40, 50)


def test_prosail_hybrid():
    pytest.importorskip("prosail")
    s = hs.prosail_spectrum(lai=4, cab=50)
    assert s.shape == (2101,) and s[400] > 0.3 > s[270]
    s2 = hs.prosail_spectrum(wavelengths_nm=[670, 860], lai=1)
    assert s2[0] < s2[1]
    P, S, wl = hs.prosail_lut(40, wavelengths_nm=WL, fwhm=np.full(WL.size, 10.0), noise=0.01)
    assert S.shape == (40, WL.size) and {"lai", "cab"} <= set(P.columns)
    rng = np.random.default_rng(2)
    lai = rng.uniform(0.5, 6, (6, 6))
    cube = np.stack([hs.prosail_spectrum(wavelengths_nm=WL, lai=v, cab=45) for v in lai.ravel()])
    cube = hs.as_cube(cube.T.reshape(WL.size, 6, 6), WL, fwhm=np.full(WL.size, 10.0))
    res = hs.hybrid_retrieval(cube, ["lai"], n=300, model="rf", fixed={"cab": 45.0},
                              n_train=300)
    est = res["maps"]["lai"].values
    assert np.corrcoef(est.ravel(), lai.ravel())[0, 1] > 0.85


# ----------------------------------------------------------------------------- plots
def test_plots(scene):
    import matplotlib

    matplotlib.use("Agg")
    c = scene[0]
    im = hs.rgb(c, "color_infrared")
    assert im.shape == (3, 40, 50) and float(im.max()) <= 1
    hs.plot_rgb(c, (640, 550, 460))
    hs.plot_spectra(hs.spectra_at(c, [(X0 + 15, Y0 - 15)]), bad_ranges=[(1340, 1455)])
    hs.plot_spectra(hs.synthetic_library())
    hs.plot_spectra(c.values[:, 0, :3].T, wavelengths_nm=WL, labels=["a", "b", "c"])
    hs.plot_band_stats(hs.band_stats(c))
    pytest.importorskip("plotly")
    fig = hs.spectral_viewer(c, [(X0 + 15, Y0 - 15)])
    assert len(fig.data) == 2
    assert len(hs.wavelength_slider(c, step=50).frames) == 5
