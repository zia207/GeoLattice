"""Tests for geolattice.spatial_analysis.terrain_analysis."""

from pathlib import Path

import numpy as np
import pytest

from geolattice.raster.raster_read_write import (
    driver_available,
    open_raster_data,
    write_geotiff,
)
from geolattice.spatial_analysis.terrain_analysis import (
    TerrainAnalysisError,
    aspect,
    contours,
    curvature,
    gaussian_curvature,
    hillshade,
    landform_classification,
    mean_curvature,
    multiscale_analysis,
    plan_curvature,
    profile_curvature,
    roughness,
    slope,
    tangential_curvature,
    tpi,
    tri,
    twi,
    viewshed,
)

pytestmark = pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff driver missing")

# Projected grid so slope / wetness cell sizes are meaningful (metres).
_GT = (500000.0, 10.0, 0.0, 4800000.0, 0.0, -10.0)
_CRS = 32617


def _hill_dem(rows: int = 40, cols: int = 40) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    dem = 100.0 + 35.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 8.0**2))
    return dem.astype(np.float32)


def _write_dem(path: Path, array=None, *, nodata=-9999.0) -> Path:
    write_geotiff(
        path,
        _hill_dem() if array is None else array,
        crs=_CRS,
        geotransform=_GT,
        nodata=nodata,
        overwrite=True,
    )
    return path


def _read(path: Path) -> np.ndarray:
    with open_raster_data(path) as dataset:
        return np.asarray(dataset.read(), dtype=np.float64)


def test_slope_degrees_and_percent(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")

    deg_path = tmp_path / "slope_deg.tif"
    pct_path = tmp_path / "slope_pct.tif"
    assert slope(dem, deg_path, units="degrees", overwrite=True) == str(deg_path)
    assert slope(dem, pct_path, units="percent", overwrite=True) == str(pct_path)

    deg = _read(deg_path)
    pct = _read(pct_path)
    valid = (deg != -9999) & (pct != -9999) & np.isfinite(deg) & np.isfinite(pct)
    assert valid.any()
    assert float(np.nanmax(deg[valid])) > 0
    assert float(np.nanmax(deg[valid])) < 90
    # percent slope should be larger than degrees for the same moderate slopes
    assert float(np.nanmean(pct[valid])) > float(np.nanmean(deg[valid]))

    with pytest.raises(TerrainAnalysisError):
        slope(dem, tmp_path / "bad.tif", units="radians", overwrite=True)


def test_aspect_hillshade_tri_tpi_roughness(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")

    asp = tmp_path / "aspect.tif"
    aspect(dem, asp, overwrite=True)
    aspect_vals = _read(asp)
    valid = aspect_vals != -9999
    assert valid.any()
    assert float(np.nanmin(aspect_vals[valid])) >= 0
    assert float(np.nanmax(aspect_vals[valid])) <= 360

    hs = tmp_path / "hillshade.tif"
    hillshade(dem, hs, azimuth=315, altitude=45, overwrite=True)
    hs_vals = _read(hs)
    assert hs_vals.dtype == np.float64 or hs_vals.dtype == np.uint8 or np.issubdtype(
        hs_vals.dtype, np.number
    )
    assert float(np.nanmax(hs_vals)) > float(np.nanmin(hs_vals))

    md = tmp_path / "hillshade_md.tif"
    hillshade(dem, md, multidirectional=True, overwrite=True)
    assert md.is_file()

    for fn, name in (
        (tri, "tri.tif"),
        (tpi, "tpi.tif"),
        (roughness, "rough.tif"),
    ):
        out = tmp_path / name
        fn(dem, out, overwrite=True)
        vals = _read(out)
        finite = vals != -9999
        assert finite.any()
        assert float(np.nanmax(vals[finite])) >= float(np.nanmin(vals[finite]))

    tpi_wide = tmp_path / "tpi11.tif"
    tpi(dem, tpi_wide, window_size=11, overwrite=True)
    assert tpi_wide.is_file()

    with pytest.raises(TerrainAnalysisError):
        hillshade(dem, tmp_path / "bad_hs.tif", combined=True, multidirectional=True, overwrite=True)


def test_curvature_suite(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    kinds = {
        "mean": mean_curvature,
        "tangential": tangential_curvature,
        "profile": profile_curvature,
        "plan": plan_curvature,
        "gaussian": gaussian_curvature,
    }
    for kind, fn in kinds.items():
        out = tmp_path / f"curv_{kind}.tif"
        assert fn(dem, out, overwrite=True) == str(out)
        vals = _read(out)
        finite = vals != -9999
        assert finite.any(), kind
        assert np.isfinite(vals[finite]).all(), kind

    generic = tmp_path / "curv_generic.tif"
    curvature(dem, generic, curvature_type="profile", overwrite=True)
    assert generic.is_file()

    with pytest.raises(TerrainAnalysisError):
        curvature(dem, tmp_path / "bad_curv.tif", curvature_type="foobar", overwrite=True)


def test_twi_landform_multiscale(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")

    twi_path = tmp_path / "twi.tif"
    twi(dem, twi_path, overwrite=True)
    twi_vals = _read(twi_path)
    finite = twi_vals != -9999
    assert finite.any()
    assert float(np.nanmin(twi_vals[finite])) > -50
    assert float(np.nanmax(twi_vals[finite])) < 50

    lf = landform_classification(
        dem,
        tmp_path / "landform.tif",
        small_window=3,
        large_window=15,
        overwrite=True,
    )
    assert Path(lf["path"]).is_file()
    assert set(lf["classes"].keys()) == {1, 2, 3, 4, 5, 6, 7}
    classes = _read(Path(lf["path"]))
    present = {int(v) for v in np.unique(classes) if v != -9999}
    assert present.issubset(set(lf["classes"]))
    assert len(present) >= 2

    ms = multiscale_analysis(
        dem,
        tmp_path / "ms_tpi.tif",
        metric="tpi",
        min_scale=3,
        max_scale=9,
        step=2,
        overwrite=True,
    )
    assert Path(ms["value"]).is_file()
    assert Path(ms["scale"]).is_file()
    scale_vals = _read(Path(ms["scale"]))
    scale_finite = scale_vals != -9999
    assert set(np.unique(scale_vals[scale_finite]).astype(int)).issubset({3, 5, 7, 9})

    stacked = multiscale_analysis(
        dem,
        tmp_path / "ms_stack.tif",
        metric="roughness",
        min_scale=3,
        max_scale=7,
        step=2,
        output="stack",
        overwrite=True,
    )
    with open_raster_data(stacked["stack"]) as dataset:
        assert dataset.band_count == 3


def test_viewshed_and_contours(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    # Centre of the hill in map coordinates
    x = _GT[0] + 20 * _GT[1] + 0.5 * _GT[1]
    y = _GT[3] + 20 * _GT[5] + 0.5 * _GT[5]

    vs_path = tmp_path / "viewshed.tif"
    viewshed(dem, vs_path, observer=(x, y), observer_height=1.8, overwrite=True)
    vis = _read(vs_path)
    assert set(np.unique(vis)).issubset({0, 1})
    assert int(vis.sum()) > 1
    assert int(vis[20, 20]) == 1

    # Multiple observers
    viewshed(
        dem,
        tmp_path / "viewshed2.tif",
        observer=[(x, y), (x + 50, y + 50)],
        overwrite=True,
    )

    ct_path = tmp_path / "contours.gpkg"
    contours(dem, ct_path, interval=5.0, overwrite=True)
    assert ct_path.is_file()

    with pytest.raises(TerrainAnalysisError):
        viewshed(dem, tmp_path / "bad_vs.tif", observer=(0.0, 0.0), overwrite=True)


def test_package_exports():
    from geolattice import spatial_analysis as raster_mod

    for name in (
        "slope",
        "aspect",
        "hillshade",
        "tri",
        "tpi",
        "twi",
        "roughness",
        "curvature",
        "mean_curvature",
        "tangential_curvature",
        "profile_curvature",
        "plan_curvature",
        "gaussian_curvature",
        "landform_classification",
        "multiscale_analysis",
        "viewshed",
        "contours",
        "TerrainAnalysisError",
    ):
        assert hasattr(raster_mod, name), name
