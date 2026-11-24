"""Tests for vector_CRS and raster_CRS."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from geolattice.raster.raster_CRS import (
    RasterCRSError,
    define_crs as define_raster_crs,
    get_raster_crs,
    reproject_raster,
    validate_raster_crs,
    warp,
    warp_options,
)
from geolattice.raster.raster_read_write import driver_available, write_geotiff
from geolattice.vector.vector_CRS import (
    VectorCRSError,
    crs_equal,
    define_crs,
    is_geographic,
    reproject_geometry,
    reproject_vector,
    validate_crs,
)


def test_define_and_validate_vector_crs():
    crs = define_crs(4326)
    assert crs.to_epsg() == 4326
    assert is_geographic(crs)
    assert crs_equal(crs, "EPSG:4326")
    assert validate_crs("EPSG:3857").to_epsg() == 3857
    with pytest.raises(VectorCRSError):
        define_crs(None)


def test_reproject_geometry_and_vector():
    point = Point(-73.9857, 40.7484)
    projected = reproject_geometry(point, 4326, 32618)
    assert projected.geom_type == "Point"
    assert abs(projected.x - 585000) < 50000  # rough UTM zone 18N

    geoms = reproject_vector([point], 32618, source_crs=4326)
    assert len(geoms) == 1
    assert geoms[0].geom_type == "Point"

    try:
        import geopandas as gpd
    except ImportError:
        return
    frame = gpd.GeoDataFrame({"name": ["empire"]}, geometry=[point], crs="EPSG:4326")
    out = reproject_vector(frame, 32618)
    assert out.crs.to_epsg() == 32618
    assert out.geometry.iloc[0].geom_type == "Point"


@pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff missing")
def test_raster_crs_validate_and_reproject(tmp_path: Path):
    src = tmp_path / "grid.tif"
    array = np.arange(100, dtype=np.float32).reshape(10, 10)
    write_geotiff(
        src,
        array,
        crs=4326,
        geotransform=(-74.0, 0.01, 0.0, 41.0, 0.0, -0.01),
        overwrite=True,
    )
    crs = get_raster_crs(src)
    assert crs is not None and crs.to_epsg() == 4326
    assert validate_raster_crs(src).to_epsg() == 4326
    assert define_raster_crs(3857).to_epsg() == 3857

    options = warp_options(dstSRS=3857, resampleAlg="bilinear", format="GTiff")
    assert options is not None

    out = tmp_path / "grid_3857.tif"
    path = reproject_raster(src, out, dst_crs=3857, resample="bilinear", overwrite=True)
    assert Path(path).is_file()
    warped_crs = get_raster_crs(path)
    assert warped_crs is not None and warped_crs.to_epsg() == 3857

    again = warp(src, tmp_path / "grid_warp.tif", dst_crs=3857, overwrite=True)
    assert Path(again).is_file()


@pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff missing")
def test_raster_crs_missing_raises(tmp_path: Path):
    src = tmp_path / "nocrs.tif"
    write_geotiff(src, np.ones((4, 4), dtype=np.float32), overwrite=True)
    # Default write still may set a unit geotransform but CRS can be absent.
    crs = get_raster_crs(src)
    if crs is None:
        with pytest.raises(RasterCRSError):
            validate_raster_crs(src, require=True)
