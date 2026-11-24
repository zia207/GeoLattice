"""Round-trip raster files through write_raster_data and read_raster_data."""

import numpy as np
import pytest

from geolattice.raster import (
    open_raster_data,
    read_raster_data,
    write_geotiff,
    write_png,
    write_raster_data,
)
from geolattice.raster.raster_read_write import driver_available


pytestmark = pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff driver missing")


def test_read_raster_data_matches_open_raster_data(tmp_path):
    path = tmp_path / "grid.tif"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_geotiff(path, array, crs=4326, geotransform=(0, 1, 0, 3, 0, -1), nodata=-9999)
    with read_raster_data(path) as dataset:
        written = dataset.shape
    with open_raster_data(path) as dataset:
        opened = dataset.shape
    assert written == opened == (3, 4)


def test_round_trip_geotiff_and_png(tmp_path):
    array = np.stack(
        [
            np.arange(12, dtype=np.uint8).reshape(3, 4),
            np.arange(12, 24, dtype=np.uint8).reshape(3, 4),
        ],
        axis=0,
    )
    tiff = tmp_path / "rgb.tif"
    write_geotiff(tiff, array, crs="EPSG:4326")
    with read_raster_data(tiff) as dataset:
        assert dataset.band_count == 2
        assert dataset.driver == "GTiff"
        assert dataset.crs is not None
        saved = dataset.read()
    assert saved.shape == (2, 3, 4)
    np.testing.assert_array_equal(saved, array)

    png = tmp_path / "gray.png"
    write_png(png, array[0])
    with read_raster_data(png) as dataset:
        assert dataset.band_count == 1
        gray = dataset.read()
    np.testing.assert_array_equal(gray, array[0])


def test_write_raster_data_overwrite_guard(tmp_path):
    path = tmp_path / "one.tif"
    write_raster_data(path, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(FileExistsError):
        write_raster_data(path, np.zeros((2, 2), dtype=np.float32))
    write_raster_data(path, np.zeros((2, 2), dtype=np.float32), overwrite=True)
    with read_raster_data(path) as dataset:
        assert dataset.read().max() == 0.0
