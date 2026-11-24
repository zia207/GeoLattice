"""Tests for geolattice.raster.basic_raster_operation."""

from pathlib import Path

import numpy as np
import pytest

from geolattice.raster.basic_raster_operation import (
    RasterOperationError,
    aggregate,
    blend,
    calc,
    class_intervals,
    clip,
    color_map,
    compare,
    contour,
    convert,
    create,
    disaggregate,
    edit,
    fill_nodata,
    focal_statistics,
    mosaic,
    neighbors,
    polygonize,
    proximity,
    raster_histogram,
    rasterize,
    reclassify,
    reclassify_by,
    resample,
    resize,
    stack,
    summarize_raster,
    unscale,
    zonal_stats,
)
from geolattice.raster.raster_read_write import (
    driver_available,
    open_raster_data,
    write_geotiff,
)

pytestmark = pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff driver missing")

_GT = (-80.0, 0.1, 0.0, 40.0, 0.0, -0.1)


def _write_grid(path: Path, array: np.ndarray, *, nodata=None) -> Path:
    write_geotiff(
        path,
        array,
        crs=4326,
        geotransform=_GT,
        nodata=nodata,
        overwrite=True,
    )
    return path


def test_create_edit_convert_resize_unscale(tmp_path: Path):
    path = tmp_path / "empty.tif"
    create(
        path,
        width=20,
        height=15,
        bands=1,
        dtype="float32",
        crs=4326,
        geotransform=_GT,
        nodata=-9999,
        fill=1.0,
        overwrite=True,
    )
    with open_raster_data(path) as dataset:
        assert dataset.width == 20
        assert dataset.height == 15
        assert dataset.crs is not None
        assert dataset.crs.to_epsg() == 4326

    edit(
        path,
        nodata=-9999,
        scale=2.0,
        offset=1.0,
        metadata={"TITLE": "unit-test"},
        overwrite=True,
    )
    with open_raster_data(path) as dataset:
        assert dataset.metadata.get("TITLE") == "unit-test"
        assert dataset.get_band(1).nodata == -9999.0

    unscaled = tmp_path / "unscaled.tif"
    unscale(path, unscaled, overwrite=True)
    assert unscaled.is_file()

    converted = tmp_path / "converted.tif"
    convert(path, converted, dtype="float64", overwrite=True)
    with open_raster_data(converted) as dataset:
        assert dataset.get_band(1).dtype == "float64"

    resized = tmp_path / "resized.tif"
    resize(path, resized, width=10, height=8, overwrite=True)
    with open_raster_data(resized) as dataset:
        assert dataset.width == 10
        assert dataset.height == 8


def test_fill_nodata_neighbors_calc(tmp_path: Path):
    src = tmp_path / "holes.tif"
    array = np.arange(15 * 20, dtype=np.float32).reshape(15, 20)
    array[5:8, 5:8] = -9999
    _write_grid(src, array, nodata=-9999)

    filled = tmp_path / "filled.tif"
    fill_nodata(src, filled, max_distance=5, overwrite=True)
    with open_raster_data(filled) as dataset:
        values = dataset.read()
        assert not np.any(values[5:8, 5:8] == -9999)

    focal = tmp_path / "focal.tif"
    neighbors(filled, focal, statistic="mean", size=3, overwrite=True)
    assert focal.is_file()
    focal_statistics(filled, tmp_path / "focal_max.tif", statistic="max", size=3, overwrite=True)

    out = tmp_path / "calc.tif"
    calc("A * 2 + 1", out, source=filled, overwrite=True)
    with open_raster_data(filled) as a_ds, open_raster_data(out) as b_ds:
        a = a_ds.read().astype(np.float64)
        b = b_ds.read().astype(np.float64)
        mask = a != -9999
        np.testing.assert_allclose(b[mask], a[mask] * 2 + 1, rtol=1e-5)


def test_blend_compare_clip_reclassify(tmp_path: Path):
    left = _write_grid(tmp_path / "left.tif", np.full((15, 20), 10.0, dtype=np.float32))
    right = _write_grid(tmp_path / "right.tif", np.full((15, 20), 0.0, dtype=np.float32))

    blended = tmp_path / "blend.tif"
    blend(left, right, blended, opacity=0.5, overwrite=True)
    with open_raster_data(blended) as dataset:
        np.testing.assert_allclose(dataset.read(), 5.0, rtol=1e-5)

    summary = compare(left, right)
    assert summary["shape_match"] is True
    assert summary["equal"] is False
    assert summary["max_abs_diff"] == 10.0

    clipped = tmp_path / "clip.tif"
    clip(left, clipped, bounds=(-79.5, 39.0, -78.5, 39.8), overwrite=True)
    with open_raster_data(clipped) as dataset:
        assert dataset.width < 20
        assert dataset.height < 15

    reclassed = tmp_path / "reclass.tif"
    values = np.arange(15 * 20, dtype=np.float32).reshape(15, 20)
    src = _write_grid(tmp_path / "classes.tif", values)
    reclassify(src, reclassed, [(0, 50, 1), (50, 200, 2), (200, 10000, 3)], overwrite=True)
    with open_raster_data(reclassed) as dataset:
        out = dataset.read()
        assert set(np.unique(out)).issubset({1.0, 2.0, 3.0})


def test_histogram_and_reclassify_by_styles(tmp_path: Path):
    rng = np.random.default_rng(42)
    values = rng.normal(100, 25, size=(40, 50)).astype(np.float32)
    src = _write_grid(tmp_path / "elev.tif", values)

    hist = raster_histogram(src, bins=8)
    assert len(hist["breaks"]) == 9
    assert len(hist["counts"]) == 8
    assert hist["n"] == values.size
    assert sum(hist["counts"]) == values.size

    fixed = class_intervals(src, style="fixed", fixed=[0, 80, 100, 120, 300])
    assert fixed[0] == 0.0
    assert fixed[-1] == 300.0

    for style in ("equal", "quantile", "pretty", "sd", "kmeans", "hclust", "bclust"):
        breaks = class_intervals(src, n=5, style=style, sample=2000, seed=1)
        assert len(breaks) >= 2
        assert breaks[0] <= breaks[-1]

    # fisher / jenks (mapclassify when available, else fallback)
    for style in ("fisher", "jenks"):
        breaks = class_intervals(src, n=4, style=style, sample=1500, seed=2)
        assert len(breaks) >= 2

    out = tmp_path / "qclass.tif"
    reclassify_by(src, out, n=5, style="quantile", overwrite=True, seed=3)
    with open_raster_data(out) as dataset:
        classes = set(np.unique(dataset.read()))
        assert classes.issubset({1.0, 2.0, 3.0, 4.0, 5.0})
        assert len(classes) >= 2

    with pytest.raises(RasterOperationError):
        class_intervals(src, style="not-a-style")


def test_resample_aggregate_disaggregate_mosaic_stack(tmp_path: Path):
    src = _write_grid(
        tmp_path / "src.tif",
        np.arange(15 * 20, dtype=np.float32).reshape(15, 20),
    )
    resampled = tmp_path / "resampled.tif"
    resample(src, resampled, resolution=0.2, method="bilinear", overwrite=True)
    with open_raster_data(resampled) as dataset:
        assert abs(dataset.geotransform[1]) == pytest.approx(0.2)

    aggregated = tmp_path / "agg.tif"
    aggregate(src, aggregated, factor=2, method="average", overwrite=True)
    with open_raster_data(aggregated) as dataset:
        assert dataset.width == 10
        assert dataset.height == 8

    disaggregated = tmp_path / "disagg.tif"
    disaggregate(aggregated, disaggregated, factor=2, method="near", overwrite=True)
    with open_raster_data(disaggregated) as dataset:
        assert dataset.width == 20
        assert dataset.height == 16

    other = _write_grid(tmp_path / "other.tif", np.ones((15, 20), dtype=np.float32))
    mosaic_path = tmp_path / "mosaic.tif"
    mosaic([src, other], mosaic_path, overwrite=True)
    assert mosaic_path.is_file()

    stacked = tmp_path / "stack.tif"
    stack([src, other], stacked, overwrite=True)
    with open_raster_data(stacked) as dataset:
        assert dataset.band_count == 2


def test_summarize_proximity_polygonize_contour_colormap_zonal(tmp_path: Path):
    dem = _write_grid(
        tmp_path / "dem.tif",
        (np.mgrid[0:15, 0:20][0] + np.mgrid[0:15, 0:20][1]).astype(np.float32),
    )
    stats = summarize_raster(dem)
    assert stats["count"] == 300
    assert stats["min"] == 0.0
    assert stats["max"] == 33.0

    targets = np.zeros((15, 20), dtype=np.float32)
    targets[7, 10] = 1
    prox_src = _write_grid(tmp_path / "targets.tif", targets)
    prox = tmp_path / "prox.tif"
    proximity(prox_src, prox, target_values=[1], overwrite=True)
    with open_raster_data(prox) as dataset:
        assert float(np.min(dataset.read())) == pytest.approx(0.0)

    cats = np.ones((15, 20), dtype=np.float32)
    cats[0:5, :] = 1
    cats[5:10, :] = 2
    cats[10:, :] = 3
    poly_src = _write_grid(tmp_path / "cats.tif", cats)
    polygons = tmp_path / "poly.gpkg"
    polygonize(poly_src, polygons, overwrite=True)
    assert polygons.is_file()

    contours = tmp_path / "contours.gpkg"
    contour(dem, contours, interval=5, overwrite=True)
    assert contours.is_file()

    cmap = tmp_path / "colors.txt"
    cmap.write_text("0 0 0 0\n10 255 0 0\n20 0 255 0\n30 0 0 255\nnv 0 0 0 0\n")
    rgb = tmp_path / "rgb.tif"
    color_map(dem, rgb, cmap, overwrite=True)
    with open_raster_data(rgb) as dataset:
        assert dataset.band_count >= 3

    zones = tmp_path / "zones.geojson"
    zones.write_text(
        """{"type":"FeatureCollection","features":[
 {"type":"Feature","properties":{"zid":1},"geometry":{"type":"Polygon","coordinates":[[[-80,40],[-78.5,40],[-78.5,39],[-80,39],[-80,40]]]}},
 {"type":"Feature","properties":{"zid":2},"geometry":{"type":"Polygon","coordinates":[[[-78.5,40],[-78,40],[-78,39],[-78.5,39],[-78.5,40]]]}}
]}"""
    )
    by_field = zonal_stats(dem, zones, zone_field="zid", stats=("count", "mean", "min", "max"))
    assert len(by_field) == 2
    assert {row["zid"] for row in by_field} == {1, 2}
    assert all(row["count"] > 0 for row in by_field)

    by_fid = zonal_stats(dem, zones, stats=("count", "mean"))
    assert len(by_fid) == 2
    assert "fid" in by_fid[0]

    burned = tmp_path / "rasterized.tif"
    rasterize(
        zones,
        burned,
        attribute="zid",
        template=dem,
        nodata=0,
        overwrite=True,
    )
    with open_raster_data(burned) as dataset:
        # template supplies resolution; extent comes from the vector layer
        assert dataset.width == 20
        assert dataset.height == 10

    from osgeo import gdal

    from geolattice.raster.basic_raster_operation import _read_band_array

    gds = gdal.Open(str(burned))
    burned_arr = _read_band_array(gds.GetRasterBand(1), gds.RasterXSize, gds.RasterYSize)
    assert {1.0, 2.0}.issubset(set(np.unique(burned_arr).tolist()))
    gds = None


def test_invalid_inputs_raise(tmp_path: Path):
    src = _write_grid(tmp_path / "src.tif", np.ones((4, 4), dtype=np.float32))
    with pytest.raises(RasterOperationError):
        create(tmp_path / "bad.tif", width=0, height=4)
    with pytest.raises(RasterOperationError):
        neighbors(src, tmp_path / "n.tif", size=2, overwrite=True)
    with pytest.raises(RasterOperationError):
        clip(src, tmp_path / "c.tif", overwrite=True)
    with pytest.raises(RasterOperationError):
        aggregate(src, tmp_path / "a.tif", factor=1, overwrite=True)
    with pytest.raises(RasterOperationError):
        calc("A * 2", tmp_path / "x.tif")
    with pytest.raises(FileExistsError):
        create(src, width=4, height=4, overwrite=False)
