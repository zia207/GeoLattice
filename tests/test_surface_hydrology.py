"""Tests for geolattice.spatial_analysis.surface_hydrology."""

from pathlib import Path

import numpy as np
import pytest

from geolattice.raster.raster_read_write import (
    driver_available,
    open_raster_data,
    write_geotiff,
)
from geolattice.spatial_analysis.surface_hydrology import (
    SurfaceHydrologyError,
    basins,
    breach_depressions,
    catchment_delineation,
    d8_flow_accumulation,
    d8_pointer,
    dinf_flow_accumulation,
    dinf_pointer,
    downslope_distance_to_stream,
    elevation_above_stream,
    extract_streams,
    fd8_pointer,
    fill_depressions,
    hydrologic_connectivity,
    mdinf_pointer,
    minimal_dispersion_pointer,
    quinn_pointer,
    rho8_pointer,
    snap_pour_points,
    stream_segmentation,
    subbasins,
    watershed,
    wetness_index,
)

pytestmark = pytest.mark.skipif(not driver_available("GTiff"), reason="GTiff driver missing")

_GT = (500000.0, 10.0, 0.0, 4800000.0, 0.0, -10.0)
_CRS = 32617


def _sloped_dem(rows: int = 40, cols: int = 40) -> np.ndarray:
    """Monotonic slope with a small interior pit."""
    dem = np.tile(np.arange(cols, dtype=np.float32), (rows, 1)) * 2.0
    dem += np.arange(rows, dtype=np.float32)[:, None] * 1.5
    dem[15:18, 15:18] = dem[15, 15] - 8.0
    return dem


def _write_dem(path: Path, array=None) -> Path:
    write_geotiff(
        path,
        _sloped_dem() if array is None else array,
        crs=_CRS,
        geotransform=_GT,
        nodata=-9999.0,
        overwrite=True,
    )
    return path


def _read(path: Path, *, band: int = 1) -> np.ndarray:
    with open_raster_data(path) as dataset:
        return np.asarray(dataset.read(bands=band), dtype=np.float64)


def _centre_xy() -> tuple[float, float]:
    x = _GT[0] + 20 * _GT[1] + 0.5 * _GT[1]
    y = _GT[3] + 20 * _GT[5] + 0.5 * _GT[5]
    return x, y


def test_dem_preprocessing(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    filled = tmp_path / "filled.tif"
    breached = tmp_path / "breached.tif"
    fill_depressions(dem, filled, overwrite=True)
    breach_depressions(dem, breached, fill_pits=True, overwrite=True)

    orig = _read(dem)
    fill_arr = _read(filled)
    assert fill_arr.shape == orig.shape
    # Pit should be raised
    assert float(np.nanmax(fill_arr[15:18, 15:18])) >= float(np.nanmax(orig[15:18, 15:18]))


def test_flow_pointers_and_accumulation(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    filled = tmp_path / "filled.tif"
    fill_depressions(dem, filled, overwrite=True)

    d8p = tmp_path / "d8p.tif"
    d8_pointer(filled, d8p, overwrite=True)
    pntr = _read(d8p)
    valid = (pntr != 0) & (pntr != -9999) & np.isfinite(pntr)
    assert valid.any()
    assert set(np.unique(pntr[valid]).astype(int)).issubset({1, 2, 4, 8, 16, 32, 64, 128})

    fa = tmp_path / "fa.tif"
    d8_flow_accumulation(filled, fa, out_type="cells", overwrite=True)
    fa_vals = _read(fa)
    assert float(np.nanmax(fa_vals)) >= float(np.nanmax(_read(filled) * 0 + 1))

    sca = tmp_path / "sca.tif"
    dinf_flow_accumulation(filled, sca, out_type="sca", overwrite=True)
    sca_vals = _read(sca)
    assert sca_vals.shape == fa_vals.shape
    assert float(np.nanmax(sca_vals)) > 0

    dinf_pointer(filled, tmp_path / "dinf.tif", overwrite=True)
    rho8_pointer(filled, tmp_path / "rho8.tif", seed=42, overwrite=True)
    fd8_pointer(filled, tmp_path / "fd8.tif", overwrite=True)
    mdinf_pointer(filled, tmp_path / "mdinf.tif", overwrite=True)
    quinn_pointer(filled, tmp_path / "quinn.tif", overwrite=True)
    minimal_dispersion_pointer(filled, tmp_path / "mdisp.tif", overwrite=True)

    with open_raster_data(tmp_path / "fd8.tif") as dataset:
        assert dataset.band_count == 8


def test_streams_watersheds_and_indices(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    filled = tmp_path / "filled.tif"
    fill_depressions(dem, filled, overwrite=True)
    d8p = tmp_path / "d8p.tif"
    d8_pointer(filled, d8p, overwrite=True)
    fa = tmp_path / "fa.tif"
    d8_flow_accumulation(filled, fa, overwrite=True)
    streams = tmp_path / "streams.tif"
    extract_streams(fa, streams, threshold=15, overwrite=True)
    st = _read(streams)
    assert int(np.nansum(st > 0)) >= 1

    links = tmp_path / "links.tif"
    stream_segmentation(streams, d8p, links, overwrite=True)
    catch = tmp_path / "catch.tif"
    catchment_delineation(streams, d8p, catch, overwrite=True)
    subbasins(d8p, streams, tmp_path / "sub.tif", overwrite=True)
    basins(d8p, tmp_path / "basins.tif", overwrite=True)

    x, y = _centre_xy()
    snapped = snap_pour_points([(x, y)], streams, snap_distance=500.0)
    assert len(snapped) == 1
    ws = tmp_path / "ws.tif"
    watershed(d8p, snapped, ws, overwrite=True)
    ws_vals = _read(ws)
    assert int(np.nansum(np.isfinite(ws_vals) & (ws_vals > 0))) >= 1

    wetness_index(filled, tmp_path / "twi.tif", method="dinf", overwrite=True)
    elevation_above_stream(filled, streams, d8p, tmp_path / "eas.tif", overwrite=True)
    downslope_distance_to_stream(streams, d8p, tmp_path / "dds.tif", overwrite=True)
    hydrologic_connectivity(filled, streams, tmp_path / "conn.tif", overwrite=True)


def test_errors(tmp_path: Path):
    dem = _write_dem(tmp_path / "dem.tif")
    filled = tmp_path / "filled.tif"
    fill_depressions(dem, filled, overwrite=True)
    fa = tmp_path / "fa.tif"
    d8_flow_accumulation(filled, fa, overwrite=True)

    with pytest.raises(SurfaceHydrologyError):
        extract_streams(fa, tmp_path / "s.tif", threshold=0, overwrite=True)

    with pytest.raises(SurfaceHydrologyError):
        d8_flow_accumulation(filled, tmp_path / "bad_fa.tif", out_type="invalid", overwrite=True)


def test_package_exports():
    from geolattice import spatial_analysis as raster_mod

    for name in (
        "SurfaceHydrologyError",
        "breach_depressions",
        "fill_depressions",
        "d8_pointer",
        "d8_flow_accumulation",
        "dinf_flow_accumulation",
        "extract_streams",
        "watershed",
        "wetness_index",
        "elevation_above_stream",
        "downslope_distance_to_stream",
        "hydrologic_connectivity",
        "impoundment_analysis",
    ):
        assert hasattr(raster_mod, name), name


def _data_interior(valid: np.ndarray) -> np.ndarray:
    """Valid cells whose eight neighbours are all valid and inside the grid."""
    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    inner = valid.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            inner &= padded[1 + dr : 1 + dr + valid.shape[0], 1 + dc : 1 + dc + valid.shape[1]]
    return inner


def test_fill_depressions_nodata_bounded_dem_drains_everywhere(tmp_path: Path):
    """A DEM surrounded by nodata, with a pit and a large flat, must drain fully.

    Regression test for two defects: the flood was seeded from the raster edge
    only (so a nodata-bounded DEM was not filled), and the flat gradient was
    lost when the result was written as float32.
    """
    rows = cols = 60
    dem = _sloped_dem(rows, cols) + 300.0  # elevations where float32 spacing is ~3e-5
    dem[20:40, 20:40] = dem[20, 20]  # large flat
    dem[45:48, 10:13] -= 8.0  # extra pit
    dem[:5, :] = -9999.0  # nodata frame: data never touch the raster edge
    dem[-5:, :] = -9999.0
    dem[:, :5] = -9999.0
    dem[:, -5:] = -9999.0
    src = _write_dem(tmp_path / "dem.tif", dem)

    filled = tmp_path / "filled.tif"
    pointer = tmp_path / "d8.tif"
    fill_depressions(src, filled, fix_flats=True, overwrite=True)
    d8_pointer(filled, pointer, overwrite=True)

    fill_arr = _read(filled)
    valid = dem != -9999.0
    assert np.all(fill_arr[~valid] == -9999.0)
    assert np.all(fill_arr[valid] >= dem[valid] - 1e-3)  # never lowered
    assert float(fill_arr[46, 11]) > float(dem[46, 11]) + 1.0  # pit raised

    ptr = _read(pointer)
    interior = _data_interior(valid)
    # Every interior cell has a flow direction after the float32 round trip.
    assert np.all(ptr[interior] > 0)


def test_fill_depressions_flat_increment(tmp_path: Path):
    dem = _sloped_dem() + 300.0
    dem[10:30, 10:30] = dem[10, 10]
    src = _write_dem(tmp_path / "dem.tif", dem)
    filled = tmp_path / "filled.tif"
    pointer = tmp_path / "d8.tif"
    fill_depressions(src, filled, fix_flats=True, flat_increment=0.01, overwrite=True)
    d8_pointer(filled, pointer, overwrite=True)
    ptr = _read(pointer)
    assert np.all(ptr[1:-1, 1:-1] > 0)

    with pytest.raises(SurfaceHydrologyError):
        fill_depressions(src, filled, flat_increment=0.0, overwrite=True)
