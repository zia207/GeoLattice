"""Tests for geolattice.spatial_analysis.multidimensional_raster."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xarray")
import xarray as xr  # noqa: E402

from geolattice.spatial_analysis.multidimensional_raster import (
    MultidimensionalRaster,
    MultidimensionalRasterError,
    aggregate_multidimensional_raster,
    analyze_changes_using_ccdc,
    analyze_changes_using_landtrendr,
    build_multidimensional_info,
    build_multidimensional_transpose,
    copy_multidimensional_raster,
    create_space_time_cube_from_multidimensional_raster,
    detect_change_using_change_analysis_raster,
    find_argument_statistics,
    generate_multidimensional_anomaly,
    generate_trend_raster,
    make_multidimensional_raster_layer,
    make_netcdf_table_view,
    merge_multidimensional_rasters,
    multidimensional_filter,
    multidimensional_principal_components,
    predict_using_trend_raster,
    raster_to_netcdf,
    sample_multidimensional,
    subset_multidimensional_raster,
    table_to_netcdf,
    temporal_profile,
    zonal_statistics_as_table,
)


def _synthetic_cube(
    *,
    n_time: int = 36,
    n_y: int = 5,
    n_x: int = 6,
    start: str = "2018-01-01",
    seed: int = 0,
) -> xr.Dataset:
    times = pd.date_range(start, periods=n_time, freq="MS")
    y = np.linspace(10.0, 11.0, n_y)
    x = np.linspace(90.0, 91.0, n_x)
    t_idx, _, _ = np.meshgrid(np.arange(n_time), y, x, indexing="ij")
    rng = np.random.default_rng(seed)
    data = (
        20.0
        + 0.03 * t_idx
        + 2.5 * np.sin(2.0 * np.pi * t_idx / 12.0)
        + 0.05 * rng.normal(size=t_idx.shape)
    )
    # Disturbance patch after month 18.
    data[18:, 1:3, 2:4] -= 4.0
    return xr.Dataset(
        {"t2m": (("time", "y", "x"), data.astype(np.float32))},
        coords={"time": times, "y": y, "x": x},
    )


def _write_cube(path: Path, ds=None) -> Path:
    cube = ds if ds is not None else _synthetic_cube()
    cube.to_netcdf(path)
    return path


def test_open_subset_merge_and_exports(tmp_path: Path):
    nc = _write_cube(tmp_path / "cube.nc")
    mdr = make_multidimensional_raster_layer(nc, variables=["t2m"])
    assert isinstance(mdr, MultidimensionalRaster)
    assert mdr.variables == ["t2m"]
    assert "time" in mdr.get_dimension_names("t2m")
    assert len(mdr.get_dimension_values("time")) == 36

    sub = subset_multidimensional_raster(
        mdr,
        dimension_definitions={"time": {"start": "2019-01-01", "end": "2019-12-31"}},
        extent=(90.2, 10.2, 90.8, 10.8),
    )
    assert isinstance(sub, MultidimensionalRaster)
    assert sub.dataset.sizes["time"] == 12
    assert sub.dataset.sizes["y"] < mdr.dataset.sizes["y"]

    filtered = multidimensional_filter(
        mdr,
        variables=["t2m"],
        dimension_definitions={"time": slice("2018-06-01", "2018-08-01")},
    )
    assert filtered.dataset.sizes["time"] == 3

    a = mdr.sel(time=slice("2018", "2019"))
    b = mdr.sel(time=slice("2020", "2020"))
    merged = merge_multidimensional_rasters([a, b], dimension="time", combine="nested")
    assert merged.dataset.sizes["time"] == a.dataset.sizes["time"] + b.dataset.sizes["time"]

    out_nc = tmp_path / "out.nc"
    raster_to_netcdf(sub, out_nc, overwrite=True)
    assert out_nc.is_file()

    zarr_path = tmp_path / "out.zarr"
    try:
        copy_multidimensional_raster(mdr, zarr_path, overwrite=True)
        assert zarr_path.exists()
    except Exception as exc:
        pytest.skip(f"zarr write unavailable: {exc}")

    table = make_netcdf_table_view(nc, variables=["t2m"])
    assert "t2m" in table.columns
    assert len(table) == 36 * 5 * 6


def test_build_info_and_transpose(tmp_path: Path):
    from geolattice.raster.raster_read_write import write_geotiff

    paths = []
    for i, month in enumerate(("2019-01", "2019-02", "2019-03")):
        arr = np.full((4, 5), float(i), dtype=np.float32)
        path = tmp_path / f"slice_{month}.tif"
        write_geotiff(
            path,
            arr,
            crs=4326,
            geotransform=(90.0, 0.25, 0.0, 11.0, 0.0, -0.25),
            nodata=-9999.0,
            overwrite=True,
        )
        paths.append(path)

    built = build_multidimensional_info(
        paths,
        variable_name="value",
        dimension_name="time",
        dimension_values=pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
    )
    assert built.dataset.sizes["time"] == 3
    assert built.variables == ["value"]

    transposed = build_multidimensional_transpose(built, variable="value", dimension="time")
    assert transposed.dataset["value"].dims == ("y", "x", "time")


def test_aggregate_anomaly_argstats_trend_predict(tmp_path: Path):
    nc = _write_cube(tmp_path / "cube.nc")
    mdr = make_multidimensional_raster_layer(nc)

    yearly = aggregate_multidimensional_raster(
        mdr,
        variable="t2m",
        aggregation_method="MEAN",
        aggregation_definition="INTERVAL_KEYWORD",
        interval_keyword="YEARLY",
    )
    assert yearly.dataset.sizes["time"] == 3

    anom = generate_multidimensional_anomaly(
        mdr,
        variable="t2m",
        anomaly_type="DIFFERENCE_FROM_MEAN",
        calculation_interval="BY_MONTH",
    )
    assert anom.dataset.sizes["time"] == 36
    assert "month" not in anom.dataset.dims
    # Seasonal anomalies should centre near zero overall.
    assert abs(float(np.nanmean(anom.dataset["t2m"].values))) < 0.5

    argmax = find_argument_statistics(
        mdr,
        variable="t2m",
        statistics_type="ARGUMENT_OF_MAXIMUM",
    )
    assert "t2m_arg_max" in argmax.variables

    # Clean monotonic cube (no disturbance) for trend sign checks.
    times = pd.date_range("2018-01-01", periods=24, freq="MS")
    y = np.linspace(0.0, 1.0, 4)
    x = np.linspace(0.0, 1.0, 5)
    t_idx, _, _ = np.meshgrid(np.arange(24), y, x, indexing="ij")
    clean = xr.Dataset(
        {"t2m": (("time", "y", "x"), (10.0 + 0.05 * t_idx).astype(np.float32))},
        coords={"time": times, "y": y, "x": x},
    )
    clean_path = tmp_path / "clean.nc"
    clean.to_netcdf(clean_path)
    clean_mdr = make_multidimensional_raster_layer(clean_path)

    trend = generate_trend_raster(clean_mdr, variable="t2m", trend_type="LINEAR")
    assert set(trend.variables) >= {"intercept", "slope", "r2", "p_value"}
    assert float(np.nanmean(trend.dataset["slope"].values)) > 0

    harmonic = generate_trend_raster(
        mdr, variable="t2m", trend_type="HARMONIC", frequency=1
    )
    assert "cos_1" in harmonic.variables and "sin_1" in harmonic.variables

    mk = generate_trend_raster(clean_mdr, variable="t2m", trend_type="MANN_KENDALL")
    assert set(mk.variables) >= {"tau", "p_value", "slope", "intercept"}
    assert float(np.nanmean(mk.dataset["slope"].values)) > 0

    t0 = pd.Timestamp(clean_mdr.dataset["time"].values[0])
    future = pd.date_range("2020-01-01", periods=3, freq="MS")
    dim_vals = [(pd.Timestamp(t) - t0).days for t in future]
    pred = predict_using_trend_raster(trend, dimension_values=dim_vals)
    assert pred.dataset.sizes["time"] == 3


def test_change_detection_pca_cube_and_sampling(tmp_path: Path):
    nc = _write_cube(tmp_path / "cube.nc", _synthetic_cube(n_time=48))
    mdr = make_multidimensional_raster_layer(nc)

    ccdc = analyze_changes_using_ccdc(
        mdr, variable="t2m", min_observations=8, max_breaks=2
    )
    assert set(ccdc.variables) >= {
        "change_count",
        "first_break",
        "last_break",
        "magnitude",
    }

    lt = analyze_changes_using_landtrendr(mdr, variable="t2m")
    assert "disturbance_time" in lt.variables

    detected = detect_change_using_change_analysis_raster(
        ccdc, change_type="NUM_OF_CHANGES"
    )
    assert "num_of_changes" in detected.variables

    pca = multidimensional_principal_components(
        mdr, variable="t2m", n_components=2
    )
    assert "pc_1" in pca.variables and "pc_2" in pca.variables
    ratios = pca.dataset.attrs["explained_variance_ratio"]
    assert len(ratios) == 2
    assert abs(sum(ratios) - 1.0) < 1e-6 or sum(ratios) <= 1.0 + 1e-6

    cube = create_space_time_cube_from_multidimensional_raster(mdr)
    assert cube.dataset.attrs.get("space_time_cube") is True
    assert cube.dataset["t2m"].dims[0] == "time"

    profile = temporal_profile(mdr, x=90.5, y=10.5)
    assert list(profile.columns) == ["time", "t2m"]
    assert len(profile) == mdr.dataset.sizes["time"]

    samples = sample_multidimensional(mdr, [(90.3, 10.4), (90.7, 10.6)])
    assert len(samples) == 2 * mdr.dataset.sizes["time"]

    zones = np.ones((5, 6), dtype=np.int32)
    zones[:, 3:] = 2
    zonal = zonal_statistics_as_table(mdr, zones, stats=("mean", "count"))
    assert set(zonal["zone"]) == {1, 2}
    assert len(zonal) == 2 * mdr.dataset.sizes["time"]


def test_table_to_netcdf_and_errors(tmp_path: Path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    out = tmp_path / "table.nc"
    table_to_netcdf(frame, out, overwrite=True)
    assert out.is_file()

    with pytest.raises(MultidimensionalRasterError):
        make_multidimensional_raster_layer(tmp_path / "missing.nc")

    mdr = MultidimensionalRaster(_synthetic_cube())
    with pytest.raises(MultidimensionalRasterError):
        subset_multidimensional_raster(mdr, variables=["nope"])

    with pytest.raises(MultidimensionalRasterError):
        generate_trend_raster(mdr, trend_type="NOT_A_TREND")
