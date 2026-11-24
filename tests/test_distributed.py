"""Tests for geolattice.distributed: backend, registry, partitioning, raster, vector, bridge, io."""

import warnings

import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

import geolattice.distributed as pdist
from geolattice.distributed import _backend

HAS_GPU = _backend.gpu_count() > 0
COMPUTES = ["cpu"] + (["gpu"] if HAS_GPU else [])
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a CUDA GPU with CuPy")

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def dem():
    rng = np.random.default_rng(0)
    z = np.cumsum(np.cumsum(rng.normal(size=(300, 260)), 0), 1).astype("float32")
    y = np.arange(300)[::-1] * 10.0
    x = np.arange(260) * 10.0
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}, name="dem")


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(3000, 2))
    return gpd.GeoDataFrame({"id": np.arange(len(xy))}, geometry=gpd.points_from_xy(*xy.T),
                            crs="EPSG:3857")


@pytest.fixture(scope="module")
def grid():
    cells = [shapely.box(i, j, i + 10, j + 10) for i in range(0, 100, 10) for j in range(0, 100, 10)]
    return gpd.GeoDataFrame({"cell": np.arange(len(cells))}, geometry=cells, crs="EPSG:3857")


# --------------------------------------------------------------------------- backend
class TestBackend:
    def test_capabilities(self):
        caps = pdist.capabilities()
        assert caps["dask"] is True
        assert isinstance(caps["gpus"], int)

    def test_resolve_cpu(self):
        assert pdist.resolve_compute("cpu") == "cpu"

    def test_resolve_invalid(self):
        with pytest.raises(ValueError):
            pdist.resolve_compute("tpu")

    def test_gpu_fallback_warns(self, monkeypatch):
        monkeypatch.setattr(_backend, "gpu_count", lambda: 0)
        _backend._warned.clear()
        with pytest.warns(RuntimeWarning, match="GPU requested"):
            assert pdist.resolve_compute("gpu", op="test.op") == "cpu"

    def test_defaults(self):
        old = pdist.get_defaults()
        try:
            assert pdist.set_defaults(npartitions=3).npartitions == 3
            with pytest.raises(TypeError):
                pdist.set_defaults(bogus=1)
        finally:
            pdist.set_defaults(compute=old.compute, npartitions=old.npartitions, chunks=old.chunks)


class TestRegistry:
    def test_dispatch_falls_back_without_gpu_impl(self):
        @pdist.register("test.cpu_only", "cpu")
        def impl(x):
            return x + 1

        fn, used = pdist.dispatch("test.cpu_only", compute="cpu")
        assert used == "cpu" and fn(1) == 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert pdist.dispatch("test.cpu_only", compute="gpu")[1] == "cpu"

    def test_unknown_op(self):
        with pytest.raises(KeyError):
            pdist.dispatch("test.missing")

    def test_list_ops(self):
        ops = pdist.list_ops()
        assert ops["raster.slope"] == ["cpu", "gpu"]
        assert ops["vector.sjoin"] == ["cpu", "gpu"]
        assert ops["bridge.polygonize"] == ["cpu"]


class TestPartition:
    def test_spatial_partition_keeps_rows(self, points):
        parts = pdist.spatial_partition(points, 7)
        assert len(parts) == 7
        assert sorted(np.concatenate([p.index for p in parts])) == list(points.index)

    def test_hilbert_partitions_are_compact(self, points):
        parts = pdist.spatial_partition(points, 16)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in pdist.partition_bounds(parts)]
        assert np.mean(areas) < 0.5 * 100 * 100

    def test_plan_chunks(self):
        ch = pdist.plan_chunks((3, 20000, 20000), "float32", halo=4)
        assert ch[0] == 3 and 256 <= ch[1] <= 20000


# --------------------------------------------------------------------------- raster
class TestRasterTerrain:
    @pytest.mark.parametrize("name", ["slope", "aspect", "hillshade", "curvature", "tri", "roughness"])
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_matches_whole_array(self, dem, name, compute):
        terrain = pytest.importorskip("geolattice.rsense.lidar.terrain")
        ref = np.asarray(getattr(terrain, name)(dem))
        out = getattr(pdist.raster, name)(dem, compute=compute, chunks={"y": 70, "x": 60})
        assert isinstance(out, xr.DataArray) and isinstance(out.data, np.ndarray)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-3)

    def test_tpi(self, dem):
        z = dem.values
        ref = z - ndi.uniform_filter(z, 9, mode="nearest")
        np.testing.assert_allclose(pdist.raster.tpi(dem, window=9, chunks=64), ref, atol=1e-3)

    def test_plain_array_needs_res(self, dem):
        with pytest.raises(ValueError, match="res="):
            pdist.raster.slope(dem.values)
        out = pdist.raster.slope(dem.values, res=10.0)
        assert isinstance(out, np.ndarray)

    def test_lazy_stays_lazy_and_on_host(self, dem):
        out = pdist.raster.slope(dem.chunk({"y": 100, "x": 100}), compute=COMPUTES[-1])
        assert pdist.is_lazy(out) and not pdist.is_gpu(out)
        assert out.compute().shape == dem.shape


class TestRasterFocal:
    @pytest.mark.parametrize("stat,ref", [("mean", ndi.uniform_filter), ("min", ndi.minimum_filter),
                                          ("max", ndi.maximum_filter), ("median", ndi.median_filter)])
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_focal(self, dem, stat, ref, compute):
        z = dem.values
        out = pdist.raster.focal(z, stat, size=5, chunks=(64, 64), compute=compute)
        np.testing.assert_allclose(out, ref(z, 5, mode="nearest"), atol=1e-3)

    def test_focal_std(self, dem):
        z = dem.values.astype("float64")
        m = ndi.uniform_filter(z, 3, mode="nearest")
        ref = np.sqrt(np.maximum(ndi.uniform_filter(z * z, 3, mode="nearest") - m * m, 0))
        np.testing.assert_allclose(pdist.raster.focal(z, "std", chunks=50), ref, rtol=1e-3, atol=1e-2)

    def test_focal_band_stack(self, dem):
        stack = xr.concat([dem, dem * 2], dim="band")
        out = pdist.raster.focal(stack, "mean", size=3, chunks=(80, 80))
        np.testing.assert_allclose(out[1], ndi.uniform_filter(dem.values * 2, 3, mode="nearest"),
                                   atol=1e-2)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_convolve(self, dem, compute):
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype="float32")
        z = dem.values
        out = pdist.raster.convolve(z, k, chunks=(64, 64), compute=compute)
        np.testing.assert_allclose(out, ndi.convolve(z, k, mode="nearest"), atol=1e-2)

    def test_even_size_rejected(self, dem):
        with pytest.raises(ValueError):
            pdist.raster.focal(dem, size=4)


class TestRasterAlgebra:
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_calc_string(self, compute):
        a = xr.DataArray(np.full((50, 40), 3.0), dims=("y", "x"))
        b = xr.DataArray(np.full((50, 40), 1.0), dims=("y", "x"))
        out = pdist.raster.calc("(a - b) / (a + b) + sqrt(b)", a=a, b=b, compute=compute)
        np.testing.assert_allclose(out, 1.5)
        assert isinstance(out.data, np.ndarray)

    def test_calc_callable_and_ndiff(self):
        a = np.full((20, 20), 4.0)
        b = np.full((20, 20), 2.0)
        np.testing.assert_allclose(pdist.raster.calc(lambda a, b: a * b, a=a, b=b), 8.0)
        np.testing.assert_allclose(pdist.raster.normalized_difference(a, b), 1 / 3)

    def test_calc_blocks_builtins(self):
        with pytest.raises(NameError):
            pdist.raster.calc("__import__('os')", a=np.zeros((4, 4)))

    def test_where(self):
        a = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("y", "x"))
        np.testing.assert_allclose(pdist.raster.where(a > 7, 1.0, 0.0).sum(), 8)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_reclassify(self, compute):
        a = np.array([[0.1, 0.4], [0.6, np.nan]])
        out = pdist.raster.reclassify(a, bins=[0.25, 0.5], values=[1, 2, 3], compute=compute)
        np.testing.assert_allclose(out, [[1, 2], [3, np.nan]])
        out = pdist.raster.reclassify(np.array([[1.0, 2.0], [3.0, 1.0]]), mapping={1: 10, 2: 20})
        np.testing.assert_allclose(out, [[10, 20], [np.nan, 10]])

    def test_normalize(self, dem):
        out = pdist.raster.normalize(dem, chunks=100)
        assert float(out.min()) == pytest.approx(0) and float(out.max()) == pytest.approx(1)
        z = pdist.raster.normalize(dem, method="zscore")
        assert float(z.mean()) == pytest.approx(0, abs=1e-4)


@gpu
class TestGPUSpecific:
    def test_gpu_input_returns_gpu(self, dem):
        import cupy as cp

        out = pdist.raster.slope(pdist.to_gpu(dem), compute="gpu", chunks={"y": 100, "x": 100})
        assert isinstance(out.data, cp.ndarray)

    def test_to_gpu_roundtrip(self, dem):
        g = pdist.to_gpu(dem)
        assert pdist.is_gpu(g)
        np.testing.assert_array_equal(pdist.to_cpu(g).values, dem.values)


# --------------------------------------------------------------------------- vector
class TestVector:
    def test_buffer_matches_geopandas(self, points):
        out = pdist.vector.buffer(points, 1.5, npartitions=4)
        assert list(out.columns) == list(points.columns)
        assert out.index.equals(points.index)
        np.testing.assert_allclose(out.area, points.buffer(1.5).area)

    def test_measures(self, grid):
        np.testing.assert_allclose(pdist.vector.area(grid, npartitions=3), 100.0)
        np.testing.assert_allclose(pdist.vector.length(grid, npartitions=3), 40.0)
        assert pdist.vector.bounds(grid, npartitions=3).equals(grid.bounds)

    def test_centroid_simplify_make_valid(self, grid):
        c = pdist.vector.centroid(grid, npartitions=2)
        assert (c.geometry.geom_type == "Point").all()
        assert pdist.vector.simplify(grid, 0.1, npartitions=2).geometry.equals(grid.geometry)
        bow = gpd.GeoDataFrame(geometry=[shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])])
        assert pdist.vector.make_valid(bow).is_valid.all()

    def test_to_crs(self, points):
        out = pdist.vector.to_crs(points, "EPSG:4326", npartitions=4)
        assert out.crs.to_epsg() == 4326
        assert out.geometry.geom_equals_exact(points.to_crs(4326).geometry, 1e-9).all()

    def test_clip_prunes_and_keeps_order(self, points):
        mask = shapely.box(10, 10, 30, 30)
        out = pdist.vector.clip(points, mask, npartitions=8)
        ref = gpd.clip(points, mask)
        assert sorted(out["id"]) == sorted(ref["id"])
        assert list(out["id"]) == sorted(out["id"])

    @pytest.mark.parametrize("predicate", ["within", "intersects"])
    def test_sjoin_matches_geopandas(self, points, grid, predicate):
        out = pdist.vector.sjoin(points, grid, predicate=predicate, npartitions=6)
        ref = gpd.sjoin(points, grid, predicate=predicate)
        key = ["id", "index_right"]
        a = out[key].sort_values(key).to_numpy()
        b = ref[key].sort_values(key).to_numpy()
        np.testing.assert_array_equal(a, b)
        assert list(out.columns) == list(ref.columns)

    def test_sjoin_left_keeps_unmatched(self, points):
        small = gpd.GeoDataFrame({"v": [1]}, geometry=[shapely.box(0, 0, 5, 5)], crs=points.crs)
        out = pdist.vector.sjoin(points, small, how="left", npartitions=5)
        assert len(out) == len(points)
        assert out["v"].notna().sum() == gpd.sjoin(points, small).shape[0]

    def test_sjoin_dwithin(self, points, grid):
        cents = grid.assign(geometry=grid.centroid)
        out = pdist.vector.sjoin(points, cents, predicate="dwithin", distance=2.0, npartitions=4)
        ref = gpd.sjoin(points, cents, predicate="dwithin", distance=2.0)
        assert len(out) == len(ref)

    def test_sjoin_crs_mismatch(self, points, grid):
        with pytest.raises(ValueError, match="CRS"):
            pdist.vector.sjoin(points, grid.to_crs(4326))

    def test_gpu_request_falls_back(self, points):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = pdist.vector.area(points, compute="gpu")
        assert len(out) == len(points)


# --------------------------------------------------------------------------- io
class TestIO:
    def test_vector_parquet_roundtrip(self, points, tmp_path):
        path = pdist.write_vector(points, tmp_path / "pts", npartitions=4)
        files = sorted((tmp_path / "pts").glob("*.parquet"))
        assert len(files) == 4
        back = pdist.read_vector(path)
        assert sorted(back["id"]) == list(points["id"])
        assert back.crs == points.crs

    def test_vector_partition_on(self, grid, tmp_path):
        g = grid.assign(half=np.where(grid["cell"] < 50, "a", "b"))
        pdist.write_vector(g, tmp_path / "hive", partition_on="half", npartitions=2)
        assert (tmp_path / "hive" / "half=a").is_dir()

    def test_vector_ogr_parallel_ranges(self, points, tmp_path):
        f = tmp_path / "pts.gpkg"
        points.to_file(f)
        back = pdist.read_vector(f, npartitions=3)
        assert len(back) == len(points)
        assert sorted(back["id"]) == list(points["id"])

    def test_raster_roundtrip(self, dem, tmp_path):
        pytest.importorskip("rioxarray")
        src = dem.rio.write_crs("EPSG:32633")
        tif = pdist.write_raster(src, tmp_path / "dem.tif", format="gtiff")
        da = pdist.read_raster(tif, chunks={"y": 128, "x": 128})
        assert pdist.is_lazy(da)
        np.testing.assert_allclose(da.values, dem.values)
        s = pdist.raster.slope(da)
        assert pdist.is_lazy(s)
        z = pdist.write_raster(s, tmp_path / "slope.zarr")
        back = pdist.read_zarr(z)
        np.testing.assert_allclose(back["slope"].values, s.values, equal_nan=True)


class TestSession:
    def test_local_session(self, dem):
        with pdist.start("local", n_workers=2, threads_per_worker=1, dashboard=False) as s:
            assert pdist.get_session() is s
            assert pdist.status()["workers"] == 2
            out = pdist.raster.slope(dem, chunks={"y": 100, "x": 100})
            assert out.shape == dem.shape
        assert pdist.get_session() is None


# --------------------------------------------------------------------------- phase 2: GPU vector
@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(2)
    polys = [shapely.Point(rng.uniform(0, 100), rng.uniform(0, 100)).buffer(rng.uniform(1, 6))
             for _ in range(150)]
    polys = [p.difference(p.centroid.buffer(0.5)) if i % 3 == 0 else p for i, p in enumerate(polys)]
    polys.append(shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(3, 3, 5, 6)]))
    return gpd.GeoDataFrame({"z": np.arange(len(polys))}, geometry=polys, crs="EPSG:3857")


@gpu
class TestGPUVector:
    @pytest.mark.parametrize("fn", ["area", "length"])
    def test_measures_polygons(self, blobs, fn):
        out = getattr(pdist.vector, fn)(blobs, compute="gpu")
        assert out.index.equals(blobs.index)
        np.testing.assert_allclose(out, getattr(blobs, fn))

    def test_bounds(self, blobs):
        np.testing.assert_allclose(pdist.vector.bounds(blobs, compute="gpu"), blobs.bounds)

    def test_lines_and_points(self, points):
        rng = np.random.default_rng(3)
        lines = gpd.GeoDataFrame(geometry=[shapely.LineString(rng.uniform(0, 10, (5, 2)))
                                           for _ in range(40)]
                                 + [shapely.MultiLineString([[(0, 0), (1, 0)], [(2, 0), (2, 3)]])])
        np.testing.assert_allclose(pdist.vector.length(lines, compute="gpu"), lines.length)
        np.testing.assert_allclose(pdist.vector.bounds(lines, compute="gpu"), lines.bounds)
        np.testing.assert_allclose(pdist.vector.bounds(points, compute="gpu"), points.bounds)

    def test_mixed_families_fall_back(self):
        mixed = gpd.GeoDataFrame(geometry=[shapely.Point(0, 0), shapely.box(0, 0, 2, 2)])
        np.testing.assert_allclose(pdist.vector.area(mixed, compute="gpu"), [0, 4])

    @pytest.mark.parametrize("predicate", ["within", "intersects"])
    def test_point_in_polygon_sjoin(self, points, blobs, predicate):
        out = pdist.vector.sjoin(points, blobs, predicate=predicate, compute="gpu")
        ref = gpd.sjoin(points, blobs, predicate=predicate)
        key = ["id", "index_right"]
        np.testing.assert_array_equal(out[key].sort_values(key).to_numpy(),
                                      ref[key].sort_values(key).to_numpy())
        assert list(out.columns) == list(ref.columns)

    def test_gpu_kernels_actually_run(self, points, blobs, monkeypatch):
        from geolattice.distributed import _gpu_geom as G

        calls = []
        for name in ("points_in_polygons", "rasterize_polygons"):
            real = getattr(G, name)
            monkeypatch.setattr(G, name, lambda *a, _r=real, _n=name, **k: calls.append(_n) or _r(*a, **k))
        pdist.vector.sjoin(points, blobs, predicate="within", compute="gpu")
        like = xr.DataArray(np.zeros((40, 40)), dims=("y", "x"),
                            coords={"y": np.arange(40)[::-1] * 2.5 + 1.25, "x": np.arange(40) * 2.5 + 1.25})
        pdist.rasterize(blobs, like, compute="gpu")
        assert {"points_in_polygons", "rasterize_polygons"} <= set(calls)

    def test_contains_swapped(self, points, blobs):
        out = pdist.vector.sjoin(blobs, points, predicate="contains", compute="gpu")
        ref = gpd.sjoin(blobs, points, predicate="contains")
        assert len(out) == len(ref) and list(out.columns) == list(ref.columns)

    def test_left_join(self, points, blobs):
        out = pdist.vector.sjoin(points, blobs, how="left", compute="gpu")
        assert len(out) == len(gpd.sjoin(points, blobs, how="left"))

    def test_knn_exact_at_large_coordinates(self):
        rng = np.random.default_rng(4)
        off = 5e6
        a = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*(rng.uniform(0, 100, (2, 300)) + off)))
        b = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*(rng.uniform(0, 100, (2, 4000)) + off)))
        g = pdist.vector.knn(a, b, k=3, compute="gpu")
        c = pdist.vector.knn(a, b, k=3, compute="cpu")
        np.testing.assert_array_equal(g["index_right"], c["index_right"])
        np.testing.assert_allclose(g["distance"], c["distance"])


class TestKNN:
    def test_knn_self_excludes_self(self, points):
        out = pdist.vector.knn(points.iloc[:500], k=2)
        assert len(out) == 1000
        assert (out["index_right"].to_numpy() != out.index.to_numpy()).all()
        assert (out.groupby(level=0)["distance"].apply(lambda d: d.is_monotonic_increasing)).all()

    def test_knn_max_distance(self, points):
        out = pdist.vector.knn(points.iloc[:200], points.iloc[200:], k=5, max_distance=1.0)
        assert (out["distance"] <= 1.0).all()

    def test_knn_needs_points(self, grid, points):
        with pytest.raises(TypeError):
            pdist.vector.knn(grid, points)


# --------------------------------------------------------------------------- phase 2: bridge
X0 = 0.0137  # keeps pixel centres off the integer box edges, so burn ties cannot occur
@pytest.fixture(scope="module")
def field():
    pytest.importorskip("rioxarray")
    rng = np.random.default_rng(5)
    h, w = 200, 250
    da = xr.DataArray(rng.random((h, w)).astype("float32"), dims=("y", "x"),
                      coords={"y": 100 - (np.arange(h) + 0.5) * 0.5,
                              "x": X0 + (np.arange(w) + 0.5) * 0.4},
                      name="field")
    return da.rio.write_crs("EPSG:3857")


class TestBridge:
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_rasterize_matches_rasterio(self, blobs, field, compute):
        from affine import Affine
        from rasterio.features import rasterize as rio_rasterize

        ref = rio_rasterize(zip(blobs.geometry, blobs.z + 1), out_shape=field.shape,
                            transform=Affine(0.4, 0, X0, 0, -0.5, 100), fill=0, dtype="float32")
        out = pdist.rasterize(blobs, field, values=blobs.z + 1, compute=compute,
                              chunks={"y": 64, "x": 48})
        assert isinstance(out.data, np.ndarray)
        np.testing.assert_array_equal(np.asarray(out), ref)

    def test_rasterize_lazy_and_all_touched(self, blobs, field):
        out = pdist.rasterize(blobs, field.chunk({"y": 50, "x": 50}), all_touched=True)
        assert pdist.is_lazy(out)
        plain = pdist.rasterize(blobs, field)
        assert float(out.sum()) >= float(plain.sum())

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_zonal_stats(self, blobs, field, compute):
        from affine import Affine
        from rasterio.features import rasterize as rio_rasterize

        lab = rio_rasterize(zip(blobs.geometry, blobs.z + 1), out_shape=field.shape,
                            transform=Affine(0.4, 0, X0, 0, -0.5, 100), fill=0, dtype="int32")
        out = pdist.zonal_stats(blobs, field, compute=compute, chunks={"y": 64, "x": 48})
        v = field.values
        for i in (0, 7, 42):
            m = lab == i + 1
            assert out["count"].iloc[i] == m.sum()
            if m.any():
                assert out["mean"].iloc[i] == pytest.approx(v[m].mean(), rel=1e-6)
                assert out["min"].iloc[i] == pytest.approx(v[m].min())
                assert out["max"].iloc[i] == pytest.approx(v[m].max())
                assert out["std"].iloc[i] == pytest.approx(v[m].std(), rel=1e-5)

    def test_zonal_stats_bands_and_empty_zone(self, blobs, field):
        stack = xr.concat([field, field * 2], dim=xr.Variable("band", [1, 2]))
        far = gpd.GeoDataFrame({"z": [-1]}, geometry=[shapely.box(1000, 1000, 1001, 1001)],
                               crs=blobs.crs)
        zones = gpd.GeoDataFrame(pd_concat([blobs, far]), crs=blobs.crs)
        out = pdist.zonal_stats(zones, stack, stats=("count", "mean"))
        assert {"mean_1", "mean_2", "count_1"} <= set(out.columns)
        np.testing.assert_allclose(out["mean_2"], 2 * out["mean_1"], equal_nan=True, rtol=1e-6)
        assert out["count_1"].iloc[-1] == 0 and np.isnan(out["mean_1"].iloc[-1])

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_extract_nearest(self, points, field, compute):
        out = pdist.extract_values(points, field, compute=compute, chunks=64)
        x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
        ins = (x >= X0) & (x < X0 + 100)
        ref = field.sel(x=xr.DataArray(x[ins]), y=xr.DataArray(y[ins]), method="nearest").values
        np.testing.assert_allclose(out["field"].to_numpy()[ins], ref)
        assert np.isnan(out["field"].to_numpy()[~ins]).all()

    def test_extract_linear(self, points, field):
        out = pdist.extract_values(points, field, method="linear", chunks=50)
        x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
        ref = field.interp(x=xr.DataArray(x), y=xr.DataArray(y)).values
        ok = ~np.isnan(ref)
        np.testing.assert_allclose(out["field"].to_numpy()[ok], ref[ok], atol=1e-5)

    def test_extract_needs_points(self, grid, field):
        with pytest.raises(TypeError):
            pdist.extract_values(grid, field)

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_polygonize_seams_merge(self, connectivity):
        rng = np.random.default_rng(6)
        cls = xr.DataArray((rng.random((60, 70)) > 0.5).astype("int16"), dims=("y", "x"),
                           coords={"y": np.arange(60)[::-1] + 0.5, "x": np.arange(70) + 0.5})
        whole = pdist.polygonize(cls, connectivity=connectivity, chunks={"y": 60, "x": 70})
        split = pdist.polygonize(cls, connectivity=connectivity, chunks={"y": 17, "x": 23})
        assert len(whole) == len(split)
        assert sorted(np.round(whole.area, 6)) == sorted(np.round(split.area, 6))
        def by_value(g):
            return g.assign(a=g.area).groupby("value")["a"].sum()

        np.testing.assert_allclose(by_value(whole), by_value(split))

    def test_polygonize_skips_nan(self):
        a = np.ones((10, 10), dtype="float32")
        a[:5] = np.nan
        da = xr.DataArray(a, dims=("y", "x"), coords={"y": np.arange(10)[::-1] + 0.5,
                                                      "x": np.arange(10) + 0.5})
        out = pdist.polygonize(da, chunks=4)
        assert len(out) == 1 and out.area.iloc[0] == pytest.approx(50)


# --------------------------------------------------------------------------- dask-geopandas
class TestDaskGeoPandas:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("dask_geopandas")

    def test_lazy_ops(self, points):
        import dask_geopandas

        dg = dask_geopandas.from_geopandas(points, npartitions=4)
        out = pdist.vector.buffer(dg, 1.0)
        assert pdist.is_lazy(out)
        np.testing.assert_allclose(out.compute().area, points.buffer(1.0).area)
        np.testing.assert_allclose(pdist.vector.area(dg).compute(), points.area)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_lazy_sjoin(self, points, grid, compute):
        import dask_geopandas

        dg = dask_geopandas.from_geopandas(points, npartitions=4)
        out = pdist.vector.sjoin(dg, grid, predicate="within", compute=compute)
        assert pdist.is_lazy(out)
        assert len(out.compute()) == len(gpd.sjoin(points, grid, predicate="within"))

    def test_read_vector_lazy(self, points, tmp_path):
        pdist.write_vector(points, tmp_path / "pts", npartitions=3)
        lazy = pdist.read_vector(tmp_path / "pts", lazy=True)
        assert pdist.is_lazy(lazy)
        assert len(lazy.compute()) == len(points)


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- phase 3: analysis
@pytest.fixture(scope="module")
def lattice():
    rng = np.random.default_rng(7)
    cells = [shapely.box(i, j, i + 1, j + 1) for j in range(25) for i in range(25)]
    g = gpd.GeoDataFrame(geometry=cells)
    y = np.array([np.sin(c.centroid.x / 5) + np.cos(c.centroid.y / 4) for c in cells])
    return g, y + rng.normal(0, 0.3, len(cells))


class TestWeights:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("libpysal")

    @pytest.mark.parametrize("kind", ["queen", "rook"])
    def test_contiguity_matches_libpysal(self, lattice, kind):
        import libpysal

        g, _ = lattice
        w = pdist.analysis.weights(g, kind, row_standardize=False)
        ref = (libpysal.weights.Queen if kind == "queen" else libpysal.weights.Rook).from_dataframe(
            g, use_index=False)
        assert (w.sparse != ref.sparse.tocsr()).nnz == 0

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_knn_matches_libpysal(self, points, compute):
        import libpysal

        w = pdist.analysis.weights(points.iloc[:800], "knn", k=5, row_standardize=False,
                                   compute=compute)
        ref = libpysal.weights.KNN.from_dataframe(points.iloc[:800], k=5)
        assert (w.sparse != ref.sparse.tocsr()).nnz == 0

    def test_distance_band(self, points):
        import libpysal

        w = pdist.analysis.weights(points.iloc[:800], "distance", threshold=4.0, row_standardize=False)
        ref = libpysal.weights.DistanceBand.from_dataframe(points.iloc[:800], threshold=4.0,
                                                           silence_warnings=True)
        assert (w.sparse != ref.sparse.tocsr()).nnz == 0

    def test_row_standardize(self, lattice):
        w = pdist.analysis.weights(lattice[0], "queen")
        np.testing.assert_allclose(np.asarray(w.sparse.sum(axis=1)).ravel(), 1.0)
        assert w.transform == "r"


class TestAutocorrelation:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("esda")

    @pytest.fixture(scope="class")
    def setup(self, lattice):
        import libpysal

        g, y = lattice
        w = pdist.analysis.weights(g, "queen")
        lw = libpysal.weights.Queen.from_dataframe(g, use_index=False)
        lw.transform = "r"
        lb = libpysal.weights.Queen.from_dataframe(g, use_index=False)
        return y, w, lw, lb

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_morans_i(self, setup, compute):
        import esda

        y, w, lw, _ = setup
        r = pdist.analysis.morans_i(y, w, permutations=199, seed=0, compute=compute)
        m = esda.Moran(y, lw, permutations=0)
        assert r.statistic == pytest.approx(m.I)
        assert r.expected == pytest.approx(m.EI)
        assert r.variance_norm == pytest.approx(m.VI_norm)
        assert r.z_norm == pytest.approx(m.z_norm)
        assert r.p_norm == pytest.approx(m.p_norm)
        assert r.p_sim == pytest.approx(1 / 200)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_gearys_c(self, setup, compute):
        import esda

        y, w, lw, _ = setup
        r = pdist.analysis.gearys_c(y, w, permutations=99, seed=0, compute=compute)
        ref = esda.Geary(y, lw, permutations=0)
        assert r.statistic == pytest.approx(ref.C)
        assert r.variance_norm == pytest.approx(ref.VC_rand)
        assert r.z_norm == pytest.approx(ref.z_rand)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_local_morans(self, setup, compute):
        import esda

        y, w, lw, _ = setup
        out = pdist.analysis.local_morans(y, w, permutations=499, seed=1, compute=compute)
        ref = esda.Moran_Local(y, lw, permutations=499, seed=1)
        np.testing.assert_allclose(out["Is"], ref.Is)
        np.testing.assert_array_equal(out["q"], ref.q)
        assert np.corrcoef(out["p_sim"], ref.p_sim)[0, 1] > 0.95

    @pytest.mark.parametrize("star", [True, False])
    def test_getis_ord(self, setup, star):
        import esda

        y, w, _, lb = setup
        out = pdist.analysis.getis_ord_local(y, w, star=star, permutations=299, seed=1)
        ref = esda.G_Local(y, lb, transform="B", star=star, permutations=299, seed=1)
        np.testing.assert_allclose(out["G"], ref.Gs)
        np.testing.assert_allclose(out["Zs"], ref.Zs)
        np.testing.assert_allclose(out["p_norm"], ref.p_norm)
        assert np.corrcoef(out["p_sim"], ref.p_sim)[0, 1] > 0.95

    def test_rejects_nan(self, setup):
        y, w, _, _ = setup
        bad = y.copy()
        bad[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            pdist.analysis.morans_i(bad, w)


class TestClustering:
    @pytest.fixture(scope="class")
    def blobs_xy(self):
        rng = np.random.default_rng(8)
        centers = np.array([[0, 0], [20, 0], [0, 20], [20, 20]])
        xy = np.vstack([c + rng.normal(0, 1, (150, 2)) for c in centers])
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(*xy.T)), np.repeat(np.arange(4), 150)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_dbscan(self, blobs_xy, compute):
        from sklearn.metrics import adjusted_rand_score

        g, truth = blobs_xy
        lab = pdist.analysis.dbscan(g, eps=1.5, min_samples=5, compute=compute)
        assert lab.index.equals(g.index)
        core = lab.to_numpy() >= 0
        assert adjusted_rand_score(truth[core], lab.to_numpy()[core]) == pytest.approx(1.0)

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_hdbscan(self, blobs_xy, compute):
        from sklearn.metrics import adjusted_rand_score

        g, truth = blobs_xy
        lab = pdist.analysis.hdbscan(g, min_cluster_size=20, compute=compute).to_numpy()
        core = lab >= 0
        assert adjusted_rand_score(truth[core], lab[core]) > 0.99

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_kmeans(self, blobs_xy, compute):
        from sklearn.metrics import adjusted_rand_score

        g, truth = blobs_xy
        lab, centers = pdist.analysis.kmeans(g, 4, compute=compute)
        assert adjusted_rand_score(truth, lab.to_numpy()) == pytest.approx(1.0)
        assert centers.shape == (4, 2)


class TestKDE:
    @pytest.mark.parametrize("kernel", ["gaussian", "quartic"])
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_matches_brute_force(self, kernel, compute):
        rng = np.random.default_rng(9)
        xy = rng.uniform(0, 50, (300, 2))
        pts = gpd.GeoDataFrame({"w": rng.uniform(1, 2, 300)}, geometry=gpd.points_from_xy(*xy.T))
        like = xr.DataArray(np.zeros((60, 70)), dims=("y", "x"),
                            coords={"y": 50 - (np.arange(60) + 0.5) * 50 / 60,
                                    "x": (np.arange(70) + 0.5) * 50 / 70})
        h = 3.0
        out = pdist.analysis.kde(pts, like, bandwidth=h, kernel=kernel, weights="w",
                                 compute=compute, chunks={"y": 17, "x": 23})
        gx, gy = np.meshgrid(like.x.values, like.y.values)
        u2 = ((gx[..., None] - xy[:, 0]) ** 2 + (gy[..., None] - xy[:, 1]) ** 2) / h**2
        k = (np.exp(-u2 / 2) / (2 * np.pi) if kernel == "gaussian"
             else np.where(u2 < 1, 3 / np.pi * (1 - u2) ** 2, 0))
        ref = (k * pts["w"].to_numpy()).sum(-1) / h**2
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-7)


# --------------------------------------------------------------------------- phase 3: network
@pytest.fixture(scope="module")
def streets():
    rng = np.random.default_rng(10)
    n = 15
    segs = [shapely.LineString([(i, j), (i + 1, j)]) for i in range(n - 1) for j in range(n)]
    segs += [shapely.LineString([(i, j), (i, j + 1)]) for i in range(n) for j in range(n - 1)]
    segs.append(shapely.LineString([(100, 100), (101, 100)]))
    lines = gpd.GeoDataFrame({"cost": rng.uniform(1, 3, len(segs))}, geometry=segs)
    return pdist.network.graph_from_lines(lines, weight="cost")


class TestNetwork:
    def test_graph(self, streets):
        assert streets.n_nodes == 15 * 15 + 2
        assert streets.n_edges == 2 * 15 * 14 + 1

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_shortest_paths_match_networkx(self, streets, compute):
        nx = pytest.importorskip("networkx")
        g = streets.to_networkx()
        d = pdist.network.shortest_path_lengths(streets, [0, 7], compute=compute)
        ref = nx.single_source_dijkstra_path_length(g, 7, weight="weight")
        np.testing.assert_allclose([d[1][k] for k in ref], [ref[k] for k in ref])
        assert np.isinf(d[0]).sum() == 2
        p = pdist.network.shortest_path(streets, 0, 15 * 15 - 1, compute=compute)
        assert p["cost"] == pytest.approx(nx.dijkstra_path_length(g, 0, 15 * 15 - 1))
        assert p["nodes"][0] == 0 and p["nodes"][-1] == 15 * 15 - 1
        assert p["geometry"].length == pytest.approx(len(p["nodes"]) - 1)

    def test_unreachable(self, streets):
        p = pdist.network.shortest_path(streets, 0, streets.n_nodes - 1)
        assert p["nodes"] == [] and np.isinf(p["cost"])

    @pytest.mark.parametrize("compute", COMPUTES)
    def test_components(self, streets, compute):
        assert len(np.unique(pdist.network.components(streets, compute=compute))) == 2

    @pytest.mark.parametrize("kind", ["betweenness", "pagerank", "degree"])
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_centrality(self, streets, kind, compute):
        nx = pytest.importorskip("networkx")
        g = streets.to_networkx()
        out = pdist.network.centrality(streets, kind, compute=compute).to_numpy()
        ref = {"betweenness": lambda: nx.betweenness_centrality(g),
               "pagerank": lambda: nx.pagerank(g, weight="weight"),
               "degree": lambda: dict(g.degree())}[kind]()
        np.testing.assert_allclose(out, [ref[i] for i in range(streets.n_nodes)], atol=1e-5)

    @gpu
    def test_betweenness_overflow_falls_back(self):
        nx = pytest.importorskip("networkx")
        n = 40
        segs = [shapely.LineString([(i, j), (i + 1, j)]) for i in range(n - 1) for j in range(n)]
        segs += [shapely.LineString([(i, j), (i, j + 1)]) for i in range(n) for j in range(n - 1)]
        g = pdist.network.graph_from_lines(gpd.GeoDataFrame(geometry=segs))
        assert pdist.network.max_shortest_path_count(g) > 1e22
        with pytest.warns(RuntimeWarning, match="overflow"):
            out = pdist.network.centrality(g, "betweenness", compute="gpu")
        ref = nx.betweenness_centrality(g.to_networkx())
        np.testing.assert_allclose(out, [ref[i] for i in range(g.n_nodes)], atol=1e-9)

    def test_nearest_nodes(self, streets):
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0.1, 10.4], [0.2, 12.6]))
        ids = pdist.network.nearest_nodes(streets, pts)
        np.testing.assert_allclose(streets.xy[ids], [[0, 0], [10, 13]])

    def test_needs_lines(self, grid):
        with pytest.raises(TypeError):
            pdist.network.graph_from_lines(grid)


# --------------------------------------------------------------------------- phase 3: hydrology
class TestHydrology:
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_flow_direction_matches_rsense(self, dem, compute):
        terrain = pytest.importorskip("geolattice.rsense.lidar.terrain")
        d = dem.astype("float64").copy()
        d.values[40:45, 60:70] = np.nan
        ref = np.asarray(terrain.flow_direction(d))
        out = pdist.hydrology.flow_direction(d, compute=compute, chunks={"y": 64, "x": 50})
        np.testing.assert_array_equal(np.asarray(out), ref)
        assert out.dtype == np.int8

    def test_accumulation_in_memory(self, dem):
        terrain = pytest.importorskip("geolattice.rsense.lidar.terrain")
        out = pdist.hydrology.flow_accumulation(dem.chunk(100))
        np.testing.assert_allclose(np.asarray(out), np.asarray(terrain.flow_accumulation(dem)))
        filled = pdist.hydrology.fill_depressions(dem.chunk(100))
        assert (np.asarray(filled) >= dem.values - 1e-9).all()


# --------------------------------------------------------------------------- phase 3: scale-out
def _has_java():
    import shutil

    return shutil.which("java") is not None


@pytest.mark.skipif(not _has_java(), reason="needs Java for Spark")
class TestSedona:
    @pytest.fixture(scope="class", autouse=True)
    def session(self):
        pytest.importorskip("sedona")
        pytest.importorskip("pyspark")
        s = pdist.sedona.spark("local[2]", config={"spark.ui.enabled": "false"})
        yield s
        pdist.sedona.stop()

    def test_packages(self):
        assert "sedona-spark-shaded" in pdist.sedona.packages()

    @pytest.mark.parametrize("predicate", ["within", "intersects"])
    def test_sjoin_matches_geopandas(self, points, grid, predicate):
        out = pdist.vector.sjoin(points, grid, predicate=predicate, engine="sedona")
        ref = gpd.sjoin(points, grid, predicate=predicate)
        key = ["id", "index_right"]
        np.testing.assert_array_equal(out[key].sort_values(key).to_numpy(),
                                      ref[key].sort_values(key).to_numpy())
        assert list(out.columns) == list(ref.columns)

    def test_spark_inputs_stay_lazy(self, points, grid):
        sp = pdist.sedona.to_spark(points.iloc[:500])
        out = pdist.vector.sjoin(sp, grid, predicate="within")
        assert pdist.sedona.is_spark(out)
        assert out.count() == len(gpd.sjoin(points.iloc[:500], grid, predicate="within"))


class TestClusterKinds:
    def test_slurm_job_script(self):
        pytest.importorskip("dask_jobqueue")
        with pdist.start("slurm", cores=2, memory="4GB", walltime="00:10:00") as s:
            script = s.cluster.job_script()
            assert "#SBATCH" in script and "00:10:00" in script
            assert s.status()["workers"] == 0

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="kubernetes"):
            pdist.start("mesos")


# --------------------------------------------------------------------------- phase 4: ml
@pytest.fixture(scope="module")
def bands():
    rng = np.random.default_rng(11)
    h, w = 120, 150
    stack = xr.DataArray(rng.random((4, h, w)).astype("float32"), dims=("band", "y", "x"),
                         coords={"band": [1, 2, 3, 4], "y": np.arange(h)[::-1] + 0.5,
                                 "x": np.arange(w) + 0.5})
    stack[:, :5, :5] = np.nan
    X = stack.values.reshape(4, -1).T
    ok = np.isfinite(X).all(1)
    y = np.where(X[:, 0] + X[:, 1] > 1, 20, 10) + (X[:, 2] > 0.5)
    return stack, X, y, ok


class TestML:
    @pytest.mark.parametrize("name", ["rf", "xgb", "lgbm"])
    @pytest.mark.parametrize("compute", COMPUTES)
    def test_fit_predict_raster(self, bands, name, compute):
        pytest.importorskip({"rf": "sklearn", "xgb": "xgboost", "lgbm": "lightgbm"}[name])
        stack, X, y, ok = bands
        rng = np.random.default_rng(0)
        idx = rng.choice(np.flatnonzero(ok), 3000, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = pdist.ml.fit(name, X[idx], y[idx], compute=compute, n_estimators=40)
        pred = pdist.ml.predict_raster(m, stack, chunks={"y": 50, "x": 60})
        assert pred.dims == ("y", "x")
        assert np.isnan(pred.values[:5, :5]).all()
        assert set(np.unique(pred.values[5:, 5:])) <= {10, 11, 20, 21}
        assert (pred.values.ravel()[ok] == y[ok]).mean() > 0.95
        proba = pdist.ml.predict_raster(m, stack, proba=True)
        assert proba.sizes["class"] == 4
        np.testing.assert_allclose(proba.sum("class").values[5:, 5:], 1, atol=1e-4)

    def test_predict_raster_lazy_and_dataset(self, bands):
        stack, X, y, ok = bands
        m = pdist.ml.fit("rf", X[ok][:2000], y[ok][:2000], compute="cpu", n_estimators=10)
        lazy = pdist.ml.predict_raster(m, stack.chunk({"y": 40, "x": 50}))
        assert pdist.is_lazy(lazy)
        ds = stack.to_dataset("band")
        np.testing.assert_array_equal(pdist.ml.predict_raster(m, ds).values, lazy.compute().values)

    def test_xgboost_on_dask_cluster(self, bands):
        pytest.importorskip("xgboost")
        import dask.array as da

        stack, X, y, ok = bands
        with pdist.start("local", n_workers=2, threads_per_worker=1, dashboard=False):
            Xd = da.from_array(X[ok][:3000], chunks=(1000, 4))
            yd = da.from_array(y[ok][:3000], chunks=1000)
            b = pdist.ml.fit("xgb", Xd, yd, compute="cpu", n_estimators=30)
            assert b.geolattice_engine == "xgboost-dask"
            pred = pdist.ml.predict_raster(b, stack).values
        assert (pred.ravel()[ok] == y[ok]).mean() > 0.95


class TestTiledInference:
    @pytest.fixture(scope="class")
    def nets(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)
        one = torch.nn.Conv2d(4, 2, 5, padding=2, padding_mode="reflect")
        two = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3, padding=1, padding_mode="reflect"),
                                  torch.nn.ReLU(),
                                  torch.nn.Conv2d(8, 2, 3, padding=1, padding_mode="reflect"))
        return one, two

    def _ref(self, net, img):
        import torch

        with torch.no_grad():
            return net(torch.from_numpy(img)[None]).numpy()[0]

    @pytest.mark.parametrize("devices", [["cpu"]] + ([["cuda:0"], ["cpu", "cuda:0"]] if HAS_GPU else []))
    def test_crop_matches_whole_image(self, bands, nets, devices):
        img = np.nan_to_num(bands[0].values)
        one, two = nets
        out = pdist.ml.predict_tiles(one, img, tile=48, overlap=8, devices=devices)
        np.testing.assert_allclose(out, self._ref(one, img), atol=1e-5)
        out2 = pdist.ml.predict_tiles(two, img, tile=48, overlap=8, devices=devices)
        np.testing.assert_allclose(out2[:, 2:-2, 2:-2], self._ref(two, img)[:, 2:-2, 2:-2], atol=1e-5)

    def test_cosine_and_dataarray(self, bands, nets):
        img = xr.DataArray(np.nan_to_num(bands[0].values), dims=("band", "y", "x"),
                           coords={"y": bands[0].y, "x": bands[0].x})
        out = pdist.ml.predict_tiles(nets[0], img, tile=48, overlap=16, blend="cosine")
        assert out.dims == ("channel", "y", "x") and out.y.equals(img.y)
        np.testing.assert_allclose(out.values, self._ref(nets[0], img.values), atol=1e-4)

    def test_rejects_odd_overlap(self, bands, nets):
        with pytest.raises(ValueError):
            pdist.ml.predict_tiles(nets[0], bands[0].values, tile=32, overlap=5)


class TestDDP:
    def test_gloo_two_processes(self):
        pytest.importorskip("torch")
        res = pdist.ml.train_ddp(pdist.ml._ddp_selftest, nprocs=2, backend="gloo")
        assert [r["sum"] for r in res] == [3.0, 3.0]
        # DDP averages gradients: (4 * 1 + 4 * 2) / 2 per weight
        assert all(r["grad"] == [[6.0, 6.0]] for r in res)

    @gpu
    def test_nccl_default(self):
        res = pdist.ml.train_ddp(pdist.ml._ddp_selftest)
        assert res[0]["backend"] == "nccl" and res[0]["sum"] == float(_backend.gpu_count() * (_backend.gpu_count() + 1) / 2)


# --------------------------------------------------------------------------- phase 4: diagnostics
class TestDiagnostics:
    def test_benchmark(self, dem):
        df = pdist.benchmark(pdist.raster.slope, dem, repeats=2, chunks={"y": 100, "x": 100})
        assert list(df.index) == COMPUTES
        assert (df["best_s"] > 0).all() and df["speedup"].iloc[0] == 1.0

    def test_check_equivalence(self, dem, points, blobs):
        rep = pdist.check_equivalence(pdist.raster.slope, dem, rtol=1e-4, atol=1e-3)
        assert rep.ok, rep.details
        rep = pdist.check_equivalence(pdist.vector.sjoin, points, blobs, predicate="within")
        assert rep.ok, rep.details
        rep = pdist.check_equivalence(pdist.analysis.morans_i, np.arange(625.0),
                                      pdist.analysis.weights(gpd.GeoDataFrame(
                                          geometry=[shapely.box(i, j, i + 1, j + 1)
                                                    for j in range(25) for i in range(25)]), "queen"),
                                      permutations=0)
        assert rep.ok, rep.details

    def test_check_equivalence_detects_mismatch(self, dem):
        if not HAS_GPU:
            pytest.skip("needs a GPU to compare")

        def biased(x, compute):
            out = pdist.raster.slope(x, compute=compute)
            return out + (1.0 if compute == "gpu" else 0.0)

        rep = pdist.check_equivalence(biased, dem)
        assert not rep.ok and rep.max_abs == pytest.approx(1.0, rel=1e-3)
        with pytest.raises(AssertionError):
            pdist.check_equivalence(biased, dem, raise_on_mismatch=True)

    def test_explain(self):
        e = pdist.explain("vector.sjoin", "gpu")
        assert e["uses"] == ("gpu" if HAS_GPU else "cpu") and e["targets"] == ["cpu", "gpu"]
        assert pdist.explain("bridge.polygonize", "gpu")["reason"] == "no GPU implementation registered"
        assert pdist.explain("bridge.rasterize", "auto")["uses"] == "cpu"
        assert pdist.explain("bridge.rasterize", "gpu")["uses"] == ("gpu" if HAS_GPU else "cpu")
        assert pdist.dispatch("analysis.dbscan", "auto")[1] == "cpu"
        with pytest.raises(KeyError, match="did you mean"):
            pdist.explain("sjoin")

    def test_memory_report(self):
        rep = pdist.memory_report()
        assert "process_rss_gb" in rep or "host_total_gb" in rep
        if HAS_GPU:
            assert rep["gpu_total_gb"] > 0

    def test_profile_local(self, dem):
        res, table = pdist.profile(pdist.raster.slope, dem.chunk({"y": 100, "x": 100}))
        assert res.shape == dem.shape and len(table) > 0

    def test_profile_session(self, dem, tmp_path):
        pytest.importorskip("bokeh")
        with pdist.start("local", n_workers=1, threads_per_worker=2, dashboard=False):
            res, path = pdist.profile(pdist.raster.slope, dem.chunk({"y": 100, "x": 100}),
                                      path=str(tmp_path / "p.html"))
        assert res.shape == dem.shape and (tmp_path / "p.html").stat().st_size > 1000


# --------------------------------------------------------------------------- remaining branches
class TestBackendHelpers:
    def test_compute_and_persist(self, dem):
        a = dem.chunk({"y": 100, "x": 100})
        one = pdist.compute(a)
        assert not pdist.is_lazy(one)
        x, y = pdist.compute(a, a * 2)
        np.testing.assert_allclose(y.values, 2 * x.values)
        kept = pdist.persist(a)
        assert pdist.is_lazy(kept)
        k1, k2 = pdist.persist(a, a + 1)
        assert pdist.is_lazy(k2)

    def test_star_import_does_not_shadow_stdlib(self):
        ns = {}
        exec("import io as _io; io = _io; from geolattice.distributed import *", ns)
        import io as stdlib_io

        assert ns["io"] is stdlib_io
        assert "raster" not in ns and "start" in ns
        for sub in pdist.__submodules__:  # documented and importable, just not star-exported
            assert hasattr(pdist, sub)

    @gpu
    def test_capabilities_gpu_memory(self):
        caps = pdist.capabilities()
        assert caps["gpu_memory_total"] >= caps["gpu_memory_free"] > 0


class TestSessionKinds:
    def test_scale_adapt_status(self):
        with pdist.start("local", n_workers=1, threads_per_worker=1, dashboard=False) as s:
            s.scale(2)
            s.client.wait_for_workers(2, timeout=60)
            assert pdist.status()["workers"] == 2
            s.adapt(minimum=1, maximum=2)
            assert pdist.dashboard_link() == s.dashboard_link
            assert "kind='local'" in repr(s)
        assert pdist.status()["kind"] == "threads" and pdist.dashboard_link() is None

    def test_connect_by_address(self):
        from dask.distributed import LocalCluster

        with LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None) as cluster:
            with pdist.start("address", address=cluster.scheduler_address) as s:
                assert s.status()["workers"] == 1
                with pytest.raises(RuntimeError):
                    s.scale(2)
                with pytest.raises(RuntimeError):
                    s.adapt()
        with pytest.raises(ValueError, match="address"):
            pdist.start("address")

    def test_start_replaces_active_session(self):
        a = pdist.start("local", n_workers=1, threads_per_worker=1, dashboard=False)
        b = pdist.start("local", n_workers=1, threads_per_worker=1, dashboard=False)
        assert pdist.get_session() is b and a.client.status == "closed"
        pdist.shutdown()
        assert pdist.get_session() is None

    @gpu
    def test_cuda_cluster(self, dem):
        pytest.importorskip("dask_cuda")
        with pdist.start("cuda", dashboard=False) as s:
            assert s.is_gpu and s.status()["workers"] == _backend.gpu_count()
            out = pdist.raster.slope(dem.chunk({"y": 100, "x": 100}), compute="gpu").compute()
        assert out.shape == dem.shape

    def test_pbs_job_script(self):
        pytest.importorskip("dask_jobqueue")
        with pdist.start("pbs", n_workers=0, cores=1, memory="1GB", walltime="00:05:00") as s:
            assert "#PBS" in s.cluster.job_script()


class TestIOBranches:
    def test_glob_and_list_paths(self, points, tmp_path):
        pdist.write_vector(points, tmp_path / "pts", npartitions=3, spatial=False)
        by_glob = pdist.read_vector(str(tmp_path / "pts" / "*.parquet"), columns=["id", "geometry"])
        files = sorted(str(p) for p in (tmp_path / "pts").glob("*.parquet"))
        by_list = pdist.read_vector(files)
        assert len(by_glob) == len(by_list) == len(points)
        with pytest.raises(FileNotFoundError):
            pdist.read_vector(str(tmp_path / "missing" / "*.parquet"))

    def test_single_parquet_and_bbox(self, points, tmp_path):
        f = pdist.write_vector(points, tmp_path / "one.parquet")
        assert f.endswith("one.parquet")
        sub = pdist.read_vector(f, bbox=(0, 0, 10, 10))
        assert 0 < len(sub) < len(points)

    def test_several_ogr_files_and_bbox(self, points, tmp_path):
        a, b = tmp_path / "a.gpkg", tmp_path / "b.gpkg"
        points.iloc[:1000].to_file(a)
        points.iloc[1000:].to_file(b)
        both = pdist.read_vector([a, b])
        assert len(both) == len(points)
        boxed = pdist.read_vector(a, bbox=(0, 0, 20, 20))
        assert len(boxed) == len(gpd.read_file(a, bbox=(0, 0, 20, 20)))

    def test_lazy_ogr(self, points, tmp_path):
        pytest.importorskip("dask_geopandas")
        f = tmp_path / "p.gpkg"
        points.to_file(f)
        lazy = pdist.read_vector(f, lazy=True, npartitions=2)
        assert pdist.is_lazy(lazy) and len(lazy.compute()) == len(points)
        with pytest.raises(ValueError):
            pdist.read_vector([f, f], lazy=True)

    def test_cog_and_gpu_read(self, dem, tmp_path):
        pytest.importorskip("rioxarray")
        from geolattice.web.cog import validate_cog

        src = dem.rio.write_crs("EPSG:32633")
        cog = pdist.write_raster(src, tmp_path / "dem_cog.tif", format="cog")
        assert validate_cog(cog).valid
        da = pdist.read_raster(cog, compute=COMPUTES[-1])
        assert pdist.is_lazy(da)
        np.testing.assert_allclose(pdist.to_cpu(da).values, dem.values)
        with pytest.raises(ValueError):
            pdist.write_raster(src, tmp_path / "x.img", format="hfa")

    def test_read_stac_local_items(self, dem, tmp_path):
        pystac = pytest.importorskip("pystac")
        pytest.importorskip("odc.stac")
        pytest.importorskip("rioxarray")
        src = dem.rio.write_crs("EPSG:32633")
        tif = pdist.write_raster(src, tmp_path / "band.tif", format="gtiff")
        t = src.rio.transform()
        bbox = [float(v) for v in src.rio.bounds()]
        import datetime as _dt

        item = pystac.Item("dem", geometry=None, bbox=bbox, datetime=_dt.datetime(2024, 1, 1),
                           properties={"proj:epsg": 32633, "proj:shape": list(dem.shape),
                                       "proj:transform": [t.a, t.b, t.c, t.d, t.e, t.f]},
                           stac_extensions=["https://stac-extensions.github.io/projection/v1.1.0/schema.json"])
        item.add_asset("elev", pystac.Asset(href=str(tif), media_type=pystac.MediaType.COG,
                                            roles=["data"]))
        ds = pdist.read_stac([item], ["elev"], chunks={"x": 128, "y": 128})
        assert pdist.is_lazy(ds)
        np.testing.assert_allclose(ds["elev"].isel(time=0).values, dem.values, rtol=1e-6)


class TestErrorPaths:
    def test_raster_errors(self, dem):
        with pytest.raises(ValueError):
            pdist.raster.slope(np.zeros(5), res=1.0)
        with pytest.raises(ValueError):
            pdist.raster.calc("a")
        with pytest.raises(ValueError):
            pdist.raster.reclassify(dem)
        with pytest.raises(ValueError):
            pdist.raster.reclassify(dem, bins=[1, 2], values=[1, 2])
        with pytest.raises(ValueError):
            pdist.raster.normalize(dem, method="robust")
        with pytest.raises(ValueError):
            pdist.raster.convolve(dem, np.ones((2, 2)))
        with pytest.raises(ValueError):
            pdist.raster.tpi(dem, window=4)
        with pytest.raises(ValueError):
            pdist.raster.slope(dem.isel(x=[0]))

    def test_raster_variants(self, dem):
        pct = pdist.raster.slope(dem, units="percent")
        rad = pdist.raster.slope(dem, units="radians")
        np.testing.assert_allclose(np.tan(rad.values) * 100, pct.values, rtol=1e-4, atol=1e-3)
        flat = xr.DataArray(np.ones((20, 20)), dims=("y", "x"),
                            coords={"y": np.arange(20.0)[::-1], "x": np.arange(20.0)})
        assert np.all(pdist.raster.aspect(flat, flat=-1) == -1)
        out = pdist.raster.map_tiles(dem, lambda b, xp: b * 2, chunks=64)
        np.testing.assert_allclose(out.values, dem.values * 2)
        np.testing.assert_allclose(pdist.raster.focal(dem.values, "sum", size=3, chunks=50),
                                   ndi.uniform_filter(dem.values.astype("float32"), 3, mode="nearest") * 9,
                                   rtol=1e-4, atol=1e-2)
        conv = pdist.raster.convolve(dem.values, np.ones((3, 3)), normalize_kernel=True)
        np.testing.assert_allclose(conv, ndi.uniform_filter(dem.values, 3, mode="nearest"), atol=1e-2)

    def test_geographic_dem_warns(self):
        pytest.importorskip("rioxarray")
        geo = xr.DataArray(np.random.default_rng(0).random((30, 30)), dims=("y", "x"),
                           coords={"y": np.linspace(45, 44, 30), "x": np.linspace(10, 11, 30)})
        geo = geo.rio.write_crs("EPSG:4326")
        with pytest.warns(UserWarning, match="geographic"):
            pdist.raster.slope(geo)

    def test_vector_errors(self, points, grid):
        with pytest.raises(TypeError):
            pdist.vector.buffer(points, np.ones(len(points)))
        with pytest.raises(ValueError):
            pdist.vector.to_crs(points.set_crs(None, allow_override=True), 4326)
        with pytest.raises(ValueError):
            pdist.vector.sjoin(points, grid, how="right")
        with pytest.raises(ValueError):
            pdist.vector.sjoin(points, grid, predicate="near")
        with pytest.raises(ValueError):
            pdist.vector.sjoin(points, grid, predicate="dwithin")
        with pytest.raises(ValueError):
            pdist.vector.sjoin(points, grid, engine="ray")
        with pytest.raises(ValueError):
            pdist.vector.knn(points, k=0)
        with pytest.raises(ValueError, match="CRS"):
            pdist.vector.knn(points, points.to_crs(4326))

    def test_vector_variants(self, points, grid):
        gs = points.geometry
        assert pdist.vector.centroid(gs).equals(gs.centroid)
        clipped = pdist.vector.clip(gs, (0, 0, 30, 30))
        assert len(clipped) == len(gpd.clip(gs, (0, 0, 30, 30)))
        assert len(pdist.vector.clip(points, grid.iloc[:5])) == len(gpd.clip(points, grid.iloc[:5]))
        assert len(pdist.vector.clip(points, (500, 500, 600, 600))) == 0
        far = gpd.GeoDataFrame({"v": [1]}, geometry=[shapely.box(500, 500, 501, 501)], crs=points.crs)
        empty = pdist.vector.sjoin(points, far)
        assert len(empty) == 0 and "index_right" in empty.columns
        one = pdist.vector.sjoin(points.iloc[:10], grid, npartitions=1)
        assert len(one) == len(gpd.sjoin(points.iloc[:10], grid))

    def test_bridge_errors(self, blobs, field):
        with pytest.raises(TypeError):
            pdist.rasterize(blobs, field.values)
        with pytest.raises(ValueError):
            pdist.rasterize(blobs, field, values=[1, 2])
        with pytest.raises(ValueError):
            pdist.zonal_stats(blobs, field, stats=("median",))
        with pytest.raises(TypeError):
            pdist.zonal_stats(blobs, field.values)
        with pytest.raises(ValueError):
            pdist.zonal_stats(blobs, field.expand_dims(a=1, b=1))
        with pytest.raises(ValueError, match="CRS"):
            pdist.zonal_stats(blobs.to_crs(4326), field)
        with pytest.raises(ValueError):
            pdist.extract_values(blobs.assign(geometry=blobs.centroid), field, method="cubic")
        with pytest.raises(TypeError):
            pdist.polygonize(field.values)
        with pytest.raises(ValueError):
            pdist.polygonize(field, connectivity=6)

    def test_bridge_variants(self, blobs, field):
        burned = pdist.rasterize(blobs, field, column="z", dtype="int32", fill=-1)
        assert burned.dtype == np.int32 and int(burned.max()) <= blobs["z"].max()
        plain = pdist.rasterize(blobs, field.values if False else field)
        assert set(np.unique(plain.values)) <= {0.0, 1.0}
        lines = gpd.GeoDataFrame(geometry=[shapely.LineString([(5, 5), (90, 90)])], crs=blobs.crs)
        assert float(pdist.rasterize(lines, field, compute="gpu").sum()) > 0
        stack = field.expand_dims(band=[1, 2])
        ev = pdist.extract_values(blobs.assign(geometry=blobs.centroid), stack, name="v")
        assert {"v_1", "v_2"} <= set(ev.columns)
        cls = xr.DataArray(np.array([[1, 1, 0], [0, 2, 2], [9, 9, 9]], dtype="int64"), dims=("y", "x"),
                           coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]})
        out = pdist.polygonize(cls, nodata=9)
        assert 9 not in set(out["value"]) and np.isclose(out.area.sum(), 6)
        boolean = pdist.polygonize(cls > 0)
        assert set(boolean["value"]) <= {0, 1}
        floats = pdist.polygonize(cls.astype("float64") / 2)
        assert np.isclose(floats.area.sum(), 9)


class TestAnalysisBranches:
    def test_weights_variants(self, points, lattice):
        idw = pdist.analysis.weights(points.iloc[:300], "distance", threshold=8.0, binary=False,
                                     row_standardize=False)
        assert idw.sparse.data.max() > 1 / 8 - 1e-9
        knn_poly = pdist.analysis.weights(lattice[0], "knn", k=4, row_standardize=False)
        assert set(knn_poly.cardinalities) == {4}
        lone = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1), shapely.box(5, 5, 6, 6)])
        w = pdist.analysis.weights(lone, "queen")
        assert list(w.islands) == [0, 1]
        with pytest.raises(ValueError):
            pdist.analysis.weights(points, "distance")
        with pytest.raises(ValueError):
            pdist.analysis.weights(points, "triangle")

    def test_weights_conversions(self, lattice):
        pytest.importorskip("libpysal")
        w = pdist.analysis.weights(lattice[0], "rook", row_standardize=False)
        lw = w.to_libpysal()
        assert lw.n == w.n and lw.s0 == pytest.approx(w.sparse.sum())
        if HAS_GPU:
            assert w.to_gpu().shape == (w.n, w.n)

    def test_stat_errors_and_no_permutations(self, lattice):
        g, y = lattice
        w = pdist.analysis.weights(g, "queen")
        with pytest.raises(ValueError, match="values"):
            pdist.analysis.morans_i(y[:-1], w)
        r = pdist.analysis.morans_i(y, w, permutations=0)
        assert r.p_sim is None and r.sims is None
        assert pdist.analysis.morans_i(y, w, permutations=99, keep_simulations=True).sims.shape == (99,)
        assert pdist.analysis.gearys_c(y, w, permutations=0).p_sim is None
        assert "p_sim" not in pdist.analysis.local_morans(y, w, permutations=0)

    def test_clustering_arrays(self):
        rng = np.random.default_rng(12)
        x = np.vstack([rng.normal(0, 0.3, (100, 3)), rng.normal(5, 0.3, (100, 3))])
        lab = pdist.analysis.dbscan(x, eps=1.0, min_samples=5, compute="cpu")
        assert isinstance(lab, np.ndarray) and len(set(lab) - {-1}) == 2
        lab1, centers = pdist.analysis.kmeans(x[:, 0], 2, compute="cpu")
        assert centers.shape == (2, 1)
        if HAS_GPU:
            g = pdist.analysis.dbscan(x, eps=1.0, min_samples=5, compute="gpu")
            assert len(set(g) - {-1}) == 2

    def test_kde_errors_and_empty_chunks(self):
        like = xr.DataArray(np.zeros((40, 40)), dims=("y", "x"),
                            coords={"y": np.arange(40)[::-1] + 0.5, "x": np.arange(40) + 0.5})
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([2.0], [2.0]))
        out = pdist.analysis.kde(pts, like, bandwidth=1.0, kernel="quartic", chunks={"y": 10, "x": 10})
        assert float(out.values[:10, :10].sum()) == 0 and float(out.max()) > 0
        lazy = pdist.analysis.kde(pts, like.chunk(20), bandwidth=1.0)
        assert pdist.is_lazy(lazy)
        with pytest.raises(ValueError):
            pdist.analysis.kde(pts, like, bandwidth=1.0, kernel="epanechnikov")


class TestNetworkBranches:
    def test_directed_cutoff_tolerance(self):
        lines = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (1, 0)]),
                                           shapely.LineString([(1.0004, 0), (2, 0)]),
                                           shapely.MultiLineString([[(2, 0), (3, 0)], [(3, 0), (4, 0)]])])
        loose = pdist.network.graph_from_lines(lines)
        snapped = pdist.network.graph_from_lines(lines, tolerance=0.01)
        assert snapped.n_nodes == loose.n_nodes - 1
        dg = pdist.network.graph_from_lines(lines, tolerance=0.01, directed=True)
        fwd = pdist.network.shortest_path_lengths(dg, 0)
        back = pdist.network.shortest_path_lengths(dg, dg.n_nodes - 1)
        assert np.isfinite(fwd).all() and np.isinf(back[0][:-1]).all()
        assert np.isinf(pdist.network.shortest_path_lengths(snapped, 0, cutoff=1.5)[0]).sum() >= 2
        assert pdist.network.shortest_path(snapped, 0, 0)["geometry"].geom_type == "Point"
        assert len(np.unique(pdist.network.components(dg))) == 1
        for c in COMPUTES:
            assert len(np.unique(pdist.network.components(dg, compute=c))) == 1
        with pytest.raises(ValueError):
            pdist.network.centrality(snapped, "eigen")


class TestHydrologyBranches:
    def test_integer_dem_and_numpy(self):
        z = (np.add.outer(np.arange(50), np.arange(60)) * 3).astype("int32")[::-1]
        fd = pdist.hydrology.flow_direction(z, res=10.0)
        assert fd.dtype == np.int8 and isinstance(fd, np.ndarray)
        filled = pdist.hydrology.fill_depressions(z.astype("float64"))
        assert filled.shape == z.shape


class TestSedonaBranches:
    @pytest.fixture(scope="class", autouse=True)
    def session(self):
        if not _has_java():
            pytest.skip("needs Java")
        pytest.importorskip("sedona")
        s = pdist.sedona.spark("local[2]", config={"spark.ui.enabled": "false"})
        yield s
        pdist.sedona.stop()

    def test_left_dwithin_and_roundtrip(self, points, grid):
        sub = points.iloc[:300]
        out = pdist.sedona.sjoin(sub, grid.iloc[:10], how="left", predicate="within")
        assert len(out) == len(gpd.sjoin(sub, grid.iloc[:10], how="left", predicate="within"))
        cents = grid.assign(geometry=grid.centroid)
        dw = pdist.sedona.sjoin(sub, cents, predicate="dwithin", distance=2.0)
        assert len(dw) == len(gpd.sjoin(sub, cents, predicate="dwithin", distance=2.0))
        back = pdist.sedona.to_geopandas(pdist.sedona.to_spark(sub), crs=sub.crs)
        assert len(back) == len(sub) and back.crs == sub.crs
        both = pdist.vector.sjoin(pdist.sedona.to_spark(sub), pdist.sedona.to_spark(grid),
                                  predicate="intersects", how="left")
        assert pdist.sedona.is_spark(both)

    def test_errors(self, points, grid):
        with pytest.raises(ValueError):
            pdist.sedona.sjoin(points, grid, how="outer")
        with pytest.raises(ValueError):
            pdist.sedona.sjoin(points, grid, predicate="near")
        with pytest.raises(ValueError):
            pdist.sedona.sjoin(points, grid, predicate="dwithin")
        with pytest.raises(ValueError, match="CRS"):
            pdist.sedona.sjoin(points, grid.to_crs(4326))

    def test_packages_override(self, monkeypatch):
        monkeypatch.setenv("GEOLATTICE_SEDONA_PACKAGES", "a:b:1")
        assert pdist.sedona.packages() == "a:b:1"


class TestMLBranches:
    def test_sklearn_estimator_on_cluster(self, bands):
        from sklearn.ensemble import ExtraTreesClassifier

        stack, X, y, ok = bands
        with pdist.start("local", n_workers=2, threads_per_worker=1, dashboard=False):
            m = pdist.ml.fit(ExtraTreesClassifier(n_estimators=20, n_jobs=-1), X[ok][:2000], y[ok][:2000])
        assert m.geolattice_engine == "sklearn"
        assert (pdist.ml.predict_raster(m, stack).values.ravel()[ok] == y[ok]).mean() > 0.9

    def test_regression_and_errors(self, bands):
        stack, X, y, ok = bands
        reg = pdist.ml.fit("rf", X[ok][:2000], X[ok][:2000, 0] * 2, task="regression", compute="cpu",
                           n_estimators=20)
        pred = pdist.ml.predict_raster(reg, stack)
        assert np.nanmax(np.abs(pred.values - 2 * stack.values[0])) < 0.3
        with pytest.raises(ValueError):
            pdist.ml.fit("svm", X[:10], y[:10])
        with pytest.raises(ValueError):
            pdist.ml.predict_raster(reg, stack.values)
        with pytest.raises(ValueError, match="classes"):
            pdist.ml.predict_raster(reg, stack, proba=True)

    def test_xgboost_dask_multiclass_proba(self, bands):
        pytest.importorskip("xgboost")
        import dask.array as da

        stack, X, y, ok = bands
        with pdist.start("local", n_workers=1, threads_per_worker=2, dashboard=False):
            b = pdist.ml.fit("xgb", da.from_array(X[ok][:2000], chunks=(500, 4)),
                             da.from_array(y[ok][:2000], chunks=500), compute="cpu", n_estimators=20)
            proba = pdist.ml.predict_raster(b, stack, proba=True)
        assert proba.sizes["class"] == 4
        np.testing.assert_allclose(proba.sum("class").values[5:, 5:], 1, atol=1e-4)

    @gpu
    def test_predict_tiles_amp_and_auto(self, bands):
        torch = pytest.importorskip("torch")
        net = torch.nn.Conv2d(4, 2, 3, padding=1, padding_mode="reflect")
        img = np.nan_to_num(bands[0].values)
        out = pdist.ml.predict_tiles(net, img, tile=48, overlap=8, devices="auto", amp=True)
        with torch.no_grad():
            ref = net(torch.from_numpy(img)[None]).numpy()[0]
        np.testing.assert_allclose(out, ref, atol=5e-2)
        assert next(net.parameters()).device.type == "cpu"
        single = pdist.ml.predict_tiles(net, img[0:1].repeat(4, 0)[0], tile=32, overlap=4) \
            if False else pdist.ml.predict_tiles(torch.nn.Conv2d(1, 1, 1), img[0], tile=32, overlap=4)
        assert single.shape == (1,) + img.shape[1:]


class TestDiagnosticsBranches:
    def test_compare_structures(self):
        from geolattice.distributed.diagnostics import _compare

        out = []
        _compare({"a": np.ones(3), "b": [1, 2]}, {"a": np.ones(3), "b": [1, 2]}, 1e-6, 0, "x", out)
        _compare(None, None, 1e-6, 0, "n", out)
        _compare(np.array(["a", "b"]), np.array(["a", "b"]), 1e-6, 0, "s", out)
        assert all(o[1] for o in out)
        bad = []
        _compare(np.ones(3), np.ones(4), 1e-6, 0, "shape", bad)
        _compare(pd_frame(a=[1]), pd_frame(b=[1]), 1e-6, 0, "cols", bad)
        _compare(np.array([np.nan]), np.array([1.0]), 1e-6, 0, "nan", bad)
        g = gpd.GeoDataFrame(geometry=[shapely.Point(0, 0)])
        _compare(g, gpd.GeoDataFrame(geometry=[shapely.Point(0, 0), shapely.Point(1, 1)]), 1e-6, 0, "len", bad)
        assert not any(o[1] for o in bad)

    def test_equivalence_on_geodataframe(self, points, grid):
        rep = pdist.check_equivalence(pdist.vector.area, grid)
        assert rep.ok
        rep = pdist.check_equivalence(pdist.vector.sjoin, points.iloc[:500], grid, computes=("cpu", "cpu"))
        assert rep.ok and rep.max_abs == 0

    def test_benchmark_cpu_only(self, dem):
        df = pdist.benchmark(pdist.raster.slope, dem, computes=("cpu",), repeats=1, warmup=0)
        assert list(df.index) == ["cpu"] and df["speedup"].iloc[0] == 1.0

    def test_profile_default_path(self, dem, tmp_path, monkeypatch):
        pytest.importorskip("bokeh")
        monkeypatch.chdir(tmp_path)
        with pdist.start("local", n_workers=1, threads_per_worker=1, dashboard=False):
            _, path = pdist.profile(pdist.raster.slope, dem.chunk({"y": 150, "x": 130}))
        assert (tmp_path / path).exists()

    def test_explain_defaults(self):
        e = pdist.explain("raster.focal")
        assert e["requested"] == "auto" and e["session"] == "local threads"
        assert pdist.explain("raster.focal", "cpu")["reason"] == "CPU requested"


def pd_frame(**cols):
    import pandas as pd

    return pd.DataFrame(cols)
