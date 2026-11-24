"""Build 08_11_diagnostics_and_performance.ipynb: ``python _builders/run.py 11``."""

from nb import Notebook

nb = Notebook("08_11_diagnostics_and_performance.ipynb")

nb.md("""
# Measuring, Checking and Tuning Distributed and GPU Workloads

A fast wrong answer is worse than a slow right one, and a "speed-up" measured badly is neither.
This notebook collects the habits that kept the rest of the chapter honest: checking that CPU and
GPU give the same answer, timing what actually happens (including the GPU's asynchronous queue and
Dask's laziness), measuring strong and weak scaling, finding where the time goes with a profiler,
and recognising the common mistakes. Every pitfall in the last section actually happened while
this chapter was written, and each is reproduced and fixed with `geolattice.distributed.diagnostics`.
""")

nb.md("""
## Overview

### Correctness first

GPU kernels, chunked algorithms and parallel reductions all compute in a different order from the
plain whole-array version, so results can differ by rounding, and bugs show up at chunk edges, at
NaN values or with unusual geometries. Every operation should be checked against a reference on
real data before it is trusted at scale: exactly for integers and geometries, within a tolerance
for floating point.

### Timing correctly

- **Laziness**: a Dask call returns at once; time the `.compute()` (or the write) instead.
- **GPU queues**: CuPy launches kernels asynchronously; synchronise the device before stopping the
  clock.
- **Warm-up**: the first call compiles kernels, fills caches and loads files into the page cache;
  time later calls, or report the first separately.
- **Repeats**: report the best or the median of several runs.

### Strong and weak scaling

- **Strong scaling**: fixed problem, more workers. Ideal: time falls as `1/n`; Amdahl's law
  (notebook 00) describes the shortfall.
- **Weak scaling**: problem grows with the workers. Ideal: constant time. Weak scaling is what
  matters when the reason for more workers is more data.

### Overheads that eat speed-ups

Task scheduling (about a millisecond per task on a cluster), serialization between processes,
host-to-device copies over PCIe, Python code that holds the GIL, and reading the same data twice.
""")

nb.md("""
## Diagnostics with GeoLattice

| Function | Does |
|---|---|
| `pdist.check_equivalence(func, *args, rtol=, atol=)` | runs `func` with `compute="cpu"` and `"gpu"` and compares arrays, DataArrays, tables, GeoDataFrames and result objects |
| `pdist.benchmark(func, *args, computes=, repeats=, warmup=)` | best, mean and spread of the run time per target, lazy results materialised and the GPU synchronised |
| `pdist.explain(op, compute=)` | which implementation an operation would use, and why |
| `pdist.memory_report()` | host, process, GPU and worker memory |
| `pdist.profile(func, *args, path=)` | a per-task timing table (local scheduler) or a Dask performance report in HTML (with a session) |

### Learning objectives

1. Check every GPU implementation used in the chapter against the CPU on real data.
2. Time lazy and GPU computations correctly with `pdist.benchmark`.
3. Measure strong and weak scaling and interpret them.
4. Profile a computation and find its most expensive tasks.
5. Recognise and fix seven common pitfalls.

### Data

| Dataset | Content |
|---|---|
| Great Plains DEM and counties | 90 m DEM (262 million pixels); 225 counties |
| Kansas buildings | 1.6 million footprints |

### Contents

- Setup
- CPU and GPU agree: one check per operation
- Benchmarks done right
- Strong and weak scaling
- Where the time goes: profiling
- Seven pitfalls, reproduced and fixed
- Summary and exercises
""")

nb.setup('''
import dask  # Dask configuration
import dask.array as da  # chunked arrays


def sync():  # wait for queued GPU work
    if caps["gpus"]:  # only with a GPU
        import cupy as cp  # GPU arrays
        cp.cuda.Device().synchronize()  # block until done


dem = dd.gp_raster("elev").compute()  # the DEM in memory
win = dem[6000:8048, 6000:8048].fillna(0)  # a 2048 x 2048 window
counties = dd.gp_counties()  # 225 counties
bld = pdist.vector.to_crs(dd.kansas_buildings(), 5070)  # 1.6 million buildings
pts = bld.set_geometry(bld.geometry.representative_point())  # one point per building
ks = counties[counties.state == "KS"][["FIPS", "geometry"]]  # Kansas counties
''')

nb.md("""
## CPU and GPU agree: one check per operation

`pdist.check_equivalence` runs a function on both targets and compares everything it returns. The
table covers every operation with a GPU implementation used in this chapter, each on real data.
""")

nb.code('''
s200 = pts.sample(20_000, random_state=0)  # a sample for the point statistics
w8 = pdist.analysis.weights(s200, "knn", k=8)  # weights for the statistics
y8 = np.log(pdist.vector.area(bld.loc[s200.index])).values  # log building area
grid = xr.DataArray(np.zeros((300, 300), np.float32), dims=("y", "x"),  # 1 km template around Wichita
                    coords={"y": 1_640_000 - 500 - 1000 * np.arange(300), "x": -250_000 + 500 + 1000 * np.arange(300)}).rio.write_crs(5070)
checks = {  # name: (function, arguments, keyword arguments)
    "raster.slope": (pdist.raster.slope, (win,), {}),
    "raster.hillshade": (pdist.raster.hillshade, (win,), {}),
    "raster.curvature": (pdist.raster.curvature, (win,), {}),
    "raster.focal median 9": (pdist.raster.focal, (win, "median"), {"size": 9}),
    "raster.focal std 9": (pdist.raster.focal, (win, "std"), {"size": 9}),
    "raster.reclassify": (pdist.raster.reclassify, (win,), {"bins": [500, 1000, 1500]}),
    "hydrology.flow_direction": (pdist.hydrology.flow_direction, (win,), {}),
    "vector.area": (pdist.vector.area, (bld.iloc[:200_000],), {}),
    "vector.bounds": (pdist.vector.bounds, (bld.iloc[:200_000],), {}),
    "vector.sjoin points in counties": (pdist.vector.sjoin, (pts, ks), {"predicate": "within"}),
    "vector.knn k=5": (pdist.vector.knn, (s200,), {"k": 5}),
    "bridge.zonal_stats": (pdist.zonal_stats, (counties, dd.gp_raster("ndvi")), {"stats": ["count", "mean", "std"]}),
    "bridge.extract_values": (pdist.extract_values, (dd.gp_points(), dd.gp_raster("preci")), {}),
    "bridge.rasterize": (pdist.rasterize, (counties, dd.gp_raster("elev")[::10, ::10].compute()), {"column": "x"}),
    "analysis.morans_i": (pdist.analysis.morans_i, (y8, w8), {"permutations": 0}),
    "analysis.getis_ord_local": (pdist.analysis.getis_ord_local, (y8, w8), {}),
    "analysis.kde": (pdist.analysis.kde, (s200, grid), {"bandwidth": 3000}),
}
rows = []  # one row per check
for name, (fn, args, kw) in checks.items():  # each operation
    tol = dict(rtol=1e-4, atol=1e-3) if name.startswith(("raster", "analysis.kde")) else dict(rtol=1e-6, atol=1e-3)  # float32 rasters; float64 tables (areas in m2)
    rep = pdist.check_equivalence(fn, *args, **tol, **kw)  # CPU vs GPU
    rows.append({"operation": name, "equivalent": rep.ok, "largest |GPU - CPU|": rep.max_abs, "parts compared": len(rep.details)})  # row
eq = pd.DataFrame(rows).set_index("operation")  # table
print("all equivalent:", bool(eq.equivalent.all()))  # verdict
eq  # show
''')

nb.md("""
Every operation passed. Integer and geometry outputs (flow direction, reclassification, join
pairs, neighbour indices) were identical; floating point outputs differed by rounding at most.
Statistics with permutations are compared without them here, since the two targets draw
different random numbers.

## Benchmarks done right

The cell below times the terrain ruggedness index (TRI, not used earlier in this notebook) on the
GPU in three misleading ways, then with `pdist.benchmark`.
""")

nb.code('''
if caps["gpus"]:  # needs a GPU
    g = pdist.to_gpu(win)  # data on the device
    t = time.perf_counter(); r = pdist.raster.tri(g); sync(); wrong1 = time.perf_counter() - t  # first call of TRI: includes kernel compilation
    t = time.perf_counter(); r = pdist.raster.tri(g); wrong2_launch = time.perf_counter() - t  # no sync: may return before the kernels finish
    t = time.perf_counter(); lz = pdist.raster.tri(dd.gp_raster("elev")); wrong3 = time.perf_counter() - t  # lazy: nothing ran
    bench = pdist.benchmark(pdist.raster.tri, win, computes=("cpu", "gpu"), repeats=5, warmup=1)  # warm-up, repeats, sync
    print(f"first call {wrong1 * 1000:.0f} ms | unsynchronised {wrong2_launch * 1000:.1f} ms | lazy call {wrong3 * 1000:.1f} ms")  # misleading numbers
    print(bench.round(4).to_string())  # the right numbers
''')

nb.md("""
The first call is the slowest because CuPy compiles TRI's kernels then; an unsynchronised call can
stop the clock before the queued kernels have finished (here the Dask machinery around each call
dominates, so the error is small); and the lazy call measured only building a task graph for the
whole DEM, not computing it. `pdist.benchmark`
avoids all three: it runs a warm-up, materialises lazy results, synchronises the GPU and repeats.

## Strong and weak scaling

Strong scaling: slope of the full DEM, zonal statistics of NDVI by county and the point-in-county
join, each with 1 to 24 threads. Weak scaling: slope with the DEM window growing in proportion to
the threads, so each thread always has about 11 million pixels.
""")

nb.code('''
ndvi = dd.gp_raster("ndvi").compute()  # NDVI in memory, so reading does not dominate
work = {"slope (raster, memory-bound)": lambda: pdist.raster.slope(dem, compute="cpu", chunks=(1024, 1024)),  # three workloads
        "focal median 5 (raster, compute-bound)": lambda: pdist.raster.focal(dem[:6000, :6000], "median", size=5, compute="cpu", chunks=(750, 750)),
        "points in counties (vector join)": lambda: pdist.vector.sjoin(pts, ks, predicate="within", compute="cpu", npartitions=48)}
threads = [1, 2, 4, 8, 16, 24]  # thread counts
strong = pd.DataFrame(index=threads, columns=list(work), dtype=float)  # seconds
for n in threads:  # each count
    with dask.config.set(scheduler="threads", num_workers=n):  # n threads
        for name, fn in work.items():  # each workload
            fn(); t = time.perf_counter(); fn(); strong.loc[n, name] = time.perf_counter() - t  # warm, then time
weak = {}  # seconds by thread count
for n in threads:  # each count
    side = int(np.sqrt(n * 11e6))  # pixels per side so each thread has about 11 million
    sub = dem[:min(side, dem.shape[0]), :min(side, dem.shape[1])]  # the window
    with dask.config.set(scheduler="threads", num_workers=n):  # n threads
        pdist.raster.focal(sub, "median", size=5, compute="cpu", chunks=(1024, 1024))  # warm-up
        t = time.perf_counter(); pdist.raster.focal(sub, "median", size=5, compute="cpu", chunks=(1024, 1024)); weak[n] = time.perf_counter() - t  # time
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))  # two panels
for col in strong:  # strong scaling curves
    ax[0].plot(threads, strong[col].iloc[0] / strong[col], "o-", label=col)  # speed-up
ax[0].plot(threads, threads, ":", color="grey", label="ideal"); ax[0].set_ylim(0, 25)  # ideal line
ax[0].set_xlabel("threads"); ax[0].set_ylabel("speed-up"); ax[0].legend(fontsize=8); ax[0].set_title("Strong scaling")  # labels
ax[1].plot(list(weak), [weak[1] / v for v in weak.values()], "o-")  # weak scaling efficiency
ax[1].axhline(1, ls=":", color="grey"); ax[1].set_ylim(0, 1.2); ax[1].set_xlabel("threads (and data)")  # ideal
ax[1].set_ylabel("efficiency (time with 1 thread / time with n)"); ax[1].set_title("Weak scaling, focal median 5 x 5")  # labels
plt.tight_layout(); plt.show()  # draw
strong.round(2)  # the seconds
''')

nb.md("""
The three workloads scale very differently on the same 24 cores. The compute-bound median scaled
more than tenfold; slope, which mostly moves memory, stopped improving after four threads; the
point-in-county join did not scale at all. Its time goes into serial steps (Hilbert-sorting 1.6
million points, querying the county tree, concatenating and re-sorting the result), which matches
notebook 04, where plain GeoPandas on one core was faster. Weak scaling of the median
holds up until the cores start to share memory bandwidth and the turbo clock falls with more
active cores. Measure scaling for your own workload before choosing hardware.

## Where the time goes: profiling

`pdist.profile` without a session uses Dask's local profiler and returns a table of seconds by
task type; with a session it writes Dask's HTML performance report (task stream, worker
profiles, bandwidth), which opens in a browser.
""")

nb.code('''
lazy_chain = lambda: pdist.raster.hillshade(pdist.raster.focal(dd.gp_raster("elev"), "mean", size=5, compute="cpu"), compute="cpu")  # read, smooth, shade
res, table = pdist.profile(lazy_chain)  # local profiler
print(f"result {res.shape}; task time summed over threads: {table['sum'].sum():.1f} s")  # total
print(table.head(8).round(2).to_string())  # the most expensive task types
with pdist.start("local", n_workers=4, threads_per_worker=6) as s:  # a cluster for the HTML report
    _, report = pdist.profile(lazy_chain, path=str(OUT / "profile_hillshade.html"))  # performance report
print("performance report:", report, f"({Path(report).stat().st_size / 1e6:.1f} MB)")  # where it is
''')

nb.md("""
Almost all of the task time is attributed to `getitem` tasks. That does not mean slicing is
slow: Dask fuses each chunk's read, halo slicing and the two computations into one task before
running the graph, and the fused task keeps the name of its first step. The table is therefore a
coarse view; the HTML report shows the same work as a timeline per thread and worker, which is
where the real bottlenecks (idle threads, transfers, one slow chunk) become visible.

## Seven pitfalls, reproduced and fixed

### 1. Computing the same graph twice

Each `.compute()` runs its graph from the start. Two statistics of the same lazy array computed
one after the other read the file twice; `dask.compute` (or `pdist.compute`) shares the work.
""")

nb.code('''
lazy = dd.gp_raster("preci")  # lazy
t = time.perf_counter(); a, b = float(lazy.mean()), float(lazy.std()); twice = time.perf_counter() - t  # two passes
t = time.perf_counter(); a2, b2 = dask.compute(lazy.mean(), lazy.std()); once = time.perf_counter() - t  # one shared pass
print(f"two .compute() calls {twice:.2f} s | one dask.compute {once:.2f} s | same values: {np.isclose(a, a2) and np.isclose(b, b2)}")  # result
''')

nb.md("""
### 2. Tiny chunks

Notebook 01 measured it: 256 pixel chunks were about nine times slower than 2048 pixel
chunks for the same slope. Check `data.npartitions` and the task count before running a big job.

### 3. Moving data to the host between GPU steps

A chain of GPU operations on host data copies every intermediate result to the host and back.
Moving the input once and keeping intermediates on the device avoids that.
""")

nb.code('''
if caps["gpus"]:  # needs a GPU
    chain = lambda d: pdist.raster.hillshade(pdist.raster.focal(d, "mean", size=5, compute="gpu"), compute="gpu")  # two GPU steps
    t = time.perf_counter(); chain(dem); sync(); host = time.perf_counter() - t  # host in, host between, host out
    t = time.perf_counter(); out = pdist.to_cpu(chain(pdist.to_gpu(dem))); sync(); device = time.perf_counter() - t  # one copy in, one out
    print(f"host input: {host:.2f} s | data kept on the GPU: {device:.2f} s")  # result
''')

nb.md("""
### 4. Reading a GPU result with NumPy

Results stay on the GPU when the input was on the GPU. NumPy cannot read CuPy memory directly;
move results with `pdist.to_cpu` (or `.get()`), and check with `pdist.is_gpu`.
""")

nb.code('''
if caps["gpus"]:  # needs a GPU
    r = pdist.raster.slope(pdist.to_gpu(win))  # a GPU result
    try:  # the mistake
        np.asarray(r.data)  # NumPy on CuPy memory
    except TypeError as e:  # what happens
        print("np.asarray on a CuPy array:", str(e)[:90])  # message
    print("is_gpu:", pdist.is_gpu(r), "| after to_cpu:", pdist.is_gpu(pdist.to_cpu(r)), type(pdist.to_cpu(r).data).__name__)  # the fix
''')

nb.md("""
### 5. `compute="auto"` on a machine with a GPU

`"auto"` picks the GPU whenever one exists, also inside the worker processes of a CPU cluster.
While writing notebook 01 this silently turned a "CPU cluster" comparison into a GPU one, and
four worker processes competed for the GPU's memory. Pass `compute="cpu"` explicitly when
measuring the CPU, or set the default for a whole session.
""")

nb.code('''
print("auto on this machine:", pdist.explain("raster.slope")["uses"])  # GPU when present
pdist.set_defaults(compute="cpu")  # every call now defaults to the CPU
print("after set_defaults(compute='cpu'):", pdist.explain("raster.slope")["uses"])  # CPU
pdist.set_defaults(compute="auto")  # back to the default
''')

nb.md("""
### 6. Importing cuDF or cuML changes CuPy's memory allocator

Importing `cudf` or `cuml` calls `cupy.cuda.set_allocator(rmm_cupy_allocator)`. Without an RMM
pool every CuPy allocation then becomes a `cudaMalloc`, which made chunked GPU work about four
times slower while this chapter was written, and memory already cached by CuPy's pool is
stranded. GeoLattice imports RAPIDS libraries through `geolattice.distributed._backend.rapids`,
which restores the previous allocator, and `pdist.capabilities()` only checks that the packages
are installed. If you import cuDF yourself, restore the allocator afterwards or configure an RMM
pool.
""")

nb.code('''
if caps["gpus"]:  # needs a GPU and RAPIDS
    import subprocess, sys  # a fresh Python process, so cuDF is imported for the first time
    probe = """
import time, numpy as np, cupy as cp
import geolattice.distributed as pdist
x = pdist.to_gpu(np.random.default_rng(0).random((8192, 8192), dtype=np.float32))
def best():
    ts = []
    for _ in range(4):
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        pdist.raster.slope(x, res=90.0, compute="gpu"); cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts[1:])
before = cp.cuda.get_allocator(); a = best()
import cudf
switched = cp.cuda.get_allocator() is not before; b = best()
cp.cuda.set_allocator(before); c = best()
print(switched, a, b, c)
"""  # times slope before and after a plain import of cuDF, then after restoring the pool
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.split()  # run it
    switched, t_pool, t_rmm, t_fixed = out[0] == "True", *map(float, out[1:4])  # parse
    print(f"allocator switched by import cudf: {switched} | slope of 67 million pixels: CuPy pool {t_pool:.3f} s, "  # result
          f"after import {t_rmm:.3f} s ({t_rmm / t_pool:.1f}x), restored {t_fixed:.3f} s")  # slowdown and fix
''')

nb.md("""
### 7. Worker memory limits, GDAL's cache and spilling

A worker that exceeds 60 % of its memory limit writes results to disk (spills), and at 95 % its
nanny restarts it. While this notebook was written, workers limited to a few hundred megabytes or
a gigabyte or two died again and again while doing nothing but reading GeoTIFFs. The cause was
GDAL: it caches decoded raster blocks up to 5 % of the machine's RAM **in every process**, here
6.6 GB per worker, far above the limit. `pdist.start("local", memory_limit=...)` now caps GDAL's
cache in the workers at a tenth of an explicit limit (set `GDAL_CACHEMAX` yourself to override).
With that, a small limit produces what it should: spilling, and a slower job.
""")

nb.code('''
for limit in ["1.5GB", "6GB"]:  # too little and enough memory per worker
    with pdist.start("local", n_workers=2, threads_per_worker=2, memory_limit=limit, dashboard=False) as s:  # GDAL cache capped
        t = time.perf_counter()  # start
        p = pdist.persist(dd.gp_raster("elev", chunks={"y": 1024, "x": 1024}))  # 1 GB into worker memory
        from dask.distributed import wait; wait(p.data)  # until loaded
        v = float(pdist.raster.slope(p, compute="cpu").mean())  # use it
        secs = time.perf_counter() - t  # elapsed
        spilled = sum(w.get("metrics", {}).get("spilled_bytes", {}).get("disk", 0) for w in s.client.scheduler_info()["workers"].values())  # bytes on disk
        used = [w["memory_gb"] for w in pdist.memory_report()["workers"].values()]  # memory per worker
        print(f"memory limit {limit:>5} per worker: {secs:.1f} s, spilled to disk {spilled / 1e9:.2f} GB, worker memory {used} GB")  # result
''')

nb.md("""
With 1.5 GB per worker part of the persisted DEM went to disk and the job took several times
longer; with 6 GB nothing spilled. A job that spills constantly needs more workers, more memory
per worker, smaller chunks, or less data persisted; a worker that dies while only reading files
points to a library cache, not to your data.

## Summary and exercises

- `check_equivalence` confirmed that every GPU implementation used in this chapter matched the
  CPU on real data: exactly for integers and geometries, to rounding for floating point.
- Correct timing needs warm-up, synchronisation, materialised lazy results and repeats;
  `pdist.benchmark` does all four.
- Strong scaling differed by workload: a compute-bound median scaled well, memory-bound slope
  did not, and the vector join was limited by its serial assembly step.
- `pdist.profile` attributes time to task types, or writes Dask's HTML performance report.
- Seven pitfalls reproduced: repeated computes, tiny chunks, host round trips between GPU steps,
  NumPy on GPU memory, `compute="auto"` inside CPU clusters, RAPIDS imports changing CuPy's
  allocator, and GDAL's per-process cache breaking worker memory limits.

**Exercises**

1. Add `pdist.network.shortest_path_lengths` and `pdist.analysis.dbscan` (with `compute="gpu"`)
   to the equivalence table. What tolerance does each need?
2. Measure strong scaling of `pdist.zonal_stats` for the counties with 1 to 24 threads. Where
   does it stop improving, and why?
3. Open the HTML performance report and identify the longest task in the task stream.
4. Repeat the spilling experiment with `memory_limit="1GB"` and with `GDAL_CACHEMAX` set to 2048
   in the environment before starting the cluster. What happens to the workers?
""")

nb.write()
