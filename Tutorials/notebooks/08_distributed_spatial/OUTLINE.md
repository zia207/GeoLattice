# Distributed and GPU Spatial Computing with GeoLattice: Notebook Outline

Chapter folder: `Tutorials/notebooks/08_distributed_spatial/`
Library: `geolattice.distributed` (imported as `pdist`, `src/distributed/`): sessions and clusters
(`start`, `Session`, `status`, `dashboard_link`), backends (`compute="auto" | "cpu" | "gpu"`,
`capabilities`, `to_gpu`, `to_cpu`, `persist`), partitioning (`spatial_partition`,
`partition_bounds`, `plan_chunks`), the operation registry (`list_ops`, `dispatch`, `explain`) and
ten submodules: `io`, `raster`, `vector`, `bridge`, `analysis`, `network`, `hydrology`, `ml`,
`sedona`, `diagnostics`.

Every notebook follows the house style: banner cell, `#` title with a one paragraph intro,
`## Overview` (theory), `## ... with GeoLattice`, Learning objectives, Data, Contents, Setup, the
working sections, and Summary and exercises. Section headings carry no numbers, every code
statement has a comment, and the markdown has no em or en dashes. Notebooks are built by scripts in
`_builders/` (now kept locally in `helper_functions/`, outside the repository; same `nb.py` / `run.py` / `sync_text.py` pattern as 09_11 and 09_12), executed before
release, and their markdown is checked against the executed outputs. Notebook files are named
`08_NN_<title>.ipynb`.

Three rules of the library frame the whole chapter and are repeated in notebook 00: every function
takes `compute=`, returns the type and memory space it was given (NumPy in, NumPy out; GeoDataFrame
in, GeoDataFrame out; Dask in, lazy Dask out), and runs on Dask's local threads unless a session was
started. Clusters are started under `if __name__ == "__main__":` in scripts.

Installation: `geolattice[distributed]` (in `geolattice[all]`), plus `geolattice[distributed-gpu]`
(CuPy, dask-cuda), `geolattice[distributed-hpc]` (SLURM, PBS, Kubernetes), `geolattice[distributed-spark]`
(Apache Sedona, Java 17+) and `geolattice[distributed-ml]` (XGBoost, LightGBM, PyTorch). cuDF, cuML
and cuGraph come from the RAPIDS index. Every notebook runs on a CPU-only machine: GPU cells fall
back with one warning, and timings are reported for whichever targets are present.

## Build machine (checked 24 November 2026)

| Resource | Value |
|---|---|
| CPU and memory | 24 cores, 123 GB RAM |
| GPU | 1 x NVIDIA RTX A5500 Laptop, 16 GB (CuPy 14.2, cuDF, dask-cuda, cuML 26.08, cuGraph 26.08; no cuSpatial, not needed) |
| Other | Dask 2026.7.1, dask-geopandas, dask-jobqueue 0.9, dask-kubernetes, Apache Sedona with OpenJDK 17, PyTorch 2.14 (CUDA), XGBoost 3.2, LightGBM 4.7, OSMnx 2.1, esda 2.9, libpysal 4.14 |
| Registered operations | 44 (`pdist.list_ops()`) |

Multi-GPU, SLURM, PBS and Kubernetes cannot be run on the build machine. Those sections show the
calls and explain the output, run a one-GPU or local stand-in, and say so in the text.

## Data inventory

Existing data are reused wherever possible. Paths are relative to `Data/`.

| Dataset | Path or source | Size | Used in |
|---|---|---|---|
| Great Plains covariate rasters | `03_spatial_analysis/Great_plain/Raster/*.tif`: elevation, slope, aspect, curvature (h, v), TPI, topographic class, slope length, NDVI, NLCD, precipitation, temperature, fragmentation, silt and clay, K factor | 15 rasters, 15,245 x 17,196 at 90 m (262 million pixels, 1.1 GB each, tiled, EPSG:5070) | 00, 02, 03, 05, 08, 09, 11, 12 |
| Great Plains soil carbon points and counties | `00_shared/data_geostatistics/Data_GP/GP_POINT_SOC.shp`, `GP_COUNTY.shp`, `03_spatial_analysis/Great_plain/GP_STATE.shp`, `gp_grid_new.csv` | 570 points, about 600 counties | 05, 06, 09, 12 |
| USDA CDL 2025 at 10 m, CONUS | `08_remote_sensing/08_11_Capstones/NISAR_crop_classification/2025_10m_cdls/2025_10m_cdls.tif` | 480,509 x 316,295 (152 billion pixels, 15 GB packbits) | 02, 03, 05, 12 |
| NYC yellow taxi trips 2023 | `06_spatialDB/yellow_tripdata_2023*.parquet` (and monthly files) | 38.3 million rows | 02, 04, 10 |
| NYC taxi zones | NYC TLC taxi zone shapefile (public download, cached on first use) | 263 polygons | 04, 10 |
| Microsoft Building Footprints | Planetary Computer `ms-buildings` (GeoParquet, partitioned by quadkey); one US state and Bangladesh | millions of polygons | 04, 05, 06, 10, 12 |
| OpenStreetMap roads | OSMnx for one metropolitan area (Dhaka or Buffalo, New York) | 10^5 to 10^6 edges | 07 |
| Landsat and Sentinel-2 | Planetary Computer STAC (`pdist.read_stac`) | lazy | 02, 09 |
| Large images for tiled inference | `08_remote_sensing/08_09_image_classification/Image_Fusion/HR/HR_Rural_July_14.tif` (1.7 GB) or the Corn Belt Sentinel-2 cube | | 09 |
| Sentinel-1 flood maps, Bangladesh | `08_remote_sensing/08_11_Capstones/s1_BD_charland/preprocessed/` (0.9 GB per date) | | 12 (option B) |

New downloads (taxi zones, building footprints, OSM roads) are cached under
`Data/08_distributed_spatial/`. Converted copies (Great Plains rasters to Zarr, CDL to a chunked Zarr
subset) are written to `outputs/` or a scratch folder and can be deleted.

## Notebook list

| Notebook | Built | Main result |
|---|---|---|
| 00 fundamentals | no | |
| 01 sessions and clusters | no | |
| 02 distributed input and output | no | |
| 03 raster processing at scale | no | |
| 04 vector processing at scale | no | |
| 05 rasters and vectors together | no | |
| 06 spatial statistics and clustering | no | |
| 07 networks | no | |
| 08 hydrology | no | |
| 09 machine learning | no | |
| 10 Apache Sedona | no | |
| 11 diagnostics and performance | no | |
| 12 capstone | no | |

### 08_00_distributed_spatial_computing_fundamentals.ipynb
Why Spatial Data Outgrow One Core, and What to Do About It
- Overview: data larger than memory, work longer than patience; the three ways out (more cores,
  more machines, a GPU) and what each costs; embarrassingly parallel versus communication-heavy
  operations; Amdahl's law and why I/O often sets the limit.
- Concepts with pictures: chunks and partitions; halos for moving windows; shuffles for joins;
  lazy task graphs (Dask) and when work actually runs; host memory versus GPU memory and the cost of
  moving data between them; spatial ordering (Hilbert curve) so that neighbours share a partition.
- Rasters versus vectors: regular grids split cleanly, irregular geometries do not; why spatial
  joins need partition pruning.
- The GeoLattice design: one API, `compute="auto" | "cpu" | "gpu"`, results in the input's type
  and memory space, Dask input stays lazy; the registry of 44 operations (`pdist.list_ops()`),
  `pdist.explain("raster.slope")` for the implementation that will run and why,
  `pdist.capabilities()` for what this machine has.
- Quick hands on: slope of the Great Plains DEM (262 million pixels) on one core with NumPy, on all
  cores with Dask, and on the GPU with CuPy; `pdist.benchmark` table; `pdist.check_equivalence`
  shows the three results agree; memory used by each.

### 08_01_sessions_and_clusters.ipynb
From a Laptop to a Cluster: Sessions, Workers and the Dashboard
- Overview: the Dask scheduler, workers, threads and processes; when threads suffice (NumPy, GDAL
  release the GIL) and when processes are needed (pure Python, GeoPandas); one worker per GPU.
- `pdist.start("local", n_workers=, threads_per_worker=, memory_limit=)`, the `Session` object,
  `status()`, `scale()`, `adapt()`, `dashboard_link()`, `close()` and the context manager.
- `pdist.start("cuda")` with dask-cuda: one worker per GPU, RMM memory pools, spilling to host.
- HPC and cloud: `start("slurm", cores=, memory=, walltime=)`, `start("pbs", ...)`,
  `start("kubernetes", image=)`, `start("address", address="tcp://...")`; job scripts generated by
  dask-jobqueue shown, not submitted on the build machine.
- Chunk planning: `pdist.plan_chunks` for a raster given worker memory; the 100 MB rule of thumb;
  what too-small and too-large chunks do to the task graph (measured).
- `persist` versus `compute`; why `.values` on a lazy array pulls everything to one process.
- Scripts: the `if __name__ == "__main__":` guard and a runnable example script in `scripts/`.

### 08_02_distributed_input_and_output.ipynb
Reading and Writing Big Spatial Data in Parallel
- Overview: Cloud-Optimized GeoTIFF internal tiles and overviews, Zarr chunks, GeoParquet row groups
  and partitions; why the storage layout should match the chunking.
- Rasters: `pdist.read_raster` with chunks aligned to the GeoTIFF blocks (128 x 128 for the Great
  Plains files), windowed reads of the 152 billion pixel CDL; `pdist.write_raster` to COG and to
  Zarr; `pdist.read_zarr`; converting the 15 Great Plains rasters to one chunked Zarr cube and the
  read speed before and after.
- STAC: `pdist.read_stac` for a lazy Landsat or Sentinel-2 time cube (odc-stac), selecting bands
  and resolution, nothing read until computed.
- Vectors: `pdist.read_vector` for partitioned GeoParquet and OGR sources; `pdist.write_vector`
  with spatially sorted partitions; `pdist.spatial_partition` (Hilbert order) and
  `pdist.partition_bounds`; reading Microsoft building footprints for a state from Planetary
  Computer; the 38 million taxi trips as a Dask DataFrame.
- Measured: read throughput for COG versus Zarr versus GeoTIFF strips, and GeoParquet versus
  Shapefile, with one and many workers.

### 08_03_raster_processing_at_scale.ipynb
Map Algebra, Focal Operations and Terrain on Chunked and GPU Rasters
- Overview: local, focal and zonal operations; why focal operations need halos; edge correctness.
- Map algebra: `pdist.raster.calc` (expression strings), `where`, `reclassify`, `normalize`,
  `normalized_difference`, `map_tiles` for any function on tiles.
- Focal: `focal` (mean, median, max, std), `convolve` with a custom kernel; chunked result equals
  the whole-array result, edges included (checked).
- Terrain on the Great Plains DEM: `slope`, `aspect`, `hillshade`, `curvature`, `tpi`, `tri`,
  `roughness`; comparison with the 90 m slope and curvature rasters that ship with the data.
- Out of core: the 10 m CONUS CDL (152 billion pixels) reclassified into cropland groups and
  counted, chunk by chunk, without ever holding it in memory; peak memory reported.
- CPU threads, local cluster and GPU timings for each operation (`pdist.benchmark`); where the GPU
  wins (focal, terrain) and where I/O dominates (reclassification of the CDL).

### 08_04_vector_processing_at_scale.ipynb
Geometry Operations, Clipping, Spatial Joins and Nearest Neighbours on Millions of Features
- Overview: per-feature operations parallelize trivially; joins do not. Bounding boxes, spatial
  indexes, partition pruning and the cost of a shuffle.
- Per-feature: `buffer`, `simplify`, `centroid`, `make_valid`, `to_crs`, `area`, `length`, `bounds`
  on a state's building footprints (GPU kernels for area, length and bounds).
- `pdist.vector.clip` to a county; `pdist.vector.sjoin` with partition pruning: buildings to
  counties or census tracts; same columns and rows as `geopandas.sjoin` (checked on a sample).
- GPU point-in-polygon: building centroids in polygons with cuDF; timing against GeoPandas and Dask.
- `pdist.vector.knn`: nearest k buildings, nearest facility for every building.
- NYC taxi: 38 million trips aggregated to the 263 taxi zones (attribute join, then a choropleth);
  pickup and drop-off flows between zones.

### 08_05_rasters_and_vectors_together.ipynb
Rasterize, Zonal Statistics, Point Extraction and Polygonize at Scale
- Overview: the two directions between models of space; why zonal statistics over many zones is a
  partitioned problem.
- `pdist.zonal_stats`: Great Plains NDVI, precipitation and slope by county; cropland area of the
  10 m CDL by county for the Great Plains states.
- `pdist.rasterize`: counties and building footprints to the 90 m and 10 m grids (and why
  `compute="auto"` keeps rasterize on the CPU).
- `pdist.extract_values`: all 15 Great Plains covariates at the 570 soil carbon points (the table
  that chapter 05's regression kriging starts from), nearest and bilinear.
- `pdist.polygonize`: field polygons from a CDL subset; chunk-edge stitching.

### 08_06_spatial_statistics_and_clustering.ipynb
Spatial Weights, Autocorrelation, Hot Spots and Clustering for Large Data
- Overview: why spatial statistics are expensive (n x n relations, 999 permutations), and how sparse
  weights and batched permutations make them tractable.
- `pdist.analysis.weights` (queen, rook, k nearest, distance band) for counties and for hundreds of
  thousands of building centroids; the `Weights` object, `row_standardize`, `to_libpysal`,
  `to_gpu`.
- Global and local statistics: `morans_i`, `gearys_c`, `local_morans`, `getis_ord_local`; results
  identical to `esda` (checked) and timings as n grows from 10^3 to 10^6.
- Clustering: `dbscan`, `hdbscan`, `kmeans` of building centroids (settlement detection), on CPU and
  GPU (cuML).
- `pdist.analysis.kde`: building density surface on a raster grid.
- Example: hot spots of Great Plains soil carbon covariates by county, and settlement clusters in
  Bangladesh building footprints.

### 08_07_networks.ipynb
Shortest Paths, Centrality and Accessibility on Large Road Networks
- Overview: graphs from line geometry, why betweenness centrality is costly, GPU graph analytics
  (cuGraph) and its limits.
- `pdist.network.graph_from_lines` from an OSMnx road network; the `Graph` object and conversion to
  SciPy, cuGraph and NetworkX.
- `shortest_path`, `shortest_path_lengths`, `nearest_nodes`; accessibility: network distance from
  every building to the nearest hospital or school.
- `centrality` (betweenness, closeness) on CPU and GPU, and why GeoLattice falls back to the CPU on
  lattice-like networks (`max_shortest_path_count` and cuGraph's counter overflow).
- `components`: connected parts of the network and data errors they reveal.

### 08_08_hydrology.ipynb
Flow Direction and Accumulation on Large DEMs
- Overview: D8 flow direction is local (parallel with a halo); depression filling and flow
  accumulation are global (information travels across the whole DEM), which limits chunking.
- `pdist.hydrology.flow_direction` chunked on the 262 million pixel Great Plains DEM (CPU and GPU).
- `fill_depressions` and `flow_accumulation` in memory: how large a DEM fits, and strategies for
  larger ones (tile by watershed, coarsen, WhiteboxTools out of core).
- Stream network and catchments of one river basin; comparison with the WhiteboxTools results of
  chapter 03.

### 08_09_machine_learning.ipynb
Training and Prediction at Scale: Random Forests, Boosting and Deep Learning on GPUs
- Overview: data parallel training, model parallel prediction, why prediction over large rasters is
  the easy part to parallelize.
- `pdist.ml.fit` with random forest, XGBoost and LightGBM on the Great Plains soil carbon table
  (cuML on GPU, scikit-learn on CPU); spatial cross-validation.
- `pdist.ml.predict_raster`: a soil carbon map over 262 million pixels from 15 covariates, chunked;
  NaN handling; class probabilities for a classification example.
- `pdist.ml.predict_tiles`: a PyTorch segmentation model over a 1.7 GB image in overlapping tiles,
  one replica per GPU, seams checked.
- `pdist.ml.train_ddp` and `ddp_model`: distributed data parallel training (run with one process on
  the build machine; multi-GPU usage explained).
- Timings: CPU versus GPU for fitting and for prediction.

### 08_10_apache_sedona.ipynb
Spatial SQL on Spark with Apache Sedona
- Overview: Spark's model (executors, partitions, the driver), Sedona's spatial RDDs and SQL
  functions; when Spark is the better engine (very large joins, existing Spark data lakes) and when
  Dask is simpler.
- `pdist.sedona.spark()`, `packages()`, `to_spark`, `to_geopandas`, `stop()`, `is_spark`.
- `pdist.sedona.sjoin`: buildings to counties on Spark, compared with `pdist.vector.sjoin` on Dask
  (rows, time, memory).
- Spatial SQL with `ST_Contains`, `ST_Intersects`, `ST_Distance`, `ST_Buffer` on the taxi zones and
  trips; GeoParquet in and out.

### 08_11_diagnostics_and_performance.ipynb
Measuring, Checking and Tuning Distributed and GPU Workloads
- Overview: correctness first (CPU and GPU must agree), then speed; strong and weak scaling;
  overheads (task scheduling, serialization, host-device transfer).
- `pdist.check_equivalence` for every operation used in the chapter, with tolerances.
- `pdist.benchmark` across chunk sizes, worker counts and targets; strong and weak scaling curves
  for slope, sjoin and Moran's I.
- `pdist.explain`, `pdist.memory_report`, `pdist.profile` (Dask performance report HTML).
- Common pitfalls, each reproduced and fixed: tiny chunks, unaligned chunks and file blocks,
  repeated `.compute()`, implicit transfers to the host, a GPU result read with NumPy, spilling.

### 08_12_capstone_*.ipynb
Capstone: one end-to-end workflow at a scale that needs the whole chapter.
- Option A (Great Plains, all data local): crop area of every Great Plains county from the 2025 10 m
  CDL (out of core), terrain and climate summaries from the 90 m covariates, a soil carbon map
  predicted over 262 million pixels, and county statistics with hot spots; timed end to end on CPU
  and GPU.
- Option B (Bangladesh): building exposure to river flooding: Microsoft building footprints of
  Bangladesh, Sentinel-1 flood maps of the char lands (chapter 09 capstone data), spatial join and
  zonal counts per upazila, settlement clusters and access to the nearest road.

## Library gaps to check while building

- `pdist.analysis.hdbscan` should be checked against `sklearn.cluster.HDBSCAN` (the stand-alone
  `hdbscan` package is not installed).
- `pdist.read_vector` on Planetary Computer's partitioned `ms-buildings` GeoParquet (signed URLs,
  quadkey partitions) needs a test.
- `pdist.hydrology.flow_accumulation` holds the whole DEM in memory; notebook 08 must state the size
  limit it measures.

## Open decisions

- Capstone: option A (Great Plains, no downloads) or option B (Bangladesh flood exposure).
- Region for building footprints and the road network (one US state and Buffalo, or Bangladesh and
  Dhaka).
- Whether to include a multi-GPU or SLURM run on another machine, or keep those parts descriptive.
