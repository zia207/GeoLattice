# GeoLattice: module organization and rebuild (session record)

A record of one Claude Code working session on GeoLattice, 13–19 October 2026. It covers reorganizing the
package, adding new subpackages, the bugs found along the way, and the tutorial notebooks written or
updated. File names are given as they are in the repository now. Several notebooks were renamed by
the author after they were written, and those renames are noted where they matter.

---

## 1. Environment and build procedure

| Item | Value |
|---|---|
| Python | pyenv 3.11.16 |
| GDAL | local build; run everything with `LD_LIBRARY_PATH=$HOME/.local/share/gdal-prefix/lib` |
| GPU | NVIDIA RTX A5500, CuPy (used by the GPU simulation engines and `rsense`) |
| Version | 0.3.8 → 0.3.9 (layout change, this session) → 0.3.11 (bumped by the author later) |

Build, install and test:

```bash
export LD_LIBRARY_PATH=$HOME/.local/share/gdal-prefix/lib
python -m build                                   # wheel + sdist into dist/
pip install --no-deps --force-reinstall dist/geolattice-0.3.11-cp311-cp311-linux_x86_64.whl
pytest tests/ --ignore=tests/rsense --ignore=tests/spatialcausal   # main suite
pytest tests/rsense tests/spatialcausal                             # optional extras
```

Subpackages are loaded lazily: `src/__init__.py` lists them in `_SUBMODULES`, so `import geolattice`
stays fast and `geolattice.geostatistics`, `geolattice.rsense`, `geolattice.spatial_analysis.spatialcausal`, `geolattice.web` and the
rest load on first use.

---

## 2. Timeline of requests

| # | Date (2026) | Request | Outcome |
|---|---|---|---|
| 1 | Oct 13 15:30 | Organize all modules in `src/`, test, rebuild, update `dist/` and README, install | 0.3.9 built and installed; 375 passed |
| 2 | Oct 13 15:51 | Test `spatial_statistics` | 91 new tests; GWGLM bug fixed; 466 passed |
| 3 | Oct 13 21:00 | Explain geostatistical simulation and its types | Written explanation |
| 4–13 | Oct 13 21:05–23:23 | Add simulators: LU, turning bands, FFT, plurigaussian, MPS (Direct Sampling), SNESIM, SIS co-sim, SGS co-sim, DSS, simulated annealing | 10 new modules; suite grew to 563 passed |
| 14 | Oct 13 23:35 | Multiple-point co-simulation | Added to Direct Sampling and SNESIM; 568 passed |
| 15–19 | Oct 13 23:47–Oct 14 00:54 | GPU engines: MPS, SGS, SIS, DSS, turning bands | 5 GPU engines; cuBLAS crash fixed; 636 passed |
| 20 | Oct 14 01:30 | Organize `src/geostatistics` into subfolders by function | 10 subfolders, old paths aliased |
| 21 | Oct 14 04:33 | Notebook: introduction to geostatistical simulation | Written and executed |
| 22 | Oct 14 05:03 | Integrate `src/rsense` as `geolattice.rsense`, update `tests/rsense` | 1024 passed, 4 skipped; GPU bug fixed in ~50 places |
| 23–29 | Oct 14 12:53–16:50 | Simulation notebooks: sgcosim, sicosim, dssim, sasim, PGS, MPS, SNESIM | All executed, all checks pass |
| 30 | Oct 14 17:03 | Explain Bayesian kriging and its types | Written explanation |
| 31 | Oct 14 17:05 | Add Bayesian kriging to `src/geostatistics` | `kriging/bayesian_kriging.py`; 1047 passed |
| 32 | Oct 14 17:21 | "Do I need `kriging/ebk.py`?" | Yes: different method, used by tests and tutorials |
| 33–38 | Oct 14 17:25–22:04 | Bayesian kriging notebooks (6) | All executed; three library improvements |
| 39 | Oct 15 00:08 | Integrate `src/spatialcausal` as `geolattice.spatial_analysis.spatialcausal` | 106 tests pass |
| 40 | Oct 15 00:55 | Update the nine `06_spatial_causal` notebooks for GeoLattice | All rerun with no errors |
| 41–46 | Oct 16 20:22–21:06 | Update the four 07_01 foundation notebooks for GeoLattice (`geolattice.rsense`, local data) | All rerun with no errors |
| 47–48 | Oct 16 21:20 | New notebook: introduction to optical images | Written and executed |
| 49 | Oct 19 14:52 | Outline a Web/Cloud GIS module | Module 08 outline (12 notebooks) |
| 50 | Oct 19 15:00 | Write any missing Web/Cloud GIS functions in `src` | New `geolattice.web`; cloud-native vector writers; 696 passed |
| 51 | Oct 19 15:57 | Start Module 08 with local data and `get_images` | 08_00 and 08_01 written |
| 52 | Oct 19 21:22 | 08_02 cloud-native vector notebook | Written; three reader problems fixed |
| 53 | Oct 19 22:41 | 08_03 spatial SQL notebook | Written; data problem in `bd_poverty.csv` documented |
| 54 | Oct 19 23:09 | This record | This file |

---

## 3. Package organization

### 3.1 First pass (request 1)

- **Geostatistics would not import.** Its modules had moved from `geostatistics/pygstat/` into
  `geostatistics/`, but `src/geostatistics/__init__.py` still imported `pygstat`. The old 0.3.8
  install hid the problem because it still had the `pygstat/` folder. The `__init__` was rewritten
  for the flat layout with the same 134 public names; `pygstat` remains as an alias.
- **Top-level access:** `src/__init__.py` exposes every subpackage lazily (15 at the time).
- **Import check:** all 192 modules under `src/` imported, except three H2O modules (optional `h2o`
  not installed). Missing optional backends (H2O, PyTorch, TensorFlow) become placeholders that only
  raise when used.
- **Out-of-date tests:** two tests looked for terrain and hydrology functions in `geolattice.raster`; they
  now live in `geolattice.spatial_analysis`.
- Version bumped to 0.3.9 in `pyproject.toml`, `src/__init__.py` and
  `src/spatial_statistics/__init__.py`. README layout tree updated.

### 3.2 Geostatistics subfolders (request 20)

| Folder | Contents |
|---|---|
| `core/` | Variograms, ordinary and simple kriging, indicator variogram |
| `kriging/` | Universal, co-, indicator, disjunctive, factorial, soft, Poisson and multivariate Poisson kriging; EBK; KT3D; E-type; Bayesian kriging (added later) |
| `spacetime/` | Space-time kriging, space-time Poisson kriging, space-time variogram models |
| `ml/` | Regression kriging (scikit-learn, H2O, PyTorch, TensorFlow, GNN); STGNN, STGTN, CNN-LSTM |
| `validation/` | `cross_validation.py` and the `spatial_cv_*` modules |
| `simulation/` | sgsim, sisim, lusim, tbsim, fftsim, sgcosim, sicosim, direct sequential, annealing, plurigaussian |
| `mps/` | `direct_sampling.py` (formerly `mps.py`) and `snesim.py` |
| `gpu/` | The five GPU engines |
| `gslib/` | GSLIB-style helpers: file I/O, normal scores, back-transform, declustering, grid coordinates |
| `io/`, `utils/` | Unchanged |

All 54 old module paths still resolve (for example `geolattice.geostatistics.mps`,
`geolattice.geostatistics.common`, `geolattice.geostatistics.kriging_STGNN`). One quirk is unchanged:
`geolattice.geostatistics.sgsim` is the function, not the module, because they share a name.

### 3.3 Subpackages added during the session

| Subpackage | Source | Extras in `pyproject.toml` |
|---|---|---|
| `geolattice.rsense` | `src/rsense` (from PyRSense, 110 modules) | `rsense`, `rsense-access`, `-sar`, `-insar`, `-lidar`, `-uav`, `-classify`, `-dl`, `-xai`, `-gee`, `-viz`, `-scale`, `-apps`, `-gpu`, `-all` |
| `geolattice.spatial_analysis.spatialcausal` | `src/spatialcausal` (from PySpatialCausal) | `spatialcausal`, `spatialcausal-viz`, `spatialcausal-causalml`, `spatialcausal-all` |
| `geolattice.web` | New | `web`, `web-cloud` |

`geolattice[all]` is installation option 6 (from 0.3.44): every group in options 1 to 5 plus web and cloud GIS, spatial databases, deep learning, Bayesian and boosting models; GPU, H2O, TensorFlow, large vision models, BioPAR and `dev` are added by name. See "Installation options" in the README.

---

## 4. Geostatistics additions

### 4.1 Simulation methods (requests 4–14)

| Method | Module | Main entry points |
|---|---|---|
| LU (Cholesky) decomposition | `simulation/lusim.py` | `lusim` |
| Turning bands | `simulation/tbsim.py` | `TBSimulator`, `tbsim` |
| FFT spectral | `simulation/fftsim.py`, `_conditioning.py` | `fftsim` |
| Plurigaussian | `simulation/pgs.py` | `PluriGaussianSimulator`, truncation rules (`hierarchical`, `from_flag`) |
| Multiple-point statistics, Direct Sampling | `mps/direct_sampling.py` | `DirectSampling` |
| SNESIM (Strebelle 2002) | `mps/snesim.py` | `SNESIM`, `snesim` |
| Sequential indicator co-simulation | `simulation/sicosim.py` | `IndicatorCoSimulator`, `sicosim`, `SoftCalibration` |
| SIS with local prior means | `simulation/sis_local_means.py` | `sis_local_means`, `sklm_cdf`, `order_relations` |
| Sequential Gaussian co-simulation | `simulation/sgcosim.py` | `GaussianCoSimulator`, `sgcosim`, `sgcosim_hierarchical` |
| Direct sequential simulation | `simulation/direct_sequential.py` | `DirectSequentialSimulator`, `dssim_direct` |
| Simulated annealing | `simulation/annealing.py` | `SimulatedAnnealing`, `sasim` |
| Multiple-point co-simulation | optional parameters on Direct Sampling and SNESIM | |

SNESIM uses a per-level list of unique training-image patterns in place of Strebelle's search tree,
a multigrid scheme, dropping of the farthest nodes when fewer than `min_replicates` patterns match,
and a servosystem for target proportions. Its defaults were tuned by measurement rather than copied
from GSLIB.

### 4.2 GPU engines (requests 15–19)

| Engine | Module | Use |
|---|---|---|
| MPS (Direct Sampling and SNESIM) | `gpu/mps_gpu.py` | `use_gpu=True` |
| Sequential Gaussian | `gpu/sgs_gpu.py` | `sgsim_gpu` |
| Sequential indicator | `gpu/sis_gpu.py` | `IndicatorCoSimulator(..., use_gpu=True)` |
| Direct sequential | `gpu/dss_gpu.py` | `DirectSequentialSimulator(..., use_gpu=True)` |
| Turning bands | `gpu/tbsim_gpu.py` | `TBSimulator(..., use_gpu=True)` |

All were tested on the real GPU. Turning bands on the GPU reproduces the CPU result for the same seed
to about 1e-14 of the value range, with speed-ups of 25–105× depending on grid and model (the
spherical model gains least).

### 4.3 Bayesian kriging (request 31, then refined in 34–37)

New module `src/geostatistics/kriging/bayesian_kriging.py`:

| Class or function | What it does |
|---|---|
| `BayesianKriging` | Exact grid posterior over range and nugget-to-sill ratio (as in geoR `krige.bayes`), optional trend or covariates; Student-t mixture predictive |
| `ConjugateBayesianKriging` | Fixed correlation; only trend and variance uncertain. `BayesianKriging.from_variogram(vg)` builds one |
| `BayesianTransformedKriging` | Adds a Box–Cox λ to the grid; back-transforms by simulation |
| `BayesianGLMKriging` | Poisson (with exposure) or binomial data by MCMC |
| `bayesian_kriging()`, `correlation_function()` | One-call wrapper; shared unit-sill correlation function |

Changes made while writing the notebooks:

- **`BayesianGLMKriging` mixing:** the intercept had an effective sample size of 12 and R-hat 1.62.
  Centred Gibbs updates (interweaving, `interweave=True` by default, `range_gibbs_every`) raised the
  intercept's ESS to about 1,400 with every R-hat at most 1.06. Chains are about 4× slower; σ² and the
  range still mix slowly.
- **`BayesianTransformedKriging`** now warns when λ < 0 produces infinite predictive draws and reports
  their fraction.
- **`bayesian_kriging()`** gained `signal=False` (predict measurements) and `return_model=True`.
- **`correlation_function`**: stable-family α handling fixed (α > 2 now raises).

`kriging/ebk.py` (`EmpiricalBayesianKriging`) was kept: it is a different, prior-free method, it
supports GPU prediction and anisotropy, and tests and two tutorials depend on it. Its docstring
refers to a `tests/test_ebk.py` that does not exist; the tests are in `tests/test_geostatistics.py`.

---

## 5. `geolattice.web` and cloud-native vector (requests 50 and 52)

### 5.1 New `src/web/` package

| Module | Functions |
|---|---|
| `cloud.py` | `to_vsi_path` (s3, gs, az, https and zip paths to GDAL `/vsi` paths), `cloud_options`, `cloud_env`, `remote_info` (network errors returned, not raised) |
| `cog.py` | `COGReport`, `validate_cog`, `to_cog` |
| `tiles.py` | Tile maths (`lonlat_to_tile`, `tile_bounds`, `tiles_for_bounds`, zoom/resolution helpers), `render_tile`, `make_xyz_tiles`, `tilejson`, `raster_web_info`, `suggest_rescale` |
| `server.py` | `create_app` (Starlette: XYZ tiles, TileJSON, OGC API – Features, `/viewer`), `serve` |
| `viewer.py` | `leaflet_html` |
| `static.py` | `static_server` / `StaticServer`: serves a folder with range requests, CORS and byte counters |

### 5.2 Cloud-native vector in `geolattice.vector`

`src/vector/write_cloud_native_vector_data.py`:

- `write_geoparquet`, `read_geoparquet`, `geoparquet_info`: GeoParquet 1.1 with a bbox covering
  column and Hilbert-sorted rows, written through GeoPandas and PyArrow because this GDAL build has
  no Parquet driver. Reads accept a bbox in another CRS and work over `http(s)://`.
- `write_flatgeobuf` (with spatial index) and `write_pmtiles` (through GDAL).

### 5.3 Reader fixes found by the 08_02 measurements

| Problem | Fix | Result |
|---|---|---|
| `open_cloud_native_vector_data` could not open GeoParquet over http(s) | `_remote_arrow_filesystem` (fsspec http, `cache_type="none"`, pre-buffer) | Works |
| A bbox query on a 103 MB remote GeoParquet transferred 2.0 GB | Row-group pruning (`_bbox_covering_paths`, `_row_groups_for_bbox`, stable ids via `_FID_KEY`) | 8.3 MB |
| `read_geoparquet` failed on http(s) URLs | fsspec http filesystem | Works |
| Opening a remote PMTiles archive decoded every tile (8.2 MB) | `_defer_layer_scan` in `read_vector_data.py` skips count/extent scans for PMTiles/MVT on `/vsi` paths | 20 KB |

Tests: `tests/test_web.py` (30 tests).

---

## 6. Bugs found and fixed

| Where | Bug | Fix |
|---|---|---|
| `geostatistics/__init__.py` | Still imported the removed `pygstat` folder | Rewritten for the flat layout |
| terrain/hydrology tests | Looked in `geolattice.raster` | Point at `geolattice.spatial_analysis` |
| `spatial_statistics/gwmodel/linear/gwglm.py` | IRLS weights divided by the variance instead of multiplied, so every Poisson and binomial GWGLM was wrong (slope 0.228 vs 0.307; diverged with 40 neighbours) | Fixed; matches statsmodels to ~1e-10 at infinite bandwidth. Earlier Poisson/binomial GWGLM results should be rerun |
| `simulation/fftsim.py` | Conditioning variance | Fixed in `_conditioning.py` |
| `ml/regression_kriging_tf.py` | Force-loaded the CUDA toolkit's cuBLAS 13.5 while PyTorch had loaded 13.4, causing a segfault on the next GPU matrix multiply | Fixed |
| `rsense` (~50 places) | `to_cpu(x.values)` raises on GPU arrays before the copy | Changed to `to_cpu(x).values`; regression test added |
| `rsense` tests | rioxarray 0.19 rejects `write_crs(None)` | Test builds a CRS-less array directly |
| `bayesian_kriging.py` | GLM mixing, infinite draws, missing options | See section 4.3 |
| `geolattice.vector` readers | Remote GeoParquet and PMTiles | See section 5.3 |

---

## 7. Test suite growth

| Point in session | Result |
|---|---|
| After reorganization (0.3.9) | 375 passed, 1 skipped (`cartopy`) |
| After spatial_statistics tests | 466 passed (91 new) |
| After the ten simulators | 563 passed |
| After the five GPU engines | 636 passed |
| After rsense integration | 1024 passed, 4 skipped |
| After Bayesian kriging | 1047 passed, 4 skipped (later 19 Bayesian kriging tests) |
| spatialcausal | 106 passed (`tests/spatialcausal/`) |
| After `geolattice.web` | 696 passed, 1 skipped (main suite, excluding rsense and spatialcausal) |
| Final main suite | 699 passed, 1 skipped (excluding rsense and spatialcausal) |

---

## 8. Tutorials

### 8.1 Module 05: geostatistics (`Tutorials/notebooks/05_geostatistics/`)

Simulation (Jura data, unless noted):

| Notebook | Topic | Notes |
|---|---|---|
| `05_07_00_introduction_geostatistical_sumulations.ipynb` | Introduction to simulation | Requested as 05_07_01; renamed by the author |
| `05_07_02_sgcosim.ipynb` | `GaussianCoSimulator`, `sgcosim`, `sgcosim_hierarchical`, `sgsim_gpu` | 14 checks pass |
| `05_07_04_sicosim.ipynb` | `IndicatorCoSimulator`, `sicosim`, `SoftCalibration` | 12 checks pass |
| `05_07_05_dssim.ipynb` | `DirectSequentialSimulator`, `dssim_direct` | |
| `05_07_06_sasim.ipynb` | `SimulatedAnnealing`, `sasim` | 9 checks; exact histograms and class proportions; abrupt changes next to data |
| `05_07_07_pgs.ipynb` | `PluriGaussianSimulator` | 9 checks; skipped-formation contacts 0.07% vs 9% for SIS (map: 0.4%) |
| `05_07_08_mps.ipynb` | Direct Sampling | |
| `05_07_09_snesim.ipynb` | SNESIM, synthetic three-facies river system | 7 checks; multigrid levels matter most |

Bayesian kriging (written into a `06_bayesian_kriging/` subfolder as 05_06_xx, then renumbered by the
author to 05_08_xx in the main folder):

| Notebook | Topic | Main finding |
|---|---|---|
| `05_08_02_bayesian_kriging.ipynb` | `BayesianKriging`, Jura log-cadmium | Conjugate case equals ordinary kriging to 1e-8; likelihood prefers ~0.19 km range vs 1.33 km least-squares fit |
| `05_08_03_bayesian_glm_kriging.ipynb` | `BayesianGLMKriging` | Counts simulated from real county rates (`lbc_rate.csv`, diabetes) so estimates can be checked |
| `05_08_04_bayesian_transformed_kriging.ipynb` | `BayesianTransformedKriging` | λ falls with skewness (r = −0.80) over ten metal datasets |
| `05_08_05_conjugate_bayesian_kriging.ipynb` | `ConjugateBayesianKriging` | Student-t matters at small n (8 samples: 90.2% vs 86.4% coverage) |
| `05_08_06_bayesian_kriging_function.ipynb` | `bayesian_kriging()` | "Range" means different distances per model (0.81, 3, 1.73 × range) |
| `05_08_07_correlation_function.ipynb` | `correlation_function` | 9 checks; validity in 2-D, smoothness near zero, conditioning |

### 8.2 Module 06: spatial causal inference (`06_spatial_causal/`)

- All nine notebooks (06_00–06_08) switched to `import geolattice.spatial_analysis.spatialcausal as sc` and
  `geolattice.spatial_statistics` and rerun with no errors.
- 06_04–06_08 reproduce every printed value; 06_01–06_03 differ only in the third decimal of a few
  machine-learning estimates.
- Helpers `countycausal.py` and `tutorial_style.py` were copied from the trashed
  `Spatial-Causal-Tutorial` folder. `countycausal.py` reads `Data/03_spatial_analysis/data_saptial_causal/county_frame.gpkg`
  (override with `SPATIAL_CAUSAL_DATA`; `ATLAS_REPO` points to the USA-Health-Data-Atlas repo).

### 8.3 Module 07: remote sensing (`07_remote_sensing/`)

The four foundation notebooks were moved from PyRSense to `geolattice.rsense` and to local data in
`Data/00_shared/data_remote_sensing` (with live STAC searches where needed). They have since been renumbered
by the author:

| Now | Topic |
|---|---|
| `07_01_foundations/07_01_00_remote_sensing_fundamentals.ipynb` | Fundamentals |
| `07_01_foundations/07_01_01_python_geospatial_environment.ipynb` | Python environment |
| `07_01_foundations/07_01_02_raster_vector_essentials.ipynb` | Raster and vector essentials |
| `07_01_foundations/07_01_03_data_discovery_access.ipynb` | Data discovery and access |
| `07_02_optical/07_02_00_introduction_optical_images.ipynb` | New: theory, applications, open and commercial optical sensors |

### 8.4 Module 08: Web and Cloud GIS (`07_web_cloud_gis/`)

The outline (request 49) is kept in `07_web_cloud_gis/README.md`: twelve notebooks, 08_00 to 08_11.

| Notebook | Status | Highlights |
|---|---|---|
| `07_00_introduction_web_cloud_gis.ipynb` | ready | A 512×512 window of a 91 MB Sentinel-2 band costs 2.2 MB; Onondaga County needs 9 tiles at z10, 896 at z14; `get_images` finds 302 Sentinel-2 and 38 Landsat scenes for summer 2025; minimal `create_app` server |
| `07_01_cloud_native_raster.ipynb` | ready | `get_images` downloads to `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/`; downloads are tiled but not COGs; ZSTD/DEFLATE with predictor smallest; NDVI cube, Zarr append, STAC catalogue with GeoParquet index |
| `07_02_cloud_native_vector.ipynb` | ready | 5,160 Bangladesh unions; remote transfer for one Dhaka query: GeoJSON 686 MB, GeoPackage 2.1 MB, FlatGeobuf 0.72 MB, sorted GeoParquet ~8 MB, unsorted 53 MB |
| `07_03_spatial_sql.ipynb` | ready | DuckDB spatial; axis-order trap (`always_xy`); three-step join repair; district dissolve; arsenic fields in Tala upazila; R-tree measurements; GeoPackage SQL; DuckDB/PostGIS/BigQuery translation table |
| 08_04 – 08_11 | planned | Web maps, serving maps, APIs, web apps, scalable processing, cloud platforms, DevOps, capstone |

---

I'll check how the existing course outline is formatted and what GeoLattice already provides for web and cloud work, so the module builds on real functions:

# Module 08: Web and Cloud GIS with GeoLattice

**Folder:** `Tutorials/notebooks/07_web_cloud_gis/`, which continues the existing 01–07 numbering.
**Level:** intermediate. It assumes Modules 01–02 (vector and raster) and 07_01_04 (data discovery).
**Goal:** take a GeoLattice analysis off the laptop. Read data straight from cloud storage, process it at scale, publish it as maps, tiles and APIs, and deploy small web apps.

### Notebooks

**08_00 · Introduction to Web and Cloud GIS**
- Theory:
  - desktop vs web vs cloud GIS;
  - client–server, REST, the OGC standards (WMS, WMTS, WFS, WCS, OGC API Features and Tiles), STAC;
  - "cloud-native" data: HTTP range requests, chunking, tiling;
  - where the compute runs: browser, server, serverless, cluster.
- A map of the ecosystem: storage (S3, GCS, Azure Blob), formats, catalogues, compute, serving, front ends.
- Costs, egress, licensing and data sovereignty.
- **Libraries:** none; mostly diagrams and tables.

**08_01 · Cloud-Native Raster Data: COG, Zarr and STAC**
- How COGs work (internal tiles, overviews, range requests) and how Zarr chunks n-dimensional data.
- Writing COGs and Zarr, checking a COG is valid, choosing chunk sizes and compression.
- Building and publishing a STAC catalogue for your own products; reading and writing stac-geoparquet.
- **GeoLattice:** `rs.write` (COG), `rs.core.write_zarr` / `read_zarr`, `rs.scale.to_zarr_cube`, `rs.scale.export_stac`, `rs.access.stac_search`.
- **Others:** `rio-cogeo`, `zarr`, `xarray`, `pystac`, `stac-geoparquet`.

**08_02 · Cloud-Native Vector Data**
- GeoParquet / GeoArrow, FlatGeobuf, PMTiles, COPC, GeoJSON sequences.
- Lakehouse tables (Iceberg v3 geometry, Delta Lake) and H3/S2 cell tables.
- Querying remote GeoParquet with bounding-box filters, e.g. Overture Maps buildings for Syracuse.
- Converting shapefiles and GeoPackages to GeoParquet, FlatGeobuf and PMTiles.
- **GeoLattice:** `geolattice.vector.open_cloud_native_vector_data`, `supported_cloud_native_formats`, `read_vector` / `write_vector`.
- **Others:** `geopandas`, `pyarrow`, `duckdb` (spatial extension), `lonboard`, `tippecanoe` / `pmtiles`.

**08_03 · Spatial SQL in the Cloud: DuckDB, PostGIS and BigQuery**
- Spatial SQL basics: joins, buffers, aggregation by H3 cell.
- DuckDB on local and remote Parquet with no server.
- PostGIS as a service (Docker or a managed database), connected through SQLAlchemy and GeoAlchemy2.
- Overview of BigQuery GIS and Earth Engine tables.
- Pushing computation to the data vs pulling data to Python.
- **Others:** `duckdb`, `sqlalchemy`, `geoalchemy2`, `psycopg`, `h3`.

**08_04 · Interactive Web Maps in Python**
- Leaflet and MapLibre concepts: tiles, layers, styles, popups.
- Folium for quick maps; `geolattice.viz.interactive_map`, `add_choropleth`, `add_geojson`, `add_layer_control`.
- leafmap for COG, STAC and PMTiles layers.
- lonboard and deck.gl for millions of features on the GPU.
- Exporting standalone HTML maps.
- **GeoLattice:** `geolattice.viz.interactive_map` and the other `geolattice.viz` interactive functions, `rs.viz.interactive_map`.
- **Others:** `folium`, `leafmap`, `ipyleaflet`, `lonboard`, `pydeck`.

**08_05 · Serving Maps: Tiles and OGC APIs**
- Dynamic raster tiles from COG and STAC with TiTiler.
- Vector tiles with PMTiles, served from static storage with no server.
- OGC API Features with pygeoapi; WMS/WMTS with GeoServer or MapServer, for context.
- Styling with Mapbox GL style JSON.
- Example: serve the Onondaga NDVI COG (07_01_03) and the county layer as tiles.
- **Others:** `titiler`, `rio-tiler`, `pygeoapi`, `pmtiles`, `owslib` (client side).

**08_06 · Building Geospatial APIs with FastAPI**
- REST design for spatial services, e.g. `/zonal-stats`, `/ndvi?bbox=&date=`.
- Wrapping GeoLattice functions as endpoints; GeoJSON in and out; validation with Pydantic.
- Background jobs for long tasks; caching.
- Packaging with Docker.
- **GeoLattice:** `rs.zonal_stats`, `rs.optical.index`, `rs.clip`, spatial-statistics functions as service back ends.
- **Others:** `fastapi`, `uvicorn`, `pydantic`, `geojson-pydantic`, `httpx`.

**08_07 · Web Apps: Streamlit, Panel and Solara**
- Dashboard patterns: sidebar inputs, map, chart, download button.
- Rebuilding a small version of the PyGIS Explorer app: draw an area, search STAC, preview, download.
- Running the GeoLattice app templates (change viewer, flood mapper, index explorer).
- State, caching and secrets; deploying to Streamlit Community Cloud or Hugging Face Spaces.
- **GeoLattice:** `rs.apps.list_templates`, `rs.apps.create_app`, `rs.apps.launch`, `rs.access.*`.
- **Others:** `streamlit`, `streamlit-folium`, `leafmap`, `panel`, `solara`.

**08_08 · Scalable Processing in the Cloud**
- Dask clusters, local then cloud (Coiled or Kubernetes); chunked processing of a multi-year Sentinel-2 cube to a monthly NDVI Zarr.
- GPU instances, using the CPU/GPU backend from 07_01_02.
- Serverless functions (AWS Lambda / Cloud Run) for tile-by-tile jobs.
- Monitoring cost and memory.
- **GeoLattice:** `rs.scale.start_cluster`, `rechunk_for`, `open_cube`, `rs.apply_tiled`, `rs.set_backend`, `rs.optical.load_sentinel2`.
- **Others:** `dask.distributed`, `coiled`, `odc-stac`, `fsspec`, `s3fs` / `gcsfs` / `adlfs`.

**08_09 · Cloud Platforms: Planetary Computer, Earth Engine and Others**
- Planetary Computer hub and data, AWS Open Data, Copernicus Data Space (openEO, Sentinel Hub), Google Earth Engine.
- Choosing a platform and avoiding lock-in.
- The same NDVI-composite task three ways: GeoLattice + STAC, Earth Engine via `gee_to_xarray`, openEO.
- **GeoLattice:** `rs.access.stac_load`, `gee_init`, `gee_to_xarray`, `cdse_search`.
- **Others:** `earthengine-api`, `xee`, `geemap`, `openeo`, `planetary-computer`.

**08_10 · DevOps for Geospatial: Reproducible Deployment**
- Containerizing a GeoLattice service: GDAL in Docker, multi-stage builds.
- CI with GitHub Actions: tests, building COG and STAC outputs.
- Infrastructure basics: object storage, IAM, secrets.
- Scheduled pipelines, e.g. a weekly NDVI update appended to Zarr and republished as STAC.
- **Others:** Docker, GitHub Actions, `prefect` or `dagster` (optional).

**08_11 · Capstone: A Cloud Web-GIS for Environmental Monitoring**
- An end-to-end project with three layers:
  - **Data:** Sentinel-2 from STAC.
  - **Processing:** a monthly NDVI / flood product computed with GeoLattice on Dask and stored as Zarr plus COG, catalogued in STAC.
  - **Serving:** tiles via TiTiler, a FastAPI zonal-statistics endpoint, and a Streamlit front end with a map and time-series charts.
- Suggested sites from the course data: Onondaga County (NY), coastal Bangladesh, or the user's own area.
- Deliverables: a deployed URL, the repository, a README with the architecture diagram.

### Data reused from the course
- Onondaga Landsat, DEM and county layers (07_01_03).
- Sentinel-2 and Landsat over Syracuse and coastal Bangladesh (07_01_04, 07_02_00).
- NY county and US county shapefiles, converted to GeoParquet / PMTiles in 08_02.
- Public cloud sources: Planetary Computer, AWS Earth Search, Overture Maps, Source Cooperative.

### Gaps in GeoLattice this module would expose
These are worth deciding on before writing the notebooks:
- **COG validation:** no `validate_cog` helper; the notebooks would rely on `rio-cogeo`.
- **Tile serving:** no tile or OGC service layer; it would come from TiTiler and pygeoapi.
- **GeoParquet / PMTiles writing:** `open_cloud_native_vector_data` reads these formats, but I didn't find writers for them in `geolattice.vector`.
- **Missing install extra:** a `geolattice[web]` extra (folium, leafmap, lonboard, duckdb, fastapi, streamlit) would make setup one line.

I can save this as `Tutorials/notebooks/07_web_cloud_gis/README.md` or add it to `GeoLattice_Outline.md`, and then start on the 08_00 notebook.


## 9. Data problems found (not changed in the data files)

| File or source | Problem |
|---|---|
| Planet image in `data_remote_sensing` | Band 1 is identical to band 2 |
| Sentinel-2 sample | B7 is identical to B6 |
| `dg_multi` | Band 1 is saturated |
| Earth Search Sentinel-2 | Newer scenes already have the BOA offset applied (`earthsearch:boa_offset_applied`); subtracting it again double-corrects. 08_01 honours the flag |
| `bd_poverty.csv` | 544 rows but 525 distinct upazila codes: codes were assigned by name, so same-named upazilas share one (e.g. four "Kaliganj" all carry `BD303334`). A plain join attaches several values to some places and none to 19. 08_03 repairs it by district code + name: exact, then fuzzy (Jaro–Winkler ≥ 0.9), then a two-row manual lookup |
| `bd_upzila_GCS.shp` | File name typo ("upzila") |

---

## 10. Open items

- Module 08 notebooks 08_04 to 08_11 are planned in the README but not written.
- `Data/00_shared/data_web_cloud_gis/08_02_outputs` is about 1.2 GB of generated files; decide whether to keep,
  regenerate on demand, or exclude it from version control.
- Fix the `tests/test_ebk.py` reference in the `ebk.py` docstring.
- Rerun any earlier analyses that used Poisson or binomial GWGLM.
