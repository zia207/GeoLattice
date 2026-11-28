# Changelog

All notable changes to GeoLattice (formerly PyGIS) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Notebook builders (the ``_builders/`` folders of chapters 08_distributed_spatial, 09_05, 09_11, 09_12 and 09_13), the maintenance scripts in ``Tutorials/_tools/``, the banner and quick reference generators in ``Images/Banner/_src/`` and two unused notebook scripts (``variogram_kriging_jax.py``, ``download_lidar.py``) moved out of the repository to a local, git-ignored ``helper_functions/`` folder that keeps their original paths
- GeoLattice home page (``Tutorials/home``, built by ``build_home.py`` into ``docs/``) with chapter tiles, Getting Started, User Guide, API Reference and a Data page that links the tutorial data on Dropbox

### Added
- ``geolattice.rsense.soil_moisture`` (``rs.soil_moisture``): satellite soil moisture. Implemented: Saxton and Rawls (2006) pedotransfer functions (``saxton_rawls``, ``porosity_from_texture``), volumetric, gravimetric and saturation conversions, the Debye free water model, the Dobson (with Peplinski) and Mironov dielectric models, Topp and its inverse, penetration depth, Fresnel and H-Q-N reflectivity, SMAP vegetation water content and VOD, the tau-omega model, its single channel inversion and a Tb sensitivity table. Outlined (signatures and docstrings, raising ``NotImplementedError``): product registry and readers for SMAP, SMOS, AMSR2, ASCAT, ESA CCI, ERA5-Land, CLMS and NISAR; Sentinel-1 change detection, Water Cloud Model and surface scattering models; downscaling, gap filling, ISMN validation, triple collocation, root-zone filters, and drought, irrigation and hazard indicators. ``tests/rsense/test_soil_moisture.py`` (25 tests) checks the Saxton and Rawls table values, the water relaxation frequency, the Fresnel and tau-omega limits and the inversion round trip
- ``docs/make_api_reference.py`` marks outlined functions (bodies that only call ``_todo``) as planned; the API reference lists ``geolattice.rsense.soil_moisture`` and no longer places ``surface_energy_balance`` under a second Visualization heading
- ``geolattice.rsense.surface_energy_balance``: evapotranspiration from satellite images by the surface energy balance. Solar geometry and scene metadata, atmosphere and Open-Meteo ERA5 weather, broadband albedo (Landsat, Sentinel-2, MODIS), LAI, emissivity, net radiation, soil heat flux, roughness, aerodynamic resistance and Monin-Obukhov stability, ASCE standardized and FAO-56 reference ET (hourly and daily), automated hot and cold anchor pixels (Allen et al. 2013), SEBAL, METRIC, SSEBop, S-SEBI, the triangle method, PT-JPL, TSEB-PT, thermal sharpening, ETrF interpolation, seasonal and per zone ET, energy closure, accuracy metrics, sensitivity and anchor Monte Carlo. ``tests/rsense/test_surface_energy_balance.py`` checks FAO-56 examples 2, 8, 18 and 19 and the anchor and energy conservation constraints of each model
- Tutorial chapter ``Tutorials/notebooks/09_remote_sensing/09_12_surface_energy_balance_modeling``: 13 executed notebooks (fundamentals, inputs, reference ET, net radiation, soil heat flux and aerodynamics, SEBAL, METRIC, SSEBop and simplified models, TSEB-PT, MODIS and Sentinel-3 with Sentinel-2 thermal sharpening, seasonal ET and crop water use, validation and uncertainty, Boro rice capstone in Bangladesh), built by ``_builders/``, with the helper ``seb_data.py`` that downloads and caches Landsat, Sentinel-2, Sentinel-3, MODIS, Copernicus DEM, CDL (CropScape), WorldCover, ERA5 (Open-Meteo) and gridMET inputs under ``Data/08_remote_sensing/08_12_surface_energy_balance``

### Fixed
- ``surface_energy_balance.candidate_anchors``: when no dense pixel passes the cold-anchor albedo screen (e.g. spring cereals), or no low-NDVI pixel lies in the hot-anchor NDVI range (e.g. Sentinel-2 bare soil), that screen is skipped with a ``RuntimeWarning`` instead of the run failing
- ``surface_energy_balance.tseb_pt``: forcing heights below the canopy (2 m air temperature over 4 m orchards) made the aerodynamic resistance NaN for every tall-canopy pixel; heights below ``hc + 2`` m are now raised to it
- ``surface_energy_balance.thermal_sharpen``: the docstring now says that only unsmoothed residuals conserve the coarse LST exactly
- README, user guide (feature section, quick start, package layout, documentation table) and API reference cover the new module

### Changed
- Renamed the library from PyGIS to GeoLattice. The distribution is now ``geolattice`` (``pip install geolattice``, extras such as ``geolattice[all]``), the import package is ``geolattice``, and ``pygis.pygisdb`` is now ``geolattice.geolatticedb`` (``geolattice.spatial_bd.geolatticedb``). Environment variables use the ``GEOLATTICE_`` prefix (``GEOLATTICE_EXPLORER``, ``GEOLATTICE_FTW_CACHE``, ``GEOLATTICE_SEDONA_PACKAGES``, ``GEOLATTICE_UAV_WEIGHTS``); the old ``PYGIS_`` names are no longer read. Project URLs point at github.com/zia207/GeoLattice
- Tutorials, docs, tests, examples and data helper scripts use the new names; files named after PyGIS were renamed (for example ``Images/geolattice_logo.png`` and ``Data/06_spatialDB/outputs/06_01_02/geolattice.duckdb``)
- PyGIS Explorer, the separate Streamlit app, keeps its name: ``geolattice.rsense.get_images`` still looks for a ``PyGIS-Explorer`` folder (or the folder in ``GEOLATTICE_EXPLORER``), and the tutorials link to the app at zia207-pygis-explorer-app-ou8ehd.streamlit.app
- The logo and every banner in ``Images/Banner`` (PNG files and PowerPoint decks) show the GeoLattice name
- Radar tutorials link to the GeoLattice user guide instead of pygis.readthedocs.io, which belongs to a different project; ``docs/source/installation.rst`` clones github.com/zia207/GeoLattice instead of a ``yourusername`` placeholder

### Deprecated
- ``pygis`` remains as an alias package (``compat/pygis``): ``import pygis`` and ``from pygis.x import y`` return the ``geolattice`` modules and emit a ``FutureWarning``. It will be removed in a future release

## [0.3.49] - 2026-11-23

### Changed
- Installing PyGIS now installs every runtime dependency. ``requirements.txt`` lists them (83 packages: the conflict-free set that ``pygis[all]`` resolves to, plus the original core of NumPy, SciPy, pandas, Shapely, GDAL and pyproj), and ``pyproject.toml`` reads ``[project].dependencies`` from it (``dynamic = ["dependencies"]``), so ``pip install`` of the wheel or the sdist installs all of them. The sdist ships ``requirements.txt``. The installation options (``pygis[core]`` ... ``pygis[all]``) still work. GPU (``compute``, ``geostatistics-gpu``, ``rsense-gpu``, ``distributed-gpu``), cluster (``distributed-hpc``), Spark (``distributed-spark``), H2O, TensorFlow, large vision models, ``rsense-biopar`` (pandas < 3) and ``dev`` stay opt-in
- User guide: the individual extras list covers all 65 extras (12 were missing, such as ``geostatistics-gam``, ``rsense-foundation`` and ``spatial-statistics-all``); the package layout adds ``install_options.py``, ``network.py``, ``stac.py`` and the ``spatialcausal`` alias; new troubleshooting section for a GDAL prefix used without ``conda activate`` (``PROJ_DATA`` and the pyarrow/abseil clash with ``LD_LIBRARY_PATH``); a note on regenerating ``requirements.txt``
- ``requirements-gpu.txt`` installs ``cupy-cuda13x`` and ``dask-cuda`` (it named ``cupy-cuda11x``, which would clash with the CUDA 13 CuPy the GPU extras install)

- Python 3.10 or newer (``requires-python = ">=3.10"``): the dependency set cannot be installed on 3.9 (libpysal 4.10 and newer need 3.10). ``uv pip compile`` resolves ``requirements.txt`` with no conflicts for Python 3.10, 3.11, 3.12, 3.13 and 3.14; the test suite was run on 3.11 and 3.14. Classifiers, black ``target-version`` and mypy ``python_version`` follow
- ``pyproject.toml``: project URLs point at github.com/zia207/PyGIS (they were ``yourusername`` placeholders); the Documentation link is removed, since pygis.readthedocs.io belongs to a different project; description and keywords mention distributed and GPU computing
- CONTRIBUTING.md rewritten for this repository: GDAL setup, the ``src/`` to ``pygis`` mapping, tests against the installed package, ``requirements.txt`` and ``tools/make_requirements.py``, building wheels, regenerating the API reference, the Sphinx command (there is no ``docs/Makefile``) and the release checklist
- README: 348 tutorial notebooks (was 333); the remote sensing chapter links to ``Tutorials/notebooks/09_remote_sensing`` (the folder was renamed from ``08_remote_sensing``) with 128 notebooks, and spatial databases has 15

### Fixed
- ``tests/test_geostatistics.py::test_geostatistics_submodules_importable`` skips ``regression_kriging_tf`` when TensorFlow (an opt-in extra with no Python 3.14 wheel) is missing, as it already did for H2O and JAX

### Added
- ``tools/make_requirements.py`` regenerates ``requirements.txt`` from ``pyproject.toml`` (``--check`` reports drift); ``tests/test_requirements.py`` checks that the file is current, covers ``pygis[all]``, keeps opt-in packages out, has one plain requirement per line, and that the installed wheel declares the same requirements

### Verified
- ``pip install --dry-run --ignore-installed -r requirements.txt`` resolves with no conflicts on Python 3.11 and 3.14. Installing the wheel with dependencies adds 32 packages to an existing PyGIS 3.11 environment without changing any installed version; on 3.14 pygam 0.12 (scipy < 1.17), botocore and python-cmr (python-dateutil < 3) and PyTorch 2.15 (exact CUDA library versions) move scipy, python-dateutil and the NVIDIA CUDA libraries to older versions that no installed package excludes
- ``snaphu`` (``rsense-insar``) is installed on Python below 3.14 only: it has no 3.14 wheel and its C source fails to compile with GCC 14 or newer, which aborted the whole install on 3.14. Without it, ``rsense.insar.unwrap(method="snaphu")`` raises an install hint and ``method="native"`` works
- ``pycausalml`` (in ``spatialcausal-causalml``) is not on PyPI; it installs from the direct URL of its GitHub release wheel, which ``requirements.txt`` keeps

## [0.3.48] - 2026-11-22

### Added
- ``pygis.distributed`` (phase 1): one API for vector and raster processing on all CPU cores, a Dask cluster or CUDA GPUs. Every function takes ``compute="auto" | "cpu" | "gpu"``, returns the input's container type and memory space, and keeps Dask input lazy; GPU requests fall back to CPU with a warning
  - Sessions: ``start("local" | "cuda" | "slurm" | "pbs" | "address")`` returns a ``Session`` (context manager) with ``status``, ``scale``, ``adapt`` and ``dashboard_link``; ``cuda`` runs one ``dask-cuda`` worker per GPU
  - Backend and registry: ``capabilities``, ``resolve_compute``, ``set_defaults``, ``to_gpu``/``to_cpu``, ``compute``/``persist``; ``register``/``dispatch``/``list_ops`` map each operation to its CPU and GPU implementation
  - Partitioning: ``spatial_partition`` (Hilbert order), ``partition_bounds``, ``plan_chunks``
  - ``raster``: ``calc`` (string or callable expressions), ``where``, ``reclassify``, ``normalize``, ``normalized_difference``, ``focal`` (mean, sum, min, max, median, std), ``convolve``, ``map_tiles``, and terrain ``slope``, ``aspect``, ``hillshade``, ``curvature``, ``tpi``, ``tri``, ``roughness``. Moving windows read a halo from neighbouring chunks, so chunked results equal whole-array results on NumPy and CuPy
  - ``vector`` (CPU): ``buffer``, ``simplify``, ``centroid``, ``make_valid``, ``to_crs``, ``area``, ``length``, ``bounds``, ``clip`` (drops partitions outside the mask) and ``sjoin`` (Hilbert-partitioned left side, each task gets only the right rows inside its partition's bounding box; ``inner``/``left``, ``dwithin``); dask-geopandas frames stay lazy
  - ``io``: ``read_vector`` (one task per GeoParquet file, or parallel row ranges of one OGR file), ``write_vector`` (Hilbert-ordered partitioned GeoParquet, ``partition_on``), ``read_raster``, ``read_zarr``, ``read_stac``, ``write_raster`` (``zarr``, ``gtiff``, ``cog``)
  - Extras ``distributed``, ``distributed-gpu`` and ``distributed-hpc``
  - Phase 2, GPU vector engine on CuPy and cuDF (no cuSpatial; it has no CUDA 13 build). Geometries go to the GPU once in the GeoArrow ragged layout (``shapely.to_ragged_array``):
    - ``vector.area``, ``length`` and ``bounds`` on GPU for polygons (holes, multipolygons), lines and points; mixed geometry families run on CPU
    - ``vector.sjoin`` on GPU for points within/intersecting polygons (and polygons containing points): the STRtree bounding-box query gives candidate pairs and one CUDA kernel runs the exact crossing-number test. On 2 million points against 2,000 257-vertex polygons the exact test takes 1.0 s on the GPU against 33.9 s for Shapely on one thread; the whole join (20.6 million rows) takes 4.7 s against 15.4 s for ``geopandas.sjoin``. Output columns come from GeoPandas' own join builder, so they match ``geopandas.sjoin``
    - ``vector.knn``: k nearest points as a long table (``index_right``, ``distance``, ``k``), ``max_distance``, self-neighbours. CPU uses SciPy's multi-threaded KD-tree; GPU an exact float64 brute-force search (cuML's float32 distances misorder neighbours at projected coordinates)
    - dask-geopandas frames stay lazy in every vector function, including GPU ``sjoin``
  - Phase 2, ``bridge`` (raster and vector together, chunk by chunk):
    - ``rasterize``: each raster chunk gets only the features whose bounding boxes touch it; polygons burn on GPU by pixel centre, points, lines and ``all_touched`` use ``rasterio``; matches ``rasterio.features.rasterize``
    - ``zonal_stats``: count, sum, mean, min, max and population std per zone for ``(y, x)`` or ``(band, y, x)`` rasters, from per-chunk partial sums (NumPy or CuPy ``bincount`` and scatter) combined at the end; NaN cells ignored
    - ``extract_values``: nearest or bilinear sampling at points, reading only the chunks that hold them; points outside the raster get NaN
    - ``polygonize``: ``rasterio.features.shapes`` per chunk, then same-valued polygons that meet across chunk seams are merged (4- or 8-connectivity), so the result equals polygonizing the whole raster
  - Phase 3, ``analysis``: ``weights`` (k nearest neighbours, distance band, queen and rook contiguity; identical to libpysal), ``morans_i``, ``gearys_c``, ``local_morans`` (LISA) and ``getis_ord_local`` (G and G*), matching esda's statistics, normal inference and pseudo p-value definition; permutations run as batched sparse matrix products on CPU or GPU, and local statistics use esda's conditional permutation scheme. ``dbscan``, ``hdbscan`` and ``kmeans`` use cuML on GPU and scikit-learn on CPU. ``kde`` (Gaussian or quartic, weighted) builds a density raster chunk by chunk, each chunk receiving only the points in kernel reach
  - Phase 3, ``network``: ``graph_from_lines`` (end-point nodes, optional snapping tolerance), ``shortest_path_lengths`` and ``shortest_path`` (SciPy Dijkstra or cuGraph SSSP), ``centrality`` (betweenness, PageRank, degree), ``components``, ``nearest_nodes``. cuGraph counts shortest paths in int64, so betweenness on large lattices (over about 10^18 equal-hop paths between some nodes, e.g. a 35 x 35 street grid) is wrong on GPU; a float64 path-count probe detects this and runs betweenness on CPU with a warning
  - Phase 3, ``hydrology``: chunked D8 ``flow_direction`` on CPU or GPU (identical to ``rsense.lidar.terrain.flow_direction``); ``fill_depressions`` and ``flow_accumulation`` run in memory on CPU, since they depend on the whole DEM
  - Phase 3, scale-out: ``sedona`` engine (``spark`` session with Maven coordinates matched to the installed Sedona and PySpark, worker Python pinned to the driver's; ``sjoin`` returns a GeoDataFrame matching ``geopandas.sjoin``, or a lazy Spark DataFrame for Spark inputs; ``vector.sjoin(engine="sedona")``). ``start("kubernetes")`` through the dask-kubernetes operator. New extra ``distributed-spark``; ``distributed-hpc`` adds dask-kubernetes
  - Phase 4, ``ml``: ``fit`` (``"rf"``, ``"xgb"``, ``"lgbm"`` or any scikit-learn estimator; cuML random forest and XGBoost on CUDA for GPU; scikit-learn fits on an active cluster through joblib's Dask backend; ``xgboost.dask`` for Dask arrays), with labels encoded for cuML and XGBoost and decoded on prediction. ``predict_raster`` predicts every pixel of a ``(band, y, x)`` stack or Dataset chunk by chunk (NaN pixels stay NaN, probabilities with ``proba=True``, the model shipped to workers once). ``predict_tiles`` runs a PyTorch model over large images in overlapping tiles with one model replica per device (several GPUs, or CPU and GPU together), ``crop`` or ``cosine`` blending and optional autocast; the caller's model is never moved. ``train_ddp`` runs a function in one process per device with ``torch.distributed`` initialised (NCCL for one GPU per process, else Gloo) and returns each rank's result; ``ddp_model`` wraps a model in ``DistributedDataParallel``
  - Phase 4, ``diagnostics``: ``benchmark`` (warm-up, GPU synchronisation, lazy results materialised, speed-up column), ``check_equivalence`` (compares CPU and GPU results for arrays, DataArrays, Series, DataFrames, GeoDataFrames and result dataclasses), ``explain`` (which implementation an operation uses and why), ``memory_report`` (host, process, GPU and cluster workers) and ``profile`` (Dask performance report as HTML with a session, per-task timing table without). New extra ``distributed-ml``
  - ``register(..., auto=False)`` marks a GPU version as opt-in: ``compute="auto"`` keeps the CPU version for ``rasterize`` and ``dbscan``, whose GPU versions measured slower (0.5x and 0.03x); ``compute="gpu"`` still selects them. Measured GPU speed-ups elsewhere (RTX A5500, 8000 x 8000 DEM): focal median 74x, Moran's I 10.5x, LISA 6x, slope 3.7x

### Changed
- ``pygis[all]`` (installation option 6) now includes ``distributed`` and ``distributed-ml``. ``distributed-gpu``, ``distributed-hpc`` and ``distributed-spark`` stay out of it, like the other GPU and cluster extras, and are installed by name. ``pygis.install_options`` knows the new extras
- ``from pygis.distributed import *`` exports functions only; the submodules (``io``, ``network``, ``raster`` ...) are no longer star-exported, so they cannot shadow the standard library ``io`` or ``pygis.network``
- ``distributed.write_vector`` writes a GeoParquet 1.1 bbox covering column by default, so ``read_vector(..., bbox=...)`` works on its output
- README and user guide: a distributed computing section with runnable examples, the new extras and measured CPU and GPU timings; API reference regenerated with a ``pygis.distributed`` section

### Fixed
- ``distributed.network.components`` on GPU for directed graphs (cuGraph 26.08 takes undirected graphs only; weak components are now computed on the undirected version)
- ``tests/test_open_cloud_native_vector_data.py::TestGeoParquet::test_h3_cell_column`` assumed the ``h3`` package was missing; it now checks the cell hexagon when ``h3`` is installed and the warning when it is not

### Tests
- ``tests/test_distributed.py`` covers every public function of ``pygis.distributed`` (95% of its lines), comparing results with GeoPandas, rasterio, scipy.ndimage, libpysal, esda, NetworkX, scikit-learn and the existing ``rsense`` functions; ``tests/test_distributed_docs.py`` runs every example in the user guide's distributed section

## [0.3.47] - 2026-11-21

### Added
- ``pygis.rsense.foundation_models``, for the 08_11 foundation model tutorials:
  - ``chip_raster`` and ``FoundationEncoder.embed_raster`` accept a ``(time, band, y, x)`` series; chips come out as ``(n, band, time, h, w)``, the layout Prithvi takes
  - ``FoundationEncoder.dense_features`` and ``embed_raster(dense=True)``: per-patch features stitched into a georeferenced map (ViT tokens with class and register tokens dropped, ResNet and Swin feature maps, DOFA tokens)
  - ``chip_metadata``: longitude, latitude, date, day of year and area of every chip; ``embed_raster(metadata=True)`` feeds it to Copernicus-FM (``metadata``) and Prithvi-TL (``temporal_coords``, ``location_coords``)
  - ``benchmark_encoders``: linear and kNN probes, embedding time and GPU memory for several encoders (or precomputed baselines) on one labelled set
  - ``sample_alphaearth`` takes several years and returns a long table with a ``year`` column
  - Embedding maps record the normalization used in their attributes
  - AlphaEarth without an Earth Engine account: ``alphaearth_to_xarray(source="zarr")`` reads the Source Cooperative GeoZarr mosaic and ``source="gcs"`` the Google Cloud Storage GeoTIFFs (from overviews, for coarse maps), both anonymously; ``source="auto"`` (the default) picks Zarr below 100 m and GCS above, and ``like=`` puts the result on an existing grid. ``sample_alphaearth`` reads points from the Zarr mosaic in parallel by default. New ``alphaearth_dequantize`` (int8 files to float embeddings) and ``alphaearth_index`` (cached tile index). Earth Engine stays available with ``source="earthengine"``
  - ``alphaearth_to_xarray`` downloads in tiles (``tile=256`` pixels, under Earth Engine's response limit) and mosaics them, so whole study areas fit in one call
  - CROMA's joint mode moves the radar chips to the model's device; ``embed_raster`` accepts radar band names (``vv``, ``vh``) for Panopticon
  - ``embed_raster`` resizes chips batch by batch instead of all at once, so overlapping small chips (for example 32 pixels every 16, resized to 224) no longer need tens of gigabytes of memory
  - ``dense_features`` takes the number of time steps from a ``(B, C, T, H, W)`` input, so Prithvi tokens of four frames (785 = 1 + 4 x 196) are not mistaken for one 28 x 28 frame
  - ``input_spec`` and ``INPUT_SPECS``: the band order, units, statistics and chip size each loadable model expects (from TorchGeo weight transforms, TerraTorch pretraining constants and model cards); ``prepare`` reorders bands, fills missing ones from the nearest wavelength, converts reflectance, DN or dB and normalizes; ``embed_raster(norm="model")`` applies it

### Tutorials
- Chapter ``Tutorials/notebooks/08_remote_sensing/08_11_foundations_models``: notebooks 08_11_00 and 08_11_03 to 08_11_13 executed (08_11_07 and 08_11_08 read AlphaEarth from its public copies, no Earth Engine account needed); built by scripts in ``_builders/`` (``run.py`` builds and executes with ``--allow-errors`` and dumps outputs; ``sync_text.py`` updates markdown while keeping executed outputs). 08_11_01 and 08_11_02 were moved here from chapter 08_09

## [0.3.46] - 2026-11-21

### Added
- ``pygis.rsense.foundation_models``: one module for remote sensing foundation models
  - Catalog of 45 models in six groups (multispectral optical, multi-sensor, time series and embedding products, very high resolution, vision-language, weather), with provider, year, architecture, modalities, embedding size, weights, loader, use cases and reference. ``list_models``, ``model_info``, ``get_model_spec`` (with aliases such as ``"dofa"``) and ``recommend(task, sensor=)``
  - ``load_model`` builds 24 models with public weights through TorchGeo (DOFA, CROMA, Copernicus-FM, Panopticon, Scale-MAE, SSL4EO-S12/S1, SatlasPretrain), TerraTorch (Prithvi-EO 1.0 and 2.0, TerraMind), open_clip (RemoteCLIP, GeoRSCLIP), ``torch.hub`` (AnySat) and ``transformers`` (DINOv3-sat), and wraps each in ``FoundationEncoder`` (``encode``, ``embed``, ``embed_raster``, ``forward_features``). Wavelength-aware models get wavelengths and bandwidths from band names (``band_wavelengths``, ``WAVELENGTHS_NM``, ``BANDWIDTHS_NM``)
  - ``chip_raster`` and ``normalize`` for model input; ``embed_raster`` returns a georeferenced ``(embedding, y, x)`` map
  - AlphaEarth Foundations (Google Satellite Embedding, Earth Engine): ``alphaearth_image``, ``sample_alphaearth`` and ``alphaearth_to_xarray``
  - ``linear_probe``, ``knn_probe`` (spatial ``groups=``), ``similarity_map``, ``embedding_search``, ``cluster_embeddings``, ``embedding_change``, ``embedding_rgb``, ``embeddings_to_frame``, ``pool_features``, ``zero_shot_classify``, ``segmentation_model`` and ``terratorch_config``
- ``pygis[rsense-foundation]`` extra (TorchGeo, TerraTorch, timm, open_clip, huggingface_hub, transformers, earthengine-api); like ``rsense-vision`` it is not in ``pygis[all]``
- ``tests/rsense/test_foundation_models.py``: catalog, preprocessing, encoder wrapper, embedding analysis, AlphaEarth with a fake Earth Engine module, and real-weight embedding tests for checkpoints already in the local cache (15 models). Model tests run on the CPU and, when PyTorch sees one, on a CUDA GPU, where embeddings must match the CPU ones (cosine similarity above 0.999)

### Fixed
- ``pygis.rsense.__version__`` was stuck at 0.3.38; it now tracks PyGIS

## [0.3.45] - 2026-11-21

### Added
- ``rsense.uav.bare_soil_line``: minimum bare-soil line for red and near-infrared digital numbers (``landsat::BSL`` minimum method, major-axis slope). Returns the slope, intercept, full-canopy point and fit size
- ``rsense.uav.cover_layers``: perpendicular vegetation index (PVI), green ground cover (PVI divided by PVI at full canopy) and WDVI on a red/NIR image (Maas and Rajan 2008). Coefficients and ``pvi_fc`` are on the Dataset attributes. ``pvi_max=None`` skips the clip at 100
- Tests in ``tests/rsense/test_uav_workflows.py``, including the 12 February 2015 Barisal wheat mosaic
- Tutorial ``08_04_04`` calls these functions for the Barisal ground-cover section

## [0.3.44] - 2026-11-18

### Added
- Six cumulative installation options; each includes everything in the one before it:
  ``pygis[core]`` (vector and raster I/O, CRS, visualization: ``io``, ``viz``),
  ``pygis[analysis]`` (+ spatial analysis: ``multidim``, ``network``, ``segregation``, ``segregation-dasymetric``, ``whitebox``),
  ``pygis[statistics]`` (+ spatial statistics and spatial causal inference: ``spatial-statistics``, ``spatial-statistics-viz``, ``spatialcausal``, ``spatialcausal-viz``),
  ``pygis[geostats]`` (+ ``geostatistics``, ``geostatistics-plot``),
  ``pygis[remote-sensing]`` (+ ``rsense``, ``rsense-access``, ``rsense-sentinel3``, ``rsense-sentinel5p``, ``rsense-sar``, ``rsense-lidar``, ``rsense-hyperspectral``, ``rsense-uav``, ``rsense-classify``, ``rsense-viz``) and
  ``pygis[all]``. All six resolve with pip on Python 3.11 and 3.14
- README "Installation options" table, and building a wheel for another Python version from the sdist
- ``pygis.install_options``: ``INSTALL_OPTIONS``, ``option_for(extra)`` and ``install_hint(extra)``, which names an extra together with the smallest installation option that contains it (read from the installed package metadata). Missing-dependency errors from ``pygis.geostatistics`` stubs and ``pygis.rsense.access`` use it, for example ``pip install "pygis[rsense-access]" (included in installation option 5, "pygis[remote-sensing]")``
- Every tutorial notebook names the installation option it needs in an ``**Installation.**`` note (in its Setup cell, or before the first code cell), worked out from its imports by ``Tutorials/_tools/notebook_install_notes.py``, with any library outside every option listed by name

### Changed
- GPU extras target CUDA 13: ``compute`` and ``geostatistics-gpu`` (previously ``cupy-cuda11x``) and ``rsense-gpu`` (previously ``cupy-cuda12x``) now install ``cupy-cuda13x>=13.6``; the README explains how to use an older CUDA, and the tutorial GPU hints name ``cupy-cuda13x``
- ``pygis[io]`` (and so ``pygis[core]``) now includes ``pyarrow`` for GeoParquet and Feather, which ``pygis.vector`` reads and writes
- ``pygis[all]`` now means installation option 6: ``remote-sensing`` plus ``spatial-statistics-all``, ``spatialcausal-all``, ``geostatistics-jax``, ``geostatistics-pytorch``, ``geostatistics-gam``, ``geostatistics-optuna``, ``rsense-all``, ``web-cloud`` and ``spatial-db-all``. It no longer includes ``dev`` or ``compute`` (CuPy for CUDA 11); add those, the other GPU groups, ``geostatistics-h2o``, ``geostatistics-tensorflow``, ``rsense-uav-dl``, ``rsense-vision`` and ``rsense-biopar`` by name

### Changed
- ``pygis.geostatistics.jax_variogram_kriging``: ``SimpleKriging``, ``OrdinaryKriging``, ``UniversalKriging`` and ``Cokriging`` now factor the covariance matrix once in ``fit`` (Cholesky, kept on the device) and predict with ``jax.jit`` kernels on fixed-size, padded batches, instead of rebuilding and LU-solving the whole bordered system on every ``predict`` call. Variogram parameters are traced arguments, so new parameters do not recompile. Predictions and standard errors match the previous code to within 1.3e-12 (relative) for all six variogram models, on CPU and GPU. On a 40,000-cell grid with 2,000 training points, a steady ``predict`` on the GPU went from 2.8 to 0.67 s (ordinary kriging), and JAX on the CPU is now faster than the NumPy core instead of slower. When the Cholesky factor is not finite (for example duplicate locations with no nugget) the classes fall back to the bordered solve
- New ``dtype="float32"`` option on the four JAX kriging classes for GPUs with slow float64 (distances stay float64 and predictions use the whitened form vᵀL⁻¹y to avoid cancellation); on the Great Plains SOC data predicting with variances it was 10 to 15 times faster than float64 on the GPU, with predictions within 0.006 of float64
- Tutorial ``05_02_05`` times kriging on a 40,000-cell grid at several training sizes, adds a float32 GPU column and a section on getting good performance from JAX kriging
- JAX variogram model functions keep float32 input in float32 (they used to cast everything to float64), and ``_pairwise`` requests full-precision matrix products so GPUs do not use TF32

### Fixed
- The JAX kriging least-squares fallback for singular systems never ran: ``_solve`` caught ``jnp.linalg.LinAlgError``, which current JAX does not define, so a singular system raised ``AttributeError``
- Install hints that named the old ``pygstat[...]`` extras (``plot``, ``gpu``, ``optuna``, ``gam``) now name the PyGIS extras (``geostatistics-plot``, ``geostatistics-gpu``, ``geostatistics-optuna``, ``geostatistics-gam``)
- Tutorials ``05_07_01`` and ``05_07_03`` imported ``pygis.geostatistics.pygstat.*``; they now import ``pygis.geostatistics.simulation.sgsim``, ``.simulation.sisim`` and ``.utils.backend``. ``08_02_02`` and ``08_11_15`` imported the retired ``pyrsense`` package; they now import ``pygis.rsense``

## [0.3.43] - 2026-11-18

### Changed
- ``pygis.spatialcausal`` moved to ``pygis.spatial_analysis.spatialcausal`` (``src/spatialcausal`` → ``src/spatial_analysis/spatialcausal``), next to ``segregation``. ``pygis.spatial_analysis`` loads it lazily on first access, so importing ``pygis.spatial_analysis`` stays light; all internal imports, tests, the README, the segregation ``export_for_causal`` docs and the spatial causal and segregation tutorials (``Tutorials/notebooks/03_spatial_analysis/03_06_*``, ``03_07_*`` and ``countycausal.py``) use the new path. The ``spatialcausal`` install extras keep their names
- ``pygis.spatialcausal`` remains as a deprecation shim: it emits a ``DeprecationWarning`` and aliases the package and every submodule (``pygis.spatialcausal.quasi`` and so on) to the new modules, so old imports return the same classes and objects

### Added
- ``rsense.hyperspectral.read_tanager(variable="surface_reflectance_uncertainty")``: the uncertainty layer has no band attributes of its own, so wavelengths, FWHM and good-band flags now come from the reflectance cube in the same file, while the fill value and units stay those of the uncertainty layer
### Fixed
- ``pygis.spatial_statistics.__version__`` was stuck at 0.3.22 while ``test_top_level_exports`` expects it to match the package version; it now tracks PyGIS (0.3.43)

### Tests
- ``test_old_import_path_is_an_alias`` for the shim and an uncertainty-layer case in ``test_tanager_reader``
- Tutorials ``09_remote_sensing/09_05_HyperSpectral/09_05_00`` to ``09_05_11`` (imaging spectroscopy fundamentals, reading and exploring cubes, preprocessing, dimensionality reduction, narrowband indices, trait retrieval, libraries and matching, unmixing, detection, methane plumes from HITRAN, thematic applications, deep learning on Pavia), built by scripts in ``_builders/``
- Banners and a vector / raster quick reference in ``Images/Banner/`` with their generators in ``Images/Banner/_src/``

## [0.3.42] - 2026-11-18

### Added
- ``rsense.hyperspectral.read_tanager``: Planet Tanager HDF-EOS5 products (``*_ortho_sr_hdf5.h5``, ``*_ortho_radiance_hdf5.h5`` and the unorthorectified ``*_basic_radiance_hdf5.h5``) as a cube. Wavelengths, FWHM and ``good_wavelengths`` come from the variable attributes (water vapour bands are flagged when a product has no good-band list), ortho grids get their UTM CRS and pixel centres from ``StructMetadata.0``, swaths keep ``(band, line, sample)`` with ``lat``/``lon`` from the file or a ``geolocation`` GeoTIFF. Reads subsets by ``bands``, ``wavelength_range``, ``window`` or lon/lat ``bbox``; fill and ``nodata_pixels`` become NaN and ``mask_clouds`` blanks the beta cloud and cirrus masks
- ``rsense.hyperspectral.read_tanager_ancillary``: the 2-D layers of a Tanager product (cloud, cirrus and no-data masks, aerosol optical depth, column water vapour, sun and sensor angles, path length, time) as a Dataset on the cube grid
- ``rsense.hyperspectral.read_ohs``: Orbita OHS (Zhuhai-1) L1 scenes stored as one GeoTIFF per band, stacked in band order with the nominal 32 band centres (``OHS_WAVELENGTHS``, overridable), metadata from ``*_meta.xml``, band and window subsets, and optional RPC orthorectification (``ortho=True``, UTM zone of the scene centre by default)
- ``rsense.hyperspectral.swath_to_grid``: nearest-neighbour gridding of any ``(band, line, sample)`` swath with per-pixel lat/lon (Tanager basic, PRISMA L2B/L2C) onto a UTM or user CRS
- ``read_hyperspectral`` now recognises Tanager ``.h5`` files and OHS scene folders or band files
- Tests for the new readers in ``tests/rsense/test_hyperspectral.py``
- ``Tutorials/notebooks/09_remote_sensing/09_05_HyperSpectral/OUTLINE.md``: plan for the hyperspectral tutorial series

## [0.3.41] - 2026-11-17

### Added
- ``rsense.uav.synthetic_dem``: GCP-free style DSM of a plot trial (domed, tilted ground with canopy plots, alleys and noise) plus its true canopy height, for testing ``ransac_flatten`` and teaching without real DEMs
- ``rsense.uav.detect_stages``: onset / mid / plateau dates (any fractions) from a ``fit_logistic`` result
- ``rsense.uav.frame_traits``: canopy cover, index means inside the vegetation mask, senescent fraction and optional plant count for one frame, ready for ``trait_table``
- ``rsense.uav.edge_agreement``: Sobel edge-map correlation of two bands away from NaN borders, a fair cross-band alignment score
- ``coregister(validate=True, min_inliers=15, max_shift=None)``: rejects registrations with too few inliers, implausible shifts or lower edge agreement and returns the band unwarped; ``info["accepted"]`` and ``info["reason"]`` report the outcome. Repetitive crop rows can lure ORB, SIFT and ECC onto a neighbouring row
- ``ransac_flatten(match="ground")``: level each date on a low percentile (bare soil) instead of the mean
- ``classify_senescence(hue_max=...)``, ``health_map(hue_max=...)`` and ``frame_traits(hue_max=...)``: expose the colour-rule hue cut (olive-green canopies sit below the 75° default; about 62° for the LUMS rice)
- ``fit_logistic(min_transition_days=..., limits=...)``: the rate is bounded so the 10-90 % transition lasts at least twice the median flight interval (noisy weekly data no longer collapse into a step), and ``limits`` keeps both asymptotes in a physical range such as ``(0, 1)``
- Tutorials ``09_remote_sensing/09_04_uav/09_04_01`` to ``09_04_10``: mission planning, reading and QC, photogrammetry, RGB indices and canopy cover, RGB + NIR co-registration and calibration, DEM correction and canopy height, plant counting and size, SAM and senescence, tiled panicle detection, season-long phenotyping and a capstone pipeline (LUMS rice and paired RGB/NIR crop data)
- Tests in ``tests/rsense/test_uav_workflows.py`` for all of the above

### Fixed
- ``rsense.uav.cosine_weight`` was a ramp peaking at the bottom-right tile corner (cosine over 0..pi, as in the LUMS ``predict_orthomosaic.py`` scripts), not a centred Hann window, so ``predict_tiled(blend="cosine")`` and ``density_count`` stitching favoured one tile in every overlap. It now uses the full 0..2pi period
- ``cwsi(method="empirical")``: the default non-water-stressed baseline ``nwsb=(-1.9, 3.4)`` made an unstressed canopy warmer than air as VPD rose. The default is now Idso's (1982) wheat baseline ``(3.38, -3.25)``, and a baseline giving Twet >= Tdry raises ``ValueError``
- ``build_catalog`` dropped the camera-rig suffix of band folders (``RGB-1``, ``RGB-2``), so ``pair_catalog`` merged two rigs by frame number and lost 966 Drybean pairs. The catalogue now has a ``rig`` column and pairs are matched within a rig

### Changed
- ``ransac_flatten`` keeps ``match="mean"`` (the LUMS script) as the default, but its docstring now notes that mean matching pulls a growing canopy down and underestimates growth

## [0.3.40] - 2026-11-13

### Added
- ``pygis.geostatistics.jax_variogram_kriging``: JAX CPU/GPU ports of the core variogram and kriging functions (theoretical models including Matérn, Matheron / Cressie–Hawkins / Dowd empirical variograms, variogram cloud, indicator variogram, ordinary, simple, universal, and cokriging, plus block kriging). ``use_gpu`` is ``"auto"``, ``True`` or ``False``. Package aliases ``JaxVariogram``, ``JaxOrdinaryKriging``, ``JaxSimpleKriging``, ``JaxUniversalKriging`` and ``JaxCokriging`` do not replace the NumPy/CuPy classes
- Tutorial ``05_02_kriging/05_02_05_comparison_simple_ok_uk_ck_jax.ipynb``: simple, ordinary, universal, and cokriging on a known field, core CPU against JAX CPU and JAX GPU (agreement, error, first-call and steady-state time)
- Optional extra ``pygis[geostatistics-jax]`` (``jax``, ``jaxlib``)
- Tests: ``tests/test_jax_variogram_kriging.py``

## [0.3.39] - 2026-11-11

### Added
- Sequential indicator simulation with local prior means in ``pygis.geostatistics.sis_local_means``: ``sis_local_means``, ``sklm_cdf``, ``sk_correction``, ``residual_covariances``, ``order_relations`` (continuous SIS when each node has a secondary prior CDF; residual indicator SK)

### Changed
- Capstone notebook ``05_09_02_arsenic_cancer_risk.ipynb`` now calls ``sis_local_means`` / ``sklm_cdf`` from the package and no longer embeds a local copy of the algorithm or manuscript comparison tables

## [Unreleased]

### Changed
- Notebook builders (the ``_builders/`` folders of chapters 08_distributed_spatial, 09_05, 09_11, 09_12 and 09_13), the maintenance scripts in ``Tutorials/_tools/``, the banner and quick reference generators in ``Images/Banner/_src/`` and two unused notebook scripts (``variogram_kriging_jax.py``, ``download_lidar.py``) moved out of the repository to a local, git-ignored ``helper_functions/`` folder that keeps their original paths
- GeoLattice home page (``Tutorials/home``, built by ``build_home.py`` into ``docs/``) with chapter tiles, Getting Started, User Guide, API Reference and a Data page that links the tutorial data on Dropbox

## [0.3.38] - 2026-11-10

### Fixed
- ``spatial_analysis.fill_depressions`` did not fill DEMs bounded by nodata. The priority flood was seeded from the raster edge only, so a DEM clipped to a polygon outline was barely filled. The flood is now seeded from every valid cell on the data boundary (raster edge or next to nodata). ``breach_depressions`` and ``stochastic_depression_analysis`` use the same routine and are fixed too
- ``spatial_analysis.fill_depressions(fix_flats=True)`` left flats undrained after writing. The automatic flat increment (about 1e-6 m) was below float32 precision at ordinary elevations, and a larger ``flat_increment`` created new pits next to the flat. Flat resolution now uses Priority-Flood + epsilon (Barnes, Lehman and Mulla, 2014) in float32 arithmetic, so every interior cell keeps a flow direction in the float32 output. ``flat_increment`` sets an optional larger step and must be positive (``SurfaceHydrologyError`` otherwise)

### Removed
- Private helper ``surface_hydrology._resolve_flats`` (replaced by the epsilon step above; no public API change)

### Added
- Tests: ``test_fill_depressions_nodata_bounded_dem_drains_everywhere`` and ``test_fill_depressions_flat_increment`` in ``tests/test_surface_hydrology.py``
- Tutorial ``03_spatial_analysis/03_07_capston/03_07_02_watershed_characterization.ipynb``: Onondaga Lake watershed capstone on the county DEM (100 m UTM grid): conditioning, D8 and D-infinity routing, stream-threshold sensitivity, Strahler order, gauge-based delineation checked against published drainage areas, sub-basins at major confluences, morphometric indices, hypsometric curves, slope, wetness, landforms and height above stream by sub-basin, Kirpich time of concentration, compound-ranking prioritisation with sensitivity checks, and a GeoPackage of results. Shared helpers in ``03_07_capston/data_processing.py``; external layers for the Module 03.07 capstones are fetched with ``03_07_capston/download_data.py``

### Changed
- Outputs of ``03_02_digital_terrain_modeling/03_02_03_hydrology_modeling.ipynb`` change when re-run, because its DEM conditioning uses the fixed ``fill_depressions``

## [0.3.37] - 2026-11-06

### Added
- ``pygis.rsense.uav`` workflow modules for UAV phenotyping, built around the LUMS infrastructure-free rice pipeline (Chaudhry, Rana & Jaleel 2026) and RGB + NIR crop datasets:
  - ``uav.io``: ``read_frame`` (PNG/JPG/TIFF to ``(band, y, x)`` with metre coordinates from a GSD, or pixel coordinates), ``as_image``, ``to_rgb_array``, ``load_pair`` (RGB + NIR into one array, optional co-registration), ``build_catalog`` / ``pair_catalog`` for dated flight folders (``crop_DDMMYYYY/RGB|NIR``, ``YYYY-MM-DD``), ``parse_date``, ``extract_frames`` (every n-th video frame), ``blur_score``, ``exposure_stats``, ``frame_quality``
  - ``uav.register``: ``coregister`` (ORB/SIFT + RANSAC homography with ECC refinement kept only when it raises correlation), ``estimate_homography``, ``refine_ecc``
  - ``uav.indices``: ``vegetation_indices`` (RGB and NIR indices through ``optical.index``, plus ``CIVE``, ``TGI_MAXNORM`` and ``EXG_MINMAX``), ``chromatic``, ``index_summary``
  - ``uav.tiling``: ``sliding_windows`` (edge-reaching overlapping tiles), ``cosine_weight``, ``predict_tiled`` (cosine/mean/max blending), ``merge_points``, ``merge_boxes`` (cross-tile NMS)
  - ``uav.dem``: ``ransac_flatten`` (one RANSAC polynomial trend fitted on a reference DEM and removed from every date; ``ground``, ``none`` or original ``lums`` normalisation), ``trend_surface``, ``doming_profile``, ``write_dems``
  - ``uav.counting``: RiceNet (``build_ricenet``, ``load_ricenet``), ``density_count`` (tiled density maps, peak extraction, cross-tile merge, adaptive plant size in px and m), ``peaks_from_density``, ``size_classes``
  - ``uav.detect``: ``tiled_detect`` (ultralytics, original YOLOv5 via torch.hub, or any callable; drops boxes cut at window seams by default), ``load_detector``, ``boxes_to_gdf``, ``count_summary``, ``detection_scores``
  - ``uav.health``: ``sam_plant_mask`` (SAM with ExG-derived point prompts), ``exg_plant_mask``, ``segveg_features`` (SegVeg's 14 colour features, original black-term behaviour by default), ``load_segveg``, ``classify_senescence`` (SegVeg model or colour rule), ``hsv_filter``, ``health_map`` (tiled), ``health_summary`` (per zone)
  - ``uav.phenology``: ``trait_table``, ``summarize_dates``, ``fit_logistic`` (inflection date, maximum rate), ``growth_rates``
  - ``uav.weights``: registry and cache for SAM ViT-B/L/H, LUMS RiceNet and YOLOv5 panicle weights (``fetch_weights``, ``list_weights``; ``$PYGIS_UAV_WEIGHTS``)
- ``uav.flight_plan``: ``camera=`` presets (``CAMERAS``: Phantom 4 Pro, Mavic 3M RGB, P4 Multispectral, RedEdge-MX, Mini 4 Pro photo and 4K video), ``camera_from_equivalent``, and video capture (``fps``, ``frame_step``: ``frame_spacing_m``, ``video_front_overlap``)
- ``uav.load_outputs``: explicit ``orthophoto``, ``dsm``, ``dtm``, ``point_cloud`` paths (Metashape/Pix4D exports)
- ``rs.core.pix_to_map`` / ``map_to_pix`` (also ``rs.uav.pix_to_map``) as public functions
- ``pygis[rsense-uav]`` now includes OpenCV, scikit-learn and gdown; new ``pygis[rsense-uav-dl]`` (segment-anything, ultralytics, xgboost, PyTorch)
- Tests: ``tests/rsense/test_uav_workflows.py`` (25 tests)

### Fixed
- ``uav.chm_from_surfaces`` aligned the DTM only when shapes differed; same-shape rasters on offset grids were subtracted unaligned. Grids are now compared by CRS and coordinates
- ``uav.count_plants`` accepts plain 2-D arrays, single-band ``(band, y, x)`` arrays and an explicit ``gsd``, handles non-square pixels, and returns ``row``/``col``
- ``uav.plants_per_area`` projects geographic polygons to UTM before computing areas (densities were per square degree) and returns ``area``
- ``uav.sam_segment`` validates prompts (``points``/``boxes``/``text_prompt``), accepts georeferenced DataArrays, and no longer defaults to Apple MPS

## [0.3.36] - 2026-11-06

### Added
- ``pygis.spatial_bd.pygisdb`` (new subpackage ``pygis.spatial_bd``; also importable as ``pygis.pygisdb``): one ``SpatialDB`` interface for PostGIS, SpatiaLite, GeoPackage and DuckDB spatial. ``connect`` (backend aliases ``pg``, ``sqlite``, ``gpkg``, ``duck``; DSN, keyword or ``PG*`` environment settings for PostGIS), ``load`` (vector files, GeoDataFrames and rasters, SRID detection, optional spatial index), ``query`` / ``read_table`` (SQL back to GeoDataFrames with CRS), ``execute``, ``tables``, ``columns``, ``geometry_column``, ``srid``, ``spatial_join`` (ten predicates), ``nearest`` (k-NN with ``max_distance``), ``buffer`` (planar or geography, dissolve), ``create_index``, ``explain`` (with ``analyze``), ``vacuum``. PostGIS extras: ``db.raster`` (``load`` via raster2pgsql, ``info``, ``zonal_stats``, ``value_at``), ``db.network`` (pgRouting ``prepare``, ``nearest_vertex``, ``shortest_path``, ``isochrone``) and ``db.tiles`` (``mvt``, ``bounds``, ``serve`` / ``stop`` for a threaded vector-tile server). Each method is also a module function taking the connection first. Tests: ``tests/test_pygisdb.py`` (PostGIS cases run when ``PYGIS_TEST_PG_DSN`` is set)
- ``pygis[spatial-db]`` (GeoPandas, Shapely, pyogrio, DuckDB), ``pygis[spatial-db-postgis]`` (SQLAlchemy 2, psycopg 3, GeoAlchemy2, rasterio) and ``pygis[spatial-db-all]`` optional dependencies; ``spatial-db`` is included in ``pygis[all]``
- Tutorial ``07_remote_sensing/07_09_object_detection/07_09_00_object_detection_intoduction.ipynb``: chapter introduction to object detection and field edge detection. Covers task types (boxes, oriented boxes, masks, boundaries, boundary-to-instance), IoU and NMS, mAP at VOC and COCO thresholds, detector families (Faster/Mask R-CNN, RetinaNet, FCOS, YOLO, DETR), boundary networks and pretrained field models (FTW, Delineate Anything, SAM 2.1), overhead-imagery adaptations, and maps of the ``ic`` and ``ed`` functions. Hands-on examples use the chapter data: tree box statistics, ``detection_tiles`` and ``merge_detections``, Sobel edges to field instances at two thresholds, and the detector and edge models built without downloads. Ends with a guide to the chapter's notebooks
- Tutorial ``07_remote_sensing/07_10_Capstones/07_09_14_edge_detection.ipynb``: field-boundary detection across four countries (``data/Edge_detection_global_data``). Trains on one site each from Mexico, Mali, Cambodia and Bangladesh (Babuganj) and tests on an independent second Bangladesh site (Chandpasha). Steps: digitised lines closed into fields, a per-site 0.5 m RGB grid and stretch, ``rasterize_boundaries``, a block split inside each site with overlapping ``tile_pairs`` and ``dihedral_augment``, classical detectors, a texture random forest, and U-Net, SegNet, FC-DenseNet56/67/103 and FCN-DKConv6 with a learning-rate sweep. Reports per-site validation and whole-site test scores (with tolerance and ROC), and compares training on Bangladesh only, other countries only and all four sites. Ends with Chandpasha field polygons and object matching
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_16_field_edge_detection_sentinel2.ipynb``: Fields of The World (``ftw_models``, ``load_ftw_model``, ``ftw_stack``, ``detect_fields_ftw``; B3, B5, B7, B5 CC-BY) versus Delineate Anything (DA-S, v1, v2) and classical edge detectors on Sentinel-2 2024/2025 over the Corn Belt AOI, scored against CDL-derived field parcels (pixel boundary F1 with tolerance, one-to-one object matching, over- and under-segmentation); FTW windows, offset and up-sampling and Delineate Anything model, date and scale chosen on 2024 and tested on 2025; year-to-year stability; fiboa export with ``to_fiboa``
- ``pygis.rsense.edge_detectors``: Fields of The World (FTW) support. ``ftw_models`` lists the FTW U-Net checkpoints (v3 ``FTW_PRUE_EFNET_B3/B5/B7``, the ``_CCBY`` variants, legacy v1/v2) with licences and release URLs; ``ftw_checkpoint`` resolves a local ``.ckpt`` or a cached registry checkpoint (download only with ``download=True``); ``load_ftw_model`` rebuilds the network with ``build_segmenter`` from the checkpoint's stored settings and loads the Lightning state dict (``weights_only=True``); ``ftw_stack`` builds the two-window B04-B03-B02-B08 input; ``predict_ftw`` / ``detect_fields_ftw`` reproduce FTW inference (÷3000, up-sampling, Hann-blended tiles) and return field polygons; ``to_fiboa`` adds the fiboa core properties. ``FieldEdgeResult`` gains an optional ``class_probability``. Three new tests
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_09_change_detection.ipynb``: Corn Belt change detection 2024 -> 2025 on Landsat 8 and Sentinel-2 with CDL as reference: ``check_coregistration`` and ``histogram_match`` (with a pseudo-invariant check), ``image_difference``, ``image_ratio``, ``change_vector_analysis`` / ``classify.cva``, ``pca_change`` with k-sigma and Otsu thresholds, ``classify.post_classification`` from-to matrices, an analysis of CDL change (rotation vs edge noise), ``build_change_model`` (Siamese-diff, Siamese-concat, early-fusion U-Net, BIT) with ``PairChipDataset`` and cross-sensor transfer to Landsat, ``change_metrics`` on all and interior pixels, ``olofsson`` area estimates with repeated-sample coverage, and a rotation-map exercise
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_08_spatio_temporal_multi_sensor_crop_mapping.ipynb``: Corn Belt crop mapping from six seasons (2020-2025) of Sentinel-2, Sentinel-1 and Landsat with CDL labels: ``pixel_time_series`` with cloud masks and gap-aware temporal interpolation, ``temporal_split`` and a leakage comparison (random vs temporal vs spatio-temporal), Random Forest on phenology metrics, ``build_temporal_model`` (TempCNN, LSTM, L-TAE, PSE-TAE on parcel proxies, ConvLSTM, U-TAE), early / middle / late optical-SAR fusion under simulated cloud (with cloud-aware training and availability-weighted late fusion), ``sar_assisted_gap_fill`` with the reconstruction flag, Wald-protocol ``pansharpen`` / ``pansharpening_quality`` for Sentinel-2 and Landsat, strict in-season mapping and a 2025 wall-to-wall map
- Tutorial ``07_remote_sensing/07_10_Capstones/07_09_13_Sentinel1_sentinel2_mapping_wildfire.ipynb``: the August 2026 Big Sur wildfire (``AOI_Big_Sur.geojson``, 15 Jul - 31 Oct 2026) from 25 Sentinel-2 L2A days and 36 Sentinel-1 RTC acquisitions on two orbits: per-pixel cloud and smoke screening, dNBR / RdNBR burn severity with an abrupt-drop test against grass curing, comparison of the ``sentinel2_burn_indices``, per-pixel burn dates and fire growth, S1C/S1D radiometric normalisation, local incidence angles per orbit, SAR burn signatures and multi-scale AUC, omnibus test, threshold and Random Forest burn maps with spatial block validation, change-point SAR burn dates, and terrain effects
- Tutorial ``07_remote_sensing/07_10_Capstones/07_09_12_Sentinel1_mapping_shrimp_area_Bangladesh.ipynb``: shrimp gher mapping in Satkhira, Khulna and Bagerhat from 32 Planetary Computer Sentinel-1 RTC dates (orbit 150, Jul 2025 - Jul 2026) for ``AOI.kml`` plus two cloud-free Sentinel-2 dry-season composites: mosaic-fill masking, mangrove stable-target check, multi-temporal speckle filtering, a river mask (water frequency, JRC history, drain hump), rules for year-round and shrimp-rice rotation ghers, gher drain and refill dates, Sentinel-2 cross-check, k-means, Random Forest (S1 vs S1 + S2) and TempCNN with spatial block validation, gher patch polygons and district statistics
- ``pygis.rsense.edge_detectors``: multispectral input (any channel count, e.g. blue-green-red-NIR2) for ``predict_boundary_probability``, ``detect_field_edges`` and ``detect_field_edges_from_raster``; ``build_edge_model("SegNet" | "FCN-DKConv6", in_channels=...)`` (new ``SegNet``, with dropout at the three deepest encoder and decoder stages, and ``FCNDKConv6`` in ``dl/models/fieldedge.py``, after Yang et al. 2020); training-data helpers ``rasterize_boundaries``, ``tile_pairs``, ``dihedral_augment``; classical baselines ``classical_edges`` (Roberts, Sobel, Prewitt, Scharr, LoG, Canny); evaluation ``edge_metrics`` (boundary and field precision/recall/F1, ROC AUC, positional tolerance) and ``select_threshold``. Four new tests in ``tests/rsense/test_edge_detectors.py``
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_12_field_edge_detection_worldview.ipynb``: replication of Yang, Ahmed et al. (2020) on the Patuakhali WorldView scene: 0.5 m boundary reference from digitised fields, 192 x 192 tiles with a block split before rotation augmentation, classical edge detectors, a texture random forest, and U-Net, SegNet, FC-DenseNet56/67/103 and FCN-DKConv6 on 3 (B, G, R) and 4 (B, G, R, NIR2) channels, with the paper's tables, learning curves, training times, ROC curves, tile comparisons and georeferenced field polygons
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_11_field_edge_detection_google_screenshots.ipynb``: field-boundary and field-instance detection on 60 labelled Bangladesh Google screenshots (Train/Valid/Test) at 0.25 m: classical Sobel+Sato baseline, FC-DenseNet56/67/103 trained in-notebook, Delineate Anything V2 (pre-trained and fine-tuned) and SAM 2.1 via ``pygis.rsense.edge_detectors``, with instance and boundary metrics on held-out tiles and georeferenced polygon export.
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_07_image_classification_sentinel_1.ipynb``: Corn Belt crop classification from five 2024 Sentinel-1 RTC dates (May–Oct) with CDL 10 m labels: speckle filtering (ENL), temporal signatures, spatial block split, pixel Random Forest baseline, nine CNN/ViT architectures, SAR-input ablations (filter, polarisation, leave-one-date-out, cumulative dates), augmentation/loss ablations, final ResNet-18 with TTA, wall-to-wall map and boundary-error analysis
- Tutorial ``07_remote_sensing/07_10_Capstones/07_09_11_Sentinel1_paddy_rice_mapping_Bangladesh.ipynb``: wetland paddy (Aman, Boro, Aus) mapping in the Kishoreganj–Netrokona haors from 42 Planetary Computer Sentinel-1 RTC dates (orbit 150, Jun 2025–Oct 2026) for ``AOI_1.kml``: multi-temporal speckle filtering, flood and harvest-drop rules, k-means, Random Forest and TempCNN with spatial block validation, district statistics and GeoTIFF/GeoPackage outputs (no field reference data; scores measure agreement with rule-derived labels)
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_01_image_processing.ipynb``: every ``image_processing`` algorithm with its mathematics, run on the ``data/Image_processing`` photo, aerial, 8-band and Planet images
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_03_sampling_accuracy_reproducible.ipynb``: data model, split strategies compared (random vs spatial block vs group vs spatial k-fold), Sentinel-2/Sentinel-1/texture feature stacks, the lightweight ``classify`` API, bootstrap accuracy, Olofsson area-adjusted estimates, ``error_map``, baselines, ``RunConfig``/``save_run`` and the command line
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_04_image_classification_CNN.ipynb``: GPU PyTorch CNN on Landsat 9 + CDL (Corn Belt); global block split vs tile-wise split with augmentation, 3-seed ablation and comparison, plus a transfer test of both models and the 12-model ablation on six non-overlapping tiles outside the AOI (mean ± SD, paired tests, 2 x 3 CDL and prediction maps; data in ``data/Corn_belt/outside_aoi``)
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_07_transfer_learning_vision_transformers.ipynb``: transfer learning with ResNet/ViT/Swin on Sentinel-2 + CDL chips (``set_trainable`` linear probe/partial/full, ``compare_transfer_modes``, ``load_rs_backbone`` SSL4EO/Satlas vs ImageNet, ``adapt_input_channels`` for 10 bands, ``vit_attention_rollout``, label-budget curves), mean ± SD over seeds.
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_06_semantic_segmentation.ipynb``: semantic segmentation on Sentinel-2 (10 bands) + CDL 10 m (``rgb_mask_to_index``, block-based patch sampling, ``build_segmenter`` for U-Net/FCN/DeepLabV3(+)/SegFormer with the math of each, ``get_loss`` Dice/Lovász/focal ablation, ``train_model(task="segmentation")``, ``sliding_window_predict`` overlap/blending, ``segmentation_metrics`` with boundary-band errors, ``write_class_map``), mean ± SD over seeds.
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_05_image_classification_advanced_CNN.ipynb``: deep-learning API walk-through on Sentinel-2 (10 bands, 10 m) + CDL 10 m chips (``set_seed``, ``extract_chips``, ``chips_from_folder``, ``ChipDataset``/``as_loader``, ``class_weights``, ``build_cnn`` from LeNet to ViT/Swin, ``receptive_field``, ``OverheadAugment``, ``cutmix``, ``get_loss``, ``train_model`` with AMP/early stopping/checkpoint/TensorBoard, ``load_checkpoint``, ``predict_proba_batches``, ``predict_chip_map``, ``classification_report``), spatial-block split, architecture and ablation tables (mean ± SD over seeds) and pixel-level map check.
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_06_semantic_segmentation.ipynb`` now refits the validation-selected U-Net + CE/Dice recipe on AOI-contained patches only, evaluates held-out AOI blocks, predicts outside-AOI pixels with overlap/Hann blending, visualizes the exterior, and exports an outside-only georeferenced GeoTIFF.
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_07_instance_segmentation.ipynb``: PyGIS Mask R-CNN instance segmentation of eight-band WorldView-2 imagery using field polygon labels, spatial training/validation/test bands, mask AP/IoU evaluation, and CRS-aware GeoPackage outputs.
- ``image_processing.read_image(window=(row0, row1, col0, col1))`` reads a pixel window of a large raster
- ``pygis.rsense.image_processing`` (new, standalone): OpenCV / scikit-image style image processing for remote-sensing rasters on ``(band, y, x)`` DataArrays or NumPy arrays, keeping CRS and rebuilding coordinates after grid changes, NaN-aware. I/O (`read_image`, `write_image`, `image_info`); color (`to_gray`, `rgb_to_hsv`, `hsv_to_rgb`, `rgb_to_lab`, `to_uint8`); geometry (`resize`, `rescale`, `downscale_local_mean`, `rotate`, `shift`, `flip`, `crop`, `affine_transform`, `perspective_transform`); radiometry (`stretch`, `adjust_gamma`, `adjust_brightness_contrast`, `histogram`, `equalize_hist`, `equalize_adapthist`, `match_histograms`); `threshold` (Otsu, Li, triangle, Yen, isodata, mean, local); filters (`convolve`, `mean_blur`, `gaussian_blur`, `median_blur`, `bilateral_denoise`, `tv_denoise`, `sharpen`, `unsharp_mask`, `add_noise`); `edges` (Sobel, Roberts, Prewitt, Scharr, Laplacian, Canny), `sobel_h`, `sobel_v`; `morphology`, `remove_small_objects`; `harris_corners`, `hog_features`, `to_vector`. Needs only ``pygis[rsense]`` (no OpenCV). Tests: ``tests/rsense/test_image_processing.py`` (20 tests)
- Tutorial ``07_remote_sensing/07_08_image_classification/07_08_00_image_classification_introduction.ipynb``: image classification tasks (scene classification, semantic segmentation, object detection, instance segmentation, change detection, time series), pixel- vs object-based, unsupervised / supervised / semi-supervised / self-supervised learning, computer vision and foundation models; a hands-on comparison of k-means, pixel RF, OBIA and self-training on Sentinel-2 with a spatial block split; and an inventory of PyGIS's classification functions mapped to the module's chapters

### Fixed
- ``image_classification.RS_WEIGHTS``: Satlas entries now use the TorchGeo 0.8 enum names (``satlas_s2_swinb_rgb/ms``, new ``satlas_s2_swint_rgb/ms``); ``set_trainable(mode="partial")`` now unfreezes the last stage plus final norm for torchvision models with ``features`` (Swin, ConvNeXt, EfficientNet) instead of training only the head.
- ``classify.image_classification.rasterize_labels`` failed on label columns with a pandas extension dtype (the default ``str`` dtype for text columns in pandas 3); the dtype check now uses ``pd.api.types.is_numeric_dtype``. Regression test added in ``tests/rsense/test_image_classification.py``

### Changed
- Version bumped to 0.3.36
- README documents ``pygis.pygisdb``: feature table, extras, package layout, a spatial-database quick start and the documentation table

## [0.3.35] - 2026-10-30

### Added
- ``pygis.rsense.edge_detectors`` for FieldEdge-compatible FC-DenseNet and PyGIS semantic-segmentation models, large-image boundary inference, field polygon/boundary outputs, and optional Delineate Anything/SAM 2 inference.
- ``pygis[rsense-edge]`` optional dependencies for deep-learning and geospatial field-edge workflows; included in ``pygis[rsense-all]``.

## [0.3.34] - 2026-10-30

### Added
- ``pygis.rsense.hyperspectral`` (new subpackage) for the "Hyperspectral Image Processing in Python" tutorial. Cubes are ``(band, y, x)`` DataArrays with ``wavelength``/``fwhm``/``good`` band coordinates.
- ``cube``: `as_cube` (µm→nm, Dataset or array input), `wavelengths`, `band_index`, `band_at` (nearest band or averaged window, with tolerance), `sel_range`, `to_matrix`/`from_matrix`, `spectra_at`, `mean_spectrum`, `band_stats` (per-band SNR), `apply_spectral` (chunked per-pixel functions)
- ``io``: `read_envi` / `write_envi` / `read_envi_header` (BSQ/BIL/BIP, big/little endian, memory-mapped band and window subsets, scale factor, data ignore value, `bbl`, map info and coordinate system string), `read_aviris`, `read_emit` (L1B/L2A NetCDF, GLT orthorectification, good-band flags, lon/lat crop), `glt_orthorectify`, `read_enmap` (SPECTRAL_IMAGE + METADATA.XML gains/offsets), `read_prisma` (L2 HE5, VNIR+SWIR merged and sorted, scale factors, L2D UTM grid), `read_geotiff_cube` (wavelengths from band descriptions or tags), `read_hyperspectral` (auto-detect), `search` (EMIT, AVIRIS-NG and PACE OCI via NASA Earthdata), `subset_file`
- ``preprocess``: `bad_bands`, `drop_bad_bands`, `estimate_noise` (shift difference or spectral residual), `destripe` (moment matching, median), `quality_mask`, `empirical_line` (gains, offsets and R² per band), `flat_field`, `iar_reflectance`, `smooth`, `derivative`, `snv`, `normalize`, `continuum_removal` (vectorised quickhull, checked against SciPy ConvexHull), `band_depth`, `resample_spectral`, `resample_to_sensor` (Sentinel-2, Landsat 8/9, MODIS, PlanetScope with the optical band names), `resample_library`
- ``reduce``: `pca`, `mnf`, `mnf_denoise` (fitted `Transform` with inverse), `ica`, `hysime`, `virtual_dimensionality`, `select_bands`
- ``indices``: 32 wavelength-defined narrowband indices with references, `index`, `list_indices`, `available`, `register_index`, `red_edge_position` (maximum first derivative or 4-point linear)
- ``library``: `SpectralLibrary`, `read_usgs` (splib07), `read_ecostress`, `read_csv`, `from_image`, `from_dict`, `merge_libraries`, `as_library`, `synthetic_library` (teaching/test spectra)
- ``match``: `sam`, `sid`, `spectral_similarity` (SAM, SID, SCA, Euclidean, SID-SAM), `classify_spectral` (with rejection threshold)
- ``endmembers``: `ppi`, `atgp`, `nfindr` (vectorised Cramer-rule sweeps), `vca`, `extract_endmembers`
- ``unmixing``: `unmix` (LS, SCLS, NNLS, FCLS), `solve_abundances` (exact vectorised NNLS/FCLS for up to 8 endmembers, matching SciPy NNLS; FISTA beyond), `project_simplex`, `mesma` (multiple endmembers per class, photometric shade, RMSE and fraction constraints), `sparse_unmix`, `reconstruct`
- ``detect``: `rx` (global; local dual-window on principal components), `matched_filter`, `ace`, `cem`, `osp`, `detection_roc` (AUC), `gas_matched_filter` (column-wise CH₄/CO₂ matched filter, ppm·m), `plume_mask`, `plume_ime`, `emission_rate`
- ``traits``: `TraitModel` (`predict_map`, optional GPR uncertainty), `fit_plsr` (CV component selection, one-SE rule, VIP, grouped CV), `fit_regressor` (RF, gradient boosting, SVR, ridge, GPR), `vip_scores`, `prosail_spectrum`, `prosail_lut` (Latin hypercube, convolved to sensor bands, noise), `hybrid_retrieval`
- ``viz``: `rgb` / `plot_rgb` (wavelength presets), `plot_spectra`, `plot_band_stats`, `spectral_viewer`, `wavelength_slider`
- ``pygis.rsense.dl.hyperspectral`` (new): `HSIPatchDataset`, `block_split`, `SpectralCNN1D`, `CNN3D`, `HybridSN`, `SpectralTransformer`, `train_hsi_classifier`, `evaluate`, `predict_hsi_map`; exported from ``pygis.rsense.dl``
- New extra ``pygis[rsense-hyperspectral]`` (h5py, scikit-learn, matplotlib, plotly, prosail), included in ``rsense-all``
- Tests: ``tests/rsense/test_hyperspectral.py`` (31 tests) and ``tests/rsense/test_hyperspectral_dl.py`` (7 tests) on synthetic cubes with known endmembers, abundances, targets and plumes, and synthetic ENVI, EMIT, EnMAP, PRISMA and GeoTIFF products

### Changed
- ``pygis.rsense`` lists ``hyperspectral`` as a lazily loaded submodule
- README: Hyperspectral section, code example, install extra, package tree and documentation table; build commands name 0.3.34
- Version bumped to 0.3.34


## [0.3.33] - 2026-10-30

### Added
Lidar coverage for the "Lidar Data Processing in Python" tutorial (Parts I–IX):
- ``pygis.rsense.lidar.filters`` (new): `statistical_outlier_removal` and `radius_outlier_removal` (drop, classify as ASPRS 7 low / 18 high noise against the surrounding clean surface, or add an `outlier` mask), `remove_outliers`, `pmf` (Progressive Morphological Filter, Zhang et al. 2003; also `classify_ground(pc, "pmf")`), `thin` (voxel, grid-min, grid-max, random), `reproject` (pyproj, including compound vertical CRSs), `clip_polygon`, `summary`, `density_map`, `vertical_accuracy` (RMSEz, mean error, NVA 95%), `swath_overlap` (inter-swath dz by `point_source_id`)
- ``pygis.rsense.lidar.tiles`` (new): `Tile`, `tile_grid` and `iter_tiles` with buffers, `trim`, `process_tiles` (thread/process pools; PointCloud, raster and GeoDataFrame results trimmed to tile cores and merged), `merge_pointclouds`, `merge_rasters`, `write_tiles`, `process_files`
- ``pygis.rsense.lidar.features`` (new): `eigen_features` (linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, change of curvature, verticality, normals; chunked), `normals`, `euclidean_clusters`, `height_profile` (gap fraction, MacArthur–Horn LAD), `foliage_height_diversity`, `structure_metrics` (gridded FHD, effective LAI, canopy relief ratio), `plot_metrics` (per field plot, polygons or points + radius), `point_feature_table` (multi-scale), `train_point_classifier` / `predict_point_classes`
- ``pygis.rsense.lidar.urban`` (new): `classify_vegetation` (ASPRS 3/4/5 by height), `classify_buildings` (local flatness, RANSAC plane test per cluster, roof-edge growing), `building_footprints` (area, height, z max), `ransac_plane`, `segment_roof_planes` (slope, aspect, normal per plane), `detect_power_lines` (ASPRS 14 and line geometries), `sun_positions`, `solar_potential` (clear-sky relative insolation with cast shadows, per-roof summary)
- ``pygis.rsense.lidar.change`` (new): `dem_of_difference` (level of detection, erosion/deposition/net volumes and areas), `c2c_distance` (optionally signed along normals), `m3c2` (Lague et al. 2013; distance, LoD95, significance), `canopy_change`
- ``pygis.rsense.lidar.registration`` (new): `icp` (point-to-point, trimmed, distance-limited; transform in world coordinates), `rigid_transform`, `apply_transform`, `fit_circle` (Kåsa + Gauss–Newton), `detect_stems` (stem positions and DBH from a breast-height slice)
- ``pygis.rsense.lidar.viz`` (new): `cross_section`, `plot_cross_section` (ASPRS colours), `plot_height_profile`, `plot_classification`
- ``pygis.rsense.dl.pointnet`` (new): `PointBlockDataset` (block sampling, label remapping, ignore classes, augmentation), `PointNetSeg`, `PointNet2Seg` (PointNet++ SSG in pure PyTorch), `train_point_segmentation` (best weights by validation mIoU), `predict_point_segmentation` (overlapping-block voting over every point); exported from ``pygis.rsense.dl``
- ``lidar.io``: `read_las_chunks`, `las_info`, `write_ply`, `pdal_pipeline` (run PDAL filters on an in-memory PointCloud); `read_pdal` gains `bounds_crs` (e.g. lon/lat boxes for USGS 3DEP EPT in EPSG:3857) and `out_crs`, and maps more PDAL dimensions (scan angle, point source id, user data, HeightAboveGround)
- ``lidar.spaceborne``: `read_gedi_l2b` (cover, PAI, FHD, RH100, optional PAVD/cover profiles, quality filtering)
- ``lidar.surfaces``: `sample_raster` (nearest or bilinear) and `colorize` (RGB/NIR from imagery into LAS 16-bit fields)
- ``lidar.terrain``: `hand` (Height Above Nearest Drainage, vectorized pointer jumping), `inundation` (HAND stage or DEM water level, seed-connected), `flatten_water` (hydro-flattening to a shoreline percentile); `flow_accumulation` gains `weights=`
- ``lidar.forest``: `segment_trees_points` (point-based tree segmentation from treetops)
- Tests: ``tests/rsense/test_lidar_extra.py`` (30 tests) and ``tests/rsense/test_lidar_dl.py`` (4 tests)

### Changed
- `lidar.flow_accumulation` solves the D8 accumulation as one sparse linear system instead of a Python loop (about 1 s for a 1000 × 1000 DEM)
- `lidar.grid_metrics` is vectorized with pandas group-bys (same outputs; about 1 s for 3 million points at 10 m)
- ``pyproject.toml``: `rsense-lidar` extra adds scikit-learn and matplotlib; new keywords; version 0.3.33
- README: Lidar section lists the full API, new Lidar code example, package tree and documentation table updated, build commands name 0.3.33

### Fixed
- `lidar.grid_metrics` could assign points on the maximum x edge of the extent to the first cell of the next row; cells are now clipped to the grid

## [0.3.32] - 2026-10-29

### Fixed
- ``pygis.rsense.sar.radarsat`` on real RCM ScanSAR GRD products (checked against the two RCM-1/RCM-3 SCLN scenes of the sea ice capstone). Products with ``pixelTimeOrdering`` ``Decreasing`` store their gain, noise and incidence curves from the last column backwards (negative ``stepSize``); these were interpolated with descending sample positions, which ``np.interp`` does not support. All three curves are now sorted to ascending column order first (new helper `_range_curve`), in `interpolate_rcm_lut`, `interpolate_rcm_noise` and `rcm_incidence_angle`
- `read_rcm_incidence` read only the first value of real ``incidenceAngles.xml`` files, which write one ``<angles>`` element per value; it now collects every element (a single space-separated element still works)
- `read_rcm_noise_levels` matched the calibration type literally, so ``kind="gamma"`` found nothing in products that write ``"Gamma"`` rather than ``"Gamma Nought"``; matching now ignores case and a trailing "Nought". Products without per-beam segments fall back to the whole-swath ``referenceNoiseLevel`` (returned as one segment, ``beam="all"``)
- `rcm_incidence_angle` docstring: the linear near-to-far fallback is off by up to about 2 degrees on a ScanSAR swath (1.3 to 1.8 degrees on the real products), not a few tenths
- ``examples/rcm_sea_ice.py``: one HV threshold for both dates failed on the real pair (HV is at the noise floor at 50 to 54 degrees incidence, so the second date mapped 7% ice). The script now geocodes the HV noise floor and uses HV only where it is at least 3 dB above it, thresholds normalized HH per date, accepts a fitted HH incidence slope only if plausible (-0.40 to -0.05 dB/degree, R² at least 0.2) and otherwise uses -0.22, no longer normalizes HV, and refines the Natural Earth land mask with bright HV land near the coast
- ``docs/sea_ice_mapping_rcm_aoi.md``: Steps 7 and 10 updated with the same lessons (HV noise check, no HV normalization, per-date thresholds); RCM SCLN GRD pixel spacing is 40 m, not 50 m

### Added
- ``Tutorials/notebooks/07_remote_sensing/07_10_Capstones/07_09_09_sea_ice_mapping_RADARSAT.ipynb``: sea ice capstone on two real RCM ScanSAR Low Noise GRD scenes over the Lower St. Lawrence Estuary (28 January and 3 February 2026), executed with outputs: footprints, AOI windows with calibration and noise removal, measured ENL and refined Lee filtering, common UTM grid, SAR-refined land mask, HV noise-floor check, incidence normalization, texture features, per-date ice/water mapping, unsupervised ice regimes and landfast candidates, ice concentration (1 km, 5 km), change, ice-cost least-cost routes, an interactive folium map (saved as HTML in the notebook, so it displays without a live widget kernel), and GeoTIFF/GeoJSON/CSV outputs
- Test ``test_rcm_real_product_layouts`` in ``tests/rsense/test_sar.py``: negative ``stepSize`` curves, one ``<angles>`` element per value, overlapping per-beam noise segments, the ``"Gamma"`` alias and the ``referenceNoiseLevel`` fallback

### Changed
- Version bumped to 0.3.32; README build commands, RADARSAT section and documentation table point to the new capstone notebook

## [0.3.31] - 2026-10-29

### Added
- ``pygis.rsense.sar.radarsat``: a Level-1 GRD product reader for RCM, so a sea-ice or ship workflow runs from the unzipped EODMS folder without SNAP. `read_rcm_grd_subset` reads only the window covering a lon/lat box for one polarisation, calibrates it with the product's own `lutSigma`/`lutBeta`/`lutGamma`, subtracts the matching NESZ from `noiseLevels_<pol>.xml`, and returns window-local GCPs and the per-pixel incidence angle in the same `{"data", "gcps", "gcp_crs"}` shape as `sentinel1.read_grd_subset`, so `sentinel1.create_stack` puts RCM dates from different beams on one map grid and `ship.land_mask_for_window` masks land in radar geometry. Supporting readers: `find_rcm_files` (imagery and per-polarisation calibration files, with the older unsuffixed names as a fallback), `read_rcm_metadata` (satellite, beam mode, product type, acquisition time, pass, size, spacing, near/far incidence, pixel/line time ordering; missing tags return `None`), `read_rcm_tie_points` (`product.xml` `imageTiePoint` grid as rasterio GCPs, used when the GeoTIFF has none), `read_rcm_incidence` and `rcm_incidence_angle` (`incidenceAngles.xml`, or a linear near-to-far fallback from the metadata that respects `pixelTimeOrdering`)
- ``pygis.rsense.sar.radarsat``: incidence-angle normalisation, `fit_incidence_slope` (least-squares dB-per-degree slope, intercept, R² and pixel count over an optional mask) and `normalize_incidence` (`sigma0 - slope * (theta - ref)`, keeps xarray coordinates and records the reference angle and slope in `attrs`)
- All eight functions are also exported flat on `pygis.rsense.sar`
- Tests in ``tests/rsense/test_sar.py`` (6 new tests, 7 cases) on a synthetic RCM GRD product folder: file discovery, metadata, tie points, incidence curve and fallback, windowed LUT/noise interpolation, calibrated subset with and without GeoTIFF GCPs, two-date `create_stack`, and slope fitting/normalisation
- ``examples/rcm_sea_ice.py``: runnable sea ice mapping script for two RCM ScanSAR GRD scenes (AOI windows, calibration and noise removal, speckle filtering, common UTM grid, incidence normalization, HV ice/water threshold, ice concentration in tenths, HH change between dates, ice-cost least-cost route), writing GeoTIFFs
- ``docs/sea_ice_mapping_rcm_aoi.md``: step-by-step case study (Lower St. Lawrence Estuary, Baie-Comeau to Matane) built on the new RCM functions: ice type, concentration, change, ice-cost surface and least-cost shipping route

### Changed
- README: RADARSAT section documents the GRD product reader and incidence normalization, a new RCM example in the remote-sensing code samples, a documentation-table row for the sea ice case study, the package tree entry, and the build/install commands now name 0.3.31
- ``pyproject.toml`` keywords add ``radarsat`` and ``sea-ice``
- `interpolate_rcm_lut` and `interpolate_rcm_noise` take a keyword `col0` (scene column of the first output column) so a window cut from the image gets the right part of the range curve; the default `col0=0` keeps the old behaviour. Gaps between ScanSAR noise segments are now filled from the segment end points, which also works when the window lies entirely inside a seam
- Version bumped to 0.3.31

## [0.3.30] - 2026-10-29

### Changed
- ``image_classification`` moved into the ``classify`` subpackage: ``src/rsense/image_classification.py`` → ``src/rsense/classify/image_classification.py``, imported as ``pygis.rsense.classify.image_classification`` (or ``from pygis.rsense.classify import image_classification``) and run as ``python -m pygis.rsense.classify.image_classification``. Its relative imports now point at ``classify.estimators`` / ``classify.obia`` / ``classify.change`` and ``rsense.dl`` / ``rsense.fusion`` / ``rsense.super_resolution`` directly
- The old path still works: ``src/rsense/image_classification.py`` is a small alias of the new module (``import pygis.rsense.image_classification`` and ``python -m pygis.rsense.image_classification`` both resolve to it), and ``pygis.rsense`` exposes it through an ``_ALIASES`` table so ``rs.image_classification`` keeps working without breaking the lazy-submodule naming contract checked by ``tests/rsense/test_viz_scale_apps.py``
- ``pygis.rsense.classify`` exports ``image_classification`` in ``__all__``
- Version bumped to 0.3.30; README (package tree, feature section, quick start, command line, tutorial index) and ``pyproject.toml`` use the new module path; example scripts import from ``pygis.rsense.classify``

### Added
- ``tests/rsense/test_classify_functions.py`` (23 tests): direct tests for every public function in ``pygis.rsense.classify`` not already called by ``test_classify.py`` / ``test_image_classification.py`` -- ``Classifier.predict_proba`` and ``predict_raster`` for six models plus regression, k-means and string labels; ``as_band_array``, ``labels_from_table``, ``spatial_blocks``, ``Normalizer.fit_transform``, ``spatial_cross_validate``, ``set_seed``, ``extract_chips``, ``rgb_mask_to_index``, ``read_class_dict``, ``chips_from_folder``, ``pixel_time_series``, ``uncertainty_scores``, ``parse_args``, ``run_workflow``, ``cutmix``, ``fixmatch_loss``, ``adapt_input_channels``, ``count_parameters``, ``detect_objects``, ``instances_from_detector``; the module alias and both ``python -m`` entry points; and ``load_rs_backbone``, ``load_foundation_encoder``, ``sam_segment``, ``text_prompt_segment``, ``load_rs_clip`` and ``zero_shot_classify`` against stand-in TorchGeo / segment-geospatial / open_clip / Hugging Face modules (no downloads). All 146 public functions, classes and methods in ``classify`` are now exercised by tests

## [0.3.29] - 2026-10-29

### Added
- ``pygis.rsense.image_classification``: one module for remote-sensing image classification and related vision tasks, implemented from the 14-part image-classification blueprint. **Data**: `load_stack` (multi-band file, folder of band files with natural band sorting, or file list; nodata to NaN; CRS, transform, original nodata/dtype, band names, dates and sensor kept in `RasterMeta`; `crs=` only fills a missing CRS; grid mismatches raise unless `align=True`, which is recorded), `labels_from_table`, `rasterize_labels`, `sample_pixels`; leakage-safe `spatial_block_split` / `spatial_kfold` / `group_split` / `temporal_split` / `geographic_holdout` returning a checked `Split`; `Normalizer` (z-score, min-max, percentile) fitted on training data only and serializable. **Paradigm and spatial unit** (independent, see `RunConfig`): `unsupervised_classify` (k-means, mini-batch k-means, GMM), `train_pixel_classifier` with spatial CV (RF, SVM, XGBoost, k-NN, logistic, GBM, MLP through `rsense.classify.Classifier`), `predict_map`, `object_based_classify`. **PyTorch foundations**: `ChipDataset`, `PairChipDataset`, `extract_chips`, `chips_from_folder` + `read_class_dict` / `rgb_mask_to_index` (DeepGlobe-style masks), `build_cnn` (built-in LeNet / VGG-style / simple CNN, torchvision ResNet / EfficientNet / ConvNeXt / VGG / ViT / Swin, any timm model), `adapt_input_channels` (mean / repeat / random stem for N bands), `receptive_field`, `OverheadAugment` (dihedral, radiometric jitter, crop), `cutmix`, `class_weights`, `get_loss` (CE, weighted CE, focal, Dice, Lovasz-Softmax, CE+Dice, GCE, SCE), `train_model` (AdamW/Adam/SGD, cosine/one-cycle/step/plateau, AMP, gradient clipping, CutMix, early stopping, checkpoints, seeds, TensorBoard / W&B / MLflow / callable loggers), `predict_proba_batches` with dihedral TTA, `predict_chip_map`. **Transfer learning**: `set_trainable` (linear probe, partial, full), `load_rs_backbone` (TorchGeo SSL4EO-S12, SeCo, Satlas weights), `compare_transfer_modes`, `vit_attention_rollout`. **Segmentation**: `build_segmenter` (torchvision FCN / DeepLabV3 / LR-ASPP; smp U-Net, U-Net++, DeepLabV3+, FPN, SegFormer, UPerNet; built-in U-Net), `sliding_window_predict` (blended windows), `write_class_map` (GeoTIFF with nodata, class-name tags and color table), `segmentation_metrics`. **Detection**: `build_object_detector` (Faster R-CNN, RetinaNet, FCOS, Mask R-CNN, YOLO / YOLO-OBB), `box_iou`, `nms`, `mean_average_precision` (VOC / COCO thresholds), `obb_to_polygon` / `polygon_iou`, `detection_tiles` (scale-aware), `merge_detections`, `detect_objects`, `count_objects`. **Instances and boundaries**: `clean_mask`, `instances_from_semantic` (connected / watershed), `boundary_weight_map` (U-Net weights), `mask_boundaries`, `polygonize` (min area, simplification, rectangle/convex/auto regularization, class, confidence, area, perimeter, compactness), `instances_from_detector`, `export_vectors` (GeoPackage, GeoJSON, Shapefile, GeoParquet; CRS kept). **Change detection**: `image_difference`, `image_ratio`, `change_vector_analysis` (Otsu robust to constant magnitudes), `pca_change`, `check_coregistration` (phase correlation), `build_change_model` (FC-EF, FC-Siam-diff, FC-Siam-conc U-Nets and a BIT-style transformer), `change_metrics`. **Spatio-temporal and multi-sensor**: `pixel_time_series`, `build_temporal_model` (TempCNN, LSTM, L-TAE, PSE-TAE, ConvLSTM, U-TAE-style), `build_fusion_model` (middle fusion), `fuse_early`, `fuse_late`, `sar_assisted_gap_fill` (adds a `reconstructed` flag), `pansharpen`, `pansharpening_quality` (RMSE, ERGAS, SAM, CC). **Self-supervised**: `nt_xent_loss`, `pretrain_contrastive` (SimCLR, MoCo with momentum encoder and queue, SeCo-style explicit pairs), `build_mae` / `pretrain_masked`, `make_encoder`, `extract_embeddings`, `linear_probe`, `check_geographic_overlap`. **Foundation models**: `FOUNDATION_MODELS` / `foundation_model_info` (Prithvi-EO-2.0, Clay, SatlasPretrain, DOFA, SkySense, SSL4EO-S12, SatMAE, Scale-MAE; licenses flagged to check), `load_foundation_encoder`, `terratorch_config`, `embedding_search`, `few_shot_prototypes`. **Promptable / vision-language**: `sam_segment` (SAM, SAM 2), `text_prompt_segment` (Grounding DINO + SAM), `load_rs_clip` / `zero_shot_classify` (RemoteCLIP, GeoRSCLIP); outputs carry `verified=False`. **Few or imperfect labels**: `uncertainty_scores`, `select_for_labeling` (uncertainty + k-means diversity), `active_learning_loop`, `pseudo_label`, `self_training`, `fixmatch_loss` / `train_fixmatch`, `weak_labels_from_product`, `coral`, `histogram_match`, `tent_adapt`, `check_split_isolation`. **Evaluation and reproducibility**: `classification_report` (per-class precision / recall / F1 / support, kappa, macro F1, bootstrap intervals), `error_map`, `compare_with_baselines`, `save_run` / `load_run` (configuration, package versions, checkpoint, normalization, class names, CRS / transform, split ids, metrics). **Command line**: `python -m pygis.rsense.image_classification` with `--task`, `--learning-mode`, `--spatial-unit`, `--bands`, `--seed` plus data options; runs pixel / object, supervised / unsupervised / semi-supervised classification and classical change detection end to end
- New optional extra ``pygis[rsense-vision]`` (``rsense-dl`` + ``rsense-classify`` + timm, ultralytics, open_clip_torch, huggingface_hub, segment-geospatial); kept out of ``rsense-all`` because of its size
- Tests in ``tests/rsense/test_image_classification.py`` (35 tests; the 14 deep-learning tests skip without PyTorch). Checked against the tutorial data in ``Tutorials/notebooks/07_remote_sensing/07_08_image_classification.ipynb/data`` (Sentinel-2 UB North Campus, WorldView-2 field boundaries, Landsat Bangladesh, Sentinel-1 VV/VH, DeepGlobe land cover)

### Changed
- Version bumped to 0.3.29
- ``pygis.rsense`` lazily exposes ``image_classification``; README documents the module (feature table, package tree, quick start, command line, extras, tutorial index)

## [0.3.28] - 2026-10-28

### Added
- ``pygis.spatial_analysis.segregation``: a lighter, faster, time-series-aware alternative to PySAL's `segregation` package. ``indices`` (dissimilarity, Gini, isolation, exposure, correlation ratio, Theil's H, Simpson diversity, multigroup dissimilarity -- cross-validated exactly against PySAL's own `segregation` package); ``local`` (location quotient, entropy, Simpson diversity, exposure deviation, dissimilarity contribution, with Benjamini-Hochberg FDR-corrected permutation significance); ``multiscale`` (a KD-tree-based, chunked spatial segregation profile -- segregation scale, area-under-curve, distortion coefficient -- that never materializes the dense n x n distance matrix PySAL's own profile does); ``neighborhoods`` (kernel, KNN via libpysal, and network via OSMnx + `scipy.sparse.csgraph.dijkstra` instead of a pandana table); ``null``/``inference`` (evenness and systematic null models, single/two-value permutation tests, a spatial-block bootstrap that resamples whole spatial blocks instead of individual units, and a Mann-Kendall panel trend test); ``decomposition`` (an exactly-additive two-term Shapley spatial-vs-composition decomposition, chainable across any number of periods); ``data`` (areal and raster-weighted/dasymetric boundary harmonization across census years); ``temporal`` (panel workflows tying harmonization, per-period indices, and trend testing together); ``report.compute()`` (one consistent entry point with optional bootstrap confidence intervals) and ``report.export_for_causal`` (a hook into ``pygis.spatialcausal`` / PyCausalML); ``viz`` (static choropleth, profile, decomposition, and trend plots). New optional extras ``pygis[segregation]`` and ``pygis[segregation-dasymetric]``
- Synthetic tutorial dataset in ``data/data_segregation/`` (``make_data.py``): a fictional city with a built-in east/west segregation pattern that partially integrates over three census years, plus a differently-bounded 1990 layer and a built-up-intensity raster for the boundary-harmonization examples
- Tutorial ``03_spatial_analysis/03_07_00_introduction_segregation_analysis.ipynb``: a hands-on tour of the whole module, executed end-to-end
- Tests in ``tests/test_segregation.py`` (25 tests), including direct cross-validation against PySAL's `segregation` package

### Changed
- Version bumped to 0.3.28
- README documents the new `pygis.spatial_analysis.segregation` module (feature table and tutorial index)

## [0.3.27] - 2026-10-28

### Added
- ``pygis.rsense.texture``: texture analysis for optical and SAR imagery, covering six method families -- statistical (`first_order_stats` local moments and entropy, `glcm_texture` Haralick 1973 gray-level co-occurrence features, `glrlm_texture` run-length emphasis), structural (`edge_density` Canny edge density, `lacunarity` gliding-box gap/clump distribution), transform-based (`gabor_texture` oriented/frequency filter bank, `fourier_texture` radial power-spectrum rings, `wavelet_texture` a pure-NumPy 2-D Haar DWT needing no extra dependency), local pattern-based (`lbp_texture` Local Binary Pattern histogram uniformity/entropy), model-based (`mrf_texture_energy` Markov Random Field clique-energy proxy, `fractal_dimension` differential box-counting (Sarkar & Chaudhuri 1994), `autoregressive_texture` a closed-form local AR(1) fit), and deep-learning-based (`deep_texture_features`: pretrained CNN backbone feature maps via `timm`, `pygis[rsense-dl]`). Every function accepts a 2-D array/DataArray or one with leading dims (band/polarization/time, e.g. a VV/VH SAR stack), broadcasts the algorithm over those leading dims, and returns a georeferenced raster the same shape as its input, preserving CRS and coordinates
- Tests in ``tests/rsense/test_texture.py``

### Changed
- Version bumped to 0.3.27
- README remote-sensing feature line and ``src/rsense/`` tree listing document the `texture` module
- `src/__init__.py`'s `__version__` corrected to match `pyproject.toml` (it had drifted one release behind since 0.3.26)

## [0.3.26] - 2026-10-27

### Added
- ``pygis.rsense.sar.ship``: CFAR ship detection on a Sentinel-1 GRD AOI in native radar geometry (`land_mask_for_window`, `estimate_enl`, `cfar`/`cfar_tiled`, `extract_targets` with PCA length/width/orientation and VH/VV contrast, `flag_azimuth_ambiguities`, `detect_ships_grd`); SLC sub-look coherence to separate coherent ships from incoherent sea clutter (`range_sublooks`, `azimuth_sublooks`, `sublook_coherence`, `coherence_at_points`); AIS validation (`read_ais_dma`, `ais_at_time`, `azimuth_displacement_m`, `match_points`, `detection_scores`). Reachable both as the `ship` submodule and flattened onto `pygis.rsense.sar` (`cfar` aliased to `ship_cfar` there, since `pygis.rsense.sar.change.cfar` already uses that name)
- Tests in ``tests/rsense/test_ship.py``

### Changed
- Version bumped to 0.3.26
- README remote-sensing feature line documents ship detection

## [0.3.25] - 2026-10-27

### Added
- ``pygis.rsense.sar.nisar``: HDF5 inspection and reading for RSLC, RIFG, RUNW, ROFF, GSLC, GCOV, GUNW, and GOFF; Level-1/Level-2 layer readers; RSLC backscatter, RIFG/GSLC interferograms, unwrapped-phase displacement, GCOV indices, offset quality processing, and quick-look Level-1 geocoding. Precision geocoding, phase unwrapping, and time-series inversion remain delegated to ISCE3, SNAPHU, and MintPy.
- Optional dependency ``h5py`` in ``pygis[rsense-sar]`` and README usage examples for NISAR products

### Changed
- Version bumped to 0.3.25

## [Unreleased]

### Changed
- Notebook builders (the ``_builders/`` folders of chapters 08_distributed_spatial, 09_05, 09_11, 09_12 and 09_13), the maintenance scripts in ``Tutorials/_tools/``, the banner and quick reference generators in ``Images/Banner/_src/`` and two unused notebook scripts (``variogram_kriging_jax.py``, ``download_lidar.py``) moved out of the repository to a local, git-ignored ``helper_functions/`` folder that keeps their original paths
- GeoLattice home page (``Tutorials/home``, built by ``build_home.py`` into ``docs/``) with chapter tiles, Getting Started, User Guide, API Reference and a Data page that links the tutorial data on Dropbox

### Added
- ``pygis.rsense.sar.flood_map(..., method="ratio")``: the UN-SPIDER Sentinel-1 change test, ``after / before`` on dB backscatter with a default threshold of 1.25. Because dB values are negative, a drop in backscatter (flooding) makes the ratio larger than 1. The Sylhet notebook follows that recommended practice (50 m smoothing, JRC seasonality, 5% slope, 8-pixel connectivity, area in hectares, exposed people, cropland and urban area).
- ``pygis.rsense.super_resolution``: classical interpolation, unsharp masking, iterative back-projection and pan-sharpening; SRCNN/VDSR/EDSR/RCAN-style multispectral models; SRGAN and conditional DDPM training/sampling; spectral-angle and coarse-observation consistency losses; tiled inference.

### Changed
- Version bumped to 0.3.24; README documents satellite super-resolution methods, optional PyTorch setup, and limitations of cross-sensor and learned-prior products.

## [0.3.22] - 2026-10-26

### Added
- ``pygis.rsense.sar.hand``: Height Above Nearest Drainage (Nobre et al. 2011) from a DEM. Depressions are filled, a D8 channel network is taken from flow accumulation, and each cell is measured above the channel it drains to
- ``pygis.rsense.sar.mask_to_polygons``: vectorize a boolean flood or water mask to polygons with pixel count and area (projected CRS, or an estimated UTM zone when the CRS is geographic)
- ``pygis.rsense.sar.flood_map`` accepts ``slope`` / ``slope_max`` and ``shadow`` (nonzero layover or shadow, including ``layover_shadow_mask``) in addition to HAND, permanent water, and the minimum mapping unit
- Tutorial ``07_remote_sensing/07_09_Capstones/07_09_05_flood_mapping_sentinel1.ipynb``: the Sylhet monsoon flood workflow runs on PyGIS (Lee filter, split-Otsu, fixed threshold, change detection, slope, layover, HAND, polygons, and accuracy). The worked example is a local Sentinel-1-like stack; ``USE_LIVE`` queries ASF
- Tests in ``tests/rsense/test_sar.py`` for HAND, slope and shadow masks, and polygon area

### Changed
- Version bumped to 0.3.22
- README remote-sensing feature line and the Sentinel-1 example mention ``hand``, ``flood_map`` masks, and ``mask_to_polygons``

## [0.3.20] - 2026-10-25

### Added
- ``pygis.rsense.optical.sentinel2_biophysical``: Sentinel-2 canopy biophysical variables (LAI, FAPAR, FCOVER, CCC, CWC) through the Copernicus Data Space BioPAR openEO service, the SNAP Sentinel-2 Toolbox Level-2 neural-network retrieval (Weiss and Baret 2016). ``estimate`` submits one batch job per variable and downloads GeoTIFFs; ``read`` stacks them. ``CWY`` and ``CW`` are accepted as CWC. Optional extra ``pygis[rsense-biopar]`` (openEO; not part of ``rsense-all``, because current ``openeo`` releases require pandas below 3). ``tests/rsense/test_sentinel2_biophysical.py``

### Changed
- Version bumped to 0.3.20

## [0.3.18] - 2026-10-25

### Added
- ``pygis.rsense.sar.radarsat.simulate_compact_pol``: simulates a right-circular-transmit, linear-receive (CTLR) compact-pol acquisition from quad-pol SLC channels, for missions with no compact-pol hardware (RADARSAT-2). Follows Guo et al. (2018, *Sensors* 18(7):2271) exactly -- the transmitted Jones vector for right-hand circular polarisation is ``R = (1/2)[1, -j]^T``; applying the Sinclair matrix gives ``E_RH = (Shh - j*Sx)/2``, ``E_RV = (Sx - j*Svv)/2``, with the usual reciprocity average ``Sx = (Shv+Svh)/2``. Returns a locally-averaged RH/RV covariance matrix (``c11``/``c22``/``c12_re``/``c12_im``) in the same naming ``rrrl_to_rhrv`` uses, so it plugs directly into ``polsar.h_alpha_dual`` or, via the new ``rhrv_to_stokes`` (the Stokes-parameter inverse of ``rrrl_to_rhrv``'s own mapping), into ``compact_pol_mchi`` unchanged -- the same function call RCM's real circular-receive product uses. Verified two independent ways: analytically exact on canonical pure-surface (``Shh=Svv``) and pure-double-bounce (``Shh=-Svv``) synthetic scatterers, which land 100% of their power in ``ps``/``pd`` respectively with zero leakage (``tests/rsense/test_sar.py::test_simulate_compact_pol``); and, on a real RADARSAT-2 Fine Quad scene, cross-checked against that same scene's independently-computed real quad-pol H-alpha zone classification -- the "low entropy surface" zone has the highest simulated surface-power fraction (0.68) of any zone, the "low entropy multiple" (high-alpha, double-bounce-like) zone has the highest double-bounce fraction (0.57), and the two quantities correlate with real quad-pol entropy/alpha in the physically expected direction (r = -0.72 and +0.70)
- Tutorial ``07_remote_sensing/07_03_radar/07_03_02_04_RADARSAT_CompPol.ipynb``: kept the existing compact-pol theory (it already matched the implemented pipeline) and added a real-data worked example. RADARSAT-2 has no compact-pol hardware, so there is no real RS2 CP product to read -- the notebook instead simulates RCM's CTLR acquisition mode directly from the real RS2 Fine Quad (FQ17) quad-pol SLC scene already used in the PolSAR tutorial (the same simulate-from-quad-pol technique Charbonneau et al. (2010) used on RADARSAT-2 data to validate what became RCM's compact-pol mode before RCM existed), runs it through the Stokes/m-chi pipeline, validates the sign convention both analytically and against the real quad-pol H-alpha classification, and produces an m-chi RGB composite that visually mirrors the real quad-pol Pauli RGB on the same scene (same lakes dark, same forest green, same village double-bounce cluster)

### Changed
- Version bumped to 0.3.18
- README remote-sensing feature line mentions ``simulate_compact_pol`` / ``rhrv_to_stokes``
- ``pygis.rsense.sar`` exports ``simulate_compact_pol``, ``rhrv_to_stokes``

## [0.3.17] - 2026-10-25

### Added
- ``pygis.rsense.sar.polsar``: quad-pol polarimetry. ``coherency_t3`` builds the locally-averaged 3×3 Pauli-basis coherency matrix T3 from HH/HV/VH/VV SLC channels (``k = [Shh+Svv, Shh-Svv, sqrt(2)(Shv+Svh)/2] / sqrt(2)``, cross-pol channels averaged first under the reciprocity assumption); ``h_alpha_quad`` is the full Cloude & Pottier (1997) H/A/α decomposition from T3's vectorised 3×3 eigendecomposition (``numpy.linalg.eigh`` over every pixel at once) -- entropy (log base 3, 0-1 for three mechanisms), anisotropy (the two smaller eigenvalues), and the power-weighted mean alpha angle (``arccos`` of each eigenvector's first, odd-bounce-axis component, the same convention the existing dual-pol ``h_alpha_dual`` uses for its 2×2 case). ``h_alpha_classify``'s H-α plane zone map needed no change to work on quad-pol output, since the zone boundaries only ever depended on entropy and alpha, not which eigendecomposition produced them. Verified against hand-computable pure-mechanism T3 matrices (pure surface: H=0, α=0°; pure double-bounce: H=0, α=90°; the textbook fully-depolarised limit point diag(1,1,1): H=1, α=60° exactly; a non-degenerate diag(4,2,1) case checked to machine precision) in ``tests/rsense/test_sar.py::test_quad_pol``, and against a real RADARSAT-2 Fine Quad scene, where it reproduces the well-known qualitative result that quad-pol α spans a much wider range than a dual-pol subset of the same pixels (median 43° vs. 20° on the same real scene)
- Tutorial ``07_remote_sensing/07_03_radar/07_03_02_03_RADARSAT_PolSAR.ipynb``: rewritten from a defunct standalone-``pyrsense``, simulated-dual-pol-only notebook into a real quad-pol run on a real RADARSAT-2 Fine Quad SLC scene (beam FQ17, 2015-06-26, Mecklenburg Lake District, Germany). Covers a real reciprocity check (``|Shv|`` vs. ``|Svh|`` after multilooking, median -0.03 dB), a Pauli RGB composite that resolves real lakes/forest/fields/a village (double-bounce) without any land-cover labels, the full H-α-plane coverage quad-pol data gives versus a dual-pol subset confined to a narrow low-α band, and a direct same-scene, same-pixels quad- vs. dual-pol alpha-histogram comparison; notes Freeman-Durden/model-based decomposition as a deliberate gap (several published variants differ in their exact volume-scattering branch and normalisation constants, and risking a subtly wrong formula was judged worse than not having it)

### Changed
- Version bumped to 0.3.17
- README remote-sensing feature line mentions ``coherency_t3`` / ``h_alpha_quad``
- ``pygis.rsense.sar`` exports ``coherency_t3``, ``h_alpha_quad``

## [0.3.16] - 2026-10-25

### Added
- ``pygis.rsense.sar.radarsat.read_rcm_annotation``: reads an RCM Level-1 **Stripmap** SLC ``product.xml`` (one continuous image, no TOPS bursts; its own precise orbit state vectors, no separate orbit-file download needed) into the same annotation shape ``pygis.rsense.sar.sentinel1.read_slc_annotation`` returns for one Sentinel-1 burst -- a single-entry ``bursts`` list covering the whole image -- so ``sentinel1.back_geocode`` and ``resample_slc`` (called with ``ann=None``, since Stripmap has no TOPS burst steering to deramp) coregister it unchanged, and the already sensor-agnostic ``pygis.rsense.insar`` chain (interferogram, coherence, Goldstein filtering, SNAPHU unwrapping, phase-to-displacement, terrain correction) needed no RCM-specific code at all. Tested in ``tests/rsense/test_sar.py::test_read_rcm_annotation`` against a synthetic ``product.xml`` matching the real schema, including the descending-row-time case (below) and the ascending case for contrast
- Tutorial ``07_remote_sensing/07_03_radar/07_03_02_02_RADARSAT_InSAR.ipynb``: the InSAR theory completed with a real two-pass run on a genuine RCM Stripmap SLC pair -- beam 16M11, relative orbit 15, 2023-09-13/2023-09-17 (RCM's 4-day repeat), Montreal, downloaded fresh from NRCan/EODMS's ``RCMImageProducts`` STAC collection and Data Delivery Service (bearer-token auth) since nothing already cached shared a beam mode. Runs coregistration, differential interferogram and coherence (mean 0.35, real and geocoded-confirmed horizontal bands of higher coherence plausibly tied to Quebec's historic strip-field land division), Goldstein filtering, SNAPHU unwrapping, phase-to-LOS conversion and terrain correction; the unmasked unwrapped field's implausible range and a coherence-threshold mask's failure to fix it are kept and explained rather than hidden -- isolated high-coherence islands surrounded by decorrelated pixels get the wrong absolute 2π count from SNAPHU, a real failure mode distinct from per-pixel noise, and exactly the problem persistent-scatterer/SBAS network methods (the PS-InSAR and SBAS tutorials) exist to avoid

### Fixed
- ``sentinel1.terrain_correct`` assumed row azimuth-time always increases with row number (true for every Sentinel-1 product used so far, false for the RCM product this tutorial uses, which stores rows in decreasing-time order) -- its internal ``np.interp`` needs ascending ``xp``, so a descending ``azimuth_time`` silently produced an all-NaN result (every row maps outside the DEM's apparent time range). Fixed by sorting ascending internally when needed, a no-op for every existing ascending-time caller; regression-tested in ``tests/rsense/test_insar_topo.py::test_terrain_correct_places_radar_pixels_on_the_dem`` with a reversed-row-order variant of the existing case

### Changed
- Version bumped to 0.3.16
- README remote-sensing feature line mentions ``read_rcm_annotation`` and the RCM Stripmap InSAR path
- ``pygis.rsense.sar`` exports ``read_rcm_annotation``

## [0.3.15] - 2026-10-25

### Added
- ``pygis.rsense.sar.radarsat``: RADARSAT Constellation Mission (RCM) Level-1 GRD radiometric calibration. ``read_rcm_lut`` / ``interpolate_rcm_lut`` parse and interpolate a ``lutSigma``/``lutBeta``/``lutGamma`` calibration XML (a range-only gain curve, constant along azimuth); ``read_rcm_noise_levels`` / ``interpolate_rcm_noise`` parse and stitch a ``noiseLevels`` XML's per-beam NESZ segments (a ScanSAR product packs several sub-beams, e.g. SC-7 through SC-14, each covering its own column range; a Stripmap product has just one), filtered to the calibration type (sigma0/beta0/gamma0 nought) matching the gain LUT used; ``calibrate_rcm`` combines both into ``sigma0 (or beta0, gamma0) = (DN² + offset) / gain``, with the NESZ floor subtracted directly in that calibrated domain -- confirmed against a real RCM-1 SCLND (ScanSAR Low Noise) product that RCM's LUT gain is used unsquared (dividing by the squared gain, the Sentinel-1 convention, put sigma0 five to six orders of magnitude below any plausible backscatter; dividing by the gain directly gives the expected -20 to 0 dB range for land). Multilook, speckle filtering and dB conversion for the calibrated result reuse the existing sensor-generic ``pygis.rsense.sar.sentinel1`` / ``pygis.rsense.sar.speckle`` functions; tested in ``tests/rsense/test_sar.py::test_radarsat_grd_calibration`` against synthetic LUT/noise-level XML matching the real schema (including overlapping ScanSAR beam segments and a DN=0 no-data case)
- Tutorial ``07_remote_sensing/07_03_radar/07_03_02_01_RADARSAT_GRD.ipynb``: the GRD backscatter workflow theory completed with a real-data run -- a real RCM-1 Level-1 GRD product (SCLND beam, VV+VH, 2023-10-25, Cox's Bazar -- Teknaf coast, Bangladesh, the same AOI as the Sentinel-1 GRD and dual-pol tutorials). Covers noise removal and calibration to sigma0 (a real 9.5% of the VH scene sits at the calibrated noise floor), a real equivalent-number-of-looks comparison (5.8 single-look to 8.2 after 2x2 multilook to 26 after Refined Lee, on a real homogeneous patch), and GCP/ellipsoid geocoding (the same fallback the Sentinel-1 GRD tutorial used for a source lacking full annotation) for a final map-projected image; notes Steps 6-7 (terrain flattening, range-Doppler terrain correction) as a known gap, since they need an orbit/geolocation-grid reader for RCM's own ``product.xml`` schema that ``pygis`` does not have

### Changed
- Version bumped to 0.3.15
- README remote-sensing feature line mentions ``pygis.rsense.sar.radarsat``'s Level-1 GRD calibration
- ``pygis.rsense.sar`` exports ``read_rcm_lut``, ``interpolate_rcm_lut``, ``read_rcm_noise_levels``, ``interpolate_rcm_noise``, ``calibrate_rcm``

## [0.3.14] - 2026-10-25

### Added
- ``pygis.rsense.sar.radarsat``: processing for RADARSAT Constellation Mission (RCM) CEOS-ARD compact-pol data, following NRCan/EODMS's own reference notebooks (https://github.com/eodms-sgdot/radarsat-notebooks). ``rr_rl_to_stokes`` (circular-basis Stokes parameters ``s0``-``s3`` and degree of polarization from the ARD product's native RR/RL backscatter and RRRL* complex correlation); ``rrrl_to_rhrv`` (lossless conversion to the linear RH/RV compact-pol covariance matrix, returned as ``{c11, c22, c12_re, c12_im}`` -- the same C2 convention ``pygis.rsense.sar.polsar`` uses, so its dual-pol H/α, Stokes and model-based decomposition functions apply directly, since Cloude's eigendecomposition is basis-independent); ``compact_pol_mchi`` (the m-χ surface/volume/double-bounce decomposition in RCM's own real-circular-transmit sign convention, which runs the opposite way from ``polsar.m_chi_decomposition``'s generic linear-pol analogy -- matches the authoritative EODMS implementation rather than the generic one); ``split_rrrl`` (real/imaginary bands of the two-band RRRL* GeoTIFF asset) and ``gamma0_to_sigma0`` (the ARD product's own precomputed, DEM-derived terrain-correction ratio, as opposed to ``sentinel1.sigma0_to_gamma0``'s ellipsoid-only approximation in the other direction). All four power-based outputs conserve total power exactly (``ps+pd+pv == s0``, ``c11+c22 == rr+rl``), tested in ``tests/rsense/test_sar.py::test_radarsat`` against synthetic compact-pol data (including a fully-correlated, fully-polarized sanity check) and an xarray-input check for aligned ``Dataset`` output
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_07_DualPol.ipynb``: the dual-pol decomposition theory completed with a real-data run -- a Sentinel-1A IW SLC burst (VV+VH, 2024-01-13, Cox's Bazar -- Teknaf coast, Bangladesh, the same acquisition and AOI as the GRD tutorial). Covers the C2 covariance matrix and multilook, ``h_alpha_dual``/``stokes_parameters``/``m_delta_decomposition``/``m_chi_decomposition``/``dual_pol_model_decomposition`` on real data (power conservation verified to float32 precision), an H-alpha plane density plot with Cloude-Pottier zone occupancy, a real double-bounce signal localised to built structures near Cox's Bazar airport (vs. under 0.05% scene-wide), and a terrain-corrected overview map; notes the known gaps this exposed (no per-channel calibration of ASF burst-extractor SLC yet, no precise-orbit option) rather than hiding them
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_08_interferometric_coherence.ipynb``: the coherence theory completed with a real-data run reusing the Sentinel-1A Mexico City SLC stack and precise orbits already cached by the DInSAR tutorial (no new download) -- four pairs against one reference at 12/36/96/150-day baselines, the last also crossing the 2025 Sentinel-1A-to-1C handover. Forms coherence with ``pygis.rsense.insar.interferogram()`` directly (rather than the DInSAR/SBAS notebooks' inline hand-rolled estimator), debursts it, reports ``coregister_stack``'s ESD diagnostics as a coherence quality check (the handover pair's -184 mm correction vs. 6-29 mm for the same-satellite pairs, with no resulting anomaly in the coherence trend), geocodes a short- and long-baseline map, and builds a data-driven coherence-vs-baseline decay curve (median coherence and % of scene above 0.3) instead of assuming land-cover polygons; notes that TOPSAR-Merge is the one workflow step with no public PyGIS function, unneeded here since every AOI in this series fits one sub-swath

### Fixed
- ``polsar.h_alpha_dual``'s and the m-χ/eigen-model decompositions' use of bare ``np.where`` on what can be an ``xr.DataArray`` is now a shared ``pygis.rsense.core._utils.where`` helper (``xr.where``/``np.where`` dispatch), reused by the new ``radarsat.compact_pol_mchi`` to avoid the same NEP-18 gap (a bare ``np.where`` on a DataArray silently drops to a plain ndarray, which ``xr.Dataset({...})`` then promotes to its own anonymous dimension coordinate instead of aligning it with the other outputs)

### Changed
- Version bumped to 0.3.14
- README remote-sensing feature line and package layout mention ``pygis.rsense.sar.radarsat``
- ``pygis.rsense.sar`` exports ``rr_rl_to_stokes``, ``rrrl_to_rhrv``, ``compact_pol_mchi``, ``split_rrrl``, ``gamma0_to_sigma0``

## [0.3.13] - 2026-10-24

### Added
- ``pygis.rsense.sar.polsar``: ``stokes_parameters`` (Stokes-vector parameters ``s0``-``s3`` and degree of polarization ``m`` from a dual-pol C2 covariance matrix, by analogy with the compact-pol hybrid-pol architecture the m-χ/m-δ framework was built for); ``m_delta_decomposition`` and ``m_chi_decomposition`` (Raney 2007 m-δ/m-χ model-based surface/double-bounce/volume power decompositions, exact for the relative co-cross phase δ and ellipticity-style angle χ respectively, each conserving total power ``ps+pd+pv == span`` by construction); ``dual_pol_model_decomposition`` (an eigen-decomposition model-based decomposition generalizing the Freeman-Durden/van Zyl surface/double-bounce/volume split to the 2×2 dual-pol case, following the degree-of-polarization / mean-α framework of dual-pol model-based decompositions such as Mascolo et al. 2022: the depolarized span ``(1-m)·span`` is attributed to volume scattering, the polarized remainder split into surface/double-bounce by whether ``h_alpha_dual``'s mean α is below/above 45°)
- ``tests/rsense/test_sar.py::test_stokes_and_model_decompositions``: degree-of-polarization sanity checks (near 1 for a single scattering mechanism, low for independent channels), exact power conservation and non-negativity of all three decompositions on synthetic C2 data, and an xarray-input check that each function returns a ``Dataset`` with its variables aligned on one dimension

### Fixed
- ``polsar.h_alpha_dual``'s entropy term used bare ``np.where`` on what can be an ``xr.DataArray`` (``p1``/``p2``); NEP-18 dispatch for ``np.where`` silently drops back to a plain ndarray, so passing ``DataArray`` C2 elements (the normal case for raster workflows) produced a plain array that, merged back into the returned ``Dataset``, worked by coincidence only because every other field happened to still carry a `DataArray`'s existing dimension name. The new ``m_chi_decomposition`` and ``dual_pol_model_decomposition`` hit the same gap directly: their bare-ndarray intermediate, once placed in ``xr.Dataset({...})``, was promoted to its own anonymous dimension *named after the variable itself* instead of aligning with the other outputs. Fixed with a small ``xr.where``/``np.where`` dispatch helper used throughout the module

### Changed
- Version bumped to 0.3.13
- README remote-sensing feature line: PolSAR now lists the Stokes/m-δ/m-χ/model-based decomposition functions
- ``pygis.rsense.sar`` exports ``stokes_parameters``, ``m_delta_decomposition``, ``m_chi_decomposition``, ``dual_pol_model_decomposition``

## [0.3.12] - 2026-10-24

### Added
- SBAS: new ``pygis.rsense.insar.sbas`` with ``sbas_network`` (pairs by temporal / perpendicular baseline limits or nearest neighbours, optionally forced connected) and ``network_components``; ``interferogram_stack`` (one call from ``sar.coregister_stack`` output: differential interferograms, deburst, multilook, coherence, Goldstein filter and SNAPHU unwrapping for every pair, plus per-date ``kz`` and geometry, with the attributes ``sar.terrain_correct`` needs); ``loop_closure`` and ``correct_unwrapping_errors`` (phase closure of interferogram triplets; minimum-L1 integer corrections by ``scipy.optimize.milp`` per distinct closure pattern, MintPy-style); ``sbas_svd`` (Berardino et al. 2002 velocity parameterisation solved by SVD/minimum norm, so split networks still invert; coherence weights and missing interferograms handled in vectorised chunks); ``dem_error_correction`` (Fattahi & Amelung 2013, per-pixel kz history, optional seasonal terms); ``tests/rsense/test_sbas.py`` (5 tests)
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_06_InSAR_SBAS.ipynb``: the SBAS theory completed with a real-data run -- 24 Sentinel-1 dates of 2025 over Mexico City, network design, unwrapping-error correction, SVD inversion (including a deliberately split network), DEM-error and atmospheric correction, validated against GNSS and compared with the DInSAR 12-date SBAS and the PS points
- PS-InSAR: new ``pygis.rsense.insar.psinsar`` with ``amplitude_dispersion`` (radiometrically normalised D_A), ``ps_candidates`` (threshold plus one candidate per amplitude peak), ``sparse_lowpass`` (Gaussian low-pass of scattered points, leave-one-out option), ``ps_periodogram`` (per-point DEM error and optional linear rate maximising temporal coherence), ``coherence_threshold`` (random-phase false-positive threshold for the same estimator), ``unwrap_points`` (sparse PS unwrapping through a SNAPHU-unwrapped coarse grid plus per-point residuals), ``atmospheric_filter`` (APS: high-pass in time, low-pass in space) and ``point_velocity``; ``sar.coregister_stack`` now also returns each date's height sensitivity ``kz``; ``sar.geocode_points`` (map positions of radar-geometry points from a terrain-correction result, with DEM-error correction along the look direction); ``tests/rsense/test_psinsar.py`` (10 tests); ``unwrap_points(..., sequential=True)`` unwraps consecutive-date differences (temporal continuity, anchored to whole cycles), ``amplitude_dispersion`` masks pixels with too few valid dates, ``ps_periodogram`` returns NaN coherence for incomplete geometry
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_05_PSInSAR.ipynb``: the PS-InSAR theory completed with a real-data run -- 24 Sentinel-1 dates of 2025 over Mexico City (descending track 41), StaMPS-style PS selection, DEM-error estimation, sparse unwrapping and APS filtering, validated against NGL GNSS stations and the SBAS velocities of the DInSAR notebook
- Differential InSAR: ``pygis.rsense.insar.dinsar`` with ``topographic_phase`` (the DEM-predicted flat-Earth + topographic phase from back-geocoding's slant-range difference), ``differential_interferogram``, ``select_reference_point`` (most coherent window inside a stable mask, optional velocity limit), ``reference_to_point`` (2-D maps and stacks) and ``enu_to_los``; ``pygis.rsense.sar.coregister_stack`` (co-register a multi-date SLC stack to one reference: back-geocoding, deramped resampling and ESD per secondary, keeping each date's range difference for topographic phase removal of any pair) and ``sar.los_unit_vector`` (exact ground-to-satellite ENU direction from the orbit, for GNSS comparison); ``tests/rsense/test_dinsar.py`` (4 tests)
- ``insar.unwrap(method="snaphu")`` silences SNAPHU's console log; ``sar.download_orbit`` and ``insar.geoid_to_ellipsoid`` retry transient network errors
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_04_DInSAR.ipynb``: the DInSAR theory and Steps 0-17 completed with a real-data implementation -- Mexico City subsidence from 12 Sentinel-1 dates in 2025 (descending track 41), one differential interferogram and a 21-pair SBAS time series, validated against NGL GNSS stations MMX1 and ICMX
- ``pygis.rsense.sar.sentinel1`` Sentinel-1 IW SLC / TOPS burst processing (the pre-processing of SNAP's TOPS InSAR graph, natively in Python): ``download_orbit`` (precise POEORB, then RESORB, from the open AWS ``s1-orbits`` bucket; cached files reused offline), ``crop_orbit``, ``apply_orbit_file``; ``read_slc_annotation`` (SAFE annotation or ASF burst-extractor metadata XML, which embeds every sub-swath -- choose with ``swath=``/``polarisation=``; bursts, valid lines/samples, FM-rate and Doppler polynomials, orbit, geolocation grid) and ``read_slc_burst`` (TOPS Split to an AOI column window); ``llh_to_ecef``, ``geo2rdr`` (zero-Doppler range-Doppler geocoding on a Hermite-interpolated orbit), ``burst_coordinates``, ``burst_window``; ``back_geocode`` (DEM-assisted co-registration plus flat-Earth range difference and phase-to-height factor per reference pixel); ``tops_deramp_phase`` / ``tops_doppler_centroid`` / ``estimate_doppler_centroid`` and ``resample_slc`` (deramp, spline interpolation, reramp); ``esd_azimuth_shift`` (Enhanced Spectral Diversity from the burst-overlap double difference); ``deburst``; ``terrain_correct`` (Range-Doppler terrain correction of radar-geometry layers onto a DEM's map grid); ``layover_shadow_mask``; ``SPEED_OF_LIGHT``. Verified on real ASF bursts: deramping brings the measured azimuth Doppler of a burst from a +-2.2 kHz sweep to -0.6 +- 1.6 Hz, and ESD converges from 0.0043 px to 0.5 mm
- New ``pygis.rsense.insar.topo`` (topographic InSAR / DEM generation): ``perpendicular_baseline`` (from orbits; matches ASF's baseline tool), ``height_of_ambiguity``, ``flat_earth_phase``, ``unwrap_snaphu_topo`` (SNAPHU ``TOPO`` statistical-cost mode via the binary bundled with ``pip install snaphu``), ``phase_to_height`` (with per-connected-component ties to a reference DEM), ``geoid_to_ellipsoid`` (EGM96/EGM2008 via PROJ), ``height_std`` (Cramer-Rao height error), ``detrend_heights`` (plane + elevation-proportional tropospheric trend), ``combine_heights`` (inverse-variance multi-pair average), ``dem_error_stats`` (bias, NMAD, RMSE, LE90), ``fill_dem_voids`` (delta-surface gap filling)
- ``pygis.rsense.access.asf_download`` reads Earthdata credentials from ``~/.netrc`` and downloads burst metadata XML with the GeoTIFF (``all_files=True``); ``snaphu`` and ``sliderule`` added to the ``rsense-insar`` extra
- ``tests/rsense/test_insar_topo.py`` (15 tests): geo2rdr zero-Doppler geometry, baselines and height of ambiguity on a synthetic orbit, back-geocoding flat-Earth/height factor, TOPS deramp/Doppler consistency, ESD shift recovery, deburst, terrain correction, layover/shadow, annotation and burst readers (including burst-extractor sub-swath selection), orbit-file selection, SNAPHU TOPO unwrapping, detrending, height statistics, void filling
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_03_insar.ipynb`` rewritten: a DEM of Death Valley (Panamint Range to Badwater Basin) from Sentinel-1 IW SLC bursts, June 2025, following the 13-step topographic InSAR chain natively in PyGIS and validated against Copernicus DEM and ICESat-2 (replaces the earlier simulated DInSAR notebook)
- ``pygis.rsense.sar.sentinel1`` multitemporal GRD stacks: ``read_grd_subset`` (calibrated sigma0 for a lon/lat window of a local or remote GRD measurement image, located with the scene's GCPs; LUTs sampled at the window's true offset; only the GCPs around the window are kept so geocoding fits the local geometry), ``common_grid`` (snapped target grid), ``create_stack`` (SNAP ``CreateStack`` equivalent: every scene geocoded onto one ``(time, y, x)`` grid, tz-aware STAC datetimes converted to naive UTC), ``multitemporal_speckle_filter`` (Quegan & Yu filter for ``xarray`` DataArrays / Datasets, wrapping ``speckle.multitemporal``), ``estimate_offset`` / ``apply_offset`` (register a GCP-geocoded stack to a terrain-corrected reference such as Planetary Computer RTC); ``read_calibration_lut`` / ``read_noise_lut`` accept http(s) URLs (e.g. Planetary Computer ``schema-calibration-*`` assets); 6 new tests in ``tests/rsense/test_sar.py``
- Tutorial ``07_remote_sensing/07_03_radar/07_03_01_02_sentinel1_multitemporal_grd.ipynb``: a 19-date Sentinel-1 GRD time series of the 2017 Rohingya-camp deforestation (Kutupalong–Balukhali), steps 1–9 from calibration to Random Forest / XGBoost / LSTM classification validated against Sentinel-2
- ``pygis.rsense.optical.sensors.sentinel3``: readers and mask functions for Sentinel-3 OLCI, SLSTR, SRAL and Synergy products (``.SEN3`` NetCDF swath data, which does not fit the lazy STAC-raster loaders the other sensors share). OLCI: ``read_olci_l1`` / ``olci_toa_reflectance`` / ``olci_l1_mask`` (EFR/ERR, with tie-point angle interpolation), ``read_olci_water`` / ``olci_water_mask`` (WFR/WRR, OC4ME and neural-net flag sets), ``read_olci_land`` / ``olci_land_mask`` (LFR/LRR). SLSTR: ``read_slstr_l1`` / ``slstr_reflectance`` / ``slstr_flag_mask`` (RBT, across its five grid resolutions), ``read_slstr_lst`` / ``slstr_lst_mask``, ``read_slstr_sst`` / ``sst_bias_correct`` / ``sst_quality_mask``, ``read_slstr_frp`` / ``frp_to_points`` (GeoDataFrame) / ``aggregate_frp``. SRAL: ``read_sral_marine`` / ``sral_marine_mask``, ``read_sral_virtual_station`` (inland water level time series). Synergy: ``read_synergy`` / ``synergy_mask``, ``read_synergy_vgt`` (already-gridded VGP/VG1/V10). Generic helpers: ``decode_flags`` / ``flag_mask`` (read ``flag_masks`` / ``flag_meanings`` from the variable itself rather than a hardcoded bit position), ``swath_to_grid`` (KD-tree or linear regridding onto a georeferenced grid), ``log_to_linear``, ``kelvin_to_celsius``, ``read_manifest``
- ``MCI`` (Gower et al. 2005) and ``OTCI`` (Dash & Curran 2004, OLCI's MTCI equivalent) spectral indices; ``SENSOR_BANDS["sentinel3_olci"]`` canonical band mapping and OLCI band-name detection in ``guess_sensor``, so the existing index catalogue (NDVI, NDCI, ...) also runs on OLCI reflectance
- ``rsense-sentinel3`` extra in ``pyproject.toml`` (``netCDF4``, ``h5netcdf``, ``pyproj``); ``rsense-all`` now includes it
- ``tests/rsense/test_sentinel3.py`` (19 tests): flag decoding and masking against synthetic CF ``flag_masks``/``flag_meanings`` metadata, the OLCI TOA reflectance and log-to-linear formulas, SST bias correction and quality filtering, SRAL editing criteria, FRP-to-points, swath-to-grid regridding, the MCI/OTCI formulas, the ``log_to_linear`` → ``swath_to_grid`` chain, and end-to-end reads of synthetic ``.SEN3``-folder NetCDF fixtures for OLCI L1, L2 water and L2 land
- New subpackage ``pygis.rsense.atmosphere`` with ``sentinel_5p``: readers and quality masking for Sentinel-5P/TROPOMI Level-2 products (one self-contained NetCDF-4 file per scene, read from its nested ``PRODUCT`` group, unlike Sentinel-3's multi-file ``.SEN3`` swaths). ``read_no2`` / ``read_so2`` / ``read_o3`` / ``read_co`` / ``read_ch4`` / ``read_hcho`` (trace-gas columns), ``read_aerosol_index`` (UV absorbing aerosol index), ``read_aerosol_layer_height`` (mid-layer height), ``read_cloud`` (fraction, top pressure/height, optical thickness); every reader accepts a ``variables=`` override and returns ``latitude``/``longitude`` as coordinates. ``qa_mask`` reads ESA's own recommended ``qa_value`` threshold straight out of the variable's ``comment`` attribute instead of a hardcoded constant (the threshold differs by product and has changed between processing baselines), with ``S5P_QA_RECOMMENDED`` giving the stricter literature-recommended cutoffs (NO2 > 0.75, AER_AI > 0.8) as an opt-in alternative; ``molecules_per_cm2`` reads the file's own ``multiplication_factor_to_convert_to_molecules_percm2`` attribute rather than typing in Avogadro's number; ``dobson_units`` for ozone and SO2 (validated against a real downloaded scene: 313.99 DU, matching climatology); ``read_metadata`` (orbit, processing mode NRTI/OFFL/RPRO, processor version); ``swath_to_grid`` re-exported from ``sentinel3`` rather than duplicated, since the resampler is sensor-agnostic
- ``sentinel_5p`` Level-2-to-3 gridding and aggregation: ``bin_to_grid`` (pixel-centre averaging onto a regular lat/lon grid, the fast default for cells >= ~0.05°), ``oversample_to_grid`` (pixel-corner area-weighted gridding via shapely polygon intersection, using ``read_geolocation_bounds`` -- more accurate on finer grids or multi-orbit composites, at real computational cost), ``temporal_composite`` (accumulates several swaths, by either method, into one monthly/seasonal mean with a minimum-observation-count cutoff)
- ``sentinel_5p`` NO2 averaging-kernel model comparison: ``tm5_pressure_levels`` (TM5 hybrid sigma-pressure layer interfaces from ``tm5_constant_a``/``tm5_constant_b`` and surface pressure) and ``apply_averaging_kernel`` (AK-smoothed model column for a fair comparison against the satellite's tropospheric column); both validated against a real NO2 scene -- the bottom TM5 layer's pressure matched surface pressure exactly, and an AK = 1 sanity check reproduced the plain hydrostatic column calculation to 6 decimal places
- ``sentinel_5p`` CH4 point-source emission estimation (Varon et al. 2018 Integrated Mass Enhancement method): ``ch4_enhancement`` (background subtraction, scene median by default), ``integrated_mass_enhancement`` (ppb enhancement to total plume mass in kg via the dry-air column), ``estimate_emission_rate`` (``Q = IME * U_eff / L``); and ``co_mixing_ratio`` (CO column to dry-air mixing ratio, XCO)
- ``sentinel_5p`` SO2 volcanic plume-height variants: ``read_so2(..., layer_height="1km"/"7km"/"15km")`` reads ``sulfurdioxide_total_vertical_column_1km`` etc. from ``PRODUCT/SUPPORT_DATA/DETAILED_RESULTS``, confirmed against a real downloaded product
- ``sentinel_5p.check_processing_consistency``: flags processor-version drift and the real 6 August 2019 along-track pixel size change (7 km to 5.5 km) across a list of product files, cheap (global attributes only, no pixel data) -- the standard caution against mixing processing baselines in a time series
- ``rsense-sentinel5p`` extra in ``pyproject.toml`` (``netCDF4``, ``h5netcdf``); ``rsense-all`` now includes it
- ``tests/rsense/test_sentinel5p.py`` (33 tests): ``qa_value`` comment-threshold parsing and override, ``molecules_per_cm2`` and ``dobson_units`` against known conversion factors, end-to-end reads of synthetic TROPOMI ``PRODUCT``-group NetCDF fixtures (including that ``latitude``/``longitude`` come back as coordinates and feed straight into ``swath_to_grid``), the SO2 layer-height reader, ``bin_to_grid``/``oversample_to_grid``/``temporal_composite`` against hand-computed cell means, the averaging-kernel formula's internal physical consistency (AK = 1 and AK = 0 sanity checks), the CH4/CO formulas against their manual hydrostatic-column derivation, ``check_processing_consistency``'s version- and era-change flagging, graceful handling of a missing ``qa_value``, and an unknown-variable error path
- ``pygis.rsense.sar.sentinel1``: ``remove_border_noise`` (GRD border-noise masking, a lightweight native approximation of SNAP's ``Remove-GRD-Border-Noise``, wired into ``process_grd(..., backend="native", border_noise=True)`` as the default first step on raw DN); ``read_orbit_file`` / ``interpolate_orbit`` (parse a precise/restituted Sentinel-1 orbit file, ``.EOF``, into ECEF state vectors and cubic-spline interpolate them to arbitrary times -- what ``Apply-Orbit-File`` substitutes for a product's own orbit metadata); ``read_geolocation_grid`` / ``incidence_angle_grid`` / ``read_platform_heading`` (parse the annotation XML's sparse geolocation grid and platform heading -- closes a real gap, since nothing previously supplied the ``incidence_deg`` that ``sigma0_to_gamma0`` and ``pygis.rsense.insar.deformation.los_to_vertical`` both require); ``local_incidence_angle`` / ``terrain_flatten`` (DEM-based terrain flattening, beta0 to true terrain-flattened gamma0 via the Small (2011) surface-normal/look-vector formulation, the native counterpart to ``sigma0_to_gamma0``'s ellipsoid-only approximation and to the ``sarsen``/``snap`` backends' full RTC). VH/VV ratio and the dual-pol radar vegetation index were already covered by ``pygis.rsense.sar.polsar.dual_pol_features``
- ``tests/rsense/test_sar.py``: 6 new tests for the above (border-noise edge trimming, orbit-file parsing and interpolation against a synthetic ``.EOF``, geolocation-grid/incidence-angle/heading parsing against a synthetic annotation XML, and the local-incidence-angle/terrain-flattening formulas, including a flat-ground sanity check against the ellipsoid case); all functions were also exercised directly against a real downloaded Sentinel-1 GRD product
- ``pygis.rsense.get_images``: ``rcm-ard`` source/collection for RADARSAT Constellation Mission (RCM) CEOS Analysis Ready Data (https://registry.opendata.aws/rcm-ceos-ard/) -- systematic compact-polarization, 30 m, C-band coverage of Canada, served as cloud-optimized GeoTIFFs from Natural Resources Canada's EODMS STAC catalogue and mirrored, open and unsigned, in the AWS ``rcm-ceos-ard`` bucket (``ca-central-1``). ``search_rcm_ard`` / ``get_rcm_ard`` wrap ``search_images``/``get_images`` with the source and collection pre-filled; ``orbit="ascending"/"descending"`` filters by ``sat:orbit_state`` client-side in Python, not as a server-side STAC query -- EODMS's catalogue does not advertise the STAC ``query`` extension that a server-side filter would need, confirmed against the live API (it silently returns both orbits otherwise, rather than erroring). ``RCM_ARD_BANDS`` lists every raster asset (``rr``/``rl`` backscatter, ``rrrl`` compact-pol covariance matrix, ``local_inc_angle``, ``gamma_to_sigma_ratio``, ``local_contributing_area``, ``data_mask``); ``RCM_ARD_DEFAULT_BANDS`` (``rr``, ``rl``, ``data_mask``) is what downloads when ``bands`` is omitted. Verified end to end against the live EODMS catalogue and the real S3 bucket: a real scene over Ottawa searched, orbit-filtered, and clipped to a small AOI in under 3 seconds. Also wired into the standalone ``PyGIS-Explorer/image_downloader.py`` CLI (``--source rcm-ard``)
- ``tests/rsense/test_get_images.py``: orbit-filtering (including the client-side-filter path and an invalid-orbit error), ``clearest`` degrading gracefully with no ``eo:cloud_cover`` field, and a download dispatch test for ``rcm-ard``
- ``pygis.web.start_server`` / ``BackgroundServer``: run ``create_app`` (or any ASGI app) with uvicorn in a background thread, for notebooks and tests (``port=0`` picks a free port; ``url``, ``stop()``, context manager)
- ``pygis.web.static_server(..., subprocess=True)`` / ``SubprocessStaticServer``: the range-request file server in a child process with the same ``url`` / ``stats`` / ``reset`` / ``stop`` interface; needed for rasterio, which holds the GIL during network reads and deadlocks against an in-process server
- ``create_app``: ``/api`` OpenAPI 3.0 document linked from the landing page as ``service-desc`` (OGC API – Features Core); ``datetime`` parameter (RFC 3339 instant or interval, open ends ``..``) filtering on a collection's single datetime column and accepted (ignored) for collections without one; temporal extent in collection documents
- Web and cloud GIS subpackage ``pygis.web`` (reachable lazily as ``pygis.web``): ``to_vsi_path`` (``s3://``, ``gs://``, ``az://``, ``https://`` and ``archive!/member`` URLs to GDAL ``/vsi`` paths), ``cloud_options`` / ``cloud_env`` (GDAL settings for fast or anonymous range reads, applied to both ``osgeo.gdal`` and rasterio and restored on exit), ``remote_info`` (one-byte range request: size, content type, range support; network failures reported, not raised); ``to_cog`` (GDAL COG driver, optional web-optimized Web Mercator grid) and ``validate_cog`` (tiling, internal overviews, IFD order, data order, ``LAYOUT=COG``; local or remote) returning a ``COGReport``; XYZ tile math (``lonlat_to_tile``, ``tile_bounds``, ``tiles_for_bounds``, ``tile_resolution``, ``zoom_for_resolution``), ``raster_web_info``, ``tilejson``, ``suggest_rescale``, ``render_tile`` (warped to the tile footprint, colormaps, transparency for no-data) and ``make_xyz_tiles`` (static pyramid with ``tilejson.json`` and a Leaflet ``index.html``); ``create_app`` (Starlette app serving XYZ tiles, TileJSON and OGC API – Features with bbox, paging and attribute filters, plus a ``/viewer`` map) and ``serve`` (uvicorn); ``leaflet_html``
- Cloud-native vector writers in ``pygis.vector`` (also exported from ``pygis.web``): ``write_geoparquet`` (GeoParquet 1.1 with a ``bbox`` covering column, Hilbert-sorted rows and configurable row groups; works without a GDAL Parquet driver), ``read_geoparquet`` (bbox push-down with optional ``bbox_crs``), ``geoparquet_info`` (footer summary), ``write_flatgeobuf`` (packed Hilbert R-tree) and ``write_pmtiles`` (GDAL PMTiles driver)
- ``pygis.web.static_server`` / ``StaticServer``: a threaded static file server with single-range requests (206, 416 for multi-range), CORS headers and request/byte counters, for testing COG, GeoParquet, FlatGeobuf and PMTiles access locally
- ``web`` and ``web-cloud`` extras in ``pyproject.toml``; ``all`` now includes ``web``
- ``tests/test_web.py`` (26 tests: tile math checked against mercantile, VSI paths, cloud settings, ``remote_info`` against a local server, GeoParquet / FlatGeobuf / PMTiles round trips, COG creation and validation, tile rendering and pyramids, and the server endpoints through Starlette's test client)
- Spatial causal inference subpackage ``pygis.spatialcausal`` (ported from PySpatialCausal 0.3.1): ``SpatialCausalData`` / ``SpatialPanel`` / result containers, exposure mappings and randomisation designs, ``InterferenceEstimator`` (AIPW and Hájek), spatial-confounding diagnostics and adjustment (``moran_eigenvectors``, ``rbf_basis``, ``SpatialBasis``, ``SpatialPlus``), ``SpatialDML`` / ``SpatialDRLearner`` with spatial block cross-fitting and Conley / block standard errors, ``GWDML`` / ``GWCausalForest``, ``GeographicRDD``, ``SpatialDiD``, ``StaggeredSpatialDiD``, ``SpatialSyntheticControl``, ``BayesianSpatialCausal``, cluster / saturation designs and power analysis, ``SpatialGranger`` / ``GCCM`` / ``SpatialNOTEARS``, simulators and plots; reachable lazily as ``pygis.spatialcausal``. It uses ``pygis.spatial_statistics`` instead of the external PySpatialStats package. ``GWCausalForest(engine=...)`` also accepts ``"spatial_statistics"`` as an alias of ``"spatialstats"``
- ``spatialcausal`` extras in ``pyproject.toml``: ``spatialcausal``, ``spatialcausal-viz``, ``spatialcausal-causalml``, ``spatialcausal-all``; ``all`` now includes ``spatialcausal`` and ``spatialcausal-viz``
- ``tests/spatialcausal`` (106 tests): the three PySpatialCausal suites ported to ``pygis`` imports, plus ``test_spatialcausal_api.py`` with direct tests for the helpers, designs, variance estimators, result containers and plots
- Bayesian kriging in ``pygis.geostatistics.kriging.bayesian_kriging``: ``BayesianKriging`` (discrete priors on range and relative nugget, conjugate normal–inverse-gamma prior on trend and variance, closed-form grid posterior ``posterior_``, Student-t mixture predictive with ``predict``, ``predict_quantiles``, ``predict_proba``, marginal or joint ``predictive_samples``, ``sample_parameters``, ``from_variogram``), ``ConjugateBayesianKriging`` (fixed correlation), ``BayesianTransformedKriging`` (Box–Cox λ on the grid, normalised transform, Monte Carlo back-transform), ``BayesianGLMKriging`` (Poisson / binomial spatial GLMM by MCMC: MALA on the whitened latent field, random-walk β and σ², discrete range grid), ``bayesian_kriging`` one-call wrapper and ``correlation_function``; tests in ``tests/test_bayesian_kriging.py``; ``BayesianGLMKriging(interweave=True)`` (default) adds centred Gibbs updates with the linear predictor fixed (β Gaussian, collapsed range | S, σ² | S) — ancillarity-sufficiency interweaving — which raises the intercept's effective sample size about 20× and brings R-hat below 1.03 on the Atlantic-county example; ``BayesianTransformedKriging`` warns when λ < 0 draws fall outside the Box–Cox domain (infinite values: the predictive mean does not exist; use medians / quantiles); ``bayesian_kriging(..., signal=True, return_model=False)`` can predict observations and return the fitted model
- Remote sensing subpackage ``pygis.rsense`` (ported from PyRSense 0.2.0.dev0): backend (CPU / CUDA dispatch, precision, memory), core raster / cube / point-cloud I/O, data access (STAC, Earth Engine, ASF, CDSE, Earthdata), optical (sensor loaders, scaling, QA masks, spectral indices, composites, LST, unmixing), SAR, InSAR, lidar, UAV, time series, classification, fusion, deep learning, XAI, visualization, scaling (Dask / Zarr / STAC export) and Streamlit app templates; ``.rs`` xarray accessor; reachable lazily as ``pygis.rsense``
- ``rsense`` extras in ``pyproject.toml``: ``rsense``, ``rsense-access``, ``rsense-sar``, ``rsense-insar``, ``rsense-lidar``, ``rsense-uav``, ``rsense-classify``, ``rsense-dl``, ``rsense-xai``, ``rsense-gee``, ``rsense-viz``, ``rsense-scale``, ``rsense-apps``, ``rsense-gpu``, ``rsense-all``; ``all`` now includes ``rsense``
- ``tests/rsense`` (390 test cases across the CPU and CUDA backends), including ``test_host_only_functions_accept_device_arrays``

### Fixed
- ``sentinel3.read_olci_water`` / ``read_olci_land`` returned each variable under its raw uppercase NetCDF name (``CHL_NN``, ``KD490_M07``, ``RC681``/``RC865``) instead of the documented lowercase canonical name; real ``.SEN3`` files were never used to write the original readers, only synthetic test fixtures that matched the same wrong assumption. Both now rename from a file-stem -> real-variable-name map confirmed against real downloaded Planetary Computer products, and the fixtures were corrected to match
- ``sentinel3.log_to_linear`` stored its back-transform note under an attribute literally named ``"transform"``, which rioxarray reserves for a GDAL-style affine geotransform; calling ``swath_to_grid`` on its output crashed in ``write_crs``. Renamed to ``"back_transform"``
- ``sentinel_5p``'s per-product readers (``read_no2`` and the rest) built their output ``Dataset`` from only the requested main variable, its ``_precision`` companion and ``qa_value`` -- ``latitude``/``longitude`` were never included, leaving every reader's output unusable for mapping, ``swath_to_grid`` or zonal stats without reopening the file separately. Fixed before the first release; caught by testing the documented ``swath_to_grid`` workflow against a real downloaded NO2 scene, not just checking that the reader returned without error
- ``sentinel_5p.temporal_composite`` validated its ``method`` argument inside the accumulation loop, so passing an unknown method with an empty ``swaths`` list silently did nothing instead of raising; the check now runs before the loop
- ``create_app``: ``/items`` and ``/items/{id}`` crashed on collections with datetime columns (timestamps were not JSON-serialisable); they are now written as ISO 8601
- ``create_app``: feature responses were encoded on the event loop, three times over (``to_json``, ``json.loads``, ``json.dumps``), so one large page blocked every other request (tiles waited up to 4.4 s during a 1,000-union page). Features are now encoded once with GEOS (``shapely.to_geojson``) in Starlette's thread pool, in chunks that yield the GIL: about 7× faster (4.9 s to 0.7 s for that page), with identical output
- ``render_tile`` warped zoomed-out tiles from full resolution even when the source had overviews (a zoom-10 tile of a remote Sentinel-2 COG took 16–18 s); it now reads from the coarsest overview that is still at least as fine as the tile (2.7 s; 0.82 MB instead of 10 MB from a plain GeoTIFF of the same scene)
- ``static_server``: request and byte counters were updated after the last byte was sent, so a fast client could read stale ``stats`` (intermittent failure of ``test_static_server_ranges_counters_and_cors``); they are now updated before each write
- ``render_tile``: a single ``rescale`` pair given as a list (``[(0, 255)]``) now applies to every band; ``create_app`` checks ``bands`` and ``rescale`` against each raster at start-up instead of failing (and dropping the connection) on the first tile request
- Opening a remote PMTiles archive with ``open_cloud_native_vector_data`` decoded every tile at the feature zoom to count features (8.2 MB of a 22 MB archive over HTTP); GDAL vector-tile layers (PMTiles, MVT) on ``/vsi`` network paths now defer the count and extent scan (4 requests, 20 KB), while local files are still counted
- ``open_cloud_native_vector_data`` could not open GeoParquet or Arrow files over ``http(s)://`` (PyArrow has no HTTP filesystem); remote reads now use fsspec's HTTP filesystem without read-ahead (the default 5 MB block multiplied transfers, e.g. 2.0 GB for a 103 MB file) and PyArrow ``pre_buffer``
- GeoParquet layers of ``open_cloud_native_vector_data`` read the whole table for a ``bbox`` query; they now skip row groups whose GeoParquet 1.1 ``bbox`` covering statistics miss the box (a remote query on a 103 MB file went from 2.0 GB to 8.3 MB transferred) and keep the feature ids of an unfiltered read
- ``read_geoparquet`` failed for ``http(s)://`` URLs; it now passes fsspec's HTTP filesystem to GeoPandas
- ``correlation_function`` ignored the stable exponent (``model='stable'`` was always exponential); it now takes ``alpha``, and the Bayesian kriging classes pass ``nu`` as the stable exponent (``BayesianKriging.from_variogram`` carries a fitted stable α)
- rsense host-side steps (time-series smoothers, CCDC, OBIA, classifiers, SAR apps, lidar / UAV metrics, plots, ...) failed on GPU-backed DataArrays because ``to_cpu(x.values)`` converted to NumPy before the device copy; now ``to_cpu(x).values``
- ``tests/rsense``: ``test_write_requires_crs`` builds a CRS-free array directly (``rio.write_crs(None)`` raises in rioxarray >= 0.19); LAS and STAC export tests skip when ``laspy`` / ``pystac`` are missing
- ``sentinel1.process_grd(backend="native")`` returned an empty (0 x 0) raster whenever ``dst_crs`` was left at its default: it reprojected to the product's own geographic GCP CRS (EPSG:4326, degrees) while ``resolution`` is a metre-valued distance (20 m default), so asking for e.g. "20-degree" pixels rounded the whole scene to zero pixels. It now auto-picks the UTM zone under the scene (new ``_utm_from_gcps``) when ``dst_crs`` is not given; confirmed on a real downloaded Sentinel-1 GRD product (previously 0 x 0, now a correctly sized, correctly valued raster) and on a real Bangladesh coastal scene used to rewrite ``07_03_01_sentinel1_grd.ipynb`` (below)
- ``sentinel1.process_grd(backend="sarsen")`` passed the bare SAFE directory path as ``sarsen.terrain_correction``'s first argument; sarsen >= 0.9 takes a pre-built ``Sentinel1SarProduct``, not a path, there. It now constructs one first. Confirmed against a real downloaded product: this previously failed before even opening the DEM (``AttributeError``/wrong-type errors deep in sarsen's own code), and now reaches sarsen's real terrain-correction logic
- ``sentinel1.interpolate_lut`` (and ``incidence_angle_grid``, built on it) always evaluated a sparse calibration/noise/geolocation LUT as if the target grid started at the scene's own line/pixel 0 -- correct for a full-scene raster, wrong for any windowed/cropped read (e.g. a ``rasterio`` window read restricted to an AOI), which silently sampled the LUT at the wrong position. Both now take ``offset=(row0, col0)``, the window's true origin in the scene's own numbering

### Changed
- Version bumped to 0.3.12
- ``tests/test_spatialcausal.py`` and ``tests/test_v03.py`` (unported PySpatialCausal copies importing ``spatialcausal`` / ``spatialstats``) moved to ``tests/spatialcausal/`` and renamed
- README: spatial causal inference feature, extras, package layout and quick start
- Spatial causal tutorials (``Tutorials/notebooks/06_spatial_causal``, 06_00–06_08) run on ``pygis.spatialcausal`` / ``pygis.spatial_statistics`` and were re-executed; their helper modules ``countycausal.py`` and ``tutorial_style.py`` now sit beside the notebooks and read the cached county frame from ``data/data_saptial_causal/county_frame.gpkg``
- rsense install hints and docstrings point to ``pip install pygis[rsense-...]`` and ``import pygis.rsense as rs``; PyRSense metadata keys (``pyrsense:sensor``, ``pyrsense:index``, ...) are kept for file compatibility
- README: remote sensing feature, extras, package layout and quick start
- Geostatistics modules grouped into subpackages (``core``, ``kriging``, ``spacetime``, ``ml``, ``validation``, ``simulation``, ``mps``, ``gpu``, ``gslib``, ``io``, ``utils``). Former flat paths such as ``pygis.geostatistics.sgsim`` stay aliases of the new modules. Tests import the new paths; ``test_legacy_flat_module_aliases`` checks the aliases directly
- README package layout updated for the geostatistics subpackages
- ``PluriGaussianSimulator.simulate``: realization ``i`` no longer depends on ``nsim``. Each Gibbs chain draws its uniforms from its own per-realization seed (chains stay vectorized and share only the sweep order), so ``simulate(3, seed=s)`` equals the first 3 of ``simulate(50, seed=s)``. Conditional realizations for a given seed differ from earlier versions; run time is unchanged. The PGS tutorial (``05_07_07_pgs``) was re-executed
- ``Tutorials/notebooks/07_remote_sensing/07_03_radar/07_03_01_sentinel1_grd.ipynb`` and ``07_03_00_sar_fundamentals.ipynb``: moved off the standalone ``pyrsense`` package (not installed in this environment; the notebooks could not previously run) onto ``pygis.rsense``. The GRD notebook was also rewritten end to end to run the real backscatter workflow -- download, thermal-noise removal, calibration, multilooking, speckle filtering, DEM-based terrain flattening, geocoding, dB conversion, quality control -- on a real Sentinel-1A GRD scene and a real Copernicus DEM GLO-30 over the Cox's Bazar -- Teknaf coast of Bangladesh, in place of the earlier notebook's synthetic-simulator stand-ins for most of those steps; re-executed, with every reported number (NESZ, ENL, the terrain-flattening correction) taken from that real run

### Planned
- Point cloud data processing implementation
- GPU/CPU compute backend implementation
- Advanced analysis tools
- Method chaining interface
- Pre-built workflows

## [0.3.11] - 2026-10-19

### Added
- ``pygis.rsense.spectral_index``: sensor-aware spectral indices for MODIS, Landsat 8/9 (and TM), Sentinel-2, PlanetScope (4-band and SuperDove), WorldView-2/3, and NAIP. Filters the optical index registry by each sensor's bands; ``available`` / ``list_for_sensor`` / ``compute`` / ``describe``; USGS Landsat Collection 2 Surface Reflectance-derived set via ``USGS_LANDSAT_SR`` and ``usgs_landsat_sr`` (NDVI, EVI, SAVI, MSAVI, NDMI, NBR, NBR2, NDSI). Tests in ``tests/rsense/test_spectral_index.py``

### Changed
- Version bumped to 0.3.11
- README remote-sensing feature, package layout, and quick start cover ``spectral_index``

## [0.3.10] - 2026-10-18

### Added
- ``pygis.rsense.get_images``: ``search_images`` and ``get_images`` download the PyGIS Explorer catalogues (Sentinel, Landsat, USDA NAIP, NOAA coastal imagery, USGS 3DEP lidar) into a chosen folder. The area is a bounding box or a shapefile, GeoJSON, or GeoPackage. Date range, collection, bands, clip or full scene, Landsat product, Sentinel-1 mode, and Copernicus product type match the Explorer sidebar. The download engine ships as ``pygis.rsense._image_downloader`` so a wheel does not need the Explorer folder beside the install. Tests in ``tests/rsense/test_get_images.py``

### Changed
- Version bumped to 0.3.10
- ``rsense`` extra now includes ``requests``, ``pystac-client``, and ``tqdm`` so catalogue search works with ``pip install pygis[rsense]``. The Planetary Computer source still needs ``planetary-computer`` from ``pygis[rsense-access]``
- README remote-sensing feature, package layout, extras, and quick start cover ``get_images``

## [0.3.9] - 2026-10-13

### Changed
- Geostatistics modules flattened from ``pygis.geostatistics.pygstat.*`` into ``pygis.geostatistics.*``; ``pygis.geostatistics.__init__`` rebuilt for the flat layout (``pygis.geostatistics.pygstat`` kept as an attribute alias)
- Top-level ``pygis`` exposes every subpackage lazily (``pygis.raster``, ``pygis.spatial_analysis``, ``pygis.geostatistics``, …)
- Terrain / hydrology export tests now target ``pygis.spatial_analysis``
- README package layout and build instructions updated; version bumped to 0.3.9

### Added
- GPU-accelerated turning bands simulation in ``pygis.geostatistics.tbsim_gpu``: ``TBSimulator(use_gpu=...)`` evaluates the line processes and the kriging conditioning on the GPU with host-side random numbers; identical to the CPU engine; 25-105x faster, including for a single realization; CPU fallback
- GPU-accelerated direct sequential (co-)simulation in ``pygis.geostatistics.dss_gpu``: ``DirectSequentialSimulator(use_gpu=...)`` (Soares: one thread per realization; lookup: one block per realization with a parallel FP32 + exact FP64 window search); identical to the CPU engine for the same seed; up to ~110x (Soares) / ~37x (lookup) faster at 1,024 realizations; memory batching and CPU fallback
- GPU-accelerated sequential indicator (co-)simulation in ``pygis.geostatistics.sis_gpu``: ``IndicatorCoSimulator(use_gpu=...)`` runs one CUDA thread per realization (per-class indicator kriging with Markov–Bayes soft data, categorical draws, or ``ordrel`` + ``beyond`` tails for continuous variables); identical to the CPU engine for the same seed; ~110x faster at 1,024 realizations; memory batching and CPU fallback
- GPU-accelerated sequential Gaussian (co-)simulation in ``pygis.geostatistics.sgs_gpu``: ``GaussianCoSimulator(use_gpu=...)`` and ``sgsim_gpu`` run one CUDA thread per realization (in-kernel step loop, memory-sized batches); bit-identical to the CPU engine for the same seed; simple / ordinary kriging and collocated MM1 secondary; warns and falls back to the CPU when the GPU is unavailable or out of memory
- ``GaussianCoSimulator(kriging_type='ordinary')`` on the CPU
- GPU-accelerated multiple-point simulation: ``use_gpu`` on ``DirectSampling`` and ``SNESIM`` runs all realizations in lockstep with one fused CUDA kernel per step (``pygis.geostatistics.mps_gpu``: ``ds_simulate_batched``, ``snesim_simulate_batched``, with a vectorized NumPy reference engine); up to 19x (Direct Sampling) and 334x (SNESIM) faster at 128 realizations; GPU SNESIM is identical to the CPU engine, GPU Direct Sampling statistically equivalent; co-simulation supported; tests in ``tests/test_geostatistics.py`` (GPU tests skip without CUDA)
- Multiple-point co-simulation: ``DirectSampling`` takes ``secondary_ti`` / ``secondary_weight`` / ``n_secondary`` and an exhaustive ``secondary`` in ``fit`` (multivariate data events, Mariethoz et al. 2010); ``SNESIM.fit`` takes ``soft`` probabilities merged by the tau model (``tau_model``, ``tau`` exponents), with the soft mean as default target proportions; ``dssim`` / ``snesim`` pass them through; tests in ``tests/test_geostatistics.py``
- Simulated annealing simulation in ``pygis.geostatistics.annealing``: ``SimulatedAnnealing`` / ``sasim`` (swap-based annealing with exact target histogram, variogram / indicator-variogram / secondary-correlation objective components normalised as in GSLIB sasim, incremental objective updates, Metropolis acceptance with geometric cooling, hard data fixed); tests in ``tests/test_geostatistics.py``
- Direct sequential simulation in ``pygis.geostatistics.direct_sequential``: ``DirectSequentialSimulator`` / ``dssim_direct`` in original units (lookup-table local distributions after Oz et al. 2003, default, or Soares 2001 with a skewness warning), declustering weights, collocated MM1 co-simulation with an exhaustive secondary, and ``GlobalDistribution``; tests in ``tests/test_geostatistics.py``
- Shared nearest-informed-neighbour search ``sequential_neighbors`` in ``pygis.geostatistics._conditioning`` (used by ``sicosim``, ``sgcosim``, ``direct_sequential``)
- Sequential Gaussian co-simulation in ``pygis.geostatistics.sgcosim``: ``GaussianCoSimulator`` / ``sgcosim`` (collocated simple cokriging under Markov Model 1, or multi-collocated MM1 cokriging via ``secondary_neighbors``, with the ``mm1_nugget_ratio_`` diagnostic to choose), ``sgcosim_hierarchical`` (joint simulation of two variables known only at the data); tests in ``tests/test_geostatistics.py``
- Sequential indicator co-simulation in ``pygis.geostatistics.sicosim``: ``IndicatorCoSimulator`` / ``sicosim`` (collocated indicator cokriging with Markov–Bayes soft data, categorical classes or continuous thresholds, plain SIS without secondary), ``SoftCalibration`` (secondary-to-soft-indicator calibration table and ``B`` per class); tests in ``tests/test_geostatistics.py``
- SNESIM in ``pygis.geostatistics.snesim``: ``SNESIM`` / ``snesim`` (multigrid, list-based pattern tables with per-facies centre counts, data-event reduction to ``min_replicates``, servosystem toward target proportions, hard-data relocation on coarse grids); tests in ``tests/test_geostatistics.py``
- Multiple-point statistics simulation in ``pygis.geostatistics.mps``: ``DirectSampling`` / ``dssim`` (Direct Sampling for categorical and continuous training images, vectorized NaN-padded scanning, post-processing passes, hard data on nearest nodes), and the diagnostics ``pattern_histogram``, ``pattern_histogram_distance``, ``connectivity``, plus ``channel_training_image``; tests in ``tests/test_geostatistics.py``
- Plurigaussian simulation in ``pygis.geostatistics.pgs``: ``TruncationRule`` (``truncated_gaussian``, ``hierarchical``, ``from_flag``, proportions for correlated latent fields), ``PluriGaussianSimulator`` / ``pgsim`` (one or two latent fields with correlation ``rho`` simulated by ``fftsim`` or ``tbsim``, Gibbs sampling of latent values at categorical data), ``category_probabilities``, ``category_proportions``; tests in ``tests/test_geostatistics.py``
- ``TBSimulator.simulate`` / ``FFTSimulator.simulate`` accept ``data_values`` to re-condition on new values at the fitted data locations; ``conditioning_xy_`` gives their order
- FFT spectral simulation in ``pygis.geostatistics.fftsim``: ``fftsim``, ``fftsim_unconditional``, and ``FFTSimulator`` (circulant embedding with automatic padding and a clipped-spectrum diagnostic, two realizations per FFT, lattice inference, nearest-node data relocation, simple / ordinary kriging conditioning, ``simulate_lattice``); tests in ``tests/test_geostatistics.py``
- Shared global dual-kriging conditioning in ``pygis.geostatistics._conditioning`` (used by ``tbsim`` and ``fftsim``)
- Turning bands simulation in ``pygis.geostatistics.tbsim``: ``tbsim`` (conditioned by simple or ordinary dual kriging), ``tbsim_unconditional``, and ``TBSimulator``; sawtooth-dilution lines for spherical and exact spectral lines for exponential / Gaussian / Matérn, 3-D lines with random rotation per realization, nugget as white noise, normal-score transform and ``zmin`` / ``zmax``; tests in ``tests/test_geostatistics.py``
- LU (Cholesky) decomposition simulation in ``pygis.geostatistics.lusim``: ``lusim`` (conditional by simple or ordinary kriging, normal-score transform, ``zmin`` / ``zmax`` tail limits), ``lusim_unconditional``, and ``LUSimulator`` (factorize once, simulate many); tests in ``tests/test_geostatistics.py``
- ``tests/test_spatial_statistics.py``: known-answer tests for every ``pygis.spatial_statistics`` subpackage (core, weights, ESDA, regression, point patterns, interpolation, clustering, sampling, Bayesian smoothing, GW models, datasets, viz)

### Fixed
- Importing ``pygis.geostatistics.regression_kriging_tf`` force-loaded the CUDA toolkit's cuBLAS even when another cuBLAS 13 (e.g. PyTorch's pip copy) was already mapped, putting two cuBLAS builds in the process; the next CuPy cuBLAS call (any GPU matrix product) segfaulted. The preload is now skipped when a cuBLAS 13 is already loaded
- ``DirectSequentialSimulator(method='lookup')`` now consumes one normal deviate per simulated node even when the kriging variance is zero, so the random stream no longer depends on the simulation (needed for GPU/CPU equivalence; changes CPU results only at zero-variance nodes)
- ``GWGLM`` (Poisson / binomial) IRLS divided the working weights by the variance instead of multiplying, so local coefficients were not the MLE and small bandwidths could diverge; fits now match a global GLM exactly at infinite bandwidth
- ``import pygis.geostatistics`` and ``import pygis.analysis`` failed from the source tree after the flatten (stale ``pygstat`` import)

## [0.3.7] - 2026-10-07

### Added
- Geostatistics package under ``pygis.geostatistics`` (``pygstat``): variograms, ordinary / simple / universal / indicator / disjunctive / factorial / soft / Poisson / cokriging, EBK, regression kriging, space-time kriging, sequential simulation (SGSIM / SISIM), KT3D, GSLIB helpers, and validation utilities
- Optional extras ``pygis[geostatistics]`` plus backend extras (``geostatistics-gpu``, ``geostatistics-h2o``, ``geostatistics-pytorch``, ``geostatistics-tensorflow``, ``geostatistics-gam``, ``geostatistics-optuna``, ``geostatistics-deep``, ``geostatistics-plot``)
- ``src/`` mapped to the ``pygis`` import namespace in ``setup.py`` / ``pyproject.toml`` so nested geostatistics modules install correctly
- ``pygis.analysis.geostatistics`` alias
- Tests in ``tests/test_geostatistics.py``

### Changed
- README updated for geostatistics workflow, package layout, and install extras
- Package description and keywords mention geostatistics / kriging / variogram
- Version bumped to 0.3.7

## [0.3.6] - 2026-10-07

### Added
- Spatial network analysis in ``pygis.spatial_analysis.network_analysis`` for the ``03_06_spatial_network`` tutorials: geographic embedding (haversine, Kansky α/β/γ, circuity), OSMnx routing / OD matrices / isochrones, libpysal spatial graphs and Moran's I, 2SFCA / Lorenz–Gini accessibility, OD gravity and dominant-flow hierarchy, metapopulation SIR, and simple / complex contagion seeding
- Optional extra ``pygis[network]`` (networkx, osmnx, libpysal, scikit-learn, geopandas)
- Tests in ``tests/test_network_analysis.py``

## [0.3.3] - 2026-10-05

### Added
- Surface hydrological analysis in ``pygis.spatial_analysis.surface_hydrology``: depression breaching and filling, D8 / Rho8 / D-infinity / FD8 / MD-infinity / Quinn / minimal-dispersion flow routing, D8 and D-infinity flow accumulation, stream extraction and segmentation, catchment and watershed delineation, TWI (``wetness_index``), elevation above stream, downslope distance to stream, stochastic depression analysis, hydrologic connectivity, and impoundment size index
- Tests in ``tests/test_surface_hydrology.py``

### Changed
- README updated for surface hydrology workflow and package layout
- Version bumped to 0.3.3

## [0.3.2] - 2026-10-05

### Added
- Terrain analysis / geomorphometry in ``pygis.spatial_analysis.terrain_analysis``: ``slope``, ``aspect``, ``hillshade``, ``tri``, ``tpi``, ``twi``, ``roughness``, curvature suite (``mean``, ``tangential``, ``profile``, ``plan``, ``gaussian``), ``landform_classification``, ``multiscale_analysis``, ``viewshed``, and ``contours``
- GDAL ``gdaldem``-backed products for first-order derivatives and roughness indices; NumPy implementations for curvature, TWI, landform, multiscale, and viewshed
- Tests in ``tests/test_terrain_analysis.py``

### Changed
- README updated for terrain analysis
- Package description and keywords mention terrain / geomorphometry
- Version bumped to 0.3.2

## [0.3.1] - 2026-10-05

### Added
- Histogram and classInt-style raster classification in ``basic_raster_operation``: ``raster_histogram``, ``class_intervals``, and ``reclassify_by``
- Styles: ``fixed``, ``sd``, ``equal``, ``pretty``, ``quantile``, ``kmeans``, ``hclust``, ``bclust``, ``fisher``, and ``jenks``
- Tests for histogram / classification helpers
- Tutorial notebook ``Tutorials/notebooks/02_raster/02_03_basic_raster_operation.ipynb``

### Changed
- README updated for histogram / quantile-based reclassification
- Version bumped to 0.3.1

## [0.3.0] - 2026-10-05

### Added
- Basic raster operations in ``pygis.raster.basic_raster_operation``: ``create``, ``edit``, ``convert``, ``resize``, ``unscale``, ``fill_nodata``, ``neighbors`` / ``focal_statistics``, ``calc`` / ``raster_algebra``, ``blend``, ``compare``, ``clip``, ``reclassify``, ``resample``, ``aggregate``, ``disaggregate``, ``mosaic``, ``stack``, ``summarize_raster``, ``zonal_stats``, ``polygonize``, ``proximity``, ``color_map``, and ``contour``
- Tests in ``tests/test_basic_raster_operation.py``

### Changed
- Package description and README updated for raster I/O and basic raster operations
- Version bumped to 0.3.0

## [0.2.0] - 2026-10-05

### Added
- Vector analysis tools in ``pygis.vector.vector_analysis``: field statistics, climb along line, point-in-polygon counts, DBSCAN / ST-DBSCAN / k-means clustering, distance matrices, nearest-hub distances, hub lines, unique values, mean coordinates, nearest-neighbour analysis, overlap analysis, shortest lines, statistics by categories, and sum line lengths
- Attribute manipulation tools: ``filter_features``, ``select_fields``, ``rename_fields``, ``calculate_fields``, ``aggregate_attributes``, ``sort_features``, ``join_attributes``, ``transform_field``, ``string_field``, ``update_where``, ``reshape_attributes``, and ``drop_geometry``
- Tutorial notebook ``Tutorials/notebooks/01_vector/01_07_vector_analysis.ipynb``
- Tests for vector analysis and attribute manipulation

### Changed
- Package description and README updated for vector analysis and attribute workflows
- Version bumped to 0.2.0

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of PyGIS
- Core spatial data abstractions
- CRS management, vector I/O, geoprocessing, and geometry tools
- Spatial data visualization in ``pygis.viz``
- WhiteboxTools integration framework
- Basic configuration and API structure
