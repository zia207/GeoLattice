# LiDAR Data Processing and Analysis in Python

A reference outline for the workflow the eight notebooks in this folder
(`07_06_01` to `07_06_08`, listed in Section 14) walk through hands-on, using
`geolattice.rsense.lidar`.

---

## 1. Foundations

**1.1 How LiDAR works**
- Time-of-flight ranging: `R = c·t/2`; combined with GNSS position + IMU orientation → georeferenced (x, y, z) points
- Discrete-return vs. full-waveform systems
- Multiple returns per pulse (first/last/intermediate) — canopy penetration reveals both canopy top and ground
- Point density (points/m²), footprint size (beam divergence × range), intensity (echo strength, wavelength-dependent)
- Platform types: airborne (ALS), terrestrial (TLS), mobile, UAV, spaceborne (GEDI, ICESat-2)

**1.2 File formats and standards**
- **LAS/LAZ**: ASPRS binary standard; LAZ = lossless compression (~7-10x)
- **COPC** (Cloud-Optimized Point Cloud): octree-organized LAZ, queryable by region/level of detail over HTTP range requests
- **EPT** (Entwine Point Tiles): tiled directory format; backbone of the USGS 3DEP AWS archive
- **E57, PLY, PCD**: other common exchange formats
- ASPRS classification codes (ground, vegetation tiers, building, water, noise, etc.)

---

## 2. Reading, I/O, and Quality Control

**2.1 Reading point clouds**
- `laspy` — pure-Python LAS/LAZ/COPC reader; `laspy.CopcReader` for windowed/level-of-detail streaming
- `PDAL` (Python bindings) — pipeline-based I/O for nearly any format, including EPT
- `geolattice.rsense.lidar.io`: `read_las(path, bounds=, resolution=, device=)`, `read_pdal(source, pipeline=)`, `usgs_3dep_ept(bbox)`, `write_las(pc, path)` — wraps laspy/PDAL behind one consistent `PointCloud` container
- Columnar point representation (`PointCloud`: aligned NumPy arrays for x/y/z + attributes) vs. row-oriented — columnar is what nearly every downstream numeric step wants

**2.2 CRS and unit checks**
- Horizontal/vertical CRS from the LAS header (`pyproj.CRS.from_user_input`)
- Feet vs. metres is the single most common silent error source — always verify `axis_info[0].unit_name`
- Reprojecting to a metric, locally-appropriate UTM zone (`geolattice.rsense.core.utm_epsg(lon, lat)`) before any area/height computation

**2.3 Inspection and QA**
- Header stats: point count, bounds, point density per m²
- Classification histogram, returns-per-pulse cross-tab (`number_of_returns` vs `return_number`)
- Cross-sections (thin profile slices, x/z or y/z scatter, coloured by class) — the fastest way to visually sanity-check a dataset
- Comparing producer-supplied classification against your own filters (confusion matrix, kappa)

**2.4 Cleaning**
- Dropping noise classes (low/high noise, typically codes 7, 18)
- Cropping to AOI, filtering by class, filtering by return number (first returns only)
- Voxel downsampling for uniform density / faster downstream processing
- Elevation-band / statistical outlier filters

---

## 3. Ground Classification (Terrain Filtering)

This is the hinge point every other product depends on.

- **SMRF** (Simple Morphological Filter, Pingel et al. 2013): progressive morphological opening of a minimum-elevation surface; points close to the resulting provisional DTM are ground. Tunable: `cell`, `slope`, `window` (must exceed the largest non-ground object, e.g. widest roof).
- **CSF** (Cloth Simulation Filter, Zhang et al. 2016): invert the cloud, drop a simulated cloth under gravity with internal springs (`rigidness`); points near the settled cloth are ground.
- **PMF, ATIN, segment-based, and deep-learning filters** (e.g. PDAL's `smrf`/`pmf` filters, or learned classifiers) for harder terrain.
- `geolattice.rsense.lidar.ground`: `smrf(pc, cell=, slope=, window=)`, `csf(pc, resolution=, rigidness=)`, `classify_ground(pc, method=)` — native NumPy implementations, with an automatic fallback to the official compiled `CSF` package if installed.
- **Validation**: overall accuracy, kappa, ground recall/precision against reference labels; always inspect disagreements along steep banks and under dense low vegetation, the two places every ground filter struggles.

---

## 4. Surface Models: DEM, DSM, CHM

- **DEM/DTM**: rasterize ground-classified points (mean/min per cell), then gap-fill (nearest, IDW, or Delaunay/TIN linear) where no ground points fall in a cell (common under dense canopy or water).
- **DSM**: rasterize first returns at max elevation per cell — the visible surface, canopy included.
- **CHM** (Canopy Height Model) = DSM minus DEM, typically pit-filled (median filter removing single-pixel inter-crown gaps).
- Gridding via scatter-reduction (bin points to cells with min/max/mean/count/sum) — vectorizable and GPU-portable.
- `geolattice.rsense.lidar.surfaces`: `rasterize(pc, res, value=, stat=)`, `fill_gaps(grid, method=)`, `dem(pc, res)`, `dsm(pc, res)`, `chm(dsm_or_pc, dem)`, `normalize_heights(pc, dtm)` (adds a per-point height-above-ground field — the basis of nearly everything in Section 6).

---

## 5. Terrain Derivatives

Computed from the DEM, feeding hydrology/soils/habitat models:

| Derivative | What it measures |
|---|---|
| Slope / Aspect | Steepness and compass direction of steepest descent (Horn 1981 gradient) |
| Curvature | Laplacian; ridges (convex) vs. hollows (concave) |
| TPI (Topographic Position Index) | Elevation relative to a local neighbourhood mean — ridge vs. valley position |
| TRI / Roughness | Local elevation variability |
| Hillshade / Illumination | Analytical shading / cosine of solar incidence |
| TWI (Topographic Wetness Index) | `ln(specific catchment area / tan(slope))` — needs flow accumulation |
| D8 flow direction/accumulation, depression filling | Hydrological routing (priority-flood algorithm, Barnes et al. 2014) |
| Geomorphons | Landform classification (ridge/valley/slope/pit/...) from multi-directional line-of-sight angles (Jasiewicz & Stepinski 2013) |

`geolattice.rsense.lidar.terrain` implements all of the above natively (local derivatives GPU-portable; hydrology is sequential/CPU). WhiteboxTools is a common alternative/complement for very large DEMs.

---

## 6. Vegetation / Forest Structure Analysis

**6.1 Height normalization**
- Subtract the (bilinear-interpolated) DEM from every point's z → per-point height above ground — required before any canopy metric makes sense.

**6.2 Area-based metrics** (the standard predictors for forest inventory/biomass models)
- Per grid cell (often matched to field plot size, e.g. 20 m): max/mean/std/percentile heights (`p25, p50, p75, p95, ...`), canopy cover (fraction of first returns above a height threshold), point density, coefficient of variation.
- `geolattice.rsense.lidar.forest.grid_metrics(pc, res=, metrics=, cover_height=)`.

**6.3 Vertical structure**
- Histograms / profiles of normalized height by class (forest vs. open), revealing canopy layering.

**6.4 Individual tree detection**
- Treetops as local maxima of the CHM, with a height-dependent search window (taller trees need wider minimum separation) or a fixed window; local maximum filter plus plateau-merging.
- `detect_trees(chm, min_height=, window=)`.

**6.5 Crown segmentation**
- Marker-controlled watershed from detected treetops over the (inverted) CHM — `segment_crowns(chm, treetops, min_height=)` returns a labeled raster plus a GeoDataFrame of crown polygons with height, area, equivalent diameter.
- Allometric relationships (height vs. crown diameter) as a sanity check / input to biomass models.
- `assign_points_to_crowns(pc, labels)` — tag every point with its tree id for per-tree point-cloud analysis (e.g. species classification from intensity/structure).

---

## 7. Infrastructure / Building Extraction

- **Label-based**: rasterize building-classified points, morphological closing to remove gaps, vectorize to polygons (`rasterio.features.shapes`), filter slivers by area, sample the normalized DSM for height.
- **Rule-based** (no labels needed): tall surfaces (normalized DSM above a threshold) with a low multiple-return fraction — roofs are solid and mostly single-return, while trees of similar height usually produce several returns per pulse. Morphological opening cleans up the result.
- Validation via Intersection-over-Union against reference labels; watch for systematic failure modes (a very large flat roof gets absorbed into the ground surface if the ground filter's window is too small).

---

## 8. Point Cloud Classification (beyond geometric filters)

- Feature engineering per point: local height, planarity/linearity/scatter (from local covariance eigenvalues), intensity, return structure, local density.
- Classical ML: Random Forest / gradient boosting on engineered features (`scikit-learn`).
- Deep learning directly on point clouds: PointNet/PointNet++, KPConv, RandLA-Net (PyTorch / `torch-points3d` / `Open3D-ML`); or voxelized/rasterized CNNs.
- Semantic segmentation of urban scenes (ground/building/vegetation/power line/vehicle, etc.) — common in MLS/TLS workflows.

---

## 9. Spaceborne LiDAR Integration

- **GEDI** (ISS, full-waveform, ~25 m footprints, 60 m track spacing): L2A relative heights (RHn — height below which n% of returned energy lies; RH98 ≈ canopy top), L4A aboveground biomass density.
- **ICESat-2 / ATLAS** (photon-counting, 17 m footprints, 6 beams): ATL08 100 m land segments → terrain height, canopy height, canopy cover.
- Reading official HDF5 layouts: `geolattice.rsense.lidar.spaceborne.read_gedi_l2a` / `read_gedi_l4a` / `read_atl08`, with quality filtering (`quality_flag`, `degrade_flag`, `sensitivity` thresholds).
- Cross-validation against airborne LiDAR: sample the same footprint geometry from an ALS CHM, account for geolocation error (~10 m for GEDI) and footprint-size mismatch before trusting a direct comparison.
- Gridding footprints to wall-to-wall maps (`footprints_to_raster`) or using them to train models against optical/SAR predictors for continuous coverage where ALS isn't available.

---

## 10. Visualization

- **2D**: raster rendering (matplotlib / `rioxarray`), choropleth of grid metrics, categorical classification maps, cross-section/profile plots.
- **3D**: `pyvista` or `Open3D` for interactive point-cloud rendering; matplotlib 3D scatter (subsampled) as a dependency-light fallback.
- Hillshades for quick visual terrain QA.

---

## 11. Scaling and Performance

- **Streaming large files**: COPC windowed/level-of-detail reads instead of loading a whole file; EPT for archive-scale data (USGS 3DEP).
- **Tiling**: process large AOIs tile-by-tile with overlap buffers to avoid edge artifacts in rasterization/filtering.
- **Backend dispatch**: vectorized NumPy operations that can run on GPU (CuPy) with no code changes — relevant for rasterization/scatter-reduction and the local terrain derivatives.
- **PDAL pipelines** for server-side, chainable, memory-efficient filtering before data ever reaches Python objects.

---

## 12. A Typical End-to-End Workflow

```
1. Acquire/download  ->  LAS/LAZ or COPC/EPT
2. Read + inspect    ->  CRS/unit check, class histogram, density, cross-section
3. Clean             ->  drop noise, crop to AOI, (optionally) voxel downsample
4. Classify ground   ->  SMRF or CSF, validate against reference labels
5. Build surfaces    ->  DEM (ground, gap-filled) -> DSM (first returns) -> CHM (DSM-DEM)
6. Terrain analysis  ->  slope/aspect/curvature/TPI/TWI/geomorphons from the DEM
7. Normalize heights ->  per-point height above ground
8a. Forest path      ->  grid_metrics -> detect_trees -> segment_crowns -> allometry
8b. Building path    ->  nDSM -> footprint mask (label or rule-based) -> polygons -> heights
9. Cross-validate    ->  vs. spaceborne LiDAR (GEDI/ICESat-2) or field plots
10. Export           ->  write LAS/LAZ, GeoTIFF rasters, GeoPackage/GeoJSON vectors
```

---

## 13. Python Tooling Summary

| Need | Library |
|---|---|
| LAS/LAZ/COPC I/O | `laspy` |
| General point-cloud pipelines, EPT | `PDAL` (Python bindings) |
| Ground filtering (compiled, exact) | `CSF` (official package) |
| Rasters, CRS, zonal ops | `rasterio`, `rioxarray`, `pyproj` |
| Vector footprints/crowns | `geopandas`, `shapely` |
| Image-processing primitives (watershed, morphology) | `scikit-image`, `scipy.ndimage` |
| 3D visualization | `pyvista`, `Open3D` |
| ML/DL point classification | `scikit-learn`, `torch` / `Open3D-ML` / `torch-points3d` |
| HDF5 (GEDI/ICESat-2) | `h5py` |
| One consistent, GPU-portable API across all of the above stages | `geolattice.rsense.lidar` (`io`, `ground`, `surfaces`, `terrain`, `forest`, `spaceborne`) |

---

## 14. Where This Folder's Notebooks Fit

| Notebook | Covers | Data |
|---|---|---|
| `07_06_01_lidar_fundamentals.ipynb` | Sections 1 and 2.1 to 2.2: how LiDAR works (ranging, georeferencing, wavelengths, footprints, multiple returns, full waveform, single photon), acquisition (platforms, scanners, flight planning, QL0 to QL3), formats and classification codes, applications and data sources; reading LAS/LAZ/COPC, fixing feet, degrees and vertical datums, classes/returns/density/intensity, cross-sections and 3D views, COPC windows and levels of detail, writing LAZ/PLY, USGS 3DEP lookup | Madison COPC, UB 2008, Ithaca 2020 |
| `07_06_02_lidar_processing.ipynb` | Sections 2.3 to 4: density and swath-overlap QC, outlier filters, selection and thinning, SMRF/CSF/PMF ground filters with validation and tuning, DEM/DSM/CHM, gap filling and cell size, vertical accuracy with checkpoints, export | Madison, UB 2008 |
| `07_06_03_lidar_terrain_hydrology.ipynb` | Section 5: seamless DEM from tiles, hillshades, slope/aspect/curvature/TPI/TRI/roughness, illumination, scale effects, geomorphons, depression filling and embankments, D8 flow, streams, TWI, HAND and flood extent, profiles | Ithaca 2020 (4 tiles) |
| `07_06_04_lidar_forest_structure.ipynb` | Section 6: height normalisation, CHM and cover, area-based and structure metrics, vertical profiles and LAD, plot metrics, treetop detection, crown segmentation, allometry, point-based tree segmentation, per-tree point clouds, effect of point density | Madison, Ithaca 2020 |
| `07_06_05_lidar_urban_buildings.ipynb` | Section 7: ground filters and large roofs, eigenvalue features, geometric and raster building classification with validation, footprints and heights, RANSAC roof planes, solar potential, wire detection, label-free buildings on a sparse survey, LoD1 block model | Autzen, UB 2008 |
| `07_06_06_lidar_point_classification.ipynb` | Section 8: labels and spatial split, multi-scale point features, random forest and XGBoost, random versus spatial holdout, feature importance, PointNet and PointNet++ on the GPU, error analysis | Autzen |
| `07_06_07_spaceborne_lidar.ipynb` | Section 9: GEDI and ICESat-2 instruments and products, CMR granule search, reading GEDI L2A and ATL08 with quality filters, RH profiles, comparison with ALS, geolocation-error and footprint-size experiment, gridding footprints | Simulated GEDI L2A and ATL08 files, Madison |
| `07_06_08_lidar_large_area_workflow.ipynb` | Sections 10 to 12: tile index from headers, memory and field selection, chunked streaming, parallel per-tile products and mosaics, buffered tiling and edge effects, GPU terrain derivatives, writing tiles, the end-to-end workflow checklist | Ithaca 2020 (all 10 tiles) |

Not yet covered in this folder's notebooks: terrestrial/mobile LiDAR (TLS/MLS)-specific
workflows (stem detection, registration and change detection with `detect_stems`, `icp` and
`m3c2` are available in `geolattice.rsense.lidar`).
