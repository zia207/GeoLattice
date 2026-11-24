# Hyperspectral Imaging with GeoLattice: Notebook Outline

Chapter folder: `Tutorials/notebooks/08_remote_sensing/08_05_HyperSpectral/`
Library: `geolattice.rsense.hyperspectral` (cube, io, preprocess, reduce, indices, library, match,
endmembers, unmixing, detect, traits, viz) and `geolattice.rsense.dl.hyperspectral`.

Every notebook follows the house style: banner cell, `#` title with a one paragraph intro,
`## Overview` (theory), `## ... with GeoLattice`, Learning objectives, Data, Contents, Setup, the
working sections, and Summary and exercises. Section headings carry no numbers.

## Data inventory

| Dataset | Path (under `data/`) | Sensor and level | Size | Main use |
|---|---|---|---|---|
| Planet Tanager core imagery, 9 scenes | `Tanger/<category>/<scene_id>/` | Tanager-1, 426 bands, ~376 to 2500 nm, ~5.5 nm FWHM; ortho TOA radiance and surface reflectance (HDF-EOS5), beta UDM, visual GeoTIFF, STAC JSON | 15 GB | Main teaching data |
| Scene index | `Tanger/scenes.json` | Category, datetime, bbox, STAC link per scene | small | Scene picker |
| Tanager basic (swath) radiance | `20250224_145149_32_4001_basic_radiance_hdf5.h5` (+ basic UDM) | Unorthorectified agriculture scene | 433 MB | Swath vs ortho, GLT ideas |
| Tanager basic radiance + geolocation array | `20250516_181930_35_4001_*` | Second swath scene with per pixel lat/lon | 566 MB | Orthorectification |
| OHS (Zhuhai Orbita OHS-2) | `10m_OH2_Hyperspectral_Sample/Spanish-OHS/HAZ2_..._CCD1/` | 32 single band L1 GeoTIFFs (B1 to B32, VNIR ~400 to 1000 nm), RPC files | 1.6 GB | Band stacking, RPC, destriping |
| Pavia Centre | `Pavia/pavia.npy` | ROSIS, 1096 x 715 x 102, uint16 | 153 MB | Benchmark for DR, unmixing, DL |
| HACARUS lab HSI | `archive/data/{coffee,rice,sugar_salt_flour,sugar_salt_flour_contamination,yatsuhashi}` | NIR camera, 96 PNG bands 1293 to 2215 nm, 192 x 256 | 11 MB | Small, fast, lab spectroscopy, contamination detection |

Tanager categories: `agriculture`, `natural-lands`, `urban`, `coastal-water-bodies`,
`energy-mining`, `GHG-plumes` (with `ortho_ql_ch4.tif` and `ql_ch4_json.geojson` quick look
CH4 products), `fire`, `snow-ice`, `ROCX2025` (cal/val campaign).

Inside each Tanager HDF5: `HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance` (or
`toa_radiance`) with attributes `wavelengths`, `fwhm`, `good_wavelengths`, `_FillValue`
(-9999), plus `beta_cloud_mask`, `beta_cirrus_mask`, `nodata_pixels`, `aerosol_optical_depth`,
`column_water_vapour`, sun and sensor geometry, and `surface_reflectance_uncertainty`.

## Notebook list

### 09_05_00_introduction_to_imaging_spectroscopy.ipynb
Introduction to Imaging Spectroscopy
- Overview: imaging spectroscopy vs multispectral, spectral resolution and sampling, FWHM,
  SNR, the data cube, radiance vs reflectance, atmospheric absorption windows.
- Sensors and missions: Tanager-1 (Planet / Carbon Mapper), EMIT, EnMAP, PRISMA, AVIRIS-NG
  and AVIRIS-3, OHS, PACE OCI, CHIME and SBG (future). Laboratory and UAV cameras.
- Spectral signatures of vegetation, soil, minerals, water, snow, urban materials.
- The GeoLattice cube convention: `xarray.DataArray` `(band, y, x)` with `wavelength`, `fwhm`,
  `good` coordinates; `as_cube`, `wavelengths`, `band_at`, `sel_range`.
- Tour of the chapter data: read `scenes.json`, map scene footprints, show thumbnails.
- Quick hands on with the HACARUS sugar/salt/flour cube (small and fast).

### 09_05_01_reading_and_exploring_hyperspectral_data.ipynb
Reading and Exploring Hyperspectral Data
- Tanager HDF5: open surface reflectance and radiance, attach wavelengths/FWHM/good bands,
  set fill to NaN, attach CRS and transform from `StructMetadata.0` or the visual GeoTIFF.
- OHS: stack 32 band GeoTIFFs into one cube with band centre wavelengths; inspect RPCs.
- Pavia: `np.load` then `as_cube` with ROSIS wavelengths (430 to 860 nm).
- HACARUS: read 96 PNGs, parse wavelength from the file name, build the cube.
- Exploring: `band_stats`, `plot_band_stats`, `mean_spectrum`, `spectra_at`, `plot_spectra`,
  `rgb` / `plot_rgb` with `true_color`, `color_infrared`, `swir`, `geology` composites,
  `spectral_viewer`, `wavelength_slider`.
- Subsetting large scenes: `sel_range`, spatial windows, `subset_file`; writing with
  `write_envi`.
- Swath vs ortho: the basic radiance product and the geolocation array, `glt_orthorectify`
  (or a lat/lon regrid) on the 20250516 scene.

### 09_05_02_preprocessing_hyperspectral_cubes.ipynb
Preprocessing Hyperspectral Cubes
- Bad bands: `good_wavelengths`, `bad_bands`, `drop_bad_bands` (water vapour 1400, 1900 nm).
- Quality masks: Tanager beta cloud, cirrus, nodata and UDM; `quality_mask`.
- Noise: `estimate_noise` (shift and residual), per band SNR, `surface_reflectance_uncertainty`.
- Striping in pushbroom data: `destripe` on OHS and Tanager basic radiance.
- Radiance to reflectance: `flat_field`, `iar_reflectance`, `empirical_line`; compare with
  the Tanager SR product (ROCX2025 cal/val scene as reference).
- Spectral smoothing and transforms: `smooth` (Savitzky Golay, moving, Gaussian),
  `derivative`, `snv`, `normalize`, `continuum_removal`, `band_depth`.
- Spectral resampling: `resample_spectral`, `resample_to_sensor` (Tanager to Sentinel-2,
  Landsat, PlanetScope) to simulate multispectral data.

### 09_05_03_dimensionality_reduction_and_band_selection.ipynb
Dimensionality Reduction and Band Selection
- Overview: Hughes phenomenon, band correlation, signal vs noise subspace.
- `pca`, `mnf`, `mnf_denoise`, `ica`; explained variance and eigenimages.
- Intrinsic dimension: `hysime`, `virtual_dimensionality`.
- Band selection: `select_bands` (unsupervised and supervised).
- Data: Pavia Centre, Tanager urban.

### 09_05_04_narrowband_indices_and_vegetation_analysis.ipynb
Narrowband Indices and Vegetation Analysis
- Overview: pigments, red edge, leaf water, dry matter, nitrogen.
- `list_indices`, `available_indices`, `index` (e.g. PRI, CRI, ARI, mNDVI705, NDWI, NDII,
  CAI), `register_index` for a custom index, `red_edge_position`.
- Data: Tanager agriculture (Argentina/Bolivia, Feb 2025) and natural-lands (Bosnia).
- Field patterns, histograms, comparison with broadband NDVI after `resample_to_sensor`.

### 09_05_05_biophysical_and_biochemical_trait_retrieval.ipynb
Biophysical and Biochemical Trait Retrieval
- Overview: radiative transfer (PROSPECT, SAIL), empirical vs hybrid retrieval.
- `prosail_spectrum`, `prosail_lut` on Tanager bands.
- `hybrid_retrieval` for LAI and Cab with GPR (with uncertainty) and RF.
- `fit_plsr`, `vip_scores`, `fit_regressor`, `TraitModel.predict_map` trained on a
  simulated LUT (no field samples ship with the data; note how to swap in real plots).
- Data: Tanager agriculture.

### 09_05_06_spectral_libraries_and_spectral_matching.ipynb
Spectral Libraries and Spectral Matching
- `SpectralLibrary`, `synthetic_library`, `from_image`, `merge_libraries`, `resample`.
- Reference libraries: `read_usgs`, `read_ecostress` (download splib07 / ECOSTRESS subsets).
- Similarity measures: `sam`, `sid`, `spectral_similarity` (SCA, Euclidean, SID SAM).
- `classify_spectral` with a rejection threshold.
- Lab case: HACARUS sugar/salt/flour and rice types. Field case: Tanager energy-mining
  (Permian Basin) with clay and carbonate band depth near 2200 and 2330 nm.

### 09_05_07_endmember_extraction_and_spectral_unmixing.ipynb
Endmember Extraction and Spectral Unmixing
- Overview: linear mixing model, abundance constraints, pure pixels.
- Extraction: `ppi`, `atgp`, `nfindr`, `vca`, `extract_endmembers`; number of endmembers from
  `hysime`.
- Unmixing: `unmix` (LS, SCLS, NNLS, FCLS), `mesma`, `sparse_unmix`; `reconstruct` and RMSE
  maps.
- Data: Pavia Centre, Tanager urban (impervious, vegetation, soil) and coastal-water-bodies.

### 09_05_08_target_and_anomaly_detection.ipynb
Target and Anomaly Detection
- Overview: background statistics, Mahalanobis distance, detection vs classification.
- Anomalies: `rx` (global and local).
- Targets: `matched_filter`, `ace`, `cem`, `osp`.
- Evaluation: `detection_roc` with a hand drawn truth mask.
- Data: HACARUS sugar_salt_flour_contamination (the eraser is the target, the clean scene
  gives the background), Tanager urban (rooftop and vehicle targets).

### 09_05_09_greenhouse_gas_plume_detection.ipynb
Greenhouse Gas Plume Detection
- Overview: CH4 absorption at 2300 nm, unit absorption spectrum, column wise matched filter,
  integrated mass enhancement (IME), emission rate.
- Data: Tanager GHG-plumes scene (Syria, Oct 2024), TOA radiance.
- Build a CH4 unit absorption spectrum on Tanager bands (HITRAN based table or the EMIT
  target generator; to be added to the data folder).
- `gas_matched_filter` (window 2100 to 2500 nm), `plume_mask`, `plume_ime`, `emission_rate`
  with an assumed wind speed.
- Validate against `ortho_ql_ch4.tif` and the plume polygons in `ql_ch4_json.geojson`.

### 09_05_10_thematic_applications_across_tanager_scenes.ipynb
Thematic Applications Across Tanager Scenes
- Fire (Dec 2024): SWIR composites, active fire detection, burn severity, K emission line
  near 766 nm.
- Snow and ice (Apr 2025): snow grain size from 1030 nm band depth, snow vs cloud.
- Coastal and inland water (Venezuela coast): chlorophyll, turbidity, CDOM band ratios,
  shallow water features.
- Energy and mining: mineral mapping with continuum removal and band depth.
- One short, self contained section per scene, reusing earlier functions.

### 09_05_11_deep_learning_for_hyperspectral_classification.ipynb
Deep Learning for Hyperspectral Classification
- Overview: spectral (1D), spectral spatial (3D CNN, HybridSN), spectral transformer.
- `block_split` to avoid spatial leakage, `HSIPatchDataset`, `SpectralCNN1D`, `CNN3D`,
  `HybridSN`, `SpectralTransformer`, `train_hsi_classifier`, `evaluate`, `predict_hsi_map`.
- Data: Pavia Centre (needs the ground truth map `pavia_gt`, download in Setup), PCA or MNF
  reduced input from notebook 03; compare with SAM and random forest baselines.

## Gaps to close before writing

1. Done (0.3.42): `read_tanager`, `read_tanager_ancillary` and `swath_to_grid` read the
   Tanager ortho and basic products; `read_hyperspectral` routes Tanager `.h5` files.
2. Done (0.3.42): `read_ohs` stacks the 32 OHS band GeoTIFFs (nominal band centres in
   `OHS_WAVELENGTHS`) and orthorectifies with the RPCs (`ortho=True`).
3. Pavia ground truth (`pavia_gt`) is missing; only the cube is present.
4. HACARUS has no labels; truth masks for notebooks 06 and 08 will be digitised by hand.
5. CH4 unit absorption spectrum is not in the data folder.
6. A banner image under `Images/Banner/08_remote_sensing/` for each notebook.
7. Scenes are large (each SR file ~1 to 2 GB); notebooks should subset windows early and
   save small ENVI clips to `outputs/` for reuse by later notebooks.
8. The stray files at the chapter root (`20250224_..._basic_beta_udm.tif`, the 20250516 basic
   radiance and geolocation array) now live in `Data/08_remote_sensing/08_05_HyperSpectral/`.
