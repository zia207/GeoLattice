# Remote Sensing Foundation Models with GeoLattice: Notebook Outline

Chapter folder: `Tutorials/notebooks/08_remote_sensing/08_11_foundations_models/`
Library: `geolattice.rsense.foundation_models` (catalog, `load_model` / `FoundationEncoder`,
`chip_raster`, `normalize`, `embed_raster`, AlphaEarth helpers, probes, similarity, clustering,
change, `segmentation_model`, `terratorch_config`, `zero_shot_classify`), with
`geolattice.rsense.classify.image_classification` (spatial splits, few-shot prototypes, SAM, metrics)
and `geolattice.rsense.dl` (tile datasets, training).

Every notebook follows the house style: banner cell, `#` title with a one paragraph intro,
`## Overview` (theory), `## ... with GeoLattice`, Learning objectives, Data, Contents, Setup, the
working sections, and Summary and exercises. Section headings carry no numbers, every code
statement has a comment, and the markdown has no em or en dashes. Notebooks are built by
scripts in `_builders/` (now kept locally in `helper_functions/`, outside the repository) and executed before release; markdown claims are checked against the
executed outputs.

Moved from the classification chapter (21 November 2026): `08_09_05_01_self_supervised_pretraining`
is now `09_11_01_self_supervised_pretraining_for_sentinel_2_crop_mapping.ipynb`, and `08_09_05_02_foundation_model` is now
`09_11_02_geospatial_foundation_models_embeddings_promptable_segmentation_and_zero_shot_classification.ipynb`. Both are executed notebooks. Only their folder path,
banner and cross-references changed; their data still comes from
`Data/08_remote_sensing/08_09_image_classification/`. Notebook 01 shows how pretraining works
by doing it at small scale. Notebook 02 is a one notebook tour (model zoo, DOFA and Prithvi
embeddings, few-shot prototypes, SAM prompting, RemoteCLIP, a trust scorecard). The planned
notebooks go deeper on each model family, measure what the embeddings know, and link back to
01 and 02 instead of repeating their SSL, SAM and few-shot parts.

Installation: `geolattice[rsense-foundation]` (TorchGeo, TerraTorch, timm, open_clip, transformers,
earthengine-api). Notebooks 07 and 08 read AlphaEarth from public copies (no account); notebook 12
needs `microsoft-aurora` and `gcsfs`.

## Data inventory

No new downloads are needed except where noted. Paths are relative to `Data/08_remote_sensing/`
(note that the Data tree numbers the capstones `08_11_Capstones`, while the tutorials number
them `08_13`).

| Dataset | Path | Content | Size | Used in |
|---|---|---|---|---|
| Corn Belt cube | `09_09_image_classification/Corn_belt/cache_20m/{s2,s1,labels}_20m.nc` | Iowa AOI, Sentinel-2 10 bands x 34 dates (2020 to 2024), Sentinel-1 at 20 m, CDL labels | 7.3 GB folder, cache much smaller | 01, 03, 04, 05, 06, 13 |
| Corn Belt scenes and CDL | `.../Corn_belt/Sentinel_2/2025_08_20250826`, `.../Corn_belt/Landsat_9/`, `.../Corn_belt/CDL/{10_m,30_m}` | 26 August 2025 Sentinel-2 L2A scene, Landsat 8/9 L2 scenes, CDL 10 m and 30 m | in the 7.3 GB | 02, 04, 06 |
| UB campus Sentinel-2 | `09_09_image_classification/Sentinel_2_UB/` | 10 band GeoTIFFs + labelled points (train/test CSV) | 6 MB | 00, 02, 03 |
| UB campus Sentinel-1 | `09_09_image_classification/Sentinel_01_UB/` | VV/VH bands and texture | 11 MB | 05 |
| WorldView-2 Bangladesh | `09_09_image_classification/WorldView2_BD/` | 8 band VHR scene, labelled data, boundary raster | 142 MB | 05, 09 |
| DeepGlobe land cover | `09_09_image_classification/Landcover_data/` | RGB tiles + masks, 7 classes | 2.8 GB | 02, 09, 10 |
| EuroSAT RGB | `08_11_Capstones/EuroSat.zip` | 27,000 64 px tiles, 10 classes | 90 MB zip | 10, 13 |
| Bug Fire Sentinel-2 | `08_11_Capstones/bug_fire_sentinel2/{pre,post}` + perimeter | L2A pre and post fire tiles | 3.4 GB | 11 |
| Bangladesh delta Landsat | `08_11_Capstones/bd_delta_annual/` | Annual Landsat 8 L2 scenes 2013 to 2020 | 1.1 GB | 11 |
| Great Plains SOC | `../00_shared/data_geostatistics/Data_GP/GP_POINT_SOC.shp` | 570 topsoil SOC points, Albers | small | 07, 08 |
| Great Plains covariates | `../03_spatial_analysis/Great_plain/` (`gp_grid_new.csv`, `Raster/`) | Prediction grid and classical covariates | small | 08 |
| AlphaEarth embeddings | public Source Cooperative Zarr mosaic and Google Cloud Storage GeoTIFFs (also Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) | 64 band annual 10 m embeddings | cached to `outputs/` as NetCDF/CSV on first run | 07, 08 |
| ERA5 sample (optional) | downloaded by notebook 12 | Two time steps of ERA5 0.25 degree fields | about 1 GB | 12 |

Weights (TorchGeo, Hugging Face and torch.hub caches; already cached on the build machine):
DOFA base, SSL4EO S2 ResNet-18 MoCo, SSL4EO S1 ResNet-50, SSL4EO ViT-S DINO, Satlas Swin-T,
Scale-MAE, CROMA base, Copernicus-FM, Panopticon, Prithvi 1.0 and 2.0-300M, TerraMind base,
AnySat, RemoteCLIP ViT-B/32, GeoRSCLIP, SAM ViT-H. Notebook 02 also downloads SAM 2 and
Grounding DINO on first use (about 7 GB in total). DINOv3-sat is gated (accept the license on
Hugging Face).

## Notebook list

Status (22 November 2026): every notebook is built by `_builders/build_08_11_XX.py`, executed without
errors, and its markdown checked against its outputs and figures (01 and 02 were moved from 08_09 and
keep their own outputs). 07 and 08 read AlphaEarth anonymously from its public Zarr and GeoTIFF copies,
so they need no Earth Engine account; run 07 before 08, which uses its cached samples. After any
`run.py` or `sync_text.py`, rerun `helper_functions/Tutorials/_tools/notebook_install_notes.py` to restore the
installation notes.

| Notebook | Built | Main result |
|---|---|---|
| 00 fundamentals | yes | DOFA patch embeddings beat band means on a small campus probe |
| 01 self-supervised pretraining | moved | |
| 02 foundation models tour | moved | |
| 03 embedding rasters | yes | wrong units change embedding direction (cosine 0.23) yet barely change the probe |
| 04 optical encoders | yes | Prithvi and SSL4EO lead in June; a U-Net from scratch beats fine-tuned Prithvi on segmentation |
| 05 multi-sensor models | yes | Panopticon transfers from Sentinel-2 to Landsat (F1 0.94), DOFA does not (0.44) |
| 06 time series | yes | Prithvi-TL maps crops by late June (F1 0.97) about three weeks before an NDVI random forest |
| 07 AlphaEarth | yes | one corn pixel finds all corn (AUC 0.99); change 2020 to 2024 maps crop rotation (AUC 0.89) |
| 08 soil mapping | yes | classical covariates beat AlphaEarth for SOC (R2 0.46 vs 0.41); embeddings track NDVI and climate |
| 09 very high resolution | yes | patch features beat RGB statistics on DeepGlobe; SAM 2 needs box prompts for small fields |
| 10 vision-language | yes | one labelled image per class beats zero-shot on EuroSAT |
| 11 change and similarity | yes | embedding change reaches AUC 0.90 against the fire perimeter (dNBR 0.975) |
| 12 weather | yes | Aurora's 6 h error is 0.49 K for 2 m temperature against 2.8 K for persistence |
| 13 benchmarking | yes | rankings change between EuroSAT RGB and Sentinel-2 crops |

### 09_11_00_what_foundation_models_are_and_why_remote_sensing_needs_them.ipynb
What Foundation Models Are and Why Remote Sensing Needs Them
- Overview: from task-specific CNNs to pretrain once, adapt many times; why labels are the
  bottleneck in Earth observation; what "foundation" does and does not mean.
- Pretraining objectives: masked autoencoders (Prithvi, SatMAE, Scale-MAE), contrastive
  learning (SSL4EO MoCo, CROMA), self-distillation (DINO, DINOv3-sat, Panopticon), joint
  embedding prediction (AnySat, Galileo), generative any-to-any (TerraMind), image-text
  (RemoteCLIP). One diagram per objective; notebook 01 then trains SimCLR, MoCo and an MAE
  at small scale, so this notebook keeps the theory short.
- What makes remote sensing different: many bands, many sensors, ground sample distance,
  time, geolocation, huge unlabelled archives. How models encode each: wavelength-conditioned
  stems (DOFA, Copernicus-FM, Panopticon), GSD-aware position encoding (Scale-MAE), time and
  location tokens (Prithvi-TL), per-sensor encoders (CROMA, AnySat).
- Tokens, patches and embeddings: what a 16 px patch token is, CLS vs mean pooling, the
  embedding dimension, pixel vs patch vs image embeddings.
- Ways to use a model: frozen embeddings + simple model, linear and kNN probes, partial or full
  fine-tuning, decoders for segmentation, zero-shot with text, precomputed embedding products.
- The landscape: `fm.CATEGORIES`, `fm.list_models()` grouped by category, filters by modality
  and `loadable`, `fm.model_info`, `fm.recommend` for common tasks. Table of provider, year,
  parameters and weights.
- Benchmarks and honest expectations: GEO-Bench and PANGAEA results in words; tuned supervised
  baselines often compete; why spatial cross-validation matters.
- Licences, compute and the carbon cost of pretraining; CPU vs GPU for inference.
- Quick hands on (as built): DOFA patch embeddings of the UB campus Sentinel-2 image (160 m
  cells), a PCA scatter by land use, and a linear probe against mean band values.

### 09_11_01_self_supervised_pretraining_for_sentinel_2_crop_mapping.ipynb (moved from 08_09_05_01, done)
Self-supervised Pretraining for Sentinel-2 Crop Mapping
- Overview: why labels are the bottleneck; contrastive learning (SimCLR, MoCo) with ordinary
  augmentations and with seasonal views (two dates of the same place), masked autoencoders.
- Data: Corn Belt 20 m Sentinel-2 cube (built by `09_09_03_02`) and CDL labels.
- Pretraining four encoders on unlabelled chips (`pretrain_contrastive`, `build_mae`,
  `pretrain_masked`), then frozen embeddings (`extract_embeddings`) scored with a linear probe
  on a geographic hold-out (`geographic_holdout`), against random weights and band means.
- Label efficiency, t-SNE of the embeddings, a crop map from the best encoder, and spatial
  leakage pitfalls. Encoders are saved with their normalization for reuse.
- Role in the chapter: the hands-on version of the pretraining objectives in notebook 00, at
  teaching scale; the planned notebooks reuse its probe and hold-out design.

### 09_11_02_geospatial_foundation_models_embeddings_promptable_segmentation_and_zero_shot_classification.ipynb (moved from 08_09_05_02, done)
Geospatial Foundation Models: Embeddings, Promptable Segmentation and Zero-Shot Classification
- Overview and model zoo (`foundation_model_info`), licences read from the model cards.
- Load DOFA and Prithvi-EO-2.0 (`load_foundation_encoder`), TerraTorch fine-tuning config.
- Embeddings as features on the Corn Belt 2025 Sentinel-2 scene and CDL: `embedding_search`,
  five-label few-shot crop maps with `few_shot_prototypes`, few-shot land use on UB campus.
- Promptable segmentation: SAM from points and boxes, `text_prompt_segment` (Grounding DINO).
- Zero-shot classification: RemoteCLIP on DeepGlobe crops, prompt wording, GeoRSCLIP.
- Trust: a scorecard for every model output, provenance written with the few-shot map.
- Role in the chapter: a fast overview before the deep dives. It uses the older
  `geolattice.rsense.classify.image_classification` loaders; notebooks 03 onward use
  `geolattice.rsense.foundation_models`, and notebook 03 shows how the two APIs map onto each other.

### 09_11_03_from_a_geotiff_to_an_embedding_map.ipynb
From a GeoTIFF to an Embedding Map
- Overview: the input contract of a model (bands and order, scale, normalization, chip size,
  time steps) and why mismatches fail silently.
- `FoundationEncoder` anatomy: `spec`, `model`, `device`, `encode`, `embed`, `embed_raster`,
  `forward_features`, `pool_features` with `pool="cls"` vs `"mean"`.
- Bands and wavelengths: GeoLattice band names, Sentinel-2 ids (`B04`), `band_wavelengths`,
  `WAVELENGTHS_NM`, `BANDWIDTHS_NM`; how DOFA, Copernicus-FM and Panopticon read them.
- Normalization: `normalize` with `standard`, `scale` (10000 for L2A, 3000 for Satlas),
  `minmax`; model card statistics vs per-scene statistics; experiment showing how wrong
  scaling changes the embeddings (cosine similarity to the correctly scaled run).
- Chipping: `chip_raster` chip size, stride, reflect padding, chip centre coordinates;
  resize to the model input; edge effects with overlapping strides.
- `embed_raster` on the UB Sentinel-2 stack with SSL4EO, Satlas, DOFA and Panopticon:
  georeferenced `(embedding, y, x)` maps, `embedding_rgb` (PCA false colour), side by side.
- Embeddings to a table: `embeddings_to_frame`, joining labelled points to the nearest chip.
- CPU vs GPU: `device=`, batch size, timing table, memory use; `rs.set_backend("cuda")`.
- Saving embeddings to NetCDF/Zarr with model name, chip size and normalization in attributes
  so they can be reused without the model.

### 09_11_04_prithvi_ssl4eo_satlaspretrain_and_scale_mae_on_sentinel_2.ipynb
Prithvi, SSL4EO, SatlasPretrain and Scale-MAE on Sentinel-2
- Overview: the optical encoders in detail. Prithvi-EO (HLS, 3-D patch embedding, 1.0 vs
  2.0, the TL variant), SSL4EO-S12 (ResNet and ViT, MoCo and DINO), SatlasPretrain (Swin,
  multi-task labels), Scale-MAE (GSD-aware), SatMAE and SpectralGPT (catalog only).
- Data: one summer date of the Corn Belt Sentinel-2 cube; matching the 6 HLS bands for
  Prithvi, the 13 or 9 band layouts for SSL4EO and Satlas (zero-filling missing bands and
  what that costs), RGB for Scale-MAE.
- Embeddings of the same AOI from each model; `embedding_rgb` maps compared with the CDL.
- What do they know? `linear_probe` and `knn_probe` on CDL crop classes with spatial block
  groups (`spatial_block_split` from `image_classification`); random vs spatial folds.
- Unsupervised view: `cluster_embeddings` (k-means, GMM) vs CDL; adjusted Rand index.
- Fine-tuning Prithvi: `segmentation_model` (UperNet decoder, default necks) on Corn Belt
  chips with CDL masks; frozen backbone vs full fine-tune; `terratorch_config` for the
  Lightning CLI route; IoU per class and compute time.
- Label efficiency: accuracy vs number of labelled chips (10, 50, 200, all) for a
  from-scratch U-Net and the Prithvi model.

### 09_11_05_one_model_for_many_sensors_dofa_panopticon_copernicus_fm_croma_and_terramind.ipynb
One Model for Many Sensors: DOFA, Panopticon, Copernicus-FM, CROMA and TerraMind
- Overview: why sensor-agnostic models matter (new satellites, odd band sets, SAR,
  hyperspectral); hypernetwork and cross-attention patch embeddings; metadata encoding.
- DOFA and Panopticon on three sensors with different bands: UB Sentinel-2 (10 bands),
  WorldView-2 (8 bands, coastal to NIR2), Sentinel-1 VV/VH (Panopticon SAR channel ids).
  Same code, different `bands=`.
- Copernicus-FM with metadata: wavelengths, bandwidths, lon/lat/time/area in the `metadata`
  tensor; effect of supplying vs omitting metadata on probe accuracy.
- CROMA optical, SAR and joint embeddings on the Corn Belt S1 + S2 pair: `modality="optical"`,
  `"sar"`, `"both"`; probe accuracy for each; simulate cloud cover by masking S2 and compare.
- TerraMind: `modalities=["S2L2A", "S1GRD"]` dict input, encoder embeddings; the any-to-any
  idea and "thinking in modalities" (TiM) in words; generation of a missing modality shown
  as an optional TerraTorch example if the generation API is exposed (see Library gaps).
- Cross-sensor transfer: train a probe on Sentinel-2 embeddings, apply to WorldView-2
  embeddings from the same wavelength-aware model; where it works and where it breaks.

### 09_11_06_time_in_foundation_models_prithvi_tl_anysat_presto_and_galileo.ipynb
Time in Foundation Models: Prithvi-TL, AnySat, Presto and Galileo
- Overview: pixel time series vs image time series; phenology; how models take time
  (3-D patches, date tokens, temporal attention, L-TAE); irregular sampling and clouds.
- Data: Corn Belt Sentinel-2 cube, 34 dates; build monthly composites and a cloud mask.
- Prithvi-EO-2.0-300M-TL with `num_frames=4` and date/location inputs; compare 1 vs 4 frames
  on crop type probes.
- AnySat with S2 time series and dates as a dict (`{"s2": ..., "s2_dates": ...}`),
  `patch_size=` in metres; tile vs patch outputs.
- Presto and Galileo: what they are and how to run them with the authors' packages (catalog
  entries are `external`); a small Presto run on pixel samples if the package installs cleanly.
- Baselines: NDVI phenology features (from `geolattice.rsense.timeseries`) + random forest, and a
  TempCNN from `09_09_03_02`; same spatial folds.
- Early season mapping: accuracy by month of the season for each approach.

### 09_11_07_alphaearth_foundations_precomputed_embeddings_for_the_whole_planet.ipynb
AlphaEarth Foundations: Precomputed Embeddings for the Whole Planet
- Overview: embedding fields; what AlphaEarth learned from (optical, SAR, lidar, climate,
  text); 64 unit-length dimensions at 10 m, one per year since 2017; strengths (no model to
  run, consistent globally) and limits (annual only, opaque dimensions, licence terms).
- Earth Engine setup: authentication, project id, `alphaearth_image(aoi, year)`; a quick map
  of three dimensions as RGB; caching results to `outputs/` so the notebook reruns offline.
- `sample_alphaearth` at the Great Plains SOC points (chunking, scale), and
  `alphaearth_to_xarray` for the Corn Belt AOI at 30 m.
- Dot products are cosine similarities: `similarity_map` from a clicked field; threshold to
  find all fields like it; compare with the CDL.
- Unsupervised land cover: `cluster_embeddings` on the AOI vs the CDL and WorldCover.
- Year to year change with `embedding_change` (2019 vs 2024) on the Corn Belt AOI; what
  rotation (corn to soybean) looks like in embedding space.
- Comparison: AlphaEarth vs DOFA vs SSL4EO embeddings on the same CDL probe.

### 09_11_08_foundation_model_embeddings_as_covariates_digital_soil_mapping_of_soc.ipynb
Foundation Model Embeddings as Covariates: Digital Soil Mapping of SOC
- Overview: environmental covariates in DSM (scorpan), how embeddings act as learned
  covariates, risks (leakage, spatial autocorrelation, interpretability), area of applicability.
- Data: 570 Great Plains SOC points, the classical covariate grid, AlphaEarth embeddings from
  notebook 07 (cached). (Planned DOFA embeddings of a Landsat composite were dropped: no
  Landsat archive of the Great Plains is in the repository.)
- Models: random forest and ridge regression with classical covariates only, embeddings only, and both;
  `linear_probe` (ridge) as a baseline; spatial block cross-validation with `groups=`.
- Dimensionality: PCA of the 64 AlphaEarth bands; how many components are needed.
- Interpretation: SHAP and permutation importance (`geolattice.rsense.xai`); what an embedding
  dimension correlates with (NDVI, elevation, climate).
- Regression kriging on residuals (`geolattice.geostatistics`) and whether embeddings remove the
  spatial structure in the residuals (variogram before and after).
- Map SOC over the Great Plains grid with prediction intervals and the area of applicability.

### 09_11_09_very_high_resolution_dinov3_sat_satlas_naip_scale_mae_and_sam_2.ipynb
Very High Resolution: DINOv3-sat, Satlas NAIP and SAM 2
- Overview: what changes at sub-metre resolution (objects, shadows, texture, off-nadir);
  DINOv3 self-distillation on SAT-493M; dense patch features vs global embeddings.
- As built: DINOv3-sat is tried and, being gated, falls back to Scale-MAE and Satlas NAIP Swin-B.
  Dense features of the WorldView-2 scene (`embed_raster(dense=True)`), crop or fallow fields from
  features sampled at 556 polygon centroids, DeepGlobe land cover per 8 m cell over 20 tiles (tiles as
  CV groups), and SAM 2 field outlines from click and box prompts scored by IoU against the drawn
  polygons (box prompts win clearly), exported to GeoPackage.

### 09_11_10_vision_language_models_zero_shot_classification_and_text_retrieval.ipynb
Vision-Language Models: Zero-Shot Classification and Text Retrieval
- Overview: CLIP training, shared image-text space, temperature, prompt templates; remote
  sensing CLIPs (RemoteCLIP, GeoRSCLIP, SkyCLIP) and chat models (GeoChat, EarthGPT,
  LHRS-Bot, EarthDial) described from the catalog.
- EuroSAT zero-shot benchmark: `zero_shot_classify` with RemoteCLIP and GeoRSCLIP; confusion
  matrix; per-class accuracy; prompt template and prompt ensembling effects.
- Linear probe on CLIP image embeddings vs zero-shot: how many labels beat zero-shot.
- Text to image retrieval: embed DeepGlobe tiles, query "a river with sandbars", "brick
  kilns", "flooded fields"; `embedding_search` top-k gallery; precision at k from masks.
- Image to image retrieval across datasets (EuroSAT query, DeepGlobe gallery).
- Limits: counting, small objects, non-RGB bands, place names; outputs flagged unverified.

### 09_11_11_change_and_similarity_in_embedding_space.ipynb
Change and Similarity in Embedding Space
- Overview: change as distance between embeddings; why it is robust to illumination and
  phenology noise compared with band differencing; false alarms and thresholds.
- Bug Fire: pre and post Sentinel-2 embeddings (DOFA, SSL4EO) and `embedding_change`;
  compare with dNBR from `spectral_index`; ROC curve against the fire perimeter.
- Bangladesh delta: seven dry-season Landsat 8 scenes 2013 to 2020 (no 2016 scene on disk), the
  largest window with data in every year, overlapping DOFA chips, change vs NDWI change, intervals of
  strong change. (AlphaEarth change is covered in notebook 07.)
- Similarity search: one example chip (a riverbank) against every chip of the 2020 map with
  `similarity_map` and `embedding_search`; FAISS is mentioned as an exercise.
- As built, change maps use pooled embeddings of overlapping chips (32 px every 16 px): patch
  features of 224 px chips produced chip-border artefacts in the DOFA change map.

### 09_11_12_weather_and_earth_system_foundation_models.ipynb
Weather and Earth System Foundation Models
- Overview: from numerical weather prediction to learned emulators; Aurora, Prithvi WxC,
  GraphCast, GenCast and FourCastNet compared (architecture, data, lead time, ensembles).
- Catalog view: `fm.list_models("weather")`, licences and hardware needs.
- As built: the full Aurora 0.25 pretrained model (1.3 B) on ERA5 from the public WeatherBench2
  Zarr store; it does not fit a 16 GB GPU at 0.25 degrees, so it runs on the CPU (about 7 minutes per
  step, GPU automatically when 24 GB or more). +6 h and +12 h forecasts scored with latitude-weighted
  RMSE against ERA5 and persistence, globally and over Iowa.
- Links to land applications: drought, evapotranspiration (`08_12`), crop yield.

### 09_11_13_benchmarking_foundation_models_on_your_own_data.ipynb
Benchmarking Foundation Models on Your Own Data
- Overview: what to measure (accuracy, label efficiency, transfer, speed, memory, licence),
  fair comparisons, spatial and temporal hold-outs, seeds and confidence intervals.
- One harness: every `loadable` model whose inputs match, on EuroSAT RGB and the Corn Belt
  CDL task; `linear_probe` and `knn_probe` with spatial groups; supervised baselines (random
  forest on bands, small CNN).
- Results table and plots: accuracy, F1, embedding time per 1000 chips on CPU and GPU, GPU
  memory, parameters.
- Decision guide that updates `fm.recommend` with what the data showed; when not to use a
  foundation model.
- Reproducibility: record model name, weights version, normalization and chip size with
  every embedding file.

## Library work (done in GeoLattice 0.3.47)

Each gap found while planning was closed in `geolattice.rsense.foundation_models` before the notebooks
were written, with tests in `tests/rsense/test_foundation_models.py`.

| Gap | Resolution | Used in |
|---|---|---|
| `embed_raster` took `(band, y, x)` only | `chip_raster` and `embed_raster` accept `(time, band, y, x)`; chips come out as `(n, band, time, h, w)` | 06 |
| no dense outputs | `FoundationEncoder.dense_features` and `embed_raster(dense=True)` (ViT tokens, ResNet and Swin maps, DOFA tokens) | 00, 04, 06, 07, 09, 11, 13 |
| no metadata helpers | `chip_metadata`; `embed_raster(metadata=True)` for Copernicus-FM and Prithvi-TL | 05, 06 |
| inputs prepared by hand per model | `input_spec`, `INPUT_SPECS`, `prepare`, `embed_raster(norm="model")` | 03 onward |
| TerraMind generation not exposed | used through TerraTorch's `terratorch_terramind_v1_base_generate` with inputs from `fm.prepare` (the registry model keeps its statistics on the CPU, so inputs are standardized outside it) | 05 |
| Presto and Galileo loaders | kept as `external` catalog entries; notebook 06 explains how to install them from the authors' repositories and evaluate their embeddings with the same probes | 06 |
| AlphaEarth per year, large areas, no account | `sample_alphaearth(points, [years])`; `alphaearth_to_xarray(source="zarr" / "gcs" / "earthengine", like=)`; `alphaearth_dequantize`, `alphaearth_index` | 07, 08 |
| no benchmark helper | `benchmark_encoders` (probes, speed, GPU memory, precomputed baselines) | 07, 13 |

Still open:

- Notebook 02 calls `ic.foundation_model_info` and `ic.load_foundation_encoder`. Either keep them as
  thin wrappers or switch the notebook to `fm.list_models` and `fm.load_model` when it is next rebuilt,
  so the chapter uses one API (notebook 03 shows the mapping).
- Banner (done): `Images/Banner/07_remote_sensing/09_11_banner_rs_foundation_models.png` (renamed from
  `..._foundation_modelspng.png`); every notebook of the chapter uses it.
