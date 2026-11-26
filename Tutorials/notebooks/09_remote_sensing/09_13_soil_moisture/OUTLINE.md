# Satellite Soil Moisture with GeoLattice: Notebook Outline

Chapter folder: `Tutorials/notebooks/09_remote_sensing/09_13_soil_moisture/`
Banner: `Images/Banner/07_remote_sensing/09_13_banner_rs_soil_moisture.png`
Course outline: `Satellite-Based Soil Moisture with Python Tutorial Outline.md` (26 modules, seven parts)
Library: `geolattice.rsense.soil_moisture` (one module, `src/rsense/soil_moisture.py`), used with
`geolattice.rsense.access` (Earthdata, ASF, STAC), `geolattice.rsense.sar` (Sentinel-1 RTC, speckle,
NISAR HDF5), `geolattice.rsense.timeseries` (climatology, anomalies, VCI) and
`geolattice.rsense.core` (`zonal_stats`, `apply_tiled`, `write`). Import it as
`import geolattice.rsense.soil_moisture as sm`.

Every notebook follows the house style of 09_12: banner cell, `#` title with a one paragraph intro,
`## Overview` (theory), `## ... with GeoLattice`, Learning objectives, Data, Contents, Setup, the
working sections, and Summary and exercises. Section headings carry no numbers, every code
statement has a comment, and the markdown has no em or en dashes. Notebooks are built by
scripts in `_builders/` (`build_09_13_XX.py`, same `nb.py` / `run.py` / `sync_text.py` pattern as
09_12) and executed before release; markdown claims are checked against the executed outputs.

Scope: the 29 notebooks of course modules 1 to 25 live in this folder (`09_13_00` to `09_13_28`).
The four capstones of module 26 go to `09_14_Capstones` as `09_14_18` to `09_14_21`, as the
09_12 capstone did (`09_14_17`).

Installation: `geolattice[remote-sensing]` (option 5) plus the soil moisture packages named in
the module docstring (`pytesmo`, `ismn`, `ascat`, `h5py`, `xgboost`, `shap`, `torch`, `filterpy`).
Every notebook needs a free NASA Earthdata login (`~/.netrc`). Optional free accounts: ISMN
(more stations), H SAF (ASCAT), EU Login (CLMS 1 km SSM), Copernicus CDS (gridded ERA5-Land),
USDA NASS Quick Stats key (county yields). Each notebook says which it can use and runs without
them through the fallback in the data inventory.

## Study areas

| Site | Why | Data checked (25 November 2026) |
|---|---|---|
| Central Iowa, Ames and the South Fork of the Iowa River (SMAPVEX16 area), bbox about -93.9, 41.8, -93.3, 42.2 | Rainfed corn and soybean on one grid of every product; a SMAP core validation region; public in situ stations; strong crop growth that tests vegetation correction | USCRN `IA_Des_Moines_17_E` hourly file for 2024; SCAN `2031:IA:SCAN` Ames (2001 to now); SMAPVEX16 networks `SV16I_TNET`, `SV16I_PNET`, `SV16I_TPSM` (May to September 2016); Planetary Computer has 15 Sentinel-1 RTC and 9 Sentinel-2 L2A scenes from 15 May to 31 August 2016; NISAR provisional soil moisture, 66 granules; USGS gauge 05470000 South Skunk River near Ames (daily flow 2024) |
| Bangladesh, Rabi season wheat districts in the northwest (Rajshahi and Rangpur divisions) and Mymensingh | Dry season drought and irrigation on small fields; the course capstone 1 | NISAR provisional soil moisture over Mymensingh, 34 granules; ESA CCI from 1978; no public in situ network, so ERA5-Land and triple collocation stand in |
| New York State counties | County drought dashboard, capstone 3 | NISAR provisional, 35 granules near Binghamton; ESA CCI and SMAP L4 cover the state |
| High Plains irrigated strip, Nebraska (Platte valley) | Center pivots next to rainfed fields for irrigation detection (notebook 27, capstone 2) | to check when notebook 27 is built |
| Europe (optional): REMEDHUS network, Spain | Only area of the CLMS 1 km SSM product (notebook 14) | needs EU Login and an ISMN account |

## Data inventory

Everything is downloaded by the notebooks on first run and cached under
`Data/08_remote_sensing/08_13_soil_moisture/` (new folder, following `08_12_surface_energy_balance`),
through a helper `sm_data.py` in this folder (same role as `seb_data.py`). Collections were
checked in NASA CMR, the Planetary Computer STAC and the provider sites on 25 November 2026.

| Dataset | Source and identifier | Account | Content | Used in |
|---|---|---|---|---|
| SMAP L3 radiometer | NSIDC `SPL3SMP` v009, `earthaccess` | Earthdata | 36 km daily AM and PM soil moisture, quality flags | 02, 04, 23 |
| SMAP L3 Enhanced | NSIDC `SPL3SMP_E` v006 | Earthdata | 9 km daily soil moisture, VWC, tau, flags | 02 to 04, 06, 15 to 18, 20 to 25, 27 |
| SMAP L4 | NSIDC `SPL4SMGP` v008 | Earthdata | 9 km 3-hourly surface and root-zone soil moisture | 05, 24, 28 |
| SMAP/Sentinel-1 active-passive | NSIDC `SPL2SMAP_S` v003 | Earthdata | 3 km and 1 km soil moisture | 14 |
| SMOS with the SMAP algorithm | NSIDC `NSIDC-0800` v2 (L3, 9 km EASE-Grid) and `NSIDC-0799` (L2) | Earthdata | SMOS Tb retrieved like SMAP | 06 |
| SMOS CATDS L3 (optional) | CATDS | CATDS (free) | ESA style L3 with RFI probability | 06 |
| AMSR2 LPRM | GES DISC `LPRM_AMSR2_DS_A_SOILM3` (downscaled 10 km) and `LPRM_AMSR2_A_SOILM3` (25 km) v001 | Earthdata | C1, C2 and X band soil moisture and VOD | 07 |
| ESA CCI SM | CEDA `neodc/esacci/soil_moisture/data/daily_files/{ACTIVE,PASSIVE,COMBINED}/v09.2/`, e.g. `ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-20230101000000-fv09.2.nc` | none | 0.25 degree daily, 1978 to 2024, flags and uncertainty; ACTIVE is ASCAT based (degree of saturation, %) | 08, 09, 10, 20, 22, 23, 25 |
| ASCAT H SAF SSM and SWI (optional) | H SAF | H SAF (free) | 12.5 km ASCAT SSM time series | 08 (ESA CCI ACTIVE is the fallback) |
| ERA5-Land soil water | Open-Meteo archive `models=era5_land` (`soil_moisture_0_to_7cm`, `7_to_28cm`, `28_to_100cm`, hourly, checked), or CDS `reanalysis-era5-land` for grids | none (CDS free key for grids) | model reference and third member for triple collocation | 10, 23, 24, 27 |
| GLDAS Noah | GES DISC `GLDAS_NOAH025_3H` v2.1 | Earthdata | 0.25 degree model soil moisture, the triple collocation member that needs no CDS key | 23 |
| Sentinel-1 RTC | Planetary Computer `sentinel-1-rtc` (`rsense.sar.load_rtc`); OPERA RTC-S1 via ASF as an option | none (Earthdata for OPERA) | 10 m gamma0 VV and VH, terrain corrected | 11 to 13, 15, 20 |
| NISAR L3 soil moisture | ASF `NISAR_L3_SME2_PROVISIONAL_V1` (from 28 October 2025) and `NISAR_L3_SME2_BETA_V1`, HDF5 | Earthdata | 200 m L-band SAR soil moisture with flags; also `NISAR_VWC` | 15, capstone 21 |
| Sentinel-2 L2A | Planetary Computer `sentinel-2-l2a` | none | NDVI, LAI, SWIR for WCM and OPTRAM | 13, 16 |
| MODIS LST, NDVI, LAI | Planetary Computer `modis-11A1-061`, `modis-13A1-061`, `modis-15A2H-061` | none | 1 km LST and 500 m NDVI for downscaling and VCI | 01, 07, 16 to 19, 25 |
| Copernicus DEM, WorldCover, CDL | Planetary Computer `cop-dem-glo-30`, `esa-worldcover`, `usda-cdl` | none | terrain masks, TWI, land cover strata, crop types | 11, 16, 22, 26, 27 |
| gridMET | Planetary Computer `gridmet` | none | daily precipitation and PET (SPI, SPEI, bucket model, events) | 05, 24, 25, 28 |
| SoilGrids 2.0 | ISRIC `files.isric.org/soilgrids/latest/data/{sand,clay,soc,bdod}/*_mean.vrt` (checked) | none | texture, organic carbon, bulk density, 250 m, six depths | 00, 01, 08, 12, 16 |
| USCRN | NCEI `pub/data/uscrn/products/hourly02/` (`CRNH0203-YYYY-IA_Des_Moines_17_E.txt`) | none | hourly soil moisture at 5, 10, 20, 50 and 100 cm | 00, 05, 13, 21, 22, 24 |
| SCAN | NRCS AWDB REST API, station `2031:IA:SCAN` Ames | none | hourly soil moisture at 2, 4, 8, 20 and 40 inches | 00, 21, 22, 24 |
| SMAPVEX16 Iowa | NSIDC `SV16I_TNET`, `SV16I_PNET`, `SV16I_TPSM` v001 | Earthdata | dense field soil moisture, May to September 2016 | 12, 13, 21 |
| ISMN (optional) | ismn.earth download | ISMN (free) | many networks in one format | 00, 21, 22 |
| USGS streamflow | `dataretrieval`, site 05470000 South Skunk River near Ames (checked) | none | daily discharge | 28 |
| County yields | USDA NASS Quick Stats (Iowa corn) | NASS key (free) | county yield by year | 26 |
| Boundaries | US Census TIGER counties; geoBoundaries ADM2 for Bangladesh | none | zones for drought and yield tables | 25, 26, capstones |

## Library: `geolattice.rsense.soil_moisture`

Status (25 November 2026): sections 1 and 2 (20 functions, soil water and microwave physics)
are implemented and tested in `tests/rsense/test_soil_moisture.py` (25 tests). The other 87
functions have their final signatures and docstrings and raise `NotImplementedError`; each is
implemented with the notebook that first needs it, in the order of the table below, with tests
added at the same time. `PRODUCTS` lists the identifiers in the data inventory above.

| Section | Functions | Status | Notebooks |
|---|---|---|---|
| Soil water | `saxton_rawls`, `porosity_from_texture`, `porosity_from_bulk_density`, `volumetric_from_gravimetric`, `degree_of_saturation`, `saturation_to_volumetric`, `soil_water_status` | implemented | 00, 08, 12, 24, 25 |
| SoilGrids | `load_soilgrids` | planned | 00, 01, 12, 16 |
| Dielectric and emission | `free_water_dielectric`, `dobson_dielectric`, `mironov_dielectric`, `topp`, `topp_inverse`, `penetration_depth`, `fresnel_reflectivity`, `rough_reflectivity`, `vwc_from_ndvi`, `vod_from_vwc`, `tau_omega`, `invert_tau_omega`, `tb_sensitivity` | implemented | 00, 01, 07 |
| Products and access | `product_info`, `product_table`, `ease2_grid`, `ease2_index`, `search_product`, `download_product`, `open_product`, `to_cache` | planned | 02, 03, then all |
| Readers | `read_smap`, `read_smos`, `read_amsr2_lprm`, `read_ascat`, `read_esa_cci`, `read_era5_land`, `read_clms_ssm`, `read_nisar_sm` | planned | 04 to 10, 14, 15 |
| Masks, cubes, trends | `mask_quality`, `decode_flags`, `daily_cube`, `harmonize`, `sm_climatology`, `sm_anomaly`, `trend` | planned | 04, 06, 09, 10 |
| Optical and thermal | `str_transform`, `optram_edges`, `optram`, `ts_vi_edges`, `tvdi`, `soil_evaporative_efficiency` | planned | 01, 16, 17 |
| SAR retrieval | `s1_soil_moisture_cube`, `sar_masks`, `normalize_incidence`, `change_detection_references`, `change_detection`, `short_term_change_detection`, `relative_to_volumetric`, `water_cloud_model`, `calibrate_wcm`, `invert_wcm`, `oh_model`, `dubois_model`, `iem_lut`, `invert_lut` | planned | 11 to 14 |
| Downscaling | `predictor_stack`, `dispatch`, `thermal_inertia_downscale`, `spatial_blocks`, `ml_downscale`, `explain_downscaling`, `unet_downscale`, `predict_unet` | planned | 16 to 19 |
| Gap filling and fusion | `gap_fraction`, `fill_climatology_anomaly`, `fill_dineof`, `fill_gp`, `merge_products` | planned | 20 |
| In situ | `read_ismn`, `read_network`, `match_stations`, `extract_at_stations`, `representativeness` | planned | 21 |
| Validation | `temporal_match`, `rescale`, `validation_metrics`, `validation_report`, `plot_validation`, `triple_collocation`, `triple_collocation_map`, `tc_weights` | planned | 10, 22, 23 |
| Root zone | `exponential_filter`, `optimize_t`, `bucket_model`, `enkf_assimilate` | planned | 24 |
| Drought and agriculture | `sm_percentile`, `ssi`, `swdi`, `smapi`, `drought_events`, `drought_class`, `lagged_correlation`, `irrigation_signal`, `irrigation_events`, `crop_stage_features` | planned | 25 to 27 |
| Hydrology | `antecedent_moisture`, `event_response`, `hazard_indicator` | planned | 28 |

`read_network` needs `"USCRN"` and `"SCAN"` readers (public), not only ISMN, so that notebooks
00, 21, 22 and 24 run without an ISMN account. `read_ascat` should accept ESA CCI ACTIVE as the
no-account ASCAT source.

## Notebook list

Status (25 November 2026): planned; none built yet. Notebooks 00 and 01 can be built now, since
they use only implemented functions (plus a direct SoilGrids read until `load_soilgrids` exists).

| Notebook | Course | Part | Main data | Main `sm` functions |
|---|---|---|---|---|
| 00 soil moisture concepts | 1.1 | I | USCRN, SCAN, SoilGrids | `saxton_rawls`, `soil_water_status`, `topp` |
| 01 microwave remote sensing | 2.1 | I | SoilGrids clay, MODIS NDVI | `mironov_dielectric`, `dobson_dielectric`, `tau_omega`, `tb_sensitivity`, `invert_tau_omega` |
| 02 missions and products | 3.1 | I | CMR collection metadata | `product_table`, `product_info` |
| 03 environment and data access | 4.1 | I | one granule of each product | `search_product`, `open_product`, `ease2_grid`, `harmonize`, `to_cache` |
| 04 SMAP L3 Enhanced | 5.1 | II | `SPL3SMP_E` 2016 to 2025, Iowa | `read_smap`, `mask_quality`, `daily_cube`, `sm_climatology` |
| 05 SMAP L4 root zone | 5.2 | II | `SPL4SMGP`, gridMET, USCRN | `read_smap`, `sm_anomaly` |
| 06 SMOS | 6.1 | II | `NSIDC-0800`, `SPL3SMP_E` | `read_smos`, `harmonize`, `validation_metrics` |
| 07 AMSR2 LPRM | 7.1 | II | `LPRM_AMSR2_DS_A_SOILM3`, MODIS NDVI | `read_amsr2_lprm`, `tau_omega` |
| 08 ASCAT | 8.1 | II | ESA CCI ACTIVE (H SAF optional), SoilGrids | `read_ascat`, `saturation_to_volumetric`, `porosity_from_texture` |
| 09 ESA CCI trends | 9.1 | II | CCI COMBINED 1978 to 2024, Bangladesh | `read_esa_cci`, `trend`, `sm_anomaly` |
| 10 CCI vs ERA5-Land | 9.2 | II | CCI COMBINED, ERA5-Land | `read_era5_land`, `temporal_match`, `validation_metrics` |
| 11 Sentinel-1 preprocessing | 10.1 | III | `sentinel-1-rtc` 2024, DEM, WorldCover | `s1_soil_moisture_cube`, `sar_masks`, `normalize_incidence` |
| 12 change detection | 11.1 | III | notebook 11 cube, SoilGrids | `change_detection_references`, `change_detection`, `relative_to_volumetric` |
| 13 Water Cloud Model | 12.1 | III | Sentinel-1 and Sentinel-2 2016, SMAPVEX16 | `water_cloud_model`, `calibrate_wcm`, `invert_wcm`, `oh_model`, `iem_lut` |
| 14 operational high-resolution products | 13.1 | III | `SPL2SMAP_S`, notebook 12 maps (CLMS optional) | `read_smap`, `harmonize`, `validation_metrics` |
| 15 NISAR | 14.1 | III | `NISAR_L3_SME2_PROVISIONAL_V1`, notebook 12, `SPL3SMP_E` | `read_nisar_sm`, `mask_quality`, `harmonize` |
| 16 optical and thermal predictors | 15.1 | IV | MODIS LST and NDVI, Sentinel-2, SoilGrids, DEM | `optram`, `tvdi`, `soil_evaporative_efficiency`, `predictor_stack` |
| 17 DisPATCh | 16.1 | IV | `SPL3SMP_E`, notebook 16 stack | `dispatch`, `thermal_inertia_downscale` |
| 18 XGBoost downscaling | 17.1 | IV | notebook 16 stack | `spatial_blocks`, `ml_downscale`, `explain_downscaling` |
| 19 U-Net downscaling | 17.2 | IV | SMAP and Sentinel-1 pairs | `unet_downscale`, `predict_unet` |
| 20 gap filling and fusion | 18.1 | IV | SMAP, notebook 12, notebook 18 | `fill_climatology_anomaly`, `fill_dineof`, `merge_products` |
| 21 in situ reference data | 19.1 | V | USCRN, SCAN, SMAPVEX16 (ISMN optional) | `read_network`, `match_stations`, `extract_at_stations`, `representativeness` |
| 22 validation metrics | 20.1 | V | SMAP L3E, CCI, notebook 12 vs notebook 21 stations | `temporal_match`, `rescale`, `validation_report`, `plot_validation` |
| 23 triple collocation | 21.1 | V | SMAP L3E, CCI ACTIVE, GLDAS Noah (CONUS, 0.25 degree) | `triple_collocation_map`, `tc_weights` |
| 24 root-zone soil moisture | 22.1 | VI | SMAP L3E, SMAP L4, USCRN 50 and 100 cm, gridMET | `exponential_filter`, `optimize_t`, `bucket_model`, `enkf_assimilate` |
| 25 agricultural drought | 23.1 | VI | SMAP L3E, gridMET, MODIS NDVI, TIGER counties | `sm_percentile`, `ssi`, `swdi`, `drought_events`, `lagged_correlation` |
| 26 crop yield | 24.1 | VI | NASS county corn yields, SMAP, CDL | `crop_stage_features`, `sm_anomaly` |
| 27 irrigation mapping | 24.2 | VI | SMAP L3E, ERA5-Land, Nebraska | `irrigation_signal`, `irrigation_events` |
| 28 floods and hazards | 25.1 | VI | SMAP L4, gridMET, USGS 05470000 | `antecedent_moisture`, `event_response`, `hazard_indicator` |

Capstones in `09_14_Capstones`:

| Notebook | Course | Main data |
|---|---|---|
| `09_14_18_capstone_rabi_season_drought_in_bangladesh.ipynb` | 26.1 | SMAP L3E, Sentinel-1 RTC change detection, CCI, geoBoundaries districts, district wheat yields (BBS yearbook tables) |
| `09_14_19_capstone_field_scale_irrigation_mapping_in_the_us_corn_belt.ipynb` | 26.2 | Sentinel-1 WCM with Sentinel-2, irrigation events, USCRN and SCAN |
| `09_14_20_capstone_county_soil_moisture_drought_dashboard_for_new_york_state.ipynb` | 26.3 | ESA CCI, SMAP L4, TIGER counties, Streamlit app in `apps/` |
| `09_14_21_capstone_nisar_vs_sentinel_1_soil_moisture_intercomparison.ipynb` | 26.4 | NISAR provisional, Sentinel-1 change detection, SMAP L3E (triple collocation), Iowa |

Each capstone writes an executed notebook, an HTML report, a short methods write-up and COG or
Zarr outputs under `Data/08_remote_sensing/08_13_soil_moisture/outputs/`.

## Part I. Foundations

### 09_13_00_soil_moisture_concepts_and_why_it_matters.ipynb
Soil Moisture: Concepts and Why It Matters
- Overview: volumetric and gravimetric water content, porosity, degree of saturation, field
  capacity, wilting point and plant available water; surface (0 to 5 cm) vs root zone
  (0 to 100 cm); sensing depth vs agronomic depth; soil moisture in the water, energy and
  carbon cycles; drought, floods, yield and fire risk.
- In situ methods: gravimetric sampling, TDR and the Topp equation, capacitance probes,
  cosmic-ray neutron sensing, GNSS-R; what each one measures and its footprint.
- Hands on: hourly USCRN Des Moines 17 E soil moisture at five depths for 2024, and SCAN Ames;
  how the wetting front reaches each depth after a storm.
- SoilGrids sand, clay, organic carbon and bulk density profiles at the stations; `saxton_rawls`
  wilting point, field capacity, saturation and ksat by depth; `soil_water_status` shows when the
  root zone fell below 50 % of available water in 2024.

### 09_13_01_microwave_remote_sensing_of_soil_moisture.ipynb
Microwave Remote Sensing of Soil Moisture
- Overview: why water dominates the permittivity of soil; passive emission vs active
  backscatter; L-, C- and X-band; confounders (vegetation optical depth, roughness, RFI, frozen
  soil, open water, topography); optical and thermal methods as a complement.
- `free_water_dielectric` (relaxation near 17 GHz), `dobson_dielectric` and `mironov_dielectric`
  for the Iowa SoilGrids texture; `topp` as the field instrument view; `penetration_depth` by band.
- `fresnel_reflectivity` and `rough_reflectivity` against incidence angle; `tau_omega` for bare
  soil and crops, with `vwc_from_ndvi` and `vod_from_vwc` from a MODIS NDVI season.
- Notebook 2.1 figure: `tb_sensitivity`, Tb and dTb/dtheta as VOD rises; `invert_tau_omega`
  round trip, and the error from a 20 % wrong VOD.
- A short preview of the LST-NDVI triangle and OPTRAM (implemented in notebook 16).

### 09_13_02_the_mission_and_product_landscape.ipynb
The Mission and Product Landscape
- Overview: SMAP, SMOS, AMSR2, ASCAT, Sentinel-1, NISAR and ESA CCI; model references (SMAP
  L4, ERA5-Land, GLDAS); product levels L1 to L4, EASE-Grid 2.0 and regular grids, latency,
  revisit and flags.
- `product_table(from_cmr=True)`: collection versions, periods and resolutions straight from
  CMR, next to the `PRODUCTS` registry; a timeline figure of mission records.

### 09_13_03_python_environment_and_data_access.ipynb
Python Environment and Data Access
- Environment and project layout; Earthdata login in `~/.netrc`; caching to Zarr and COG.
- One call per source: `search_product` and `open_product` for SMAP, SMOS, AMSR2, CCI,
  ERA5-Land, Sentinel-1 RTC and NISAR, each returning a tidy `time, y, x` Dataset for the Iowa
  AOI; HDF5 and NetCDF structure with `h5py`; EASE-Grid 2.0 with `ease2_grid` and `ease2_index`.
- `harmonize` onto the 9 km grid, `dask` chunking for multi-year cubes, `to_cache`.
- Output: `sm_data.py` functions that every later notebook calls.

## Part II. Coarse-resolution global products

### 09_13_04_smap_l3_enhanced_daily_cube_and_climatology.ipynb
SMAP L3 Enhanced: a Daily Cube and its Climatology
- Product family (L2, L3 36 km, L3 Enhanced 9 km, L4); HDF5 groups, AM and PM overpasses.
- `read_smap`, `decode_flags` and `mask_quality` (recommended quality, frozen ground, water);
  how much of each season survives the mask.
- `daily_cube` for 2016 to 2025 over Iowa; `sm_climatology` (31-day day-of-year window); the
  2023 dry summer as an anomaly.

### 09_13_05_smap_l4_surface_and_root_zone_soil_moisture.ipynb
SMAP L4: Surface and Root-Zone Soil Moisture
- What L4 assimilates and why it is gap free; 3-hourly fields.
- Surface vs root zone at a cropland pixel after rain events (gridMET precipitation): lag
  and damping; comparison with USCRN 5 cm and 50 to 100 cm.

### 09_13_06_smos_and_smap_over_a_shared_period.ipynb
SMOS and SMAP over a Shared Period
- SMOS L-band interferometric radiometer; RFI and its geography.
- `NSIDC-0800` (SMOS Tb with the SMAP algorithm) vs `SPL3SMP_E`; the CATDS L3 product as an
  option; `harmonize` and agreement statistics by season.

### 09_13_07_amsr2_lprm_soil_moisture_and_vegetation_optical_depth.ipynb
AMSR2 LPRM Soil Moisture and Vegetation Optical Depth
- C- and X-band sensitivity; the Land Parameter Retrieval Model; VOD as a by-product.
- C1 soil moisture and VOD against MODIS NDVI through the season; `tau_omega` shows where
  vegetation saturates the C-band signal.

### 09_13_08_ascat_and_the_tu_wien_change_detection_approach.ipynb
ASCAT and the TU Wien Change Detection Approach
- C-band scatterometer; dry and wet references; degree of saturation; Soil Water Index.
- ESA CCI ACTIVE (ASCAT based) by default, H SAF SSM with the `ascat` package as an option;
  `saturation_to_volumetric` with `porosity_from_texture` from SoilGrids; comparison with SMAP.

### 09_13_09_esa_cci_long_term_trends_and_anomalies.ipynb
ESA CCI: Long-Term Trends and Anomalies
- ACTIVE, PASSIVE and COMBINED; merging logic, sensor changes, uncertainty and flags.
- Bangladesh 1978 to 2024: `trend` (Mann-Kendall and Sen's slope per pixel), anomaly maps,
  and how sensor changes show up in the record.

### 09_13_10_esa_cci_vs_era5_land.ipynb
ESA CCI vs ERA5-Land
- What a reanalysis soil layer is (ERA5-Land layer 1 is 0 to 7 cm).
- Spatial and temporal agreement of CCI COMBINED and ERA5-Land over Iowa and Bangladesh:
  `temporal_match`, `validation_metrics` on raw values and anomalies.

## Part III. High-resolution SAR soil moisture

### 09_13_11_sentinel_1_preprocessing_for_soil_moisture.ipynb
Sentinel-1 Preprocessing for Soil Moisture
- GRD vs RTC; calibration, terrain flattening, speckle; ready-made RTC sources.
- `s1_soil_moisture_cube` (Planetary Computer `sentinel-1-rtc`, one relative orbit, 2024,
  multitemporal speckle filter); `normalize_incidence`; `sar_masks` for water, urban, frozen soil,
  steep slopes, layover and shadow with the Copernicus DEM and WorldCover.

### 09_13_12_change_detection_soil_moisture_from_sentinel_1.ipynb
Change Detection Soil Moisture from Sentinel-1
- The TU Wien approach per pixel and short-term change detection, with their assumptions.
- `change_detection_references`, `change_detection`, `short_term_change_detection` scaled by
  SMAP; `relative_to_volumetric` with SoilGrids porosity; check at USCRN and SCAN.

### 09_13_13_water_cloud_model_and_surface_scattering_inversion.ipynb
Water Cloud Model and Surface Scattering Inversion
- WCM; Oh, Dubois and IEM models and their validity ranges; Sentinel-2 NDVI and LAI as the
  vegetation descriptor.
- 2016 Sentinel-1 and Sentinel-2 with the SMAPVEX16 temporary network and ThetaProbe points:
  `calibrate_wcm` per crop, `invert_wcm` maps, error by crop stage; `iem_lut` and `invert_lut`
  on bare fields.

### 09_13_14_operational_high_resolution_products.ipynb
Operational High-Resolution Products
- SPL2SMAP_S (3 km and 1 km) and the CLMS 1 km SSM (Europe).
- SPL2SMAP_S vs the notebook 12 change detection maps over Iowa; CLMS over REMEDHUS as an
  optional section.

### 09_13_15_nisar_l_band_soil_moisture.ipynb
NISAR L-Band Soil Moisture
- Mission: L-band (NASA) and S-band (ISRO), 12-day repeat; the NASA L3 product at 200 m and the
  ISRO 100 m product over India.
- `asf_search` for `NISAR_L3_SME2_PROVISIONAL_V1` over Iowa (66 granules on 25 November 2026),
  `read_nisar_sm` with its flags; comparison with Sentinel-1 change detection and SMAP L3E.
- Note: re-check the collection version before release; the beta collection
  `NISAR_L3_SME2_BETA_V1` is kept for comparison.

## Part IV. Downscaling, gap filling and fusion

### 09_13_16_optical_and_thermal_predictors.ipynb
Optical and Thermal Predictors
- MODIS LST and NDVI, Sentinel-2 SWIR; `str_transform`, `optram_edges`, `optram`; `ts_vi_edges`,
  `tvdi`; `soil_evaporative_efficiency`.
- `predictor_stack`: a 1 km stack (LST, NDVI, OPTRAM, SoilGrids texture, slope, TWI, land cover)
  aligned to the SMAP 9 km grid.

### 09_13_17_dispatch_physics_based_downscaling.ipynb
DisPATCh: Physics-Based Downscaling
- Soil evaporative efficiency as the link between LST and near-surface moisture.
- `dispatch` of SMAP 9 km to 1 km over Iowa; `thermal_inertia_downscale` as the alternative; check
  that the fine map averages back to the coarse value; check at stations.

### 09_13_18_machine_learning_downscaling_with_spatial_cross_validation.ipynb
Machine Learning Downscaling with Spatial Cross-Validation
- Random forest, XGBoost and LightGBM on the predictor stack; why random CV leaks.
- `spatial_blocks`, `ml_downscale` (XGBoost, residual correction), `explain_downscaling` (SHAP
  and partial dependence).

### 09_13_19_deep_learning_downscaling_with_a_u_net.ipynb
Deep Learning Downscaling with a U-Net
- CNN and U-Net super-resolution; residual learning on SMAP and Sentinel-1 pairs.
- `unet_downscale` (GPU with CPU fallback) and `predict_unet`; comparison with notebook 18.

### 09_13_20_gap_filling_and_multi_sensor_fusion.ipynb
Gap Filling and Multi-Sensor Fusion
- Temporal and spatial gaps; climatology-anomaly interpolation, DINEOF-style EOFs, Gaussian
  processes.
- `gap_fraction`, `fill_climatology_anomaly`, `fill_dineof`; `merge_products` of SMAP, Sentinel-1
  and ML downscaling into a seamless daily 1 km cube, weights from notebook 23.

## Part V. Validation and uncertainty

### 09_13_21_in_situ_reference_data.ipynb
In Situ Reference Data
- ISMN, USCRN, SCAN and campaign networks; quality flags and depth selection; sparse networks.
- `read_network` for USCRN and SCAN (ISMN optional), SMAPVEX16 for density; `match_stations` to
  SMAP, CCI and Sentinel-1 pixels; `representativeness` of points within 9 km cells.

### 09_13_22_validation_metrics.ipynb
Validation Metrics
- Temporal matching, rescaling (mean-std, CDF matching), anomalies; bias, RMSE, ubRMSE, R and
  anomaly R; the SMAP 0.04 m3 m-3 ubRMSE target.
- `validation_report` for SMAP L3E, CCI and the notebook 12 retrievals, stratified by land cover
  and season, with bootstrap confidence intervals; `plot_validation`.

### 09_13_23_triple_collocation_and_error_characterization.ipynb
Triple Collocation and Error Characterization
- Classical and extended triple collocation; assumptions and failure cases.
- `triple_collocation_map` for SMAP L3E (passive), CCI ACTIVE (ASCAT) and GLDAS Noah (model) over
  CONUS at 0.25 degree; `tc_weights` saved for notebook 20.

## Part VI. Applications

### 09_13_24_root_zone_soil_moisture_estimation.ipynb
Root-Zone Soil Moisture Estimation
- Exponential filter and the characteristic time T; a bucket model; the Ensemble Kalman Filter.
- `exponential_filter` and `optimize_t` against USCRN 50 and 100 cm; `bucket_model` with
  gridMET; `enkf_assimilate` of SMAP into the bucket; comparison with SMAP L4 root zone.

### 09_13_25_agricultural_drought_monitoring.ipynb
Agricultural Drought Monitoring
- Percentiles, SSI, SWDI and SMAPI; SPI and SPEI; vegetation response and lags.
- `sm_percentile`, `ssi`, `swdi` (with `saxton_rawls` limits), `drought_events` and `drought_class`
  for the 2023 Iowa season; `lagged_correlation` with SPI and MODIS VCI; county maps.

### 09_13_26_soil_moisture_and_crop_yield.ipynb
Soil Moisture and Crop Yield
- Moisture at critical stages as a yield predictor; panel regression and gradient boosting;
  a causal question answered with double ML (`geolattice.spatial_analysis.spatialcausal`).
- `crop_stage_features` for Iowa counties (CDL corn mask, NASS yields).

### 09_13_27_irrigation_mapping_from_satellite_and_model_soil_moisture.ipynb
Irrigation Mapping from Satellite and Model Soil Moisture
- Why satellites see irrigation and models do not.
- `irrigation_signal` (SMAP minus ERA5-Land after CDF matching) over the Nebraska Platte valley
  with rainfed Iowa as a control; `irrigation_events` on Sentinel-1 retrievals.

### 09_13_28_floods_and_hazards_antecedent_moisture.ipynb
Floods and Hazards: Antecedent Moisture
- Antecedent moisture and runoff; landslide and wildfire indicators.
- `antecedent_moisture` and `event_response` for storms at USGS 05470000 (South Skunk River near
  Ames) with gridMET rain and SMAP L4; `hazard_indicator`.
