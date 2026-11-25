# Satellite-Based Soil Moisture with Python: Tutorial Outline

Nov 25, 2026 · @Zia

## Overview

A 26-module, seven-part Python tutorial that takes learners from microwave physics to operational soil moisture products, field-scale SAR retrievals, downscaling, validation and applied drought and agriculture analytics. Each module ships as an executed Jupyter notebook plus a rendered HTML page.

**Audience.** Graduate students, agronomists, hydrologists and data scientists who know basic Python and want to work with satellite soil moisture end to end.

**Prerequisites.** Python basics (NumPy, pandas), introductory GIS (CRS, rasters, vectors), and a free NASA Earthdata login. Optional: Google Earth Engine and Copernicus Data Space accounts.

**Learning outcomes.** By the end, learners can:

1. Explain how L-, C- and X-band microwave signals relate to volumetric soil moisture, vegetation and roughness.
2. Discover, download and harmonize SMAP, SMOS, AMSR2, ASCAT, ESA CCI, Sentinel-1 and NISAR products in Python.
3. Retrieve field-scale soil moisture from SAR with change detection and Water Cloud Model inversion.
4. Downscale coarse products to 1 km and below with thermal/optical and machine learning methods.
5. Validate any product against ISMN in situ data using standard metrics and triple collocation.
6. Build drought, irrigation, crop yield and root-zone analyses on top of the gridded data.

**Module template.** Every module follows the same structure: learning objectives, concepts (short), data used, hands-on notebook(s), exercises, and further reading.

**Core environment.** `xarray`, `rioxarray`, `h5netcdf`/`h5py`, `netCDF4`, `geopandas`, `earthaccess`, `pystac-client`, `odc-stac`, `asf_search`, `earthengine-api`/`geemap`, `pytesmo`, `ismn`, `scikit-learn`, `xgboost`, `pytorch`, `matplotlib`, `cartopy`, `hvplot`. A `conda` environment file and a CPU/GPU note are provided in Module 4.

## Part I. Foundations (Modules 1 to 4)

Part I builds the physical and computational base every later module relies on.

**Module 1. Soil moisture: concepts and why it matters**

- Definitions: volumetric (m³/m³) vs gravimetric water content, degree of saturation, field capacity, wilting point, plant-available water.
- Surface (0 to 5 cm) vs root-zone (0 to 100 cm) soil moisture; sensing depth vs agronomic depth.
- Role in the water, energy and carbon cycles; land-atmosphere coupling; drought, floods, yield, fire risk.
- In situ methods: gravimetric, TDR, capacitance, cosmic-ray neutron (COSMOS), GNSS-R.
- Notebook 1.1: load an ISMN station time series and a SoilGrids texture profile; compute field capacity and wilting point with pedotransfer functions.

**Module 2. Microwave remote sensing of soil moisture**

- Dielectric constant of wet soil; dielectric mixing models (Dobson, Mironov) and the role of texture.
- Passive (brightness temperature, emissivity, tau-omega model) vs active (backscatter σ⁰) sensing.
- Frequency effects: L-band (\~1.4 GHz) vs C-band (\~5.4 GHz) vs X-band; penetration depth.
- Confounders: vegetation optical depth (VOD), vegetation water content, surface roughness, RFI, frozen soil, open water, topography.
- Optical/thermal approaches as a complement: LST-NDVI triangle, OPTRAM.
- Notebook 2.1: implement the Mironov dielectric model and a simple tau-omega forward model; plot Tb sensitivity to soil moisture under rising VOD.

**Module 3. The mission and product landscape**

- Passive: SMAP (L-band radiometer, 2015 to present), SMOS (L-band, 2009 to present), AMSR2 (C/X-band).
- Active: ASCAT (C-band scatterometer), Sentinel-1 (C-band SAR), NISAR (L/S-band SAR, launched July 2025).
- Merged climate records: ESA CCI SM (ACTIVE, PASSIVE, COMBINED; 1978 onward).
- Model and reanalysis references: SMAP L4, ERA5-Land, GLDAS, NLDAS.
- Product levels (L1 to L4), grids (EASE-Grid 2.0, WGS84 regular), latency, revisit, quality flags.
- Notebook 3.1: build a comparison table of products programmatically from CMR collection metadata.

**Module 4. Python environment and data access**

- Environment setup (conda/mamba, Jupyter, optional GPU); project layout and caching.
- NASA Earthdata with `earthaccess` (search, download, cloud-direct streaming); ASF with `asf_search`.
- Copernicus Data Space and STAC with `pystac-client` and `odc-stac`; Google Earth Engine and `geemap`; openEO.
- Reading HDF5/NetCDF with `h5py` and `xarray`; EASE-Grid 2.0 geolocation; reprojection with `rioxarray`.
- Chunking with `dask` for large time series; writing Zarr and Cloud-Optimized GeoTIFF.
- Notebook 4.1: one helper function per data source that returns a tidy `xarray.Dataset` for an AOI and date range.

## Part II. Coarse-resolution global products (Modules 5 to 9)

Part II covers the global 9 to 40 km products that underpin most climate and drought work.

**Module 5. SMAP**

- Product family: L2/L3 radiometer (36 km), Enhanced L3 (9 km), L4 surface and root-zone (9 km, 3-hourly, assimilated).
- HDF5 structure, AM/PM overpasses, retrieval quality flags, recommended-quality masks.
- Notebook 5.1: download SMAP L3 Enhanced for a region, mask by quality flag, build a daily cube and a monthly climatology.
- Notebook 5.2: SMAP L4 root-zone soil moisture time series for a cropland site; compare surface vs root-zone dynamics after rainfall.

**Module 6. SMOS**

- ESA L2 and CATDS L3 products (\~25 km EASE grid); RFI flags and their geography.
- Notebook 6.1: read SMOS L3 NetCDF, filter RFI, and compare with SMAP over a shared overlap period.

**Module 7. AMSR2 and the LPRM algorithm**

- C/X-band sensitivity, LPRM retrievals, VOD as a by-product.
- Notebook 7.1: AMSR2 LPRM soil moisture and VOD; show where vegetation saturates the signal.

**Module 8. ASCAT and the TU Wien change detection approach**

- C-band scatterometer, H SAF surface soil moisture (degree of saturation), Soil Water Index (SWI).
- Notebook 8.1: read ASCAT time series with the `ascat` package; convert degree of saturation to volumetric using porosity.

**Module 9. ESA CCI SM long-term climate record**

- ACTIVE, PASSIVE and COMBINED products (0.25°); merging logic; uncertainty and flag layers; version changes.
- Notebook 9.1: 40+ year trend analysis (Mann-Kendall, Sen's slope) and anomaly maps for a country of choice.
- Notebook 9.2: compare CCI COMBINED with ERA5-Land layer 1 for spatial and temporal agreement.

## Part III. High-resolution SAR soil moisture (Modules 10 to 14)

Part III moves from 9 km grids to field scale (10 m to 1 km) using C- and L-band SAR.

**Module 10. Sentinel-1 preprocessing for soil moisture**

- GRD vs RTC products; orbit, thermal noise removal, calibration to σ⁰, terrain flattening, speckle filtering.
- Ready-made RTC sources (OPERA RTC-S1 via ASF, Sentinel-1 RTC on Microsoft Planetary Computer, GEE `COPERNICUS/S1_GRD`).
- Incidence angle normalization; VV vs VH; masking water, urban, frozen and steep terrain.
- Notebook 10.1: build an analysis-ready VV/VH time-series cube for an agricultural AOI.

**Module 11. Change detection retrieval**

- TU Wien style approach: dry and wet reference backscatter per pixel; relative soil moisture index.
- Short-term change detection and its assumptions (stable vegetation and roughness between acquisitions).
- Notebook 11.1: implement change detection on the Sentinel-1 cube; scale to volumetric with SoilGrids porosity.

**Module 12. Physical and semi-empirical inversion**

- Water Cloud Model (WCM) for vegetation; surface scattering models (Oh, Dubois, IEM) and their validity ranges.
- Vegetation descriptors from Sentinel-2 (NDVI, LAI) fused with Sentinel-1.
- Look-up-table inversion and calibration with `scipy.optimize`.
- Notebook 12.1: calibrate the WCM on in situ points; invert to soil moisture maps; report error by crop stage.

**Module 13. Operational high-resolution products**

- SMAP/Sentinel-1 active-passive product (SPL2SMAP\_S, 3 km and 1 km).
- Copernicus Land Monitoring Service surface soil moisture at 1 km (Sentinel-1 based, Europe).
- Notebook 13.1: compare SPL2SMAP\_S with the Module 11 retrievals over the same AOI.

**Module 14. NISAR L-band soil moisture**

- Mission overview: L-band (NASA) and S-band (ISRO) SAR, 12-day repeat, ascending and descending.
- NASA L3 soil moisture (200 m, beta release via ASF DAAC); ISRO 100 m L4 product via Bhoonidhi (India).
- Product structure (HDF5), flags for dense vegetation, urban, water and snow.
- Notebook 14.1: search and read NISAR soil moisture with `asf_search` and `h5py`; compare with Sentinel-1 change detection and SMAP L3 Enhanced.
- Note for authors: re-check product version and availability before publishing, since NISAR products are still maturing.

## Part IV. Downscaling, gap filling and fusion (Modules 15 to 18)

Part IV closes the gap between coarse daily products and the field-scale detail users need.

**Module 15. Optical and thermal predictors**

- MODIS/VIIRS LST and NDVI, Landsat and Sentinel-2 indices, Sentinel-3 SLSTR.
- Temperature-Vegetation Dryness Index (TVDI), soil evaporative efficiency, OPTRAM.
- Static covariates: SoilGrids texture, DEM-derived terrain (slope, TWI), land cover.
- Notebook 15.1: assemble a 1 km predictor stack aligned to the SMAP 9 km grid.

**Module 16. Physics-based downscaling**

- DisPATCh (soil evaporative efficiency) and thermal inertia methods.
- Notebook 16.1: implement DisPATCh-style downscaling of SMAP from 9 km to 1 km.

**Module 17. Machine learning and deep learning downscaling**

- Random forest, XGBoost and LightGBM regression on predictor stacks; spatial cross-validation to avoid leakage.
- CNN and U-Net super-resolution; residual learning on SMAP/Sentinel-1 pairs.
- Explainability with SHAP and partial dependence.
- Notebook 17.1: XGBoost downscaling with spatial block CV and SHAP.
- Notebook 17.2: PyTorch U-Net downscaling on GPU (CPU fallback).

**Module 18. Gap filling and multi-sensor fusion**

- Temporal gaps (revisit, RFI, frozen soil) and spatial gaps (swath edges, dense vegetation).
- Methods: climatology-anomaly interpolation, DINEOF-style EOF reconstruction, Gaussian processes, ConvLSTM.
- Merging products using triple collocation weights (preview of Module 21).
- Notebook 18.1: produce a seamless daily 1 km soil moisture cube by fusing SMAP, Sentinel-1 and ML downscaling.

## Part V. Validation and uncertainty (Modules 19 to 21)

Part V teaches learners to judge any product before trusting it.

**Module 19. In situ reference data**

- International Soil Moisture Network (ISMN): download, the `ismn` reader, quality flags, depth selection.
- Regional networks (USCRN, SCAN, COSMOS-UK) and sparse-network caveats.
- Point-to-pixel scale mismatch and representativeness error.
- Notebook 19.1: extract all ISMN stations in an AOI and match them to SMAP, CCI and Sentinel-1 pixels.

**Module 20. Validation metrics with pytesmo**

- Temporal matching, scaling (mean-std, CDF matching), and anomaly computation.
- Metrics: bias, RMSE, unbiased RMSE (ubRMSE), Pearson and Spearman R, anomaly R; SMAP's 0.04 m³/m³ ubRMSE target.
- Stratifying error by land cover, VOD, aridity and season.
- Notebook 20.1: full validation report for three products, with bootstrap confidence intervals.

**Module 21. Triple collocation and error characterization**

- Classical and extended triple collocation; assumptions and failure cases.
- Error maps without in situ data (e.g., SMAP + ASCAT + ERA5-Land).
- Notebook 21.1: global triple collocation error maps and their use as fusion weights in Module 18.

## Part VI. Applications (Modules 22 to 25)

Part VI turns validated soil moisture into decisions for agriculture, drought and hydrology.

**Module 22. Root-zone soil moisture estimation**

- Exponential filter (SWI) with tuned characteristic time T; comparison with SMAP L4 root zone.
- Intro to data assimilation: a 1-D bucket model plus an Ensemble Kalman Filter (`filterpy` or NumPy).
- Notebook 22.1: derive root-zone estimates from surface retrievals and validate at deep ISMN sensors.

**Module 23. Agricultural drought monitoring**

- Indices: soil moisture percentiles, standardized soil moisture index (SSI), SWDI, SMAPI.
- Linking to SPI/SPEI and vegetation response (NDVI, VCI) with lag analysis.
- Notebook 23.1: monthly drought maps and onset/termination detection for a growing season.

**Module 24. Crop yield and irrigation**

- Soil moisture anomalies at critical crop stages as yield predictors (panel regression, gradient boosting).
- Irrigation detection: satellite vs model soil moisture divergence; Sentinel-1 based field irrigation events.
- Causal framing: effect of in-season moisture stress on yield using double ML (links to PyCausalML).
- Notebook 24.1: district-level wheat or maize yield model with soil moisture features.
- Notebook 24.2: irrigated area mapping from SMAP-ERA5-Land divergence.

**Module 25. Hydrology, floods and hazards**

- Antecedent moisture and runoff response; flood potential indicators.
- Soil moisture in landslide and wildfire risk.
- Notebook 25.1: event analysis linking pre-storm soil moisture to streamflow peaks from USGS gauges (`dataretrieval`).

## Part VII. Capstone projects (Module 26)

Each capstone chains acquisition, retrieval, validation and an application into one reproducible notebook.

1. **Rabi-season drought in Bangladesh.** SMAP L3 Enhanced + Sentinel-1 change detection over wheat districts; drought percentiles; link to district yield statistics.
2. **Field-scale irrigation mapping in the US Corn Belt.** Sentinel-1 WCM retrievals with Sentinel-2 vegetation; irrigation event detection; validation against USCRN/SCAN.
3. **County-level soil moisture drought dashboard for New York State.** ESA CCI and SMAP L4 aggregated to counties; Streamlit app with anomaly maps and time series.
4. **NISAR vs Sentinel-1 intercomparison.** Same AOI, same dates; triple collocation with SMAP; discussion of L- vs C-band behavior under crops.

Deliverables per capstone: executed notebook, HTML report, a short methods write-up, and exported COG/Zarr outputs.

## Appendices

**A. Dataset quick reference** (resolutions are nominal; check current versions before publishing)

| Product | Band / type | Spatial resolution | Revisit | Period | Access | Modules |
| --- | --- | --- | --- | --- | --- | --- |
| SMAP L3 / L3 Enhanced | L-band passive | 36 km / 9 km | 2 to 3 days | 2015 to present | NSIDC via `earthaccess`, GEE | 5, 13, 23 |
| SMAP L4 SM | Model + assimilation | 9 km | 3-hourly | 2015 to present | NSIDC, GEE | 5, 22 |
| SPL2SMAP\_S | SMAP + Sentinel-1 | 3 km / 1 km | \~12 days | 2015 to present | NSIDC | 13 |
| SMOS L3 | L-band passive | \~25 km | 2 to 3 days | 2010 to present | CATDS, ESA | 6 |
| AMSR2 LPRM | C/X-band passive | 0.1° to 0.25° | daily | 2012 to present | GES DISC | 7 |
| ASCAT H SAF SSM / SWI | C-band scatterometer | 12.5 to 25 km | daily | 2007 to present | H SAF, Copernicus | 8, 22 |
| ESA CCI SM | Merged | 0.25° | daily | 1978 to present | CEDA, ESA CCI | 9, 21, 23 |
| Sentinel-1 GRD / RTC | C-band SAR | 10 to 20 m | 6 to 12 days | 2014 to present | CDSE, ASF, Planetary Computer, GEE | 10 to 12 |
| CLMS SSM 1 km | C-band SAR | 1 km | \~daily (Europe) | 2015 to present | Copernicus Land | 13 |
| NISAR soil moisture | L-band SAR | 200 m (NASA); 100 m (ISRO, India) | \~2 per 12 days | 2025 to present | ASF DAAC; Bhoonidhi | 14 |
| ERA5-Land | Reanalysis | 0.1° | hourly | 1950 to present | Copernicus CDS, GEE | 9, 21, 24 |
| ISMN | In situ | point | sub-daily | varies | ismn.earth | 19 to 21 |

**B. Package stack by part**

- Access: `earthaccess`, `asf_search`, `pystac-client`, `odc-stac`, `earthengine-api`, `geemap`, `openeo`, `cdsapi`.
- Data handling: `xarray`, `rioxarray`, `dask`, `h5py`, `netCDF4`, `zarr`, `geopandas`.
- Soil moisture specific: `pytesmo`, `ismn`, `ascat`, `pyresample`.
- Modeling: `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `pytorch`, `shap`, `filterpy`.
- Visualization and apps: `matplotlib`, `cartopy`, `hvplot`, `folium`, `streamlit`.

**C. Compute and packaging notes**

- Every notebook runs on a laptop CPU with a small AOI; Modules 17 and 18 add optional GPU paths.
- Cache downloads to Zarr; keep credentials in `~/.netrc` and environment variables, never in notebooks.
- Optional: wrap the retrieval, downscaling and validation helpers as a `soilmoisture` module in PyRSense so the tutorial and package stay in sync, as with the remote sensing series.

**Sources**

- [NISAR Beta Soil Moisture (Version 1), data.gov](https://catalog.data.gov/dataset/nisar-beta-soil-moisture-version-1)
- [ISRO: Soil moisture products at 100 m from NISAR (Feb 2026)](https://www.isro.gov.in/NisarData_soil_moisture_products.html)
- [NASA Earthdata: NISAR data release (Feb 2026)](https://earthdata.nasa.gov/data/alerts-outages/over-100000-new-data-files-available-from-nisar)
