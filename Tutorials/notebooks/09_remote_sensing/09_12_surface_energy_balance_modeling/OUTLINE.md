# Surface Energy Balance Modeling from Satellite Images with GeoLattice: Notebook Outline

Chapter folder: `Tutorials/notebooks/09_remote_sensing/09_12_surface_energy_balance_modeling/`
Banner: `Images/Banner/07_remote_sensing/09_12_surface_energy_balance_modeling.png`
Library: `geolattice.rsense.surface_energy_balance` (one module, `src/rsense/surface_energy_balance.py`),
used with `geolattice.rsense.optical` (`scale`, `qa`, `indices`, `lst`) and
`geolattice.rsense.get_images` (STAC access). Import it as
`import geolattice.rsense.surface_energy_balance as seb`.

Every notebook follows the house style: banner cell, `#` title with a one paragraph intro,
`## Overview` (theory), `## ... with GeoLattice`, Learning objectives, Data, Contents, Setup, the
working sections, and Summary and exercises. Section headings carry no numbers, every code
statement has a comment, and the markdown has no em or en dashes. Notebooks are built by
scripts in `_builders/` (now kept locally in `helper_functions/`, outside the repository; same `nb.py` / `run.py` / `sync_text.py` pattern as 08_11) and executed
before release; markdown claims are checked against the executed outputs.

Scope: the methods behind the OpenET ensemble (https://etdata.org/methods/), meaning SEBAL,
METRIC (eeMETRIC), SSEBop, geeSEBAL, PT-JPL, DisALEXI/TSEB and SIMS. Each one is written from
its equations and checked on real scenes. SEBAL and METRIC get the most space because the
other models reuse their radiation, soil heat flux and aerodynamic parts.

Installation: `geolattice[remote-sensing]` (option 5). Notebook 08 can also use `pyTSEB` for a
cross-check; notebook 11 can use an AmeriFlux account (free) for flux tower data and an OpenET
API key (free) for comparison. Both are optional, and the notebook runs without them.

## Study areas

| Site | Why | Scenes |
|---|---|---|
| Westside San Joaquin Valley, California (Five Points, Fresno and Kings counties), bbox about -120.25, 36.30, -119.95, 36.55 | Irrigated crops next to fallow and bare fields, so hot and cold anchors are easy to find; clear summer skies; USDA CDL for crop labels; METRIC and OpenET have been tested here | Landsat 8/9 C2 L2, path 42 row 35: 40 scenes under 5% cloud from April to October 2024 (checked 23 November 2026), e.g. `LC08_L2SP_042035_20240705_02_T1`, `LC09_L2SP_042035_20240729_02_T1` |
| Boro rice belt, Mymensingh, Bangladesh, bbox about 90.2, 24.3, 90.5, 24.6 | Dry season irrigated rice, the capstone | Landsat path 137 row 43, clear on 1 and 9 March 2024; more dates from January to April in 2023 and 2025 |

## Data inventory

Everything is downloaded by the notebooks on first run and cached under
`Data/08_remote_sensing/08_12_surface_energy_balance/` (new folder). Sources were checked on
23 November 2026.

| Dataset | Source | Content | Used in |
|---|---|---|---|
| Landsat 8/9 Collection 2 Level-2 | Planetary Computer `landsat-c2-l2` | SR bands 1 to 7, `ST_B10`, `ST_EMIS`, `ST_ATRAN`, `ST_DRAD`, `ST_URAD`, `QA_PIXEL`, solar angles from MTL | 01 to 08, 10 to 12 |
| Landsat Collection 2 Level-1 | Planetary Computer `landsat-c2-l1` | Band 10 DN and MTL constants, to retrieve LST from scratch | 03 |
| HLS v2.0 S30 and L30 | Planetary Computer `hls2-s30`, `hls2-l30` | Sentinel-2 and Landsat at 30 m on one grid | 09, 10 |
| Sentinel-3 SLSTR LST | Planetary Computer `sentinel-3-slstr-lst-l2-netcdf` | 1 km daily LST, for thermal sharpening | 09 |
| MODIS | Planetary Computer `modis-11A1-061` (LST), `modis-43A4-061` (NBAR), `modis-15A2H-061` (LAI), `modis-16A3GF-061` (annual ET) | 500 m to 1 km inputs and the MOD16 ET product | 09, 11 |
| Copernicus DEM GLO-30 | Planetary Computer `cop-dem-glo-30` | Elevation, slope and aspect for pressure and terrain corrected radiation | 01, 03 |
| USDA CDL 2024 | Planetary Computer `usda-cdl` | Crop types for anchor screening and per crop ET | 01, 05, 06, 10 |
| gridMET | Planetary Computer `gridmet` | Daily 4 km weather and reference ET (ETo, ETr) | 02, 06, 10 |
| ERA5 hourly | Open-Meteo archive API (no key) and Planetary Computer `era5-pds` | Hourly air temperature, humidity, wind, shortwave at overpass time | 02 to 06, 12 |
| CIMIS station 2 (Five Points) | CIMIS web API (free key, optional) | Station hourly weather and ETo, to check the gridded weather | 02, 11 |
| AmeriFlux / OpenET (optional) | AmeriFlux (free account), OpenET API (free key) | Flux tower LE and OpenET ensemble ET for validation | 11 |

## Library: `geolattice.rsense.surface_energy_balance`

Added 23 November 2026. Every function takes NumPy arrays or 2-D xarray DataArrays and returns
the same kind; model runs return an `xr.Dataset` on the input grid (CRS kept) whose
calibration history and anchors are stored as JSON attributes, so results save to NetCDF.
Units: K for the energy balance, C for reference ET, W m-2, kPa, m s-1, s m-1, mm.

| Section | Functions | Notebooks |
|---|---|---|
| Constants | `SIGMA`, `VON_KARMAN`, `CP`, `GSC`, `ALBEDO_WEIGHTS`, `latent_heat_of_vaporization` | all |
| Solar geometry | `day_of_year`, `solar_declination`, `inverse_relative_distance`, `equation_of_time`, `hour_angle`, `solar_position`, `scene_geometry` (MTL or STAC), `slope_aspect`, `cos_incidence`, `sunset_hour_angle`, `extraterrestrial_radiation_daily`, `extraterrestrial_radiation_hourly` | 01, 02, 03 |
| Atmosphere and weather | `pressure_from_elevation`, `psychrometric_constant`, `saturation_vapour_pressure`, `slope_vapour_pressure_curve`, `actual_vapour_pressure`, `air_density`, `precipitable_water`, `broadband_transmissivity`, `weather_open_meteo` (ERA5 via Open-Meteo, no key), `interpolate_weather` | 01, 02, 03 |
| Vegetation and radiation | `ndvi`, `savi`, `leaf_area_index`, `fractional_cover`, `albedo` (Liang OLI, Tasumi, Sentinel-2, MODIS weights), `albedo_from_toa`, `surface_emissivity`, `incoming_shortwave`, `air_emissivity`, `incoming_longwave`, `outgoing_longwave`, `net_radiation`, `daily_net_radiation`, `datum_temperature` | 03 |
| Soil heat and aerodynamics | `soil_heat_flux` (Bastiaanssen, METRIC), `momentum_roughness` (LAI, NDVI, height, land cover table), `wind_at_blending_height`, `wind_2m`, `friction_velocity`, `aerodynamic_resistance`, `monin_obukhov_length`, `stability_corrections` | 04 |
| Reference ET | `asce_pm_daily`, `asce_pm_hourly` (short and tall), `fao56_eto` | 02, 06, 10 |
| Anchors | `candidate_anchors` (Allen et al. 2013 percentiles, albedo and CDL screens), `select_anchors` (homogeneity filter), `anchor_at`, `anchor_report` | 05, 06 |
| SEBAL and METRIC | `calibrate_dt`, `sensible_heat_iterative`, `latent_heat_flux`, `evaporative_fraction`, `instantaneous_et`, `calibration_history`, `run_sebal`, `run_metric`, `etrf` | 05, 06 |
| SSEBop and simple models | `cold_reference_temperature`, `dt_from_climate`, `run_ssebop`, `ssebi`, `triangle_method`, `priestley_taylor_jpl` | 07 |
| Two-source | `tseb_pt` (parallel network, Kustas and Norman 1999 soil resistance, bare soil one-source branch), `partition_et` | 08 |
| Sharpening | `aggregate_lst` (emissive power mean), `thermal_sharpen` (tree, forest or boosting DMS with residual correction) | 09 |
| Upscaling | `instantaneous_to_daily`, `interpolate_etrf` (gap limited, no bottleneck needed), `seasonal_et`, `zonal_et` (raster zones or polygons) | 06, 10, 12 |
| Validation | `energy_closure`, `compare_to_tower`, `sensitivity`, `monte_carlo_anchors` | 11 |

Tests: `tests/rsense/test_surface_energy_balance.py` (33 tests) check FAO-56 examples 2, 8,
18 and 19 (pressure, Ra, daily ETo 3.9 mm/d, hourly ETo 0.63 mm/h), the anchor constraints of
SEBAL (no LE at hot, no H at cold) and METRIC (ETrF 1.05 and 0), energy conservation in TSEB,
and the edge fits, sharpening, interpolation and zonal statistics on synthetic scenes.

Check on real data (23 November 2026): `LC08_L2SP_042035_20240705_02_T1`, Westside window
(896 x 926 pixels) with Open-Meteo weather. METRIC converged in 5 iterations; ET24 was about
10.8 mm/d on irrigated fields (ETr24 12.0) and 2.2 mm/d on bare land; SEBAL with constant EF
gave lower values (median 3.8 mm/d), the advection effect notebook 06 should explain; TSEB
gave T/ET 0.84 over irrigated crops and closed the energy balance exactly.

## Notebook list

Status (23 November 2026): every notebook is built by `_builders/build_08_12_XX.py`, executed
without errors in the `python3` kernel (Python 3.11), and its markdown checked against its outputs
and figures. The section folder was renamed from `08_12_...` to `09_12_...` on 23 November 2026;
notebook files keep the `08_12_` prefix like the other sections. Run 02, 06, 07 and 10 before 11
(it reads their outputs); every other notebook runs on its own through `seb_data.py`. After any
`run.py` or `sync_text.py`, rerun the installation notes (`helper_functions/Tutorials/_tools/notebook_install_notes.py`,
or its `update()` for these notebooks only).

| Notebook | Built | Main result |
|---|---|---|
| 00 fundamentals | yes | fallow 9 to 14 K hotter than crops; LST and NDVI correlation -0.93; METRIC energy partition by crop group |
| 01 inputs | yes | 27 clear 2024 scenes cached (563 MB); cloud fringes 1.2 K cooler than clear land; masks must not inherit `_FillValue` 0 |
| 02 reference ET | yes | FAO-56 examples reproduced (3.88 and 0.627); 5 July 2024 ETr 15.6 mm; ERA5 vs gridMET R2 0.93; aridity adjustment -12 % |
| 03 radiation balance | yes | radiative transfer LST equals ST_B10 within 0.15 K; albedo vs MCD43A4 RMSE 0.018; orchards about 80 W m-2 more Rn than fallow |
| 04 soil heat and aerodynamics | yes | SEBAL G 27 to 41 W m-2 above METRIC over crops; stability triples H at a hot anchor's dT of 5 K |
| 05 SEBAL | yes | hot pixel rah 60.5 to 17.2 s/m in 5 iterations; SEBAL daily ET 4 to 5.5 mm for crops on a 15.6 mm ETr day |
| 06 METRIC | yes | cold anchor needs 657 W m-2 against 605 of Rn - G; 37 % of pixels H < 0; METRIC daily ET 2.1 to 2.4 times SEBAL |
| 07 SSEBop and simple models | yes | thermal models correlate 0.94 to 0.99; PT-JPL far lower; ensemble drops PT-JPL at 50 % and METRIC at 21 % of pixels |
| 08 two-source model | yes | T/ET 0.86 tomatoes, 0.44 almonds; +13 W m-2 LE per K of air temperature; tall canopies need forcing heights above the canopy |
| 09 MODIS, Sentinel-3 and Sentinel-2 | yes | MODIS METRIC +3.1 mm/day vs Landsat (mixed cold anchor); DMS halves bilinear RMSE (2.3 vs 4.5 K); S2 + sharpened S3 ETrF R2 0.79 vs next-day Landsat |
| 10 seasonal ET | yes | annual ET 1,102 mm (window mean) after rain and winter adjustments; tomatoes, cotton, wheat Kc match FAO-56; fallow 357 mm in a 38 mm season (bias) |
| 11 validation and uncertainty | yes | dry-land excess about 300 mm (fallow) per season; MOD16 293 mm/yr; uniform 2 K LST error 0.5 %, 10 % ETr error up to 23 %; crop budget about 25 % |
| 12 capstone | yes | 20,600 ha of Boro rice; 494 (METRIC) to 575 mm (SSEBop) ET from 15 Jan to 30 Apr; 100 to 120 million m3 consumed; anchors only 2.5 to 10 K apart |

Changes from the plan: notebook 10 measures the effect of revisit by thinning the image record
instead of adding HLS (HLS has no thermal band); notebook 11 validates with a dry-land water balance,
MOD16, reference ET choices and an uncertainty budget, and leaves AmeriFlux and OpenET to readers with
free accounts; notebook 12 is the Bangladesh capstone (Trishal and Bhaluka, Mymensingh district).

### 09_12_00_evapotranspiration_and_the_surface_energy_balance.ipynb
Evapotranspiration and the Surface Energy Balance
- Overview: what ET is (evaporation, transpiration, interception); why it matters for water
  rights, irrigation scheduling, drought and climate; the water balance vs the energy balance
  view; latent heat and the conversion from W m-2 to mm.
- The energy balance `Rn = G + H + LE`: each term, typical midday values over crops, bare
  soil and water; diagrams of the flux paths; energy closure at flux towers.
- Why thermal images: surface temperature as the signal of evaporative cooling; the
  Ts minus Ta gradient; why surface temperature is not aerodynamic temperature.
- A family tree of satellite ET methods: one-source residual models (SEBAL, METRIC, SEBS),
  simplified and contextual models (SSEBop, S-SEBI, triangle), two-source (TSEB, ALEXI and
  DisALEXI), Priestley-Taylor (PT-JPL), Penman-Monteith (MOD16), crop coefficient (SIMS).
  The OpenET ensemble and what each member needs.
- Instantaneous to daily to seasonal: the evaporative fraction and reference ET fraction
  ideas, in words; satellite revisit and clouds.
- Quick hands on: one Landsat 9 scene of the Westside site; NDVI, LST and a scatter of LST
  against NDVI coloured by CDL crop, showing the cool irrigated and hot dry ends that anchor
  every model in this chapter.

### 09_12_01_building_an_analysis_ready_stack_for_energy_balance_models.ipynb
Building an Analysis-Ready Stack for Energy Balance Models
- Overview: what every SEB model needs (reflectance, LST, emissivity, elevation, land cover,
  weather at overpass, daily reference ET) and where errors come from.
- Landsat C2 L2 from STAC (`get_images`), scale factors, `QA_PIXEL` cloud, shadow and
  water masks with a dilation buffer; why one missed cloud edge ruins a cold anchor.
- Overpass time and solar geometry from the MTL file; sun elevation, sun-Earth distance.
- Copernicus DEM on the Landsat grid, slope and aspect; CDL 2024 on the same grid.
- Weather at overpass: ERA5 hourly from Open-Meteo for the scene centre and a small grid;
  gridMET daily; wind at blending height.
- One `xr.Dataset` per scene saved to NetCDF with all inputs and their units, so later
  notebooks load it in one line.

### 09_12_02_reference_et_with_the_asce_standardized_penman_monteith_equation.ipynb
Reference ET with the ASCE Standardized Penman-Monteith Equation
- Overview: grass (ETo) and alfalfa (ETr) references; the Penman-Monteith equation term by
  term; hourly vs daily constants; why METRIC calibrates against ETr.
- `asce_pm_hourly` and `asce_pm_daily` checked against ASCE (2005) worked examples.
- ERA5 vs gridMET vs CIMIS station: hourly ETr at overpass and daily ETr for the 2024 season;
  bias of gridded weather over irrigated land (the aridity effect) and a simple correction.
- Output: hourly ETr at each overpass and a daily ETr series, saved for notebooks 06 and 10.

### 09_12_03_net_radiation_from_landsat_albedo_emissivity_and_surface_temperature.ipynb
Net Radiation from Landsat: Albedo, Emissivity and Surface Temperature
- Overview: shortwave and longwave balance; broadband albedo from narrow bands; emissivity;
  atmospheric transmissivity; terrain effects.
- LST two ways: the Collection 2 `ST_B10` product, and from scratch from Level-1 band 10
  (radiance, brightness temperature, NDVI emissivity, atmospheric correction with
  `ST_ATRAN`, `ST_URAD`, `ST_DRAD`); differences by land cover.
- Albedo: Tasumi et al. (2008) and Liang (2001) weights; comparison with MODIS MCD43
  albedo aggregated to 500 m.
- Incoming shortwave on sloping ground, incoming longwave from air emissivity, outgoing
  longwave; `net_radiation` maps; Rn vs NDVI by crop.

### 09_12_04_soil_heat_flux_roughness_and_aerodynamic_resistance.ipynb
Soil Heat Flux, Roughness and Aerodynamic Resistance
- Overview: G/Rn and its dependence on cover; the log wind profile, roughness length and
  displacement height; friction velocity; aerodynamic resistance between two heights;
  atmospheric stability and Monin-Obukhov theory.
- `soil_heat_flux_bastiaanssen` vs the METRIC LAI form; values over water and wet soil.
- Momentum roughness from NDVI, from LAI and from a CDL lookup; effect on rah.
- Stability: neutral rah vs stable and unstable corrections; a small experiment showing how
  H changes for the same Ts minus Ta under different stability.

### 09_12_05_sebal_solving_for_sensible_heat_with_hot_and_cold_pixels.ipynb
SEBAL: Solving for Sensible Heat with Hot and Cold Pixels
- Overview: Bastiaanssen et al. (1998); the linear dT = a + b Ts assumption and why it
  removes the need for air temperature over every pixel; the iteration for stability.
- Anchor pixels: manual choice on the NDVI and LST maps, then `candidate_anchors` and
  `select_anchors` (Allen et al. 2013 percentile method) with CDL screens; how much the result
  moves with the anchors.
- `calibrate_dt` and `sensible_heat_iterative`; convergence plot of rah and H at the hot
  pixel; maps of H, LE and the evaporative fraction.
- Instantaneous LE to daily ET with constant EF and daily net radiation.
- Checks: LE at the cold pixel vs available energy, negative H and LE, water pixels.

### 09_12_06_metric_internal_calibration_against_reference_et.ipynb
METRIC: Internal Calibration against Reference ET
- Overview: Allen et al. (2007); what METRIC changes from SEBAL (cold pixel at 1.05 ETr,
  hot pixel with a soil water balance, slope and aspect corrections, ETrF for daily
  scaling); eeMETRIC in OpenET.
- `calibrate_metric` with hourly ETr from notebook 02; ETrF map; daily ET from ETrF and
  daily ETr.
- SEBAL vs METRIC on the same scene: maps, difference by crop, where they disagree and why
  (anchor ET assumptions, daily scaling).
- Repeat for five dates in 2024 to see how the calibration coefficients a and b change over
  the season.

### 09_12_07_ssebop_s_sebi_the_triangle_method_and_pt_jpl.ipynb
SSEBop, S-SEBI, the Triangle Method and PT-JPL
- Overview: simplified models that skip the aerodynamic solution; SSEBop (Senay et al.
  2013, 2023) with a cold reference from Tmax and a climatological dT; S-SEBI with
  albedo and LST edges; the Ts and NDVI triangle; PT-JPL (Fisher et al. 2008).
- Each model on the same scene; comparison with METRIC; time and inputs needed.
- A small ensemble (median of the models), in the OpenET spirit, and its spread as an
  uncertainty map.

### 09_12_08_two_source_energy_balance_separating_soil_evaporation_and_transpiration.ipynb
Two-Source Energy Balance: Separating Soil Evaporation and Transpiration
- Overview: Norman et al. (1995); why one-source models struggle with partial cover; the
  soil and canopy temperatures; parallel and series resistances; Priestley-Taylor start for the canopy;
  ALEXI and DisALEXI in words.
- `tseb_pt` (parallel resistance network) on the Westside scene with LAI from Landsat; E and T maps; T/ET by crop and
  cover fraction.
- Optional cross-check with `pyTSEB` on the same inputs.

### 09_12_09_et_at_other_resolutions_modis_sentinel_3_and_sentinel_2_with_thermal_sharpening.ipynb
ET at Other Resolutions: MODIS, Sentinel-3 and Sentinel-2 with Thermal Sharpening
- Overview: the trade between pixel size and revisit; Sentinel-2 has no thermal band; MODIS
  and Sentinel-3 have daily thermal at 1 km; the MOD16 Penman-Monteith product.
- METRIC on MODIS (MOD11A1 LST, MCD43A4, MOD15 LAI) at 1 km; comparison with Landsat
  METRIC aggregated to 1 km.
- Thermal sharpening: Sentinel-3 SLSTR LST sharpened to 20 m with Sentinel-2 reflectance
  (`thermal_sharpen`, Gao et al. 2012 data mining sharpener); check by sharpening Landsat LST
  aggregated to 1 km and comparing with the original 30 m.
- Sentinel-2 plus sharpened LST ET at 20 m on a date with no Landsat pass.

### 09_12_10_from_snapshots_to_seasonal_et_and_crop_water_use.ipynb
From Snapshots to Seasonal ET and Crop Water Use
- Overview: why clouds and a 8 to 16 day revisit need gap filling; interpolating ETrF
  (not ET) between dates; the reference ET as the daily driver.
- All clear 2024 scenes through `run_metric`; `interpolate_etrf` and `seasonal_et`; monthly
  and seasonal ET maps.
- Crop water use per field: `zonal_et` with CDL; ET by crop (almonds, pistachios, tomatoes,
  cotton, fallow); satellite crop coefficients compared with FAO-56 Kc curves.
- HLS (Landsat plus Sentinel-2) for a denser ETrF series and its effect on seasonal totals.

### 09_12_11_how_good_is_the_et_map_validation_sensitivity_and_uncertainty.ipynb
How Good Is the ET Map? Validation, Sensitivity and Uncertainty
- Overview: flux tower footprints, energy closure, scale mismatch; what OpenET found in its
  accuracy assessment (Volk et al. 2024), in words.
- Sensitivity: one at a time and Sobol style sensitivity of daily ET to anchors, wind,
  air temperature, albedo and LST error.
- `monte_carlo_anchors`: many anchor pairs from the candidate pool and the spread of ET.
- Comparison with MOD16 annual ET at 500 m and, if keys are set, AmeriFlux towers and
  OpenET ensemble values; RMSE, bias and R2 tables.

### 09_12_12_capstone_water_use_of_boro_rice_in_bangladesh.ipynb
Capstone: Water Use of Boro Rice in Bangladesh
- Question: how much water does dry season Boro rice use around Mymensingh, and how does it
  compare with groundwater pumping estimates?
- Full chain on Landsat 8/9 from January to April: inputs, ERA5 reference ET, METRIC and
  SSEBop, flooded paddies as cold anchors, seasonal ET, rice area from NDWI and NDVI.
- Discussion: humid tropics vs the California test site, haze and cloud, what limits
  accuracy, and how to scale to the whole country with HLS.

## Open decisions

- Library home: decided, one module `geolattice.rsense.surface_energy_balance`.
- Capstone: decided, Bangladesh Boro rice.
- Optional keys (CIMIS, AmeriFlux, OpenET): keep them optional with a skip message, as above.

## References

- Allen, R. G., Tasumi, M., and Trezza, R. (2007). Satellite-based energy balance for mapping
  evapotranspiration with internalized calibration (METRIC): model. J. Irrig. Drain. Eng.
- Allen, R. G., et al. (2013). Automated calibration of the METRIC-Landsat evapotranspiration
  process. JAWRA.
- ASCE-EWRI (2005). The ASCE standardized reference evapotranspiration equation.
- Bastiaanssen, W. G. M., et al. (1998). A remote sensing surface energy balance algorithm for
  land (SEBAL). J. Hydrol.
- Fisher, J. B., Tu, K. P., and Baldocchi, D. D. (2008). Global estimates of the land
  atmosphere water flux (PT-JPL). Remote Sens. Environ.
- Gao, F., Kustas, W. P., and Anderson, M. C. (2012). A data mining approach for sharpening
  thermal satellite imagery over land. Remote Sens.
- Melton, F. S., et al. (2022). OpenET: filling a critical data gap in water management for
  the western United States. JAWRA.
- Norman, J. M., Kustas, W. P., and Humes, K. S. (1995). Source approach for estimating soil
  and vegetation energy fluxes (TSEB). Agric. For. Meteorol.
- Senay, G. B., et al. (2013, 2023). SSEBop. JAWRA; Remote Sens. Environ.
- Tasumi, M., Allen, R. G., and Trezza, R. (2008). At-surface reflectance and albedo from
  satellite for operational calculation of land surface energy balance. J. Hydrol. Eng.
- Volk, J. M., et al. (2024). Assessing the accuracy of OpenET satellite-based
  evapotranspiration data. Nature Water.
