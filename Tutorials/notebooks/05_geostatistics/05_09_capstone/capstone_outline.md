# Geostatistics Capstones: Outline of Candidate Projects

Module: `Tutorials/notebooks/05_geostatistics/05_09_capstone`
Package: `src/geostatistics` (`geolattice.geostatistics`, v0.1.1)
Data: `Data/00_shared/data_geostatistics`

## 1. Purpose

Chapters 05_01 to 05_08 teach one method per notebook. The capstones should do the opposite: start from an applied question, pick the methods the question needs, and carry the analysis through to a decision product (a risk map, an uncertainty statement, a model ranking). Each capstone below integrates at least three chapters and exercises a distinct part of `src/geostatistics`.

Following the convention in `04_10_capstone`, each capstone is one notebook plus one `.qmd` guideline in the same folder.

## 2. Candidate capstones at a glance

| No. | Capstone | Main data | Chapters integrated | Core package classes |
|---|---|---|---|---|
| 05_09_01 | Groundwater arsenic risk in Bangladesh | `BGS_BD/bgs_geochemical.csv` (about 3,500 wells), `bd_grid.csv` | 01, 02, 04, 07 | `IndicatorKriging`, `DisjunctiveKriging`, `SoftKriging`, `sisim`, `kt3d` |
| 05_09_02 | Arsenic transfer from irrigation water to soil to rice (Tala) | `Arsenic - Tala/water_soil_rice_arsenic_data.csv` (263 fields, 2 years), `Tala_grid.csv` | 01, 02, 04, 07 | `Cokriging`, `fit_lmc`, `ColocatedCokriging`, `sgcosim`, `FactorialKriging` |
| 05_09_03 | Digital soil mapping of SOC with uncertainty (Great Plains) | `gp_data.csv` (471 sites), `gp_data_5000.csv`, `gp_grid.csv`, `Data_GP/RASTER` | 02, 03, 07, 08 | `RegressionKriging`, `RegressionKrigingH2O`, `DeepRegressionKriging`, `GNNRegressionKriging`, `KFoldCV_RK`, `sgsim` |
| 05_09_04 | Soil fertility and micronutrient deficiency mapping (Rajshahi) | `BD_Raj/Soil_Data/Raj_all_data.csv` (about 4,600 sites), train/val/test splits, `RAJ_Raster` | 02, 03, 04, 08 | `RegressionKriging`, `MultivariateCokriging`, `IndicatorKriging`, `EmpiricalBayesianKriging` |
| 05_09_05 | Space-time PM2.5 exposure in California, 2025 | `CA_pm25_2025.csv` (about 59,600 daily records) | 01, 05 | `STKriging`, `fit_st_variogram`, `krige_st_local`, `CNNLSTMRegressionKriging`, `STGNNRegressionKriging`, `STGTNRegressionKriging` |
| 05_09_06 | Lung and bronchus cancer mortality risk, USA 1998 to 2012 | `LBC_1998_2012_USA.csv` (about 46,600 county-years), `lbc_rate.csv`, `cancer_areas.shp`, `cancer_points.csv`, `US_COUNTY.shp` | 06, 05 | `PoissonKriging`, `AreaPoissonKriging`, `poisson_cokrige`, `shared_risk`, `STPoissonKriging` |
| 05_09_07 | Continental soil geochemistry (USGS) | `USA_Geochemical/usa_geochemical.csv` (about 4,850 sites, 45+ elements) | 01, 02, 04, 08 | `Variogram`, `FactorialKriging`, `fit_nested_variogram`, `MultivariateCokriging`, `BayesianTransformedKriging` |
| 05_09_08 | Subsurface facies and property modeling | `GSLIB_data/twowell.dat`, `cluster.dat`, `true.dat`, `ydata.dat` | 01, 04, 07 | `declus_cell`, `nscore_forward`, `sgsim`, `sisim`, `pgsim`, `SNESIM`, `DirectSampling`, `kt3d` |
| 05_09_09 | Method benchmark on Jura heavy metals | `jura_train.csv`, `jura_test.csv`, `jura_grid.csv`, `jura_transect.csv` | 02, 04, 08 | all kriging classes, `loo_cv`, `kfold_cv`, `BayesianKriging`, `ConjugateBayesianKriging` |

Together the nine cover every subpackage: `core`, `kriging`, `ml`, `spacetime`, `simulation`, `mps`, `gslib`, `validation` and `gpu`.

## 3. Common notebook structure

Every capstone follows the same ten sections so readers can compare them.

1. Problem statement, decision to be supported and research questions
2. Data description, provenance and loading
3. Data preparation: projection, duplicates, censored values, transformation (`transform_log`, `transform_boxcox`, `nscore_forward`), declustering (`declus_cell`)
4. Exploratory spatial analysis: maps, histograms, variogram cloud, directional variograms, anisotropy
5. Variogram or covariance modeling with justification of the chosen model
6. Prediction with two or more competing methods
7. Validation: leave-one-out, k-fold and spatially blocked CV, with RMSE, MAE, bias, standardized error and coverage of prediction intervals
8. Uncertainty: kriging variance, conditional simulation, probability of exceedance
9. Decision product: final maps and a summary table for a non-technical reader
10. Limitations, extensions and exercises

## 4. Capstone outlines

### 05_09_01 Groundwater arsenic risk in Bangladesh

Question: where does arsenic in tube wells exceed the WHO (10 µg/L) and Bangladesh (50 µg/L) limits, and how certain is that classification?

- Data: BGS/DPHE national survey with coordinates, well depth, well type, construction year and about 20 co-analytes (Fe, Mn, P, SO4 and others).
- Preparation: handle left-censored values (entries such as `< 0.04`), project to BUTM, log transform, decluster.
- Variography: omnidirectional and directional variograms of log As; indicator variograms at several thresholds (`fit_indicator_variogram`).
- Prediction: ordinary kriging of log As with back transformation; indicator kriging for exceedance probability; disjunctive kriging as a comparison; soft kriging with Fe or well depth as soft information (`markov_bayes_calibrate`).
- Depth: split shallow and deep aquifers, or treat depth as the third coordinate with `kt3d`.
- Simulation: `sisim` realizations to estimate the area and number of unions above each threshold, with intervals.
- Validation: `IndicatorKrigingCV`, misclassification rates (false safe, false unsafe).
- Product: probability map, classification map at a chosen risk tolerance, upazila-level summary table.

### 05_09_02 Arsenic from irrigation water to soil to rice grain (Tala)

Question: can soil and water arsenic improve the mapping of grain arsenic, and where is dietary exposure of concern?

- Data: 263 paddy fields over two years with water As, soil As (total, phosphate and oxalate extractable), grain and straw As, soil properties, elevation, years of irrigation, distance to shallow tube well. Source paper is in the folder (Ahmed et al., STOTEN 2011).
- Variography: direct and cross variograms for water, soil and grain As; fit a linear model of coregionalization (`fit_lmc`, `check_lmc_validity`).
- Prediction: ordinary kriging of grain As as baseline; cokriging with soil As as secondary variable; colocated cokriging; regression kriging with soil and landscape covariates.
- Scale analysis: factorial kriging to separate short-range (field management) from long-range (landscape, flooding) components.
- Simulation: `sgcosim` joint realizations of soil and grain As; propagate to a daily intake estimate per grid cell.
- Validation: `KFoldCV_Cokriging`; gain of cokriging over univariate kriging.
- Product: grain As map with uncertainty, probability of exceeding a food-safety guideline, year-to-year comparison.

### 05_09_03 Digital soil mapping of SOC with uncertainty (Great Plains)

Question: which regression kriging engine predicts SOC best across four states, and what is the uncertainty of the regional SOC estimate?

- Data: 471 sites with SOC and nine covariates (elevation, slope, aspect, TPI, K factor, MAP, MAT, NDVI, silt plus clay, NLCD, FRG); prediction grid of about 10,700 cells; a 5,000-point version for scaling tests.
- Baselines: ordinary kriging and universal kriging with drift.
- Model comparison: `RegressionKriging` (random forest, gradient boosting, GAM), `RegressionKrigingH2O` (AutoML and stacked ensemble), `DeepRegressionKriging`, `DeepRegressionKrigingTF`, `GNNRegressionKriging`.
- Validation: random versus spatially blocked k-fold (`KFoldCV_RK`, `KFoldCV_H2O_RK`, `KFoldCV_PyTorch_RK`) to show optimism of random CV; residual variograms to check what each learner leaves behind.
- Uncertainty: quantile outputs, kriging variance of residuals, `sgsim` on residuals to get the distribution of the regional mean.
- Bayesian comparison: `EmpiricalBayesianKriging` and `BayesianKriging` on the same split.
- Scaling: CPU versus GPU timing on the 5,000-point set.
- Product: ranked model table, final SOC map with prediction interval width, state-level means with intervals.

The existing `gp_prediction_RK*.csv` files suggest the chapter 03 tutorials already use this dataset. The capstone should therefore focus on the head-to-head comparison and uncertainty, which the single-engine tutorials do not do.

### 05_09_04 Soil fertility and micronutrient deficiency mapping (Rajshahi)

Question: where are Zn, B and S below critical levels, and what fertilizer recommendation zones follow?

- Data: about 4,600 samples with pH, SOM, N, P, K, Ca, Mg, S, B, Cu, Fe, Mn, Zn; covariates DEM, NDVI, NDFI, AEZ, land type, drainage, permeability, texture and soil type; ready train, validation and test splits; a full prediction grid.
- Preparation: mixed numeric and categorical covariates, skewed nutrients, administrative units for reporting.
- Prediction: regression kriging per nutrient; multivariate cokriging for correlated nutrients (for example Ca and Mg, Zn and SOM); EBK for a locally varying variogram.
- Deficiency: indicator kriging at critical limits; combine probabilities into a multi-nutrient deficiency index.
- Validation: independent test set plus spatial CV by upazila.
- Product: deficiency probability maps, union-level recommendation classes.

### 05_09_05 Space-time PM2.5 exposure in California, 2025

Question: how well can daily PM2.5 be interpolated between monitors, and which approach handles wildfire smoke episodes?

- Data: daily PM2.5 and AQI at monitoring sites for 2025 (`load_ca_pm25_panel`), with `make_prediction_grid`.
- Exploration: station time series, seasonality, missing days, spatial and temporal marginal variograms.
- Space-time variography: `empirical_st_variogram`; fit separable, product-sum, metric and sum-metric models (`fit_st_variogram`); compare with `variogram_surface`.
- Prediction: `STKriging` and `krige_st_local`; purely spatial day-by-day kriging as baseline.
- Deep learning: `CNNLSTMRegressionKriging`, `STGNNRegressionKriging` (T-GCN, DCRNN), `STGTNRegressionKriging` (STTN, GMAN).
- Validation: leave-station-out and leave-time-block-out; separate scores for smoke days and normal days.
- Product: daily surfaces for a selected episode, annual mean map, days above 35 µg/m³ per county.

Chapter 05 already uses this panel, so the capstone value is the full model comparison and the exposure summary.

### 05_09_06 Lung and bronchus cancer mortality risk, USA 1998 to 2012

Question: what is the underlying mortality risk once small-population noise is removed, and how did it change over 15 years?

- Data: county LBC mortality rate by year with smoking, poverty, PM2.5, NO2, SO2, urban-rural class and coal production; a northeast subset with population (`lbc_rate.csv`, `cancer_areas`, `cancer_points`); county and state boundaries.
- Motivation: show instability of raw rates in small counties; empirical Bayes smoothing (`eb_risk`) as a first fix.
- Poisson kriging: population-weighted variogram (`pck_variogram`); area-to-area and area-to-point kriging (`AreaPoissonKriging`, `PointSupport`) to produce a continuous risk surface that preserves county totals.
- Multivariate: Poisson cokriging with a second outcome, for example diabetes from `diabetes_usa.csv`, and a shared-risk component (`poisson_cokrige`, `shared_risk`).
- Space-time: `STPoissonKriging` across 1998 to 2012; trend maps by region.
- Covariates: residual risk after adjusting for smoking and poverty, to show what air quality adds.
- Validation: coherence check (aggregated ATP predictions equal ATA), comparison with raw and EB rates.
- Product: smoothed risk maps by period, list of persistent high-risk counties, change map.

### 05_09_07 Continental soil geochemistry (USGS)

Question: at what spatial scales do soil elements vary across the conterminous USA, and which signals are geogenic versus anthropogenic?

- Data: top-soil (0 to 20 cm) concentrations of 45+ elements at about 4,850 sites with land cover.
- Preparation: censored values (`<1`, `N.D.`), projection to Albers, log or Box-Cox transform.
- Variography: nested models for selected elements (As, Pb, Hg, Zn, organic C) with `fit_nested_variogram`.
- Scale decomposition: factorial kriging maps of short-range and regional components.
- Multivariate: `MultivariateCokriging` for an under-sampled element using correlated elements.
- Bayesian: `BayesianTransformedKriging` and EBK for non-stationary continental data.
- Product: element maps with uncertainty, exceedance probability for a soil screening level, a short interpretation by physiographic region.

### 05_09_08 Subsurface facies and property modeling

Question: how do two-point and multiple-point methods differ in reproducing geological continuity and flow-relevant connectivity?

- Data: GSLIB reference sets: two-well section with facies, clay volume, porosity and permeability; clustered samples with declustering weights; exhaustive `true.dat` for checking.
- Preparation: `read_gslib`, cell declustering, normal-score transform.
- Continuous properties: `sgsim` for porosity; `sgcosim` for permeability conditioned on porosity; `dssim_direct` and `sasim` as alternatives.
- Facies: `sisim`, `pgsim` with a truncation rule, `SNESIM` and `DirectSampling` with `channel_training_image`.
- Checks: histogram and variogram reproduction, `connectivity`, `pattern_histogram_distance`, comparison with the exhaustive image.
- Performance: CPU versus GPU versions in `gpu/`.
- Product: realization gallery, E-type and probability maps, a summary of which method suits which geology.

### 05_09_09 Method benchmark on Jura heavy metals

Question: for a small, well-known dataset, which kriging variant gives the best prediction and the most honest uncertainty?

- Data: 259 training and 100 test locations with Cd, Co, Cr, Cu, Ni, Pb, Zn, land use and rock type; prediction grid and transect.
- Methods: simple, ordinary and universal kriging; cokriging; indicator and disjunctive kriging; regression kriging with categorical covariates; EBK; `BayesianKriging`, `ConjugateBayesianKriging`, `BayesianTransformedKriging`.
- Scoring: RMSE, MAE, mean standardized error, interval coverage and CRPS on the independent test set.
- Product: one leaderboard table and a short discussion of when added complexity pays off.

This is the shortest capstone and works as a bridge between the tutorials and the larger projects.

## 5. Suggested build order

1. 05_09_01 Groundwater arsenic: strong applied story, covers indicator methods and simulation.
2. 05_09_06 Cancer mortality: the only capstone for Poisson kriging and it links to the USA Health Data Atlas.
3. 05_09_03 SOC digital soil mapping: showcases all `ml` engines and spatial CV.
4. 05_09_05 PM2.5 space-time: showcases `spacetime` and deep learning models.
5. 05_09_02, 05_09_04, 05_09_07, 05_09_08, 05_09_09 as time allows.

The first four alone cover every chapter from 05_01 to 05_08.

## 6. Points to confirm before building

- Units in `bgs_geochemical.csv` and the treatment of censored values need to be checked against the BGS documentation before thresholds are applied.
- Which datasets each existing tutorial already uses was inferred from file names and loader functions, not from the notebooks themselves. Capstones 03 and 05 are written to add to those tutorials, not repeat them.
- `LBC_1998_2012_USA.csv` holds rates, not counts. Poisson kriging needs population by county and year, which may have to be joined from another source.
- `meuse.csv` and `meuse_grid.csv` are left for the tutorials and for unit tests.
